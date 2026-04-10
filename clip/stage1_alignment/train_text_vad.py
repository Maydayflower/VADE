import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import os
import sys

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor, get_cosine_schedule_with_warmup

# 添加父目录到Python路径
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# 导入core模块
from core.vad_data import (
    DEFAULT_VAD,
    ImageTextExample,
    load_image_text_file,
    save_vad_stats,
)

logger = logging.getLogger(__name__)


class SimpleImageTextDataset(Dataset):
    """Simple dataset that returns image paths and texts (stylized captions)."""
    
    def __init__(self, examples: Sequence) -> None:
        self.examples = list(examples)
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict:
        example = self.examples[idx]
        return {
            "image_path": example.image_path,
            "text": example.sentence,  # stylized caption
        }


def load_images(paths) -> List[Image.Image]:
    """Load images from a list of paths or a single path string."""
    if isinstance(paths, str):
        paths = [paths]
    elif not isinstance(paths, (list, tuple)):
        # Handle tensor or other types
        paths = list(paths) if hasattr(paths, '__iter__') else [str(paths)]
    
    images: List[Image.Image] = []
    for path in paths:
        path_str = str(path) if not isinstance(path, str) else path
        try:
            with Image.open(path_str) as img:
                images.append(img.convert("RGB").copy())
        except Exception as e:
            raise FileNotFoundError(f"Failed to load image from {path_str}: {e}")
    return images


class ClipImageTextAlignmentModel(nn.Module):
    """CLIP-based model that aligns image features with stylized text features.
    
    Training Strategy:
        - Align image features with stylized caption text features in CLIP semantic space
        - Loss: MSE(image_features, text_features) - make them closer
        - This leverages CLIP's powerful semantic alignment capability
    
    Inference Strategy:
        - Image → CLIP Image Encoder → image_features
        - image_features → VAD Predictor → VAD
        - Since image_features are aligned with stylized text features,
          they naturally contain emotional information for VAD prediction
    
    Key Insight:
        Instead of directly training image→VAD, we train image→stylized_text in semantic space.
        The aligned features naturally encode emotional semantics from stylized captions.
    """

    def __init__(
        self,
        pretrained_path: str,
        freeze_vision_layers: Optional[int] = None,
        freeze_text_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.clip = CLIPModel.from_pretrained(pretrained_path)
        projection_dim = getattr(self.clip.config, "projection_dim", None)
        if projection_dim is None:
            raise ValueError(f"CLIP model at {pretrained_path} does not expose projection_dim.")

        # Get total number of layers
        vision_encoder = self.clip.vision_model.encoder
        total_vision_layers = len(vision_encoder.layers)
        
        # Log layer information
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"CLIP Vision Encoder: {total_vision_layers} layers total")

        # Freeze text encoder (we only fine-tune vision encoder)
        if freeze_text_encoder:
            for param in self.clip.text_model.parameters():
                param.requires_grad = False
            logger.info("Frozen text encoder (only fine-tuning vision encoder)")

        # Freeze early vision layers if specified
        if freeze_vision_layers is not None and freeze_vision_layers > 0:
            if freeze_vision_layers > total_vision_layers:
                logger.warning(f"freeze_vision_layers ({freeze_vision_layers}) > total layers ({total_vision_layers}), freezing all vision layers")
                freeze_vision_layers = total_vision_layers
            for layer in vision_encoder.layers[:freeze_vision_layers]:
                for param in layer.parameters():
                    param.requires_grad = False
            logger.info(f"Frozen {freeze_vision_layers}/{total_vision_layers} vision encoder layers")

    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        Forward pass for training: align image and text features.
        
        Args:
            pixel_values: Image tensor [batch_size, 3, H, W]
            input_ids: Text tokens [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            image_features: Image features [batch_size, projection_dim]
            text_features: Text features [batch_size, projection_dim]
        """
        # Extract features from CLIP encoders
        image_features = self.clip.get_image_features(pixel_values=pixel_values)  # [batch_size, projection_dim]
        text_features = self.clip.get_text_features(input_ids=input_ids, attention_mask=attention_mask)  # [batch_size, projection_dim]
        
        return image_features, text_features
    
    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Extract image features for inference.
        
        Args:
            pixel_values: Image tensor [batch_size, 3, H, W]
            
        Returns:
            image_features: Image features [batch_size, projection_dim]
        """
        return self.clip.get_image_features(pixel_values=pixel_values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune CLIP vision encoder to align with stylized text features.")
    parser.add_argument("--train_file", type=str, required=True, help="Training TSV with `image_path<TAB>original_captions<TAB>emotional_caption` per line")
    parser.add_argument("--val_file", type=str, default=None, help="Optional validation TSV")
    parser.add_argument("--pretrained_clip", type=str, default="/workspace/models/clip-vit-base-patch32")
    parser.add_argument("--max_length", type=int, default=77, help="Max length for text tokenizer")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--freeze_vision_layers", type=int, default=2, help="Number of earliest vision layers to freeze")
    parser.add_argument("--freeze_text_encoder", action="store_true", help="Freeze text encoder (only train vision)")
    parser.add_argument("--gradient_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, help="Force device e.g. cuda or cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    """Custom collate function to properly handle string lists."""
    if not batch:
        raise ValueError("Empty batch")
    
    image_paths = []
    texts = []
    
    for item in batch:
        image_paths.append(str(item["image_path"]))
        texts.append(str(item["text"]))
    
    return {
        "image_path": image_paths,
        "text": texts,
    }


def create_dataloader(
    dataset: Optional[Dataset],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> Optional[DataLoader]:
    if dataset is None:
        return None
    # Use num_workers=0 to avoid collate_fn issues in worker processes
    effective_num_workers = 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=effective_num_workers,
        pin_memory=torch.cuda.is_available() and effective_num_workers == 0,
        collate_fn=collate_fn,
    )


def evaluate(
    model: ClipImageTextAlignmentModel,
    processor: CLIPProcessor,
    dataloader: Optional[DataLoader],
    device: torch.device,
    criterion,
    max_length: int,
) -> Dict[str, float]:
    """
    Evaluate model: L = MSE(image_features, text_features)
    Measure how well image features align with stylized text features.
    """
    if dataloader is None:
        return {"loss": 0.0, "cosine_sim": 0.0}
    model.eval()
    losses: List[float] = []
    cosine_sims: List[float] = []
    
    with torch.no_grad():
        for batch in dataloader:
            image_paths = batch["image_path"]
            texts = batch["text"]
            images = load_images(image_paths)
            
            # Process images and texts
            inputs = processor(
                text=texts,
                images=images,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            pixel_values = inputs["pixel_values"].to(device)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            # Get features
            image_features, text_features = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Normalize features
            image_features_norm = torch.nn.functional.normalize(image_features, p=2, dim=1)
            text_features_norm = torch.nn.functional.normalize(text_features, p=2, dim=1)
            
            # Loss: MSE between image and text features
            loss = criterion(image_features, text_features)
            losses.append(loss.item())
            
            # Also compute cosine similarity for monitoring
            cosine_sim = (image_features_norm * text_features_norm).sum(dim=1).mean()
            cosine_sims.append(cosine_sim.item())
    
    if not losses:
        return {"loss": 0.0, "cosine_sim": 0.0}
    mean_loss = float(sum(losses) / len(losses))
    mean_cosine = float(sum(cosine_sims) / len(cosine_sims))
    return {"loss": mean_loss, "cosine_sim": mean_cosine}


def train() -> None:
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logger.info("Arguments: %s", vars(args))

    set_seed(args.seed)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    logger.info("Using device: %s", device)

    # Load data
    train_examples = load_image_text_file(args.train_file)
    val_examples = load_image_text_file(args.val_file) if args.val_file else None
    logger.info("Loaded %d training examples", len(train_examples))
    if val_examples:
        logger.info("Loaded %d validation examples", len(val_examples))

    # Create datasets (simple: just return image_path and text)
    train_dataset = SimpleImageTextDataset(train_examples)
    val_dataset = SimpleImageTextDataset(val_examples) if val_examples else None

    # Create dataloaders
    train_loader = create_dataloader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = create_dataloader(val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Load CLIP processor
    processor = CLIPProcessor.from_pretrained(args.pretrained_clip)
    effective_max_length = min(args.max_length, getattr(processor.tokenizer, "model_max_length", args.max_length))

    # Initialize image-text alignment model
    model = ClipImageTextAlignmentModel(
        pretrained_path=args.pretrained_clip,
        freeze_vision_layers=args.freeze_vision_layers,
        freeze_text_encoder=args.freeze_text_encoder,
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    criterion = nn.MSELoss()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_path = output_dir / "best_clip_alignment.pt"

    logger.info("=" * 80)
    logger.info("Training Strategy: Align Image Features with Stylized Text Features")
    logger.info("Loss: MSE(image_features, text_features) in CLIP semantic space")
    logger.info("Key Idea: Make image features closer to stylized/emotional text features")
    logger.info("Inference: Image features → VAD Predictor → VAD")
    logger.info("=" * 80)

    for epoch in range(args.epochs):
        model.train()
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        epoch_losses = []
        epoch_cosine_sims = []
        
        for batch in progress:
            optimizer.zero_grad()

            image_paths = batch["image_path"]
            texts = batch["text"]  # Stylized captions
            images = load_images(image_paths)

            # Process images and texts together
            inputs = processor(
                text=texts,
                images=images,
                padding="max_length",
                truncation=True,
                max_length=effective_max_length,
                return_tensors="pt",
            )
            pixel_values = inputs["pixel_values"].to(device)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            # Forward: get image and text features
            image_features, text_features = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Loss: MSE between image and text features (align in semantic space)
            loss = criterion(image_features, text_features)
            loss.backward()

            if args.gradient_clip and args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)

            optimizer.step()
            scheduler.step()
            
            # Compute cosine similarity for monitoring
            with torch.no_grad():
                image_norm = torch.nn.functional.normalize(image_features, p=2, dim=1)
                text_norm = torch.nn.functional.normalize(text_features, p=2, dim=1)
                cosine_sim = (image_norm * text_norm).sum(dim=1).mean().item()
                epoch_cosine_sims.append(cosine_sim)
            
            epoch_losses.append(loss.item())

            progress.set_postfix(
                {
                    "loss": loss.item(),
                    "cosine": f"{cosine_sim:.3f}",
                    "lr": scheduler.get_last_lr()[0],
                }
            )

        # Log epoch statistics
        mean_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        mean_epoch_cosine = sum(epoch_cosine_sims) / len(epoch_cosine_sims)
        logger.info(
            "Epoch %d training - loss: %.4f, cosine_sim: %.4f",
            epoch + 1,
            mean_epoch_loss,
            mean_epoch_cosine,
        )

        # Validation
        val_metrics = (
            evaluate(model, processor, val_loader, device, criterion, effective_max_length)
            if val_loader is not None
            else {"loss": 0.0, "cosine_sim": 0.0}
        )
        logger.info(
            "Epoch %d validation - loss: %.4f, cosine_sim: %.4f",
            epoch + 1,
            val_metrics["loss"],
            val_metrics["cosine_sim"],
        )
        if val_loader is not None and val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), best_path)
            logger.info("Saved new best model to %s", best_path)

    final_path = output_dir / "last_clip_alignment.pt"
    torch.save(model.state_dict(), final_path)
    logger.info("Saved final model to %s", final_path)


if __name__ == "__main__":
    train()

