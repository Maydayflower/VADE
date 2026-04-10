#!/usr/bin/env python3
"""
训练下游VAD预测器

Pipeline:
    1. 使用对齐好的CLIP模型提取图像特征（冻结）
    2. 使用VAD Predictor从风格化caption生成VAD标签
    3. 训练MLP: 图像特征 → VAD

架构:
    Image → CLIP (frozen) → features [512] → MLP → VAD [3]
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import CLIPProcessor, BertTokenizer, get_cosine_schedule_with_warmup

# 添加父目录到Python路径
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# 导入其他目录的模块
from stage1_alignment.train_text_vad import ClipImageTextAlignmentModel, SimpleImageTextDataset, load_images
from core.vad_data import load_image_text_file
from core.vad_predictor import VADPredictor

logger = logging.getLogger(__name__)


class ImageFeatureVADModel(nn.Module):
    """Simple MLP that predicts VAD from CLIP image features.
    
    Input: CLIP image features [batch_size, 512]
    Output: VAD predictions [batch_size, 3]
    """
    
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3),  # VAD: [V, A, D]
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: CLIP image features [batch_size, 512]
        Returns:
            vad: VAD predictions [batch_size, 3]
        """
        return self.mlp(features)


class ClipVADModel(nn.Module):
    """End-to-end model: CLIP + VAD Predictor.
    
    可以选择性地训练CLIP或仅训练VAD预测头。
    """
    
    def __init__(
        self, 
        clip_model: ClipImageTextAlignmentModel,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        freeze_clip: bool = False,
    ):
        super().__init__()
        self.clip_model = clip_model
        self.vad_predictor = ImageFeatureVADModel(
            input_dim=512,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        
        # 控制CLIP是否冻结
        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False
            logger.info("CLIP model frozen")
        else:
            for param in self.clip_model.parameters():
                param.requires_grad = True
            logger.info("CLIP model trainable")
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: Images [batch_size, 3, H, W]
        Returns:
            vad: VAD predictions [batch_size, 3]
        """
        # Extract features
        features = self.clip_model.get_image_features(pixel_values)
        # Predict VAD
        vad = self.vad_predictor(features)
        return vad


class ImageFeatureVADDataset(Dataset):
    """Dataset that returns image path, pre-computed CLIP features, and VAD labels."""
    
    def __init__(self, examples, clip_features: torch.Tensor, vad_labels: torch.Tensor):
        """
        Args:
            examples: List of ImageTextExample
            clip_features: Pre-computed CLIP features [N, 512]
            vad_labels: Pre-computed VAD labels [N, 3]
        """
        self.examples = list(examples)
        self.clip_features = clip_features
        self.vad_labels = vad_labels
        assert len(self.examples) == len(self.clip_features) == len(self.vad_labels)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return {
            "image_path": self.examples[idx].image_path,
            "features": self.clip_features[idx],
            "vad": self.vad_labels[idx],
        }


class ImageVADDataset(Dataset):
    """Dataset that returns image path and VAD labels (for end-to-end training)."""
    
    def __init__(self, examples, vad_labels: torch.Tensor):
        """
        Args:
            examples: List of ImageTextExample
            vad_labels: Pre-computed VAD labels [N, 3]
        """
        self.examples = list(examples)
        self.vad_labels = vad_labels
        assert len(self.examples) == len(self.vad_labels)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return {
            "image_path": self.examples[idx].image_path,
            "vad": self.vad_labels[idx],
        }


def collate_fn(batch):
    """Collate function for pre-computed features."""
    features = torch.stack([item["features"] for item in batch])
    vads = torch.stack([item["vad"] for item in batch])
    image_paths = [item["image_path"] for item in batch]
    
    return {
        "image_path": image_paths,
        "features": features,
        "vad": vads,
    }


def collate_fn_endtoend(batch):
    """Collate function for end-to-end training."""
    vads = torch.stack([item["vad"] for item in batch])
    image_paths = [item["image_path"] for item in batch]
    
    return {
        "image_path": image_paths,
        "vad": vads,
    }


def extract_clip_features(
    clip_model: ClipImageTextAlignmentModel,
    processor: CLIPProcessor,
    examples,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Extract CLIP image features for all examples.
    
    Returns:
        features: [N, 512] tensor of CLIP features
    """
    logger.info("Extracting CLIP features from %d images...", len(examples))
    clip_model.eval()
    all_features = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(examples), batch_size), desc="Extracting CLIP features"):
            batch_examples = examples[i:i + batch_size]
            image_paths = [ex.image_path for ex in batch_examples]
            
            # Load images
            images = load_images(image_paths)
            
            # Process images
            inputs = processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            
            # Extract features
            features = clip_model.get_image_features(pixel_values=pixel_values)
            all_features.append(features.cpu())
    
    all_features = torch.cat(all_features, dim=0)
    logger.info("Extracted CLIP features: %s", all_features.shape)
    return all_features


def generate_vad_labels(
    vad_predictor: VADPredictor,
    vad_tokenizer: BertTokenizer,
    examples,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate VAD labels from stylized captions using VAD Predictor.
    
    Returns:
        vad_labels: [N, 3] tensor of VAD labels
    """
    logger.info("Generating VAD labels from %d stylized captions...", len(examples))
    vad_predictor.eval()
    all_vads = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(examples), batch_size), desc="Generating VAD labels"):
            batch_examples = examples[i:i + batch_size]
            texts = [ex.sentence for ex in batch_examples]  # Stylized captions
            
            # Tokenize
            encoded = vad_tokenizer(
                texts,
                padding='max_length',
                truncation=True,
                max_length=77,
                return_tensors='pt'
            )
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            
            # Predict VAD
            vad = vad_predictor(input_ids, attention_mask)
            all_vads.append(vad.cpu())
    
    all_vads = torch.cat(all_vads, dim=0)
    logger.info("Generated VAD labels: %s", all_vads.shape)
    return all_vads


def evaluate(
    model,
    dataloader: DataLoader,
    device: torch.device,
    criterion,
    processor=None,
    is_endtoend=False,
) -> Dict[str, float]:
    """Evaluate the model.
    
    Args:
        model: VAD prediction model
        dataloader: DataLoader
        device: torch device
        criterion: Loss function
        processor: CLIP processor (required if is_endtoend=True)
        is_endtoend: Whether using end-to-end mode (loads images) or feature mode
    """
    model.eval()
    losses = []
    
    with torch.no_grad():
        for batch in dataloader:
            vad_targets = batch["vad"].to(device)
            
            if is_endtoend:
                # End-to-end: 实时加载并处理图像
                image_paths = batch["image_path"]
                images = []
                for img_path in image_paths:
                    img = Image.open(img_path).convert("RGB")
                    images.append(img)
                inputs = processor(images=images, return_tensors="pt", padding=True)
                pixel_values = inputs["pixel_values"].to(device)
                vad_pred = model(pixel_values)
            else:
                # Feature extraction: 使用预计算的特征
                features = batch["features"].to(device)
                vad_pred = model(features)
            
            # Loss
            loss = criterion(vad_pred, vad_targets)
            losses.append(loss.item())
    
    return {"loss": float(np.mean(losses))}


def parse_args():
    parser = argparse.ArgumentParser(description="Train downstream VAD predictor from CLIP features")
    parser.add_argument("--train_file", type=str, required=True, help="Training TSV file")
    parser.add_argument("--val_file", type=str, default=None, help="Validation TSV file")
    parser.add_argument("--clip_checkpoint", type=str, default=None, help="Path to aligned CLIP checkpoint (None=use original CLIP)")
    parser.add_argument("--clip_path", type=str, default="/workspace/models/clip-vit-base-patch32")
    parser.add_argument("--vad_predictor_path", type=str, default="/workspace/compare_model/bart/clip/vad_predictor.pt")
    parser.add_argument("--vad_bert_path", type=str, default="/workspace/models/bert-base-uncased")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--feature_batch_size", type=int, default=64, help="Batch size for feature extraction")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for VAD MLP")
    parser.add_argument("--clip_learning_rate", type=float, default=1e-5, help="Learning rate for CLIP (if trainable)")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gradient_clip", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    # 新增参数：控制训练模式
    parser.add_argument("--train_clip", action="store_true", help="Train CLIP model together with VAD predictor (end-to-end)")
    parser.add_argument("--freeze_clip_layers", type=int, default=0, help="Number of CLIP vision layers to freeze (0=train all)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logger.info("Arguments: %s", vars(args))
    
    # Set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # Device
    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    logger.info("Using device: %s", device)
    
    # Load data
    logger.info("Loading data...")
    train_examples = load_image_text_file(args.train_file)
    val_examples = load_image_text_file(args.val_file) if args.val_file else None
    logger.info("Loaded %d training examples", len(train_examples))
    if val_examples:
        logger.info("Loaded %d validation examples", len(val_examples))
    
    # Load CLIP model
    processor = CLIPProcessor.from_pretrained(args.clip_path)
    clip_model = ClipImageTextAlignmentModel(args.clip_path)
    
    if args.clip_checkpoint:
        logger.info("Loading aligned CLIP checkpoint from %s", args.clip_checkpoint)
        clip_model.load_state_dict(torch.load(args.clip_checkpoint, map_location="cpu"))
        logger.info("Using aligned CLIP model")
    else:
        logger.info("Using original CLIP model (no alignment checkpoint)")
    
    clip_model.to(device)
    
    # 根据训练模式决定是否冻结CLIP
    if not args.train_clip:
        clip_model.eval()
        for param in clip_model.parameters():
            param.requires_grad = False
        logger.info("CLIP model loaded and frozen (feature extraction mode)")
    else:
        clip_model.train()
        # 可选：冻结部分层
        if args.freeze_clip_layers > 0:
            vision_encoder = clip_model.clip.vision_model.encoder
            for layer in vision_encoder.layers[:args.freeze_clip_layers]:
                for param in layer.parameters():
                    param.requires_grad = False
            logger.info(f"CLIP model loaded, frozen first {args.freeze_clip_layers} layers (end-to-end mode)")
        else:
            for param in clip_model.parameters():
                param.requires_grad = True
            logger.info("CLIP model loaded and trainable (end-to-end mode)")
    
    # Load VAD Predictor
    logger.info("Loading VAD Predictor from %s", args.vad_predictor_path)
    vad_tokenizer = BertTokenizer.from_pretrained(args.vad_bert_path)
    vad_predictor = VADPredictor(bert_model_name=args.vad_bert_path)
    vad_predictor.load_state_dict(torch.load(args.vad_predictor_path, map_location="cpu"))
    vad_predictor.to(device)
    vad_predictor.eval()
    for param in vad_predictor.parameters():
        param.requires_grad = False
    logger.info("VAD Predictor loaded and frozen")
    
    # Generate VAD labels from stylized captions  
    logger.info("=" * 80)
    logger.info("Generating VAD labels from stylized captions")
    logger.info("=" * 80)
    train_vads = generate_vad_labels(vad_predictor, vad_tokenizer, train_examples, args.feature_batch_size, device)
    val_vads = generate_vad_labels(vad_predictor, vad_tokenizer, val_examples, args.feature_batch_size, device) if val_examples else None
    
    # Print VAD statistics
    logger.info("=" * 80)
    logger.info("VAD Label Statistics")
    logger.info("=" * 80)
    logger.info("Training VAD - mean: V=%.3f, A=%.3f, D=%.3f", 
                train_vads[:, 0].mean(), train_vads[:, 1].mean(), train_vads[:, 2].mean())
    logger.info("Training VAD - std:  V=%.3f, A=%.3f, D=%.3f", 
                train_vads[:, 0].std(), train_vads[:, 1].std(), train_vads[:, 2].std())
    if val_vads is not None:
        logger.info("Validation VAD - mean: V=%.3f, A=%.3f, D=%.3f", 
                    val_vads[:, 0].mean(), val_vads[:, 1].mean(), val_vads[:, 2].mean())
    
    # 根据训练模式创建不同的数据集和模型
    if args.train_clip:
        # End-to-end模式：训练CLIP + VAD预测器
        logger.info("=" * 80)
        logger.info("Training Mode: End-to-End (CLIP + VAD Predictor)")
        logger.info("=" * 80)
        
        # 创建端到端数据集（不预计算特征）
        train_dataset = ImageVADDataset(train_examples, train_vads)
        val_dataset = ImageVADDataset(val_examples, val_vads) if val_examples else None
        
        # 创建dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn_endtoend,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn_endtoend,
        ) if val_dataset else None
        
        # 创建端到端模型
        model = ClipVADModel(
            clip_model=clip_model,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            freeze_clip=False,
        )
        model.to(device)
        
        # 统计参数
        clip_params = sum(p.numel() for p in model.clip_model.parameters() if p.requires_grad)
        mlp_params = sum(p.numel() for p in model.vad_predictor.parameters())
        logger.info("Model parameters: CLIP=%d, MLP=%d, Total=%d", clip_params, mlp_params, clip_params + mlp_params)
        
    else:
        # Feature extraction模式：只训练MLP
        logger.info("=" * 80)
        logger.info("Training Mode: Feature Extraction (Frozen CLIP + Train MLP)")
        logger.info("=" * 80)
        
        # 预计算CLIP特征
        logger.info("Extracting CLIP features...")
        train_features = extract_clip_features(clip_model, processor, train_examples, args.feature_batch_size, device)
        val_features = extract_clip_features(clip_model, processor, val_examples, args.feature_batch_size, device) if val_examples else None
        
        # 创建特征数据集
        train_dataset = ImageFeatureVADDataset(train_examples, train_features, train_vads)
        val_dataset = ImageFeatureVADDataset(val_examples, val_features, val_vads) if val_examples else None
        
        # 创建dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        ) if val_dataset else None
        
        # 创建MLP模型
        model = ImageFeatureVADModel(
            input_dim=512,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        )
        model.to(device)
        logger.info("Model parameters: %d", sum(p.numel() for p in model.parameters()))
    
    # Optimizer and scheduler
    if args.train_clip:
        # 为CLIP和MLP使用不同的学习率
        optimizer = AdamW([
            {'params': model.clip_model.parameters(), 'lr': args.clip_learning_rate},
            {'params': model.vad_predictor.parameters(), 'lr': args.learning_rate},
        ], weight_decay=args.weight_decay)
        logger.info("Optimizer: CLIP lr=%.2e, MLP lr=%.2e", args.clip_learning_rate, args.learning_rate)
    else:
        optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        logger.info("Optimizer: MLP lr=%.2e", args.learning_rate)
    
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    criterion = nn.MSELoss()
    
    # Training loop
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_loss = float("inf")
    best_path = output_dir / "best_vad_downstream.pt"
    
    for epoch in range(args.epochs):
        model.train()
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        epoch_losses = []
        
        for batch in progress:
            optimizer.zero_grad()
            
            vad_targets = batch["vad"].to(device)
            
            # 根据训练模式处理输入
            if args.train_clip:
                # End-to-end: 实时加载并处理图像
                image_paths = batch["image_path"]
                images = []
                for img_path in image_paths:
                    img = Image.open(img_path).convert("RGB")
                    images.append(img)
                inputs = processor(images=images, return_tensors="pt", padding=True)
                pixel_values = inputs["pixel_values"].to(device)
                vad_pred = model(pixel_values)
            else:
                # Feature extraction: 使用预计算的特征
                features = batch["features"].to(device)
                vad_pred = model(features)
            
            # Loss
            loss = criterion(vad_pred, vad_targets)
            loss.backward()
            
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            
            optimizer.step()
            scheduler.step()
            
            epoch_losses.append(loss.item())
            progress.set_postfix({"loss": loss.item(), "lr": scheduler.get_last_lr()[0]})
        
        # Log epoch statistics
        mean_epoch_loss = np.mean(epoch_losses)
        logger.info("Epoch %d training loss: %.4f", epoch + 1, mean_epoch_loss)
        
        # Validation
        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, device, criterion, processor if args.train_clip else None, args.train_clip)
            logger.info("Epoch %d validation loss: %.4f", epoch + 1, val_metrics["loss"])
            
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                torch.save(model.state_dict(), best_path)
                logger.info("Saved new best model to %s", best_path)
    
    # Save final model
    final_path = output_dir / "last_vad_downstream.pt"
    torch.save(model.state_dict(), final_path)
    logger.info("Saved final model to %s", final_path)
    
    logger.info("=" * 80)
    logger.info("Training completed!")
    logger.info("Best validation loss: %.4f", best_val_loss)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

