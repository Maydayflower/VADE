import argparse
import logging
from pathlib import Path
from typing import List

import os
import sys

import torch
from PIL import Image
from transformers import CLIPProcessor, BertTokenizer

try:
    from .train_text_vad import ClipImageTextAlignmentModel
    from .vad_predictor import VADPredictor
except ImportError:  # pragma: no cover - fallback for direct script execution
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    if CURRENT_DIR not in sys.path:
        sys.path.append(CURRENT_DIR)
    from train_text_vad import ClipImageTextAlignmentModel  # type: ignore
    from vad_predictor import VADPredictor  # type: ignore

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CLIP-based image-to-VAD inference (image features → VAD Predictor → VAD).")
    parser.add_argument("--alignment_checkpoint", type=str, required=True, help="Fine-tuned CLIP alignment model checkpoint (.pt)")
    parser.add_argument("--vad_predictor_path", type=str, default="/workspace/compare_model/bart/clip/vad_predictor.pt", help="Path to VAD predictor model")
    parser.add_argument("--vad_bert_path", type=str, default="/workspace/models/bert-base-uncased", help="Path to BERT model for VAD predictor")
    parser.add_argument("--clip_path", type=str, default="/workspace/models/clip-vit-base-patch32")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Optional: save results to file (TSV format: image_path<TAB>v<TAB>a<TAB>d)"
    )
    return parser.parse_args()


def load_image(path: str) -> Image.Image:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Image not found at {path}")
    return Image.open(source).convert("RGB")




def run(image_path: str) -> tuple[float, float, float]:
    """
    Run inference with a single image (no text required).
    
    Pipeline:
        1. Image → CLIP Image Encoder → image_features (aligned with stylized text)
        2. image_features → VAD Predictor → VAD
    
    Args:
        image_path: Single image path
        
    Returns:
        (v, a, d): VAD values predicted from the image
    """
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    logger.info("Using device: %s", device)

    # Validate input
    image_path_obj = Path(image_path)
    if not image_path_obj.exists():
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    logger.info(f"Processing image: {image_path}")

    # Load CLIP alignment model
    logger.info("Loading CLIP alignment model from %s", args.clip_path)
    processor = CLIPProcessor.from_pretrained(args.clip_path)
    clip_model = ClipImageTextAlignmentModel(args.clip_path)
    
    logger.info("Loading alignment checkpoint from %s", args.alignment_checkpoint)
    checkpoint = torch.load(args.alignment_checkpoint, map_location="cpu")
    clip_model.load_state_dict(checkpoint)
    clip_model.to(device)
    clip_model.eval()
    logger.info("CLIP alignment model loaded successfully")

    # Load VAD Predictor
    logger.info("Loading VAD Predictor from %s", args.vad_predictor_path)
    vad_tokenizer = BertTokenizer.from_pretrained(args.vad_bert_path)
    vad_predictor = VADPredictor(bert_model_name=args.vad_bert_path)
    vad_predictor.load_state_dict(torch.load(args.vad_predictor_path, map_location="cpu"))
    vad_predictor.to(device)
    vad_predictor.eval()
    logger.info("VAD Predictor loaded successfully")

    # Load image
    try:
        image = load_image(image_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load image {image_path}: {e}")
    
    # Step 1: Image → CLIP → image_features (aligned with stylized text features)
    inputs = processor(
        images=[image],
        return_tensors="pt",
    )
    pixel_values = inputs["pixel_values"].to(device)
    
    with torch.no_grad():
        # Get image features from aligned CLIP model
        image_features = clip_model.get_image_features(pixel_values=pixel_values)  # [1, 512]
        
        # Step 2: image_features → VAD Predictor → VAD
        # VAD Predictor expects BERT-style input, but we use image features directly
        # We need to adapt the features
        # Option: Use image features as if they were text embeddings
        # This requires that VAD Predictor accepts feature input
        
        # For now, we use a workaround: treat image features as pooled BERT output
        # VAD Predictor forward: input_ids, attention_mask → BERT → pooled → VAD
        # We'll directly use the pooled-like features
        
        # Since VAD Predictor uses BERT internally, we need to bypass BERT
        # Let's directly use the VAD predictor's regression head
        vad_output = vad_predictor(image_features)  # [1, 3]
    
    # Extract results
    v, a, d = vad_output[0].cpu().tolist()
    
    # Print results
    logger.info("=" * 80)
    logger.info(f"Image: {image_path}")
    logger.info(f"VAD Prediction: V={v:.3f}, A={a:.3f}, D={d:.3f}")
    logger.info("=" * 80)
    logger.info("Pipeline: Image → CLIP (aligned) → image_features → VAD Predictor → VAD")
    
    # Save results to file if specified
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("image_path\tv\ta\td\n")
            f.write(f"{image_path}\t{v:.6f}\t{a:.6f}\t{d:.6f}\n")
        logger.info(f"Saved results to {output_path}")
    
    logger.info("Inference completed successfully")
    return v, a, d


if __name__ == "__main__":
    # ============================================================================
    # Input Data Configuration
    # ============================================================================
    # Configure your image input here (no text needed!)
    
    image_path = "/workspace/compare_model/bart/clip/dataset/coco2017/val2017/000000268378.jpg"
    
    # ============================================================================
    # Run Inference
    # ============================================================================
    # Training: Image features learn to predict VAD from stylized captions
    # Inference: Image -> VAD (leveraging CLIP's shared semantic space)
    run(image_path=image_path)

