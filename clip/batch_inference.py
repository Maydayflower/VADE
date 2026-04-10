#!/usr/bin/env python3
"""
批量图像VAD推理脚本

用法：
    python batch_inference.py \
        --image_dir /path/to/images \
        --checkpoint output/image_vad_model/best_image_vad.pt \
        --output results.tsv
"""

import argparse
import logging
from pathlib import Path
from typing import List
import os
import sys

import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor

try:
    from .train_text_vad import ClipImageVADModel
except ImportError:
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    if CURRENT_DIR not in sys.path:
        sys.path.append(CURRENT_DIR)
    from train_text_vad import ClipImageVADModel

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="批量图像VAD推理")
    parser.add_argument("--image_dir", type=str, required=True, help="图像目录路径")
    parser.add_argument("--image_list", type=str, default=None, help="可选：图像路径列表文件（每行一个路径）")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型checkpoint路径")
    parser.add_argument("--clip_path", type=str, default="/workspace/models/clip-vit-base-patch32")
    parser.add_argument("--output", type=str, required=True, help="输出TSV文件路径")
    parser.add_argument("--batch_size", type=int, default=32, help="批处理大小")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--image_extensions", type=str, default="jpg,jpeg,png,bmp", help="图像扩展名（逗号分隔）")
    return parser.parse_args()


def load_image(path: str) -> Image.Image:
    """加载单张图像"""
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        logger.warning(f"Failed to load image {path}: {e}")
        return None


def get_image_paths(image_dir: str, image_list: str = None, extensions: List[str] = None) -> List[str]:
    """获取所有图像路径"""
    if extensions is None:
        extensions = ["jpg", "jpeg", "png", "bmp"]
    
    image_paths = []
    
    if image_list:
        # 从列表文件读取
        with open(image_list, 'r', encoding='utf-8') as f:
            for line in f:
                path = line.strip()
                if path and Path(path).exists():
                    image_paths.append(path)
    else:
        # 扫描目录
        image_dir_path = Path(image_dir)
        for ext in extensions:
            image_paths.extend(list(image_dir_path.glob(f"**/*.{ext}")))
            image_paths.extend(list(image_dir_path.glob(f"**/*.{ext.upper()}")))
        
        image_paths = [str(p) for p in image_paths]
    
    return sorted(image_paths)


def batch_inference(
    model: ClipImageVADModel,
    processor: CLIPProcessor,
    image_paths: List[str],
    batch_size: int,
    device: torch.device,
) -> List[tuple]:
    """批量推理
    
    Returns:
        List of (image_path, v, a, d) tuples
    """
    results = []
    model.eval()
    
    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
            batch_paths = image_paths[i:i + batch_size]
            
            # 加载图像
            images = []
            valid_paths = []
            for path in batch_paths:
                img = load_image(path)
                if img is not None:
                    images.append(img)
                    valid_paths.append(path)
            
            if not images:
                continue
            
            # 处理图像
            inputs = processor(
                images=images,
                return_tensors="pt",
            )
            pixel_values = inputs["pixel_values"].to(device)
            
            # 预测VAD
            vads = model(pixel_values=pixel_values)
            
            # 保存结果
            for path, vad in zip(valid_paths, vads):
                v, a, d = vad.cpu().tolist()
                results.append((path, v, a, d))
    
    return results


def main():
    args = parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    
    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    logger.info(f"Using device: {device}")
    
    # 加载模型
    logger.info(f"Loading CLIP model from {args.clip_path}")
    processor = CLIPProcessor.from_pretrained(args.clip_path)
    model = ClipImageVADModel(args.clip_path)
    
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    logger.info("Model loaded successfully")
    
    # 获取图像路径
    extensions = args.image_extensions.split(',')
    image_paths = get_image_paths(args.image_dir, args.image_list, extensions)
    logger.info(f"Found {len(image_paths)} images")
    
    if not image_paths:
        logger.error("No images found!")
        return
    
    # 批量推理
    logger.info("Starting batch inference...")
    results = batch_inference(model, processor, image_paths, args.batch_size, device)
    logger.info(f"Processed {len(results)} images successfully")
    
    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("image_path\tv\ta\td\n")
        for path, v, a, d in results:
            f.write(f"{path}\t{v:.6f}\t{a:.6f}\t{d:.6f}\n")
    
    logger.info(f"Results saved to {output_path}")
    
    # 统计信息
    if results:
        vs = [r[1] for r in results]
        a_s = [r[2] for r in results]
        ds = [r[3] for r in results]
        
        logger.info("=" * 60)
        logger.info("Statistics:")
        logger.info(f"  Valence:   mean={sum(vs)/len(vs):.3f}, min={min(vs):.3f}, max={max(vs):.3f}")
        logger.info(f"  Arousal:   mean={sum(a_s)/len(a_s):.3f}, min={min(a_s):.3f}, max={max(a_s):.3f}")
        logger.info(f"  Dominance: mean={sum(ds)/len(ds):.3f}, min={min(ds):.3f}, max={max(ds):.3f}")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()

