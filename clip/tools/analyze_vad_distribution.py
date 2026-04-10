#!/usr/bin/env python3
"""
分析数据集中文本的VAD向量分布

使用VAD Predictor生成所有文本的VAD向量，然后分析其分布特征：
- 统计量（均值、标准差、最小值、最大值、分位数）
- 相关性分析
- 分布可视化（直方图、散点图）
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
from transformers import BertTokenizer
from tqdm.auto import tqdm
from collections import defaultdict

# 添加父目录到Python路径
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from core.vad_data import load_image_text_file
from core.vad_predictor import VADPredictor

logger = logging.getLogger(__name__)


def generate_vad_for_texts(
    texts: List[str],
    vad_predictor: VADPredictor,
    tokenizer: BertTokenizer,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """为文本列表生成VAD向量
    
    Args:
        texts: 文本列表
        vad_predictor: VAD预测器模型
        tokenizer: BERT tokenizer
        batch_size: 批处理大小
        device: 设备
        
    Returns:
        VAD向量 [N, 3]
    """
    vad_predictor.eval()
    all_vads = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating VAD vectors"):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            
            # Predict
            vad_output = vad_predictor(input_ids, attention_mask)
            all_vads.append(vad_output.cpu())
    
    all_vads = torch.cat(all_vads, dim=0).numpy()
    return all_vads


def normalize_vad_to_range(vads: np.ndarray, target_range: tuple = (-1, 1)) -> tuple:
    """将VAD向量归一化到指定范围
    
    Args:
        vads: VAD向量 [N, 3]
        target_range: 目标范围，默认(-1, 1)
        
    Returns:
        (normalized_vads, scaling_params): 归一化后的VAD向量和缩放参数
    """
    normalized_vads = np.zeros_like(vads)
    scaling_params = []
    
    target_min, target_max = target_range
    dimension_names = ['Valence', 'Arousal', 'Dominance']
    
    for i, name in enumerate(dimension_names):
        values = vads[:, i]
        
        # 记录原始范围
        original_min = float(np.min(values))
        original_max = float(np.max(values))
        
        # Min-Max归一化到目标范围
        # formula: (x - min) / (max - min) * (target_max - target_min) + target_min
        if original_max - original_min > 1e-8:  # 避免除零
            normalized_values = (values - original_min) / (original_max - original_min)
            normalized_values = normalized_values * (target_max - target_min) + target_min
        else:
            # 如果范围太小，直接设为范围中点
            normalized_values = np.full_like(values, (target_min + target_max) / 2)
        
        normalized_vads[:, i] = normalized_values
        
        scaling_params.append({
            'dimension': name,
            'original_min': original_min,
            'original_max': original_max,
            'target_min': target_min,
            'target_max': target_max,
        })
    
    return normalized_vads, scaling_params


def compute_statistics(vads: np.ndarray) -> dict:
    """计算VAD向量的统计信息
    
    Args:
        vads: VAD向量 [N, 3]
        
    Returns:
        统计信息字典
    """
    stats = {}
    dimension_names = ['Valence', 'Arousal', 'Dominance']
    
    for i, name in enumerate(dimension_names):
        values = vads[:, i]
        stats[name] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'q25': float(np.percentile(values, 25)),
            'q50': float(np.percentile(values, 50)),  # median
            'q75': float(np.percentile(values, 75)),
        }
    
    # 计算相关性
    corr_matrix = np.corrcoef(vads.T)
    stats['correlations'] = {
        'V-A': float(corr_matrix[0, 1]),
        'V-D': float(corr_matrix[0, 2]),
        'A-D': float(corr_matrix[1, 2]),
    }
    
    return stats


def print_statistics(stats: dict):
    """打印统计信息"""
    logger.info("=" * 80)
    logger.info("VAD向量分布统计")
    logger.info("=" * 80)
    
    for dim in ['Valence', 'Arousal', 'Dominance']:
        dim_stats = stats[dim]
        logger.info(f"\n{dim}:")
        logger.info(f"  均值:    {dim_stats['mean']:.4f}")
        logger.info(f"  标准差:  {dim_stats['std']:.4f}")
        logger.info(f"  最小值:  {dim_stats['min']:.4f}")
        logger.info(f"  最大值:  {dim_stats['max']:.4f}")
        logger.info(f"  25分位: {dim_stats['q25']:.4f}")
        logger.info(f"  50分位: {dim_stats['q50']:.4f} (中位数)")
        logger.info(f"  75分位: {dim_stats['q75']:.4f}")
    
    logger.info("\n相关性:")
    corr = stats['correlations']
    logger.info(f"  Valence-Arousal:    {corr['V-A']:+.4f}")
    logger.info(f"  Valence-Dominance:  {corr['V-D']:+.4f}")
    logger.info(f"  Arousal-Dominance:  {corr['A-D']:+.4f}")
    logger.info("=" * 80)


def save_statistics(stats: dict, output_file: Path, scaling_params=None):
    """保存统计信息到文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("VAD向量分布统计报告\n")
        f.write("=" * 80 + "\n\n")
        
        # 如果有归一化信息，先显示
        if scaling_params:
            f.write("数据归一化信息:\n")
            for param in scaling_params:
                f.write(f"{param['dimension']}:\n")
                f.write(f"  原始范围: [{param['original_min']:.6f}, {param['original_max']:.6f}]\n")
                f.write(f"  归一化后: [{param['target_min']:.6f}, {param['target_max']:.6f}]\n")
            f.write("\n" + "=" * 80 + "\n\n")
        
        for dim in ['Valence', 'Arousal', 'Dominance']:
            dim_stats = stats[dim]
            f.write(f"{dim}:\n")
            f.write(f"  均值:    {dim_stats['mean']:.6f}\n")
            f.write(f"  标准差:  {dim_stats['std']:.6f}\n")
            f.write(f"  最小值:  {dim_stats['min']:.6f}\n")
            f.write(f"  最大值:  {dim_stats['max']:.6f}\n")
            f.write(f"  25分位: {dim_stats['q25']:.6f}\n")
            f.write(f"  50分位: {dim_stats['q50']:.6f}\n")
            f.write(f"  75分位: {dim_stats['q75']:.6f}\n")
            f.write("\n")
        
        f.write("相关性:\n")
        corr = stats['correlations']
        f.write(f"  Valence-Arousal:    {corr['V-A']:+.6f}\n")
        f.write(f"  Valence-Dominance:  {corr['V-D']:+.6f}\n")
        f.write(f"  Arousal-Dominance:  {corr['A-D']:+.6f}\n")
        f.write("=" * 80 + "\n")


def save_vad_vectors(texts: List[str], vads: np.ndarray, output_file: Path):
    """保存文本和对应的VAD向量"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("text\tvalence\tarousal\tdominance\n")
        for text, vad in zip(texts, vads):
            v, a, d = vad
            # 转义制表符和换行符
            text_clean = text.replace('\t', ' ').replace('\n', ' ')
            f.write(f"{text_clean}\t{v:.6f}\t{a:.6f}\t{d:.6f}\n")


def create_visualizations(vads: np.ndarray, output_dir: Path):
    """创建可视化图表
    
    Args:
        vads: VAD向量 [N, 3]
        output_dir: 输出目录
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # 非交互式后端
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        dimension_names = ['Valence', 'Arousal', 'Dominance']
        
        # 1. 直方图 - 每个维度的分布
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for i, (ax, name) in enumerate(zip(axes, dimension_names)):
            ax.hist(vads[:, i], bins=50, alpha=0.7, edgecolor='black')
            ax.set_xlabel(name)
            ax.set_ylabel('Frequency')
            ax.set_title(f'{name} Distribution')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        hist_path = output_dir / 'vad_histograms.png'
        plt.savefig(hist_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"已保存直方图: {hist_path}")
        
        # 2. 散点图矩阵
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # V-A
        axes[0, 0].scatter(vads[:, 0], vads[:, 1], alpha=0.3, s=10)
        axes[0, 0].set_xlabel('Valence')
        axes[0, 0].set_ylabel('Arousal')
        axes[0, 0].set_title('Valence vs Arousal')
        axes[0, 0].grid(True, alpha=0.3)
        
        # V-D
        axes[0, 1].scatter(vads[:, 0], vads[:, 2], alpha=0.3, s=10)
        axes[0, 1].set_xlabel('Valence')
        axes[0, 1].set_ylabel('Dominance')
        axes[0, 1].set_title('Valence vs Dominance')
        axes[0, 1].grid(True, alpha=0.3)
        
        # A-D
        axes[0, 2].scatter(vads[:, 1], vads[:, 2], alpha=0.3, s=10)
        axes[0, 2].set_xlabel('Arousal')
        axes[0, 2].set_ylabel('Dominance')
        axes[0, 2].set_title('Arousal vs Dominance')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 箱线图
        for i, (ax, name) in enumerate(zip(axes[1, :], dimension_names)):
            ax.boxplot([vads[:, i]], labels=[name])
            ax.set_ylabel('Value')
            ax.set_title(f'{name} Box Plot')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        scatter_path = output_dir / 'vad_scatter_matrix.png'
        plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"已保存散点图矩阵: {scatter_path}")
        
        # 3. 热力图（2D密度）
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        pairs = [(0, 1, 'Valence', 'Arousal'), 
                 (0, 2, 'Valence', 'Dominance'),
                 (1, 2, 'Arousal', 'Dominance')]
        
        for ax, (i, j, name_i, name_j) in zip(axes, pairs):
            # 2D直方图
            h = ax.hist2d(vads[:, i], vads[:, j], bins=50, cmap='YlOrRd')
            ax.set_xlabel(name_i)
            ax.set_ylabel(name_j)
            ax.set_title(f'{name_i} vs {name_j} Density')
            plt.colorbar(h[3], ax=ax)
        
        plt.tight_layout()
        density_path = output_dir / 'vad_density.png'
        plt.savefig(density_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"已保存密度图: {density_path}")
        
        logger.info("所有可视化图表已生成")
        
    except ImportError as e:
        logger.warning(f"无法生成可视化图表，缺少依赖: {e}")
        logger.warning("请安装: pip install matplotlib seaborn")


def parse_args():
    parser = argparse.ArgumentParser(description="分析数据集中文本的VAD向量分布")
    parser.add_argument("--data_file", type=str, required=True, help="数据文件路径（TSV格式）")
    parser.add_argument("--vad_predictor_path", type=str, 
                        default="/workspace/compare_model/bart/clip/vad_predictor.pt",
                        help="VAD预测器模型路径")
    parser.add_argument("--vad_bert_path", type=str,
                        default="/workspace/models/bert-base-uncased",
                        help="BERT模型路径")
    parser.add_argument("--batch_size", type=int, default=64, help="批处理大小")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save_vectors", action="store_true", help="保存所有文本的VAD向量")
    parser.add_argument("--visualize", action="store_true", help="生成可视化图表")
    parser.add_argument("--normalize", action="store_true", help="将VAD向量归一化到[-1, 1]范围")
    parser.add_argument("--normalize_range", type=float, nargs=2, default=[-1.0, 1.0], 
                        help="归一化目标范围 (默认: -1 1)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logger.info("Arguments: %s", vars(args))
    
    # Device
    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    logger.info("Using device: %s", device)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    logger.info("加载数据文件: %s", args.data_file)
    examples = load_image_text_file(args.data_file)
    texts = [ex.sentence for ex in examples]  # 使用风格化的文本（sentence字段）
    logger.info("加载了 %d 条文本", len(texts))
    
    # 加载VAD Predictor
    logger.info("加载VAD Predictor: %s", args.vad_predictor_path)
    tokenizer = BertTokenizer.from_pretrained(args.vad_bert_path)
    vad_predictor = VADPredictor(bert_model_name=args.vad_bert_path)
    vad_predictor.load_state_dict(torch.load(args.vad_predictor_path, map_location="cpu"))
    vad_predictor.to(device)
    vad_predictor.eval()
    logger.info("VAD Predictor加载完成")
    
    # 生成VAD向量
    logger.info("=" * 80)
    logger.info("为所有文本生成VAD向量...")
    logger.info("=" * 80)
    vads = generate_vad_for_texts(texts, vad_predictor, tokenizer, args.batch_size, device)
    logger.info("VAD向量生成完成: %s", vads.shape)
    
    # 归一化（可选）
    scaling_params = None
    if args.normalize:
        logger.info("=" * 80)
        logger.info("归一化VAD向量到范围 [%.2f, %.2f]...", args.normalize_range[0], args.normalize_range[1])
        logger.info("=" * 80)
        
        # 先打印原始统计
        logger.info("原始数据统计:")
        stats_original = compute_statistics(vads)
        print_statistics(stats_original)
        
        # 保存原始统计
        stats_original_file = output_dir / "vad_statistics_original.txt"
        save_statistics(stats_original, stats_original_file)
        logger.info("原始统计已保存: %s", stats_original_file)
        
        # 归一化
        vads, scaling_params = normalize_vad_to_range(vads, tuple(args.normalize_range))
        logger.info("归一化完成")
        
        # 打印缩放参数
        logger.info("\n归一化映射:")
        for param in scaling_params:
            logger.info(f"  {param['dimension']}: [{param['original_min']:.4f}, {param['original_max']:.4f}] → [{param['target_min']:.4f}, {param['target_max']:.4f}]")
        logger.info("")
    
    # 计算统计信息（归一化后或原始）
    logger.info("=" * 80)
    logger.info("计算统计信息...")
    logger.info("=" * 80)
    stats = compute_statistics(vads)
    
    # 打印统计信息
    print_statistics(stats)
    
    # 保存统计报告
    stats_file = output_dir / ("vad_statistics_normalized.txt" if args.normalize else "vad_statistics.txt")
    save_statistics(stats, stats_file, scaling_params)
    logger.info("统计报告已保存: %s", stats_file)
    
    # 保存VAD向量（可选）
    if args.save_vectors:
        vectors_file = output_dir / ("vad_vectors_normalized.tsv" if args.normalize else "vad_vectors.tsv")
        save_vad_vectors(texts, vads, vectors_file)
        logger.info("VAD向量已保存: %s", vectors_file)
    
    # 生成可视化（可选）
    if args.visualize:
        logger.info("=" * 80)
        logger.info("生成可视化图表...")
        logger.info("=" * 80)
        create_visualizations(vads, output_dir)
    
    logger.info("=" * 80)
    logger.info("分析完成！")
    logger.info("输出目录: %s", output_dir)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

