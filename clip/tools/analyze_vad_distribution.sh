#!/bin/bash
# 分析数据集中文本的VAD向量分布

cd /workspace/compare_model/bart/clip

# 分析训练集（归一化到[-1, 1]）
echo "分析训练集VAD分布（归一化到[-1, 1]）..."
python tools/analyze_vad_distribution.py \
    --data_file /workspace/compare_model/bart/clip/data/emotion_train.tsv \
    --vad_predictor_path /workspace/compare_model/bart/clip/vad_predictor.pt \
    --vad_bert_path /workspace/models/bert-base-uncased \
    --batch_size 128 \
    --output_dir /workspace/compare_model/bart/clip/output/vad_analysis/train \
    --device cuda \
    --save_vectors \
    --visualize \
    --normalize \
    --normalize_range -1 1

echo ""
echo "分析验证集VAD分布（归一化到[-1, 1]）..."
python tools/analyze_vad_distribution.py \
    --data_file /workspace/compare_model/bart/clip/data/val_pairs_emotion.tsv \
    --vad_predictor_path /workspace/compare_model/bart/clip/vad_predictor.pt \
    --vad_bert_path /workspace/models/bert-base-uncased \
    --batch_size 128 \
    --output_dir /workspace/compare_model/bart/clip/output/vad_analysis/val \
    --device cuda \
    --save_vectors \
    --visualize \
    --normalize \
    --normalize_range -1 1

echo ""
echo "================================"
echo "分析完成！"
echo "训练集结果: output/vad_analysis/train/"
echo "验证集结果: output/vad_analysis/val/"
echo "================================"

