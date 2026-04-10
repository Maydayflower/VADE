#!/bin/bash
# 训练下游VAD预测器
# 
# Pipeline:
# 1. 加载对齐好的CLIP模型（冻结）→ 提取图像特征
# 2. 使用VAD Predictor从风格化caption生成VAD标签
# 3. 训练MLP: 图像特征 → VAD

cd /workspace/compare_model/bart/clip

python train_vad_downstream.py \
    --train_file /workspace/compare_model/bart/clip/data/emotion_train.tsv \
    --val_file /workspace/compare_model/bart/clip/data/emotion_val.tsv \
    --clip_checkpoint /workspace/compare_model/bart/clip/output/clip_alignment_model/best_clip_alignment.pt \
    --clip_path /workspace/models/clip-vit-base-patch32 \
    --vad_predictor_path /workspace/compare_model/bart/clip/vad_predictor.pt \
    --vad_bert_path /workspace/models/bert-base-uncased \
    --batch_size 128 \
    --feature_batch_size 64 \
    --epochs 30 \
    --learning_rate 1e-3 \
    --hidden_dim 256 \
    --dropout 0.1 \
    --output_dir /workspace/compare_model/bart/clip/output/vad_downstream_model \
    --device cuda

echo ""
echo "训练完成！模型保存在: /workspace/compare_model/bart/clip/output/vad_downstream_model"
echo "最佳模型: best_vad_downstream.pt"
echo "最终模型: last_vad_downstream.pt"
echo ""
echo "推理方法："
echo "python infer_vad_downstream.py \\"
echo "    --clip_checkpoint output/clip_alignment_model/best_clip_alignment.pt \\"
echo "    --vad_checkpoint output/vad_downstream_model/best_vad_downstream.pt"

