#!/bin/bash
# 训练脚本：对齐图像特征和风格化文本特征
# 
# 核心思想（改进版）：
# 1. CLIP的图像和文本在同一语义空间
# 2. 训练目标：让图像特征和风格化caption的文本特征更接近（对齐）
# 3. 推理流程：图像 → CLIP → 图像特征 → VAD Predictor → VAD
# 4. 优势：利用CLIP语义对齐能力，图像特征自然包含情感信息

cd /workspace/compare_model/bart/clip

python train_text_vad.py \
    --train_file /workspace/compare_model/bart/clip/data/emotion_train.tsv \
    --val_file /workspace/compare_model/bart/clip/data/val_pairs_emotion.tsv \
    --pretrained_clip /workspace/models/clip-vit-base-patch32 \
    --max_length 77 \
    --batch_size 32 \
    --epochs 20 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --freeze_vision_layers 2 \
    --freeze_text_encoder \
    --gradient_clip 1.0 \
    --num_workers 0 \
    --output_dir /workspace/compare_model/bart/clip/output/clip_alignment_model \
    --seed 42 \
    --device cuda

echo "训练完成！模型保存在: /workspace/compare_model/bart/clip/output/clip_alignment_model"
echo "最佳模型: best_clip_alignment.pt"
echo "最终模型: last_clip_alignment.pt"
echo ""
echo "推理方法："
echo "python image_to_vad.py \\"
echo "    --alignment_checkpoint output/clip_alignment_model/best_clip_alignment.pt \\"
echo "    --vad_predictor_path vad_predictor.pt"

