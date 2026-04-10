python /workspace/compare_model/bart/clip/train_text_vad.py \
     --vad_predictor_path /workspace/compare_model/bart/clip/vad_predictor.pt \
     --vad_bert_path /workspace/models/bert-base-uncased \
     --train_file /workspace/compare_model/bart/clip/data/emotion_train.tsv \
     --val_file /workspace/compare_model/bart/clip/data/emotion_val.tsv \
     --pretrained_clip /workspace/models/clip-vit-base-patch32 \
     --output_dir /workspace/compare_model/bart/clip/checkpoints/clip_text_vad