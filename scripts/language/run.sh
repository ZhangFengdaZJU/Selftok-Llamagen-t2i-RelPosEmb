# 单机多卡（建议用 torchrun 或你现有的 init_distributed_mode）
torchrun --nproc_per_node=8 language/extract_t5_feature_imagenet.py \
  --parquet-dir /home/jovyan/zsn/imagenet_caption_all \
  --out-dir ./output/t5_embs_out \
  --data-start 0 --data-end 206 \
  --pattern "train-{idx:05d}-of-00207.parquet" \
  --t5-model-path ./pretrained_models/t5-ckpt \
  --t5-model-type flan-t5-xl \
  --precision bf16
