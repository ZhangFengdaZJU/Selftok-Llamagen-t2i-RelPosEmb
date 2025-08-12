# 单机多卡（建议用 torchrun 或你现有的 init_distributed_mode）
torchrun --nproc_per_node=8 language/extract_t5_feature_imagenet.py \
  --parquet-dir /home/jovyan/zfd/imagenet_long_caption_internvl \
  --out-dir /home/jovyan/zfd/datasets/code-t2i-imagenet-FSQ-W1/text_token \
  --data-start 0 --data-end 206 \
  --pattern "train-{idx:05d}-of-00207.parquet" \
  --t5-model-path /home/jovyan/zfd/llamagen-finetune/pretrained_models/t5-ckpt \
  --t5-model-type flan-t5-xl \
  --precision bf16
