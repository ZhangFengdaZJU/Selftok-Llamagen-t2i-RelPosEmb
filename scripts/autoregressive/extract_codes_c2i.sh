# !/bin/bash
set -x

torchrun \
    --nnodes=1 --nproc_per_node=8 --node_rank=0 \
    --master_port=12336 \
    autoregressive/train/extract_codes_c2i.py \
    --yml-path /home/jovyan/zfd/Selftok-Llamagen-t2i-RelPosEmb-W1/config/FSQ-W1.yml \
    --vq-ckpt /home/jovyan/zfd/selftok-tokenizer-FSQ-W1/W1_FSQ.safetensors \
    --sd3-pretrained /home/jovyan/zfd/llamagen-finetune/stable-diffusion-3-medium/sd3_medium.safetensors \
    --data-path /home/jovyan/zfd/imagenet/data \
    --code-path /home/jovyan/zfd/datasets/code-t2i-imagenet-FSQ-W1/image_token \
    --image-size 512
