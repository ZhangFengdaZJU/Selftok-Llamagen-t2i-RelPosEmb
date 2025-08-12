# !/bin/bash
set -x
export PYTHONPATH="${PWD}:${PYTHONPATH}"
export WANDB_NAME="run_$(date +%Y%m%d_%H%M%S)"
export WANDB_PROJECT="c2i_selftok"
export WANDB_API_KEY="0b30f581d65172381c1f1a45f928210cab80f1de"

torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 \
    --master_port=20045 \
    autoregressive/train/train_t2i.py \
    --image-token-path /home/jovyan/datasets/code-t2i-imagenet-E31/imagenet512_codes \
    --text-token-path /home/jovyan/zfd/LlamaGen/output/t5_embs_merged \
    --gpt-model GPT-XL --gpt-type t2i \
    --dataset t2i_code \
    --cls-token-num 256 \
    --vocab-size 16384 \
    --image-size 512 \
    --global-batch-size 64 \
    --gradient-accumulation-steps 2 \
    --cloud-save-path ./results_E31_t2i \
    --no-local-save 
