# !/bin/bash
set -x

torchrun \
    --nnodes=1 --nproc_per_node=1 --node_rank=0 \
    --master_port=12346 \
    autoregressive/sample/sample_t2i.py \
    --yml-path="/home/jovyan/zfd/LlamaGen/config/renderer-eval.yml" \
    --vq-ckpt="/home/jovyan/zfd/llamagen-finetune/tokenizer-ckpt-E31/E31_renderer.safetensors" \
    --gpt-model="GPT-XL" \
    --gpt-ckpt="/home/jovyan/zfd/LlamaGen/results_E31_t2i/2025-08-07-04-58-42/000-GPT-XL/checkpoints/0350000.pt" \
    --sd3-pretrained="/home/jovyan/zfd/llamagen-finetune/SD3-ckpt/sd3_medium.safetensors" \
    --gpt-type="t2i" \
    --codebook-size=16384 \
    --image-size=512 \
    --cfg-scale 3.0 \
	--top-p 1.0 \
	--top-k 0 \
	--temperature 1.0 \
