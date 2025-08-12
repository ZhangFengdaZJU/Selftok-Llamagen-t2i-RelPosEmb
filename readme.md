（1）AR codebase：https://github.com/jiachunp/llamagen-finetune/tree/main

（2）ImageNet数据集：https://huggingface.co/datasets/visual-layer/imagenet-1k-vl-enriched/tree/main/data
# Make sure hf CLI is installed: pip install -U "huggingface_hub[cli]"
hf download visual-layer/imagenet-1k-vl-enriched --repo-type=dataset --local-dir

（3）tokenizer ckpt：https://huggingface.co/xiaoxiao012/E31/tree/main
# Make sure hf CLI is installed: pip install -U "huggingface_hub[cli]"
hf download xiaoxiao012/E31 --local-dir

（4）SD3 ckpt：https://huggingface.co/stabilityai/stable-diffusion-3-medium
# Make sure hf CLI is installed: pip install -U "huggingface_hub[cli]"
hf download stabilityai/stable-diffusion-3-medium --local-dir

（5）提image token
bash scripts/autoregressive/extract_codes_c2i.sh

（6）裁剪image token，只保留前面一部分
bash autoregressive/train/cut_token.py

（7）AR训练
bash scripts/autoregressive/train_c2i_fsdp.sh
