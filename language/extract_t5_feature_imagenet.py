#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.append("./")

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils.distributed import init_distributed_mode
from language.t5 import T5Embedder

#################################################################################
#                                  Dataset                                      #
#################################################################################
class ParquetCaptionDataset(Dataset):
    """
    读取指定目录下 [start, end] 范围内的 parquet（通过 pattern 构造文件名），
    收集两列：image_id, caption。__getitem__ 返回 (caption, parquet_stem, image_id)。
    """
    def __init__(self, parquet_dir: str, start: int, end: int, pattern: str = "train-{idx:05d}-of-00207.parquet",
                 trunc_caption: bool = False):
        self.items = []
        pdir = Path(parquet_dir)
        assert pdir.is_dir(), f"parquet_dir not found: {parquet_dir}"

        for i in range(start, end + 1):
            fname = pattern.format(idx=i)
            fpath = pdir / fname
            if not fpath.exists():
                print(f"[WARN] file not found, skip: {fpath}")
                continue

            try:
                df = pd.read_parquet(fpath, columns=["image_id", "caption"])
            except Exception as e:
                print(f"[ERROR] read parquet failed: {fpath} -> {e}")
                continue

            if "image_id" not in df.columns or "caption" not in df.columns:
                print(f"[ERROR] parquet missing columns in {fpath.name}, has {df.columns.tolist()}")
                continue

            stem = fpath.stem  # e.g. train-00000-of-00207
            for _, row in df.iterrows():
                cap = row["caption"]
                if isinstance(cap, str):
                    if trunc_caption:
                        cap = cap.split(".")[0]
                else:
                    # 非字符串的 caption，跳过或转空
                    cap = "" if cap is None else str(cap)

                img_id = str(row["image_id"])
                self.items.append((cap, stem, img_id))

        if len(self.items) == 0:
            print("[WARN] No items collected from given range/files.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        cap, stem, img_id = self.items[idx]
        return cap, stem, img_id


#################################################################################
#                                  Main Loop                                    #
#################################################################################
def main(args):
    assert torch.cuda.is_available(), "Requires at least one GPU."

    # DDP init
    init_distributed_mode(args)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    if rank == 0:
        print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")

    # Output root
    if rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)

    # Dataset & Loader
    if rank == 0:
        print("Building dataset ...")
    dataset = ParquetCaptionDataset(
        parquet_dir=args.parquet_dir,
        start=args.data_start,
        end=args.data_end,
        pattern=args.pattern,
        trunc_caption=args.trunc_caption
    )

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=1,            # 保持 1，方便逐条变长处理
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    if rank == 0:
        print(f"Dataset size: {len(dataset):,}")

    # T5 embedder
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    assert os.path.exists(args.t5_model_path), f"t5_model_path not found: {args.t5_model_path}"

    t5 = T5Embedder(
        device=device,
        local_cache=True,
        cache_dir=args.t5_model_path,
        dir_or_name=args.t5_model_type,
        torch_dtype=precision
    )

    # Loop
    for caption, parquet_stem, image_id in loader:
        # DataLoader batch_size=1 -> 取出标量
        caption = caption[0]
        parquet_stem = parquet_stem[0]
        image_id = image_id[0]

        # 获取 token embeddings & mask
        # caption_embs: [1, L, D]; emb_masks: [1, L] (1 有效, 0 padding)
        caption_embs, emb_masks = t5.get_text_embeddings([caption])  # 传 list[str] 以适配某些实现
        # 取有效长度
        valid_len = int(emb_masks[0].sum().item())
        # 截掉 padding，并去掉 batch 维 -> [L, D]
        valid_caption_embs = caption_embs[0, :valid_len].to(torch.float32).detach().cpu().numpy()

        # 保存到 {out_dir}/{parquet_stem}/{image_id}.npy
        save_dir = Path(args.out_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{image_id}.npy"
        np.save(save_path, valid_caption_embs)

        # if rank == 0:
        #     print(f"saved: {save_path}")

    dist.destroy_process_group()


#################################################################################
#                                    CLI                                        #
#################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Extract T5 token embeddings from parquet captions")
    parser.add_argument("--parquet-dir", type=str, required=True, help="输入 parquet 目录")
    parser.add_argument("--out-dir", type=str, required=True, help="输出根目录（将生成 out/parquet_stem/image_id.npy）")
    parser.add_argument("--data-start", type=int, required=True, help="起始索引（含）")
    parser.add_argument("--data-end", type=int, required=True, help="结束索引（含）")
    parser.add_argument("--pattern", type=str, default="train-{idx:05d}-of-00207.parquet", help="文件名模式")
    parser.add_argument("--trunc-caption", action="store_true", default=False, help="是否仅取句号前的首句")
    parser.add_argument("--t5-model-path", type=str, default="./pretrained_models/t5-ckpt", help="T5 缓存/权重目录")
    parser.add_argument("--t5-model-type", type=str, default="flan-t5-xl", help="如 flan-t5-xl / flan-t5-xxl / t5-11b 等")
    parser.add_argument("--precision", type=str, default="bf16", choices=["none", "fp16", "bf16"])
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    # DDP args (utils.distributed 里会读取环境变量等)
    parser.add_argument("--dist-url", default="env://")
    args = parser.parse_args()
    main(args)
