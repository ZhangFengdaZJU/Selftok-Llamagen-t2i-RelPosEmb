import glob, io, os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class ParquetImageNetDataset(Dataset):
    def __init__(self, parquet_dir, transform=None):
        self.parquet_files = sorted(glob.glob(f"{parquet_dir}/train-*.parquet"))
        self.dfs = []          # 先存各自 df，稍后 concat
        self.paths = []        # 全局 idx 对应的 “parquet路径::图像名”
        for file in self.parquet_files:
            df = pd.read_parquet(file)
            df = df.reset_index(drop=True)

            # 猜测图像名列（按常见字段优先级）
            name_col = "image_id"
            names = df[name_col].astype(str).tolist()
            
            # 记录全局路径映射
            self.paths.extend(["/home/jovyan/zfd/LlamaGen/output/t5_embs_out/"+file.split("/")[-1].split(".")[0]+"/"+n+".npy" for n in names])
            # 标注来源，调试时也有用
            df["__source_parquet__"] = file
            self.dfs.append(df)

        self.data = pd.concat(self.dfs, ignore_index=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def get_image_path(self, idx: int) -> str:
        """返回形如 /.../train-00012-of-00207.parquet::ILSVRC2012_train_00000001.JPEG 的伪路径"""
        return self.paths[idx]

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_bytes = row["image"]["bytes"]
        label = int(row["label"])
        if isinstance(image_bytes, str):  # 可能是base64
            import base64
            image_bytes = base64.b64decode(image_bytes)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        # 如果你希望一起返回路径，取消下一行注释
        # return img, label, self.get_image_path(idx)
        return img, label


ds = ParquetImageNetDataset("/home/jovyan/data/imagenet-1k/data")
for i in range(N):
    npy_path = ds.get_image_path(i)
    
