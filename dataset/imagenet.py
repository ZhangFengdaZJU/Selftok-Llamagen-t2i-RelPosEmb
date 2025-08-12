import torch
import numpy as np
import os
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import io
import glob
import pandas as pd
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, feature_dir, label_dir):
        self.feature_dir = feature_dir
        self.label_dir = label_dir
        self.flip = 'flip' in self.feature_dir

        if 'ten_crop/' in feature_dir:
            aug_feature_dir = feature_dir.replace('ten_crop/', 'ten_crop_105/')
            aug_label_dir = label_dir.replace('ten_crop/', 'ten_crop_105/')
            if os.path.exists(aug_feature_dir) and os.path.exists(aug_label_dir):
                self.aug_feature_dir = aug_feature_dir
                self.aug_label_dir = aug_label_dir
        else:
            self.aug_feature_dir = None
            self.aug_label_dir = None

        # self.feature_files = sorted(os.listdir(feature_dir))
        # self.label_files = sorted(os.listdir(label_dir))
        # TODO: make it configurable
        self.feature_files = [f"{i}.npy" for i in range(1281167)]
        self.label_files = [f"{i}.npy" for i in range(1281167)]

    def __len__(self):
        assert len(self.feature_files) == len(self.label_files), \
            "Number of feature files and label files should be same"
        return len(self.feature_files)

    def __getitem__(self, idx):
        if self.aug_feature_dir is not None and torch.rand(1) < 0.5:
            feature_dir = self.aug_feature_dir
            label_dir = self.aug_label_dir
        else:
            feature_dir = self.feature_dir
            label_dir = self.label_dir
                   
        feature_file = self.feature_files[idx]
        label_file = self.label_files[idx]

        features = np.load(os.path.join(feature_dir, feature_file))
        
        # aug_idx = torch.randint(low=0, high=features.shape[1], size=(1,)).item()
        # features = features[:, aug_idx]
        labels = np.load(os.path.join(label_dir, label_file))
        return torch.from_numpy(features), torch.from_numpy(labels)

# class ParquetImageNetDataset(Dataset):
#     def __init__(self, parquet_dir, transform=None):
#         self.parquet_files = sorted(glob.glob(f"{parquet_dir}/train-*.parquet"))
#         self.data = []
#         for file in self.parquet_files:
#             df = pd.read_parquet(file)
#             self.data.append(df)
#         self.data = pd.concat(self.data, ignore_index=True)
#         self.transform = transform

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         row = self.data.iloc[idx]
#         image_bytes = row['image']['bytes']
#         label = int(row['label'])
#         # If image_bytes is a string, decode base64, else use bytes directly
#         if isinstance(image_bytes, str):
#             import base64
#             image_bytes = base64.b64decode(image_bytes)
#         img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
#         if self.transform:
#             img = self.transform(img)
#         return img, label

class ParquetImageNetDataset(Dataset):
    def __init__(self, parquet_dir, transform=None):
        self.parquet_files = sorted(glob.glob(f"{parquet_dir}/train-*.parquet"))
        self.data = []
        for file in self.parquet_files:
            df = pd.read_parquet(file)
            self.data.append(df)
        self.data = pd.concat(self.data, ignore_index=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_bytes = row['image']['bytes']
        label = int(row['label'])
        path = row['image_id']
        # If image_bytes is a string, decode base64, else use bytes directly
        if isinstance(image_bytes, str):
            import base64
            image_bytes = base64.b64decode(image_bytes)
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label, path


def build_imagenet(args, transform):
    return ParquetImageNetDataset(args.data_path, transform=transform)

def build_imagenet_code(args):
    feature_dir = f"{args.code_path}/imagenet{args.image_size}_codes"
    label_dir = f"{args.code_path}/imagenet{args.image_size}_labels"
    assert os.path.exists(feature_dir) and os.path.exists(label_dir), \
        f"please first run: bash scripts/autoregressive/extract_codes_c2i.sh ..."
    return CustomDataset(feature_dir, label_dir)

import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import sys 
sys.path.append('./')
from mimogpt.infer.SelftokPipeline import NormalizeToTensor

def resize_and_pad(image, target_size=256, padding_color="black"):
    """
    Resizes the image, upsampling if necessary, and pads to create a square.

    Args:
        image: PIL Image object.
        target_size: Target size for the *largest* dimension.
        padding_color: 'black' or 'white' (or any valid PIL color).

    Returns:
        A new PIL Image object, square and padded.
    """
    width, height = image.size

    # Calculate scaling factor, upsampling if needed
    scale = target_size / max(width, height)
    new_width = int(width * scale)
    new_height = int(height * scale)

    # Resize (upsample or downsample)
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)

    # Create a new square image
    square_image = Image.new("RGB", (target_size, target_size), color=padding_color)

    # Calculate paste position (center)
    paste_x = (target_size - new_width) // 2
    paste_y = (target_size - new_height) // 2

    square_image.paste(resized_image, (paste_x, paste_y))
    return square_image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/home/jovyan/zfd/imagenet/data",
                        help="Path to ImageNet parquet folder (for 'imagenet')")
    parser.add_argument("--image_size", type=int, default=512,
                        help="Image size used in dataset")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for testing")
    args = parser.parse_args()

    # Simple transform for images
    transform = transforms.Compose([
                transforms.Lambda(lambda pil_image: resize_and_pad(pil_image, self.image_size, 'white')),
                NormalizeToTensor(),
            ])


    dataset = build_imagenet(args, transform)

    images, labels = dataset[0]

    # loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)

    # print(f"Dataset length: {len(dataset)} samples")
    # for i, batch in enumerate(loader):

    #     images, labels = batch
    #     print(f"Batch {i}: images {images.shape}, labels {labels}")

    #     if i == 2:  # Show first 3 batches
    #         break

if __name__ == "__main__":
    main()
