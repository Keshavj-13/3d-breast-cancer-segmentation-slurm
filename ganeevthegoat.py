#!/usr/bin/env python3
"""
Multi-GPU 3D Medical Image Segmentation (DDP + MONAI)
Dataset: MAMA-MIA
Model: Attention UNet (FlexibleUNet + SCSE)
"""

import os
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.distributed as dist

from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from sklearn.model_selection import train_test_split

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    ScaleIntensityRanged, NormalizeIntensityd, CropForegroundd,
    SpatialPadD, DivisiblePadd,
    RandCropByPosNegLabeld, RandFlipd, RandRotate90d, RandAffined,
    EnsureTyped
)
from monai.data import DataLoader, PersistentDataset, Dataset
from monai.networks.nets import FlexibleUNet
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric
from monai.utils import set_determinism

from tqdm import tqdm

# =========================================================
# CONFIG
# =========================================================
BASE_DIR = "/hostd/mama_mia_dataset/MAMA-MIA"
CACHE_DIR = "/hostd/mama_mia_cache"
OUTPUT_DIR = "/hostd/mama_mia_output_ddp"

MAX_EPOCHS = 500
BATCH_SIZE = 2          # per GPU
NUM_WORKERS = 8         # per GPU
LR = 1e-4
WEIGHT_DECAY = 1e-5

PATCH_SIZE = (128, 128, 96)

set_determinism(42)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# DISTRIBUTED HELPERS
# =========================================================
def is_distributed():
    return int(os.environ.get("WORLD_SIZE", 1)) > 1

def get_rank():
    return int(os.environ.get("RANK", 0))

def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", 0))

def is_main_process():
    return get_rank() == 0

def init_distributed():
    if is_distributed():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(get_local_rank())
        device = torch.device(f"cuda:{get_local_rank()}")
    else:
        device = torch.device("cuda")
    return device

def cleanup():
    if is_distributed():
        dist.destroy_process_group()

# =========================================================
# MODEL
# =========================================================
class SCSEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        self.sSE = nn.Sequential(
            nn.Conv3d(channels, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)

class AttentionUNet(FlexibleUNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attn = nn.ModuleList([
            SCSEBlock(c) for c in kwargs["decoder_channels"]
        ])

    def forward(self, x):
        feats = self.encoder(x)
        skips = [f for f in feats[:-1] if f is not None][::-1]
        x = feats[-1]

        for i, block in enumerate(self.decoder.blocks):
            x = block(x, skips[i] if i < len(skips) else None)
            x = self.attn[i](x)

        return self.segmentation_head(x)

# =========================================================
# DATA
# =========================================================
def prepare_files():
    images = os.path.join(BASE_DIR, "images")
    labels = os.path.join(BASE_DIR, "segmentations", "expert")

    files = []
    for pid in sorted(os.listdir(images)):
        img = os.path.join(images, pid, f"{pid}_0002.nii")
        lab = os.path.join(labels, f"{pid}.nii")
        if os.path.exists(img) and os.path.exists(lab):
            files.append({"image": img, "label": lab})

    return train_test_split(files, test_size=0.2, random_state=42)

def pre_transforms():
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1,1,1),
                 mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"], a_min=0, a_max=1555,
            b_min=0, b_max=1, clip=True),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadD(keys=["image", "label"], spatial_size=PATCH_SIZE),
        DivisiblePadd(keys=["image", "label"], k=32),
    ])

def rand_transforms():
    return Compose([
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=PATCH_SIZE,
            pos=1, neg=1,
            num_samples=2,
            image_key="image"),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0,1,2]),
        RandRotate90d(keys=["image", "label"], prob=0.5),
        RandAffined(keys=["image", "label"], prob=0.3, rotate_range=0.1),
        EnsureTyped(keys=["image", "label"]),
    ])

# =========================================================
# TRAIN
# =========================================================
def train():
    device = init_distributed()

    if is_main_process():
        print("🚀 Starting training")
        print("GPUs:", torch.cuda.device_count())

    train_files, val_files = prepare_files()

    base_train = PersistentDataset(
        train_files,
        transform=pre_transforms(),
        cache_dir=os.path.join(CACHE_DIR, "train")
    )

    train_ds = Dataset(base_train, transform=rand_transforms())

    train_sampler = DistributedSampler(train_ds) if is_distributed() else None

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    model = AttentionUNet(
        in_channels=1,
        out_channels=1,
        backbone="resnet50",
        pretrained=False,
        decoder_channels=(512,256,128,64,32),
        spatial_dims=3,
    ).to(device)

    if is_distributed():
        model = DDP(model, device_ids=[get_local_rank()])

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = DiceFocalLoss(sigmoid=True, include_background=False)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(MAX_EPOCHS):
        if train_sampler:
            train_sampler.set_epoch(epoch)

        model.train()
        epoch_loss = 0

        if is_main_process():
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        else:
            pbar = train_loader

        for batch in pbar:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():
                y_hat = model(x)
                loss = loss_fn(y_hat, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        if is_main_process():
            print(f"Epoch {epoch+1} | Loss {epoch_loss/len(train_loader):.4f}")

    if is_main_process():
        torch.save(model.module.state_dict() if is_distributed()
                   else model.state_dict(),
                   os.path.join(OUTPUT_DIR, "final_model.pth"))

    cleanup()

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    train()
S