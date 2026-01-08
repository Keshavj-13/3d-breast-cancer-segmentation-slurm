#!/usr/bin/env python3
"""
3D Medical Image Segmentation Training Script
Yunnan Dataset
SwinUNETR and UNETR Only
TRUE DDP, memory safe retries
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data.distributed import DistributedSampler

from sklearn.model_selection import train_test_split

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    EnsureTyped, ScaleIntensityRanged, NormalizeIntensityd,
    RandGaussianNoised, RandAdjustContrastd,
    CropForegroundd, RandCropByPosNegLabeld, DivisiblePadd,
    RandFlipd, RandRotate90d, RandAffined, SpatialPadD
)
from monai.data import DataLoader, CacheDataset
from monai.utils import set_determinism
from monai.networks.nets import SwinUNETR, UNETR
from monai.losses import DiceLoss, DiceFocalLoss, DiceCELoss
from monai.metrics import DiceMetric

# ENV
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
set_determinism(42)

# CONFIG
BASE_DIR = "/hostd/yunnan"
OUTPUT_ROOT = "/hostd/yunnan_ablation"
MAX_PATIENTS = 100
MAX_EPOCHS = 500
VAL_INTERVAL = 5
GLOBAL_BATCH_SIZE_TRAIN = 3
BATCH_SIZE_VAL = 1
NUM_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", 2))
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# DDP INIT
def init_ddp():
    if "RANK" not in os.environ:
        return False, 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return True, rank, world_size, local_rank

# MODELS
def build_model(name, device):
    if name == "swin_unetr":
        return SwinUNETR(
            in_channels=1,
            out_channels=1,
            feature_size=48,
            use_checkpoint=True,
            spatial_dims=3
        ).to(device)

    if name == "unetr":
        return UNETR(
            in_channels=1,
            out_channels=1,
            img_size=(128, 128, 96),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            norm_name="instance",
            spatial_dims=3
        ).to(device)

    raise ValueError("Unknown model")

# LOSSES
class HybridDiceCEFocalTverskyLoss(nn.Module):
    def __init__(self, dice_ce_weight, tversky_weight, alpha, beta, gamma):
        super().__init__()
        self.dice_ce_weight = dice_ce_weight
        self.tversky_weight = tversky_weight
        self.dice_ce = DiceCELoss(
            to_onehot_y=False,
            sigmoid=True,
            lambda_dice=1.0,
            lambda_ce=1.0,
        )
        self.tversky = DiceFocalLoss(
            include_background=False,
            sigmoid=True,
            squared_pred=False,
            lambda_dice=1.0,
            lambda_focal=1.0,
            alpha=alpha,
            gamma=gamma,
            reduction="mean",
        )

    def forward(self, preds, targets):
        return (
            self.dice_ce_weight * self.dice_ce(preds, targets)
            + self.tversky_weight * self.tversky(preds, targets)
        )

def build_loss(name):
    if name == "dice_ce":
        return DiceCELoss(
            to_onehot_y=False,
            sigmoid=True,
            lambda_dice=1.5,
            lambda_ce=1.15,
        )

    if name == "hybrid_dice_ce_focal_tversky":
        return HybridDiceCEFocalTverskyLoss(
            dice_ce_weight=0.5,
            tversky_weight=0.5,
            alpha=0.36246607666571395,
            beta=0.7883473198308575,
            gamma=1.7644243682253733,
        )

    raise ValueError("Unknown loss")

# DATA
def prepare_data():
    cases = sorted([
        d for d in os.listdir(BASE_DIR)
        if os.path.isdir(os.path.join(BASE_DIR, d))
    ])[:MAX_PATIENTS]

    files = []
    for c in cases:
        img = os.path.join(BASE_DIR, c, "P5.nii")
        lbl = os.path.join(BASE_DIR, c, "GT.nii")
        if os.path.exists(img) and os.path.exists(lbl):
            files.append({"image": img, "label": lbl})

    return train_test_split(files, test_size=0.2, random_state=42)

# TRANSFORMS unchanged


def get_train_transforms():
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1, 1, 1), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=0, a_max=1555, b_min=0, b_max=1, clip=True),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadD(keys=["image", "label"], spatial_size=(128, 128, 96)),
        DivisiblePadd(keys=["image", "label"], k=32),
        RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", spatial_size=(128, 128, 96), pos=1, neg=1, num_samples=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0, 1, 2]),
        RandRotate90d(keys=["image", "label"], prob=0.5),
        RandAffined(keys=["image", "label"], prob=0.5, rotate_range=0.1, scale_range=0.1, translate_range=10, mode=("bilinear", "nearest")),
        RandGaussianNoised(keys=["image"], prob=0.3, std=0.01),
        RandAdjustContrastd(keys=["image"], prob=0.3),
        EnsureTyped(keys=["image", "label"]),
    ])


def get_val_transforms():
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1, 1, 1), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=0, a_max=1555, b_min=0, b_max=1, clip=True),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadD(keys=["image", "label"], spatial_size=(128, 128, 96)),
        DivisiblePadd(keys=["image", "label"], k=32),
        EnsureTyped(keys=["image", "label"]),
    ])

# TRAIN
def train_one(cfg, train_files, val_files, global_bs, distributed, rank, world_size, local_rank):
    attempt_bs = global_bs

    while attempt_bs >= 1:
        device = torch.device("cuda", local_rank)
        out_dir = os.path.join(OUTPUT_ROOT, cfg["name"] + f"_bs{attempt_bs}")

        if rank == 0:
            os.makedirs(out_dir, exist_ok=True)
            print(f"Starting {cfg['name']} with batch size {attempt_bs}")

        # logging buffers
        epochs = []
        losses = []
        train_dice_scores = []
        val_dice_scores = []

        try:
            train_ds = CacheDataset(train_files, get_train_transforms(), cache_rate=0.0)
            val_ds = CacheDataset(val_files, get_val_transforms(), cache_rate=0.0)

            train_sampler = DistributedSampler(train_ds, world_size, rank, shuffle=True) if distributed else None
            val_sampler = DistributedSampler(val_ds, world_size, rank, shuffle=False) if distributed else None

            train_loader = DataLoader(train_ds, batch_size=attempt_bs, sampler=train_sampler, shuffle=False, num_workers=NUM_WORKERS)
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE_VAL, sampler=val_sampler, shuffle=False, num_workers=NUM_WORKERS)

            model = build_model(cfg["model"], device)
            if distributed:
                model = DDP(model, device_ids=[local_rank])

            loss_fn = build_loss(cfg["loss"])
            optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

            train_metric = DiceMetric(include_background=False, reduction="mean")
            val_metric = DiceMetric(include_background=False, reduction="mean")

            best = -1.0

            for epoch in range(MAX_EPOCHS):
                if distributed:
                    train_sampler.set_epoch(epoch)

                model.train()
                train_metric.reset()
                total_loss = 0.0
                steps = 0

                for batch in train_loader:
                    steps += 1
                    imgs = batch["image"].to(device)
                    lbls = batch["label"].to(device)

                    optimizer.zero_grad()
                    out = model(imgs)
                    loss = loss_fn(out, lbls)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    with torch.no_grad():
                        preds = (torch.sigmoid(out) > 0.5).float()
                        train_metric(preds, lbls)

                avg_loss = total_loss / max(1, steps)
                train_dice = train_metric.aggregate().item()

                if rank == 0:
                    print(f"Epoch {epoch + 1}, avg loss {avg_loss:.4f}, train dice {train_dice:.4f}")

                    epochs.append(epoch + 1)
                    losses.append(avg_loss)
                    train_dice_scores.append(train_dice)

                if (epoch + 1) % VAL_INTERVAL == 0:
                    model.eval()
                    val_metric.reset()
                    with torch.no_grad():
                        for batch in val_loader:
                            imgs = batch["image"].to(device)
                            lbls = batch["label"].to(device)
                            preds = (torch.sigmoid(model(imgs)) > 0.5).float()
                            val_metric(preds, lbls)

                    val_score = val_metric.aggregate().item()

                    if rank == 0:
                        print(f"Validation dice {val_score:.4f}")
                        val_dice_scores.append(val_score)

                        if val_score > best:
                            best = val_score
                            torch.save(
                                getattr(model, "module", model).state_dict(),
                                os.path.join(out_dir, "best_model.pth")
                            )
                else:
                    if rank == 0:
                        val_dice_scores.append(np.nan)

                if rank == 0:
                    np.save(os.path.join(out_dir, "epochs.npy"), np.array(epochs))
                    np.save(os.path.join(out_dir, "loss_values.npy"), np.array(losses))
                    np.save(os.path.join(out_dir, "train_dice_scores.npy"), np.array(train_dice_scores))
                    np.save(os.path.join(out_dir, "val_dice_scores.npy"), np.array(val_dice_scores))

            if rank == 0:
                torch.save(
                    getattr(model, "module", model).state_dict(),
                    os.path.join(out_dir, "final_model.pth")
                )

            return

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                if attempt_bs == 1:
                    if rank == 0:
                        print("OOM at batch size one, skipping run")
                    return
                attempt_bs //= 2
            else:
                raise

# MAIN
if __name__ == "__main__":
    distributed, rank, world_size, local_rank = init_ddp()
    train_files, val_files = prepare_data()

    CONFIGS = [
        # {"name": "swin_unetr_dice_ce", "model": "swin_unetr", "loss": "dice_ce"},
        {"name": "swin_unetr_hybrid", "model": "swin_unetr", "loss": "hybrid_dice_ce_focal_tversky"},
        {"name": "unetr_dice_ce", "model": "unetr", "loss": "dice_ce"},
        {"name": "unetr_hybrid", "model": "unetr", "loss": "hybrid_dice_ce_focal_tversky"},
    ]

    for cfg in CONFIGS:
        train_one(cfg, train_files, val_files, GLOBAL_BATCH_SIZE_TRAIN, distributed, rank, world_size, local_rank)
        torch.cuda.empty_cache()

    if distributed and dist.is_initialized():
        dist.destroy_process_group()

    print("All runs completed.")
