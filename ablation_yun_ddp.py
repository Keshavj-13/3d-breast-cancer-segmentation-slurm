#!/usr/bin/env python3
"""
3D Medical Image Segmentation Training Script
Yunnan Dataset, Multi Model Ablation with TRUE DDP

Single script, single entry point.
Each architecture is trained, with automatic memory safe retry logic.
If a run triggers GPU out of memory, the code reduces batch size until one.
If still OOM at batch size one, the model is skipped with a logged warning.
DDP is initialized and destroyed per run to avoid state leakage.
"""

import os
import sys
import time
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
from monai.networks.nets import UNet, UNETR, SwinUNETR, FlexibleUNet
from monai.losses import DiceLoss, DiceFocalLoss
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
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.00001

# DDP INIT helper
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
class SCSEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cse = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, max(1, in_channels // reduction), 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(max(1, in_channels // reduction), in_channels, 1),
            nn.Sigmoid(),
        )
        self.sse = nn.Sequential(nn.Conv3d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return self.cse(x) * x + self.sse(x) * x


class AttentionAllDecoderUNet(FlexibleUNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        channels = kwargs.get("decoder_channels", (512, 256, 128, 64, 32))
        self.attn = nn.ModuleList([SCSEBlock(c) for c in channels])

    def forward(self, x):
        feats = self.encoder(x)
        skips = [f for f in feats[:-1] if f is not None][::-1]
        x = feats[-1]
        for i, blk in enumerate(self.decoder.blocks):
            x = blk(x, skips[i] if i < len(skips) else None)
            x = self.attn[i](x)
        return self.segmentation_head(x)


def build_model(name, device):
    if name == "unet_baseline":
        return UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm="INSTANCE",
        ).to(device)

    if name == "nnunet_style":
        return UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(32, 64, 128, 256, 320),
            strides=(2, 2, 2, 2),
            num_res_units=3,
            norm="INSTANCE",
        ).to(device)

    if name == "attention_all_decoder_unet":
        return AttentionAllDecoderUNet(
            in_channels=1,
            out_channels=1,
            backbone="resnet50",
            spatial_dims=3,
            decoder_channels=(512, 256, 128, 64, 32),
            norm=("instance", {"affine": True}),
            dropout=0.2,
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
        ).to(device)

    if name == "swin_unetr":
        return SwinUNETR(
            img_size=(128, 128, 96),
            in_channels=1,
            out_channels=1,
            feature_size=48,
            use_checkpoint=False,
        ).to(device)

    raise ValueError(f"Unknown model {name}")

# LOSSES
from monai.losses import DiceCELoss

# Custom hybrid loss wrapper using MONAI primitives
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
        loss_dice_ce = self.dice_ce(preds, targets)
        loss_tversky = self.tversky(preds, targets)
        return (
            self.dice_ce_weight * loss_dice_ce
            + self.tversky_weight * loss_tversky
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

    if name == "dice_focal":
        return DiceFocalLoss(
            include_background=False,
            sigmoid=True,
            squared_pred=True,
            lambda_dice=1.0,
            lambda_focal=1.0,
            gamma=2.0,
            alpha=0.7,
            reduction="mean",
        )

    if name == "dice":
        return DiceLoss(include_background=False, sigmoid=True)

    raise ValueError("Unknown loss {}".format(name))

# DATA
def prepare_data():
    cases = sorted([d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))])[:MAX_PATIENTS]
    files = []
    for c in cases:
        img = os.path.join(BASE_DIR, c, "P5.nii")
        lbl = os.path.join(BASE_DIR, c, "GT.nii")
        if os.path.exists(img) and os.path.exists(lbl):
            files.append({"image": img, "label": lbl})
    return train_test_split(files, test_size=0.2, random_state=42)


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

# TRAIN ONE MODEL with memory safe retries
def train_one(cfg, train_files, val_files, global_bs, distributed, rank, world_size, local_rank):
    attempt_bs = global_bs
    successful = False
    last_exception = None

    while attempt_bs >= 1 and not successful:
        device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

        out_dir = os.path.join(OUTPUT_ROOT, cfg["name"] + f"_bs{attempt_bs}")
        if rank == 0:
            os.makedirs(out_dir, exist_ok=True)
            print(f"Starting run {cfg['name']} with batch size {attempt_bs}")

        # per-run logging lists, kept local to run and saved each epoch
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

            dice_metric = DiceMetric(include_background=False, reduction="mean")
            train_metric = DiceMetric(include_background=False, reduction="mean")

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
                    total_loss += float(loss.item())
                    with torch.no_grad():
                        preds = (torch.sigmoid(out) > 0.5).float()
                        train_metric(preds, lbls)

                avg_loss = total_loss / max(1, steps)
                # aggregate train dice metric safely
                try:
                    t_dice = train_metric.aggregate().item()
                except Exception:
                    t_dice = float("nan")

                if rank == 0:
                    print(f"Epoch {epoch + 1}, avg loss {avg_loss:.4f}, train dice {t_dice:.4f}" if not math.isnan(t_dice) else f"Epoch {epoch + 1}, avg loss {avg_loss:.4f}")

                # Append epoch level values
                if rank == 0:
                    epochs.append(epoch + 1)
                    losses.append(avg_loss)
                    train_dice_scores.append(t_dice)

                # Validation block, and val_dice_scores update
                if (epoch + 1) % VAL_INTERVAL == 0:
                    model.eval()
                    dice_metric.reset()
                    with torch.no_grad():
                        for batch in val_loader:
                            imgs = batch["image"].to(device)
                            lbls = batch["label"].to(device)
                            preds = (torch.sigmoid(model(imgs)) > 0.5).float()
                            dice_metric(preds, lbls)
                    try:
                        score = dice_metric.aggregate().item()
                    except Exception:
                        score = float("nan")
                    if rank == 0:
                        val_dice_scores.append(score)
                        print(f"Validation dice {score:.4f}" if not math.isnan(score) else "Validation dice nan")
                        if score > best:
                            best = score
                            save_path = os.path.join(out_dir, "best_model.pth")
                            torch.save(getattr(model, 'module', model).state_dict(), save_path)
                else:
                    if rank == 0:
                        # keep arrays aligned by placing nan for non validation epochs
                        val_dice_scores.append(np.nan)

                # Save per epoch logs to out_dir, rank zero only
                if rank == 0:
                    np.save(os.path.join(out_dir, "epochs.npy"), np.array(epochs))
                    np.save(os.path.join(out_dir, "loss_values.npy"), np.array(losses))
                    np.save(os.path.join(out_dir, "train_dice_scores.npy"), np.array(train_dice_scores))
                    np.save(os.path.join(out_dir, "val_dice_scores.npy"), np.array(val_dice_scores))

            if rank == 0:
                torch.save(getattr(model, 'module', model).state_dict(), os.path.join(out_dir, "final_model.pth"))

            successful = True

        except RuntimeError as e:
            last_exception = e
            msg = str(e).lower()
            if "out of memory" in msg or "cuda out of memory" in msg:
                if rank == 0:
                    print(f"OOM at batch size {attempt_bs}, clearing cache and retrying with smaller batch size")
                torch.cuda.empty_cache()
                if attempt_bs == 1:
                    if rank == 0:
                        print(f"Skipping model {cfg['name']}, OOM at batch size one")
                    break
                attempt_bs = max(1, attempt_bs // 2)
                continue
            else:
                # unknown runtime error, re raise after cleanup
                if rank == 0:
                    print(f"Runtime error during run {cfg['name']}: {e}")
                raise

    if not successful and last_exception is not None:
        # top level warning log
        print(f"Model {cfg['name']} failed, last exception: {repr(last_exception)}")

# MAIN
if __name__ == "__main__":
    distributed, rank, world_size, local_rank = init_ddp()
    train_files, val_files = prepare_data()

    ARCHS = [
        "nnunet_style",
        "attention_all_decoder_unet",
        "unetr",
        "swin_unetr",
    ]
    CONFIGS = []
    for a in ARCHS:
        CONFIGS.append({"name": f"{a}_dice_ce", "model": a, "loss": "dice_ce"})
        CONFIGS.append({"name": f"{a}_hybrid_dice_ce_focal_tversky", "model": a, "loss": "hybrid_dice_ce_focal_tversky"})

    for cfg in CONFIGS:
        train_one(cfg, train_files, val_files, GLOBAL_BATCH_SIZE_TRAIN, distributed, rank, world_size, local_rank)
        torch.cuda.empty_cache()

    if distributed and dist.is_initialized():
        dist.destroy_process_group()
    print("All runs completed.")