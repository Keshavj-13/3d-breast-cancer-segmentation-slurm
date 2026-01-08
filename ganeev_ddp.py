#!/usr/bin/env python3
"""
DDP-ready training script for MAMA-MIA segmentation
Features added or changed:
- Proper DDP setup using environment provided by torchrun
- Mixed precision training (automatic mixed precision)
- Per-GPU batch size and distributed sampler
- Memory fragmentation mitigations via PYTORCH_CUDA_ALLOC_CONF
- Per-epoch numpy and PNG exports (many plots + npy preserved)
- Checkpointing and best-model saving from rank 0
- Reduced default per-GPU batch size to avoid OOM
- Useful logging only from rank 0
"""

import os
import sys
import time
import json
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    RandGaussianNoised, RandGaussianSmoothd, RandAdjustContrastd,
    CropForegroundd, RandCropByPosNegLabeld, DivisiblePadd,
    RandFlipd, RandRotate90d, RandAffined, SpatialPadD
)
from monai.data import DataLoader, Dataset
from monai.networks.nets import FlexibleUNet
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric
from monai.utils import set_determinism

from tqdm import tqdm

# ------------------- Environment and memory tweaks -------------------
# Try to reduce fragmentation and encourage larger allocation granularity
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Optional: uncomment the next line if fragmentation still happens
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"

# ------------------- DDP helpers -------------------

def setup_ddp():
    """Initialise process group and set local device."""
    # torchrun / torch.distributed.run will provide RANK, LOCAL_RANK, WORLD_SIZE
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # default backend for NCCL GPUs
    dist.init_process_group(backend="nccl")

    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_ddp():
    try:
        dist.barrier()
    except Exception:
        pass
    dist.destroy_process_group()


# ------------------- Config -------------------
BASE_DIR = "/hostd/mama_mia_dataset/MAMA-MIA"
OUTPUT_DIR = "/hostd/mama_mia_output_ddp"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_PATIENTS = 100
MAX_EPOCHS = 500
VAL_INTERVAL = 1
PER_GPU_BATCH = 1  # conservative default to avoid OOM; increase if you have room
LR = 1e-4
WEIGHT_DECAY = 1e-5
CHECKPOINT_FREQ = 5

# ------------------- Model pieces -------------------
class SCSEBlock(nn.Module):
    """Spatial and Channel Squeeze and Excitation block, safely built for any channel count."""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        reduced = max(1, in_channels // reduction)
        # Channel squeeze and excitation
        self.channel_excitation = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, reduced, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(reduced, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        # Spatial squeeze and excitation
        self.spatial_excitation = nn.Sequential(
            nn.Conv3d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        chn = self.channel_excitation(x) * x
        spa = self.spatial_excitation(x) * x
        return chn + spa


class AttentionAllDecoderUNet(FlexibleUNet):
    """
    FlexibleUNet with attention applied at decoder outputs.
    Attention modules are created lazily on first forward pass using the
    actual channel dimensionality of the decoder outputs.
    This avoids channel mismatch when MONAI internals reorder or reduce channels.
    """
    def __init__(self, *args, **kwargs):
        # preserve any decoder_channels provided by user, but do not rely on them
        # for strict sizing. We will create attention modules from tensor shapes.
        self._user_decoder_channels = kwargs.get("decoder_channels", None)
        super().__init__(*args, **kwargs)
        # placeholder ModuleList where modules will be set on first forward
        self.attention_blocks = nn.ModuleList()
        # create empty slots matching decoder.blocks count, so registered in state dict
        for _ in range(len(self.decoder.blocks)):
            # use Identity placeholder until we know actual channel count
            self.attention_blocks.append(nn.Identity())

    def forward(self, x):
        features = self.encoder(x)
        skips = [f for f in features[:-1] if f is not None][::-1]
        x = features[-1]

        for i, decoder_block in enumerate(self.decoder.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

            # If the stored attention module is Identity, replace it with a properly sized SCSEBlock
            if isinstance(self.attention_blocks[i], nn.Identity):
                ch = x.shape[1]
                att = SCSEBlock(ch)
                # move module to same device and register it
                att.to(x.device)
                # replace the Identity with the real module, preserves registration
                self.attention_blocks[i] = att

            # apply attention only if module is callable
            att_mod = self.attention_blocks[i]
            if not isinstance(att_mod, nn.Identity):
                x = att_mod(x)

        x = self.segmentation_head(x)
        return x


# ------------------- Data utils -------------------

def prepare_data(max_patients=MAX_PATIENTS):
    images = os.path.join(BASE_DIR, "images")
    labels = os.path.join(BASE_DIR, "segmentations", "expert")
    ids = sorted([d for d in os.listdir(images) if os.path.isdir(os.path.join(images,d))])[:max_patients]

    files = []
    for pid in ids:
        img = os.path.join(images, pid, f"{pid}_0002.nii")
        lbl = os.path.join(labels, f"{pid}.nii")
        if os.path.exists(img) and os.path.exists(lbl):
            files.append({"image": img, "label": lbl})
    train, val = train_test_split(files, test_size=0.2, random_state=42)
    return train, val


def get_train_tf():
    return Compose([
        LoadImaged(keys=["image","label"]),
        EnsureChannelFirstd(keys=["image","label"]),
        Spacingd(keys=["image","label"], pixdim=(1,1,1), mode=("bilinear","nearest")),
        Orientationd(keys=["image","label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=0, a_max=1555, b_min=0, b_max=1, clip=True),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image","label"], source_key="image"),
        SpatialPadD(keys=["image","label"], spatial_size=(128,128,96)),
        DivisiblePadd(keys=["image","label"], k=32),
        RandCropByPosNegLabeld(keys=["image","label"], label_key="label",
                               spatial_size=(128,128,96), pos=1, neg=1, num_samples=4),
        RandFlipd(keys=["image","label"], prob=0.5, spatial_axis=[0,1,2]),
        RandRotate90d(keys=["image","label"], prob=0.5),
        RandGaussianNoised(keys=["image"], prob=0.3),
        EnsureTyped(keys=["image","label"]) 
    ])


def get_val_tf():
    return Compose([
        LoadImaged(keys=["image","label"]),
        EnsureChannelFirstd(keys=["image","label"]),
        Spacingd(keys=["image","label"], pixdim=(1,1,1), mode=("bilinear","nearest")),
        Orientationd(keys=["image","label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=0, a_max=1555, b_min=0, b_max=1, clip=True),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image","label"], source_key="image"),
        SpatialPadD(keys=["image","label"], spatial_size=(128,128,96)),
        DivisiblePadd(keys=["image","label"], k=32),
        EnsureTyped(keys=["image","label"]) 
    ])


# ------------------- Plotting and save utilities -------------------

def save_training_curves(epochs, loss_values, train_dices, val_dices,
                         outdir=OUTPUT_DIR, epoch=None):

    if dist.is_initialized() and dist.get_rank() != 0:
        return

    os.makedirs(outdir, exist_ok=True)

    np.save(os.path.join(outdir, 'epochs.npy'), np.array(epochs))
    np.save(os.path.join(outdir, 'loss_values.npy'), np.array(loss_values))
    np.save(os.path.join(outdir, 'train_dice_scores.npy'), np.array(train_dices))
    np.save(os.path.join(outdir, 'val_dice_scores.npy'), np.array(val_dices))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))

    ax1_twin = ax1.twinx()
    ax1.plot(epochs, loss_values, color='tab:red', linewidth=2, label='Train Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.grid(True, alpha=0.3)

    ax1_twin.plot(epochs, val_dices, color='tab:blue', linewidth=2, label='Val Dice')
    ax1_twin.set_ylabel('Validation Dice', color='tab:blue')
    ax1_twin.tick_params(axis='y', labelcolor='tab:blue')

    title_epoch = epoch if epoch is not None else epochs[-1]
    ax1.set_title(f'Training Progress, Epoch {title_epoch}', fontweight='bold')

    ax2.plot(epochs, train_dices, color='tab:green', linestyle='--', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Training Dice')
    ax2.set_title('Training Dice Score', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    fname = 'training_curves_current.png' if epoch is None else f'training_curves_epoch_{epoch}.png'
    plt.savefig(os.path.join(outdir, fname), dpi=150, bbox_inches='tight')
    plt.close()

def save_prediction_sample(img, pred, gt, outdir=OUTPUT_DIR, epoch=None):

    if dist.is_initialized() and dist.get_rank() != 0:
        return

    os.makedirs(outdir, exist_ok=True)

    if img.ndim == 4:
        img = img[0]
    if pred.ndim == 4:
        pred = pred[0]
    if gt.ndim == 4:
        gt = gt[0]

    zidx = int(np.argmax(gt.sum(axis=(1, 2))))

    base = f'sample_epoch_{epoch}' if epoch is not None else 'sample_current'
    np.save(os.path.join(outdir, base + '_img.npy'), img)
    np.save(os.path.join(outdir, base + '_pred.npy'), pred)
    np.save(os.path.join(outdir, base + '_gt.npy'), gt)

    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(img[zidx], cmap='gray')
    plt.title('Input Image', fontweight='bold')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(img[zidx], cmap='gray')
    plt.imshow(gt[zidx], cmap='Greens', alpha=0.5)
    plt.title('Ground Truth', fontweight='bold')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(img[zidx], cmap='gray')
    plt.imshow(pred[zidx], cmap='Reds', alpha=0.5)
    plt.title('Prediction', fontweight='bold')
    plt.axis('off')

    plt.suptitle(f'Segmentation @ Epoch {epoch}', fontweight='bold')
    plt.tight_layout()

    fname = 'prediction_current.png' if epoch is None else f'prediction_epoch_{epoch}.png'
    plt.savefig(os.path.join(outdir, fname), dpi=150, bbox_inches='tight')
    plt.close()


# ------------------- Training loop -------------------

def train():
    rank, world_size, local_rank = setup_ddp()
    device = torch.device('cuda', local_rank)

    free_mem, total_mem = torch.cuda.mem_get_info(device)
    free_gb = free_mem / (1024 ** 3)

    if free_gb < 10:
        raise RuntimeError(
            f"Insufficient free GPU memory on rank {rank}: {free_gb:.2f} GiB available"
        )

    if rank == 0:
        print(f'Running DDP on world_size={world_size}, per-gpu-batch={PER_GPU_BATCH}')

    train_files, val_files = prepare_data()

    train_ds = Dataset(train_files, get_train_tf())
    val_ds = Dataset(val_files, get_val_tf())

    train_sampler = DistributedSampler(train_ds)
    val_sampler = DistributedSampler(val_ds, shuffle=False)

    train_loader = DataLoader(train_ds, batch_size=PER_GPU_BATCH, sampler=train_sampler,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, sampler=val_sampler,
                            num_workers=1, pin_memory=True)

    # instantiate model
    model = AttentionAllDecoderUNet(
        in_channels=1,
        out_channels=1,
        backbone="resnet50",
        pretrained=False,
        decoder_channels=(512, 256, 128, 64, 32),
        spatial_dims=3,
        norm=('instance', {'affine': True}),
        act=('leakyrelu', {'inplace': True, 'negative_slope': 0.01}),
        dropout=0.2,
        decoder_bias=False,
        upsample='deconv',
        interp_mode='trilinear',
        is_pad=False
    ).to(device)

    # wrap
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    loss_fn = DiceFocalLoss(
        include_background=False,
        sigmoid=True,
        squared_pred=True,
        lambda_dice=1.0,
        lambda_focal=1.0,
        gamma=2.0,
        alpha=0.7,
        reduction="mean",
    )
    opt = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    train_dice_metric = DiceMetric(include_background=False, reduction='mean')
    val_dice_metric = DiceMetric(include_background=False, reduction='mean')

    best_val = -1.0

    # Only rank 0 collects and writes epoch-level summaries
    if rank == 0:
        epochs_list, loss_list, train_dices, val_dices = [], [], [], []

    for epoch in range(MAX_EPOCHS):
        train_sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        train_dice_metric.reset()

        for batch in tqdm(train_loader, desc=f'[Rank {rank}] Train E{epoch+1}', disable=(rank!=0)):
            x = batch['image'].to(device)
            y = batch['label'].to(device)

            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=True):
                out = model(x)
                loss = loss_fn(out, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            epoch_loss += loss.item()

            # update training dice metric, use sigmoid then binarize
            with torch.no_grad():
                preds = torch.sigmoid(out)
                preds_bin = (preds > 0.5).float()
                train_dice_metric(preds_bin, y)
            del preds, preds_bin, out, x, y

        # reduce epoch loss and training dice across workers
        loss_tensor = torch.tensor(epoch_loss, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        epoch_loss_reduced = loss_tensor.item() / world_size

        train_dice_val = train_dice_metric.aggregate().item()
        td_tensor = torch.tensor(train_dice_val, device=device)
        dist.all_reduce(td_tensor, op=dist.ReduceOp.SUM)
        train_dice_reduced = td_tensor.item() / world_size

        if rank == 0:
            epochs_list.append(epoch+1)
            loss_list.append(epoch_loss_reduced)
            train_dices.append(train_dice_reduced)

        # Validation
        if (epoch + 1) % VAL_INTERVAL == 0:
            model.eval()
            val_dice_metric.reset()
            sample = None
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
                for batch in tqdm(val_loader, desc=f'[Rank {rank}] Val E{epoch+1}', disable=(rank!=0)):
                    x = batch['image'].to(device, non_blocking=True)
                    y = batch['label'].to(device, non_blocking=True)

                    p = torch.sigmoid(model(x))
                    pb = (p > 0.5).float()
                    val_dice_metric(pb, y)

                    if sample is None and rank == 0:
                        sample = (x.cpu().numpy(), pb.cpu().numpy(), y.cpu().numpy())

                    # explicit cleanup
                    del x, y, p, pb
                    torch.cuda.empty_cache()

            val_score = val_dice_metric.aggregate().item()
            vs_tensor = torch.tensor(val_score, device=device)
            dist.all_reduce(vs_tensor, op=dist.ReduceOp.SUM)
            val_score_reduced = vs_tensor.item() / world_size

            # rank 0 writes plots, checkpoints
            if rank == 0:
                val_dices.append(val_score_reduced)
                # save per-epoch arrays and PNGs
                save_training_curves(epochs_list, loss_list, train_dices, val_dices, epoch=epoch+1)
                if sample is not None:
                    # sample[0] shape: (B,C,Z,Y,X)
                    save_prediction_sample(sample[0][0,0], sample[1][0,0], sample[2][0,0], epoch=epoch+1)

                # checkpoint
                ckpt = {
                    'epoch': epoch+1,
                    'state_dict': model.module.state_dict(),
                    'optimizer': opt.state_dict(),
                    'scaler': scaler.state_dict(),
                    'loss_history': loss_list,
                    'train_dice_history': train_dices,
                    'val_dice_history': val_dices
                }
                torch.save(ckpt, os.path.join(OUTPUT_DIR, f'checkpoint_epoch_{epoch+1}.pth'))

                # keep best model
                if val_score_reduced > best_val:
                    best_val = val_score_reduced
                    torch.save(model.module.state_dict(), os.path.join(OUTPUT_DIR, 'best_model.pth'))
                    print(f'[Rank 0] Saved new best model at epoch {epoch+1}, val={best_val:.4f}')

        # save periodic checkpoint even if not validation epoch
        if rank == 0 and ((epoch+1) % CHECKPOINT_FREQ == 0 and (epoch+1) % VAL_INTERVAL != 0):
            torch.save({'epoch': epoch+1, 'state_dict': model.module.state_dict()},
                       os.path.join(OUTPUT_DIR, f'checkpoint_epoch_{epoch+1}.pth'))

    # final save
    if rank == 0:
        torch.save(model.module.state_dict(), os.path.join(OUTPUT_DIR, 'final_model.pth'))
        # final comprehensive files
        save_training_curves(epochs_list, loss_list, train_dices, val_dices, epoch='final')

    cleanup_ddp()


if __name__ == '__main__':
    train()
