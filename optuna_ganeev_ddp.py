#!/usr/bin/env python3
"""
DDP-ready training script for MAMA-MIA segmentation with Optuna hyperparameter search.

Key fixes applied per option 1:
- Use Optuna ask/tell pattern on rank 0 to propose trials, then broadcast sampled params to all ranks
- Avoid study.add_trial(create_trial(...)) which fails when storage already has distributions
- Save sampled params per trial for reproducibility
- Keep plotting helpers overwriting the "current" files by default
- All ranks run trial training, rank 0 records trial results with study.tell
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

# optuna
try:
    import optuna
except Exception:
    optuna = None

# ------------------- Environment and memory tweaks -------------------
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ------------------- DDP helpers -------------------

def setup_ddp():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_ddp():
    try:
        dist.barrier()
    except Exception:
        pass
    try:
        dist.destroy_process_group()
    except Exception:
        pass

# ------------------- Config -------------------
BASE_DIR = "/hostd/mama_mia_dataset/MAMA-MIA"
OUTPUT_DIR = "/hostd/mama_mia_output_ddp"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_PATIENTS = int(os.environ.get("MAX_PATIENTS", 1506))
MAX_EPOCHS = int(os.environ.get("MAX_EPOCHS", 500))
VAL_INTERVAL = int(os.environ.get("VAL_INTERVAL", 1))
PER_GPU_BATCH = int(os.environ.get("PER_GPU_BATCH", 1))
LR = float(os.environ.get("LR", 1e-4))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", 1e-5))
CHECKPOINT_FREQ = int(os.environ.get("CHECKPOINT_FREQ", 5))

PHASE1_TRIALS = int(os.environ.get("PHASE1_TRIALS", 12))
PHASE2_TRIALS = int(os.environ.get("PHASE2_TRIALS", 24))
TRIAL_EPOCHS = int(os.environ.get("TRIAL_EPOCHS", 2))

STUDY_DB = os.path.join(OUTPUT_DIR, "optuna_studies.sqlite")

# ------------------- Model pieces -------------------
class SCSEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        reduced = max(1, in_channels // reduction)
        self.channel_excitation = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, reduced, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(reduced, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.spatial_excitation = nn.Sequential(
            nn.Conv3d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        chn = self.channel_excitation(x) * x
        spa = self.spatial_excitation(x) * x
        return chn + spa


class AttentionAllDecoderUNet(FlexibleUNet):
    def __init__(self, *args, **kwargs):
        self._user_decoder_channels = kwargs.get("decoder_channels", None)
        super().__init__(*args, **kwargs)
        self.attention_blocks = nn.ModuleList()
        for _ in range(len(self.decoder.blocks)):
            self.attention_blocks.append(nn.Identity())
    def forward(self, x):
        features = self.encoder(x)
        skips = [f for f in features[:-1] if f is not None][::-1]
        x = features[-1]
        for i, decoder_block in enumerate(self.decoder.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
            if isinstance(self.attention_blocks[i], nn.Identity):
                ch = x.shape[1]
                att = SCSEBlock(ch)
                att.to(x.device)
                self.attention_blocks[i] = att
            att_mod = self.attention_blocks[i]
            if not isinstance(att_mod, nn.Identity):
                x = att_mod(x)
        x = self.segmentation_head(x)
        return x

# ------------------- Data utils -------------------

def prepare_data(max_patients=MAX_PATIENTS):
    images = os.path.join(BASE_DIR, "images")
    labels = os.path.join(BASE_DIR, "segmentations", "expert")
    ids = sorted([d for d in os.listdir(images) if os.path.isdir(os.path.join(images, d))])[:max_patients]
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
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1, 1, 1), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=0, a_max=1555, b_min=0, b_max=1, clip=True),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadD(keys=["image", "label"], spatial_size=(128, 128, 96)),
        DivisiblePadd(keys=["image", "label"], k=32),
        RandCropByPosNegLabeld(keys=["image", "label"], label_key="label",
                               spatial_size=(128, 128, 96), pos=1, neg=1, num_samples=4),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0, 1, 2]),
        RandRotate90d(keys=["image", "label"], prob=0.5),
        RandGaussianNoised(keys=["image"], prob=0.3),
        EnsureTyped(keys=["image", "label"], track_meta=False)
    ])


def get_val_tf():
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
        EnsureTyped(keys=["image", "label"], track_meta=False)
    ])

# ------------------- Plotting helpers (overwrites current files by default) -------------------

def save_training_curves(epochs, loss_values, train_dices, val_dices, outdir=OUTPUT_DIR, epoch=None):
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
    title_epoch = epoch if epoch is not None else (epochs[-1] if len(epochs) > 0 else 'current')
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

# ------------------- Build model and optimizer -------------------

def build_model_and_optim(params, device):
    model = AttentionAllDecoderUNet(
        in_channels=1,
        out_channels=1,
        backbone="resnet50",
        pretrained=False,
        decoder_channels=(512, 256, 128, 64, 32),
        spatial_dims=3,
        norm=('group', {'num_groups': 8, 'affine': True}),  # Use GroupNorm for DDP safety
        act=('leakyrelu', {'inplace': True, 'negative_slope': 0.01}),
        dropout=params.get('dropout', 0.2),
        decoder_bias=False,
        upsample='deconv',
        interp_mode='trilinear',
        is_pad=False
    ).to(device)
    opt = AdamW(model.parameters(), lr=params.get('lr', LR), weight_decay=params.get('weight_decay', WEIGHT_DECAY))
    loss_fn = DiceFocalLoss(
        include_background=False,
        sigmoid=True,
        squared_pred=True,
        lambda_dice=params.get('lambda_dice', 1.0),
        lambda_focal=params.get('lambda_focal', 1.0),
        gamma=params.get('gamma', 2.0),
        alpha=params.get('alpha', 0.7),
        reduction="mean",
    )
    return model, opt, loss_fn

# ------------------- Training runner -------------------

def run_training_for_params(params, epochs, rank, world_size, local_rank, train_files, val_files, write_outputs=False):
    device = torch.device('cuda', local_rank)
    model, opt, loss_fn = build_model_and_optim(params, device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    train_ds = Dataset(train_files, get_train_tf())
    val_ds = Dataset(val_files, get_val_tf())
    train_sampler = DistributedSampler(train_ds)
    val_sampler = DistributedSampler(val_ds, shuffle=False)
    train_loader = DataLoader(train_ds, batch_size=PER_GPU_BATCH, sampler=train_sampler,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, sampler=val_sampler, num_workers=1, pin_memory=True)
    train_metric = DiceMetric(include_background=False, reduction='mean')
    val_metric = DiceMetric(include_background=False, reduction='mean')
    best_val_local = -1.0
    epochs_list = []
    loss_values = []
    train_dice_scores = []
    val_dice_scores = []
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        train_metric.reset()
        epoch_loss = 0.0
        for batch in train_loader:
            x = batch['image'].to(device)
            y = batch['label'].to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=True):
                out = model(x)
                loss = loss_fn(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            epoch_loss += float(loss.item())
            with torch.no_grad():
                preds = torch.sigmoid(out)
                preds_bin = (preds > 0.5).float()
                train_metric(preds_bin, y)
            del preds, preds_bin, out, x, y
        loss_tensor = torch.tensor(epoch_loss, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        epoch_loss_reduced = loss_tensor.item() / world_size
        train_val = train_metric.aggregate().item()
        t_tensor = torch.tensor(train_val, device=device)
        dist.all_reduce(t_tensor, op=dist.ReduceOp.SUM)
        train_val_reduced = t_tensor.item() / world_size
        val_score_reduced = None
        sample = None
        if (epoch + 1) % VAL_INTERVAL == 0:
            model.eval()
            val_metric.reset()
            with torch.no_grad():
                for batch in val_loader:
                    x = batch['image'].to(device)
                    y = batch['label'].to(device)
                    p = torch.sigmoid(model(x))
                    pb = (p > 0.5).float()
                    val_metric(pb, y)
                    if sample is None and dist.get_rank() == 0:
                        sample = (x.cpu().numpy(), pb.cpu().numpy(), y.cpu().numpy())
                    del x, y, p, pb
            val_score = val_metric.aggregate().item()
            vs = torch.tensor(val_score, device=device)
            dist.all_reduce(vs, op=dist.ReduceOp.SUM)
            val_score_reduced = vs.item() / world_size
        # Always append, even if not a val interval, to keep lists aligned
        epochs_list.append(epoch + 1)
        loss_values.append(epoch_loss_reduced)
        train_dice_scores.append(train_val_reduced)
        # If val_score_reduced is None (not a val interval), repeat last or use np.nan
        if val_score_reduced is not None:
            val_dice_scores.append(val_score_reduced)
        else:
            val_dice_scores.append(val_dice_scores[-1] if val_dice_scores else float('nan'))
        if write_outputs and dist.get_rank() == 0:
            save_training_curves(epochs_list, loss_values, train_dice_scores, val_dice_scores, epoch=None)
            if sample is not None:
                save_prediction_sample(sample[0][0, 0], sample[1][0, 0], sample[2][0, 0], epoch=None)
            if val_score_reduced is not None and val_score_reduced > best_val_local:
                best_val_local = val_score_reduced
                torch.save(model.module.state_dict(), os.path.join(OUTPUT_DIR, 'best_model.pth'))
    model.eval()
    val_metric.reset()
    with torch.no_grad():
        for batch in val_loader:
            x = batch['image'].to(device)
            y = batch['label'].to(device)
            p = torch.sigmoid(model(x))
            pb = (p > 0.5).float()
            val_metric(pb, y)
            del x, y, p, pb
    val_score = val_metric.aggregate().item()
    vs = torch.tensor(val_score, device=device)
    dist.all_reduce(vs, op=dist.ReduceOp.SUM)
    val_score_reduced = vs.item() / world_size
    # --- Explicitly delete all CUDA objects and force garbage collection ---
    try:
        del model, opt, loss_fn, scaler, train_loader, val_loader, train_ds, val_ds, train_sampler, val_sampler, train_metric, val_metric
    except Exception:
        pass
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    return float(val_score_reduced)

# ------------------- Optuna orchestration (ask/tell) -------------------

def ddp_two_phase_optuna(rank, world_size, local_rank, train_files, val_files):
    if optuna is None:
        if rank == 0:
            print("Optuna not available, skipping hyperparameter search.")
        return None
    storage_url = f"sqlite:///{STUDY_DB}"
    if rank == 0:
        print(f"[Optuna] Using storage: {storage_url}")

    def broadcast_params(params):
        obj = [params]
        dist.broadcast_object_list(obj, src=0)
        return obj[0]

    # Phase 1 study creation
    if rank == 0:
        study1 = optuna.create_study(study_name="phase1_base", storage=storage_url, load_if_exists=True, direction="maximize")
    else:
        study1 = None

    for t in range(PHASE1_TRIALS):
        if rank == 0:
            trial = study1.ask()
            p = {}
            p['lr'] = trial.suggest_float('lr', 1e-5, 5e-4, log=True)
            p['weight_decay'] = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)
            p['dropout'] = trial.suggest_float('dropout', 0.0, 0.4)
            # save sampled params
            with open(os.path.join(OUTPUT_DIR, f"phase1_trial_{t}.json"), "w") as fh:
                json.dump(p, fh)
        else:
            p = None
        p = broadcast_params(p)
        val_score = run_training_for_params(p, TRIAL_EPOCHS, rank, world_size, local_rank, train_files, val_files, write_outputs=False)
        if rank == 0:
            study1.tell(trial, val_score)

    if rank == 0:
        try:
            best1_trial = study1.best_trial
            best1 = best1_trial.params
        except Exception:
            bests = []
            for f in os.listdir(OUTPUT_DIR):
                if f.startswith("phase1_trial_") and f.endswith('.json'):
                    with open(os.path.join(OUTPUT_DIR, f)) as fh:
                        bests.append(json.load(fh))
            best1 = bests[0] if bests else {"lr": LR, "weight_decay": WEIGHT_DECAY, "dropout": 0.2}
        print(f"[Optuna] Phase 1 best params: {best1}")
    else:
        best1 = None
    best1 = broadcast_params(best1)

    # Phase 2 study creation
    if rank == 0:
        study2 = optuna.create_study(study_name="phase2_loss", storage=storage_url, load_if_exists=True, direction="maximize")
    else:
        study2 = None

    for t in range(PHASE2_TRIALS):
        if rank == 0:
            trial2 = study2.ask()
            p2 = {}
            p2['lambda_dice'] = trial2.suggest_float('lambda_dice', 0.5, 2.0)
            p2['lambda_focal'] = trial2.suggest_float('lambda_focal', 0.0, 2.0)
            p2['gamma'] = trial2.suggest_float('gamma', 1.0, 4.0)
            p2['alpha'] = trial2.suggest_float('alpha', 0.3, 0.9)
            p_comb = dict(best1)
            p_comb.update(p2)
            with open(os.path.join(OUTPUT_DIR, f"phase2_trial_{t}.json"), "w") as fh:
                json.dump(p_comb, fh)
        else:
            p_comb = None
        p_comb = broadcast_params(p_comb)
        val_score = run_training_for_params(p_comb, TRIAL_EPOCHS, rank, world_size, local_rank, train_files, val_files, write_outputs=False)
        if rank == 0:
            study2.tell(trial2, val_score)

    if rank == 0:
        try:
            best2 = study2.best_trial.params
        except Exception:
            bests = []
            for f in os.listdir(OUTPUT_DIR):
                if f.startswith("phase2_trial_") and f.endswith('.json'):
                    with open(os.path.join(OUTPUT_DIR, f)) as fh:
                        bests.append(json.load(fh))
            best2 = bests[0] if bests else best1
        print(f"[Optuna] Phase 2 best params (combined): {best2}")
    else:
        best2 = None
    best2 = broadcast_params(best2)
    return best2

# ------------------- Training entrypoint with Optuna orchestration -------------------

def train():
    rank, world_size, local_rank = setup_ddp()
    device = torch.device('cuda', local_rank)
    try:
        free_mem, total_mem = torch.cuda.mem_get_info(device)
        free_gb = free_mem / (1024 ** 3)
        if free_gb < 6 and rank == 0:
            print(f"[Warning] free GPU memory low: {free_gb:.2f} GiB. Consider reducing PER_GPU_BATCH.")
    except Exception:
        pass
    if rank == 0:
        print(f'Running DDP on world_size={world_size}, per-gpu-batch={PER_GPU_BATCH}')
    train_files, val_files = prepare_data()
    best_params = None
    if optuna is not None:
        best_params = ddp_two_phase_optuna(rank, world_size, local_rank, train_files, val_files)
    if best_params is None:
        best_params = {
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "dropout": 0.2,
            "lambda_dice": 1.0,
            "lambda_focal": 1.0,
            "gamma": 2.0,
            "alpha": 0.7
        }
    final_val = run_training_for_params(best_params, MAX_EPOCHS, rank, world_size, local_rank, train_files, val_files, write_outputs=True)
    if rank == 0:
        print(f"[Final] Completed full training with best params. Final validation dice: {final_val:.4f}")
    cleanup_ddp()


if __name__ == '__main__':
    train()
