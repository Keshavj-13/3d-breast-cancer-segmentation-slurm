#!/usr/bin/env python3
"""
hybrid_diffusion_ddp_full.py

DDP ready training script for MAMA MIA, implementing 5 channel DCE prior
plus diffusion refiner that conditions on the 5 channel image and noisy prior.
Uses MONAI DiceMetric correctly in distributed training using Method A.
"""

import os
import sys
import time
import json
import math
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
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from sklearn.model_selection import train_test_split

from monai.transforms import (
    Compose, EnsureTyped, ScaleIntensityRanged, EnsureChannelFirstd,
    SpatialPadD, DivisiblePadd, CropForegroundd
)
from monai.networks.nets import FlexibleUNet, UNet
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric
from monai.utils import set_determinism

from tqdm import tqdm

# -------------------------
# CONFIG
# -------------------------
BASE_DIR = "/hostd/mama_mia_dataset/MAMA-MIA"   # set to your dataset root
OUTPUT_DIR = "/hostd/mama_mia_output_5phase_ddp"  # set to desired outputs
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_PATIENTS = 500
MAX_EPOCHS = 500
VAL_INTERVAL = 1
PER_GPU_BATCH = 1        # per GPU, increase if memory allows
NUM_WORKERS = 4
LR = 1e-4
WEIGHT_DECAY = 1e-5
CHECKPOINT_FREQ = 5
CACHE_RATE = 0.0         # 0.0 means no caching; tune 0.1 or 1.0 if you want CacheDataset
REQUIRED_PHASES = 3      # number of DCE phases expected
RNG_SEED = 42


# memory fragmentation mitigation recommended
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# deterministic behavior for reproducibility
set_determinism(RNG_SEED)

# Set NUM_WORKERS = 0 to debug DataLoader exceptions in main process

def worker_init_fn(worker_id):
    base = RNG_SEED
    rank = dist.get_rank() if dist.is_initialized() else 0
    np.random.seed(base + worker_id + rank)


# -------------------------
# DDP helpers
# -------------------------
def setup_ddp():
    """Initialize process group and set local CUDA device."""
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
    dist.destroy_process_group()


def is_rank0():
    return dist.is_initialized() and dist.get_rank() == 0


# -------------------------
# Dataset and IO
# -------------------------
def discover_records(base_dir, max_patients=MAX_PATIENTS):
    """
    Build list of records with explicit, validated paths.
    Each record dict contains
      - patient_id
      - phase_paths: list of 5 explicit phase paths in correct order
      - label_path
    Raises FileNotFoundError if any expected file is missing.
    """
    images_dir = os.path.join(base_dir, "images")
    labels_dir = os.path.join(base_dir, "segmentations", "expert")

    patient_ids = sorted([d for d in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, d))])
    patient_ids = patient_ids[:max_patients]

    records = []
    missing = []
    for pid in patient_ids:
        phase_paths = []
        for t in range(REQUIRED_PHASES):
            p = os.path.join(images_dir, pid, f"{pid}_000{t}.nii")
            if not os.path.exists(p):
                missing.append(p)
            phase_paths.append(p)
        label_path = os.path.join(labels_dir, f"{pid}.nii")
        if not os.path.exists(label_path):
            missing.append(label_path)
        records.append({"patient_id": pid, "phase_paths": phase_paths, "label_path": label_path})

    if len(missing) > 0:
        # Fail loudly, list same missing on rank 0
        msg = "Missing required files, aborting. Example missing paths:\n" + "\n".join(missing[:10])
        raise FileNotFoundError(msg)

    if is_rank0() or "RANK" not in os.environ:
        print(f"Discovered {len(records)} patient records, each with {REQUIRED_PHASES} phases.")
    return records


class MAMAMIA5PhaseDataset(Dataset):
    """Loads five DCE phases as channels and the expert mask as single channel."""

    def __init__(self, records, transforms=None):
        self.records = records
        self.transforms = transforms

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        # load each phase explicitly from disk, preserving order 0000..0004
        phases = []
        for p in rec["phase_paths"]:
            arr = nib.load(p).get_fdata().astype(np.float32)
            phases.append(arr)
        image = np.stack(phases, axis=0)  # shape: C=5, Z, Y, X

        lbl = nib.load(rec["label_path"]).get_fdata().astype(np.float32)
        lbl = (lbl > 0.5).astype(np.float32)[None]  # shape: 1, Z, Y, X

        sample = {"image": image, "label": lbl}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample


# -------------------------
# Model building blocks
# -------------------------
class SCSEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        reduced = max(1, in_channels // reduction)
        self.ch = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, reduced, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(reduced, in_channels, 1),
            nn.Sigmoid()
        )
        self.sp = nn.Sequential(
            nn.Conv3d(in_channels, 1, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.ch(x) + x * self.sp(x)


class AttentionAllDecoderUNet(FlexibleUNet):
    """
    FlexibleUNet with SCSE attention applied at decoder outputs.
    Accepts arbitrary in_channels via args.
    """
    def __init__(self, *args, **kwargs):
        # keep decoder_channels from kwargs if present
        self._user_decoder_channels = kwargs.get("decoder_channels", None)
        super().__init__(*args, **kwargs)
        # build attention list lazily based on decoder blocks
        self.attention_blocks = nn.ModuleList()
        for _ in range(len(self.decoder.blocks)):
            # placeholder identity, will be replaced on first forward if sizes are uncertain
            self.attention_blocks.append(nn.Identity())

    def forward(self, x):
        features = self.encoder(x)
        skips = [f for f in features[:-1] if f is not None][::-1]
        x = features[-1]
        for i, decoder_block in enumerate(self.decoder.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
            # replace identity with correct sized SCSE when we can inspect x channels
            if isinstance(self.attention_blocks[i], nn.Identity):
                ch = x.shape[1]
                att = SCSEBlock(ch)
                att.to(x.device)
                self.attention_blocks[i] = att
            if not isinstance(self.attention_blocks[i], nn.Identity):
                x = self.attention_blocks[i](x)
        x = self.segmentation_head(x)
        return x


# -------------------------
# Diffusion utilities
# -------------------------
def linear_beta_schedule(T):
    return torch.linspace(1e-4, 2e-2, T)

class DiffusionProcess:
    def __init__(self, timesteps, device):
        self.timesteps = timesteps
        betas = linear_beta_schedule(timesteps).to(device)
        alphas = 1.0 - betas
        self.alpha_bar = torch.cumprod(alphas, dim=0)

    def q_sample(self, x0, t, noise):
        # t is a vector of timesteps for each sample
        a = self.alpha_bar[t].view(-1, 1, 1, 1, 1)
        return torch.sqrt(a) * x0 + torch.sqrt(1.0 - a) * noise


# -------------------------
# Plotting and saving helpers
# -------------------------
def save_training_curves(epochs, loss_values, train_dices, val_dices, outdir, epoch=None):
    # only rank 0 writes
    if dist.is_initialized() and not is_rank0():
        return
    os.makedirs(outdir, exist_ok=True)
    np.save(os.path.join(outdir, "epochs.npy"), np.array(epochs))
    np.save(os.path.join(outdir, "loss_values.npy"), np.array(loss_values))
    np.save(os.path.join(outdir, "train_dice_scores.npy"), np.array(train_dices))
    np.save(os.path.join(outdir, "val_dice_scores.npy"), np.array(val_dices))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    ax1_twin = ax1.twinx()
    ax1.plot(epochs, loss_values, linewidth=2, label="Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax1_twin.plot(epochs, val_dices, linewidth=2, label="Val Dice")
    ax1_twin.set_ylabel("Val Dice")
    ax2.plot(epochs, train_dices, linestyle="--", linewidth=2, label="Train Dice")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Train Dice")
    title_epoch = epoch if epoch is not None else (epochs[-1] if len(epochs) else "NA")
    ax1.set_title(f"Training Progress, Epoch {title_epoch}")
    plt.tight_layout()
    fname = "training_curves_current.png" if epoch is None else f"training_curves_epoch_{epoch}.png"
    plt.savefig(os.path.join(outdir, fname), dpi=150, bbox_inches="tight")
    plt.close()

def save_prediction_sample(img, pred, gt, outdir, epoch=None):
    # only rank 0 writes
    if dist.is_initialized() and not is_rank0():
        return
    os.makedirs(outdir, exist_ok=True)

    # img pred gt expected as numpy arrays: C,Z,Y,X or Z,Y,X depending
    if img.ndim == 4:  # C,Z,Y,X
        img = img[0]
    if pred.ndim == 4:
        pred = pred[0]
    if gt.ndim == 4:
        gt = gt[0]

    zidx = int(np.argmax(gt.sum(axis=(1,2))))
    base = f"sample_epoch_{epoch}" if epoch is not None else "sample_current"
    np.save(os.path.join(outdir, base + "_img.npy"), img)
    np.save(os.path.join(outdir, base + "_pred.npy"), pred)
    np.save(os.path.join(outdir, base + "_gt.npy"), gt)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.imshow(img[zidx], cmap="gray"); plt.title("Input"); plt.axis("off")
    plt.subplot(1, 3, 2); plt.imshow(img[zidx], cmap="gray"); plt.imshow(gt[zidx], cmap="Greens", alpha=0.5); plt.title("GT"); plt.axis("off")
    plt.subplot(1, 3, 3); plt.imshow(img[zidx], cmap="gray"); plt.imshow(pred[zidx], cmap="Reds", alpha=0.5); plt.title("Pred"); plt.axis("off")
    plt.suptitle(f"Segmentation @ Epoch {epoch}")
    plt.tight_layout()
    fname = "prediction_current.png" if epoch is None else f"prediction_epoch_{epoch}.png"
    plt.savefig(os.path.join(outdir, fname), dpi=150, bbox_inches="tight")
    plt.close()


# -------------------------
# Main training function
# -------------------------
def train():
    rank, world_size, local_rank = setup_ddp()
    device = torch.device("cuda", local_rank)

    if is_rank0():
        print(f"Starting hybrid DDP training, world_size {world_size}, per gpu batch {PER_GPU_BATCH}")
        print(f"BASE_DIR: {BASE_DIR}")
        print(f"OUTPUT_DIR: {OUTPUT_DIR}")

    # discover and validate records
    try:
        records = discover_records(BASE_DIR, max_patients=MAX_PATIENTS)
    except FileNotFoundError as e:
        # show on rank 0 then re-raise to abort all processes
        if is_rank0():
            print(str(e))
        raise

    # split deterministic
    train_records, val_records = train_test_split(records, test_size=0.2, random_state=RNG_SEED)
    if is_rank0():
        print(f"Train {len(train_records)} val {len(val_records)}")

    # transforms: minimal here, keep heavy augmentations out of DDP seed complexity for now
    train_transforms = Compose([
        EnsureChannelFirstd(keys=["image", "label"], channel_dim=0),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=0.0,
            a_max=1555.0,
            b_min=0.0,
            b_max=1.0,
            clip=True
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadD(keys=["image", "label"], spatial_size=(128, 128, 96)),
        DivisiblePadd(keys=["image", "label"], k=32),
        EnsureTyped(keys=["image", "label"])
    ])
    val_transforms = train_transforms

    train_ds = MAMAMIA5PhaseDataset(train_records, transforms=train_transforms)
    val_ds   = MAMAMIA5PhaseDataset(val_records, transforms=val_transforms)

    train_sampler = DistributedSampler(train_ds, shuffle=True)
    val_sampler = DistributedSampler(val_ds, shuffle=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=PER_GPU_BATCH,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=worker_init_fn,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        sampler=val_sampler,
        num_workers=max(1, NUM_WORKERS//2),
        pin_memory=True,
        drop_last=False,
        worker_init_fn=worker_init_fn,
        persistent_workers=True
    )

    # build models
    prior = AttentionAllDecoderUNet(
        in_channels=REQUIRED_PHASES,
        out_channels=1,
        backbone="resnet50",
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

    refiner = UNet(
        spatial_dims=3,
        in_channels=REQUIRED_PHASES + 1,  # 5 image channels plus 1 noisy prior channel => 6
        out_channels=1,
        channels=(64, 128, 256),
        strides=(2, 2),
        num_res_units=1
    ).to(device)

    # wrap with DDP
    prior = DDP(prior, device_ids=[local_rank], find_unused_parameters=False)
    refiner = DDP(refiner, device_ids=[local_rank], find_unused_parameters=False)

    # loss, optimizer, scaler
    seg_loss_fn = DiceFocalLoss(include_background=False, sigmoid=True, squared_pred=True,
                                lambda_dice=1.0, lambda_focal=1.0, gamma=2.0, alpha=0.7, reduction="mean")
    mse_loss = nn.MSELoss()
    optimizer = AdamW(list(prior.parameters()) + list(refiner.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # MONAI DiceMetric objects; Method A approach used below
    train_metric = DiceMetric(include_background=False, reduction="mean")
    val_metric   = DiceMetric(include_background=False, reduction="mean")

    diffusion = DiffusionProcess(timesteps=1000, device=device)

    best_val = -1.0
    if is_rank0():
        epoch_list = []
        loss_history = []
        train_dice_history = []
        val_dice_history = []

    # training loop
    for epoch in range(MAX_EPOCHS):
        train_sampler.set_epoch(epoch)
        prior.train()
        refiner.train()

        epoch_loss = 0.0
        train_metric.reset()

        pbar = tqdm(train_loader, desc=f"[Rank {rank}] Train E{epoch+1}", disable=(not is_rank0()))
        for batch in pbar:
            x = batch["image"].to(device, non_blocking=True)  # shape B,5,Z,Y,X
            y = batch["label"].to(device, non_blocking=True)  # shape B,1,Z,Y,X

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=True):
                logits = prior(x)                    # B,1,Z,Y,X
                seg_loss = seg_loss_fn(logits, y)

                # diffusion training
                prior_prob = torch.sigmoid(logits).detach()  # condition
                bsz = prior_prob.shape[0]
                t = torch.randint(0, diffusion.timesteps, (bsz,), device=device, dtype=torch.long)
                noise = torch.randn_like(prior_prob)
                noisy_prior = diffusion.q_sample(prior_prob, t, noise)

                cond = torch.cat([x, noisy_prior], dim=1)   # B,6,Z,Y,X
                noise_pred = refiner(cond)
                diff_loss = mse_loss(noise_pred, noise)

                total_loss = seg_loss + 0.5 * diff_loss

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += float(total_loss.item())

            # update training dice metric with binarized predictions
            with torch.no_grad():
                preds_bin = (torch.sigmoid(logits) > 0.5).float()
                train_metric(preds_bin, y)

            # free some memory
            del x, y, logits, prior_prob, noisy_prior, cond, noise_pred, preds_bin, noise
            torch.cuda.empty_cache()

            if is_rank0():
                pbar.set_postfix(loss=total_loss.item())

        # aggregate epoch loss across ranks
        loss_tensor = torch.tensor(epoch_loss, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        epoch_loss_reduced = loss_tensor.item() / world_size

        # Method A: aggregate local metric, then all_reduce aggregated scalar
        try:
            local_train_dice = train_metric.aggregate().item()
        except Exception:
            # if metric empty, set NaN
            local_train_dice = float("nan")
        # reset local metric buffers for next epoch
        train_metric.reset()

        # reduce the aggregated value across ranks
        td_tensor = torch.tensor(local_train_dice, device=device)
        # if NaN, replace with -1 so all_reduce works; we'll treat negative as NaN
        if math.isnan(local_train_dice):
            td_tensor = torch.tensor(-1.0, device=device)
        dist.all_reduce(td_tensor, op=dist.ReduceOp.SUM)
        # if any rank was NaN all ranks had -1 contribution; reconstruct
        summed = td_tensor.item()
        # compute averaged metric; if any rank reported -1, handle as NaN
        if summed < 0:
            averaged_train_dice = float("nan")
        else:
            averaged_train_dice = summed / world_size

        # logging arrays update on rank0
        if is_rank0():
            epoch_list.append(epoch + 1)
            loss_history.append(epoch_loss_reduced)
            train_dice_history.append(averaged_train_dice)
            print(f"[Epoch {epoch+1}] Train Loss: {epoch_loss_reduced:.4f}, Train Dice: {averaged_train_dice:.4f}")

        # Validation
        val_metric.reset()
        prior.eval()
        refiner.eval()

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"[Rank {rank}] Val E{epoch+1}", disable=(not is_rank0())):
                x = batch["image"].to(device, non_blocking=True)
                y = batch["label"].to(device, non_blocking=True)
                logits = prior(x)
                preds_bin = (torch.sigmoid(logits) > 0.5).float()
                val_metric(preds_bin, y)
                # keep sample from first batch on rank0 for plotting
                if is_rank0() and 'val_sample' not in locals():
                    val_sample = (x.cpu().numpy(), preds_bin.cpu().numpy(), y.cpu().numpy())
                del x, y, logits, preds_bin
                torch.cuda.empty_cache()

        # aggregate val metric properly
        try:
            local_val_dice = val_metric.aggregate().item()
        except Exception:
            local_val_dice = float("nan")
        val_metric.reset()

        vd_tensor = torch.tensor(local_val_dice, device=device)
        if math.isnan(local_val_dice):
            vd_tensor = torch.tensor(-1.0, device=device)
        dist.all_reduce(vd_tensor, op=dist.ReduceOp.SUM)
        summed_v = vd_tensor.item()
        if summed_v < 0:
            averaged_val_dice = float("nan")
        else:
            averaged_val_dice = summed_v / world_size

        # rank0 saves plots and checkpoints
        if is_rank0():
            val_dice_history.append(averaged_val_dice)
            save_training_curves(epoch_list, loss_history, train_dice_history, val_dice_history, OUTPUT_DIR, epoch=epoch+1)
            if 'val_sample' in locals():
                # val_sample[0] shape: B,C,Z,Y,X
                save_prediction_sample(val_sample[0][0, :], val_sample[1][0, :], val_sample[2][0, :], OUTPUT_DIR, epoch=epoch+1)
                del val_sample
            # checkpoint
            ckpt = {
                "epoch": epoch+1,
                "prior_state": prior.module.state_dict(),
                "refiner_state": refiner.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "loss_history": loss_history,
                "train_dice_history": train_dice_history,
                "val_dice_history": val_dice_history
            }
            torch.save(ckpt, os.path.join(OUTPUT_DIR, f"checkpoint_epoch_{epoch+1}.pth"))

            # best model saving
            if not math.isnan(averaged_val_dice) and averaged_val_dice > best_val:
                best_val = averaged_val_dice
                torch.save(prior.module.state_dict(), os.path.join(OUTPUT_DIR, "best_prior.pth"))
                torch.save(refiner.module.state_dict(), os.path.join(OUTPUT_DIR, "best_refiner.pth"))
                print(f"[Rank 0] New best val {best_val:.4f} at epoch {epoch+1}")

        else:
            # non rank0 clear val_sample if set locally
            if 'val_sample' in locals():
                del val_sample

        # periodic checkpoint if not a validation epoch and to reduce loss of long runs
        if is_rank0() and ((epoch + 1) % CHECKPOINT_FREQ == 0 and (epoch + 1) % VAL_INTERVAL != 0):
            torch.save({
                "epoch": epoch+1,
                "prior_state": prior.module.state_dict(),
                "refiner_state": refiner.module.state_dict()
            }, os.path.join(OUTPUT_DIR, f"checkpoint_epoch_{epoch+1}.pth"))

    # final saves
    if is_rank0():
        torch.save(prior.module.state_dict(), os.path.join(OUTPUT_DIR, "final_prior.pth"))
        torch.save(refiner.module.state_dict(), os.path.join(OUTPUT_DIR, "final_refiner.pth"))
        save_training_curves(epoch_list, loss_history, train_dice_history, val_dice_history, OUTPUT_DIR, epoch="final")
        print("Training complete, best val:", best_val)

    cleanup_ddp()


if __name__ == "__main__":
    train()
