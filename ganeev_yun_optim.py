#!/usr/bin/env python3
"""
3D Medical Image Segmentation Training Script
Adapted to run on SLURM with the Yunnan dataset as the data source.
Only the data preparation and dataloader selection were changed.
Everything else from your original script is preserved.
"""

import os
import sys
import glob
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import AdamW

from sklearn.model_selection import train_test_split

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    EnsureTyped, ScaleIntensityRanged, NormalizeIntensityd,
    RandGaussianNoised, RandGaussianSmoothd, RandAdjustContrastd,
    CropForegroundd, RandCropByPosNegLabeld, DivisiblePadd,
    RandFlipd, RandRotate90d, RandAffined, SpatialPadD
)
from monai.data import DataLoader, CacheDataset
from monai.utils import set_determinism
from monai.networks.nets import FlexibleUNet
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric

from tqdm import tqdm

# Set memory management
# os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Set deterministic behavior
set_determinism(42)

# Configuration, keep these as in your original script
# Note: change BASE_DIR below to your Yunnan dataset root if needed
BASE_DIR = '/hostd/yunnan'  # <--- Yunnan dataset root, numeric case folders under this
MAX_PATIENTS = 100
MAX_EPOCHS = 500
VAL_INTERVAL = 5
BATCH_SIZE_TRAIN = 3
BATCH_SIZE_VAL = 1
NUM_WORKERS = 2  # fallback, will try to read SLURM_CPUS_PER_TASK if present
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# If you still want to filter a known corrupted file, set path here, otherwise None
CORRUPTED_FILE = None

# Persistent output directory
OUTPUT_DIR = "/hostd/yunnan_output_ganeev_optim"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory: {OUTPUT_DIR}")

BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model20.pth")
FINAL_MODEL_PATH = os.path.join(OUTPUT_DIR, "final_model.pth")
CHECKPOINT_FREQ = 5  # save checkpoint every few epochs
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "checkpoint.pth")

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"CUDA devices available: {torch.cuda.device_count()}")

# Respect SLURM allocated cpus if available for num_workers
try:
    slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", NUM_WORKERS))
    NUM_WORKERS = max(1, slurm_cpus)
except Exception:
    NUM_WORKERS = max(1, NUM_WORKERS)
print(f"Using num_workers for dataloaders: {NUM_WORKERS}")

# ====================================================================
# Custom Attention Block
# ====================================================================
class SCSEBlock(nn.Module):
    """Spatial and Channel Squeeze & Excitation Block"""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.channel_excitation = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, max(1, in_channels // reduction), 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(max(1, in_channels // reduction), in_channels, 1),
            nn.Sigmoid()
        )
        self.spatial_excitation = nn.Sequential(
            nn.Conv3d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        chn_se = self.channel_excitation(x) * x
        spa_se = self.spatial_excitation(x) * x
        return chn_se + spa_se


class AttentionAllDecoderUNet(FlexibleUNet):
    """FlexibleUNet with attention blocks in all decoder stages"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        decoder_channels = kwargs.get("decoder_channels", (512, 256, 128, 64, 32))
        self.attention_blocks = nn.ModuleList([
            SCSEBlock(ch) for ch in decoder_channels
        ])

    def forward(self, x):
        features = self.encoder(x)
        skips = [f for f in features[:-1] if f is not None][::-1]
        x = features[-1]

        for i, decoder_block in enumerate(self.decoder.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
            x = self.attention_blocks[i](x)

        x = self.segmentation_head(x)
        return x


# ====================================================================
# Data Validation Functions
# ====================================================================
def is_valid_nifti(path):
    """Check if NIfTI file is valid and readable"""
    try:
        img = nib.load(path)
        _ = np.asanyarray(img.dataobj)
        return True
    except Exception as e:
        print(f"[BAD] {path} -> {e}")
        return False


def pair_ok(rec):
    """Check if image-label pair is valid"""
    return is_valid_nifti(rec["image"]) and is_valid_nifti(rec["label"])


def keep_ok(rec):
    """Filter out known corrupted files"""
    if CORRUPTED_FILE is None:
        return True
    return (rec["image"] != CORRUPTED_FILE) and (rec["label"] != CORRUPTED_FILE)


# ====================================================================
# Data Preparation adapted for Yunnan dataset and SLURM
# ====================================================================
def prepare_data():
    """
    Prepare training and validation file lists for the Yunnan dataset.
    - Expects numeric case folders under BASE_DIR
    - Each case folder must contain P5.nii and GT.nii
    - Keeps same output structure as original prepare_data so rest of script is untouched
    """
    print("Preparing data from Yunnan dataset...")
    images_root = BASE_DIR  # each numeric folder is a case folder
    # find numeric case folders
    all_patient_ids = sorted([
        d for d in os.listdir(images_root)
        if d.isdigit() and os.path.isdir(os.path.join(images_root, d))
    ])

    if len(all_patient_ids) == 0:
        # fallback: also accept folders that are not strictly numeric
        all_patient_ids = sorted([
            d for d in os.listdir(images_root)
            if os.path.isdir(os.path.join(images_root, d))
        ])

    patient_ids_to_process = all_patient_ids[:MAX_PATIENTS]
    print(f"Found {len(all_patient_ids)} total patient folders. Processing first {len(patient_ids_to_process)}.")

    all_files = []
    for patient_id in patient_ids_to_process:
        case_dir = os.path.join(images_root, patient_id)
        image_path = os.path.join(case_dir, "P5.nii")
        label_path = os.path.join(case_dir, "GT.nii")

        # If naming variations exist try some common alternatives
        if not os.path.exists(image_path):
            # try lowercase
            alt = os.path.join(case_dir, "p5.nii")
            if os.path.exists(alt):
                image_path = alt
        if not os.path.exists(label_path):
            alt = os.path.join(case_dir, "gt.nii")
            if os.path.exists(alt):
                label_path = alt

        if os.path.exists(image_path) and os.path.exists(label_path):
            all_files.append({"image": image_path, "label": label_path})
        else:
            # print missing case info for debugging on SLURM
            print(f"[SKIP] case {patient_id}, missing files: "
                  f"{'P5.nii' if not os.path.exists(image_path) else ''} "
                  f"{'GT.nii' if not os.path.exists(label_path) else ''}")

    print(f"Created list with {len(all_files)} image/label pairs from Yunnan.")

    # Validate NIfTI files, optional heavy step. If you want to skip validation set SKIP_VALIDATION env var.
    skip_validation = os.environ.get("SKIP_VALIDATION", "0") == "1"
    if skip_validation:
        clean_files = all_files
        print("Skipping NIfTI validation as SKIP_VALIDATION env var is set.")
    else:
        print("Validating NIfTI files... (this can be slow for many cases)")
        clean_files = [r for r in all_files if pair_ok(r)]
        print(f"Valid files after check: {len(clean_files)}/{len(all_files)}")

    # Filter corrupted files if any configured
    clean_files = [r for r in clean_files if keep_ok(r)]
    print(f"After filtering known corrupted files: {len(clean_files)}")

    # Keep the same splitting behaviour as original script
    train_files, val_files = train_test_split(clean_files, test_size=0.2, random_state=42)
    print(f"Training cases: {len(train_files)}")
    print(f"Validation cases: {len(val_files)}")

    # Print first few entries for quick verification in SLURM logs
    for i, rec in enumerate(train_files[:3]):
        print(f"Train sample {i}: image {rec['image']}, label {rec['label']}")
    for i, rec in enumerate(val_files[:3]):
        print(f"Val sample {i}: image {rec['image']}, label {rec['label']}")

    return train_files, val_files


# ====================================================================
# Transforms (unchanged)
# ====================================================================
def get_train_transforms():
    """Define training transforms with augmentation"""
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"], a_min=0.0, a_max=1555.0,
            b_min=0.0, b_max=1.0, clip=True
        ),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadD(
            keys=["image", "label"], spatial_size=(128, 128, 96),
            method="end", mode="constant"
        ),
        DivisiblePadd(keys=["image", "label"], k=32),
        RandCropByPosNegLabeld(
            keys=["image", "label"], label_key="label",
            spatial_size=(128, 128, 96), pos=1, neg=1, num_samples=1,
            image_key="image", image_threshold=0
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0, 1, 2]),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        RandAffined(
            keys=["image", "label"], prob=0.5,
            rotate_range=(-0.1, 0.1), scale_range=(-0.1, 0.1),
            shear_range=0.1, translate_range=10,
            mode=("bilinear", "nearest")
        ),
        RandGaussianNoised(keys=["image"], prob=0.3, std=0.01),
        RandGaussianSmoothd(keys=["image"], prob=0.3, sigma_x=(0.5, 1.0)),
        RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.3)),
        EnsureTyped(keys=["image", "label"]),
    ])


def get_val_transforms():
    """Define validation transforms without augmentation"""
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"], a_min=0.0, a_max=1555.0,
            b_min=0.0, b_max=1.0, clip=True
        ),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadD(
            keys=["image", "label"], spatial_size=(128, 128, 96),
            method="end", mode="constant"
        ),
        DivisiblePadd(keys=["image", "label"], k=32),
        EnsureTyped(keys=["image", "label"]),
    ])


# ====================================================================
# Plotting and Saving Functions (unchanged)
# ====================================================================
def save_training_curves(epochs, loss_values, train_dice_scores, dice_scores,
                         epoch_num, overwrite=True):
    np.save(os.path.join(OUTPUT_DIR, 'epochs.npy'), np.array(epochs))
    np.save(os.path.join(OUTPUT_DIR, 'loss_values.npy'), np.array(loss_values))
    np.save(os.path.join(OUTPUT_DIR, 'train_dice_scores.npy'), np.array(train_dice_scores))
    np.save(os.path.join(OUTPUT_DIR, 'val_dice_scores.npy'), np.array(dice_scores))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))
    ax1_twin = ax1.twinx()
    ax1.plot(epochs, loss_values, color='tab:red', label="Train Loss", linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", color='tab:red', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.grid(True, alpha=0.3)

    ax1_twin.plot(epochs, dice_scores, color='tab:blue', label="Val Dice", linewidth=2)
    ax1_twin.set_ylabel("Validation Dice", color='tab:blue', fontsize=12)
    ax1_twin.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_title(f"Training Progress - Epoch {epoch_num}", fontsize=14, fontweight='bold')

    ax2.plot(epochs, train_dice_scores, color='tab:green', linestyle='--', linewidth=2)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Training Dice", fontsize=12)
    ax2.set_title("Training Dice Score", fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = 'training_curves_current.png' if overwrite else f'training_curves_epoch_{epoch_num}.png'
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()


def save_prediction_sample(val_images, val_outputs, val_labels, epoch_num, overwrite=True):
    img = val_images[0, 0].cpu().numpy()
    pred = val_outputs[0, 0].cpu().numpy()
    gt = val_labels[0, 0].cpu().numpy()

    np.save(os.path.join(OUTPUT_DIR, 'sample_image.npy'), img)
    np.save(os.path.join(OUTPUT_DIR, 'sample_prediction.npy'), pred)
    np.save(os.path.join(OUTPUT_DIR, 'sample_gt.npy'), gt)

    slice_idx = int(np.argmax(gt.sum(axis=(1, 2))))

    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(img[slice_idx], cmap='gray')
    plt.title('Input Image', fontsize=12, fontweight='bold')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(img[slice_idx], cmap='gray')
    plt.imshow(gt[slice_idx], cmap='Greens', alpha=0.5)
    plt.title('Ground Truth', fontsize=12, fontweight='bold')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(img[slice_idx], cmap='gray')
    plt.imshow(pred[slice_idx], cmap='Reds', alpha=0.5)
    plt.title('Prediction', fontsize=12, fontweight='bold')
    plt.axis('off')

    plt.suptitle(f"Segmentation @ Epoch {epoch_num}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    filename = 'prediction_current.png' if overwrite else f'prediction_epoch_{epoch_num}.png'
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()


def create_final_summary_plot(epochs, loss_values, train_dice_scores, dice_scores,
                              val_images, val_outputs, val_labels, best_dice):
    img = val_images[0, 0].cpu().numpy()
    pred = val_outputs[0, 0].cpu().numpy()
    gt = val_labels[0, 0].cpu().numpy()

    np.save(os.path.join(OUTPUT_DIR, 'final_image.npy'), img)
    np.save(os.path.join(OUTPUT_DIR, 'final_prediction.npy'), pred)
    np.save(os.path.join(OUTPUT_DIR, 'final_gt.npy'), gt)

    gt_sum = gt.sum(axis=(1, 2))
    roi_slices = np.where(gt_sum > 0)[0]

    if len(roi_slices) > 0:
        if len(roi_slices) >= 3:
            slice_indices = [roi_slices[0], roi_slices[len(roi_slices) // 2], roi_slices[-1]]
        else:
            slice_indices = [int(np.argmax(gt_sum))]
    else:
        slice_indices = [img.shape[0] // 2]

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, :2])
    ax1_twin = ax1.twinx()
    ax1.plot(epochs, loss_values, color='tab:red', label="Train Loss", linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", color='tab:red', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.grid(True, alpha=0.3)
    ax1_twin.plot(epochs, dice_scores, color='tab:blue', label="Val Dice", linewidth=2)
    ax1_twin.set_ylabel("Validation Dice", color='tab:blue', fontsize=12)
    ax1_twin.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_title("Training Loss & Validation Dice", fontsize=14, fontweight='bold')

    ax2 = fig.add_subplot(gs[0, 2:])
    ax2.plot(epochs, train_dice_scores, color='tab:green', linewidth=2)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Training Dice", fontsize=12)
    ax2.set_title("Training Dice Score", fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    for i, slice_idx in enumerate(slice_indices):
        row = 1 + i // 3
        col_offset = (i % 3) * 3

        ax = fig.add_subplot(gs[row, col_offset])
        ax.imshow(img[slice_idx], cmap='gray')
        ax.set_title(f'Slice {slice_idx}: Input', fontsize=10)
        ax.axis('off')

        ax = fig.add_subplot(gs[row, col_offset + 1])
        ax.imshow(img[slice_idx], cmap='gray')
        ax.imshow(gt[slice_idx], cmap='Greens', alpha=0.5)
        ax.set_title(f'Slice {slice_idx}: GT', fontsize=10)
        ax.axis('off')

        ax = fig.add_subplot(gs[row, col_offset + 2])
        ax.imshow(img[slice_idx], cmap='gray')
        ax.imshow(pred[slice_idx], cmap='Reds', alpha=0.5)
        ax.set_title(f'Slice {slice_idx}: Pred', fontsize=10)
        ax.axis('off')

    fig.suptitle(f'Final Training Summary - Best Dice: {best_dice:.4f}', fontsize=16, fontweight='bold')
    plt.savefig(os.path.join(OUTPUT_DIR, 'FINAL_SUMMARY.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Final summary plot saved: {os.path.join(OUTPUT_DIR, 'FINAL_SUMMARY.png')}")


# ====================================================================
# Training Functions (unchanged except dataloader worker numbers)
# ====================================================================
def train():
    """Main training loop"""
    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80 + "\n")

    # Prepare data
    train_files, val_files = prepare_data()

    # Create datasets
    print("\nCreating datasets...")
    train_ds = CacheDataset(
        data=train_files,
        transform=get_train_transforms(),
        cache_rate=0.0
    )
    val_ds = CacheDataset(
        data=val_files,
        transform=get_val_transforms(),
        cache_rate=0.0
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE_VAL,
        shuffle=False,
        num_workers=NUM_WORKERS if NUM_WORKERS > 1 else 1,
        pin_memory=False
    )

    # Create model
    print("\nInitializing model...")
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

    # Loss, optimizer, metrics
    loss_function = DiceFocalLoss(
        include_background=False,
        sigmoid=True,
        squared_pred=True,
        lambda_dice=1.0,
        lambda_focal=1.0,
        gamma=2.0,
        alpha=0.7,
        reduction="mean",
    )

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    train_dice_metric = DiceMetric(include_background=False, reduction="mean")

    # Training tracking
    best_dice = -1
    epochs_list, dice_scores, train_dice_scores, loss_values = [], [], [], []

    final_val_images = None
    final_val_outputs = None
    final_val_labels = None

    print(f"\nTraining for {MAX_EPOCHS} epochs...")
    print(f"Monitoring plots will be saved to: {OUTPUT_DIR}")
    print(f"  - training_curves_current.png (updates each epoch)")
    print(f"  - prediction_current.png (updates each epoch)")
    print(f"  - FINAL_SUMMARY.png (created at end)")
    print("=" * 80 + "\n")

    for epoch in range(MAX_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{MAX_EPOCHS}")
        print("-" * 40)

        # Training phase
        model.train()
        epoch_loss = 0
        step = 0
        train_dice_metric.reset()

        for batch_data in tqdm(train_loader, desc="Training"):
            step += 1
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            with torch.no_grad():
                preds = torch.sigmoid(outputs)
                preds_bin = (preds > 0.5).float()
                train_dice_metric(preds_bin, labels)

        avg_loss = epoch_loss / step if step > 0 else float("nan")
        try:
            train_dice = train_dice_metric.aggregate().item()
        except Exception:
            train_dice = float("nan")

        loss_values.append(avg_loss)
        train_dice_scores.append(train_dice)
        epochs_list.append(epoch + 1)

        print(f"Avg Train Loss: {avg_loss:.4f} | Train Dice: {train_dice:.4f}")

        # Validation phase
        if (epoch + 1) % VAL_INTERVAL == 0:
            model.eval()
            dice_metric.reset()

            with torch.no_grad():
                val_images_sample = None
                val_outputs_sample = None
                val_labels_sample = None

                for val_data in tqdm(val_loader, desc="Validation"):
                    val_images = val_data["image"].to(device)
                    val_labels = val_data["label"].to(device)

                    val_outputs = model(val_images)
                    val_outputs = torch.sigmoid(val_outputs)
                    val_outputs_bin = (val_outputs > 0.5).float()

                    dice_metric(val_outputs_bin, val_labels)

                    if val_images_sample is None:
                        val_images_sample = val_images
                        val_outputs_sample = val_outputs_bin
                        val_labels_sample = val_labels

            try:
                mean_dice = dice_metric.aggregate().item()
            except Exception:
                mean_dice = float("nan")
            dice_scores.append(mean_dice)

            print(f"Validation Dice: {mean_dice:.4f}")

            if mean_dice > best_dice:
                best_dice = mean_dice
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                print(f"Saved best model with Dice {best_dice:.4f}")

                final_val_images = val_images_sample
                final_val_outputs = val_outputs_sample
                final_val_labels = val_labels_sample

            if (epoch + 1) % 5 == 0:
                save_training_curves(
                    epochs_list,
                    loss_values,
                    train_dice_scores,
                    dice_scores,
                    epoch + 1,
                    overwrite=True
                )

                if val_images_sample is not None:
                    save_prediction_sample(
                        val_images_sample,
                        val_outputs_sample,
                        val_labels_sample,
                        epoch + 1,
                        overwrite=True
                    )

            if (epoch + 1) % 50 == 0:
                save_training_curves(
                    epochs_list,
                    loss_values,
                    train_dice_scores,
                    dice_scores,
                    epoch + 1,
                    overwrite=False
                )

                if val_images_sample is not None:
                    save_prediction_sample(
                        val_images_sample,
                        val_outputs_sample,
                        val_labels_sample,
                        epoch + 1,
                        overwrite=False
                    )

        else:
            dice_scores.append(np.nan)

    # Save final model
    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    print(f"\n✅ Saved final model to {FINAL_MODEL_PATH}")

    # Create comprehensive final summary
    if final_val_images is not None:
        print("\nCreating final summary plot...")
        create_final_summary_plot(
            epochs_list, loss_values, train_dice_scores, dice_scores,
            final_val_images, final_val_outputs, final_val_labels, best_dice
        )

    print("\n" + "=" * 80)
    print("Training Complete!")
    print(f"Best Validation Dice: {best_dice:.4f}")
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    train()