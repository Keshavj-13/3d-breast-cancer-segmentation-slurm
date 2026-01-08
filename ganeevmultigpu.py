#!/usr/bin/env python3
"""
3D Medical Image Segmentation Training Script
MAMA-MIA Dataset - UNet with Attention
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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
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
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Set deterministic behavior
set_determinism(42)

# Configuration
BASE_DIR = '/hostd/mama_mia_dataset/MAMA-MIA'
MAX_PATIENTS = 1506
MAX_EPOCHS = 500
VAL_INTERVAL = 1
BATCH_SIZE_TRAIN = 3
BATCH_SIZE_VAL = 1
NUM_WORKERS = 2 
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
CORRUPTED_FILE = "/hostd/mama_mia_dataset/MAMA-MIA/images/DUKE_034/DUKE_034_0002.nii"

# Device will be set in init_distributed_and_device()
device = None


# ============================================================================
# Distributed Training Helpers
# ============================================================================
def is_distributed():
    """Check if running in distributed mode"""
    return int(os.environ.get('WORLD_SIZE', 1)) > 1


def get_local_rank():
    """Get local rank from environment"""
    return int(os.environ.get('LOCAL_RANK', 0))


def get_global_rank():
    """Get global rank from environment"""
    return int(os.environ.get('RANK', 0))


def is_main_process():
    """Check if current process is rank 0"""
    return get_global_rank() == 0


def init_distributed_and_device():
    """Initialize distributed training and set device"""
    global device
    
    # Initialize process group if distributed
    if is_distributed():
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(get_local_rank())
        device = torch.device(f"cuda:{get_local_rank()}")
        if is_main_process():
            print(f"Initialized distributed training on {dist.get_world_size()} GPUs")
            print(f"Using device: {device}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if is_main_process():
            print(f"Using device: {device}")
    
    # Create output directory on main process
    if is_main_process():
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"Output directory: {OUTPUT_DIR}")
        print(f"CUDA devices available: {torch.cuda.device_count()}")
        print(f"Global rank: {get_global_rank()}, Local rank: {get_local_rank()}")


def destroy_process_group():
    """Clean up distributed training"""
    if is_distributed():
        dist.destroy_process_group()


# Persistent output directory (like your CFD reference)
OUTPUT_DIR = "/hostd/mama_mia_output"
# Note: Output directory will be created by init_distributed_and_device() before use

BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model20.pth")
FINAL_MODEL_PATH = os.path.join(OUTPUT_DIR, "final_model.pth")


# ============================================================================
# Custom Attention Block
# ============================================================================
class SCSEBlock(nn.Module):
    """Spatial and Channel Squeeze & Excitation Block"""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.channel_excitation = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction, in_channels, 1),
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


# ============================================================================
# Data Validation Functions
# ============================================================================
def is_valid_nifti(path):
    """Check if NIfTI file is valid and readable"""
    try:
        img = nib.load(path)
        _ = np.asanyarray(img.dataobj)
        return True
    except Exception as e:
        if is_main_process():
            print(f"[BAD] {path} -> {e}")
        return False


def pair_ok(rec):
    """Check if image-label pair is valid"""
    return is_valid_nifti(rec["image"]) and is_valid_nifti(rec["label"])


def keep_ok(rec):
    """Filter out known corrupted files"""
    return (rec["image"] != CORRUPTED_FILE) and (rec["label"] != CORRUPTED_FILE)


# ============================================================================
# Data Preparation
# ============================================================================
def prepare_data():
    """Load and prepare training/validation data splits"""
    if is_main_process():
        print("Preparing data...")
    
    images_dir = os.path.join(BASE_DIR, 'images')
    labels_dir = os.path.join(BASE_DIR, 'segmentations', 'expert')
    
    all_patient_ids = sorted([
        folder_name for folder_name in os.listdir(images_dir)
        if os.path.isdir(os.path.join(images_dir, folder_name))
    ])
    
    patient_ids_to_process = all_patient_ids[:MAX_PATIENTS]
    if is_main_process():
        print(f"Found {len(all_patient_ids)} total patients. Processing first {len(patient_ids_to_process)}.")
    
    all_files = []
    for patient_id in patient_ids_to_process:
        image_path = os.path.join(images_dir, patient_id, f'{patient_id}_0002.nii')
        label_path = os.path.join(labels_dir, f'{patient_id}.nii')
        
        if os.path.exists(image_path) and os.path.exists(label_path):
            all_files.append({"image": image_path, "label": label_path})
    
    if is_main_process():
        print(f"Created list with {len(all_files)} image/label pairs.")
    
    # Validate files
    if is_main_process():
        print("Validating NIfTI files...")
    clean_files = [r for r in all_files if pair_ok(r)]
    if is_main_process():
        print(f"Valid files: {len(clean_files)}/{len(all_files)}")
    
    # Filter corrupted files
    clean_files = [r for r in clean_files if keep_ok(r)]
    if is_main_process():
        print(f"After filtering corrupted files: {len(clean_files)}")
    
    # Split data
    train_files, val_files = train_test_split(clean_files, test_size=0.2, random_state=42)
    if is_main_process():
        print(f"Training cases: {len(train_files)}")
        print(f"Validation cases: {len(val_files)}")
    
    return train_files, val_files


# ============================================================================
# Transforms
# ============================================================================
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
            spatial_size=(128, 128, 96), pos=1, neg=1, num_samples=4,
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


# ============================================================================
# Plotting and Saving Functions
# ============================================================================
def save_training_curves(epochs, loss_values, train_dice_scores, dice_scores, 
                         epoch_num, overwrite=True):
    """
    Save training curves - overwrites the same file for monitoring progress
    Also saves numpy arrays for later plotting
    """
    if not is_main_process():
        return
    
    # Save numpy data
    np.save(os.path.join(OUTPUT_DIR, 'epochs.npy'), np.array(epochs))
    np.save(os.path.join(OUTPUT_DIR, 'loss_values.npy'), np.array(loss_values))
    np.save(os.path.join(OUTPUT_DIR, 'train_dice_scores.npy'), np.array(train_dice_scores))
    np.save(os.path.join(OUTPUT_DIR, 'val_dice_scores.npy'), np.array(dice_scores))
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))
    
    # Loss and Val Dice
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
    
    # Train Dice
    ax2.plot(epochs, train_dice_scores, color='tab:green', linestyle='--', linewidth=2)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Training Dice", fontsize=12)
    ax2.set_title("Training Dice Score", fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Overwrite the same file for easy monitoring
    if overwrite:
        filename = 'training_curves_current.png'
    else:
        filename = f'training_curves_epoch_{epoch_num}.png'
    
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()


def save_prediction_sample(val_images, val_outputs, val_labels, epoch_num, overwrite=True):
    """
    Save prediction visualization - overwrites same file for monitoring
    Also saves numpy arrays of the sample
    """
    if not is_main_process():
        return
    
    img = val_images[0, 0].cpu().numpy()
    pred = val_outputs[0, 0].cpu().numpy()
    gt = val_labels[0, 0].cpu().numpy()
    
    # Save numpy data
    np.save(os.path.join(OUTPUT_DIR, 'sample_image.npy'), img)
    np.save(os.path.join(OUTPUT_DIR, 'sample_prediction.npy'), pred)
    np.save(os.path.join(OUTPUT_DIR, 'sample_gt.npy'), gt)
    
    # Find slice with most ground truth
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
    
    # Overwrite for monitoring
    if overwrite:
        filename = 'prediction_current.png'
    else:
        filename = f'prediction_epoch_{epoch_num}.png'
    
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()


def create_final_summary_plot(epochs, loss_values, train_dice_scores, dice_scores,
                              val_images, val_outputs, val_labels, best_dice):
    """
    Create comprehensive final summary plot with all metrics and predictions
    Shows region of interest with multiple slices
    """
    if not is_main_process():
        return
    
    img = val_images[0, 0].cpu().numpy()
    pred = val_outputs[0, 0].cpu().numpy()
    gt = val_labels[0, 0].cpu().numpy()
    
    # Save final numpy data
    np.save(os.path.join(OUTPUT_DIR, 'final_image.npy'), img)
    np.save(os.path.join(OUTPUT_DIR, 'final_prediction.npy'), pred)
    np.save(os.path.join(OUTPUT_DIR, 'final_gt.npy'), gt)
    
    # Find region of interest (slices with segmentation)
    gt_sum = gt.sum(axis=(1, 2))
    roi_slices = np.where(gt_sum > 0)[0]
    
    if len(roi_slices) > 0:
        # Get 3 representative slices from ROI
        if len(roi_slices) >= 3:
            slice_indices = [
                roi_slices[0],  # Start of ROI
                roi_slices[len(roi_slices)//2],  # Middle of ROI
                roi_slices[-1]  # End of ROI
            ]
        else:
            slice_indices = [int(np.argmax(gt_sum))]  # Just use best slice
    else:
        slice_indices = [img.shape[0]//2]  # Fallback to middle slice
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Top row: Training curves
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
    
    # Bottom rows: Predictions for multiple slices
    for i, slice_idx in enumerate(slice_indices):
        row = 1 + i // 3
        col_offset = (i % 3) * 3
        
        # Input image
        ax = fig.add_subplot(gs[row, col_offset])
        ax.imshow(img[slice_idx], cmap='gray')
        ax.set_title(f'Slice {slice_idx}: Input', fontsize=10)
        ax.axis('off')
        
        # Ground truth overlay
        ax = fig.add_subplot(gs[row, col_offset + 1])
        ax.imshow(img[slice_idx], cmap='gray')
        ax.imshow(gt[slice_idx], cmap='Greens', alpha=0.5)
        ax.set_title(f'Slice {slice_idx}: GT', fontsize=10)
        ax.axis('off')
        
        # Prediction overlay
        ax = fig.add_subplot(gs[row, col_offset + 2])
        ax.imshow(img[slice_idx], cmap='gray')
        ax.imshow(pred[slice_idx], cmap='Reds', alpha=0.5)
        ax.set_title(f'Slice {slice_idx}: Pred', fontsize=10)
        ax.axis('off')
    
    fig.suptitle(f'Final Training Summary - Best Dice: {best_dice:.4f}', 
                 fontsize=16, fontweight='bold')
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'FINAL_SUMMARY.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    if is_main_process():
        print(f"\n✅ Final summary plot saved: {os.path.join(OUTPUT_DIR, 'FINAL_SUMMARY.png')}")


# ============================================================================
# Training Functions
# ============================================================================
def train():
    """Main training loop"""
    if is_main_process():
        print("\n" + "="*80)
        print("Starting Training")
        print("="*80 + "\n")
    
    # Prepare data
    train_files, val_files = prepare_data()
    
    # Create datasets
    if is_main_process():
        print("\nCreating datasets...")
    train_ds = CacheDataset(
        data=train_files,
        transform=get_train_transforms(),
        cache_rate=1.0
    )
    val_ds = CacheDataset(
        data=val_files,
        transform=get_val_transforms(),
        cache_rate=1.0
    )
    
    # Create samplers for distributed training
    train_sampler = DistributedSampler(
        train_ds,
        num_replicas=dist.get_world_size() if is_distributed() else 1,
        rank=get_global_rank(),
        shuffle=True,
        seed=42
    ) if is_distributed() else None
    
    val_sampler = DistributedSampler(
        val_ds,
        num_replicas=dist.get_world_size() if is_distributed() else 1,
        rank=get_global_rank(),
        shuffle=False,
        seed=42
    ) if is_distributed() else None
    
    # Create dataloaders
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE_TRAIN,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE_VAL,
        sampler=val_sampler,
        shuffle=False, num_workers=1
    )
    
    # Create model
    if is_main_process():
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
    
    # Wrap model with DistributedDataParallel if distributed
    if is_distributed():
        model = DDP(model, device_ids=[get_local_rank()])
    
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
    
    # For final plot
    final_val_images = None
    final_val_outputs = None
    final_val_labels = None
    
    if is_main_process():
        print(f"\nTraining for {MAX_EPOCHS} epochs...")
        print(f"Monitoring plots will be saved to: {OUTPUT_DIR}")
        print(f"  - training_curves_current.png (updates each epoch)")
        print(f"  - prediction_current.png (updates each epoch)")
        print(f"  - FINAL_SUMMARY.png (created at end)")
        print("="*80 + "\n")
    
    for epoch in range(MAX_EPOCHS):
        # Set epoch for sampler (important for shuffle)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)
        
        if is_main_process():
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
        
        avg_loss = epoch_loss / step
        train_dice = train_dice_metric.aggregate().item()
        loss_values.append(avg_loss)
        train_dice_scores.append(train_dice)
        
        if is_main_process():
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
                    
                    # Save first batch for visualization
                    if val_images_sample is None:
                        val_images_sample = val_images
                        val_outputs_sample = val_outputs_bin
                        val_labels_sample = val_labels
            
            mean_dice = dice_metric.aggregate().item()
            dice_scores.append(mean_dice)
            epochs_list.append(epoch + 1)
            
            if is_main_process():
                print(f"Validation Dice: {mean_dice:.4f}")
            
            # Save best model (only on main process)
            if is_main_process() and mean_dice > best_dice:
                best_dice = mean_dice
                # Get underlying model if using DDP
                model_state = model.module.state_dict() if is_distributed() else model.state_dict()
                torch.save(model_state, BEST_MODEL_PATH)
                print(f"✅ Saved best model with Dice {best_dice:.4f}")
                
                # Save best samples for final plot
                final_val_images = val_images_sample
                final_val_outputs = val_outputs_sample
                final_val_labels = val_labels_sample
            
            # Save monitoring plots (overwrite same files)
            save_training_curves(epochs_list, loss_values, train_dice_scores,
                               dice_scores, epoch + 1, overwrite=True)
            save_prediction_sample(val_images_sample, val_outputs_sample,
                                 val_labels_sample, epoch + 1, overwrite=True)
            
            # Save milestone plots (keep separate files every 50 epochs)
            if (epoch + 1) % 50 == 0:
                save_training_curves(epochs_list, loss_values, train_dice_scores,
                                   dice_scores, epoch + 1, overwrite=False)
                save_prediction_sample(val_images_sample, val_outputs_sample,
                                     val_labels_sample, epoch + 1, overwrite=False)
    
    # Save final model (only on main process)
    if is_main_process():
        model_state = model.module.state_dict() if is_distributed() else model.state_dict()
        torch.save(model_state, FINAL_MODEL_PATH)
        print(f"\n✅ Saved final model to {FINAL_MODEL_PATH}")
        
        # Create comprehensive final summary
        if final_val_images is not None:
            print("\nCreating final summary plot...")
            create_final_summary_plot(
                epochs_list, loss_values, train_dice_scores, dice_scores,
                final_val_images, final_val_outputs, final_val_labels, best_dice
            )
        
        print("\n" + "="*80)
        print("Training Complete!")
        print(f"Best Validation Dice: {best_dice:.4f}")
        print(f"All outputs saved to: {OUTPUT_DIR}")
        print("="*80)


if __name__ == "__main__":
    init_distributed_and_device()
    try:
        train()
    finally:
        destroy_process_group()