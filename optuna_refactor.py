#!/usr/bin/env python3
"""
Optuna ready training script, robust for Slurm runs and shared sqlite DB access.
Save as optuna_refactor_fixed_with_logs.py
"""

import os
import sys
import time
import traceback
import sqlite3
import platform
import shutil
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import AdamW

import optuna
from sklearn.model_selection import train_test_split

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    EnsureTyped, ScaleIntensityRanged, NormalizeIntensityd,
    RandGaussianNoised, RandGaussianSmoothd, RandAdjustContrastd,
    CropForegroundd, RandCropByPosNegLabeld, DivisiblePadd,
    RandFlipd, RandRotate90d, RandAffined, SpatialPadD
)
from monai.data import DataLoader, Dataset
from monai.utils import set_determinism
from monai.networks.nets import FlexibleUNet
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric

from tqdm import tqdm

# -------------------- Configurable values, override with env if needed --------------------
BASE_DIR = os.environ.get('MAMA_MIA_BASE', '/hostd/mama_mia_dataset/MAMA-MIA')
OUTPUT_ROOT = os.environ.get('MAMA_MIA_OUT', '/hostd/mama_mia_optuna_runs')
OPTUNA_DB_DIR = os.environ.get('OPTUNA_DB_DIR', '/home/neeraj/optuna')  # default to home
os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.makedirs(OPTUNA_DB_DIR, exist_ok=True)

MAX_PATIENTS = int(os.environ.get('MAX_PATIENTS', '1506'))
DEFAULT_EPOCHS = int(os.environ.get('OPTUNA_EPOCHS', '12'))
VAL_INTERVAL = int(os.environ.get('OPTUNA_VAL_INTERVAL', '2'))

set_determinism(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

optuna.logging.set_verbosity(optuna.logging.WARNING)

# -------------------- Small utilities --------------------
def ensure_sqlite_wal(db_path):
    """Create DB file if missing and set journal mode to WAL so concurrent access works"""
    created = False
    if not os.path.exists(db_path):
        # create empty file by connecting
        open(db_path, 'a').close()
        created = True
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute('PRAGMA journal_mode=WAL;')
        cur.execute('PRAGMA synchronous=NORMAL;')
        cur.execute('PRAGMA wal_checkpoint(TRUNCATE);')
        conn.commit()
        cur.execute('PRAGMA journal_mode;')
        mode = cur.fetchone()
        print(f'[DB] WAL mode returned: {mode}')
    finally:
        conn.close()
    if created:
        print(f'[DB] Created new sqlite file at {db_path}')
    else:
        print(f'[DB] Using existing sqlite file at {db_path}')
    # also print permissions and size
    try:
        st = os.stat(db_path)
        print(f'[DB] File size bytes: {st.st_size}, mode: {oct(st.st_mode)}')
    except Exception as e:
        print(f'[DB] Could not stat file: {e}')

def print_env_info():
    print('========== ENV INFO ==========')
    print('Python', sys.version.replace('\n', ' '))
    print('Platform', platform.platform())
    print('Working dir', os.getcwd())
    print('BASE_DIR', BASE_DIR)
    print('OUTPUT_ROOT', OUTPUT_ROOT)
    print('OPTUNA_DB_DIR', OPTUNA_DB_DIR)
    print('CUDA_VISIBLE_DEVICES', os.environ.get('CUDA_VISIBLE_DEVICES'))
    print('Torch version', torch.__version__)
    print('CUDA available', torch.cuda.is_available())
    if torch.cuda.is_available():
        try:
            ndev = torch.cuda.device_count()
            print('CUDA device count', ndev)
            for i in range(ndev):
                props = torch.cuda.get_device_properties(i)
                print(f'GPU {i}, name {props.name}, SM count {props.multi_processor_count}, total mem GB {props.total_memory / 1e9:.2f}')
        except Exception as e:
            print('[ENV] Could not query GPU properties', e)
    print('================================')

def safe_makedirs(path):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        print(f'[FS] Could not create directory {path}: {e}')
        raise

# -------------------- Model pieces --------------------
class SCSEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        reduced = max(1, in_channels // reduction)
        self.channel_excitation = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, reduced, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(reduced, in_channels, 1),
            nn.Sigmoid()
        )
        self.spatial_excitation = nn.Sequential(
            nn.Conv3d(in_channels, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.channel_excitation(x) * x + self.spatial_excitation(x) * x

class AttentionAllDecoderUNet(FlexibleUNet):
    def __init__(self, *args, **kwargs):
        decoder_channels = kwargs.get('decoder_channels', (512, 256, 128, 64, 32))
        super().__init__(*args, **kwargs)
        self.attention_blocks = nn.ModuleList([SCSEBlock(ch) for ch in decoder_channels])

    def forward(self, x):
        features = self.encoder(x)
        skips = [f for f in features[:-1] if f is not None][::-1]
        x = features[-1]
        for i, block in enumerate(self.decoder.blocks):
            x = block(x, skips[i] if i < len(skips) else None)
            x = self.attention_blocks[i](x)
        return self.segmentation_head(x)

# -------------------- Data prep --------------------
def prepare_data(max_patients=MAX_PATIENTS):
    images_dir = os.path.join(BASE_DIR, 'images')
    labels_dir = os.path.join(BASE_DIR, 'segmentations', 'expert')
    print(f'[DATA] Looking for images in {images_dir}')
    print(f'[DATA] Looking for labels in {labels_dir}')
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f'Images directory not found: {images_dir}')
    patient_ids = sorted([p for p in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, p))])[:max_patients]
    files = []
    for pid in patient_ids:
        img = os.path.join(images_dir, pid, f'{pid}_0002.nii')
        lbl = os.path.join(labels_dir, f'{pid}.nii')
        if os.path.exists(img) and os.path.exists(lbl):
            files.append({'image': img, 'label': lbl})
    if len(files) == 0:
        raise RuntimeError('No image, label pairs found, check BASE_DIR layout')
    train, val = train_test_split(files, test_size=0.2, random_state=42)
    print(f'[DATA] Found {len(files)} pairs, train {len(train)}, val {len(val)}')
    return train, val

def get_train_tf():
    return Compose([
        LoadImaged(keys=['image', 'label']),
        EnsureChannelFirstd(keys=['image', 'label']),
        Spacingd(keys=['image', 'label'], pixdim=(1.0, 1.0, 1.0), mode=('bilinear', 'nearest')),
        Orientationd(keys=['image', 'label'], axcodes='RAS'),
        ScaleIntensityRanged(keys=['image'], a_min=0.0, a_max=1555.0, b_min=0.0, b_max=1.0, clip=True),
        NormalizeIntensityd(keys=['image'], nonzero=True, channel_wise=True),
        CropForegroundd(keys=['image', 'label'], source_key='image'),
        SpatialPadD(keys=['image', 'label'], spatial_size=(128, 128, 96), method='end', mode='constant'),
        DivisiblePadd(keys=['image', 'label'], k=32),
        RandCropByPosNegLabeld(keys=['image', 'label'], label_key='label', spatial_size=(128, 128, 96), pos=1, neg=1, num_samples=4, image_key='image', image_threshold=0),
        RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=[0,1,2]),
        RandRotate90d(keys=['image', 'label'], prob=0.5, max_k=3),
        RandAffined(keys=['image','label'], prob=0.5, rotate_range=(-0.1,0.1), translate_range=(-10,10), scale_range=(0.9,1.1), shear_range=None, mode=('bilinear','nearest')),
        RandGaussianNoised(keys=['image'], prob=0.3, std=0.01),
        RandGaussianSmoothd(keys=['image'], prob=0.3, sigma_x=(0.5, 1.0)),
        RandAdjustContrastd(keys=['image'], prob=0.3, gamma=(0.7,1.3)),
        EnsureTyped(keys=['image','label']),
    ])

def get_val_tf():
    return Compose([
        LoadImaged(keys=['image','label']),
        EnsureChannelFirstd(keys=['image','label']),
        Spacingd(keys=['image','label'], pixdim=(1.0,1.0,1.0), mode=('bilinear','nearest')),
        Orientationd(keys=['image','label'], axcodes='RAS'),
        ScaleIntensityRanged(keys=['image'], a_min=0.0, a_max=1555.0, b_min=0.0, b_max=1.0, clip=True),
        NormalizeIntensityd(keys=['image'], nonzero=True, channel_wise=True),
        CropForegroundd(keys=['image','label'], source_key='image'),
        SpatialPadD(keys=['image','label'], spatial_size=(128,128,96), method='end', mode='constant'),
        DivisiblePadd(keys=['image','label'], k=32),
        EnsureTyped(keys=['image','label']),
    ])

# -------------------- Objective --------------------
def objective(trial):
    print(f'[TRIAL] Starting trial number {trial.number}')
    lr = trial.suggest_float('lr', 1e-5, 3e-4, log=True)
    wd = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)
    dropout = trial.suggest_float('dropout', 0.1, 0.4)
    batch_size = int(trial.suggest_categorical('batch_size', [1, 2, 3]))
    n_epochs = int(trial.suggest_int('n_epochs', 6, DEFAULT_EPOCHS))

    print(f'[TRIAL] params lr={lr}, wd={wd}, dropout={dropout}, batch_size={batch_size}, n_epochs={n_epochs}')

    run_dir = os.path.join(OUTPUT_ROOT, f'trial_{trial.number}')
    safe_makedirs(run_dir)

    train_files, val_files = prepare_data()

    train_ds = Dataset(data=train_files, transform=get_train_tf())
    val_ds = Dataset(data=val_files, transform=get_val_tf())

    # conservative num_workers on shared node
    num_cpus = max(1, os.cpu_count() or 1)
    train_workers = min(4, max(0, num_cpus - 2))
    val_workers = min(2, max(0, num_cpus - 2))

    print(f'[DATA] train size {len(train_ds)}, val size {len(val_ds)}')
    print(f'[IO] train_workers {train_workers}, val_workers {val_workers}, batch_size {batch_size}')

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=train_workers, pin_memory=torch.cuda.is_available(), persistent_workers=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=val_workers, pin_memory=torch.cuda.is_available(), persistent_workers=False)

    print('[MODEL] Building model on device', device)
    model = AttentionAllDecoderUNet(
        in_channels=1, out_channels=1, backbone='resnet50', pretrained=False,
        decoder_channels=(512,256,128,64,32), spatial_dims=3,
        dropout=dropout, norm=('instance', {'affine': True}),
        act=('leakyrelu', {'inplace': True, 'negative_slope': 0.01}),
        upsample='deconv', interp_mode='trilinear', is_pad=False
    ).to(device)

    loss_fn = DiceFocalLoss(sigmoid=True, squared_pred=True)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    dice_metric = DiceMetric(include_background=False, reduction='mean')

    best_dice = 0.0

    try:
        for epoch in range(n_epochs):
            t_epoch_start = time.time()
            model.train()
            total_loss = 0.0
            steps = 0
            for batch in train_loader:
                x = batch['image'].to(device)
                y = batch['label'].to(device)
                optimizer.zero_grad()
                out = model(x)
                loss = loss_fn(out, y)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().cpu().numpy())
                steps += 1

            avg_loss = total_loss / max(1, steps)
            t_epoch_end = time.time()
            print(f'[TRAIN] trial {trial.number} epoch {epoch+1}/{n_epochs} avg_loss {avg_loss:.6f} time_sec {(t_epoch_end - t_epoch_start):.1f}')

            if (epoch + 1) % VAL_INTERVAL == 0:
                t_val_start = time.time()
                model.eval()
                dice_metric.reset()
                with torch.no_grad():
                    for batch in val_loader:
                        x = batch['image'].to(device)
                        y = batch['label'].to(device)
                        p = torch.sigmoid(model(x))
                        dice_metric((p > 0.5).float(), y)
                score = float(dice_metric.aggregate().item())
                t_val_end = time.time()
                print(f'[VAL] trial {trial.number} epoch {epoch+1} dice {score:.6f} val_time_sec {(t_val_end - t_val_start):.1f}')
                best_dice = max(best_dice, score)
                trial.report(score, epoch)
                if trial.should_prune():
                    print(f'[PRUNE] trial {trial.number} pruning requested at epoch {epoch+1}')
                    raise optuna.TrialPruned()

    except optuna.TrialPruned:
        print(f'[TRIAL] Trial {trial.number} pruned at epoch {epoch + 1}')
        raise
    except Exception:
        tb = traceback.format_exc()
        with open(os.path.join(run_dir, 'error.log'), 'w') as f:
            f.write(tb)
        print('[ERROR] Exception in trial, trace saved to', os.path.join(run_dir, 'error.log'))
        print(tb)
        raise

    model_path = os.path.join(run_dir, 'model.pth')
    torch.save(model.state_dict(), model_path)
    print(f'[SAVE] Saved model to {model_path}')
    summary_path = os.path.join(run_dir, 'trial_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f'best_dice={best_dice}, lr={lr}, wd={wd}, dropout={dropout}, batch_size={batch_size}\n')
    print(f'[SAVE] Wrote trial summary to {summary_path}')

    return best_dice

# -------------------- Entrypoint --------------------
if __name__ == '__main__':
    start_time = time.time()
    print_env_info()

    db_path = os.path.join(OPTUNA_DB_DIR, 'mama_mia_optuna.db')
    print(f'[MAIN] Optuna sqlite path {db_path}')
    ensure_sqlite_wal(db_path)
    storage = f'sqlite:///{db_path}'

    try:
        print('[MAIN] Creating or loading study')
        study = optuna.create_study(study_name='mama_mia_a100_2', storage=storage, direction='maximize', load_if_exists=True)
        print(f'[MAIN] Study loaded name {study.study_name}')
        print('[MAIN] Starting optimize for a single trial')
        study.optimize(objective, n_trials=1)
        print('[MAIN] study.optimize returned')
        # print best trial if any
        try:
            best = study.best_trial
            print(f'[MAIN] Best trial number {best.number} value {best.value}')
            print('[MAIN] Best params:')
            for k, v in best.params.items():
                print(f'  {k}: {v}')
        except Exception as e:
            print('[MAIN] Could not fetch best trial', e)

    except Exception:
        print('[MAIN] Optuna study failed to start or run')
        traceback.print_exc()
        sys.exit(2)
    finally:
        elapsed = time.time() - start_time
        print(f'[MAIN] Total runtime seconds {elapsed:.1f}')
