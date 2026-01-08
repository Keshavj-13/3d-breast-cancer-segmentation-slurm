import os
import time
import numpy as np
import matplotlib.pyplot as plt

# ================= USER CONFIG =================
SOURCE_DIR = "/home/neeraj/yunnan_ablation/swin_unetr_hybrid_bs3"
REFRESH_SECONDS = 5
OUT_DIR = os.path.abspath(".")
# ==============================================


def safe_load(path):
    if not os.path.exists(path):
        return None
    try:
        return np.load(path)
    except Exception:
        return None


def load_full_history():
    epochs = safe_load(os.path.join(SOURCE_DIR, "epochs.npy"))
    loss = safe_load(os.path.join(SOURCE_DIR, "loss_values.npy"))
    train_dice = safe_load(os.path.join(SOURCE_DIR, "train_dice_scores.npy"))
    val_dice = safe_load(os.path.join(SOURCE_DIR, "val_dice_scores.npy"))

    if epochs is None or loss is None or train_dice is None or val_dice is None:
        return None

    epochs = epochs.astype(int)

    # Mask out NaNs in validation
    valid = ~np.isnan(val_dice)
    val_epochs = epochs[valid]
    val_dice = val_dice[valid]

    return epochs, loss, train_dice, val_epochs, val_dice

def debug_val_dice():
    val_dice = safe_load(os.path.join(SOURCE_DIR, "val_dice_scores.npy"))
    epochs = safe_load(os.path.join(SOURCE_DIR, "epochs.npy"))

    if val_dice is None or epochs is None:
        print("Validation or epochs array missing")
        return

    print("Total val_dice length:", len(val_dice))

    # Print first 20 entries
    print("\nFirst 20 val_dice values:")
    for i in range(min(20, len(val_dice))):
        print(f"Epoch {epochs[i]} -> {val_dice[i]}")

    # Print last 20 entries
    print("\nLast 20 val_dice values:")
    for i in range(len(val_dice) - 20, len(val_dice)):
        print(f"Epoch {epochs[i]} -> {val_dice[i]}")

    # Detect actual changes
    print("\nDetected value changes:")
    last = val_dice[0]
    for i in range(1, len(val_dice)):
        if not np.isclose(val_dice[i], last, rtol=1e-6, atol=1e-8):
            print(f"Epoch {epochs[i]}: {last} -> {val_dice[i]}")
            last = val_dice[i]


def plot_training_curves():
    data = load_full_history()
    if data is None:
        return

    epochs, loss, train_dice, val_epochs, val_dice = data

    plt.figure(figsize=(15, 4))

    # Training loss
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(epochs, loss, linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.grid(True, alpha=0.3)

    # Dice scores
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(
        epochs,
        train_dice,
        label="Train Dice",
        linewidth=1.5,
        alpha=0.5
    )
    ax2.plot(
        val_epochs,
        val_dice,
        label="Validation Dice",
        linewidth=1.5
    )

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Dice")
    ax2.set_title("Dice Scores")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "training_curves_current.png"), dpi=150)
    plt.close()


def plot_prediction_sample():
    img = safe_load(os.path.join(SOURCE_DIR, "sample_image.npy"))
    pred = safe_load(os.path.join(SOURCE_DIR, "sample_prediction.npy"))
    gt = safe_load(os.path.join(SOURCE_DIR, "sample_gt.npy"))

    if img is None or pred is None or gt is None:
        return

    slice_idx = int(np.argmax(gt.sum(axis=(1, 2))))

    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(img[slice_idx], cmap="gray")
    plt.title("Input")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(img[slice_idx], cmap="gray")
    plt.imshow(gt[slice_idx], cmap="Greens", alpha=0.5)
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(img[slice_idx], cmap="gray")
    plt.imshow(pred[slice_idx], cmap="Reds", alpha=0.5)
    plt.title("Prediction")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "prediction_current.png"), dpi=150)
    plt.close()


def main():
    print("Live plotter running")
    print("Reading from:", SOURCE_DIR)

    # debug_val_dice()

    while True:
        try:
            plot_training_curves()
            plot_prediction_sample()
            time.sleep(REFRESH_SECONDS)
        except KeyboardInterrupt:
            print("Stopped")
            break
        except Exception as e:
            print("Plot error:", e)
            time.sleep(REFRESH_SECONDS)


if __name__ == "__main__":
    main()
