"""
Regenerate training_curves.png for all checkpoints_* directories
from their metrics.csv, with train and val MSE on the same scale.

The bug: train_recon/train_mse was logged as sum over d_in (128), while
val_mse uses F.mse_loss (mean over all elements). Dividing by d_in=128
makes them comparable.
"""

import csv
import os
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D_IN  = 128
D_HID = D_IN * 8  # expansion=8


def save_plots_rand(log: list[dict], ckpt_dir: str):
    steps       = [r["step"]               for r in log]
    train_recon = [r["train_recon"] / D_IN for r in log]  # sum→mean over d_in
    val_loss    = [r["val_loss"]           for r in log]
    val_mse     = [r["val_mse"]            for r in log]
    val_l0      = [r["val_l0"]             for r in log]
    dead_frac   = [r["dead_frac"]          for r in log]

    # Infer lam from first row with non-zero L1: lam = (train_loss - train_recon) / train_L1
    lam = None
    for r in log:
        if r["train_L1"] > 0:
            lam = (r["train_loss"] - r["train_recon"]) / r["train_L1"]
            break
    if lam is not None:
        # Reconstruct normalized loss: recon/d_in + lam * L1/d_hid
        train_loss = [r["train_recon"] / D_IN + lam * r["train_L1"] / D_HID for r in log]
        loss_label = f"train_loss (norm, λ≈{lam:.2e})"
    else:
        train_loss = [r["train_loss"] for r in log]
        loss_label = "train_loss (raw)"

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("SAE Training Curves")

    ax = axes[0, 0]
    ax.semilogy(steps, train_loss, label=loss_label)
    ax.semilogy(steps, val_loss,   label="val_loss")
    ax.set_xlabel("step"); ax.set_ylabel("loss (log)"); ax.set_title("Loss")
    ax.legend()

    ax = axes[0, 1]
    ax.semilogy(steps, train_recon, label="train_recon/128 (EMA)")
    ax.semilogy(steps, val_mse,     label="val_mse")
    ax.set_xlabel("step"); ax.set_ylabel("MSE (log)"); ax.set_title("Reconstruction MSE")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(steps, val_l0)
    ax.set_xlabel("step"); ax.set_ylabel("L0 (features/node)"); ax.set_title("Val L0 (sparsity)")

    ax = axes[1, 1]
    ax.plot(steps, dead_frac)
    ax.set_xlabel("step"); ax.set_ylabel("fraction"); ax.set_title("Dead features")
    ax.set_ylim(0, 1)

    plt.tight_layout()
    path = os.path.join(ckpt_dir, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  -> {path}")


def save_plots_topk(log: list[dict], ckpt_dir: str):
    steps      = [r["step"]                   for r in log]
    train_mse  = [r["train_mse"] / D_IN       for r in log]  # normalize sum→mean
    val_mse    = [r["val_mse"]                for r in log]
    val_l0     = [r["val_l0"]                 for r in log]
    dead_frac  = [r["dead_frac"]              for r in log]
    k = log[0]["k"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f"Top-K SAE Training (K={k})")

    ax = axes[0]
    ax.semilogy(steps, train_mse, label="train_mse/128 (EMA)")
    ax.semilogy(steps, val_mse,   label="val_mse")
    ax.set_xlabel("step"); ax.set_ylabel("MSE (log)"); ax.set_title("Reconstruction MSE")
    ax.legend()

    ax = axes[1]
    ax.plot(steps, val_l0)
    ax.axhline(k, color="gray", linestyle="--", label=f"K={k}")
    ax.set_xlabel("step"); ax.set_ylabel("L0"); ax.set_title("Val L0 (should converge to K)")
    ax.legend()

    ax = axes[2]
    ax.plot(steps, dead_frac)
    ax.set_xlabel("step"); ax.set_ylabel("fraction"); ax.set_title("Dead features")
    ax.set_ylim(0, 1)

    plt.tight_layout()
    path = os.path.join(ckpt_dir, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  -> {path}")


def load_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append({k: (int(v) if k in ("step", "epoch") else
                             (int(v) if k == "k" else float(v)))
                         for k, v in row.items()})
    return rows


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dirs = sorted(glob.glob(os.path.join(script_dir, "checkpoints_*")))

    if not ckpt_dirs:
        print("No checkpoints_* directories found.")
        return

    for ckpt_dir in ckpt_dirs:
        csv_path = os.path.join(ckpt_dir, "metrics.csv")
        if not os.path.exists(csv_path):
            print(f"Skipping {ckpt_dir} (no metrics.csv)")
            continue

        log = load_csv(csv_path)
        if not log:
            print(f"Skipping {ckpt_dir} (empty metrics.csv)")
            continue

        print(f"Replotting {os.path.basename(ckpt_dir)} ({len(log)} rows)...")
        if "train_recon" in log[0]:
            save_plots_rand(log, ckpt_dir)
        else:
            save_plots_topk(log, ckpt_dir)


if __name__ == "__main__":
    main()
