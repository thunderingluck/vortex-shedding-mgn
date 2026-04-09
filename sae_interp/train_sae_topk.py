"""
train_sae_topk.py

Train a Top-K SparseAutoencoder on node-level MGN embeddings.
Architecture: hard top-K activation, MSE-only loss (no L1, no lambda).

For a quick smoke test:
    python train_sae_topk.py --k 32 --max_trajs 10 --max_epochs 2 --val_every 200 --patience 2

Full run (from sae_interp/):
    python train_sae_topk.py --k 32 --emb_dir ../sae_embeddings/consolidated
"""

import argparse
import csv
import os
import glob
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from sae_topk import SparseAutoencoderTopK


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--emb_dir", default="../sae_embeddings/consolidated")
    p.add_argument("--ckpt_dir", default=None,
                   help="Checkpoint directory (default: checkpoints_topk_K{k})")
    p.add_argument("--resume_ckpt", default=None,
                   help="Path to sae_latest.pt to resume from")
    p.add_argument("--d_in", type=int, default=128)
    p.add_argument("--expansion", type=int, default=8)
    p.add_argument("--k", type=int, default=32,
                   help="Number of active features per node (Top-K)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--max_epochs", type=int, default=50)
    p.add_argument("--val_every", type=int, default=10000)
    p.add_argument("--patience", type=int, default=8,
                   help="Early-stop after this many eval cycles without improvement")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--max_trajs", type=int, default=None)
    return p.parse_args()


def traj_id_from_path(path: str) -> str:
    m = re.search(r"traj_(\d+)", os.path.basename(path))
    if m is None:
        raise ValueError(f"Cannot parse traj id from {path}")
    return m.group(1)


def load_split(files: list[str], desc: str) -> torch.Tensor:
    arrays = []
    for f in files:
        if f.endswith(".npy"):
            arrays.append(np.load(f))
        else:
            arrays.append(np.load(f)["hL"].astype(np.float32))
    data = np.concatenate(arrays, axis=0)
    print(f"  {desc}: {len(files)} files, {data.shape[0]:,} nodes")
    return torch.from_numpy(data)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(sae, val_data: torch.Tensor, device: str, batch_size: int = 8192):
    sae.eval()
    total_mse = 0.0
    total_l0 = 0.0
    n_samples = val_data.shape[0]
    d_hid = sae.d_hid

    feature_fired = torch.zeros(d_hid, dtype=torch.bool, device=device)

    for start in range(0, n_samples, batch_size):
        h = val_data[start:start + batch_size].to(device)
        h_hat, z = sae(h)
        mse = F.mse_loss(h_hat, h, reduction="mean")
        total_mse += mse.item() * h.shape[0]
        total_l0 += (z > 0).float().sum(dim=1).mean().item() * h.shape[0]
        feature_fired |= (z > 0).any(dim=0)

    total_mse /= n_samples
    total_l0 /= n_samples
    dead_frac = (~feature_fired).float().mean().item()

    sae.train()
    return total_mse, total_l0, dead_frac


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

def save_plots(log: list[dict], ckpt_dir: str):
    if not log:
        return
    steps      = [r["step"]       for r in log]
    train_mse  = [r["train_mse"]  for r in log]
    val_mse    = [r["val_mse"]    for r in log]
    val_l0     = [r["val_l0"]     for r in log]
    dead_frac  = [r["dead_frac"]  for r in log]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f"Top-K SAE Training (K={log[0]['k']})")

    ax = axes[0]
    ax.semilogy(steps, train_mse, label="train_mse (EMA)")
    ax.semilogy(steps, val_mse,   label="val_mse")
    ax.set_xlabel("step"); ax.set_ylabel("MSE (log)"); ax.set_title("Reconstruction MSE")
    ax.legend()

    ax = axes[1]
    ax.plot(steps, val_l0)
    ax.axhline(log[0]["k"], color="gray", linestyle="--", label=f"K={log[0]['k']}")
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
    print(f"  -> saved training curves to {path}")


# ---------------------------------------------------------------------------
# main training loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    if args.ckpt_dir is None:
        args.ckpt_dir = f"./checkpoints_topk_K{args.k}"

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  K={args.k}  |  ckpt_dir={args.ckpt_dir}")

    # ------------------------------------------------------------------
    # 1. Discover files and split by trajectory
    # ------------------------------------------------------------------
    all_files = sorted(
        glob.glob(os.path.join(args.emb_dir, "traj_*.npy")) or
        glob.glob(os.path.join(args.emb_dir, "traj_*.npz"))
    )
    if not all_files:
        raise FileNotFoundError(f"No traj_*.npy/npz files in {args.emb_dir}")

    traj_ids = sorted(set(traj_id_from_path(f) for f in all_files))
    if args.max_trajs is not None:
        traj_ids = traj_ids[:args.max_trajs]
        all_files = [f for f in all_files if traj_id_from_path(f) in set(traj_ids)]
    n_traj = len(traj_ids)
    n_val_traj = max(1, round(n_traj * args.val_frac))
    n_train_traj = n_traj - n_val_traj

    shuffled_traj = rng.permutation(traj_ids)
    train_traj_set = set(shuffled_traj[:n_train_traj])
    val_traj_set   = set(shuffled_traj[n_train_traj:])

    train_files = [f for f in all_files if traj_id_from_path(f) in train_traj_set]
    val_files   = [f for f in all_files if traj_id_from_path(f) in val_traj_set]

    print(f"Trajectories total={n_traj}, train={n_train_traj}, val={n_val_traj}")

    # ------------------------------------------------------------------
    # 2. Load data into RAM
    # ------------------------------------------------------------------
    print("Loading training data...")
    train_data = load_split(train_files, "train")
    print("Loading validation data...")
    val_data = load_split(val_files, "val")

    emb_mean = train_data.mean(dim=0)
    emb_std  = train_data.std(dim=0).clamp_min(1e-6)
    train_data = (train_data - emb_mean) / emb_std
    val_data   = (val_data   - emb_mean) / emb_std
    print(f"  Normalized: mean={emb_mean.mean():.4f}  std={emb_std.mean():.4f}")

    n_train = train_data.shape[0]

    # ------------------------------------------------------------------
    # 3. Build model and optimizer
    # ------------------------------------------------------------------
    sae = SparseAutoencoderTopK(d_in=args.d_in, expansion=args.expansion, k=args.k).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=args.lr)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 4. Resume from checkpoint if requested
    # ------------------------------------------------------------------
    global_step = 0
    best_val_mse = float("inf")
    patience_count = 0
    ema_mse = 0.0
    start_epoch = 1

    if args.resume_ckpt is not None:
        print(f"Resuming from {args.resume_ckpt}")
        ckpt = torch.load(args.resume_ckpt, map_location=device)
        sae.load_state_dict(ckpt["sae_state"])
        opt.load_state_dict(ckpt["opt_state"])
        global_step   = ckpt["step"]
        best_val_mse  = ckpt["best_val_mse"]
        patience_count = ckpt["patience_count"]
        ema_mse       = ckpt["ema_mse"]
        # Resume from the start of the next epoch so we don't re-run a partial epoch
        start_epoch   = ckpt["epoch"] + 1
        print(f"  Resumed at step={global_step}, epoch={ckpt['epoch']}, "
              f"best_val_mse={best_val_mse:.4e}, patience={patience_count}")

    # ------------------------------------------------------------------
    # 5. Metric logging
    # ------------------------------------------------------------------
    csv_path = os.path.join(args.ckpt_dir, "metrics.csv")
    csv_fields = ["step", "epoch", "k", "train_mse", "val_mse", "val_l0", "dead_frac"]

    # On resume, read existing log for plotting continuity then append
    metric_log = []
    if args.resume_ckpt is not None and os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                metric_log.append({
                    "step": int(row["step"]), "epoch": int(row["epoch"]),
                    "k": int(row["k"]), "train_mse": float(row["train_mse"]),
                    "val_mse": float(row["val_mse"]), "val_l0": float(row["val_l0"]),
                    "dead_frac": float(row["dead_frac"]),
                })
        csv_file = open(csv_path, "a", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    else:
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
        csv_writer.writeheader()

    # ------------------------------------------------------------------
    # 6. Training loop
    # ------------------------------------------------------------------
    ema_alpha = 0.98

    print(f"\nStarting training: K={args.k}, batch_size={args.batch_size}, "
          f"val_every={args.val_every}, patience={args.patience}\n")

    try:
        for epoch in range(start_epoch, args.max_epochs + 1):
            perm = torch.from_numpy(rng.permutation(n_train))

            for start in range(0, n_train, args.batch_size):
                idx = perm[start:start + args.batch_size]
                if len(idx) == 0:
                    continue
                h = train_data[idx].to(device)

                recon, _, _ = sae.loss(h)
                opt.zero_grad(set_to_none=True)
                recon.backward()
                torch.nn.utils.clip_grad_norm_(sae.parameters(), max_norm=1.0)
                opt.step()
                sae.renorm_decoder_rows_()

                global_step += 1
                ema_mse = ema_alpha * ema_mse + (1 - ema_alpha) * recon.item()

                if global_step % args.val_every == 0:
                    val_mse, val_l0, dead_frac = validate(sae, val_data, device)

                    print(
                        f"[step {global_step:7d} | ep {epoch}] "
                        f"train_mse={ema_mse:.4e}  "
                        f"val_mse={val_mse:.4e}  val_L0={val_l0:.1f}  dead={dead_frac:.3f}"
                    )

                    row = {
                        "step": global_step, "epoch": epoch, "k": args.k,
                        "train_mse": ema_mse, "val_mse": val_mse,
                        "val_l0": val_l0, "dead_frac": dead_frac,
                    }
                    csv_writer.writerow(row)
                    csv_file.flush()
                    metric_log.append(row)
                    save_plots(metric_log, args.ckpt_dir)

                    if val_mse < best_val_mse:
                        best_val_mse = val_mse
                        patience_count = 0
                        ckpt_path = os.path.join(args.ckpt_dir, "sae_best.pt")
                        torch.save({
                            "sae_state": sae.state_dict(),
                            "opt_state": opt.state_dict(),
                            "d_in": args.d_in,
                            "expansion": args.expansion,
                            "k": args.k,
                            "step": global_step,
                            "epoch": epoch,
                            "best_val_mse": best_val_mse,
                            "val_mse": best_val_mse,
                            "val_l0": val_l0,
                            "dead_frac": dead_frac,
                            "patience_count": patience_count,
                            "ema_mse": ema_mse,
                            "emb_mean": emb_mean.cpu(),
                            "emb_std": emb_std.cpu(),
                            "args": vars(args),
                        }, ckpt_path)
                        print(f"  -> saved best checkpoint (val_mse={best_val_mse:.4e})")
                    else:
                        patience_count += 1
                        print(f"  -> no improvement ({patience_count}/{args.patience})")

                    if patience_count >= args.patience:
                        print("\nEarly stopping.")
                        return

                    torch.save({
                        "sae_state": sae.state_dict(),
                        "opt_state": opt.state_dict(),
                        "d_in": args.d_in,
                        "expansion": args.expansion,
                        "k": args.k,
                        "step": global_step,
                        "epoch": epoch,
                        "best_val_mse": best_val_mse,
                        "val_mse": val_mse,
                        "val_l0": val_l0,
                        "dead_frac": dead_frac,
                        "patience_count": patience_count,
                        "ema_mse": ema_mse,
                        "emb_mean": emb_mean.cpu(),
                        "emb_std": emb_std.cpu(),
                        "args": vars(args),
                    }, os.path.join(args.ckpt_dir, "sae_latest.pt"))

            print(f"Epoch {epoch} done  (step={global_step})")

        print(f"\nTraining complete. Best val_mse={best_val_mse:.4e}")

    finally:
        csv_file.close()
        save_plots(metric_log, args.ckpt_dir)


if __name__ == "__main__":
    main()
