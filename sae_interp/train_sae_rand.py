"""
train_sae_rand.py

For testing training on small number of trajectories, 
run: 
python train_sae_rand.py --max_trajs 10 --max_epochs 2 --val_every 200 --patience 2

Train a SparseAutoencoder on node-level embeddings from ../sae_embeddings/raw/.

Key design choices:
- 80/20 train/val split at the TRAJECTORY level (all steps of a traj go to the same split)
- Each NODE is one training sample (not each graph/snapshot)
- Full flattened matrix loaded into RAM; random permutation each epoch
- Validation every `val_every` steps: MSE, L0, dead-feature fraction
- Early stopping with configurable patience (in eval cycles)
- Best checkpoint saved whenever val loss improves
- Metrics logged to CSV; training curves saved as PNG on exit
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

from sae import SparseAutoencoder


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--emb_dir", default="../sae_embeddings/raw")
    p.add_argument("--ckpt_dir", default="./checkpoints_rand")
    p.add_argument("--d_in", type=int, default=128)
    p.add_argument("--expansion", type=int, default=8)
    p.add_argument("--lam", type=float, default=3e-4,
                   help="L1 sparsity coefficient (paper: 3e-4)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=128,
                   help="Node-level mini-batch size (paper: 128)")
    p.add_argument("--max_epochs", type=int, default=50)
    p.add_argument("--val_every", type=int, default=2000,
                   help="Validate every this many optimizer steps")
    p.add_argument("--patience", type=int, default=8,
                   help="Early-stop after this many eval cycles without improvement")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--max_trajs", type=int, default=None,
                   help="Cap the number of trajectories loaded (useful for smoke tests)")
    return p.parse_args()


def traj_id_from_path(path: str) -> str:
    """Extract trajectory identifier from filename.
    Handles both traj_0000_step_0000.npz and consolidated traj_0000.npy."""
    m = re.search(r"traj_(\d+)", os.path.basename(path))
    if m is None:
        raise ValueError(f"Cannot parse traj id from {path}")
    return m.group(1)


def load_split(files: list[str], desc: str) -> torch.Tensor:
    """Load all files, concatenate node embeddings -> (total_nodes, d_in) float32.
    Accepts both .npy (consolidated, already float32) and .npz (raw per-step)."""
    arrays = []
    for f in files:
        if f.endswith(".npy"):
            arrays.append(np.load(f))           # (N*T, 128) already float32
        else:
            arrays.append(np.load(f)["hL"].astype(np.float32))  # (N, 128)
    data = np.concatenate(arrays, axis=0)
    print(f"  {desc}: {len(files)} files, {data.shape[0]:,} nodes")
    return torch.from_numpy(data)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(sae, val_data: torch.Tensor, lam: float, device: str,
             batch_size: int = 8192):
    sae.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_l1 = 0.0
    total_l0 = 0.0
    n_samples = val_data.shape[0]
    d_hid = sae.d_hid

    # track which features fired at all
    feature_fired = torch.zeros(d_hid, dtype=torch.bool, device=device)

    for start in range(0, n_samples, batch_size):
        h = val_data[start:start + batch_size].to(device)
        h_hat, z = sae(h)
        mse = F.mse_loss(h_hat, h, reduction="mean")
        l1 = z.abs().mean()
        total_loss += (mse + lam * l1).item() * h.shape[0]
        total_mse += mse.item() * h.shape[0]
        total_l1 += l1.item() * h.shape[0]
        total_l0 += (z > 0).float().sum(dim=1).mean().item() * h.shape[0]
        feature_fired |= (z > 0).any(dim=0)

    total_loss /= n_samples
    total_mse /= n_samples
    total_l1 /= n_samples
    total_l0 /= n_samples
    dead_frac = (~feature_fired).float().mean().item()

    sae.train()
    return total_loss, total_mse, total_l0, dead_frac


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

def save_plots(log: list[dict], ckpt_dir: str):
    if not log:
        return
    steps       = [r["step"]        for r in log]
    train_loss  = [r["train_loss"]  for r in log]
    train_recon = [r["train_recon"] for r in log]
    val_loss    = [r["val_loss"]    for r in log]
    val_mse     = [r["val_mse"]     for r in log]
    val_l0      = [r["val_l0"]      for r in log]
    dead_frac   = [r["dead_frac"]   for r in log]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("SAE Training Curves")

    ax = axes[0, 0]
    ax.semilogy(steps, train_loss, label="train_loss (EMA)")
    ax.semilogy(steps, val_loss,   label="val_loss")
    ax.set_xlabel("step"); ax.set_ylabel("loss (log)"); ax.set_title("Loss")
    ax.legend()

    ax = axes[0, 1]
    ax.semilogy(steps, train_recon, label="train_recon (EMA)")
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
    print(f"  -> saved training curves to {path}")


# ---------------------------------------------------------------------------
# main training loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # 1. Discover files and split by trajectory
    # ------------------------------------------------------------------
    all_files = sorted(
        glob.glob(os.path.join(args.emb_dir, "traj_*.npy")) or
        glob.glob(os.path.join(args.emb_dir, "traj_*.npz"))
    )
    if not all_files:
        raise FileNotFoundError(f"No traj_*.npz files in {args.emb_dir}")

    traj_ids = sorted(set(traj_id_from_path(f) for f in all_files))
    if args.max_trajs is not None:
        traj_ids = traj_ids[:args.max_trajs]
        all_files = [f for f in all_files if traj_id_from_path(f) in set(traj_ids)]
    n_traj = len(traj_ids)
    n_val_traj = max(1, round(n_traj * args.val_frac))
    n_train_traj = n_traj - n_val_traj

    shuffled_traj = rng.permutation(traj_ids)
    train_traj_set = set(shuffled_traj[:n_train_traj])
    val_traj_set = set(shuffled_traj[n_train_traj:])

    train_files = [f for f in all_files if traj_id_from_path(f) in train_traj_set]
    val_files = [f for f in all_files if traj_id_from_path(f) in val_traj_set]

    print(f"Trajectories total={n_traj}, train={n_train_traj}, val={n_val_traj}")

    # ------------------------------------------------------------------
    # 2. Load data into RAM
    # ------------------------------------------------------------------
    print("Loading training data...")
    train_data = load_split(train_files, "train")  # (N_train, 128)
    print("Loading validation data...")
    val_data = load_split(val_files, "val")        # (N_val, 128)

    n_train = train_data.shape[0]

    # ------------------------------------------------------------------
    # 3. Build model and optimizer
    # ------------------------------------------------------------------
    sae = SparseAutoencoder(d_in=args.d_in, expansion=args.expansion).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=args.lr)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 4. Metric logging setup
    # ------------------------------------------------------------------
    csv_path = os.path.join(args.ckpt_dir, "metrics.csv")
    csv_fields = ["step", "epoch", "train_loss", "train_recon", "train_L1",
                  "val_loss", "val_mse", "val_l0", "dead_frac"]
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    csv_writer.writeheader()
    metric_log = []  # in-memory copy for plotting at the end

    # ------------------------------------------------------------------
    # 5. Training loop
    # ------------------------------------------------------------------
    global_step = 0
    best_val_loss = float("inf")
    patience_count = 0
    ema_loss = ema_recon = ema_spars = 0.0
    ema_alpha = 0.98

    print(f"\nStarting training: batch_size={args.batch_size}, "
          f"val_every={args.val_every} steps, patience={args.patience} evals\n")

    try:
        for epoch in range(1, args.max_epochs + 1):
            # Fresh shuffle of all training nodes each epoch
            perm = torch.from_numpy(rng.permutation(n_train))

            for start in range(0, n_train, args.batch_size):
                idx = perm[start:start + args.batch_size]
                if len(idx) == 0:
                    continue
                h = train_data[idx].to(device)

                loss, recon, spars = sae.loss(h, lam=args.lam)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(sae.parameters(), max_norm=1.0)
                opt.step()
                sae.renorm_decoder_rows_()

                global_step += 1
                ema_loss  = ema_alpha * ema_loss  + (1 - ema_alpha) * loss.item()
                ema_recon = ema_alpha * ema_recon + (1 - ema_alpha) * recon.item()
                ema_spars = ema_alpha * ema_spars + (1 - ema_alpha) * spars.item()

                # ----------------------------------------------------------
                # Validation
                # ----------------------------------------------------------
                if global_step % args.val_every == 0:
                    val_loss, val_mse, val_l0, dead_frac = validate(
                        sae, val_data, args.lam, device)

                    print(
                        f"[step {global_step:7d} | ep {epoch}] "
                        f"train_loss={ema_loss:.4e}  train_recon={ema_recon:.4e}  train_L1={ema_spars:.4e}  "
                        f"val_loss={val_loss:.4e}  val_mse={val_mse:.4e}  "
                        f"val_L0={val_l0:.1f}  dead={dead_frac:.3f}"
                    )

                    row = {
                        "step": global_step, "epoch": epoch,
                        "train_loss": ema_loss, "train_recon": ema_recon,
                        "train_L1": ema_spars, "val_loss": val_loss,
                        "val_mse": val_mse, "val_l0": val_l0,
                        "dead_frac": dead_frac,
                    }
                    csv_writer.writerow(row)
                    csv_file.flush()
                    metric_log.append(row)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_count = 0
                        ckpt_path = os.path.join(args.ckpt_dir, "sae_best.pt")
                        torch.save({
                            "sae_state": sae.state_dict(),
                            "d_in": args.d_in,
                            "expansion": args.expansion,
                            "lam": args.lam,
                            "step": global_step,
                            "epoch": epoch,
                            "val_loss": best_val_loss,
                            "val_mse": val_mse,
                            "val_l0": val_l0,
                            "dead_frac": dead_frac,
                        }, ckpt_path)
                        print(f"  -> saved best checkpoint (val_loss={best_val_loss:.4e})")
                    else:
                        patience_count += 1
                        print(f"  -> no improvement ({patience_count}/{args.patience})")
                        if patience_count >= args.patience:
                            print("\nEarly stopping triggered.")
                            return

            print(f"Epoch {epoch} done  (step={global_step})")

        print(f"\nTraining complete. Best val_loss={best_val_loss:.4e}")

    finally:
        csv_file.close()
        save_plots(metric_log, args.ckpt_dir)


if __name__ == "__main__":
    main()
