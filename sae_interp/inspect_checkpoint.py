"""
Print the contents of a sae_best.pt checkpoint.

Usage:
    python inspect_checkpoint.py                          # default: ./checkpoints_rand/sae_best.pt
    python inspect_checkpoint.py path/to/sae_best.pt
"""

import sys
import torch

path = sys.argv[1] if len(sys.argv) > 1 else "./checkpoints_rand/sae_best.pt"
ckpt = torch.load(path, map_location="cpu", weights_only=False)

print(f"Checkpoint: {path}\n")

# Training args (new runs only)
if "args" in ckpt:
    print("=== Training args ===")
    for k, v in sorted(ckpt["args"].items()):
        print(f"  {k}: {v}")
    print()

# Best-checkpoint metrics
print("=== Metrics at best checkpoint ===")
for k in ("step", "epoch", "val_loss", "val_mse", "val_l0", "dead_frac"):
    if k in ckpt:
        print(f"  {k}: {ckpt[k]}")
print()

# Model shape
print("=== Model ===")
for k in ("d_in", "expansion", "lam"):
    if k in ckpt:
        print(f"  {k}: {ckpt[k]}")
if "sae_state" in ckpt:
    print("  sae_state keys:", list(ckpt["sae_state"].keys()))
print()

# Normalisation stats
if "emb_mean" in ckpt:
    m, s = ckpt["emb_mean"], ckpt["emb_std"]
    print(f"=== Embedding normalisation ===")
    print(f"  emb_mean: shape={tuple(m.shape)}  mean={m.mean():.4f}  std={m.std():.4f}")
    print(f"  emb_std:  shape={tuple(s.shape)}  mean={s.mean():.4f}  std={s.std():.4f}")
