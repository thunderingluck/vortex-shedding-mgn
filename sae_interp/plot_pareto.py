"""Plot L0 vs val_MSE Pareto frontier across L1 and Top-K SAE runs.

Each run is represented by its best checkpoint (lowest val loss/MSE on val).
MSE is element-wise on per-dim normalized embeddings (unit variance).
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

L1_RUNS = [
    ("checkpoints_rand_1e-2",      "λ=1e-2"),
    ("checkpoints_rand_1e-3",      "λ=1e-3"),
    ("checkpoints_rand_3e-4",      "λ=3e-4"),
    ("checkpoints_rand_3e-4_long", "λ=3e-4 (long)"),
    ("checkpoints_rand_1e-4_p20",  "λ=1e-4 (p20)"),
]
TOPK_RUNS = [
    ("checkpoints_topk_K16", 16),
    ("checkpoints_topk_K32", 32),
    ("checkpoints_topk_K48", 48),
    ("checkpoints_topk_K64", 64),
]
TOPK_AUX_RUNS = [
    ("checkpoints_topk_aux_K16",  16),
    ("checkpoints_topk_aux_K32",  32),
    ("checkpoints_topk_aux_K48",  48),
    ("checkpoints_topk_aux_K64",  64),
    ("checkpoints_topk_aux_K128", 128),
    ("checkpoints_topk_aux_K150", 150),
]


def best_row(csv_path: str, mse_col: str, l0_col: str = "val_l0",
             dead_col: str = "dead_frac"):
    """Return (L0, MSE, dead_frac) from row minimizing mse_col."""
    if not os.path.exists(csv_path):
        return None
    best = None
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                mse = float(row[mse_col])
            except (KeyError, ValueError):
                continue
            if best is None or mse < best[1]:
                best = (float(row[l0_col]), mse, float(row[dead_col]))
    return best


def main():
    l1_pts, topk_pts, topkaux_pts = [], [], []

    for ckpt, label in L1_RUNS:
        csv_path = os.path.join(HERE, ckpt, "metrics.csv")
        r = best_row(csv_path, mse_col="val_mse")
        if r is not None:
            l1_pts.append((ckpt, label, *r))

    for ckpt, k in TOPK_RUNS:
        csv_path = os.path.join(HERE, ckpt, "metrics.csv")
        r = best_row(csv_path, mse_col="val_mse")
        if r is not None:
            topk_pts.append((ckpt, f"K={k}", *r))

    for ckpt, k in TOPK_AUX_RUNS:
        csv_path = os.path.join(HERE, ckpt, "metrics.csv")
        r = best_row(csv_path, mse_col="val_mse")
        if r is not None:
            topkaux_pts.append((ckpt, f"K={k}", *r))

    fig, ax = plt.subplots(figsize=(9, 6))

    def scatter(pts, color, marker, label):
        if not pts:
            return
        l0s  = [p[2] for p in pts]
        mses = [p[3] for p in pts]
        ax.scatter(l0s, mses, c=color, marker=marker, s=90, label=label,
                   edgecolors="black", linewidths=0.6, zorder=3)
        order = sorted(range(len(pts)), key=lambda i: l0s[i])
        ax.plot([l0s[i] for i in order], [mses[i] for i in order],
                color=color, alpha=0.4, zorder=2)
        for _, tag, l0, mse, _dead in pts:
            ax.annotate(tag, (l0, mse), xytext=(5, 5),
                        textcoords="offset points", fontsize=8, color=color)

    scatter(l1_pts,      "tab:red",   "o", "L1 SAE")
    scatter(topk_pts,    "tab:blue",  "s", "Top-K (plain)")
    scatter(topkaux_pts, "tab:green", "D", "Top-K + aux")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("L0 (features active per node)")
    ax.set_ylabel("val MSE (element-wise, on normalized embeddings)")
    ax.set_title("SAE Pareto frontier: sparsity vs. reconstruction")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right")

    out = os.path.join(HERE, "figures", "pareto_l0_vs_mse.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"saved {out}")

    print("\nRun summary (L0, val_MSE, dead_frac):")
    for group, pts in [("L1", l1_pts), ("TopK", topk_pts), ("TopK+aux", topkaux_pts)]:
        for ckpt, tag, l0, mse, dead in pts:
            print(f"  [{group:9s}] {tag:14s} L0={l0:7.2f}  MSE={mse:.3e}  dead={dead:.1%}  ({ckpt})")


if __name__ == "__main__":
    main()
