"""
make_paper_figures.py

Reproduces Figures 2 and 3 from the paper:
  Fig 2: Aggregated salient latent dimensions on mesh, for eta in [20, 85, 300],
          at 4 representative time steps.
  Fig 3: Individual latent dimension activations (top-3 by mean_abs),
          for eta=100, at 4 representative time steps.

Usage:
    cd sae_interp/
    python make_paper_figures.py \
        --ckpt checkpoints_rand_3e-4/sae_best.pt \
        --emb_dir ../sae_embeddings/raw \
        --traj_id 0006 \
        --out_dir ./figures
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import torch

# Allow running from within sae_interp/ or from repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sae import SparseAutoencoder
from saliency import select_topk_global


# ── helpers ──────────────────────────────────────────────────────────────────

def load_trajectory(emb_dir: str, traj_id: str, step_indices: list = None):
    """Return list of (hL, mesh_pos, cells) for steps of traj_id, sorted.

    If step_indices is given, only load those indices (0-based into the sorted
    file list) rather than all steps.
    """
    pattern = f"traj_{traj_id}_step_"
    files = sorted(
        f for f in os.listdir(emb_dir) if f.startswith(pattern) and f.endswith(".npz")
    )
    if not files:
        raise FileNotFoundError(
            f"No files matching 'traj_{traj_id}_step_*.npz' in {emb_dir}"
        )
    if step_indices is not None:
        files = [files[i] for i in step_indices if i < len(files)]
    steps = []
    for fname in files:
        # extract step number from filename, e.g. traj_0006_step_0042.npz -> 42
        stem = fname[len(pattern):]            # e.g. "0042.npz"
        step_num = int(stem.split(".")[0])
        d = np.load(os.path.join(emb_dir, fname))
        steps.append(
            dict(
                hL=d["hL"].astype(np.float32),
                mesh_pos=d["mesh_pos"],
                cells=d["cells"],
                step_num=step_num,
            )
        )
    return steps


def make_triangulation(mesh_pos, cells):
    x, y = mesh_pos[:, 0], mesh_pos[:, 1]
    cells = np.asarray(cells, dtype=np.int32)
    if cells.shape[1] == 4:                       # quads → two triangles
        a, b, c, d = cells[:, 0], cells[:, 1], cells[:, 2], cells[:, 3]
        cells = np.concatenate(
            [np.stack([a, b, c], 1), np.stack([a, c, d], 1)], axis=0
        )
    return mtri.Triangulation(x, y, cells)


def encode_all(steps, sae, device):
    """Return Z_list: list of (N_t, d_hid) numpy arrays."""
    Z_list = []
    for s in steps:
        h = torch.from_numpy(s["hL"]).to(device)
        with torch.no_grad():
            z = sae.encode(h).cpu().numpy()
        Z_list.append(np.maximum(z, 0.0))   # ReLU activations
    return Z_list


def top_eta(a, eta):
    eta = min(int(eta), len(a))
    idx = np.argpartition(a, -eta)[-eta:]
    return idx[np.argsort(a[idx])[::-1]]


def aggregate(Z_t, dims):
    return Z_t[:, dims].sum(axis=1)


# ── Figure 2 ─────────────────────────────────────────────────────────────────

def figure2(Z_list, steps_data, topk_dims, t_indices, eta_list, out_dir, metric, traj_id):
    """
    Rows = time steps, Cols = eta values.
    Shows red scatter of top-eta nodes on the mesh background.
    """
    nrows = len(t_indices)
    ncols = len(eta_list)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 3.2, nrows * 2.4),
        squeeze=False,
    )

    for col, eta in enumerate(eta_list):
        for row, t in enumerate(t_indices):
            ax = axes[row][col]
            Z_t = Z_list[t]
            mesh_pos = steps_data[t]["mesh_pos"]
            cells = steps_data[t]["cells"]

            a = aggregate(Z_t, topk_dims)
            hot_idx = top_eta(a, eta)
            hot_xy = mesh_pos[hot_idx, :2]

            tri = make_triangulation(mesh_pos, cells)
            a_disp = np.clip(a, 0, np.percentile(a, 99))

            ax.tripcolor(tri, a_disp, shading="gouraud", cmap="Blues", alpha=0.6)
            ax.scatter(hot_xy[:, 0], hot_xy[:, 1], s=6, color="red", linewidths=0)
            ax.set_aspect("equal")
            ax.set_axis_off()

            if row == 0:
                ax.set_title(f"$\\eta$={eta}", fontsize=10)
            if col == 0:
                ax.text(-0.02, 0.5, f"step {steps_data[t]['step_num']}",
                        transform=ax.transAxes, fontsize=9,
                        va="center", ha="right", rotation=90)

    fig.suptitle(
        f"Fig 2 – Aggregated salient dims (traj={traj_id}, metric={metric}, K={len(topk_dims)})",
        fontsize=11,
    )
    fig.tight_layout()
    out = os.path.join(out_dir, "fig2_aggregated.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[fig2] saved -> {out}")


# ── Figure 3 ─────────────────────────────────────────────────────────────────

def figure3(Z_list, steps_data, topk_dims, t_indices, eta, out_dir, traj_id, n_dims=3):
    """
    Shows n_dims individual latent dimensions (top n_dims from mean_abs Top-K),
    as columns; rows = time steps.
    """
    # Pick n_dims representative dims from the top-K selection
    chosen_dims = topk_dims[:n_dims]

    nrows = len(t_indices)
    ncols = n_dims
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 3.2, nrows * 2.4),
        squeeze=False,
    )

    for col, dim in enumerate(chosen_dims):
        for row, t in enumerate(t_indices):
            ax = axes[row][col]
            Z_t = Z_list[t]
            mesh_pos = steps_data[t]["mesh_pos"]
            cells = steps_data[t]["cells"]

            a = Z_t[:, dim]   # single dimension activation
            hot_idx = top_eta(a, eta)
            hot_xy = mesh_pos[hot_idx, :2]

            tri = make_triangulation(mesh_pos, cells)
            a_disp = np.clip(a, 0, np.percentile(a, 99))

            ax.tripcolor(tri, a_disp, shading="gouraud", cmap="Blues", alpha=0.6)
            ax.scatter(hot_xy[:, 0], hot_xy[:, 1], s=6, color="red", linewidths=0)
            ax.set_aspect("equal")
            ax.set_axis_off()

            if row == 0:
                ax.set_title(f"Dim {dim}", fontsize=10)
            if col == 0:
                ax.text(-0.02, 0.5, f"step {steps_data[t]['step_num']}",
                        transform=ax.transAxes, fontsize=9,
                        va="center", ha="right", rotation=90)

    fig.suptitle(
        f"Fig 3 – Individual latent dimensions (traj={traj_id}, mean_abs Top-{n_dims}, $\\eta$={eta})",
        fontsize=11,
    )
    fig.tight_layout()
    out = os.path.join(out_dir, "fig3_individual_dims.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[fig3] saved -> {out}")


# ── Figure 1 (Jaccard) ───────────────────────────────────────────────────────

def figure1_jaccard(Z_list, topk_dims_global, K, out_dir):
    """
    Jaccard similarity between global and time-local Top-K, for 3 metrics.
    """
    from saliency import select_topk_global, jaccard

    metrics = ["variance", "mean_abs", "entropy"]
    colors  = ["tab:blue", "tab:orange", "tab:green"]
    results = {m: [] for m in metrics}

    global_sets = {}
    for m in metrics:
        g_dims, _ = select_topk_global(
            np.concatenate(Z_list, axis=0), k=K, metric=m
        )
        global_sets[m] = set(map(int, g_dims))

    T = len(Z_list)
    for t, Z_t in enumerate(Z_list):
        for m in metrics:
            local_dims, _ = select_topk_global(Z_t, k=K, metric=m)
            local_set = set(map(int, local_dims))
            j = jaccard(global_sets[m], local_set)
            results[m].append(j)

    fig, ax = plt.subplots(figsize=(8, 4))
    for m, c in zip(metrics, colors):
        ax.plot(range(T), results[m], label=m, color=c, lw=1.0)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Jaccard coefficient")
    ax.set_title(f"Fig 1 – Global vs. Time-Local Top-{K} Salient Coordinates")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(out_dir, "fig1_jaccard.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[fig1] saved -> {out}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints_rand_3e-4/sae_best.pt")
    p.add_argument("--emb_dir", default="../sae_embeddings/raw")
    p.add_argument("--traj_id", default="0006",
                   help="4-digit trajectory id, e.g. 0006")
    p.add_argument("--out_dir", default="./figures")
    p.add_argument("--metric", default="mean_abs",
                   choices=["variance", "mean_abs", "entropy"])
    p.add_argument("--topk", type=int, default=50)
    p.add_argument("--eta_list", default="20,85,300",
                   help="comma-separated eta values for Fig 2")
    p.add_argument("--eta_fig3", type=int, default=100)
    p.add_argument("--t_steps", default="10,50,200,500",
                   help="comma-separated snapshot indices for plotting")
    p.add_argument("--jaccard", action="store_true",
                   help="also produce Fig 1 (Jaccard) — slow for large traj")
    p.add_argument("--fig3_only", action="store_true",
                   help="only produce Fig 3; loads only the needed timesteps to save memory")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # ── load SAE ────────────────────────────────────────────────────────────
    ckpt_path = args.ckpt
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sae  = SparseAutoencoder(
        d_in=int(ckpt.get("d_in", 128)),
        expansion=int(ckpt.get("expansion", 8)),
    ).to(device)
    sae.load_state_dict(ckpt["sae_state"], strict=True)
    sae.eval()
    print(f"loaded SAE from {ckpt_path}")

    # ── load trajectory ──────────────────────────────────────────────────────
    emb_dir = args.emb_dir
    if not os.path.isabs(emb_dir):
        emb_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), emb_dir)

    t_indices_raw = [int(x) for x in args.t_steps.split(",")]
    selective_indices = t_indices_raw if args.fig3_only else None
    steps_data = load_trajectory(emb_dir, args.traj_id, step_indices=selective_indices)
    T = len(steps_data)
    print(f"loaded trajectory {args.traj_id}: {T} steps, "
          f"{steps_data[0]['hL'].shape[0]} nodes")

    # ── encode ──────────────────────────────────────────────────────────────
    Z_list = encode_all(steps_data, sae, device)

    # ── global top-K using mean_abs (for Figs 2 & 3) ─────────────────────
    Z_all = np.concatenate(Z_list, axis=0)
    topk_dims, scores = select_topk_global(Z_all, k=args.topk, metric=args.metric)
    np.save(os.path.join(args.out_dir, "global_topk_dims.npy"), topk_dims)
    np.save(os.path.join(args.out_dir, "global_scores.npy"), scores)
    print(f"top-{args.topk} dims (metric={args.metric}): {topk_dims[:10]} ...")

    # ── Figure 3 only: use mean_abs over the same (small) Z_all ─────────────
    from saliency import select_topk_global as stg
    dims_mean_abs, _ = stg(Z_all, k=10, metric="mean_abs")
    del Z_all  # free the large concatenated array

    # ── time step indices ────────────────────────────────────────────────────
    if args.fig3_only:
        # steps_data contains exactly the requested steps; use sequential indices
        t_indices = list(range(T))
    else:
        t_indices = [min(int(x), T - 1) for x in args.t_steps.split(",")]
    eta_list  = [int(x) for x in args.eta_list.split(",")]

    # ── Figure 2 ────────────────────────────────────────────────────────────
    if not args.fig3_only:
        figure2(Z_list, steps_data, topk_dims, t_indices, eta_list, args.out_dir, args.metric, args.traj_id)

    # ── Figure 3 ────────────────────────────────────────────────────────────
    figure3(Z_list, steps_data, dims_mean_abs, t_indices, args.eta_fig3, args.out_dir, args.traj_id, n_dims=3)

    # ── Figure 1 (optional, slow) ────────────────────────────────────────────
    if args.jaccard:
        figure1_jaccard(Z_list, topk_dims, args.topk, args.out_dir)

    print("done.")


if __name__ == "__main__":
    main()
