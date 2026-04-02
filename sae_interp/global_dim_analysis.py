"""
global_dim_analysis.py

Finds the 5 most prominent SAE latent dimensions across ALL trajectories,
measures how much the per-trajectory ranking of those dimensions varies,
and produces one figure per prominent dimension showing a few trajectories
across 4 representative time steps.

Memory-efficient: streams all trajectories once (O(d_hid) working memory),
then loads only the required snapshot files for plotting.

Usage (from sae_interp/):
    python global_dim_analysis.py \
        --ckpt   checkpoints_rand_3e-4_long/sae_best.pt \
        --emb_dir ../sae_embeddings/raw \
        --out_dir ./figures/global_dims \
        --metric  mean_abs \
        --n_dims  5 \
        --n_trajs 4 \
        --t_steps 10,50,200,500
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
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sae import SparseAutoencoder


# ── I/O ───────────────────────────────────────────────────────────────────────

def list_trajectories(emb_dir: str):
    """Return sorted list of unique traj-id strings found in emb_dir."""
    ids = set()
    for f in os.listdir(emb_dir):
        if f.startswith("traj_") and f.endswith(".npz"):
            ids.add(f[5:9])
    return sorted(ids)


def sorted_traj_files(emb_dir: str, traj_id: str):
    pattern = f"traj_{traj_id}_step_"
    return sorted(
        f for f in os.listdir(emb_dir)
        if f.startswith(pattern) and f.endswith(".npz")
    )


def load_snap(emb_dir: str, fname: str):
    d = np.load(os.path.join(emb_dir, fname))
    return dict(
        hL=d["hL"].astype(np.float32),
        mesh_pos=d["mesh_pos"],
        cells=d["cells"],
    )


def load_snap_with_stepnum(emb_dir: str, fname: str, pattern: str):
    step_num = int(fname[len(pattern):].split(".")[0])
    s = load_snap(emb_dir, fname)
    s["step_num"] = step_num
    return s


def load_sae(ckpt_path: str, device: str) -> SparseAutoencoder:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sae  = SparseAutoencoder(
        d_in=int(ckpt.get("d_in", 128)),
        expansion=int(ckpt.get("expansion", 8)),
    ).to(device)
    sae.load_state_dict(ckpt["sae_state"], strict=True)
    sae.eval()
    return sae


# ── streaming score accumulators ──────────────────────────────────────────────

class MeanAbsAccum:
    """Streaming mean |z| per dimension. O(d_hid) memory."""
    def __init__(self, d):
        self.sum_abs = np.zeros(d, dtype=np.float64)
        self.count   = 0

    def update(self, Z: np.ndarray):   # Z: (N, d)
        self.sum_abs += np.abs(Z).sum(axis=0)
        self.count   += Z.shape[0]

    def result(self):
        return self.sum_abs / max(1, self.count)


class VarianceAccum:
    """Batch Welford online variance. O(d_hid) memory."""
    def __init__(self, d):
        self.n    = 0
        self.mean = np.zeros(d, dtype=np.float64)
        self.M2   = np.zeros(d, dtype=np.float64)

    def update(self, Z: np.ndarray):   # Z: (N, d)
        n2    = Z.shape[0]
        mean2 = Z.mean(axis=0)
        M2_2  = ((Z - mean2) ** 2).sum(axis=0)
        n     = self.n + n2
        delta = mean2 - self.mean
        self.mean += delta * (n2 / n)
        self.M2   += M2_2 + delta ** 2 * (self.n * n2 / n)
        self.n = n

    def result(self):
        return self.M2 / max(1, self.n)   # population variance


def make_accum(metric: str, d: int):
    if metric == "mean_abs":
        return MeanAbsAccum(d)
    elif metric == "variance":
        return VarianceAccum(d)
    else:
        raise ValueError(f"Streaming not supported for metric='{metric}'. "
                         "Use 'mean_abs' or 'variance'.")


# ── encoding ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_hL(hL: np.ndarray, sae, device) -> np.ndarray:
    h = torch.from_numpy(hL).to(device)
    z = sae.encode(h).cpu().numpy()
    return np.maximum(z, 0.0)


@torch.no_grad()
def encode_hL_batched(hL: np.ndarray, sae, device, batch: int = 65536) -> np.ndarray:
    """Encode a large (M, d_in) array in GPU-sized chunks. Returns (M, d_hid)."""
    out = []
    for i in range(0, len(hL), batch):
        chunk = torch.from_numpy(hL[i:i + batch]).to(device)
        out.append(sae.encode(chunk).cpu().numpy())
    Z = np.concatenate(out, axis=0)
    return np.maximum(Z, 0.0)


# ── streaming pass ────────────────────────────────────────────────────────────

def streaming_pass(all_traj_ids, emb_dir, sae, device, metric):
    """
    Single pass over all trajectories.

    Per trajectory: loads all hL arrays (~570 MB for 599 steps × 1850 nodes × 128
    dims), concatenates them, then encodes in large GPU batches — reducing 59 900
    tiny forward passes to ~1 500 large ones.

    Returns:
      global_accum   – score accumulator over all node-steps
      traj_scores    – list of per-traj score vectors (d_hid,)
      traj_n_steps   – list of step counts
    """
    d_hid = sae.d_hid
    global_accum = make_accum(metric, d_hid)
    traj_scores  = []
    traj_n_steps = []

    for tid in all_traj_ids:
        files = sorted_traj_files(emb_dir, tid)
        if not files:
            print(f"  [warn] no files for traj {tid}, skipping")
            traj_scores.append(np.zeros(d_hid))
            traj_n_steps.append(0)
            continue

        # Load only hL (d_in=128) for all steps — one npz key access each.
        # Memory: 599 × 1850 × 128 × 4 B ≈ 570 MB, released after this loop.
        hL_all = np.concatenate(
            [np.load(os.path.join(emb_dir, f))["hL"].astype(np.float32)
             for f in files],
            axis=0,
        )                                          # (T*N, d_in)

        # Single large encode — far fewer GPU round-trips
        Z = encode_hL_batched(hL_all, sae, device)    # (T*N, d_hid)
        del hL_all                                      # free ~570 MB

        traj_accum = make_accum(metric, d_hid)
        global_accum.update(Z)
        traj_accum.update(Z)
        del Z                                           # free Z immediately

        traj_scores.append(traj_accum.result())
        traj_n_steps.append(len(files))
        print(f"  traj {tid}: {len(files)} steps")

    return global_accum, traj_scores, traj_n_steps


# ── top-k selection ───────────────────────────────────────────────────────────

def select_topk(score_vec: np.ndarray, k: int):
    """Return top-k indices sorted by descending score."""
    idx = np.argpartition(score_vec, -k)[-k:]
    return idx[np.argsort(score_vec[idx])[::-1]]


# ── rank-stability analysis ───────────────────────────────────────────────────

def rank_stability(traj_scores, top_dims):
    """
    traj_scores : list of (d_hid,) score vectors, one per trajectory
    top_dims    : (n_top,) global top-dim indices

    Returns:
      rank_matrix   (n_trajs, n_top) – rank of each top dim in each traj (1=best)
      rank_mean     (n_top,)
      rank_std      (n_top,)
      avg_spearman  float – mean pairwise Spearman r across all traj pairs
    """
    d_hid   = traj_scores[0].shape[0]
    n_trajs = len(traj_scores)
    n_top   = len(top_dims)

    rank_matrix = np.zeros((n_trajs, n_top), dtype=float)
    for ti, s in enumerate(traj_scores):
        order      = np.argsort(s)[::-1]
        dense_rank = np.empty(d_hid, dtype=float)
        dense_rank[order] = np.arange(1, d_hid + 1, dtype=float)
        rank_matrix[ti]   = dense_rank[top_dims]

    rank_mean = rank_matrix.mean(axis=0)
    rank_std  = rank_matrix.std(axis=0)

    # Average pairwise Spearman r over the full score vector
    rhos = []
    for i in range(n_trajs):
        for j in range(i + 1, n_trajs):
            rho, _ = spearmanr(traj_scores[i], traj_scores[j])
            rhos.append(rho)
    avg_spearman = float(np.mean(rhos)) if rhos else float("nan")

    return rank_matrix, rank_mean, rank_std, avg_spearman


def print_rank_table(top_dims, rank_matrix, rank_mean, rank_std,
                     avg_spearman, traj_ids):
    print("\n── Rank stability of global top dims ───────────────────────────────")
    # only print ranks for first 20 trajs to keep output manageable
    show = traj_ids[:20]
    header = (f"{'dim':>6}  {'mean_rank':>10}  {'std_rank':>9}  " +
              "  ".join(f"{tid:>8}" for tid in show))
    print(header)
    for k, dim in enumerate(top_dims):
        per = "  ".join(f"{rank_matrix[ti, k]:>8.0f}"
                        for ti in range(len(show)))
        print(f"{dim:>6}  {rank_mean[k]:>10.1f}  {rank_std[k]:>9.1f}  {per}")
    print(f"\nAverage pairwise Spearman r (all dims, all traj pairs): "
          f"{avg_spearman:.4f}")
    print("────────────────────────────────────────────────────────────────────\n")


# ── mesh helper ───────────────────────────────────────────────────────────────

def make_tri(mesh_pos, cells):
    x, y  = mesh_pos[:, 0], mesh_pos[:, 1]
    cells = np.asarray(cells, dtype=np.int32)
    if cells.shape[1] == 4:
        a, b, c, d = cells.T
        cells = np.concatenate([np.stack([a, b, c], 1),
                                 np.stack([a, c, d], 1)])
    return mtri.Triangulation(x, y, cells)


# ── per-dimension figure ──────────────────────────────────────────────────────

def figure_for_dim(dim, plot_snaps, plot_Z, traj_ids,
                   rank_mean_k, rank_std_k, out_path):
    """
    plot_snaps : list[list[dict]]  – [traj][t_step] snapshot dicts
    plot_Z     : list[list[ndarray]] – [traj][t_step] Z arrays (N, d_hid)
    Rows = trajectories, Cols = time steps.
    """
    n_trajs = len(plot_snaps)
    n_cols  = len(plot_snaps[0])

    fig, axes = plt.subplots(
        n_trajs, n_cols,
        figsize=(n_cols * 3.2, n_trajs * 2.2),
        squeeze=False,
    )

    for row, (tid, snaps_row, Z_row) in enumerate(
            zip(traj_ids, plot_snaps, plot_Z)):
        for col, (s, Z_t) in enumerate(zip(snaps_row, Z_row)):
            ax  = axes[row][col]
            a   = Z_t[:, dim]
            tri = make_tri(s["mesh_pos"], s["cells"])
            a_c = np.clip(a, 0, np.percentile(a, 99) + 1e-12)

            im = ax.tripcolor(tri, a_c, shading="gouraud", cmap="viridis")
            ax.set_aspect("equal")
            ax.set_axis_off()

            if row == 0:
                ax.set_title(f"step {s['step_num']}", fontsize=9)
            if col == 0:
                ax.text(-0.04, 0.5, f"traj {tid}",
                        transform=ax.transAxes, fontsize=8,
                        va="center", ha="right", rotation=90)

            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    fig.suptitle(
        f"Latent dim {dim}  "
        f"(global rank mean={rank_mean_k:.0f} ± {rank_std_k:.1f})",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[fig] dim {dim} -> {out_path}")


# ── rank-variation summary figure ─────────────────────────────────────────────

def figure_rank_summary(top_dims, rank_matrix, rank_mean, rank_std,
                         traj_ids, out_path):
    """
    Box plot of per-trajectory ranks for the top global dims, showing
    distribution across all trajectories, with individual points.
    """
    n_dims = len(top_dims)
    data   = [rank_matrix[:, k] for k in range(n_dims)]
    labels = [f"dim {d}" for d in top_dims]

    fig, ax = plt.subplots(figsize=(max(6, n_dims * 1.8), 4))
    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                    medianprops=dict(color="black", linewidth=1.5))

    cmap = plt.get_cmap("tab10")
    for patch, color in zip(bp["boxes"], [cmap(k) for k in range(n_dims)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # scatter individual traj points
    rng = np.random.default_rng(0)
    for k, col_data in enumerate(data):
        jitter = rng.uniform(-0.15, 0.15, size=len(col_data))
        ax.scatter(np.full(len(col_data), k + 1) + jitter, col_data,
                   s=12, alpha=0.5, color="black", zorder=3)

    ax.set_ylabel("Rank (1 = most salient)")
    ax.invert_yaxis()
    ax.set_title(
        f"Per-trajectory rank distribution of global top-{n_dims} dims  "
        f"(n={len(traj_ids)} trajs)"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[fig] rank summary -> {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",     default="checkpoints_rand_3e-4/sae_best.pt")
    p.add_argument("--emb_dir",  default="../sae_embeddings/raw")
    p.add_argument("--out_dir",  default="./figures/global_dims")
    p.add_argument("--metric",   default="mean_abs",
                   choices=["mean_abs", "variance"],
                   help="Salience metric (streaming-compatible)")
    p.add_argument("--n_dims",   type=int, default=5,
                   help="Number of globally prominent dims to study")
    p.add_argument("--n_trajs",  type=int, default=4,
                   help="How many trajectories to show in each figure")
    p.add_argument("--t_steps",  default="10,50,200,500",
                   help="4 comma-separated snapshot indices (0-based) for figure columns")
    p.add_argument("--scores_dir", default=None,
                   help="Load precomputed global_scores.npy, rank_matrix.npy, "
                        "global_top_dims.npy from this directory and skip pass 1")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    base      = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = args.ckpt    if os.path.isabs(args.ckpt)    else os.path.join(base, args.ckpt)
    emb_dir   = args.emb_dir if os.path.isabs(args.emb_dir) else os.path.join(base, args.emb_dir)

    sae = load_sae(ckpt_path, device)
    print(f"loaded SAE  d_in={sae.d_in}  d_hid={sae.d_hid}")

    all_traj_ids = list_trajectories(emb_dir)
    if not all_traj_ids:
        raise RuntimeError(f"No trajectory files found in {emb_dir}")
    print(f"found {len(all_traj_ids)} trajectories")

    t_indices = [int(x) for x in args.t_steps.split(",")]

    # ── streaming pass OR load precomputed scores ─────────────────────────────
    if args.scores_dir:
        sd = args.scores_dir if os.path.isabs(args.scores_dir) \
             else os.path.join(base, args.scores_dir)
        print(f"\n[pass 1] loading precomputed scores from {sd} ...")
        global_score_vec = np.load(os.path.join(sd, "global_scores.npy"))
        rank_matrix      = np.load(os.path.join(sd, "rank_matrix.npy"))
        top_dims         = np.load(os.path.join(sd, "global_top_dims.npy"))
        # Recompute summary stats from loaded rank_matrix
        rank_mean    = rank_matrix.mean(axis=0)
        rank_std     = rank_matrix.std(axis=0)
        # Read Spearman r from saved text file if present
        spearman_path = os.path.join(sd, "spearman.txt")
        if os.path.exists(spearman_path):
            with open(spearman_path) as f:
                avg_spearman = float(f.readline().split("=")[1])
        else:
            avg_spearman = float("nan")
        print(f"  loaded global_scores, rank_matrix ({rank_matrix.shape}), "
              f"top_dims {top_dims.tolist()}")
    else:
        print(f"\n[pass 1] streaming all trajectories (metric={args.metric}) ...")
        global_accum, traj_scores, _ = streaming_pass(
            all_traj_ids, emb_dir, sae, device, args.metric
        )

        global_score_vec = global_accum.result()
        top_dims = select_topk(global_score_vec, args.n_dims)

        rank_matrix, rank_mean, rank_std, avg_spearman = rank_stability(
            traj_scores, top_dims
        )

        np.save(os.path.join(args.out_dir, "global_top_dims.npy"), top_dims)
        np.save(os.path.join(args.out_dir, "global_scores.npy"), global_score_vec)
        np.save(os.path.join(args.out_dir, "rank_matrix.npy"), rank_matrix)
        np.savetxt(
            os.path.join(args.out_dir, "rank_summary.txt"),
            np.column_stack([top_dims, rank_mean, rank_std]),
            header="dim  rank_mean  rank_std",
            fmt=["%d", "%.2f", "%.2f"],
        )
        with open(os.path.join(args.out_dir, "spearman.txt"), "w") as f:
            f.write(f"avg_pairwise_spearman_r = {avg_spearman:.6f}\n")
            f.write(f"n_trajectories = {len(all_traj_ids)}\n")
            f.write(f"metric = {args.metric}\n")

    print(f"\nGlobal top-{args.n_dims} dims: {top_dims.tolist()}")
    print(f"  scores: {global_score_vec[top_dims]}")
    print_rank_table(top_dims, rank_matrix, rank_mean, rank_std,
                     avg_spearman, all_traj_ids)

    # ── choose plotting trajectories (evenly spaced) ──────────────────────────
    n_trajs = min(args.n_trajs, len(all_traj_ids))
    step    = max(1, len(all_traj_ids) // n_trajs)
    plot_indices  = list(range(0, len(all_traj_ids), step))[:n_trajs]
    plot_traj_ids = [all_traj_ids[i] for i in plot_indices]
    print(f"\n[pass 2] loading {n_trajs} plot trajectories: {plot_traj_ids}")

    # Load only the needed snapshot files (4 per trajectory)
    plot_snaps = []   # [traj][t_step]  – snapshot dicts
    plot_Z     = []   # [traj][t_step]  – (N, d_hid) arrays

    for tid in plot_traj_ids:
        files   = sorted_traj_files(emb_dir, tid)
        pattern = f"traj_{tid}_step_"
        T       = len(files)

        snaps_row, Z_row = [], []
        for t in t_indices:
            t_safe = min(t, T - 1)
            s = load_snap_with_stepnum(emb_dir, files[t_safe], pattern)
            Z = encode_hL(s["hL"], sae, device)
            snaps_row.append(s)
            Z_row.append(Z)

        plot_snaps.append(snaps_row)
        plot_Z.append(Z_row)
        print(f"  traj {tid}: loaded {len(snaps_row)} steps")

    # ── per-dimension figures ─────────────────────────────────────────────────
    print("\n[figs] generating per-dimension figures ...")
    for k, dim in enumerate(top_dims):
        out_path = os.path.join(args.out_dir, f"dim_{dim:04d}.png")
        figure_for_dim(
            dim=int(dim),
            plot_snaps=plot_snaps,
            plot_Z=plot_Z,
            traj_ids=plot_traj_ids,
            rank_mean_k=rank_mean[k],
            rank_std_k=rank_std[k],
            out_path=out_path,
        )

    # ── rank summary figure ───────────────────────────────────────────────────
    figure_rank_summary(
        top_dims, rank_matrix, rank_mean, rank_std,
        all_traj_ids,
        os.path.join(args.out_dir, "rank_summary.png"),
    )

    print("\ndone.")


if __name__ == "__main__":
    main()
