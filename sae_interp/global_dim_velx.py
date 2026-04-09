"""
global_dim_velx.py

Re-renders the global_dim_analysis figures but with:
  - background: velocity_x field (Greens colormap)
  - overlay:    top-eta nodes by SAE dim activation (red scatter)

By default loads precomputed scores from figures/global_dims/ so
pass 1 (the full streaming sweep) can be skipped.

Usage (from sae_interp/):
    python global_dim_velx.py \
        --ckpt       checkpoints_rand_3e-4_long/sae_best.pt \
        --emb_dir    ../sae_embeddings/raw \
        --phys_dir   ../sae_embeddings/phys \
        --scores_dir ./figures/global_dims \
        --out_dir    ./figures/global_dims \
        --n_trajs    4 \
        --t_steps    10,50,200,500 \
        --eta        100
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


def load_sae(ckpt_path: str, device: str) -> SparseAutoencoder:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sae  = SparseAutoencoder(
        d_in=int(ckpt.get("d_in", 128)),
        expansion=int(ckpt.get("expansion", 8)),
    ).to(device)
    sae.load_state_dict(ckpt["sae_state"], strict=True)
    sae.eval()
    return sae


# ── streaming helpers ─────────────────────────────────────────────────────────

class MeanAbsAccum:
    def __init__(self, d):
        self.sum_abs = np.zeros(d, dtype=np.float64)
        self.count   = 0

    def update(self, Z: np.ndarray):
        self.sum_abs += np.abs(Z).sum(axis=0)
        self.count   += Z.shape[0]

    def result(self):
        return self.sum_abs / max(1, self.count)


class VarianceAccum:
    def __init__(self, d):
        self.n    = 0
        self.mean = np.zeros(d, dtype=np.float64)
        self.M2   = np.zeros(d, dtype=np.float64)

    def update(self, Z: np.ndarray):
        n2    = Z.shape[0]
        mean2 = Z.mean(axis=0)
        M2_2  = ((Z - mean2) ** 2).sum(axis=0)
        n     = self.n + n2
        delta = mean2 - self.mean
        self.mean += delta * (n2 / n)
        self.M2   += M2_2 + delta ** 2 * (self.n * n2 / n)
        self.n = n

    def result(self):
        return self.M2 / max(1, self.n)


def make_accum(metric: str, d: int):
    if metric == "mean_abs":
        return MeanAbsAccum(d)
    elif metric == "variance":
        return VarianceAccum(d)
    else:
        raise ValueError(f"Streaming not supported for metric='{metric}'.")


@torch.no_grad()
def encode_hL_batched(hL: np.ndarray, sae, device, batch: int = 65536) -> np.ndarray:
    out = []
    for i in range(0, len(hL), batch):
        chunk = torch.from_numpy(hL[i:i + batch]).to(device)
        out.append(sae.encode(chunk).cpu().numpy())
    Z = np.concatenate(out, axis=0)
    return np.maximum(Z, 0.0)


@torch.no_grad()
def encode_hL(hL: np.ndarray, sae, device) -> np.ndarray:
    h = torch.from_numpy(hL).to(device)
    z = sae.encode(h).cpu().numpy()
    return np.maximum(z, 0.0)


def streaming_pass(all_traj_ids, emb_dir, sae, device, metric):
    d_hid = sae.d_hid
    global_accum = make_accum(metric, d_hid)
    traj_scores  = []

    for tid in all_traj_ids:
        files = sorted_traj_files(emb_dir, tid)
        if not files:
            print(f"  [warn] no files for traj {tid}, skipping")
            traj_scores.append(np.zeros(d_hid))
            continue

        hL_all = np.concatenate(
            [np.load(os.path.join(emb_dir, f))["hL"].astype(np.float32)
             for f in files],
            axis=0,
        )
        Z = encode_hL_batched(hL_all, sae, device)
        del hL_all

        traj_accum = make_accum(metric, d_hid)
        global_accum.update(Z)
        traj_accum.update(Z)
        del Z

        traj_scores.append(traj_accum.result())
        print(f"  traj {tid}: {len(files)} steps")

    return global_accum, traj_scores


def select_topk(score_vec: np.ndarray, k: int):
    idx = np.argpartition(score_vec, -k)[-k:]
    return idx[np.argsort(score_vec[idx])[::-1]]


def rank_stability(traj_scores, top_dims):
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

    rhos = []
    for i in range(n_trajs):
        for j in range(i + 1, n_trajs):
            rho, _ = spearmanr(traj_scores[i], traj_scores[j])
            rhos.append(rho)
    avg_spearman = float(np.mean(rhos)) if rhos else float("nan")

    return rank_matrix, rank_mean, rank_std, avg_spearman


# ── mesh / scatter helpers ────────────────────────────────────────────────────

def make_tri(mesh_pos, cells):
    x, y  = mesh_pos[:, 0], mesh_pos[:, 1]
    cells = np.asarray(cells, dtype=np.int32)
    if cells.shape[1] == 4:
        a, b, c, d = cells.T
        cells = np.concatenate([np.stack([a, b, c], 1),
                                 np.stack([a, c, d], 1)])
    return mtri.Triangulation(x, y, cells)


def top_eta(a, eta):
    eta = min(int(eta), len(a))
    idx = np.argpartition(a, -eta)[-eta:]
    return idx[np.argsort(a[idx])[::-1]]


# ── phys loading ──────────────────────────────────────────────────────────────

def load_phys_velx(phys_dir: str, traj_id: str, step_num: int):
    """Load velocity_x (N,) from phys dir for a given traj/step."""
    fname = f"traj_{traj_id}_step_{step_num:04d}.npz"
    fpath = os.path.join(phys_dir, fname)
    d = np.load(fpath)
    return d["velocity"][:, 0]


# ── per-dimension figure ──────────────────────────────────────────────────────

def figure_for_dim_velx(dim, plot_snaps, plot_Z, traj_ids,
                        rank_mean_k, rank_std_k, eta, phys_dir, out_path):
    """
    Like global_dim_analysis's figure_for_dim, but:
      - background: velocity_x (Greens, shading=flat)
      - overlay: top-eta nodes by SAE activation (red scatter)
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
            ax       = axes[row][col]
            a        = Z_t[:, dim]
            tri      = make_tri(s["mesh_pos"], s["cells"])
            hot_idx  = top_eta(a, eta)
            hot_xy   = s["mesh_pos"][hot_idx, :2]

            # Load velocity_x for this snapshot
            vel_x = load_phys_velx(phys_dir, tid, s["step_num"])
            vx_disp = np.clip(vel_x,
                              np.percentile(vel_x, 1),
                              np.percentile(vel_x, 99))

            ax.tripcolor(tri, vx_disp, shading="flat", cmap="Greens")
            ax.scatter(hot_xy[:, 0], hot_xy[:, 1],
                       s=6, color="red", linewidths=0)
            ax.set_aspect("equal")
            ax.set_axis_off()

            if row == 0:
                ax.set_title(f"step {s['step_num']}", fontsize=9)
            if col == 0:
                ax.text(-0.04, 0.5, f"traj {tid}",
                        transform=ax.transAxes, fontsize=8,
                        va="center", ha="right", rotation=90)

    fig.suptitle(
        f"Latent dim {dim}  "
        f"(rank mean={rank_mean_k:.0f} ± {rank_std_k:.1f})  "
        f"– velocity_x + top-$\\eta$ nodes ($\\eta$={eta})",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[fig] dim {dim} -> {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",       default="checkpoints_rand_3e-4_long/sae_best.pt")
    p.add_argument("--emb_dir",    default="../sae_embeddings/raw")
    p.add_argument("--phys_dir",   default="../sae_embeddings/phys")
    p.add_argument("--out_dir",    default="./figures/global_dims")
    p.add_argument("--metric",     default="mean_abs",
                   choices=["mean_abs", "variance"])
    p.add_argument("--n_dims",     type=int, default=5)
    p.add_argument("--n_trajs",    type=int, default=4)
    p.add_argument("--t_steps",    default="10,50,200,500")
    p.add_argument("--eta",        type=int, default=100,
                   help="Number of top-activated nodes to highlight in red")
    p.add_argument("--scores_dir", default=None,
                   help="Load precomputed global_scores.npy, rank_matrix.npy, "
                        "global_top_dims.npy from this dir and skip streaming pass")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    base      = os.path.dirname(os.path.abspath(__file__))
    def resolve(q):
        return q if os.path.isabs(q) else os.path.join(base, q)

    ckpt_path = resolve(args.ckpt)
    emb_dir   = resolve(args.emb_dir)
    phys_dir  = resolve(args.phys_dir)

    sae = load_sae(ckpt_path, device)
    print(f"loaded SAE  d_in={sae.d_in}  d_hid={sae.d_hid}")

    all_traj_ids = list_trajectories(emb_dir)
    if not all_traj_ids:
        raise RuntimeError(f"No trajectory files found in {emb_dir}")
    print(f"found {len(all_traj_ids)} trajectories")

    t_indices = [int(x) for x in args.t_steps.split(",")]

    # ── load scores (precomputed or streaming) ────────────────────────────────
    if args.scores_dir:
        sd = resolve(args.scores_dir)
        print(f"\n[pass 1] loading precomputed scores from {sd} ...")
        global_score_vec = np.load(os.path.join(sd, "global_scores.npy"))
        rank_matrix      = np.load(os.path.join(sd, "rank_matrix.npy"))
        top_dims         = np.load(os.path.join(sd, "global_top_dims.npy"))
        rank_mean        = rank_matrix.mean(axis=0)
        rank_std         = rank_matrix.std(axis=0)
        print(f"  top dims: {top_dims.tolist()}")
    else:
        print(f"\n[pass 1] streaming all trajectories (metric={args.metric}) ...")
        global_accum, traj_scores = streaming_pass(
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
        print(f"  top dims: {top_dims.tolist()}")

    # ── choose plotting trajectories (evenly spaced) ──────────────────────────
    n_trajs = min(args.n_trajs, len(all_traj_ids))
    step    = max(1, len(all_traj_ids) // n_trajs)
    plot_traj_ids = [all_traj_ids[i] for i in range(0, len(all_traj_ids), step)][:n_trajs]
    print(f"\n[pass 2] loading {n_trajs} plot trajectories: {plot_traj_ids}")

    plot_snaps = []
    plot_Z     = []

    for tid in plot_traj_ids:
        files   = sorted_traj_files(emb_dir, tid)
        pattern = f"traj_{tid}_step_"
        T       = len(files)

        snaps_row, Z_row = [], []
        for t in t_indices:
            t_safe   = min(t, T - 1)
            fname    = files[t_safe]
            step_num = int(fname[len(pattern):].split(".")[0])
            d        = np.load(os.path.join(emb_dir, fname))
            hL       = d["hL"].astype(np.float32)
            Z        = encode_hL(hL, sae, device)
            snap     = dict(
                hL=hL, mesh_pos=d["mesh_pos"], cells=d["cells"],
                step_num=step_num,
            )
            snaps_row.append(snap)
            Z_row.append(Z)

        plot_snaps.append(snaps_row)
        plot_Z.append(Z_row)
        print(f"  traj {tid}: loaded {len(snaps_row)} steps")

    # ── per-dimension velx figures ────────────────────────────────────────────
    print("\n[figs] generating velocity-x figures ...")
    for k, dim in enumerate(top_dims):
        out_path = os.path.join(args.out_dir, f"dim_{dim:04d}_velx.png")
        figure_for_dim_velx(
            dim=int(dim),
            plot_snaps=plot_snaps,
            plot_Z=plot_Z,
            traj_ids=plot_traj_ids,
            rank_mean_k=rank_mean[k],
            rank_std_k=rank_std[k],
            eta=args.eta,
            phys_dir=phys_dir,
            out_path=out_path,
        )

    print("\ndone.")


if __name__ == "__main__":
    main()
