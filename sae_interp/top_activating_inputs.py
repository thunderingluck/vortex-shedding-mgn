"""
top_activating_inputs.py

For each globally prominent SAE latent dimension, shows which physical
inputs (velocity u/v, pressure, flow speed, mesh x/y position) are
associated with its highest activations.

Output per dim:
  <out_dir>/dim_XXXX_inputs.png
    2×3 scatter grid: activation vs (u, v, p, speed, x, y)
    each point coloured by activation magnitude

Memory-efficient: only the selected-dim columns of the SAE output are ever
held in memory.  The large hL, Z arrays are freed per trajectory.

Usage (from sae_interp/):
    python top_activating_inputs.py \\
        --ckpt      checkpoints_rand_3e-4_long/sae_best.pt \\
        --phys_dir  ../sae_embeddings/phys \\
        --dims_file ./figures/global_dims/global_top_dims.npy \\
        --out_dir   ./figures/global_dims \\
        --max_pts   30000 \\
        --n_trajs   all
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sae import SparseAutoencoder


# ── I/O helpers ───────────────────────────────────────────────────────────────

def list_trajectories(phys_dir: str):
    ids = set()
    for f in os.listdir(phys_dir):
        if f.startswith("traj_") and f.endswith(".npz"):
            ids.add(f[5:9])
    return sorted(ids)


def sorted_traj_files(phys_dir: str, traj_id: str):
    pattern = f"traj_{traj_id}_step_"
    return sorted(
        f for f in os.listdir(phys_dir)
        if f.startswith(pattern) and f.endswith(".npz")
    )


def load_sae(ckpt_path: str, device: str) -> SparseAutoencoder:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sae = SparseAutoencoder(
        d_in=int(ckpt.get("d_in", 128)),
        expansion=int(ckpt.get("expansion", 8)),
    ).to(device)
    sae.load_state_dict(ckpt["sae_state"], strict=True)
    sae.eval()
    return sae


# ── encoding (keeps only selected dims) ───────────────────────────────────────

@torch.no_grad()
def encode_extract_dims(hL: np.ndarray, sae, device,
                        dims: np.ndarray, batch: int = 65536) -> np.ndarray:
    """
    Encode hL (M, d_in) and return only the requested dim columns.
    Output shape: (M, len(dims)), dtype float32, already ReLU'd.
    """
    out = []
    for i in range(0, len(hL), batch):
        chunk = torch.from_numpy(hL[i : i + batch]).to(device)
        z = sae.encode(chunk)          # (batch, d_hid) — encode already applies ReLU
        out.append(z[:, dims].cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


# ── stratified sampler ────────────────────────────────────────────────────────

class StratifiedSampler:
    """
    Accumulates a fixed number of random samples per trajectory
    into per-dim arrays of shape (n_samples, 7):
      [activation, u, v, p, speed, x, y]
    """
    NCOLS = 7

    def __init__(self, n_dims: int, max_pts_total: int, n_trajs: int, seed: int = 0):
        self.n_dims         = n_dims
        self.max_pts_total  = max_pts_total
        self.pts_per_traj   = max(1, max_pts_total // n_trajs)
        self.rng            = np.random.default_rng(seed)
        # Pre-allocate; will grow lazily
        self.buffers        = [[] for _ in range(n_dims)]

    def update(self, Z_dims: np.ndarray,
               vel: np.ndarray, pres: np.ndarray, pos: np.ndarray):
        """
        Z_dims : (T*N, n_dims)  activations for selected dims
        vel    : (T*N, 2)       [u, v]
        pres   : (T*N, 1)       pressure
        pos    : (T*N, 2)       [x, y]  (mesh_pos repeated T times)
        """
        M = len(Z_dims)
        n_sample = min(self.pts_per_traj, M)
        idx = self.rng.choice(M, n_sample, replace=False)

        u     = vel[idx, 0]
        v     = vel[idx, 1]
        p     = pres[idx, 0]
        speed = np.sqrt(u ** 2 + v ** 2)
        x     = pos[idx, 0]
        y     = pos[idx, 1]

        for ki in range(self.n_dims):
            acts = Z_dims[idx, ki]
            block = np.column_stack([acts, u, v, p, speed, x, y]).astype(np.float32)
            self.buffers[ki].append(block)

    def get(self, ki: int) -> np.ndarray:
        """Return (n, 7) array for dim index ki, subsampled to max_pts_total if needed."""
        if not self.buffers[ki]:
            return np.empty((0, self.NCOLS), dtype=np.float32)
        data = np.concatenate(self.buffers[ki], axis=0)
        if len(data) > self.max_pts_total:
            idx  = self.rng.choice(len(data), self.max_pts_total, replace=False)
            data = data[idx]
        return data


# ── figure ────────────────────────────────────────────────────────────────────

_FIELD_LABELS = ["u (m/s)", "v (m/s)", "pressure", "speed (m/s)", "x position", "y position"]
_FIELD_COLS   = [1, 2, 3, 4, 5, 6]   # column indices in the (7,) row


def figure_for_dim(dim: int, data: np.ndarray, global_rank: int,
                   rank_mean: float, rank_std: float, out_path: str):
    """
    data : (n, 7)  columns = [activation, u, v, p, speed, x, y]
    Produces a 2×3 scatter grid: activation (y) vs each physical field (x),
    coloured by activation magnitude.
    """
    if len(data) == 0:
        print(f"[warn] dim {dim}: no data, skipping figure")
        return

    acts   = data[:, 0]
    vmax   = np.percentile(acts, 99)
    norm   = plt.Normalize(vmin=0, vmax=max(vmax, 1e-12))
    cmap   = plt.cm.viridis
    colors = cmap(norm(acts))

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))

    for col_i, (ax, label, col_idx) in enumerate(
            zip(axes.flat, _FIELD_LABELS, _FIELD_COLS)):
        phys_vals = data[:, col_idx]

        sc = ax.scatter(phys_vals, acts, c=colors, s=4, alpha=0.35,
                        rasterized=True, linewidths=0)
        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel("activation" if col_i % 3 == 0 else "", fontsize=9)
        ax.tick_params(labelsize=8)

        # Overlay a trend line (mean activation in equal-width bins)
        try:
            bins  = np.linspace(np.percentile(phys_vals, 1),
                                np.percentile(phys_vals, 99), 25)
            bin_idx = np.digitize(phys_vals, bins)
            bin_means = [acts[bin_idx == b].mean()
                         for b in range(1, len(bins))
                         if (bin_idx == b).any()]
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            valid = [(c, m) for c, b, m in zip(bin_centers,
                     range(1, len(bins)),
                     [acts[bin_idx == b].mean() if (bin_idx == b).any() else np.nan
                      for b in range(1, len(bins))])
                     if not np.isnan(m)]
            if valid:
                bx, by = zip(*valid)
                ax.plot(bx, by, color="red", lw=1.5, alpha=0.8, label="bin mean")
        except Exception:
            pass

    # Shared colour bar on the right
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes.ravel().tolist(),
                 label="activation", fraction=0.02, pad=0.02)

    fig.suptitle(
        f"Latent dim {dim}  (global rank #{global_rank+1},  "
        f"per-traj mean rank = {rank_mean:.0f} ± {rank_std:.1f})\n"
        f"n = {len(data):,} sampled node-timesteps",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 0.97, 1])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[fig] dim {dim} -> {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",       default="checkpoints_rand_3e-4_long/sae_best.pt")
    p.add_argument("--phys_dir",   default="../sae_embeddings/phys")
    p.add_argument("--dims_file",  default="./figures/global_dims/global_top_dims.npy",
                   help="Path to global_top_dims.npy (output of global_dim_analysis.py)")
    p.add_argument("--rank_matrix_file", default="./figures/global_dims/rank_matrix.npy")
    p.add_argument("--out_dir",    default="./figures/global_dims")
    p.add_argument("--max_pts",    type=int, default=30000,
                   help="Total scatter points to collect per dim")
    p.add_argument("--n_trajs",    default="all",
                   help="Number of trajectories to use ('all' or integer)")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    base      = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = args.ckpt      if os.path.isabs(args.ckpt)      else os.path.join(base, args.ckpt)
    phys_dir  = args.phys_dir  if os.path.isabs(args.phys_dir)  else os.path.join(base, args.phys_dir)
    dims_file = args.dims_file if os.path.isabs(args.dims_file) else os.path.join(base, args.dims_file)

    # ── load SAE + top dims ────────────────────────────────────────────────────
    sae = load_sae(ckpt_path, device)
    print(f"loaded SAE  d_in={sae.d_in}  d_hid={sae.d_hid}")

    top_dims = np.load(dims_file).astype(int)
    print(f"top dims: {top_dims.tolist()}")

    # Load rank stats for figure titles
    rank_means = np.full(len(top_dims), np.nan)
    rank_stds  = np.full(len(top_dims), np.nan)
    rmat_path  = args.rank_matrix_file if os.path.isabs(args.rank_matrix_file) \
                 else os.path.join(base, args.rank_matrix_file)
    if os.path.exists(rmat_path):
        rmat       = np.load(rmat_path)   # (n_trajs, n_dims)
        rank_means = rmat.mean(axis=0)
        rank_stds  = rmat.std(axis=0)

    # ── discover trajectories ─────────────────────────────────────────────────
    all_traj_ids = list_trajectories(phys_dir)
    if not all_traj_ids:
        raise RuntimeError(f"No trajectory files found in {phys_dir}")
    print(f"found {len(all_traj_ids)} trajectories")

    if args.n_trajs == "all":
        traj_ids = all_traj_ids
    else:
        n = int(args.n_trajs)
        step = max(1, len(all_traj_ids) // n)
        traj_ids = all_traj_ids[::step][:n]
    print(f"using {len(traj_ids)} trajectories")

    # ── streaming pass ────────────────────────────────────────────────────────
    sampler = StratifiedSampler(
        n_dims=len(top_dims),
        max_pts_total=args.max_pts,
        n_trajs=len(traj_ids),
    )

    for ti, tid in enumerate(traj_ids):
        files = sorted_traj_files(phys_dir, tid)
        if not files:
            print(f"  [warn] no files for traj {tid}, skipping")
            continue

        T = len(files)
        N = None  # determined from first file

        # Batch-load all arrays for this trajectory
        hL_list   = []
        vel_list  = []
        pres_list = []

        for fname in files:
            d = np.load(os.path.join(phys_dir, fname))
            hL_list.append(d["hL"].astype(np.float32))
            vel_list.append(d["velocity"].astype(np.float32))   # (N, 2)
            pres_list.append(d["pressure"].astype(np.float32))  # (N, 1)
            if N is None:
                mesh_pos = d["mesh_pos"].astype(np.float32)     # (N, 2) — static
                N = mesh_pos.shape[0]

        hL_all   = np.concatenate(hL_list,   axis=0)   # (T*N, d_in)
        vel_all  = np.concatenate(vel_list,  axis=0)   # (T*N, 2)
        pres_all = np.concatenate(pres_list, axis=0)   # (T*N, 1)
        pos_all  = np.tile(mesh_pos, (T, 1))           # (T*N, 2)
        del hL_list, vel_list, pres_list

        # Encode and extract only the needed dim columns
        Z_dims = encode_extract_dims(hL_all, sae, device, top_dims)
        del hL_all

        sampler.update(Z_dims, vel_all, pres_all, pos_all)
        del Z_dims, vel_all, pres_all, pos_all

        print(f"  traj {tid} ({ti+1}/{len(traj_ids)}): {T} steps, {T*N:,} node-timesteps")

    # ── generate figures ──────────────────────────────────────────────────────
    print("\n[figs] generating per-dimension input figures ...")
    for ki, dim in enumerate(top_dims):
        data     = sampler.get(ki)
        out_path = os.path.join(args.out_dir, f"dim_{dim:04d}_inputs.png")
        figure_for_dim(
            dim        = int(dim),
            data       = data,
            global_rank= ki,
            rank_mean  = float(rank_means[ki]),
            rank_std   = float(rank_stds[ki]),
            out_path   = out_path,
        )

    print("\ndone.")


if __name__ == "__main__":
    main()
