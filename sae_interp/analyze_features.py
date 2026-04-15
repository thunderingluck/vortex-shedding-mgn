"""
analyze_features.py

Correlates SAE feature activations with physical fields (velocity, pressure)
and produces:
  1. features_correlation.csv  – per-feature Pearson r with u, v, p, speed
  2. top_features_spatial.png  – spatial activation maps for top-N features
  3. top_features_temporal.png – mean activation over time for top-N features
  4. top_features_scatter.png  – activation vs best-correlated physical field

Usage (from sae_interp/):
    python analyze_features.py \
        --ckpt checkpoints_rand_3e-4_long/sae_best.pt \
        --phys_dir ../sae_embeddings/phys \
        --out_dir ./figures/phys_analysis \
        --traj_id 0          # which trajectory to use for spatial/temporal plots
        --topn 12            # how many features to plot
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sae import SparseAutoencoder


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_trajectory(phys_dir: str, traj_id: int):
    """Return sorted list of snapshot dicts for a given trajectory."""
    prefix = f"traj_{traj_id:04d}_step_"
    files  = sorted(
        f for f in os.listdir(phys_dir)
        if f.startswith(prefix) and f.endswith(".npz")
    )
    if not files:
        raise FileNotFoundError(f"No files for traj {traj_id:04d} in {phys_dir}")
    snaps = []
    for fname in files:
        d = np.load(os.path.join(phys_dir, fname))
        snaps.append({k: d[k] for k in d.files})
    return snaps


def list_trajectories(phys_dir: str):
    ids = set()
    for f in os.listdir(phys_dir):
        if f.startswith("traj_") and f.endswith(".npz"):
            ids.add(int(f[5:9]))
    return sorted(ids)


def load_sae(ckpt_path: str, device: str) -> SparseAutoencoder:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sae  = SparseAutoencoder(
        d_in=int(ckpt.get("d_in", 128)),
        expansion=int(ckpt.get("expansion", 8)),
    ).to(device)
    sae.load_state_dict(ckpt["sae_state"], strict=True)
    sae.eval()
    return sae


def make_tri(mesh_pos, cells):
    x, y  = mesh_pos[:, 0], mesh_pos[:, 1]
    cells = np.asarray(cells, dtype=np.int32)
    if cells.shape[1] == 4:
        a, b, c, d = cells.T
        cells = np.concatenate([np.stack([a,b,c],1), np.stack([a,c,d],1)])
    return mtri.Triangulation(x, y, cells)


# ── encoding ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_snaps(snaps, sae, device):
    """Returns Z (M*N, d_hid), vel (M*N, 2), pres (M*N, 1) stacked over all snapshots."""
    Z_list, vel_list, pres_list = [], [], []
    for s in snaps:
        h = torch.from_numpy(s["hL"]).to(device)
        z = sae.encode(h).detach().cpu().numpy()
        Z_list.append(z)
        vel_list.append(s["velocity"])              # (N, 2)
        pres_list.append(s["pressure"])             # (N, 1)
    return (
        np.concatenate(Z_list,    axis=0),   # (M*N, d_hid)
        np.concatenate(vel_list,  axis=0),   # (M*N, 2)
        np.concatenate(pres_list, axis=0),   # (M*N, 1)
    )


# ── correlation ───────────────────────────────────────────────────────────────

class CorrAccumulator:
    """
    Streams snapshots one at a time and accumulates the sufficient statistics
    needed for Pearson r (n, Σz, Σz², Σy, Σy², Σzy) without storing Z in memory.

    Physical fields tracked: [u, v, p, speed]  (indices 0-3).
    """

    _FIELDS = ["r_u", "r_v", "r_p", "r_speed"]

    def __init__(self, d_hid: int):
        self.d      = d_hid
        self.n      = 0
        self.sum_z  = np.zeros(d_hid, dtype=np.float64)
        self.sum_z2 = np.zeros(d_hid, dtype=np.float64)
        self.sum_y  = np.zeros(4,     dtype=np.float64)
        self.sum_y2 = np.zeros(4,     dtype=np.float64)
        self.sum_zy = np.zeros((d_hid, 4), dtype=np.float64)  # (d, 4)

    def update(self, Z: np.ndarray, vel: np.ndarray, pres: np.ndarray):
        """Z: (M, d_hid)  vel: (M, 2)  pres: (M, 1)"""
        u     = vel[:, 0].astype(np.float64)
        v     = vel[:, 1].astype(np.float64)
        p     = pres[:, 0].astype(np.float64)
        speed = np.sqrt(u**2 + v**2)
        Y     = np.stack([u, v, p, speed], axis=1)   # (M, 4)
        Z64   = Z.astype(np.float64)

        self.n       += Z64.shape[0]
        self.sum_z   += Z64.sum(axis=0)
        self.sum_z2  += (Z64 ** 2).sum(axis=0)
        self.sum_y   += Y.sum(axis=0)
        self.sum_y2  += (Y ** 2).sum(axis=0)
        self.sum_zy  += Z64.T @ Y                    # (d, 4)

    def finalize(self) -> dict:
        """Return dict of arrays shape (d_hid,): r_u, r_v, r_p, r_speed, r_abs_max."""
        n       = self.n
        num     = n * self.sum_zy - self.sum_z[:, None] * self.sum_y[None, :]  # (d, 4)
        denom_z = np.maximum(n * self.sum_z2 - self.sum_z ** 2, 0.0)           # (d,)
        denom_y = np.maximum(n * self.sum_y2 - self.sum_y ** 2, 0.0)           # (4,)
        denom   = np.sqrt(denom_z[:, None] * denom_y[None, :])                 # (d, 4)
        r       = np.where(denom > 1e-12, num / denom, 0.0)                    # (d, 4)
        r_abs_max = np.max(np.abs(r), axis=1)
        return dict(r_u=r[:, 0], r_v=r[:, 1], r_p=r[:, 2],
                    r_speed=r[:, 3], r_abs_max=r_abs_max)


def save_csv(corr, out_path):
    import csv
    d = len(corr["r_u"])
    rows = sorted(range(d), key=lambda i: -corr["r_abs_max"][i])
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature", "r_u", "r_v", "r_p", "r_speed", "r_abs_max", "best_field"])
        for i in rows:
            vals = [corr["r_u"][i], corr["r_v"][i], corr["r_p"][i], corr["r_speed"][i]]
            best = ["r_u","r_v","r_p","r_speed"][int(np.argmax(np.abs(vals)))]
            w.writerow([i, *[f"{x:.4f}" for x in vals],
                        f"{corr['r_abs_max'][i]:.4f}", best])
    print(f"[corr] saved -> {out_path}")


# ── temporal profile ──────────────────────────────────────────────────────────

@torch.no_grad()
def temporal_profiles(snaps, sae, device, dims):
    """mean activation per timestep for selected dims. Returns (T, len(dims))."""
    out = []
    for s in snaps:
        h = torch.from_numpy(s["hL"]).to(device)
        z = sae.encode(h).detach().cpu().numpy()          # (N, d_hid)
        out.append(z[:, dims].mean(axis=0))      # (len(dims),)
    return np.stack(out)                          # (T, len(dims))


# ── figures ───────────────────────────────────────────────────────────────────

def fig_spatial(snaps, sae, device, dims, t_indices, out_path):
    """Activation maps on mesh for selected dims at selected timesteps."""
    nrows = len(t_indices)
    ncols = len(dims)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols*3, nrows*2.2), squeeze=False)
    for col, dim in enumerate(dims):
        for row, t in enumerate(t_indices):
            ax  = axes[row][col]
            s   = snaps[t]
            h   = torch.from_numpy(s["hL"]).to(device)
            z   = sae.encode(h).detach().cpu().numpy()
            a   = z[:, dim]
            tri = make_tri(s["mesh_pos"], s["cells"])
            a_c = np.clip(a, 0, np.percentile(a, 99) + 1e-12)
            ax.tripcolor(tri, a_c, shading="gouraud", cmap="viridis")
            ax.set_aspect("equal"); ax.set_axis_off()
            if row == 0:
                ax.set_title(f"feat {dim}", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"t={t}", fontsize=8)
    fig.suptitle("Spatial activation maps (top features by |r|)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[fig] saved -> {out_path}")


def fig_temporal(profiles, dims, corr, out_path):
    """Mean activation over time for each selected dim."""
    T    = profiles.shape[0]
    ts   = np.arange(T) * 0.01        # dt=0.01s from meta.json
    ncols = min(4, len(dims))
    nrows = (len(dims) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols*3.5, nrows*2.2), squeeze=False)
    for k, dim in enumerate(dims):
        ax = axes[k // ncols][k % ncols]
        ax.plot(ts, profiles[:, k], lw=0.9)
        r  = corr["r_abs_max"][dim]
        best_fields = {"r_u":"u", "r_v":"v", "r_p":"p", "r_speed":"speed"}
        vals  = [corr["r_u"][dim], corr["r_v"][dim],
                 corr["r_p"][dim], corr["r_speed"][dim]]
        best  = best_fields[["r_u","r_v","r_p","r_speed"][int(np.argmax(np.abs(vals)))]]
        ax.set_title(f"feat {dim}  |r|={r:.3f} ({best})", fontsize=8)
        ax.set_xlabel("time (s)", fontsize=7)
        ax.set_ylabel("mean act.", fontsize=7)
        ax.tick_params(labelsize=7)
    # hide unused axes
    for k in range(len(dims), nrows*ncols):
        axes[k // ncols][k % ncols].set_visible(False)
    fig.suptitle("Temporal mean activation (top features by |r|)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[fig] saved -> {out_path}")


def fig_scatter(snaps, sae, device, dims, corr, out_path, max_pts=5000):
    """Scatter: feature activation vs best physical field (uses one trajectory)."""
    zi_lists   = {dim: [] for dim in dims}
    phys_lists = {k: [] for k in ("u", "v", "p", "speed")}

    for s in snaps:
        h = torch.from_numpy(s["hL"]).to(device)
        z = sae.encode(h).detach().cpu().numpy()
        u = s["velocity"][:, 0]; v = s["velocity"][:, 1]
        p = s["pressure"][:, 0]
        for dim in dims:
            zi_lists[dim].append(z[:, dim])
        phys_lists["u"].append(u); phys_lists["v"].append(v)
        phys_lists["p"].append(p)
        phys_lists["speed"].append(np.sqrt(u**2 + v**2))

    zi_cat   = {dim: np.concatenate(zi_lists[dim])   for dim in dims}
    phys_cat = {k:   np.concatenate(phys_lists[k])   for k in phys_lists}
    fields   = {"r_u": phys_cat["u"], "r_v": phys_cat["v"],
                "r_p": phys_cat["p"], "r_speed": phys_cat["speed"]}

    ncols = min(4, len(dims))
    nrows = (len(dims) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols*3, nrows*2.5), squeeze=False)
    rng = np.random.default_rng(0)

    for k, dim in enumerate(dims):
        ax  = axes[k // ncols][k % ncols]
        zi  = zi_cat[dim]
        vals = [corr["r_u"][dim], corr["r_v"][dim],
                corr["r_p"][dim], corr["r_speed"][dim]]
        best_key = ["r_u","r_v","r_p","r_speed"][int(np.argmax(np.abs(vals)))]
        best_r   = vals[int(np.argmax(np.abs(vals)))]
        phys     = fields[best_key]

        mask = zi > 0
        idx  = np.where(mask)[0]
        if len(idx) > max_pts:
            idx = rng.choice(idx, max_pts, replace=False)
        ax.scatter(phys[idx], zi[idx], s=1, alpha=0.3, rasterized=True)
        ax.set_xlabel(best_key[2:], fontsize=8)
        ax.set_ylabel("activation", fontsize=8)
        ax.set_title(f"feat {dim}  r={best_r:.3f}", fontsize=8)
        ax.tick_params(labelsize=7)

    for k in range(len(dims), nrows*ncols):
        axes[k // ncols][k % ncols].set_visible(False)
    fig.suptitle("Feature activation vs best-correlated physical field", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[fig] saved -> {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",      default="checkpoints_rand_3e-4_long/sae_best.pt")
    p.add_argument("--phys_dir",  default="../sae_embeddings/phys")
    p.add_argument("--out_dir",   default="./figures/phys_analysis")
    p.add_argument("--traj_id",   type=int, default=0,
                   help="Trajectory to use for spatial/temporal plots")
    p.add_argument("--topn",      type=int, default=12,
                   help="Number of top-correlated features to plot")
    p.add_argument("--t_steps",   default="50,150,300,450",
                   help="Snapshot indices for spatial plots")
    p.add_argument("--all_traj",  action="store_true",
                   help="Use all available trajectories for correlation (slow)")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # resolve paths relative to script dir
    base = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(base, args.ckpt)
    phys_dir  = args.phys_dir if os.path.isabs(args.phys_dir) else os.path.join(base, args.phys_dir)

    sae = load_sae(ckpt_path, device)
    print(f"loaded SAE  d_in={sae.d_in}  d_hid={sae.d_hid}")

    # ── choose trajectories for correlation ───────────────────────────────────
    traj_ids = list_trajectories(phys_dir)
    if not traj_ids:
        raise RuntimeError(f"No trajectory files found in {phys_dir}")
    print(f"found trajectories: {traj_ids}")

    corr_traj_ids = traj_ids if args.all_traj else [traj_ids[0]]
    print(f"using trajectories for correlation: {corr_traj_ids}")

    # ── streaming correlation (no Z_flat in memory) ───────────────────────────
    # Encode one full trajectory at a time (batched for GPU efficiency), then
    # immediately discard Z after updating the accumulator.
    acc        = CorrAccumulator(sae.d_hid)
    total_rows = 0
    for tid in corr_traj_ids:
        snaps = load_trajectory(phys_dir, tid)
        H   = np.concatenate([s["hL"]       for s in snaps], axis=0)  # (T*N, d_in)
        vel = np.concatenate([s["velocity"] for s in snaps], axis=0)
        pre = np.concatenate([s["pressure"] for s in snaps], axis=0)
        with torch.no_grad():
            z = sae.encode(torch.from_numpy(H).to(device)).cpu().numpy()
        acc.update(z, vel, pre)
        total_rows += z.shape[0]
        del H, z
        print(f"  traj {tid:04d}: {len(snaps)} snaps accumulated")

    print(f"total: {total_rows} node-steps, {sae.d_hid} features")
    print("computing correlations ...")
    corr = acc.finalize()
    save_csv(corr, os.path.join(args.out_dir, "features_correlation.csv"))

    top_dims = np.argsort(corr["r_abs_max"])[::-1][:args.topn].tolist()
    print(f"top-{args.topn} features by |r|: {top_dims}")
    print("  feature  |  r_u    r_v    r_p   r_speed  best")
    for d in top_dims:
        vals = [corr["r_u"][d], corr["r_v"][d], corr["r_p"][d], corr["r_speed"][d]]
        best = ["u","v","p","speed"][int(np.argmax(np.abs(vals)))]
        print(f"  {d:5d}    | {vals[0]:+.3f}  {vals[1]:+.3f}  {vals[2]:+.3f}  {vals[3]:+.3f}   {best}")

    # ── load the plotting trajectory ──────────────────────────────────────────
    plot_tid = args.traj_id if args.traj_id in traj_ids else traj_ids[0]
    plot_snaps = load_trajectory(phys_dir, plot_tid)
    t_indices  = [min(int(x), len(plot_snaps)-1)
                  for x in args.t_steps.split(",")]

    # ── figures ───────────────────────────────────────────────────────────────
    fig_spatial(
        plot_snaps, sae, device, top_dims, t_indices,
        os.path.join(args.out_dir, "top_features_spatial.png"),
    )

    profiles = temporal_profiles(plot_snaps, sae, device, top_dims)
    fig_temporal(
        profiles, top_dims, corr,
        os.path.join(args.out_dir, "top_features_temporal.png"),
    )

    fig_scatter(
        plot_snaps, sae, device, top_dims, corr,
        os.path.join(args.out_dir, "top_features_scatter.png"),
    )

    print("done.")


if __name__ == "__main__":
    main()
