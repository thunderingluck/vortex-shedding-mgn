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
        z = sae.encode(h).cpu().numpy()
        Z_list.append(z)
        vel_list.append(s["velocity"])              # (N, 2)
        pres_list.append(s["pressure"])             # (N, 1)
    return (
        np.concatenate(Z_list,    axis=0),   # (M*N, d_hid)
        np.concatenate(vel_list,  axis=0),   # (M*N, 2)
        np.concatenate(pres_list, axis=0),   # (M*N, 1)
    )


# ── correlation ───────────────────────────────────────────────────────────────

def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r between two flat arrays."""
    xm = x - x.mean()
    ym = y - y.mean()
    denom = np.sqrt((xm**2).sum() * (ym**2).sum())
    return float(np.dot(xm, ym) / denom) if denom > 1e-12 else 0.0


def compute_correlations(Z, vel, pres):
    """
    Returns dict of arrays shape (d_hid,):
        r_u, r_v, r_p, r_speed, r_abs_max
    """
    u     = vel[:, 0]
    v     = vel[:, 1]
    p     = pres[:, 0]
    speed = np.sqrt(u**2 + v**2)

    d = Z.shape[1]
    r_u     = np.zeros(d)
    r_v     = np.zeros(d)
    r_p     = np.zeros(d)
    r_speed = np.zeros(d)

    for i in range(d):
        zi = Z[:, i]
        if zi.max() < 1e-12:   # dead feature – skip
            continue
        r_u[i]     = pearson_r(zi, u)
        r_v[i]     = pearson_r(zi, v)
        r_p[i]     = pearson_r(zi, p)
        r_speed[i] = pearson_r(zi, speed)

    r_abs_max = np.max(np.abs(np.stack([r_u, r_v, r_p, r_speed])), axis=0)
    return dict(r_u=r_u, r_v=r_v, r_p=r_p, r_speed=r_speed, r_abs_max=r_abs_max)


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
        z = sae.encode(h).cpu().numpy()          # (N, d_hid)
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
            z   = sae.encode(h).cpu().numpy()
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


def fig_scatter(Z_flat, vel, pres, dims, corr, out_path, max_pts=5000):
    """Scatter: feature activation vs best physical field."""
    u, v = vel[:, 0], vel[:, 1]
    p    = pres[:, 0]
    speed = np.sqrt(u**2 + v**2)
    fields = {"r_u": u, "r_v": v, "r_p": p, "r_speed": speed}

    ncols = min(4, len(dims))
    nrows = (len(dims) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols*3, nrows*2.5), squeeze=False)
    rng = np.random.default_rng(0)

    for k, dim in enumerate(dims):
        ax  = axes[k // ncols][k % ncols]
        zi  = Z_flat[:, dim]
        vals = [corr["r_u"][dim], corr["r_v"][dim],
                corr["r_p"][dim], corr["r_speed"][dim]]
        best_key  = ["r_u","r_v","r_p","r_speed"][int(np.argmax(np.abs(vals)))]
        best_r    = vals[int(np.argmax(np.abs(vals)))]
        phys      = fields[best_key]

        # subsample for readability
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

    Z_all, vel_all, pres_all = [], [], []
    for tid in corr_traj_ids:
        snaps = load_trajectory(phys_dir, tid)
        Z, vel, pres = encode_snaps(snaps, sae, device)
        Z_all.append(Z); vel_all.append(vel); pres_all.append(pres)
        print(f"  traj {tid:04d}: {len(snaps)} snaps, {Z.shape[0]} node-steps")

    Z_flat   = np.concatenate(Z_all,    axis=0)
    vel_flat = np.concatenate(vel_all,  axis=0)
    pres_flat= np.concatenate(pres_all, axis=0)
    print(f"total: {Z_flat.shape[0]} node-steps, {Z_flat.shape[1]} features")

    # ── correlations ──────────────────────────────────────────────────────────
    print("computing correlations ...")
    corr = compute_correlations(Z_flat, vel_flat, pres_flat)
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
        Z_flat, vel_flat, pres_flat, top_dims, corr,
        os.path.join(args.out_dir, "top_features_scatter.png"),
    )

    print("done.")


if __name__ == "__main__":
    main()
