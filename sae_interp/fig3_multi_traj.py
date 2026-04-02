"""
fig3_multi_traj.py

Randomly selects 5 trajectories. Finds the most salient latent dimensions
aggregated across all selected trajectories (by mean_abs). Produces one
figure per salient dimension showing how that dimension activates across
all trajectories and time steps, alongside the ground-truth velocity_x field.

Layout per figure (one per salient dimension):
  rows = n_trajs * 2  (alternating: dim activation row, then velocity_x row)
  cols = 4 evenly-spaced time stamps
  Left label distinguishes "dim XXXX" vs "vel_x"

Usage:
    cd sae_interp/
    python fig3_multi_traj.py \
        --ckpt checkpoints_rand_3e-4_long/sae_best.pt \
        --emb_dir ../sae_embeddings/raw \
        --phys_dir ../sae_embeddings/phys \
        --out_dir ./figures/multi_traj \
        --n_trajs 5 \
        --n_dims 5 \
        --eta 100 \
        --metric mean_abs \
        --seed 42
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


# ── helpers ────────────────────────────────────────────────────────────────

def list_traj_files(emb_dir: str, traj_id: str):
    """Return sorted list of (step_num, filepath) for the trajectory."""
    pattern = f"traj_{traj_id}_step_"
    entries = []
    for fname in os.listdir(emb_dir):
        if fname.startswith(pattern) and fname.endswith(".npz"):
            step_num = int(fname[len(pattern):].split(".")[0])
            entries.append((step_num, os.path.join(emb_dir, fname)))
    if not entries:
        raise FileNotFoundError(
            f"No files matching 'traj_{traj_id}_step_*.npz' in {emb_dir}"
        )
    return sorted(entries)


def encode_file(fpath: str, sae, device):
    """Load one raw .npz, encode through SAE, return (z, mesh_pos, cells)."""
    d = np.load(fpath)
    hL = torch.from_numpy(d["hL"].astype(np.float32)).to(device)
    with torch.no_grad():
        z = sae.encode(hL).cpu().numpy()
    return np.maximum(z, 0.0), d["mesh_pos"], d["cells"]


def load_phys_file(fpath: str):
    """Load one phys .npz, return velocity_x (N,)."""
    d = np.load(fpath)
    return d["velocity"][:, 0]   # velocity_x


def make_triangulation(mesh_pos, cells):
    x, y = mesh_pos[:, 0], mesh_pos[:, 1]
    cells = np.asarray(cells, dtype=np.int32)
    if cells.shape[1] == 4:
        a, b, c, d_ = cells[:, 0], cells[:, 1], cells[:, 2], cells[:, 3]
        cells = np.concatenate(
            [np.stack([a, b, c], 1), np.stack([a, c, d_], 1)], axis=0
        )
    return mtri.Triangulation(x, y, cells)


def top_eta(a, eta):
    eta = min(int(eta), len(a))
    idx = np.argpartition(a, -eta)[-eta:]
    return idx[np.argsort(a[idx])[::-1]]


def pick_t_indices(T, n=4):
    """n evenly-spaced indices spanning [0, T-1]."""
    return [int(round(i * (T - 1) / (n - 1))) for i in range(n)]


# ── streaming pass ─────────────────────────────────────────────────────────

def stream_trajectory(traj_id, emb_dir, phys_dir, sae, device, n_timestamps=4):
    """
    Streams all steps of a trajectory.
    Returns:
        scores : dict with keys 'mean_abs' and 'variance', each (d_hid,)
        snapshots : list of dicts with z, vel_x, mesh_pos, cells, step_num
                    for the n_timestamps evenly-spaced steps
    """
    raw_entries  = list_traj_files(emb_dir, traj_id)
    phys_entries = list_traj_files(phys_dir, traj_id)
    phys_by_step = {sn: fp for sn, fp in phys_entries}

    T = len(raw_entries)
    t_indices = set(pick_t_indices(T, n=n_timestamps))

    # Accumulators for mean_abs and variance (Welford online)
    per_step_mean = None   # will be (T, d_hid), built incrementally
    snapshots_by_t = {}

    print(f"  [traj {traj_id}] streaming {T} steps ...", flush=True)
    for t, (step_num, fpath) in enumerate(raw_entries):
        z, mesh_pos, cells = encode_file(fpath, sae, device)
        step_mean = np.mean(np.abs(z), axis=0)   # (d_hid,)
        if per_step_mean is None:
            per_step_mean = np.empty((T, z.shape[1]), dtype=np.float64)
        per_step_mean[t] = step_mean
        if t in t_indices:
            vel_x = load_phys_file(phys_by_step[step_num])
            snapshots_by_t[t] = dict(
                z=z, vel_x=vel_x,
                mesh_pos=mesh_pos, cells=cells, step_num=step_num,
            )

    scores = {
        "mean_abs": per_step_mean.mean(axis=0),
        "variance": per_step_mean.var(axis=0),
    }
    snapshots = [snapshots_by_t[t] for t in sorted(t_indices)]
    return scores, snapshots, per_step_mean


# ── figure rendering ───────────────────────────────────────────────────────

def render_dim_figure(dim, all_snapshots, traj_ids, eta, out_dir):
    """
    Two figures for latent dimension `dim`:
      _activation.png  – SAE dim activation (Blues) + red nodes
      _velx.png        – velocity_x (Greens) + red nodes
    Both have rows = trajectories, cols = time stamps.
    """
    n_trajs = len(all_snapshots)
    n_times = len(all_snapshots[0])

    fig_act, axes_act = plt.subplots(
        n_trajs, n_times, figsize=(n_times * 3.2, n_trajs * 1.8), squeeze=False,
    )
    fig_vx, axes_vx = plt.subplots(
        n_trajs, n_times, figsize=(n_times * 3.2, n_trajs * 1.8), squeeze=False,
    )

    for i, (traj_id, snapshots) in enumerate(zip(traj_ids, all_snapshots)):
        for col, snap in enumerate(snapshots):
            z        = snap["z"]
            vel_x    = snap["vel_x"]
            mesh_pos = snap["mesh_pos"]
            cells    = snap["cells"]
            step_num = snap["step_num"]

            tri     = make_triangulation(mesh_pos, cells)
            a       = z[:, dim]
            hot_idx = top_eta(a, eta)
            hot_xy  = mesh_pos[hot_idx, :2]

            # ── activation panel ──
            ax = axes_act[i][col]
            a_disp = np.clip(a, 0, np.percentile(a, 99))
            ax.tripcolor(tri, a_disp, shading="gouraud", cmap="Blues", alpha=0.6)
            ax.scatter(hot_xy[:, 0], hot_xy[:, 1], s=6, color="red", linewidths=0)
            ax.set_aspect("equal")
            ax.set_axis_off()
            if i == 0:
                ax.set_title(f"step {step_num}", fontsize=9)
            if col == 0:
                ax.text(-0.02, 0.5, f"traj {traj_id}",
                        transform=ax.transAxes, fontsize=8,
                        va="center", ha="right", rotation=90)

            # ── velocity_x panel ──
            ax = axes_vx[i][col]
            vx_disp = np.clip(vel_x,
                              np.percentile(vel_x, 1),
                              np.percentile(vel_x, 99))
            ax.tripcolor(tri, vx_disp, shading="flat", cmap="Greens")
            ax.scatter(hot_xy[:, 0], hot_xy[:, 1], s=6, color="red", linewidths=0)
            ax.set_aspect("equal")
            ax.set_axis_off()
            if i == 0:
                ax.set_title(f"step {step_num}", fontsize=9)
            if col == 0:
                ax.text(-0.02, 0.5, f"traj {traj_id}",
                        transform=ax.transAxes, fontsize=8,
                        va="center", ha="right", rotation=90)

    for fig, suffix, title in [
        (fig_act, "activation", f"Dim {dim} – SAE activation ($\\eta$={eta})"),
        (fig_vx,  "velx",      f"Dim {dim} – velocity_x + top-$\\eta$ nodes ($\\eta$={eta})"),
    ]:
        fig.suptitle(title, fontsize=11)
        fig.tight_layout()
        out = os.path.join(out_dir, f"fig3_dim_{dim:04d}_{suffix}.png")
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"  [dim {dim}] saved -> {out}", flush=True)


# ── per-trajectory activation figure ──────────────────────────────────────

def render_traj_figure(traj_id, snapshots, top_dims, eta, out_dir):
    """
    One figure per trajectory: rows = 4 time steps, cols = top salient dims.
    Saved as fig3_traj_XXXX_activation.png.
    """
    n_times = len(snapshots)
    n_dims  = len(top_dims)

    fig, axes = plt.subplots(
        n_times, n_dims,
        figsize=(n_dims * 3.2, n_times * 1.8),
        squeeze=False,
    )

    for row, snap in enumerate(snapshots):
        z        = snap["z"]
        mesh_pos = snap["mesh_pos"]
        cells    = snap["cells"]
        step_num = snap["step_num"]
        tri      = make_triangulation(mesh_pos, cells)

        for col, dim in enumerate(top_dims):
            ax     = axes[row][col]
            a      = z[:, dim]
            hot_idx = top_eta(a, eta)
            hot_xy  = mesh_pos[hot_idx, :2]
            a_disp  = np.clip(a, 0, np.percentile(a, 99))
            ax.tripcolor(tri, a_disp, shading="gouraud", cmap="Blues", alpha=0.6)
            ax.scatter(hot_xy[:, 0], hot_xy[:, 1], s=6, color="red", linewidths=0)
            ax.set_aspect("equal")
            ax.set_axis_off()
            if row == 0:
                ax.set_title(f"dim {dim}", fontsize=9)
            if col == 0:
                ax.text(-0.02, 0.5, f"step {step_num}",
                        transform=ax.transAxes, fontsize=8,
                        va="center", ha="right", rotation=90)

    fig.suptitle(f"Traj {traj_id} – top salient dims ($\\eta$={eta})", fontsize=11)
    fig.tight_layout()
    out = os.path.join(out_dir, f"fig3_traj_{traj_id}_activation.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  [traj {traj_id}] saved -> {out}", flush=True)


# ── main ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",     default="checkpoints_rand_3e-4/sae_best.pt")
    p.add_argument("--emb_dir",  default="../sae_embeddings/raw")
    p.add_argument("--phys_dir", default="../sae_embeddings/phys")
    p.add_argument("--out_dir",  default="./figures/multi_traj")
    p.add_argument("--n_trajs", type=int, default=5,
                   help="number of trajectories to sample at random")
    p.add_argument("--n_dims",  type=int, default=5,
                   help="number of top salient dimensions to plot (one figure each)")
    p.add_argument("--eta",     type=int, default=100,
                   help="number of highlighted nodes in dim activation panels")
    p.add_argument("--metric", default="mean_abs",
                   choices=["mean_abs", "variance"],
                   help="mean_abs: globally active dims; variance: spiky/transient dims")
    p.add_argument("--seed",    type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    def resolve(p_):
        return p_ if os.path.isabs(p_) else os.path.join(script_dir, p_)
    ckpt_path = resolve(args.ckpt)
    emb_dir   = resolve(args.emb_dir)
    phys_dir  = resolve(args.phys_dir)

    # Load SAE
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sae  = SparseAutoencoder(
        d_in=int(ckpt.get("d_in", 128)),
        expansion=int(ckpt.get("expansion", 8)),
    ).to(device)
    sae.load_state_dict(ckpt["sae_state"], strict=True)
    sae.eval()
    print(f"loaded SAE from {ckpt_path}")

    # Discover trajectory IDs present in both emb_dir and phys_dir
    def traj_ids_in(d):
        return set(
            f.split("_step_")[0][len("traj_"):]
            for f in os.listdir(d)
            if f.startswith("traj_") and f.endswith(".npz")
        )
    all_ids = sorted(traj_ids_in(emb_dir) & traj_ids_in(phys_dir))
    print(f"trajectories available in both dirs: {len(all_ids)}")
    if len(all_ids) < args.n_trajs:
        raise ValueError(f"Only {len(all_ids)} trajectories found in both dirs, need {args.n_trajs}")

    chosen_ids = sorted(rng.choice(all_ids, size=args.n_trajs, replace=False).tolist())
    print(f"randomly selected trajectories (seed={args.seed}): {chosen_ids}")

    # Stream each trajectory
    all_scores = []
    all_snapshots = []
    for traj_id in chosen_ids:
        scores, snapshots, _ = stream_trajectory(traj_id, emb_dir, phys_dir, sae, device)
        all_scores.append(scores[args.metric])
        all_snapshots.append(snapshots)

    # Aggregate chosen metric across trajectories → top-n_dims
    agg_scores = np.mean(np.stack(all_scores, axis=0), axis=0)
    top_dims = np.argsort(agg_scores)[-args.n_dims:][::-1]
    print(f"\nmetric={args.metric}, aggregated top-{args.n_dims} dims: {top_dims.tolist()}")

    # One figure per dimension
    print()
    for dim in top_dims:
        render_dim_figure(int(dim), all_snapshots, chosen_ids, args.eta, args.out_dir)

    # Per-trajectory activation figure (rows=timesteps, cols=top dims)
    print()
    for traj_id, snapshots in zip(chosen_ids, all_snapshots):
        render_traj_figure(traj_id, snapshots, top_dims, args.eta, args.out_dir)

    print("\ndone.")


if __name__ == "__main__":
    main()
