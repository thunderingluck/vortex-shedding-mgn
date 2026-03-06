# examples/cfd/vortex_shedding_mgn/sae_interp/viz_mesh.py
#
# Visualize SAE activations on the vortex-shedding mesh.
# - Loads per-snapshot embeddings saved by extract_embeddings.py (emb_XXXXXX.npz)
# - Loads trained SAE checkpoint (sae_best.pt)
# - Computes z = SAE.encode(hL)
# - Selects Top-K latent dims (global or time-local) using variance/mean_abs/entropy
# - Aggregates node activation a_{i,t} = sum_{d in K_t} z_{i,t}^{(d)}
# - Plots mesh heatmap of a_{i,t} and highlights top-eta nodes
#
# Produces per-snapshot PNGs (and optionally an MP4 if ffmpeg exists)

import glob
import os
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

if __package__ in (None, ""):
    # Supports `python viz_mesh.py` and `python -m viz_mesh` from within sae_interp/.
    from sae import SparseAutoencoder
    from saliency import (
        select_topk_global,
        select_topk_time_local,
    )
else:
    # Supports package execution, e.g. `python -m sae_interp.viz_mesh`.
    from .sae import SparseAutoencoder
    from .saliency import (
        select_topk_global,
        select_topk_time_local,
    )

Metric = Literal["variance", "mean_abs", "entropy"]
TopKMode = Literal["global", "time_local"]


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))


def _resolve_existing_path(path: str) -> str:
    """
    Resolve a path robustly for both repo-root and sae_interp working directories.

    Lookup order:
      1) as provided (absolute or relative to CWD)
      2) relative to this file's directory (sae_interp/)
      3) relative to repo root
    """
    candidates = [
        path,
        os.path.join(THIS_DIR, path),
        os.path.join(REPO_ROOT, path),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return os.path.abspath(path)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_npz_files(npz_dir: str) -> List[str]:
    npz_dir_resolved = _resolve_existing_path(npz_dir)
    files = sorted(glob.glob(os.path.join(npz_dir_resolved, "emb_*.npz")))
    if not files:
        raise FileNotFoundError(
            "No emb_*.npz files found. "
            f"Configured embeddings_dir={npz_dir!r}, resolved={npz_dir_resolved!r}."
        )
    return files


def _cells_to_triangles(cells: np.ndarray) -> np.ndarray:
    """
    Convert cells to triangles suitable for matplotlib.tri.Triangulation.

    Expected shapes:
      - (T, 3): triangles already
      - (Q, 4): quads -> split each quad into two triangles
      - (T, 3, 1) or similar: squeeze
    """
    c = np.asarray(cells)
    c = np.squeeze(c)

    if c.ndim != 2:
        raise ValueError(f"cells expected 2D after squeeze, got shape {c.shape}")

    if c.shape[1] == 3:
        return c.astype(np.int32)

    if c.shape[1] == 4:
        # split (a,b,c,d) into (a,b,c) and (a,c,d)
        a = c[:, 0]
        b = c[:, 1]
        cc = c[:, 2]
        d = c[:, 3]
        tri1 = np.stack([a, b, cc], axis=1)
        tri2 = np.stack([a, cc, d], axis=1)
        tris = np.concatenate([tri1, tri2], axis=0)
        return tris.astype(np.int32)

    raise ValueError(f"Unsupported cell arity: {c.shape[1]} (expected 3 or 4)")


def _make_triangulation(mesh_pos: np.ndarray, cells: np.ndarray) -> mtri.Triangulation:
    """
    mesh_pos: (N, 2) or (N, 3). We'll use x,y.
    cells: triangles/quads indices.
    """
    pos = np.asarray(mesh_pos)
    if pos.shape[1] < 2:
        raise ValueError(f"mesh_pos must have at least 2 columns, got {pos.shape}")
    x, y = pos[:, 0], pos[:, 1]
    triangles = _cells_to_triangles(cells)
    return mtri.Triangulation(x, y, triangles)


def _score_metric(Z: np.ndarray, metric: Metric) -> np.ndarray:
    if metric == "variance":
        return Z.var(axis=0)
    if metric == "mean_abs":
        return np.mean(np.abs(Z), axis=0)
    if metric == "entropy":
        # lightweight entropy implementation (bins=50)
        eps = 1e-12
        d = Z.shape[1]
        out = np.zeros(d, dtype=np.float64)
        for j in range(d):
            hist, _ = np.histogram(Z[:, j], bins=50, density=True)
            p = hist / (hist.sum() + eps)
            out[j] = -(p * np.log(p + eps)).sum()
        return out
    raise ValueError(metric)


def _safe_normalize(a: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    mn = np.min(a)
    mx = np.max(a)
    if mx - mn < eps:
        return np.zeros_like(a)
    return (a - mn) / (mx - mn)


def _aggregate_node_activation(Z: np.ndarray, topk_dims: np.ndarray) -> np.ndarray:
    """
    Z: (N, d_hid)
    topk_dims: (K,)
    Returns: a (N,) aggregated activation (sum over selected dims)
    """
    if Z.ndim != 2:
        raise ValueError(f"Z must be 2D (N, d_hid), got {Z.shape}")
    if topk_dims.size == 0:
        return np.zeros(Z.shape[0], dtype=np.float32)
    return Z[:, topk_dims].sum(axis=1)


def _top_eta_indices(a: np.ndarray, eta: int) -> np.ndarray:
    eta = int(eta)
    if eta <= 0:
        return np.array([], dtype=np.int64)
    eta = min(eta, a.shape[0])
    # argpartition for speed; then sort descending among top-eta
    idx = np.argpartition(a, -eta)[-eta:]
    idx = idx[np.argsort(a[idx])[::-1]]
    return idx


def _load_sae_from_ckpt(ckpt_path: str, device: str = "cpu") -> SparseAutoencoder:
    ckpt_path_resolved = _resolve_existing_path(ckpt_path)
    ckpt = torch.load(ckpt_path_resolved, map_location=device)
    sae_cfg = ckpt.get("cfg", {})
    d_in = int(sae_cfg.get("d_in", 128))
    expansion = int(sae_cfg.get("expansion", 8))

    sae = SparseAutoencoder(d_in=d_in, expansion=expansion).to(device)
    sae.load_state_dict(ckpt["sae"], strict=True)
    sae.eval()
    return sae


@dataclass
class VizConfig:
    embeddings_dir: str                  # cfg.sae.out_dir
    sae_ckpt_path: str                   # e.g. cfg.sae.train.ckpt_dir/sae_best.pt
    out_dir: str = "./sae_viz"
    metric: Metric = "variance"          # cfg.sae.analysis.metric
    topk: int = 50                       # cfg.sae.analysis.topk
    eta: int = 100                       # cfg.sae.analysis.eta
    topk_mode: TopKMode = "global"       # "global" or "time_local"
    snapshot_stride: int = 1             # visualize every n-th snapshot
    max_snapshots: Optional[int] = None  # cap number of frames
    use_relu_only: bool = True           # keep Z as nonnegative (paper uses ReLU)
    clamp_percentile: float = 99.0       # for display stability
    make_video: bool = False             # if True, tries ffmpeg
    video_fps: int = 10


def run_viz(cfg: VizConfig) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _ensure_dir(cfg.out_dir)

    files = _load_npz_files(cfg.embeddings_dir)
    if cfg.max_snapshots is not None:
        files = files[: int(cfg.max_snapshots)]

    sae = _load_sae_from_ckpt(cfg.sae_ckpt_path, device=device)

    # Pass 1: compute Top-K dims (global or per-snapshot).
    # We keep activations on CPU numpy for simplicity.
    if cfg.topk_mode == "global":
        # Pool Z over many snapshots (can be heavy). We'll subsample by stride.
        Z_pool = []
        for i, f in enumerate(files[:: cfg.snapshot_stride]):
            d = np.load(f)
            hL = d["hL"].astype(np.float32)  # (N, 128)
            with torch.no_grad():
                z = sae.encode(torch.from_numpy(hL).to(device)).detach().cpu().numpy()
            if cfg.use_relu_only:
                z = np.maximum(z, 0.0)
            Z_pool.append(z)
        Z_all = np.concatenate(Z_pool, axis=0)  # (sum N, d_hid)

        topk_dims, scores = select_topk_global(Z_all, k=int(cfg.topk), metric=cfg.metric)
        # Save scores for inspection
        np.save(os.path.join(cfg.out_dir, "global_feature_scores.npy"), scores)
        np.save(os.path.join(cfg.out_dir, "global_topk_dims.npy"), topk_dims)
        topk_dims_per_snapshot = None

    elif cfg.topk_mode == "time_local":
        # Compute Top-K per snapshot
        Z_by_t = []
        for f in files:
            d = np.load(f)
            hL = d["hL"].astype(np.float32)
            with torch.no_grad():
                z = sae.encode(torch.from_numpy(hL).to(device)).detach().cpu().numpy()
            if cfg.use_relu_only:
                z = np.maximum(z, 0.0)
            Z_by_t.append(z)

        topk_dims_per_snapshot = select_topk_time_local(
            Z_by_t, k=int(cfg.topk), metric=cfg.metric
        )
        # Save dims list
        np.save(os.path.join(cfg.out_dir, "time_local_topk_dims.npy"),
                np.stack(topk_dims_per_snapshot, axis=0))
        topk_dims = None
    else:
        raise ValueError(f"Unknown topk_mode: {cfg.topk_mode}")

    # Pass 2: render frames
    frame_paths = []
    for t, f in enumerate(files):
        if (t % cfg.snapshot_stride) != 0:
            continue

        d = np.load(f)
        hL = d["hL"].astype(np.float32)
        mesh_pos = d["mesh_pos"]
        cells = d["cells"]

        tri = _make_triangulation(mesh_pos, cells)

        with torch.no_grad():
            z = sae.encode(torch.from_numpy(hL).to(device)).detach().cpu().numpy()
        if cfg.use_relu_only:
            z = np.maximum(z, 0.0)

        if cfg.topk_mode == "global":
            dims = topk_dims
        else:
            dims = topk_dims_per_snapshot[t]

        a = _aggregate_node_activation(z, dims)  # (N,)

        # For display: clamp extreme values and normalize (optional but helps)
        if cfg.clamp_percentile is not None and 0 < cfg.clamp_percentile < 100:
            hi = np.percentile(a, cfg.clamp_percentile)
            a_disp = np.clip(a, 0, hi)
        else:
            a_disp = a

        # Find top-eta nodes on *raw* a (not clipped)
        hot_idx = _top_eta_indices(a, cfg.eta)
        hot_xy = mesh_pos[hot_idx, :2]

        # Plot
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(
            f"SAE node activation (mode={cfg.topk_mode}, metric={cfg.metric}, K={cfg.topk}, eta={cfg.eta})\n"
            f"snapshot={t:06d}"
        )

        # tripcolor expects per-node scalar if shading='gouraud', or per-triangle if flat.
        # We'll use gouraud to smoothly show nodewise activation.
        tc = ax.tripcolor(tri, a_disp, shading="gouraud")
        fig.colorbar(tc, ax=ax, fraction=0.046, pad=0.04, label="activation")

        # Overlay hot nodes
        if hot_xy.shape[0] > 0:
            ax.scatter(hot_xy[:, 0], hot_xy[:, 1], s=8, marker="o", linewidths=0)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_axis_off()

        out_path = os.path.join(cfg.out_dir, f"sae_act_{t:06d}.png")
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

        frame_paths.append(out_path)

    print(f"[viz_mesh] wrote {len(frame_paths)} frames to: {cfg.out_dir}")

    if cfg.make_video and len(frame_paths) > 0:
        _try_make_video(cfg.out_dir, fps=cfg.video_fps)


def _try_make_video(out_dir: str, fps: int = 10) -> None:
    """
    Attempts to create an mp4 using ffmpeg:
      ffmpeg -y -framerate FPS -i sae_act_%06d.png -pix_fmt yuv420p sae_act.mp4
    """
    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("[viz_mesh] ffmpeg not found; skipping video.")
        return

    cmd = [
        ffmpeg, "-y",
        "-framerate", str(int(fps)),
        "-i", os.path.join(out_dir, "sae_act_%06d.png"),
        "-pix_fmt", "yuv420p",
        os.path.join(out_dir, "sae_act.mp4"),
    ]
    try:
        subprocess.run(cmd, check=True)
        print(f"[viz_mesh] wrote video -> {os.path.join(out_dir, 'sae_act.mp4')}")
    except subprocess.CalledProcessError as e:
        print(f"[viz_mesh] ffmpeg failed: {e}")


# --- Optional Hydra-friendly wrapper -----------------------------------------
#
# If you want to call this from run_sae_pipeline.py after training,
# build VizConfig from your DictConfig like:
#
#   vc = VizConfig(
#       embeddings_dir=to_absolute_path(cfg.sae.out_dir),
#       sae_ckpt_path=os.path.join(to_absolute_path(cfg.sae.train.ckpt_dir), "sae_best.pt"),
#       out_dir=to_absolute_path(cfg.sae.viz.out_dir),
#       metric=cfg.sae.analysis.metric,
#       topk=cfg.sae.analysis.topk,
#       eta=cfg.sae.analysis.eta,
#       topk_mode=cfg.sae.viz.topk_mode,
#       snapshot_stride=cfg.sae.viz.snapshot_stride,
#       max_snapshots=cfg.sae.viz.max_snapshots,
#       clamp_percentile=cfg.sae.viz.clamp_percentile,
#       make_video=cfg.sae.viz.make_video,
#       video_fps=cfg.sae.viz.video_fps,
#   )
#   run_viz(vc)
#
# and add to sae.yaml something like:
#
# sae:
#   viz:
#     out_dir: ./sae_viz
#     topk_mode: global
#     snapshot_stride: 1
#     max_snapshots: null
#     clamp_percentile: 99.0
#     make_video: false
#     video_fps: 10
#
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    # Minimal manual run (edit paths as needed):
    vc = VizConfig(
        embeddings_dir=os.path.join(REPO_ROOT, "sae_embeddings"),
        sae_ckpt_path=os.path.join(REPO_ROOT, "sae_ckpts", "sae_best.pt"),
        out_dir=os.path.join(REPO_ROOT, "sae_viz"),
        metric="variance",
        topk=50,
        eta=100,
        topk_mode="global",
        snapshot_stride=1,
        make_video=False,
    )
    run_viz(vc)
