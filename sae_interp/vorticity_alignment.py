"""
vorticity_alignment.py

Replicates the quantitative evaluation of Section 5.4 of
Hu & Liu (2025) arXiv:2507.16069.

Pipeline (per trajectory):
  1. Encode all snapshots with the frozen SAE.
  2. Select K globally-salient feature dimensions using three criteria
     (variance, mean_abs, entropy) as defined in Table 1 of the paper.
  3. For each snapshot, compute a node-level saliency score by summing
     activations across the K selected dimensions:
         a_i = Σ_{d∈K} z^(d)_i   (Section 4.5)
  4. Treat the top-vortex_pct nodes by |ω| as ground truth (Section 5.4).
     Vorticity ω = ∂v/∂x − ∂u/∂y is computed via Green-Gauss on the
     triangular mesh.
  5. Compare the top-eta nodes by a_i against the vorticity mask and record
     precision, recall, F1, Jaccard (averaged over snapshots and trajectories).
  6. Repeat for three baselines: embedding-norm, PCA (m = ⌊K/κ⌋ components),
     and random.

Outputs:
  vorticity_alignment_summary.csv  — one row per method (replicates Table 2)
  vorticity_alignment_per_feature.csv — per-feature breakdown (bonus)

Usage (from sae_interp/):
    python vorticity_alignment.py \\
        --phys_dir ../sae_embeddings/phys \\
        --out_dir  ./figures/phys_analysis \\
        --all_traj \\
        --K 50 --eta 100 --vortex_pct 0.10
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_features import load_trajectory, load_sae, list_trajectories


# ── vorticity ─────────────────────────────────────────────────────────────────

def precompute_vorticity_geometry(mesh_pos: np.ndarray, cells: np.ndarray):
    """
    Precompute the static (mesh-only) factors for vorticity.
    Call once per trajectory; reuse across all snapshots.

    Returns a callable fast_vorticity(velocity) -> (N,).
    """
    N  = mesh_pos.shape[0]
    x, y = mesh_pos[:, 0], mesh_pos[:, 1]
    i0, i1, i2 = cells[:, 0], cells[:, 1], cells[:, 2]

    x0, y0 = x[i0], y[i0]
    x1, y1 = x[i1], y[i1]
    x2, y2 = x[i2], y[i2]

    A2     = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    A2     = np.where(np.abs(A2) < 1e-14, 1e-14, A2)
    area_T = np.abs(A2) / 2.0

    # shape function gradient coefficients (C,) per corner
    dvdx = [(y1 - y2) / A2, (y2 - y0) / A2, (y0 - y1) / A2]
    dudy = [(x2 - x1) / A2, (x0 - x2) / A2, (x1 - x0) / A2]

    # precompute node denominators (sum of adjacent triangle areas)
    denom = np.zeros(N)
    corner_ids = [i0, i1, i2]
    for c in corner_ids:
        denom += np.bincount(c, weights=area_T, minlength=N)
    denom = np.where(denom < 1e-14, 1.0, denom)

    # precompute scatter index arrays for np.bincount
    all_ids = np.concatenate([i0, i1, i2])  # (3C,)
    C = len(cells)

    def fast_vorticity(velocity: np.ndarray) -> np.ndarray:
        u, v = velocity[:, 0], velocity[:, 1]
        # vorticity per triangle
        omega_T = (
            dvdx[0] * v[i0] + dvdx[1] * v[i1] + dvdx[2] * v[i2]
          - dudy[0] * u[i0] - dudy[1] * u[i1] - dudy[2] * u[i2]
        )
        # area-weighted scatter via np.bincount (faster than np.add.at)
        weights = np.tile(omega_T * area_T, 3)
        num = np.bincount(all_ids, weights=weights, minlength=N)
        return num / denom

    return fast_vorticity


# ── saliency scoring (Table 1) ────────────────────────────────────────────────

def score_variance(Z: np.ndarray) -> np.ndarray:
    """Z: (M, d_hid) -> (d_hid,)  variance of activations."""
    return Z.var(axis=0)


def score_mean_abs(Z: np.ndarray) -> np.ndarray:
    """Z: (M, d_hid) -> (d_hid,)  mean absolute activation.
    SAE activations are always >= 0 (ReLU), so abs is a no-op."""
    return Z.mean(axis=0)


def score_entropy(Z: np.ndarray, B: int = 50) -> np.ndarray:
    """Z: (M, d_hid) -> (d_hid,)  entropy of activation histogram (B bins).

    Transposes Z once so each feature's values are a contiguous row, avoiding
    cache-thrashing from non-contiguous column access on large row-major arrays.
    """
    Zt = np.ascontiguousarray(Z.T)   # (d_hid, M) — one-time copy
    d  = Zt.shape[0]
    s  = np.zeros(d)
    for i in range(d):
        counts, _ = np.histogram(Zt[i], bins=B)
        p = counts / max(counts.sum(), 1)
        p = p[p > 0]
        s[i] = -np.dot(p, np.log(p))
    return s


def global_top_k(Z: np.ndarray, score_fn, K: int) -> np.ndarray:
    """Return indices of K features with highest global score."""
    return np.argsort(score_fn(Z))[-K:]


# ── PCA baseline ──────────────────────────────────────────────────────────────

def fit_pca(H: np.ndarray, m: int):
    """
    Fit PCA on H (M, d_in) and return (components (m, d_in), mean (d_in,)).
    Forms the (d_in, d_in) covariance matrix first — O(M·d²) as a single
    BLAS dgemm call, much faster than calling SVD on the full (M, d_in) matrix.
    """
    mean = H.mean(axis=0)
    Hc   = H - mean
    cov  = Hc.T @ Hc                          # (d_in, d_in)
    _, Vt = np.linalg.eigh(cov)               # eigenvalues ascending
    return Vt[:, ::-1].T[:m], mean            # (m, d_in), (d_in,)


def pca_projection_norm(hL: np.ndarray, components: np.ndarray,
                        mean: np.ndarray) -> np.ndarray:
    """Node-level L2 norm of PCA projection. hL: (N, d_in) -> (N,)."""
    proj = (hL - mean) @ components.T   # (N, m)
    return np.linalg.norm(proj, axis=1)


# ── metrics helpers ───────────────────────────────────────────────────────────

def prf1j(top_set: set, mask_set: set, eta: int, k_vort: int) -> tuple:
    tp   = len(top_set & mask_set)
    prec = tp / eta
    rec  = tp / k_vort
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    jacc = tp / len(top_set | mask_set)
    return prec, rec, f1, jacc


# ── per-trajectory evaluation ─────────────────────────────────────────────────

@torch.no_grad()
def evaluate_trajectory(snaps, sae, device, K: int, eta: int,
                        vortex_pct: float, entropy_bins: int = 50,
                        expansion: int = 8):
    """
    Encode all snapshots, select K global features by each criterion, then
    evaluate aggregated node saliency vs vorticity mask for every snapshot.

    Returns dict: method_name -> list of (prec, rec, f1, jacc) per snapshot.
    """
    mesh_pos = snaps[0]["mesh_pos"]
    cells    = snaps[0]["cells"].astype(np.int32)
    N        = mesh_pos.shape[0]
    T        = len(snaps)
    k_vort   = max(1, int(vortex_pct * N))
    m_pca    = max(1, K // expansion)   # ⌊K/κ⌋ PCA components (paper Sec 5.4)

    # ── precompute static mesh geometry for vorticity ────────────────────────
    fast_vorticity = precompute_vorticity_geometry(mesh_pos, cells)

    # ── batch encode full trajectory (one GPU call) ───────────────────────────
    H = np.concatenate([s["hL"] for s in snaps], axis=0)          # (T*N, d_in)
    with torch.no_grad():
        Z = sae.encode(torch.from_numpy(H).to(device)).cpu().numpy()  # (T*N, d_hid)

    # ── global K selection ────────────────────────────────────────────────────
    K_var  = global_top_k(Z, score_variance,                       K)
    K_abs  = global_top_k(Z, score_mean_abs,                       K)
    K_ent  = global_top_k(Z, lambda z: score_entropy(z, entropy_bins), K)

    # ── PCA fit ───────────────────────────────────────────────────────────────
    pca_comp, pca_mean = fit_pca(H, m_pca)

    # ── per-snapshot evaluation ───────────────────────────────────────────────
    methods = ["SAE (variance)", "SAE (mean_abs)", "SAE (entropy)",
               "Embedding-norm", "PCA", "Random"]
    results = {m: [] for m in methods}
    rng = np.random.default_rng(0)

    for t, s in enumerate(snaps):
        omega     = fast_vorticity(s["velocity"])
        vort_mask = set(np.argsort(np.abs(omega))[-k_vort:].tolist())

        z_t  = Z[t * N:(t + 1) * N]    # (N, d_hid)
        hL_t = H[t * N:(t + 1) * N]    # (N, d_in)

        # SAE variants: aggregate activations over K selected dims
        for name, K_dims in [("SAE (variance)", K_var),
                              ("SAE (mean_abs)", K_abs),
                              ("SAE (entropy)",  K_ent)]:
            scores  = z_t[:, K_dims].sum(axis=1)
            top_eta = set(np.argsort(scores)[-eta:].tolist())
            results[name].append(prf1j(top_eta, vort_mask, eta, k_vort))

        # Embedding-norm baseline: rank by ||hL||_2
        emb_norm = np.linalg.norm(hL_t, axis=1)
        top_eta  = set(np.argsort(emb_norm)[-eta:].tolist())
        results["Embedding-norm"].append(prf1j(top_eta, vort_mask, eta, k_vort))

        # PCA baseline: rank by ||PCA projection||_2
        proj_norm = pca_projection_norm(hL_t, pca_comp, pca_mean)
        top_eta   = set(np.argsort(proj_norm)[-eta:].tolist())
        results["PCA"].append(prf1j(top_eta, vort_mask, eta, k_vort))

        # Random baseline
        top_eta = set(rng.choice(N, size=eta, replace=False).tolist())
        results["Random"].append(prf1j(top_eta, vort_mask, eta, k_vort))

    return results


# ── aggregation ───────────────────────────────────────────────────────────────

def aggregate(all_results: dict) -> dict:
    """
    all_results: method -> list of (prec, rec, f1, jacc) across all snapshots
                           and all trajectories.
    Returns: method -> {precision, recall, f1, jaccard}
    """
    out = {}
    for method, vals in all_results.items():
        arr = np.array(vals)   # (total_snaps, 4)
        out[method] = dict(precision=float(arr[:, 0].mean()),
                           recall   =float(arr[:, 1].mean()),
                           f1       =float(arr[:, 2].mean()),
                           jaccard  =float(arr[:, 3].mean()))
    return out


# ── I/O ───────────────────────────────────────────────────────────────────────

def save_summary_csv(summary: dict, out_path: str):
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "precision", "recall", "f1", "jaccard"])
        for method, m in summary.items():
            w.writerow([method, f"{m['precision']:.4f}", f"{m['recall']:.4f}",
                        f"{m['f1']:.4f}", f"{m['jaccard']:.4f}"])
    print(f"[vort] summary saved -> {out_path}")


def print_table(summary: dict):
    print(f"\n{'Method':<20}  {'Precision':>9}  {'Recall':>6}  "
          f"{'F1':>6}  {'Jaccard':>7}")
    print("-" * 58)
    for method, m in summary.items():
        print(f"{method:<20}  {m['precision']:>9.4f}  {m['recall']:>6.4f}  "
              f"{m['f1']:>6.4f}  {m['jaccard']:>7.4f}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",         default="checkpoints_rand_3e-4_long/sae_best.pt")
    p.add_argument("--phys_dir",     default="../sae_embeddings/phys")
    p.add_argument("--out_dir",      default="./figures/phys_analysis")
    p.add_argument("--traj_id",      type=int, default=0,
                   help="Trajectory to use (ignored when --all_traj is set)")
    p.add_argument("--all_traj",     action="store_true",
                   help="Evaluate across all available trajectories")
    p.add_argument("--K",            type=int, default=50,
                   help="Number of salient SAE dimensions to select (default 50, "
                        "matching Figure 1 of the paper)")
    p.add_argument("--eta",          type=int, default=100,
                   help="Number of top-activated nodes to compare against "
                        "the vorticity mask (default 100)")
    p.add_argument("--vortex_pct",   type=float, default=0.10,
                   help="Fraction of nodes treated as high-vorticity (default 0.10)")
    p.add_argument("--entropy_bins", type=int, default=50,
                   help="Histogram bins for entropy scoring (default 50)")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    base      = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = args.ckpt     if os.path.isabs(args.ckpt)     else os.path.join(base, args.ckpt)
    phys_dir  = args.phys_dir if os.path.isabs(args.phys_dir) else os.path.join(base, args.phys_dir)

    sae = load_sae(ckpt_path, device)
    print(f"loaded SAE  d_in={sae.d_in}  d_hid={sae.d_hid}  "
          f"expansion={sae.d_hid // sae.d_in}")
    expansion = sae.d_hid // sae.d_in

    traj_ids = list_trajectories(phys_dir)
    eval_ids = traj_ids if args.all_traj else (
        [args.traj_id] if args.traj_id in traj_ids else [traj_ids[0]])
    print(f"evaluating {len(eval_ids)} trajectories  "
          f"K={args.K}  eta={args.eta}  vortex_pct={args.vortex_pct:.0%}")

    methods  = ["SAE (variance)", "SAE (mean_abs)", "SAE (entropy)",
                "Embedding-norm", "PCA", "Random"]
    all_res  = {m: [] for m in methods}

    for tid in eval_ids:
        snaps = load_trajectory(phys_dir, tid)
        print(f"  traj {tid:04d}: {len(snaps)} snaps, "
              f"{snaps[0]['mesh_pos'].shape[0]} nodes ...", end=" ", flush=True)
        res = evaluate_trajectory(snaps, sae, device,
                                  K=args.K, eta=args.eta,
                                  vortex_pct=args.vortex_pct,
                                  entropy_bins=args.entropy_bins,
                                  expansion=expansion)
        for m in methods:
            all_res[m].extend(res[m])
        best_f1 = max(np.mean([v[2] for v in res[m]]) for m in methods[:3])
        print(f"best SAE F1={best_f1:.4f}")

    summary = aggregate(all_res)
    print_table(summary)
    save_summary_csv(summary,
                     os.path.join(args.out_dir, "vorticity_alignment_summary.csv"))
    print("done.")


if __name__ == "__main__":
    main()
