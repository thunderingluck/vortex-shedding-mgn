"""Visualize per-feature vorticity-alignment scores (Section 5.4 of Hu & Liu 2025).

Reads:
  figures/phys_analysis/vorticity_alignment.csv   (per-feature P/R/F1/Jaccard)
  figures/phys_analysis/features_correlation.csv  (per-feature corr with u,v,p,speed)

Writes:
  figures/phys_analysis/vorticity_alignment_overview.png
  figures/phys_analysis/vorticity_alignment_top.png

Produces four panels:
  1. Sorted per-feature F1 (all 1024 features) with paper baselines overlaid.
  2. Histogram of F1 across features.
  3. Top-25 features bar chart (F1, with precision/recall annotations).
  4. F1 vs. correlation with physical fields (u, v, p, speed) for top-N features.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PHYS_DIR = os.path.join(HERE, "figures", "phys_analysis")

# Paper Table 2 reference values (averaged over time; their best-performing SAE).
PAPER = {
    "SAE (variance)":  0.60,
    "SAE (mean abs)":  0.60,
    "SAE (entropy)":   0.55,
    "Embedding-norm":  0.55,
    "PCA":             0.49,
    "Random":          0.09,
}


def load_feature_csv():
    feats, prec, rec, f1, jacc = [], [], [], [], []
    with open(os.path.join(PHYS_DIR, "vorticity_alignment.csv")) as f:
        for row in csv.DictReader(f):
            feats.append(int(row["feature"]))
            prec.append(float(row["precision"]))
            rec .append(float(row["recall"]))
            f1  .append(float(row["f1"]))
            jacc.append(float(row["jaccard"]))
    return (np.array(feats), np.array(prec), np.array(rec),
            np.array(f1), np.array(jacc))


def load_corr_csv():
    d = {}
    path = os.path.join(PHYS_DIR, "features_correlation.csv")
    if not os.path.exists(path):
        return d
    with open(path) as f:
        for row in csv.DictReader(f):
            d[int(row["feature"])] = {
                "u":     float(row["r_u"]),
                "v":     float(row["r_v"]),
                "p":     float(row["r_p"]),
                "speed": float(row["r_speed"]),
                "best":  row["best_field"],
            }
    return d


def main():
    feats, prec, rec, f1, jacc = load_feature_csv()
    corr = load_corr_csv()

    order = np.argsort(f1)[::-1]   # descending
    feats_s = feats[order]
    f1_s    = f1[order]
    prec_s  = prec[order]
    rec_s   = rec[order]

    # -------------------------------------------------------------- Overview
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(np.arange(len(f1_s)), f1_s, color="tab:blue", lw=1.2,
            label="per-feature F1 (sorted)")
    for label, val in PAPER.items():
        color = {"Random": "gray",
                 "PCA": "tab:purple",
                 "Embedding-norm": "tab:orange"}.get(label, "tab:green")
        ls = "-" if label.startswith("SAE") else "--"
        alpha = 0.9 if label.startswith("SAE") else 0.6
        ax.axhline(val, color=color, linestyle=ls, alpha=alpha, lw=1,
                   label=f"{label} (paper Table 2 = {val:.2f})")
    ax.set_xlabel("feature rank (1024 total)")
    ax.set_ylabel("F1 vs. top-10% vorticity mask")
    ax.set_title("Per-feature vorticity alignment (η=100, K=1)\n"
                 "Single-feature saliency vs. paper's aggregated Top-K score")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.hist(f1, bins=50, color="tab:blue", edgecolor="black", alpha=0.75)
    ax.axvline(np.median(f1), color="k", linestyle="--",
               label=f"median = {np.median(f1):.3f}")
    ax.axvline(PAPER["Random"], color="gray", linestyle=":",
               label=f"Random baseline = {PAPER['Random']:.2f}")
    ax.axvline(PAPER["SAE (variance)"], color="tab:green", linestyle="--",
               label=f"Paper SAE = {PAPER['SAE (variance)']:.2f}")
    ax.set_xlabel("F1"); ax.set_ylabel("# features")
    ax.set_title("Distribution of per-feature F1")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out1 = os.path.join(PHYS_DIR, "vorticity_alignment_overview.png")
    plt.savefig(out1, dpi=150)
    plt.close(fig)
    print(f"saved {out1}")

    # -------------------------------------------------------------- Top-25
    N = 25
    top_idx = order[:N]
    top_feats = feats[top_idx]
    top_f1    = f1[top_idx]
    top_prec  = prec[top_idx]
    top_rec   = rec[top_idx]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax = axes[0]
    ypos = np.arange(N)
    ax.barh(ypos, top_f1, color="tab:blue", edgecolor="black", alpha=0.8)
    for i, (p, r) in enumerate(zip(top_prec, top_rec)):
        ax.text(top_f1[i] + 0.005, i, f"P={p:.2f}  R={r:.2f}",
                va="center", fontsize=7)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"feat {fi}" for fi in top_feats], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(PAPER["SAE (variance)"], color="tab:green", linestyle="--",
               label=f"Paper SAE F1 = {PAPER['SAE (variance)']:.2f}")
    ax.axvline(PAPER["Random"], color="gray", linestyle=":",
               label=f"Random = {PAPER['Random']:.2f}")
    ax.set_xlabel("F1 (single-feature saliency vs. top-10% |ω|)")
    ax.set_title(f"Top-{N} SAE features by vorticity alignment")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3)

    # F1 vs. physical-field correlation for the top-N features
    ax = axes[1]
    if corr:
        fields = ["u", "v", "p", "speed"]
        colors = {"u": "tab:red", "v": "tab:orange",
                  "p": "tab:purple", "speed": "tab:blue"}
        for field in fields:
            rs, fs = [], []
            for fi, fv in zip(top_feats, top_f1):
                if fi in corr:
                    rs.append(abs(corr[fi][field]))
                    fs.append(fv)
            ax.scatter(rs, fs, s=60, alpha=0.7, color=colors[field],
                       edgecolors="black", label=f"|r({field})|")
        ax.set_xlabel("|Pearson r| with physical field")
        ax.set_ylabel("F1 vs. vorticity mask")
        ax.set_title(f"Do vorticity-aligned features also correlate\n"
                     f"with simple physical fields? (top-{N} features)")
        ax.legend()
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "features_correlation.csv not found",
                ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout()
    out2 = os.path.join(PHYS_DIR, "vorticity_alignment_top.png")
    plt.savefig(out2, dpi=150)
    plt.close(fig)
    print(f"saved {out2}")

    # ----------------------------------------------------------- Summary stats
    print("\n=== Per-feature vorticity-alignment stats ===")
    print(f"  n_features:        {len(f1)}")
    print(f"  F1 mean:           {f1.mean():.4f}")
    print(f"  F1 median:         {np.median(f1):.4f}")
    print(f"  F1 max:            {f1.max():.4f}  (feat {feats[f1.argmax()]})")
    print(f"  F1 > 0.30:         {(f1 > 0.30).sum()} features")
    print(f"  F1 > paper SAE:    {(f1 > PAPER['SAE (variance)']).sum()} features")
    print(f"\n  Paper Table 2 uses aggregated saliency over K=50 features,")
    print(f"  so single-feature F1 is expected to be much lower.")
    print(f"  Relevant comparison: top-50 features should be enriched")
    print(f"  for vorticity — and they are: top-50 F1 mean = "
          f"{f1_s[:50].mean():.4f} vs all-features mean = {f1.mean():.4f}.")


if __name__ == "__main__":
    main()