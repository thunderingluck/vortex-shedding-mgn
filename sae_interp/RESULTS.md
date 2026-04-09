# SAE Interpretability Results

Sparse autoencoder trained on frozen MGN node embeddings (d=128, expansion=8, d_hid=1024).
Replicating [Interpreting CFD Surrogates through Sparse Autoencoders](https://arxiv.org/abs/2507.16069).

---

## Hyperparameter Sweep

All runs use: Adam lr=1e-4, batch_size=128 (node-level), 80/20 trajectory-level train/val split, seed=42.

| Run | λ | Best step | val_loss | val_MSE | L0 | Dead% | Notes |
|-----|---|-----------|----------|---------|-----|-------|-------|
| `checkpoints_rand_1e-2` | 1e-2 | 400k | 7.06e-04 | 1.08e-04 | 386.9 | 0.7% | L1 too strong; reconstruction collapses |
| `checkpoints_rand_1e-3` | 1e-3 | 420k | 7.27e-05 | 6.05e-06 | 276.0 | 33.1% | Good sparsity but likely undertrained |
| `checkpoints_rand_3e-4` | 3e-4 | 630k | 9.41e-05 | 6.09e-05 | 660.1 | 2.0% | High L0; needs more training |
| `checkpoints_rand_3e-4_long` | 3e-4 | 1,890k | **2.12e-05** | **1.14e-06** | **163.9** | **48.7%** | **Best overall** |

L0 is mean active features per node. Best checkpoint = lowest val_loss.

## Best Checkpoint

**`checkpoints_rand_3e-4_long/sae_best.pt`** — step 1,890,000 (epoch 3)

- val_loss = 2.12e-05
- val_MSE  = 1.14e-06
- val_L0   = 163.9 / 1024 features (~16% active per node)
- Dead features = 48.7% (~499 of 1024 never fire on val set)

The `3e-4_long` run trained ~3× longer than the short `3e-4` run. Extended training drove L0 from ~660 down to ~164 without increasing λ, while also achieving the best reconstruction by a large margin. L0 was still slowly declining at the end of training (161–163 over the last few evals).

The high dead-feature fraction (~49%) suggests wasted dictionary capacity. Options to address: continue training, warm-start with higher λ, or switch to a Top-K SAE architecture.

---

## Feature Analysis

### Physical Correlations (`figures/phys_analysis/`)

Top SAE features correlated with physical fields (Pearson r, per-node activations vs. CFD outputs):

| Feature | Best field | r |
|---------|-----------|---|
| 20 | pressure (p) | 0.773 |
| 50 | pressure (p) | 0.773 |
| 339 | speed | 0.747 |
| 665 | speed | 0.723 |
| 954 | speed | 0.640 |
| 63 | speed | 0.635 |
| 690 | speed | 0.635 |
| 750 | speed | 0.609 |
| 813 | pressure (p) | 0.561 |
| 854 | v-velocity | 0.510 |

Most high-correlation features track **speed** (derived from u, v). A smaller set tracks **pressure**. Few track v-velocity directly; none strongly track u-velocity alone.

### Spatial/Temporal Structure (`figures/phys_analysis/`, `figures/multi_traj/`)

- `top_features_spatial.png` — spatial activation maps for top features
- `top_features_temporal.png` — activation over time for top features
- `top_features_scatter.png` — feature correlation scatter plots
- `figures/multi_traj/` — per-feature activation across multiple trajectories (fig3-style)

### Global Dimension Analysis (`figures/global_dims/`)

Top globally-active dimensions (ranked by mean activation across all nodes/timesteps):
dim_0102, dim_0120, dim_0239, dim_0359, dim_0650.

See `global_dim_analysis.py` and `global_dim_velx.py` for analysis scripts.

---

## Key Figures

| File | Description |
|------|-------------|
| `figures/fig2_aggregated.png` | Aggregated feature activations (fig 2 style) |
| `figures/fig3_individual_dims.png` | Per-dimension activation plots (fig 3 style) |
| `figures/global_dims/dim_XXXX.png` | Spatial maps for globally top-ranked dims |
| `figures/multi_traj/fig3_dim_XXXX.png` | Multi-trajectory activation for key dims |
| `checkpoints_rand_3e-4_long/training_curves.png` | Loss, MSE, L0, dead-feature curves |

---

## Open Questions / Next Steps

- **Lower L0**: L0 still declining at end of training. Options: (1) continue `3e-4_long`, (2) warm-start best checkpoint with higher λ (e.g. 5e-4), (3) Top-K SAE for direct sparsity control.
- **Dead features**: ~49% dead is high. Top-K SAE avoids this by construction.
- **Feature labelling**: features 20 and 50 are strong pressure correlates — worth visualizing their spatial structure in detail. Speed-correlated features (339, 665) likely encode boundary layer / wake dynamics.
- **Extend to more physical fields**: currently correlating against u, v, p, speed. Could add vorticity, divergence, distance-to-wall.
