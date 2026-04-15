# SAE Interpretability Results

Sparse autoencoder trained on frozen MGN node embeddings (d=128, expansion=8, d_hid=1024).
Replicating [Interpreting CFD Surrogates through Sparse Autoencoders](https://arxiv.org/abs/2507.16069).

---

## Hyperparameter Sweep

All runs use: Adam lr=1e-4, batch_size=128 (node-level), 80/20 trajectory-level train/val split, seed=42.
`val_MSE` is element-wise MSE on per-dim normalized embeddings (unit variance → 1.0 = "predict zero" baseline).

### L1 SAEs (ReLU + L1 penalty)

| Run | λ | Best step | val_loss | val_MSE | L0 | Dead% | Notes |
|-----|---|-----------|----------|---------|-----|-------|-------|
| `checkpoints_rand_1e-2` | 1e-2 | 400k | 7.06e-04 | 1.08e-04 | 386.9 | 0.7% | L1 too strong; reconstruction collapses |
| `checkpoints_rand_1e-3` | 1e-3 | 420k | 7.27e-05 | 6.05e-06 | 276.0 | 33.1% | Good sparsity but likely undertrained |
| `checkpoints_rand_3e-4` | 3e-4 | 630k | 9.41e-05 | 6.09e-05 | 660.1 | 2.0% | High L0; needs more training |
| `checkpoints_rand_3e-4_long` | 3e-4 | 1,890k | 2.12e-05 | 1.14e-06 | 163.9 | 48.7% | Previous best L1 |
| `checkpoints_rand_1e-4_p20` | 1e-4 | — | ~7e-6 | ~3e-6 | **151.3** | 62.3% | Warm-started from `3e-4_long`; L0 stuck at ~151 for thousands of evals |

**L1 hits a sparsity floor at L0 ≈ 150.** Pushing λ higher raises MSE and kills features without further reducing L0. Dead fraction grows monotonically with training time at low λ — wasted dictionary capacity.

### Top-K SAEs (hard top-K activation, MSE-only loss)

Plain Top-K (no auxiliary loss):

| Run | K | val_MSE | L0 | Dead% |
|-----|---|---------|-----|-------|
| `checkpoints_topk_K16` | 16 | 0.1148 | 16.0 | 50.6% |
| `checkpoints_topk_K32` | 32 | 0.0614 | 32.0 | 33.4% |
| `checkpoints_topk_K48` | 48 | 0.0355 | 48.0 | 21.4% |
| `checkpoints_topk_K64` | 64 | 0.0192 | 64.0 | 0.0% |

Top-K with auxiliary dead-feature loss (`sae_topk.loss_with_aux`, α = 1/32):

| Run | K | val_MSE | L0 | Dead% |
|-----|---|---------|-----|-------|
| `checkpoints_topk_aux_K16` | 16 | 0.0990 | 16.0 | **0.0%** |
| `checkpoints_topk_aux_K32` | 32 | 0.0578 | 32.0 | **0.0%** |
| `checkpoints_topk_aux_K48` | 48 | 0.0346 | 48.0 | **0.0%** |
| `checkpoints_topk_aux_K64` | 64 | 0.0192 | 64.0 | **0.0%** |
| `checkpoints_topk_aux_K128` | 128 | *pending* | 128.0 | — | L0-match against L1 `1e-4_p20` |
| `checkpoints_topk_aux_K150` | 150 | *pending* | 150.0 | — | L0-match against L1 `1e-4_p20` |

**The auxiliary loss eliminates dead features entirely** — 100% of 1024 dictionary atoms are used at every K. Plain Top-K leaves up to 50% of features dead at low K.

### L0-matched comparison

The L1 `1e-4_p20` run has val_MSE ≈ 3e-6 at L0 = 151, which looks far better than any Top-K run above. **This is a fair-comparison artifact, not a real win:** MSE can only be compared at matched L0. L1 cannot reach L0 = 32, so the apples-to-apples comparison requires Top-K at K ≈ 150.

The two new `checkpoints_topk_aux_K128` and `checkpoints_topk_aux_K150` runs (launched via `train_sae_topk_aux_highK.sh`) close this gap. Expected outcome: val_MSE competitive with L1 at matched L0, with 0% dead features vs. L1's 62%.

## Best Checkpoint

**`checkpoints_topk_aux_K32/sae_best.pt`** — canonical SAE for downstream interpretability.

- val_MSE ≈ 5.78e-02 (normalized; ~94% variance explained)
- L0 = 32.0 (exact, by construction)
- Dead = 0% (all 1024 atoms used)
- ~5× sparser than best L1 (L0 = 151) with full dictionary utilization

For higher-fidelity reconstruction, use K = 48 (val_MSE 3.46e-02, ~97% var) or K = 64 (val_MSE 1.92e-02, ~98% var).

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

- **L0-matched K=128/K=150 Top-K runs** launched via `train_sae_topk_aux_highK.sh`. Fill in val_MSE once complete to confirm Top-K dominates L1 across the full Pareto frontier.
- **Feature labelling on Top-K K=32**: rerun `analyze_features.py`, `top_activating_inputs.py`, `vorticity_alignment.py` against the canonical `checkpoints_topk_aux_K32/sae_best.pt`.
- **Pareto plot**: L0 vs val_MSE across all runs (L1 + Top-K + Top-K+aux) to visualize dominance.
- **Extend to more physical fields**: currently correlating against u, v, p, speed. Could add vorticity, divergence, distance-to-wall.
- **Bug**: `train_sae_3e-5.sh` passes `--lam 1e-2` (not 3e-5) — `checkpoints_rand_3e-5` is a duplicate of `checkpoints_rand_1e-2`.
