# figures/phys_analysis

Output of `analyze_features.py`. Correlates SAE feature activations with physical fields
(u, v, p, speed) and produces spatial and temporal visualisation of the top features.

## Method

For a single trajectory, each node's SAE activation vector z is computed at every
timestep. Pearson r is computed between each of the 1024 feature activations and each
physical field (u, v, p, speed = sqrt(u²+v²)) across all nodes × timesteps.
`r_abs_max` is the maximum absolute correlation across the four fields.

## Files

| File | Description |
|------|-------------|
| `features_correlation.csv` | Per-feature correlations: columns `feature, r_u, r_v, r_p, r_speed, r_abs_max, best_field` |
| `top_features_spatial.png` | Spatial activation maps (on the mesh) for the top-N features by `r_abs_max` |
| `top_features_temporal.png` | Mean activation over time (averaged across all nodes) for the top-N features |
| `top_features_scatter.png` | Scatter of feature activation vs best-correlated physical field for top-N features |

## Key results (all 1024 features, traj 0000)

| r_abs_max threshold | Feature count |
|--------------------|--------------|
| > 0.7 | 4 |
| > 0.5 | 19 |
| > 0.3 | 136 |
| > 0.1 | 589 |

Most high-correlation features track **speed** (magnitude). A smaller subset tracks
**pressure**. Few features correlate strongly with u or v individually.

Top features by `r_abs_max`:

| Feature | Best field | r |
|---------|-----------|---|
| 20 | pressure | 0.773 |
| 50 | pressure | 0.773 |
| 339 | speed | 0.747 |
| 665 | speed | 0.723 |
| 954 | speed | 0.640 |

Note: the ~435 features with r_abs_max < 0.1 are likely dead or near-dead on this
trajectory. The L0 analysis suggests ~49% of features are globally dead.

## How to reproduce

```bash
cd sae_interp/
python analyze_features.py \
    --ckpt      checkpoints_rand_3e-4_long/sae_best.pt \
    --phys_dir  ../sae_embeddings/phys \
    --out_dir   ./figures/phys_analysis \
    --traj_id   0 \
    --topn      12
```
