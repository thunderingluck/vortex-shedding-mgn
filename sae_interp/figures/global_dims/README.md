# figures/global_dims

Output of `global_dim_analysis.py`. Identifies the most globally prominent SAE latent
dimensions across all 100 trajectories and measures how consistently they rank across
trajectories.

## How dimensions are selected

Each trajectory's node embeddings are passed through the SAE encoder. Per-dimension
importance is scored by **mean absolute activation** (`mean_abs`) across all nodes and
timesteps. Dimensions are ranked globally (averaged over all trajectories) and the top 5
are selected.

## Files

| File | Description |
|------|-------------|
| `global_scores.npy` | Shape `(1024,)` — mean-abs activation score for each SAE dimension, averaged over all 100 trajectories |
| `global_top_dims.npy` | Indices of the top-5 globally prominent dimensions |
| `rank_matrix.npy` | Shape `(100, 5)` — rank of each top dimension within each trajectory (rank 1 = most active) |
| `rank_summary.txt` | Per-dimension mean and std of rank across trajectories |
| `rank_summary.png` | Bar chart of rank stability for the top 5 dimensions |
| `spearman.txt` | Average pairwise Spearman r across all trajectory pairs (measures global ranking consistency) |
| `dim_XXXX.png` | Spatial activation map for dimension XXXX across 4 representative trajectories × 4 timesteps (t=10, 50, 200, 500) |

## Key results

Top 5 globally prominent dimensions (by mean_abs across 100 trajectories):

| Rank | Dim | Mean rank across trajs | Rank std |
|------|-----|----------------------|----------|
| 1 | 359 | 5.4 | 3.2 |
| 2 | 650 | 7.8 | 9.3 |
| 3 | 102 | 8.6 | 9.5 |
| 4 | 239 | 39.6 | 102.6 |
| 5 | 120 | 71.2 | 161.0 |

Average pairwise Spearman r = **0.881** across all trajectory pairs — the global
dimension ranking is highly consistent across different flow geometries.

Dims 359, 650, and 102 are robustly top-ranked (low rank std). Dims 239 and 120 are
globally prominent on average but show high variance — they dominate some trajectories
and are unimportant in others.

## How to reproduce

```bash
cd sae_interp/
python global_dim_analysis.py \
    --ckpt   checkpoints_rand_3e-4_long/sae_best.pt \
    --emb_dir ../sae_embeddings/raw \
    --out_dir ./figures/global_dims \
    --metric  mean_abs \
    --n_dims  5 \
    --n_trajs 4 \
    --t_steps 10,50,200,500
```
