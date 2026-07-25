# figures/multi_traj

Output of `fig3_multi_traj.py`. Produces paper-style figures (analogous to Fig. 3 in
the replication target) showing how the most salient SAE dimensions activate across
multiple randomly-sampled trajectories alongside the ground-truth velocity_x field.

## Layout

Each `fig3_dim_XXXX.png` shows a grid:
- **Rows**: alternating pairs — SAE activation for dim XXXX, then ground-truth vel_x —
  one pair per trajectory
- **Columns**: 4 evenly-spaced timesteps

Each `fig3_traj_XXXX_activation.png` shows all salient dimensions for a single
trajectory across time.

## Files

| Pattern | Description |
|---------|-------------|
| `fig3_dim_XXXX.png` | Multi-trajectory activation map for dimension XXXX (activation + vel_x side-by-side) |
| `fig3_dim_XXXX_activation.png` | Activation-only panel for dimension XXXX across selected trajectories |
| `fig3_dim_XXXX_velx.png` | Ground-truth vel_x panel matching the activation layout |
| `fig3_traj_XXXX_activation.png` | All salient dimensions for trajectory XXXX across time |
| `fig3_traj_XXXX.png` | Full multi-panel figure for trajectory XXXX |

## Dimensions shown

Dims 102, 120, 239, 359, 650 — the top-5 globally prominent dimensions from
`global_dim_analysis.py`. Trajectories were randomly sampled (seed=42): 0000, 0003,
0004, 0005, 0007, 0008, 0043, 0064, 0075, 0099.

## How to reproduce

```bash
cd sae_interp/
python fig3_multi_traj.py \
    --ckpt      checkpoints_rand_3e-4_long/sae_best.pt \
    --emb_dir   ../sae_embeddings/raw \
    --phys_dir  ../sae_embeddings/phys \
    --out_dir   ./figures/multi_traj \
    --n_trajs   5 \
    --n_dims    5 \
    --metric    mean_abs \
    --seed      42
```
