#!/bin/bash
#SBATCH --job-name=global_dim_analysis
#SBATCH --output=figures/global_dims/output.log
#SBATCH --error=figures/global_dims/error.log
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4

cd /home/evag/code/physicsnemo/examples/cfd/vortex_shedding_mgn/sae_interp
source ~/miniconda3/bin/activate
conda activate nemo311

mkdir -p figures/global_dims

python global_dim_analysis.py \
    --ckpt   checkpoints_rand_3e-4_long/sae_best.pt \
    --emb_dir ../sae_embeddings/raw \
    --out_dir ./figures/global_dims \
    --metric  mean_abs \
    --n_dims  5 \
    --n_trajs 4 \
    --t_steps 10,50,200,500

