#!/bin/bash
#SBATCH --job-name=analyze_features
#SBATCH --output=figures/phys_analysis/output.log
#SBATCH --error=figures/phys_analysis/error.log
#SBATCH --time=04:00:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4

cd /orcd/home/002/evag/code/physicsnemo/examples/cfd/vortex_shedding_mgn/sae_interp
source ~/miniconda3/bin/activate
conda activate nemo311

mkdir -p figures/phys_analysis

python analyze_features.py \
    --ckpt checkpoints_rand_3e-4_long/sae_best.pt \
    --phys_dir ../sae_embeddings/phys \
    --out_dir ./figures/phys_analysis \
    --all_traj
