#!/bin/bash
#SBATCH --job-name=vorticity_alignment
#SBATCH --output=figures/phys_analysis/vorticity_alignment.log
#SBATCH --error=figures/phys_analysis/vorticity_alignment.err
#SBATCH --time=06:00:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4

cd /orcd/home/002/evag/code/physicsnemo/examples/cfd/vortex_shedding_mgn/sae_interp
source ~/miniconda3/bin/activate
conda activate nemo311

mkdir -p figures/phys_analysis

python vorticity_alignment.py \
    --phys_dir ../sae_embeddings/phys \
    --out_dir  ./figures/phys_analysis \
    --all_traj \
    --K 50 --eta 100 --vortex_pct 0.10
