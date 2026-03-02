#!/bin/bash
#SBATCH -J vortex_mgn_4gpu
#SBATCH -p mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o vortex_%j.out
#SBATCH -e vortex_%j.err

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nemo311

cd /orcd/home/002/evag/code/physicsnemo/examples/cfd/vortex_shedding_mgn
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# W&B key should be exported before sbatch
export WANDB_SILENT=true

CKPT="checkpoints/checkpoint.0.12.pt"

srun --ntasks=$SLURM_NTASKS python -u train.py \
  wandb_mode=online resume=true checkpoint_path="$CKPT"
