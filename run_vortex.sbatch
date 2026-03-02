#!/bin/bash
#SBATCH -J vortex_mgn
#SBATCH -p mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH -t 06:00:00
#SBATCH -o vortex_%j.out
#SBATCH -e vortex_%j.err

set -euo pipefail

echo "JobID: ${SLURM_JOB_ID:-N/A}"
echo "Node:  $(hostname)"
date
nvidia-smi || true

source ~/miniconda3/etc/profile.d/conda.sh
conda activate nemo311

cd /orcd/home/002/evag/code/physicsnemo/examples/cfd/vortex_shedding_mgn

export HYDRA_FULL_ERROR=1

# Launch under SLURM so it sets the right env vars
srun --ntasks=1 --gpus-per-task=1 python -u train.py
