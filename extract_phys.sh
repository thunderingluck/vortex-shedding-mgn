#!/bin/bash
#SBATCH --job-name=extract_phys
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/extract_phys_%j.out
#SBATCH --error=logs/extract_phys_%j.err

mkdir -p logs

source /home/evag/miniconda3/etc/profile.d/conda.sh
conda activate nemo311

cd /orcd/home/002/evag/code/physicsnemo/examples/cfd/vortex_shedding_mgn

python sae_interp/extract_phys_embeddings.py \
    --num_traj  100 \
    --num_steps 600 \
    --resume
