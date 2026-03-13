#!/bin/bash
#SBATCH --job-name=train_sae
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

cd /home/evag/code/physicsnemo/examples/cfd/vortex_shedding_mgn/sae_interp
source ~/miniconda3/bin/activate
conda activate nemo311

python train_sae_rand.py --emb_dir ../sae_embeddings/consolidated --lam 1e-2