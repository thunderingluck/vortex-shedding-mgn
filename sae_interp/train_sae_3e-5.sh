#!/bin/bash
#SBATCH --job-name=train_sae_3e-5
#SBATCH --output=output_lam3e-5.log
#SBATCH --error=error_lam3e-5.log
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=4

cd /home/evag/code/physicsnemo/examples/cfd/vortex_shedding_mgn/sae_interp
source ~/miniconda3/bin/activate
conda activate nemo311

python -u train_sae_rand.py --emb_dir ../sae_embeddings/consolidated --lam 1e-2 --val_every 10000 --ckpt_dir checkpoints_rand_3e-5

