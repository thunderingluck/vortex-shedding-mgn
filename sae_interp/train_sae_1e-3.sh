#!/bin/bash
#SBATCH --job-name=train_sae_1e-3
#SBATCH --output=checkpoints_rand_1e-3/output.log_1e-3
#SBATCH --error=checkpoints_rand_1e-3/error.log_1e-3
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=4

cd /home/evag/code/physicsnemo/examples/cfd/vortex_shedding_mgn/sae_interp
source ~/miniconda3/bin/activate
conda activate nemo311

python -u train_sae_rand.py --emb_dir ../sae_embeddings/consolidated --lam 1e-3 --val_every 10000 --ckpt_dir checkpoints_rand_1e-3 --l0_patience 10 --l0_tol 0.1
