#!/bin/bash
#SBATCH --job-name=train_sae_1e-4_resume
#SBATCH --output=checkpoints_rand_1e-4_resume/output.log
#SBATCH --error=checkpoints_rand_1e-4_resume/error.log
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=4

cd /home/evag/code/physicsnemo/examples/cfd/vortex_shedding_mgn/sae_interp
source ~/miniconda3/bin/activate
conda activate nemo311

mkdir -p checkpoints_rand_1e-4_resume

python -u train_sae_rand.py \
    --emb_dir ../sae_embeddings/consolidated \
    --ckpt_dir checkpoints_rand_1e-4_resume \
    --resume_ckpt checkpoints_rand_3e-4_long/sae_latest.pt \
    --lam 1e-4 \
    --val_every 10000 \
    --l0_patience 10 \
    --l0_tol 0.1
