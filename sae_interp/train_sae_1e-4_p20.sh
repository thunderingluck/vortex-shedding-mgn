#!/bin/bash
#SBATCH --job-name=train_sae_1e-4_p20
#SBATCH --output=checkpoints_rand_1e-4_p20/output.log
#SBATCH --error=checkpoints_rand_1e-4_p20/error.log
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=4

cd /home/evag/code/physicsnemo/examples/cfd/vortex_shedding_mgn/sae_interp
source ~/miniconda3/bin/activate
conda activate nemo311

mkdir -p checkpoints_rand_1e-4_p20

python -u train_sae_rand.py \
    --emb_dir ../sae_embeddings/consolidated \
    --ckpt_dir checkpoints_rand_1e-4_p20 \
    --resume_ckpt checkpoints_rand_1e-4_resume/sae_best.pt \
    --lam 1e-4 \
    --val_every 10000 \
    --patience 20
