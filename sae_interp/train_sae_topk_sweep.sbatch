#!/bin/bash
#SBATCH --job-name=sae_topk_sweep
#SBATCH --output=checkpoints_topk_K%a/output.log
#SBATCH --error=checkpoints_topk_K%a/error.log
#SBATCH --array=16,32,48,64
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=4

# SLURM_ARRAY_TASK_ID is the K value directly (16, 32, 48, or 64)
K=$SLURM_ARRAY_TASK_ID

mkdir -p checkpoints_topk_K${K}

cd /home/evag/code/physicsnemo/examples/cfd/vortex_shedding_mgn/sae_interp
source ~/miniconda3/bin/activate
conda activate nemo311

echo "Starting Top-K SAE training with K=${K}"

python train_sae_topk.py \
    --k          ${K} \
    --emb_dir    ../sae_embeddings/consolidated \
    --ckpt_dir   checkpoints_topk_K${K} \
    --val_every  10000 \
    --patience   8
