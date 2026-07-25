#!/bin/bash
#SBATCH --job-name=sae_topk_sweep
#SBATCH --output=checkpoints_topk_K%a/output.log
#SBATCH --error=checkpoints_topk_K%a/error.log
#SBATCH --array=16,32,48,64
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=4

# SLURM_ARRAY_TASK_ID is the K value directly (16, 32, 48, or 64)
K=$SLURM_ARRAY_TASK_ID
CKPT_DIR=checkpoints_topk_K${K}

mkdir -p ${CKPT_DIR}

cd /home/evag/code/physicsnemo/examples/cfd/vortex_shedding_mgn/sae_interp
source ~/miniconda3/bin/activate
conda activate nemo311

# Auto-resume from latest checkpoint if one exists
RESUME=""
if [ -f "${CKPT_DIR}/sae_latest.pt" ]; then
    RESUME="--resume_ckpt ${CKPT_DIR}/sae_latest.pt"
    echo "Resuming from ${CKPT_DIR}/sae_latest.pt"
else
    echo "Starting fresh training with K=${K}"
fi

python -u train_sae_topk.py \
    --k          ${K} \
    --emb_dir    ../sae_embeddings/consolidated \
    --ckpt_dir   ${CKPT_DIR} \
    --val_every  10000 \
    --patience   8 \
    ${RESUME}
