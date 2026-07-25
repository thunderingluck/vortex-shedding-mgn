#!/bin/bash
#SBATCH --job-name=top_activating_inputs
#SBATCH --output=figures/global_dims/top_inputs_output.log
#SBATCH --error=figures/global_dims/top_inputs_error.log
#SBATCH --time=04:00:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4

cd /orcd/home/002/evag/code/physicsnemo/examples/cfd/vortex_shedding_mgn/sae_interp
source ~/miniconda3/bin/activate
conda activate nemo311

mkdir -p figures/global_dims

python top_activating_inputs.py \
    --ckpt      checkpoints_rand_3e-4_long/sae_best.pt \
    --phys_dir  ../sae_embeddings/phys \
    --dims_file ./figures/global_dims/global_top_dims.npy \
    --out_dir   ./figures/global_dims \
    --max_pts   30000 \
    --n_trajs   all
