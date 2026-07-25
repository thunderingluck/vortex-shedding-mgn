#!/bin/bash
# Removes corrupted traj_0007 npz files and regenerates them.
# Trajectories 0-6 are skipped via --resume (their last-step file exists).

PHYS_DIR="$(dirname "$0")/../sae_embeddings/phys"

echo "Removing corrupted traj_0007 files from $PHYS_DIR ..."
rm -f "$PHYS_DIR"/traj_0007_step_*.npz

echo "Re-extracting traj 0007 ..."
cd "$(dirname "$0")/.."
python sae_interp/extract_phys_embeddings.py \
    --num_traj 8 \
    --resume
