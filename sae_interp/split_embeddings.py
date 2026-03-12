"""
Split extracted test embeddings into SAE train/val partitions.

Splitting is done at the trajectory level (not snapshot level) to prevent
data leakage. All snapshots belonging to a trajectory go to the same partition.

Reads from:  <out_root>/test/emb_*.npz
Writes to:   <out_root>/train/emb_*.npz  (1 - val_fraction of trajectories)
             <out_root>/val/emb_*.npz    (val_fraction of trajectories)
"""
import os
import shutil
import numpy as np
from hydra.utils import to_absolute_path


def split_embeddings(cfg):
    out_root = to_absolute_path(cfg.sae.out_root)
    src_dir = os.path.join(out_root, "test")
    train_dir = os.path.join(out_root, "train")
    val_dir = os.path.join(out_root, "val")

    files = sorted(f for f in os.listdir(src_dir) if f.endswith(".npz"))
    if not files:
        raise FileNotFoundError(f"No emb_*.npz files found in {src_dir}")

    # Group files by trajectory_id
    traj_to_files: dict[int, list[str]] = {}
    for fname in files:
        d = np.load(os.path.join(src_dir, fname), allow_pickle=True)
        tid = int(d["trajectory_id"])
        traj_to_files.setdefault(tid, []).append(fname)

    traj_ids = sorted(traj_to_files.keys())
    n_traj = len(traj_ids)
    n_val = max(1, round(n_traj * cfg.sae.val_fraction))
    n_train = n_traj - n_val

    rng = np.random.default_rng(cfg.sae.split_seed)
    shuffled = rng.permutation(traj_ids).tolist()
    train_trajs = set(shuffled[:n_train])
    val_trajs = set(shuffled[n_train:])

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    train_counter, val_counter = 0, 0
    train_meta, val_meta = [], []

    for tid in sorted(traj_to_files.keys()):
        dest_dir = train_dir if tid in train_trajs else val_dir
        partition = "train" if tid in train_trajs else "val"

        for fname in sorted(traj_to_files[tid]):
            if partition == "train":
                dest_name = f"emb_{train_counter:06d}.npz"
                train_counter += 1
                train_meta.append({"file": dest_name, "src": fname, "trajectory_id": tid})
            else:
                dest_name = f"emb_{val_counter:06d}.npz"
                val_counter += 1
                val_meta.append({"file": dest_name, "src": fname, "trajectory_id": tid})

            shutil.copy2(
                os.path.join(src_dir, fname),
                os.path.join(dest_dir, dest_name),
            )

    np.save(os.path.join(train_dir, "index.npy"), np.array(train_meta, dtype=object))
    np.save(os.path.join(val_dir, "index.npy"), np.array(val_meta, dtype=object))

    print(
        f"[SAE] split_embeddings: {n_traj} trajectories → "
        f"{n_train} train ({train_counter} snapshots), "
        f"{n_val} val ({val_counter} snapshots)"
    )
    print(f"  train → {train_dir}")
    print(f"  val   → {val_dir}")
