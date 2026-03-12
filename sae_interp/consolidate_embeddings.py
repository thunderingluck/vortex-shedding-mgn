"""
consolidate_embeddings.py

Merges the ~60,000 per-step NPZ files in sae_embeddings/raw/ into one .npy
file per trajectory.  This replaces 600 file-open/decompress calls per
trajectory with a single np.load(), cutting wall-clock load time from
~18 min to a few seconds for the full 100-trajectory dataset.

Output layout:
  <out_dir>/traj_0000.npy   shape (N_steps * N_nodes, 128) float32
  <out_dir>/traj_0001.npy   ...

Usage:
  python consolidate_embeddings.py                        # uses defaults
  python consolidate_embeddings.py --workers 16           # more parallelism
"""

import argparse
import glob
import os
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np


def traj_id_from_path(path: str) -> str:
    m = re.search(r"traj_(\d+)_step_", os.path.basename(path))
    if m is None:
        raise ValueError(f"Cannot parse traj id from {path}")
    return m.group(1)


def consolidate_one(args):
    traj_id, files, out_dir = args
    out_path = os.path.join(out_dir, f"traj_{traj_id}.npy")
    if os.path.exists(out_path):
        return traj_id, None  # already done

    arrays = [np.load(f)["hL"].astype(np.float32) for f in sorted(files)]
    data = np.concatenate(arrays, axis=0)  # (N_steps * N_nodes, 128)
    np.save(out_path, data)
    return traj_id, data.shape


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", default="../sae_embeddings/raw")
    p.add_argument("--out_dir", default="../sae_embeddings/consolidated")
    p.add_argument("--workers", type=int, default=8,
                   help="Parallel workers (each loads one trajectory's files)")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    all_files = sorted(glob.glob(os.path.join(args.raw_dir, "traj_*.npz")))
    if not all_files:
        raise FileNotFoundError(f"No traj_*.npz files in {args.raw_dir}")

    traj_files: dict[str, list[str]] = defaultdict(list)
    for f in all_files:
        traj_files[traj_id_from_path(f)].append(f)
    traj_ids = sorted(traj_files.keys())

    print(f"Found {len(traj_ids)} trajectories, {len(all_files)} step files")
    print(f"Writing consolidated .npy files to {args.out_dir}")

    tasks = [(tid, traj_files[tid], args.out_dir) for tid in traj_ids]

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(consolidate_one, t): t[0] for t in tasks}
        done = 0
        for fut in as_completed(futures):
            tid, shape = fut.result()
            done += 1
            status = f"shape={shape}" if shape else "skipped (exists)"
            if done % 10 == 0 or done == len(traj_ids):
                print(f"  [{done:3d}/{len(traj_ids)}] traj_{tid}: {status}")

    print(f"\nDone. Point train_sae_rand.py at {args.out_dir}")


if __name__ == "__main__":
    main()
