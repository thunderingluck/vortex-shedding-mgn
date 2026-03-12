"""
Inspect .npz embedding files in a directory.

Usage:
    python inspect_embeddings.py [dir]          # default: sae_embeddings/train
    python inspect_embeddings.py sae_embeddings/val --n 5
"""
import argparse
import os
import numpy as np


def inspect_npz(path: str) -> dict:
    d = np.load(path, allow_pickle=True)
    info = {}
    for k in d.files:
        v = d[k]
        if v.ndim == 0:
            info[k] = {"value": v.item(), "dtype": str(v.dtype)}
        else:
            info[k] = {"shape": v.shape, "dtype": str(v.dtype)}
    return info


def print_file_summary(path: str, info: dict):
    fname = os.path.basename(path)
    print(f"\n  {fname}")
    for k, meta in info.items():
        if "shape" in meta:
            print(f"    {k:20s}  shape={meta['shape']}  dtype={meta['dtype']}")
        else:
            print(f"    {k:20s}  value={meta['value']!r}  dtype={meta['dtype']}")


def main():
    parser = argparse.ArgumentParser(description="Inspect SAE embedding .npz files.")
    parser.add_argument("dir", nargs="?", default="sae_embeddings/train",
                        help="Directory containing .npz files")
    parser.add_argument("--n", type=int, default=3,
                        help="Number of files to inspect in detail (default: 3)")
    args = parser.parse_args()

    emb_dir = args.dir
    if not os.path.isdir(emb_dir):
        raise FileNotFoundError(f"Directory not found: {emb_dir}")

    files = sorted(f for f in os.listdir(emb_dir) if f.endswith(".npz"))
    if not files:
        print(f"No .npz files found in {emb_dir}")
        return

    print(f"Directory : {os.path.abspath(emb_dir)}")
    print(f"Total .npz: {len(files)}")

    # Detailed view of first --n files
    sample = files[: args.n]
    print(f"\n--- Sample ({len(sample)} of {len(files)}) ---")
    for fname in sample:
        info = inspect_npz(os.path.join(emb_dir, fname))
        print_file_summary(os.path.join(emb_dir, fname), info)

    # Aggregate stats over all files for hL
    print("\n--- Aggregate stats (hL across all files) ---")
    num_nodes_list = []
    hL_means, hL_stds = [], []
    for fname in files:
        d = np.load(os.path.join(emb_dir, fname), allow_pickle=True)
        hL = d["hL"]
        num_nodes_list.append(hL.shape[0])
        hL_means.append(hL.mean())
        hL_stds.append(hL.std())

    hL_dim = np.load(os.path.join(emb_dir, files[0]), allow_pickle=True)["hL"].shape[1]
    print(f"  hL dim         : {hL_dim}")
    print(f"  num_nodes min  : {min(num_nodes_list)}")
    print(f"  num_nodes max  : {max(num_nodes_list)}")
    print(f"  num_nodes mean : {np.mean(num_nodes_list):.1f}")
    print(f"  hL mean (avg)  : {np.mean(hL_means):.4f}")
    print(f"  hL std  (avg)  : {np.mean(hL_stds):.4f}")


if __name__ == "__main__":
    main()
