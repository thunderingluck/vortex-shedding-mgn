"""
extract_phys_embeddings.py

Extracts MGN hidden embeddings (hL) alongside physical fields (velocity,
pressure) from test.tfrecord, saving one .npz per snapshot.

Output per file (sae_embeddings/phys/traj_XXXX_step_YYYY.npz):
    hL         (N, 128)  – MGN processor output
    velocity   (N, 2)    – raw (un-normalised) velocity [u, v]
    pressure   (N, 1)    – raw pressure
    mesh_pos   (N, 2)    – node coordinates
    cells      (C, 3)    – triangles
    node_type  (N, 1)    – node type integer

Usage (from repo root):
    python sae_interp/extract_phys_embeddings.py --num_traj 100
    python sae_interp/extract_phys_embeddings.py --num_traj 100 --resume
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

# ── resolve repo root (script lives in sae_interp/) ──────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

from tfrecord.torch.dataset import TFRecordDataset
from physicsnemo.models.meshgraphnet import MeshGraphNet
from physicsnemo.datapipes.gnn.vortex_shedding_dataset import VortexSheddingDataset

# ── default paths (all relative to REPO_ROOT) ────────────────────────────────
DEFAULT_DATA_DIR  = os.path.join(REPO_ROOT, "raw_dataset/cylinder_flow/cylinder_flow")
DEFAULT_CKPT_DIR  = os.path.join(REPO_ROOT, "checkpoints")
DEFAULT_STATS_DIR = os.path.join(REPO_ROOT, "outputs")
DEFAULT_OUT_DIR   = os.path.join(REPO_ROOT, "sae_embeddings/phys")


# ── MGN embedding helper ──────────────────────────────────────────────────────

@torch.no_grad()
def get_hL(model, node_x, edge_attr, graph):
    edge_e = model.edge_encoder(edge_attr)
    node_e = model.node_encoder(node_x)
    return model.processor(node_e, edge_e, graph)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_traj",   type=int, default=100,
                   help="Number of test trajectories to process")
    p.add_argument("--num_steps",  type=int, default=600,
                   help="Timesteps per trajectory (max 600)")
    p.add_argument("--data_dir",   default=DEFAULT_DATA_DIR)
    p.add_argument("--ckpt_dir",   default=DEFAULT_CKPT_DIR)
    p.add_argument("--stats_dir",  default=DEFAULT_STATS_DIR)
    p.add_argument("--out_dir",    default=DEFAULT_OUT_DIR)
    p.add_argument("--resume",     action="store_true",
                   help="Skip trajectories whose files are already complete")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # ── load normalisation stats ──────────────────────────────────────────────
    def load_json(path):
        with open(path) as f:
            raw = json.load(f)
        return {k: torch.tensor(v) for k, v in raw.items()}

    edge_stats = load_json(os.path.join(args.stats_dir, "edge_stats.json"))
    node_stats = load_json(os.path.join(args.stats_dir, "node_stats.json"))

    # ── load MGN (uses from_checkpoint to avoid distributed manager) ──────────
    ckpt_path = os.path.join(args.ckpt_dir, "MeshGraphNet.0.32.mdlus")
    model = MeshGraphNet.from_checkpoint(ckpt_path).to(device)
    model.eval()
    print(f"loaded MGN from {ckpt_path}")

    # ── open tfrecord ─────────────────────────────────────────────────────────
    tfrecord_path = os.path.join(args.data_dir, "test.tfrecord")
    with open(os.path.join(args.data_dir, "meta.json")) as f:
        meta = json.load(f)

    description = {k: "byte" for k in meta["field_names"]}
    dataset = TFRecordDataset(
        tfrecord_path,
        index_path=None,
        description=description,
        transform=lambda rec: VortexSheddingDataset._decode_record(rec, meta),
    )

    # ── iterate trajectories ──────────────────────────────────────────────────
    total_snapshots = 0
    for traj_id, data_np in enumerate(dataset):
        if traj_id >= args.num_traj:
            break

        T = min(args.num_steps, data_np["velocity"].shape[0])
        expected_steps = T - 1

        # resume: skip trajectories where all steps already exist
        if args.resume:
            last_fname = os.path.join(
                args.out_dir, f"traj_{traj_id:04d}_step_{expected_steps - 1:04d}.npz"
            )
            if os.path.exists(last_fname):
                print(f"  traj {traj_id:04d}: already complete, skipping")
                total_snapshots += expected_steps
                continue

        data_np = {k: v[:T] for k, v in data_np.items()}

        # build static graph
        src, dst = VortexSheddingDataset.cell_to_adj(data_np["cells"][0])
        graph = VortexSheddingDataset.create_graph(src, dst, dtype=torch.int32)
        graph = VortexSheddingDataset.add_edge_features(graph, data_np["mesh_pos"][0])
        graph.edge_attr = VortexSheddingDataset.normalize_edge(
            graph, edge_stats["edge_mean"], edge_stats["edge_std"]
        )

        node_type    = torch.tensor(data_np["node_type"][0], dtype=torch.uint8)
        node_type_oh = VortexSheddingDataset._one_hot_encode(node_type).float()

        velocity = torch.tensor(data_np["velocity"], dtype=torch.float32)  # (T, N, 2)
        vel_norm = VortexSheddingDataset.normalize_node(
            velocity, node_stats["velocity_mean"], node_stats["velocity_std"]
        )

        mesh_pos_np  = data_np["mesh_pos"][0].astype(np.float32)
        cells_np     = data_np["cells"][0]
        node_type_np = data_np["node_type"][0].astype(np.int32)

        # pressure: (T, N, 1)
        pressure_np = data_np["pressure"].astype(np.float32)

        graph        = graph.to(device)
        node_type_oh = node_type_oh.to(device)

        for step_id in range(T - 1):
            fname = f"traj_{traj_id:04d}_step_{step_id:04d}.npz"
            fpath = os.path.join(args.out_dir, fname)

            # resume: skip individual steps that are already written
            if args.resume and os.path.exists(fpath):
                total_snapshots += 1
                continue

            node_x    = torch.cat([vel_norm[step_id].to(device), node_type_oh], dim=-1)
            graph.x   = node_x
            hL        = get_hL(model, graph.x, graph.edge_attr, graph)
            hL_np     = hL.cpu().numpy().astype(np.float32)

            np.savez_compressed(
                fpath,
                hL        = hL_np,
                velocity  = velocity[step_id].numpy().astype(np.float32),  # (N, 2)
                pressure  = pressure_np[step_id],                           # (N, 1)
                mesh_pos  = mesh_pos_np,
                cells     = cells_np,
                node_type = node_type_np,
                trajectory_id = traj_id,
                step_id       = step_id,
            )
            total_snapshots += 1

        print(f"  traj {traj_id:04d}: {T-1} snapshots written")

    print(f"\ndone — {total_snapshots} snapshots saved to {args.out_dir}")


if __name__ == "__main__":
    main()
