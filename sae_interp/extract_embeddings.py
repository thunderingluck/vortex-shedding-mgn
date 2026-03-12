import json
import os
import numpy as np
import torch
from hydra.utils import to_absolute_path
from tfrecord.torch.dataset import TFRecordDataset

from physicsnemo.models.meshgraphnet import MeshGraphNet
from physicsnemo.datapipes.gnn.vortex_shedding_dataset import VortexSheddingDataset
from physicsnemo.utils import load_checkpoint


@torch.no_grad()
def get_hL_embeddings(model: MeshGraphNet, node_x, edge_attr, graph):
    """
    Returns processor output h^L (node embeddings) before the node decoder.
    """
    edge_e = model.edge_encoder(edge_attr)
    node_e = model.node_encoder(node_x)
    hL = model.processor(node_e, edge_e, graph)  # (num_nodes, d_in)
    return hL


def _load_stats(stats_dir: str) -> tuple[dict, dict]:
    """Load pre-computed edge and node normalisation stats from JSON."""
    def _load(path):
        with open(path) as f:
            raw = json.load(f)
        return {k: torch.tensor(v) for k, v in raw.items()}

    edge_stats = _load(os.path.join(stats_dir, "edge_stats.json"))
    node_stats = _load(os.path.join(stats_dir, "node_stats.json"))
    return edge_stats, node_stats


def _extract_split(cfg, model, device, split: str, out_root: str, stats_dir: str):
    """Stream one trajectory at a time to avoid loading the full dataset into RAM."""
    out_dir = os.path.join(out_root, "raw")
    os.makedirs(out_dir, exist_ok=True)

    data_dir = to_absolute_path(cfg.data_dir)
    num_steps = cfg.sae.num_steps   # timesteps to use per trajectory
    num_samples = cfg.sae.num_samples

    edge_stats, node_stats = _load_stats(stats_dir)

    # Load tfrecord meta and build decoder
    with open(os.path.join(data_dir, "meta.json")) as f:
        meta = json.load(f)
    description = {k: "byte" for k in meta["field_names"]}
    tfrecord_path = os.path.join(data_dir, f"{split}.tfrecord")
    index_path = tfrecord_path.replace(".tfrecord", ".tfindex")
    if not os.path.exists(index_path):
        index_path = None

    tfr_dataset = TFRecordDataset(
        tfrecord_path,
        index_path,
        description,
        transform=lambda rec: VortexSheddingDataset._decode_record(rec, meta),
    )

    snapshot_meta = []
    dataset_idx = 0

    for traj_id, data_np in enumerate(tfr_dataset):
        if traj_id >= num_samples:
            break

        # Slice to requested num_steps
        data_np = {k: v[:num_steps] for k, v in data_np.items()}

        # Build graph (static per trajectory)
        src, dst = VortexSheddingDataset.cell_to_adj(data_np["cells"][0])
        graph = VortexSheddingDataset.create_graph(src, dst, dtype=torch.int32)
        graph = VortexSheddingDataset.add_edge_features(graph, data_np["mesh_pos"][0])
        graph.edge_attr = VortexSheddingDataset.normalize_edge(
            graph, edge_stats["edge_mean"], edge_stats["edge_std"]
        )

        node_type = torch.tensor(data_np["node_type"][0], dtype=torch.uint8)
        node_type_oh = VortexSheddingDataset._one_hot_encode(node_type).float()

        # Per-trajectory static arrays (saved once per snapshot for self-contained files)
        mesh_pos_np = data_np["mesh_pos"][0].astype(np.float32)
        cells_np = data_np["cells"][0]
        rollout_mask_np = VortexSheddingDataset._get_rollout_mask(node_type).numpy()

        velocity = torch.tensor(data_np["velocity"], dtype=torch.float32)  # (T, N, 2)
        vel_norm = VortexSheddingDataset.normalize_node(
            velocity, node_stats["velocity_mean"], node_stats["velocity_std"]
        )

        graph = graph.to(device)
        node_type_oh = node_type_oh.to(device)

        # Iterate over timesteps (drop last, same as dataset)
        for step_id in range(num_steps - 1):
            node_x = torch.cat([vel_norm[step_id].to(device), node_type_oh], dim=-1)
            graph.x = node_x

            hL = get_hL_embeddings(model, graph.x, graph.edge_attr, graph)
            hL = hL.detach().cpu().numpy().astype(np.float32)

            fname = f"traj_{traj_id:04d}_step_{step_id:04d}.npz"
            np.savez_compressed(
                os.path.join(out_dir, fname),
                hL=hL,
                mesh_pos=mesh_pos_np,
                cells=cells_np,
                rollout_mask=rollout_mask_np,
                split=split,
                dataset_idx=dataset_idx,
                trajectory_id=traj_id,
                step_id=step_id,
                num_nodes=hL.shape[0],
            )

            snapshot_meta.append({
                "dataset_idx": dataset_idx,
                "file": fname,
                "split": split,
                "trajectory_id": traj_id,
                "step_id": step_id,
                "num_nodes": int(hL.shape[0]),
            })
            dataset_idx += 1

        print(f"[SAE] traj {traj_id:04d} done ({num_steps - 1} snapshots)")

    np.save(os.path.join(out_dir, "index.npy"), np.array(snapshot_meta, dtype=object))
    print(f"[SAE] Saved {len(snapshot_meta)} snapshots for split='{split}' to {out_dir}")


def extract_and_save(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_root = to_absolute_path(cfg.sae.out_root)
    os.makedirs(out_root, exist_ok=True)

    # Stats JSONs are written to the Hydra output dir (cwd after chdir)
    stats_dir = os.getcwd()

    model = MeshGraphNet(
        cfg.num_input_features,
        cfg.num_edge_features,
        cfg.num_output_features,
        mlp_activation_fn="silu" if cfg.recompute_activation else "relu",
        do_concat_trick=cfg.do_concat_trick,
        num_processor_checkpoint_segments=cfg.num_processor_checkpoint_segments,
        recompute_activation=cfg.recompute_activation,
    ).to(device)
    model.eval()
    load_checkpoint(to_absolute_path(cfg.ckpt_path), models=model, device=device)

    splits = cfg.sae.splits
    if isinstance(splits, str):
        splits = [splits]

    for split in splits:
        _extract_split(cfg, model, device, split, out_root, stats_dir)

    print(f"[SAE] Finished extracting splits={splits} into {out_root}")


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    _CONF = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "conf"))

    @hydra.main(version_base="1.3", config_path=_CONF, config_name="config")
    def main(cfg: DictConfig):
        extract_and_save(cfg)

    main()