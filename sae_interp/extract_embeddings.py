import os
import numpy as np
import torch
from torch_geometric.loader import DataLoader as PyGDataLoader
from hydra.utils import to_absolute_path

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


def _unpack_batch(batch, split: str):
    """
    Handle possible dataset return-format differences across splits.
    Expected:
      - non-train: (graph, cells, rollout_mask)
      - train: maybe just graph, or maybe also tuple-like
    """
    if isinstance(batch, (list, tuple)):
        if len(batch) == 3:
            graph, cells, rollout_mask = batch
        elif len(batch) == 1:
            graph = batch[0]
            cells, rollout_mask = None, None
        else:
            raise ValueError(f"Unexpected batch format for split={split}: len={len(batch)}")
    else:
        graph = batch
        cells, rollout_mask = None, None

    return graph, cells, rollout_mask


def _extract_split(cfg, model, device, split: str, out_root: str):
    out_dir = os.path.join(out_root, split)
    os.makedirs(out_dir, exist_ok=True)

    dataset = VortexSheddingDataset(
        name=f"vortex_shedding_sae_{split}",
        data_dir=to_absolute_path(cfg.data_dir),
        split=split,
        num_samples=cfg.sae.num_samples,
        num_steps=cfg.sae.num_steps,
    )
    loader = PyGDataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)

    meta = []

    for idx, batch in enumerate(loader):
        graph, cells, rollout_mask = _unpack_batch(batch, split)
        graph = graph.to(device)

        hL = get_hL_embeddings(model, graph.x, graph.edge_attr, graph)
        hL = hL.detach().cpu().numpy().astype(np.float32)

        # mesh positions, if present
        mesh_pos = None
        if "mesh_pos" in graph:
            mesh_pos = graph["mesh_pos"].detach().cpu().numpy().astype(np.float32)

        # cells / rollout mask may not exist for train split
        cells_np = None
        if cells is not None:
            if hasattr(cells, "numpy"):
                cells_np = np.squeeze(cells.numpy())
            else:
                cells_np = np.asarray(cells)

        mask_np = None
        if rollout_mask is not None:
            if hasattr(rollout_mask, "detach"):
                mask_np = rollout_mask.detach().cpu().numpy()
            else:
                mask_np = np.asarray(rollout_mask)

        save_path = os.path.join(out_dir, f"emb_{idx:06d}.npz")
        np.savez_compressed(
            save_path,
            hL=hL,
            mesh_pos=mesh_pos,
            cells=cells_np,
            rollout_mask=mask_np,
            split=split,
            dataset_idx=idx,
            num_nodes=hL.shape[0],
        )

        meta.append({
            "dataset_idx": idx,
            "file": f"emb_{idx:06d}.npz",
            "split": split,
            "num_nodes": int(hL.shape[0]),
        })

    np.save(os.path.join(out_dir, "index.npy"), np.array(meta, dtype=object))
    print(f"[SAE] Saved {len(meta)} snapshots for split='{split}' to {out_dir}")


def extract_and_save(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_root = to_absolute_path(cfg.sae.out_root)
    os.makedirs(out_root, exist_ok=True)

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
        _extract_split(cfg, model, device, split, out_root)

    print(f"[SAE] Finished extracting splits={splits} into {out_root}")