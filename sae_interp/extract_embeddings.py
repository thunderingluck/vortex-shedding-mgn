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
    This is the 'node embedding space' used in the paper for SAE training. :contentReference[oaicite:5]{index=5}
    """
    edge_e = model.edge_encoder(edge_attr)
    node_e = model.node_encoder(node_x)
    hL = model.processor(node_e, edge_e, graph)  # (num_nodes, 128)
    return hL


def extract_and_save(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = to_absolute_path(cfg.sae.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    dataset = VortexSheddingDataset(
        name="vortex_shedding_sae",
        data_dir=to_absolute_path(cfg.data_dir),
        split=cfg.sae.split,          # typically "test" like the paper :contentReference[oaicite:6]{index=6}
        num_samples=cfg.sae.num_samples,
        num_steps=cfg.sae.num_steps,
    )
    loader = PyGDataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)

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

    # Save as a list of per-snapshot arrays (N varies across samples).
    # Each item corresponds to one dataset index = (sample, time).
    meta = []
    for idx, batch in enumerate(loader):
        # dataset returns (graph, cells, rollout_mask) for split != train
        graph, cells, rollout_mask = batch
        graph = graph.to(device)

        hL = get_hL_embeddings(model, graph.x, graph.edge_attr, graph)  # (N, 128)
        hL = hL.detach().cpu().numpy()

        # mesh_pos stored in graph["mesh_pos"] in inference.py; keep it for viz later
        mesh_pos = graph["mesh_pos"].detach().cpu().numpy()
        cells_np = np.squeeze(cells.numpy())
        mask_np = rollout_mask.detach().cpu().numpy()

        np.savez_compressed(
            os.path.join(out_dir, f"emb_{idx:06d}.npz"),
            hL=hL,
            mesh_pos=mesh_pos,
            cells=cells_np,
            rollout_mask=mask_np,
        )
        meta.append((idx, hL.shape[0]))

    np.save(os.path.join(out_dir, "index.npy"), np.array(meta, dtype=object))
    print(f"[SAE] Saved {len(meta)} snapshots to {out_dir}")