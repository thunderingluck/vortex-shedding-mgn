# run_eval.py
import torch
from torch.utils.data import DataLoader
from hydra.utils import to_absolute_path
import os
import hydra

from .sae import SparseAutoencoder
from .eval_sae import eval_sae
from .train_sae import EmbeddingNPZDataset, collate_concat


def run_eval(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(
        os.path.join(cfg.sae.train.ckpt_dir, "sae_best.pt"),
        map_location=device,
    )
    sae = SparseAutoencoder(d_in=cfg.sae.d_in, expansion=cfg.sae.expansion).to(device)
    sae.load_state_dict(ckpt["sae"])

    ds = EmbeddingNPZDataset(to_absolute_path(cfg.sae.out_dir))
    loader = DataLoader(
        ds,
        batch_size=cfg.sae.train.snapshot_batch_size,
        shuffle=False,
        num_workers=cfg.sae.train.num_workers,
        collate_fn=collate_concat,
    )

    metrics = eval_sae(sae, loader, device, cfg.sae.lam)
    for k, v in metrics.items():
        print(f"{k}: {v:.4e}" if isinstance(v, float) else f"{k}: {v}")


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg):
    run_eval(cfg)

if __name__ == "__main__":
    main()
