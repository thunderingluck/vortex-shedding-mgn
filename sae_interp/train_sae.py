import glob
import itertools
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from hydra.utils import to_absolute_path

from .sae import SparseAutoencoder


class EmbeddingNPZDataset(Dataset):
    def __init__(self, npz_dir: str):
        self.files = sorted(glob.glob(os.path.join(npz_dir, "emb_*.npz")))
        if not self.files:
            raise FileNotFoundError(f"No emb_*.npz found in {npz_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(self.files[i])
        hL = d["hL"].astype(np.float32)  # (N, 128)
        return torch.from_numpy(hL)


def collate_concat(batch):
    # batch is list of (N_i, 128) -> concat nodes across snapshots for SGD
    return torch.cat(batch, dim=0)


def train_sae(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = EmbeddingNPZDataset(to_absolute_path(cfg.sae.out_dir))
    # Paper uses batch size 128 (samples). Here snapshots have variable N,
    # so we batch snapshots and then concatenate nodes. :contentReference[oaicite:7]{index=7}
    loader = DataLoader(
        ds,
        batch_size=cfg.sae.train.snapshot_batch_size,
        shuffle=True,
        num_workers=cfg.sae.train.num_workers,
        collate_fn=collate_concat,
        drop_last=True,
    )

    sae = SparseAutoencoder(d_in=cfg.sae.d_in, expansion=cfg.sae.expansion).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=cfg.sae.train.lr)

    best_recon = float("inf")
    os.makedirs(cfg.sae.train.ckpt_dir, exist_ok=True)

    for step, h in enumerate(itertools.cycle(loader), start=1):
        h = h.to(device)

        loss, recon, spars = sae.loss(h, lam=cfg.sae.lam)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        # required by paper after *every* step :contentReference[oaicite:8]{index=8}
        sae.renorm_decoder_rows_()

        if step % cfg.sae.train.log_every == 0:
            print(f"step={step:6d} loss={loss.item():.4e} recon={recon.item():.4e} spars={spars.item():.4e}")

        # crude early stopping on recon
        if step % cfg.sae.train.ckpt_every == 0:
            if recon.item() < best_recon:
                best_recon = recon.item()
                path = os.path.join(cfg.sae.train.ckpt_dir, "sae_best.pt")
                torch.save({"sae": sae.state_dict(), "cfg": dict(cfg.sae)}, path)
                print(f"[SAE] saved best checkpoint -> {path} (recon={best_recon:.4e})")

        if step >= cfg.sae.train.max_steps:
            break