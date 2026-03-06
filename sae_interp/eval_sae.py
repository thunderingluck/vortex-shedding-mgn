# eval_sae.py
import torch
import numpy as np


@torch.no_grad()
def eval_sae(sae, loader, device, lam, max_batches=50):
    sae.eval()
    total_recon, total_spars, total_l0 = 0.0, 0.0, 0.0
    all_alive = torch.zeros(sae.d_hid, device=device)
    n_batches = 0

    for h in loader:
        h = h.to(device)
        h_hat, z = sae(h)

        recon = (h_hat - h).pow(2).mean()
        spars = z.abs().mean()
        l0 = (z > 0).float().mean(dim=0).sum()  # avg features active per input

        # track which features ever fire
        all_alive += (z > 0).any(dim=0).float()

        total_recon += recon.item()
        total_spars += spars.item()
        total_l0 += l0.item()
        n_batches += 1
        if n_batches >= max_batches:
            break

    n = max(n_batches, 1)
    dead_frac = (all_alive == 0).float().mean().item()

    sae.train()
    return {
        "recon": total_recon / n,
        "sparsity_l1": total_spars / n,
        "l0": total_l0 / n,
        "dead_feature_frac": dead_frac,
    }