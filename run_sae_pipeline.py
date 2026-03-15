## python run_sae_pipeline.py +sae=sae

import os
import sys

import hydra
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig

from sae_interp.extract_embeddings import extract_and_save


def _run_with_args(main_fn, args_list):
    """Call an argparse-based main() with a synthetic argv."""
    old_argv = sys.argv
    sys.argv = [sys.argv[0]] + args_list
    try:
        main_fn()
    finally:
        sys.argv = old_argv


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    orig_cwd = get_original_cwd()
    sae_interp_dir = os.path.join(orig_cwd, "sae_interp")
    raw_dir = to_absolute_path(os.path.join(cfg.sae.out_root, "raw"))
    consolidated_dir = to_absolute_path(os.path.join(cfg.sae.out_root, "consolidated"))
    ckpt_dir = to_absolute_path(cfg.sae.train.ckpt_dir)

    # sae_interp/ must be on sys.path so that `from sae import ...` works
    # inside train_sae_rand.py and make_paper_figures.py
    if sae_interp_dir not in sys.path:
        sys.path.insert(0, sae_interp_dir)

    # 1) extract h^L embeddings from pretrained MGN → sae_embeddings/raw/
    if cfg.run_extraction:
        extract_and_save(cfg)

    # 2) merge per-step NPZ files into per-trajectory .npy → sae_embeddings/consolidated/
    if cfg.run_consolidation:
        from sae_interp.consolidate_embeddings import main as consolidate_main
        _run_with_args(consolidate_main, [
            "--raw_dir", raw_dir,
            "--out_dir", consolidated_dir,
            "--workers", str(cfg.sae.get("consolidate_workers", 8)),
        ])

    # 3) train SAE on consolidated embeddings → checkpoints/
    if cfg.run_train:
        from sae_interp.train_sae_rand import main as train_main
        _run_with_args(train_main, [
            "--emb_dir",      consolidated_dir,
            "--ckpt_dir",     ckpt_dir,
            "--d_in",         str(cfg.sae.d_in),
            "--expansion",    str(cfg.sae.expansion),
            "--lam",          str(cfg.sae.lam),
            "--lr",           str(cfg.sae.train.lr),
            "--max_epochs",   str(cfg.sae.train.max_epochs),
            "--val_every",    str(cfg.sae.train.val_every),
            "--patience",     str(cfg.sae.train.patience),
            "--l0_patience",  str(cfg.sae.train.l0_patience),
            "--l0_tol",       str(cfg.sae.train.l0_tol),
            "--seed",         str(cfg.sae.seed),
            "--val_frac",     str(cfg.sae.val_fraction),
        ])

    # 4) generate paper figures from raw embeddings + trained SAE
    if cfg.run_figures:
        from sae_interp.make_paper_figures import main as figures_main
        figures_dir = os.path.join(sae_interp_dir, "figures")
        _run_with_args(figures_main, [
            "--ckpt",     os.path.join(ckpt_dir, "sae_best.pt"),
            "--emb_dir",  raw_dir,
            "--out_dir",  figures_dir,
            "--metric",   cfg.sae.analysis.metric,
            "--topk",     str(cfg.sae.analysis.topk),
            "--eta_list", "20,85,300",
            "--eta_fig3", str(cfg.sae.analysis.eta),
        ])


if __name__ == "__main__":
    main()
