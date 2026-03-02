## python run_sae_pipeline.py +sae=sae

import hydra
from omegaconf import DictConfig

from sae_interp.extract_embeddings import extract_and_save
from sae_interp.train_sae import train_sae


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # 1) extract h^L embeddings from pretrained MGN
    extract_and_save(cfg)

    # 2) train SAE on those embeddings
    train_sae(cfg)

    # 3) analysis/viz step (you’ll add once SAE is trained)
    # from sae_interp.viz_mesh import run_viz
    # run_viz(cfg)

if __name__ == "__main__":
    main()