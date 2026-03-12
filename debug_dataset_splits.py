import hydra
from hydra.utils import to_absolute_path
from physicsnemo.datapipes.gnn.vortex_shedding_dataset import VortexSheddingDataset

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg):

    data_dir = to_absolute_path(cfg.data_dir)
    print("Using data_dir:", data_dir)

    for split in ["train", "val", "test"]:
        try:
            ds = VortexSheddingDataset(
                name=f"debug_{split}",
                data_dir=data_dir,
                split=split,
                num_samples=cfg.sae.num_samples,
                num_steps=cfg.sae.num_steps,
            )
            print(f"{split}: len = {len(ds)}")

        except Exception as e:
            print(f"{split}: ERROR -> {repr(e)}")

if __name__ == "__main__":
    main()
