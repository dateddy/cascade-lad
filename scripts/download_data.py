import hydra
from omegaconf import DictConfig, OmegaConf

from cascadelad.data.download import fetch_dataset


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    fetch_dataset(OmegaConf.to_container(cfg.dataset, resolve=True))


if __name__ == "__main__":
    main()
