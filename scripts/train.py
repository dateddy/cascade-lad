import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from cascadelad.utils.run import start_run


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    exp = HydraConfig.get().runtime.choices.get("experiment")
    config_path = f"configs/experiment/{exp}.yaml" if exp else "configs/config.yaml"
    run_dir, logger = start_run(cfg, config_path)
    logger.info("[outputs]")
    logger.info("  run_dir: %s", run_dir)


if __name__ == "__main__":
    main()
