import hydra
from omegaconf import DictConfig, OmegaConf

from cascadelad.data.prepare import prepare
from cascadelad.utils.run import start_run


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run_dir, logger = start_run(cfg, "configs/experiment/prepare_hdfs.yaml")
    out_dir = prepare(OmegaConf.to_container(cfg, resolve=True), logger)
    logger.info("[outputs]")
    logger.info("  run_dir: %s", run_dir)
    logger.info("  processed_dir: %s", out_dir)


if __name__ == "__main__":
    main()
