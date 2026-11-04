import json

import hydra
from omegaconf import DictConfig, OmegaConf

from cascadelad.data.parser_check import run_check
from cascadelad.utils.run import start_run


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run_dir, logger = start_run(cfg, "configs/config.yaml")
    metrics = run_check(
        OmegaConf.to_container(cfg.dataset, resolve=True),
        OmegaConf.to_container(cfg.parser, resolve=True),
        logger,
    )
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
