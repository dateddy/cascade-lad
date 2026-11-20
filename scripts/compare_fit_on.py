import json

import hydra
from omegaconf import DictConfig, OmegaConf

from cascadelad.data.fit_on_compare import run_compare
from cascadelad.utils.run import start_run


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run_dir, logger = start_run(cfg, "configs/experiment/fit_on_hdfs_2k.yaml")
    metrics = run_compare(OmegaConf.to_container(cfg, resolve=True), logger)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
