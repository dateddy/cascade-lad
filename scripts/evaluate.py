from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from cascadelad.data.prepare import SPLITS, load_split, prepare
from cascadelad.data.vocab import Vocab
from cascadelad.methods.pipeline import run_pipeline
from cascadelad.utils.run import start_run


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    exp = HydraConfig.get().runtime.choices.get("experiment")
    config_path = f"configs/experiment/{exp}.yaml" if exp else "configs/config.yaml"
    run_dir, logger = start_run(cfg, config_path)
    plain = OmegaConf.to_container(cfg, resolve=True)
    out_dir = prepare(plain, logger)
    splits = {name: load_split(out_dir, name) for name in SPLITS}
    vocab = Vocab.from_json(Path(out_dir) / "vocab.json")
    run_pipeline(plain, splits, vocab.size, logger, run_dir)


if __name__ == "__main__":
    main()
