from pathlib import Path

from omegaconf import OmegaConf

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_eval_config_keys():
    cfg = OmegaConf.load(CONFIGS / "evaluation" / "default.yaml")
    assert cfg.bootstrap.n_resamples == 1000
    assert cfg.bootstrap.seed == 0
    assert cfg.bootstrap.ci_level == 0.95
    assert cfg.bootstrap.block_length is None
    assert list(cfg.flag_rates) == [0.1, 0.2]
    assert cfg.std_ddof == 1
    assert cfg.decimals == 4


def test_eval_group_is_in_default_composition():
    root = OmegaConf.load(CONFIGS / "config.yaml")
    assert {"evaluation": "default"} in [dict(d) for d in root.defaults if not isinstance(d, str)]
