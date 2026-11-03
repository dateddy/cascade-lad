import logging
import re

import numpy as np
import pytest
from omegaconf import OmegaConf

from cascadelad.utils.run import create_run_dir, start_run
from cascadelad.utils.seed import set_seed


@pytest.fixture
def cfg(tmp_path):
    return OmegaConf.create(
        {
            "run_name": "t",
            "seed": 0,
            "device": "cpu",
            "overwrite": False,
            "paths": {"runs_dir": str(tmp_path)},
        }
    )


@pytest.fixture(autouse=True)
def close_log_handlers():
    yield
    logger = logging.getLogger("cascadelad")
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def test_start_run_writes_files(cfg, tmp_path):
    run_dir, _ = start_run(cfg, "configs/config.yaml")
    assert run_dir == tmp_path / "t"
    for name in ("config.yaml", "git_commit.txt", "run.log"):
        assert (run_dir / name).is_file()
    log = (run_dir / "run.log").read_text(encoding="utf-8")
    assert "[setup]" in log
    assert "seed: 0" in log
    assert re.match(
        r"^(\w+|unknown) (clean|dirty)$", (run_dir / "git_commit.txt").read_text().strip()
    )


def test_create_run_dir_refuses_non_empty(tmp_path):
    run_dir = create_run_dir(tmp_path, "r")
    (run_dir / "x.txt").write_text("x")
    with pytest.raises(FileExistsError):
        create_run_dir(tmp_path, "r")
    assert create_run_dir(tmp_path, "r", overwrite=True) == run_dir


def test_set_seed_reproducible():
    set_seed(1)
    a = np.random.rand()
    set_seed(1)
    assert np.random.rand() == a
