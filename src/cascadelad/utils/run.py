import logging
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from cascadelad.utils.seed import set_seed

LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"


def git_state(cwd: Path | None = None) -> tuple[str, bool]:
    """Return (HEAD sha, dirty flag); ("unknown", True) if git is unavailable."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown", True
    return sha, bool(status.strip())


def create_run_dir(runs_dir: Path, run_name: str, overwrite: bool = False) -> Path:
    """Create runs_dir/run_name; refuse to reuse a non-empty dir unless overwrite is set."""
    run_dir = Path(runs_dir) / run_name
    if run_dir.exists() and any(run_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"run dir {run_dir} is not empty; set overwrite=true to reuse it")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_git_commit(run_dir: Path, sha: str, dirty: bool) -> None:
    state = "dirty" if dirty else "clean"
    (run_dir / "git_commit.txt").write_text(f"{sha} {state}\n", encoding="utf-8")


def save_config(cfg: DictConfig, run_dir: Path) -> None:
    OmegaConf.save(cfg, run_dir / "config.yaml", resolve=True)


def setup_logging(run_dir: Path, name: str = "cascadelad") -> logging.Logger:
    """Log to console and run_dir/run.log in plain text."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    formatter = logging.Formatter(LOG_FORMAT)
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(run_dir / "run.log", encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def log_section(logger: logging.Logger, section: str, items: Mapping[str, object]) -> None:
    logger.info("[%s]", section)
    for key, value in items.items():
        logger.info("  %s: %s", key, value)


def start_run(cfg: DictConfig, config_path: str) -> tuple[Path, logging.Logger]:
    """Create the run dir, snapshot config and git state, and log the [setup] section."""
    run_dir = create_run_dir(Path(cfg.paths.runs_dir), cfg.run_name, cfg.overwrite)
    set_seed(cfg.seed)
    sha, dirty = git_state()
    write_git_commit(run_dir, sha, dirty)
    save_config(cfg, run_dir)
    logger = setup_logging(run_dir)
    log_section(
        logger,
        "setup",
        {
            "git_sha": f"{sha} ({'dirty' if dirty else 'clean'})",
            "seed": cfg.seed,
            "device": cfg.device,
            "config": config_path,
        },
    )
    return run_dir, logger
