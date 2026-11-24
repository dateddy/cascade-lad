"""Run the Stage 5 pipeline on prepared splits: baselines + primary detector, one table.

`scripts/evaluate.py` composes the config and loads the splits; everything after that lives
here so it can be tested without Hydra or a data download.
"""

import json
import logging
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from cascadelad.detectors import get_detector
from cascadelad.detectors.base import Detector
from cascadelad.evaluation.report import format_ci_table, format_table
from cascadelad.methods.detector_only import RULE, MethodResult, run_detector_only
from cascadelad.methods.outputs import metrics_json, scores_frame
from cascadelad.utils.run import log_section

METRICS_FILE = "metrics.json"
SCORES_FILE = "scores.parquet"


def describe_policy(threshold_cfg: Mapping) -> str:
    """e.g. "val_percentile(percentile=99)"."""
    args = ", ".join(f"{k}={v}" for k, v in threshold_cfg.items() if k != "method")
    return f"{threshold_cfg['method']}({args})"


def describe_detector(detector: Detector) -> str:
    """What the detector learned, for the [model] log section."""
    if hasattr(detector, "n_components_"):
        return f"{detector.n_components_} components"
    if hasattr(detector, "seen"):
        return f"{len(detector.seen)} train sequences"
    return "no parameters"


def run_pipeline(
    cfg: Mapping,
    splits: Mapping[str, pd.DataFrame],
    n_features: int,
    logger: logging.Logger,
    run_dir: Path,
) -> dict:
    """Evaluate the trivial baselines and the primary detector on the test split.

    Baselines come first in the table (they are decided by a fixed rule, D-47fc); the primary
    detector (cfg["model"]) goes through the threshold policy cfg["threshold"]. Writes
    metrics.json and scores.parquet to `run_dir`.
    """
    policy = cfg["threshold"]
    detectors: dict[str, Detector] = {}
    rules: dict[str, float | None] = {}
    for name, bcfg in cfg["baselines"].items():
        detectors[name] = get_detector(bcfg, n_features)
        rules[name] = bcfg["rule_threshold"]
    primary = cfg["model"]["name"]
    detectors[primary] = get_detector(cfg["model"], n_features)
    rules[primary] = None

    results: dict[str, MethodResult] = {
        name: run_detector_only(name, det, splits, policy, rules[name], cfg["evaluation"])
        for name, det in detectors.items()
    }

    log_section(
        logger,
        "model",
        {
            **{f"{name}": describe_detector(det) for name, det in detectors.items()},
            "n_features": n_features,
            **{
                f"{name} threshold": f"{res.threshold:.6g} ({res.threshold_method}, "
                f"val flag rate {res.val_flag_rate:.4f})"
                for name, res in results.items()
            },
        },
    )

    rows = {name: [res.metrics] for name, res in results.items()}
    ev = cfg["evaluation"]
    common = {
        "dataset": cfg["dataset"]["name"],
        "split": cfg["split"]["mode"],
        "seeds": [cfg["seed"]],
        "decimals": ev["decimals"],
    }
    threshold_label = f"{describe_policy(policy)}; trivial baselines: {RULE}"
    logger.info("[eval]")
    logger.info(
        "\n%s", format_table(rows, threshold=threshold_label, ddof=ev["std_ddof"], **common)
    )
    logger.info(
        "\n%s",
        format_ci_table(rows, ci_level=ev["bootstrap"]["ci_level"], **common),
    )

    run_dir = Path(run_dir)
    metrics = metrics_json(
        results[primary],
        {name: res for name, res in results.items() if name != primary},
        max_lines=cfg["dataset"].get("max_lines"),
    )
    (run_dir / METRICS_FILE).write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    scores_frame(list(results.values()), splits).to_parquet(run_dir / SCORES_FILE, index=False)
    log_section(
        logger,
        "outputs",
        {"run_dir": run_dir, "metrics": run_dir / METRICS_FILE, "scores": run_dir / SCORES_FILE},
    )
    return {"metrics": metrics, "results": results}
