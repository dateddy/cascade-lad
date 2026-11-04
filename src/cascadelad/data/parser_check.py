"""Check the Drain parser against a Loghub structured file (ground-truth EventIds)."""

import logging
import re
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from cascadelad.data.parsing import grouping_accuracy, parse_cached
from cascadelad.utils.run import log_section


def read_contents(dataset_cfg: Mapping) -> list[str]:
    line_re = re.compile(dataset_cfg["line_regex"])
    path = Path(dataset_cfg["raw_dir"]) / dataset_cfg["log_file"]
    contents = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, start=1):
            m = line_re.match(line.rstrip("\r\n"))
            if m is None:
                raise ValueError(f"line {i}: does not match line_regex: {line[:100]!r}")
            contents.append(m["content"])
    return contents


def run_check(
    dataset_cfg: Mapping, parser_cfg: Mapping, logger: logging.Logger
) -> dict[str, float | int | bool]:
    structured = pd.read_csv(
        Path(dataset_cfg["raw_dir"]) / dataset_cfg["structured_file"],
        dtype=str,
        keep_default_na=False,
    )
    truth_contents = structured[dataset_cfg["content_column"]].tolist()
    truth_events = structured[dataset_cfg["event_column"]].tolist()
    contents = read_contents(dataset_cfg)
    if len(contents) != len(truth_contents):
        raise ValueError(f"log has {len(contents)} lines, csv has {len(truth_contents)}")
    for i, (a, b) in enumerate(zip(contents, truth_contents, strict=True), start=1):
        if a != b:
            raise ValueError(f"line {i}: log content {a!r} != csv content {b!r}")

    assignments, templates, cache_hit = parse_cached(contents, contents, parser_cfg)
    id_of = dict(zip(assignments["content"], assignments["template_id"], strict=True))
    pred = [id_of[c] for c in contents]
    metrics: dict[str, float | int | bool] = {
        "lines": len(contents),
        "predicted_templates": len(set(pred)),
        "true_templates": len(set(truth_events)),
        "n_unk": sum(p == parser_cfg["unk_token"] for p in pred),
        "grouping_accuracy": grouping_accuracy(pred, truth_events),
        "cache_hit": cache_hit,
    }
    log_section(
        logger,
        "parse",
        {
            **metrics,
            "grouping_accuracy": f"{metrics['grouping_accuracy']:.4f}",
            "cache": "hit" if cache_hit else "miss",
        },
    )
    return metrics
