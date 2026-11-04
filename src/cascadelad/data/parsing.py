"""Drain log parsing: contents -> template ids.

fit() learns templates; transform() only matches and never learns, so lines whose
template was not seen in fit() map to unk_token (research invariant 2).
"""

import hashlib
import importlib.metadata
import json
import logging
import shutil
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import pandas as pd
from drain3 import TemplateMiner
from drain3.masking import MaskingInstruction
from drain3.template_miner_config import TemplateMinerConfig

logger = logging.getLogger(__name__)


class DrainParser:
    def __init__(self, cfg: Mapping) -> None:
        self.cfg = cfg
        tm_cfg = TemplateMinerConfig()
        tm_cfg.drain_depth = cfg["depth"]
        tm_cfg.drain_sim_th = cfg["sim_th"]
        tm_cfg.drain_max_children = cfg["max_children"]
        tm_cfg.drain_max_clusters = cfg["max_clusters"]
        tm_cfg.drain_extra_delimiters = list(cfg["extra_delimiters"])
        tm_cfg.parametrize_numeric_tokens = cfg["parametrize_numeric_tokens"]
        tm_cfg.masking_instructions = [
            MaskingInstruction(m["regex"], m["mask_with"]) for m in cfg["masking"]
        ]
        self._miner = TemplateMiner(config=tm_cfg)
        self._fitted = False

    def fit(self, contents: Iterable[str]) -> "DrainParser":
        for content in dict.fromkeys(contents):
            self._miner.add_log_message(content)
        self._fitted = True
        return self

    def transform(self, contents: Iterable[str]) -> list[str]:
        if not self._fitted:
            raise RuntimeError("transform() called before fit()")
        out = []
        for content in contents:
            cluster = self._miner.match(content, full_search_strategy=self.cfg["match_strategy"])
            out.append(self.cfg["unk_token"] if cluster is None else f"E{cluster.cluster_id}")
        return out

    def templates(self) -> dict[str, str]:
        return {f"E{c.cluster_id}": c.get_template() for c in self._miner.drain.clusters}


def grouping_accuracy(pred: Sequence[str], truth: Sequence[str]) -> float:
    """Loghub grouping accuracy: a line is correct iff the set of lines sharing its
    predicted id equals the set of lines sharing its true id."""
    if len(pred) != len(truth):
        raise ValueError(f"length mismatch: {len(pred)} predictions vs {len(truth)} labels")
    if not len(pred):
        raise ValueError("empty input")
    pred_groups: dict[str, list[int]] = defaultdict(list)
    true_groups: dict[str, list[int]] = defaultdict(list)
    for i, (p, t) in enumerate(zip(pred, truth, strict=True)):
        pred_groups[p].append(i)
        true_groups[t].append(i)
    correct = sum(len(idx) for idx in pred_groups.values() if idx == true_groups[truth[idx[0]]])
    return correct / len(pred)


CACHE_FILES = ("assignments.csv", "templates.json", "meta.json")


def cache_key(cfg: Mapping, fit_contents: Sequence[str], apply_contents: Sequence[str]) -> str:
    """Hash of the parser config (minus parsed_dir) and both content lists."""
    h = hashlib.sha256()
    cfg_part = {k: v for k, v in cfg.items() if k != "parsed_dir"}
    h.update(json.dumps(cfg_part, sort_keys=True, default=str).encode("utf-8"))
    for section in (fit_contents, apply_contents):
        body = "\n".join(section).encode("utf-8")
        h.update(f"|{len(section)}:{len(body)}|".encode())
        h.update(body)
    return h.hexdigest()[:16]


def parse_cached(
    fit_contents: Sequence[str], apply_contents: Sequence[str], cfg: Mapping
) -> tuple[pd.DataFrame, dict[str, str], bool]:
    """Fit on fit_contents, assign template ids to the unique apply_contents.

    Returns (assignments[content, template_id], templates, cache_hit). Results are
    cached in <parsed_dir>/<key>/; a hit never builds a parser.
    """
    cache_dir = Path(cfg["parsed_dir"]) / cache_key(cfg, fit_contents, apply_contents)
    if all((cache_dir / f).exists() for f in CACHE_FILES):
        logger.info("cache hit: %s", cache_dir)
        assignments = pd.read_csv(cache_dir / "assignments.csv", dtype=str, keep_default_na=False)
        templates = json.loads((cache_dir / "templates.json").read_text(encoding="utf-8"))
        return assignments, templates, True

    unique_fit = list(dict.fromkeys(fit_contents))
    logger.info("cache miss: fitting on %d unique lines", len(unique_fit))
    parser = DrainParser(cfg).fit(unique_fit)
    unique_apply = list(dict.fromkeys(apply_contents))
    assignments = pd.DataFrame(
        {"content": unique_apply, "template_id": parser.transform(unique_apply)}
    )
    templates = parser.templates()
    meta = {
        "n_fit": len(unique_fit),
        "n_apply": len(unique_apply),
        "n_templates": len(templates),
        "n_unk": int((assignments["template_id"] == cfg["unk_token"]).sum()),
        "drain3_version": importlib.metadata.version("drain3"),
        "cfg": dict(cfg),
    }
    tmp_dir = cache_dir.with_name(cache_dir.name + ".tmp")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True)
    assignments.to_csv(tmp_dir / "assignments.csv", index=False)
    (tmp_dir / "templates.json").write_text(json.dumps(templates, indent=1), encoding="utf-8")
    (tmp_dir / "meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    tmp_dir.rename(cache_dir)
    return assignments, templates, False
