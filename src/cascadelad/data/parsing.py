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

    def add(self, contents: Iterable[str]) -> list[tuple[str, str]]:
        """Online mode: feed lines in order, no dedup. Per line returns (template_id, template)
        as they were right after that line was added; earlier results are never revised."""
        out = []
        for content in contents:
            res = self._miner.add_log_message(content)
            out.append((f"E{res['cluster_id']}", res["template_mined"]))
        return out

    @property
    def n_clusters(self) -> int:
        return len(self._miner.drain.clusters)

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


CACHE_FILES = ("lines.parquet", "templates.json", "meta.json")
DATA_DIR = Path(__file__).resolve().parent


def source_hash(root: Path = DATA_DIR) -> str:
    """Hash of every .py file under root (path + bytes), so editing data code changes cache keys."""
    h = hashlib.sha256()
    for path in sorted(root.rglob("*.py"), key=lambda p: p.relative_to(root).as_posix()):
        body = path.read_bytes()
        h.update(f"{path.relative_to(root).as_posix()}\0{len(body)}\0".encode())
        h.update(body)
    return h.hexdigest()[:16]


def drain3_version() -> str:
    return importlib.metadata.version("drain3")


FIT_ON_MODES = ("stream", "train")


def _check_mode(cfg: Mapping, n: int, fit_mask: Sequence[bool] | None) -> str:
    mode = cfg["fit_on"]
    if mode not in FIT_ON_MODES:
        raise ValueError(f"fit_on must be one of {FIT_ON_MODES}, got {mode!r}")
    if mode == "stream":
        if fit_mask is not None:
            raise ValueError("fit_on=stream takes no fit_mask")
        return mode
    if fit_mask is None:
        raise ValueError("fit_on=train needs a fit_mask")
    if len(fit_mask) != n:
        raise ValueError(f"fit_mask length {len(fit_mask)} != {n} contents")
    if not any(fit_mask):
        raise ValueError("fit_mask must select at least one line")
    return mode


def cache_key(cfg: Mapping, contents: Sequence[str], fit_mask: Sequence[bool] | None = None) -> str:
    """Hash of parser config (minus parsed_dir, incl. fit_on), data code, drain3 version, contents.

    Under fit_on=train the fit mask is part of the key (D-6f30); under stream it is not, so a
    new split over the same stream is a cache hit.
    """
    h = hashlib.sha256()
    cfg_part = {k: v for k, v in cfg.items() if k != "parsed_dir"}
    cfg_part["_code"] = source_hash()
    cfg_part["_drain3"] = drain3_version()
    h.update(json.dumps(cfg_part, sort_keys=True, default=str).encode("utf-8"))
    body = "\n".join(contents).encode("utf-8")
    h.update(f"|{len(contents)}:{len(body)}|".encode())
    h.update(body)
    if cfg["fit_on"] == "train" and fit_mask is not None:
        h.update(b"|mask|")
        h.update(bytes(bool(m) for m in fit_mask))
    return h.hexdigest()[:16]


def _parse(
    contents: Sequence[str], cfg: Mapping, mode: str, fit_mask: Sequence[bool] | None
) -> tuple[pd.DataFrame, dict[str, str], dict]:
    parser = DrainParser(cfg)
    if mode == "stream":
        pairs = parser.add(contents)
        ids = [i for i, _ in pairs]
        templates = [t for _, t in pairs]
        extra = {"n_clusters_final": parser.n_clusters}
        known = parser.templates()
    else:
        unique_fit = list(dict.fromkeys(c for c, m in zip(contents, fit_mask, strict=True) if m))
        parser.fit(unique_fit)
        n_fit = parser.n_clusters
        unique_all = list(dict.fromkeys(contents))
        id_of = dict(zip(unique_all, parser.transform(unique_all), strict=True))
        known = parser.templates()
        text_of = {c: known.get(i, cfg["unk_token"]) for c, i in id_of.items()}
        ids = [id_of[c] for c in contents]
        templates = [text_of[c] for c in contents]
        extra = {"n_clusters_fit": n_fit, "n_clusters_after_apply": parser.n_clusters}
    return pd.DataFrame({"template_id": ids, "template": templates}), known, extra


def parse_lines(
    contents: Sequence[str], cfg: Mapping, fit_mask: Sequence[bool] | None = None
) -> tuple[pd.DataFrame, bool]:
    """Template id and text for every line, rows aligned with `contents`.

    fit_on=stream: one online Drain pass in order; ids and texts are as of each line.
    fit_on=train: Drain learns on the lines where fit_mask is True, then only matches;
    unmatched lines get unk_token. Results are cached in <parsed_dir>/<key>/; a hit never
    builds a parser. Returns (DataFrame[template_id, template], cache_hit).
    """
    mode = _check_mode(cfg, len(contents), fit_mask)
    cache_dir = Path(cfg["parsed_dir"]) / cache_key(cfg, contents, fit_mask)
    if all((cache_dir / f).exists() for f in CACHE_FILES):
        logger.info("cache hit: %s", cache_dir)
        return pd.read_parquet(cache_dir / "lines.parquet"), True

    logger.info("cache miss: parsing %d lines (fit_on=%s)", len(contents), mode)
    lines, templates, extra = _parse(contents, cfg, mode, fit_mask)
    meta = {
        "fit_on": mode,
        "n_lines": len(contents),
        "n_templates": int(lines["template_id"].nunique()),
        "n_unk": int((lines["template_id"] == cfg["unk_token"]).sum()),
        **extra,
        "drain3_version": drain3_version(),
        "code_hash": source_hash(),
        "cfg": dict(cfg),
    }
    tmp_dir = cache_dir.with_name(cache_dir.name + ".tmp")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True)
    lines.to_parquet(tmp_dir / "lines.parquet", index=False)
    (tmp_dir / "templates.json").write_text(json.dumps(templates, indent=1), encoding="utf-8")
    (tmp_dir / "meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    tmp_dir.rename(cache_dir)
    return lines, False
