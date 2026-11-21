"""Checks around the Drain fit_on modes: stream prefix stability (S2-3) and stream-vs-train
comparison of processed outputs (S2-6)."""

import copy
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from cascadelad.data.loaders import get_loader
from cascadelad.data.parsing import DrainParser
from cascadelad.data.prepare import load_split, prepare
from cascadelad.data.vocab import Vocab
from cascadelad.utils.run import log_section

KEY = ["group_id", "line_pos"]


def prefix_mismatches(contents: Sequence[str], cfg: Mapping, train_ratio: float) -> int:
    """Count lines in the first int(n * train_ratio) whose stream template id differs between
    parsing that prefix alone and parsing the full stream (must be 0: ids never change)."""
    n_prefix = int(len(contents) * train_ratio)
    alone = DrainParser(cfg).add(contents[:n_prefix])
    full = DrainParser(cfg).add(contents)[:n_prefix]
    return sum(a[0] != f[0] for a, f in zip(alone, full, strict=True))


def _lines(out_dir: Path, split: str) -> tuple[pd.DataFrame, Vocab]:
    """One row per line of `split`: group_id, line_pos, event_id (vocab id)."""
    df = load_split(out_dir, split)
    lines = df[[*KEY, "event_ids"]].explode(["line_pos", "event_ids"])
    lines = lines.rename(columns={"event_ids": "event_id"}).astype({"line_pos": "int64"})
    lines["event_id"] = lines["event_id"].astype("int64")
    if lines.duplicated(KEY).any():
        raise ValueError(f"{out_dir}: duplicate (group_id, line_pos) keys in {split}")
    return lines.reset_index(drop=True), Vocab.from_json(out_dir / "vocab.json")


def compare_fit_on(stream_dir: Path, train_dir: Path, split: str = "test") -> dict[str, int]:
    """Join the processed `split` of a fit_on=stream and a fit_on=train run line by line.

    Returns line counts and `unk_train_invocab_stream`: lines that are <UNK> under train but
    in-vocab under stream (anomalies that stream parsing absorbs into known templates).
    Each run's event ids are compared against its own vocab's unk id.
    """
    s_lines, s_vocab = _lines(stream_dir, split)
    t_lines, t_vocab = _lines(train_dir, split)
    if set(s_lines["group_id"]) != set(t_lines["group_id"]):
        raise ValueError("stream and train runs have different groups; use the same split")
    joined = s_lines.merge(t_lines, on=KEY, how="outer", suffixes=("_s", "_t"), indicator=True)
    if (joined["_merge"] != "both").any():
        raise ValueError("stream and train lines do not join one-to-one on (group_id, line_pos)")
    unk_s = joined["event_id_s"] == s_vocab.unk_id
    unk_t = joined["event_id_t"] == t_vocab.unk_id
    return {
        "lines": len(s_lines),
        "joined": len(joined),
        "unk_stream": int(unk_s.sum()),
        "unk_train": int(unk_t.sum()),
        "unk_train_invocab_stream": int((unk_t & ~unk_s).sum()),
    }


def run_compare(cfg: Mapping, logger: logging.Logger) -> dict[str, int]:
    """Prepare the dataset under fit_on=stream and fit_on=train, then check S2-3 and S2-6."""
    dirs = {}
    for mode in ("stream", "train"):
        run_cfg = copy.deepcopy(dict(cfg))
        run_cfg["parser"]["fit_on"] = mode
        dirs[mode] = prepare(run_cfg, logger)
    contents = get_loader(cfg["dataset"]["loader"], cfg["dataset"]).load()["content"].tolist()
    stream_cfg = {**cfg["parser"], "fit_on": "stream"}
    metrics = {
        "prefix_mismatches": prefix_mismatches(
            contents, stream_cfg, cfg["compare"]["prefix_ratio"]
        ),
        **compare_fit_on(dirs["stream"], dirs["train"]),
    }
    log_section(logger, "compare", {**metrics, "prefix_ratio": cfg["compare"]["prefix_ratio"]})
    return metrics
