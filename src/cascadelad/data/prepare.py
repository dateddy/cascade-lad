"""Raw log -> processed train/val/test windows + vocab, cached under data/processed/<hash>/."""

import hashlib
import json
import logging
import shutil
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from cascadelad.data.grouping import build_windows, session_bounds
from cascadelad.data.loaders import get_loader
from cascadelad.data.parsing import parse_cached
from cascadelad.data.splitting import chronological_split
from cascadelad.data.vocab import Vocab
from cascadelad.utils.run import log_section

SPLITS = ("train", "val", "test")
OUT_COLUMNS = ["group_id", "event_ids", "timestamp", "label"]
OUT_FILES = (*(f"{s}.parquet" for s in SPLITS), "vocab.json", "meta.json")


def processed_key(cfg: Mapping) -> str:
    """Hash of the dataset, parser (minus parsed_dir), split and vocab configs."""
    parts = {
        "dataset": cfg["dataset"],
        "parser": {k: v for k, v in cfg["parser"].items() if k != "parsed_dir"},
        "split": cfg["split"],
        "vocab": cfg["vocab"],
    }
    blob = json.dumps(parts, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def prepare(cfg: Mapping, logger: logging.Logger) -> Path:
    """Build (or reuse) the processed dataset and return its directory."""
    key = processed_key(cfg)
    out_dir = Path(cfg["paths"]["processed_dir"]) / key
    if all((out_dir / f).exists() for f in OUT_FILES):
        stats = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))["stats"]
        stats["processed cache"] = "hit"
        log_section(logger, "data", stats)
        return out_dir

    lines = get_loader(cfg["dataset"]["name"], cfg["dataset"]).load()
    split = chronological_split(session_bounds(lines), cfg["split"])
    ids = {"train": split.train, "val": split.val, "test": split.test}
    lines = lines[lines["group_id"].isin(set().union(*ids.values()))]

    train_lines = lines[lines["group_id"].isin(set(split.train))]
    assignments, _, parse_hit = parse_cached(
        train_lines["content"].tolist(), lines["content"].tolist(), cfg["parser"]
    )
    id_of = dict(zip(assignments["content"], assignments["template_id"], strict=True))
    lines = lines.assign(template_id=lines["content"].map(id_of))

    windows = {
        name: build_windows(lines[lines["group_id"].isin(set(ids[name]))], "template_id")
        for name in SPLITS
    }
    vocab = Vocab.build(windows["train"]["templates"], cfg["vocab"])
    for w in windows.values():
        w["event_ids"] = w["templates"].map(vocab.encode)

    stats = {
        "raw_dir": cfg["dataset"]["raw_dir"],
        "processed key": key,
        "processed cache": "miss",
        "parse cache": "hit" if parse_hit else "miss",
        **{f"groups {s}": len(windows[s]) for s in SPLITS},
        "purged train": split.n_purged["train"],
        "purged val": split.n_purged["val"],
        "train anomalous dropped": split.n_train_anomalous_dropped,
        **{f"anomaly % {s}": f"{100 * windows[s]['label'].mean():.2f}" for s in SPLITS},
        "vocab size": vocab.size,
        "test unk rate": f"{vocab.unk_rate(windows['test']['templates']):.4f}",
    }
    log_section(logger, "data", stats)

    tmp_dir = out_dir.with_name(out_dir.name + ".tmp")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True)
    for name, w in windows.items():
        pd.DataFrame(w[OUT_COLUMNS]).to_parquet(tmp_dir / f"{name}.parquet", index=False)
    vocab.to_json(tmp_dir / "vocab.json")
    meta = {
        "key": key,
        "stats": stats,
        "cfg": {k: cfg[k] for k in ("dataset", "parser", "split", "vocab")},
    }
    (tmp_dir / "meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp_dir.rename(out_dir)
    return out_dir
