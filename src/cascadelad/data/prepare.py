"""Raw log -> processed train/val/test windows + vocab, cached under data/processed/<hash>/."""

import hashlib
import json
import logging
import shutil
from collections.abc import Mapping
from itertools import islice
from pathlib import Path

import pandas as pd

from cascadelad.data.grouping import build_windows, session_bounds
from cascadelad.data.loaders import get_loader
from cascadelad.data.parsing import drain3_version, parse_cached, source_hash
from cascadelad.data.splitting import chronological_split
from cascadelad.data.vocab import Vocab
from cascadelad.utils.run import log_section

SPLITS = ("train", "val", "test")
OUT_COLUMNS = ["group_id", "event_ids", "timestamp", "label"]
OUT_FILES = (*(f"{s}.parquet" for s in SPLITS), "vocab.json", "meta.json")
HASH_CHUNK_BYTES = 1 << 20
BUILD_STATS = ("purged train", "purged val", "train anomalous dropped")


def _file_md5(path: Path, max_lines: int | None = None) -> dict:
    """Size and md5 of a file, or md5 of only its first max_lines lines (None = whole file).

    With max_lines the size is left out: lines past max_lines are never read, so they must not
    change the key.
    """
    h = hashlib.md5()  # noqa: S324 - fingerprint, not security
    with open(path, "rb") as f:
        if max_lines is None:
            for chunk in iter(lambda: f.read(HASH_CHUNK_BYTES), b""):
                h.update(chunk)
            return {"size": path.stat().st_size, "md5": h.hexdigest()}
        for line in islice(f, max_lines):
            h.update(line)
    return {"md5": h.hexdigest()}


def raw_fingerprint(dataset_cfg: Mapping) -> dict:
    """Fingerprint of the raw files a load reads: the first max_lines of the log, the labels."""
    raw_dir = Path(dataset_cfg["raw_dir"])
    out = {"log_file": _file_md5(raw_dir / dataset_cfg["log_file"], dataset_cfg.get("max_lines"))}
    if dataset_cfg.get("label_file"):
        out["label_file"] = _file_md5(raw_dir / dataset_cfg["label_file"])
    return out


def _key_parts(cfg: Mapping) -> dict:
    return {
        "dataset": cfg["dataset"],
        "parser": {k: v for k, v in cfg["parser"].items() if k != "parsed_dir"},
        "split": cfg["split"],
        "vocab": cfg["vocab"],
        "code": source_hash(),
        "raw": raw_fingerprint(cfg["dataset"]),
        "drain3": drain3_version(),
    }


def _hash_parts(parts: Mapping) -> str:
    blob = json.dumps(parts, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def processed_key(cfg: Mapping) -> str:
    """Hash of the configs (parser minus parsed_dir), data code, raw files and drain3 version."""
    return _hash_parts(_key_parts(cfg))


def _hit_stats(cfg: Mapping, key: str, out_dir: Path) -> dict:
    """[data] stats of this run on a cache hit, recomputed from the cached files.

    Build-time counts (purged, anomalous dropped) are not in the outputs, so they come from
    meta.json, which is written with the outputs, and are labeled "(from build)".
    """
    frames = {s: pd.read_parquet(out_dir / f"{s}.parquet") for s in SPLITS}
    vocab = Vocab.from_json(out_dir / "vocab.json")
    test_ids = [e for ids in frames["test"]["event_ids"] for e in ids]
    build = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))["stats"]
    return {
        "raw_dir": cfg["dataset"]["raw_dir"],
        "processed key": key,
        "processed dir": out_dir,
        "processed cache": "hit",
        "parse cache": "not run",
        **{f"groups {s}": len(frames[s]) for s in SPLITS},
        **{f"{k} (from build)": build[k] for k in BUILD_STATS},
        **{f"anomaly % {s}": f"{100 * frames[s]['label'].mean():.2f}" for s in SPLITS},
        "vocab size": vocab.size,
        "test unk rate": f"{sum(e == vocab.unk_id for e in test_ids) / len(test_ids):.4f}",
    }


def prepare(cfg: Mapping, logger: logging.Logger) -> Path:
    """Build (or reuse) the processed dataset and return its directory."""
    key_parts = _key_parts(cfg)
    key = _hash_parts(key_parts)
    out_dir = Path(cfg["paths"]["processed_dir"]) / key
    if all((out_dir / f).exists() for f in OUT_FILES):
        log_section(logger, "data", _hit_stats(cfg, key, out_dir))
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
        "processed dir": out_dir,
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
        "key_parts": key_parts,
        "stats": stats,
        "cfg": {k: cfg[k] for k in ("dataset", "parser", "split", "vocab")},
    }
    (tmp_dir / "meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp_dir.rename(out_dir)
    return out_dir
