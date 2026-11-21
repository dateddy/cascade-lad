"""Raw log -> processed train/val/test windows + vocab, cached under data/processed/<hash>/."""

import hashlib
import json
import logging
import shutil
from collections.abc import Mapping
from itertools import islice
from pathlib import Path

import pandas as pd

from cascadelad.data.grouping import assign_fixed_windows, build_windows, session_bounds
from cascadelad.data.loaders import get_loader
from cascadelad.data.parsing import drain3_version, parse_lines, source_hash
from cascadelad.data.splitting import split_groups
from cascadelad.data.vocab import Vocab
from cascadelad.utils.run import log_section

SPLITS = ("train", "val", "test")
OUT_COLUMNS = ["group_id", "event_ids", "timestamp", "label", "line_labels", "line_pos"]
LIST_COLUMNS = ("event_ids", "line_labels", "line_pos")  # per-line lists, one cell per window
OUT_FILES = (*(f"{s}.parquet" for s in SPLITS), "vocab.json", "meta.json")
HASH_CHUNK_BYTES = 1 << 20
BUILD_STATS = (
    "purged train",
    "purged val",
    "train anomalous dropped",
    "train anomalous kept",
    "val anomalous dropped",
    "tail lines dropped",
)
GROUPING_MODES = ("session", "fixed")


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


def max_lines_label(dataset_cfg: Mapping) -> str:
    """max_lines for the [data] log: the number, or "all" when it is null (D-346d)."""
    n = dataset_cfg.get("max_lines")
    return "all" if n is None else str(n)


def anomaly_count(labels: pd.Series) -> str:
    """Anomalous groups as "<count> (<percent>%)"; an empty split prints "0 (n/a)"."""
    if labels.empty:
        return "0 (n/a)"
    n = int((labels != 0).sum())
    return f"{n} ({100 * n / len(labels):.2f}%)"


def load_split(processed_dir: Path, name: str) -> pd.DataFrame:
    """Read one processed split with its list columns as list[int].

    pd.read_parquet returns list cells as numpy arrays, where `arr + [0, 0]` adds elementwise
    instead of padding (D-c9f0). Row order is the file's: (timestamp, group_id) chronological.
    """
    if name not in SPLITS:
        raise ValueError(f"unknown split {name!r}, expected one of {SPLITS}")
    path = Path(processed_dir) / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"processed split not found: {path}")
    df = pd.read_parquet(path)
    missing = [c for c in OUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}, expected {OUT_COLUMNS}")
    for col in LIST_COLUMNS:
        df[col] = [[int(x) for x in cell] for cell in df[col]]
    return df


def _hit_stats(cfg: Mapping, key: str, out_dir: Path) -> dict:
    """[data] stats of this run on a cache hit, recomputed from the cached files.

    Build-time counts (purged, anomalous dropped) are not in the outputs, so they come from
    meta.json, which is written with the outputs, and are labeled "(from build)".
    """
    frames = {s: load_split(out_dir, s) for s in SPLITS}
    vocab = Vocab.from_json(out_dir / "vocab.json")
    test_ids = [e for ids in frames["test"]["event_ids"] for e in ids]
    build = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))["stats"]
    return {
        "raw_dir": cfg["dataset"]["raw_dir"],
        "processed key": key,
        "processed dir": out_dir,
        "max_lines": max_lines_label(cfg["dataset"]),
        "processed cache": "hit",
        "parse cache": "not run",
        "fit_on": cfg["parser"]["fit_on"],
        "grouping": cfg["dataset"]["grouping"]["mode"],
        "split mode": cfg["split"]["mode"],
        **{f"groups {s}": len(frames[s]) for s in SPLITS},
        **{f"{k} (from build)": build[k] for k in BUILD_STATS},
        **{f"anomalies {s}": anomaly_count(frames[s]["label"]) for s in SPLITS},
        "vocab size": vocab.size,
        "test unk rate": f"{sum(e == vocab.unk_id for e in test_ids) / len(test_ids):.4f}",
    }


def group_lines(lines: pd.DataFrame, grouping: Mapping) -> tuple[pd.DataFrame, int, str, bool]:
    """Apply the dataset's grouping mode to loaded lines (log order, original index kept).

    Returns (grouped lines, tail lines dropped, label_rule, purge allowed). Sessions may
    overlap in time and use the split's purge flag; fixed windows never overlap, so they are
    split without purging and labeled by the max of their line labels.
    """
    if grouping["mode"] == "session":
        return lines, 0, "same", True
    if grouping["mode"] == "fixed":
        windowed, n_tail = assign_fixed_windows(lines, grouping["size"], grouping["step"])
        return windowed, n_tail, "max", False
    raise ValueError(f"unknown grouping mode {grouping['mode']!r}, expected {GROUPING_MODES}")


def prepare(cfg: Mapping, logger: logging.Logger) -> Path:
    """Build (or reuse) the processed dataset and return its directory."""
    key_parts = _key_parts(cfg)
    key = _hash_parts(key_parts)
    out_dir = Path(cfg["paths"]["processed_dir"]) / key
    if all((out_dir / f).exists() for f in OUT_FILES):
        log_section(logger, "data", _hit_stats(cfg, key, out_dir))
        return out_dir

    lines = get_loader(cfg["dataset"]["loader"], cfg["dataset"]).load()
    lines["line_pos"] = range(len(lines))  # stream position, joins stream and train runs
    grouped, n_tail, label_rule, purge_ok = group_lines(lines, cfg["dataset"]["grouping"])
    split_cfg = {**cfg["split"], "purge": cfg["split"]["purge"] and purge_ok}
    split = split_groups(session_bounds(grouped, label_rule), split_cfg)
    ids = {"train": split.train, "val": split.val, "test": split.test}

    # Parse every loaded row (purged groups and a dropped window tail included): a stream parse
    # must not depend on the split. fit_on=train fits on every train line, anomalous ones
    # included when drop_train_anomalies is false (D-3c69).
    train_pos = set(grouped.loc[grouped["group_id"].isin(set(split.train)), "line_pos"])
    mask = (
        lines["line_pos"].isin(train_pos).tolist() if cfg["parser"]["fit_on"] == "train" else None
    )
    parsed, parse_hit = parse_lines(lines["content"].tolist(), cfg["parser"], mask)
    pos = grouped["line_pos"].to_numpy()  # line_pos is the row number in `lines` and `parsed`
    grouped = grouped.assign(
        template_id=parsed["template_id"].to_numpy()[pos],
        template=parsed["template"].to_numpy()[pos],
    )
    grouped = grouped[grouped["group_id"].isin(set().union(*ids.values()))]

    windows = {
        name: build_windows(
            grouped[grouped["group_id"].isin(set(ids[name]))],
            "template_id",
            extra_cols=("line_pos",),
            label_rule=label_rule,
        )
        for name in SPLITS
    }
    vocab = Vocab.build(windows["train"]["templates"], cfg["vocab"])
    for w in windows.values():
        w["event_ids"] = w["templates"].map(vocab.encode)

    stats = {
        "raw_dir": cfg["dataset"]["raw_dir"],
        "processed key": key,
        "processed dir": out_dir,
        "max_lines": max_lines_label(cfg["dataset"]),
        "processed cache": "miss",
        "parse cache": "hit" if parse_hit else "miss",
        "fit_on": cfg["parser"]["fit_on"],
        "grouping": cfg["dataset"]["grouping"]["mode"],
        "split mode": cfg["split"]["mode"],
        **{f"groups {s}": len(windows[s]) for s in SPLITS},
        "purged train": split.n_purged["train"],
        "purged val": split.n_purged["val"],
        "train anomalous dropped": split.n_train_anomalous_dropped,
        "train anomalous kept": split.n_train_anomalous_kept,
        "val anomalous dropped": split.n_val_anomalous_dropped,
        "tail lines dropped": n_tail,
        **{f"anomalies {s}": anomaly_count(windows[s]["label"]) for s in SPLITS},
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
