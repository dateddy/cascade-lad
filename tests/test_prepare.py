import json
import logging
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from cascadelad.data import prepare as prepare_mod
from cascadelad.data.prepare import prepare, processed_key
from cascadelad.data.vocab import Vocab

ROOT = Path(__file__).resolve().parents[1]
ANOMALOUS = {"blk_2", "blk_8"}  # blk_2 sits fully in train, blk_8 in test


def write_raw(raw: Path) -> None:
    """10 blocks, block k starts at hour k. blk_6 (last train candidate) ends in hour 7, i.e.
    after the first val block starts, so it is purged. blk_9 (test) has a template train lacks."""
    rows = []
    for k in range(10):
        end_hour = 7 if k == 6 else k
        rows.append((f"{k:02d}0000", f"Received block blk_{k} of size 5 from /10.0.0.{k}"))
        if k == 9:
            rows.append((f"{k:02d}0010", f"completely unrelated message here now blk_{k}"))
        rows.append((f"{end_hour:02d}0030", f"Deleting block blk_{k} file x"))
    rows.sort(key=lambda r: r[0])
    log = "".join(f"081109 {t} 1 INFO dfs.DataNode: {c}\n" for t, c in rows)
    raw.mkdir(parents=True)
    (raw / "HDFS.log").write_text(log, encoding="utf-8")
    labels = "BlockId,Label\n" + "".join(
        f"blk_{k},{'Anomaly' if f'blk_{k}' in ANOMALOUS else 'Normal'}\n" for k in range(10)
    )
    (raw / "anomaly_label.csv").write_text(labels, encoding="utf-8")


def make_cfg(tmp_path: Path) -> dict:
    raw = tmp_path / "raw"
    write_raw(raw)
    load = lambda *p: OmegaConf.load(ROOT / "configs" / Path(*p))  # noqa: E731
    cfg = OmegaConf.create(
        {
            "dataset": load("dataset", "hdfs.yaml"),
            "parser": load("parser", "drain.yaml"),
            "split": load("split", "chrono.yaml"),
            "vocab": load("vocab", "default.yaml"),
            "paths": {"processed_dir": str(tmp_path / "processed")},
        }
    )
    cfg.dataset.raw_dir = str(raw)
    cfg.dataset.label_file = "anomaly_label.csv"
    cfg.parser.parsed_dir = str(tmp_path / "parsed")
    return OmegaConf.to_container(cfg, resolve=True)


def test_outputs_leakage_and_vocab(tmp_path, caplog):
    cfg = make_cfg(tmp_path)
    logger = logging.getLogger("test_prepare")
    caplog.set_level(logging.INFO, logger="test_prepare")
    out = prepare(cfg, logger)

    assert sorted(p.name for p in out.iterdir()) == [
        "meta.json",
        "test.parquet",
        "train.parquet",
        "val.parquet",
        "vocab.json",
    ]
    train, val, test = (pd.read_parquet(out / f"{s}.parquet") for s in ("train", "val", "test"))
    for df in (train, val, test):
        assert list(df.columns) == ["group_id", "event_ids", "timestamp", "label"]

    assert set(train["group_id"]) == {
        "blk_0",
        "blk_1",
        "blk_3",
        "blk_4",
        "blk_5",
    }  # 2 dropped, 6 purged
    assert list(val["group_id"]) == ["blk_7"]
    assert list(test["group_id"]) == ["blk_8", "blk_9"]
    assert (train["label"] == 0).all()
    assert train["timestamp"].max() < test["timestamp"].min()

    vocab = Vocab.from_json(out / "vocab.json")
    train_events = {e for ids in train["event_ids"] for e in ids}
    assert train_events <= set(vocab.token_to_id.values())
    assert vocab.unk_id not in train_events
    test_ids = [e for ids in test["event_ids"] for e in ids]
    assert vocab.unk_id in test_ids  # blk_9's unseen template

    text = caplog.text
    for needle in (
        "[data]",
        "groups train: 5",
        "purged train: 1",
        "train anomalous dropped: 1",
        "vocab size:",
        "test unk rate:",
        "processed cache: miss",
    ):
        assert needle in text, needle


def test_second_run_is_cache_hit_without_loading(tmp_path, monkeypatch, caplog):
    cfg = make_cfg(tmp_path)
    logger = logging.getLogger("test_prepare_hit")
    caplog.set_level(logging.INFO, logger="test_prepare_hit")
    first = prepare(cfg, logger)
    miss_unk = [ln for ln in caplog.text.splitlines() if "test unk rate:" in ln]

    def boom(*args, **kwargs):
        raise AssertionError("loader called on processed cache hit")

    monkeypatch.setattr(prepare_mod, "get_loader", boom)
    caplog.clear()
    assert prepare(cfg, logger) == first
    assert "processed cache: hit" in caplog.text
    assert "parse cache: not run" in caplog.text
    assert "groups train: 5" in caplog.text
    for name in ("purged train", "purged val", "train anomalous dropped"):
        assert f"{name} (from build):" in caplog.text
    assert "vocab size:" in caplog.text
    assert [ln for ln in caplog.text.splitlines() if "test unk rate:" in ln] == miss_unk
    assert json.loads((first / "meta.json").read_text())["key"] == processed_key(cfg)


def test_key_ignores_parsed_dir_but_not_split(tmp_path):
    cfg = make_cfg(tmp_path)
    key = processed_key(cfg)
    other_dir = {**cfg, "parser": {**cfg["parser"], "parsed_dir": "elsewhere"}}
    assert processed_key(other_dir) == key
    other_split = {**cfg, "split": {**cfg["split"], "train_ratio": 0.6, "test_ratio": 0.3}}
    assert processed_key(other_split) != key


def edit_raw(cfg: dict, name: str, old: str, new: str, *, last: bool = False) -> None:
    path = Path(cfg["dataset"]["raw_dir"]) / name
    text = path.read_text(encoding="utf-8")
    assert old in text
    if last:
        head, _, tail = text.rpartition(old)
        text = head + new + tail
    else:
        text = text.replace(old, new, 1)
    path.write_text(text, encoding="utf-8")


def test_key_changes_with_data_code(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    key = processed_key(cfg)
    monkeypatch.setattr(prepare_mod, "source_hash", lambda: "other-code")
    assert processed_key(cfg) != key


def test_key_changes_with_drain3_version(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    key = processed_key(cfg)
    monkeypatch.setattr(prepare_mod, "drain3_version", lambda: "0.0.0-other")
    assert processed_key(cfg) != key


def test_key_changes_when_raw_log_edited_or_truncated(tmp_path):
    cfg = make_cfg(tmp_path)
    key = processed_key(cfg)
    edit_raw(cfg, "HDFS.log", "size 5", "size 6")  # same length, different bytes
    edited = processed_key(cfg)
    assert edited != key
    log = Path(cfg["dataset"]["raw_dir"]) / "HDFS.log"
    lines = log.read_text(encoding="utf-8").splitlines(keepends=True)
    log.write_text("".join(lines[:-1]), encoding="utf-8")
    assert processed_key(cfg) != edited


def test_key_changes_when_label_file_edited(tmp_path):
    cfg = make_cfg(tmp_path)
    key = processed_key(cfg)
    edit_raw(cfg, "anomaly_label.csv", "blk_0,Normal", "blk_0,Anomaly")
    assert processed_key(cfg) != key


def test_key_ignores_lines_past_max_lines(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg["dataset"]["max_lines"] = 2
    key = processed_key(cfg)
    edit_raw(cfg, "HDFS.log", "file x", "file y", last=True)  # past line 2
    assert processed_key(cfg) == key
    log = Path(cfg["dataset"]["raw_dir"]) / "HDFS.log"
    with open(log, "a", encoding="utf-8") as f:
        f.write("081109 230000 1 INFO dfs.DataNode: extra line blk_0\n")
    assert processed_key(cfg) == key
    lines = log.read_text(encoding="utf-8").splitlines(keepends=True)
    log.write_text("".join(lines[:3]), encoding="utf-8")  # truncated, first 2 lines kept
    assert processed_key(cfg) == key


def test_key_changes_when_line_within_max_lines_edited(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg["dataset"]["max_lines"] = 2
    key = processed_key(cfg)
    log = Path(cfg["dataset"]["raw_dir"]) / "HDFS.log"
    lines = log.read_text(encoding="utf-8").splitlines(keepends=True)
    lines[1] = lines[1].replace("INFO", "WARN", 1)
    log.write_text("".join(lines), encoding="utf-8")
    assert processed_key(cfg) != key


def test_stale_cache_not_reused_after_raw_change(tmp_path, caplog):
    cfg = make_cfg(tmp_path)
    logger = logging.getLogger("test_prepare_stale")
    caplog.set_level(logging.INFO, logger="test_prepare_stale")
    first = prepare(cfg, logger)
    edit_raw(cfg, "HDFS.log", "size 5", "size 6")
    caplog.clear()
    second = prepare(cfg, logger)
    assert second != first
    assert "processed cache: miss" in caplog.text


def test_hit_stats_come_from_files_not_meta(tmp_path, caplog):
    cfg = make_cfg(tmp_path)
    logger = logging.getLogger("test_prepare_meta")
    caplog.set_level(logging.INFO, logger="test_prepare_meta")
    out = prepare(cfg, logger)
    real_vocab = Vocab.from_json(out / "vocab.json").size

    meta_path = out / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["stats"].update({"vocab size": 999, "parse cache": "miss", "groups train": 999})
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    caplog.clear()
    prepare(cfg, logger)
    assert f"vocab size: {real_vocab}" in caplog.text
    assert "999" not in caplog.text
    assert "parse cache: not run" in caplog.text
    assert "parse cache: miss" not in caplog.text
