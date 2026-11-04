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

    def boom(*args, **kwargs):
        raise AssertionError("loader called on processed cache hit")

    monkeypatch.setattr(prepare_mod, "get_loader", boom)
    caplog.clear()
    assert prepare(cfg, logger) == first
    assert "processed cache: hit" in caplog.text
    assert "vocab size:" in caplog.text
    assert json.loads((first / "meta.json").read_text())["key"] == processed_key(cfg)


def test_key_ignores_parsed_dir_but_not_split(tmp_path):
    cfg = make_cfg(tmp_path)
    key = processed_key(cfg)
    other_dir = {**cfg, "parser": {**cfg["parser"], "parsed_dir": "elsewhere"}}
    assert processed_key(other_dir) == key
    other_split = {**cfg, "split": {**cfg["split"], "train_ratio": 0.6, "test_ratio": 0.3}}
    assert processed_key(other_split) != key
