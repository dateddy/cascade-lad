from pathlib import Path

import pandas as pd
import pytest
from omegaconf import OmegaConf

from cascadelad.data.loaders import get_loader

LOG = """\
081109 203518 35 INFO dfs.FSNamesystem: BLOCK* allocate blk_-100 to node
081109 203519 36 INFO dfs.DataNode$PacketResponder: Received block blk_200 of size 5
081109 203520 37 WARN dfs.FSNamesystem: replicate blk_-100 and blk_200 now
081109 203521 38 INFO dfs.DataNode: heartbeat without any id
081109 203522 39 INFO dfs.DataNode: Deleting block blk_300 file x
081109 203523 40 INFO dfs.DataNode: again blk_-100
"""
LABELS = "BlockId,Label\nblk_-100,Anomaly\nblk_200,Normal\nblk_300,Normal\n"


def make_cfg(tmp_path: Path, **overrides):
    (tmp_path / "HDFS.log").write_text(LOG, encoding="utf-8")
    (tmp_path / "anomaly_label.csv").write_text(LABELS, encoding="utf-8")
    cfg = OmegaConf.load(Path(__file__).parents[1] / "configs" / "dataset" / "hdfs.yaml")
    cfg.raw_dir = str(tmp_path)
    cfg.label_file = "anomaly_label.csv"  # flat fixture layout; the real config uses preprocessed/
    cfg.update(overrides)
    return OmegaConf.to_container(cfg, resolve=True)


def load(tmp_path: Path, **overrides) -> pd.DataFrame:
    cfg = make_cfg(tmp_path, **overrides)
    return get_loader("hdfs", cfg).load()


def test_columns_header_and_timestamp(tmp_path):
    df = load(tmp_path)
    assert list(df.columns) == ["timestamp", "content", "group_id", "label"]
    assert df["content"].iloc[0] == "BLOCK* allocate blk_-100 to node"
    assert df["timestamp"].iloc[0] == pd.Timestamp("2008-11-09 20:35:18")


def test_multi_block_line_gives_two_rows_and_no_block_line_dropped(tmp_path):
    df = load(tmp_path)
    assert (df["content"] == "replicate blk_-100 and blk_200 now").sum() == 2
    assert not df["content"].str.contains("heartbeat").any()
    assert len(df) == 6


def test_labels_are_joined(tmp_path):
    df = load(tmp_path)
    by_block = df.groupby("group_id")["label"].first().to_dict()
    assert by_block == {"blk_-100": 1, "blk_200": 0, "blk_300": 0}


def test_max_lines_keeps_first_lines_only(tmp_path):
    df = load(tmp_path, max_lines=3)
    assert set(df["group_id"]) == {"blk_-100", "blk_200"}
    assert len(df) == 4


def test_unlabeled_block_raises(tmp_path):
    cfg = make_cfg(tmp_path)
    (tmp_path / "anomaly_label.csv").write_text("BlockId,Label\nblk_-100,Anomaly\n")
    with pytest.raises(ValueError, match="no label"):
        get_loader("hdfs", cfg).load()


def test_malformed_line_raises_with_line_number(tmp_path):
    cfg = make_cfg(tmp_path)
    (tmp_path / "HDFS.log").write_text(LOG + "garbage line\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 7"):
        get_loader("hdfs", cfg).load()
