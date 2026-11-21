import logging
from pathlib import Path

import pandas as pd
import pytest
from omegaconf import OmegaConf
from test_prepare import make_cfg

from cascadelad.data import fit_on_compare
from cascadelad.data.fit_on_compare import compare_fit_on, prefix_mismatches, run_compare
from cascadelad.data.parsing import DrainParser
from cascadelad.data.vocab import Vocab

ROOT = Path(__file__).resolve().parents[1]
SHARED_KEYS = (
    "loader",
    "line_regex",
    "timestamp_format",
    "block_regex",
    "label_column",
    "block_column",
    "label_map",
)


def parser_cfg() -> dict:
    return OmegaConf.to_container(OmegaConf.load(ROOT / "configs" / "parser" / "drain.yaml"))


CONTENTS = [
    "user alice logged in now",
    "Received block blk_1 of size 5 from /10.1.1.1",
    "user bob logged in now",
    "Deleting block blk_1 file /mnt/x",
    "user carol logged in now",
    "Received block blk_2 of size 7 from /10.1.1.2",
    "something else entirely new here",
    "user dave logged in now",
    "Deleting block blk_2 file /mnt/y",
    "last message of the stream ok",
]


def test_prefix_mismatches_zero_on_stream():
    assert prefix_mismatches(CONTENTS, parser_cfg(), 0.7) == 0


def test_prefix_mismatches_counts_differing_ids(monkeypatch):
    real = DrainParser.add

    def shifted(self, contents):
        out = real(self, contents)
        # a parser that saw the longer input numbers its ids differently
        return [(f"X{i}", t) for i, t in out] if len(out) == len(CONTENTS) else out

    monkeypatch.setattr(fit_on_compare.DrainParser, "add", shifted)
    assert prefix_mismatches(CONTENTS, parser_cfg(), 0.7) == 7


def test_hdfs_2k_cfg_matches_hdfs_regexes():
    load = lambda n: OmegaConf.to_container(  # noqa: E731
        OmegaConf.load(ROOT / "configs" / "dataset" / f"{n}.yaml")
    )
    full, small = load("hdfs"), load("hdfs_2k")
    for key in SHARED_KEYS:
        assert small[key] == full[key], key
    assert small["max_lines"] is None
    label = (ROOT / "data" / "raw" / "hdfs_2k" / small["label_file"]).resolve().as_posix()
    assert label.endswith("data/raw/hdfs/preprocessed/anomaly_label.csv")


# --- compare_fit_on (S2-6) -------------------------------------------------------------------


def write_split(
    out: Path, vocab_tokens: list[str], test_rows: list[tuple[str, list[int], list[int]]]
) -> Path:
    out.mkdir(parents=True)
    token_to_id = {"<PAD>": 0, "<UNK>": 1, **{t: i for i, t in enumerate(vocab_tokens, start=2)}}
    Vocab(token_to_id, "<PAD>", "<UNK>").to_json(out / "vocab.json")
    pd.DataFrame(
        {
            "group_id": [r[0] for r in test_rows],
            "event_ids": [r[1] for r in test_rows],
            "line_pos": [r[2] for r in test_rows],
            "line_labels": [[0] * len(r[1]) for r in test_rows],
            "timestamp": pd.Timestamp("2020-01-01"),
            "label": 0,
        }
    ).to_parquet(out / "test.parquet", index=False)
    return out


def test_compare_counts_unk_train_invocab_stream(tmp_path):
    # stream vocab knows E1,E2 (ids 2,3); train vocab only E1 (id 2)
    stream = write_split(
        tmp_path / "s", ["E1", "E2"], [("g1", [2, 3], [5, 6]), ("g2", [3], [9]), ("g3", [1], [11])]
    )
    train = write_split(
        tmp_path / "t", ["E1"], [("g1", [2, 1], [5, 6]), ("g2", [1], [9]), ("g3", [1], [11])]
    )
    res = compare_fit_on(stream, train)
    assert res == {
        "lines": 4,
        "joined": 4,
        "unk_stream": 1,
        "unk_train": 3,
        "unk_train_invocab_stream": 2,
    }


def test_compare_raises_on_different_groups(tmp_path):
    stream = write_split(tmp_path / "s", ["E1"], [("g1", [2], [0])])
    train = write_split(tmp_path / "t", ["E1"], [("g2", [2], [0])])
    with pytest.raises(ValueError, match="group"):
        compare_fit_on(stream, train)


def test_compare_raises_on_different_positions(tmp_path):
    stream = write_split(tmp_path / "s", ["E1"], [("g1", [2, 2], [0, 1])])
    train = write_split(tmp_path / "t", ["E1"], [("g1", [2, 2], [0, 2])])
    with pytest.raises(ValueError, match="one-to-one"):
        compare_fit_on(stream, train)


def test_compare_raises_on_duplicate_keys(tmp_path):
    stream = write_split(tmp_path / "s", ["E1"], [("g1", [2, 2], [4, 4])])
    train = write_split(tmp_path / "t", ["E1"], [("g1", [2, 2], [4, 4])])
    with pytest.raises(ValueError, match="duplicate"):
        compare_fit_on(stream, train)


def test_run_compare_end_to_end(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg["compare"] = {"prefix_ratio": 0.7}
    res = run_compare(cfg, logging.getLogger("t_compare"))
    assert res["prefix_mismatches"] == 0
    assert res["joined"] == res["lines"] > 0
    assert res["unk_stream"] >= 1  # blk_9's template is absent from train
    assert res["unk_train_invocab_stream"] == 0
