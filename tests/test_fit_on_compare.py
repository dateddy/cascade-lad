from pathlib import Path

from omegaconf import OmegaConf

from cascadelad.data import fit_on_compare
from cascadelad.data.fit_on_compare import prefix_mismatches
from cascadelad.data.parsing import DrainParser

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
