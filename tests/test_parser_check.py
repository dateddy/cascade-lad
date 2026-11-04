import logging
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from cascadelad.data.parser_check import run_check

ROOT = Path(__file__).resolve().parents[1]
CONTENTS = [
    ("Received block blk_-100 of size 5 from /10.1.1.1", "E1"),
    ("Received block blk_200 of size 7 from /10.1.1.2", "E1"),
    ("Received block blk_300 of size 9 from /10.1.1.3", "E1"),
    ("Deleting block blk_-100 file /mnt/data/blk_-100", "E2"),
    ("Deleting block blk_200 file /mnt/data/blk_200", "E2"),
    ("Deleting block blk_300 file /mnt/data/blk_300", "E2"),
]


def make_cfgs(tmp_path: Path, csv_contents=None):
    ds = OmegaConf.to_container(OmegaConf.load(ROOT / "configs" / "dataset" / "hdfs_2k.yaml"))
    ds["raw_dir"] = str(tmp_path / "raw")
    raw = Path(ds["raw_dir"])
    raw.mkdir()
    log = [f"081109 20361{i} 1{i} INFO dfs.DataNode: {c}" for i, (c, _) in enumerate(CONTENTS)]
    (raw / ds["log_file"]).write_text("\n".join(log) + "\n", encoding="utf-8")
    shown = csv_contents or CONTENTS
    rows = ["LineId,Content,EventId"]
    for i, ((c, _), (_, e)) in enumerate(zip(shown, CONTENTS, strict=True), start=1):
        rows.append(f'{i},"{c}",{e}')
    (raw / ds["structured_file"]).write_text("\n".join(rows) + "\n", encoding="utf-8")
    parser = OmegaConf.to_container(OmegaConf.load(ROOT / "configs" / "parser" / "drain.yaml"))
    parser["parsed_dir"] = str(tmp_path / "parsed")
    return ds, parser


def test_perfect_grouping_then_cache_hit(tmp_path):
    ds, parser = make_cfgs(tmp_path)
    log = logging.getLogger("t")
    m1 = run_check(ds, parser, log)
    assert m1["grouping_accuracy"] == 1.0 and m1["cache_hit"] is False
    assert m1["lines"] == 6 and m1["true_templates"] == 2
    assert run_check(ds, parser, log)["cache_hit"] is True


def test_content_mismatch_raises(tmp_path):
    bad = [*CONTENTS[:-1], ("something else", "E2")]
    ds, parser = make_cfgs(tmp_path, csv_contents=bad)
    with pytest.raises(ValueError, match="line 6"):
        run_check(ds, parser, logging.getLogger("t"))
