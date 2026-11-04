import hashlib
import os
import zipfile
from pathlib import Path

import pytest

from cascadelad.data.download import fetch_dataset


def make_zip(path: Path, members: dict[str, str]) -> str:
    with zipfile.ZipFile(path, "w") as zf:
        for name, text in members.items():
            zf.writestr(name, text)
    return hashlib.md5(path.read_bytes()).hexdigest()


def make_cfg(tmp_path: Path, archive: Path, md5: str) -> dict:
    return {
        "raw_dir": str(tmp_path / "raw"),
        "log_file": "HDFS.log",
        "label_file": "preprocessed/anomaly_label.csv",
        "zenodo": {"filename": "x.zip", "url": archive.as_uri(), "md5": md5},
        "download": {"chunk_bytes": 16},
    }


GOOD = {"HDFS.log": "line\n", "preprocessed/anomaly_label.csv": "BlockId,Label\n"}


def test_correct_md5_extracts_files(tmp_path):
    archive = tmp_path / "src.zip"
    cfg = make_cfg(tmp_path, archive, make_zip(archive, GOOD))
    raw = fetch_dataset(cfg)
    assert (raw / "HDFS.log").read_text() == "line\n"
    assert (raw / "preprocessed" / "anomaly_label.csv").exists()
    assert not list(raw.glob("*.part"))


def test_wrong_md5_raises_and_leaves_nothing(tmp_path):
    archive = tmp_path / "src.zip"
    make_zip(archive, GOOD)
    cfg = make_cfg(tmp_path, archive, "0" * 32)
    with pytest.raises(ValueError, match="md5 mismatch"):
        fetch_dataset(cfg)
    raw = Path(cfg["raw_dir"])
    assert list(raw.iterdir()) == []


def test_second_call_is_noop(tmp_path):
    archive = tmp_path / "src.zip"
    cfg = make_cfg(tmp_path, archive, make_zip(archive, GOOD))
    raw = fetch_dataset(cfg)
    log = raw / "HDFS.log"
    os.utime(log, (1_000_000, 1_000_000))
    archive.unlink()  # a re-download would now fail
    fetch_dataset(cfg)
    assert log.stat().st_mtime == 1_000_000


def test_unsafe_member_raises(tmp_path):
    archive = tmp_path / "src.zip"
    cfg = make_cfg(tmp_path, archive, make_zip(archive, {**GOOD, "../evil": "x"}))
    with pytest.raises(ValueError, match="unsafe path"):
        fetch_dataset(cfg)
    assert not (tmp_path / "evil").exists()
