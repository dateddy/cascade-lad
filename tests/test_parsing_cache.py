import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from cascadelad.data import parsing
from cascadelad.data.parsing import DrainParser, cache_key, parse_lines, source_hash

ROOT = Path(__file__).resolve().parents[1]
LINES = [
    "Received block blk_-100 of size 5 from /10.1.1.1",
    "Received block blk_200 of size 7 from /10.1.1.2",
    "NA",
    "null",
]


def cfg_for(tmp_path: Path, **overrides) -> dict:
    cfg = OmegaConf.load(ROOT / "configs" / "parser" / "drain.yaml")
    cfg.parsed_dir = str(tmp_path / "parsed")
    cfg.update(overrides)
    return OmegaConf.to_container(cfg, resolve=True)


MASK = [True, True, False, False]


def test_miss_then_hit_without_fitting(tmp_path, monkeypatch):
    cfg = cfg_for(tmp_path)
    d1, hit1 = parse_lines(LINES, cfg)
    assert hit1 is False
    (cache_dir,) = (tmp_path / "parsed").iterdir()
    assert sorted(p.name for p in cache_dir.iterdir()) == [
        "lines.parquet",
        "meta.json",
        "templates.json",
    ]

    def boom(self, contents):
        raise AssertionError("parser used on cache hit")

    monkeypatch.setattr(DrainParser, "add", boom)
    monkeypatch.setattr(DrainParser, "fit", boom)
    d2, hit2 = parse_lines(LINES, cfg)
    assert hit2 is True
    assert d1.equals(d2)


def test_output_aligned_with_input_and_keeps_duplicates(tmp_path):
    lines = [LINES[0], LINES[1], LINES[0]]
    d, _ = parse_lines(lines, cfg_for(tmp_path))
    assert list(d.columns) == ["template_id", "template"] and len(d) == 3
    assert d["template_id"].iloc[0] == d["template_id"].iloc[2]


def test_fit_mask_rules(tmp_path):
    stream = cfg_for(tmp_path)
    train = cfg_for(tmp_path, fit_on="train")
    with pytest.raises(ValueError, match="fit_mask"):
        parse_lines(LINES, stream, MASK)
    with pytest.raises(ValueError, match="fit_mask"):
        parse_lines(LINES, train)
    with pytest.raises(ValueError, match="length"):
        parse_lines(LINES, train, MASK[:2])
    with pytest.raises(ValueError, match="at least one"):
        parse_lines(LINES, train, [False] * 4)
    with pytest.raises(ValueError, match="fit_on"):
        parse_lines(LINES, cfg_for(tmp_path, fit_on="nope"))


def test_train_mode_unseen_is_unk_and_cluster_count_fixed(tmp_path):
    cfg = cfg_for(tmp_path, fit_on="train")
    lines = [*LINES[:2], "something else entirely new here"]
    d, _ = parse_lines(lines, cfg, [True, True, False])
    assert d["template_id"].iloc[-1] == cfg["unk_token"] == d["template"].iloc[-1]
    assert d["template_id"].iloc[0].startswith("E")
    (cache_dir,) = (tmp_path / "parsed").iterdir()
    meta = json.loads((cache_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["n_clusters_fit"] == meta["n_clusters_after_apply"] == 1
    assert meta["n_unk"] == 1


def test_key_changes_with_fit_on_and_each_drain_param(tmp_path):
    base = cfg_for(tmp_path)
    key = cache_key(base, LINES)
    changes = {
        "fit_on": "train",
        "depth": 5,
        "sim_th": 0.9,
        "max_children": 50,
        "max_clusters": 10,
        "parametrize_numeric_tokens": False,
        "extra_delimiters": ["_"],
        "masking": [],
        "match_strategy": "never",
        "unk_token": "<X>",
    }
    for k, v in changes.items():
        other = cache_key(cfg_for(tmp_path, **{k: v}), LINES, MASK if k == "fit_on" else None)
        assert other != key, k
    assert cache_key(base, LINES[:2]) != key
    assert cache_key(cfg_for(tmp_path, parsed_dir="elsewhere"), LINES) == key


def test_train_key_changes_with_fit_mask(tmp_path):
    cfg = cfg_for(tmp_path, fit_on="train")
    assert cache_key(cfg, LINES, MASK) != cache_key(cfg, LINES, [True, True, True, False])


def test_stream_key_ignores_fit_mask(tmp_path):
    cfg = cfg_for(tmp_path)
    assert cache_key(cfg, LINES, MASK) == cache_key(cfg, LINES, None)


def test_source_hash_tracks_nested_py_files(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "sub" / "b.py").write_text("y = 2\n", encoding="utf-8")
    base = source_hash(tmp_path)
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")
    assert source_hash(tmp_path) == base
    (tmp_path / "sub" / "b.py").write_text("y = 3\n", encoding="utf-8")
    assert source_hash(tmp_path) != base


def test_key_changes_with_code_hash(tmp_path, monkeypatch):
    cfg = cfg_for(tmp_path)
    key = cache_key(cfg, LINES)
    monkeypatch.setattr(parsing, "source_hash", lambda: "other-code")
    assert cache_key(cfg, LINES) != key


def test_key_changes_with_drain3_version(tmp_path, monkeypatch):
    cfg = cfg_for(tmp_path)
    key = cache_key(cfg, LINES)
    monkeypatch.setattr(parsing, "drain3_version", lambda: "0.0.0-other")
    assert cache_key(cfg, LINES) != key


def test_key_content_boundary_is_unambiguous(tmp_path):
    cfg = cfg_for(tmp_path)
    assert cache_key(cfg, ["a", "b"]) != cache_key(cfg, ["a\nb"])
    assert cache_key(cfg, ["ab", "c"]) != cache_key(cfg, ["a", "bc"])


def test_na_and_null_round_trip_as_strings(tmp_path):
    cfg = cfg_for(tmp_path)
    d1, _ = parse_lines(LINES, cfg)
    d2, hit = parse_lines(LINES, cfg)
    assert hit is True
    assert d1.equals(d2)
    assert d2["template_id"].map(type).eq(str).all() and d2["template"].map(type).eq(str).all()
    assert d2["template"].iloc[2] == "NA" and d2["template"].iloc[3] == "null"


@pytest.fixture(autouse=True)
def _no_stray_tmp(tmp_path):
    yield
    assert not list((tmp_path / "parsed").glob("*.tmp")) if (tmp_path / "parsed").exists() else True
