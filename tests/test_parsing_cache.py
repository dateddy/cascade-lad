from pathlib import Path

import pytest
from omegaconf import OmegaConf

from cascadelad.data.parsing import DrainParser, cache_key, parse_cached

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


def test_miss_then_hit_without_fitting(tmp_path, monkeypatch):
    cfg = cfg_for(tmp_path)
    a1, t1, hit1 = parse_cached(LINES, LINES, cfg)
    assert hit1 is False
    (cache_dir,) = (tmp_path / "parsed").iterdir()
    assert sorted(p.name for p in cache_dir.iterdir()) == [
        "assignments.csv",
        "meta.json",
        "templates.json",
    ]

    def boom(self, contents):
        raise AssertionError("fit called on cache hit")

    monkeypatch.setattr(DrainParser, "fit", boom)
    a2, t2, hit2 = parse_cached(LINES, LINES, cfg)
    assert hit2 is True
    assert a1.equals(a2) and t1 == t2


def test_key_depends_on_cfg_and_fit_contents_not_parsed_dir(tmp_path):
    base = cfg_for(tmp_path)
    key = cache_key(base, LINES, LINES)
    assert cache_key(cfg_for(tmp_path, sim_th=0.9), LINES, LINES) != key
    assert cache_key(base, LINES[:2], LINES) != key
    assert cache_key(cfg_for(tmp_path, parsed_dir="elsewhere"), LINES, LINES) == key


def test_key_section_boundary_is_unambiguous(tmp_path):
    cfg = cfg_for(tmp_path)
    assert cache_key(cfg, ["a", "b"], ["c"]) != cache_key(cfg, ["a"], ["b", "c"])


def test_na_and_null_round_trip_as_strings(tmp_path):
    cfg = cfg_for(tmp_path)
    parse_cached(LINES, LINES, cfg)
    a, _, hit = parse_cached(LINES, LINES, cfg)
    assert hit is True
    assert list(a["content"]) == LINES
    assert a["template_id"].map(type).eq(str).all()


def test_apply_unseen_is_unk_and_counted(tmp_path):
    cfg = cfg_for(tmp_path)
    a, _, _ = parse_cached(LINES[:2], [*LINES[:2], "something else entirely new here"], cfg)
    assert a["template_id"].iloc[-1] == cfg["unk_token"]


@pytest.fixture(autouse=True)
def _no_stray_tmp(tmp_path):
    yield
    assert not list((tmp_path / "parsed").glob("*.tmp")) if (tmp_path / "parsed").exists() else True
