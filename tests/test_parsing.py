from pathlib import Path

import pytest
from omegaconf import OmegaConf

from cascadelad.data.parsing import DrainParser, grouping_accuracy

ROOT = Path(__file__).resolve().parents[1]


def parser_cfg() -> dict:
    cfg = OmegaConf.load(ROOT / "configs" / "parser" / "drain.yaml")
    return OmegaConf.to_container(cfg, resolve=True)


TRAIN = [
    "Received block blk_-100 of size 67108864 from /10.250.19.102",
    "Received block blk_200 of size 5 from /10.251.1.2",
    "Deleting block blk_300 file /mnt/hadoop/dfs/data/current/blk_300",
]


def test_same_template_despite_block_ip_number():
    p = DrainParser(parser_cfg()).fit(TRAIN)
    a, b = p.transform(TRAIN[:2])
    assert a == b


def test_unseen_content_maps_to_unk():
    cfg = parser_cfg()
    p = DrainParser(cfg).fit(TRAIN)
    assert p.transform(["totally different words appear here now"]) == [cfg["unk_token"]]


def test_transform_does_not_learn():
    p = DrainParser(parser_cfg()).fit(TRAIN)
    before = len(p.templates())
    p.transform(["brand new message never seen before ok", TRAIN[0]])
    assert len(p.templates()) == before


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        DrainParser(parser_cfg()).transform(TRAIN)


def test_grouping_accuracy_perfect_and_relabelled():
    assert grouping_accuracy(["a", "a", "b"], ["x", "x", "y"]) == 1.0
    assert grouping_accuracy(["b", "b", "a"], ["x", "x", "y"]) == 1.0


def test_grouping_accuracy_merged_cluster():
    # truth {0,1} and {2,3}; pred merges all: no predicted group equals a true group -> 0
    assert grouping_accuracy(["a"] * 4, ["x", "x", "y", "y"]) == 0.0
    # pred splits {0,1} right, merges {2,3} with 4: 2 of 5 lines correct
    assert grouping_accuracy(["a", "a", "b", "b", "b"], ["x", "x", "y", "y", "z"]) == 2 / 5


def test_grouping_accuracy_errors():
    with pytest.raises(ValueError):
        grouping_accuracy(["a"], ["x", "y"])
    with pytest.raises(ValueError):
        grouping_accuracy([], [])
