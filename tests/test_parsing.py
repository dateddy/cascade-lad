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


EVOLVE = ["user alice logged in now", "user bob logged in now", "user carol logged in now"]


def test_template_ids_are_E_strings():
    out = DrainParser(parser_cfg()).add(TRAIN)
    assert all(isinstance(i, str) and i.startswith("E") and i[1:].isdigit() for i, _ in out)
    assert all(isinstance(t, str) for _, t in out)


def test_stream_ids_fixed_when_added():
    p = DrainParser(parser_cfg())
    first = p.add(EVOLVE[:1])[0][0]
    p.add(EVOLVE[1:] + TRAIN)
    assert p.add(EVOLVE[:1])[0][0] == first


def test_stream_template_is_template_at_that_moment():
    out = DrainParser(parser_cfg()).add(EVOLVE)
    assert out[0][0] == out[1][0] == out[2][0]
    assert out[0][1] == EVOLVE[0]  # exact text when first seen
    assert out[1][1] == "user <*> logged in now"  # generalised once bob arrives
    assert out[0][1] != out[2][1]  # earlier line keeps its older text


def test_stream_prefix_stable():
    cfg = parser_cfg()
    stream = [*TRAIN, *EVOLVE, "another new kind of message appears"]
    full = DrainParser(cfg).add(stream)
    prefix = DrainParser(cfg).add(stream[:4])
    assert [i for i, _ in prefix] == [i for i, _ in full[:4]]


def test_add_keeps_duplicates():
    p = DrainParser(parser_cfg())
    out = p.add([TRAIN[0], TRAIN[0]])
    assert len(out) == 2 and out[0][0] == out[1][0]


def test_train_cluster_count_unchanged_after_apply():
    p = DrainParser(parser_cfg()).fit(TRAIN)
    n = p.n_clusters
    assert n == len(p.templates())
    p.transform(["brand new message never seen before ok", *TRAIN, *EVOLVE])
    assert p.n_clusters == n


def test_train_unseen_is_unk_string():
    cfg = parser_cfg()
    p = DrainParser(cfg).fit(TRAIN)
    out = p.transform(["totally different words appear here now", TRAIN[0]])
    assert out[0] == cfg["unk_token"] == "<UNK>"
    assert out[1] != cfg["unk_token"]
