import pytest

from cascadelad.data.vocab import Vocab

CFG = {"pad_token": "<PAD>", "unk_token": "<UNK>", "min_count": 1}
TRAIN = [["E1", "E2", "E1"], ["E1", "E3"], ["E2"]]  # counts: E1=3, E2=2, E3=1


def test_ids_pad_unk_then_by_frequency():
    v = Vocab.build(TRAIN, CFG)
    assert v.token_to_id == {"<PAD>": 0, "<UNK>": 1, "E1": 2, "E2": 3, "E3": 4}
    assert v.size == 5


def test_count_ties_broken_by_token():
    v = Vocab.build([["b", "a"]], CFG)
    assert v.token_to_id["a"] < v.token_to_id["b"]


def test_unseen_token_maps_to_unk():
    v = Vocab.build(TRAIN, CFG)
    assert v.encode(["E1", "E9", "E3"]) == [2, 1, 4]


def test_parser_unk_token_never_gets_its_own_id():
    v = Vocab.build([["E1", "<UNK>", "<UNK>"]], CFG)
    assert v.size == 3
    assert v.encode(["<UNK>"]) == [v.unk_id]


def test_built_only_from_given_sequences():
    v = Vocab.build([["E1"]], CFG)
    assert "E2" not in v.token_to_id
    assert v.encode(["E2"]) == [v.unk_id]


def test_min_count_filters_rare_tokens():
    v = Vocab.build(TRAIN, {**CFG, "min_count": 2})
    assert "E3" not in v.token_to_id
    assert v.encode(["E3"]) == [v.unk_id]


def test_deterministic():
    assert Vocab.build(TRAIN, CFG).token_to_id == Vocab.build(TRAIN[::-1], CFG).token_to_id


def test_json_round_trip(tmp_path):
    v = Vocab.build(TRAIN, CFG)
    v.to_json(tmp_path / "vocab.json")
    w = Vocab.from_json(tmp_path / "vocab.json")
    assert w.token_to_id == v.token_to_id
    assert (w.pad_id, w.unk_id) == (v.pad_id, v.unk_id)
    assert w.encode(["E1", "zzz"]) == v.encode(["E1", "zzz"])


def test_unk_rate_hand_made():
    v = Vocab.build(TRAIN, CFG)
    # 5 events: E1, E9, E2, E8, <UNK> -> 3 unk
    assert v.unk_rate([["E1", "E9"], ["E2", "E8", "<UNK>"]]) == pytest.approx(3 / 5)


def test_unk_rate_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        Vocab.build(TRAIN, CFG).unk_rate([])
