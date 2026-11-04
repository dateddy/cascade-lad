import pandas as pd
import pytest

from cascadelad.data.splitting import chronological_split

CFG = {"train_ratio": 0.5, "val_ratio": 0.25, "test_ratio": 0.25}


def make_bounds(rows: list[tuple[str, int, int, int]]) -> pd.DataFrame:
    """rows: (group_id, start_second, end_second, label)"""
    base = pd.Timestamp("2020-01-01")
    return pd.DataFrame(
        {
            "group_id": [r[0] for r in rows],
            "start": [base + pd.Timedelta(seconds=r[1]) for r in rows],
            "end": [base + pd.Timedelta(seconds=r[2]) for r in rows],
            "label": [r[3] for r in rows],
        }
    )


# 8 blocks, starts 0..7. Candidates: train = g0..g3, val = g4..g5, test = g6..g7.
# t1 = start(g4) = 4s, t2 = start(g6) = 6s.
#   g2 ends at 5s (>= t1) -> straddles train/val -> purged from train
#   g5 ends at 6s (>= t2) -> straddles val/test  -> purged from val
#   g1 is anomalous and fully inside train       -> dropped from train, not purged
BOUNDS = make_bounds(
    [
        ("g0", 0, 1, 0),
        ("g1", 1, 2, 1),
        ("g2", 2, 5, 0),
        ("g3", 3, 3, 0),
        ("g4", 4, 5, 0),
        ("g5", 5, 6, 1),
        ("g6", 6, 7, 0),
        ("g7", 7, 9, 1),
    ]
)


def test_split_membership_and_counts():
    r = chronological_split(BOUNDS, CFG)
    assert r.train == ["g0", "g3"]
    assert r.val == ["g4"]
    assert r.test == ["g6", "g7"]
    assert r.n_purged == {"train": 1, "val": 1, "test": 0}
    assert r.n_train_anomalous_dropped == 1


def test_no_temporal_leakage():
    r = chronological_split(BOUNDS, CFG)
    b = BOUNDS.set_index("group_id")
    assert b.loc[r.train, "end"].max() < b.loc[r.val, "start"].min()
    assert b.loc[r.val, "end"].max() < b.loc[r.test, "start"].min()
    assert b.loc[r.train, "end"].max() < b.loc[r.test, "start"].min()


def test_train_has_no_anomalous_block_but_val_test_keep_both_labels():
    r = chronological_split(BOUNDS, CFG)
    label = BOUNDS.set_index("group_id")["label"]
    assert (label[r.train] == 0).all()
    assert label[r.test].tolist() == [0, 1]


def test_groups_are_disjoint_and_cover_only_input():
    r = chronological_split(BOUNDS, CFG)
    all_ids = r.train + r.val + r.test
    assert len(all_ids) == len(set(all_ids))
    assert set(all_ids) <= set(BOUNDS["group_id"])


def test_deterministic_and_independent_of_input_order():
    shuffled = BOUNDS.sample(frac=1, random_state=0).reset_index(drop=True)
    assert chronological_split(shuffled, CFG) == chronological_split(BOUNDS, CFG)


def test_start_ties_broken_by_group_id():
    tied = make_bounds([(g, 0, 1, 0) for g in ("d", "c", "b", "a")])
    r = chronological_split(tied, {"train_ratio": 0.5, "val_ratio": 0.25, "test_ratio": 0.25})
    # all blocks start at 0 so t1 = t2 = 0: every candidate ends >= t1 and is purged
    assert r.n_purged["train"] == 2 and r.n_purged["val"] == 1
    assert r.test == ["d"]


@pytest.mark.parametrize(
    "cfg",
    [
        {"train_ratio": 0.5, "val_ratio": 0.5, "test_ratio": 0.5},
        {"train_ratio": 0.6, "val_ratio": 0.1, "test_ratio": 0.1},
    ],
)
def test_ratios_must_sum_to_one(cfg):
    with pytest.raises(ValueError, match="sum to 1"):
        chronological_split(BOUNDS, cfg)


def test_too_few_blocks_for_three_splits_raises():
    with pytest.raises(ValueError, match="too few"):
        chronological_split(BOUNDS.iloc[:2], CFG)
