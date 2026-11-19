import pandas as pd
import pytest

from cascadelad.data.splitting import chronological_split, random_split, split_groups

RATIOS = {"train_ratio": 0.5, "val_ratio": 0.25, "test_ratio": 0.25}
FLAGS = {"drop_train_anomalies": True, "drop_val_anomalies": True, "purge": True, "seed": 0}
CFG = {"mode": "chronological", **RATIOS, **FLAGS}


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
    assert r.n_val_anomalous_dropped == 0  # g5 (anomalous) was purged first
    assert r.n_train_anomalous_kept == 0


def test_no_temporal_leakage():
    r = chronological_split(BOUNDS, CFG)
    b = BOUNDS.set_index("group_id")
    assert b.loc[r.train, "end"].max() < b.loc[r.val, "start"].min()
    assert b.loc[r.val, "end"].max() < b.loc[r.test, "start"].min()
    assert b.loc[r.train, "end"].max() < b.loc[r.test, "start"].min()


def test_train_has_no_anomalous_block_but_val_test_keep_both_labels():
    r = chronological_split(BOUNDS, {**CFG, "drop_val_anomalies": False})
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
    r = chronological_split(tied, CFG)
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
        chronological_split(BOUNDS, {**CFG, **cfg})


def test_too_few_blocks_for_three_splits_raises():
    with pytest.raises(ValueError, match="too few"):
        chronological_split(BOUNDS.iloc[:2], CFG)


# 12 blocks without straddlers, block k spans [k, k] seconds.
# Anomalous: g01 (train), g07 (val), g10 (test).
# Ratios 0.5/0.25/0.25 -> train g00..g05, val g06..g08, test g09..g11.
CLEAN = make_bounds([(f"g{k:02d}", k, k, int(k in (1, 7, 10))) for k in range(12)])
LABEL = CLEAN.set_index("group_id")["label"]


def test_session_start_strictly_before_next_split():
    """S3-1 (session data): max(train start) < min(val start), and the same for val -> test."""
    for bounds in (BOUNDS, CLEAN):
        r = chronological_split(bounds, CFG)
        start = bounds.set_index("group_id")["start"]
        assert start[r.train].max() < start[r.val].min()
        assert start[r.val].max() < start[r.test].min()


def test_val_normal_only_by_default():
    """S3-4: val has 0 anomalies by default; test keeps its anomalies."""
    r = chronological_split(CLEAN, CFG)
    assert r.val == ["g06", "g08"]
    assert r.n_val_anomalous_dropped == 1
    assert (LABEL[r.val] == 0).all()
    assert LABEL[r.test].sum() == 1


def test_drop_val_false_keeps_val_anomalies():
    r = chronological_split(CLEAN, {**CFG, "drop_val_anomalies": False})
    assert r.val == ["g06", "g07", "g08"]
    assert r.n_val_anomalous_dropped == 0


def test_drop_train_false_keeps_rows_and_counts_anomalies():
    """S3-3: with drop_train_anomalies=false no train group is removed; anomalies are counted."""
    r = chronological_split(CLEAN, {**CFG, "drop_train_anomalies": False})
    assert r.train == [f"g{k:02d}" for k in range(6)]
    assert r.n_train_anomalous_dropped == 0
    assert r.n_train_anomalous_kept == 1


def test_purge_false_keeps_straddlers():
    r = chronological_split(BOUNDS, {**CFG, "purge": False})
    assert r.n_purged == {"train": 0, "val": 0, "test": 0}
    assert "g2" in r.train


RANDOM = {**CFG, "mode": "random"}


def test_random_same_seed_same_split():
    """S3-2: same seed -> same split."""
    assert random_split(CLEAN, RANDOM) == random_split(CLEAN, RANDOM)


def test_random_without_seed_raises():
    """Seed is only in configs/split/random.yaml; split.mode=random on chrono fails clearly."""
    with pytest.raises(ValueError, match="split.seed"):
        random_split(CLEAN, {k: v for k, v in RANDOM.items() if k != "seed"})


def test_random_different_seed_differs():
    """S3-2: a different seed gives a different split."""
    a = random_split(CLEAN, {**RANDOM, "seed": 0})
    b = random_split(CLEAN, {**RANDOM, "seed": 1})
    assert (a.train, a.val, a.test) != (b.train, b.val, b.test)


def test_random_independent_of_input_order():
    shuffled = CLEAN.sample(frac=1, random_state=3).reset_index(drop=True)
    assert random_split(shuffled, RANDOM) == random_split(CLEAN, RANDOM)


def test_random_no_purge():
    """Random mode never purges; with both drop flags off every group lands in some split."""
    cfg = {**RANDOM, "drop_train_anomalies": False, "drop_val_anomalies": False}
    r = random_split(BOUNDS, cfg)
    assert r.n_purged == {"train": 0, "val": 0, "test": 0}
    assert sorted(r.train + r.val + r.test) == sorted(BOUNDS["group_id"])


def test_random_applies_drop_flags():
    r = random_split(CLEAN, RANDOM)
    assert (LABEL[r.train] == 0).all() and (LABEL[r.val] == 0).all()


def test_split_groups_dispatches_on_mode():
    assert split_groups(CLEAN, CFG) == chronological_split(CLEAN, CFG)
    assert split_groups(CLEAN, RANDOM) == random_split(CLEAN, RANDOM)


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="unknown split mode"):
        split_groups(CLEAN, {**CFG, "mode": "shuffled"})
