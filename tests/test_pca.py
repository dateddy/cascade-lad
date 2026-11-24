import numpy as np
import pandas as pd
import pytest

from cascadelad.detectors import get_detector

N_FEATURES = 6  # id 0 = <PAD>, id 1 = <UNK>
CFG = {
    "name": "pca",
    "decision": "threshold",
    "term_weighting": "tfidf",
    "normalization": "zero_mean",
    "n_components": 0.95,
}


def windows(sequences, start=0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "group_id": [f"g{start + i}" for i in range(len(sequences))],
            "event_ids": sequences,
            "timestamp": pd.date_range("2026-01-01", periods=len(sequences), freq="min"),
        }
    )


def normal_train(n=60) -> pd.DataFrame:
    """Counts of ids 2 and 3 move together; k = 0 windows lack them, so their idf is above 0."""
    rng = np.random.default_rng(0)
    seqs = []
    for _ in range(n):
        k = int(rng.integers(0, 5))
        seqs.append([2] * k + [3] * k + [4])
    return windows(seqs)


def make(**over):
    return get_detector({**CFG, **over}, n_features=N_FEATURES)


def test_pca_deterministic():
    # S5-5: two independent fits give bit-identical scores.
    train, test = normal_train(), windows([[2, 3, 4], [2, 2, 5], [1, 4]], start=100)
    runs = []
    for _ in range(2):
        det = make()
        det.fit(train)
        runs.append(det.score(test))
    assert np.array_equal(runs[0], runs[1])


def test_pca_train_point_scores_below_outlier():
    train = normal_train()
    det = make()
    det.fit(train)
    inlier = det.score(windows([[2, 2, 3, 3, 4]], start=100))[0]
    outlier = det.score(windows([[2, 2, 2, 5, 5]], start=101))[0]  # breaks the 2-3 pairing
    assert outlier > inlier


def test_pca_unk_raises_score():
    # <UNK> (id 1) never occurs in train: df = 0 gives it the largest idf and it lies
    # outside the normal subspace.
    train = normal_train()
    det = make()
    det.fit(train)
    known = det.score(windows([[2, 3, 4]], start=100))[0]
    with_unk = det.score(windows([[2, 3, 4, 1]], start=101))[0]
    assert with_unk > known


def test_pca_tfidf_none_matches_counts():
    # No weighting: score = squared residual of the centered count vector after projecting on
    # the first principal component, computed by hand.
    train = normal_train()
    test = windows([[2, 3, 3, 5], [4, 4]], start=100)
    det = make(term_weighting="none", n_components=1)
    det.fit(train)

    def counts(df):
        return np.array(
            [np.bincount(ids, minlength=N_FEATURES) for ids in df["event_ids"]], dtype=float
        )

    x = counts(train)
    mean = x.mean(axis=0)
    _, _, vt = np.linalg.svd(x - mean, full_matrices=False)
    p = vt[:1].T
    y = counts(test) - mean
    expected = ((y - y @ p @ p.T) ** 2).sum(axis=1)
    assert np.allclose(det.score(test), expected)


def test_pca_rejects_out_of_range_id():
    det = make()
    with pytest.raises(ValueError, match="id"):
        det.fit(windows([[1, N_FEATURES]]))
    det.fit(normal_train())
    with pytest.raises(ValueError, match="id"):
        det.score(windows([[2, N_FEATURES + 3]]))


def test_pca_n_components_int():
    det = make(n_components=1)
    det.fit(normal_train())
    assert det.n_components_ == 1


def test_pca_float_components_keep_explained_variance():
    det = make(n_components=0.95)
    det.fit(normal_train())
    assert 1 <= det.n_components_ < N_FEATURES


def test_pca_rejects_empty_train():
    with pytest.raises(ValueError, match="empty"):
        make().fit(windows([]))


def test_pca_score_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        make().score(windows([[2, 3]]))


def test_pca_constant_train_scores_zero_for_same_window():
    # All train windows identical: zero variance, no components; the train point scores 0.
    det = make()
    det.fit(windows([[2, 3, 4]] * 10))
    assert det.score(windows([[2, 3, 4]], start=100))[0] == pytest.approx(0.0)
