"""Cross-validation splits: grouping, stratification, repeats and clear failures."""

from __future__ import annotations

import numpy as np
import pytest

from acoustic_feature_lab import DatasetError, split_indices
from acoustic_feature_lab.splitting import has_groups, repeat_seeds


def test_grouped_regression_folds_never_split_a_speaker(rng):
    groups = np.repeat([f"s{i}" for i in range(10)], 3)
    y = rng.uniform(0, 100, 30)
    folds = split_indices(y, groups, task="regression", n_splits=5, random_state=0)
    assert sorted(np.concatenate([test for _, test in folds]).tolist()) == list(range(30))
    for train, test in folds:
        assert not set(groups[train]) & set(groups[test])


def _speaker_folds(groups, seed):
    folds = split_indices(np.zeros(groups.size), groups, task="regression", n_splits=5,
                          random_state=seed)  # fmt: skip
    return [sorted({int(name[1:]) for name in groups[test]}) for _, test in folds]


def test_equal_sized_speakers_get_the_same_folds_everywhere():
    # 20 speakers x 3 recordings is the default synthetic dataset, where every
    # group ties with every other one on size.
    groups = np.repeat([f"s{i:02d}" for i in range(20)], 3)
    assert _speaker_folds(groups, 0) == [
        [0, 1, 16, 17],
        [4, 6, 9, 14],
        [3, 11, 12, 15],
        [2, 5, 7, 18],
        [8, 10, 13, 19],
    ]


def test_grouped_folds_do_not_depend_on_the_tie_order_of_argsort(monkeypatch):
    groups = np.repeat([f"s{i:02d}" for i in range(20)], 3)
    expected = _speaker_folds(groups, 0)
    real_argsort = np.argsort

    def ties_last_first(a, *args, **kwargs):
        values = np.asarray(a)
        if values.ndim != 1:
            return real_argsort(a, *args, **kwargs)
        return values.size - 1 - real_argsort(values[::-1], kind="stable")

    monkeypatch.setattr(np, "argsort", ties_last_first)
    assert _speaker_folds(groups, 0) == expected


def test_grouped_folds_put_large_speakers_first_and_balance_the_folds():
    sizes = [5, 1, 3, 3, 2, 4, 1]
    groups = np.repeat([f"g{i}" for i in range(len(sizes))], sizes)
    folds = split_indices(np.zeros(groups.size), groups, task="regression", n_splits=3,
                          random_state=4)  # fmt: skip
    assert [sorted(set(groups[test])) for _, test in folds] == [
        ["g0", "g1", "g6"],
        ["g4", "g5"],
        ["g2", "g3"],
    ]
    assert [test.size for _, test in folds] == [7, 6, 6]


def test_repeats_with_groups_differ(rng):
    groups = np.repeat([f"s{i}" for i in range(12)], 2)
    y = rng.uniform(size=24)
    first, second = (
        split_indices(y, groups, task="regression", n_splits=4, random_state=seed)
        for seed in repeat_seeds(0, 2)
    )
    assert any(not np.array_equal(a[1], b[1]) for a, b in zip(first, second, strict=True))


def test_stratified_folds_keep_class_proportions():
    y = np.array(["a"] * 20 + ["b"] * 10)
    folds = split_indices(y, None, task="classification", n_splits=5, random_state=1)
    for _, test in folds:
        assert (y[test] == "b").sum() == 2


def test_impossible_splits_are_explained():
    with pytest.raises(DatasetError, match="groups"):
        split_indices(np.arange(6.0), ["a", "a", "b", "b", "c", "c"], task="regression",
                      n_splits=4, random_state=0)  # fmt: skip
    with pytest.raises(DatasetError, match="smallest class"):
        split_indices(np.array(["a"] * 8 + ["b"] * 2), None, task="classification",
                      n_splits=3, random_state=0)  # fmt: skip
    with pytest.raises(DatasetError, match="folds from"):
        split_indices(np.arange(3.0), None, task="regression", n_splits=4, random_state=0)


def test_group_detection():
    assert not has_groups(None) and not has_groups(np.array(["", ""]))
    assert has_groups(["a", ""])
