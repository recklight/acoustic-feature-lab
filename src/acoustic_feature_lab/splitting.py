"""Cross-validation splits that respect speakers and class balance.

* Classification uses stratified folds, so every fold keeps the class
  proportions; with a ``group`` column the folds are also grouped
  (``StratifiedGroupKFold``), so no speaker contributes to both sides.
* Regression uses shuffled k-fold, or, with groups, puts whole groups on the
  folds largest first, each on the fold with the fewest items so far (the
  ``GroupKFold`` rule). Groups of equal size are taken in an order drawn from
  the seed, so repeated splits differ and do not depend on how a NumPy release
  orders ties when sorting.

The splits depend only on the targets, the groups and the seed, not on the
features, so every feature setting of an ablation study is evaluated on the
same folds. That is what allows the settings to be compared fold by fold in a
paired test.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.model_selection import KFold, StratifiedGroupKFold, StratifiedKFold

from .errors import DatasetError

#: Learning tasks accepted by ``data.task``.
TASKS: tuple[str, ...] = ("regression", "classification")

Split = tuple[NDArray[np.intp], NDArray[np.intp]]


def has_groups(groups: ArrayLike | None) -> bool:
    """Whether a group vector carries information (not missing, not all empty).

    Examples
    --------
    >>> has_groups(None), has_groups(["", ""]), has_groups(["s1", "s2"])
    (False, False, True)
    """
    if groups is None:
        return False
    values = np.asarray(groups).astype(str)
    return bool(values.size) and bool(np.any(values != ""))


def split_indices(
    y: ArrayLike,
    groups: ArrayLike | None = None,
    *,
    task: str,
    n_splits: int,
    random_state: int,
) -> list[Split]:
    """Train and test indices of every fold.

    Parameters
    ----------
    y : array_like, shape (n_items,)
        Targets (class names or ratings).
    groups : array_like, shape (n_items,), optional
        Speaker or session of every item; empty strings mean "no grouping".
    task : {"regression", "classification"}
        Chooses stratification.
    n_splits : int
        Number of folds.
    random_state : int
        Seed of the shuffle.

    Returns
    -------
    list of tuple of numpy.ndarray
        ``(train, test)`` index arrays; every item is tested exactly once.

    Raises
    ------
    DatasetError
        If there are fewer items, groups or class members than folds.

    Examples
    --------
    >>> y = np.repeat(["healthy", "dysphonic"], 6)
    >>> groups = np.repeat([f"s{i}" for i in range(6)], 2)
    >>> folds = split_indices(y, groups, task="classification", n_splits=3, random_state=0)
    >>> sorted(int(i) for _, test in folds for i in test) == list(range(12))
    True
    >>> all(not set(groups[a]) & set(groups[b]) for a, b in folds)
    True
    """
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; choose one of {list(TASKS)}")
    targets = np.asarray(y)
    n_items = targets.shape[0]
    if n_splits > n_items:
        raise DatasetError(f"cannot make {n_splits} folds from {n_items} items")
    grouped = has_groups(groups)
    group_codes = None
    if grouped:
        labels = np.asarray(groups).astype(str)
        unique, group_codes = np.unique(labels, return_inverse=True)
        if unique.size < n_splits:
            raise DatasetError(
                f"cannot make {n_splits} grouped folds from {unique.size} groups; "
                "lower evaluation.n_splits or add speakers"
            )
    placeholder = np.zeros((n_items, 1))
    if task == "classification":
        _, counts = np.unique(targets.astype(str), return_counts=True)
        if counts.min() < n_splits:
            raise DatasetError(
                f"the smallest class has {int(counts.min())} items, fewer than the "
                f"{n_splits} folds; lower the number of folds"
            )
        if grouped:
            splitter = StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state
            )
            folds = splitter.split(placeholder, targets.astype(str), group_codes)
        else:
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            folds = splitter.split(placeholder, targets.astype(str))
    elif grouped:
        assert group_codes is not None
        folds = _grouped_folds(group_codes, n_splits, random_state)
    else:
        folds = KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(placeholder)
    return [(np.asarray(train), np.asarray(test)) for train, test in folds]


def _grouped_folds(codes: NDArray[np.intp], n_splits: int, random_state: int) -> list[Split]:
    """Folds that keep every group whole, balanced by item count.

    Same rule as ``GroupKFold``: largest group first, each onto the fold with
    the fewest items so far (the first such fold on a tie). ``GroupKFold``
    breaks ties between equal-sized groups by the order ``np.argsort`` leaves
    them in, which differs between NumPy releases, so the same seed would give
    different folds on different installations. Here a seeded rank of each
    group breaks the tie.
    """
    sizes = np.bincount(codes)
    rank = np.random.default_rng(random_state).permutation(sizes.size)
    # Ranks are unique, so this order has no ties left to break.
    order = np.lexsort((-rank, -sizes))
    load = np.zeros(n_splits, dtype=np.int64)
    fold_of_group = np.empty(sizes.size, dtype=np.intp)
    for group in order:
        fold = int(np.argmin(load))
        fold_of_group[group] = fold
        load[fold] += sizes[group]
    fold_of_item = fold_of_group[codes]
    items = np.arange(codes.size)
    return [(items[fold_of_item != fold], items[fold_of_item == fold]) for fold in range(n_splits)]


def repeat_seeds(seed: int, n_repeats: int) -> Sequence[int]:
    """Independent split seeds for repeated cross-validation, derived from one master seed.

    Examples
    --------
    >>> seeds = repeat_seeds(0, 3)
    >>> len(seeds), len(set(seeds)), list(repeat_seeds(0, 3)) == list(seeds)
    (3, 3, True)
    """
    children = np.random.SeedSequence(seed).spawn(n_repeats)
    return [int(child.generate_state(1)[0] % (2**31 - 1)) for child in children]
