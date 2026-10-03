"""Financial cross-validation: expanding / rolling walk-forward, purged K-fold with
embargo, and nested validation.

Purging removes training observations whose label window [t, label_end] overlaps the
test window. Embargo additionally removes observations immediately *after* the test
window (serial correlation of features). Both are essential when horizons overlap.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass
class Split:
    train: np.ndarray
    test: np.ndarray
    fold: int

    def describe(self, index: pd.DatetimeIndex) -> str:
        return (f"fold {self.fold}: train {index[self.train[0]].date()}..{index[self.train[-1]].date()} "
                f"({len(self.train)}), test {index[self.test[0]].date()}..{index[self.test[-1]].date()} ({len(self.test)})")


def purge(train_idx: np.ndarray, test_idx: np.ndarray, index: pd.DatetimeIndex, label_end: pd.Series,
          embargo: int = 0) -> np.ndarray:
    """Drop train obs whose label interval overlaps the test interval, plus an embargo after it."""
    if len(test_idx) == 0:
        return train_idx
    t0, t1 = index[test_idx[0]], index[test_idx[-1]]
    t1_label = label_end.iloc[test_idx].max()
    t1_label = t1 if pd.isna(t1_label) else max(t1, t1_label)
    le = label_end.iloc[train_idx]
    starts = index[train_idx]
    # overlap if train start <= test label end AND train label end >= test start
    overlap = (starts <= t1_label) & ((le >= t0) | le.isna()).values
    keep = train_idx[~overlap]
    if embargo > 0:
        emb_end_pos = min(len(index) - 1, test_idx[-1] + embargo)
        emb = (keep > test_idx[-1]) & (keep <= emb_end_pos)
        keep = keep[~emb]
    return keep


def walk_forward(index: pd.DatetimeIndex, label_end: pd.Series, n_splits: int = 5, min_train: int = 500,
                 test_size: int | None = None, mode: str = "expanding", rolling_window: int | None = None,
                 embargo: int = 5) -> Iterator[Split]:
    len(index)
    valid = np.where(label_end.notna().values)[0]
    if len(valid) == 0:
        return
    last = valid[-1] + 1
    test_size = test_size or max(20, (last - min_train) // n_splits)
    start_test = last - n_splits * test_size
    if start_test < min_train:
        start_test = min_train
        test_size = max(20, (last - min_train) // n_splits)
    for k in range(n_splits):
        a = start_test + k * test_size
        b = min(last, a + test_size)
        if a >= b:
            break
        test = np.arange(a, b)
        lo = 0 if mode == "expanding" else max(0, a - (rolling_window or min_train))
        train = np.arange(lo, a)
        train = purge(train, test, index, label_end, embargo)
        train = train[label_end.iloc[train].notna().values]
        if len(train) < min(min_train, 100):
            continue
        yield Split(train, test, k)


def purged_kfold(index: pd.DatetimeIndex, label_end: pd.Series, n_splits: int = 5, embargo: int = 5) -> Iterator[Split]:
    len(index)
    valid = np.where(label_end.notna().values)[0]
    folds = np.array_split(valid, n_splits)
    for k, test in enumerate(folds):
        train = np.setdiff1d(valid, test)
        train = purge(train, test, index, label_end, embargo)
        yield Split(train, test, k)


def nested_walk_forward(index: pd.DatetimeIndex, label_end: pd.Series, outer_splits: int = 4, inner_splits: int = 3,
                        min_train: int = 500, embargo: int = 5):
    """Yield (outer_split, inner_splits_on_outer_train). Hyper-parameters / model choice are
    selected only on the inner splits; the outer test fold is touched once for evaluation."""
    for outer in walk_forward(index, label_end, outer_splits, min_train, embargo=embargo):
        sub_index = index[outer.train]
        sub_le = label_end.iloc[outer.train]
        inner = list(walk_forward(sub_index, sub_le, inner_splits, max(250, min_train // 2), embargo=embargo))
        inner = [Split(outer.train[s.train], outer.train[s.test], s.fold) for s in inner]
        yield outer, inner


def sample_weights(index: pd.DatetimeIndex, scheme: str = "full", halflife_years: float = 5.0,
                   recent_years: float = 8.0) -> np.ndarray:
    """Regime-adaptive training weights: full | recent | exp_weighted."""
    age_years = (index[-1] - index).days.values / 365.25
    if scheme == "recent":
        return (age_years <= recent_years).astype(float)
    if scheme == "exp_weighted":
        return 0.5 ** (age_years / halflife_years)
    return np.ones(len(index))
