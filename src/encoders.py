# src/encoders.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

RARE_TOKEN = "__RARE__"

@dataclass
class RareCategoryGrouper:
    min_count: int = 20

    def fit(self, s: pd.Series) -> "RareCategoryGrouper":
        vc = s.value_counts(dropna=True)
        self.valid_ = set(vc[vc >= self.min_count].index.tolist())
        return self

    def transform(self, s: pd.Series) -> pd.Series:
        mask = ~s.isin(getattr(self, "valid_", set()))
        out = s.astype("string").copy()
        out[mask] = RARE_TOKEN
        return out

    def fit_transform(self, s: pd.Series) -> pd.Series:
        return self.fit(s).transform(s)

@dataclass
class FrequencyEncoder:
    smoothing: float = 1.0

    def fit(self, s: pd.Series) -> "FrequencyEncoder":
        n = len(s)
        vc = s.value_counts(dropna=True)
        self.freq_ = (vc / max(n, 1)).to_dict()
        return self

    def transform(self, s: pd.Series) -> pd.Series:
        d = getattr(self, "freq_", {})
        return s.map(d).fillna(0.0).astype("float32")

    def fit_transform(self, s: pd.Series) -> pd.Series:
        return self.fit(s).transform(s)

@dataclass
class KFoldTargetEncoder:
    """
    Leakage-safe OOF target encoding:
      - fit_transform(train, y, folds): returns OOF means for train.
      - transform(test): uses FULL train mapping for test.
    """
    cols: Sequence[str]
    kfold: int = 5
    smoothing_alpha: float = 10.0
    _global_means: Optional[Dict[Tuple, float]] = None
    _prior: float = np.nan

    def _key(self, row: pd.Series) -> Tuple:
        return tuple(row[c] for c in self.cols)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series, folds: pd.Series) -> pd.Series:
        assert folds is not None, "folds are required for OOF target encoding"
        df = X[self.cols].copy()
        df["_y"] = y.values
        df["_fold"] = folds.values
        oof = pd.Series(index=df.index, dtype="float32")
        self._prior = float(df["_y"].mean())
        # Full mapping for test-time use
        full = (
            df.groupby(list(self.cols))["_y"]
              .agg(["sum","count"])
              .rename(columns={"sum":"s","count":"n"})
              .reset_index()
        )
        full["mean_smooth"] = (full["s"] + self.smoothing_alpha * self._prior) / (full["n"] + self.smoothing_alpha)
        self._global_means = {tuple(row[list(self.cols)]): row["mean_smooth"] for _, row in full.iterrows()}

        # OOF per fold
        for k in np.sort(df["_fold"].unique()):
            tr = df[df["_fold"] != k]
            va = df[df["_fold"] == k]
            agg = (
                tr.groupby(list(self.cols))["_y"]
                  .agg(["sum","count"])
                  .rename(columns={"sum":"s","count":"n"})
                  .reset_index()
            )
            agg["mean_smooth"] = (agg["s"] + self.smoothing_alpha * self._prior) / (agg["n"] + self.smoothing_alpha)
            mapping = {tuple(row[list(self.cols)]): row["mean_smooth"] for _, row in agg.iterrows()}
            oof.loc[va.index] = [
                mapping.get(tuple(va.loc[i, list(self.cols)]), self._prior) for i in va.index
            ]
        return oof.astype("float32")

    def transform(self, X: pd.DataFrame) -> pd.Series:
        prior = self._prior if not np.isnan(self._prior) else 0.5
        mapping = self._global_means or {}
        vals = [mapping.get(tuple(X.loc[i, list(self.cols)]), prior) for i in X.index]
        return pd.Series(vals, index=X.index, dtype="float32")
