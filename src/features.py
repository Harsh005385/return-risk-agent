"""Feature matrix helpers."""
from __future__ import annotations

import pandas as pd

from src.data_prep import FEATURE_COLS, TARGET_COL


def feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    X = df[FEATURE_COLS].fillna(0)
    y = df[TARGET_COL].astype(int)
    return X, y


def row_to_features(row: dict | pd.Series) -> dict:
    if isinstance(row, pd.Series):
        row = row.to_dict()
    return {c: float(row.get(c, 0) or 0) for c in FEATURE_COLS}
