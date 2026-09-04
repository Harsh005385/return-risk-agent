"""Feature-ablation stress test (evaluation only - does not overwrite production model).

Retrains the same soft-voting ensemble on the chronological split with dominant
feature(s) removed, then writes metrics for MODEL_CARD.md.

Usage (from repo root):
  set PYTHONPATH=.
  python scripts/feature_ablation.py
  python scripts/feature_ablation.py --sample-n 20000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_prep import FEATURE_COLS, TARGET_COL, prepare_dataset, time_split
from src.models.train_ensemble import build_ensemble, evaluate
from src.paths import DATA_PROCESSED, METRICS_PATH

OUT_PATH = DATA_PROCESSED / "ablation_metrics.json"


def _feature_matrix(df, cols: list[str]):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")
    X = df[cols].fillna(0)
    y = df[TARGET_COL].astype(int)
    return X, y


def _rf_importance_rank(X_train, y_train, cols: list[str], top_k: int = 5) -> list[tuple[str, float]]:
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        min_samples_leaf=5,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=42,
    )
    rf.fit(X_train, y_train)
    order = np.argsort(rf.feature_importances_)[::-1]
    return [(cols[i], float(rf.feature_importances_[i])) for i in order[:top_k]]


def _eval_ablation(train_df, test_df, drop: list[str]) -> dict:
    cols = [c for c in FEATURE_COLS if c not in drop]
    X_train, y_train = _feature_matrix(train_df, cols)
    X_test, y_test = _feature_matrix(test_df, cols)
    model = build_ensemble()
    model.fit(X_train, y_train)
    refund_inr = test_df["refund_amount_requested_usd"].fillna(0).to_numpy() * 83.0
    metrics = evaluate(model, X_test, y_test, refund_inr)
    return {
        "ablated_features": drop,
        "n_features": len(cols),
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics.get("pr_auc"),
        "n_test": metrics["n_test"],
    }


def run(sample_n: int | None = 20000, test_ratio: float = 0.2) -> dict:
    baseline = {}
    if METRICS_PATH.exists():
        baseline = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    df = prepare_dataset(sample_n=sample_n)
    train_df, test_df = time_split(df, test_ratio=test_ratio)

    full_cols = list(FEATURE_COLS)
    X_train_full, y_train = _feature_matrix(train_df, full_cols)
    top_importance = _rf_importance_rank(X_train_full, y_train, full_cols, top_k=5)
    top3 = [f for f, _ in top_importance[:3]]

    without_rate = _eval_ablation(train_df, test_df, ["return_rate_pct"])
    without_top3 = _eval_ablation(train_df, test_df, top3)

    report = {
        "disclaimer": (
            "Ablation evaluation only. Does not replace the production model bundle. "
            "Same chronological split and ensemble builder as train_ensemble."
        ),
        "sample_n": sample_n,
        "test_ratio": test_ratio,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "rf_importance_top5_full_set": [
            {"feature": f, "importance": round(v, 6)} for f, v in top_importance
        ],
        "baseline_full_feature_set": {
            "precision": baseline.get("precision"),
            "recall": baseline.get("recall"),
            "f1": baseline.get("f1"),
            "roc_auc": baseline.get("roc_auc"),
            "source": str(METRICS_PATH.name) if baseline else None,
        },
        "without_return_rate_pct": without_rate,
        "without_top3_importance_features": without_top3,
    }

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved ablation report -> {OUT_PATH}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature-ablation stress test (docs only)")
    parser.add_argument("--sample-n", type=int, default=20000)
    parser.add_argument("--full", action="store_true", help="Use full dataset (sample_n=None)")
    args = parser.parse_args()
    run(sample_n=None if args.full else args.sample_n)
