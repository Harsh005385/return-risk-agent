"""Train soft-voting ensemble: RandomForest + XGBoost + GradientBoosting."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from src.cost.roi import cost_summary
from src.data_prep import FEATURE_COLS, prepare_dataset, save_processed, time_split
from src.features import feature_matrix
from src.paths import ARTIFACT_PATH, DATA_PROCESSED, METRICS_PATH, MODEL_DIR


def build_ensemble(random_state: int = 42) -> VotingClassifier:
    rf = RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        min_samples_leaf=5,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=random_state,
    )
    xgb = XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.8,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=random_state,
    )
    gb = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.08,
        random_state=random_state,
    )
    return VotingClassifier(
        estimators=[("rf", rf), ("xgb", xgb), ("gb", gb)],
        voting="soft",
        n_jobs=-1,
    )


def evaluate(model, X_test, y_test, refund_amounts: np.ndarray) -> dict:
    y_test = np.asarray(y_test).astype(int)
    proba_full = model.predict_proba(X_test)
    # Binary class index for abuse=1; fall back to last column
    classes = list(getattr(model, "classes_", [0, 1]))
    pos_idx = classes.index(1) if 1 in classes else -1
    proba = proba_full[:, pos_idx]
    pred = (proba >= 0.5).astype(int)
    precision = float(precision_score(y_test, pred, average="binary", zero_division=0))
    recall = float(recall_score(y_test, pred, average="binary", zero_division=0))
    f1 = float(f1_score(y_test, pred, average="binary", zero_division=0))
    try:
        roc = float(roc_auc_score(y_test, proba))
    except ValueError:
        roc = float("nan")
    try:
        pr_auc = float(average_precision_score(y_test, proba))
    except ValueError:
        pr_auc = float("nan")
    costs = cost_summary(
        y_true=np.asarray(y_test),
        y_pred=pred,
        refund_amounts_inr=refund_amounts,
        precision=precision,
        recall=recall,
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc,
        "pr_auc": pr_auc,
        "n_test": int(len(y_test)),
        "positive_rate_test": float(np.mean(y_test)),
        **costs,
        "disclaimer": (
            "Metrics computed on synthetic/held-out chronological split. "
            "Not equivalent to production performance."
        ),
    }


def train(
    sample_n: int | None = 20000,
    test_ratio: float = 0.2,
    artifact_path: Path | None = None,
) -> dict:
    artifact_path = artifact_path or ARTIFACT_PATH
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    df = prepare_dataset(sample_n=sample_n)
    save_processed(df)
    train_df, test_df = time_split(df, test_ratio=test_ratio)
    X_train, y_train = feature_matrix(train_df)
    X_test, y_test = feature_matrix(test_df)

    model = build_ensemble()
    model.fit(X_train, y_train)

    # INR proxy: USD * 83
    refund_inr = test_df["refund_amount_requested_usd"].fillna(0).to_numpy() * 83.0
    metrics = evaluate(model, X_test, y_test, refund_inr)

    bundle = {
        "model": model,
        "feature_cols": FEATURE_COLS,
        "metrics": metrics,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "sample_n": sample_n,
    }
    joblib.dump(bundle, artifact_path)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    # Keep a lightweight case index for agent tools / graph signals
    case_index = df[
        [
            "order_id",
            "customer_id",
            "device_type",
            "payment_method",
            "country",
            "abuse_label",
            "refund_amount_requested_usd",
            "return_rate_pct",
            "product_category",
            "return_reason",
            "days_to_return",
            "multiple_accounts_flag",
            "address_change_before_delivery",
            "refund_to_different_account",
        ]
        + [c for c in FEATURE_COLS if c in df.columns]
    ].copy()
    case_index.to_csv(DATA_PROCESSED / "case_index.csv", index=False)
    print(json.dumps(metrics, indent=2))
    print(f"Saved model bundle -> {artifact_path}")
    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-n", type=int, default=20000)
    parser.add_argument("--full", action="store_true", help="Train on full dataset")
    args = parser.parse_args()
    train(sample_n=None if args.full else args.sample_n)
