"""SHAP explanations for ensemble risk scores (cached explainer)."""
from __future__ import annotations

from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from src.paths import ARTIFACT_PATH


@lru_cache(maxsize=1)
def load_bundle(path: str | None = None):
    p = path or str(ARTIFACT_PATH)
    return joblib.load(p)


@lru_cache(maxsize=1)
def _get_explainer():
    bundle = load_bundle()
    model = bundle["model"]
    estimators = dict(model.named_estimators_)
    tree_model = estimators.get("rf") or estimators.get("gb")
    try:
        import shap

        return shap.TreeExplainer(tree_model), tree_model, bundle["feature_cols"], model
    except Exception:
        return None, tree_model, bundle["feature_cols"], model


def warmup() -> None:
    """Load model + SHAP explainer into memory."""
    _get_explainer()


def get_shap_explanation(features: dict, top_k: int = 3) -> dict:
    """Return top SHAP drivers for a single feature dict."""
    explainer, tree_model, cols, model = _get_explainer()
    row = pd.DataFrame([{c: float(features.get(c, 0) or 0) for c in cols}])

    try:
        if explainer is None:
            raise RuntimeError("no explainer")
        values = explainer.shap_values(row)
        if isinstance(values, list):
            sv = np.asarray(values[1][0])
        else:
            sv = np.asarray(values[0])
            if sv.ndim > 1:
                sv = sv[:, -1] if sv.shape[-1] == 2 else sv.ravel()
        method = "shap_tree_rf_cached"
    except Exception:
        importances = getattr(tree_model, "feature_importances_", np.ones(len(cols)))
        classes = list(getattr(model, "classes_", [0, 1]))
        pos_idx = classes.index(1) if 1 in classes else -1
        proba = float(model.predict_proba(row)[0, pos_idx])
        signed = importances * (1 if proba >= 0.5 else -1)
        sv = signed
        method = "importance_fallback"

    order = np.argsort(np.abs(sv))[::-1][:top_k]
    drivers = [
        {
            "feature": cols[i],
            "shap_value": float(sv[i]),
            "feature_value": float(row.iloc[0][cols[i]]),
        }
        for i in order
    ]
    return {"top_drivers": drivers, "method": method}
