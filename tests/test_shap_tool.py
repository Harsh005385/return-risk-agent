"""SHAP explanation determinism tests."""
from __future__ import annotations

import pytest

from src.agent.scenarios import get_scenario
from src.explain.shap_tool import get_shap_explanation
from src.features import row_to_features


@pytest.fixture(scope="module", autouse=True)
def _warm_shap():
    try:
        from src.bootstrap import warm_runtime

        warm_runtime()
    except Exception:
        pytest.skip("Model artifacts not available")


def test_shap_top3_stable():
    features = row_to_features(get_scenario("safe")["features"])
    a = get_shap_explanation(features, top_k=3)
    b = get_shap_explanation(features, top_k=3)
    da = a["top_drivers"]
    db = b["top_drivers"]
    assert len(da) == len(db) == 3
    for x, y in zip(da, db):
        assert x["feature"] == y["feature"]
        assert (x["shap_value"] >= 0) == (y["shap_value"] >= 0)
