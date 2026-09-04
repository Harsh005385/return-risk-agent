"""One-shot runtime warmup: model, case index, graph, SHAP."""
from __future__ import annotations


def warm_runtime() -> None:
    from src.agent.tools import (
        _build_shared_signal_graph,
        _load_cases,
        _load_model_bundle,
    )
    from src.explain.shap_tool import warmup

    _load_model_bundle()
    _load_cases()
    _build_shared_signal_graph()
    warmup()
