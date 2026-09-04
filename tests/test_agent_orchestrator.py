"""Agent orchestrator integration tests."""
from __future__ import annotations

import os

import pytest

from src.agent.orchestrator import run_agent
from src.agent.scenarios import get_scenario

REQUIRED_KEYS = {
    "action",
    "risk_score",
    "rationale",
    "evidence",
    "trace",
    "defense_only",
}


@pytest.fixture(scope="module", autouse=True)
def _warm_artifacts():
    try:
        from src.bootstrap import warm_runtime

        warm_runtime()
    except Exception:
        pytest.skip("Model artifacts not available - run train_ensemble first")


def test_run_agent_safe_fast():
    case = get_scenario("safe")
    verdict = run_agent(case, fast=True)
    assert REQUIRED_KEYS <= set(verdict.keys())
    assert verdict["defense_only"] is True
    assert verdict["action"] in {"ALLOW", "MONITOR", "REVIEW", "HOLD"}


def test_run_agent_risky_fast():
    case = get_scenario("risky")
    verdict = run_agent(case, fast=True)
    assert REQUIRED_KEYS <= set(verdict.keys())
    assert verdict["defense_only"] is True


def test_run_agent_heuristic_sequential_no_llm(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    case = get_scenario("safe")
    verdict = run_agent(case, fast=False)
    assert REQUIRED_KEYS <= set(verdict.keys())
    assert len(verdict.get("trace") or []) <= 4
