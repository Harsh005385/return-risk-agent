"""Policy engine threshold boundary tests."""
from __future__ import annotations

import pytest

from src.policy.policy_engine import decide


@pytest.mark.parametrize(
    "score,expected",
    [
        (0.39, "ALLOW"),
        (0.40, "MONITOR"),
        (0.64, "MONITOR"),
        (0.65, "REVIEW"),
        (0.84, "REVIEW"),
        (0.85, "HOLD"),
    ],
)
def test_policy_score_boundaries(score, expected):
    assert decide(score, evidence={}).action == expected


def test_policy_confidence_gate_override():
    d = decide(0.50, evidence={}, confidence_gap=0.05)
    assert d.action == "REVIEW"


def test_policy_confidence_gate_outside_band():
    d = decide(0.20, evidence={}, confidence_gap=0.05)
    assert d.action == "ALLOW"


def test_policy_pattern_flags_review():
    d = decide(0.50, evidence={"pattern": {"flag_count": 2}})
    assert d.action == "REVIEW"


def test_policy_single_syndicate_link_review():
    d = decide(0.50, evidence={"shared_signals": {"linked_flagged_customers": 1}})
    assert d.action == "REVIEW"
