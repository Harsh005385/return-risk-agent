"""Unit tests for policy, ROI, and agent tools (offline)."""
from __future__ import annotations

import numpy as np

from src.cost.roi import cost_summary, net_protected_value, times_roi
from src.policy.policy_engine import decide


def test_policy_allow_low_score():
    d = decide(0.1, evidence={})
    assert d.action == "ALLOW"


def test_policy_hold_high_score():
    d = decide(0.9, evidence={})
    assert d.action == "HOLD"


def test_policy_confidence_gate():
    d = decide(0.5, evidence={}, confidence_gap=0.02)
    assert d.action == "REVIEW"


def test_policy_syndicate_signal():
    d = decide(
        0.5,
        evidence={"shared_signals": {"linked_flagged_customers": 3}},
    )
    assert d.action == "HOLD"


def test_net_protected_value_framing():
    assert net_protected_value(1000, 200) == 800
    assert times_roi(1000, 200) == 5.0


def test_cost_summary_keys():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 0])
    refunds = np.array([100.0, 50.0, 80.0, 40.0])
    out = cost_summary(y_true, y_pred, refunds, precision=0.5, recall=0.5)
    assert "net_protected_value_inr" in out
    assert "times_roi" in out
    assert "story" in out
