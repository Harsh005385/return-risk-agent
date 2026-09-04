"""Deterministic policy engine - LLM never decides final action."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PolicyDecision:
    action: str  # ALLOW | MONITOR | REVIEW | HOLD
    rationale: str
    risk_score: float
    confidence: str


def decide(
    risk_score: float,
    evidence: dict | None = None,
    confidence_gap: float | None = None,
) -> PolicyDecision:
    """
    Map score + evidence to a defense-only action.
    HOLD/REVIEW always imply human review - never auto-punish.
    """
    evidence = evidence or {}
    shared = evidence.get("shared_signals") or {}
    syndicate_hits = int(shared.get("linked_flagged_customers", 0) or 0)
    pattern_flags = int((evidence.get("pattern") or {}).get("flag_count", 0) or 0)

    # Confidence gate: ambiguous mid-band scores force REVIEW
    if confidence_gap is not None and confidence_gap < 0.08 and 0.35 <= risk_score <= 0.65:
        return PolicyDecision(
            action="REVIEW",
            rationale="Confidence gate: score near decision boundary; route to human review.",
            risk_score=risk_score,
            confidence="low",
        )

    # Shared-signal links come from coarse device/payment/country fingerprints. On a
    # case the model itself scores as clearly low risk they justify monitoring, not a hold.
    if risk_score < 0.25 and syndicate_hits >= 1 and pattern_flags < 2:
        return PolicyDecision(
            action="MONITOR",
            rationale=(
                "Low risk score, but shares device/payment/address fingerprint with "
                "previously flagged accounts. Allow refund path with heightened monitoring."
            ),
            risk_score=risk_score,
            confidence="medium",
        )

    if risk_score >= 0.85 or syndicate_hits >= 3:
        return PolicyDecision(
            action="HOLD",
            rationale=(
                "High risk score and/or multi-customer shared-signal cluster. "
                "Queue for human review before any refund decision."
            ),
            risk_score=risk_score,
            confidence="high" if risk_score >= 0.85 else "medium",
        )

    if risk_score >= 0.65 or pattern_flags >= 2 or syndicate_hits >= 1:
        return PolicyDecision(
            action="REVIEW",
            rationale="Elevated risk or behavioral/shared signals warrant analyst review.",
            risk_score=risk_score,
            confidence="medium",
        )

    if risk_score >= 0.40:
        return PolicyDecision(
            action="MONITOR",
            rationale="Moderate risk - allow refund path with heightened monitoring.",
            risk_score=risk_score,
            confidence="medium",
        )

    return PolicyDecision(
        action="ALLOW",
        rationale="Low risk with no material adverse signals.",
        risk_score=risk_score,
        confidence="high",
    )
