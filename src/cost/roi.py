"""Business-impact ROI framing on top of the INR cost model."""
from __future__ import annotations

import numpy as np


INR_PER_USD = 83.0
# Assumed friction cost when a legitimate return is wrongly held for review
DEFAULT_FP_FRICTION_INR = 120.0
# Assumed recovery fraction when true abuse is caught before payout
DEFAULT_RECOVERY_RATE = 0.85


def estimate_fp_friction_inr(
    n_false_positives: int, friction_per_fp: float = DEFAULT_FP_FRICTION_INR
) -> float:
    return float(n_false_positives * friction_per_fp)


def estimate_fraud_loss_prevented_inr(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    refund_amounts_inr: np.ndarray,
    recovery_rate: float = DEFAULT_RECOVERY_RATE,
) -> float:
    mask = (y_true == 1) & (y_pred == 1)
    return float(np.sum(refund_amounts_inr[mask]) * recovery_rate)


def net_protected_value(
    fraud_loss_prevented: float, fp_friction_cost: float
) -> float:
    """Net Protected Value = Fraud Loss Prevented − FP Friction Cost."""
    return float(fraud_loss_prevented - fp_friction_cost)


def times_roi(fraud_loss_prevented: float, fp_friction_cost: float) -> float | None:
    """x-times ROI relative to FP friction cost."""
    if fp_friction_cost <= 0:
        return None
    return float(fraud_loss_prevented / fp_friction_cost)


def cost_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    refund_amounts_inr: np.ndarray,
    precision: float,
    recall: float,
    friction_per_fp: float = DEFAULT_FP_FRICTION_INR,
    recovery_rate: float = DEFAULT_RECOVERY_RATE,
) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    refund_amounts_inr = np.asarray(refund_amounts_inr, dtype=float)

    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    prevented = estimate_fraud_loss_prevented_inr(
        y_true, y_pred, refund_amounts_inr, recovery_rate
    )
    friction = estimate_fp_friction_inr(fp, friction_per_fp)
    npv = net_protected_value(prevented, friction)
    roi = times_roi(prevented, friction)

    return {
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "fp_friction_inr": friction,
        "fraud_loss_prevented_inr": prevented,
        "net_protected_value_inr": npv,
        "times_roi": roi,
        "assumptions": {
            "fp_friction_inr_per_case": friction_per_fp,
            "recovery_rate_on_caught_abuse": recovery_rate,
            "fx_usd_to_inr": INR_PER_USD,
        },
        "story": (
            f"Net Protected Value = Fraud Loss Prevented (₹{prevented:,.0f}) "
            f"− FP Friction Cost (₹{friction:,.0f}) = ₹{npv:,.0f}"
            + (f" ({roi:.1f}x ROI)." if roi is not None else ".")
        ),
        "precision_used": precision,
        "recall_used": recall,
    }
