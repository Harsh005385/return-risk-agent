"""Razorpay test-mode order helper (optional dashboard integration)."""
from __future__ import annotations

import os
from typing import Any

from src.env import load_env

load_env()


def is_configured() -> bool:
    return bool(os.getenv("RAZORPAY_KEY_ID") and os.getenv("RAZORPAY_KEY_SECRET"))


def create_test_order(amount_inr: float, notes: dict[str, Any] | None = None) -> dict:
    """
    Create a Razorpay test-mode order.
    Returns {order_id, amount, currency, status}.
    """
    if not is_configured():
        raise RuntimeError("Razorpay keys not configured (RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET)")

    import razorpay

    client = razorpay.Client(
        auth=(os.environ["RAZORPAY_KEY_ID"], os.environ["RAZORPAY_KEY_SECRET"])
    )
    amount_paise = int(round(float(amount_inr) * 100))
    payload = {
        "amount": max(amount_paise, 100),
        "currency": "INR",
        "notes": notes or {},
    }
    order = client.order.create(data=payload)
    return {
        "order_id": order.get("id"),
        "amount": order.get("amount"),
        "currency": order.get("currency", "INR"),
        "status": order.get("status", "created"),
    }
