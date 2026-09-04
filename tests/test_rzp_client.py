"""Razorpay client tests (mocked SDK)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.rzp import client as rzp


def test_create_test_order_shape(monkeypatch):
    monkeypatch.setenv("RAZORPAY_KEY_ID", "rzp_test_key")
    monkeypatch.setenv("RAZORPAY_KEY_SECRET", "rzp_test_secret")

    mock_order = MagicMock()
    mock_order.create.return_value = {
        "id": "order_test123",
        "amount": 50000,
        "currency": "INR",
        "status": "created",
    }
    mock_client = MagicMock()
    mock_client.order = mock_order

    with patch("razorpay.Client", return_value=mock_client):
        out = rzp.create_test_order(500.0, {"case_id": "CASE-1"})

    assert out["order_id"] == "order_test123"
    assert out["amount"] == 50000
    assert out["currency"] == "INR"
    assert out["status"] == "created"


def test_not_configured_raises(monkeypatch):
    monkeypatch.delenv("RAZORPAY_KEY_ID", raising=False)
    monkeypatch.delenv("RAZORPAY_KEY_SECRET", raising=False)
    with pytest.raises(RuntimeError, match="not configured"):
        rzp.create_test_order(100.0, {})
