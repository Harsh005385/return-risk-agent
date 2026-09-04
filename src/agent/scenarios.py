"""Synthetic demo scenarios for interactive attack/scenario simulator."""
from __future__ import annotations

from copy import deepcopy

# Feature defaults aligned with FEATURE_COLS; tuned for demo archetypes.
_BASE = {
    "age": 32,
    "account_age_days": 400,
    "avg_order_value_usd": 80.0,
    "refund_amount_requested_usd": 75.0,
    "is_high_value_item": 0,
    "discount_used": 0,
    "days_to_return": 10,
    "total_orders_lifetime": 20,
    "total_returns_lifetime": 2,
    "return_rate_pct": 10.0,
    "item_returned_opened": 1,
    "return_packaging_intact": 1,
    "photo_evidence_provided": 1,
    "tracking_number_valid": 1,
    "address_change_before_delivery": 0,
    "refund_to_different_account": 0,
    "multiple_accounts_flag": 0,
    "customer_support_contacts": 0,
    "previous_dispute_count": 0,
    "wishlist_to_cart_time_hrs": 12.0,
    "review_left_after_return": 0,
    "customer_segment_enc": 1,
    "device_type_enc": 1,
    "payment_method_enc": 1,
    "product_category_enc": 1,
    "return_reason_enc": 1,
    "platform_enc": 1,
    "shipping_carrier_enc": 1,
    "sla_days_proxy": 10.0,
    "return_cost_proxy": 12.0,
}


SCENARIOS: dict[str, dict] = {
    "safe": {
        "label": "Safe return",
        "description": "Long-tenured customer, low return rate, intact packaging, evidence provided.",
        "case": {
            "order_id": "DEMO-SAFE-001",
            "customer_id": "CUST-SAFE-001",
            "device_type": "iPhone",
            "payment_method": "Credit Card",
            "country": "US",
            "product_category": "Books",
            "return_reason": "Changed Mind",
            "features": {
                **_BASE,
                "account_age_days": 1200,
                "return_rate_pct": 3.0,
                "total_returns_lifetime": 1,
                "photo_evidence_provided": 1,
                "tracking_number_valid": 1,
            },
        },
    },
    "risky": {
        "label": "Risky abuse pattern",
        "description": "High return rate, refund to different account, address change, high value.",
        "case": {
            "order_id": "DEMO-RISKY-001",
            "customer_id": "CUST-RISKY-001",
            "device_type": "Android",
            "payment_method": "Crypto",
            "country": "US",
            "product_category": "Electronics",
            "return_reason": "Defective/Broken",
            "features": {
                **_BASE,
                "account_age_days": 14,
                "is_high_value_item": 1,
                "refund_amount_requested_usd": 890.0,
                "return_rate_pct": 62.0,
                "total_returns_lifetime": 11,
                "days_to_return": 1,
                "address_change_before_delivery": 1,
                "refund_to_different_account": 1,
                "multiple_accounts_flag": 1,
                "previous_dispute_count": 4,
                "photo_evidence_provided": 0,
                "tracking_number_valid": 0,
            },
        },
    },
    "ambiguous": {
        "label": "Ambiguous mid-band",
        "description": "Mixed signals near decision boundary - should hit confidence gate / REVIEW.",
        "case": {
            "order_id": "DEMO-AMBIG-001",
            "customer_id": "CUST-AMBIG-001",
            "device_type": "iPhone",
            "payment_method": "PayPal",
            "country": "IN",
            "product_category": "Apparel",
            "return_reason": "Size Issue",
            "features": {
                **_BASE,
                "account_age_days": 90,
                "return_rate_pct": 28.0,
                "total_returns_lifetime": 4,
                "days_to_return": 5,
                "customer_support_contacts": 2,
                "photo_evidence_provided": 0,
                "refund_amount_requested_usd": 140.0,
            },
        },
    },
    "syndicate": {
        "label": "Abuse-ring / syndicate probe",
        "description": "Forces shared-signal graph check; mimics organized multi-account abuse.",
        "case": {
            "order_id": "DEMO-RING-001",
            "customer_id": "CUST-RING-001",
            "force_syndicate_check": True,
            "device_type": "Android",
            "payment_method": "Crypto",
            "country": "US",
            "product_category": "Electronics",
            "return_reason": "Defective/Broken",
            "features": {
                **_BASE,
                "account_age_days": 20,
                "return_rate_pct": 48.0,
                "multiple_accounts_flag": 1,
                "refund_to_different_account": 1,
                "is_high_value_item": 1,
                "refund_amount_requested_usd": 640.0,
                "days_to_return": 2,
            },
        },
    },
}


def list_scenarios() -> list[dict]:
    return [
        {"id": k, "label": v["label"], "description": v["description"]}
        for k, v in SCENARIOS.items()
    ]


def get_scenario(name: str) -> dict | None:
    item = SCENARIOS.get(name)
    if not item:
        return None
    return deepcopy(item["case"])


def build_manual_case(
    customer_id: str,
    order_id: str,
    refund_usd: float,
    return_rate_pct: float,
    force_syndicate_check: bool = False,
) -> dict:
    """Build a live-score case from dashboard / API form inputs."""
    refund = float(refund_usd or 0)
    return {
        "customer_id": customer_id or "CUST-SAFE-001",
        "order_id": order_id or "DEMO-MANUAL-001",
        "force_syndicate_check": force_syndicate_check,
        "features": {
            **_BASE,
            "avg_order_value_usd": refund,
            "refund_amount_requested_usd": refund,
            "is_high_value_item": int(refund > 300),
            "return_rate_pct": float(return_rate_pct or 0),
            "days_to_return": 8,
            "total_orders_lifetime": 15,
            "total_returns_lifetime": 2,
            "wishlist_to_cart_time_hrs": 10.0,
        },
    }
