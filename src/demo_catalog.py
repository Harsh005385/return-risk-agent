"""Lookup helpers for dashboard customer/order pickers from case_index."""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from src.data_prep import FEATURE_COLS
from src.features import row_to_features
from src.paths import DATA_PROCESSED


@lru_cache(maxsize=1)
def load_index() -> pd.DataFrame:
    path = DATA_PROCESSED / "case_index.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def customer_options(limit: int = 5000) -> list[dict]:
    df = load_index()
    if df.empty:
        return []
    agg = (
        df.groupby("customer_id", as_index=False)
        .agg(
            return_rate_pct=("return_rate_pct", "mean"),
            n_orders=("order_id", "count"),
            prior_abuse=("abuse_label", "sum"),
        )
        .sort_values("customer_id")
        .head(limit)
    )
    return [
        {
            "label": (
                f"{r.customer_id} · {int(r.n_orders)} orders · "
                f"{float(r.return_rate_pct):.0f}% return rate"
            ),
            "value": r.customer_id,
        }
        for r in agg.itertuples()
    ]


def order_options(customer_id: str | None = None, limit: int = 5000) -> list[dict]:
    df = load_index()
    if df.empty:
        return []
    if customer_id:
        df = df[df["customer_id"].astype(str) == str(customer_id)]
    df = df.sort_values("order_id").head(limit)
    return [
        {
            "label": (
                f"{r.order_id} · {r.customer_id} · "
                f"${float(r.refund_amount_requested_usd):.0f} · "
                f"{float(r.return_rate_pct):.0f}%"
            ),
            "value": r.order_id,
        }
        for r in df.itertuples()
    ]


def customer_detail(customer_id: str | None) -> dict | None:
    if not customer_id:
        return None
    df = load_index()
    if df.empty:
        return None
    hist = df[df["customer_id"].astype(str) == str(customer_id)]
    if hist.empty:
        return None
    return {
        "customer_id": str(customer_id),
        "n_orders": int(len(hist)),
        "mean_return_rate_pct": float(hist["return_rate_pct"].mean()),
        "prior_abuse_flags": int(hist["abuse_label"].sum()),
        "categories": hist["product_category"].value_counts().head(3).to_dict(),
        "countries": hist["country"].value_counts().head(2).to_dict(),
        "recent_refund_usd": float(hist["refund_amount_requested_usd"].iloc[-1]),
    }


def order_detail(order_id: str | None) -> dict | None:
    if not order_id:
        return None
    df = load_index()
    if df.empty:
        return None
    hit = df[df["order_id"].astype(str) == str(order_id)]
    if hit.empty:
        return None
    row = hit.iloc[0]
    return {
        "order_id": str(order_id),
        "customer_id": str(row["customer_id"]),
        "refund_amount_requested_usd": float(row["refund_amount_requested_usd"]),
        "return_rate_pct": float(row["return_rate_pct"]),
        "product_category": row.get("product_category"),
        "return_reason": row.get("return_reason"),
        "days_to_return": float(row.get("days_to_return", 0)),
        "device_type": row.get("device_type"),
        "payment_method": row.get("payment_method"),
        "country": row.get("country"),
        "abuse_label": int(row.get("abuse_label", 0)),
    }


@lru_cache(maxsize=1)
def manual_feature_template() -> dict:
    """Median feature vector from the dataset. Used as the baseline for manual entry
    so that untouched fields carry realistic values instead of zeros."""
    df = load_index()
    template = {c: 0.0 for c in FEATURE_COLS}
    if df.empty:
        return template
    for c in FEATURE_COLS:
        if c in df.columns:
            series = pd.to_numeric(df[c], errors="coerce").dropna()
            if not series.empty:
                template[c] = float(series.median())
    return template


@lru_cache(maxsize=8)
def category_values(col: str) -> list[str]:
    """Distinct labels for a categorical column, sorted."""
    df = load_index()
    if df.empty or col not in df.columns:
        return []
    return sorted(str(v) for v in df[col].dropna().unique())


@lru_cache(maxsize=64)
def encode_label(col: str, label: str | None) -> int:
    """Map a categorical label to the integer code the model was trained on."""
    df = load_index()
    enc_col = f"{col}_enc"
    if df.empty or col not in df.columns or enc_col not in df.columns or label is None:
        return 0
    hit = df[df[col].astype(str) == str(label)]
    if hit.empty:
        return 0
    return int(hit[enc_col].iloc[0])


def build_case_from_selection(
    customer_id: str | None,
    order_id: str | None,
    refund_usd: float | None = None,
    return_rate_pct: float | None = None,
    force_syndicate_check: bool = False,
) -> dict:
    """Build agent case; use catalog row when order exists, else manual features."""
    df = load_index()
    order_id = str(order_id or "").strip()
    customer_id = str(customer_id or "").strip()

    if not df.empty and order_id:
        hit = df[df["order_id"].astype(str) == order_id]
        if not hit.empty:
            row = hit.iloc[0]
            features = row_to_features(row)
            if refund_usd is not None:
                features["refund_amount_requested_usd"] = float(refund_usd)
                features["avg_order_value_usd"] = float(refund_usd)
                features["is_high_value_item"] = int(float(refund_usd) > 300)
            if return_rate_pct is not None:
                features["return_rate_pct"] = float(return_rate_pct)
            return {
                "order_id": order_id,
                "customer_id": str(row["customer_id"]),
                "device_type": row.get("device_type"),
                "payment_method": row.get("payment_method"),
                "country": row.get("country"),
                "product_category": row.get("product_category"),
                "return_reason": row.get("return_reason"),
                "force_syndicate_check": force_syndicate_check,
                "features": features,
            }

    from src.agent.scenarios import build_manual_case

    return build_manual_case(
        customer_id=customer_id or "CUST-SAFE-001",
        order_id=order_id or "DEMO-MANUAL-001",
        refund_usd=float(refund_usd or 0),
        return_rate_pct=float(return_rate_pct or 0),
        force_syndicate_check=force_syndicate_check,
    )
