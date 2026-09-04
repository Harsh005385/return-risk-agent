"""Load and merge return-abuse primary data with returns-management supplement."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.paths import ABUSE_CSV, DATA_PROCESSED, RETURNS_CSV


FEATURE_COLS = [
    "age",
    "account_age_days",
    "avg_order_value_usd",
    "refund_amount_requested_usd",
    "is_high_value_item",
    "discount_used",
    "days_to_return",
    "total_orders_lifetime",
    "total_returns_lifetime",
    "return_rate_pct",
    "item_returned_opened",
    "return_packaging_intact",
    "photo_evidence_provided",
    "tracking_number_valid",
    "customer_support_contacts",
    "previous_dispute_count",
    "wishlist_to_cart_time_hrs",
    "review_left_after_return",
    "customer_segment_enc",
    "device_type_enc",
    "payment_method_enc",
    "product_category_enc",
    "return_reason_enc",
    "platform_enc",
    "shipping_carrier_enc",
    "sla_days_proxy",
    "return_cost_proxy",
]

TARGET_COL = "abuse_label"
TIME_COL = "return_date"


def _encode_categoricals(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            out[col] = "unknown"
        codes, _ = pd.factorize(out[col].astype(str).fillna("unknown"))
        out[f"{col}_enc"] = codes.astype(int)
    return out


def load_abuse(path: Path | None = None) -> pd.DataFrame:
    path = path or ABUSE_CSV
    df = pd.read_csv(path)
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["return_date"] = pd.to_datetime(df["return_date"], errors="coerce")
    return df


def load_returns_supplement(path: Path | None = None) -> pd.DataFrame:
    path = path or RETURNS_CSV
    df = pd.read_csv(path)
    # Normalize join keys - datasets do not share IDs; use category+reason aggregates.
    df = df.rename(
        columns={
            "Product_Category": "product_category",
            "Return_Reason": "return_reason",
            "Days_to_Return": "days_to_return_sup",
            "Return_Cost": "return_cost",
            "Order_Value": "order_value_sup",
        }
    )
    agg = (
        df.groupby(["product_category", "return_reason"], dropna=False)
        .agg(
            sla_days_proxy=("days_to_return_sup", "median"),
            return_cost_proxy=("return_cost", "median"),
        )
        .reset_index()
    )
    return agg


def prepare_dataset(sample_n: int | None = None) -> pd.DataFrame:
    abuse = load_abuse()
    if sample_n is not None and sample_n < len(abuse):
        abuse = abuse.sample(n=sample_n, random_state=42)

    supplement = load_returns_supplement()
    merged = abuse.merge(
        supplement,
        how="left",
        on=["product_category", "return_reason"],
    )
    merged["sla_days_proxy"] = merged["sla_days_proxy"].fillna(
        merged["days_to_return"].median()
    )
    merged["return_cost_proxy"] = merged["return_cost_proxy"].fillna(
        merged["refund_amount_requested_usd"].median() * 0.15
    )

    cat_cols = [
        "customer_segment",
        "device_type",
        "payment_method",
        "product_category",
        "return_reason",
        "platform",
        "shipping_carrier",
    ]
    merged = _encode_categoricals(merged, cat_cols)
    merged = merged.dropna(subset=[TIME_COL, TARGET_COL]).sort_values(TIME_COL)
    # Force binary target (dataset occasionally carries non-{0,1} codes)
    merged[TARGET_COL] = (merged[TARGET_COL].astype(float) > 0).astype(int)
    return merged.reset_index(drop=True)


def time_split(
    df: pd.DataFrame, test_ratio: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split - avoids random-split leakage."""
    n = len(df)
    cut = int(n * (1 - test_ratio))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def save_processed(df: pd.DataFrame, name: str = "prepared.csv") -> Path:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    path = DATA_PROCESSED / name
    df.to_csv(path, index=False)
    meta = {"rows": len(df), "columns": list(df.columns)}
    (DATA_PROCESSED / "prepared_meta.json").write_text(json.dumps(meta, indent=2))
    return path


if __name__ == "__main__":
    prepared = prepare_dataset()
    path = save_processed(prepared)
    print(f"Prepared {len(prepared)} rows -> {path}")
