"""Data leakage and feature exclusion checks."""
from __future__ import annotations

from src.data_prep import FEATURE_COLS, TIME_COL, prepare_dataset, time_split


def test_chronological_split_no_time_leakage():
    df = prepare_dataset(sample_n=2000)
    assert df[TIME_COL].is_monotonic_increasing
    train_df, test_df = time_split(df, test_ratio=0.2)
    assert not train_df.empty and not test_df.empty
    # Split is index-based on time-sorted frame - no test row precedes train row.
    assert int(train_df.index.max()) < int(test_df.index.min())


def test_proxy_flags_excluded_from_feature_cols():
    excluded = {
        "multiple_accounts_flag",
        "refund_to_different_account",
        "address_change_before_delivery",
    }
    for col in excluded:
        assert col not in FEATURE_COLS
