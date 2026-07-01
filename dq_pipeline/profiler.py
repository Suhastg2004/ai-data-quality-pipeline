"""
Data profiler for the DQ pipeline.

Computes per-column and dataset-level quality metrics on any DataFrame.
Designed to run before anomaly detection — it catches the obvious issues
(nulls, duplicates, invalid values) using pure statistics, no ML involved.

The profile output is a plain dictionary so it can be:
  - Printed to console
  - Written to JSON for downstream systems
  - Consumed by the report generator in Step 8
"""

import json
import numpy as np
import pandas as pd

import config


def profile_dataset(df: pd.DataFrame) -> dict:
    """
    Runs a full DQ profile on the given DataFrame.

    Computes:
        - Dataset-level metrics (row count, duplicate %, fault rate)
        - Per-column metrics (null %, type, stats for numerics, top values for categoricals)
        - Validity checks (negative quantities, invalid payment methods)

    Args:
        df: The DataFrame to profile (works on both clean and dirty data)

    Returns:
        A nested dictionary with all metrics. Structure:
        {
            "dataset": { ...overall metrics... },
            "columns": {
                "column_name": { ...per-column metrics... },
                ...
            },
            "validity": { ...rule-based checks... }
        }
    """

    profile = {
        "dataset": _profile_dataset_level(df),
        "columns": _profile_columns(df),
        "validity": _profile_validity(df),
    }

    return profile


def _profile_dataset_level(df: pd.DataFrame) -> dict:
    """
    Computes metrics that describe the dataset as a whole, not individual columns.
    These are the headline numbers you'd show a stakeholder first.
    """
    total_rows = len(df)

    # Duplicate detection: compare every row against every other row.
    # duplicated() returns True for every row that is an exact copy of a previous row.
    # The original is not marked — only the copies are.
    duplicate_mask = df.drop(
        columns=[config.COL_IS_ANOMALY, config.COL_ANOMALY_TYPE], errors="ignore"
    ).duplicated()
    duplicate_count = int(duplicate_mask.sum())

    # Fault rate: only meaningful if ground truth labels exist
    # If running on real data without labels, this will just be 0
    anomaly_count = int(df[config.COL_IS_ANOMALY].sum()) \
        if config.COL_IS_ANOMALY in df.columns else 0

    # Total null count across the entire dataset (all columns combined)
    total_nulls = int(df.isnull().sum().sum())

    # Completeness: % of cells that have a value (not null)
    total_cells = total_rows * len(df.columns)
    completeness_pct = round((1 - total_nulls / total_cells) * 100, 2)

    return {
        "total_rows": total_rows,
        "total_columns": len(df.columns),
        "total_null_cells": total_nulls,
        "completeness_pct": completeness_pct,
        "duplicate_rows": duplicate_count,
        "duplicate_pct": round(duplicate_count / total_rows * 100, 2),
        "anomalous_rows": anomaly_count,
        "fault_rate_pct": round(anomaly_count / total_rows * 100, 2),
    }


def _profile_columns(df: pd.DataFrame) -> dict:
    """
    Profiles each column individually.
    Numeric columns get statistical summaries.
    Categorical columns get frequency distributions.
    """
    column_profiles = {}

    # These are metadata columns — we profile the actual data columns only
    skip_cols = {config.COL_IS_ANOMALY, config.COL_ANOMALY_TYPE}

    for col in df.columns:
        if col in skip_cols:
            continue

        col_data = df[col]
        null_count = int(col_data.isnull().sum())
        null_pct = round(null_count / len(df) * 100, 2)
        distinct_count = int(col_data.nunique())

        base = {
            "dtype": str(col_data.dtype),
            "null_count": null_count,
            "null_pct": null_pct,
            "distinct_count": distinct_count,
        }

        if pd.api.types.is_numeric_dtype(col_data):
            # For numeric columns, compute full statistical summary.
            # These exact values (mean, std, Q1, Q3) feed into Z-score and IQR detectors later.
            non_null = col_data.dropna()
            q1 = float(non_null.quantile(0.25))
            q3 = float(non_null.quantile(0.75))
            iqr = q3 - q1

            base.update({
                "min":    round(float(non_null.min()), 2),
                "max":    round(float(non_null.max()), 2),
                "mean":   round(float(non_null.mean()), 2),
                "median": round(float(non_null.median()), 2),
                "std":    round(float(non_null.std()), 2),
                "q1":     round(q1, 2),
                "q3":     round(q3, 2),
                "iqr":    round(iqr, 2),
                # Skewness: measures how asymmetric the distribution is.
                # 0 = perfectly symmetric, >1 = heavily right-skewed (like our prices).
                # High skew here is why Z-score will struggle on price columns.
                "skewness": round(float(non_null.skew()), 4),
            })

        else:
            # For categorical/string columns, show the top 5 most frequent values.
            # This quickly surfaces things like dominant payment methods or busiest stores.
            top_values = (
                col_data.value_counts()
                        .head(5)
                        .rename_axis("value")
                        .reset_index(name="count")
                        .assign(pct=lambda x: (x["count"] / len(df) * 100).round(2))
                        .to_dict(orient="records")
            )
            base["top_values"] = top_values

        column_profiles[col] = base

    return column_profiles


def _profile_validity(df: pd.DataFrame) -> dict:
    """
    Rule-based validity checks — the things that are wrong by definition,
    not just statistically unusual.

    These are the checks your existing SQL tool likely already runs.
    The profiler makes them programmatic and quantified.
    """

    # Rule 1: Quantity must be positive (you can't sell -3 items)
    invalid_qty_mask = df[config.COL_QUANTITY] < 0
    invalid_qty_count = int(invalid_qty_mask.sum())

    # Rule 2: Payment method must be in the allowed list
    invalid_payment_mask = ~df[config.COL_PAYMENT_METHOD].isin(config.PAYMENT_METHODS)
    invalid_payment_count = int(invalid_payment_mask.sum())

    # Rule 3: Total amount should roughly equal quantity * unit_price
    # We allow for TOTAL_PRICE_NOISE tolerance since we added noise during generation.
    # Beyond 20% discrepancy is suspicious — could indicate data corruption.
    expected_total = df[config.COL_QUANTITY] * df[config.COL_UNIT_PRICE]
    price_mismatch_mask = (
        (df[config.COL_TOTAL_AMOUNT] - expected_total).abs() / expected_total.abs() > 0.20
    )
    price_mismatch_count = int(price_mismatch_mask.sum())

    return {
        "negative_quantity": {
            "count": invalid_qty_count,
            "pct": round(invalid_qty_count / len(df) * 100, 2),
            "rule": "quantity must be >= 0"
        },
        "invalid_payment_method": {
            "count": invalid_payment_count,
            "pct": round(invalid_payment_count / len(df) * 100, 2),
            "rule": f"payment_method must be one of {config.PAYMENT_METHODS}"
        },
        "total_amount_mismatch": {
            "count": price_mismatch_count,
            "pct": round(price_mismatch_count / len(df) * 100, 2),
            "rule": "total_amount should be within 20% of quantity * unit_price"
        },
    }


def print_profile(profile: dict) -> None:
    """
    Prints the profile in a readable format to the console.
    For a cleaner look than raw JSON dumps.
    """
    print("\n" + "=" * 60)
    print("DATASET-LEVEL SUMMARY")
    print("=" * 60)
    ds = profile["dataset"]
    print(f"  Total rows          : {ds['total_rows']}")
    print(f"  Total columns       : {ds['total_columns']}")
    print(f"  Completeness        : {ds['completeness_pct']}%")
    print(f"  Duplicate rows      : {ds['duplicate_rows']} ({ds['duplicate_pct']}%)")
    print(f"  Anomalous rows      : {ds['anomalous_rows']} ({ds['fault_rate_pct']}%)")

    print("\n" + "=" * 60)
    print("COLUMN-LEVEL SUMMARY")
    print("=" * 60)
    for col, stats in profile["columns"].items():
        print(f"\n  [{col}]  dtype: {stats['dtype']}")
        print(f"    Nulls       : {stats['null_count']} ({stats['null_pct']}%)")
        print(f"    Distinct    : {stats['distinct_count']}")
        if "mean" in stats:
            print(f"    Mean/Median : {stats['mean']} / {stats['median']}")
            print(f"    Std         : {stats['std']}")
            print(f"    Min/Max     : {stats['min']} / {stats['max']}")
            print(f"    Q1/Q3/IQR   : {stats['q1']} / {stats['q3']} / {stats['iqr']}")
            print(f"    Skewness    : {stats['skewness']}")
        if "top_values" in stats:
            top = stats["top_values"][:3]
            top_str = ", ".join([f"{t['value']} ({t['pct']}%)" for t in top])
            print(f"    Top values  : {top_str}")

    print("\n" + "=" * 60)
    print("VALIDITY CHECKS")
    print("=" * 60)
    for check, result in profile["validity"].items():
        status = "PASS" if result["count"] == 0 else "FAIL"
        print(f"  [{status}] {check}: {result['count']} rows ({result['pct']}%)")
        print(f"         Rule: {result['rule']}")


def save_profile(profile: dict, path: str = "output/profile_report.json") -> None:
    """Saves the profile dictionary as a JSON file for downstream consumption."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(profile, f, indent=2, default=str)
    print(f"\nProfile saved to {path}")


if __name__ == "__main__":
    from data_generator import generate_clean_data, inject_anomalies

    clean_df = generate_clean_data()
    dirty_df = inject_anomalies(clean_df)

    print("Profiling dirty dataset (with injected anomalies)...")
    profile = profile_dataset(dirty_df)

    print_profile(profile)
    save_profile(profile)