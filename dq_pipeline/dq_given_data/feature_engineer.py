# dq_given_data/feature_engineer.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import config


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["date"]      = pd.to_datetime(df["date"], errors="coerce")

    # Group 1: Math validity
    df["expected_sales"]   = df["quantity"] * df["unit_price"]
    df["sales_math_error"] = (
        (df["sales_amount"] - df["expected_sales"]).abs()
        / (df["sales_amount"].abs() + 1e-9)
    ).round(6)

    df["sales_per_unit"] = (
        df["sales_amount"] / (df["quantity"].abs() + 1e-9)
    ).round(6)
    df["sales_per_unit_deviation"] = (
        (df["sales_per_unit"] - df["unit_price"]).abs()
        / (df["unit_price"].abs() + 1e-9)
    ).round(6)

    # Group 2: Violation flags
    df["is_negative_price"]    = (df["unit_price"] < 0).astype(int)
    df["is_zero_quantity"]     = (df["quantity"] == 0).astype(int)
    df["is_negative_quantity"] = (df["quantity"] < 0).astype(int)

    discount_col = df["discount"].fillna(0)
    
    df["unauthorized_discount"] = (
        (discount_col > 0) & (df["is_promotion"] == 0)
    ).astype(int)

    # Group 3: Quantity outlier per product
    qty_stats = (
        df.groupby("product_id")["quantity"]
          .agg(qty_mean="mean", qty_std="std")
          .reset_index()
    )
    df = df.merge(qty_stats, on="product_id", how="left")
    df["qty_zscore"]   = (
        (df["quantity"] - df["qty_mean"]) / (df["qty_std"] + 1e-9)
    ).abs().round(4)
    df["log_quantity"] = np.log1p(df["quantity"].clip(lower=0))

    # Group 4: Time features
    df["hour_of_day"] = df["timestamp"].dt.hour.fillna(12).astype(int)
    df["day_of_week"] = df["timestamp"].dt.dayofweek.fillna(0).astype(int)
    df["month"]       = df["date"].dt.month.fillna(1).astype(int)
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)

    df["log_unit_price"]   = np.log1p(df["unit_price"].clip(lower=0))
    df["log_sales_amount"] = np.log1p(df["sales_amount"].clip(lower=0))

    # Group 5: Optional categoricals
    if "payment_type" in df.columns:
        le = LabelEncoder()
        df["payment_type_encoded"] = le.fit_transform(
            df["payment_type"].fillna("Unknown")
        )
    else:
        df["payment_type_encoded"] = 0

    if "store_type" in df.columns:
        le2 = LabelEncoder()
        df["store_type_encoded"] = le2.fit_transform(
            df["store_type"].fillna("Unknown")
        )
    else:
        df["store_type_encoded"] = 0

    # Group 6: Precomputed columns (dataset2 only)
    for col in config.ACTIVE_DATASET_PRECOMPUTED:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0

    return df


def get_feature_columns() -> list:
    base = [
        "sales_math_error", "expected_sales", "sales_per_unit_deviation",
        "is_negative_price", "is_zero_quantity", "is_negative_quantity",
        "unauthorized_discount", "qty_zscore", "log_quantity",
        "hour_of_day", "day_of_week", "month", "is_weekend",
        "log_unit_price", "log_sales_amount",
        "payment_type_encoded", "store_type_encoded",
    ]
    return base + config.ACTIVE_DATASET_PRECOMPUTED


def prepare_training_data(df: pd.DataFrame):
    print("  Engineering features...")
    featured_df  = engineer_features(df)
    feature_cols = get_feature_columns()
    X = featured_df[feature_cols]
    y = featured_df[config.O_ANOMALY_FLAG]

    print(f"  Features : {X.shape[1]}")
    print(f"  Rows     : {X.shape[0]:,}")
    print(f"  Anomalies: {int(y.sum()):,} ({y.mean()*100:.2f}%)")
    return X, y, featured_df


if __name__ == "__main__":
    from data_loader import load_training_data
    import os # Add this import for directory management
    
    print("\n" + "=" * 55)
    print(f"  FEATURE ENGINEER — {config.ACTIVE_DATASET.upper()}")
    print("=" * 55)
    
    df = load_training_data()
    X, y, featured_df = prepare_training_data(df)
    
    print(f"\n  Feature columns ({len(get_feature_columns())}):")
    for col in get_feature_columns():
        print(f"    {col}")
        
    nulls = X.isnull().sum()
    nulls = nulls[nulls > 0]
    print("\n  Null check:")
    print("    No nulls." if len(nulls) == 0 else nulls.to_string())
    
    # ---------------------------------------------------------
    # NEW CODE: Save the feature-engineered dataset to CSV
    # ---------------------------------------------------------
    # output_dir = "dq_given_data/output"
    # os.makedirs(output_dir, exist_ok=True) 
    
    # # You can name this based on the active dataset in config
    # output_file = f"{output_dir}/engineered_{config.ACTIVE_DATASET}.csv"
    
    # featured_df.to_csv(output_file, index=False)
    # print(f"\n  [+] Saved engineered dataset to: {output_file}")
    # ---------------------------------------------------------

    print("\n" + "=" * 55)
    print("  FEATURE ENGINEER COMPLETE")
    print("=" * 55)