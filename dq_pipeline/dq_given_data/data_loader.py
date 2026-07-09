# dq_given_data/data_loader.py

import pandas as pd
import config


def load_training_data() -> pd.DataFrame:
    path    = config.ACTIVE_DATASET_PATH
    col_map = config.ACTIVE_DATASET_COL_MAP

    df = pd.read_csv(path)
    print(f"  Loaded : {path}")
    print(f"  Shape  : {df.shape[0]:,} rows × {df.shape[1]} columns")

    rename_map = {v: k for k, v in col_map.items() if v in df.columns}
    df = df.rename(columns=rename_map)

    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], dayfirst=True, errors="coerce"
        )
    else:
        df["timestamp"] = df["date"]

    if "discount" not in df.columns:
        df["discount"] = 0.0
    df["discount"] = df["discount"].fillna(0.0)

    required = ["date", "store_id", "product_id", "quantity",
                "unit_price", "sales_amount", "anomaly_flag"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns after renaming: {missing}\n"
            f"Check ACTIVE_DATASET_COL_MAP in config.py.\n"
            f"Available columns: {list(df.columns)}"
        )

    print(f"  Anomalies: {int(df['anomaly_flag'].sum()):,} "
          f"({df['anomaly_flag'].mean()*100:.2f}%)")
    return df


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print(f"  DATA LOADER — {config.ACTIVE_DATASET.upper()}")
    print("=" * 55)

    df = load_training_data()
    print(f"\n  Columns : {list(df.columns)}")
    print(f"\n  Anomaly breakdown:")
    print(df["anomaly_flag"].value_counts().to_string())

    if "anomaly_type" in df.columns:
        print(f"\n  Anomaly types:")
        print(df["anomaly_type"].value_counts().to_string())

    show_cols = ["store_id", "product_id", "date",
                 "quantity", "unit_price", "sales_amount",
                 "anomaly_flag", "anomaly_type"]
    available = [c for c in show_cols if c in df.columns]
    print(f"\n  Sample rows:")
    print(df[available].head(3).to_string())
    print("\n" + "=" * 55)
    print("  DATA LOADER COMPLETE")
    print("=" * 55)