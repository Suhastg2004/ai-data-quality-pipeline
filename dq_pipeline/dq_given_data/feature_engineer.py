# dq_given_data/feature_engineer.py

"""
Feature engineering for the DQ pipeline.

Takes the enriched master DataFrame from data_loader.py and produces
a numeric feature matrix ready for ML model training or scoring.

Why does feature engineering matter so much?
    The model never sees raw data — it sees numbers.
    A transaction like "qty=0, price=5.67, sales=22.68" needs to become
    a number like "sales_math_error=1.0" before a model can learn from it.
    Every anomaly type we discovered maps to one or more features here.

Anomaly types we're targeting and their feature:
    Type 1 (negative_unit_price) → is_negative_price flag
    Type 2 (wrong_price)         → price_deviation from catalog
    Type 4 (zero_quantity)       → is_zero_quantity flag
    Type 5 (unauthorized_discount) → unauthorized_discount flag
    Type 6 (quantity_outlier)    → quantity zscore, sales_per_unit mismatch
    Type 7 (sales_math_error)    → sales_math_error ratio
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import config


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes all ML features from the enriched master DataFrame.

    Adds new columns to the DataFrame — does not remove original columns.
    Call get_feature_columns() to get the list of columns the model uses.

    Args:
        df: Enriched DataFrame from data_loader.enrich_orders()

    Returns:
        DataFrame with all engineered feature columns added.
    """
    df = df.copy()

    # Ensure timestamp is parsed
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"]      = pd.to_datetime(df["date"])


    # ── Group 1: Math / Validity Features ─────────────────────────────────────
    # These catch anomaly types 1, 2, 4, 7
    # They are deterministic — the model doesn't need to guess, the math is wrong

    # Expected sales based on qty × price
    # For type 7 (sales_math_error): qty=20, price=0.71, sales=0.71
    # expected = 20 × 0.71 = 14.2, actual = 0.71 → error = 95%
    df["expected_sales"] = df["quantity"] * df["unit_price"]

    df["sales_math_error"] = (
        (df["sales_amount"] - df["expected_sales"]).abs()
        / (df["sales_amount"].abs() + 1e-9)
    ).round(6)
    # 1e-9 prevents division by zero when sales_amount = 0

    # Price deviation from catalog selling price
    # For type 2 (wrong_price): if selling_price=5.99 but unit_price=8.50
    # deviation = (8.50 - 5.99) / 5.99 = 0.42 → 42% above catalog
    df["price_deviation"] = (
        (df["unit_price"] - df["selling_price"])
        / (df["selling_price"].abs() + 1e-9)
    ).round(6)

    # Absolute price deviation (direction doesn't matter for flagging)
    df["abs_price_deviation"] = df["price_deviation"].abs()

    # Actual margin vs expected margin
    # If catalog says margin should be 24% but actual is 80% — something is wrong
    df["actual_margin"] = (
        (df["unit_price"] - df["cost_price"])
        / (df["unit_price"].abs() + 1e-9)
    ).round(6)

    df["margin_deviation"] = (
        df["actual_margin"] - df["margin_pct"]
    ).abs().round(6)

    # Sales amount per unit — should be close to unit_price
    # For type 6 (qty_outlier): qty=120, sales=27.78
    # sales_per_unit = 27.78/120 = 0.23, but unit_price = 4.63 → 95% off
    df["sales_per_unit"] = (
        df["sales_amount"] / (df["quantity"].abs() + 1e-9)
    ).round(6)

    df["sales_per_unit_deviation"] = (
        (df["sales_per_unit"] - df["unit_price"]).abs()
        / (df["unit_price"].abs() + 1e-9)
    ).round(6)


    # ── Group 2: Direct Violation Flags ───────────────────────────────────────
    # Binary 0/1 flags for things that are wrong by definition
    # These give the model an explicit "this is illegal" signal

    # Type 1: negative unit price — physically impossible
    df["is_negative_price"] = (df["unit_price"] < 0).astype(int)

    # Type 4: zero quantity — can't sell nothing
    df["is_zero_quantity"] = (df["quantity"] == 0).astype(int)

    # Negative quantity — also impossible
    df["is_negative_quantity"] = (df["quantity"] < 0).astype(int)

    # Type 5: discount applied but no promotion flagged
    # O0000006: discount=0.5, is_promotion=0 → unauthorized
    discount_col = df["discount"].fillna(0)
    df["unauthorized_discount"] = (
        (discount_col > 0) & (df["is_promotion"] == 0)
    ).astype(int)

    # Discount exists but no active promotion in the promotions table either
    df["discount_no_active_promo"] = (
        (discount_col > 0) & (df["promo_discount_pct"] == 0)
    ).astype(int)

    # Promotion flagged in order but no promotion found in promotions table
    df["promo_flag_no_promo_record"] = (
        (df["is_promotion"] == 1) & (df["promo_discount_pct"] == 0)
    ).astype(int)


    # ── Group 3: Quantity Outlier Features ────────────────────────────────────
    # For type 6: qty=120 is extreme — use statistical distance from normal

    # Z-score of quantity per product
    # Why per product? A qty of 20 might be normal for tissue paper
    # but extreme for an expensive item. Per-product baseline is fairer.
    qty_stats = (
        df.groupby("product_id")["quantity"]
          .agg(qty_mean="mean", qty_std="std")
          .reset_index()
    )
    df = df.merge(qty_stats, on="product_id", how="left")

    df["qty_zscore"] = (
        (df["quantity"] - df["qty_mean"])
        / (df["qty_std"] + 1e-9)
    ).abs().round(4)

    # Raw quantity as a feature too — model can learn its own threshold
    df["log_quantity"] = np.log1p(df["quantity"].clip(lower=0))


    # ── Group 4: Time / Store Features ────────────────────────────────────────
    # Context features — not wrong by themselves but important signals

    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek   # 0=Mon, 6=Sun
    df["month"]       = df["timestamp"].dt.month
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)

    # Off-hours transaction at a store that is not open 24x7
    # A transaction at 2am at a non-24hr store is suspicious
    df["is_off_hours"] = (
        ((df["hour_of_day"] < 6) | (df["hour_of_day"] > 22))
        & (df["open_24x7"] == 0)
    ).astype(int)

    # Store age in days at the time of the transaction
    # New stores have less transaction history — context for baseline anomalies
    df["store_open_date"] = pd.to_datetime(
        df["opening_year"].astype(str) + "-01-01"
    )
    df["store_age_days"] = (df["date"] - df["store_open_date"]).dt.days

    # Transaction value vs store's expected average
    # If a store's avg transaction is $14 and this one is $800, that's unusual
    df["txn_vs_store_avg"] = (
        (df["sales_amount"] - df["avg_transaction_value"])
        / (df["avg_transaction_value"] + 1e-9)
    ).round(4)


    # ── Group 5: Weather / Holiday Context ────────────────────────────────────
    # These help the model understand if an anomaly is explained by context
    # High sales on a holiday = expected. High sales on a random Tuesday = suspicious.

    df["is_holiday"]   = df["is_holiday"].fillna(0).astype(int)
    df["temperature"]  = df["temperature"].fillna(df["temperature"].median())
    df["rainfall"]     = df["rainfall"].fillna(0)
    df["humidity"]     = df["humidity"].fillna(df["humidity"].median())


    # ── Group 6: Product Context Features ─────────────────────────────────────

    # Seasonal mismatch — selling a Summer product in winter
    # products.seasonality = "Summer" but month is December
    summer_months = [6, 7, 8]
    winter_months = [12, 1, 2]

    df["is_summer_product"] = (df["seasonality"] == "Summer").astype(int)
    df["is_winter_product"] = (df["seasonality"] == "Winter").astype(int)
    df["is_all_season"]     = (df["seasonality"] == "All").astype(int)

    df["seasonal_mismatch"] = (
        ((df["is_summer_product"] == 1) & (df["month"].isin(winter_months))) |
        ((df["is_winter_product"] == 1) & (df["month"].isin(summer_months)))
    ).astype(int)

    # Refrigerated product — higher sensitivity to data errors
    df["is_refrigerated"] = (df["storage_type"] == "Refrigerated").astype(int)

    # Log-transform price features to reduce skew for LR
    df["log_unit_price"]   = np.log1p(df["unit_price"].clip(lower=0))
    df["log_sales_amount"] = np.log1p(df["sales_amount"].clip(lower=0))
    df["log_selling_price"] = np.log1p(df["selling_price"].clip(lower=0))

    # Encode categorical columns as numbers
    # Logistic Regression needs all numbers — can't handle strings
    le_category   = LabelEncoder()
    le_store_type = LabelEncoder()
    le_payment    = LabelEncoder()
    le_region     = LabelEncoder()

    df["category_encoded"]    = le_category.fit_transform(df["category"].fillna("Unknown"))
    df["store_type_encoded"]  = le_store_type.fit_transform(df["store_type"].fillna("Unknown"))
    df["payment_type_encoded"] = le_payment.fit_transform(df["payment_type"].fillna("Unknown"))
    df["region_encoded"]      = le_region.fit_transform(df["region"].fillna("Unknown"))

    # ── Group 7: Inventory Features ───────────────────────────────────────────
    # These catch contradictions between order data and inventory records.
    # No other table can catch these — inventory is the only source of truth
    # for what physically existed in the store on a given day.

    # Stock math error from inventory reconciliation
    # If inventory says stock_math_error > 0, inventory records are corrupt
    # That same store/product/day having an order is doubly suspicious
    df["inv_stock_math_error"] = df["stock_math_error"].fillna(0)

    # Order quantity vs what inventory says was sold
    # If order qty=50 but inventory sold_qty=2 — they contradict each other
    df["qty_vs_inv_sold"] = (
        (df["quantity"] - df[config.INV_SOLD_QTY]).abs()
    ).fillna(0)

    # Did the order quantity exceed available stock?
    # available = begin_stock + received_qty
    # If order qty > available → physically impossible to fulfil
    df["order_exceeds_stock"] = (
        (df["quantity"] > df["stock_available"])
        & (df["stock_available"] > 0)  # only flag when we have inventory data
    ).astype(int)

    # Stock availability ratio — how much of available stock did this order use?
    # A single order consuming 80%+ of daily stock is unusual
    df["stock_consumption_ratio"] = (
        df["quantity"] / (df["stock_available"] + 1e-9)
    ).clip(upper=10).round(4)
    # clip at 10 to prevent extreme values when stock_available is tiny
    
    return df


def get_feature_columns() -> list:
    """
    Returns the exact list of columns the ML model trains and predicts on.

    Centralised here so training and scoring always use identical features.
    If you add a feature in engineer_features(), add it here too.

    Why this separation?
        When you load the trained model 6 months later and score new data,
        you call get_feature_columns() to ensure you pass the exact same
        features the model was trained on. Without this, column mismatches
        cause silent wrong predictions.
    """
    return [
        # Math / validity
        "sales_math_error",
        "price_deviation",
        "abs_price_deviation",
        "margin_deviation",
        "sales_per_unit_deviation",
        "expected_sales",

        # Direct violation flags
        "is_negative_price",
        "is_zero_quantity",
        "is_negative_quantity",
        "unauthorized_discount",
        "discount_no_active_promo",
        "promo_flag_no_promo_record",

        # Quantity features
        "qty_zscore",
        "log_quantity",

        # Time / store features
        "hour_of_day",
        "day_of_week",
        "month",
        "is_weekend",
        "is_off_hours",
        "store_age_days",
        "txn_vs_store_avg",

        # Weather / holiday
        "is_holiday",
        "temperature",
        "rainfall",
        "humidity",

        # Inventory features
        "inv_stock_math_error",
        "qty_vs_inv_sold",
        "order_exceeds_stock",
        "stock_consumption_ratio",
        
        # Product context
        "seasonal_mismatch",
        "is_refrigerated",
        "is_summer_product",
        "is_winter_product",

        # Log-transformed prices
        "log_unit_price",
        "log_sales_amount",
        "log_selling_price",

        # Categorical encodings
        "category_encoded",
        "store_type_encoded",
        "payment_type_encoded",
        "region_encoded",
    ]


def prepare_training_data(df: pd.DataFrame):
    """
    Engineers features on the labelled dataset and returns
    X (feature matrix) and y (target) ready for sklearn.

    Args:
        df: Enriched labelled DataFrame from data_loader.load_training_data()

    Returns:
        X: DataFrame of features
        y: Series of labels (0=normal, 1=anomaly)
        featured_df: Full DataFrame with all engineered columns (for inspection)
    """
    print("  Engineering features...")
    featured_df  = engineer_features(df)
    feature_cols = get_feature_columns()

    X = featured_df[feature_cols]
    y = featured_df[config.O_ANOMALY_FLAG]

    print(f"  Feature matrix : {X.shape[0]:,} rows × {X.shape[1]} features")
    print(f"  Anomalies      : {int(y.sum()):,} ({y.mean()*100:.2f}%)")
    print(f"  Normal rows    : {int((y==0).sum()):,}")

    return X, y, featured_df


def prepare_scoring_data(df: pd.DataFrame):
    """
    Engineers features on unlabelled production data for scoring.
    No y returned — there are no labels on real data.

    Args:
        df: Enriched DataFrame from data_loader.load_scoring_data()

    Returns:
        X: DataFrame of features
        featured_df: Full DataFrame with all engineered columns
    """
    print("  Engineering features for scoring...")
    featured_df  = engineer_features(df)
    feature_cols = get_feature_columns()

    X = featured_df[feature_cols]
    print(f"  Scoring matrix : {X.shape[0]:,} rows × {X.shape[1]} features")

    return X, featured_df


if __name__ == "__main__":
    from data_loader import load_training_data

    print("\n" + "=" * 55)
    print("  FEATURE ENGINEER — VERIFICATION RUN")
    print("=" * 55)

    print("\nLoading training data...")
    train_df = load_training_data()

    print("\nEngineering features...")
    X, y, featured_df = prepare_training_data(train_df)

    print("\n[Feature matrix sample — first 3 rows]")
    print(X.head(3).to_string())

    print("\n[Feature stats]")
    print(X.describe().round(3).to_string())

    print("\n[Null check — features with missing values]")
    null_counts = X.isnull().sum()
    null_counts = null_counts[null_counts > 0]
    if len(null_counts) == 0:
        print("  No nulls in feature matrix — clean.")
    else:
        print(null_counts.to_string())

    print("\n[Flag features — how many rows triggered each flag]")
    flag_cols = [
        "is_negative_price", "is_zero_quantity", "is_negative_quantity",
        "unauthorized_discount", "discount_no_active_promo",
        "promo_flag_no_promo_record", "is_off_hours", "seasonal_mismatch",
    ]
    for col in flag_cols:
        if col in featured_df.columns:
            count = int(featured_df[col].sum())
            pct   = count / len(featured_df) * 100
            print(f"  {col:35s}: {count:>6,} rows ({pct:.2f}%)")

    print("\n" + "=" * 55)
    print("  FEATURE ENGINEER COMPLETE")
    print("=" * 55)