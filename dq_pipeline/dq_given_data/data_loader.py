# dq_pipeline/data_loader.py

"""
Loads all 7 CSV datasets, validates them, and joins them into one
enriched master DataFrame — one row per order, with full context attached.

Why join everything here rather than in feature_engineer.py?
    Keeping joins separate from feature creation makes both easier to debug.
    If a join silently drops rows, we catch it here with assertions.
    If a feature formula is wrong, we fix it in feature_engineer.py.
    One responsibility per file.

Output of this file:
    master_df — enriched orders DataFrame with columns from all 7 tables
    Each row = one transaction with store, product, promotion,
    weather, and holiday context already attached.
"""

import datetime
import pandas as pd
import config


# ── Step 1: Load raw CSVs ──────────────────────────────────────────────────────

def load_raw_tables() -> dict:
    """
    Loads all 7 CSVs into separate DataFrames.
    Validates that expected columns exist in each table.
    Parses date columns to proper datetime objects.

    Returns a dictionary of table_name -> DataFrame.
    """

    print("  Loading raw CSV files...")

    tables = {
    "orders":           pd.read_csv(config.PATH_ORDERS),
    "orders_anomalies": pd.read_csv(config.PATH_ORDERS_ANOMALIES),
    "products":         pd.read_csv(config.PATH_PRODUCTS),
    "stores":           pd.read_csv(config.PATH_STORES),
    "promotions":       pd.read_csv(config.PATH_PROMOTIONS),
    "weather":          pd.read_csv(config.PATH_WEATHER),
    "holidays":         pd.read_csv(config.PATH_HOLIDAYS),
    "inventory":        pd.read_csv(config.PATH_INVENTORY),  # ADD THIS
    }

    # Print row counts so you can see what was loaded
    for name, df in tables.items():
        print(f"    {name:20s}: {len(df):>7,} rows  |  {len(df.columns)} columns")

    # Validate that critical columns exist in each table
    # If a column is missing, better to fail here with a clear message
    # than silently produce wrong features later
    _validate_columns(tables)

    # Parse date columns — they come in as strings from CSV
    # pandas needs them as actual date objects for comparisons and joins
    tables["orders"]["date"]           = pd.to_datetime(tables["orders"]["date"])
    tables["orders"]["timestamp"]      = pd.to_datetime(tables["orders"]["timestamp"])

    tables["orders_anomalies"]["date"]      = pd.to_datetime(tables["orders_anomalies"]["date"])
    tables["orders_anomalies"]["timestamp"] = pd.to_datetime(tables["orders_anomalies"]["timestamp"])

    tables["weather"]["date"]    = pd.to_datetime(tables["weather"]["date"])
    tables["holidays"]["date"]   = pd.to_datetime(tables["holidays"]["date"])

    # Fill missing discount column with 0 in orders
    # orders.csv doesn't have a discount column — normal orders have no discount
    if "discount" not in tables["orders"].columns:
        tables["orders"]["discount"] = 0.0

    if "discount" not in tables["orders_anomalies"].columns:
        tables["orders_anomalies"]["discount"] = 0.0

    # Fill nulls in discount with 0
    tables["orders"]["discount"]           = tables["orders"]["discount"].fillna(0.0)
    tables["orders_anomalies"]["discount"] = tables["orders_anomalies"]["discount"].fillna(0.0)

    print("  All tables loaded and parsed.\n")
    return tables


def _validate_columns(tables: dict) -> None:
    """
    Checks that the columns we expect in each table actually exist.
    Raises a clear error if something is missing — much better than
    a cryptic KeyError 3 steps later.
    """
    required = {
        "orders": [
            config.O_ORDER_ID, config.O_DATE, config.O_STORE_ID,
            config.O_PRODUCT_ID, config.O_QUANTITY,
            config.O_UNIT_PRICE, config.O_SALES_AMOUNT,
        ],
        "orders_anomalies": [
            config.O_ORDER_ID, config.O_DATE, config.O_ANOMALY_FLAG,
        ],
        "products": [
            config.P_PRODUCT_ID, config.P_SELLING_PRICE,
            config.P_COST_PRICE, config.P_SEASONALITY,
        ],
        "stores": [
            config.S_STORE_ID, config.S_CITY,
            config.S_OPEN_24X7, config.S_OPENING_YEAR,
        ],
        "promotions": [
            config.PR_PRODUCT_ID, config.PR_START_OFFSET,
            config.PR_DURATION, config.PR_DISCOUNT_PCT,
        ],
        "weather": [config.W_DATE, config.W_CITY, config.W_TEMPERATURE],
        "holidays": [config.H_DATE, config.H_IS_HOLIDAY],
    }

    for table_name, cols in required.items():
        df = tables[table_name]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Table '{table_name}' is missing columns: {missing}\n"
                f"Found columns: {list(df.columns)}"
            )


# ── Step 2: Prepare the promotion lookup ──────────────────────────────────────

def build_promotion_lookup(promotions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts promotions from offset-based to date-based, then expands
    each promotion into one row per day it is active.

    Why expand to one row per day?
        The promotions table uses start_offset + duration_days, not
        actual dates. To join against orders (which have actual dates),
        we need to know "is product P00361 on promotion on 2024-04-14?"
        Expanding to daily rows lets us do a simple date + product_id join.

    Example:
        promotion_id=PR15795, product_id=P00361, start_offset=102, duration=7
        Reference date = 2024-01-01
        start_date = 2024-01-01 + 102 days = 2024-04-12
        end_date   = 2024-04-12 + 7 days   = 2024-04-19
        Expanded to 7 rows: one for each date from 2024-04-12 to 2024-04-18

    Returns:
        DataFrame with columns: product_id, date, promo_discount_pct
        One row per (product, active_promotion_date)
    """
    ref = config.PROMO_REFERENCE_DATE
    rows = []

    for _, promo in promotions_df.iterrows():
        start_date = ref + datetime.timedelta(days=int(promo[config.PR_START_OFFSET]))
        end_date   = start_date + datetime.timedelta(days=int(promo[config.PR_DURATION]))

        # Generate one row for each active date
        current = start_date
        while current < end_date:
            rows.append({
                "product_id":       promo[config.PR_PRODUCT_ID],
                "date":             pd.Timestamp(current),
                "promo_discount_pct": promo[config.PR_DISCOUNT_PCT],
            })
            current += datetime.timedelta(days=1)

    promo_lookup = pd.DataFrame(rows)
    print(f"  Promotion lookup built: {len(promo_lookup):,} product-day combinations")
    return promo_lookup

def build_inventory_lookup(inventory_df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts inventory from offset-based dates to actual calendar dates.

    inventory.date_offset works the same as promotions.start_offset —
    it's the number of days since PROMO_REFERENCE_DATE (2024-01-01).

    After conversion, we compute two pre-calculated columns:
        stock_available : begin_stock + received_qty (total available that day)
        stock_math_ok   : 1 if end_stock = begin + received - sold, else 0

    Returns one row per (store_id, product_id, date) — joinable to orders.
    """
    import datetime
    ref = config.PROMO_REFERENCE_DATE
    inv = inventory_df.copy()

    # Convert offset to actual date
    inv["date"] = inv[config.INV_DATE_OFFSET].apply(
        lambda x: pd.Timestamp(ref + datetime.timedelta(days=int(x)))
    )

    # Pre-compute stock health columns
    inv["stock_available"] = (
        inv[config.INV_BEGIN_STOCK] + inv[config.INV_RECEIVED_QTY]
    )

    inv["stock_math_error"] = (
        inv[config.INV_END_STOCK]
        - (inv[config.INV_BEGIN_STOCK]
           + inv[config.INV_RECEIVED_QTY]
           - inv[config.INV_SOLD_QTY])
    ).abs()

    # Keep only columns needed for the join
    result = inv[[
        "store_id", "product_id", "date",
        "stock_available",
        "stock_math_error",
        config.INV_BEGIN_STOCK,
        config.INV_SOLD_QTY,
    ]].copy()

    print(f"  Inventory lookup built: {len(result):,} store-product-day combinations")
    return result


# ── Step 3: Enrich orders with all table context ───────────────────────────────

def enrich_orders(orders_df: pd.DataFrame, tables: dict) -> pd.DataFrame:
    """
    Joins all 7 tables onto the orders DataFrame to produce one
    enriched row per transaction.

    Join order and why:
        1. products  — every order has a product_id, always present
        2. stores    — every order has a store_id, always present
        3. promotions — not every product is on promo every day (left join)
        4. weather   — join on store city + date (left join — some cities may not match)
        5. holidays  — join on date (left join — not every date is a holiday)

    After every join, we assert the row count hasn't changed.
    A join that drops rows means a foreign key exists in orders but
    not in the reference table — that itself is a DQ issue.

    Args:
        orders_df: raw orders DataFrame (either orders or orders_anomalies)
        tables:    dictionary of all loaded raw tables

    Returns:
        Enriched DataFrame with all context columns attached
    """
    df = orders_df.copy()
    original_count = len(df)

    # Join 1: products
    # Select only the product columns we need — avoids column name collisions
    product_cols = [
        config.P_PRODUCT_ID,
        config.P_SELLING_PRICE,
        config.P_COST_PRICE,
        config.P_MARGIN_PCT,
        config.P_SHELF_LIFE,
        config.P_STORAGE_TYPE,
        config.P_SEASONALITY,
        config.P_CATEGORY,
    ]
    df = df.merge(
        tables["products"][product_cols],
        on="product_id",
        how="left",
    )
    _assert_row_count(df, original_count, "products join")

    # Join 2: stores
    store_cols = [
        config.S_STORE_ID,
        config.S_CITY,
        config.S_STORE_TYPE,
        config.S_OPEN_24X7,
        config.S_OPENING_YEAR,
        config.S_AVG_TXN_VALUE,
        config.S_REGION,
    ]
    df = df.merge(
        tables["stores"][store_cols],
        on="store_id",
        how="left",
    )
    _assert_row_count(df, original_count, "stores join")

    # Join 3: promotions (left join — not every product is on promo today)
    promo_lookup = build_promotion_lookup(tables["promotions"])

    # Ensure date types match before joining
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    promo_lookup["date"] = pd.to_datetime(promo_lookup["date"]).dt.normalize()

    df = df.merge(
        promo_lookup,
        on=["product_id", "date"],
        how="left",
    )
    _assert_row_count(df, original_count, "promotions join")

    # promo_discount_pct is NaN if no promotion is active — fill with 0
    df["promo_discount_pct"] = df["promo_discount_pct"].fillna(0.0)

    # Join 4: weather (join on store city + date)
    # We need the store's city to match weather data
    # Rename weather city to avoid confusion
    weather = tables["weather"].rename(columns={"date": "date"})
    weather["date"] = pd.to_datetime(weather["date"]).dt.normalize()

    df = df.merge(
        weather,
        left_on=["city", "date"],
        right_on=[config.W_CITY, config.W_DATE],
        how="left",
    )
    _assert_row_count(df, original_count, "weather join")

    # Join 5: holidays (join on date only)
    holidays = tables["holidays"][[config.H_DATE, config.H_IS_HOLIDAY, config.H_HOLIDAY_NAME]].copy()
    holidays["date"] = pd.to_datetime(holidays["date"]).dt.normalize()

    # A date can appear multiple times in holidays (different regions)
    # Keep just one row per date — take max is_holiday
    holidays = holidays.groupby("date").agg(
        is_holiday=("is_holiday", "max"),
        holiday_name=("holiday_name", "first"),
    ).reset_index()

    df = df.merge(
        holidays,
        on="date",
        how="left",
    )
    _assert_row_count(df, original_count, "holidays join")

    # Fill holiday nulls — dates not in holidays table are not holidays
    df["is_holiday"]    = df["is_holiday"].fillna(0).astype(int)
    df["holiday_name"]  = df["holiday_name"].fillna("")

    # Join 6: inventory (left join — not every product has inventory data)
    inv_lookup = build_inventory_lookup(tables["inventory"])
    inv_lookup["date"] = pd.to_datetime(inv_lookup["date"]).dt.normalize()

    df = df.merge(
        inv_lookup,
        on=["store_id", "product_id", "date"],
        how="left",
    )
    _assert_row_count(df, original_count, "inventory join")

    # Fill nulls — products with no inventory record get 0 for these
    df["stock_available"]  = df["stock_available"].fillna(0)
    df["stock_math_error"] = df["stock_math_error"].fillna(0)
    df[config.INV_BEGIN_STOCK] = df[config.INV_BEGIN_STOCK].fillna(0)
    df[config.INV_SOLD_QTY]    = df[config.INV_SOLD_QTY].fillna(0)
    
    print(f"  Enriched {len(df):,} orders with context from all tables")
    return df


def _assert_row_count(df: pd.DataFrame, expected: int, step: str) -> None:
    """
    Checks that a join didn't silently drop or duplicate rows.
    A dropped row means a foreign key in orders has no match in the reference table.
    A duplicated row means the reference table has multiple matches (many-to-one issue).
    Either is a DQ problem worth knowing about.
    """
    actual = len(df)
    if actual != expected:
        print(
            f"  WARNING: Row count changed after {step}. "
            f"Expected {expected:,}, got {actual:,}. "
            f"Difference: {actual - expected:+,}"
        )
    else:
        print(f"  OK: {step} — row count stable at {actual:,}")


# ── Step 4: Public entry point ─────────────────────────────────────────────────

def load_training_data() -> pd.DataFrame:
    """
    Loads and enriches orders_with_anomalies — the labelled training dataset.
    This is what the ML model trains and evaluates on.

    Returns:
        Enriched DataFrame with anomaly_flag and anomaly_type columns intact.
    """
    tables = load_raw_tables()
    master = enrich_orders(tables["orders_anomalies"], tables)
    return master


def load_scoring_data() -> pd.DataFrame:
    """
    Loads and enriches orders — the unlabelled production dataset.
    This is what the trained model scores to find real anomalies.

    Returns:
        Enriched DataFrame without anomaly labels (those don't exist in production).
    """
    tables = load_raw_tables()
    master = enrich_orders(tables["orders"], tables)
    return master


# ── Main: run standalone to verify joins ──────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  DATA LOADER — VERIFICATION RUN")
    print("=" * 55)

    print("\n[Training data — orders_with_anomalies]")
    train_df = load_training_data()

    print(f"\n  Final shape  : {train_df.shape}")
    print(f"  Columns      : {list(train_df.columns)}")
    print(f"\n  Anomaly breakdown:")
    print(train_df["anomaly_flag"].value_counts().to_string())

    if "anomaly_type" in train_df.columns:
        print(f"\n  Anomaly types:")
        print(train_df["anomaly_type"].value_counts().to_string())

    print("\n[Scoring data — orders]")
    score_df = load_scoring_data()
    print(f"  Final shape  : {score_df.shape}")

    print("\n[Sample enriched row]")
    sample_cols = [
        "order_id", "date", "store_id", "product_id",
        "unit_price", "selling_price", "city",
        "temperature", "is_holiday", "promo_discount_pct",
    ]
    available = [c for c in sample_cols if c in train_df.columns]
    print(train_df[available].head(3).to_string())

    print("\n" + "=" * 55)
    print("  DATA LOADER COMPLETE")
    print("=" * 55)