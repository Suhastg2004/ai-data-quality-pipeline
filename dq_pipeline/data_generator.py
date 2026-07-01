"""
Generates synthetic retail transaction data and injects configurable DQ issues.

Two public functions:
    generate_clean_data()  -> DataFrame of realistic, problem-free transactions
    inject_anomalies(df)   -> Same DataFrame with DQ issues + ground truth labels added

Kept separate so the clean baseline can be inspected independently.
"""

import uuid
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import config


def generate_clean_data() -> pd.DataFrame:
    """
    Generates NUM_ROWS of clean retail transactions using realistic distributions.
    No nulls, no duplicates, no outliers — pure normal data.
    
    Returns a DataFrame with all columns defined in config, plus is_anomaly=0
    and anomaly_type="" for every row (will be updated during injection).
    """
    rng = np.random.default_rng(config.RANDOM_SEED)
    n = config.NUM_ROWS

    # Transaction IDs — universally unique, no collisions possible
   # REPLACE with this
    transaction_ids = [str(uuid.uuid4()) for _ in range(n)]

    # Stores — randomly assigned, weighted so some stores are busier than others
    # rng.choice picks from the list; p= sets probability weights
    store_weights = rng.dirichlet(np.ones(len(config.STORES)))
    store_ids = rng.choice(config.STORES, size=n, p=store_weights)

    # Timestamps — spread across the last 90 days, random time within each day
    base_date = datetime(2024, 1, 1)
    timestamps = [
        base_date + timedelta(days=int(rng.integers(0, 90)),
                              hours=int(rng.integers(8, 22)),
                              minutes=int(rng.integers(0, 60)))
        for _ in range(n)
    ]

    # SKUs — simple product codes like "SKU_0042"
    skus = [f"SKU_{rng.integers(1, 500):04d}" for _ in range(n)]

    # Categories — uniform random pick from the list
    categories = rng.choice(config.CATEGORIES, size=n)

    # Quantity — Poisson distribution
    # Why Poisson? It models "count of events" naturally — always a positive integer,
    # clusters around a mean, right-skewed. Perfect for units per transaction.
    # clip(1) ensures no zero-quantity transactions, which wouldn't make sense.
    quantities = rng.poisson(lam=config.QUANTITY_LAMBDA, size=n).clip(1)

    # Unit price — log-normal distribution
    # Why log-normal? Real prices cluster at low values with a long right tail.
    # e.g. most items are Rs.50-500 but some are Rs.5000+. Normal distribution
    # can't model this — it would generate negative prices too.
    unit_prices = np.round(
        rng.lognormal(mean=np.log(config.PRICE_MEAN), sigma=config.PRICE_SIGMA, size=n),
        decimals=2
    )

    # Total amount — qty * price with small noise to simulate rounding/discount
    # This mirrors real-world data where totals don't always multiply out perfectly
    noise = rng.uniform(1 - config.TOTAL_PRICE_NOISE, 1 + config.TOTAL_PRICE_NOISE, size=n)
    total_amounts = np.round(quantities * unit_prices * noise, decimals=2)

    # Payment methods — uniform random
    payment_methods = rng.choice(config.PAYMENT_METHODS, size=n)

    # Customer IDs — realistic mix: some customers return (repeat IDs), some are new
    # We generate 600 unique customer IDs and sample from them — creates natural repeats
    unique_customers = [f"CUST_{i:05d}" for i in range(600)]
    customer_ids = rng.choice(unique_customers, size=n)

    df = pd.DataFrame({
        config.COL_TRANSACTION_ID:  transaction_ids,
        config.COL_STORE_ID:        store_ids,
        config.COL_TIMESTAMP:       timestamps,
        config.COL_SKU:             skus,
        config.COL_CATEGORY:        categories,
        config.COL_QUANTITY:        quantities,
        config.COL_UNIT_PRICE:      unit_prices,
        config.COL_TOTAL_AMOUNT:    total_amounts,
        config.COL_PAYMENT_METHOD:  payment_methods,
        config.COL_CUSTOMER_ID:     customer_ids,
        config.COL_IS_ANOMALY:      0,   # all clean rows start as 0
        config.COL_ANOMALY_TYPE:    "",  # no anomaly type yet
    })

    return df
def inject_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Injects 4 types of DQ issues into the clean DataFrame.
    Marks every affected row with is_anomaly=1 and records the anomaly_type.

    The 4 types and why they were chosen:
        null          - Missing values are the most common real-world DQ issue
        duplicate     - Retail POS systems often send duplicate transactions on retry
        outlier       - Data entry errors or fraud produce extreme numeric values
        invalid_format - Upstream system changes can corrupt categorical/type fields

    Args:
        df: Clean DataFrame from generate_clean_data()

    Returns:
        DataFrame with DQ issues injected and ground truth labels set.
        Original row order is preserved — anomalies are scattered throughout.
    """
    df = df.copy()  # never mutate the original clean dataset
    rng = np.random.default_rng(config.RANDOM_SEED + 1)
    # +1 so injection randomness is independent from generation randomness
    # If both used the same seed, the same rows that got certain prices
    # would also be the ones getting nulls — artificial correlation

    n = len(df)

    # Track which rows are already flagged to avoid double-injecting
    # A row that gets a null AND an outlier would be harder to evaluate cleanly
    flagged_indices = set()

    def sample_unflagged(count):
        """Pick `count` rows that haven't been touched yet."""
        available = list(set(range(n)) - flagged_indices)
        count = min(count, len(available))
        chosen = rng.choice(available, size=count, replace=False)
        flagged_indices.update(chosen)
        return chosen

    # 1. NULL INJECTION
    # Nulls go into unit_price and customer_id — the two most impactful columns
    # for a retail business (can't calculate revenue, can't attribute to customer)
    null_count = int(n * config.NULL_RATE)
    null_idx = sample_unflagged(null_count)

    # Split evenly: half get null unit_price, half get null customer_id
    half = len(null_idx) // 2
    df.loc[null_idx[:half], config.COL_UNIT_PRICE] = np.nan
    df.loc[null_idx[half:], config.COL_CUSTOMER_ID] = np.nan
    df.loc[null_idx, config.COL_IS_ANOMALY] = 1
    df.loc[null_idx, config.COL_ANOMALY_TYPE] = "null"

    # 2. DUPLICATE INJECTION
    # Pick existing clean rows and append exact copies
    # Real cause: POS terminal retries a failed network request, sends transaction twice
    dup_count = int(n * config.DUPLICATE_RATE)
    dup_source_idx = rng.choice(list(set(range(n)) - flagged_indices),
                                size=dup_count, replace=False)

    dup_rows = df.iloc[dup_source_idx].copy()
    dup_rows[config.COL_IS_ANOMALY] = 1
    dup_rows[config.COL_ANOMALY_TYPE] = "duplicate"

    df = pd.concat([df, dup_rows], ignore_index=True)
    # Note: duplicates are appended at the end here, but main.py will shuffle
    # the final DataFrame so they're scattered — detectors shouldn't benefit
    # from positional patterns

    # 3. OUTLIER INJECTION
    # Multiply unit_price by OUTLIER_MULTIPLIER (10x) to simulate:
    # - Data entry error (extra zero added)
    # - Fraud (inflated invoice)
    # - Unit mismatch (price per case entered instead of per unit)
    outlier_count = int(n * config.OUTLIER_RATE)
    outlier_idx = sample_unflagged(outlier_count)

    df.loc[outlier_idx, config.COL_UNIT_PRICE] = (
        df.loc[outlier_idx, config.COL_UNIT_PRICE] * config.OUTLIER_MULTIPLIER
    )
    # Recalculate total_amount so it's consistent with the outlier price
    # If we didn't do this, total_amount would be an outlier too — double-flagging
    df.loc[outlier_idx, config.COL_TOTAL_AMOUNT] = (
        df.loc[outlier_idx, config.COL_QUANTITY] *
        df.loc[outlier_idx, config.COL_UNIT_PRICE]
    )
    df.loc[outlier_idx, config.COL_IS_ANOMALY] = 1
    df.loc[outlier_idx, config.COL_ANOMALY_TYPE] = "outlier"

    # 4. INVALID FORMAT INJECTION
    # Two sub-types:
    #   a) Negative quantity — physically impossible, suggests upstream system bug
    #   b) Invalid payment method — value outside the allowed domain (e.g. a new
    #      payment system was added upstream but not registered in the master list)
    invalid_count = int(n * config.INVALID_FORMAT_RATE)
    invalid_idx = sample_unflagged(invalid_count)

    half_inv = len(invalid_idx) // 2
    # Negative quantities
    df.loc[invalid_idx[:half_inv], config.COL_QUANTITY] = (
        df.loc[invalid_idx[:half_inv], config.COL_QUANTITY] * -1
    )
    # Invalid payment method strings
    df.loc[invalid_idx[half_inv:], config.COL_PAYMENT_METHOD] = "UNKNOWN_METHOD"
    df.loc[invalid_idx, config.COL_IS_ANOMALY] = 1
    df.loc[invalid_idx, config.COL_ANOMALY_TYPE] = "invalid_format"

    return df


def save_data(df: pd.DataFrame, path: str = "output/synthetic_data.csv") -> None:
    """
    Saves the DataFrame to a CSV file.
    Creates the output directory if it doesn't exist.
    """
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Data saved to {path}")
    
if __name__ == "__main__":
    # Step 1: generate clean data
    clean_df = generate_clean_data()
    print(f"Clean data: {len(clean_df)} rows")
    save_data(clean_df, "output/synthetic_data_clean.csv")

    # Step 2: inject anomalies
    dirty_df = inject_anomalies(clean_df)
    print(f"After injection: {len(dirty_df)} rows")
    save_data(dirty_df, "output/synthetic_data_with_anomalies.csv")

    # Step 3: print a summary of what was injected
    print("\nGround truth summary:")
    summary = dirty_df.groupby(config.COL_ANOMALY_TYPE)[config.COL_IS_ANOMALY].count()
    print(summary.rename("row_count").to_string())

    total = len(dirty_df)
    anomaly_count = dirty_df[config.COL_IS_ANOMALY].sum()
    print(f"\nTotal rows     : {total}")
    print(f"Anomalous rows : {int(anomaly_count)}")
    print(f"Fault rate     : {anomaly_count / total * 100:.1f}%")