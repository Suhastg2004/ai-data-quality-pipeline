"""
Isolation Forest based anomaly detector.

How it works (recap):
    Treats each row as a point in multi-dimensional space where every
    numeric column is one axis. Builds 100 random decision trees, each
    trying to isolate individual points by making random splits across
    random columns.

    Points that get isolated in fewer splits = anomalies (they sit alone
    in empty space). Points that take many splits to isolate = normal
    (they're packed in a dense cluster with similar rows).

    The average path length across all 100 trees becomes the anomaly score.
    Rows with the lowest scores (isolated fastest) get flagged.

Why sklearn's IsolationForest?
    sklearn's implementation is well-tested, handles missing values cleanly
    via imputation, and exposes both the binary prediction (-1/1) and the
    raw anomaly score — which we use for ranking how anomalous each row is.

Key parameters from config:
    IF_N_ESTIMATORS  = 100   number of trees in the ensemble
    IF_CONTAMINATION = 0.13  expected fraction of anomalies in the data
    RANDOM_SEED      = 42    ensures reproducible tree building
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

import config


def detect(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs Isolation Forest anomaly detection across all numeric columns.

    Unlike Z-score and IQR which look at one column at a time, Isolation
    Forest receives all numeric columns simultaneously — each row becomes
    a point in multi-dimensional space.

    Null values are imputed with column medians before fitting.
    We use median (not mean) because our data is skewed — median is more
    representative of the typical value and won't be pulled by the outliers
    we injected.

    Args:
        df: Input DataFrame (the dirty dataset with injected anomalies)

    Returns:
        Original DataFrame with two new columns added:
            if_anomaly (int)  : 1 if flagged as anomaly, 0 if not
            if_score (float)  : raw anomaly score — more negative = more anomalous
                                useful for ranking rows by how suspicious they are
    """
    df = df.copy()

    # Columns to exclude from the model — labels and non-numeric identifiers
    skip_cols = {
        config.COL_TRANSACTION_ID,
        config.COL_IS_ANOMALY,
        config.COL_ANOMALY_TYPE,
    }

    # Select only numeric columns that aren't metadata
    feature_cols = [
        col for col in df.select_dtypes(include=[np.number]).columns
        if col not in skip_cols
    ]

    X = df[feature_cols].copy()

    # Impute nulls with column medians before fitting
    # Isolation Forest cannot handle NaN values directly.
    # We use median imputation because:
    #   1. Our unit_price is heavily right-skewed — median is more stable than mean
    #   2. We're imputing only to make the model run, not to fix the data
    #      The profiler already flagged these nulls separately
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    # Build and fit the Isolation Forest
    model = IsolationForest(
        n_estimators=config.IF_N_ESTIMATORS,
        contamination=config.IF_CONTAMINATION,
        random_state=config.RANDOM_SEED,
        # max_samples="auto" means sklearn picks min(256, n_rows) samples per tree
        # This is intentional — using a subset per tree increases diversity
        # across the 100 trees, which improves the ensemble's accuracy
        max_samples="auto",
    )

    model.fit(X_imputed)

    # Predictions: sklearn returns -1 for anomaly, 1 for normal
    # We convert to 0/1 to match our ground truth column format
    raw_predictions = model.predict(X_imputed)
    df["if_anomaly"] = (raw_predictions == -1).astype(int)

    # Anomaly scores: negative float, more negative = more anomalous
    # score_samples() returns the negative average path length
    # We store this because it lets us rank rows — useful for root cause analysis
    df["if_score"] = model.score_samples(X_imputed).round(6)

    return df


def get_top_anomalies(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Returns the n rows with the most extreme anomaly scores.
    Useful for investigating the worst offenders in the dataset.

    Requires detect() to have been run first (needs if_score column).

    Args:
        df  : DataFrame output from detect()
        n   : Number of top anomalies to return (default 10)

    Returns:
        DataFrame of the n most anomalous rows, sorted by score ascending
        (most anomalous first since scores are negative)
    """
    if "if_score" not in df.columns:
        raise ValueError("Run detect() first to generate if_score column.")

    return (
        df[df["if_anomaly"] == 1]
        .sort_values("if_score", ascending=True)
        .head(n)[[
            config.COL_STORE_ID,
            config.COL_QUANTITY,
            config.COL_UNIT_PRICE,
            config.COL_TOTAL_AMOUNT,
            config.COL_IS_ANOMALY,
            config.COL_ANOMALY_TYPE,
            "if_score",
        ]]
    )


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from data_generator import generate_clean_data, inject_anomalies
    from detectors import zscore, iqr, isolation_forest

    clean_df = generate_clean_data()
    dirty_df = inject_anomalies(clean_df)

    # Run all three detectors
    df_z  = zscore.detect(dirty_df)
    df_iq = iqr.detect(dirty_df)
    df_if = isolation_forest.detect(dirty_df)

    print("=" * 50)
    print("DETECTION SUMMARY — All Three Methods")
    print("=" * 50)

    total        = len(dirty_df)
    true_anomaly = int(dirty_df[config.COL_IS_ANOMALY].sum())

    print(f"\nTotal rows        : {total}")
    print(f"True anomalies    : {true_anomaly} ({true_anomaly/total*100:.1f}%)")

    print(f"\nZ-score flagged   : {df_z['zscore_anomaly'].sum()}")
    print(f"IQR flagged       : {df_iq['iqr_anomaly'].sum()}")
    print(f"Isolation Forest  : {df_if['if_anomaly'].sum()}")

    print("\nTop 5 most anomalous rows (Isolation Forest):")
    print(isolation_forest.get_top_anomalies(df_if, n=5).to_string())