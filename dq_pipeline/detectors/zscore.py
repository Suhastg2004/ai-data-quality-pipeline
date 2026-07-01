"""
Z-score based anomaly detector.

How it works:
    For each numeric column, compute how many standard deviations a value
    sits away from the column mean. That distance is the Z-score.

    Z = (value - mean) / std

    If |Z| > threshold( here z = 3 initially), the row is flagged.

Why Z = 3 as the threshold?
    In a perfect normal distribution, 99.7% of values fall within 3 standard
    deviations of the mean. Anything beyond that is in the extreme 0.3% tail —
    statistically unusual enough to flag.

Core assumption (and its weakness):
    Z-score assumes the column follows a normal distribution. If it doesn't —
    like our unit_price which is log-normal and heavily right-skewed — the mean
    and std are both distorted by the tail. This pushes the threshold too high,
    causing Z-score to miss real outliers. You will see this in the evaluation.

Why use it at all then?
    It's fast, interpretable, and works well on columns that are close to normal
    (like quantity, which is Poisson but roughly symmetric for our lambda). It's
    also the most widely taught baseline — good for showing what classical
    statistics can and can't do.
"""

import numpy as np
import pandas as pd

import config

def detect(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs Z-score anomaly detection on all numeric columns.

    For each numeric column, computes Z-scores and flags any row where at
    least one column's Z-score exceeds the configured threshold.

    Args:
        df: Input DataFrame (the dirty dataset with injected anomalies)

    Returns:
        Original DataFrame with one new column added:
            zscore_anomaly (int): 1 if flagged as anomaly, 0 if not
    """
    df = df.copy()

    # Columns we don't want to score — they're labels, not features
    skip_cols = {
        config.COL_TRANSACTION_ID,
        config.COL_IS_ANOMALY,
        config.COL_ANOMALY_TYPE,
    }

    # Select only numeric columns that aren't in the skip list
    numeric_cols = [
        col for col in df.select_dtypes(include=[np.number]).columns
        if col not in skip_cols
    ]

    # This will track whether ANY column flagged each row
    # Start with all False, flip to True as we find anomalies
    flagged = pd.Series(False, index=df.index)

    for col in numeric_cols:
        col_data = df[col].dropna()

        mean = col_data.mean()
        std = col_data.std()

        # If std is 0, every value is identical — Z-score is undefined.
        # Skip the column rather than dividing by zero.
        if std == 0:
            continue

        # Compute Z-score for every row in this column
        # Rows with nulls get NaN z-scores — fillna(0) treats them as
        # "not anomalous via Z-score" since nulls are caught by the profiler
        z_scores = ((df[col] - mean) / std).fillna(0).abs()

        # Flag this row if its Z-score exceeds the threshold
        flagged = flagged | (z_scores > config.ZSCORE_THRESHOLD)

    # Convert boolean Series to integer column (1 = anomaly, 0 = normal)
    df["zscore_anomaly"] = flagged.astype(int)

    return df


def get_column_zscores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame showing the Z-score of every numeric value.
    Useful for debugging: lets you see exactly how far each value sits
    from its column mean, not just whether it was flagged.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame of Z-scores with same index as input, numeric columns only
    """
    skip_cols = {
        config.COL_TRANSACTION_ID,
        config.COL_IS_ANOMALY,
        config.COL_ANOMALY_TYPE,
    }

    numeric_cols = [
        col for col in df.select_dtypes(include=[np.number]).columns
        if col not in skip_cols
    ]

    z_df = pd.DataFrame(index=df.index)

    for col in numeric_cols:
        col_data = df[col].dropna()
        mean = col_data.mean()
        std = col_data.std()
        if std == 0:
            z_df[f"{col}_zscore"] = 0.0
        else:
            z_df[f"{col}_zscore"] = ((df[col] - mean) / std).abs().round(4)

    return z_df