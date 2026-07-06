"""
IQR (Interquartile Range) based anomaly detector.

How it works:
    Compute Q1 (25th percentile) and Q3 (75th percentile) for each numeric column.
    The IQR is the distance between them: IQR = Q3 - Q1.

    Lower fence = Q1 - 1.5 * IQR
    Upper fence = Q3 + 1.5 * IQR

    Any value outside these fences is flagged as an anomaly.
    This is called Tukey's rule, introduced by statistician John Tukey in 1977 —
    the same method used to draw the whiskers on a box plot.

Why is IQR better than Z-score on skewed data?
    Q1 and Q3 are percentile-based — they only look at the middle 50% of the data.
    Extreme values in the tail (our injected outliers) don't shift Q1 or Q3 at all.
    This makes IQR robust to the very outliers it's trying to catch.

    Z-score uses the mean and std — both get pulled toward extreme values,
    which widens the acceptable range and causes Z-score to miss real outliers.

Limitation:
    Like Z-score, IQR is univariate — it looks at one column at a time.
    A transaction with a normal price AND a normal quantity is never flagged,
    even if the combination (very high qty × very low price) is suspicious.
    That's the gap Isolation Forest fills.
"""

import numpy as np
import pandas as pd

import config


def detect(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs IQR anomaly detection on all numeric columns.

    Flags any row where at least one numeric column falls outside
    the Tukey fence (Q1 - 1.5*IQR, Q3 + 1.5*IQR).

    Args:
        df: Input DataFrame (the dirty dataset with injected anomalies)

    Returns:
        Original DataFrame with one new column added:
            iqr_anomaly (int): 1 if flagged as anomaly, 0 if not
    """
    df = df.copy()

    skip_cols = {
        config.COL_TRANSACTION_ID,
        config.COL_IS_ANOMALY,
        config.COL_ANOMALY_TYPE,
    }

    numeric_cols = [
        col for col in df.select_dtypes(include=[np.number]).columns
        if col not in skip_cols
    ]

    flagged = pd.Series(False, index=df.index)

    for col in numeric_cols:
        col_data = df[col].dropna()

        q1 = col_data.quantile(0.25)
        q3 = col_data.quantile(0.75)
        iqr = q3 - q1

        # If IQR is 0, all values in the middle 50% are identical.
        # No meaningful fence can be set — skip to avoid flagging everything.
        if iqr == 0:
            continue

        lower_fence = q1 - config.IQR_MULTIPLIER * iqr
        upper_fence = q3 + config.IQR_MULTIPLIER * iqr

        # Flag rows where this column's value is outside the fences
        # fillna(False) means null values are not flagged here —
        # they're already caught by the profiler's null check
        outside_fence = (
            (df[col] < lower_fence) | (df[col] > upper_fence)
        ).fillna(False)

        flagged = flagged | outside_fence

    df["iqr_anomaly"] = flagged.astype(int)

    return df


def get_column_fences(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a summary of IQR fences for every numeric column.
    Useful for understanding exactly where each column's boundaries sit
    and why certain values were flagged.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with one row per column showing Q1, Q3, IQR, and fences
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

    rows = []
    for col in numeric_cols:
        col_data = df[col].dropna()
        q1 = col_data.quantile(0.25)
        q3 = col_data.quantile(0.75)
        iqr = q3 - q1
        rows.append({
            "column": col,
            "q1": round(q1, 2),
            "q3": round(q3, 2),
            "iqr": round(iqr, 2),
            "lower_fence": round(q1 - config.IQR_MULTIPLIER * iqr, 2),
            "upper_fence": round(q3 + config.IQR_MULTIPLIER * iqr, 2),
        })

    return pd.DataFrame(rows).set_index("column")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from data_generator import generate_clean_data, inject_anomalies
    from detectors import zscore, iqr

    clean_df = generate_clean_data()
    dirty_df = inject_anomalies(clean_df)

    # Run both detectors
    df_z = zscore.detect(dirty_df)
    df_iq = iqr.detect(dirty_df)

    print("Z-score results:")
    print(f"  Flagged : {df_z['zscore_anomaly'].sum()} rows")
    print(f"  Missed  : {len(df_z) - df_z['zscore_anomaly'].sum()} rows\n")

    print("IQR results:")
    print(f"  Flagged : {df_iq['iqr_anomaly'].sum()} rows")
    print(f"  Missed  : {len(df_iq) - df_iq['iqr_anomaly'].sum()} rows\n")

    print("IQR fences per column:")
    print(iqr.get_column_fences(dirty_df).to_string())

    print("\nZ-scores (first 5 anomalous rows):")
    z_scores = zscore.get_column_zscores(dirty_df)
    print(z_scores[df_z["zscore_anomaly"] == 1].head().to_string())