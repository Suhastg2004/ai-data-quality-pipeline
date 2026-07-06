# dq_pipeline/evaluator.py

"""
Evaluates all three anomaly detectors against ground truth labels.

Computes per-method precision, recall, F1, and confusion matrix.
Produces a comparison table showing which method performs best and why.

This is the quantitative answer to the core question:
    "Can ML-based detection outperform classical statistics on retail DQ?"
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

import config


def evaluate_all(df: pd.DataFrame) -> dict:
    """
    Evaluates all three detectors against the ground truth is_anomaly column.

    Expects the DataFrame to have already been passed through all three
    detectors, so these columns exist:
        zscore_anomaly, iqr_anomaly, if_anomaly

    Args:
        df: DataFrame with ground truth + all three detector prediction columns

    Returns:
        Dictionary with per-method metrics and a combined comparison table
    """

    ground_truth = df[config.COL_IS_ANOMALY]

    # Define which prediction column belongs to which method
    methods = {
        "Z-Score":          "zscore_anomaly",
        "IQR":              "iqr_anomaly",
        "Isolation Forest": "if_anomaly",
    }

    results = {}

    for method_name, pred_col in methods.items():
        if pred_col not in df.columns:
            raise ValueError(
                f"Column '{pred_col}' not found. "
                f"Run the {method_name} detector before evaluating."
            )

        predictions = df[pred_col]

        # Compute core metrics
        precision = precision_score(ground_truth, predictions, zero_division=0)
        recall    = recall_score(ground_truth, predictions, zero_division=0)
        f1        = f1_score(ground_truth, predictions, zero_division=0)

        # Confusion matrix gives us the four raw counts
        # tn = true negatives  (normal, correctly not flagged)
        # fp = false positives (normal, wrongly flagged)
        # fn = false negatives (anomaly, missed by detector)
        # tp = true positives  (anomaly, correctly flagged)
        tn, fp, fn, tp = confusion_matrix(ground_truth, predictions).ravel()

        results[method_name] = {
            "true_positives":  int(tp),
            "false_positives": int(fp),
            "true_negatives":  int(tn),
            "false_negatives": int(fn),
            "total_flagged":   int(tp + fp),
            "precision":       round(float(precision), 4),
            "recall":          round(float(recall), 4),
            "f1_score":        round(float(f1), 4),
        }

    return results


def build_comparison_table(results: dict) -> pd.DataFrame:
    """
    Converts the results dictionary into a clean comparison DataFrame.
    One row per method, one column per metric.
    This is the table you show in your write-up.

    Args:
        results: Output from evaluate_all()

    Returns:
        DataFrame with methods as rows and metrics as columns
    """
    rows = []
    for method, metrics in results.items():
        rows.append({
            "Method":          method,
            "True Positives":  metrics["true_positives"],
            "False Positives": metrics["false_positives"],
            "False Negatives": metrics["false_negatives"],
            "Total Flagged":   metrics["total_flagged"],
            "Precision":       metrics["precision"],
            "Recall":          metrics["recall"],
            "F1 Score":        metrics["f1_score"],
        })

    df = pd.DataFrame(rows).set_index("Method")
    return df


def print_evaluation(results: dict, total_rows: int, true_anomaly_count: int) -> None:
    """
    Prints a readable evaluation report to the console.

    Args:
        results             : Output from evaluate_all()
        total_rows          : Total rows in the dataset
        true_anomaly_count  : Total ground truth anomaly rows
    """
    print("\n" + "=" * 60)
    print("ANOMALY DETECTION EVALUATION REPORT")
    print("=" * 60)
    print(f"  Total rows in dataset  : {total_rows}")
    print(f"  True anomalies (ground truth) : {true_anomaly_count} "
          f"({true_anomaly_count/total_rows*100:.1f}%)")

    print("\n" + "=" * 60)
    print("PER-METHOD RESULTS")
    print("=" * 60)

    for method, metrics in results.items():
        print(f"\n  [{method}]")
        print(f"    Total flagged    : {metrics['total_flagged']}")
        print(f"    True Positives   : {metrics['true_positives']}  "
              f"(anomalies correctly caught)")
        print(f"    False Positives  : {metrics['false_positives']}  "
              f"(normal rows wrongly flagged)")
        print(f"    False Negatives  : {metrics['false_negatives']}  "
              f"(anomalies missed)")
        print(f"    Precision        : {metrics['precision']:.4f}")
        print(f"    Recall           : {metrics['recall']:.4f}")
        print(f"    F1 Score         : {metrics['f1_score']:.4f}")

    print("\n" + "=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60)
    table = build_comparison_table(results)
    print(table.to_string())

    # Identify best method by F1 score
    best_method = max(results, key=lambda m: results[m]["f1_score"])
    best_f1     = results[best_method]["f1_score"]
    print(f"\n  Best method by F1: {best_method} (F1 = {best_f1:.4f})")

    # Interpretation guidance — this is what you say in your write-up
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    zscore_recall = results["Z-Score"]["recall"]
    iqr_recall    = results["IQR"]["recall"]
    if_recall     = results["Isolation Forest"]["recall"]

    if if_recall > zscore_recall and if_recall > iqr_recall:
        print("  Isolation Forest achieved the highest recall — it found the most")
        print("  true anomalies. This is expected because it evaluates all numeric")
        print("  columns simultaneously rather than one at a time.")

    zscore_precision = results["Z-Score"]["precision"]
    if_precision     = results["Isolation Forest"]["precision"]

    if zscore_precision > if_precision:
        print("\n  Z-Score had higher precision than Isolation Forest — when it")
        print("  flagged a row, it was more likely to be correct. But it missed")
        print(f"  more anomalies overall (recall = {zscore_recall:.4f} vs "
              f"{if_recall:.4f}).")

    print("\n  The F1 score balances both. Use it as the single comparison metric.")


def save_evaluation(results: dict, path: str = "output/evaluation_report.json") -> None:
    """Saves evaluation results to JSON for downstream consumption."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nEvaluation saved to {path}")


if __name__ == "__main__":
    from data_generator import generate_clean_data, inject_anomalies
    from detectors import zscore, iqr, isolation_forest

    # Build the full pipeline up to this point
    clean_df = generate_clean_data()
    dirty_df = inject_anomalies(clean_df)

    # Run all three detectors and merge predictions into one DataFrame
    df_z  = zscore.detect(dirty_df)
    df_iq = iqr.detect(dirty_df)
    df_if = isolation_forest.detect(dirty_df)

    # Each detector returns the full DataFrame with its prediction column added.
    # We only need to pull the prediction columns from df_iq and df_if
    # since df_z already has all the original columns.
    combined_df = df_z.copy()
    combined_df["iqr_anomaly"] = df_iq["iqr_anomaly"]
    combined_df["if_anomaly"]  = df_if["if_anomaly"]
    combined_df["if_score"]    = df_if["if_score"]

    # Run evaluation
    results = evaluate_all(combined_df)

    total_rows          = len(combined_df)
    true_anomaly_count  = int(combined_df[config.COL_IS_ANOMALY].sum())

    print_evaluation(results, total_rows, true_anomaly_count)
    save_evaluation(results)