"""
Comparative report across all detection methods.
Imports results directly from ml_detector.py — no refitting here.
"""

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

from detectors import zscore, iqr, isolation_forest
import ml_detector  # this runs ml_detector.py and gives us access to its variables


# Grab the dataset and ground truth from ml_detector
dirty_df     = ml_detector.dirty_df
ground_truth = dirty_df["is_anomaly"]

# Run the three unsupervised methods on the same dataset
z_preds   = zscore.detect(dirty_df)["zscore_anomaly"]
iqr_preds = iqr.detect(dirty_df)["iqr_anomaly"]
if_preds  = isolation_forest.detect(dirty_df)["if_anomaly"]

# Grab LR predictions directly from ml_detector — already trained and predicted
lr_preds = ml_detector.y_pred
y_test   = ml_detector.y_test


# Score helper
def score(y_true, y_pred):
    return {
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
        "Recall":    round(recall_score(y_true, y_pred, zero_division=0), 3),
        "F1":        round(f1_score(y_true, y_pred, zero_division=0), 3),
        "Flagged":   int(y_pred.sum()),
    }


# Build comparison table
results = pd.DataFrame({
    "Z-Score (Classical)":       score(ground_truth, z_preds),
    "IQR (Classical)":           score(ground_truth, iqr_preds),
    "Isolation Forest (ML)":     score(ground_truth, if_preds),
    "Logistic Regression (ML)":  score(y_test, lr_preds),
}).T


# Print report
print("\n" + "=" * 55)
print("  DATA QUALITY — METHOD COMPARISON REPORT")
print("=" * 55)
print(f"  Dataset : {len(dirty_df)} rows | "
      f"True anomalies : {int(ground_truth.sum())} ({ground_truth.mean()*100:.1f}%)")
print("=" * 55)
print(results.to_string())
print("=" * 55)

best = results["F1"].idxmax()
print(f"\n  Best method  : {best}")
print(f"  Best F1 score: {results.loc[best, 'F1']}")

print("""
  Summary:
  - Z-Score catches numeric outliers only, misses nulls/duplicates
  - IQR is more robust than Z-Score on skewed data like unit_price
  - Isolation Forest improves recall by looking at all columns together
  - Logistic Regression scores highest by learning directly from labels
  - No single method catches everything — all four layers are needed
""")