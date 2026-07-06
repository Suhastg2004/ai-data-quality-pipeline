# dq_pipeline/main.py

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

import config
from data_generator import generate_clean_data, inject_anomalies
from profiler import profile_dataset, print_profile
from detectors import zscore, iqr, isolation_forest
import logistic_regression


print("\n" + "=" * 60)
print("  AI/ML DATA QUALITY PIPELINE")
print("=" * 60)

# Stage 1: Generate data
print("\n[STAGE 1] Generating data...")
dirty_df     = inject_anomalies(generate_clean_data())
ground_truth = dirty_df["is_anomaly"]
print(f"  Rows: {len(dirty_df)} | Anomalies: {int(ground_truth.sum())} ({ground_truth.mean()*100:.1f}%)")

# Stage 2: Profile
print("\n[STAGE 2] Profiling...")
print_profile(profile_dataset(dirty_df))

# Stage 3: Classical detectors
print("\n[STAGE 3] Classical detectors...")
z_preds   = zscore.detect(dirty_df)["zscore_anomaly"]
iqr_preds = iqr.detect(dirty_df)["iqr_anomaly"]
print(f"  Z-Score flagged : {int(z_preds.sum())}")
print(f"  IQR flagged     : {int(iqr_preds.sum())}")

# Stage 4: ML detectors
print("\n[STAGE 4] ML detectors...")
if_result = isolation_forest.detect(dirty_df)
if_preds  = if_result["if_anomaly"]
print(f"  Isolation Forest flagged : {int(if_preds.sum())}")

y_test, lr_preds, importance = logistic_regression.run(dirty_df)
print(f"  Logistic Regression — top feature: {importance.iloc[0]['feature']}")

# Stage 5: Comparison
print("\n[STAGE 5] Comparison report...")

def score(y_true, y_pred):
    return {
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
        "Recall":    round(recall_score(y_true, y_pred, zero_division=0), 3),
        "F1":        round(f1_score(y_true, y_pred, zero_division=0), 3),
    }

results = pd.DataFrame({
    "Z-Score (Classical)":      score(ground_truth, z_preds),
    "IQR (Classical)":          score(ground_truth, iqr_preds),
    "Isolation Forest (ML)":    score(ground_truth, if_preds),
    "Logistic Regression (ML)": score(y_test, lr_preds),
}).T

print("\n" + "=" * 60)
print("  DATA QUALITY — METHOD COMPARISON REPORT")
print("=" * 60)
print(f"  Dataset: {len(dirty_df)} rows | Anomalies: {int(ground_truth.sum())} ({ground_truth.mean()*100:.1f}%)")
print("=" * 60)
print(results.to_string())
print("=" * 60)

best = results["F1"].idxmax()
print(f"\n  Best method : {best} (F1 = {results.loc[best, 'F1']})")
print("\n" + "=" * 60)
print("  PIPELINE COMPLETE")
print("=" * 60)