# dq_given_data/rf_clean.py

import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)
import config
from data_loader import load_training_data
from feature_engineer import prepare_training_data


def infer_anomaly_type(row):
    if row["predicted_anomaly"] == 0:
        return "normal"
    if row["is_negative_price"] == 1:
        return "negative_unit_price"
    if row["is_zero_quantity"] == 1:
        return "zero_quantity"
    if row["unauthorized_discount"] == 1:
        return "unauthorized_discount"
    if row["sales_math_error"] > 0.5:
        return "sales_math_error"
    if row["qty_zscore"] > 3.0:
        return "quantity_outlier"
    return "wrong_price"


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    print("\n" + "=" * 60)
    print(f"  RF CLEAN — PREDICT ALL — {config.ACTIVE_DATASET.upper()}")
    print("=" * 60)

    print("\n[1/4] Loading data...")
    train_df = load_training_data()
    X, y, featured_df = prepare_training_data(train_df)

    print("\n[2/4] Splitting...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.RANDOM_SEED, stratify=y)
    test_indices = X_test.index
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

    print("\n[3/4] Training Random Forest...")
    imputer     = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp  = imputer.transform(X_test)
    model = RandomForestClassifier(
        n_estimators=200, max_depth=10,
        class_weight="balanced",
        random_state=config.RANDOM_SEED, n_jobs=-1,
    )
    model.fit(X_train_imp, y_train)
    print("  Done.")

    print("\n[4/4] Predicting and saving...")
    predictions   = model.predict(X_test_imp)
    probabilities = model.predict_proba(X_test_imp)[:, 1]

    precision = precision_score(y_test, predictions, zero_division=0)
    recall    = recall_score(y_test, predictions, zero_division=0)
    f1        = f1_score(y_test, predictions, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()

    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"\n  TP: {tp:,}  FP: {fp:,}  FN: {fn:,}  TN: {tn:,}")
    print(classification_report(y_test, predictions, target_names=["Normal", "Anomaly"]))

    result_df = featured_df.loc[test_indices].copy()
    result_df["predicted_anomaly"]   = predictions
    result_df["anomaly_probability"] = probabilities.round(4)
    result_df["predicted_type"]      = result_df.apply(infer_anomaly_type, axis=1)

    training_cols = [c for c in train_df.columns
                     if c not in {config.O_ANOMALY_FLAG, config.O_ANOMALY_TYPE}]
    ground_truth_cols = [c for c in [config.O_ANOMALY_FLAG, config.O_ANOMALY_TYPE]
                         if c in result_df.columns]
    output_cols = training_cols + ["predicted_anomaly", "anomaly_probability",
                                   "predicted_type"] + ground_truth_cols
    available = [c for c in output_cols if c in result_df.columns]
    result_df[available].to_csv("output/rf_predicted_all.csv", index=False)

    print(f"\n  Total scored : {len(result_df):,}")
    print(f"  Flagged (1s) : {int(result_df['predicted_anomaly'].sum()):,}")
    print(f"  Normal  (0s) : {int((result_df['predicted_anomaly']==0).sum()):,}")
    print(f"\n  Breakdown by predicted type:")
    print(result_df["predicted_type"].value_counts().to_string())
    print(f"\n  Saved → output/rf_predicted_all.csv")
    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)