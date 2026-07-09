# dq_given_data/xg_clean.py

import os
import json
import pandas as pd
from xgboost import XGBClassifier
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
    print(f"  XGBOOST CLEAN — {config.ACTIVE_DATASET.upper()}")
    print("=" * 60)

    print("\n[1/4] Loading data...")
    train_df = load_training_data()
    X, y, featured_df = prepare_training_data(train_df)

    print("\n[2/4] Splitting...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.RANDOM_SEED, stratify=y)
    test_indices = X_test.index
    test_meta    = featured_df.loc[test_indices].copy()
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

    print("\n[3/4] Training XGBoost...")
    imputer     = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp  = imputer.transform(X_test)

    neg   = int((y_train == 0).sum())
    pos   = int((y_train == 1).sum())
    ratio = round(neg / pos, 2)
    print(f"  Class ratio: {ratio}")

    model = XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        scale_pos_weight=ratio, random_state=config.RANDOM_SEED,
        eval_metric="logloss", verbosity=0, n_jobs=-1,
    )
    model.fit(X_train_imp, y_train)
    print("  Done.")

    print("\n[4/4] Evaluating and predicting...")
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

    print("=" * 60)
    print("  PER ANOMALY TYPE BREAKDOWN")
    print("=" * 60)
    test_meta["y_true"] = y_test.values
    test_meta["y_pred"] = predictions
    test_meta["type_name"] = (
        test_meta[config.O_ANOMALY_TYPE].astype(str)
        .map({str(k): v for k, v in config.ANOMALY_TYPE_MAP.items()})
        .fillna("unknown")
    )
    rows = []
    for atype, group in test_meta.groupby("type_name"):
        if atype == "normal":
            continue
        total  = len(group)
        caught = int(group["y_pred"].sum())
        p = precision_score(group["y_true"], group["y_pred"], zero_division=0)
        r = recall_score(group["y_true"], group["y_pred"], zero_division=0)
        f = f1_score(group["y_true"], group["y_pred"], zero_division=0)
        rows.append({"anomaly_type": atype, "total": total, "caught": caught,
                     "missed": total-caught, "precision": round(p,3),
                     "recall": round(r,3), "f1": round(f,3)})

    per_type_df = pd.DataFrame(rows).sort_values("f1", ascending=False)
    print(per_type_df.to_string(index=False))

    results = {
        "model": "XGBoost", "dataset": config.ACTIVE_DATASET,
        "precision": round(precision,4), "recall": round(recall,4),
        "f1": round(f1,4), "tp": int(tp), "fp": int(fp),
        "fn": int(fn), "tn": int(tn),
        "per_type": per_type_df.to_dict(orient="records"),
    }
    with open("output/xgb_results.json", "w") as f:
        json.dump(results, f, indent=2)

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
    result_df[available].to_csv("output/xgb_predicted_all.csv", index=False)

    print(f"\n  Total scored : {len(result_df):,}")
    print(f"  Flagged (1s) : {int(result_df['predicted_anomaly'].sum()):,}")
    print(f"  Normal  (0s) : {int((result_df['predicted_anomaly']==0).sum()):,}")
    print(f"\n  Breakdown by predicted type:")
    print(result_df["predicted_type"].value_counts().to_string())
    print(f"\n  Saved → output/xgb_predicted_all.csv")
    print(f"  Saved → output/xgb_results.json")
    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)