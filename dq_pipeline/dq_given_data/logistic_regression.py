# dq_given_data/logistic_regression.py

import os
import json
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)
import config
from data_loader import load_training_data
from feature_engineer import prepare_training_data, get_feature_columns


def train(X_train, y_train):
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    model = LogisticRegression(
        random_state=config.RANDOM_SEED,
        max_iter=1000,
        class_weight="balanced",
        C=0.1,
        solver="lbfgs",
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)
    return imputer, scaler, model


def evaluate(model, imputer, scaler, X_test, y_test, test_meta):
    X_test_imp    = imputer.transform(X_test)
    X_test_scaled = scaler.transform(X_test_imp)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    print("\n" + "=" * 60)
    print("  LOGISTIC REGRESSION — EVALUATION")
    print("=" * 60)
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"\n  TP: {tp:,}  FP: {fp:,}  FN: {fn:,}  TN: {tn:,}")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Anomaly"]))

    print("=" * 60)
    print("  PER ANOMALY TYPE BREAKDOWN")
    print("=" * 60)
    meta = test_meta.copy()
    meta["y_true"] = y_test.values
    meta["y_pred"] = y_pred
    meta["type_name"] = (
        meta[config.O_ANOMALY_TYPE].astype(str)
        .map({str(k): v for k, v in config.ANOMALY_TYPE_MAP.items()})
        .fillna("unknown")
    )
    rows = []
    for atype, group in meta.groupby("type_name"):
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

    return {"model": "Logistic Regression", "precision": round(precision,4),
            "recall": round(recall,4), "f1": round(f1,4),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "per_type": per_type_df.to_dict(orient="records")}, y_pred, y_prob


def print_feature_importance(model, feature_cols):
    importance = pd.DataFrame({
        "feature": feature_cols,
        "coefficient": model.coef_[0],
    }).sort_values("coefficient", key=abs, ascending=False)
    print("\n" + "=" * 60)
    print("  FEATURE IMPORTANCE (Top 15)")
    print("=" * 60)
    print("  Positive = pushes toward ANOMALY")
    print("  Negative = pushes toward NORMAL\n")
    print(importance.head(15).to_string(index=False))
    return importance


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    print("\n" + "=" * 60)
    print(f"  LOGISTIC REGRESSION — {config.ACTIVE_DATASET.upper()}")
    print("=" * 60)

    print("\n[1/4] Loading data...")
    train_df = load_training_data()
    X, y, featured_train = prepare_training_data(train_df)

    print("\n[2/4] Splitting...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.RANDOM_SEED, stratify=y)
    test_indices = X_test.index
    test_meta    = featured_train.loc[test_indices].copy()
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

    print("\n[3/4] Training...")
    imputer, scaler, model = train(X_train, y_train)
    print("  Done.")

    print("\n[4/4] Evaluating...")
    results, y_pred, y_prob = evaluate(model, imputer, scaler, X_test, y_test, test_meta)
    print_feature_importance(model, get_feature_columns())

    # Score held-out test set as production simulation
    result_df = featured_train.loc[test_indices].copy()
    result_df["predicted_anomaly"]   = y_pred
    result_df["anomaly_probability"] = y_prob.round(4)

    training_cols = [c for c in train_df.columns
                     if c not in {config.O_ANOMALY_FLAG, config.O_ANOMALY_TYPE}]
    ground_truth_cols = [c for c in [config.O_ANOMALY_FLAG, config.O_ANOMALY_TYPE]
                         if c in result_df.columns]
    output_cols = training_cols + ["predicted_anomaly", "anomaly_probability"] + ground_truth_cols
    available = [c for c in output_cols if c in result_df.columns]
    result_df[available].to_csv("output/lr_predicted_all.csv", index=False)

    with open("output/lr_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Flagged: {int(result_df['predicted_anomaly'].sum()):,}")
    print("  Saved → output/lr_predicted_all.csv")
    print("  Saved → output/lr_results.json")
    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)