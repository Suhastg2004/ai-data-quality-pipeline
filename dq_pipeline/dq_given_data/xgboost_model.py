# dq_given_data/xgboost_model.py

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


def train(X_train, y_train):
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    neg   = int((y_train == 0).sum())
    pos   = int((y_train == 1).sum())
    ratio = round(neg / pos, 2)
    print(f"  Class ratio (normal/anomaly): {ratio}")
    model = XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        scale_pos_weight=ratio, random_state=config.RANDOM_SEED,
        eval_metric="logloss", verbosity=0, n_jobs=-1,
    )
    model.fit(X_train_imp, y_train)
    return imputer, model


def evaluate(model, imputer, X_test, y_test, test_meta):
    X_test_imp = imputer.transform(X_test)
    y_pred     = model.predict(X_test_imp)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    print("\n" + "=" * 60)
    print("  XGBOOST — EVALUATION")
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

    return {"model": "XGBoost", "precision": round(precision,4),
            "recall": round(recall,4), "f1": round(f1,4),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "per_type": per_type_df.to_dict(orient="records")}


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    print("\n" + "=" * 60)
    print(f"  XGBOOST — {config.ACTIVE_DATASET.upper()}")
    print("=" * 60)

    print("\n[1/3] Loading data...")
    train_df = load_training_data()
    X, y, featured_train = prepare_training_data(train_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.RANDOM_SEED, stratify=y)
    test_meta = featured_train.loc[X_test.index].copy()
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

    print("\n[2/3] Training...")
    imputer, model = train(X_train, y_train)
    print("  Done.")

    print("\n[3/3] Evaluating...")
    results = evaluate(model, imputer, X_test, y_test, test_meta)

    with open("output/xgb_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n  Saved → output/xgb_results.json")
    print("\n" + "=" * 60)
    print("  XGBOOST COMPLETE")
    print("=" * 60)