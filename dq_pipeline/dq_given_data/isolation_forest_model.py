# dq_given_data/isolation_forest_model.py

"""
Isolation Forest anomaly detector.

Unsupervised — does NOT use anomaly_flag labels during training.
Finds anomalies purely from the shape and density of the data.

Why include an unsupervised method?
    Supervised models need labels to train. In production, new data
    arrives without confirmed labels — departments haven't reviewed it yet.
    Isolation Forest works on day one, with zero labelled examples.
    It's the only method here that could run on raw orders.csv directly.

How it works (brief):
    Builds 100 random trees. Each tree randomly picks a feature and
    a split value, partitioning the data recursively until each row
    is isolated. Anomalies are isolated in fewer splits because they
    sit alone in sparse regions. The average path length across all
    100 trees becomes the anomaly score.

Evaluated against ground truth after prediction —
but ground truth was never used during training.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)

import config
from data_loader import load_training_data
from feature_engineer import prepare_training_data


def train(X_train, contamination):
    """
    Imputes nulls and fits Isolation Forest on training features only.
    Labels are never passed in — this is intentional.

    contamination: expected fraction of anomalies in the data.
    We set this to match the known anomaly rate from training labels
    (1.35%) — in real production you'd estimate this from domain knowledge.
    """
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)

    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        # tells the model what fraction of rows to flag as anomalies
        # set to match our known 1.35% anomaly rate
        # in production without labels, you'd set this based on
        # how many alerts your team can realistically investigate
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
    )

    model.fit(X_train_imp)
    return imputer, model


def evaluate(model, imputer, X_test, y_test, test_meta):
    """
    Evaluates Isolation Forest predictions against ground truth labels.

    Important: these labels were NOT used during training.
    We're measuring how well an unsupervised method performs
    compared to what we know the actual answers are.

    sklearn Isolation Forest returns:
        -1 → anomaly
         1 → normal
    We convert to 0/1 to match ground truth format.
    """
    X_test_imp = imputer.transform(X_test)

    raw_preds  = model.predict(X_test_imp)
    y_pred     = (raw_preds == -1).astype(int)

    # Anomaly score: more negative = more anomalous
    # We invert and normalise to get a 0-1 probability-like score
    raw_scores = model.score_samples(X_test_imp)
    y_prob     = 1 - (raw_scores - raw_scores.min()) / (
        raw_scores.max() - raw_scores.min() + 1e-9
    )

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    print("\n" + "=" * 60)
    print("  ISOLATION FOREST — EVALUATION RESULTS")
    print("=" * 60)
    print("  (Unsupervised — labels not used during training)")
    print(f"\n  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"\n  TP: {tp:,}  FP: {fp:,}  FN: {fn:,}  TN: {tn:,}")
    print("\n[Classification Report]")
    print(classification_report(y_test, y_pred,
                                target_names=["Normal", "Anomaly"]))

    # Per anomaly type breakdown
    print("=" * 60)
    print("  PER ANOMALY TYPE BREAKDOWN")
    print("=" * 60)

    meta = test_meta.copy()
    meta["y_true"] = y_test.values
    meta["y_pred"] = y_pred

    meta["type_name"] = (
        meta[config.O_ANOMALY_TYPE]
        .astype(str)
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
        rows.append({
            "anomaly_type": atype,
            "total": total, "caught": caught,
            "missed": total - caught,
            "precision": round(p, 3),
            "recall": round(r, 3),
            "f1": round(f, 3),
        })

    per_type_df = pd.DataFrame(rows).sort_values("f1", ascending=False)
    print(per_type_df.to_string(index=False))

    return {
        "model":     "Isolation Forest",
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "tp": int(tp), "fp": int(fp),
        "fn": int(fn), "tn": int(tn),
        "per_type":  per_type_df.to_dict(orient="records"),
    }


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)

    print("\n" + "=" * 60)
    print("  ISOLATION FOREST — REAL DATA PIPELINE")
    print("=" * 60)

    print("\n[1/3] Loading and preparing data...")
    train_df = load_training_data()
    X, y, featured_train = prepare_training_data(train_df)

    # Split — we train IF on X_train only, evaluate on X_test
    # y_train is never passed to the model
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=config.RANDOM_SEED,
        stratify=y,
    )
    test_meta = featured_train.loc[X_test.index].copy()

    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

    # Contamination = known anomaly rate from labels
    # In production without labels, estimate from domain knowledge
    contamination = round(float(y_train.mean()), 4)
    print(f"  Contamination set to: {contamination} (matches training anomaly rate)")

    print("\n[2/3] Training Isolation Forest (no labels used)...")
    imputer, model = train(X_train, contamination)
    print("  Done.")

    print("\n[3/3] Evaluating against ground truth...")
    results = evaluate(model, imputer, X_test, y_test, test_meta)

    with open("output/if_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n  Results saved → output/if_results.json")

    print("\n" + "=" * 60)
    print("  ISOLATION FOREST COMPLETE")
    print("=" * 60)