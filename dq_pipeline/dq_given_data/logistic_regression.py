# dq_given_data/logistic_regression.py

"""
Supervised anomaly detection using Logistic Regression.
Trained on orders_with_anomalies (labelled), scored on orders (unlabelled).

What's different from the synthetic version:
    - Data comes from data_loader + feature_engineer, not generated here
    - Evaluates per anomaly_type, not just aggregate
    - Scores unlabelled production data and saves flagged transactions
    - Reports which features drove each prediction (top coefficients)

Why Logistic Regression on this data?
    We have 503,840 rows with labels. LR learns directly from those labels —
    it doesn't guess from data shape like Isolation Forest does. On this
    dataset where anomalies are well-defined (math errors, negative prices,
    unauthorized discounts), a linear model with good features will perform
    very well because the decision boundaries are mostly linear:
    "if sales_math_error > threshold → anomaly" is a linear rule.
"""

import os
import json
import numpy as np
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
from data_loader import load_training_data, load_scoring_data
from feature_engineer import prepare_training_data, prepare_scoring_data


# ── Step 1: Train the model ────────────────────────────────────────────────────

def train(X_train, y_train):
    """
    Imputes nulls, scales features, fits Logistic Regression.

    Returns the fitted imputer, scaler, and model separately so they
    can each be applied to the test and scoring sets without refitting.

    Why return three objects instead of a Pipeline?
        Transparency. You can inspect what the imputer filled and what
        the scaler's mean/std values are. A Pipeline hides those steps.
    """
    # Impute: fill any remaining nulls with column median
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)

    # Scale: standardise to mean=0, std=1
    # Critical for LR — unit_price and is_holiday are on very different scales
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)

    # Logistic Regression
    model = LogisticRegression(
        random_state=config.RANDOM_SEED,
        max_iter=1000,
        class_weight="balanced",
        # balanced: anomalies are ~1.35% of data — without this the model
        # predicts "normal" for everything and gets 98.6% accuracy but
        # catches zero anomalies. Balanced weighting fixes this.
        C=0.1,
        # C=0.1: moderate regularisation — keeps weights small and prevents
        # the model from over-fitting to the specific anomaly patterns
        # in training data that might not generalise.
        solver="lbfgs",
        n_jobs=-1,
    )

    model.fit(X_train_scaled, y_train)
    return imputer, scaler, model


# ── Step 2: Evaluate on test set ──────────────────────────────────────────────

def evaluate(model, imputer, scaler, X_test, y_test, test_meta):
    """
    Evaluates the model on the held-out test set.

    test_meta: the original DataFrame rows corresponding to X_test,
    used to break down results by anomaly_type.

    Returns a results dictionary with overall and per-type metrics.
    """
    X_test_imp    = imputer.transform(X_test)
    X_test_scaled = scaler.transform(X_test_imp)
    y_pred        = model.predict(X_test_scaled)
    y_prob        = model.predict_proba(X_test_scaled)[:, 1]

    # Overall metrics
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS — TEST SET")
    print("=" * 60)
    print(f"  Test rows      : {len(y_test):,}")
    print(f"  True anomalies : {int(y_test.sum()):,} ({y_test.mean()*100:.2f}%)")
    print(f"\n  Precision      : {precision:.4f}")
    print(f"  Recall         : {recall:.4f}")
    print(f"  F1 Score       : {f1:.4f}")
    print(f"\n  True Positives : {tp:,}  (anomalies correctly caught)")
    print(f"  False Positives: {fp:,}  (normal rows wrongly flagged)")
    print(f"  False Negatives: {fn:,}  (anomalies missed)")

    print("\n[Full Classification Report]")
    print(classification_report(y_test, y_pred,
                                target_names=["Normal", "Anomaly"]))

    # Per-anomaly-type breakdown
    # This is the most useful table — shows which types the model handles well
    print("=" * 60)
    print("  PER ANOMALY TYPE BREAKDOWN")
    print("=" * 60)

    test_meta = test_meta.copy()
    test_meta["y_true"] = y_test.values
    test_meta["y_pred"] = y_pred
    test_meta["y_prob"] = y_prob

    # Map numeric anomaly types to readable names
    test_meta["type_name"] = (
        test_meta[config.O_ANOMALY_TYPE]
        .astype(str)
        .map({str(k): v for k, v in config.ANOMALY_TYPE_MAP.items()})
        .fillna("unknown")
    )

    per_type_rows = []
    for atype, group in test_meta.groupby("type_name"):
        if atype == "normal":
            continue
        total  = len(group)
        caught = int(group["y_pred"].sum())
        missed = total - caught
        p = precision_score(group["y_true"], group["y_pred"], zero_division=0)
        r = recall_score(group["y_true"], group["y_pred"], zero_division=0)
        f = f1_score(group["y_true"], group["y_pred"], zero_division=0)
        per_type_rows.append({
            "anomaly_type": atype,
            "total":        total,
            "caught":       caught,
            "missed":       missed,
            "precision":    round(p, 3),
            "recall":       round(r, 3),
            "f1":           round(f, 3),
        })

    per_type_df = pd.DataFrame(per_type_rows).sort_values("f1", ascending=False)
    print(per_type_df.to_string(index=False))

    return {
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "tp": int(tp), "fp": int(fp),
        "fn": int(fn), "tn": int(tn),
        "per_type": per_type_df.to_dict(orient="records"),
    }


# ── Step 3: Feature importance ─────────────────────────────────────────────────

def print_feature_importance(model, feature_cols):
    """
    Prints LR coefficients as feature importance.

    Positive coefficient → feature pushes prediction toward ANOMALY
    Negative coefficient → feature pushes prediction toward NORMAL
    Magnitude → how strongly that feature influences the decision

    This is your explainability layer — for any flagged transaction,
    the top positive-coefficient features explain why it was flagged.
    """
    importance = pd.DataFrame({
        "feature":     feature_cols,
        "coefficient": model.coef_[0],
    }).sort_values("coefficient", key=abs, ascending=False)

    print("\n" + "=" * 60)
    print("  FEATURE IMPORTANCE (Top 15)")
    print("=" * 60)
    print("  Positive = pushes toward ANOMALY")
    print("  Negative = pushes toward NORMAL\n")
    print(importance.head(15).to_string(index=False))
    return importance


# ── Step 4: Score production data ─────────────────────────────────────────────

def score_production(model, imputer, scaler, X_score, featured_df):
    """
    Scores the unlabelled orders dataset and returns flagged transactions.

    For each flagged row, attaches:
        - anomaly_probability: model's confidence (0-1)
        - top_feature: the single most influential feature for this row
          (the feature that deviates most from normal, weighted by coefficient)

    Args:
        model, imputer, scaler: trained objects from train()
        X_score: feature matrix from prepare_scoring_data()
        featured_df: full enriched DataFrame for context columns

    Returns:
        flagged_df: only the rows predicted as anomalies, with evidence
    """
    X_imp    = imputer.transform(X_score)
    X_scaled = scaler.transform(X_imp)

    predictions  = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]

    scored_df = featured_df.copy()
    scored_df["anomaly_predicted"] = predictions
    scored_df["anomaly_probability"] = probabilities.round(4)

    flagged_df = scored_df[scored_df["anomaly_predicted"] == 1].copy()

    print(f"\n  Production scoring complete:")
    print(f"  Scored         : {len(scored_df):,} orders")
    print(f"  Flagged        : {len(flagged_df):,} ({len(flagged_df)/len(scored_df)*100:.2f}%)")

    return flagged_df


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)

    print("\n" + "=" * 60)
    print("  LOGISTIC REGRESSION — REAL DATA PIPELINE")
    print("=" * 60)

    # Load and prepare training data
    print("\n[1/4] Loading training data...")
    train_df = load_training_data()
    X, y, featured_train = prepare_training_data(train_df)

    # Train/test split
    print("\n[2/4] Splitting and training...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=config.RANDOM_SEED,
        stratify=y,
        # stratify=y: keeps the same 1.35% anomaly ratio in both sets.
        # Without this, you might get no anomalies in the test set by chance.
    )

    # Keep test metadata for per-type evaluation
    test_indices = X_test.index
    test_meta    = featured_train.loc[test_indices].copy()

    print(f"  Train : {len(X_train):,} rows | {int(y_train.sum()):,} anomalies")
    print(f"  Test  : {len(X_test):,} rows  | {int(y_test.sum()):,} anomalies")

    # Train
    imputer, scaler, model = train(X_train, y_train)
    print("  Model trained.")

    # Evaluate
    print("\n[3/4] Evaluating...")
    results = evaluate(model, imputer, scaler, X_test, y_test, test_meta)

    # Feature importance
    from feature_engineer import get_feature_columns
    importance_df = print_feature_importance(model, get_feature_columns())

    # Score production data
    print("\n[4/4] Scoring production orders...")
    score_df = load_scoring_data()
    X_score, featured_score = prepare_scoring_data(score_df)
    flagged_df = score_production(model, imputer, scaler, X_score, featured_score)

    # Save flagged transactions
    output_cols = [
        "order_id", "date", "store_id", "product_id",
        "quantity", "unit_price", "sales_amount", "discount",
        "is_promotion", "payment_type", "cashier_id",
        "anomaly_predicted", "anomaly_probability",
        "selling_price", "city", "category",
        "temperature", "is_holiday",
        "sales_math_error", "price_deviation",
        "unauthorized_discount", "is_negative_price",
        "is_zero_quantity", "qty_zscore",
    ]
    available_cols = [c for c in output_cols if c in flagged_df.columns]
    flagged_df[available_cols].to_csv(
        "output/flagged_transactions.csv", index=False
    )
    print("  Flagged transactions saved to output/flagged_transactions.csv")

    # Save evaluation results
    with open("output/model_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("  Results saved to output/model_results.json")

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)