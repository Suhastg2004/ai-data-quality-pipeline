import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.impute import SimpleImputer

import config


def run(dirty_df):
    """Trains Logistic Regression and returns predictions + metadata."""

    df = dirty_df.copy()
    df["price_qty_ratio"]     = df["unit_price"] / (df["quantity"].abs() + 1e-9)
    df["expected_total"]      = df["quantity"] * df["unit_price"]
    df["total_discrepancy"]   = (
        (df["total_amount"] - df["expected_total"]).abs()
        / (df["expected_total"].abs() + 1e-9)
    )
    df["log_unit_price"]      = np.log1p(df["unit_price"].clip(lower=0))
    df["log_total_amount"]    = np.log1p(df["total_amount"].clip(lower=0))
    df["unit_price_is_null"]  = df["unit_price"].isnull().astype(int)
    df["customer_id_is_null"] = df["customer_id"].isnull().astype(int)
    df["is_negative_qty"]     = (df["quantity"] < 0).astype(int)
    df["is_invalid_payment"]  = (~df["payment_method"].isin(config.PAYMENT_METHODS)).astype(int)

    feature_cols = [
        "quantity", "unit_price", "total_amount",
        "price_qty_ratio", "expected_total", "total_discrepancy",
        "log_unit_price", "log_total_amount",
        "unit_price_is_null", "customer_id_is_null",
        "is_negative_qty", "is_invalid_payment",
    ]

    X, y = df[feature_cols], df["is_anomaly"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.RANDOM_SEED, stratify=y
    )

    imputer        = SimpleImputer(strategy="median")
    X_train_imp    = imputer.fit_transform(X_train)
    X_test_imp     = imputer.transform(X_test)

    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled  = scaler.transform(X_test_imp)

    model = LogisticRegression(
        random_state=config.RANDOM_SEED,
        max_iter=1000,
        class_weight="balanced",
        C=0.1,
    )
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    importance = pd.DataFrame({
        "feature":     feature_cols,
        "coefficient": model.coef_[0],
    }).sort_values("coefficient", key=abs, ascending=False)

    return y_test, y_pred, importance


if __name__ == "__main__":
    from data_generator import generate_clean_data, inject_anomalies

    dirty_df = inject_anomalies(generate_clean_data())
    y_test, y_pred, importance = run(dirty_df)

    print(f"\nPrecision : {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall    : {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"F1 Score  : {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Anomaly"]))
    print(importance.to_string(index=False))