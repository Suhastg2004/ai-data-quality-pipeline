"""
Supervised anomaly detection using Logistic Regression.

Steps:
    1. Engineer features from raw columns
    2. Split into train/test
    3. Train Logistic Regression
    4. Evaluate and print results
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.impute import SimpleImputer #used to fill the missing values with mean,median etc

import config
from data_generator import generate_clean_data, inject_anomalies


#Generate Data 

clean_df = generate_clean_data()
dirty_df = inject_anomalies(clean_df)
print(f"Dataset: {len(dirty_df)} rows, {int(dirty_df['is_anomaly'].sum())} anomalies")


# Feature Engineering
# We create new columns that give the model stronger signals than raw columns alone.

df = dirty_df.copy()

# Ratio of price to quantity — inflated price shows up clearly here
df["price_qty_ratio"] = df["unit_price"] / (df["quantity"].abs() + 1e-9) #1e-9 to avoid dividing by zero

# How far is total_amount from what qty * price predicts?
df["expected_total"]     = df["quantity"] * df["unit_price"]
df["total_discrepancy"]  = (
    (df["total_amount"] - df["expected_total"]).abs()
    / (df["expected_total"].abs() + 1e-9)
)

# Log transform prices — reduces the effect of extreme values on a linear model
df["log_unit_price"]   = np.log1p(df["unit_price"].clip(lower=0))
df["log_total_amount"] = np.log1p(df["total_amount"].clip(lower=0))

# Null indicator flags — tell the model explicitly "this field was missing"
df["unit_price_is_null"]  = df["unit_price"].isnull().astype(int)
df["customer_id_is_null"] = df["customer_id"].isnull().astype(int)

# Rule-based flags — direct signal for known DQ violations
df["is_negative_qty"]    = (df["quantity"] < 0).astype(int)
df["is_invalid_payment"] = (~df["payment_method"].isin(config.PAYMENT_METHODS)).astype(int)


# Select Features and Target

feature_cols = [
    "quantity",
    "unit_price",
    "total_amount",
    "price_qty_ratio",
    "expected_total",
    "total_discrepancy",
    "log_unit_price",
    "log_total_amount",
    "unit_price_is_null",
    "customer_id_is_null",
    "is_negative_qty",
    "is_invalid_payment",
]

X = df[feature_cols]
y = df["is_anomaly"]


# Train/Test Split
# stratify=y ensures both sets have the same anomaly ratio (~13%)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=config.RANDOM_SEED,
    stratify=y,
)

print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")


# Impute + Scale + Train
# Impute: fill nulls with median (LR can't handle NaN)
# Scale:  put all features on the same scale (LR is sensitive to feature magnitude)

imputer = SimpleImputer(strategy="median")
X_train_imp = imputer.fit_transform(X_train)
X_test_imp  = imputer.transform(X_test)   # transform only, don't refit on test data

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imp)
X_test_scaled  = scaler.transform(X_test_imp)   # same rule — don't refit

model = LogisticRegression(
    random_state=config.RANDOM_SEED,
    max_iter=1000,
    class_weight="balanced",  # prevents model from ignoring the minority anomaly class
    C=0.1,                    # regularisation — keeps model simple, avoids overfitting
)

model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)

print("\n── Results ───────────────────────────────────")
print(f"Precision : {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"F1 Score  : {f1_score(y_test, y_pred, zero_division=0):.4f}")
print("\nFull Report:")
print(classification_report(y_test, y_pred, target_names=["Normal", "Anomaly"]))


# Feature Importance
# Logistic Regression coefficients show which features drive predictions.
# Positive = pushes toward anomaly. Negative = pushes toward normal.

importance = pd.DataFrame({
    "feature":     feature_cols,
    "coefficient": model.coef_[0],
}).sort_values("coefficient", key=abs, ascending=False)

print("── Top Features by Influence ──────────────────")
print(importance.to_string(index=False))