# dq_given_data/config.py

import datetime

RANDOM_SEED    = 42
IF_N_ESTIMATORS = 100
O_ANOMALY_FLAG = "anomaly_flag"
O_ANOMALY_TYPE = "anomaly_type"

ACTIVE_DATASET = "dataset2"  # change to "dataset2" for new data

DATASETS = {
    "dataset1": {
        "path": "datasets/orders_with_anomalies.csv",
        "col_map": {
            "date":         "date",
            "store_id":     "store_id",
            "product_id":   "product_id",
            "quantity":     "quantity",
            "unit_price":   "unit_price",
            "sales_amount": "sales_amount",
            "is_promotion": "is_promotion",
            "discount":     "discount",
            "anomaly_flag": "anomaly_flag",
            "anomaly_type": "anomaly_type",
        },
        "precomputed": [],
    },
    "dataset2": {
        "path": "datasets/ml_features_orders.csv",
        "col_map": {
            "date":         "date",
            "store_id":     "store_id",
            "product_id":   "product_id",
            "quantity":     "qty",
            "unit_price":   "price",
            "sales_amount": "sales",
            "is_promotion": "is_promotion",
            "discount":     "discount",
            "anomaly_flag": "anomaly_flag",
            "anomaly_type": "anomaly_type",
        },
        "precomputed": [
            "sales_7d_avg", "sales_14d_avg", "sales_std_7d",
            "qty_7d_avg", "price_14d_avg", "demand_trend",
            "temperature", "rainfall", "humidity",
            "is_holiday", "size_sqft", "footfall_multiplier",
            "avg_transaction_value", "open_24x7", "demand_volatility",
            "sales_spike", "qty_spike", "price_deviation",
            "negative_inventory", "promotion_mismatch", "rule_score",
            "inventory",
        ],
    },
}

ACTIVE_DATASET_PATH        = DATASETS[ACTIVE_DATASET]["path"]
ACTIVE_DATASET_COL_MAP     = DATASETS[ACTIVE_DATASET]["col_map"]
ACTIVE_DATASET_PRECOMPUTED = DATASETS[ACTIVE_DATASET]["precomputed"]

ANOMALY_TYPE_MAPS = {
    "dataset1": {
        0: "normal",
        1: "negative_unit_price",
        2: "wrong_price",
        3: "unknown",
        4: "zero_quantity",
        5: "unauthorized_discount",
        6: "quantity_outlier",
        7: "sales_math_error",
    },
    "dataset2": {
        0: "normal",
        1: "sales_spike",
        2: "qty_spike",
        3: "price_deviation",
        4: "negative_inventory",
        5: "promotion_mismatch",
        6: "combined_anomaly",
    },
}

ANOMALY_TYPE_MAP = ANOMALY_TYPE_MAPS[ACTIVE_DATASET]