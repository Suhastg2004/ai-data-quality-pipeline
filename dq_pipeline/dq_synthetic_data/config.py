"""
Central configuration for the DQ pipeline.
All tuneable parameters live here.
Changing a value here affects the entire pipeline — no other file needs touching.
"""

import datetime

# Reproducibility
RANDOM_SEED = 42
# A fixed seed means every "random" operation produces the same result every run.
# Without it, precision/recall numbers shift each run, making comparisons meaningless.

# Dataset size
NUM_ROWS = 1000

# Retail schema: the columns we'll generate
STORES = [f"STORE_{i:03d}" for i in range(1, 21)]   # 20 stores across the country
CATEGORIES = ["Electronics", "Grocery", "Apparel", "HomeGoods", "Pharmacy"]
PAYMENT_METHODS = ["CASH", "CARD", "WALLET", "UPI"]

# Realistic value ranges for a clean transaction
PRICE_MEAN = 250.0       # mean unit price in rupees (log-normal, so this isn't the literal mean)
PRICE_SIGMA = 0.8        # log-normal sigma — controls spread, higher = more skew
QUANTITY_LAMBDA = 3      # Poisson lambda — average units per transaction
TOTAL_PRICE_NOISE = 0.02 # +/- 2% rounding noise on total_amount vs qty * unit_price

# DQ issue injection rates (as % of total rows)
NULL_RATE = 0.04            # 4% of rows get a null injected into a numeric column
DUPLICATE_RATE = 0.03       # 3% of rows are duplicated (exact copy)
OUTLIER_RATE = 0.04         # 4% of rows get an extreme value injected
INVALID_FORMAT_RATE = 0.02  # 2% of rows get a broken value (wrong type or domain)
# Total designed anomaly rate = ~13%. Realistic for raw retail ingestion data.

# Outlier injection: how extreme is "extreme"
OUTLIER_MULTIPLIER = 10  # outlier unit_price = normal_price x 10

# Anomaly detection thresholds
ZSCORE_THRESHOLD = 3.0  # flag if |z| > 3 (covers 99.7% of a normal distribution)
IQR_MULTIPLIER = 1.5    # Tukey's rule: flag if value falls outside Q1/Q3 +/- 1.5*IQR

IF_CONTAMINATION = 0.13  # tell Isolation Forest to expect ~13% anomalies
# Matches our injection rate. In real deployment you'd estimate this from historical data.
IF_N_ESTIMATORS = 100    # number of trees in the Isolation Forest ensemble


# All CSVs live in the datasets/ folder inside dq_pipeline/

DATA_DIR = "datasets"

PATH_ORDERS            = f"{DATA_DIR}/orders.csv"
PATH_ORDERS_ANOMALIES  = f"{DATA_DIR}/orders_with_anomalies.csv"
PATH_PRODUCTS          = f"{DATA_DIR}/products.csv"
PATH_STORES            = f"{DATA_DIR}/stores.csv"
PATH_PROMOTIONS        = f"{DATA_DIR}/promotions.csv"
PATH_WEATHER           = f"{DATA_DIR}/weather.csv"
PATH_HOLIDAYS          = f"{DATA_DIR}/holidays.csv"

# Column names: orders / orders_with_anomalies
O_ORDER_ID       = "order_id"
O_TIMESTAMP      = "timestamp"
O_DATE           = "date"
O_STORE_ID       = "store_id"
O_PRODUCT_ID     = "product_id"
O_QUANTITY       = "quantity"
O_UNIT_PRICE     = "unit_price"
O_SALES_AMOUNT   = "sales_amount"
O_PAYMENT_TYPE   = "payment_type"
O_CASHIER_ID     = "cashier_id"
O_IS_PROMOTION   = "is_promotion"
O_DISCOUNT       = "discount"

# Ground truth columns (only in orders_with_anomalies)
O_ANOMALY_FLAG   = "anomaly_flag"
O_ANOMALY_TYPE   = "anomaly_type"

# Column names: products 
P_PRODUCT_ID     = "product_id"
P_CATEGORY       = "category"
P_COST_PRICE     = "cost_price"
P_SELLING_PRICE  = "selling_price"
P_MARGIN_PCT     = "margin_pct"
P_SHELF_LIFE     = "shelf_life_days"
P_STORAGE_TYPE   = "storage_type"
P_SEASONALITY    = "seasonality"

# Column names: stores 
S_STORE_ID       = "store_id"
S_STORE_TYPE     = "store_type"
S_CITY           = "city"
S_OPEN_24X7      = "open_24x7"
S_OPENING_YEAR   = "opening_year"
S_AVG_TXN_VALUE  = "avg_transaction_value"
S_REGION         = "region"

# Column names: promotions
PR_PROMO_ID      = "promotion_id"
PR_PRODUCT_ID    = "product_id"
PR_START_OFFSET  = "start_offset"
PR_DURATION      = "duration_days"
PR_DISCOUNT_PCT  = "discount_pct"

# Column names: weather 
W_DATE           = "date"
W_CITY           = "city"
W_TEMPERATURE    = "temperature"
W_RAINFALL       = "rainfall"
W_HUMIDITY       = "humidity"

# Column names: holidays
H_DATE           = "date"
H_HOLIDAY_NAME   = "holiday_name"
H_IS_HOLIDAY     = "is_holiday"

# Promotion date reference
# start_offset in promotions.csv is days since this date
# e.g. start_offset=102 means the promo starts on day 102 of 2024
PROMO_REFERENCE_DATE = datetime.date(2024, 1, 1)

# Feature engineering thresholds
SALES_MATH_TOLERANCE  = 0.01   # allow 1% rounding error in sales amount check
PRICE_DEVIATION_LIMIT = 0.20   # flag if unit_price deviates >20% from selling_price