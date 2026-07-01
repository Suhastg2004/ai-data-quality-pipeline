"""
Central configuration for the DQ pipeline.
All tuneable parameters live here.
Changing a value here affects the entire pipeline — no other file needs touching.
"""

# Reproducibility
RANDOM_SEED = 42
# A fixed seed means every "random" operation produces the same result every run.
# Without it, precision/recall numbers shift each run, making comparisons meaningless.

# Dataset size
NUM_ROWS = 1000

# Retail schema: the columns we'll generate
STORES = [f"STORE_{i:03d}" for i in range(1, 21)]   # 20 stores across the country
CATEGORIES = ["Electronics", "Grocery", "Apparel", "HomeGoods", "Pharmacy"]
PAYMENT_METHODS = ["cash", "card", "UPI", "wallet"]

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

# Column name constants — defined once so a rename doesn't break multiple files
COL_TRANSACTION_ID  = "transaction_id"
COL_STORE_ID        = "store_id"
COL_TIMESTAMP       = "timestamp"
COL_SKU             = "sku"
COL_CATEGORY        = "category"
COL_QUANTITY        = "quantity"
COL_UNIT_PRICE      = "unit_price"
COL_TOTAL_AMOUNT    = "total_amount"
COL_PAYMENT_METHOD  = "payment_method"
COL_CUSTOMER_ID     = "customer_id"

# Ground truth columns — added during generation, consumed during evaluation
COL_IS_ANOMALY   = "is_anomaly"    # 0 = normal, 1 = anomaly
COL_ANOMALY_TYPE = "anomaly_type"  # "null", "duplicate", "outlier", "invalid_format", or ""