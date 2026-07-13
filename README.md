# AI Data Quality Pipeline

## Project Overview
This repository contains a machine learning pipeline designed to identify fraudulent and erroneous Point-of-Sale (POS) transactions. The primary objective of this project is to automate the detection of transactional anomalies to reduce inventory shrinkage, correct systemic errors, and prevent revenue leakage.

Through extensive evaluation, the pipeline demonstrates that supervised learning, paired with explicit feature engineering on a focused dataset, is the most effective approach for this business case.

## Anomaly Classification Taxonomy
The model categorizes transactions into eight distinct states. This taxonomy enables the business to detect anomalies and automatically route them to the appropriate department for remediation.

* **Type 0:** Normal Transaction
* **Type 1:** Negative Unit Price (System/Catalog Error)
* **Type 2:** Wrong Price (Merchandising Error)
* **Type 3:** Unknown (Uncategorized Deviation)
* **Type 4:** Zero Quantity (Cashier/System Error)
* **Type 5:** Unauthorized Discount (Potential Shrinkage/Sweethearting)
* **Type 6:** Quantity Outlier (Supply Chain Risk)
* **Type 7:** Sales Math Error (System Glitch)

## Methodology: The Impact of Feature Engineering
Initial testing revealed that raw transactional data was insufficient for optimal model performance. An A/B test was conducted on the Random Forest architecture to measure the impact of a custom feature engineering pipeline (which calculates Z-scores, math validity flags, and logical thresholds).

| Model Configuration | Precision | Recall | F1 Score |
| :--- | :--- | :--- | :--- |
| Random Forest (Raw Data) | 0.9355 | 0.8498 | 0.8906 |
| **Random Forest (Engineered Features)** | **1.0000** | **0.8908** | **0.9423** |

The injection of domain knowledge via feature engineering successfully translated abstract business rules into mathematical signals, boosting the overall F1 Score by over 5%.

## Model Evaluation

### Baseline Performance (Initial Dataset)
Four algorithms were evaluated on the primary engineered dataset. The results proved that supervised models significantly outperform unsupervised structural outlier detection (Isolation Forest) for this specific business use case.

| Algorithm | Precision | Recall | F1 Score |
| :--- | :--- | :--- | :--- |
| **Random Forest** | 1.0000 | 0.8908 | 0.9423 |
| **XGBoost** | 1.0000 | 0.8908 | 0.9423 |
| Logistic Regression | 0.7820 | 0.8857 | 0.8306 |
| Isolation Forest | 0.4537 | 0.4491 | 0.4514 |

### Production Simulation (New Dataset)
To stress-test the top two performing models (Random Forest and XGBoost), they were deployed against a newly ingested, unseen dataset containing exactly 1,365 actual anomalies.

| Algorithm | Precision | Recall | F1 Score | Total Predicted Anomalies |
| :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | 0.8765 | 0.7802 | 0.8256 | 1,215 |
| **XGBoost** | 0.6805 | 0.7912 | 0.7317 | 1,587 |

**Behavioral Analysis:**
* **XGBoost (Aggressive):** XGBoost proved highly sensitive. It successfully caught slightly more real anomalies (Recall: 0.7912) but generated a significant volume of false positives. It predicted 1,587 total anomalies against a reality of only 1,365, which reduced its Precision to 0.6805.
* **Random Forest (Precise):** Random Forest proved conservative but highly accurate. While it missed a small percentage of anomalies (predicting only 1,215 total), when it did flag a transaction as anomalous, it was correct 87.65% of the time.

## Deployment Recommendation
The transition away from highly-dimensional, joined datasets toward a focused, feature-engineered order pipeline was highly successful. 

Based on the final evaluation metrics, the **Random Forest** model is the recommended architecture for production deployment. In a retail and transactional environment, high volumes of false positives can cause alert fatigue and overwhelm store managers. The Random Forest model's superior Precision (0.8765) ensures that when a transaction is flagged for review or blocked at the POS, the alert is highly credible and warrants immediate business intervention.
