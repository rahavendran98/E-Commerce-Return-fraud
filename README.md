# E-Commerce Return Fraud Detection System

A machine learning system that predicts whether a return request is fraudulent, using customer behaviour and order history. Includes a full preprocessing pipeline, three-model comparison, SHAP explainability, and a Streamlit web application for real-time scoring.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline Summary](#pipeline-summary)
- [Model Performance](#model-performance)
- [Streamlit Application](#streamlit-application)
- [Installation](#installation)
- [Usage](#usage)
- [Key Findings](#key-findings)
- [Reports](#reports)
- [Tech Stack](#tech-stack)

---

## Project Overview

Return fraud — where customers abuse e-commerce return policies by returning used, swapped, or never-purchased items — is a growing operational and financial problem. This project builds an end-to-end fraud detection system that:

- Ingests raw transactional data with 50,500 records across 30 columns
- Runs a 14-step preprocessing pipeline that cleans, corrects, and engineers 41 model-ready features
- Trains and compares three classifiers: Logistic Regression, Random Forest, and XGBoost
- Selects XGBoost as the final model based on ROC-AUC, PR-AUC, and business cost analysis
- Explains every prediction using SHAP values so fraud operations teams understand why a decision was made
- Serves real-time predictions through a Streamlit web application with three-tier risk categorisation

The dataset is synthetic, meaning fraud follows clean, rule-based patterns and model scores approach 1.0. The project's value is in the correct workflow — leakage-free feature engineering, reproducible pipeline decisions, and operationally useful output — not the inflated accuracy numbers that synthetic data produces.

---

## Dataset

| Attribute | Detail |
|-----------|--------|
| File | `data/raw/ecommerce_return_fraud_dataset.csv` |
| Records | 50,500 raw → 50,000 after deduplication |
| Features | 30 raw columns → 41 model features |
| Target | `return_fraud` (0 = Genuine, 1 = Fraud) |
| Class distribution | Fraud: 35,487 (70.3%) / Genuine: 15,013 (29.7%) |

### Feature Groups

| Group | Columns |
|-------|---------|
| Customer profile | `customer_age`, `gender`, `city_tier`, `membership_type`, `account_age_days` |
| Transaction history | `total_orders`, `previous_returns`, `return_rate`, `customer_rating`, `customer_support_tickets` |
| Order details | `product_category`, `product_price`, `discount_percent`, `order_amount`, `payment_method` |
| Return behaviour | `return_reason`, `days_to_return`, `delivery_days` |
| Technical signals | `device_type`, `ip_risk_score` |
| Dates | `order_date`, `delivery_date`, `return_request_date`, `refund_processed_date` |
| Target | `return_fraud` |

> **Leakage columns removed before training:** `fraud_investigation_status`, `refund_approved`, `final_fraud_decision`, `refund_processed_date` — these are generated after a fraud review completes and do not exist at the time a prediction is needed.

---

## Project Structure

```
E-commerce/
│
├── app.py                              # Streamlit web application
├── requirements.txt                    # Python dependencies
├── README.md
│
├── data/
│   ├── raw/
│   │   └── ecommerce_return_fraud_dataset.csv
│   └── processed/
│       └── modeling_dataset.csv        # Clean 50,000-row dataset (41 features)
│
├── notebooks/
│   ├── data_understanding.ipynb        # EDA and initial data profiling
│   ├── data_quality.ipynb              # Data quality assessment (7 checks)
│   ├── data_preprocessing.ipynb        # 14-step cleaning and feature engineering
│   ├── model_building.ipynb            # Model training, comparison, SHAP
│   ├── fraud_model.pkl                 # Saved XGBoost model
│   ├── scaler.pkl                      # Fitted StandardScaler
│   └── feature_columns.pkl             # Ordered list of 41 feature names
│
├── src/
│   ├── feature_engineering.py          # Encoding, scaling, feature selection utilities
│   └── models.py                       # Training, evaluation, hyperparameter tuning
│
└── reports/
    ├── Data_Quality_Report.docx        # Formatted data quality report
    ├── Preprocessing_and_Evaluation_Report.md
    ├── confusion_matrix.png
    ├── Precision_Recall_Curve.png
    ├── outlier_boxplots.png
    ├── target_distribution.png
    ├── numeric_eda.png
    ├── numeric_correlation.png
    ├── categorical_correlation.png
    ├── categorical_association.png
    └── data_quality_report.txt
```

---

## Pipeline Summary

The preprocessing pipeline transforms 50,500 raw records into a clean 50,000-row, 41-feature modeling dataset in 14 sequential steps.

| Step | Operation | Records / Columns Affected |
|------|-----------|---------------------------|
| 1 | Drop exact duplicate rows | 500 removed → 50,000 remain |
| 2 | Standardise `gender` and `membership_type` casing and abbreviations | ~8% of records each |
| 3 | Strip ₹ from `product_price`, convert to float64 | 7,046 rows |
| 4 | Strip % from `discount_percent`, convert to float64 | 7,057 rows |
| 5 | Null out invalid `product_price` values (sentinel 99999999, negatives, zeros) | ~1,077 rows |
| 6 | Null out `customer_age` outside range 13–100 | 201 rows |
| 7 | Cap `previous_returns` at `total_orders`; recompute `return_rate` | 5,935 rows |
| 8 | Parse all 4 date columns with `format='mixed'`; drop corrupted date features | Mixed formats in 6% of rows |
| 9 | Median imputation for numeric columns; mode for `membership_type` | 4 columns, 12.9–15% missing |
| 10 | IQR upper-fence capping on `total_orders`, `order_amount`, `product_price` | < 0.15% of rows per column |
| 11 | Binary/ordinal/one-hot encoding for 8 categorical columns | 30 → 47 columns before leakage removal |
| 12 | Engineer 6 behavioural features (see table below) | New derived columns |
| 13 | Drop leakage and identifier columns | 6 columns removed |
| 14 | 80/20 stratified train-test split; StandardScaler on train only | 40,000 train / 10,000 test |

### Engineered Features

| Feature | Formula | Purpose |
|---------|---------|---------|
| `account_age_years` | `account_age_days / 365` | Human-readable format |
| `is_new_account` | `account_age_days < 90 → 1` | Detects new-account fraud ring pattern |
| `high_return_rate` | `return_rate > 0.5 → 1` | Isolates chronic returners |
| `support_intensity` | `tickets / total_orders` | Volume-normalised complaint ratio |
| `high_ip_risk` | `ip_risk_score > 70 → 1` | Converts continuous signal into actionable flag |
| `fast_return` | `days_to_return ≤ 2 → 1` | Flags wardrobing and serial-abuse patterns |

---

## Model Performance

Three models were trained on identical 40,000-row train splits and evaluated on a held-out 10,000-row test set.

### Metrics Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|-------|----------|-----------|--------|----|---------|--------|
| Logistic Regression | 0.9137 | 0.9701 | 0.9051 | 0.9365 | 0.9754 | 0.9863 |
| Random Forest | 0.9995 | 0.9993 | 1.0000 | 0.9996 | 0.9999 | 0.9999 |
| **XGBoost** ✓ | **0.9995** | **0.9994** | **0.9999** | **0.9996** | **1.0000** | **1.0000** |

### Confusion Matrix (10,000 test samples)

| Model | True Positive | False Positive | False Negative | True Negative |
|-------|:------------:|:--------------:|:--------------:|:-------------:|
| Logistic Regression | 6,359 | 196 | 667 | 2,778 |
| Random Forest | 7,026 | 5 | 0 | 2,969 |
| **XGBoost** | **7,025** | **4** | **1** | **2,970** |

### Business Cost Analysis

Assumptions: FN (missed fraud) = ₹5,000 loss per case | FP (false alarm) = ₹1,000 investigation cost

| Model | FP | FN | Total Cost (₹) |
|-------|:--:|:--:|---------------:|
| Logistic Regression | 196 | 667 | 3,531,000 |
| Random Forest | 5 | 0 | 5,000 |
| **XGBoost** | **4** | **1** | **9,000** |

The cost difference between Logistic Regression and the tree models is **390×** on the same 10,000-row test set.

### Imbalance Handling — `scale_pos_weight` vs SMOTE

Both methods were tested on XGBoost. `scale_pos_weight` was selected because it produces 1 false negative vs. SMOTE's 2, without generating synthetic training samples.

| Method | Accuracy | F1 | FP | FN |
|--------|----------|----|----|-----|
| XGB + scale_pos_weight | 0.9995 | 0.9996 | 4 | **1** |
| XGB + SMOTE | 0.9994 | 0.9996 | 4 | 2 |

### SHAP Feature Importance (Top 5)

| Rank | Feature | Direction |
|------|---------|-----------|
| 1 | `return_rate` | High → Fraud (dominant signal) |
| 2 | `ip_risk_score` | High → Fraud (context-dependent) |
| 3 | `previous_returns` | High count → Fraud |
| 4 | `high_return_rate` | Present → Fraud |
| 5 | `support_intensity` | High ratio → Fraud |

Demographic features (`gender`, `device_type`, `city_tier`, `customer_age`) have SHAP values near zero — the model correctly learned to ignore them.

---

## Streamlit Application

The `app.py` file serves a real-time fraud risk scoring interface.

### Features

- **Input panel** — 12 fields covering customer profile, order details, and return behaviour
- **Fraud probability** — displayed as a percentage with a progress bar
- **Risk tier** — colour-coded badge (Green / Yellow / Red)
- **Actionable recommendation** — plain-language decision guidance for each tier
- **SHAP explanation** — top 5 features that drove the prediction, with direction (increases / reduces risk)

### Risk Tiers

| Probability | Category | Recommended Action |
|-------------|----------|--------------------|
| < 30% | 🟢 Low Risk | Auto-approve. Process refund normally. |
| 30% – 70% | 🟡 Medium Risk | Manual review. Verify order history and return reason. |
| > 70% | 🔴 High Risk | Hold refund. Flag for fraud team. Check IP and return history. |

### Screenshots

| Initial State | Prediction Output |
|--------------|------------------|
| ![Initial](reports/streamlit_initial.png) | ![Output](reports/streamlit_prediction_output.png) |

---

## Installation

### Prerequisites

- Python 3.9+
- Windows / macOS / Linux

### Steps

```bash
# Clone the repository
git clone https://github.com/rahavendran98/ecommerce-return-fraud-detection.git
cd ecommerce-return-fraud-detection

# Create and activate virtual environment
python -m venv commerceenv
# Windows
commerceenv\Scripts\activate
# macOS / Linux
source commerceenv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Run the Streamlit App

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

### Run Notebooks in Order

```
1. notebooks/data_understanding.ipynb     # Explore the dataset
2. notebooks/data_quality.ipynb           # Quality assessment
3. notebooks/data_preprocessing.ipynb     # Clean and engineer features
4. notebooks/model_building.ipynb         # Train, compare, and save models
```

### Use the Saved Model Directly

```python
import joblib
import pandas as pd

model = joblib.load("notebooks/fraud_model.pkl")
scaler = joblib.load("notebooks/scaler.pkl")
feature_columns = joblib.load("notebooks/feature_columns.pkl")

# Build a row with all 41 features
input_data = {col: 0 for col in feature_columns}
input_data.update({
    "return_rate": 0.75,
    "ip_risk_score": 80,
    "previous_returns": 15,
    "total_orders": 20,
    # ... other fields
})

df = pd.DataFrame([input_data])[feature_columns]
df_scaled = scaler.transform(df)

fraud_prob = model.predict_proba(df_scaled)[0][1]
print(f"Fraud probability: {fraud_prob:.1%}")
```

---

## Key Findings

1. **Return rate is the strongest fraud signal.** Customers with a return rate above 50% are near-certain fraud cases. Those below 10% are almost always genuine. This single feature dominates the SHAP importance ranking.

2. **Fraud is behaviour-driven, not demographic.** Gender, city tier, device type, and customer age show no statistical association with fraud. The model correctly assigns them near-zero importance.

3. **Logistic Regression cannot learn the fraud boundary.** The decision boundary is a return-rate threshold — a non-linear structure that a linear classifier structurally cannot replicate, regardless of class weighting.

4. **IP risk is a supporting signal, not a standalone predictor.** Median IP risk score is 56 for fraud vs. 41 for genuine orders. It provides lift when combined with return rate but is weak on its own.

5. **Post-event columns must be excluded.** `fraud_investigation_status`, `refund_approved`, and `final_fraud_decision` are generated after a review completes. Including them causes data leakage — 100% training accuracy that collapses to zero in production.

6. **Near-perfect scores reflect synthetic data, not overfitting.** 5-fold cross-validation F1 = 0.9997 ± 0.0002. The train-test gap is near zero. The scores are high because fraud follows clean, rule-based patterns in this synthetic dataset — not because the model memorised the training set.

---

## Reports

| Report | Location | Description |
|--------|----------|-------------|
| Data Quality Report | `reports/Data_Quality_Report.docx` | 7-category quality assessment across 50,500 records |
| Preprocessing & Evaluation | `reports/Preprocessing_and_Evaluation_Report.md` | 14-step pipeline decisions + model comparison |
| Data Quality Summary | `reports/data_quality_report.txt` | Machine-readable quality summary |
| Outlier Box Plots | `reports/outlier_boxplots.png` | IQR fence visualisation for 11 numeric columns |
| Confusion Matrices | `reports/confusion_matrix.png` | Side-by-side matrix for all 3 models |
| Precision-Recall Curve | `reports/Precision_Recall_Curve.png` | PR curve comparison across all 3 models |
| Numeric EDA | `reports/numeric_eda.png` | Distribution plots for numeric features |
| Correlation Matrix | `reports/numeric_correlation.png` | Pearson correlation for numeric features |

---

## Tech Stack

| Category | Library | Version |
|----------|---------|---------|
| Data manipulation | pandas | 3.0.3 |
| Numerical computing | numpy | 2.4.6 |
| Machine learning | scikit-learn | 1.8.0 |
| Gradient boosting | xgboost | 3.2.0 |
| Imbalance handling | imbalanced-learn | 0.14.1 |
| Model explainability | shap | 0.52.0 |
| Web application | streamlit | 1.58.0 |
| Visualisation | matplotlib, seaborn | 3.10.9, 0.13.2 |
| Model serialisation | joblib | 1.5.3 |

---

## License

This project is for educational and portfolio purposes. The dataset is synthetic and does not represent real customer or transaction data.
