import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

from evidently.report import Report
from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    ClassificationPreset
)

DATA_PATH = "data/synthetic_ltd_claims_soa_expanded.xlsx"
MODEL_PATH = "model/ltd_best_model_pipeline.pkl"
REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)

# ----------------------------
# Load data
# ----------------------------
df = pd.read_excel(DATA_PATH)
df["is_covid_period"] = (df["exposure_year"] >= 2020).astype(int)

X = df.drop(columns=[
    "policy_id", "claim_incident", "incurred_year", "coverage_start_year",
    "exposure_year", "recovery_status", "claim_duration_months",
    "return_to_work_flag"
])
y = df["claim_incident"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# --------------------------------------------------
#  Simulate drift for demonstration
# --------------------------------------------------
X_test_drifted = X_test.copy()

# Shift numeric features
X_test_drifted["age"] = X_test_drifted["age"] + 5
X_test_drifted["benefit_pct"] = X_test_drifted["benefit_pct"] * 1.1

# Change categorical distribution slightly
X_test_drifted["employment_status"] = "Part-time"

# Replace X_test with drifted version
X_test = X_test_drifted

# ----------------------------
# Load model
# ----------------------------
best_clf = joblib.load(MODEL_PATH)

train_proba = best_clf.predict_proba(X_train)[:, 1]
test_proba  = best_clf.predict_proba(X_test)[:, 1]

reference = X_train.copy()
reference["target"] = y_train.values
reference["prediction"] = train_proba

current = X_test.copy()
current["target"] = y_test.values
current["prediction"] = test_proba

# ==============================
# 1️⃣ Input Data Drift
# ==============================
input_drift = Report(metrics=[DataDriftPreset()])
input_drift.run(
    reference_data=X_train,
    current_data=X_test
)
input_drift.save_html(os.path.join(REPORT_DIR, "01_input_data_drift.html"))

# ==============================
# 2️⃣ Output Drift
# ==============================
output_drift = Report(metrics=[DataDriftPreset()])
output_drift.run(
    reference_data=reference[["prediction"]],
    current_data=current[["prediction"]]
)
output_drift.save_html(os.path.join(REPORT_DIR, "02_output_drift.html"))

# ==============================
# 3️⃣ Target Drift + Performance
# ==============================
performance = Report(metrics=[ClassificationPreset()])
performance.run(
    reference_data=reference,
    current_data=current
)
performance.save_html(os.path.join(REPORT_DIR, "03_target_performance_drift.html"))

# ==============================
# 4️⃣ Data Quality / Anomalies
# ==============================
data_quality = Report(metrics=[DataQualityPreset()])
data_quality.run(
    reference_data=X_train,
    current_data=X_test
)
data_quality.save_html(os.path.join(REPORT_DIR, "04_data_quality.html"))

# ==============================
# Quick metric comparison
# ==============================
roc_ref = roc_auc_score(y_train, train_proba)
roc_cur = roc_auc_score(y_test,  test_proba)
pr_ref  = average_precision_score(y_train, train_proba)
pr_cur  = average_precision_score(y_test,  test_proba)

print("✅ Reports generated in 'reports/' folder")
print(f"Target rate ref={y_train.mean():.4f} cur={y_test.mean():.4f}")
print(f"ROC-AUC ref={roc_ref:.3f} cur={roc_cur:.3f}")
print(f"PR-AUC  ref={pr_ref:.3f} cur={pr_cur:.3f}")
