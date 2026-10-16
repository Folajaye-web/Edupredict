"""Phase 5 candidate predictor guarded by current feature/release evidence.

This module does not authorise a school pilot. It only runs the candidate
artifacts after the corrected feature pipeline, Phase 5 report and Phase 7C
technical release manifest agree on provenance and hashes.
"""
from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "nigeria_validation"
REPORT_DIR = ROOT / "reports" / "nigeria_validation"
PHASE9_REPORT_DIR = REPORT_DIR / "phase9"
CLASSIFICATION_MODEL = MODEL_DIR / "phase5_classification_hgb.joblib"
REGRESSION_MODEL = MODEL_DIR / "phase5_regression_hgb.joblib"
FEATURE_REPORT = REPORT_DIR / "feature_build_report.json"
PHASE5_REPORT = REPORT_DIR / "phase5_final_holdout_results.json"
PHASE7_CONTRACT = REPORT_DIR / "phase7_production_contract.json"
PHASE7C_REPORT = REPORT_DIR / "phase7C_production_release_gate.json"
RELEASE_MANIFEST = REPORT_DIR / "phase7C_release_manifest.json"
FEATURE_PIPELINE_VERSION = "2.0-period-safe-school-aware"
REQUIRED_SKLEARN_VERSION = "1.9.0"
THRESHOLD = 0.40

FEATURES = [
    "School", "Term", "Class", "Arm", "Gender", "Subject", "Term_Number",
    "CA1", "CA2", "Assignment", "Project", "Attendance", "Previous_Term_Total",
    "Previous_Year_Average", "Previous_Term_Grade_Numeric", "Historical_Subject_Average",
    "Historical_Subject_Count", "Historical_Overall_Average", "Historical_Overall_Count",
]
FORBIDDEN = {
    "Exam", "Final_Year_Result", "Grade", "Intervention", "Promotion_Status", "Total",
}
NUMERIC_FEATURES = [
    "Term_Number", "CA1", "CA2", "Assignment", "Project", "Attendance",
    "Previous_Term_Total", "Previous_Year_Average", "Previous_Term_Grade_Numeric",
    "Historical_Subject_Average", "Historical_Subject_Count", "Historical_Overall_Average",
    "Historical_Overall_Count",
]


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path):
    if not path.is_file():
        raise RuntimeError(f"Required validation evidence is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Validation evidence is unreadable: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Validation evidence must be a JSON object: {path}")
    return value


def verify_artifacts():
    if sklearn.__version__ != REQUIRED_SKLEARN_VERSION:
        raise RuntimeError(
            f"Model runtime mismatch: artifacts require scikit-learn {REQUIRED_SKLEARN_VERSION}; "
            f"current runtime is {sklearn.__version__}. Activate the project .venv and install requirements."
        )
    for path in (CLASSIFICATION_MODEL, REGRESSION_MODEL):
        if not path.is_file():
            raise FileNotFoundError(f"Required Phase 5 model is missing: {path}")

    feature_report = _read_json(FEATURE_REPORT)
    phase5_report = _read_json(PHASE5_REPORT)
    contract = _read_json(PHASE7_CONTRACT)
    gate = _read_json(PHASE7C_REPORT)
    manifest = _read_json(RELEASE_MANIFEST)

    if feature_report.get("status") != "PASS" or feature_report.get("feature_pipeline_version") != FEATURE_PIPELINE_VERSION:
        raise RuntimeError("Corrected period-safe feature pipeline has not passed. Run the Phase 10E revalidation workflow first.")
    if phase5_report.get("status") != "PASS" or phase5_report.get("feature_pipeline_version") != FEATURE_PIPELINE_VERSION:
        raise RuntimeError("Phase 5 models are stale or were not trained from the corrected feature pipeline.")
    if phase5_report.get("feature_input_sha256") != feature_report.get("input_sha256"):
        raise RuntimeError("Phase 5 report and feature report refer to different source CSV versions.")
    if contract.get("status") != "TECHNICAL_QUALIFICATION_PASS":
        raise RuntimeError(f"Phase 7 qualification is not cleared: {contract.get('status')!r}.")
    if gate.get("status") != "TECHNICAL_RELEASE_READY" or manifest.get("status") != "TECHNICAL_RELEASE_READY":
        raise RuntimeError("Phase 7C technical release gate has not passed; prediction is disabled.")
    if manifest.get("pilot_authorised", False) is True or gate.get("pilot_authorised", False) is True:
        # This code path must never be the mechanism that grants pilot approval.
        raise RuntimeError("Invalid release evidence: pilot authorisation must be handled separately by Phase 10C.")
    if abs(float(manifest.get("threshold", -1)) - THRESHOLD) > 1e-12:
        raise RuntimeError("Production threshold differs from the Phase 7C release manifest.")

    cls_hash = sha256_file(CLASSIFICATION_MODEL)
    reg_hash = sha256_file(REGRESSION_MODEL)
    if cls_hash != manifest.get("classification_sha256"):
        raise RuntimeError("Classification model hash does not match the Phase 7C manifest.")
    if reg_hash != manifest.get("regression_sha256"):
        raise RuntimeError("Regression model hash does not match the Phase 7C manifest.")

    return {
        "classification_sha256": cls_hash,
        "regression_sha256": reg_hash,
        "feature_pipeline_version": FEATURE_PIPELINE_VERSION,
        "scikit_learn": sklearn.__version__,
        "technical_release_status": gate["status"],
        "pilot_authorised": False,
    }


def validate_input(df):
    missing = sorted(set(FEATURES) - set(df.columns))
    if missing:
        raise ValueError(f"Production feature contract failed. Missing: {missing}")
    forbidden_used = sorted(set(FEATURES) & FORBIDDEN)
    if forbidden_used:
        raise ValueError(f"Forbidden predictors entered production feature list: {forbidden_used}")
    if df.empty:
        raise ValueError("Production input contains zero rows.")
    numeric = df[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
    required_numeric = ["Term_Number", "CA1", "CA2", "Assignment", "Project", "Attendance", "Historical_Subject_Count", "Historical_Overall_Count"]
    if not numeric[required_numeric].notna().all().all():
        raise ValueError("Required numeric production features contain missing or non-numeric values.")
    optional_history = ["Previous_Term_Total", "Previous_Year_Average", "Previous_Term_Grade_Numeric", "Historical_Subject_Average", "Historical_Overall_Average"]
    for col in optional_history:
        non_null = numeric[col].dropna()
        if not np.isfinite(non_null.to_numpy(dtype=float)).all():
            raise ValueError(f"{col} contains non-finite values other than missing history.")
    for col in ["CA1", "CA2", "Assignment", "Project"]:
        if not df[col].between(0, 10).all():
            raise ValueError(f"{col} must be within the trained 0–10 scoring scale.")
    if not df["Attendance"].between(0, 100).all():
        raise ValueError("Attendance must be a percentage between 0 and 100.")
    if not df["Term_Number"].isin([1, 2, 3]).all():
        raise ValueError("Term_Number must be 1, 2, or 3.")
    if (df[["Historical_Subject_Count", "Historical_Overall_Count"]] < 0).any().any():
        raise ValueError("Historical counts cannot be negative.")
    return True


def predict(df):
    hashes = verify_artifacts()
    validate_input(df)
    cls = joblib.load(CLASSIFICATION_MODEL)
    reg = joblib.load(REGRESSION_MODEL)
    X = df[FEATURES]

    classes = list(getattr(cls, "classes_", []))
    if 1 not in classes:
        raise RuntimeError(f"Classifier does not contain positive class 1: {classes}")
    positive_index = classes.index(1)
    risk_probability = cls.predict_proba(X)[:, positive_index]
    predicted_exam_score = np.asarray(reg.predict(X), dtype=float)

    if not np.isfinite(risk_probability).all() or ((risk_probability < 0) | (risk_probability > 1)).any():
        raise RuntimeError("Classifier returned invalid probabilities.")
    if not np.isfinite(predicted_exam_score).all() or ((predicted_exam_score < 0) | (predicted_exam_score > 60)).any():
        raise RuntimeError("Regressor returned scores outside the trained 0–60 exam scale.")

    early_warning = risk_probability >= THRESHOLD
    output = pd.DataFrame({
        "risk_probability": risk_probability,
        "early_warning": early_warning,
        "predicted_exam_score": predicted_exam_score,
    })
    metadata = {
        "model_version": "phase5_nigeria_candidate",
        "classification_sha256": hashes["classification_sha256"],
        "regression_sha256": hashes["regression_sha256"],
        "feature_pipeline_version": hashes["feature_pipeline_version"],
        "scikit_learn": hashes["scikit_learn"],
        "threshold": THRESHOLD,
        "rows": int(len(df)),
        "early_warning_count": int(early_warning.sum()),
        "early_warning_rate": float(np.mean(early_warning)),
        "risk_probability_min": float(np.min(risk_probability)),
        "risk_probability_max": float(np.max(risk_probability)),
        "predicted_exam_score_min": float(np.min(predicted_exam_score)),
        "predicted_exam_score_max": float(np.max(predicted_exam_score)),
        "pilot_authorised": False,
    }
    PHASE9_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    return output, metadata
