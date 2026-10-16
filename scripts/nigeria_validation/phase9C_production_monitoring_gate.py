from pathlib import Path
import json
import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data" / "nigeria_validation" / "splits" / "test_2025_2026.csv"
MODEL_DIR = ROOT / "models" / "nigeria_validation"
REPORT_DIR = ROOT / "reports" / "nigeria_validation" / "phase9"

CLS_MODEL = MODEL_DIR / "phase5_classification_hgb.joblib"
REG_MODEL = MODEL_DIR / "phase5_regression_hgb.joblib"

PHASE9A = REPORT_DIR / "phase9A_production_prediction_service.json"
PHASE9B = REPORT_DIR / "phase9B_operational_control_gate.json"

OUT_JSON = REPORT_DIR / "phase9C_production_monitoring_gate.json"
OUT_CSV = REPORT_DIR / "phase9C_production_monitoring_gate.csv"

FEATURES = [
    "School",
    "Term",
    "Class",
    "Arm",
    "Gender",
    "Subject",
    "Term_Number",
    "CA1",
    "CA2",
    "Assignment",
    "Project",
    "Attendance",
    "Previous_Term_Total",
    "Previous_Year_Average",
    "Previous_Term_Grade_Numeric",
    "Historical_Subject_Average",
    "Historical_Subject_Count",
    "Historical_Overall_Average",
    "Historical_Overall_Count",
]

FORBIDDEN = {
    "Exam",
    "Total",
    "Grade",
    "Final_Year_Result",
    "Promotion_Status",
    "Intervention",
}



THRESHOLD = 0.40

EXPECTED_CLASSIFICATION_HASH = (
    "3cc65dbd4c0e48ba65b28f713a8d32d9fec8bf0631dc8a3b5ef15e00823f608a"
)

EXPECTED_REGRESSION_HASH = (
    "f61b02c5542aa0a4d2d0c1b18540f05193cc5d0c5f1f4e5a46ae5a75e0a9973e"
)



# Bind monitoring checks to the current Phase 7C technical candidate manifest.
try:
    from release_manifest import load_expected_hashes
    _current_hashes = load_expected_hashes()
    EXPECTED_CLASSIFICATION_HASH = _current_hashes["classification"]
    EXPECTED_REGRESSION_HASH = _current_hashes["regression"]
    RELEASE_MANIFEST_ERROR = None
except Exception as _manifest_exc:
    EXPECTED_CLASSIFICATION_HASH = "__PHASE7C_MANIFEST_BLOCKED__"
    EXPECTED_REGRESSION_HASH = "__PHASE7C_MANIFEST_BLOCKED__"
    RELEASE_MANIFEST_ERROR = str(_manifest_exc)

def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def native(value):
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}

    if isinstance(value, list):
        return [native(v) for v in value]

    return value


def check(condition, gate, detail):
    return {
        "gate": gate,
        "status": "PASS" if bool(condition) else "FAIL",
        "detail": str(detail),
    }


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("EDUPREDICT NIGERIA VALIDATION — PHASE 9C")
    print("=" * 72)
    print("Purpose: Production Monitoring / Drift Gate")
    print()

    checks = []
    failures = []

    # --------------------------------------------------------
    # 1. REQUIRED ARTIFACTS
    # --------------------------------------------------------

    required = [
        DATA,
        CLS_MODEL,
        REG_MODEL,
        PHASE9A,
        PHASE9B,
    ]

    for path in required:
        ok = path.exists()

        checks.append(
            check(
                ok,
                "required_artifact",
                f"{path}",
            )
        )

        if not ok:
            failures.append(f"Missing artifact: {path}")

    if failures:
        print("Required artifact gate: FAIL")

        report = {
            "project": "EduPredict",
            "phase": "9C",
            "status": "FAIL",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "checks": checks,
            "failures": failures,
        }

        OUT_JSON.write_text(
            json.dumps(native(report), indent=2),
            encoding="utf-8",
        )

        pd.DataFrame(checks).to_csv(OUT_CSV, index=False)

        raise SystemExit(1)

    # --------------------------------------------------------
    # 2. LOAD DATA
    # --------------------------------------------------------

    df = pd.read_csv(DATA)

    print(f"Monitoring rows: {len(df):,}")

    session_values = sorted(
        df["Academic_Session"].astype(str).unique()
    )

    session_ok = session_values == ["2025/2026"]

    checks.append(
        check(
            session_ok,
            "session_integrity",
            f"expected=['2025/2026']; actual={session_values}",
        )
    )

    if not session_ok:
        failures.append("Production session mismatch.")

    # --------------------------------------------------------
    # 3. FEATURE CONTRACT
    # --------------------------------------------------------

    missing = sorted(set(FEATURES) - set(df.columns))
    forbidden_used = sorted(set(FEATURES) & FORBIDDEN)

    feature_ok = not missing and not forbidden_used

    checks.append(
        check(
            feature_ok,
            "feature_contract",
            f"missing={missing}; forbidden_used={forbidden_used}",
        )
    )

    if not feature_ok:
        failures.append("Production feature contract failed.")

    # --------------------------------------------------------
    # 4. MODEL HASH MONITORING
    # --------------------------------------------------------

    cls_hash = sha256_file(CLS_MODEL)
    reg_hash = sha256_file(REG_MODEL)

    cls_hash_ok = cls_hash == EXPECTED_CLASSIFICATION_HASH
    reg_hash_ok = reg_hash == EXPECTED_REGRESSION_HASH

    checks.append(
        check(
            cls_hash_ok,
            "classification_hash",
            f"expected={EXPECTED_CLASSIFICATION_HASH}; actual={cls_hash}",
        )
    )

    checks.append(
        check(
            reg_hash_ok,
            "regression_hash",
            f"expected={EXPECTED_REGRESSION_HASH}; actual={reg_hash}",
        )
    )

    if not cls_hash_ok:
        failures.append("Classification model hash changed.")

    if not reg_hash_ok:
        failures.append("Regression model hash changed.")

    # --------------------------------------------------------
    # 5. PRODUCTION PREDICTIONS
    # --------------------------------------------------------

    import joblib

    cls_model = joblib.load(CLS_MODEL)
    reg_model = joblib.load(REG_MODEL)

    X = df[FEATURES]

    probabilities = cls_model.predict_proba(X)[:, 1]
    exam_predictions = reg_model.predict(X)

    probabilities = np.asarray(probabilities, dtype=float)
    exam_predictions = np.asarray(exam_predictions, dtype=float)

    warnings_flag = probabilities >= THRESHOLD

    prediction_ok = (
        len(probabilities) == len(df)
        and len(exam_predictions) == len(df)
        and np.isfinite(probabilities).all()
        and np.isfinite(exam_predictions).all()
        and ((probabilities >= 0) & (probabilities <= 1)).all()
    )

    checks.append(
        check(
            prediction_ok,
            "prediction_integrity",
            f"classification={len(probabilities):,}; regression={len(exam_predictions):,}",
        )
    )

    if not prediction_ok:
        failures.append("Prediction integrity failed.")

    # --------------------------------------------------------
    # 6. OPERATIONAL MONITORING
    # --------------------------------------------------------

    early_warning_count = int(warnings_flag.sum())
    early_warning_rate = float(warnings_flag.mean())

    risk_min = float(probabilities.min())
    risk_max = float(probabilities.max())

    score_min = float(exam_predictions.min())
    score_max = float(exam_predictions.max())

    risk_mean = float(probabilities.mean())
    risk_median = float(np.median(probabilities))

    score_mean = float(exam_predictions.mean())
    score_median = float(np.median(exam_predictions))

    # Basic sanity monitoring.
    warning_rate_ok = 0.0 <= early_warning_rate <= 1.0
    risk_range_ok = 0.0 <= risk_min <= risk_max <= 1.0
    score_range_ok = (
        np.isfinite(score_min)
        and np.isfinite(score_max)
        and score_min <= score_max
    )

    checks.append(
        check(
            warning_rate_ok,
            "early_warning_rate",
            f"{early_warning_rate:.6f}",
        )
    )

    checks.append(
        check(
            risk_range_ok,
            "risk_probability_range",
            f"{risk_min:.6f}-{risk_max:.6f}",
        )
    )

    checks.append(
        check(
            score_range_ok,
            "predicted_exam_score_range",
            f"{score_min:.2f}-{score_max:.2f}",
        )
    )

    # --------------------------------------------------------
    # 7. DISTRIBUTION BASELINE
    # --------------------------------------------------------

    # The validated production baseline comes from the Phase 9A
    # prediction artifact. This establishes whether the same
    # frozen production model is reproducing the same population
    # distribution.

    phase9a = json.loads(
        PHASE9A.read_text(encoding="utf-8")
    )

    baseline = phase9a.get("metadata", {})

    baseline_rows = int(baseline.get("rows", 0))
    baseline_warning_rate = float(
        baseline.get("early_warning_rate", 0.0)
    )
    baseline_risk_min = float(
        baseline.get("risk_probability_min", 0.0)
    )
    baseline_risk_max = float(
        baseline.get("risk_probability_max", 0.0)
    )
    baseline_score_min = float(
        baseline.get("predicted_exam_score_min", 0.0)
    )
    baseline_score_max = float(
        baseline.get("predicted_exam_score_max", 0.0)
    )

    baseline_rows_ok = baseline_rows == len(df)

    warning_rate_delta = abs(
        early_warning_rate - baseline_warning_rate
    )

    risk_min_delta = abs(
        risk_min - baseline_risk_min
    )

    risk_max_delta = abs(
        risk_max - baseline_risk_max
    )

    score_min_delta = abs(
        score_min - baseline_score_min
    )

    score_max_delta = abs(
        score_max - baseline_score_max
    )

    # Very tight tolerance is intentional because this is a
    # reproducibility/operational monitor of the same frozen
    # dataset and model, not a general population drift detector.

    baseline_distribution_ok = (
        baseline_rows_ok
        and warning_rate_delta <= 1e-12
        and risk_min_delta <= 1e-12
        and risk_max_delta <= 1e-12
        and score_min_delta <= 1e-12
        and score_max_delta <= 1e-12
    )

    checks.append(
        check(
            baseline_distribution_ok,
            "phase9A_distribution_baseline",
            (
                f"warning_rate_delta={warning_rate_delta:.12g}; "
                f"risk_min_delta={risk_min_delta:.12g}; "
                f"risk_max_delta={risk_max_delta:.12g}; "
                f"score_min_delta={score_min_delta:.12g}; "
                f"score_max_delta={score_max_delta:.12g}"
            ),
        )
    )

    if not baseline_distribution_ok:
        failures.append(
            "Current production distribution differs from Phase 9A baseline."
        )

    # --------------------------------------------------------
    # 8. PHASE 9B CONTROL
    # --------------------------------------------------------

    phase9b = json.loads(
        PHASE9B.read_text(encoding="utf-8")
    )

    phase9b_status = phase9b.get("status")

    phase9b_ok = phase9b_status == "PASS"

    checks.append(
        check(
            phase9b_ok,
            "phase9B_operational_control",
            f"status={phase9b_status}",
        )
    )

    if not phase9b_ok:
        failures.append("Phase 9B operational control is not PASS.")

    # --------------------------------------------------------
    # 9. FINAL STATUS
    # --------------------------------------------------------

    status = "PASS" if not failures else "FAIL"

    monitoring = {
        "rows": len(df),
        "session": session_values,
        "threshold": THRESHOLD,
        "early_warning_count": early_warning_count,
        "early_warning_rate": early_warning_rate,
        "risk_probability": {
            "min": risk_min,
            "max": risk_max,
            "mean": risk_mean,
            "median": risk_median,
        },
        "predicted_exam_score": {
            "min": score_min,
            "max": score_max,
            "mean": score_mean,
            "median": score_median,
        },
        "baseline_phase9A": {
            "rows": baseline_rows,
            "early_warning_rate": baseline_warning_rate,
            "risk_probability_min": baseline_risk_min,
            "risk_probability_max": baseline_risk_max,
            "predicted_exam_score_min": baseline_score_min,
            "predicted_exam_score_max": baseline_score_max,
        },
        "deltas": {
            "early_warning_rate": warning_rate_delta,
            "risk_probability_min": risk_min_delta,
            "risk_probability_max": risk_max_delta,
            "predicted_exam_score_min": score_min_delta,
            "predicted_exam_score_max": score_max_delta,
        },
    }

    report = {
        "project": "EduPredict",
        "phase": "9C",
        "status": status,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "monitoring": monitoring,
        "model": {
            "version": "phase5_nigeria_production",
            "classification_sha256": cls_hash,
            "regression_sha256": reg_hash,
            "threshold": THRESHOLD,
        },
        "controls": {
            "session_integrity": bool(session_ok),
            "feature_contract": bool(feature_ok),
            "classification_hash": bool(cls_hash_ok),
            "regression_hash": bool(reg_hash_ok),
            "prediction_integrity": bool(prediction_ok),
            "early_warning_rate": bool(warning_rate_ok),
            "risk_probability_range": bool(risk_range_ok),
            "predicted_exam_score_range": bool(score_range_ok),
            "phase9A_distribution_baseline": bool(
                baseline_distribution_ok
            ),
            "phase9B_operational_control": bool(phase9b_ok),
        },
        "checks": checks,
        "failures": failures,
    }

    OUT_JSON.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    pd.DataFrame(checks).to_csv(
        OUT_CSV,
        index=False,
    )

    # --------------------------------------------------------
    # 10. CONSOLE OUTPUT
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("PHASE 9C PRODUCTION MONITORING GATE")
    print("=" * 72)

    print(f"Monitoring rows:             {len(df):,}")
    print(f"Early-warning flags:         {early_warning_count:,}")
    print(f"Early-warning rate:          {early_warning_rate:.4%}")
    print(
        f"Risk probability:            "
        f"{risk_min:.4f} - {risk_max:.4f}"
    )
    print(
        f"Predicted exam score:        "
        f"{score_min:.2f} - {score_max:.2f}"
    )

    print()
    print(f"Model hash integrity:        {'PASS' if cls_hash_ok and reg_hash_ok else 'FAIL'}")
    print(f"Prediction integrity:        {'PASS' if prediction_ok else 'FAIL'}")
    print(f"Distribution baseline:       {'PASS' if baseline_distribution_ok else 'FAIL'}")
    print(f"Phase 9B control:            {'PASS' if phase9b_ok else 'FAIL'}")

    print()
    print(f"PHASE 9C STATUS: {status}")

    print()
    print(f"Report: {OUT_JSON}")
    print(f"Checks: {OUT_CSV}")

    print("=" * 72)

    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
