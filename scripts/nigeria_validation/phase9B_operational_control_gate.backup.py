from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd


# ============================================================
# EDUPREDICT PHASE 9B
# OPERATIONAL CONTROL / AUDIT GATE
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "nigeria_validation" / "splits"
MODEL_DIR = ROOT / "models" / "nigeria_validation"
REPORT_DIR = ROOT / "reports" / "nigeria_validation"
PHASE9_DIR = REPORT_DIR / "phase9"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
PHASE9_DIR.mkdir(parents=True, exist_ok=True)

TEST = DATA_DIR / "test_2025_2026.csv"

CLS_MODEL = MODEL_DIR / "phase5_classification_hgb.joblib"
REG_MODEL = MODEL_DIR / "phase5_regression_hgb.joblib"

PHASE9A_REPORT = PHASE9_DIR / "phase9A_production_prediction_service.json"

EXPECTED_CLASSIFICATION_HASH = (
    "3cc65dbd4c0e48ba65b28f713a8d32d9fec8bf0631dc8a3b5ef15e00823f608a"
)

EXPECTED_REGRESSION_HASH = (
    "f61b02c5542aa0a4d2d0c1b18540f05193cc5d0c5f1f4e5a46ae5a75e0a9973e"
)

EXPECTED_THRESHOLD = 0.40

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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def check(name, condition, detail, checks, failures):
    status = "PASS" if condition else "FAIL"

    checks.append(
        {
            "check": name,
            "status": status,
            "detail": str(detail),
        }
    )

    if not condition:
        failures.append(
            {
                "check": name,
                "detail": str(detail),
            }
        )

    return condition


def main():

    print("=" * 72)
    print("EDUPREDICT NIGERIA VALIDATION — PHASE 9B")
    print("=" * 72)
    print("Purpose: Operational Control / Audit Gate")
    print()

    checks = []
    failures = []

    # --------------------------------------------------------
    # 1. REQUIRED ARTIFACTS
    # --------------------------------------------------------

    check(
        "final_test_dataset",
        TEST.exists(),
        TEST,
        checks,
        failures,
    )

    check(
        "classification_model",
        CLS_MODEL.exists(),
        CLS_MODEL,
        checks,
        failures,
    )

    check(
        "regression_model",
        REG_MODEL.exists(),
        REG_MODEL,
        checks,
        failures,
    )

    check(
        "phase9A_report",
        PHASE9A_REPORT.exists(),
        PHASE9A_REPORT,
        checks,
        failures,
    )

    if failures:
        status = "FAIL"

        print("=" * 72)
        print("PHASE 9B RESULT")
        print("=" * 72)
        print("Required artifact checks failed.")
        print(f"Failures: {len(failures)}")
        print()
        print("PHASE 9B STATUS: FAIL")

        report = {
            "project": "EduPredict",
            "phase": "9B",
            "status": status,
            "failures": failures,
        }

        path = PHASE9_DIR / "phase9B_operational_control_gate.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        pd.DataFrame(checks).to_csv(
            PHASE9_DIR / "phase9B_operational_control_gate.csv",
            index=False,
        )

        return

    # --------------------------------------------------------
    # 2. LOAD PHASE 9A CONTRACT
    # --------------------------------------------------------

    phase9a = json.loads(
        PHASE9A_REPORT.read_text(encoding="utf-8")
    )

    check(
        "phase9A_status",
        phase9a.get("status") == "PASS",
        phase9a.get("status"),
        checks,
        failures,
    )

    metadata = phase9a.get("metadata", {})

    # --------------------------------------------------------
    # 3. VERIFY FROZEN MODEL HASHES
    # --------------------------------------------------------

    cls_hash = sha256_file(CLS_MODEL)
    reg_hash = sha256_file(REG_MODEL)

    check(
        "classification_hash",
        cls_hash == EXPECTED_CLASSIFICATION_HASH,
        f"expected={EXPECTED_CLASSIFICATION_HASH}; actual={cls_hash}",
        checks,
        failures,
    )

    check(
        "regression_hash",
        reg_hash == EXPECTED_REGRESSION_HASH,
        f"expected={EXPECTED_REGRESSION_HASH}; actual={reg_hash}",
        checks,
        failures,
    )

    check(
        "phase9A_classification_hash",
        metadata.get("classification_sha256") == cls_hash,
        f"phase9A={metadata.get('classification_sha256')}; actual={cls_hash}",
        checks,
        failures,
    )

    check(
        "phase9A_regression_hash",
        metadata.get("regression_sha256") == reg_hash,
        f"phase9A={metadata.get('regression_sha256')}; actual={reg_hash}",
        checks,
        failures,
    )

    # --------------------------------------------------------
    # 4. VERIFY THRESHOLD
    # --------------------------------------------------------

    threshold = float(metadata.get("threshold", -1))

    check(
        "threshold_contract",
        threshold == EXPECTED_THRESHOLD,
        f"expected={EXPECTED_THRESHOLD}; actual={threshold}",
        checks,
        failures,
    )

    # --------------------------------------------------------
    # 5. LOAD PRODUCTION DATA
    # --------------------------------------------------------

    df = pd.read_csv(TEST)

    check(
        "input_rows",
        len(df) == int(metadata.get("rows", -1)),
        f"dataset={len(df)}; phase9A={metadata.get('rows')}",
        checks,
        failures,
    )

    # --------------------------------------------------------
    # 6. FEATURE CONTRACT
    # --------------------------------------------------------

    missing = sorted(set(FEATURES) - set(df.columns))
    forbidden_used = sorted(set(FEATURES) & FORBIDDEN)

    check(
        "feature_contract",
        not missing and not forbidden_used,
        f"missing={missing}; forbidden_used={forbidden_used}",
        checks,
        failures,
    )

    # --------------------------------------------------------
    # 7. SESSION CONTRACT
    # --------------------------------------------------------

    sessions = sorted(
        df["Academic_Session"].astype(str).unique()
    )

    check(
        "production_session",
        sessions == ["2025/2026"],
        f"expected=['2025/2026']; actual={sessions}",
        checks,
        failures,
    )

    # --------------------------------------------------------
    # 8. MODEL LOADABILITY
    # --------------------------------------------------------

    try:
        cls_model = joblib.load(CLS_MODEL)
        cls_loaded = True
        cls_detail = type(cls_model).__name__
    except Exception as exc:
        cls_loaded = False
        cls_detail = repr(exc)

    check(
        "classification_model_load",
        cls_loaded,
        cls_detail,
        checks,
        failures,
    )

    try:
        reg_model = joblib.load(REG_MODEL)
        reg_loaded = True
        reg_detail = type(reg_model).__name__
    except Exception as exc:
        reg_loaded = False
        reg_detail = repr(exc)

    check(
        "regression_model_load",
        reg_loaded,
        reg_detail,
        checks,
        failures,
    )

    # --------------------------------------------------------
    # 9. RE-RUN PRODUCTION PREDICTIONS
    # --------------------------------------------------------

    prediction_contract_ok = False
    reproducibility_ok = False

    if cls_loaded and reg_loaded and not missing:

        X = df[FEATURES]

        try:
            risk = cls_model.predict_proba(X)[:, 1]
            score = reg_model.predict(X)

            early_warning = (risk >= threshold).astype(int)

            prediction_contract_ok = (
                len(risk) == len(df)
                and len(score) == len(df)
                and len(early_warning) == len(df)
                and np.isfinite(risk).all()
                and np.isfinite(score).all()
            )

            # ------------------------------------------------
            # 10. RANGE VALIDATION
            # ------------------------------------------------

            range_ok = (
                float(np.min(risk)) >= 0.0
                and float(np.max(risk)) <= 1.0
                and float(np.min(score)) >= 0.0
                and float(np.max(score)) <= 100.0
            )

            check(
                "prediction_contract",
                prediction_contract_ok,
                f"rows={len(risk)}",
                checks,
                failures,
            )

            check(
                "prediction_ranges",
                range_ok,
                (
                    f"risk={float(np.min(risk)):.4f}-"
                    f"{float(np.max(risk)):.4f}; "
                    f"score={float(np.min(score)):.4f}-"
                    f"{float(np.max(score)):.4f}"
                ),
                checks,
                failures,
            )

            # ------------------------------------------------
            # 11. REPRODUCIBILITY AGAINST PHASE 9A
            # ------------------------------------------------

            phase9a_risk_min = float(
                metadata["risk_probability_min"]
            )

            phase9a_risk_max = float(
                metadata["risk_probability_max"]
            )

            phase9a_score_min = float(
                metadata["predicted_exam_score_min"]
            )

            phase9a_score_max = float(
                metadata["predicted_exam_score_max"]
            )

            reproducibility_ok = (
                abs(float(np.min(risk)) - phase9a_risk_min) < 1e-12
                and abs(float(np.max(risk)) - phase9a_risk_max) < 1e-12
                and abs(float(np.min(score)) - phase9a_score_min) < 1e-12
                and abs(float(np.max(score)) - phase9a_score_max) < 1e-12
            )

            check(
                "phase9A_reproducibility",
                reproducibility_ok,
                (
                    f"risk_min_error="
                    f"{abs(float(np.min(risk)) - phase9a_risk_min):.12g}; "
                    f"risk_max_error="
                    f"{abs(float(np.max(risk)) - phase9a_risk_max):.12g}; "
                    f"score_min_error="
                    f"{abs(float(np.min(score)) - phase9a_score_min):.12g}; "
                    f"score_max_error="
                    f"{abs(float(np.max(score)) - phase9a_score_max):.12g}"
                ),
                checks,
                failures,
            )

            # ------------------------------------------------
            # 12. EARLY-WARNING CONTRACT
            # ------------------------------------------------

            warning_count = int(early_warning.sum())

            expected_warning_count = int(
                metadata["early_warning_count"]
            )

            check(
                "early_warning_contract",
                warning_count == expected_warning_count,
                f"expected={expected_warning_count}; actual={warning_count}",
                checks,
                failures,
            )

        except Exception as exc:

            check(
                "production_prediction_execution",
                False,
                repr(exc),
                checks,
                failures,
            )

    # --------------------------------------------------------
    # 13. BUILD AUDIT RECORD
    # --------------------------------------------------------

    status = "PASS" if not failures else "FAIL"

    audit_record = {
        "project": "EduPredict",
        "phase": "9B",
        "status": status,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "dataset": str(TEST),
            "rows": int(len(df)),
            "session": sessions,
        },
        "model": {
            "version": metadata.get(
                "model_version",
                "unknown",
            ),
            "classification_sha256": cls_hash,
            "regression_sha256": reg_hash,
            "threshold": threshold,
        },
        "prediction": {
            "rows": int(len(df)),
            "early_warning_count": (
                int(early_warning.sum())
                if "early_warning" in locals()
                else None
            ),
            "early_warning_rate": (
                float(np.mean(early_warning))
                if "early_warning" in locals()
                else None
            ),
            "risk_probability_min": (
                float(np.min(risk))
                if "risk" in locals()
                else None
            ),
            "risk_probability_max": (
                float(np.max(risk))
                if "risk" in locals()
                else None
            ),
            "predicted_exam_score_min": (
                float(np.min(score))
                if "score" in locals()
                else None
            ),
            "predicted_exam_score_max": (
                float(np.max(score))
                if "score" in locals()
                else None
            ),
        },
        "controls": {
            "artifact_integrity": cls_hash == EXPECTED_CLASSIFICATION_HASH
            and reg_hash == EXPECTED_REGRESSION_HASH,
            "feature_contract": not missing and not forbidden_used,
            "session_contract": sessions == ["2025/2026"],
            "prediction_contract": prediction_contract_ok,
            "reproducibility": reproducibility_ok,
        },
        "failures": failures,
    }

    # --------------------------------------------------------
    # 14. WRITE REPORTS
    # --------------------------------------------------------

    report_path = (
        PHASE9_DIR
        / "phase9B_operational_control_gate.json"
    )

    checks_path = (
        PHASE9_DIR
        / "phase9B_operational_control_gate.csv"
    )

    audit_path = (
        PHASE9_DIR
        / "phase9B_production_audit_record.json"
    )

    report_path.write_text(
        json.dumps(audit_record, indent=2),
        encoding="utf-8",
    )

    pd.DataFrame(checks).to_csv(
        checks_path,
        index=False,
    )

    audit_path.write_text(
        json.dumps(audit_record, indent=2),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # 15. CONSOLE OUTPUT
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("PHASE 9B RESULT")
    print("=" * 72)

    for item in checks:
        print(
            f"{item['check']:<34} "
            f"{item['status']:<7} "
            f"{item['detail']}"
        )

    print()
    print(f"PHASE 9B STATUS: {status}")

    print()
    print(f"Report:  {report_path}")
    print(f"Checks:  {checks_path}")
    print(f"Audit:   {audit_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()