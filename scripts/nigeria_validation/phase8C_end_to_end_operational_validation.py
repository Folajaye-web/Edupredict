from pathlib import Path
import hashlib
import json
import sys
import traceback

import joblib
import numpy as np
import pandas as pd


# ============================================================
# EDUPREDICT
# PHASE 8C — END-TO-END OPERATIONAL VALIDATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

APP = ROOT / "app" / "streamlit_app.py"
SRC = ROOT / "src"
PACKAGE = SRC / "edupredict"

DATA_DIR = ROOT / "data" / "nigeria_validation" / "splits"
MODEL_DIR = ROOT / "models" / "nigeria_validation"
REPORT_DIR = ROOT / "reports" / "nigeria_validation"

TEST_DATA = DATA_DIR / "test_2025_2026.csv"

CLS_MODEL = MODEL_DIR / "phase5_classification_hgb.joblib"
REG_MODEL = MODEL_DIR / "phase5_regression_hgb.joblib"

PHASE7A = REPORT_DIR / "phase7A_artifact_gate.json"
PHASE7B = REPORT_DIR / "phase7B_subgroup_safety_gate.json"
PHASE7C = REPORT_DIR / "phase7C_production_release_gate.json"
PHASE8A = REPORT_DIR / "phase8A_production_integration.json"
PHASE8B = REPORT_DIR / "phase8B_application_model_validation.json"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


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
    "Final_Year_Result",
    "Grade",
    "Intervention",
    "Promotion_Status",
    "Total",
}

THRESHOLD = 0.40


def sha256_file(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def check(name, status, detail):
    return {
        "check": name,
        "status": status,
        "detail": detail,
    }


def load_json(path):
    if not path.exists():
        return None

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def main():

    print("=" * 72)
    print("EDUPREDICT NIGERIA VALIDATION — PHASE 8C")
    print("=" * 72)
    print("Purpose: End-to-End Operational Validation")
    print()

    checks = []

    # ========================================================
    # 1. APPLICATION EXISTENCE
    # ========================================================

    app_ok = APP.exists()

    checks.append(
        check(
            "application_present",
            "PASS" if app_ok else "FAIL",
            str(APP),
        )
    )

    # ========================================================
    # 2. SOURCE/PACKAGE EXISTENCE
    # ========================================================

    checks.append(
        check(
            "source_package",
            "PASS" if PACKAGE.exists() else "FAIL",
            str(PACKAGE),
        )
    )

    # ========================================================
    # 3. PRODUCTION DATA
    # ========================================================

    data_ok = TEST_DATA.exists()

    checks.append(
        check(
            "final_test_dataset",
            "PASS" if data_ok else "FAIL",
            str(TEST_DATA),
        )
    )

    if not data_ok:
        raise SystemExit("Final-test dataset missing.")

    test = pd.read_csv(TEST_DATA)

    checks.append(
        check(
            "final_test_load",
            "PASS",
            f"{len(test):,} rows",
        )
    )

    # ========================================================
    # 4. SESSION
    # ========================================================

    sessions = sorted(
        test["Academic_Session"]
        .astype(str)
        .unique()
        .tolist()
    )

    session_ok = sessions == ["2025/2026"]

    checks.append(
        check(
            "production_session",
            "PASS" if session_ok else "FAIL",
            f"expected=['2025/2026']; actual={sessions}",
        )
    )

    # ========================================================
    # 5. FEATURE CONTRACT
    # ========================================================

    missing = sorted(
        set(FEATURES) - set(test.columns)
    )

    forbidden_used = sorted(
        set(FEATURES) & FORBIDDEN
    )

    contract_ok = (
        not missing
        and not forbidden_used
    )

    checks.append(
        check(
            "production_feature_contract",
            "PASS" if contract_ok else "FAIL",
            f"missing={missing}; forbidden_used={forbidden_used}",
        )
    )

    # ========================================================
    # 6. MODEL LOAD
    # ========================================================

    cls_ok = CLS_MODEL.exists()
    reg_ok = REG_MODEL.exists()

    checks.append(
        check(
            "classification_model_artifact",
            "PASS" if cls_ok else "FAIL",
            str(CLS_MODEL),
        )
    )

    checks.append(
        check(
            "regression_model_artifact",
            "PASS" if reg_ok else "FAIL",
            str(REG_MODEL),
        )
    )

    if not cls_ok or not reg_ok:
        raise SystemExit("Production model artifact missing.")

    classification_model = joblib.load(CLS_MODEL)
    regression_model = joblib.load(REG_MODEL)

    checks.append(
        check(
            "classification_model_load",
            "PASS",
            type(classification_model).__name__,
        )
    )

    checks.append(
        check(
            "regression_model_load",
            "PASS",
            type(regression_model).__name__,
        )
    )

    # ========================================================
    # 7. PRODUCTION PREDICTION
    # ========================================================

    X = test[FEATURES].copy()

    probabilities = (
        classification_model
        .predict_proba(X)[:, 1]
    )

    regression_predictions = (
        regression_model.predict(X)
    )

    classification_ok = (
        len(probabilities) == len(test)
        and np.isfinite(probabilities).all()
        and np.all(probabilities >= 0)
        and np.all(probabilities <= 1)
    )

    regression_ok = (
        len(regression_predictions) == len(test)
        and np.isfinite(regression_predictions).all()
    )

    checks.append(
        check(
            "classification_prediction",
            "PASS" if classification_ok else "FAIL",
            (
                f"{len(probabilities):,} predictions; "
                f"range={probabilities.min():.4f}-"
                f"{probabilities.max():.4f}"
            ),
        )
    )

    checks.append(
        check(
            "regression_prediction",
            "PASS" if regression_ok else "FAIL",
            (
                f"{len(regression_predictions):,} predictions; "
                f"range={regression_predictions.min():.2f}-"
                f"{regression_predictions.max():.2f}"
            ),
        )
    )

    # ========================================================
    # 8. EARLY-WARNING DECISION
    # ========================================================

    risk_flags = (
        probabilities >= THRESHOLD
    ).astype(int)

    threshold_ok = THRESHOLD == 0.40

    checks.append(
        check(
            "early_warning_threshold",
            "PASS" if threshold_ok else "FAIL",
            (
                f"threshold={THRESHOLD:.2f}; "
                f"flagged={int(risk_flags.sum())}; "
                f"population={len(risk_flags):,}"
            ),
        )
    )

    # ========================================================
    # 9. PREDICTION OUTPUT CONTRACT
    # ========================================================

    prediction_output = pd.DataFrame(
        {
            "risk_probability": probabilities,
            "early_warning": risk_flags,
            "predicted_exam_score": regression_predictions,
        }
    )

    output_ok = (
        len(prediction_output) == len(test)
        and prediction_output[
            "risk_probability"
        ].between(0, 1).all()
        and np.isfinite(
            prediction_output[
                "predicted_exam_score"
            ]
        ).all()
    )

    checks.append(
        check(
            "prediction_output_contract",
            "PASS" if output_ok else "FAIL",
            (
                f"rows={len(prediction_output):,}; "
                f"columns={list(prediction_output.columns)}"
            ),
        )
    )

    # ========================================================
    # 10. APPLICATION SOURCE VALIDATION
    # ========================================================

    app_text = APP.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    source_markers = [
        "streamlit",
        "edupredict",
        "model",
        "Dashboard",
        "Students",
        "Attendance",
    ]

    marker_results = {}

    for marker in source_markers:

        found = marker.lower() in app_text.lower()

        marker_results[marker] = found

        checks.append(
            check(
                "application_source_marker",
                "PASS" if found else "REVIEW",
                f"{marker}: found={found}",
            )
        )

    # ========================================================
    # 11. STREAMLIT APPTEST
    # ========================================================

    app_test_status = "REVIEW"
    app_test_detail = ""

    try:

        from streamlit.testing.v1 import AppTest

        if str(SRC) not in sys.path:
            sys.path.insert(0, str(SRC))

        print("Running Streamlit AppTest...")

        at = AppTest.from_file(
            str(APP),
            default_timeout=30,
        )

        at.run(timeout=30)

        if len(at.exception) == 0:

            app_test_status = "PASS"

            app_test_detail = (
                f"application executed successfully; "
                f"exceptions=0; "
                f"dataframes={len(at.dataframe)}; "
                f"buttons={len(at.button)}; "
                f"selectboxes={len(at.selectbox)}"
            )

        else:

            app_test_status = "FAIL"

            app_test_detail = (
                f"application exceptions={len(at.exception)}"
            )

    except Exception as exc:

        app_test_status = "FAIL"

        app_test_detail = (
            f"{type(exc).__name__}: {exc}"
        )

    checks.append(
        check(
            "streamlit_app_test",
            app_test_status,
            app_test_detail,
        )
    )

    # ========================================================
    # 12. REPRODUCIBILITY
    # ========================================================

    probabilities_repeat = (
        classification_model
        .predict_proba(X)[:, 1]
    )

    regression_repeat = (
        regression_model.predict(X)
    )

    classification_error = float(
        np.max(
            np.abs(
                probabilities -
                probabilities_repeat
            )
        )
    )

    regression_error = float(
        np.max(
            np.abs(
                regression_predictions -
                regression_repeat
            )
        )
    )

    reproducibility_ok = (
        classification_error == 0.0
        and regression_error == 0.0
    )

    checks.append(
        check(
            "prediction_reproducibility",
            "PASS" if reproducibility_ok else "FAIL",
            (
                f"classification_error="
                f"{classification_error}; "
                f"regression_error="
                f"{regression_error}"
            ),
        )
    )

    # ========================================================
    # 13. MODEL HASHES
    # ========================================================

    cls_hash = sha256_file(CLS_MODEL)
    reg_hash = sha256_file(REG_MODEL)

    checks.append(
        check(
            "classification_hash",
            "PASS",
            cls_hash,
        )
    )

    checks.append(
        check(
            "regression_hash",
            "PASS",
            reg_hash,
        )
    )

    # ========================================================
    # 14. PREVIOUS GATES
    # ========================================================

    previous_gates = {
        "phase7A": PHASE7A,
        "phase7B": PHASE7B,
        "phase7C": PHASE7C,
        "phase8A": PHASE8A,
        "phase8B": PHASE8B,
    }

    previous_gate_status = {}

    for name, path in previous_gates.items():

        data = load_json(path)

        if data is None:

            status = "FAIL"
            detail = "gate report missing"

        else:

            value = str(
                data.get("status", "")
            )

            if name == "phase7C":

                status = (
                    "PASS"
                    if value == "TECHNICAL_RELEASE_READY"
                    else "FAIL"
                )

            else:

                status = (
                    "PASS"
                    if value == "PASS"
                    else "FAIL"
                )

            detail = f"status={value}"

        previous_gate_status[name] = status

        checks.append(
            check(
                "previous_gate",
                status,
                f"{name}: {detail}",
            )
        )

    # ========================================================
    # 15. FINAL STATUS
    # ========================================================

    failures = [
        c for c in checks
        if c["status"] == "FAIL"
    ]

    reviews = [
        c for c in checks
        if c["status"] == "REVIEW"
    ]

    if failures:
        final_status = "NOT_QUALIFIED"
    elif reviews:
        final_status = "CONDITIONAL_REVIEW"
    else:
        final_status = "PASS"

    report = {
        "project": "EduPredict",
        "phase": "8C",
        "gate": "End-to-End Operational Validation",
        "status": final_status,
        "final_test_rows": len(test),
        "production_threshold": THRESHOLD,
        "risk_flags": int(risk_flags.sum()),
        "classification_probability_range": [
            float(probabilities.min()),
            float(probabilities.max()),
        ],
        "predicted_exam_score_range": [
            float(regression_predictions.min()),
            float(regression_predictions.max()),
        ],
        "classification_sha256": cls_hash,
        "regression_sha256": reg_hash,
        "previous_gates": previous_gate_status,
        "checks": checks,
        "failures": failures,
        "reviews": reviews,
    }

    json_path = (
        REPORT_DIR /
        "phase8C_end_to_end_operational_validation.json"
    )

    csv_path = (
        REPORT_DIR /
        "phase8C_end_to_end_operational_validation.csv"
    )

    json_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    pd.DataFrame(checks).to_csv(
        csv_path,
        index=False,
    )

    # ========================================================
    # CONSOLE
    # ========================================================

    print()
    print("=" * 72)
    print("PHASE 8C RESULT")
    print("=" * 72)

    for item in checks:

        print(
            f"{item['check']:<32} "
            f"{item['status']:<8} "
            f"{item['detail']}"
        )

    print()
    print(f"Final-test rows:       {len(test):,}")
    print(f"Early-warning flags:   {int(risk_flags.sum()):,}")
    print(
        f"Risk probability:      "
        f"{probabilities.min():.4f}-"
        f"{probabilities.max():.4f}"
    )
    print(
        f"Predicted exam score:  "
        f"{regression_predictions.min():.2f}-"
        f"{regression_predictions.max():.2f}"
    )

    print()
    print(
        f"PHASE 8C STATUS: {final_status}"
    )

    print()
    print(f"Report: {json_path}")
    print(f"Checks: {csv_path}")
    print("=" * 72)

    if final_status == "NOT_QUALIFIED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()