from pathlib import Path
import hashlib
import json
import sys

import joblib
import numpy as np
import pandas as pd


# ============================================================
# EDUPREDICT
# PHASE 8B — APPLICATION / MODEL INTEGRATION VALIDATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "nigeria_validation" / "splits"
MODEL_DIR = ROOT / "models" / "nigeria_validation"
REPORT_DIR = ROOT / "reports" / "nigeria_validation"
APP_DIR = ROOT / "app"
SRC_DIR = ROOT / "src"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# FROZEN PRODUCTION CONTRACT
# ------------------------------------------------------------

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

TARGET_CLS = "target_below_credit"
TARGET_REG = "target_exam_score"

CLASSIFICATION_MODEL = (
    MODEL_DIR / "phase5_classification_hgb.joblib"
)

REGRESSION_MODEL = (
    MODEL_DIR / "phase5_regression_hgb.joblib"
)

FINAL_TEST = (
    DATA_DIR / "test_2025_2026.csv"
)

PHASE7A = (
    REPORT_DIR / "phase7A_artifact_gate.json"
)

PHASE7B = (
    REPORT_DIR / "phase7B_subgroup_safety_gate.json"
)

PHASE7C = (
    REPORT_DIR / "phase7C_production_release_gate.json"
)

PHASE7_CONTRACT = (
    REPORT_DIR / "phase7_production_contract.json"
)

PRODUCTION_THRESHOLD = 0.40


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def result(check, item, status, detail):
    return {
        "check": check,
        "item": item,
        "status": status,
        "detail": detail,
    }


def pass_fail(condition):
    return "PASS" if condition else "FAIL"


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    print("=" * 72)
    print("EDUPREDICT NIGERIA VALIDATION — PHASE 8B")
    print("=" * 72)
    print("Purpose: Application / Model Integration Validation")
    print()

    checks = []

    # --------------------------------------------------------
    # 1. PROJECT STRUCTURE
    # --------------------------------------------------------

    required_paths = [
        ("application", APP_DIR / "streamlit_app.py"),
        ("source", SRC_DIR),
        ("package", SRC_DIR / "edupredict"),
        ("classification_model", CLASSIFICATION_MODEL),
        ("regression_model", REGRESSION_MODEL),
        ("final_test_dataset", FINAL_TEST),
    ]

    for name, path in required_paths:

        ok = path.exists()

        checks.append(
            result(
                "required_artifact",
                name,
                pass_fail(ok),
                str(path),
            )
        )

    # --------------------------------------------------------
    # 2. LOAD FINAL TEST DATA
    # --------------------------------------------------------

    test = pd.read_csv(FINAL_TEST)

    required_columns = set(FEATURES + [
        TARGET_CLS,
        TARGET_REG,
    ])

    missing = sorted(
        required_columns - set(test.columns)
    )

    forbidden_used = sorted(
        set(FEATURES).intersection(FORBIDDEN)
    )

    checks.append(
        result(
            "feature_contract",
            "final_test_dataset",
            pass_fail(
                not missing and not forbidden_used
            ),
            f"missing={missing}; forbidden_used={forbidden_used}",
        )
    )

    # --------------------------------------------------------
    # 3. SESSION CONTRACT
    # --------------------------------------------------------

    sessions = sorted(
        test["Academic_Session"]
        .astype(str)
        .unique()
        .tolist()
    )

    session_ok = sessions == ["2025/2026"]

    checks.append(
        result(
            "session_contract",
            "final_test_dataset",
            pass_fail(session_ok),
            f"expected=['2025/2026']; actual={sessions}",
        )
    )

    # --------------------------------------------------------
    # 4. LOAD FROZEN MODELS
    # --------------------------------------------------------

    print("Loading frozen production models...")

    classification_model = joblib.load(
        CLASSIFICATION_MODEL
    )

    regression_model = joblib.load(
        REGRESSION_MODEL
    )

    checks.append(
        result(
            "model_load",
            "classification",
            "PASS",
            type(classification_model).__name__,
        )
    )

    checks.append(
        result(
            "model_load",
            "regression",
            "PASS",
            type(regression_model).__name__,
        )
    )

    # --------------------------------------------------------
    # 5. PREDICTION INPUT
    # --------------------------------------------------------

    X_test = test[FEATURES].copy()

    prediction_input_ok = (
        list(X_test.columns) == FEATURES
        and len(X_test) == len(test)
    )

    checks.append(
        result(
            "prediction_input_contract",
            "production_features",
            pass_fail(prediction_input_ok),
            f"columns={len(X_test.columns)}; rows={len(X_test)}",
        )
    )

    # --------------------------------------------------------
    # 6. CLASSIFICATION PREDICTION
    # --------------------------------------------------------

    probabilities = (
        classification_model
        .predict_proba(X_test)[:, 1]
    )

    probability_ok = (
        len(probabilities) == len(test)
        and np.isfinite(probabilities).all()
        and np.all(probabilities >= 0)
        and np.all(probabilities <= 1)
    )

    checks.append(
        result(
            "classification_prediction",
            "predict_proba",
            pass_fail(probability_ok),
            (
                f"rows={len(probabilities)}; "
                f"range={probabilities.min():.4f}-"
                f"{probabilities.max():.4f}"
            ),
        )
    )

    # --------------------------------------------------------
    # 7. PRODUCTION THRESHOLD
    # --------------------------------------------------------

    predictions = (
        probabilities >= PRODUCTION_THRESHOLD
    ).astype(int)

    threshold_ok = PRODUCTION_THRESHOLD == 0.40

    checks.append(
        result(
            "threshold_contract",
            "classification",
            pass_fail(threshold_ok),
            (
                f"expected=0.40; "
                f"actual={PRODUCTION_THRESHOLD:.2f}; "
                f"flagged={int(predictions.sum())}"
            ),
        )
    )

    # --------------------------------------------------------
    # 8. REGRESSION PREDICTION
    # --------------------------------------------------------

    regression_predictions = (
        regression_model.predict(X_test)
    )

    regression_ok = (
        len(regression_predictions) == len(test)
        and np.isfinite(regression_predictions).all()
    )

    checks.append(
        result(
            "regression_prediction",
            "predict",
            pass_fail(regression_ok),
            (
                f"rows={len(regression_predictions)}; "
                f"range={regression_predictions.min():.2f}-"
                f"{regression_predictions.max():.2f}"
            ),
        )
    )

    # --------------------------------------------------------
    # 9. APPLICATION IMPORT
    # --------------------------------------------------------

    application_import_ok = False
    application_detail = ""

    try:

        if str(SRC_DIR) not in sys.path:
            sys.path.insert(0, str(SRC_DIR))

        import edupredict

        application_import_ok = True
        application_detail = (
            f"edupredict imported from "
            f"{getattr(edupredict, '__file__', 'unknown')}"
        )

    except Exception as exc:

        application_detail = repr(exc)

    checks.append(
        result(
            "package_import",
            "edupredict",
            pass_fail(application_import_ok),
            application_detail,
        )
    )

    # --------------------------------------------------------
    # 10. STREAMLIT SOURCE INTEGRATION MARKERS
    # --------------------------------------------------------

    app_path = APP_DIR / "streamlit_app.py"

    app_text = ""

    if app_path.exists():
        app_text = app_path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

    markers = {
        "streamlit": "streamlit",
        "edupredict": "edupredict",
        "model": "model",
    }

    marker_results = {}

    for name, marker in markers.items():

        found = marker.lower() in app_text.lower()

        marker_results[name] = found

        checks.append(
            result(
                "application_marker",
                name,
                pass_fail(found),
                f"found={found}",
            )
        )

    # --------------------------------------------------------
    # 11. MODEL HASH INTEGRITY
    # --------------------------------------------------------

    actual_cls_hash = sha256_file(
        CLASSIFICATION_MODEL
    )

    actual_reg_hash = sha256_file(
        REGRESSION_MODEL
    )

    expected_cls_hash = None
    expected_reg_hash = None

    if PHASE7C.exists():

        phase7c = json.loads(
            PHASE7C.read_text(
                encoding="utf-8"
            )
        )

        expected_cls_hash = (
            phase7c.get("classification_sha256")
            or phase7c.get("model_hashes", {})
            .get("classification_model")
        )

        expected_reg_hash = (
            phase7c.get("regression_sha256")
            or phase7c.get("model_hashes", {})
            .get("regression_model")
        )

    cls_hash_ok = (
        expected_cls_hash is None
        or actual_cls_hash == expected_cls_hash
    )

    reg_hash_ok = (
        expected_reg_hash is None
        or actual_reg_hash == expected_reg_hash
    )

    checks.append(
        result(
            "model_hash",
            "classification",
            pass_fail(cls_hash_ok),
            (
                f"expected={expected_cls_hash}; "
                f"actual={actual_cls_hash}"
            ),
        )
    )

    checks.append(
        result(
            "model_hash",
            "regression",
            pass_fail(reg_hash_ok),
            (
                f"expected={expected_reg_hash}; "
                f"actual={actual_reg_hash}"
            ),
        )
    )

    # --------------------------------------------------------
    # 12. REPRODUCIBILITY
    # --------------------------------------------------------

    probabilities_repeat = (
        classification_model
        .predict_proba(X_test)[:, 1]
    )

    regression_repeat = (
        regression_model.predict(X_test)
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
        result(
            "reproducibility",
            "production_predictions",
            pass_fail(reproducibility_ok),
            (
                f"classification_error="
                f"{classification_error}; "
                f"regression_error="
                f"{regression_error}"
            ),
        )
    )

    # --------------------------------------------------------
    # 13. PHASE 7 GATE DEPENDENCIES
    # --------------------------------------------------------

    prior_gates = {
        "phase7A": PHASE7A,
        "phase7B": PHASE7B,
        "phase7C": PHASE7C,
        "phase7_contract": PHASE7_CONTRACT,
    }

    for name, path in prior_gates.items():

        ok = path.exists()

        status = "FAIL"

        detail = str(path)

        if ok:

            try:

                data = json.loads(
                    path.read_text(
                        encoding="utf-8"
                    )
                )

                status_value = str(data.get("status", ""))
                expected_status = {
                    "phase7A": "PASS",
                    "phase7B": "PASS",
                    "phase7C": "TECHNICAL_RELEASE_READY",
                    "phase7_contract": "TECHNICAL_QUALIFICATION_PASS",
                }.get(name)
                status = "PASS" if expected_status and status_value == expected_status else "FAIL"
                detail = f"status={status_value}; expected={expected_status}"

            except Exception as exc:

                status = "FAIL"
                detail = repr(exc)

        checks.append(
            result(
                "prior_gate",
                name,
                status,
                detail,
            )
        )

    # --------------------------------------------------------
    # 14. OVERALL STATUS
    # --------------------------------------------------------

    failures = [
        x for x in checks
        if x["status"] != "PASS"
    ]

    release_status = (
        "PASS"
        if not failures
        else "NOT_QUALIFIED"
    )

    report = {
        "project": "EduPredict",
        "phase": "8B",
        "gate": "Application / Model Integration Validation",
        "status": release_status,
        "production_threshold": PRODUCTION_THRESHOLD,
        "final_test_rows": len(test),
        "classification_probability_range": [
            float(probabilities.min()),
            float(probabilities.max()),
        ],
        "regression_prediction_range": [
            float(regression_predictions.min()),
            float(regression_predictions.max()),
        ],
        "classification_sha256": actual_cls_hash,
        "regression_sha256": actual_reg_hash,
        "checks": checks,
        "failures": failures,
    }

    report_path = (
        REPORT_DIR /
        "phase8B_application_model_validation.json"
    )

    csv_path = (
        REPORT_DIR /
        "phase8B_application_model_validation.csv"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2
        ),
        encoding="utf-8",
    )

    pd.DataFrame(checks).to_csv(
        csv_path,
        index=False,
    )

    # --------------------------------------------------------
    # CONSOLE OUTPUT
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("PHASE 8B RESULT")
    print("=" * 72)

    for check in checks:

        print(
            f"{check['check']:<32} "
            f"{check['status']:<6} "
            f"{check['item']:<28} "
            f"{check['detail']}"
        )

    print()
    print(
        f"Final-test rows:              "
        f"{len(test):,}"
    )

    print(
        f"Classification probability:   "
        f"{probabilities.min():.4f}-"
        f"{probabilities.max():.4f}"
    )

    print(
        f"Regression prediction:        "
        f"{regression_predictions.min():.2f}-"
        f"{regression_predictions.max():.2f}"
    )

    print()
    print(
        f"PHASE 8B STATUS: {release_status}"
    )

    print()
    print(f"Report:  {report_path}")
    print(f"Checks:  {csv_path}")

    print("=" * 72)

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()