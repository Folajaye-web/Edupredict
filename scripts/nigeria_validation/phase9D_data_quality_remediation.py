from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import joblib


# ============================================================
# EDUPREDICT — PHASE 9D
# Continuous Validation / Production Health Gate
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "nigeria_validation"
    / "splits"
    / "test_2025_2026.csv"
)

MODEL_DIR = PROJECT_ROOT / "models" / "nigeria_validation"

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "nigeria_validation"
    / "phase9"
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = REPORT_DIR / "phase9D_continuous_validation_gate.json"
CHECKS_PATH = REPORT_DIR / "phase9D_continuous_validation_gate.csv"

REMEDIATION_REPORT = (
    REPORT_DIR / "phase9D_data_quality_remediation.json"
)

PHASE9A_REPORT = (
    REPORT_DIR / "phase9A_production_prediction_service.json"
)

PHASE9B_REPORT = (
    REPORT_DIR / "phase9B_operational_control_gate.json"
)

PHASE9C_REPORT = (
    REPORT_DIR / "phase9C_production_monitoring_gate.json"
)

CLASSIFICATION_MODEL = (
    MODEL_DIR / "phase5_classification_hgb.joblib"
)

REGRESSION_MODEL = (
    MODEL_DIR / "phase5_regression_hgb.joblib"
)


EXPECTED_SESSION = "2025/2026"


WARNING_THRESHOLD = 0.40

EXPECTED_CLASSIFICATION_HASH = (
    "3cc65dbd4c0e48ba65b28f713a8d32d9fec8bf0631dc8a3b5ef15e00823f608a"
)

EXPECTED_REGRESSION_HASH = (
    "f61b02c5542aa0a4d2d0c1b18540f05193cc5d0c5f1f4e5a46ae5a75e0a9973e"
)


PRODUCTION_FEATURES = [
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

FORBIDDEN_PREDICTORS = [
    "Exam",
    "Final_Year_Result",
    "Grade",
    "Intervention",
    "Promotion_Status",
    "Total",
]


NUMERIC_FEATURES = [
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

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def add_check(checks, gate, status, detail):
    checks.append(
        {
            "gate": gate,
            "status": status,
            "detail": str(detail),
        }
    )


def load_json(path: Path):
    if not path.exists():
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def safe_bool(value):
    return bool(value)


def main():

    print("=" * 72)
    print("EDUPREDICT NIGERIA VALIDATION — PHASE 9D")
    print("=" * 72)
    print("Purpose: Continuous Validation / Production Health Gate")
    print()

    checks = []
    failures = []

    # --------------------------------------------------------
    # 1. REQUIRED ARTIFACTS
    # --------------------------------------------------------

    required_artifacts = [
        DATA_PATH,
        CLASSIFICATION_MODEL,
        REGRESSION_MODEL,
        PHASE9A_REPORT,
        PHASE9B_REPORT,
        PHASE9C_REPORT,
        REMEDIATION_REPORT,
    ]

    for path in required_artifacts:

        exists = path.exists()

        add_check(
            checks,
            "required_artifact",
            "PASS" if exists else "FAIL",
            str(path),
        )

        if not exists:
            failures.append(
                f"Missing required artifact: {path}"
            )

    # Stop before model/data operations if critical files are missing.
    if failures:
        status = "HEALTH_REVIEW_REQUIRED"

        report = {
            "project": "EduPredict",
            "phase": "9D",
            "status": status,
            "timestamp_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "purpose": (
                "Continuous production health validation "
                "without automatic model replacement."
            ),
            "checks": checks,
            "failures": failures,
            "operational_policy": {
                "automatic_model_replacement": False,
                "automatic_retraining": False,
                "human_review_on_failure": True,
                "production_model_identity_locked": True,
            },
        }

        REPORT_PATH.write_text(
            json.dumps(
                report,
                indent=2,
                default=lambda o: (
                    o.item()
                    if hasattr(o, "item")
                    else str(o)
                ),
            ),
            encoding="utf-8",
        )

        pd.DataFrame(checks).to_csv(
            CHECKS_PATH,
            index=False,
        )

        print("PHASE 9D STATUS: HEALTH_REVIEW_REQUIRED")
        print(f"Report: {REPORT_PATH}")
        print(f"Checks: {CHECKS_PATH}")
        return 1

    # --------------------------------------------------------
    # 2. LOAD DATASET
    # --------------------------------------------------------

    try:
        df = pd.read_csv(DATA_PATH)
        dataset_loaded = True
    except Exception as exc:

        dataset_loaded = False

        add_check(
            checks,
            "dataset_load",
            "FAIL",
            str(exc),
        )

        failures.append(
            f"Dataset load failed: {exc}"
        )

        df = pd.DataFrame()

    if dataset_loaded:

        print(
            f"Health-check rows: {len(df):,}"
        )
        print()

        # ----------------------------------------------------
        # 3. SESSION HEALTH
        # ----------------------------------------------------

        actual_sessions = sorted(
            df["Academic_Session"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        ) if "Academic_Session" in df.columns else []

        session_ok = actual_sessions == [EXPECTED_SESSION]

        add_check(
            checks,
            "session_health",
            "PASS" if session_ok else "FAIL",
            (
                f"expected=['{EXPECTED_SESSION}']; "
                f"actual={actual_sessions}"
            ),
        )

        if not session_ok:
            failures.append(
                "Production session health failed."
            )

        # ----------------------------------------------------
        # 4. FEATURE CONTRACT
        # ----------------------------------------------------

        missing_features = [
            col
            for col in PRODUCTION_FEATURES
            if col not in df.columns
        ]

        forbidden_used = [
            col
            for col in FORBIDDEN_PREDICTORS
            if col in df.columns
        ]

        feature_ok = (
            len(missing_features) == 0
            and len(forbidden_used) == 0
        )

        add_check(
            checks,
            "feature_health",
            "PASS" if feature_ok else "FAIL",
            (
                f"missing={missing_features}; "
                f"forbidden_used={forbidden_used}"
            ),
        )

        if not feature_ok:
            failures.append(
                "Production feature contract failed."
            )

        # ----------------------------------------------------
        # 5. DATA QUALITY
        # ----------------------------------------------------

        # IMPORTANT:
        # df is deliberately loaded BEFORE duplicated().
        # This fixes the previous NameError.
        duplicate_rows = int(
            df.duplicated().sum()
        )

        invalid_attendance = 0

        if "Attendance" in df.columns:

            attendance_numeric = pd.to_numeric(
                df["Attendance"],
                errors="coerce",
            )

            invalid_attendance = int(
                (
                    attendance_numeric.notna()
                    & (
                        (attendance_numeric < 0)
                        | (attendance_numeric > 100)
                    )
                ).sum()
            )

        # ----------------------------------------------------
        # Numeric conversion / missingness
        # ----------------------------------------------------

        numeric_missing_by_feature = {}

        for col in NUMERIC_FEATURES:

            if col not in df.columns:
                numeric_missing_by_feature[col] = len(df)
                continue

            numeric_values = pd.to_numeric(
                df[col],
                errors="coerce",
            )

            numeric_missing_by_feature[col] = int(
                numeric_values.isna().sum()
            )

        # Structural first-term historical missingness
        # is expected and allowed.

        subject_missing = (
            numeric_missing_by_feature.get(
                "Historical_Subject_Average",
                0,
            )
        )

        overall_missing = (
            numeric_missing_by_feature.get(
                "Historical_Overall_Average",
                0,
            )
        )

        structural_missing = (
            subject_missing + overall_missing
        )

        unexpected_missing = 0

        # Historical averages are permitted to be missing
        # only for First Term.

        if "Term" in df.columns:

            first_term_mask = (
                df["Term"]
                .astype(str)
                .eq("First Term")
            )

            for feature in [
                "Historical_Subject_Average",
                "Historical_Overall_Average",
            ]:

                if feature not in df.columns:
                    continue

                numeric_values = pd.to_numeric(
                    df[feature],
                    errors="coerce",
                )

                missing_mask = numeric_values.isna()

                unexpected_missing += int(
                    (
                        missing_mask
                        & ~first_term_mask
                    ).sum()
                )

        # Any missingness outside the two structural
        # historical fields is an automatic failure.

        for feature in NUMERIC_FEATURES:

            if feature in [
                "Historical_Subject_Average",
                "Historical_Overall_Average",
            ]:
                continue

            unexpected_missing += (
                numeric_missing_by_feature.get(
                    feature,
                    0,
                )
            )

        invalid_numeric_cells = unexpected_missing

        # Historical count consistency:
        # if historical average is missing, its corresponding
        # historical count must be zero.

        historical_count_consistency = True

        subject_count_mismatch = 0
        overall_count_mismatch = 0

        if (
            "Historical_Subject_Average" in df.columns
            and "Historical_Subject_Count" in df.columns
        ):

            subject_avg = pd.to_numeric(
                df["Historical_Subject_Average"],
                errors="coerce",
            )

            subject_count = pd.to_numeric(
                df["Historical_Subject_Count"],
                errors="coerce",
            )

            subject_count_mismatch = int(
                (
                    subject_avg.isna()
                    & subject_count.ne(0)
                ).sum()
            )

        if (
            "Historical_Overall_Average" in df.columns
            and "Historical_Overall_Count" in df.columns
        ):

            overall_avg = pd.to_numeric(
                df["Historical_Overall_Average"],
                errors="coerce",
            )

            overall_count = pd.to_numeric(
                df["Historical_Overall_Count"],
                errors="coerce",
            )

            overall_count_mismatch = int(
                (
                    overall_avg.isna()
                    & overall_count.ne(0)
                ).sum()
            )

        if (
            subject_count_mismatch > 0
            or overall_count_mismatch > 0
        ):
            historical_count_consistency = False

        # Production data quality is healthy when:
        #
        # 1. no duplicates
        # 2. no invalid attendance
        # 3. no unexpected missing values
        # 4. historical count relationship is valid
        #
        # The 4,409 structural first-term historical cells
        # are NOT treated as a production data failure.

        data_quality_ok = (
            duplicate_rows == 0
            and invalid_attendance == 0
            and unexpected_missing == 0
            and historical_count_consistency
        )

        add_check(
            checks,
            "data_quality",
            "PASS" if data_quality_ok else "FAIL",
            (
                f"structural_historical_missing="
                f"{structural_missing}; "
                f"unexpected_missing="
                f"{unexpected_missing}; "
                f"invalid_attendance="
                f"{invalid_attendance}; "
                f"duplicate_rows="
                f"{duplicate_rows}; "
                f"invalid_numeric_cells="
                f"{invalid_numeric_cells}"
            ),
        )

        if not data_quality_ok:
            failures.append(
                "Production data quality health failed."
            )

        # ----------------------------------------------------
        # 6. MODEL IDENTITY
        # ----------------------------------------------------

        actual_classification_hash = sha256_file(
            CLASSIFICATION_MODEL
        )

        actual_regression_hash = sha256_file(
            REGRESSION_MODEL
        )

        classification_identity_ok = (
            actual_classification_hash
            == EXPECTED_CLASSIFICATION_HASH
        )

        regression_identity_ok = (
            actual_regression_hash
            == EXPECTED_REGRESSION_HASH
        )

        add_check(
            checks,
            "classification_model_identity",
            "PASS"
            if classification_identity_ok
            else "FAIL",
            (
                f"expected={EXPECTED_CLASSIFICATION_HASH}; "
                f"actual={actual_classification_hash}"
            ),
        )

        add_check(
            checks,
            "regression_model_identity",
            "PASS"
            if regression_identity_ok
            else "FAIL",
            (
                f"expected={EXPECTED_REGRESSION_HASH}; "
                f"actual={actual_regression_hash}"
            ),
        )

        if not classification_identity_ok:
            failures.append(
                "Classification model identity changed."
            )

        if not regression_identity_ok:
            failures.append(
                "Regression model identity changed."
            )

        # ----------------------------------------------------
        # 7. MODEL EXECUTION
        # ----------------------------------------------------

        classification_predictions = None
        regression_predictions = None

        prediction_execution_ok = True

        try:

            classification_model = joblib.load(
                CLASSIFICATION_MODEL
            )

            regression_model = joblib.load(
                REGRESSION_MODEL
            )

            production_X = df[
                PRODUCTION_FEATURES
            ].copy()

            classification_predictions = (
                classification_model.predict_proba(
                    production_X
                )[:, 1]
            )

            regression_predictions = (
                regression_model.predict(
                    production_X
                )
            )

        except Exception as exc:

            prediction_execution_ok = False

            add_check(
                checks,
                "prediction_execution_health",
                "FAIL",
                str(exc),
            )

            failures.append(
                f"Prediction execution failed: {exc}"
            )

        if prediction_execution_ok:

            add_check(
                checks,
                "prediction_execution_health",
                "PASS",
                (
                    f"rows={len(df):,}; "
                    f"classification_predictions="
                    f"{len(classification_predictions):,}; "
                    f"regression_predictions="
                    f"{len(regression_predictions):,}"
                ),
            )

        # ----------------------------------------------------
        # 8. PREDICTION HEALTH
        # ----------------------------------------------------

        early_warning_count = 0
        early_warning_rate = 0.0

        risk_min = None
        risk_max = None
        risk_mean = None

        exam_min = None
        exam_max = None
        exam_mean = None

        prediction_output_ok = False
        reproducibility_ok = False

        if prediction_execution_ok:

            risk_probability = np.asarray(
                classification_predictions,
                dtype=float,
            )

            predicted_exam_score = np.asarray(
                regression_predictions,
                dtype=float,
            )

            early_warning = (
                risk_probability >= WARNING_THRESHOLD
            )

            early_warning_count = int(
                early_warning.sum()
            )

            early_warning_rate = float(
                early_warning.mean()
            )

            risk_min = float(
                risk_probability.min()
            )

            risk_max = float(
                risk_probability.max()
            )

            risk_mean = float(
                risk_probability.mean()
            )

            exam_min = float(
                predicted_exam_score.min()
            )

            exam_max = float(
                predicted_exam_score.max()
            )

            exam_mean = float(
                predicted_exam_score.mean()
            )

            prediction_output_ok = (
                np.isfinite(risk_probability).all()
                and np.isfinite(predicted_exam_score).all()
                and (
                    (risk_probability >= 0).all()
                    and (risk_probability <= 1).all()
                )
                and (
                    predicted_exam_score >= 0
                ).all()
                and (
                    predicted_exam_score <= 100
                ).all()
            )

            add_check(
                checks,
                "early_warning_health",
                "PASS"
                if (
                    0
                    <= early_warning_rate
                    <= 1
                )
                else "FAIL",
                (
                    f"threshold={WARNING_THRESHOLD}; "
                    f"flags={early_warning_count:,}; "
                    f"rate={early_warning_rate:.6f}"
                ),
            )

            add_check(
                checks,
                "prediction_output_range",
                "PASS"
                if prediction_output_ok
                else "FAIL",
                (
                    f"risk="
                    f"{risk_min:.6f}-"
                    f"{risk_max:.6f}; "
                    f"exam_score="
                    f"{exam_min:.2f}-"
                    f"{exam_max:.2f}"
                ),
            )

            if not prediction_output_ok:
                failures.append(
                    "Prediction output range health failed."
                )

            # Repeat prediction for reproducibility.
            try:

                classification_predictions_2 = (
                    classification_model
                    .predict_proba(production_X)[:, 1]
                )

                regression_predictions_2 = (
                    regression_model.predict(
                        production_X
                    )
                )

                classification_error = float(
                    np.max(
                        np.abs(
                            classification_predictions
                            - classification_predictions_2
                        )
                    )
                )

                regression_error = float(
                    np.max(
                        np.abs(
                            regression_predictions
                            - regression_predictions_2
                        )
                    )
                )

                reproducibility_ok = (
                    classification_error == 0.0
                    and regression_error == 0.0
                )

                add_check(
                    checks,
                    "prediction_reproducibility",
                    "PASS"
                    if reproducibility_ok
                    else "FAIL",
                    (
                        f"classification_error="
                        f"{classification_error}; "
                        f"regression_error="
                        f"{regression_error}"
                    ),
                )

                if not reproducibility_ok:
                    failures.append(
                        "Prediction reproducibility failed."
                    )

            except Exception as exc:

                reproducibility_ok = False

                add_check(
                    checks,
                    "prediction_reproducibility",
                    "FAIL",
                    str(exc),
                )

                failures.append(
                    "Prediction reproducibility failed."
                )

        # ----------------------------------------------------
        # 9. PREVIOUS PHASE HEALTH
        # ----------------------------------------------------

        phase9a = load_json(PHASE9A_REPORT)
        phase9b = load_json(PHASE9B_REPORT)
        phase9c = load_json(PHASE9C_REPORT)

        phase9a_status = (
            phase9a.get("status")
            if phase9a
            else None
        )

        phase9b_status = (
            phase9b.get("status")
            if phase9b
            else None
        )

        phase9c_status = (
            phase9c.get("status")
            if phase9c
            else None
        )

        phase9a_ok = phase9a_status == "PASS"
        phase9b_ok = phase9b_status == "PASS"
        phase9c_ok = phase9c_status == "PASS"

        add_check(
            checks,
            "phase9A_health",
            "PASS" if phase9a_ok else "FAIL",
            f"status={phase9a_status}",
        )

        add_check(
            checks,
            "phase9B_health",
            "PASS" if phase9b_ok else "FAIL",
            f"status={phase9b_status}",
        )

        add_check(
            checks,
            "phase9C_health",
            "PASS" if phase9c_ok else "FAIL",
            f"status={phase9c_status}",
        )

        if not phase9a_ok:
            failures.append(
                "Phase 9A health failed."
            )

        if not phase9b_ok:
            failures.append(
                "Phase 9B health failed."
            )

        if not phase9c_ok:
            failures.append(
                "Phase 9C health failed."
            )

        # ----------------------------------------------------
        # 10. CONTINUOUS VALIDATION CONTRACT
        # ----------------------------------------------------

        continuous_contract_ok = (
            session_ok
            and feature_ok
            and data_quality_ok
            and classification_identity_ok
            and regression_identity_ok
            and prediction_execution_ok
            and prediction_output_ok
            and reproducibility_ok
            and phase9a_ok
            and phase9b_ok
            and phase9c_ok
        )

        add_check(
            checks,
            "continuous_validation_contract",
            "PASS"
            if continuous_contract_ok
            else "FAIL",
            (
                "all production health controls passed"
                if continuous_contract_ok
                else "one or more health controls failed"
            ),
        )

        if not continuous_contract_ok:
            failures.append(
                "Continuous validation contract failed."
            )

    # --------------------------------------------------------
    # 11. FINAL STATUS
    # --------------------------------------------------------

    status = (
        "PASS"
        if len(failures) == 0
        else "HEALTH_REVIEW_REQUIRED"
    )

    report = {
        "project": "EduPredict",
        "phase": "9D",
        "status": status,
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "purpose": (
            "Continuous production health validation "
            "without automatic model replacement."
        ),
        "health_snapshot": {
            "rows": int(len(df))
            if dataset_loaded
            else 0,
            "session": (
                sorted(
                    df["Academic_Session"]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )
                if dataset_loaded
                and "Academic_Session" in df.columns
                else []
            ),
            "warning_threshold": WARNING_THRESHOLD,
            "early_warning_count": early_warning_count,
            "early_warning_rate": early_warning_rate,
            "risk_probability_min": risk_min,
            "risk_probability_max": risk_max,
            "risk_probability_mean": risk_mean,
            "predicted_exam_score_min": exam_min,
            "predicted_exam_score_max": exam_max,
            "predicted_exam_score_mean": exam_mean,
            "classification_hash": (
                actual_classification_hash
                if dataset_loaded
                else None
            ),
            "regression_hash": (
                actual_regression_hash
                if dataset_loaded
                else None
            ),
        },
        "data_quality": {
            "duplicate_rows": (
                duplicate_rows
                if dataset_loaded
                else None
            ),
            "invalid_attendance": (
                invalid_attendance
                if dataset_loaded
                else None
            ),
            "structural_historical_missing_cells": (
                structural_missing
                if dataset_loaded
                else None
            ),
            "unexpected_missing_cells": (
                unexpected_missing
                if dataset_loaded
                else None
            ),
            "subject_historical_missing": (
                subject_missing
                if dataset_loaded
                else None
            ),
            "overall_historical_missing": (
                overall_missing
                if dataset_loaded
                else None
            ),
        },
        "previous_phases": {
            "phase9A": phase9a_status
            if dataset_loaded
            else None,
            "phase9B": phase9b_status
            if dataset_loaded
            else None,
            "phase9C": phase9c_status
            if dataset_loaded
            else None,
        },
        "controls": {
            "session_health": (
                session_ok
                if dataset_loaded
                else False
            ),
            "feature_health": (
                feature_ok
                if dataset_loaded
                else False
            ),
            "data_quality": (
                data_quality_ok
                if dataset_loaded
                else False
            ),
            "classification_model_identity": (
                classification_identity_ok
                if dataset_loaded
                else False
            ),
            "regression_model_identity": (
                regression_identity_ok
                if dataset_loaded
                else False
            ),
            "prediction_execution": (
                prediction_execution_ok
                if dataset_loaded
                else False
            ),
            "early_warning_health": (
                (
                    0
                    <= early_warning_rate
                    <= 1
                )
                if dataset_loaded
                else False
            ),
            "prediction_output_range": (
                prediction_output_ok
                if dataset_loaded
                else False
            ),
            "prediction_reproducibility": (
                reproducibility_ok
                if dataset_loaded
                else False
            ),
            "phase9A_health": (
                phase9a_ok
                if dataset_loaded
                else False
            ),
            "phase9B_health": (
                phase9b_ok
                if dataset_loaded
                else False
            ),
            "phase9C_health": (
                phase9c_ok
                if dataset_loaded
                else False
            ),
            "continuous_validation_contract": (
                continuous_contract_ok
                if dataset_loaded
                else False
            ),
        },
        "checks": checks,
        "failures": failures,
        "operational_policy": {
            "automatic_model_replacement": False,
            "automatic_retraining": False,
            "human_review_on_failure": True,
            "production_model_identity_locked": True,
        },
        "model": {
            "version": "phase5_nigeria_production",
            "classification_sha256": EXPECTED_CLASSIFICATION_HASH,
            "regression_sha256": EXPECTED_REGRESSION_HASH,
            "threshold": WARNING_THRESHOLD,
        },
    }

    # --------------------------------------------------------
    # 12. WRITE REPORTS
    # --------------------------------------------------------

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
            default=lambda o: (
                o.item()
                if hasattr(o, "item")
                else str(o)
            ),
        ),
        encoding="utf-8",
    )

    pd.DataFrame(checks).to_csv(
        CHECKS_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # 13. CONSOLE OUTPUT
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("PHASE 9D CONTINUOUS VALIDATION / HEALTH GATE")
    print("=" * 72)

    print(
        f"Production rows:             "
        f"{len(df):,}"
        if dataset_loaded
        else "Production rows:             0"
    )

    if dataset_loaded:

        sessions = sorted(
            df["Academic_Session"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        print(
            f"Session:                     {sessions}"
        )

        print(
            f"Early-warning flags:         "
            f"{early_warning_count:,}"
        )

        print(
            f"Early-warning rate:          "
            f"{early_warning_rate:.4%}"
        )

        if risk_min is not None:
            print(
                f"Risk probability:            "
                f"{risk_min:.4f} - {risk_max:.4f}"
            )

        if exam_min is not None:
            print(
                f"Predicted exam score:        "
                f"{exam_min:.2f} - {exam_max:.2f}"
            )

        print()

        print(
            f"Data quality:                "
            f"{'PASS' if data_quality_ok else 'FAIL'}"
        )

        print(
            f"Model identity:              "
            f"{'PASS' if classification_identity_ok and regression_identity_ok else 'FAIL'}"
        )

        print(
            f"Prediction execution:        "
            f"{'PASS' if prediction_execution_ok else 'FAIL'}"
        )

        print(
            f"Reproducibility:             "
            f"{'PASS' if reproducibility_ok else 'FAIL'}"
        )

        print(
            f"Previous phases:             "
            f"{'PASS' if phase9a_ok and phase9b_ok and phase9c_ok else 'FAIL'}"
        )

        print(
            f"Continuous contract:         "
            f"{'PASS' if continuous_contract_ok else 'FAIL'}"
        )

    print()
    print(
        f"PHASE 9D STATUS: {status}"
    )

    print()
    print(f"Report: {REPORT_PATH}")
    print(f"Checks: {CHECKS_PATH}")
    print("=" * 72)

    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())