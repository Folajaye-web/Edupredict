from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import joblib


# ============================================================
# EDUPREDICT — PHASE 9D
# Continuous Validation / Production Health Gate
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT
    / "data"
    / "nigeria_validation"
    / "splits"
    / "test_2025_2026.csv"
)

MODEL_DIR = ROOT / "models" / "nigeria_validation"

REPORT_DIR = (
    ROOT
    / "reports"
    / "nigeria_validation"
    / "phase9"
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)

CLASSIFICATION_MODEL = (
    MODEL_DIR / "phase5_classification_hgb.joblib"
)

REGRESSION_MODEL = (
    MODEL_DIR / "phase5_regression_hgb.joblib"
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

OUTPUT_JSON = (
    REPORT_DIR / "phase9D_continuous_validation_gate.json"
)

OUTPUT_CSV = (
    REPORT_DIR / "phase9D_continuous_validation_gate.csv"
)


# ============================================================
# Production feature contract
# ============================================================

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

FORBIDDEN = [
    "Exam",
    "Final_Year_Result",
    "Grade",
    "Intervention",
    "Promotion_Status",
    "Total",
]

EXPECTED_SESSION = ["2025/2026"]

EXPECTED_CLASSIFICATION_HASH = (
    "3cc65dbd4c0e48ba65b28f713a8d32d9fec8bf0631dc8a3b5ef15e00823f608a"
)

EXPECTED_REGRESSION_HASH = (
    "f61b02c5542aa0a4d2d0c1b18540f05193cc5d0c5f1f4e5a46ae5a75e0a9973e"
)



WARNING_THRESHOLD = 0.40


# ============================================================
# Helpers
# ============================================================


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

def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def safe_json_value(value):
    if isinstance(value, (np.bool_,)):
        return bool(value)

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    return value


def load_json(path: Path):
    if not path.exists():
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def status(value: bool) -> str:
    return "PASS" if bool(value) else "FAIL"


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 72)
    print("EDUPREDICT NIGERIA VALIDATION — PHASE 9D")
    print("=" * 72)
    print("Purpose: Continuous Validation / Production Health Gate")
    print()

    checks = []

    # --------------------------------------------------------
    # Required artifacts
    # --------------------------------------------------------

    required_artifacts = [
        DATA_PATH,
        CLASSIFICATION_MODEL,
        REGRESSION_MODEL,
        PHASE9A_REPORT,
        PHASE9B_REPORT,
        PHASE9C_REPORT,
    ]

    for artifact in required_artifacts:

        ok = artifact.exists()

        checks.append(
            {
                "gate": "required_artifact",
                "status": status(ok),
                "detail": str(artifact),
            }
        )

        if not ok:
            print(f"ERROR: Missing artifact: {artifact}")

    # --------------------------------------------------------
    # Load production dataset
    # --------------------------------------------------------

    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    df = pd.read_csv(DATA_PATH)

    print(f"Health-check rows: {len(df):,}")
    print()

    # --------------------------------------------------------
    # Session health
    # --------------------------------------------------------

    actual_sessions = sorted(
        df["Academic_Session"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    session_health = actual_sessions == EXPECTED_SESSION

    checks.append(
        {
            "gate": "session_health",
            "status": status(session_health),
            "detail": (
                f"expected={EXPECTED_SESSION}; "
                f"actual={actual_sessions}"
            ),
        }
    )

    # --------------------------------------------------------
    # Feature contract
    # --------------------------------------------------------

    missing_features = [
        x for x in FEATURES if x not in df.columns
    ]

    forbidden_used = [
        x for x in FORBIDDEN if x in df.columns
    ]

    feature_health = (
        len(missing_features) == 0
        and len(forbidden_used) == 0
    )

    checks.append(
        {
            "gate": "feature_health",
            "status": status(feature_health),
            "detail": (
                f"missing={missing_features}; "
                f"forbidden_used={forbidden_used}"
            ),
        }
    )

    # --------------------------------------------------------
    # Data quality
    #
    # First-term historical averages are structurally allowed
    # to be missing because no previous history exists.
    # --------------------------------------------------------

    historical_features = [
        "Historical_Subject_Average",
        "Historical_Overall_Average",
    ]

    historical_missing = {
        col: int(df[col].isna().sum())
        for col in historical_features
    }

    structural_missing = 0
    unexpected_missing = 0

    if "Term" in df.columns:

        first_term_mask = (
            df["Term"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("first term")
        )

        for col in historical_features:

            missing_mask = df[col].isna()

            allowed = int(
                (missing_mask & first_term_mask).sum()
            )

            unexpected = int(
                (missing_mask & ~first_term_mask).sum()
            )

            structural_missing += allowed
            unexpected_missing += unexpected

    else:

        unexpected_missing = sum(
            historical_missing.values()
        )

    # Missing values in non-historical production features
    non_historical_features = [
        x for x in FEATURES
        if x not in historical_features
    ]

    non_historical_missing = int(
        df[non_historical_features]
        .isna()
        .sum()
        .sum()
    )

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

    # Numeric production features
    numeric_features = [
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

    invalid_numeric_cells = 0

    for col in numeric_features:

        if col not in df.columns:
            continue

        converted = pd.to_numeric(
            df[col],
            errors="coerce",
        )

        invalid_numeric_cells += int(
            (
                converted.isna()
                & df[col].notna()
            ).sum()
        )

    total_missing_cells = int(
        df[FEATURES].isna().sum().sum()
    )

    data_quality = (
        duplicate_rows == 0
        and invalid_attendance == 0
        and invalid_numeric_cells == 0
        and unexpected_missing == 0
        and non_historical_missing == 0
    )

    checks.append(
        {
            "gate": "data_quality",
            "status": status(data_quality),
            "detail": (
                f"missing_cells={total_missing_cells}; "
                f"structural_missing={structural_missing}; "
                f"unexpected_missing={unexpected_missing}; "
                f"duplicate_rows={duplicate_rows}; "
                f"invalid_attendance={invalid_attendance}; "
                f"invalid_numeric_cells={invalid_numeric_cells}"
            ),
        }
    )

    # --------------------------------------------------------
    # Model identity
    # --------------------------------------------------------

    actual_classification_hash = sha256(
        CLASSIFICATION_MODEL
    )

    actual_regression_hash = sha256(
        REGRESSION_MODEL
    )

    classification_identity = (
        actual_classification_hash
        == EXPECTED_CLASSIFICATION_HASH
    )

    regression_identity = (
        actual_regression_hash
        == EXPECTED_REGRESSION_HASH
    )

    checks.append(
        {
            "gate": "classification_model_identity",
            "status": status(classification_identity),
            "detail": (
                f"expected={EXPECTED_CLASSIFICATION_HASH}; "
                f"actual={actual_classification_hash}"
            ),
        }
    )

    checks.append(
        {
            "gate": "regression_model_identity",
            "status": status(regression_identity),
            "detail": (
                f"expected={EXPECTED_REGRESSION_HASH}; "
                f"actual={actual_regression_hash}"
            ),
        }
    )

    # --------------------------------------------------------
    # Load frozen production models
    # --------------------------------------------------------

    classification_model = joblib.load(
        CLASSIFICATION_MODEL
    )

    regression_model = joblib.load(
        REGRESSION_MODEL
    )

    X = df[FEATURES].copy()

    # --------------------------------------------------------
    # Production prediction execution
    # --------------------------------------------------------

    prediction_execution = False
    reproducibility = False

    early_warning_count = None
    early_warning_rate = None

    risk_min = None
    risk_max = None
    risk_mean = None

    exam_min = None
    exam_max = None
    exam_mean = None

    classification_error = None
    regression_error = None

    try:

        probabilities = classification_model.predict_proba(X)[:, 1]

        predicted_exam = regression_model.predict(X)

        probabilities = np.asarray(
            probabilities,
            dtype=float,
        )

        predicted_exam = np.asarray(
            predicted_exam,
            dtype=float,
        )

        prediction_execution = (
            len(probabilities) == len(df)
            and len(predicted_exam) == len(df)
            and np.isfinite(probabilities).all()
            and np.isfinite(predicted_exam).all()
        )

        if prediction_execution:

            early_warning = (
                probabilities >= WARNING_THRESHOLD
            )

            early_warning_count = int(
                early_warning.sum()
            )

            early_warning_rate = float(
                early_warning.mean()
            )

            risk_min = float(probabilities.min())
            risk_max = float(probabilities.max())
            risk_mean = float(probabilities.mean())

            exam_min = float(predicted_exam.min())
            exam_max = float(predicted_exam.max())
            exam_mean = float(predicted_exam.mean())

            # Re-run frozen models to verify deterministic output
            probabilities_2 = np.asarray(
                classification_model.predict_proba(X)[:, 1],
                dtype=float,
            )

            predicted_exam_2 = np.asarray(
                regression_model.predict(X),
                dtype=float,
            )

            classification_error = float(
                np.max(
                    np.abs(
                        probabilities
                        - probabilities_2
                    )
                )
            )

            regression_error = float(
                np.max(
                    np.abs(
                        predicted_exam
                        - predicted_exam_2
                    )
                )
            )

            reproducibility = (
                classification_error == 0.0
                and regression_error == 0.0
            )

    except Exception as exc:

        prediction_execution = False

        checks.append(
            {
                "gate": "prediction_execution_error",
                "status": "FAIL",
                "detail": repr(exc),
            }
        )

    # --------------------------------------------------------
    # Prediction execution health
    # --------------------------------------------------------

    checks.append(
        {
            "gate": "prediction_execution_health",
            "status": status(prediction_execution),
            "detail": (
                f"rows={len(df):,}; "
                f"classification_predictions="
                f"{len(probabilities) if prediction_execution else 0:,}; "
                f"regression_predictions="
                f"{len(predicted_exam) if prediction_execution else 0:,}"
            ),
        }
    )

    # --------------------------------------------------------
    # Early-warning health
    # --------------------------------------------------------

    early_warning_health = (
        prediction_execution
        and early_warning_count is not None
        and 0 <= early_warning_rate <= 1
    )

    checks.append(
        {
            "gate": "early_warning_health",
            "status": status(early_warning_health),
            "detail": (
                f"threshold={WARNING_THRESHOLD}; "
                f"flags={early_warning_count}; "
                f"rate={early_warning_rate}"
            ),
        }
    )

    # --------------------------------------------------------
    # Prediction range
    # --------------------------------------------------------

    prediction_output_range = (
        prediction_execution
        and 0.0 <= risk_min <= risk_max <= 1.0
        and exam_min >= 0.0
        and exam_max <= 100.0
    )

    checks.append(
        {
            "gate": "prediction_output_range",
            "status": status(prediction_output_range),
            "detail": (
                f"risk={risk_min}-{risk_max}; "
                f"exam_score={exam_min}-{exam_max}"
            ),
        }
    )

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    checks.append(
        {
            "gate": "prediction_reproducibility",
            "status": status(reproducibility),
            "detail": (
                f"classification_error={classification_error}; "
                f"regression_error={regression_error}"
            ),
        }
    )

    # --------------------------------------------------------
    # Previous phase health
    # --------------------------------------------------------

    phase9a = load_json(PHASE9A_REPORT)
    phase9b = load_json(PHASE9B_REPORT)
    phase9c = load_json(PHASE9C_REPORT)

    phase9a_health = (
        isinstance(phase9a, dict)
        and phase9a.get("status") == "PASS"
    )

    phase9b_health = (
        isinstance(phase9b, dict)
        and phase9b.get("status") == "PASS"
    )

    phase9c_health = (
        isinstance(phase9c, dict)
        and phase9c.get("status") == "PASS"
    )

    checks.append(
        {
            "gate": "phase9A_health",
            "status": status(phase9a_health),
            "detail": (
                f"status="
                f"{phase9a.get('status') if phase9a else None}"
            ),
        }
    )

    checks.append(
        {
            "gate": "phase9B_health",
            "status": status(phase9b_health),
            "detail": (
                f"status="
                f"{phase9b.get('status') if phase9b else None}"
            ),
        }
    )

    checks.append(
        {
            "gate": "phase9C_health",
            "status": status(phase9c_health),
            "detail": (
                f"status="
                f"{phase9c.get('status') if phase9c else None}"
            ),
        }
    )

    # --------------------------------------------------------
    # Continuous contract
    # --------------------------------------------------------

    continuous_validation_contract = all(
        [
            session_health,
            feature_health,
            data_quality,
            classification_identity,
            regression_identity,
            prediction_execution,
            early_warning_health,
            prediction_output_range,
            reproducibility,
            phase9a_health,
            phase9b_health,
            phase9c_health,
        ]
    )

    checks.append(
        {
            "gate": "continuous_validation_contract",
            "status": status(
                continuous_validation_contract
            ),
            "detail": (
                "all production health controls passed"
                if continuous_validation_contract
                else
                "one or more health controls failed"
            ),
        }
    )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    final_status = (
        "PASS"
        if continuous_validation_contract
        else "HEALTH_REVIEW_REQUIRED"
    )

    failures = [
        item["detail"]
        for item in checks
        if item["status"] == "FAIL"
        and item["gate"]
        not in {
            "required_artifact",
            "prediction_execution_error",
        }
    ]

    # --------------------------------------------------------
    # Audit record
    # --------------------------------------------------------

    audit_record = {
        "project": "EduPredict",
        "phase": "9D",
        "status": final_status,
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "purpose": (
            "Continuous production health validation "
            "without automatic model replacement."
        ),

        "health_snapshot": {
            "rows": int(len(df)),
            "session": actual_sessions,
            "warning_threshold": WARNING_THRESHOLD,
            "early_warning_count": safe_json_value(
                early_warning_count
            ),
            "early_warning_rate": safe_json_value(
                early_warning_rate
            ),
            "risk_probability_min": safe_json_value(
                risk_min
            ),
            "risk_probability_max": safe_json_value(
                risk_max
            ),
            "risk_probability_mean": safe_json_value(
                risk_mean
            ),
            "predicted_exam_score_min": safe_json_value(
                exam_min
            ),
            "predicted_exam_score_max": safe_json_value(
                exam_max
            ),
            "predicted_exam_score_mean": safe_json_value(
                exam_mean
            ),
            "classification_hash": actual_classification_hash,
            "regression_hash": actual_regression_hash,
        },

        "data_quality": {
            "total_feature_missing_cells": int(
                total_missing_cells
            ),
            "structural_missing_cells": int(
                structural_missing
            ),
            "unexpected_missing_cells": int(
                unexpected_missing
            ),
            "non_historical_missing_cells": int(
                non_historical_missing
            ),
            "duplicate_rows": int(
                duplicate_rows
            ),
            "invalid_attendance": int(
                invalid_attendance
            ),
            "invalid_numeric_cells": int(
                invalid_numeric_cells
            ),
            "historical_missing": historical_missing,
        },

        "previous_phases": {
            "phase9A": (
                phase9a.get("status")
                if phase9a
                else None
            ),
            "phase9B": (
                phase9b.get("status")
                if phase9b
                else None
            ),
            "phase9C": (
                phase9c.get("status")
                if phase9c
                else None
            ),
        },

        "controls": {
            "session_health": bool(session_health),
            "feature_health": bool(feature_health),
            "data_quality": bool(data_quality),
            "classification_model_identity": bool(
                classification_identity
            ),
            "regression_model_identity": bool(
                regression_identity
            ),
            "prediction_execution": bool(
                prediction_execution
            ),
            "early_warning_health": bool(
                early_warning_health
            ),
            "prediction_output_range": bool(
                prediction_output_range
            ),
            "prediction_reproducibility": bool(
                reproducibility
            ),
            "phase9A_health": bool(
                phase9a_health
            ),
            "phase9B_health": bool(
                phase9b_health
            ),
            "phase9C_health": bool(
                phase9c_health
            ),
            "continuous_validation_contract": bool(
                continuous_validation_contract
            ),
        },

        "operational_policy": {
            "automatic_model_replacement": False,
            "automatic_retraining": False,
            "human_review_on_failure": True,
            "production_model_identity_locked": True,
        },

        "model": {
            "version": "phase5_nigeria_production",
            "classification_sha256": actual_classification_hash,
            "regression_sha256": actual_regression_hash,
            "threshold": WARNING_THRESHOLD,
        },

        "failures": failures,
        "checks": checks,
    }

    # --------------------------------------------------------
    # Write reports
    # --------------------------------------------------------

    OUTPUT_JSON.write_text(
        json.dumps(
            audit_record,
            indent=2,
            default=safe_json_value,
        ),
        encoding="utf-8",
    )

    pd.DataFrame(checks).to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("PHASE 9D CONTINUOUS VALIDATION / HEALTH GATE")
    print("=" * 72)

    print(
        f"Production rows:             {len(df):,}"
    )

    print(
        f"Session:                     {actual_sessions}"
    )

    print(
        f"Early-warning flags:         "
        f"{early_warning_count if early_warning_count is not None else 'None'}"
    )

    print(
        f"Early-warning rate:          "
        f"{f'{early_warning_rate:.4%}' if early_warning_rate is not None else 'N/A'}"
    )

    print()

    print(
        f"Data quality:                "
        f"{status(data_quality)}"
    )

    print(
        f"Model identity:              "
        f"{status(classification_identity and regression_identity)}"
    )

    print(
        f"Prediction execution:        "
        f"{status(prediction_execution)}"
    )

    print(
        f"Reproducibility:             "
        f"{status(reproducibility)}"
    )

    print(
        f"Previous phases:             "
        f"{status(phase9a_health and phase9b_health and phase9c_health)}"
    )

    print(
        f"Continuous contract:         "
        f"{status(continuous_validation_contract)}"
    )

    print()

    print(
        f"PHASE 9D STATUS: {final_status}"
    )

    print()

    print(
        f"Report: {OUTPUT_JSON}"
    )

    print(
        f"Checks: {OUTPUT_CSV}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()
