from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# EDUPREDICT — PHASE 9D
# Continuous Validation / Production Health Gate
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "nigeria_validation"
    / "splits"
    / "test_2025_2026.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "nigeria_validation"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "nigeria_validation"
    / "phase9"
)

REPORT_JSON = REPORT_DIR / "phase9D_continuous_validation_gate.json"
REPORT_CSV = REPORT_DIR / "phase9D_continuous_validation_gate.csv"

REMEDIATION_JSON = (
    REPORT_DIR / "phase9D_data_quality_remediation.json"
)

PHASE9A_JSON = (
    REPORT_DIR / "phase9A_production_prediction_service.json"
)

PHASE9B_JSON = (
    REPORT_DIR / "phase9B_operational_control_gate.json"
)

PHASE9C_JSON = (
    REPORT_DIR / "phase9C_production_monitoring_gate.json"
)

CLASSIFICATION_MODEL = (
    MODEL_DIR / "phase5_classification_hgb.joblib"
)

REGRESSION_MODEL = (
    MODEL_DIR / "phase5_regression_hgb.joblib"
)

EXPECTED_SESSION = "2025/2026"
THRESHOLD = 0.40

EXPECTED_CLASSIFICATION_SHA256 = (
    "3cc65dbd4c0e48ba65b28f713a8d32d9fec8bf0631dc8a3b5ef15e00823f608a"
)

EXPECTED_REGRESSION_SHA256 = (
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


def json_safe(value):
    """
    Convert NumPy/Pandas/path scalar values into standard JSON values.
    """
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]

    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        value = float(value)
        if np.isnan(value) or np.isinf(value):
            return None
        return value

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, Path):
        return str(value)

    return value


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def add_check(checks, gate, ok, detail):
    checks.append(
        {
            "gate": gate,
            "status": "PASS" if bool(ok) else "FAIL",
            "detail": str(detail),
        }
    )


def load_json(path: Path):
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def status_from_report(path: Path):
    data = load_json(path)

    if not isinstance(data, dict):
        return None

    return data.get("status")


def normalize_prediction_output(result, row_count: int) -> pd.DataFrame:
    """
    Normalize the production prediction service output.

    Expected contract:

        risk_probability
        early_warning
        predicted_exam_score

    The production predictor may return a DataFrame, dict, tuple/list,
    or a NumPy structure. This function converts supported forms into
    one canonical DataFrame.
    """

    required = [
        "risk_probability",
        "early_warning",
        "predicted_exam_score",
    ]

    # ------------------------------------------------------------------
    # DataFrame
    # ------------------------------------------------------------------
    if isinstance(result, pd.DataFrame):
        out = result.copy()

        if all(col in out.columns for col in required):
            return out[required].reset_index(drop=True)

        # Common alternative names.
        aliases = {
            "risk_probability": [
                "risk_probability",
                "risk",
                "probability",
                "risk_prob",
            ],
            "early_warning": [
                "early_warning",
                "warning",
                "flag",
            ],
            "predicted_exam_score": [
                "predicted_exam_score",
                "predicted_score",
                "exam_score",
                "prediction",
            ],
        }

        resolved = {}

        for target, candidates in aliases.items():
            for candidate in candidates:
                if candidate in out.columns:
                    resolved[target] = candidate
                    break

        if len(resolved) == 3:
            return pd.DataFrame(
                {
                    "risk_probability": out[resolved["risk_probability"]],
                    "early_warning": out[resolved["early_warning"]],
                    "predicted_exam_score": out[
                        resolved["predicted_exam_score"]
                    ],
                }
            ).reset_index(drop=True)

        raise ValueError(
            "Prediction DataFrame does not contain the required "
            f"columns. Received columns={list(out.columns)}"
        )

    # ------------------------------------------------------------------
    # Dictionary
    # ------------------------------------------------------------------
    if isinstance(result, dict):
        # Direct production contract.
        if all(key in result for key in required):
            return pd.DataFrame(
                {
                    key: np.asarray(result[key]).reshape(-1)
                    for key in required
                }
            )

        # Nested prediction/result contract.
        for nested_key in ["predictions", "prediction", "results", "data"]:
            nested = result.get(nested_key)

            if isinstance(nested, dict):
                return normalize_prediction_output(
                    nested,
                    row_count,
                )

            if isinstance(nested, pd.DataFrame):
                return normalize_prediction_output(
                    nested,
                    row_count,
                )

    # ------------------------------------------------------------------
    # Tuple/list
    # ------------------------------------------------------------------
    if isinstance(result, (tuple, list)) and len(result) >= 2:

        # Case:
        # (risk_probability, predicted_exam_score)
        if len(result) == 2:
            risk = np.asarray(result[0]).reshape(-1)
            score = np.asarray(result[1]).reshape(-1)

            if len(risk) == row_count and len(score) == row_count:
                warning = risk >= THRESHOLD

                return pd.DataFrame(
                    {
                        "risk_probability": risk,
                        "early_warning": warning,
                        "predicted_exam_score": score,
                    }
                )

        # Case:
        # (risk_probability, early_warning, predicted_exam_score)
        if len(result) >= 3:
            risk = np.asarray(result[0]).reshape(-1)
            warning = np.asarray(result[1]).reshape(-1)
            score = np.asarray(result[2]).reshape(-1)

            if (
                len(risk) == row_count
                and len(warning) == row_count
                and len(score) == row_count
            ):
                return pd.DataFrame(
                    {
                        "risk_probability": risk,
                        "early_warning": warning,
                        "predicted_exam_score": score,
                    }
                )

    raise ValueError(
        "Unsupported production prediction output type: "
        f"{type(result).__name__}"
    )


def run_production_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Call the real production prediction service.

    The project package is imported only after PROJECT_ROOT/src has
    been inserted into sys.path.
    """

    src_dir = PROJECT_ROOT / "src"

    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from edupredict.production_predictor import predict

    result = predict(df)

    return normalize_prediction_output(
        result,
        len(df),
    )


def main():

    print("=" * 72)
    print("EDUPREDICT NIGERIA VALIDATION — PHASE 9D")
    print("=" * 72)
    print("Purpose: Continuous Validation / Production Health Gate")
    print()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    checks = []
    failures = []

    # ==================================================================
    # 1. REQUIRED ARTIFACTS
    # ==================================================================

    required_artifacts = [
        DATA_PATH,
        CLASSIFICATION_MODEL,
        REGRESSION_MODEL,
        PHASE9A_JSON,
        PHASE9B_JSON,
        PHASE9C_JSON,
        REMEDIATION_JSON,
    ]

    for artifact in required_artifacts:
        ok = artifact.exists()

        add_check(
            checks,
            "required_artifact",
            ok,
            str(artifact),
        )

        if not ok:
            failures.append(
                f"Required artifact missing: {artifact}"
            )

    # ==================================================================
    # 2. DATASET LOAD
    # ==================================================================

    if not DATA_PATH.exists():
        print("Cannot continue: production dataset is missing.")
        return

    df = pd.read_csv(DATA_PATH)

    rows = len(df)

    print(f"Health-check rows: {rows:,}")
    print()

    # ==================================================================
    # 3. SESSION HEALTH
    # ==================================================================

    session_column = "Academic_Session"

    if session_column in df.columns:
        sessions = sorted(
            df[session_column]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
    else:
        sessions = []

    session_ok = sessions == [EXPECTED_SESSION]

    add_check(
        checks,
        "session_health",
        session_ok,
        (
            f"expected=['{EXPECTED_SESSION}']; "
            f"actual={sessions}"
        ),
    )

    if not session_ok:
        failures.append(
            "Production session contract failed."
        )

    # ==================================================================
    # 4. FEATURE CONTRACT
    # ==================================================================

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
        feature_ok,
        (
            f"missing={missing_features}; "
            f"forbidden_used={forbidden_used}"
        ),
    )

    if not feature_ok:
        failures.append(
            "Production feature contract failed."
        )

    # ==================================================================
    # 5. DATA QUALITY
    #
    # Structural first-term historical missingness is allowed.
    # Unexpected missingness is not allowed.
    # ==================================================================

    remediation = load_json(REMEDIATION_JSON)

    remediation_ok = False
    structural_missing_cells = 0
    unexpected_missing_cells = 0
    non_historical_missing_cells = 0
    duplicate_rows = 0
    invalid_attendance = 0

    if isinstance(remediation, dict):

        remediation_ok = (
            remediation.get("status") == "PASS"
            and remediation.get("session") == [EXPECTED_SESSION]
            and remediation.get("unexpected_missing_cells", -1) == 0
            and remediation.get("non_historical_missing_cells", -1) == 0
        )

        structural_missing_cells = int(
            remediation.get(
                "structural_missing_cells",
                0,
            )
        )

        unexpected_missing_cells = int(
            remediation.get(
                "unexpected_missing_cells",
                0,
            )
        )

        non_historical_missing_cells = int(
            remediation.get(
                "non_historical_missing_cells",
                0,
            )
        )

        duplicate_rows = int(
            remediation.get(
                "duplicate_rows",
                0,
            )
        )

    # Independent duplicate check.
    actual_duplicate_rows = int(
        df.duplicated().sum()
    )

    duplicate_rows = max(
        duplicate_rows,
        actual_duplicate_rows,
    )

    # Independent attendance check.
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

    data_quality_ok = (
        remediation_ok
        and session_ok
        and unexpected_missing_cells == 0
        and non_historical_missing_cells == 0
        and duplicate_rows == 0
        and invalid_attendance == 0
    )

    add_check(
        checks,
        "data_quality",
        data_quality_ok,
        (
            f"missing_cells={structural_missing_cells}; "
            f"structural_missing_cells={structural_missing_cells}; "
            f"unexpected_missing_cells={unexpected_missing_cells}; "
            f"non_historical_missing_cells={non_historical_missing_cells}; "
            f"duplicate_rows={duplicate_rows}; "
            f"invalid_attendance={invalid_attendance}; "
            f"remediation_contract="
            f"{'PASS' if remediation_ok else 'FAIL'}"
        ),
    )

    if not data_quality_ok:
        failures.append(
            "Production data quality health failed."
        )

    # ==================================================================
    # 6. MODEL IDENTITY
    # ==================================================================

    cls_hash = None
    reg_hash = None

    cls_hash_ok = False
    reg_hash_ok = False

    if CLASSIFICATION_MODEL.exists():
        cls_hash = sha256_file(
            CLASSIFICATION_MODEL
        )

        cls_hash_ok = (
            cls_hash
            == EXPECTED_CLASSIFICATION_SHA256
        )

    if REGRESSION_MODEL.exists():
        reg_hash = sha256_file(
            REGRESSION_MODEL
        )

        reg_hash_ok = (
            reg_hash
            == EXPECTED_REGRESSION_SHA256
        )

    add_check(
        checks,
        "classification_model_identity",
        cls_hash_ok,
        (
            f"expected={EXPECTED_CLASSIFICATION_SHA256}; "
            f"actual={cls_hash}"
        ),
    )

    add_check(
        checks,
        "regression_model_identity",
        reg_hash_ok,
        (
            f"expected={EXPECTED_REGRESSION_SHA256}; "
            f"actual={reg_hash}"
        ),
    )

    if not cls_hash_ok:
        failures.append(
            "Classification model identity failed."
        )

    if not reg_hash_ok:
        failures.append(
            "Regression model identity failed."
        )

    # ==================================================================
    # 7. PRODUCTION PREDICTION EXECUTION
    # ==================================================================

    prediction_1 = None
    prediction_2 = None

    prediction_execution_ok = False

    prediction_error = None

    try:

        prediction_1 = run_production_prediction(df)

        required_prediction_columns = [
            "risk_probability",
            "early_warning",
            "predicted_exam_score",
        ]

        prediction_columns_ok = all(
            col in prediction_1.columns
            for col in required_prediction_columns
        )

        prediction_rows_ok = (
            len(prediction_1) == rows
        )

        prediction_execution_ok = (
            prediction_columns_ok
            and prediction_rows_ok
        )

        detail = (
            f"rows={len(prediction_1):,}; "
            f"required_columns="
            f"{prediction_columns_ok}"
        )

    except Exception as exc:

        prediction_error = (
            f"{type(exc).__name__}: {exc}"
        )

        detail = prediction_error

    add_check(
        checks,
        "prediction_execution_health",
        prediction_execution_ok,
        detail,
    )

    if not prediction_execution_ok:
        failures.append(
            "Production prediction execution failed."
        )

    # ==================================================================
    # 8. PREDICTION VALUE HEALTH
    # ==================================================================

    warning_count = None
    warning_rate = None

    risk_min = None
    risk_max = None
    risk_mean = None

    score_min = None
    score_max = None
    score_mean = None

    warning_health_ok = False
    output_range_ok = False

    if prediction_execution_ok:

        probabilities = pd.to_numeric(
            prediction_1["risk_probability"],
            errors="coerce",
        )

        scores = pd.to_numeric(
            prediction_1["predicted_exam_score"],
            errors="coerce",
        )

        warnings = prediction_1[
            "early_warning"
        ]

        warning_numeric = (
            warnings.astype(str)
            .str.strip()
            .str.lower()
            .map(
                {
                    "true": 1,
                    "false": 0,
                    "1": 1,
                    "0": 0,
                    "yes": 1,
                    "no": 0,
                }
            )
        )

        # If the production service returns numeric booleans,
        # fall back to numeric conversion.
        if warning_numeric.isna().any():
            warning_numeric = pd.to_numeric(
                warnings,
                errors="coerce",
            )

        probabilities_valid = (
            probabilities.notna().all()
            and np.isfinite(probabilities).all()
        )

        scores_valid = (
            scores.notna().all()
            and np.isfinite(scores).all()
        )

        warnings_valid = (
            warning_numeric.notna().all()
            and warning_numeric.isin([0, 1]).all()
        )

        warning_health_ok = (
            probabilities_valid
            and warnings_valid
            and len(probabilities) == rows
        )

        if warning_health_ok:

            expected_warning = (
                probabilities >= THRESHOLD
            ).astype(int)

            actual_warning = (
                warning_numeric.astype(int)
            )

            warning_health_ok = bool(
                np.array_equal(
                    expected_warning.to_numpy(),
                    actual_warning.to_numpy(),
                )
            )

        output_range_ok = (
            probabilities_valid
            and scores_valid
            and bool(
                (probabilities >= 0).all()
                and (probabilities <= 1).all()
                and (scores >= 0).all()
                and (scores <= 100).all()
            )
        )

        warning_count = int(
            warning_numeric.fillna(0).sum()
        )

        warning_rate = (
            warning_count / rows
            if rows
            else 0.0
        )

        risk_min = float(
            probabilities.min()
        )

        risk_max = float(
            probabilities.max()
        )

        risk_mean = float(
            probabilities.mean()
        )

        score_min = float(
            scores.min()
        )

        score_max = float(
            scores.max()
        )

        score_mean = float(
            scores.mean()
        )

    add_check(
        checks,
        "early_warning_health",
        warning_health_ok,
        (
            f"threshold={THRESHOLD}; "
            f"flags={warning_count}; "
            f"rate="
            f"{warning_rate:.6f}"
            if warning_rate is not None
            else "prediction output unavailable"
        ),
    )

    add_check(
        checks,
        "prediction_output_range",
        output_range_ok,
        (
            f"risk={risk_min:.6f}-{risk_max:.6f}; "
            f"exam_score={score_min:.2f}-{score_max:.2f}"
            if risk_min is not None
            else "prediction output unavailable"
        ),
    )

    if not warning_health_ok:
        failures.append(
            "Early-warning prediction health failed."
        )

    if not output_range_ok:
        failures.append(
            "Prediction output range health failed."
        )

    # ==================================================================
    # 9. REPRODUCIBILITY
    # ==================================================================

    reproducibility_ok = False
    classification_error = None
    regression_error = None

    if prediction_execution_ok:

        try:

            prediction_2 = run_production_prediction(df)

            p1_risk = pd.to_numeric(
                prediction_1["risk_probability"],
                errors="coerce",
            ).to_numpy()

            p2_risk = pd.to_numeric(
                prediction_2["risk_probability"],
                errors="coerce",
            ).to_numpy()

            p1_score = pd.to_numeric(
                prediction_1["predicted_exam_score"],
                errors="coerce",
            ).to_numpy()

            p2_score = pd.to_numeric(
                prediction_2["predicted_exam_score"],
                errors="coerce",
            ).to_numpy()

            p1_warning = (
                prediction_1[
                    "early_warning"
                ]
                .astype(str)
                .to_numpy()
            )

            p2_warning = (
                prediction_2[
                    "early_warning"
                ]
                .astype(str)
                .to_numpy()
            )

            classification_error = float(
                np.max(
                    np.abs(
                        p1_risk - p2_risk
                    )
                )
            )

            regression_error = float(
                np.max(
                    np.abs(
                        p1_score - p2_score
                    )
                )
            )

            warning_equal = bool(
                np.array_equal(
                    p1_warning,
                    p2_warning,
                )
            )

            reproducibility_ok = bool(
                classification_error == 0.0
                and regression_error == 0.0
                and warning_equal
            )

            reproducibility_detail = (
                f"classification_error="
                f"{classification_error}; "
                f"regression_error="
                f"{regression_error}; "
                f"warning_equal="
                f"{warning_equal}"
            )

        except Exception as exc:

            reproducibility_detail = (
                f"{type(exc).__name__}: {exc}"
            )

    else:

        reproducibility_detail = (
            "prediction execution unavailable"
        )

    add_check(
        checks,
        "prediction_reproducibility",
        reproducibility_ok,
        reproducibility_detail,
    )

    if not reproducibility_ok:
        failures.append(
            "Production prediction reproducibility failed."
        )

    # ==================================================================
    # 10. PREVIOUS PHASES
    # ==================================================================

    phase9a_status = status_from_report(
        PHASE9A_JSON
    )

    phase9b_status = status_from_report(
        PHASE9B_JSON
    )

    phase9c_status = status_from_report(
        PHASE9C_JSON
    )

    phase9a_ok = phase9a_status == "PASS"
    phase9b_ok = phase9b_status == "PASS"
    phase9c_ok = phase9c_status == "PASS"

    add_check(
        checks,
        "phase9A_health",
        phase9a_ok,
        f"status={phase9a_status}",
    )

    add_check(
        checks,
        "phase9B_health",
        phase9b_ok,
        f"status={phase9b_status}",
    )

    add_check(
        checks,
        "phase9C_health",
        phase9c_ok,
        f"status={phase9c_status}",
    )

    if not phase9a_ok:
        failures.append(
            "Phase 9A health gate failed."
        )

    if not phase9b_ok:
        failures.append(
            "Phase 9B health gate failed."
        )

    if not phase9c_ok:
        failures.append(
            "Phase 9C health gate failed."
        )

    # ==================================================================
    # 11. CONTINUOUS VALIDATION CONTRACT
    # ==================================================================

    contract_ok = bool(
        session_ok
        and feature_ok
        and data_quality_ok
        and cls_hash_ok
        and reg_hash_ok
        and prediction_execution_ok
        and warning_health_ok
        and output_range_ok
        and reproducibility_ok
        and phase9a_ok
        and phase9b_ok
        and phase9c_ok
    )

    add_check(
        checks,
        "continuous_validation_contract",
        contract_ok,
        (
            "all production health controls satisfied"
            if contract_ok
            else "one or more health controls failed"
        ),
    )

    if not contract_ok:
        failures.append(
            "Continuous validation contract failed."
        )

    # ==================================================================
    # 12. HEALTH SNAPSHOT
    # ==================================================================

    health_snapshot = {
        "rows": int(rows),
        "session": sessions,
        "warning_threshold": THRESHOLD,
        "early_warning_count": warning_count,
        "early_warning_rate": warning_rate,
        "risk_probability_min": risk_min,
        "risk_probability_max": risk_max,
        "risk_probability_mean": risk_mean,
        "predicted_exam_score_min": score_min,
        "predicted_exam_score_max": score_max,
        "predicted_exam_score_mean": score_mean,
        "data_quality_remediation": {
            "status": (
                "PASS"
                if remediation_ok
                else "FAIL"
            ),
            "session": (
                remediation.get(
                    "session",
                    []
                )
                if isinstance(
                    remediation,
                    dict,
                )
                else []
            ),
            "structural_missing_cells": (
                structural_missing_cells
            ),
            "unexpected_missing_cells": (
                unexpected_missing_cells
            ),
            "non_historical_missing_cells": (
                non_historical_missing_cells
            ),
            "duplicate_rows": duplicate_rows,
            "contract": bool(
                remediation_ok
            ),
        },
        "classification_hash": cls_hash,
        "regression_hash": reg_hash,
    }

    # ==================================================================
    # 13. FINAL REPORT
    # ==================================================================

    final_status = (
        "PASS"
        if contract_ok
        else "HEALTH_REVIEW_REQUIRED"
    )

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
        "health_snapshot": health_snapshot,
        "previous_phases": {
            "phase9A": phase9a_status,
            "phase9B": phase9b_status,
            "phase9C": phase9c_status,
        },
        "controls": {
            "session_health": bool(session_ok),
            "feature_health": bool(feature_ok),
            "data_quality": bool(data_quality_ok),
            "classification_model_identity": bool(
                cls_hash_ok
            ),
            "regression_model_identity": bool(
                reg_hash_ok
            ),
            "prediction_execution": bool(
                prediction_execution_ok
            ),
            "early_warning_health": bool(
                warning_health_ok
            ),
            "prediction_output_range": bool(
                output_range_ok
            ),
            "prediction_reproducibility": bool(
                reproducibility_ok
            ),
            "continuous_validation_contract": bool(
                contract_ok
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
            "classification_sha256": cls_hash,
            "regression_sha256": reg_hash,
            "threshold": THRESHOLD,
        },
    }

    audit_record = json_safe(
        audit_record
    )

    REPORT_JSON.write_text(
        json.dumps(
            audit_record,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    pd.DataFrame(checks).to_csv(
        REPORT_CSV,
        index=False,
    )

    # ==================================================================
    # 14. CONSOLE OUTPUT
    # ==================================================================

    print()
    print("=" * 72)
    print("PHASE 9D CONTINUOUS VALIDATION / HEALTH GATE")
    print("=" * 72)

    print(
        f"Production rows:             {rows:,}"
    )

    print(
        f"Session:                     {sessions}"
    )

    if warning_count is not None:
        print(
            f"Early-warning flags:         "
            f"{warning_count:,}"
        )

        print(
            f"Early-warning rate:          "
            f"{warning_rate:.4%}"
        )

        print(
            f"Risk probability:            "
            f"{risk_min:.4f} - {risk_max:.4f}"
        )

        print(
            f"Predicted exam score:        "
            f"{score_min:.2f} - {score_max:.2f}"
        )

    else:
        print(
            "Early-warning flags:         None"
        )

        print(
            "Early-warning rate:          N/A"
        )

    print()
    print(
        "Data quality:                "
        f"{'PASS' if data_quality_ok else 'FAIL'}"
    )

    print(
        "Model identity:              "
        f"{'PASS' if cls_hash_ok and reg_hash_ok else 'FAIL'}"
    )

    print(
        "Prediction execution:        "
        f"{'PASS' if prediction_execution_ok else 'FAIL'}"
    )

    print(
        "Reproducibility:             "
        f"{'PASS' if reproducibility_ok else 'FAIL'}"
    )

    print(
        "Previous phases:             "
        f"{'PASS' if phase9a_ok and phase9b_ok and phase9c_ok else 'FAIL'}"
    )

    print(
        "Continuous contract:         "
        f"{'PASS' if contract_ok else 'FAIL'}"
    )

    print()

    print(
        f"PHASE 9D STATUS: {final_status}"
    )

    print()
    print(
        f"Report: {REPORT_JSON}"
    )

    print(
        f"Checks: {REPORT_CSV}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()