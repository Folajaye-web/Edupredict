from pathlib import Path
import hashlib
import json
import sys
import subprocess

import joblib
import numpy as np
import pandas as pd


# ============================================================
# EDUPREDICT PHASE 7C
# PRODUCTION RELEASE GATE
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "nigeria_validation" / "splits"
MODEL_DIR = ROOT / "models" / "nigeria_validation"
REPORT_DIR = ROOT / "reports" / "nigeria_validation"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN = DATA_DIR / "train_2023_2024.csv"
VALID = DATA_DIR / "validation_2024_2025.csv"
TEST = DATA_DIR / "test_2025_2026.csv"

CLS_MODEL = MODEL_DIR / "phase5_classification_hgb.joblib"
REG_MODEL = MODEL_DIR / "phase5_regression_hgb.joblib"

PHASE7A_REPORT = REPORT_DIR / "phase7A_artifact_gate.json"
PHASE7B_REPORT = REPORT_DIR / "phase7B_subgroup_safety_gate.json"

PHASE7_REPORT = REPORT_DIR / "phase7_production_qualification.json"
PHASE7_CONTRACT = REPORT_DIR / "phase7_production_contract.json"
FEATURE_REPORT = REPORT_DIR / "feature_build_report.json"
PHASE5_REPORT = REPORT_DIR / "phase5_final_holdout_results.json"

TARGET_CLS = "target_below_credit"
TARGET_REG = "target_exam_score"

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

SELECTED_THRESHOLD = 0.40

EXPECTED_SESSIONS = {
    TRAIN: "2023/2024",
    VALID: "2024/2025",
    TEST: "2025/2026",
}


# ============================================================
# UTILITIES
# ============================================================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def check_file(path, label, checks, failures):
    if path.exists():
        checks.append({
            "gate": "artifact_exists",
            "item": label,
            "status": "PASS",
            "detail": str(path),
        })
        return True

    checks.append({
        "gate": "artifact_exists",
        "item": label,
        "status": "FAIL",
        "detail": str(path),
    })

    failures.append(f"Missing artifact: {path}")
    return False


def load_dataset(path, expected_session, checks, failures):

    try:
        df = pd.read_csv(path)
    except Exception as exc:

        checks.append({
            "gate": "dataset_load",
            "item": path.name,
            "status": "FAIL",
            "detail": str(exc),
        })

        failures.append(
            f"Could not load {path.name}: {exc}"
        )

        return None

    checks.append({
        "gate": "dataset_load",
        "item": path.name,
        "status": "PASS",
        "detail": f"{len(df):,} rows",
    })

    if "Academic_Session" not in df.columns:

        checks.append({
            "gate": "session_integrity",
            "item": path.name,
            "status": "FAIL",
            "detail": "Academic_Session column missing",
        })

        failures.append(
            f"{path.name}: Academic_Session missing"
        )

    else:

        sessions = sorted(
            df["Academic_Session"]
            .astype(str)
            .unique()
        )

        if sessions == [expected_session]:

            checks.append({
                "gate": "session_integrity",
                "item": path.name,
                "status": "PASS",
                "detail": (
                    f"expected={expected_session}; "
                    f"actual={sessions}"
                ),
            })

        else:

            checks.append({
                "gate": "session_integrity",
                "item": path.name,
                "status": "FAIL",
                "detail": (
                    f"expected={expected_session}; "
                    f"actual={sessions}"
                ),
            })

            failures.append(
                f"{path.name}: session mismatch"
            )

    missing = sorted(
        set(FEATURES + [TARGET_CLS, TARGET_REG])
        - set(df.columns)
    )

    forbidden_used = sorted(
        set(FEATURES).intersection(FORBIDDEN)
    )

    if missing:

        checks.append({
            "gate": "feature_contract",
            "item": path.name,
            "status": "FAIL",
            "detail": f"missing={missing}",
        })

        failures.append(
            f"{path.name}: missing features {missing}"
        )

    elif forbidden_used:

        checks.append({
            "gate": "feature_contract",
            "item": path.name,
            "status": "FAIL",
            "detail": (
                f"forbidden predictors used={forbidden_used}"
            ),
        })

        failures.append(
            f"{path.name}: forbidden predictors used"
        )

    else:

        checks.append({
            "gate": "feature_contract",
            "item": path.name,
            "status": "PASS",
            "detail": "production feature contract valid",
        })

    return df


def run_prediction_smoke_test(model, df, kind):

    X = df[FEATURES]

    if kind == "classification":

        if not hasattr(model, "predict_proba"):
            return False, "predict_proba unavailable"

        predictions = model.predict_proba(X)

        if predictions.shape[0] != len(df):
            return False, "prediction row count mismatch"

        if predictions.shape[1] != 2:
            return False, "unexpected probability shape"

        if not np.isfinite(predictions).all():
            return False, "non-finite classification predictions"

        return True, f"{len(df):,} predictions"

    if kind == "regression":

        if not hasattr(model, "predict"):
            return False, "predict unavailable"

        predictions = model.predict(X)

        if len(predictions) != len(df):
            return False, "prediction row count mismatch"

        if not np.isfinite(predictions).all():
            return False, "non-finite regression predictions"

        return True, f"{len(df):,} predictions"

    return False, "unknown model type"


def load_json(path):

    if not path.exists():
        return None

    try:
        return json.loads(
            path.read_text(encoding="utf-8")
        )
    except Exception:
        return None


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("EDUPREDICT NIGERIA VALIDATION — PHASE 7C")
    print("=" * 72)
    print("Purpose: Production Release Gate")
    print()

    checks = []
    failures = []
    warnings = []

    # --------------------------------------------------------
    # 1. REQUIRED ARTIFACTS
    # --------------------------------------------------------

    print("Checking required production artifacts...")

    required = [
        (TRAIN, "train_2023_2024.csv"),
        (VALID, "validation_2024_2025.csv"),
        (TEST, "test_2025_2026.csv"),
        (CLS_MODEL, "phase5_classification_hgb.joblib"),
        (REG_MODEL, "phase5_regression_hgb.joblib"),
        (PHASE7A_REPORT, "phase7A_artifact_gate.json"),
        (PHASE7B_REPORT, "phase7B_subgroup_safety_gate.json"),
    ]

    for path, label in required:
        check_file(
            path,
            label,
            checks,
            failures,
        )

    if failures:
        print()
        print("Required artifact gate: FAIL")

        for failure in failures:
            print("  -", failure)

        raise SystemExit(1)

    print("Required artifact gate: PASS")

    # --------------------------------------------------------
    # 2. LOAD DATA
    # --------------------------------------------------------

    print()
    print("Loading frozen temporal datasets...")

    train = load_dataset(
        TRAIN,
        "2023/2024",
        checks,
        failures,
    )

    valid = load_dataset(
        VALID,
        "2024/2025",
        checks,
        failures,
    )

    test = load_dataset(
        TEST,
        "2025/2026",
        checks,
        failures,
    )

    if train is None or valid is None or test is None:
        print()
        print("Dataset gate: FAIL")
        raise SystemExit(1)

    # --------------------------------------------------------
    # 3. FROZEN ARTIFACT HASHES
    # --------------------------------------------------------

    print()
    print("Calculating frozen model hashes...")

    cls_hash = sha256_file(CLS_MODEL)
    reg_hash = sha256_file(REG_MODEL)

    print("Classification SHA256:", cls_hash)
    print("Regression SHA256:    ", reg_hash)

    # --------------------------------------------------------
    # 4. LOAD MODELS
    # --------------------------------------------------------

    print()
    print("Loading frozen models...")

    try:
        cls_model = joblib.load(CLS_MODEL)

        checks.append({
            "gate": "model_loadability",
            "item": "classification",
            "status": "PASS",
            "detail": type(cls_model).__name__,
        })

    except Exception as exc:

        checks.append({
            "gate": "model_loadability",
            "item": "classification",
            "status": "FAIL",
            "detail": str(exc),
        })

        failures.append(
            f"classification model load failed: {exc}"
        )

        cls_model = None

    try:
        reg_model = joblib.load(REG_MODEL)

        checks.append({
            "gate": "model_loadability",
            "item": "regression",
            "status": "PASS",
            "detail": type(reg_model).__name__,
        })

    except Exception as exc:

        checks.append({
            "gate": "model_loadability",
            "item": "regression",
            "status": "FAIL",
            "detail": str(exc),
        })

        failures.append(
            f"regression model load failed: {exc}"
        )

        reg_model = None

    if cls_model is None or reg_model is None:
        raise SystemExit(1)

    # --------------------------------------------------------
    # 5. FINAL-TEST SMOKE TEST
    # --------------------------------------------------------

    print()
    print("Running final-test prediction smoke test...")

    cls_ok, cls_detail = run_prediction_smoke_test(
        cls_model,
        test,
        "classification",
    )

    checks.append({
        "gate": "classification_prediction",
        "item": "2025/2026 final test",
        "status": "PASS" if cls_ok else "FAIL",
        "detail": cls_detail,
    })

    if not cls_ok:
        failures.append(
            f"classification prediction failed: {cls_detail}"
        )

    reg_ok, reg_detail = run_prediction_smoke_test(
        reg_model,
        test,
        "regression",
    )

    checks.append({
        "gate": "regression_prediction",
        "item": "2025/2026 final test",
        "status": "PASS" if reg_ok else "FAIL",
        "detail": reg_detail,
    })

    if not reg_ok:
        failures.append(
            f"regression prediction failed: {reg_detail}"
        )

    # --------------------------------------------------------
    # 6. REPRODUCIBILITY
    # --------------------------------------------------------

    print()
    print("Checking prediction reproducibility...")

    p1 = cls_model.predict_proba(
        test[FEATURES]
    )[:, 1]

    p2 = cls_model.predict_proba(
        test[FEATURES]
    )[:, 1]

    reproducibility_error = float(
        np.max(np.abs(p1 - p2))
    )

    reproducibility_pass = (
        reproducibility_error == 0.0
    )

    checks.append({
        "gate": "reproducibility",
        "item": "classification predictions",
        "status": (
            "PASS"
            if reproducibility_pass
            else "FAIL"
        ),
        "detail": (
            f"max_absolute_difference="
            f"{reproducibility_error:.12g}"
        ),
    })

    if not reproducibility_pass:
        failures.append(
            "classification prediction reproducibility failed"
        )

    # --------------------------------------------------------
    # 7. PHASE 7A RESULT
    # --------------------------------------------------------

    print()
    print("Reading Phase 7A gate...")

    phase7a = load_json(PHASE7A_REPORT)

    phase7a_status = (
        phase7a.get("status")
        if phase7a
        else None
    )

    phase7a_pass = phase7a_status == "PASS"

    checks.append({
        "gate": "phase7A",
        "item": "artifact_and_contract_gate",
        "status": "PASS" if phase7a_pass else "FAIL",
        "detail": str(phase7a_status),
    })

    if not phase7a_pass:
        failures.append(
            "Phase 7A is not PASS"
        )

    # --------------------------------------------------------
    # 8. PHASE 7B RESULT
    # --------------------------------------------------------

    print()
    print("Reading Phase 7B gate...")

    phase7b = load_json(PHASE7B_REPORT)

    phase7b_status = (
        phase7b.get("status")
        if phase7b
        else None
    )

    phase7b_pass = phase7b_status == "PASS"

    checks.append({
        "gate": "phase7B",
        "item": "subgroup_safety_gate",
        "status": "PASS" if phase7b_pass else "FAIL",
        "detail": str(phase7b_status),
    })

    if not phase7b_pass:
        failures.append(
            "Phase 7B is not PASS"
        )

    # --------------------------------------------------------
    # 9. THRESHOLD POLICY
    # --------------------------------------------------------

    threshold_path = (
        REPORT_DIR / "phase7_threshold_policy.csv"
    )

    print()
    print("Checking frozen threshold policy...")

    threshold_pass = False
    threshold_detail = ""

    if threshold_path.exists():

        try:

            threshold_df = pd.read_csv(
                threshold_path
            )

            threshold_match = threshold_df[
                np.isclose(
                    threshold_df["threshold"],
                    SELECTED_THRESHOLD,
                )
            ]

            if len(threshold_match) == 1:

                threshold_pass = True

                threshold_detail = (
                    f"selected_threshold="
                    f"{SELECTED_THRESHOLD:.2f}"
                )

            else:

                threshold_detail = (
                    "selected threshold 0.40 "
                    "not uniquely represented"
                )

        except Exception as exc:

            threshold_detail = str(exc)

    else:

        threshold_detail = (
            "phase7_threshold_policy.csv missing"
        )

    checks.append({
        "gate": "threshold_policy",
        "item": "classification threshold",
        "status": (
            "PASS"
            if threshold_pass
            else "FAIL"
        ),
        "detail": threshold_detail,
    })

    if not threshold_pass:
        failures.append(
            f"Threshold policy invalid: {threshold_detail}"
        )

    # --------------------------------------------------------
    # 10. PRODUCTION CONTRACT CONSISTENCY
    # --------------------------------------------------------

    print()
    print("Checking Phase 7 production contract...")

    phase7_contract = load_json(
        PHASE7_CONTRACT
    )

    contract_pass = False
    contract_detail = ""

    if phase7_contract:

        contract_features = phase7_contract.get(
            "features",
            [],
        )

        contract_forbidden = set(
            phase7_contract.get(
                "forbidden_predictors",
                [],
            )
        )

        contract_threshold = (
            phase7_contract
            .get("threshold_policy", {})
            .get("selected_threshold")
        )

        contract_status = phase7_contract.get("status")
        if (
            contract_status == "TECHNICAL_QUALIFICATION_PASS"
            and contract_features == FEATURES
            and not (contract_forbidden & set(FEATURES))
            and contract_threshold is not None
            and abs(float(contract_threshold) - SELECTED_THRESHOLD) < 1e-12
        ):
            contract_pass = True
            contract_detail = (
                "qualification status, feature contract, forbidden predictors "
                "and threshold agree"
            )
        else:
            contract_detail = (
                f"qualification_status={contract_status}; required=TECHNICAL_QUALIFICATION_PASS; "
                "or feature/forbidden-predictor/threshold contract mismatch"
            )

    else:

        contract_detail = (
            "phase7_production_contract.json missing"
        )

    checks.append({
        "gate": "production_contract",
        "item": "Phase 7 contract",
        "status": (
            "PASS"
            if contract_pass
            else "FAIL"
        ),
        "detail": contract_detail,
    })

    if not contract_pass:
        failures.append(
            f"Production contract invalid: {contract_detail}"
        )

    # --------------------------------------------------------
    # 11. ENVIRONMENT RECORD
    # --------------------------------------------------------

    print()
    print("Recording production environment...")

    try:
        import sklearn
        import numpy
        import pandas
        import joblib as joblib_module

        environment = {
            "python": sys.version,
            "python_executable": sys.executable,
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib_module.__version__,
        }

        environment_status = "PASS"

    except Exception as exc:

        environment = {
            "error": str(exc)
        }

        environment_status = "REVIEW"

        warnings.append(
            f"Environment metadata incomplete: {exc}"
        )

    checks.append({
        "gate": "environment_record",
        "item": "runtime dependencies",
        "status": environment_status,
        "detail": json.dumps(environment),
    })

    # --------------------------------------------------------
    # 12. FEATURE PIPELINE / MODEL PROVENANCE
    # --------------------------------------------------------
    feature_report = load_json(FEATURE_REPORT)
    phase5_report = load_json(PHASE5_REPORT)
    provenance_pass = bool(
        feature_report
        and phase5_report
        and feature_report.get("status") == "PASS"
        and feature_report.get("feature_pipeline_version") == "2.0-period-safe-school-aware"
        and phase5_report.get("status") == "PASS"
        and phase5_report.get("feature_pipeline_version") == feature_report.get("feature_pipeline_version")
        and phase5_report.get("feature_input_sha256") == feature_report.get("input_sha256")
        and environment.get("scikit_learn") == "1.9.0"
    )
    provenance_detail = (
        f"feature_version={feature_report.get('feature_pipeline_version') if feature_report else None}; "
        f"phase5_version={phase5_report.get('feature_pipeline_version') if phase5_report else None}; "
        f"source_hash_match={bool(feature_report and phase5_report and phase5_report.get('feature_input_sha256') == feature_report.get('input_sha256'))}; "
        f"scikit_learn={environment.get('scikit_learn')}"
    )
    checks.append({
        "gate": "feature_pipeline_provenance",
        "item": "Corrected feature/model provenance",
        "status": "PASS" if provenance_pass else "FAIL",
        "detail": provenance_detail,
    })
    if not provenance_pass:
        failures.append("Feature pipeline/model provenance is stale or incompatible.")

    # --------------------------------------------------------
    # 13. RELEASE DECISION
    # --------------------------------------------------------

    # 7C is a technical artifact gate, not a school-pilot approval.
    # It requires Phase 7 qualification to explicitly clear review, in
    # addition to artifact, feature provenance, calibration-policy,
    # reproducibility, and Phase 7A/7B checks.

    release_pass = (
        len(failures) == 0
        and phase7a_pass
        and phase7b_pass
        and threshold_pass
        and contract_pass
        and reproducibility_pass
        and cls_ok
        and reg_ok
    )

    # This is a technical artifact gate only; it never authorises a school pilot.
    status = (
        "TECHNICAL_RELEASE_READY"
        if release_pass
        else "CONDITIONAL_REVIEW"
    )

    # --------------------------------------------------------
    # 13. WRITE REPORT
    # --------------------------------------------------------

    report = {
        "project": "EduPredict",
        "phase": "7C",
        "gate": "Production Release Gate",
        "status": status,
        "release_ready": False,
        "technical_release_ready": release_pass,
        "pilot_authorised": False,
        "scope_note": "Technical artifact gate only; school pilot authorisation is controlled separately by Phase 10C governance.",
        "test_session": "2025/2026",
        "threshold": SELECTED_THRESHOLD,
        "model_artifacts": {
            "classification": str(CLS_MODEL),
            "classification_sha256": cls_hash,
            "regression": str(REG_MODEL),
            "regression_sha256": reg_hash,
        },
        "datasets": {
            "training": str(TRAIN),
            "validation": str(VALID),
            "final_test": str(TEST),
            "training_rows": int(len(train)),
            "validation_rows": int(len(valid)),
            "final_test_rows": int(len(test)),
        },
        "gates": checks,
        "phase7A_status": phase7a_status,
        "phase7B_status": phase7b_status,
        "reproducibility_error": reproducibility_error,
        "environment": environment,
        "warnings": warnings,
        "failures": failures,
    }

    report_path = (
        REPORT_DIR
        / "phase7C_production_release_gate.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # 14. WRITE CSV
    # --------------------------------------------------------

    checks_df = pd.DataFrame(checks)

    checks_path = (
        REPORT_DIR
        / "phase7C_production_release_gate.csv"
    )

    checks_df.to_csv(
        checks_path,
        index=False,
    )

    # --------------------------------------------------------
    # 15. RELEASE MANIFEST
    # --------------------------------------------------------

    manifest = {
        "project": "EduPredict",
        "release_gate": "Phase 7C",
        "status": status,
        "technical_release_ready": release_pass,
        "pilot_authorised": False,
        "classification_model": str(CLS_MODEL),
        "classification_sha256": cls_hash,
        "regression_model": str(REG_MODEL),
        "regression_sha256": reg_hash,
        "training_session": "2023/2024",
        "validation_session": "2024/2025",
        "final_test_session": "2025/2026",
        "threshold": SELECTED_THRESHOLD,
        "feature_count": len(FEATURES),
        "feature_pipeline_version": (feature_report or {}).get("feature_pipeline_version"),
        "features": FEATURES,
        "forbidden_predictors": sorted(FORBIDDEN),
        "phase7A": phase7a_status,
        "phase7B": phase7b_status,
        "reproducibility_error": reproducibility_error,
        "python_executable": sys.executable,
        "environment": environment,
    }

    manifest_path = (
        REPORT_DIR
        / "phase7C_release_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # 16. CONSOLE OUTPUT
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("PHASE 7C PRODUCTION RELEASE GATE")
    print("=" * 72)

    print()
    print(
        f"Phase 7A:              "
        f"{'PASS' if phase7a_pass else 'FAIL'}"
    )

    print(
        f"Phase 7B:              "
        f"{'PASS' if phase7b_pass else 'FAIL'}"
    )

    print(
        f"Threshold policy:      "
        f"{'PASS' if threshold_pass else 'FAIL'}"
    )

    print(
        f"Production contract:   "
        f"{'PASS' if contract_pass else 'FAIL'}"
    )

    print(
        f"Classification smoke:  "
        f"{'PASS' if cls_ok else 'FAIL'}"
    )

    print(
        f"Regression smoke:      "
        f"{'PASS' if reg_ok else 'FAIL'}"
    )

    print(
        f"Reproducibility:       "
        f"{'PASS' if reproducibility_pass else 'FAIL'}"
    )

    print(
        f"Final-test rows:       {len(test):,}"
    )

    print(
        f"Classification SHA256: {cls_hash}"
    )

    print(
        f"Regression SHA256:     {reg_hash}"
    )

    if warnings:

        print()
        print("WARNINGS:")

        for warning in warnings:
            print("  -", warning)

    if failures:

        print()
        print("RELEASE BLOCKERS:")

        for failure in failures:
            print("  -", failure)

    print()
    print(
        f"PHASE 7C STATUS: {status}"
    )

    print()
    print(f"Report:   {report_path}")
    print(f"Checks:   {checks_path}")
    print(f"Manifest: {manifest_path}")

    print("=" * 72)

    if not release_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
