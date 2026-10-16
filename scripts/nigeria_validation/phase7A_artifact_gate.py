from pathlib import Path
import hashlib
import json
import joblib
import pandas as pd


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


EXPECTED_SESSIONS = {
    TRAIN: "2023/2024",
    VALID: "2024/2025",
    TEST: "2025/2026",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def check_file(path: Path):
    return {
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
    }


def main():

    print("=" * 72)
    print("EDUPREDICT NIGERIA VALIDATION — PHASE 7A")
    print("=" * 72)
    print("Purpose: Production Artifact & Contract Gate")
    print()

    checks = []
    failures = []

    # --------------------------------------------------------
    # 1. REQUIRED ARTIFACTS
    # --------------------------------------------------------

    required_files = [
        TRAIN,
        VALID,
        TEST,
        CLS_MODEL,
        REG_MODEL,
    ]

    print("Checking required artifacts...")

    for path in required_files:

        result = check_file(path)

        passed = result["exists"] and result["size_bytes"] > 0

        checks.append(
            {
                "gate": "artifact_exists",
                "item": str(path.relative_to(ROOT)),
                "status": "PASS" if passed else "FAIL",
                "detail": (
                    f"{result['size_bytes']:,} bytes"
                    if passed
                    else "missing or empty"
                ),
            }
        )

        if not passed:
            failures.append(f"Missing or empty artifact: {path}")

    # --------------------------------------------------------
    # 2. LOAD DATASETS
    # --------------------------------------------------------

    datasets = {}

    for path in [TRAIN, VALID, TEST]:

        if not path.exists():
            continue

        try:
            df = pd.read_csv(path)
            datasets[path] = df

            passed = len(df) > 0

            checks.append(
                {
                    "gate": "dataset_load",
                    "item": path.name,
                    "status": "PASS" if passed else "FAIL",
                    "detail": f"{len(df):,} rows",
                }
            )

            if not passed:
                failures.append(f"Empty dataset: {path}")

        except Exception as exc:

            checks.append(
                {
                    "gate": "dataset_load",
                    "item": path.name,
                    "status": "FAIL",
                    "detail": str(exc),
                }
            )

            failures.append(f"Could not load dataset: {path}: {exc}")

    # --------------------------------------------------------
    # 3. SESSION INTEGRITY
    # --------------------------------------------------------

    for path, df in datasets.items():

        expected = EXPECTED_SESSIONS[path]

        if "Academic_Session" not in df.columns:

            checks.append(
                {
                    "gate": "session_integrity",
                    "item": path.name,
                    "status": "FAIL",
                    "detail": "Academic_Session column missing",
                }
            )

            failures.append(
                f"{path.name}: Academic_Session column missing"
            )

            continue

        actual = sorted(
            df["Academic_Session"]
            .astype(str)
            .dropna()
            .unique()
            .tolist()
        )

        passed = actual == [expected]

        checks.append(
            {
                "gate": "session_integrity",
                "item": path.name,
                "status": "PASS" if passed else "FAIL",
                "detail": f"expected={expected}; actual={actual}",
            }
        )

        if not passed:
            failures.append(
                f"{path.name}: session mismatch: {actual}"
            )

    # --------------------------------------------------------
    # 4. FEATURE CONTRACT
    # --------------------------------------------------------

    print("Checking feature contract...")

    for path, df in datasets.items():

        missing = [
            feature
            for feature in FEATURES
            if feature not in df.columns
        ]

        forbidden_used = sorted(
            set(FEATURES).intersection(FORBIDDEN)
        )

        passed = not missing and not forbidden_used

        detail = []

        if missing:
            detail.append(f"missing={missing}")

        if forbidden_used:
            detail.append(
                f"forbidden_used={forbidden_used}"
            )

        if not detail:
            detail.append("all production features present")

        checks.append(
            {
                "gate": "feature_contract",
                "item": path.name,
                "status": "PASS" if passed else "FAIL",
                "detail": "; ".join(detail),
            }
        )

        if not passed:
            failures.append(
                f"{path.name}: feature contract violation"
            )

    # --------------------------------------------------------
    # 5. MODEL LOADABILITY
    # --------------------------------------------------------

    models = {}

    for name, path in [
        ("classification", CLS_MODEL),
        ("regression", REG_MODEL),
    ]:

        if not path.exists():
            continue

        try:

            model = joblib.load(path)
            models[name] = model

            checks.append(
                {
                    "gate": "model_loadability",
                    "item": name,
                    "status": "PASS",
                    "detail": type(model).__name__,
                }
            )

        except Exception as exc:

            checks.append(
                {
                    "gate": "model_loadability",
                    "item": name,
                    "status": "FAIL",
                    "detail": str(exc),
                }
            )

            failures.append(
                f"{name} model could not be loaded: {exc}"
            )

    # --------------------------------------------------------
    # 6. MODEL INTERFACE
    # --------------------------------------------------------

    if TEST in datasets:

        test = datasets[TEST]

        missing = [
            feature
            for feature in FEATURES
            if feature not in test.columns
        ]

        if not missing and models.get("classification") is not None:

            try:

                X_test = test[FEATURES]

                probabilities = (
                    models["classification"]
                    .predict_proba(X_test)[:, 1]
                )

                passed = (
                    len(probabilities) == len(test)
                    and
                    probabilities.shape[0] == len(test)
                )

                checks.append(
                    {
                        "gate": "classification_interface",
                        "item": "predict_proba",
                        "status": "PASS" if passed else "FAIL",
                        "detail": (
                            f"{len(probabilities):,} predictions"
                        ),
                    }
                )

                if not passed:
                    failures.append(
                        "Classification prediction count mismatch"
                    )

            except Exception as exc:

                checks.append(
                    {
                        "gate": "classification_interface",
                        "item": "predict_proba",
                        "status": "FAIL",
                        "detail": str(exc),
                    }
                )

                failures.append(
                    f"Classification prediction failed: {exc}"
                )

        if not missing and models.get("regression") is not None:

            try:

                X_test = test[FEATURES]

                predictions = models["regression"].predict(X_test)

                passed = (
                    len(predictions) == len(test)
                )

                checks.append(
                    {
                        "gate": "regression_interface",
                        "item": "predict",
                        "status": "PASS" if passed else "FAIL",
                        "detail": (
                            f"{len(predictions):,} predictions"
                        ),
                    }
                )

                if not passed:
                    failures.append(
                        "Regression prediction count mismatch"
                    )

            except Exception as exc:

                checks.append(
                    {
                        "gate": "regression_interface",
                        "item": "predict",
                        "status": "FAIL",
                        "detail": str(exc),
                    }
                )

                failures.append(
                    f"Regression prediction failed: {exc}"
                )

    # --------------------------------------------------------
    # 7. MODEL HASHES
    # --------------------------------------------------------

    hashes = {}

    for name, path in [
        ("classification_model", CLS_MODEL),
        ("regression_model", REG_MODEL),
    ]:

        if path.exists():

            digest = sha256_file(path)
            hashes[name] = digest

            checks.append(
                {
                    "gate": "artifact_hash",
                    "item": name,
                    "status": "PASS",
                    "detail": digest,
                }
            )

    # --------------------------------------------------------
    # 8. FINAL DECISION
    # --------------------------------------------------------

    status = "PASS" if not failures else "FAIL"

    report = {
        "project": "EduPredict",
        "phase": "7A",
        "gate": "Production Artifact and Contract Gate",
        "status": status,
        "features": FEATURES,
        "forbidden_predictors": sorted(FORBIDDEN),
        "expected_sessions": {
            str(k.relative_to(ROOT)): v
            for k, v in EXPECTED_SESSIONS.items()
        },
        "artifacts": {
            "classification_model": str(CLS_MODEL),
            "regression_model": str(REG_MODEL),
            "classification_sha256": hashes.get(
                "classification_model"
            ),
            "regression_sha256": hashes.get(
                "regression_model"
            ),
        },
        "checks": checks,
        "failures": failures,
    }

    json_path = (
        REPORT_DIR /
        "phase7A_artifact_gate.json"
    )

    csv_path = (
        REPORT_DIR /
        "phase7A_artifact_gate.csv"
    )

    json_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    pd.DataFrame(checks).to_csv(
        csv_path,
        index=False,
    )

    # --------------------------------------------------------
    # 9. CONSOLE RESULT
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("PHASE 7A RESULT")
    print("=" * 72)

    for row in checks:

        print(
            f"{row['gate']:<25} "
            f"{row['status']:<5} "
            f"{row['item']}"
        )

    print()
    print(
        f"PHASE 7A STATUS: {status}"
    )

    print()
    print(f"Report: {json_path}")
    print(f"Checks: {csv_path}")

    if failures:

        print()
        print("FAILURES:")

        for failure in failures:
            print(f" - {failure}")

    print("=" * 72)

    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
