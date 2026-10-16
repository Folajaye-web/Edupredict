from pathlib import Path
import hashlib
import json
import importlib.util
import sys
import time

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "nigeria_validation" / "splits"
MODEL_DIR = ROOT / "models" / "nigeria_validation"
REPORT_DIR = ROOT / "reports" / "nigeria_validation"
APP_DIR = ROOT / "app"
SRC_DIR = ROOT / "src"

TEST = DATA_DIR / "test_2025_2026.csv"
CLS_MODEL = MODEL_DIR / "phase5_classification_hgb.joblib"
REG_MODEL = MODEL_DIR / "phase5_regression_hgb.joblib"

PHASE7A = REPORT_DIR / "phase7A_artifact_gate.json"
PHASE7B = REPORT_DIR / "phase7B_subgroup_safety_gate.json"
PHASE7C = REPORT_DIR / "phase7C_production_release_gate.json"
PHASE7_CONTRACT = REPORT_DIR / "phase7_production_contract.json"

APP_CANDIDATES = [
    APP_DIR / "streamlit_app.py",
    APP_DIR / "app.py",
    ROOT / "streamlit_app.py",
]

FEATURES = [
    "School", "Term", "Class", "Arm", "Gender", "Subject",
    "Term_Number", "CA1", "CA2", "Assignment", "Project",
    "Attendance", "Previous_Term_Total", "Previous_Year_Average",
    "Previous_Term_Grade_Numeric", "Historical_Subject_Average",
    "Historical_Subject_Count", "Historical_Overall_Average",
    "Historical_Overall_Count",
]

FORBIDDEN = {
    "Exam", "Final_Year_Result", "Grade",
    "Intervention", "Promotion_Status", "Total",
}

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def check(label, ok, detail):
    return {
        "check": label,
        "status": "PASS" if ok else "FAIL",
        "detail": str(detail),
    }

def load_json(path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    checks = []

    print("=" * 72)
    print("EDUPREDICT NIGERIA VALIDATION — PHASE 8A")
    print("=" * 72)
    print("Purpose: Production Integration Gate")
    print()

    # 1. Project structure
    for label, path in [
        ("project_root", ROOT),
        ("data_directory", DATA_DIR),
        ("model_directory", MODEL_DIR),
        ("report_directory", REPORT_DIR),
        ("source_directory", SRC_DIR),
    ]:
        checks.append(check(label, path.exists(), path))

    # 2. Production release evidence
    for label, path in [
        ("phase7A_gate", PHASE7A),
        ("phase7B_gate", PHASE7B),
        ("phase7C_gate", PHASE7C),
        ("phase7_contract", PHASE7_CONTRACT),
    ]:
        checks.append(check(label, path.exists(), path))

    p7a = load_json(PHASE7A)
    p7b = load_json(PHASE7B)
    p7c = load_json(PHASE7C)
    contract = load_json(PHASE7_CONTRACT)

    checks.append(check(
        "phase7A_status",
        isinstance(p7a, dict) and p7a.get("status") == "PASS",
        p7a.get("status") if isinstance(p7a, dict) else "missing/invalid",
    ))
    checks.append(check(
        "phase7B_status",
        isinstance(p7b, dict) and p7b.get("status") == "PASS",
        p7b.get("status") if isinstance(p7b, dict) else "missing/invalid",
    ))
    checks.append(check(
        "phase7C_status",
        isinstance(p7c, dict) and p7c.get("status") == "TECHNICAL_RELEASE_READY",
        p7c.get("status") if isinstance(p7c, dict) else "missing/invalid",
    ))
    checks.append(check(
        "phase7_contract_status",
        isinstance(contract, dict) and contract.get("status") == "TECHNICAL_QUALIFICATION_PASS",
        contract.get("status") if isinstance(contract, dict) else "missing/invalid",
    ))

    # 3. Frozen production artifacts
    for label, path in [
        ("classification_model", CLS_MODEL),
        ("regression_model", REG_MODEL),
        ("final_test_dataset", TEST),
    ]:
        checks.append(check(label, path.exists(), path))

    # 4. Dataset integration contract
    df = None
    if TEST.exists():
        try:
            df = pd.read_csv(TEST)
            checks.append(check("final_test_load", True, f"{len(df):,} rows"))
        except Exception as exc:
            checks.append(check("final_test_load", False, exc))

    if df is not None:
        missing = sorted(set(FEATURES) - set(df.columns))
        forbidden_used = sorted(set(FEATURES) & FORBIDDEN)
        session_values = sorted(df["Academic_Session"].astype(str).unique()) if "Academic_Session" in df else []

        checks.append(check(
            "production_feature_contract",
            not missing and not forbidden_used,
            f"missing={missing}; forbidden_used={forbidden_used}",
        ))
        checks.append(check(
            "production_session",
            session_values == ["2025/2026"],
            session_values,
        ))

    # 5. Model loading and prediction integration
    cls = reg = None
    try:
        cls = joblib.load(CLS_MODEL)
        checks.append(check("classification_model_load", True, type(cls).__name__))
    except Exception as exc:
        checks.append(check("classification_model_load", False, exc))

    try:
        reg = joblib.load(REG_MODEL)
        checks.append(check("regression_model_load", True, type(reg).__name__))
    except Exception as exc:
        checks.append(check("regression_model_load", False, exc))

    if df is not None and cls is not None:
        try:
            p = cls.predict_proba(df[FEATURES])[:, 1]
            checks.append(check(
                "classification_production_smoke",
                len(p) == len(df) and bool(pd.Series(p).notna().all()),
                f"{len(p):,} predictions; range={float(p.min()):.4f}-{float(p.max()):.4f}",
            ))
        except Exception as exc:
            checks.append(check("classification_production_smoke", False, exc))

    if df is not None and reg is not None:
        try:
            yhat = reg.predict(df[FEATURES])
            checks.append(check(
                "regression_production_smoke",
                len(yhat) == len(df) and bool(pd.Series(yhat).notna().all()),
                f"{len(yhat):,} predictions; range={float(yhat.min()):.2f}-{float(yhat.max()):.2f}",
            ))
        except Exception as exc:
            checks.append(check("regression_production_smoke", False, exc))

    # 6. Frozen artifact hashes must agree with Phase 7 contract
    if isinstance(contract, dict):
        artifacts = contract.get("artifacts", {})
        expected_cls = artifacts.get("classification_sha256")
        expected_reg = artifacts.get("regression_sha256")

        actual_cls = sha256_file(CLS_MODEL) if CLS_MODEL.exists() else None
        actual_reg = sha256_file(REG_MODEL) if REG_MODEL.exists() else None

        checks.append(check(
            "classification_hash_contract",
            expected_cls is not None and actual_cls == expected_cls,
            f"expected={expected_cls}; actual={actual_cls}",
        ))
        checks.append(check(
            "regression_hash_contract",
            expected_reg is not None and actual_reg == expected_reg,
            f"expected={expected_reg}; actual={actual_reg}",
        ))

    # 7. Application integration discovery
    app_path = next((p for p in APP_CANDIDATES if p.exists()), None)
    checks.append(check(
        "streamlit_application_present",
        app_path is not None,
        app_path if app_path else "No recognised Streamlit entrypoint found",
    ))

    if app_path:
        text = app_path.read_text(encoding="utf-8", errors="ignore")
        integration_markers = [
            "streamlit",
            "edupredict",
            "model",
        ]
        found = [m for m in integration_markers if m.lower() in text.lower()]
        checks.append(check(
            "application_integration_markers",
            len(found) >= 2,
            f"found={found}",
        ))

    # 8. Python import sanity for project package
    package_candidates = [
        SRC_DIR / "edupredict",
        ROOT / "edupredict",
    ]
    package_path = next((p for p in package_candidates if p.exists()), None)
    checks.append(check(
        "edupredict_package_present",
        package_path is not None,
        package_path if package_path else "No edupredict package directory found",
    ))

    # 9. Produce machine-readable gate
    failures = [c for c in checks if c["status"] != "PASS"]
    status = "PASS" if not failures else "NOT_READY"

    result = {
        "project": "EduPredict",
        "phase": "8A",
        "gate": "Production Integration Gate",
        "status": status,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "root": str(ROOT),
        "checks": checks,
        "failures": failures,
        "production_artifacts": {
            "classification_model": str(CLS_MODEL),
            "regression_model": str(REG_MODEL),
            "final_test_dataset": str(TEST),
        },
    }

    json_path = REPORT_DIR / "phase8A_production_integration.json"
    csv_path = REPORT_DIR / "phase8A_production_integration.csv"

    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(checks).to_csv(csv_path, index=False)

    print()
    print("=" * 72)
    print("PHASE 8A RESULT")
    print("=" * 72)
    for item in checks:
        print(f"{item['check']:<34} {item['status']:<6} {item['detail']}")
    print()
    print(f"PHASE 8A STATUS: {status}")
    print()
    print(f"Report: {json_path}")
    print(f"Checks: {csv_path}")
    print("=" * 72)

    return 0 if status == "PASS" else 1

if __name__ == "__main__":
    raise SystemExit(main())
