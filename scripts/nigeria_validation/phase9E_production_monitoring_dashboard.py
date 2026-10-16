"""
EduPredict — Phase 9E Production Monitoring Dashboard / Operational Health Gate

Purpose:
    Build a read-only production monitoring snapshot from the PASSING Phase 9D
    continuous-validation contract. This stage does not retrain, replace, or
    modify the frozen production models.

Inputs:
    reports/nigeria_validation/phase9/phase9D_continuous_validation_gate.json
    reports/nigeria_validation/phase9/phase9C_production_monitoring_gate.json
    reports/nigeria_validation/phase9/phase9B_operational_control_gate.json
    reports/nigeria_validation/phase9/phase9A_production_prediction_service.json
    models/nigeria_validation/*.joblib
    data/nigeria_validation/splits/test_2025_2026.csv

Outputs:
    reports/nigeria_validation/phase9/phase9E_production_monitoring_dashboard.json
    reports/nigeria_validation/phase9/phase9E_production_monitoring_dashboard.csv

Policy:
    - Phase 9D PASS is required.
    - Model hashes remain locked.
    - No automatic retraining.
    - No automatic model replacement.
    - A failed control produces HEALTH_REVIEW_REQUIRED.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "reports" / "nigeria_validation" / "phase9"
MODEL_DIR = ROOT / "models" / "nigeria_validation"
DATA_DIR = ROOT / "data" / "nigeria_validation" / "splits"

PHASE9A = REPORT_DIR / "phase9A_production_prediction_service.json"
PHASE9B = REPORT_DIR / "phase9B_operational_control_gate.json"
PHASE9C = REPORT_DIR / "phase9C_production_monitoring_gate.json"
PHASE9D = REPORT_DIR / "phase9D_continuous_validation_gate.json"

CLASS_MODEL = MODEL_DIR / "phase5_classification_hgb.joblib"
REG_MODEL = MODEL_DIR / "phase5_regression_hgb.joblib"
TEST_DATA = DATA_DIR / "test_2025_2026.csv"

OUT_JSON = REPORT_DIR / "phase9E_production_monitoring_dashboard.json"
OUT_CSV = REPORT_DIR / "phase9E_production_monitoring_dashboard.csv"

EXPECTED_CLASS_HASH = (
    "3cc65dbd4c0e48ba65b28f713a8d32d9fec8bf0631dc8a3b5ef15e00823f608a"
)
EXPECTED_REG_HASH = (
    "f61b02c5542aa0a4d2d0c1b18540f05193cc5d0c5f1f4e5a46ae5a75e0a9973e"
)

EXPECTED_SESSION = ["2025/2026"]


EXPECTED_THRESHOLD = 0.40



# Bind monitoring checks to the current Phase 7C technical candidate manifest.
try:
    from release_manifest import load_expected_hashes
    _current_hashes = load_expected_hashes()
    EXPECTED_CLASS_HASH = _current_hashes["classification"]
    EXPECTED_REG_HASH = _current_hashes["regression"]
    RELEASE_MANIFEST_ERROR = None
except Exception as _manifest_exc:
    EXPECTED_CLASS_HASH = "__PHASE7C_MANIFEST_BLOCKED__"
    EXPECTED_REG_HASH = "__PHASE7C_MANIFEST_BLOCKED__"
    RELEASE_MANIFEST_ERROR = str(_manifest_exc)

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def add_check(checks, gate, status, detail):
    checks.append({
        "gate": gate,
        "status": status,
        "detail": str(detail),
    })


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("EDUPREDICT NIGERIA VALIDATION — PHASE 9E")
    print("=" * 72)
    print("Purpose: Production Monitoring Dashboard / Operational Health Gate")
    print()

    checks = []
    failures = []

    required = {
        "phase9A_report": PHASE9A,
        "phase9B_report": PHASE9B,
        "phase9C_report": PHASE9C,
        "phase9D_report": PHASE9D,
        "classification_model": CLASS_MODEL,
        "regression_model": REG_MODEL,
        "production_dataset": TEST_DATA,
    }

    for name, path in required.items():
        ok = path.exists()
        add_check(checks, "required_artifact", "PASS" if ok else "FAIL",
                  f"{name}: {path}")
        if not ok:
            failures.append(f"Missing required artifact: {path}")

    if failures:
        return write_result(checks, failures, None)

    phase9a = load_json(PHASE9A)
    phase9b = load_json(PHASE9B)
    phase9c = load_json(PHASE9C)
    phase9d = load_json(PHASE9D)

    df = pd.read_csv(TEST_DATA)

    actual_session = sorted(
        str(x) for x in df["Academic_Session"].dropna().unique()
    ) if "Academic_Session" in df.columns else []

    session_ok = actual_session == EXPECTED_SESSION
    add_check(checks, "session_health", "PASS" if session_ok else "FAIL",
              f"expected={EXPECTED_SESSION}; actual={actual_session}")
    if not session_ok:
        failures.append("Production session contract failed.")

    # Phase 9D is the authoritative continuous-health contract.
    p9d_status = str(phase9d.get("status", "UNKNOWN"))
    p9d_contract = bool(
        phase9d.get("controls", {}).get("continuous_validation_contract", False)
    )
    p9d_ok = p9d_status == "PASS" and p9d_contract
    add_check(
        checks,
        "phase9D_health_contract",
        "PASS" if p9d_ok else "FAIL",
        f"status={p9d_status}; continuous_validation_contract={p9d_contract}",
    )
    if not p9d_ok:
        failures.append("Phase 9D health contract is not PASS.")

    # Frozen model identity.
    class_hash = sha256(CLASS_MODEL)
    reg_hash = sha256(REG_MODEL)

    class_ok = class_hash == EXPECTED_CLASS_HASH
    reg_ok = reg_hash == EXPECTED_REG_HASH

    add_check(
        checks, "classification_model_identity",
        "PASS" if class_ok else "FAIL",
        f"expected={EXPECTED_CLASS_HASH}; actual={class_hash}",
    )
    add_check(
        checks, "regression_model_identity",
        "PASS" if reg_ok else "FAIL",
        f"expected={EXPECTED_REG_HASH}; actual={reg_hash}",
    )

    if not class_ok:
        failures.append("Classification production model hash changed.")
    if not reg_ok:
        failures.append("Regression production model hash changed.")

    # Pull the already-validated production metrics.
    snapshot = phase9d.get("health_snapshot", {})
    data_quality = phase9d.get("data_quality", {})
    controls = phase9d.get("controls", {})

    rows = int(snapshot.get("rows", len(df)))
    warning_count = int(snapshot.get("early_warning_count", 0))
    warning_rate = float(snapshot.get("early_warning_rate", 0.0))
    risk_min = float(snapshot.get("risk_probability_min", 0.0))
    risk_max = float(snapshot.get("risk_probability_max", 0.0))
    risk_mean = float(snapshot.get("risk_probability_mean", 0.0))
    score_min = float(snapshot.get("predicted_exam_score_min", 0.0))
    score_max = float(snapshot.get("predicted_exam_score_max", 0.0))
    score_mean = float(snapshot.get("predicted_exam_score_mean", 0.0))

    metric_ranges_ok = (
        rows == len(df)
        and 0.0 <= risk_min <= risk_max <= 1.0
        and 0.0 <= warning_rate <= 1.0
        and score_min >= 0.0
        and score_max <= 100.0
        and 0 <= warning_count <= rows
    )
    add_check(
        checks,
        "production_metric_ranges",
        "PASS" if metric_ranges_ok else "FAIL",
        (
            f"rows={rows}; warning_rate={warning_rate:.6f}; "
            f"risk={risk_min:.6f}-{risk_max:.6f}; "
            f"exam_score={score_min:.2f}-{score_max:.2f}"
        ),
    )
    if not metric_ranges_ok:
        failures.append("Production monitoring metric range check failed.")

    # Historical first-term missingness is structural and remains allowed.
    structural_missing = int(data_quality.get("structural_missing_cells", 0))
    unexpected_missing = int(data_quality.get("unexpected_missing_cells", 0))
    non_historical_missing = int(
        data_quality.get("non_historical_missing_cells", 0)
    )
    duplicate_rows = int(data_quality.get("duplicate_rows", 0))
    invalid_attendance = int(data_quality.get("invalid_attendance", 0))
    invalid_numeric = int(data_quality.get("invalid_numeric_cells", 0))

    dq_ok = (
        unexpected_missing == 0
        and non_historical_missing == 0
        and duplicate_rows == 0
        and invalid_attendance == 0
        and invalid_numeric == 0
    )
    add_check(
        checks,
        "data_quality_policy",
        "PASS" if dq_ok else "FAIL",
        (
            f"structural_missing={structural_missing}; "
            f"unexpected_missing={unexpected_missing}; "
            f"non_historical_missing={non_historical_missing}; "
            f"duplicates={duplicate_rows}; invalid_attendance={invalid_attendance}; "
            f"invalid_numeric={invalid_numeric}"
        ),
    )
    if not dq_ok:
        failures.append("Production data-quality policy failed.")

    # Preserve the existing operational policy.
    policy = {
        "automatic_model_replacement": False,
        "automatic_retraining": False,
        "human_review_on_failure": True,
        "production_model_identity_locked": True,
    }

    status = "PASS" if not failures else "HEALTH_REVIEW_REQUIRED"

    dashboard = {
        "project": "EduPredict",
        "phase": "9E",
        "status": status,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Production monitoring snapshot built on the passing Phase 9D "
            "continuous-validation contract."
        ),
        "source_contract": {
            "phase9A": str(phase9a.get("status", "UNKNOWN")),
            "phase9B": str(phase9b.get("status", "UNKNOWN")),
            "phase9C": str(phase9c.get("status", "UNKNOWN")),
            "phase9D": p9d_status,
        },
        "health_snapshot": {
            "rows": rows,
            "session": actual_session,
            "warning_threshold": EXPECTED_THRESHOLD,
            "early_warning_count": warning_count,
            "early_warning_rate": warning_rate,
            "risk_probability_min": risk_min,
            "risk_probability_max": risk_max,
            "risk_probability_mean": risk_mean,
            "predicted_exam_score_min": score_min,
            "predicted_exam_score_max": score_max,
            "predicted_exam_score_mean": score_mean,
        },
        "data_quality": {
            "structural_missing_cells": structural_missing,
            "unexpected_missing_cells": unexpected_missing,
            "non_historical_missing_cells": non_historical_missing,
            "duplicate_rows": duplicate_rows,
            "invalid_attendance": invalid_attendance,
            "invalid_numeric_cells": invalid_numeric,
        },
        "model": {
            "version": "phase5_nigeria_production",
            "classification_sha256": class_hash,
            "regression_sha256": reg_hash,
            "threshold": EXPECTED_THRESHOLD,
        },
        "controls": {
            "phase9D_health_contract": p9d_ok,
            "classification_model_identity": class_ok,
            "regression_model_identity": reg_ok,
            "production_metric_ranges": metric_ranges_ok,
            "data_quality_policy": dq_ok,
            "session_health": session_ok,
        },
        "operational_policy": policy,
        "failures": failures,
        "checks": checks,
        "next_action": (
            "Continue production monitoring; investigate manually if any "
            "future Phase 9D control fails. Do not automatically retrain "
            "or replace the frozen production model."
            if status == "PASS"
            else
            "STOP automatic model changes and route the failed controls "
            "to human review."
        ),
    }

    OUT_JSON.write_text(
        json.dumps(dashboard, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["gate", "status", "detail"])
        writer.writeheader()
        writer.writerows(checks)

    print()
    print("=" * 72)
    print("PHASE 9E PRODUCTION MONITORING DASHBOARD")
    print("=" * 72)
    print(f"Phase 9D contract:       {'PASS' if p9d_ok else 'FAIL'}")
    print(f"Model identity:          {'PASS' if class_ok and reg_ok else 'FAIL'}")
    print(f"Data quality policy:     {'PASS' if dq_ok else 'FAIL'}")
    print(f"Production rows:         {rows:,}")
    print(f"Early-warning flags:    {warning_count:,}")
    print(f"Early-warning rate:     {warning_rate:.4%}")
    print(f"Risk probability:       {risk_min:.4f} - {risk_max:.4f}")
    print(f"Predicted exam score:   {score_min:.2f} - {score_max:.2f}")
    print()
    print(f"PHASE 9E STATUS: {status}")
    print()
    print(f"Dashboard: {OUT_JSON}")
    print(f"Checks:    {OUT_CSV}")
    print("=" * 72)

    return 0 if status == "PASS" else 1


def write_result(checks, failures, dashboard):
    status = "HEALTH_REVIEW_REQUIRED"
    payload = {
        "project": "EduPredict",
        "phase": "9E",
        "status": status,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "failures": failures,
        "checks": checks,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["gate", "status", "detail"])
        writer.writeheader()
        writer.writerows(checks)
    print(f"PHASE 9E STATUS: {status}")
    print(f"Dashboard: {OUT_JSON}")
    print(f"Checks:    {OUT_CSV}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
