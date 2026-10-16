import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(r"C:\Users\l2e\OneDrive\Desktop\EduPredict_MASTER_FINAL")
REPORTS = ROOT / "reports" / "nigeria_validation"
PHASE9 = REPORTS / "phase9"
PHASE10 = REPORTS / "phase10"
MODELS = ROOT / "models" / "nigeria_validation"

EXPECTED_MODELS = {
    "classification": (
        MODELS / "phase5_classification_hgb.joblib",
        "3cc65dbd4c0e48ba65b28f713a8d32d9fec8bf0631dc8a3b5ef15e00823f608a",
    ),
    "regression": (
        MODELS / "phase5_regression_hgb.joblib",
        "f61b02c5542aa0a4d2d0c1b18540f05193cc5d0c5f1f4e5a46ae5a75e0a9973e",
    ),
}

REQUIRED_CONTROLS = [
    ("pilot_scope_approved", "Written pilot scope and participating-school approval"),
    ("data_processing_authorised", "Documented lawful basis, privacy notice and data-processing responsibilities"),
    ("data_access_reviewed", "Approved user roles, least-privilege access and access-review procedure"),
    ("retention_and_deletion_defined", "Documented retention, deletion and backup arrangements"),
    ("human_review_procedure_approved", "Named staff responsible for reviewing and challenging warnings"),
    ("consequential_decision_safeguard", "Written prohibition on using predictions alone for promotion, punishment or exclusion"),
    ("fairness_evaluation_approved", "Evaluation plan for relevant groups with suitable sample-size and uncertainty checks"),
    ("real_world_metrics_defined", "Pre-agreed accuracy, calibration, false-positive and false-negative monitoring measures"),
    ("intervention_workflow_defined", "Documented support actions, responsible staff and follow-up recording"),
    ("incident_and_appeal_process_approved", "Documented correction, complaint, appeal and incident-response process"),
    ("rollback_and_shutdown_tested", "Tested procedure to suspend pilot predictions and return to normal school processes"),
    ("staff_training_completed", "Evidence that pilot staff understand limitations and responsible use"),
    ("pilot_success_criteria_approved", "Written acceptance thresholds, review date and named approval authority"),
]

CHECKLIST_PATH = PHASE10 / "phase10C_pilot_governance_checklist.csv"
REPORT_PATH = PHASE10 / "phase10C_controlled_pilot_readiness.json"

FIELDS = [
    "control", "status", "evidence_required",
    "evidence_reference", "reviewer", "review_date", "notes",
]

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def read_json(path):
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None

def load_checklist():
    """Preserve existing evidence entries; never infer approval."""
    old = {}
    if CHECKLIST_PATH.is_file():
        try:
            with CHECKLIST_PATH.open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                for row in csv.DictReader(handle):
                    key = row.get("control", "")
                    if key:
                        old[key] = row
        except (OSError, csv.Error):
            print("WARNING: Existing checklist could not be read; "
                  "controls will remain unverified.")

    rows = []
    for control, required in REQUIRED_CONTROLS:
        prior = old.get(control, {})
        status = str(prior.get("status", "NOT_VERIFIED")).strip().upper()

        # Only explicitly VERIFIED rows with all evidence metadata
        # qualify as verified. No other status is accepted.
        reference = str(prior.get("evidence_reference", "")).strip()
        reviewer = str(prior.get("reviewer", "")).strip()
        review_date = str(prior.get("review_date", "")).strip()

        if not (
            status == "VERIFIED"
            and reference and reviewer and review_date
        ):
            status = "NOT_VERIFIED"

        rows.append({
            "control": control,
            "status": status,
            "evidence_required": required,
            "evidence_reference": reference,
            "reviewer": reviewer,
            "review_date": review_date,
            "notes": str(prior.get("notes", "")).strip(),
        })
    return rows

def main():
    PHASE10.mkdir(parents=True, exist_ok=True)
    checks = []

    def add_check(name, passed, detail):
        checks.append({
            "check": name,
            "status": "PASS" if passed else "FAIL",
            "detail": str(detail),
        })

    # Technical checks are independent of school onboarding approvals.
    add_check("project_root", ROOT.is_dir(), ROOT)

    for name, (path, expected_hash) in EXPECTED_MODELS.items():
        exists = path.is_file()
        add_check(f"{name}_model_exists", exists, path)
        if exists:
            actual_hash = sha256(path)
            add_check(
                f"{name}_model_identity",
                actual_hash == expected_hash,
                f"expected={expected_hash}; actual={actual_hash}",
            )

    phase9_path = PHASE9 / "phase9G_continuous_production_orchestrator.json"
    phase9 = read_json(phase9_path)
    current_hashes = {}
    for name, (model_path, _approved_hash) in EXPECTED_MODELS.items():
        if model_path.is_file():
            current_hashes[name] = sha256(model_path)
    phase9a_path = PHASE9 / "phase9A_production_prediction_service.json"
    phase9a = read_json(phase9a_path)
    phase9a_metadata = phase9a.get("metadata", {}) if phase9a else {}
    phase9_hashes_current = bool(
        current_hashes.get("classification")
        and current_hashes.get("regression")
        and phase9a_metadata.get("classification_sha256") == current_hashes["classification"]
        and phase9a_metadata.get("regression_sha256") == current_hashes["regression"]
    )
    add_check(
        "phase9G_evidence",
        bool(phase9 and phase9.get("status") == "PASS" and phase9_hashes_current),
        f"{phase9_path}; status={phase9.get('status') if phase9 else 'missing'}; phase9A_current_hashes={phase9_hashes_current}",
    )

    feature_report_path = REPORTS / "feature_build_report.json"
    feature_report = read_json(feature_report_path)
    add_check(
        "corrected_feature_pipeline",
        bool(feature_report and feature_report.get("status") == "PASS" and feature_report.get("feature_pipeline_version") == "2.0-period-safe-school-aware"),
        f"{feature_report_path}; version={feature_report.get('feature_pipeline_version') if feature_report else 'missing'}",
    )

    phase10b_path = PHASE10 / "phase10B_production_readiness_audit.json"
    phase10b = read_json(phase10b_path)
    add_check(
        "phase10B_evidence",
        bool(
            phase10b
            and phase10b.get("status") == "PASS"
            and phase10b.get("checks_failed") == 0
        ),
        phase10b_path,
    )

    technical_failures = [
        item for item in checks if item["status"] != "PASS"
    ]
    development_status = (
        "READY" if not technical_failures else "NEEDS_ATTENTION"
    )

    checklist = load_checklist()
    with CHECKLIST_PATH.open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(checklist)

    verified = sum(row["status"] == "VERIFIED" for row in checklist)
    unverified = len(checklist) - verified

    governance_status = (
        "READY_FOR_AUTHORISED_REVIEW"
        if unverified == 0
        else "PENDING_EVIDENCE"
    )

    # This script never grants live-pilot authorisation.
    # That requires a separate, documented human decision.
    report = {
        "project": "EduPredict",
        "phase": "10C",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Separate technical development readiness, school onboarding, "
            "governance evidence and live-pilot authorisation"
        ),
        "status": (
            "FAIL" if technical_failures
            else "BLOCKED" if unverified
            else "READY_FOR_AUTHORISED_REVIEW"
        ),
        "development_status": development_status,
        "school_onboarding_status": "PENDING_SCHOOL_ENGAGEMENT",
        "governance_status": governance_status,
        "pilot_authorisation_status": "NOT_AUTHORISED",
        "pilot_authorised": False,
        "technical_checks_passed": len(checks) - len(technical_failures),
        "technical_checks_failed": len(technical_failures),
        "technical_checks": checks,
        "governance_controls_total": len(checklist),
        "governance_controls_verified": verified,
        "governance_controls_unverified": unverified,
        "governance_blockers": [
            row["control"] for row in checklist
            if row["status"] != "VERIFIED"
        ],
        "governance_checklist_csv": str(CHECKLIST_PATH),
        "production_database_modified": False,
        "production_models_modified": False,
        "automatic_model_replacement": False,
        "automatic_retraining": False,
        "automatic_retraining_performed": False,
        "scheduled_task_triggered": False,
        "limitations": [
            "Development readiness is not live-school or production authorisation.",
            "School onboarding remains pending until applicable school-specific "
            "requirements are supplied and reviewed.",
            "A governance row requires VERIFIED status, an evidence reference, "
            "reviewer and review date; this script does not validate the "
            "substance or authenticity of evidence.",
            "This script never authorises a live pilot automatically.",
            "Do not place student names, IDs or other personal data in the "
            "governance checklist.",
            "Predictions remain advisory and must not independently determine "
            "promotion, punishment, exclusion or other consequential outcomes.",
        ],
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print("=" * 68)
    print("EDUPREDICT PHASE 10C — SEPARATED READINESS STATUS")
    print("=" * 68)
    for item in checks:
        print(f"{item['status']:4}  {item['check']}: {item['detail']}")
    print()
    print(f"Development status: {development_status}")
    print(f"School onboarding: {report['school_onboarding_status']}")
    print(f"Governance status: {governance_status}")
    print(f"Governance verified: {verified}/{len(checklist)}")
    print(f"Pilot authorisation: {report['pilot_authorisation_status']}")
    print(f"Technical checks failed: {len(technical_failures)}")
    print(f"JSON report: {REPORT_PATH}")
    print(f"Governance checklist: {CHECKLIST_PATH}")
    print("=" * 68)

    # Missing governance evidence does not count as a technical failure.
    return 1 if technical_failures else 0

if __name__ == "__main__":
    sys.exit(main())
