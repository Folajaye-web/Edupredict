"""One-command, backup-first EduPredict model-pipeline revalidation.

This rebuilds the historical features and candidate models, then runs the
quality gates in order. It never imports data into the live SQLite database,
never retrains a model automatically from the app, and never authorises a pilot.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASET = Path(r"C:\Users\l2e\OneDrive\Desktop\Nigeria_Student_Records_2023_2026_Combined.csv")
REPORT_ROOT = ROOT / "reports" / "nigeria_validation"
PHASE10_DIR = REPORT_ROOT / "phase10"
REPORT_PATH = PHASE10_DIR / "phase10E_integrated_revalidation.json"
EXPECTED_SKLEARN = "1.9.0"

CORE_STAGES = [
    ("dataset_audit", "audit_dataset.py", []),
    ("phase10D_onboarding_validation", "phase10D_school_onboarding.py", ["--input", str(DATASET)]),
    ("feature_build", "build_features.py", []),
    ("temporal_splits", "create_temporal_splits.py", []),
    ("phase5_holdout", "phase5_final_holdout_validation.py", []),
    ("phase6_calibration", "phase6_stability_calibration.py", []),
    ("phase7_qualification", "phase7_production_qualification.py", []),
    ("phase7A_artifact_gate", "phase7A_artifact_gate.py", []),
    ("phase7B_subgroup_gate", "phase7B_subgroup_safety_gate.py", []),
    ("phase7C_technical_release_gate", "phase7C_production_release_gate.py", []),
]
POST_RELEASE_STAGES = [
    ("phase8A_integration", "phase8A_production_integration.py", []),
    ("phase8B_app_model_validation", "phase8B_application_model_validation.py", []),
    ("phase8C_end_to_end", "phase8C_end_to_end_operational_validation.py", []),
    ("phase9G_continuous_validation", "phase9G_continuous_production_orchestrator.py", []),
    ("phase10B_readiness", "phase10B_production_readiness_audit.py", []),
    ("phase10C_pilot_readiness", "phase10C_controlled_pilot_readiness.py", []),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def backup_existing() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = ROOT / "backups" / f"phase10E_revalidation_{stamp}"
    destination.mkdir(parents=True, exist_ok=False)
    for relative in [
        Path("models") / "nigeria_validation",
        Path("data") / "nigeria_validation",
        Path("reports") / "nigeria_validation",
    ]:
        source = ROOT / relative
        if source.exists():
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(source, target, dirs_exist_ok=True)
            else:
                shutil.copy2(source, target)
    return destination


def run_stage(name: str, script_name: str, args: list[str], logs_dir: Path):
    script = ROOT / "scripts" / "nigeria_validation" / script_name
    record = {
        "stage": name,
        "script": str(script),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    if not script.is_file():
        record.update({"status": "FAIL", "return_code": None, "stderr": f"Missing script: {script}"})
        return record
    env = None
    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    try:
        completed = subprocess.run(
            [sys.executable, str(script), *args], cwd=ROOT, env=env,
            capture_output=True, text=True, timeout=3600, check=False,
        )
        record["return_code"] = completed.returncode
        record["stdout_tail"] = completed.stdout[-8000:]
        record["stderr_tail"] = completed.stderr[-8000:]
        record["status"] = "PASS" if completed.returncode == 0 else "FAIL"
    except subprocess.TimeoutExpired as exc:
        record.update({"status": "TIMEOUT", "return_code": None, "stderr_tail": str(exc)})
    record["completed_utc"] = datetime.now(timezone.utc).isoformat()
    logs_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    (logs_dir / f"{safe_name}.log").write_text(
        "STDOUT\n" + record.get("stdout_tail", "") + "\nSTDERR\n" + record.get("stderr_tail", ""),
        encoding="utf-8",
    )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-rebuild", action="store_true", help="Required: back up and rebuild candidate features/models/reports.")
    args = parser.parse_args()
    PHASE10_DIR.mkdir(parents=True, exist_ok=True)

    if not args.confirm_rebuild:
        print("No changes made. Re-run with --confirm-rebuild to back up and rebuild candidate features/models.")
        return 2
    if not DATASET.is_file():
        raise FileNotFoundError(f"Historical CSV not found: {DATASET}")
    import sklearn
    if sklearn.__version__ != EXPECTED_SKLEARN:
        raise RuntimeError(f"Use the project venv with scikit-learn {EXPECTED_SKLEARN}; current={sklearn.__version__}.")

    backup_path = backup_existing()
    logs_dir = PHASE10_DIR / "logs"
    records = []
    status = "RUNNING"
    stop_reason = None

    for name, script, stage_args in CORE_STAGES:
        record = run_stage(name, script, stage_args, logs_dir)
        records.append(record)
        print(f"[{record['status']}] {name}")
        if name == "phase7C_technical_release_gate":
            gate = read_json(REPORT_ROOT / "phase7C_production_release_gate.json")
            if not gate or gate.get("status") != "TECHNICAL_RELEASE_READY":
                status = "BLOCKED_PHASE7_REVIEW"
                stop_reason = (
                    "Phase 7C did not clear technical release. Phase 8/9/10 downstream stages were intentionally not run. "
                    "Review the Phase 7C report and calibration/qualification evidence."
                )
                # A gate block is a valid stopping condition, not a runner defect.
                break
            if record["status"] != "PASS":
                status = "FAILED_CORE_STAGE"
                stop_reason = f"{name} reported technical release ready but the process returned an error. See {logs_dir}."
                break
        elif record["status"] != "PASS":
            status = "FAILED_CORE_STAGE"
            stop_reason = f"{name} did not pass; downstream stages were not run. See {logs_dir}."
            break

    if status == "RUNNING":
        governance_stage_status = {}
        for name, script, stage_args in POST_RELEASE_STAGES:
            record = run_stage(name, script, stage_args, logs_dir)
            records.append(record)
            print(f"[{record['status']}] {name}")
            gate_name = {
                "phase10B_readiness": "phase10B_production_readiness_audit.json",
                "phase10C_pilot_readiness": "phase10C_controlled_pilot_readiness.json",
            }.get(name)
            if gate_name:
                report = read_json(PHASE10_DIR / gate_name)
                governance_stage_status[name] = report.get("status") if report else "MISSING_REPORT"
            # Continue after Phase 10B/10C blocks: these are expected human
            # approval gates and must be reported rather than hidden.
            if name.startswith("phase8") or name.startswith("phase9"):
                if record["status"] != "PASS":
                    status = "FAILED_INTEGRATION_STAGE"
                    stop_reason = f"{name} failed; see its log and report before proceeding."
                    break
        if status == "RUNNING":
            p10c = read_json(PHASE10_DIR / "phase10C_controlled_pilot_readiness.json")
            status = "TECHNICAL_VALIDATION_COMPLETE_GOVERNANCE_BLOCKED"
            stop_reason = "Technical validation stages ran. Phase 10B/10C approval and governance checks remain separate gates; the runner never authorises a pilot."

    report = {
        "project": "EduPredict",
        "phase": "10E",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "purpose": "Single backup-first revalidation workflow to align the historical feature pipeline and dependent quality gates.",
        "dataset": str(DATASET),
        "dataset_sha256": sha256(DATASET),
        "feature_pipeline_version_expected": "2.0-period-safe-school-aware",
        "runtime_scikit_learn": sklearn.__version__,
        "backup_directory": str(backup_path),
        "stages": records,
        "stop_reason": stop_reason,
        "database_modified": False,
        "production_model_directory_modified": False,
        "candidate_model_directory_rebuilt": any(x["stage"] == "phase5_holdout" and x["status"] == "PASS" for x in records),
        "school_file_import_performed": False,
        "automatic_model_replacement": False,
        "automatic_pilot_authorisation": False,
        "next_step": (
            "Resolve the Phase 7 calibration/qualification review before Phase 8/9 integration."
            if status == "BLOCKED_PHASE7_REVIEW" else
            "Review Phase 10B/10C evidence and obtain explicit human governance approval before any pilot."
            if status == "TECHNICAL_VALIDATION_COMPLETE_GOVERNANCE_BLOCKED" else
            "Correct the failed stage using its saved log, then rerun this same workflow."
        ),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n" + "=" * 72)
    print(f"PHASE 10E RESULT: {status}")
    print(f"Backup: {backup_path}")
    print(f"Report: {REPORT_PATH}")
    print(f"Next step: {report['next_step']}")
    print("Database modified: False | Pilot authorised: False")
    print("=" * 72)
    return 0 if status == "TECHNICAL_VALIDATION_COMPLETE_GOVERNANCE_BLOCKED" else 2


if __name__ == "__main__":
    sys.exit(main())
