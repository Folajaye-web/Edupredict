import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(r"C:\Users\l2e\OneDrive\Desktop\EduPredict_MASTER_FINAL")
REPORT_DIR = ROOT / "reports" / "nigeria_validation" / "phase10"
PHASE9_DIR = ROOT / "reports" / "nigeria_validation" / "phase9"
MODEL_DIR = ROOT / "models" / "nigeria_validation"
TASK_NAME = "EduPredict-Phase9G-ContinuousValidation"

EXPECTED_HASHES = {
    "classification": (
        MODEL_DIR / "phase5_classification_hgb.joblib",
        "3cc65dbd4c0e48ba65b28f713a8d32d9fec8bf0631dc8a3b5ef15e00823f608a",
    ),
    "regression": (
        MODEL_DIR / "phase5_regression_hgb.joblib",
        "f61b02c5542aa0a4d2d0c1b18540f05193cc5d0c5f1f4e5a46ae5a75e0a9973e",
    ),
}

checks = []

def add(name, passed, detail):
    checks.append({
        "check": name,
        "status": "PASS" if passed else "FAIL",
        "detail": str(detail),
    })

def read_json(path):
    try:
        with path.open("r", encoding="utf-8") as f:
            value = json.load(f)
        if not isinstance(value, dict):
            raise ValueError("JSON root is not an object")
        return value, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

# 1. Project and model artifact integrity
add("project_root_exists", ROOT.is_dir(), ROOT)
for name, (path, expected) in EXPECTED_HASHES.items():
    exists = path.is_file()
    add(f"{name}_model_exists", exists, path)
    if exists:
        actual = sha256(path)
        add(
            f"{name}_model_hash",
            actual == expected,
            f"expected={expected}; actual={actual}",
        )

# 2. Phase 9G report and stage evidence
p9_path = PHASE9_DIR / "phase9G_continuous_production_orchestrator.json"
p9, p9_error = read_json(p9_path)
add("phase9G_report_exists_and_readable", p9 is not None,
    p9_path if p9 is not None else f"{p9_path}; {p9_error}")

if p9 is not None:
    add("phase9G_overall_status", p9.get("status") == "PASS",
        f"status={p9.get('status')}")
    expected_stages = {"phase9A", "phase9B", "phase9C", "phase9D", "phase9E"}
    executed = set(p9.get("stages_executed", []))
    add("phase9G_expected_stages_executed",
        executed == expected_stages,
        f"expected={sorted(expected_stages)}; actual={sorted(executed)}")

    stages = p9.get("stages", [])
    stage_results = {
        item.get("stage"): item
        for item in stages if isinstance(item, dict)
    }
    all_stage_reports_pass = True
    stage_details = []
    for stage in sorted(expected_stages):
        item = stage_results.get(stage)
        if item is None:
            all_stage_reports_pass = False
            stage_details.append(f"{stage}=MISSING")
            continue
        ok = (
            item.get("status") == "PASS"
            and item.get("report_status") == "PASS"
            and item.get("return_code") == 0
        )
        if not ok:
            all_stage_reports_pass = False
        stage_details.append(
            f"{stage}: status={item.get('status')}, "
            f"report_status={item.get('report_status')}, "
            f"return_code={item.get('return_code')}"
        )
    add("phase9G_individual_stage_results",
        all_stage_reports_pass, "; ".join(stage_details))

    policy = p9.get("operational_policy", {})
    add("automatic_model_replacement_disabled",
        policy.get("automatic_model_replacement") is False,
        policy.get("automatic_model_replacement"))
    add("automatic_retraining_disabled",
        policy.get("automatic_retraining") is False,
        policy.get("automatic_retraining"))
    add("stop_on_failed_gate_enabled",
        policy.get("stop_on_failed_gate") is True,
        policy.get("stop_on_failed_gate"))
    add("human_review_on_failure_enabled",
        policy.get("human_review_on_failure") is True,
        policy.get("human_review_on_failure"))
    add("production_model_identity_locked",
        policy.get("production_model_identity_locked") is True,
        policy.get("production_model_identity_locked"))

# 3. Phase 10A failure-injection evidence
p10a_path = REPORT_DIR / "phase10A_failure_injection_recovery.json"
p10a, p10a_error = read_json(p10a_path)
add("phase10A_report_exists_and_readable", p10a is not None,
    p10a_path if p10a is not None else f"{p10a_path}; {p10a_error}")

if p10a is not None:
    add("phase10A_overall_status", p10a.get("status") == "PASS",
        p10a.get("status"))
    add("phase10A_all_tests_passed",
        p10a.get("tests_passed") == 9 and p10a.get("tests_failed") == 0,
        f"passed={p10a.get('tests_passed')}; failed={p10a.get('tests_failed')}")
    add("phase10A_production_database_unchanged",
        p10a.get("production_database_modified") is False,
        p10a.get("production_database_modified"))
    add("phase10A_production_models_unchanged",
        p10a.get("production_models_modified") is False,
        p10a.get("production_models_modified"))

# 4. Inspect scheduled task without starting it or changing its settings
ps_command = (
    "$ErrorActionPreference='Stop'; "
    "$t=Get-ScheduledTask -TaskName '" + TASK_NAME + "'; "
    "$i=Get-ScheduledTaskInfo -InputObject $t; "
    "$a=@($t.Actions | ForEach-Object { "
    "[pscustomobject]@{Execute=$_.Execute; Arguments=$_.Arguments; "
    "WorkingDirectory=$_.WorkingDirectory} }); "
    "[pscustomobject]@{TaskName=$t.TaskName; TaskPath=$t.TaskPath; "
    "State=[string]$t.State; LastRunTime=$i.LastRunTime.ToString('o'); "
    "LastTaskResult=$i.LastTaskResult; "
    "NextRunTime=$(if($i.NextRunTime){$i.NextRunTime.ToString('o')}else{$null}); "
    "Actions=$a} | ConvertTo-Json -Depth 5 -Compress"
)
try:
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive",
         "-ExecutionPolicy", "Bypass", "-Command", ps_command],
        capture_output=True, text=True, timeout=30
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    task = json.loads(result.stdout.strip().splitlines()[-1])
    if isinstance(task, list):
        task = task[0]
    add("scheduled_task_found", task.get("TaskName") == TASK_NAME, task.get("TaskName"))
    add("scheduled_task_enabled",
        task.get("State") not in ("Disabled", "Unknown"),
        task.get("State"))
    actions = task.get("Actions") or []
    if isinstance(actions, dict):
        actions = [actions]
    expected_python = str(
        Path(r"C:\Users\l2e\OneDrive\Desktop\.venv\Scripts\python.exe")
    ).lower()
    expected_script = str(
        ROOT / "scripts" / "nigeria_validation" /
        "phase9G_continuous_production_orchestrator.py"
    ).lower()
    matching_action = any(
        str(a.get("Execute", "")).strip('"').lower() == expected_python
        and expected_script in str(a.get("Arguments", "")).lower()
        and str(a.get("WorkingDirectory", "")).strip('"').lower() == str(ROOT).lower()
        for a in actions
    )
    add("scheduled_task_action_matches_project",
        matching_action, json.dumps(actions))
    add("scheduled_task_last_run_success",
        task.get("LastTaskResult") == 0,
        f"last_result={task.get('LastTaskResult')}; last_run={task.get('LastRunTime')}")
except Exception as exc:
    add("scheduled_task_inspection", False,
        f"{type(exc).__name__}: {exc}")

# 5. Save evidence
failed = [c for c in checks if c["status"] != "PASS"]
report = {
    "project": "EduPredict",
    "phase": "10B",
    "status": "PASS" if not failed else "FAIL",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "purpose": "Read-only production readiness and operational evidence audit",
    "production_database_modified": False,
    "production_models_modified": False,
    "automatic_retraining_performed": False,
    "scheduled_task_triggered": False,
    "checks_passed": len(checks) - len(failed),
    "checks_failed": len(failed),
    "checks": checks,
    "limitations": [
        "This audit verifies local artifacts, recorded reports and task configuration only.",
        "It does not establish live-school performance, fairness, privacy compliance or unrestricted production approval.",
        "A scheduled task's last successful result does not guarantee future runs will succeed."
    ],
}
REPORT_DIR.mkdir(parents=True, exist_ok=True)
json_path = REPORT_DIR / "phase10B_production_readiness_audit.json"
csv_path = REPORT_DIR / "phase10B_production_readiness_audit.csv"
json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=["check", "status", "detail"])
    writer.writeheader()
    writer.writerows(checks)

print("=" * 72)
print("EDUPREDICT PHASE 10B — PRODUCTION READINESS AUDIT")
print("=" * 72)
for item in checks:
    print(f"{item['status']:4}  {item['check']}: {item['detail']}")
print("-" * 72)
print(f"STATUS: {report['status']}")
print(f"Checks passed: {report['checks_passed']}")
print(f"Checks failed: {report['checks_failed']}")
print(f"Database modified: {report['production_database_modified']}")
print(f"Models modified: {report['production_models_modified']}")
print(f"JSON report: {json_path}")
print(f"CSV report:  {csv_path}")
sys.exit(0 if report["status"] == "PASS" else 1)
