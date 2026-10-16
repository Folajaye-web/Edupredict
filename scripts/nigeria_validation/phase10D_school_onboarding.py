import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(r"C:\Users\l2e\OneDrive\Desktop\EduPredict_MASTER_FINAL")
REPORT_DIR = ROOT / "reports" / "nigeria_validation" / "phase10"
REPORT_PATH = REPORT_DIR / "phase10D_school_onboarding.json"

# Baseline results/assessment template. A school with another format
# should map its fields to this template before any production import.
REQUIRED_COLUMNS = [
    "Student_ID", "School", "Academic_Session", "Term",
    "Class", "Subject", "CA1", "CA2", "Assignment",
    "Project", "Exam", "Total", "Grade",
]

SCORE_LIMITS = {"CA1": 10, "CA2": 10, "Assignment": 10, "Project": 10, "Exam": 60, "Total": 100}
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser(
        description="Read-only EduPredict school-file onboarding validation."
    )
    parser.add_argument(
        "--input",
        help="Path to a school CSV or Excel file. Omit for a readiness-only run."
    )
    parser.add_argument(
        "--sheet",
        default=None,
        help="Optional Excel worksheet name. Defaults to the first worksheet."
    )
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    checks = []
    issues = []
    source = None
    df = None

    def check(name, passed, detail):
        checks.append({
            "check": name,
            "status": "PASS" if passed else "FAIL",
            "detail": str(detail),
        })

    check("project_root", ROOT.is_dir(), ROOT)
    check("phase10C_report_exists",
          (REPORT_DIR / "phase10C_controlled_pilot_readiness.json").is_file(),
          REPORT_DIR / "phase10C_controlled_pilot_readiness.json")

    if not args.input:
        report = {
            "project": "EduPredict",
            "phase": "10D",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "status": "PENDING_INPUT",
            "purpose": "Read-only school-file onboarding and data-quality validation",
            "input_supplied": False,
            "input_file": None,
            "file_sha256": None,
            "rows": None,
            "columns": [],
            "checks": checks,
            "issues": [
                "No school file supplied. Run again with --input and a CSV or Excel file."
            ],
            "database_modified": False,
            "models_modified": False,
            "automatic_retraining": False,
            "automatic_model_replacement": False,
            "import_performed": False,
            "pilot_authorised": False,
            "limitations": [
                "This run does not validate school records because no file was supplied.",
                "Passing file checks does not establish record authenticity or legal authority to process data.",
                "No records are imported into the EduPredict database."
            ],
        }
        REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0

    source = Path(args.input).expanduser().resolve()
    if not source.is_file():
        issues.append(f"Input file not found: {source}")
    elif source.suffix.lower() not in ALLOWED_EXTENSIONS:
        issues.append(f"Unsupported file type: {source.suffix}")
    else:
        try:
            if source.suffix.lower() == ".csv":
                df = pd.read_csv(source, low_memory=False)
            else:
                df = pd.read_excel(source, sheet_name=args.sheet or 0)
        except Exception as exc:
            issues.append(f"Could not read input file: {type(exc).__name__}: {exc}")

    file_hash = sha256(source) if source and source.is_file() else None
    missing_columns = []
    duplicate_count = 0
    missing_required_cells = {}
    invalid_scores = {}
    total_mismatch_count = None

    if df is not None:
        # Normalize header whitespace only; do not silently rename columns.
        df.columns = [str(c).strip() for c in df.columns]
        duplicate_headers = df.columns[df.columns.duplicated()].tolist()
        check("non_empty_file", len(df) > 0, f"rows={len(df)}")
        check("unique_column_headers", not duplicate_headers, duplicate_headers or "none")

        missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        check("required_columns", not missing_columns,
              f"missing={missing_columns}" if missing_columns else "all present")

        if "Student_ID" in df.columns and "Subject" in df.columns:
            keys = [c for c in
                    ["School", "Academic_Session", "Term", "Student_ID", "Subject"]
                    if c in df.columns]
            if len(keys) >= 3:
                duplicate_count = int(df.duplicated(subset=keys, keep=False).sum())
                check("duplicate_result_keys", duplicate_count == 0,
                      f"rows participating in duplicate keys={duplicate_count}; keys={keys}")

        required_present = [c for c in REQUIRED_COLUMNS if c in df.columns]
        for col in required_present:
            missing_required_cells[col] = int(df[col].isna().sum())
        missing_cells_total = sum(missing_required_cells.values())
        check("required_cells_complete", missing_cells_total == 0,
              f"missing required cells={missing_cells_total}")

        for col, upper_limit in SCORE_LIMITS.items():
            if col not in df.columns:
                continue
            raw_values = df[col]
            values = pd.to_numeric(raw_values, errors="coerce")
            non_numeric = int((raw_values.notna() & values.isna()).sum())
            out_of_range = int((values.notna() & ~values.between(0, upper_limit)).sum())
            invalid = non_numeric + out_of_range
            invalid_scores[col] = invalid
            check(f"{col}_range_0_to_{upper_limit}", invalid == 0,
                  f"non-numeric={non_numeric}; out-of-range={out_of_range}; expected scale=0-{upper_limit}")

        if all(c in df.columns for c in ["CA1", "CA2", "Assignment", "Project", "Exam", "Total"]):
            numeric = df[["CA1", "CA2", "Assignment", "Project", "Exam", "Total"]].apply(
                pd.to_numeric, errors="coerce"
            )
            complete = numeric.notna().all(axis=1)
            expected_total = numeric[["CA1", "CA2", "Assignment", "Project", "Exam"]].sum(axis=1)
            total_mismatch_count = int(
                (complete & ((expected_total - numeric["Total"]).abs() > 0.01)).sum()
            )
            check("total_matches_component_sum", total_mismatch_count == 0,
                  f"mismatches={total_mismatch_count}; confirm school scoring rules before use")

        for col in ["Student_ID", "School", "Academic_Session", "Term", "Class", "Subject"]:
            if col in df.columns:
                blank = int(df[col].astype("string").str.strip().eq("").fillna(False).sum())
                check(f"{col}_not_blank", blank == 0, f"blank values={blank}")

        if missing_columns:
            issues.append(f"Missing required columns: {missing_columns}")
        if duplicate_count:
            issues.append(f"Duplicate result keys detected in {duplicate_count} rows.")
        if missing_cells_total:
            issues.append(f"Missing required cells detected: {missing_required_cells}")
        if any(invalid_scores.values()):
            issues.append(f"Score range problems detected: {invalid_scores}")
        if total_mismatch_count:
            issues.append(f"Total calculation mismatches: {total_mismatch_count}")

    failed_checks = [c for c in checks if c["status"] == "FAIL"]
    if df is None:
        status = "NEEDS_CORRECTION"
    elif failed_checks or issues:
        status = "NEEDS_CORRECTION"
    else:
        status = "PASS"

    report = {
        "project": "EduPredict",
        "phase": "10D",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "purpose": "Read-only school-file onboarding and data-quality validation",
        "input_supplied": True,
        "input_file": str(source) if source else None,
        "file_sha256": file_hash,
        "rows": len(df) if df is not None else None,
        "columns": list(df.columns) if df is not None else [],
        "required_columns": REQUIRED_COLUMNS,
        "missing_columns": missing_columns,
        "missing_required_cells": missing_required_cells,
        "invalid_score_counts": invalid_scores,
        "duplicate_key_rows": duplicate_count,
        "total_mismatch_rows": total_mismatch_count,
        "checks": checks,
        "issues": issues,
        "database_modified": False,
        "models_modified": False,
        "automatic_retraining": False,
        "automatic_model_replacement": False,
        "import_performed": False,
        "pilot_authorised": False,
        "limitations": [
            "This is a read-only validation; no records are imported.",
            "The Phase 5 model was trained for CA1/CA2/Assignment/Project on 0-10, Exam on 0-60 and Total on 0-100. Confirm the school's scoring scheme before use.",
            "Grade labels and promotion decisions are not independently validated by this initial script.",
            "Passing checks does not prove the source records are authentic, that a school approved processing, or that the model is suitable for live use.",
            "Use only data you are authorised to access; avoid including unnecessary personal data in reports."
        ],
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("=" * 68)
    print("EDUPREDICT PHASE 10D — SCHOOL ONBOARDING VALIDATION")
    print("=" * 68)
    print(f"Status: {status}")
    print(f"Input: {source}")
    print(f"Rows: {len(df) if df is not None else 'not read'}")
    print(f"Failed checks: {len(failed_checks)}")
    for issue in issues:
        print(f"ISSUE: {issue}")
    print(f"Report: {REPORT_PATH}")
    print("Database modified: False")
    print("Models modified: False")
    print("Import performed: False")
    print("Pilot authorised: False")
    print("=" * 68)
    return 1 if status == "NEEDS_CORRECTION" else 0

if __name__ == "__main__":
    sys.exit(main())

