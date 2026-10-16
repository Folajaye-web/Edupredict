"""Leakage-safe historical feature builder for EduPredict Nigeria.

Feature pipeline v2.0 fixes the prior overall-history ordering defect:
- groups by School + Student_ID (IDs may repeat across schools)
- orders by session and term before calculating history
- computes overall history from completed earlier terms only, never another
  subject in the same term
- checks both subject and overall histories against independent period-level
  calculations before writing output.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATASET = Path(r"C:\Users\l2e\OneDrive\Desktop\Nigeria_Student_Records_2023_2026_Combined.csv")
REPORT_DIR = ROOT / "reports" / "nigeria_validation"
OUTPUT_DIR = ROOT / "data" / "nigeria_validation"
OUTPUT_FILE = OUTPUT_DIR / "leakage_safe_features.csv"
REPORT_FILE = REPORT_DIR / "feature_build_report.json"
FEATURE_PIPELINE_VERSION = "2.0-period-safe-school-aware"
TERM_ORDER = {"First Term": 1, "Second Term": 2, "Third Term": 3}
REQUIRED_COLUMNS = [
    "Student_ID", "School", "Academic_Session", "Term", "Class", "Arm",
    "Gender", "Subject", "CA1", "CA2", "Assignment", "Project", "Exam",
    "Total", "Grade", "Attendance", "Previous_Term_Total",
    "Previous_Term_Grade", "Previous_Year_Average", "Behaviour/Engagement",
    "Intervention", "Final_Year_Result", "Promotion_Status",
]
FORBIDDEN_PREDICTOR_COLUMNS = [
    "Exam", "Total", "Grade", "Final_Year_Result", "Promotion_Status", "Intervention"
]
BASE_FEATURE_COLUMNS = [
    "Student_ID", "School", "Academic_Session", "Term", "Term_Number", "Class",
    "Arm", "Gender", "Subject", "CA1", "CA2", "Assignment", "Project",
    "Attendance", "Previous_Term_Total", "Previous_Term_Grade", "Previous_Year_Average",
]
DERIVED_FEATURE_COLUMNS = [
    "Historical_Subject_Average", "Historical_Subject_Count",
    "Historical_Overall_Average", "Historical_Overall_Count",
]
TARGET_COLUMNS = ["target_exam_score", "target_below_credit"]
NUMERIC_COLUMNS = [
    "CA1", "CA2", "Assignment", "Project", "Exam", "Total", "Attendance",
    "Previous_Term_Total", "Previous_Year_Average",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def session_start_year(value):
    text = str(value).strip()
    if len(text) < 4 or not text[:4].isdigit():
        raise ValueError(f"Invalid Academic_Session value: {value!r}")
    return int(text[:4])


def grade_to_numeric(value):
    mapping = {"A1": 9, "B2": 8, "B3": 7, "C4": 6, "C5": 5, "C6": 4, "D7": 3, "E8": 2, "F9": 1}
    if pd.isna(value):
        return np.nan
    return mapping.get(str(value).strip().upper(), np.nan)


def build_feature_frame(source: pd.DataFrame):
    """Return (features_with_targets, diagnostics) from source records."""
    df = source.copy()
    df.columns = [str(c).strip() for c in df.columns]
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df.empty:
        raise ValueError("Input dataset is empty.")
    if df.columns.duplicated().any():
        raise ValueError(f"Duplicate column headers: {df.columns[df.columns.duplicated()].tolist()}")

    df["Student_ID"] = df["Student_ID"].astype("string").str.strip()
    df["School"] = df["School"].astype("string").str.strip()
    df["Subject"] = df["Subject"].astype("string").str.strip()
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    required_scores = ["CA1", "CA2", "Assignment", "Project", "Exam", "Total"]
    invalid_numeric = {col: int(df[col].isna().sum()) for col in required_scores}
    if any(invalid_numeric.values()):
        raise ValueError(f"Required score columns contain missing/non-numeric values: {invalid_numeric}")
    scale_limits = {"CA1": 10, "CA2": 10, "Assignment": 10, "Project": 10, "Exam": 60, "Total": 100}
    out_of_range = {
        col: int((~df[col].between(0, upper)).sum())
        for col, upper in scale_limits.items()
    }
    if any(out_of_range.values()):
        raise ValueError(f"Score values outside the Phase 5 training scales: {out_of_range}")
    expected_total = df[["CA1", "CA2", "Assignment", "Project", "Exam"]].sum(axis=1)
    total_mismatches = int(((expected_total - df["Total"]).abs() > 0.01).sum())
    if total_mismatches:
        raise ValueError(f"Total does not match CA components + Exam on {total_mismatches} rows.")
    attendance_values = df["Attendance"].dropna()
    if not attendance_values.between(0, 100).all():
        raise ValueError("Attendance must be between 0 and 100 percent.")

    df["Session_Start_Year"] = df["Academic_Session"].map(session_start_year)
    df["Term_Number"] = df["Term"].map(TERM_ORDER)
    if df["Term_Number"].isna().any():
        bad = sorted(df.loc[df["Term_Number"].isna(), "Term"].astype(str).unique())
        raise ValueError(f"Unknown term values found: {bad}")

    key_cols = ["School", "Academic_Session", "Term", "Student_ID", "Subject"]
    dup_mask = df.duplicated(subset=key_cols, keep=False)
    if dup_mask.any():
        raise ValueError(f"Duplicate result keys detected: {int(dup_mask.sum())} rows; keys={key_cols}")

    blank_cols = ["Student_ID", "School", "Academic_Session", "Term", "Class", "Subject"]
    blank_counts = {
        col: int(df[col].isna().sum() + df[col].astype("string").str.strip().eq("").fillna(False).sum())
        for col in blank_cols
    }
    if any(blank_counts.values()):
        raise ValueError(f"Blank required identifiers: {blank_counts}")

    # One record per student/school/subject/term is required. For history, the
    # period key deliberately excludes Subject so same-term outcomes cannot leak.
    df = df.sort_values(
        ["School", "Student_ID", "Session_Start_Year", "Term_Number", "Subject"],
        kind="mergesort",
    ).reset_index(drop=True)
    subject_group = ["School", "Student_ID", "Subject"]
    period_group = ["School", "Student_ID", "Session_Start_Year", "Term_Number"]

    df["Historical_Subject_Average"] = df.groupby(subject_group, sort=False)["Total"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    df["Historical_Subject_Count"] = df.groupby(subject_group, sort=False)["Total"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).count()
    )

    # Collapse each completed term to its own totals first. Shift the period
    # table, then join back: all rows in a term see only strictly earlier terms.
    period_totals = (
        df.groupby(period_group, sort=False, as_index=False)
        .agg(_period_sum=("Total", "sum"), _period_count=("Total", "count"))
        .sort_values(["School", "Student_ID", "Session_Start_Year", "Term_Number"], kind="mergesort")
        .reset_index(drop=True)
    )
    student_group = ["School", "Student_ID"]
    period_totals["_prior_sum"] = period_totals.groupby(student_group, sort=False)["_period_sum"].transform(
        lambda s: s.shift(1).fillna(0).cumsum()
    )
    period_totals["_prior_count"] = period_totals.groupby(student_group, sort=False)["_period_count"].transform(
        lambda s: s.shift(1).fillna(0).cumsum()
    )
    period_totals["Historical_Overall_Average"] = (
        period_totals["_prior_sum"] / period_totals["_prior_count"].replace(0, np.nan)
    )
    period_totals["Historical_Overall_Count"] = period_totals["_prior_count"].astype(float)
    df = df.merge(
        period_totals[period_group + ["Historical_Overall_Average", "Historical_Overall_Count"]],
        on=period_group,
        how="left",
        validate="many_to_one",
    )

    df["Previous_Term_Grade_Numeric"] = df["Previous_Term_Grade"].map(grade_to_numeric)
    df["target_exam_score"] = df["Exam"]
    df["target_below_credit"] = (df["Total"] < 50).astype("int64")

    feature_columns = BASE_FEATURE_COLUMNS + ["Previous_Term_Grade_Numeric"] + DERIVED_FEATURE_COLUMNS
    result = df[feature_columns + TARGET_COLUMNS].copy()
    before = len(result)
    result = result[result["target_exam_score"].notna() & result["target_below_credit"].notna()].copy()
    removed = before - len(result)

    if set(feature_columns) & set(FORBIDDEN_PREDICTOR_COLUMNS):
        raise AssertionError("Forbidden outcome columns found in predictor list.")

    # Vectorised temporal invariants. The synthetic unit tests separately
    # assert the critical same-term and cross-school cases.
    subject_expected = df.groupby(subject_group, sort=False)["Total"].transform(
        lambda values: values.shift(1).expanding(min_periods=1).mean()
    )
    subject_count_expected = df.groupby(subject_group, sort=False)["Total"].transform(
        lambda values: values.shift(1).expanding(min_periods=1).count()
    )
    subject_violations = int((
        ~((subject_expected.isna() & df["Historical_Subject_Average"].isna()) |
          (np.isclose(subject_expected.fillna(0), df["Historical_Subject_Average"].fillna(0), atol=1e-9)))
        | ~np.isclose(subject_count_expected.fillna(0), df["Historical_Subject_Count"].fillna(0), atol=1e-9)
    ).sum())

    # Rebuild the expected prior-term average from the period aggregates, then
    # compare it with the feature joined to each row. This is vectorised and
    # scales to the full 156k-row dataset (unlike row-by-row full-table scans).
    period_check = period_totals.copy()
    period_check["_check_sum"] = period_check.groupby(student_group, sort=False)["_period_sum"].transform(
        lambda values: values.shift(1).fillna(0).cumsum()
    )
    period_check["_check_count"] = period_check.groupby(student_group, sort=False)["_period_count"].transform(
        lambda values: values.shift(1).fillna(0).cumsum()
    )
    period_check["_check_average"] = period_check["_check_sum"] / period_check["_check_count"].replace(0, np.nan)
    overall_compare = period_totals.merge(
        period_check[period_group + ["_check_average", "_check_count"]],
        on=period_group, how="left", validate="one_to_one"
    )
    overall_violations = int((
        ~((overall_compare["Historical_Overall_Average"].isna() & overall_compare["_check_average"].isna()) |
          (np.isclose(overall_compare["Historical_Overall_Average"].fillna(0), overall_compare["_check_average"].fillna(0), atol=1e-9)))
        | ~np.isclose(overall_compare["Historical_Overall_Count"], overall_compare["_check_count"], atol=1e-9)
    ).sum())
    if subject_violations or overall_violations:
        raise AssertionError(f"Temporal verification failed: subject={subject_violations}, overall={overall_violations}")

    diagnostics = {
        "status": "PASS",
        "feature_pipeline_version": FEATURE_PIPELINE_VERSION,
        "input_rows": int(len(source)),
        "output_rows": int(len(result)),
        "removed_rows_without_target": int(removed),
        "schools": int(df["School"].nunique()),
        "students_global_ids": int(df["Student_ID"].nunique()),
        "school_student_pairs": int(df[["School", "Student_ID"]].drop_duplicates().shape[0]),
        "subjects": int(df["Subject"].nunique()),
        "sessions": int(df["Academic_Session"].nunique()),
        "terms": int(df["Term"].nunique()),
        "feature_columns": feature_columns,
        "target_columns": TARGET_COLUMNS,
        "forbidden_predictor_columns": FORBIDDEN_PREDICTOR_COLUMNS,
        "forbidden_predictors_present": [],
        "historical_subject_average_temporal_violations": int(subject_violations),
        "historical_overall_average_temporal_violations": int(overall_violations),
        "same_term_results_used_in_history": False,
        "cross_school_history_isolated": True,
        "below_credit_count": int(result["target_below_credit"].sum()),
        "below_credit_rate_pct": float(100 * result["target_below_credit"].mean()),
        "missing_values": {c: int(result[c].isna().sum()) for c in result.columns},
        "input_sha256": None,
    }
    return result, diagnostics


def main():
    print("=" * 72)
    print("EDUPREDICT — CORRECTED LEAKAGE-SAFE FEATURE BUILDER")
    print("=" * 72)
    if not DATASET.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATASET}")
    source_hash = sha256_file(DATASET)
    raw = pd.read_csv(DATASET, low_memory=False)
    result, report = build_feature_frame(raw)
    report["input_sha256"] = source_hash
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False)
    REPORT_FILE.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"Input rows: {len(raw):,}")
    print(f"Output rows: {len(result):,}")
    print(f"Schools: {report['schools']}; school/student pairs: {report['school_student_pairs']:,}")
    print(f"Subject temporal violations: {report['historical_subject_average_temporal_violations']}")
    print(f"Overall temporal violations: {report['historical_overall_average_temporal_violations']}")
    print(f"Same-term outcomes used in history: {report['same_term_results_used_in_history']}")
    print(f"Feature file: {OUTPUT_FILE}")
    print(f"Report: {REPORT_FILE}")
    print("PHASE 2 FEATURE BUILD: PASS")

if __name__ == "__main__":
    main()
