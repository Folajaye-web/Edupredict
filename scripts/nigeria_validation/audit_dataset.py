from pathlib import Path
import json
import pandas as pd
import numpy as np

DATASET = Path(
    r"C:\Users\l2e\OneDrive\Desktop\Nigeria_Student_Records_2023_2026_Combined.csv"
)

REPORT_DIR = Path(__file__).resolve().parents[2] / "reports" / "nigeria_validation"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = REPORT_DIR / "dataset_audit.json"

REQUIRED_COLUMNS = [
    "Student_ID",
    "School",
    "Academic_Session",
    "Term",
    "Class",
    "Arm",
    "Gender",
    "Subject",
    "CA1",
    "CA2",
    "Assignment",
    "Project",
    "Exam",
    "Total",
    "Grade",
    "Attendance",
    "Previous_Term_Total",
    "Previous_Term_Grade",
    "Previous_Year_Average",
    "Behaviour/Engagement",
    "Intervention",
    "Final_Year_Result",
    "Promotion_Status",
]

NUMERIC_LIMITS = {
    "CA1": (0, 10),
    "CA2": (0, 10),
    "Assignment": (0, 10),
    "Project": (0, 10),
    "Exam": (0, 60),
    "Total": (0, 100),
    "Attendance": (0, 100),
    "Previous_Term_Total": (0, 100),
    "Previous_Year_Average": (0, 100),
}

GRADE_BANDS = [
    (75, "A1"),
    (70, "B2"),
    (65, "B3"),
    (60, "C4"),
    (55, "C5"),
    (50, "C6"),
    (45, "D7"),
    (40, "E8"),
    (0, "F9"),
]

LEAKAGE_COLUMNS = [
    "Exam",
    "Total",
    "Grade",
    "Final_Year_Result",
    "Promotion_Status",
]


def grade_from_total(score):
    if pd.isna(score):
        return None

    score = float(score)

    for minimum, grade in GRADE_BANDS:
        if score >= minimum:
            return grade

    return "F9"


def main():
    if not DATASET.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET}")

    print("=" * 70)
    print("EDUPREDICT NIGERIAN DATASET MODEL-QUALITY AUDIT")
    print("=" * 70)
    print(f"Dataset: {DATASET}")

    df = pd.read_csv(DATASET)

    print(f"\nRows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    report = {
        "dataset": str(DATASET),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "required_columns_missing": [],
        "duplicate_rows": 0,
        "duplicate_prediction_keys": 0,
        "range_violations": {},
        "total_formula_errors": 0,
        "grade_formula_errors": 0,
        "sessions": {},
        "terms": {},
        "schools": int(df["School"].nunique()),
        "students": int(df["Student_ID"].nunique()),
        "subjects": int(df["Subject"].nunique()),
        "classes": int(df["Class"].nunique()),
        "arms": int(df["Arm"].nunique()),
        "behaviour_categories": {},
        "temporal_checks": {},
        "leakage_columns": LEAKAGE_COLUMNS,
        "status": "FAIL",
    }

    # ------------------------------------------------------------
    # 1. Required columns
    # ------------------------------------------------------------

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    report["required_columns_missing"] = missing

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print("\n[1] Required columns: PASS")

    # ------------------------------------------------------------
    # 2. Duplicate records
    # ------------------------------------------------------------

    report["duplicate_rows"] = int(df.duplicated().sum())

    prediction_key = [
        "Student_ID",
        "Academic_Session",
        "Term",
        "Subject",
    ]

    report["duplicate_prediction_keys"] = int(
        df.duplicated(subset=prediction_key).sum()
    )

    print(
        f"[2] Duplicate complete rows: {report['duplicate_rows']}"
    )
    print(
        f"    Duplicate prediction keys: "
        f"{report['duplicate_prediction_keys']}"
    )

    # ------------------------------------------------------------
    # 3. Numeric ranges
    # ------------------------------------------------------------

    print("\n[3] Numeric range checks")

    for column, (minimum, maximum) in NUMERIC_LIMITS.items():
        values = pd.to_numeric(df[column], errors="coerce")

        invalid = (
            values.notna()
            & ((values < minimum) | (values > maximum))
        )

        count = int(invalid.sum())

        report["range_violations"][column] = count

        print(
            f"    {column:24s}: "
            f"{count:6d} violations"
        )

    # ------------------------------------------------------------
    # 4. Total consistency
    # ------------------------------------------------------------

    calculated_total = (
        pd.to_numeric(df["CA1"], errors="coerce")
        + pd.to_numeric(df["CA2"], errors="coerce")
        + pd.to_numeric(df["Assignment"], errors="coerce")
        + pd.to_numeric(df["Project"], errors="coerce")
        + pd.to_numeric(df["Exam"], errors="coerce")
    )

    supplied_total = pd.to_numeric(
        df["Total"], errors="coerce"
    )

    total_errors = (
        calculated_total.notna()
        & supplied_total.notna()
        & ~np.isclose(
            calculated_total,
            supplied_total,
            atol=1e-9,
        )
    )

    report["total_formula_errors"] = int(total_errors.sum())

    print(
        f"\n[4] Total formula errors: "
        f"{report['total_formula_errors']}"
    )

    # ------------------------------------------------------------
    # 5. Grade consistency
    # ------------------------------------------------------------

    calculated_grade = supplied_total.map(grade_from_total)

    supplied_grade = (
        df["Grade"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    grade_errors = (
        calculated_grade.notna()
        & supplied_grade.notna()
        & (calculated_grade != supplied_grade)
    )

    report["grade_formula_errors"] = int(grade_errors.sum())

    print(
        f"[5] Grade consistency errors: "
        f"{report['grade_formula_errors']}"
    )

    # ------------------------------------------------------------
    # 6. Dataset dimensions
    # ------------------------------------------------------------

    print("\n[6] Dataset dimensions")

    for session, count in (
        df["Academic_Session"]
        .value_counts()
        .sort_index()
        .items()
    ):
        report["sessions"][str(session)] = int(count)
        print(f"    {session}: {count:,}")

    for term, count in (
        df["Term"]
        .value_counts()
        .sort_index()
        .items()
    ):
        report["terms"][str(term)] = int(count)
        print(f"    {term}: {count:,}")

    print(f"    Schools:   {report['schools']}")
    print(f"    Students:  {report['students']}")
    print(f"    Subjects:  {report['subjects']}")
    print(f"    Classes:   {report['classes']}")
    print(f"    Arms:      {report['arms']}")

    # ------------------------------------------------------------
    # 7. Behaviour categories
    # ------------------------------------------------------------

    behaviour = (
        df["Behaviour/Engagement"]
        .fillna("<MISSING>")
        .astype(str)
        .str.strip()
    )

    report["behaviour_categories"] = {
        str(k): int(v)
        for k, v in behaviour.value_counts().items()
    }

    print("\n[7] Behaviour/Engagement categories")

    for value, count in report["behaviour_categories"].items():
        print(f"    {value}: {count:,}")

    # ------------------------------------------------------------
    # 8. Temporal reconstruction
    # ------------------------------------------------------------

    print("\n[8] Temporal reconstruction")

    work = df[
        [
            "Student_ID",
            "Academic_Session",
            "Term",
            "Subject",
            "Total",
            "Previous_Term_Total",
        ]
    ].copy()

    work["_session_order"] = (
        work["Academic_Session"]
        .astype(str)
        .str[:4]
        .astype(int)
    )

    term_order = {
        "First Term": 1,
        "Second Term": 2,
        "Third Term": 3,
    }

    work["_term_order"] = work["Term"].map(term_order)

    work = work.sort_values(
        [
            "Student_ID",
            "Subject",
            "_session_order",
            "_term_order",
        ]
    )

    work["expected_previous"] = (
        work.groupby(
            ["Student_ID", "Subject"]
        )["Total"]
        .shift(1)
    )

    comparable = work["expected_previous"].notna()

    matches = (
        comparable
        & np.isclose(
            pd.to_numeric(
                work["Previous_Term_Total"],
                errors="coerce",
            ),
            pd.to_numeric(
                work["expected_previous"],
                errors="coerce",
            ),
            atol=1e-9,
            equal_nan=False,
        )
    )

    mismatches = comparable & ~matches

    report["temporal_checks"][
        "previous_term_comparable_rows"
    ] = int(comparable.sum())

    report["temporal_checks"][
        "previous_term_matches"
    ] = int(matches.sum())

    report["temporal_checks"][
        "previous_term_mismatches"
    ] = int(mismatches.sum())

    report["temporal_checks"][
        "previous_term_no_prior_record"
    ] = int((~comparable).sum())

    mismatch_rate = (
        100.0
        * mismatches.sum()
        / max(1, comparable.sum())
    )

    report["temporal_checks"][
        "previous_term_mismatch_rate_pct"
    ] = float(mismatch_rate)

    print(
        f"    Comparable rows: {comparable.sum():,}"
    )
    print(
        f"    Matches:         {matches.sum():,}"
    )
    print(
        f"    Mismatches:      {mismatches.sum():,}"
    )
    print(
        f"    Mismatch rate:   {mismatch_rate:.4f}%"
    )

    # ------------------------------------------------------------
    # 9. End-of-year fields
    # ------------------------------------------------------------

    print("\n[9] End-of-year field timing")

    final_by_term = (
        df.groupby("Term")["Final_Year_Result"]
        .apply(lambda x: int(x.notna().sum()))
        .to_dict()
    )

    promotion_by_term = (
        df.groupby("Term")["Promotion_Status"]
        .apply(lambda x: int(x.notna().sum()))
        .to_dict()
    )

    report["temporal_checks"][
        "final_year_result_by_term"
    ] = final_by_term

    report["temporal_checks"][
        "promotion_status_by_term"
    ] = promotion_by_term

    print("    Final_Year_Result:")
    for term, count in final_by_term.items():
        print(f"      {term}: {count:,}")

    print("    Promotion_Status:")
    for term, count in promotion_by_term.items():
        print(f"      {term}: {count:,}")

    # ------------------------------------------------------------
    # 10. Leakage boundary
    # ------------------------------------------------------------

    print("\n[10] Leakage boundary")

    print("    EXCLUDED from predictors:")
    for column in LEAKAGE_COLUMNS:
        print(f"      - {column}")

    # ------------------------------------------------------------
    # 11. Final status
    # ------------------------------------------------------------

    range_errors = sum(
        report["range_violations"].values()
    )

    hard_failures = [
        report["required_columns_missing"],
        report["duplicate_prediction_keys"],
        range_errors,
        report["total_formula_errors"],
        report["grade_formula_errors"],
        report["temporal_checks"][
            "previous_term_mismatches"
        ],
    ]

    report["status"] = (
        "PASS"
        if not any(hard_failures)
        else "FAIL"
    )

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print(f"AUDIT STATUS: {report['status']}")
    print(f"Report: {REPORT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()