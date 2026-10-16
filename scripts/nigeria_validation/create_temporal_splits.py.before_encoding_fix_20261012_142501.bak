from pathlib import Path
import json
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "nigeria_validation"
    / "leakage_safe_features.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "nigeria_validation"
    / "splits"
)

REPORT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "nigeria_validation"
    / "temporal_split_report.json"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)


SESSION_ORDER = [
    "2023/2024",
    "2024/2025",
    "2025/2026",
]


SPLIT_MAP = {
    "2023/2024": "train",
    "2024/2025": "validation",
    "2025/2026": "test",
}


def session_start_year(value):
    return int(str(value).strip()[:4])


def main():

    print("=" * 72)
    print("EDUPREDICT PHASE 3")
    print("TIME-AWARE HISTORICAL TRAIN / VALIDATION / TEST SPLIT")
    print("=" * 72)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"Input rows: {len(df):,}")
    print(f"Input columns: {len(df.columns)}")

    required = [
        "Student_ID",
        "School",
        "Academic_Session",
        "Term",
        "Subject",
        "target_exam_score",
        "target_below_credit",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # ---------------------------------------------------------------
    # Session validation
    # ---------------------------------------------------------------

    actual_sessions = sorted(
        df["Academic_Session"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    print()
    print("[1] Session inventory")
    for session in actual_sessions:
        print(f"    {session}")

    if set(actual_sessions) != set(SESSION_ORDER):
        raise ValueError(
            "Unexpected session set.\n"
            f"Expected: {SESSION_ORDER}\n"
            f"Found:    {actual_sessions}"
        )

    # ---------------------------------------------------------------
    # Explicit chronological ordering
    # ---------------------------------------------------------------

    df["_session_start_year"] = (
        df["Academic_Session"]
        .map(session_start_year)
    )

    expected_years = {
        "2023/2024": 2023,
        "2024/2025": 2024,
        "2025/2026": 2025,
    }

    for session, expected_year in expected_years.items():

        actual_years = (
            df.loc[
                df["Academic_Session"] == session,
                "_session_start_year",
            ]
            .unique()
            .tolist()
        )

        if actual_years != [expected_year]:
            raise ValueError(
                f"Invalid chronological value for {session}: "
                f"{actual_years}"
            )

    print("[2] Chronological session ordering: PASS")

    # ---------------------------------------------------------------
    # Assign split
    # ---------------------------------------------------------------

    df["split"] = (
        df["Academic_Session"]
        .map(SPLIT_MAP)
    )

    if df["split"].isna().any():
        raise ValueError(
            "Some records could not be assigned to a split."
        )

    train = df[
        df["split"] == "train"
    ].copy()

    validation = df[
        df["split"] == "validation"
    ].copy()

    test = df[
        df["split"] == "test"
    ].copy()

    print()
    print("[3] Split sizes")

    print(
        f"    Train:      {len(train):,}"
    )

    print(
        f"    Validation: {len(validation):,}"
    )

    print(
        f"    Test:       {len(test):,}"
    )

    if len(train) == 0:
        raise ValueError("Training split is empty.")

    if len(validation) == 0:
        raise ValueError("Validation split is empty.")

    if len(test) == 0:
        raise ValueError("Test split is empty.")

    # ---------------------------------------------------------------
    # Ensure each split contains exactly one session
    # ---------------------------------------------------------------

    split_sessions = {
        "train": sorted(
            train["Academic_Session"]
            .unique()
            .tolist()
        ),
        "validation": sorted(
            validation["Academic_Session"]
            .unique()
            .tolist()
        ),
        "test": sorted(
            test["Academic_Session"]
            .unique()
            .tolist()
        ),
    }

    expected_split_sessions = {
        "train": ["2023/2024"],
        "validation": ["2024/2025"],
        "test": ["2025/2026"],
    }

    if split_sessions != expected_split_sessions:
        raise AssertionError(
            "Temporal split session assignment failed.\n"
            f"Actual:   {split_sessions}\n"
            f"Expected: {expected_split_sessions}"
        )

    print(
        "[4] One-session-per-split check: PASS"
    )

    # ---------------------------------------------------------------
    # Ensure no session overlap
    # ---------------------------------------------------------------

    train_sessions = set(
        train["Academic_Session"]
    )

    validation_sessions = set(
        validation["Academic_Session"]
    )

    test_sessions = set(
        test["Academic_Session"]
    )

    if train_sessions & validation_sessions:
        raise AssertionError(
            "Train/validation session overlap detected."
        )

    if train_sessions & test_sessions:
        raise AssertionError(
            "Train/test session overlap detected."
        )

    if validation_sessions & test_sessions:
        raise AssertionError(
            "Validation/test session overlap detected."
        )

    print(
        "[5] Session overlap check: PASS"
    )

    # ---------------------------------------------------------------
    # Strict chronological ordering
    # ---------------------------------------------------------------

    max_train_year = train[
        "_session_start_year"
    ].max()

    min_validation_year = validation[
        "_session_start_year"
    ].min()

    max_validation_year = validation[
        "_session_start_year"
    ].max()

    min_test_year = test[
        "_session_start_year"
    ].min()

    if not (
        max_train_year < min_validation_year
        and
        max_validation_year < min_test_year
    ):
        raise AssertionError(
            "Chronological ordering between splits failed."
        )

    print(
        "[6] Strict chronological ordering: PASS"
    )

    # ---------------------------------------------------------------
    # Student overlap analysis
    #
    # Students appearing in multiple years are NOT automatically
    # considered leakage. Their earlier records are historical
    # information available when predicting their later performance.
    # ---------------------------------------------------------------

    train_students = set(
        train["Student_ID"]
    )

    validation_students = set(
        validation["Student_ID"]
    )

    test_students = set(
        test["Student_ID"]
    )

    train_validation_overlap = (
        train_students
        & validation_students
    )

    train_test_overlap = (
        train_students
        & test_students
    )

    validation_test_overlap = (
        validation_students
        & test_students
    )

    print()
    print("[7] Student continuity analysis")

    print(
        f"    Train students: "
        f"{len(train_students):,}"
    )

    print(
        f"    Validation students: "
        f"{len(validation_students):,}"
    )

    print(
        f"    Test students: "
        f"{len(test_students):,}"
    )

    print(
        f"    Train -> Validation overlap: "
        f"{len(train_validation_overlap):,}"
    )

    print(
        f"    Train -> Test overlap: "
        f"{len(train_test_overlap):,}"
    )

    print(
        f"    Validation -> Test overlap: "
        f"{len(validation_test_overlap):,}"
    )

    # ---------------------------------------------------------------
    # Feature leakage boundary
    # ---------------------------------------------------------------

    forbidden = [
        "Exam",
        "Total",
        "Grade",
        "Final_Year_Result",
        "Promotion_Status",
        "Intervention",
    ]

    predictor_columns = [
        column
        for column in df.columns
        if column not in {
            "target_exam_score",
            "target_below_credit",
            "split",
            "_session_start_year",
        }
    ]

    forbidden_present = [
        column
        for column in forbidden
        if column in predictor_columns
    ]

    if forbidden_present:
        raise AssertionError(
            "Forbidden predictor columns detected: "
            f"{forbidden_present}"
        )

    print()
    print(
        "[8] Predictor leakage boundary: PASS"
    )

    # ---------------------------------------------------------------
    # Target distributions
    # ---------------------------------------------------------------

    print()
    print("[9] Target distributions")

    split_target_stats = {}

    for name, frame in [
        ("train", train),
        ("validation", validation),
        ("test", test),
    ]:

        below_credit_count = int(
            frame["target_below_credit"].sum()
        )

        below_credit_rate = (
            100.0
            * frame["target_below_credit"].mean()
        )

        exam_mean = float(
            frame["target_exam_score"].mean()
        )

        exam_median = float(
            frame["target_exam_score"].median()
        )

        split_target_stats[name] = {
            "rows": int(len(frame)),
            "below_credit_count": below_credit_count,
            "below_credit_rate_pct": below_credit_rate,
            "exam_mean": exam_mean,
            "exam_median": exam_median,
        }

        print(
            f"    {name.capitalize():11s}: "
            f"rows={len(frame):,}, "
            f"below-credit={below_credit_rate:.2f}%, "
            f"exam_mean={exam_mean:.2f}"
        )

    # ---------------------------------------------------------------
    # Save clean split files
    # ---------------------------------------------------------------

    columns_to_drop = [
        "split",
        "_session_start_year",
    ]

    train_output = train.drop(
        columns=columns_to_drop
    )

    validation_output = validation.drop(
        columns=columns_to_drop
    )

    test_output = test.drop(
        columns=columns_to_drop
    )

    train_file = (
        OUTPUT_DIR / "train_2023_2024.csv"
    )

    validation_file = (
        OUTPUT_DIR / "validation_2024_2025.csv"
    )

    test_file = (
        OUTPUT_DIR / "test_2025_2026.csv"
    )

    train_output.to_csv(
        train_file,
        index=False,
    )

    validation_output.to_csv(
        validation_file,
        index=False,
    )

    test_output.to_csv(
        test_file,
        index=False,
    )

    # ---------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------

    report = {
        "status": "PASS",
        "input_file": str(INPUT_FILE),
        "input_rows": int(len(df)),
        "sessions": SESSION_ORDER,
        "split_mapping": SPLIT_MAP,
        "split_sessions": split_sessions,
        "split_rows": {
            "train": int(len(train)),
            "validation": int(len(validation)),
            "test": int(len(test)),
        },
        "student_counts": {
            "train": int(len(train_students)),
            "validation": int(len(validation_students)),
            "test": int(len(test_students)),
        },
        "student_overlap": {
            "train_validation": int(
                len(train_validation_overlap)
            ),
            "train_test": int(
                len(train_test_overlap)
            ),
            "validation_test": int(
                len(validation_test_overlap)
            ),
        },
        "target_statistics": split_target_stats,
        "forbidden_predictors": forbidden,
        "forbidden_predictors_present": (
            forbidden_present
        ),
        "outputs": {
            "train": str(train_file),
            "validation": str(validation_file),
            "test": str(test_file),
        },
    }

    REPORT_FILE.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("PHASE 3 STATUS: PASS")
    print("=" * 72)

    print()
    print("Training:")
    print(train_file)

    print()
    print("Validation:")
    print(validation_file)

    print()
    print("Final test:")
    print(test_file)

    print()
    print("Report:")
    print(REPORT_FILE)


if __name__ == "__main__":
    main()