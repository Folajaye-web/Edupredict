import importlib.util
from pathlib import Path
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "nigeria_validation" / "build_features.py"
SPEC = importlib.util.spec_from_file_location("edupredict_build_features", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(school, session, term, student, subject, total):
    return {
        "Student_ID": student,
        "School": school,
        "Academic_Session": session,
        "Term": term,
        "Class": "JSS1",
        "Arm": "A",
        "Gender": "Female",
        "Subject": subject,
        "CA1": 7,
        "CA2": 8,
        "Assignment": 8,
        "Project": 8,
        "Exam": total - 31,
        "Total": total,
        "Grade": "C4",
        "Attendance": 90,
        "Previous_Term_Total": None,
        "Previous_Term_Grade": None,
        "Previous_Year_Average": None,
        "Behaviour/Engagement": None,
        "Intervention": None,
        "Final_Year_Result": None,
        "Promotion_Status": None,
    }


def test_overall_history_uses_prior_terms_only_not_other_subjects_in_same_term():
    data = pd.DataFrame([
        row("School A", "2023/2024", "First Term", "STU1", "Mathematics", 60),
        row("School A", "2023/2024", "First Term", "STU1", "English Language", 80),
        row("School A", "2023/2024", "Second Term", "STU1", "Mathematics", 70),
        row("School A", "2023/2024", "Second Term", "STU1", "English Language", 90),
        row("School A", "2024/2025", "First Term", "STU1", "Mathematics", 50),
    ])
    result, report = MODULE.build_feature_frame(data)
    key = ["School", "Academic_Session", "Term", "Student_ID", "Subject"]
    indexed = result.set_index(key)

    # No record can see another subject's result from the same term.
    assert pd.isna(indexed.loc[("School A", "2023/2024", "First Term", "STU1", "English Language"), "Historical_Overall_Average"])
    assert pd.isna(indexed.loc[("School A", "2023/2024", "First Term", "STU1", "Mathematics"), "Historical_Overall_Average"])

    # Second term sees only both first-term results (60 and 80).
    second = indexed.loc[("School A", "2023/2024", "Second Term", "STU1", "Mathematics")]
    assert second["Historical_Overall_Average"] == pytest.approx(70.0)
    assert second["Historical_Overall_Count"] == 2

    # Next session sees four records from completed earlier terms only.
    next_year = indexed.loc[("School A", "2024/2025", "First Term", "STU1", "Mathematics")]
    assert next_year["Historical_Overall_Average"] == pytest.approx(75.0)
    assert next_year["Historical_Overall_Count"] == 4
    assert next_year["Historical_Subject_Average"] == pytest.approx(65.0)
    assert report["same_term_results_used_in_history"] is False
    assert report["cross_school_history_isolated"] is True


def test_same_student_id_in_different_schools_does_not_share_history():
    data = pd.DataFrame([
        row("School A", "2023/2024", "First Term", "SAMEID", "Mathematics", 90),
        row("School B", "2023/2024", "Second Term", "SAMEID", "Mathematics", 40),
    ])
    result, _ = MODULE.build_feature_frame(data)
    b = result[result["School"] == "School B"].iloc[0]
    assert pd.isna(b["Historical_Subject_Average"])
    assert pd.isna(b["Historical_Overall_Average"])


def test_duplicate_result_keys_fail_closed():
    first = row("School A", "2023/2024", "First Term", "STU1", "Mathematics", 60)
    data = pd.DataFrame([first, first])
    with pytest.raises(ValueError, match="Duplicate result keys"):
        MODULE.build_feature_frame(data)
