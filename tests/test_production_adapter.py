import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from edupredict.db import (
    Base, School, AcademicSession, ClassGroup, Student, Subject,
    StudentPlacement, StudentSubjectEnrollment, ContinuousAssessment,
    Attendance, Result,
)
from edupredict.production_adapter import build_production_features


def setup_database():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    s = Session()
    school = School(name="Adapter Test School", state="Ekiti")
    s.add(school); s.flush()
    old1 = AcademicSession(school_id=school.id, session_name="2024/2025", term="Third Term", is_current=False)
    old2 = AcademicSession(school_id=school.id, session_name="2025/2026", term="First Term", is_current=False)
    current = AcademicSession(school_id=school.id, session_name="2025/2026", term="Second Term", is_current=True)
    s.add_all([old1, old2, current]); s.flush()
    groups = [
        ClassGroup(school_id=school.id, session_id=p.id, class_level="SSS2", arm="A", display_name="SSS2 A")
        for p in (old1, old2, current)
    ]
    s.add_all(groups); s.flush()
    student = Student(school_id=school.id, student_code="AD-001", first_name="Ada", last_name="Test", gender="Female", class_level="SSS2", arm="A")
    math = Subject(code="MATH", name="Mathematics", stage="SSS")
    english = Subject(code="ENG", name="English Language", stage="SSS")
    s.add_all([student, math, english]); s.flush()
    for p, g in zip((old1, old2, current), groups):
        s.add(StudentPlacement(session_id=p.id, student_id=student.id, class_group_id=g.id))
    s.add(StudentSubjectEnrollment(session_id=current.id, student_id=student.id, subject_id=math.id, active=True))
    s.add(ContinuousAssessment(session_id=current.id, student_id=student.id, subject_id=math.id, ca1=7, ca2=8, assignment=9, project=10))
    s.add(Attendance(session_id=current.id, student_id=student.id, school_days=100, days_present=92, days_absent=8))
    # Completed prior periods: Mathematics 65 and 75; English 80.
    s.add_all([
        Result(session_id=old1.id, student_id=student.id, subject_id=math.id, ca1=7, ca2=8, assignment=8, project=8, exam=34, total=65, grade="C4", grade_point=4),
        Result(session_id=old2.id, student_id=student.id, subject_id=math.id, ca1=7, ca2=8, assignment=8, project=8, exam=44, total=75, grade="B3", grade_point=7),
        Result(session_id=old2.id, student_id=student.id, subject_id=english.id, ca1=7, ca2=8, assignment=8, project=8, exam=42, total=73, grade="B3", grade_point=7),
        # A current-period result must never leak into current pre-exam features.
        Result(session_id=current.id, student_id=student.id, subject_id=math.id, ca1=7, ca2=8, assignment=9, project=10, exam=60, total=94, grade="A1", grade_point=9),
    ])
    s.commit()
    return s, current, student, math


def test_adapter_uses_only_strictly_prior_periods_and_exam_free_features():
    s, current, student, math = setup_database()
    features, ca_total = build_production_features(s, session_id=current.id, student_id=student.id, subject_id=math.id)
    assert ca_total == pytest.approx(34.0)
    assert features["Class"] == "SS2"  # normalizes the app's SSS spelling to training spelling
    assert features["Attendance"] == pytest.approx(92.0)
    assert features["Previous_Term_Total"] == pytest.approx(75.0)
    assert features["Previous_Term_Grade_Numeric"] == pytest.approx(7.0)
    assert features["Historical_Subject_Average"] == pytest.approx(70.0)
    assert features["Historical_Subject_Count"] == 2
    assert features["Historical_Overall_Average"] == pytest.approx((65 + 75 + 73) / 3)
    assert features["Historical_Overall_Count"] == 3
    assert "Exam" not in features and "Total" not in features and "Grade" not in features


def test_adapter_blocks_missing_current_ca():
    s, current, student, math = setup_database()
    ca = s.query(ContinuousAssessment).filter_by(session_id=current.id, student_id=student.id, subject_id=math.id).one()
    ca.project = None
    s.commit()
    with pytest.raises(ValueError, match="Complete CA1"):
        build_production_features(s, session_id=current.id, student_id=student.id, subject_id=math.id)


def test_adapter_blocks_missing_attendance():
    s, current, student, math = setup_database()
    s.query(Attendance).filter_by(session_id=current.id, student_id=student.id).delete()
    s.commit()
    with pytest.raises(ValueError, match="Attendance is missing"):
        build_production_features(s, session_id=current.id, student_id=student.id, subject_id=math.id)


def test_predictor_converts_exam_score_to_overall_total(monkeypatch):
    import pandas as pd
    import edupredict.ai.predictor as predictor

    sample_features = {
        "School": "School A", "Term": "First Term", "Class": "SS2", "Arm": "A",
        "Gender": "Female", "Subject": "Mathematics", "Term_Number": 1,
        "CA1": 7, "CA2": 8, "Assignment": 9, "Project": 10, "Attendance": 92,
        "Previous_Term_Total": float("nan"), "Previous_Year_Average": float("nan"),
        "Previous_Term_Grade_Numeric": float("nan"), "Historical_Subject_Average": float("nan"),
        "Historical_Subject_Count": 0, "Historical_Overall_Average": float("nan"), "Historical_Overall_Count": 0,
    }
    monkeypatch.setattr(predictor, "build_production_features", lambda *args, **kwargs: (sample_features, 34.0))
    monkeypatch.setattr(predictor, "predict_phase5", lambda df: (
        pd.DataFrame([{"risk_probability": 0.4, "early_warning": True, "predicted_exam_score": 35.2}]),
        {"threshold": 0.4, "model_version": "phase5_nigeria_candidate"},
    ))
    result = predictor.predict_pre_exam(object(), session_id=1, student_id=1, subject_id=1)
    assert result["predicted_exam_score"] == pytest.approx(35.2)
    assert result["predicted_total"] == pytest.approx(69.2)
    assert result["predicted_score"] == pytest.approx(69.2)
    assert result["early_warning"] is True


def test_risk_band_matches_phase5_warning_threshold():
    from edupredict.risk import risk_from_score
    assert risk_from_score(85, 0.40)["level"] == "Moderate"
    assert risk_from_score(85, 0.46)["level"] == "High"
    assert risk_from_score(85, 0.75)["level"] == "Critical"
