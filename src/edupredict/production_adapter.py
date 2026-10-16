"""Build the Phase 5 model's 19 features from pre-exam database evidence.

All historical aggregates are restricted to completed periods strictly before
(the session, term) being predicted. Current examination/result columns are
never queried for the active period.
"""
from __future__ import annotations
import math
from sqlalchemy.orm import Session
from .db import (
    AcademicSession, AssessmentPolicy, Attendance, ClassGroup,
    ContinuousAssessment, Result, School, Student, StudentPlacement,
    StudentSubjectEnrollment, Subject,
)

TERM_ORDER = {"First Term": 1, "Second Term": 2, "Third Term": 3}
GRADE_ORDER = {"A1": 9, "B2": 8, "B3": 7, "C4": 6, "C5": 5, "C6": 4, "D7": 3, "E8": 2, "F9": 1}
EXPECTED_WEIGHTS = (10.0, 10.0, 10.0, 10.0, 60.0)


def _session_key(session: AcademicSession):
    text = str(session.session_name).strip()
    if len(text) < 4 or not text[:4].isdigit():
        raise ValueError(f"Invalid academic session name: {session.session_name!r}")
    term_number = TERM_ORDER.get(str(session.term).strip())
    if term_number is None:
        raise ValueError(f"Unsupported academic term: {session.term!r}")
    return int(text[:4]), term_number


def _class_for_model(value):
    text = str(value or "").strip()
    # The historical model data uses SS1/SS2/SS3; the app UI also accepts SSS1/2/3.
    return {"SSS1": "SS1", "SSS2": "SS2", "SSS3": "SS3"}.get(text.upper(), text or None)


def _number(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def build_production_features(s: Session, *, session_id: int, student_id: int, subject_id: int):
    """Return (feature_dict, current_CA_total) or raise a useful ValueError.

    Missing prior history is represented as NaN for the model's fitted imputer;
    current CA and attendance are mandatory to avoid predictions from fabricated
    zero values.
    """
    period = s.get(AcademicSession, session_id)
    student = s.get(Student, student_id)
    subject = s.get(Subject, subject_id)
    if not period or not student or not subject:
        raise ValueError("The selected academic period, student, or subject no longer exists.")
    if period.school_id != student.school_id:
        raise ValueError("Student and academic period belong to different schools.")
    school = s.get(School, student.school_id)
    if not school:
        raise ValueError("Student's school record is missing.")

    enrollment = s.query(StudentSubjectEnrollment).filter_by(
        session_id=session_id, student_id=student_id, subject_id=subject_id, active=True
    ).first()
    if not enrollment:
        raise ValueError("Student is not actively enrolled in this subject for the selected period.")

    placement = s.query(StudentPlacement).filter_by(session_id=session_id, student_id=student_id).first()
    if not placement:
        raise ValueError("Student has no class/stream placement for the selected period.")
    group = s.get(ClassGroup, placement.class_group_id)
    if not group or group.school_id != school.id:
        raise ValueError("Student's class/stream placement is invalid for this school.")

    policy = s.query(AssessmentPolicy).filter_by(school_id=school.id).first()
    if policy:
        weights = (policy.ca1_weight, policy.ca2_weight, policy.assignment_weight, policy.project_weight, policy.exam_weight)
        if any(abs(float(actual) - expected) > 1e-9 for actual, expected in zip(weights, EXPECTED_WEIGHTS)):
            raise ValueError("Phase 5 is validated for CA1/CA2/Assignment/Project = 10 points each and Exam = 60. Align the school's assessment policy before prediction.")

    ca = s.query(ContinuousAssessment).filter_by(
        session_id=session_id, student_id=student_id, subject_id=subject_id
    ).first()
    if not ca:
        raise ValueError("No continuous assessment record exists for this student and subject yet.")
    ca_values = [_number(getattr(ca, field)) for field in ("ca1", "ca2", "assignment", "project")]
    if any(value is None for value in ca_values):
        raise ValueError("Complete CA1, CA2, Assignment, and Project before generating a prediction.")
    if any(value < 0 or value > 10 for value in ca_values):
        raise ValueError("CA1, CA2, Assignment, and Project must each be between 0 and 10 for the validated model.")
    current_ca_total = float(sum(ca_values))

    attendance = s.query(Attendance).filter_by(session_id=session_id, student_id=student_id).first()
    if not attendance or not attendance.school_days or attendance.school_days <= 0:
        raise ValueError("Attendance is missing or has zero school days. Record valid attendance before predicting.")
    if attendance.days_present < 0 or attendance.days_present > attendance.school_days:
        raise ValueError("Attendance record is inconsistent; days present must be between zero and school days.")
    attendance_pct = 100.0 * float(attendance.days_present) / float(attendance.school_days)

    current_key = _session_key(period)
    history = []
    for result, result_period in (
        s.query(Result, AcademicSession)
        .join(AcademicSession, Result.session_id == AcademicSession.id)
        .filter(Result.student_id == student_id, AcademicSession.school_id == school.id)
        .all()
    ):
        if result.total is None:
            continue
        key = _session_key(result_period)
        # Strictly earlier period only: never include any current-term result,
        # even for another subject or if the current exam has already been saved.
        if key < current_key:
            history.append((key, result, result_period))

    history.sort(key=lambda item: (item[0][0], item[0][1], item[1].subject_id))
    prior_subject = [item for item in history if item[1].subject_id == subject_id]
    overall_totals = [float(item[1].total) for item in history if _number(item[1].total) is not None]
    subject_totals = [float(item[1].total) for item in prior_subject if _number(item[1].total) is not None]
    latest_subject = prior_subject[-1] if prior_subject else None

    current_start_year = current_key[0]
    previous_year_totals = [
        float(item[1].total) for item in history
        if _session_key(item[2])[0] == current_start_year - 1 and _number(item[1].total) is not None
    ]

    previous_grade = latest_subject[1].grade if latest_subject else None
    features = {
        "School": school.name,
        "Term": period.term,
        "Class": _class_for_model(group.class_level or student.class_level),
        "Arm": group.arm or student.arm,
        "Gender": student.gender,
        "Subject": subject.name,
        "Term_Number": int(current_key[1]),
        "CA1": ca_values[0],
        "CA2": ca_values[1],
        "Assignment": ca_values[2],
        "Project": ca_values[3],
        "Attendance": float(attendance_pct),
        "Previous_Term_Total": float(latest_subject[1].total) if latest_subject else float("nan"),
        "Previous_Year_Average": float(sum(previous_year_totals) / len(previous_year_totals)) if previous_year_totals else float("nan"),
        "Previous_Term_Grade_Numeric": float(GRADE_ORDER.get(str(previous_grade).strip().upper(), float("nan"))) if previous_grade else float("nan"),
        "Historical_Subject_Average": float(sum(subject_totals) / len(subject_totals)) if subject_totals else float("nan"),
        "Historical_Subject_Count": float(len(subject_totals)),
        "Historical_Overall_Average": float(sum(overall_totals) / len(overall_totals)) if overall_totals else float("nan"),
        "Historical_Overall_Count": float(len(overall_totals)),
    }
    return features, current_ca_total
