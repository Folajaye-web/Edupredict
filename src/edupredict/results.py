"""Transactional result handling for EduPredict.

Rules:
- one current result per (academic session, student, subject)
- student must be placed in the selected session
- student must be actively enrolled in the subject
- every component is validated 0..100
- final score is derived only from the school's configured assessment policy
- corrections update the current row and snapshot the previous value in ResultRevision
- duplicate legacy rows can be repaired explicitly, never silently at startup
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import shutil

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .db import (
    AcademicSession,
    AssessmentPolicy,
    ClassGroup,
    Result,
    ResultRevision,
    Student,
    StudentPlacement,
    StudentSubjectEnrollment,
    Subject,
)
from .grading import grade_from_score
from .db import DB_PATH, ensure_result_unique_guard

COMPONENTS = ("ca1", "ca2", "assignment", "project", "exam")


def validate_policy(policy: AssessmentPolicy) -> None:
    weights = [
        float(policy.ca1_weight or 0),
        float(policy.ca2_weight or 0),
        float(policy.assignment_weight or 0),
        float(policy.project_weight or 0),
        float(policy.exam_weight or 0),
    ]
    if any(w < 0 or w > 100 for w in weights):
        raise ValueError("Assessment weights must each be between 0% and 100%.")
    if abs(sum(weights) - 100.0) > 1e-6:
        raise ValueError("Assessment weights must add up to exactly 100%.")
    if not 0 <= float(policy.credit_threshold) <= 100:
        raise ValueError("Credit threshold must be between 0% and 100%.")


def validate_components(values: dict) -> dict[str, float]:
    clean = {}
    for name in COMPONENTS:
        value = values.get(name, 0)
        if value is None or value == "":
            raise ValueError(f"{name.upper()} is required. Use 0 for a genuine zero mark.")
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name.upper()} must be numeric.") from exc
        if not 0 <= value <= 100:
            raise ValueError(f"{name.upper()} must be between 0 and 100.")
        clean[name] = round(value, 4)
    return clean


def calculate_final_score(values: dict, policy: AssessmentPolicy) -> tuple[float, str, float]:
    validate_policy(policy)
    clean = validate_components(values)
    weights = {
        "ca1": policy.ca1_weight,
        "ca2": policy.ca2_weight,
        "assignment": policy.assignment_weight,
        "project": policy.project_weight,
        "exam": policy.exam_weight,
    }
    total = sum(clean[k] * float(weights[k]) / 100.0 for k in COMPONENTS)
    grade, grade_point = grade_from_score(total)
    return round(total, 2), grade, grade_point


def _assert_eligibility(s: Session, session_id: int, student_id: int, subject_id: int) -> None:
    placement = s.query(StudentPlacement).filter_by(session_id=session_id, student_id=student_id).first()
    if not placement:
        raise ValueError("Student is not placed in a class/stream for this academic period.")

    enrollment = s.query(StudentSubjectEnrollment).filter_by(
        session_id=session_id, student_id=student_id, subject_id=subject_id, active=True
    ).first()
    if not enrollment:
        raise ValueError("Student is not actively enrolled in this subject for this academic period.")


def _snapshot(result: Result, action: str, reason: str | None = None) -> ResultRevision:
    return ResultRevision(
        result_id=result.id,
        session_id=result.session_id,
        student_id=result.student_id,
        subject_id=result.subject_id,
        ca1=result.ca1,
        ca2=result.ca2,
        assignment=result.assignment,
        project=result.project,
        exam=result.exam,
        total=result.total,
        grade=result.grade,
        grade_point=result.grade_point,
        teacher_comment=result.teacher_comment,
        action=action,
        reason=reason,
    )


def upsert_result(
    s: Session,
    *,
    session_id: int,
    student_id: int,
    subject_id: int,
    policy: AssessmentPolicy,
    ca1: float,
    ca2: float,
    assignment: float,
    project: float,
    exam: float,
    teacher_comment: str | None = None,
    reason: str | None = None,
) -> Result:
    """Create or update exactly one current result for a student/subject/period."""
    _assert_eligibility(s, session_id, student_id, subject_id)
    total, grade, grade_point = calculate_final_score(
        {"ca1": ca1, "ca2": ca2, "assignment": assignment, "project": project, "exam": exam}, policy
    )

    result = s.query(Result).filter_by(session_id=session_id, student_id=student_id, subject_id=subject_id).first()
    now = datetime.now(timezone.utc)
    if result:
        s.add(_snapshot(result, "UPDATED", reason or "Result corrected/updated."))
        result.ca1, result.ca2 = ca1, ca2
        result.assignment, result.project, result.exam = assignment, project, exam
        result.total, result.grade, result.grade_point = total, grade, grade_point
        result.teacher_comment = teacher_comment
        result.updated_at = now
    else:
        result = Result(
            session_id=session_id, student_id=student_id, subject_id=subject_id,
            ca1=ca1, ca2=ca2, assignment=assignment, project=project, exam=exam,
            total=total, grade=grade, grade_point=grade_point, teacher_comment=teacher_comment,
            created_at=now, updated_at=now,
        )
        s.add(result)
    try:
        s.flush()
    except IntegrityError as exc:
        s.rollback()
        raise ValueError("A result already exists for this student, subject and academic period. Reload and update it instead.") from exc
    return result


def duplicate_result_groups(s: Session) -> list[dict]:
    rows = s.execute(
        select(Result.session_id, Result.student_id, Result.subject_id, func.count(Result.id))
        .group_by(Result.session_id, Result.student_id, Result.subject_id)
        .having(func.count(Result.id) > 1)
    ).all()
    return [
        {"session_id": a, "student_id": b, "subject_id": c, "count": int(n)}
        for a, b, c, n in rows
    ]


def backup_database() -> Path:
    """Create a timestamped SQLite backup before any duplicate repair."""
    backup_dir = DB_PATH.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    target = backup_dir / f"edupredict_before_result_repair_{stamp}.db"
    shutil.copy2(DB_PATH, target)
    return target


def repair_duplicate_results(s: Session, *, reason: str = "Legacy duplicate repair") -> int:
    """Collapse legacy duplicates, keeping the latest row and auditing every retired row.

    This function intentionally does not run during startup. Call it only from an explicit
    maintenance operation after a database backup.
    """
    groups = duplicate_result_groups(s)
    retired = 0
    for g in groups:
        rows = s.query(Result).filter_by(
            session_id=g["session_id"], student_id=g["student_id"], subject_id=g["subject_id"]
        ).order_by(Result.updated_at.desc(), Result.created_at.desc(), Result.id.desc()).all()
        keep = rows[0]
        for old in rows[1:]:
            snapshot = _snapshot(old, "DUPLICATE_RETIRED", reason)
            snapshot.result_id = None
            s.add(snapshot)
            s.delete(old)
            retired += 1
        s.flush()
    s.commit()
    ensure_result_unique_guard()
    return retired


def result_integrity_report(s: Session) -> dict:
    groups = duplicate_result_groups(s)
    return {
        "duplicate_groups": len(groups),
        "duplicate_rows": sum(max(0, g["count"] - 1) for g in groups),
        "unique_index_expected": len(groups) == 0,
    }
