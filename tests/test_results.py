import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from edupredict.db import (
    Base, School, AcademicSession, ClassGroup, Student, StudentPlacement,
    Subject, StudentSubjectEnrollment, AssessmentPolicy, Result,
)
from edupredict.results import calculate_final_score, upsert_result, duplicate_result_groups, repair_duplicate_results


def make_session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)()


def seed(s):
    school = School(name="Test School", state="Lagos", school_type="Private")
    s.add(school); s.flush()
    term = AcademicSession(school_id=school.id, session_name="2026/2027", term="First Term", is_current=True)
    s.add(term); s.flush()
    group = ClassGroup(school_id=school.id, session_id=term.id, class_level="SSS2", arm="A", display_name="SSS2 A")
    student = Student(school_id=school.id, student_code="ST001", first_name="Test", last_name="Student", class_level="SSS2", arm="A")
    subject = Subject(code="MATH", name="Mathematics", stage="SSS", compulsory=True)
    policy = AssessmentPolicy(school_id=school.id, name="Test", ca1_weight=10, ca2_weight=10, assignment_weight=10, project_weight=10, exam_weight=60, credit_threshold=50)
    s.add_all([group, student, subject, policy]); s.flush()
    s.add(StudentPlacement(session_id=term.id, student_id=student.id, class_group_id=group.id))
    s.add(StudentSubjectEnrollment(session_id=term.id, student_id=student.id, subject_id=subject.id, active=True))
    s.commit()
    return term, student, subject, policy


def test_score_calculation_uses_configured_weights():
    policy = AssessmentPolicy(ca1_weight=10, ca2_weight=10, assignment_weight=10, project_weight=10, exam_weight=60, credit_threshold=50)
    total, grade, gp = calculate_final_score(dict(ca1=50, ca2=50, assignment=50, project=50, exam=60), policy)
    assert total == 56.0
    assert grade == "C5"
    assert gp == 2.7


def test_upsert_twice_keeps_one_current_result_and_audits_update():
    s = make_session(); term, student, subject, policy = seed(s)
    first = upsert_result(s, session_id=term.id, student_id=student.id, subject_id=subject.id, policy=policy, ca1=40, ca2=40, assignment=40, project=40, exam=40)
    s.commit()
    second = upsert_result(s, session_id=term.id, student_id=student.id, subject_id=subject.id, policy=policy, ca1=60, ca2=60, assignment=60, project=60, exam=60, reason="Correction")
    s.commit()
    assert first.id == second.id
    assert s.query(Result).filter_by(session_id=term.id, student_id=student.id, subject_id=subject.id).count() == 1
    assert s.query(Result).filter_by(id=first.id).one().total == 60
    assert s.query(__import__('edupredict.db', fromlist=['ResultRevision']).ResultRevision).count() == 1


def test_result_requires_active_subject_enrollment():
    s = make_session(); term, student, subject, policy = seed(s)
    s.query(StudentSubjectEnrollment).delete(); s.commit()
    with pytest.raises(ValueError, match="not actively enrolled"):
        upsert_result(s, session_id=term.id, student_id=student.id, subject_id=subject.id, policy=policy, ca1=1, ca2=1, assignment=1, project=1, exam=1)


def test_duplicate_repair_keeps_latest_and_audits_retired(monkeypatch):
    s = make_session(); term, student, subject, policy = seed(s)
    # Recreate results without the ORM unique constraint to simulate a legacy database.
    s.execute(__import__('sqlalchemy').text("DROP TABLE results"))
    s.execute(__import__('sqlalchemy').text("""CREATE TABLE results (
        id INTEGER PRIMARY KEY, session_id INTEGER NOT NULL, student_id INTEGER NOT NULL, subject_id INTEGER NOT NULL,
        ca1 FLOAT, ca2 FLOAT, assignment FLOAT, project FLOAT, exam FLOAT, total FLOAT NOT NULL, grade VARCHAR(3) NOT NULL,
        grade_point FLOAT NOT NULL, teacher_comment TEXT, created_at DATETIME, updated_at DATETIME
    )"""))
    s.execute(__import__('sqlalchemy').text("""INSERT INTO results (id,session_id,student_id,subject_id,ca1,ca2,assignment,project,exam,total,grade,grade_point,created_at,updated_at)
        VALUES (1,1,1,1,20,20,20,20,20,20,'F9',0,'2026-01-01','2026-01-01'),
               (2,1,1,1,70,70,70,70,70,70,'B2',3.7,'2026-02-01','2026-02-01')"""))
    s.commit()
    monkeypatch.setattr('edupredict.results.ensure_result_unique_guard', lambda: True)
    assert len(duplicate_result_groups(s)) == 1
    retired = repair_duplicate_results(s)
    assert retired == 1
    assert len(duplicate_result_groups(s)) == 0
    current = s.execute(__import__('sqlalchemy').text("SELECT total FROM results")).scalar_one()
    assert current == 70
    assert s.query(__import__('edupredict.db', fromlist=['ResultRevision']).ResultRevision).count() == 1
