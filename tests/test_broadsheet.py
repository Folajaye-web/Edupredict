import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from edupredict.db import Base, School, AcademicSession, ClassGroup, Student, StudentPlacement, Subject, StudentSubjectEnrollment, Result, Attendance
from edupredict.broadsheet import build_broadsheet, build_broadsheet_display, broadsheet_missing_results, broadsheet_kpis


def make_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    S = sessionmaker(bind=engine, expire_on_commit=False)
    s = S()
    school = School(name="Test School", state="Lagos", lga="Test", school_type="Private")
    s.add(school); s.flush()
    period = AcademicSession(school_id=school.id, session_name="2026/2027", term="Second Term", is_current=True)
    s.add(period); s.flush()
    group = ClassGroup(school_id=school.id, session_id=period.id, class_level="SSS2", arm="A", display_name="SSS2 A")
    s.add(group); s.flush()
    students = [
        Student(school_id=school.id, student_code="T001", first_name="Ada", last_name="One", class_level="SSS2", arm="A"),
        Student(school_id=school.id, student_code="T002", first_name="Bola", last_name="Two", class_level="SSS2", arm="A"),
        Student(school_id=school.id, student_code="T003", first_name="Chidi", last_name="Three", class_level="SSS2", arm="A"),
    ]
    s.add_all(students); s.flush()
    for st in students:
        s.add(StudentPlacement(session_id=period.id, student_id=st.id, class_group_id=group.id))
        s.add(Attendance(session_id=period.id, student_id=st.id, school_days=60, days_present=55, days_absent=5))
    math = Subject(code="MATH", name="Mathematics", stage="SSS", compulsory=True)
    phy = Subject(code="PHY", name="Physics", stage="SSS", compulsory=False)
    chem = Subject(code="CHEM", name="Chemistry", stage="SSS", compulsory=False)
    s.add_all([math, phy, chem]); s.flush()
    # T001: all three enrolled, Chemistry missing.
    for sub in [math, phy, chem]: s.add(StudentSubjectEnrollment(session_id=period.id, student_id=students[0].id, subject_id=sub.id, active=True))
    # T002: only Mathematics and Physics; Chemistry is N/E.
    for sub in [math, phy]: s.add(StudentSubjectEnrollment(session_id=period.id, student_id=students[1].id, subject_id=sub.id, active=True))
    # T003: enrolled Mathematics but no result at all.
    s.add(StudentSubjectEnrollment(session_id=period.id, student_id=students[2].id, subject_id=math.id, active=True))
    def result(st, sub, score):
        return Result(session_id=period.id, student_id=st.id, subject_id=sub.id, ca1=score, ca2=score, assignment=score, project=score, exam=score, total=score, grade="A1", grade_point=4)
    s.add_all([result(students[0], math, 80), result(students[0], phy, 60), result(students[1], math, 70), result(students[1], phy, 50)])
    s.commit()
    return s, school, period


def test_broadsheet_distinguishes_missing_and_not_enrolled():
    s, school, period = make_db()
    numeric = build_broadsheet(s, school.id, period.session_name, period.term, "SSS2", "A")
    assert len(numeric) == 3
    assert "Chemistry" in numeric.columns
    t001 = numeric[numeric["Student ID"] == "T001"].iloc[0]
    assert pd.isna(t001["Chemistry"])  # enrolled but missing
    assert t001["Missing Results"] == 1
    assert t001["Subjects Enrolled"] == 3
    t002 = numeric[numeric["Student ID"] == "T002"].iloc[0]
    assert pd.isna(t002["Chemistry"])  # numeric frame uses NaN for N/E too; display resolves state
    assert t002["Subjects Enrolled"] == 2
    display = build_broadsheet_display(s, school.id, period.session_name, period.term, "SSS2", "A")
    d1 = display[display["Student ID"] == "T001"].iloc[0]
    d2 = display[display["Student ID"] == "T002"].iloc[0]
    assert d1["Chemistry"] == "—"
    assert d2["Chemistry"] == "N/E"
    assert d1["Average"] == "70.0"
    assert d1["Ranking Status"] == "Ranked — incomplete"
    assert d2["Ranking Status"] == "Ranked — complete"


def test_broadsheet_missing_report_and_kpis():
    s, school, period = make_db()
    missing = broadsheet_missing_results(s, school.id, period.session_name, period.term, "SSS2", "A")
    assert len(missing) == 2
    assert set(missing["Student ID"]) == {"T001", "T003"}
    bs = build_broadsheet(s, school.id, period.session_name, period.term, "SSS2", "A")
    k = broadsheet_kpis(bs)
    assert k["students"] == 3
    assert k["subjects"] == 3
    assert k["enrolled_records"] == 6
    assert k["recorded_results"] == 4
    assert k["missing_results"] == 2
    assert round(k["completion_rate"], 1) == round(4/6*100, 1)
