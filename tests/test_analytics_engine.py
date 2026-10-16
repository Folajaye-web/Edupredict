import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from edupredict.db import (
    Base, School, AcademicSession, ClassGroup, Student, StudentPlacement,
    Subject, StudentSubjectEnrollment, Result, Attendance, Engagement,
)
from edupredict.analytics import (
    analytics_dataset, student_period_metrics, subject_analytics,
    class_analytics, trend_analytics, student_trend_by_subject,
    performance_distribution, ranking_table, identify_outliers,
    data_quality_summary, broadsheet,
)


def make_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    S = sessionmaker(bind=engine, expire_on_commit=False)
    s = S()
    school = School(name="Test School", state="Lagos", lga="Test", school_type="Private")
    s.add(school); s.flush()
    p1 = AcademicSession(school_id=school.id, session_name="2025/2026", term="Third Term", is_current=False)
    p2 = AcademicSession(school_id=school.id, session_name="2026/2027", term="First Term", is_current=True)
    s.add_all([p1, p2]); s.flush()
    cg1 = ClassGroup(school_id=school.id, session_id=p1.id, class_level="SSS2", arm="A", display_name="SSS2 A")
    cg2 = ClassGroup(school_id=school.id, session_id=p2.id, class_level="SSS2", arm="A", display_name="SSS2 A")
    s.add_all([cg1, cg2]); s.flush()
    students = [Student(school_id=school.id, student_code=f"T{i}", first_name=f"Student{i}", last_name="Test", class_level="SSS2", arm="A") for i in range(1,5)]
    s.add_all(students); s.flush()
    for cg in [cg1, cg2]:
        for st in students:
            s.add(StudentPlacement(session_id=cg.session_id, student_id=st.id, class_group_id=cg.id))
    subs = [Subject(code="MATH", name="Mathematics", stage="SSS", compulsory=True), Subject(code="PHY", name="Physics", stage="SSS")]
    s.add_all(subs); s.flush()
    for sess in [p1, p2]:
        for st in students:
            for sub in subs:
                s.add(StudentSubjectEnrollment(session_id=sess.id, student_id=st.id, subject_id=sub.id, active=True))
        for st in students:
            s.add(Attendance(session_id=sess.id, student_id=st.id, school_days=100, days_present=90 if st.id != students[-1].id else 60, days_absent=10 if st.id != students[-1].id else 40))
            s.add(Engagement(session_id=sess.id, student_id=st.id, homework_rate=80, participation=75, behaviour=90, study_hours_week=8))
    scores1 = [(80,70),(60,55),(45,50),(35,40)]
    scores2 = [(90,80),(70,65),(45,50),(35,40)]
    for st, (m,p) in zip(students, scores1):
        s.add(Result(session_id=p1.id, student_id=st.id, subject_id=subs[0].id, ca1=m, ca2=m, assignment=m, project=m, exam=m, total=m, grade="A1", grade_point=4))
        s.add(Result(session_id=p1.id, student_id=st.id, subject_id=subs[1].id, ca1=p, ca2=p, assignment=p, project=p, exam=p, total=p, grade="B2", grade_point=3.7))
    for st, (m,p) in zip(students, scores2):
        s.add(Result(session_id=p2.id, student_id=st.id, subject_id=subs[0].id, ca1=m, ca2=m, assignment=m, project=m, exam=m, total=m, grade="A1", grade_point=4))
        s.add(Result(session_id=p2.id, student_id=st.id, subject_id=subs[1].id, ca1=p, ca2=p, assignment=p, project=p, exam=p, total=p, grade="B2", grade_point=3.7))
    s.commit()
    return s, school, p1, p2, students, subs


def test_full_analytics_engine():
    s, school, p1, p2, students, subs = make_db()
    df = analytics_dataset(s, school.id, p2.id)
    assert len(df) == 8
    assert df.result_recorded.sum() == 8
    m = student_period_metrics(s, school.id, students[0].id, p2.id)
    assert m["average"] == 85.0
    assert m["completion_rate"] == 100.0
    assert m["attendance_pct"] == 90.0
    assert len(m["weak_subjects"]) == 0

    sa = subject_analytics(s, school.id, p2.id)
    assert set(sa["Subject Code"]) == {"MATH", "PHY"}
    assert all(sa["Completion %"] == 100.0)

    ca = class_analytics(s, school.id, p2.id, "SSS2", "A")
    assert ca["students"] == 4
    assert ca["subjects"] == 2
    assert ca["average"] == 59.38
    ranked = ranking_table(ca["student_table"])
    assert ranked.iloc[0]["Rank"] == 1

    tr = trend_analytics(s, school.id, class_level="SSS2", arm="A")
    assert len(tr) == 2
    assert tr.iloc[1]["Direction"] == "Improving"

    strend = student_trend_by_subject(s, school.id, students[0].id)
    assert len(strend) == 4
    assert "Improving" in set(strend["Direction"].dropna())

    dist = performance_distribution([85,65,40,30])
    assert dist["count"] == 4
    assert dist["bands"]["A1"] == 1

    out = identify_outliers(s, school.id, p2.id, "SSS2", "A", "MATH")
    assert "Outlier" in out.columns

    dq = data_quality_summary(s, school.id, p2.id)
    assert dq["completion_rate"] == 100.0
    assert dq["attendance_coverage"] == 100.0
    assert dq["engagement_coverage"] == 100.0

    bs = broadsheet(s, school.id, p2.session_name, p2.term, "SSS2", "A")
    assert set(["Average", "Completion %", "Attendance %"]).issubset(bs.columns)
    assert len(bs) == 4


def test_class_overview_handles_period_with_no_subject_enrollment():
    from edupredict.analytics import class_overview
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    S = sessionmaker(bind=engine, expire_on_commit=False)
    s = S()
    school = School(name="No Enrollment School", state="Lagos", lga="Test", school_type="Private")
    s.add(school); s.flush()
    period = AcademicSession(school_id=school.id, session_name="2026/2027", term="First Term", is_current=True)
    student = Student(school_id=school.id, student_code="NE001", first_name="No", last_name="Enrollment", class_level="SSS2", arm="Science")
    s.add_all([period, student]); s.commit()
    overview = class_overview(s, school.id, period.id)
    assert overview["students"] == 1
    assert overview["students_with_results"] == 0
    assert overview["below_credit"] == 0
    assert overview["results_below_credit"] == 0
