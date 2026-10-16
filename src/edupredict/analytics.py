"""Academic analytics engine for EduPredict.

The engine is deliberately derived from the transactional database. It does not
create a duplicate broadsheet table. Missing results are represented as missing
values and are never silently converted to zero or failure.
"""
from __future__ import annotations

from typing import Iterable, Optional
import math
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from .db import (
    Student, Result, Attendance, AcademicSession, Subject,
    StudentPlacement, ClassGroup, StudentSubjectEnrollment, Engagement,
)
from .grading import grade_from_score


TERM_ORDER = {"First Term": 1, "Second Term": 2, "Third Term": 3}
DEFAULT_CREDIT_THRESHOLD = 50.0
DEFAULT_PASS_THRESHOLD = 40.0


def _safe_float(v):
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def attendance_pct(att):
    if att and att.school_days and att.school_days > 0:
        return round(att.days_present / att.school_days * 100, 1)
    return None


def _period(s: Session, school_id, session_name, term):
    return s.query(AcademicSession).filter_by(
        school_id=school_id, session_name=session_name, term=term
    ).first()


def _placement_for(s: Session, student_id, session_id):
    return (
        s.query(StudentPlacement, ClassGroup)
        .join(ClassGroup, StudentPlacement.class_group_id == ClassGroup.id)
        .filter(
            StudentPlacement.student_id == student_id,
            StudentPlacement.session_id == session_id,
        )
        .first()
    )


def _period_sort_key(session_name: str, term: str, session_id: int = 0):
    # Academic sessions are normally YYYY/YYYY; nonstandard names are sorted last.
    try:
        year = int(str(session_name).split("/")[0])
    except (ValueError, IndexError):
        year = 999999
    return (year, TERM_ORDER.get(term, 99), session_id)


def period_history(s: Session, school_id: int) -> list[AcademicSession]:
    """Return school periods in chronological order."""
    periods = s.query(AcademicSession).filter_by(school_id=school_id).all()
    return sorted(periods, key=lambda x: _period_sort_key(x.session_name, x.term, x.id))


def _enrollment_frame(s: Session, school_id: int, session_id: int, class_level=None, arm=None):
    q = (
        s.query(Student, StudentPlacement, ClassGroup, StudentSubjectEnrollment, Subject)
        .join(StudentPlacement, StudentPlacement.student_id == Student.id)
        .join(ClassGroup, StudentPlacement.class_group_id == ClassGroup.id)
        .join(
            StudentSubjectEnrollment,
            StudentSubjectEnrollment.student_id == Student.id,
        )
        .join(Subject, StudentSubjectEnrollment.subject_id == Subject.id)
        .filter(
            Student.school_id == school_id,
            Student.active.is_(True),
            StudentPlacement.session_id == session_id,
            StudentSubjectEnrollment.session_id == session_id,
            StudentSubjectEnrollment.active.is_(True),
        )
    )
    if class_level:
        q = q.filter(ClassGroup.class_level == class_level)
    if arm is not None:
        q = q.filter(ClassGroup.arm == arm)
    rows = []
    for st, _pl, cg, en, sub in q.all():
        rows.append({
            "student_id": st.id,
            "student_code": st.student_code,
            "student_name": f"{st.first_name} {st.last_name}".strip(),
            "class_level": cg.class_level,
            "arm": cg.arm or "",
            "subject_id": sub.id,
            "subject_code": sub.code,
            "subject": sub.name,
        })
    return pd.DataFrame(rows)


def result_records(
    s: Session, school_id, session_name=None, term=None,
    class_level=None, arm=None, subject_code=None,
):
    q = (
        s.query(Result, Student, Subject, AcademicSession)
        .join(Student, Result.student_id == Student.id)
        .join(Subject, Result.subject_id == Subject.id)
        .join(AcademicSession, Result.session_id == AcademicSession.id)
        .filter(Student.school_id == school_id, Student.active.is_(True))
    )
    if session_name:
        q = q.filter(AcademicSession.session_name == session_name)
    if term:
        q = q.filter(AcademicSession.term == term)
    rows = []
    for r, st, sub, sess in q.all():
        placement = _placement_for(s, st.id, sess.id)
        actual_class = placement[1].class_level if placement else st.class_level
        actual_arm = (placement[1].arm or "") if placement else (st.arm or "")
        if class_level and actual_class != class_level:
            continue
        if arm is not None and actual_arm != arm:
            continue
        if subject_code and sub.code != subject_code:
            continue
        rows.append({
            "student_id": st.id, "student_code": st.student_code,
            "student_name": f"{st.first_name} {st.last_name}".strip(),
            "class_level": actual_class, "arm": actual_arm,
            "session_id": sess.id, "session": sess.session_name, "term": sess.term,
            "subject_id": sub.id, "subject_code": sub.code, "subject": sub.name,
            "ca1": r.ca1, "ca2": r.ca2, "assignment": r.assignment,
            "project": r.project, "exam": r.exam, "total": r.total,
            "grade": r.grade, "grade_point": r.grade_point,
        })
    return pd.DataFrame(rows)


def _merge_results(enrollment_df: pd.DataFrame, result_df: pd.DataFrame) -> pd.DataFrame:
    if enrollment_df.empty:
        return enrollment_df.copy()
    if result_df.empty:
        out = enrollment_df.copy()
        for c in ["ca1", "ca2", "assignment", "project", "exam", "total", "grade", "grade_point"]:
            out[c] = np.nan
        return out
    cols = ["student_id", "subject_id", "ca1", "ca2", "assignment", "project", "exam", "total", "grade", "grade_point"]
    return enrollment_df.merge(result_df[cols], on=["student_id", "subject_id"], how="left")


def _grade_info(score, credit_threshold=DEFAULT_CREDIT_THRESHOLD, pass_threshold=DEFAULT_PASS_THRESHOLD):
    x = _safe_float(score)
    if x is None:
        return "No result", False, False, None
    grade, gp = grade_from_score(x)
    return grade, x >= credit_threshold, x >= pass_threshold, gp


def analytics_dataset(
    s: Session, school_id: int, session_id: int,
    class_level=None, arm=None,
    credit_threshold: float = DEFAULT_CREDIT_THRESHOLD,
    pass_threshold: float = DEFAULT_PASS_THRESHOLD,
) -> pd.DataFrame:
    """Return one row per enrolled student-subject with result, attendance and grade fields."""
    sess = s.get(AcademicSession, session_id)
    if not sess:
        return pd.DataFrame()
    enr = _enrollment_frame(s, school_id, session_id, class_level, arm)
    rdf = result_records(s, school_id, sess.session_name, sess.term, class_level, arm)
    out = _merge_results(enr, rdf)
    if out.empty:
        return out
    out["result_recorded"] = out["total"].notna()
    out["credit_pass"] = out["total"].ge(credit_threshold) & out["total"].notna()
    out["pass"] = out["total"].ge(pass_threshold) & out["total"].notna()
    out["below_credit"] = out["total"].lt(credit_threshold) & out["total"].notna()
    out["grade"] = out.apply(lambda r: _grade_info(r["total"], credit_threshold, pass_threshold)[0], axis=1)
    out["grade_point"] = out.apply(lambda r: _grade_info(r["total"], credit_threshold, pass_threshold)[3], axis=1)

    att_rows = []
    att_q = s.query(Attendance).filter(Attendance.session_id == session_id)
    for a in att_q.all():
        att_rows.append({"student_id": a.student_id, "attendance_pct": attendance_pct(a), "school_days": a.school_days, "days_present": a.days_present, "days_absent": a.days_absent})
    if att_rows:
        out = out.merge(pd.DataFrame(att_rows), on="student_id", how="left")
    else:
        out["attendance_pct"] = np.nan
        out["school_days"] = np.nan
        out["days_present"] = np.nan
        out["days_absent"] = np.nan

    eng_rows = []
    for e in s.query(Engagement).filter_by(session_id=session_id).all():
        eng_rows.append({
            "student_id": e.student_id, "homework_rate": e.homework_rate,
            "participation": e.participation, "behaviour": e.behaviour,
            "study_hours_week": e.study_hours_week,
        })
    if eng_rows:
        out = out.merge(pd.DataFrame(eng_rows), on="student_id", how="left")
    else:
        for c in ["homework_rate", "participation", "behaviour", "study_hours_week"]:
            out[c] = np.nan
    return out


def student_period_metrics(
    s: Session, school_id: int, student_id: int, session_id: int,
    credit_threshold: float = DEFAULT_CREDIT_THRESHOLD,
    pass_threshold: float = DEFAULT_PASS_THRESHOLD,
) -> dict:
    """Compute a complete student-period academic profile."""
    df = analytics_dataset(s, school_id, session_id, credit_threshold=credit_threshold, pass_threshold=pass_threshold)
    df = df[df.student_id == student_id].copy() if not df.empty else df
    if df.empty:
        return {
            "student_id": student_id, "subjects_enrolled": 0, "results_recorded": 0,
            "completion_rate": 0.0, "average": None, "median": None,
            "lowest": None, "highest": None, "std_dev": None,
            "credit_pass_count": 0, "below_credit_count": 0, "pass_count": 0,
            "fail_count": 0, "grade_point_average": None, "attendance_pct": None,
            "missing_subjects": [], "weak_subjects": [], "strong_subjects": [],
        }
    scores = pd.to_numeric(df["total"], errors="coerce")
    recorded = scores.dropna()
    missing = df.loc[scores.isna(), "subject"].tolist()
    weak = df.loc[scores.notna() & scores.lt(credit_threshold), ["subject", "total", "grade"]].to_dict("records")
    strong = df.loc[scores.notna() & scores.ge(75), ["subject", "total", "grade"]].to_dict("records")
    att = df["attendance_pct"].dropna()
    gp = pd.to_numeric(df["grade_point"], errors="coerce").dropna()
    return {
        "student_id": student_id,
        "subjects_enrolled": int(len(df)),
        "results_recorded": int(recorded.size),
        "completion_rate": round(recorded.size / len(df) * 100, 1),
        "average": round(float(recorded.mean()), 2) if not recorded.empty else None,
        "median": round(float(recorded.median()), 2) if not recorded.empty else None,
        "lowest": round(float(recorded.min()), 2) if not recorded.empty else None,
        "highest": round(float(recorded.max()), 2) if not recorded.empty else None,
        "std_dev": round(float(recorded.std(ddof=0)), 2) if recorded.size else None,
        "credit_pass_count": int(df["credit_pass"].sum()),
        "below_credit_count": int(df["below_credit"].sum()),
        "pass_count": int(df["pass"].sum()),
        "fail_count": int((scores.lt(pass_threshold) & scores.notna()).sum()),
        "grade_point_average": round(float(gp.mean()), 2) if not gp.empty else None,
        "attendance_pct": round(float(att.mean()), 1) if not att.empty else None,
        "missing_subjects": missing,
        "weak_subjects": weak,
        "strong_subjects": strong,
    }


def student_subject_report(s: Session, school_id, student_id, session_id):
    student = s.get(Student, student_id)
    if not student:
        return pd.DataFrame()
    df = analytics_dataset(s, school_id, session_id)
    df = df[df.student_id == student_id].copy() if not df.empty else df
    if df.empty:
        return pd.DataFrame()
    cols = ["subject_code", "subject", "ca1", "ca2", "assignment", "project", "exam", "total", "grade", "grade_point", "result_recorded"]
    out = df[cols].copy()
    out.columns = ["Code", "Subject", "CA1", "CA2", "Assignment", "Project", "Exam", "Final Score", "Grade", "Grade Point", "Result Recorded"]
    out.loc[~out["Result Recorded"], "Grade"] = "No result"
    return out


def subject_analytics(
    s: Session, school_id: int, session_id: int,
    class_level=None, arm=None,
    credit_threshold: float = DEFAULT_CREDIT_THRESHOLD,
    pass_threshold: float = DEFAULT_PASS_THRESHOLD,
) -> pd.DataFrame:
    """Subject-level performance, completion, grade and assessment-component analytics."""
    df = analytics_dataset(s, school_id, session_id, class_level, arm, credit_threshold, pass_threshold)
    if df.empty:
        return pd.DataFrame()
    rows = []
    for (sid, code, name), g in df.groupby(["subject_id", "subject_code", "subject"], dropna=False):
        scores = pd.to_numeric(g["total"], errors="coerce").dropna()
        recorded = int(scores.size)
        enrolled = int(len(g))
        grade_counts = g.loc[g.result_recorded, "grade"].value_counts()
        rows.append({
            "Subject Code": code, "Subject": name,
            "Enrolled": enrolled, "Results": recorded,
            "Completion %": round(recorded / enrolled * 100, 1) if enrolled else 0,
            "Average %": round(float(scores.mean()), 2) if recorded else None,
            "Median %": round(float(scores.median()), 2) if recorded else None,
            "Minimum %": round(float(scores.min()), 2) if recorded else None,
            "Maximum %": round(float(scores.max()), 2) if recorded else None,
            "Std Dev": round(float(scores.std(ddof=0)), 2) if recorded else None,
            "Credit Pass %": round(float(scores.ge(credit_threshold).mean() * 100), 1) if recorded else None,
            "Pass %": round(float(scores.ge(pass_threshold).mean() * 100), 1) if recorded else None,
            "A1 %": round(float((scores.ge(75)).mean() * 100), 1) if recorded else None,
            "F9 %": round(float((scores.lt(40)).mean() * 100), 1) if recorded else None,
            "CA1 Avg": round(float(pd.to_numeric(g.ca1, errors="coerce").mean()), 2) if g.ca1.notna().any() else None,
            "CA2 Avg": round(float(pd.to_numeric(g.ca2, errors="coerce").mean()), 2) if g.ca2.notna().any() else None,
            "Assignment Avg": round(float(pd.to_numeric(g.assignment, errors="coerce").mean()), 2) if g.assignment.notna().any() else None,
            "Project Avg": round(float(pd.to_numeric(g.project, errors="coerce").mean()), 2) if g.project.notna().any() else None,
            "Exam Avg": round(float(pd.to_numeric(g.exam, errors="coerce").mean()), 2) if g.exam.notna().any() else None,
            "Grade Distribution": ", ".join(f"{k}:{int(v)}" for k, v in grade_counts.items()),
        })
    return pd.DataFrame(rows).sort_values(["Average %", "Subject"], ascending=[True, True], na_position="last").reset_index(drop=True)


def class_analytics(
    s: Session, school_id: int, session_id: int,
    class_level=None, arm=None,
    credit_threshold: float = DEFAULT_CREDIT_THRESHOLD,
    pass_threshold: float = DEFAULT_PASS_THRESHOLD,
) -> dict:
    """Class/stream cohort analytics including student averages and attendance."""
    df = analytics_dataset(s, school_id, session_id, class_level, arm, credit_threshold, pass_threshold)
    if df.empty:
        return {
            "students": 0, "subjects": 0, "enrolled_records": 0, "results_recorded": 0,
            "completion_rate": 0.0, "average": None, "median": None, "std_dev": None,
            "credit_rate": None, "pass_rate": None, "attendance_average": None,
            "student_table": pd.DataFrame(), "subject_table": pd.DataFrame(),
        }
    scores = pd.to_numeric(df.total, errors="coerce")
    student_avg = df.groupby(["student_id", "student_code", "student_name"], as_index=False)["total"].mean()
    student_avg = student_avg.rename(columns={"total": "Average %"})
    student_avg["Average %"] = student_avg["Average %"].round(2)
    student_avg["Rank"] = student_avg["Average %"].rank(method="min", ascending=False).astype("Int64")
    n_students = int(df.student_id.nunique())
    recorded = int(scores.notna().sum())
    enrolled = int(len(df))
    recorded_scores = scores.dropna()
    att = df.groupby("student_id")["attendance_pct"].first().dropna()
    student_avg["Performance Band"] = student_avg["Average %"].map(lambda x: grade_from_score(x)[0] if pd.notna(x) else "No result")
    return {
        "students": n_students,
        "subjects": int(df.subject_id.nunique()),
        "enrolled_records": enrolled,
        "results_recorded": recorded,
        "completion_rate": round(recorded / enrolled * 100, 1) if enrolled else 0.0,
        "average": round(float(recorded_scores.mean()), 2) if not recorded_scores.empty else None,
        "median": round(float(recorded_scores.median()), 2) if not recorded_scores.empty else None,
        "std_dev": round(float(recorded_scores.std(ddof=0)), 2) if not recorded_scores.empty else None,
        "credit_rate": round(float(recorded_scores.ge(credit_threshold).mean() * 100), 1) if not recorded_scores.empty else None,
        "pass_rate": round(float(recorded_scores.ge(pass_threshold).mean() * 100), 1) if not recorded_scores.empty else None,
        "attendance_average": round(float(att.mean()), 1) if not att.empty else None,
        "student_table": student_avg.sort_values(["Rank", "student_code"]).reset_index(drop=True),
        "subject_table": subject_analytics(s, school_id, session_id, class_level, arm, credit_threshold, pass_threshold),
    }


def class_overview(s: Session, school_id: int, session_id=None, credit_threshold=DEFAULT_CREDIT_THRESHOLD):
    students = s.query(Student).filter_by(school_id=school_id, active=True).all()
    ids = [x.id for x in students]
    if not ids:
        return {"students": 0, "students_with_results": 0, "average": 0.0, "below_credit": 0, "results_below_credit": 0}
    if session_id:
        ca = class_analytics(s, school_id, session_id, credit_threshold=credit_threshold)
        st = ca["student_table"]
        # analytics_dataset() is allowed to be empty when the period has no
        # subject enrollments yet. Missing enrollment/result data is not a failure.
        adf = analytics_dataset(s, school_id, session_id)
        results_below_credit = int(adf["below_credit"].sum()) if "below_credit" in adf.columns else 0
        return {
            "students": len(students),
            "students_with_results": int(st["Average %"].notna().sum()) if not st.empty else 0,
            "average": ca["average"] or 0.0,
            "below_credit": int((st["Average %"] < credit_threshold).sum()) if not st.empty else 0,
            "results_below_credit": results_below_credit,
        }
    q = s.query(Result).filter(Result.student_id.in_(ids))
    results = q.all()
    if not results:
        return {"students": len(students), "students_with_results": 0, "average": 0.0, "below_credit": 0, "results_below_credit": 0}
    rdf = pd.DataFrame([{"student_id": r.student_id, "total": r.total} for r in results])
    student_avg = rdf.groupby("student_id", as_index=False).total.mean()
    return {
        "students": len(students), "students_with_results": len(student_avg),
        "average": round(float(student_avg.total.mean()), 1),
        "below_credit": int((student_avg.total < credit_threshold).sum()),
        "results_below_credit": int((rdf.total < credit_threshold).sum()),
    }


def trend_analytics(
    s: Session, school_id: int, student_id: int | None = None,
    class_level=None, arm=None, subject_code=None,
    credit_threshold: float = DEFAULT_CREDIT_THRESHOLD,
) -> pd.DataFrame:
    """Chronological performance trend by term, with change from previous period."""
    periods = period_history(s, school_id)
    rows = []
    for sess in periods:
        df = analytics_dataset(s, school_id, sess.id, class_level, arm, credit_threshold=credit_threshold)
        if student_id is not None:
            df = df[df.student_id == student_id]
        if subject_code is not None and not df.empty:
            df = df[df.subject_code == subject_code]
        if df.empty:
            continue
        scores = pd.to_numeric(df.total, errors="coerce").dropna()
        if scores.empty:
            avg = None
        else:
            avg = float(scores.mean())
        rows.append({
            "session_id": sess.id, "Session": sess.session_name, "Term": sess.term,
            "Period": f"{sess.session_name} — {sess.term}",
            "Average %": round(avg, 2) if avg is not None else None,
            "Results": int(scores.size), "Enrolled": int(len(df)),
            "Completion %": round(scores.size / len(df) * 100, 1) if len(df) else 0,
            "Credit Pass %": round(float(scores.ge(credit_threshold).mean() * 100), 1) if not scores.empty else None,
            "Attendance %": round(float(df.groupby("student_id").attendance_pct.first().dropna().mean()), 1) if df.attendance_pct.notna().any() else None,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["Change vs Previous"] = out["Average %"].diff().round(2)
    out["Direction"] = out["Change vs Previous"].map(lambda x: "Improving" if pd.notna(x) and x > 0.5 else ("Declining" if pd.notna(x) and x < -0.5 else ("Stable" if pd.notna(x) else "—")))
    return out


def subject_trends(s: Session, school_id: int, subject_code: str, class_level=None, arm=None, credit_threshold=DEFAULT_CREDIT_THRESHOLD):
    return trend_analytics(s, school_id, class_level=class_level, arm=arm, subject_code=subject_code, credit_threshold=credit_threshold)


def student_trend_by_subject(s: Session, school_id: int, student_id: int, credit_threshold=DEFAULT_CREDIT_THRESHOLD):
    periods = period_history(s, school_id)
    rows = []
    for sess in periods:
        df = analytics_dataset(s, school_id, sess.id, credit_threshold=credit_threshold)
        if df.empty:
            continue
        df = df[df.student_id == student_id]
        for sub, g in df.groupby(["subject_code", "subject"]):
            scores = pd.to_numeric(g.total, errors="coerce").dropna()
            rows.append({
                "session_id": sess.id, "Session": sess.session_name, "Term": sess.term,
                "Subject Code": sub[0], "Subject": sub[1],
                "Score %": round(float(scores.mean()), 2) if not scores.empty else None,
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["Subject Code", "session_id"])
    out["Change"] = out.groupby("Subject Code")["Score %"].diff().round(2)
    out["Direction"] = out["Change"].map(lambda x: "Improving" if pd.notna(x) and x > 0.5 else ("Declining" if pd.notna(x) and x < -0.5 else ("Stable" if pd.notna(x) else "—")))
    return out.reset_index(drop=True)


def performance_distribution(scores: Iterable[float], credit_threshold=DEFAULT_CREDIT_THRESHOLD, pass_threshold=DEFAULT_PASS_THRESHOLD) -> dict:
    vals = pd.to_numeric(pd.Series(list(scores)), errors="coerce").dropna()
    if vals.empty:
        return {"count": 0, "bands": {}, "mean": None, "median": None}
    bands = {"A1": 0, "B2": 0, "B3": 0, "C4": 0, "C5": 0, "C6": 0, "D7": 0, "E8": 0, "F9": 0}
    for x in vals:
        bands[grade_from_score(x)[0]] += 1
    return {
        "count": int(vals.size), "bands": bands,
        "mean": round(float(vals.mean()), 2), "median": round(float(vals.median()), 2),
        "pass_rate": round(float(vals.ge(pass_threshold).mean() * 100), 1),
        "credit_rate": round(float(vals.ge(credit_threshold).mean() * 100), 1),
    }


def ranking_table(class_result: pd.DataFrame) -> pd.DataFrame:
    """Rank students by average, preserving ties and excluding no-result students."""
    if class_result.empty:
        return class_result.copy()
    out = class_result.copy()
    out["Rank"] = out["Average %"].rank(method="min", ascending=False).astype("Int64")
    n = int(out["Average %"].notna().sum())
    out["Percentile"] = out["Average %"].rank(pct=True, ascending=True).mul(100).round(1) if n else np.nan
    out["Class Size"] = n
    return out.sort_values(["Rank", "student_code"]).reset_index(drop=True)


def identify_outliers(s: Session, school_id: int, session_id: int, class_level=None, arm=None, subject_code=None) -> pd.DataFrame:
    """Flag unusually low/high scores using IQR within the selected cohort/subject."""
    df = analytics_dataset(s, school_id, session_id, class_level, arm)
    if subject_code and not df.empty:
        df = df[df.subject_code == subject_code]
    empty_cols = ["Student ID", "Student", "Subject Code", "Subject", "Score %", "Outlier", "IQR Low", "IQR High"]
    if df.empty:
        return pd.DataFrame(columns=empty_cols)
    rows = []
    for (code, name), g in df.groupby(["subject_code", "subject"]):
        vals = pd.to_numeric(g.total, errors="coerce")
        vals = vals.dropna()
        if len(vals) < 4:
            continue
        q1, q3 = vals.quantile([0.25, 0.75]); iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        for _, r in g[g.total.notna()].iterrows():
            if r.total < low or r.total > high:
                rows.append({"Student ID": r.student_code, "Student": r.student_name, "Subject Code": code, "Subject": name, "Score %": r.total, "Outlier": "Low" if r.total < low else "High", "IQR Low": round(float(low), 2), "IQR High": round(float(high), 2)})
    return pd.DataFrame(rows, columns=empty_cols)


def broadsheet(s: Session, school_id, session_name=None, term=None, class_level=None, arm=None, subject_code=None):
    """Derived class broadsheet. Enrolled students/subjects are the population; missing results stay blank."""
    if session_name and term:
        sess = _period(s, school_id, session_name, term)
        if not sess:
            return pd.DataFrame()
        df = analytics_dataset(s, school_id, sess.id, class_level, arm)
        if subject_code and not df.empty:
            df = df[df.subject_code == subject_code]
    else:
        df = result_records(s, school_id, session_name, term, class_level, arm, subject_code)
        if df.empty:
            return df
        df["result_recorded"] = True
        df["attendance_pct"] = np.nan
    if df.empty:
        return pd.DataFrame()
    idx = ["student_id", "student_code", "student_name", "class_level", "arm"]
    pivot = df.pivot(index=idx, columns="subject", values="total").reset_index()
    subject_cols = [c for c in pivot.columns if c not in set(idx)]
    numeric = pivot[subject_cols].apply(pd.to_numeric, errors="coerce")
    pivot["Average"] = numeric.mean(axis=1).round(1)
    pivot["Subjects Enrolled"] = df.groupby("student_id").subject_id.nunique().reindex(pivot.student_id).fillna(0).astype(int).to_numpy()
    pivot["Subjects Recorded"] = numeric.notna().sum(axis=1)
    pivot["Completion %"] = (pivot["Subjects Recorded"] / pivot["Subjects Enrolled"].replace(0, np.nan) * 100).round(1)
    pivot["Below Credit"] = (numeric < DEFAULT_CREDIT_THRESHOLD).sum(axis=1)
    pivot["Credit Pass %"] = (numeric.ge(DEFAULT_CREDIT_THRESHOLD).sum(axis=1) / numeric.notna().sum(axis=1).replace(0, np.nan) * 100).round(1)
    pivot["Performance Band"] = pivot["Average"].map(lambda x: grade_from_score(x)[0] if pd.notna(x) else "No result")
    pivot["S/N"] = range(1, len(pivot) + 1)
    att = df.groupby("student_id")["attendance_pct"].first().reindex(pivot.student_id)
    pivot["Attendance %"] = att.to_numpy()
    ordered = ["S/N", "student_code", "student_name", "class_level", "arm"] + subject_cols + ["Average", "Performance Band", "Subjects Enrolled", "Subjects Recorded", "Completion %", "Below Credit", "Credit Pass %", "Attendance %"]
    return pivot[ordered].sort_values(["class_level", "arm", "student_code"]).reset_index(drop=True)


def data_quality_summary(s: Session, school_id: int, session_id: int) -> dict:
    """Measure completeness and integrity indicators used to qualify analytics."""
    df = analytics_dataset(s, school_id, session_id)
    if df.empty:
        return {"enrolled_subject_records": 0, "result_records": 0, "completion_rate": 0.0, "missing_results": 0, "attendance_coverage": 0.0, "engagement_coverage": 0.0, "duplicate_result_keys": 0}
    result_records_count = int(df.result_recorded.sum())
    attendance_coverage = float(df.groupby("student_id").attendance_pct.first().notna().mean() * 100)
    engagement_cols = ["homework_rate", "participation", "behaviour", "study_hours_week"]
    engagement_coverage = float(df.groupby("student_id")[engagement_cols].first().notna().any(axis=1).mean() * 100)
    duplicate_result_keys = int(df[df.result_recorded].duplicated(["student_id", "subject_id"], keep=False).sum())
    return {
        "enrolled_subject_records": int(len(df)),
        "result_records": result_records_count,
        "completion_rate": round(result_records_count / len(df) * 100, 1),
        "missing_results": int(len(df) - result_records_count),
        "attendance_coverage": round(attendance_coverage, 1),
        "engagement_coverage": round(engagement_coverage, 1),
        "duplicate_result_keys": duplicate_result_keys,
    }
