import json
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from edupredict.db import Base, School, AcademicSession, Student, Subject, StudentSubjectEnrollment, ContinuousAssessment, FinalExamination
from edupredict.ai.features import build_pre_exam_features

def test_pre_exam_features_exclude_current_exam():
    e=create_engine("sqlite:///:memory:")
    Base.metadata.create_all(e)
    S=sessionmaker(bind=e,expire_on_commit=False); s=S()
    school=School(name="Boundary School",state="Lagos"); s.add(school); s.flush()
    period=AcademicSession(school_id=school.id,session_name="2026/2027",term="First Term",is_current=True); s.add(period); s.flush()
    student=Student(school_id=school.id,student_code="BOUND001",first_name="Test",last_name="Boundary",class_level="SSS2"); sub=Subject(code="MATH",name="Mathematics",stage="SSS",compulsory=True); s.add_all([student,sub]); s.flush()
    s.add(StudentSubjectEnrollment(session_id=period.id,student_id=student.id,subject_id=sub.id,active=True))
    s.add(ContinuousAssessment(session_id=period.id,student_id=student.id,subject_id=sub.id,ca1=70,ca2=60,assignment=80,project=70))
    s.add(FinalExamination(session_id=period.id,student_id=student.id,subject_id=sub.id,exam_score=99))
    s.commit()
    f=build_pre_exam_features(s,session_id=period.id,student_id=student.id,subject_id=sub.id)
    assert "exam" not in f and "exam_score" not in f and "G3" not in f
    assert f["_current_ca"]==70.0
