import json
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from edupredict.db import Base, School, AcademicSession, ClassGroup, Student, Subject, StudentPlacement, StudentSubjectEnrollment, ContinuousAssessment, Attendance, Engagement, Result
from edupredict.services.engagement_service import upsert_engagement, get_engagement
from edupredict.ai.native import build_native_features
from edupredict.ingest import upsert_students

def setup():
    e=create_engine('sqlite:///:memory:')
    Base.metadata.create_all(e)
    S=sessionmaker(bind=e,expire_on_commit=False)
    s=S()
    school=School(name='Phase5',state='Lagos'); s.add(school); s.flush()
    period=AcademicSession(school_id=school.id,session_name='2026/2027',term='First Term'); s.add(period); s.flush()
    group=ClassGroup(school_id=school.id,session_id=period.id,class_level='SSS2',arm='A',display_name='SSS2 A'); s.add(group)
    st=Student(school_id=school.id,student_code='P5-001',first_name='Ada',last_name='Test',class_level='SSS2',arm='A'); sub=Subject(code='MATH',name='Mathematics',stage='SSS'); s.add_all([st,sub]); s.flush()
    s.add_all([StudentPlacement(session_id=period.id,student_id=st.id,class_group_id=group.id),StudentSubjectEnrollment(session_id=period.id,student_id=st.id,subject_id=sub.id)])
    s.commit()
    return s,school,period,st,sub

def test_engagement_upsert_is_idempotent_and_validates():
    s,school,p,st,sub=setup()
    upsert_engagement(s,session_id=p.id,student_id=st.id,homework_rate=80,participation=70,behaviour=90,study_hours_week=8)
    upsert_engagement(s,session_id=p.id,student_id=st.id,homework_rate=85,participation=75,behaviour=90,study_hours_week=9)
    assert s.query(Engagement).count()==1
    assert get_engagement(s,p.id,st.id).homework_rate==85

def test_native_features_do_not_read_current_exam():
    s,school,p,st,sub=setup()
    s.add(ContinuousAssessment(session_id=p.id,student_id=st.id,subject_id=sub.id,ca1=70,ca2=60,assignment=80,project=70))
    s.add(Attendance(session_id=p.id,student_id=st.id,school_days=60,days_present=54,days_absent=6))
    s.commit()
    f1=build_native_features(s,session_id=p.id,student_id=st.id,subject_id=sub.id)
    r=Result(session_id=p.id,student_id=st.id,subject_id=sub.id,ca1=70,ca2=60,assignment=80,project=70,exam=99,total=0,grade='A1',grade_point=4)
    s.add(r); s.commit()
    f2=build_native_features(s,session_id=p.id,student_id=st.id,subject_id=sub.id)
    assert f1==f2
    assert 'exam' not in f2 and 'target_exam_score' not in f2

def test_profile_import_preserves_395_style_profile_fields():
    s,school,p,st,sub=setup()
    df=pd.DataFrame([{'student_code':'P5-002','sex':'F','age':16,'study_time':3,'previous_failures':1,'absence_count':4,'higher_education_intent':'yes'}])
    assert upsert_students(df,s,school.id)==1
    row=s.query(Student).filter_by(student_code='P5-002').first()
    profile=s.query(__import__('edupredict.db',fromlist=['StudentProfile']).StudentProfile).filter_by(student_id=row.id).first()
    assert 'age' in json.loads(profile.profile_json)
