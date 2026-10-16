import pandas as pd
from edupredict.db import Base, engine, SessionLocal, School, Student, StudentProfile
from edupredict.ingest import upsert_students

def test_clean_profile_import_has_no_exam_fields(tmp_path):
    Base.metadata.create_all(engine)
    s=SessionLocal()
    try:
        school=s.query(School).first()
        if not school:
            school=School(name='Profile Import Test School',state='Test'); s.add(school); s.commit()
        df=pd.DataFrame([
            {'student_code':'CLEANP001','source_school':'MS','sex':'M','age':17,'study_time':2,'previous_failures':1,'absence_count':5},
            {'student_code':'CLEANP002','source_school':'GP','sex':'F','age':16,'study_time':3,'previous_failures':0,'absence_count':2},
        ])
        n=upsert_students(df,s,school.id)
        assert n==2
        students=s.query(Student).filter(Student.student_code.in_(['CLEANP001','CLEANP002'])).all()
        assert len(students)==2
        profiles=s.query(StudentProfile).filter(StudentProfile.student_id.in_([x.id for x in students])).all()
        assert len(profiles)==2
        assert all(not any(k in p.profile_json for k in ['G1','G2','G3','grade','credit_pass']) for p in profiles)
        # Re-importing the same rows is idempotent and reports processed rows.
        assert upsert_students(df,s,school.id)==2
        assert s.query(Student).filter(Student.student_code.in_(['CLEANP001','CLEANP002'])).count()==2
    finally:
        s.close()
