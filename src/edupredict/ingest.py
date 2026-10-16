import pandas as pd
from sqlalchemy.orm import Session
from .db import Student, StudentProfile, Subject, AcademicSession, Result, Attendance, School
from .grading import grade_from_score

SUBJECTS={
 'ENG':'English Language','MATH':'Mathematics','BIO':'Biology','CHEM':'Chemistry','PHY':'Physics','CIV':'Citizenship and Heritage Studies',
 'GOV':'Government','ECO':'Economics','GEO':'Geography','AGR':'Agricultural Science','COM':'Commerce','ICT':'Information Technology','FURM':'Further Mathematics'
}

def ensure_subjects(s:Session):
    for code,name in SUBJECTS.items():
        if not s.query(Subject).filter_by(code=code).first(): s.add(Subject(code=code,name=name,stage='JSS/SSS',compulsory=code in {'ENG','MATH'}))
    s.commit()

def ensure_session(s, school_id, session_name, term):
    obj=s.query(AcademicSession).filter_by(school_id=school_id,session_name=session_name,term=term).first()
    if not obj:
        obj=AcademicSession(school_id=school_id,session_name=session_name,term=term,is_current=True); s.add(obj); s.commit()
    return obj

def _read_profile_value(r, *keys):
    for key in keys:
        if key in r and pd.notna(r[key]):
            return r[key]
    return None

def upsert_students(df,s,school_id):
    """Import student/profile CSVs without inventing academic outcomes.

    Native student fields are used when present. Any additional non-examination
    fields are preserved as StudentProfile JSON. Missing names/classes are left
    as explicit import placeholders rather than inferred from unrelated columns.
    """
    n=0
    required={'student_code'}
    missing=required-set(df.columns)
    if missing:
        raise ValueError(f'Missing required student field(s): {", ".join(sorted(missing))}')

    for r in df.to_dict('records'):
        code=str(r['student_code']).strip()
        if not code or code.lower()=='nan':
            raise ValueError('A student row has an empty student_code.')
        obj=s.query(Student).filter_by(school_id=school_id,student_code=code).first()

        first=_read_profile_value(r,'first_name')
        last=_read_profile_value(r,'last_name')
        gender=_read_profile_value(r,'gender','sex')
        cls=_read_profile_value(r,'class_level')
        arm=_read_profile_value(r,'arm')
        admission=_read_profile_value(r,'admission_year')

        if not obj:
            obj=Student(
                school_id=school_id, student_code=code,
                first_name=str(first).strip() if first is not None else '',
                last_name=str(last).strip() if last is not None else '',
                gender=str(gender).strip() if gender is not None else None,
                class_level=str(cls).strip() if cls is not None else 'Unassigned',
                arm=str(arm).strip() if arm is not None else None,
                admission_year=int(admission) if admission is not None and pd.notna(admission) else None
            )
            s.add(obj); s.flush()
        else:
            if first is not None: obj.first_name=str(first).strip()
            if last is not None: obj.last_name=str(last).strip()
            if gender is not None: obj.gender=str(gender).strip()
            if cls is not None: obj.class_level=str(cls).strip()
            if arm is not None: obj.arm=str(arm).strip()
            if admission is not None and pd.notna(admission): obj.admission_year=int(admission)

        profile={k: (None if pd.isna(v) else v.item() if hasattr(v,'item') else v) for k,v in r.items() if k not in {'student_code','first_name','last_name','gender','class_level','arm','admission_year'}}
        existing=s.query(StudentProfile).filter_by(student_id=obj.id).first()
        if existing:
            import json
            existing.profile_json=json.dumps(profile, default=str)
            existing.source_format='EduPredict pre-exam profile CSV'
        else:
            import json
            s.add(StudentProfile(student_id=obj.id,profile_json=json.dumps(profile, default=str),source_format='EduPredict pre-exam profile CSV'))
        # Count successfully processed rows, including idempotent updates.
        # This makes repeated imports report what was processed rather than
        # misleadingly returning zero after the first import.
        n += 1
    s.commit(); return n

def import_results(df, s, school_id, policy):
    """Strict, transactional result import. Existing result keys are updated, never duplicated."""
    ensure_subjects(s)
    from .results import upsert_result

    prepared=[]
    for row_no, r in enumerate(df.to_dict('records'), start=2):
        st=s.query(Student).filter_by(school_id=school_id,student_code=str(r['student_code']).strip()).first()
        if not st:
            raise ValueError(f"Row {row_no}: Student ID {r['student_code']} does not exist in this school.")
        sess=ensure_session(s,school_id,str(r['session']).strip(),str(r['term']).strip())
        sub=s.query(Subject).filter_by(code=str(r['subject_code']).strip().upper()).first()
        if not sub:
            raise ValueError(f"Row {row_no}: Subject {r['subject_code']} is not in the subject catalogue.")
        prepared.append((st,sess,sub,r))

    count=0
    try:
        for st,sess,sub,r in prepared:
            upsert_result(
                s, session_id=sess.id, student_id=st.id, subject_id=sub.id, policy=policy,
                ca1=r.get('ca1',0), ca2=r.get('ca2',0), assignment=r.get('assignment',0),
                project=r.get('project',0), exam=r.get('exam',0),
                teacher_comment=r.get('teacher_comment'), reason='CSV import correction/update'
            )
            count += 1
        s.commit()
    except Exception:
        s.rollback()
        raise
    return count

def import_attendance(df,s,school_id):
    count=0
    for r in df.to_dict('records'):
        st=s.query(Student).filter_by(school_id=school_id,student_code=str(r['student_code'])).first()
        if not st: continue
        sess=ensure_session(s,school_id,str(r['session']),str(r['term']))
        present=int(r['days_present']); absent=int(r['days_absent']); total=int(r['school_days'])
        if present+absent>total: continue
        obj=s.query(Attendance).filter_by(session_id=sess.id,student_id=st.id).first()
        if obj:
            obj.school_days=total; obj.days_present=present; obj.days_absent=absent
        else:
            s.add(Attendance(session_id=sess.id,student_id=st.id,school_days=total,days_present=present,days_absent=absent))
        count+=1
    s.commit(); return count
