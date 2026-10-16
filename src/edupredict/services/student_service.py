from sqlalchemy.orm import Session
from ..db import Student

def create_student(s: Session, *, school_id, student_code, first_name, last_name, gender=None, class_level='SSS1', arm=None, admission_year=None):
    code=student_code.strip()
    if not code or not first_name.strip() or not last_name.strip(): raise ValueError('Student ID, first name and last name are required.')
    if s.query(Student).filter_by(school_id=school_id,student_code=code).first(): raise ValueError('That Student ID already exists.')
    obj=Student(school_id=school_id,student_code=code,first_name=first_name.strip(),last_name=last_name.strip(),gender=gender,class_level=class_level,arm=arm.strip() if arm else None,admission_year=admission_year,active=True)
    s.add(obj); s.commit(); return obj

def update_student(s: Session, student_id, **values):
    obj=s.get(Student,student_id)
    if not obj: raise ValueError('Student not found.')
    if 'student_code' in values:
        code=values['student_code'].strip()
        dup=s.query(Student).filter(Student.school_id==obj.school_id,Student.student_code==code,Student.id!=obj.id).first()
        if dup: raise ValueError('That Student ID already exists.')
        values['student_code']=code
    for k,v in values.items():
        if hasattr(obj,k): setattr(obj,k,v.strip() if isinstance(v,str) else v)
    s.commit(); return obj

def deactivate_student(s: Session, student_id): return update_student(s,student_id,active=False)
