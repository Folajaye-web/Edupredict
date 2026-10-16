from sqlalchemy.orm import Session
from ..db import ContinuousAssessment, StudentSubjectEnrollment, StudentPlacement
from .common import validate_score, utcnow

FIELDS = ('ca1','ca2','assignment','project')

def _eligible(s, session_id, student_id, subject_id):
    if not s.query(StudentPlacement).filter_by(session_id=session_id, student_id=student_id).first():
        raise ValueError('Student is not placed in a class/stream for this academic period.')
    if not s.query(StudentSubjectEnrollment).filter_by(session_id=session_id, student_id=student_id, subject_id=subject_id, active=True).first():
        raise ValueError('Student is not actively enrolled in this subject for this academic period.')

def get_assessment(s: Session, session_id, student_id, subject_id):
    return s.query(ContinuousAssessment).filter_by(session_id=session_id, student_id=student_id, subject_id=subject_id).first()

def upsert_assessment(s: Session, *, session_id, student_id, subject_id, ca1=None, ca2=None, assignment=None, project=None, comment=None):
    _eligible(s, session_id, student_id, subject_id)
    values={k: validate_score(v,k.upper()) for k,v in {'ca1':ca1,'ca2':ca2,'assignment':assignment,'project':project}.items()}
    obj=get_assessment(s,session_id,student_id,subject_id)
    if obj is None:
        obj=ContinuousAssessment(session_id=session_id,student_id=student_id,subject_id=subject_id,**values,teacher_comment=comment)
        s.add(obj)
    else:
        for k,v in values.items(): setattr(obj,k,v)
        obj.teacher_comment=comment
        obj.updated_at=utcnow()
    s.commit()
    return obj
