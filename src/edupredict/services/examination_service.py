from sqlalchemy.orm import Session
from ..db import FinalExamination, StudentSubjectEnrollment, StudentPlacement
from .common import validate_score, utcnow

def get_examination(s: Session, session_id, student_id, subject_id):
    return s.query(FinalExamination).filter_by(session_id=session_id,student_id=student_id,subject_id=subject_id).first()

def upsert_examination(s: Session, *, session_id, student_id, subject_id, exam_score, exam_date=None, comment=None):
    if not s.query(StudentPlacement).filter_by(session_id=session_id,student_id=student_id).first():
        raise ValueError('Student is not placed in a class/stream for this academic period.')
    if not s.query(StudentSubjectEnrollment).filter_by(session_id=session_id,student_id=student_id,subject_id=subject_id,active=True).first():
        raise ValueError('Student is not actively enrolled in this subject for this academic period.')
    score=validate_score(exam_score,'Examination score')
    if score is None: raise ValueError('Examination score is required.')
    obj=get_examination(s,session_id,student_id,subject_id)
    if obj is None:
        obj=FinalExamination(session_id=session_id,student_id=student_id,subject_id=subject_id,exam_score=score,exam_date=exam_date,comment=comment)
        s.add(obj)
    else:
        obj.exam_score=score; obj.exam_date=exam_date; obj.comment=comment; obj.updated_at=utcnow()
    s.commit(); return obj
