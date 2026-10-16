from sqlalchemy.orm import Session
from ..db import Engagement, StudentPlacement
from .common import utcnow

def get_engagement(s: Session, session_id, student_id):
    return s.query(Engagement).filter_by(session_id=session_id, student_id=student_id).first()

def upsert_engagement(s: Session, *, session_id, student_id, homework_rate=None, participation=None, behaviour=None, study_hours_week=None):
    if not s.query(StudentPlacement).filter_by(session_id=session_id, student_id=student_id).first():
        raise ValueError('Student is not placed in a class/stream for this academic period.')
    vals={}
    for name,value in [('homework_rate',homework_rate),('participation',participation),('behaviour',behaviour)]:
        if value is not None and not 0 <= float(value) <= 100:
            raise ValueError(f'{name} must be between 0 and 100.')
        vals[name]=None if value is None else float(value)
    if study_hours_week is not None and float(study_hours_week) < 0:
        raise ValueError('Study hours per week cannot be negative.')
    vals['study_hours_week']=None if study_hours_week is None else float(study_hours_week)
    obj=get_engagement(s,session_id,student_id)
    if obj is None:
        obj=Engagement(session_id=session_id,student_id=student_id,**vals); s.add(obj)
    else:
        for k,v in vals.items(): setattr(obj,k,v)
    s.commit()
    return obj
