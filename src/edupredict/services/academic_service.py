from sqlalchemy.orm import Session
from ..db import AcademicSession, ClassGroup
from ..structure import ensure_class_group

def create_or_update_period(s: Session, school_id, session_name, term, current=False):
    from ..ingest import ensure_session
    obj=ensure_session(s,school_id,session_name.strip(),term)
    if current:
        s.query(AcademicSession).filter_by(school_id=school_id).update({AcademicSession.is_current:False})
    obj.is_current=current; s.commit(); return obj

def update_class_group(s: Session, group_id, class_level, arm=None, display_name=None):
    obj=s.get(ClassGroup,group_id)
    if not obj: raise ValueError('Class/stream not found.')
    obj.class_level=class_level; obj.arm=(arm or '').strip() or None; obj.display_name=(display_name or f'{class_level} {obj.arm or ""}').strip(); s.commit(); return obj

def deactivate_class_group(s: Session, group_id):
    obj=s.get(ClassGroup,group_id)
    if not obj: raise ValueError('Class/stream not found.')
    obj.active=False; s.commit(); return obj
