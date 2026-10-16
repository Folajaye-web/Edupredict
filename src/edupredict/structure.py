from sqlalchemy.orm import Session
from .db import AcademicSession, ClassGroup, Student, StudentPlacement, Subject, SubjectOffering, StudentSubjectEnrollment, Result
from .ingest import ensure_session

TERMS=['First Term','Second Term','Third Term']

def ensure_class_group(s: Session, school_id: int, session_id: int, class_level: str, arm: str|None):
    arm=(arm or '').strip() or None
    obj=s.query(ClassGroup).filter_by(session_id=session_id,class_level=class_level,arm=arm).first()
    if not obj:
        label=f"{class_level} {arm or ''}".strip()
        obj=ClassGroup(school_id=school_id,session_id=session_id,class_level=class_level,arm=arm,display_name=label)
        s.add(obj); s.commit()
    return obj

def assign_student(s: Session, student_id: int, class_group_id: int, session_id: int):
    obj=s.query(StudentPlacement).filter_by(session_id=session_id,student_id=student_id).first()
    if obj: obj.class_group_id=class_group_id
    else: s.add(StudentPlacement(session_id=session_id,student_id=student_id,class_group_id=class_group_id))
    sess=s.get(AcademicSession, session_id)
    if sess and sess.is_current:
        student=s.get(Student,student_id); group=s.get(ClassGroup,class_group_id)
        if student and group:
            student.class_level=group.class_level
            student.arm=group.arm
    s.commit()

def enroll_subject(s: Session, school_id: int, session_id: int, student_id: int, subject_id: int):
    obj=s.query(StudentSubjectEnrollment).filter_by(session_id=session_id,student_id=student_id,subject_id=subject_id).first()
    if obj: obj.active=True
    else: s.add(StudentSubjectEnrollment(session_id=session_id,student_id=student_id,subject_id=subject_id,active=True))
    s.commit()

def unenroll_subject(s: Session, session_id: int, student_id: int, subject_id: int):
    obj=s.query(StudentSubjectEnrollment).filter_by(session_id=session_id,student_id=student_id,subject_id=subject_id).first()
    if obj:
        obj.active=False; s.commit()

def offer_subject_to_class(s: Session, class_group_id: int, subject_id: int, compulsory: bool=False):
    obj=s.query(SubjectOffering).filter_by(class_group_id=class_group_id,subject_id=subject_id).first()
    if not obj:
        obj=SubjectOffering(class_group_id=class_group_id,subject_id=subject_id,compulsory=compulsory); s.add(obj)
    else: obj.compulsory=compulsory
    s.commit()

def enroll_class_subject(s: Session, class_group_id: int, subject_id: int, compulsory: bool=False):
    cg=s.get(ClassGroup,class_group_id)
    if not cg: return 0
    offer_subject_to_class(s,class_group_id,subject_id,compulsory)
    placements=s.query(StudentPlacement).filter_by(session_id=cg.session_id,class_group_id=class_group_id).all()
    n=0
    for p in placements:
        obj=s.query(StudentSubjectEnrollment).filter_by(session_id=cg.session_id,student_id=p.student_id,subject_id=subject_id).first()
        if not obj:
            s.add(StudentSubjectEnrollment(session_id=cg.session_id,student_id=p.student_id,subject_id=subject_id,active=True)); n+=1
        else: obj.active=True
    s.commit(); return n

def backfill_enrollments_from_results(s: Session):
    """Make every existing result imply enrollment; does not mark missing subjects as failures."""
    n=0
    for r in s.query(Result).all():
        obj=s.query(StudentSubjectEnrollment).filter_by(session_id=r.session_id,student_id=r.student_id,subject_id=r.subject_id).first()
        if not obj:
            s.add(StudentSubjectEnrollment(session_id=r.session_id,student_id=r.student_id,subject_id=r.subject_id,active=True)); n+=1
    s.commit(); return n

def subject_usage(s: Session, subject_id: int):
    """Return usage counts used to protect subjects with historical records."""
    offerings=s.query(SubjectOffering).filter_by(subject_id=subject_id).count()
    enrollments=s.query(StudentSubjectEnrollment).filter_by(subject_id=subject_id).count()
    results=s.query(Result).filter_by(subject_id=subject_id).count()
    return {"offerings": offerings, "enrollments": enrollments, "results": results}


def update_subject(s: Session, subject_id: int, code: str, name: str, stage: str, compulsory: bool):
    """Update catalogue metadata while preserving the subject record and history."""
    obj=s.get(Subject, subject_id)
    if not obj:
        raise ValueError("Subject not found.")
    code=(code or '').strip().upper()
    name=(name or '').strip()
    if not code or not name:
        raise ValueError("Subject code and subject name are required.")
    duplicate=s.query(Subject).filter(Subject.code==code, Subject.id!=subject_id).first()
    if duplicate:
        raise ValueError("That subject code is already in use.")
    obj.code=code; obj.name=name; obj.stage=stage; obj.compulsory=compulsory
    s.commit()
    return obj


def set_subject_active(s: Session, subject_id: int, active: bool):
    """Activate/deactivate a subject without deleting historical records."""
    obj=s.get(Subject, subject_id)
    if not obj:
        raise ValueError("Subject not found.")
    obj.active=active
    s.commit()
    return obj

