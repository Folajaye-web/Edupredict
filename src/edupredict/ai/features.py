from sqlalchemy.orm import Session
from ..db import ContinuousAssessment, Attendance, Result, AcademicSession, StudentSubjectEnrollment, StudentProfile

def _num(value, default=None):
    try:
        if value is None: return default
        return float(value)
    except (TypeError, ValueError):
        return default

def _yes_no(value, default=0):
    if value is None: return default
    return 1 if str(value).strip().lower() in {'yes','y','true','1'} else 0

def build_pre_exam_features(s: Session, *, session_id: int, student_id: int, subject_id: int) -> dict:
    """Build prediction features from pre-examination evidence only.

    Current examination/final-result fields are deliberately excluded. Student
    profile values imported from the cleaned source are read from StudentProfile.
    G1/G2 remain only as internal compatibility feature names for the legacy
    model bundle; they are derived from current CA/prior EduPredict results,
    never imported from legacy G1/G2 columns.
    """
    ca=s.query(ContinuousAssessment).filter_by(session_id=session_id,student_id=student_id,subject_id=subject_id).first()
    att=s.query(Attendance).filter_by(session_id=session_id,student_id=student_id).first()
    profile_row=s.query(StudentProfile).filter_by(student_id=student_id).first()
    profile={}
    if profile_row:
        import json
        try: profile=json.loads(profile_row.profile_json or '{}')
        except Exception: profile={}

    ca_values=[v for v in [getattr(ca,'ca1',None),getattr(ca,'ca2',None),getattr(ca,'assignment',None),getattr(ca,'project',None)] if v is not None] if ca else []
    current_ca=sum(ca_values)/len(ca_values) if ca_values else None
    prior=s.query(Result).filter(Result.student_id==student_id,Result.subject_id==subject_id,Result.session_id!=session_id).all()
    prior_scores=[float(r.total) for r in prior if r.total is not None]
    previous_average=sum(prior_scores)/len(prior_scores) if prior_scores else current_ca

    profile_absences=_num(profile.get('absence_count'),0)
    absences=int(att.days_absent) if att else int(profile_absences or 0)
    attendance_pct=(float(att.days_present)/att.school_days*100) if att and att.school_days else None

    g1=current_ca if current_ca is not None else previous_average if previous_average is not None else 50.0
    g2=previous_average if previous_average is not None else g1
    failures=max(int(_num(profile.get('previous_failures'),0) or 0), sum(1 for x in prior_scores if x < 50))

    return {
        'G1_pct': float(g1), 'G2_pct': float(g2), 'absences': absences,
        'studytime': int(_num(profile.get('study_time'),2) or 2),
        'failures': failures,
        'schoolsup': _yes_no(profile.get('school_support')),
        'famsup': _yes_no(profile.get('family_support')),
        'internet': _yes_no(profile.get('internet_access'),1),
        'famrel': int(_num(profile.get('family_relationship'),3) or 3),
        'freetime': int(_num(profile.get('free_time'),3) or 3),
        'goout': int(_num(profile.get('social_outings'),3) or 3),
        'health': int(_num(profile.get('self_reported_health'),3) or 3),
        'traveltime': int(_num(profile.get('travel_time'),2) or 2),
        'higher': _yes_no(profile.get('higher_education_intent'),1),
        '_current_ca': current_ca, '_previous_average': previous_average, '_attendance_pct': attendance_pct,
    }

