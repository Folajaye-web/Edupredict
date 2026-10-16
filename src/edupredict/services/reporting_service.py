from sqlalchemy.orm import Session
from ..analytics import analytics_dataset, student_period_metrics, class_analytics

def student_report(s: Session, school_id, student_id, session_id):
    return student_period_metrics(s, school_id, student_id, session_id)

def class_report(s: Session, school_id, session_id, class_level=None, arm=None):
    return class_analytics(s, school_id, session_id, class_level, arm)

def subject_report(s: Session, school_id, session_id, class_level=None, arm=None):
    ca=class_analytics(s, school_id, session_id, class_level, arm)
    return ca['subject_table']
