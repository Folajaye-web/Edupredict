from sqlalchemy.orm import Session
from ..db import Prediction, FinalExamination, PredictionEvaluation

def evaluate_prediction(s: Session, prediction_id):
    p=s.get(Prediction,prediction_id)
    if not p: raise ValueError('Prediction not found.')
    exam=s.query(FinalExamination).filter_by(session_id=p.session_id,student_id=p.student_id,subject_id=p.subject_id).first()
    if not exam: return None
    error=float(exam.exam_score)-float(p.predicted_score)
    abs_error=abs(error)
    existing=s.query(PredictionEvaluation).filter_by(prediction_id=p.id).first()
    if existing:
        existing.actual_exam_score=exam.exam_score; existing.error=error; existing.absolute_error=abs_error; existing.correct_risk=(('HIGH' in p.risk_level.upper() and exam.exam_score < 50) or ('CRITICAL' in p.risk_level.upper() and exam.exam_score < 40) or ('STABLE' in p.risk_level.upper() and exam.exam_score >= 50))
        s.commit(); return existing
    obj=PredictionEvaluation(prediction_id=p.id,session_id=p.session_id,student_id=p.student_id,subject_id=p.subject_id,actual_exam_score=exam.exam_score,error=error,absolute_error=abs_error,correct_risk=None)
    s.add(obj); s.commit(); return obj

def evaluate_available(s: Session, session_id):
    preds=s.query(Prediction).filter_by(session_id=session_id).all()
    out=[]
    for p in preds:
        obj=evaluate_prediction(s,p.id)
        if obj: out.append(obj)
    return out
