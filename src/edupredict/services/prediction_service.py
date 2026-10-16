from sqlalchemy.orm import Session
from ..db import Prediction
from .common import utcnow
import json

def _normalize_snapshot(snapshot):
    if snapshot is None:
        return ''
    if not isinstance(snapshot, str):
        snapshot = json.dumps(snapshot, default=str, sort_keys=True, separators=(',', ':'))
    else:
        try:
            snapshot = json.dumps(json.loads(snapshot), default=str, sort_keys=True, separators=(',', ':'))
        except Exception:
            snapshot = str(snapshot)
    return snapshot

def create_prediction(s: Session, *, session_id, student_id, subject_id, predicted_score, risk_level, below_credit_probability, rationale, model_version='unknown', feature_snapshot=None):
    snapshot = _normalize_snapshot(feature_snapshot)

    existing = (
        s.query(Prediction)
        .filter_by(
            session_id=session_id,
            student_id=student_id,
            subject_id=subject_id,
            model_version=model_version,
            feature_snapshot=snapshot,
        )
        .order_by(Prediction.id.desc())
        .first()
    )
    if existing:
        return existing

    obj=Prediction(
        session_id=session_id,
        student_id=student_id,
        subject_id=subject_id,
        predicted_score=float(predicted_score),
        risk_level=risk_level,
        below_credit_probability=float(below_credit_probability),
        rationale=rationale,
        model_version=model_version,
        feature_snapshot=snapshot,
    )
    s.add(obj)
    s.commit()
    return obj

def list_predictions(s: Session, session_id=None, student_id=None, subject_id=None):
    q=s.query(Prediction)
    for field,value in [('session_id',session_id),('student_id',student_id),('subject_id',subject_id)]:
        if value is not None: q=q.filter_by(**{field:value})
    return q.order_by(Prediction.created_at.desc()).all()
