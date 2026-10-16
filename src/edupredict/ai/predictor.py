"""Single prediction path: the Phase 5 Nigerian model only.

Legacy/native fallback was removed because it silently selected a different,
non-equivalent model and feature schema when the native bundle was absent.
"""
import pandas as pd
from ..production_adapter import build_production_features
from ..production_predictor import FEATURES, predict as predict_phase5
from ..risk import risk_from_score


def predict_pre_exam(s, *, session_id, student_id, subject_id):
    features, current_ca_total = build_production_features(
        s, session_id=session_id, student_id=student_id, subject_id=subject_id
    )
    model_input = pd.DataFrame([[features[name] for name in FEATURES]], columns=FEATURES)
    output, metadata = predict_phase5(model_input)
    row = output.iloc[0]
    predicted_exam_score = float(row["predicted_exam_score"])
    # The regressor target is Exam (0–60), not Total (0–100). Convert to an
    # estimated overall total only by adding the four current CA components.
    predicted_total = max(0.0, min(100.0, current_ca_total + predicted_exam_score))
    probability = float(row["risk_probability"])
    warning = bool(row["early_warning"])
    risk = risk_from_score(predicted_total, probability)
    rationale = (
        f"Estimated probability of total below 50: {probability:.1%}. "
        f"Early-warning threshold: {metadata['threshold']:.0%}. "
        f"Predicted examination score: {predicted_exam_score:.1f}/60; "
        f"estimated overall total: {predicted_total:.1f}/100. "
        "This is advisory decision support only; Phase 10C has not authorised a school pilot."
    )
    return {
        "predicted_score": predicted_total,
        "predicted_total": predicted_total,
        "predicted_exam_score": predicted_exam_score,
        "below_credit_probability": probability,
        "early_warning": warning,
        "risk_level": risk["level"],
        "rationale": rationale,
        "features": features,
        "model_version": metadata["model_version"],
        "model_metadata": metadata,
    }
