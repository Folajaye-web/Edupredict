def risk_from_score(score: float, probability_below_credit: float | None = None) -> dict:
    """Return an advisory risk band aligned with the Phase 5 warning threshold.

    Phase 5 class 1 is the probability that overall Total is below 50. The
    technical candidate's warning threshold is 0.40; risk bands above that
    threshold take precedence over the regression-derived total.
    """
    s = float(score)
    if probability_below_credit is not None:
        p = float(probability_below_credit)
        if p >= 0.70:
            return {"level": "Critical", "priority": 5, "reason": "High estimated probability of scoring below 50%."}
        if p >= 0.45:
            return {"level": "High", "priority": 4, "reason": "Substantial estimated probability of scoring below 50%."}
        if p >= 0.40:
            return {"level": "Moderate", "priority": 3, "reason": "Estimated probability has reached the Phase 5 early-warning threshold."}
    if s < 40:
        return {"level": "Critical", "priority": 5, "reason": "Estimated overall total is below the 40% pass threshold."}
    if s < 50:
        return {"level": "High", "priority": 4, "reason": "Estimated overall total is below the credit threshold."}
    if s < 60:
        return {"level": "Moderate", "priority": 3, "reason": "Estimated overall total is in the lower credit range."}
    if s < 70:
        return {"level": "Low", "priority": 2, "reason": "Estimated overall total is acceptable but below B-grade territory."}
    return {"level": "Stable", "priority": 1, "reason": "Estimated overall total is at least 70%."}
