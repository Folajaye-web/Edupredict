from datetime import datetime, timezone


def utcnow():
    return datetime.now(timezone.utc)


def validate_score(value, field='score'):
    if value is None or value == '':
        return None
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{field} must be numeric.') from exc
    if not 0 <= value <= 100:
        raise ValueError(f'{field} must be between 0 and 100.')
    return round(value, 4)
