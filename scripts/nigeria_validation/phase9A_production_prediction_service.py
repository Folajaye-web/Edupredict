import json
from pathlib import Path

import pandas as pd

from edupredict.production_predictor import predict


ROOT = Path(__file__).resolve().parents[2]

TEST_DATA = (
    ROOT
    / "data"
    / "nigeria_validation"
    / "splits"
    / "test_2025_2026.csv"
)

REPORT_DIR = ROOT / "reports" / "nigeria_validation" / "phase9"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def main():

    print("=" * 72)
    print("EDUPREDICT NIGERIA VALIDATION — PHASE 9A")
    print("=" * 72)
    print("Purpose: Production Prediction Service Gate")
    print()

    df = pd.read_csv(TEST_DATA)

    predictions, metadata = predict(df)

    assert len(predictions) == len(df)

    assert list(predictions.columns) == [
        "risk_probability",
        "early_warning",
        "predicted_exam_score",
    ]

    assert predictions["risk_probability"].between(0, 1).all()

    assert predictions["predicted_exam_score"].notna().all()

    report = {
        "project": "EduPredict",
        "phase": "9A",
        "status": "PASS",
        "input_rows": len(df),
        "prediction_rows": len(predictions),
        "prediction_columns": list(predictions.columns),
        "metadata": metadata,
    }

    report_path = REPORT_DIR / "phase9A_production_prediction_service.json"

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    predictions.to_csv(
        REPORT_DIR / "phase9A_prediction_sample.csv",
        index=False,
    )

    print("=" * 72)
    print("PHASE 9A RESULT")
    print("=" * 72)

    print(f"Input rows:             {len(df):,}")
    print(f"Prediction rows:        {len(predictions):,}")
    print(
        f"Early-warning flags:    "
        f"{metadata['early_warning_count']:,}"
    )
    print(
        f"Early-warning rate:     "
        f"{metadata['early_warning_rate']:.4%}"
    )
    print(
        f"Risk probability range: "
        f"{metadata['risk_probability_min']:.4f}"
        f" - "
        f"{metadata['risk_probability_max']:.4f}"
    )
    print(
        f"Exam-score range:       "
        f"{metadata['predicted_exam_score_min']:.2f}"
        f" - "
        f"{metadata['predicted_exam_score_max']:.2f}"
    )

    print()
    print("Production artifact verification: PASS")
    print("Feature contract:                 PASS")
    print("Prediction contract:              PASS")
    print("Prediction range validation:      PASS")
    print()
    print("PHASE 9A STATUS: PASS")
    print()
    print(f"Report: {report_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()