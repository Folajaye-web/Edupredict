from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "production"

CLASSIFICATION_MODEL = MODEL_DIR / "edupredict_classification.joblib"
REGRESSION_MODEL = MODEL_DIR / "edupredict_regression.joblib"

PRODUCTION_THRESHOLD = 0.40

CAT_FEATURES = [
    "School",
    "Term",
    "Class",
    "Arm",
    "Gender",
    "Subject",
]

NUM_FEATURES = [
    "Term_Number",
    "CA1",
    "CA2",
    "Assignment",
    "Project",
    "Attendance",
    "Previous_Term_Total",
    "Previous_Year_Average",
    "Previous_Term_Grade_Numeric",
    "Historical_Subject_Average",
    "Historical_Subject_Count",
    "Historical_Overall_Average",
    "Historical_Overall_Count",
]

FEATURES = CAT_FEATURES + NUM_FEATURES

FORBIDDEN = {
    "Exam",
    "Total",
    "Grade",
    "Final_Year_Result",
    "Promotion_Status",
    "Intervention",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_models():
    if not CLASSIFICATION_MODEL.exists():
        raise FileNotFoundError(CLASSIFICATION_MODEL)

    if not REGRESSION_MODEL.exists():
        raise FileNotFoundError(REGRESSION_MODEL)

    return (
        joblib.load(CLASSIFICATION_MODEL),
        joblib.load(REGRESSION_MODEL),
    )


def validate_features(df: pd.DataFrame):
    missing = sorted(set(FEATURES) - set(df.columns))

    if missing:
        raise ValueError(
            f"Production prediction input is missing features: {missing}"
        )

    forbidden = sorted(set(df.columns) & FORBIDDEN)

    # Forbidden columns may exist in a wider application dataframe,
    # but they must never enter the model feature matrix.
    return {
        "missing": missing,
        "forbidden_present": forbidden,
    }


def build_model_input(df: pd.DataFrame) -> pd.DataFrame:
    validate_features(df)

    return df[FEATURES].copy()


def predict(df: pd.DataFrame) -> pd.DataFrame:
    classifier, regressor = load_models()

    X = build_model_input(df)

    probabilities = classifier.predict_proba(X)[:, 1]
    exam_predictions = regressor.predict(X)

    risk_flag = probabilities >= PRODUCTION_THRESHOLD

    result = pd.DataFrame(index=df.index)

    result["risk_probability"] = probabilities
    result["risk_flag"] = risk_flag.astype(bool)
    result["predicted_exam_score"] = np.asarray(exam_predictions, dtype=float)

    result["production_threshold"] = PRODUCTION_THRESHOLD

    return result


def production_manifest():
    return {
        "project": "EduPredict",
        "stage": "Phase 8 Production Integration",
        "classification_model": str(CLASSIFICATION_MODEL),
        "regression_model": str(REGRESSION_MODEL),
        "classification_sha256": sha256_file(CLASSIFICATION_MODEL),
        "regression_sha256": sha256_file(REGRESSION_MODEL),
        "threshold": PRODUCTION_THRESHOLD,
        "features": FEATURES,
        "forbidden_predictors": sorted(FORBIDDEN),
    }


if __name__ == "__main__":
    print(json.dumps(production_manifest(), indent=2))
