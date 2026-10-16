from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA = PROJECT_ROOT / "data" / "raw" / "student-mat.csv"
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed" / "training_data.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "edupredict_bundle.joblib"
METRICS_PATH = PROJECT_ROOT / "reports" / "model_metrics.json"
FEATURE_IMPORTANCE_PATH = PROJECT_ROOT / "reports" / "feature_importance.csv"
NATIVE_MODEL_PATH = PROJECT_ROOT / "models" / "edupredict_native_bundle.joblib"
NATIVE_METRICS_PATH = PROJECT_ROOT / "reports" / "native_model_metrics.json"

# Reporting policy. This is an internal WAEC-style interpretation layer, not an
# official WASSCE grade calculator.
GRADE_BANDS = [
    (75, 100, "A1", "Excellent", True),
    (70, 74.999999, "B2", "Very Good", True),
    (65, 69.999999, "B3", "Good", True),
    (60, 64.999999, "C4", "Credit", True),
    (55, 59.999999, "C5", "Credit", True),
    (50, 54.999999, "C6", "Credit", True),
    (45, 49.999999, "D7", "Pass", False),
    (40, 44.999999, "E8", "Pass", False),
    (0, 39.999999, "F9", "Fail", False),
]

# Configurable school assessment policy; do not assume this is universal.
CA_WEIGHT = 0.40
EXAM_WEIGHT = 0.60

MODEL_FEATURES = [
    "G1_pct",
    "G2_pct",
    "absences",
    "studytime",
    "failures",
    "schoolsup",
    "famsup",
    "internet",
    "famrel",
    "freetime",
    "goout",
    "health",
    "traveltime",
    "higher",
]
