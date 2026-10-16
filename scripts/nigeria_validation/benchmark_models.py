from pathlib import Path
import json
import math

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
)
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "nigeria_validation"
    / "splits"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "nigeria_validation"
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_DIR / "train_2023_2024.csv"
VALIDATION_FILE = DATA_DIR / "validation_2024_2025.csv"

REPORT_FILE = (
    REPORT_DIR / "phase4_benchmark_results.json"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "nigeria_validation"
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)


ID_COLUMNS = [
    "Student_ID",
]

TARGET_COLUMNS = [
    "target_exam_score",
    "target_below_credit",
]


# These are identifiers/context fields rather than direct numeric
# predictors. They are included through categorical preprocessing.
CATEGORICAL_FEATURES = [
    "School",
    "Academic_Session",
    "Term",
    "Class",
    "Arm",
    "Gender",
    "Subject",
]


NUMERIC_FEATURES = [
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


FEATURES = (
    CATEGORICAL_FEATURES
    + NUMERIC_FEATURES
)


def safe_float(value):
    if value is None:
        return None

    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return None

    return float(value)


def regression_metrics(y_true, predictions):
    rmse = math.sqrt(
        mean_squared_error(
            y_true,
            predictions,
        )
    )

    return {
        "mae": safe_float(
            mean_absolute_error(
                y_true,
                predictions,
            )
        ),
        "rmse": safe_float(rmse),
        "r2": safe_float(
            r2_score(
                y_true,
                predictions,
            )
        ),
    }


def classification_metrics(y_true, probabilities):
    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    result = {
        "roc_auc": None,
        "pr_auc": None,
        "precision": safe_float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": safe_float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1": safe_float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "accuracy": safe_float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),
        "brier": safe_float(
            brier_score_loss(
                y_true,
                probabilities,
            )
        ),
        "threshold": 0.50,
    }

    if len(np.unique(y_true)) == 2:
        result["roc_auc"] = safe_float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        )

        result["pr_auc"] = safe_float(
            average_precision_score(
                y_true,
                probabilities,
            )
        )

    return result


def make_one_hot_preprocessor():
    numeric = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        [
            (
                "numeric",
                numeric,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical,
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def make_tree_preprocessor():
    numeric = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
        ]
    )

    categorical = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
                ),
            (
                "ordinal",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        [
            (
                "numeric",
                numeric,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical,
                CATEGORICAL_FEATURES,
            ),
        ],
        sparse_threshold=0,
    )


def build_regression_models():

    return {
        "linear_regression": Pipeline(
            [
                (
                    "preprocessor",
                    make_one_hot_preprocessor(),
                ),
                (
                    "model",
                    LinearRegression(),
                ),
            ]
        ),

        "random_forest": Pipeline(
            [
                (
                    "preprocessor",
                    make_tree_preprocessor(),
                ),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        min_samples_leaf=3,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),

        "hist_gradient_boosting": Pipeline(
            [
                (
                    "preprocessor",
                    make_tree_preprocessor(),
                ),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=250,
                        learning_rate=0.05,
                        max_leaf_nodes=31,
                        l2_regularization=1.0,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def build_classification_models():

    return {
        "logistic_regression": Pipeline(
            [
                (
                    "preprocessor",
                    make_one_hot_preprocessor(),
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),

        "random_forest": Pipeline(
            [
                (
                    "preprocessor",
                    make_tree_preprocessor(),
                ),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=3,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),

        "hist_gradient_boosting": Pipeline(
            [
                (
                    "preprocessor",
                    make_tree_preprocessor(),
                ),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=250,
                        learning_rate=0.05,
                        max_leaf_nodes=31,
                        l2_regularization=1.0,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def evaluate_regression_baselines(
    train,
    validation,
):
    results = {}

    y_train = train[
        "target_exam_score"
    ].astype(float)

    y_validation = validation[
        "target_exam_score"
    ].astype(float)

    # ---------------------------------------------------------------
    # Previous-term baseline
    # ---------------------------------------------------------------

    prediction = validation[
        "Previous_Term_Total"
    ].astype(float).copy()

    fallback = validation[
        "Previous_Year_Average"
    ].astype(float)

    prediction = prediction.fillna(
        fallback
    )

    prediction = prediction.fillna(
        train["target_exam_score"].mean()
    )

    results[
        "previous_term_baseline"
    ] = regression_metrics(
        y_validation,
        prediction,
    )

    # ---------------------------------------------------------------
    # Previous-year baseline
    # ---------------------------------------------------------------

    prediction = validation[
        "Previous_Year_Average"
    ].astype(float).copy()

    prediction = prediction.fillna(
        validation["Previous_Term_Total"]
    )

    prediction = prediction.fillna(
        train["target_exam_score"].mean()
    )

    results[
        "previous_year_baseline"
    ] = regression_metrics(
        y_validation,
        prediction,
    )

    # ---------------------------------------------------------------
    # ML candidates
    # ---------------------------------------------------------------

    X_train = train[FEATURES]
    X_validation = validation[FEATURES]

    for name, model in build_regression_models().items():

        print(
            f"    Training regression: {name}"
        )

        model.fit(
            X_train,
            y_train,
        )

        prediction = model.predict(
            X_validation
        )

        results[name] = regression_metrics(
            y_validation,
            prediction,
        )

    return results


def evaluate_classification_baselines(
    train,
    validation,
):
    results = {}

    y_train = train[
        "target_below_credit"
    ].astype(int)

    y_validation = validation[
        "target_below_credit"
    ].astype(int)

    # ---------------------------------------------------------------
    # Previous-term baseline
    #
    # A prior total below 50 becomes the baseline probability.
    # ---------------------------------------------------------------

    previous_term = validation[
        "Previous_Term_Total"
    ].astype(float)

    previous_year = validation[
        "Previous_Year_Average"
    ].astype(float)

    probabilities = (
        previous_term < 50
    ).astype(float)

    missing = previous_term.isna()

    probabilities.loc[missing] = (
        previous_year.loc[missing] < 50
    ).astype(float)

    missing = (
        previous_term.isna()
        & previous_year.isna()
    )

    probabilities.loc[missing] = (
        y_train.mean()
    )

    results[
        "previous_term_baseline"
    ] = classification_metrics(
        y_validation,
        probabilities,
    )

    # ---------------------------------------------------------------
    # Previous-year baseline
    # ---------------------------------------------------------------

    probabilities = (
        previous_year < 50
    ).astype(float)

    missing = previous_year.isna()

    probabilities.loc[missing] = (
        previous_term.loc[missing] < 50
    ).astype(float)

    missing = (
        previous_year.isna()
        & previous_term.isna()
    )

    probabilities.loc[missing] = (
        y_train.mean()
    )

    results[
        "previous_year_baseline"
    ] = classification_metrics(
        y_validation,
        probabilities,
    )

    # ---------------------------------------------------------------
    # ML candidates
    # ---------------------------------------------------------------

    X_train = train[FEATURES]
    X_validation = validation[FEATURES]

    for name, model in build_classification_models().items():

        print(
            f"    Training classification: {name}"
        )

        model.fit(
            X_train,
            y_train,
        )

        probabilities = model.predict_proba(
            X_validation
        )[:, 1]

        results[name] = classification_metrics(
            y_validation,
            probabilities,
        )

    return results


def main():

    print("=" * 72)
    print("EDUPREDICT PHASE 4")
    print("BASELINES + CANDIDATE MODEL BENCHMARK")
    print("=" * 72)

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Training file not found: {TRAIN_FILE}"
        )

    if not VALIDATION_FILE.exists():
        raise FileNotFoundError(
            f"Validation file not found: {VALIDATION_FILE}"
        )

    train = pd.read_csv(
        TRAIN_FILE
    )

    validation = pd.read_csv(
        VALIDATION_FILE
    )

    print(
        f"Training rows:    {len(train):,}"
    )

    print(
        f"Validation rows:  {len(validation):,}"
    )

    # ---------------------------------------------------------------
    # Confirm test set is not being loaded.
    # ---------------------------------------------------------------

    print()
    print(
        "[1] Final test set: NOT LOADED"
    )

    # ---------------------------------------------------------------
    # Feature validation
    # ---------------------------------------------------------------

    missing_features = [
        column
        for column in FEATURES
        if column not in train.columns
        or column not in validation.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing benchmark features: "
            f"{missing_features}"
        )

    print(
        "[2] Required model features: PASS"
    )

    # ---------------------------------------------------------------
    # Verify forbidden current-result fields are absent.
    # ---------------------------------------------------------------

    forbidden = [
        "Exam",
        "Total",
        "Grade",
        "Final_Year_Result",
        "Promotion_Status",
        "Intervention",
    ]

    forbidden_present = [
        column
        for column in forbidden
        if column in FEATURES
    ]

    if forbidden_present:
        raise AssertionError(
            "Forbidden predictors found: "
            f"{forbidden_present}"
        )

    print(
        "[3] Leakage boundary: PASS"
    )

    # ---------------------------------------------------------------
    # Regression benchmark
    # ---------------------------------------------------------------

    print()
    print("[4] REGRESSION BENCHMARK")
    print("-" * 72)

    regression_results = (
        evaluate_regression_baselines(
            train,
            validation,
        )
    )

    for name, metrics in regression_results.items():

        print(
            f"    {name:28s} "
            f"MAE={metrics['mae']:.4f} "
            f"RMSE={metrics['rmse']:.4f} "
            f"R2={metrics['r2']:.4f}"
        )

    # ---------------------------------------------------------------
    # Classification benchmark
    # ---------------------------------------------------------------

    print()
    print("[5] CLASSIFICATION BENCHMARK")
    print("-" * 72)

    classification_results = (
        evaluate_classification_baselines(
            train,
            validation,
        )
    )

    for name, metrics in (
        classification_results.items()
    ):

        roc = metrics["roc_auc"]
        pr = metrics["pr_auc"]

        roc_text = (
            f"{roc:.4f}"
            if roc is not None
            else "N/A"
        )

        pr_text = (
            f"{pr:.4f}"
            if pr is not None
            else "N/A"
        )

        print(
            f"    {name:28s} "
            f"ROC-AUC={roc_text} "
            f"PR-AUC={pr_text} "
            f"F1={metrics['f1']:.4f} "
            f"Brier={metrics['brier']:.4f}"
        )

    # ---------------------------------------------------------------
    # Baseline reference
    # ---------------------------------------------------------------

    classification_prevalence = float(
        validation[
            "target_below_credit"
        ].mean()
    )

    # ---------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------

    report = {
        "status": "PASS",
        "training_file": str(TRAIN_FILE),
        "validation_file": str(
            VALIDATION_FILE
        ),
        "final_test_used": False,
        "training_rows": int(len(train)),
        "validation_rows": int(
            len(validation)
        ),
        "features": FEATURES,
        "categorical_features": (
            CATEGORICAL_FEATURES
        ),
        "numeric_features": NUMERIC_FEATURES,
        "forbidden_predictors": forbidden,
        "forbidden_predictors_present": (
            forbidden_present
        ),
        "classification_validation_prevalence": (
            classification_prevalence
        ),
        "regression": regression_results,
        "classification": classification_results,
        "random_state": 42,
    }

    REPORT_FILE.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("PHASE 4 STATUS: PASS")
    print("=" * 72)

    print()
    print(
        "Benchmark report:"
    )

    print(
        REPORT_FILE
    )

    print()
    print(
        "IMPORTANT: 2025/2026 final test data "
        "was not used."
    )


if __name__ == "__main__":
    main()