from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sys

import joblib
import numpy as np
import pandas as pd

from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
)
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import phase7_production_qualification as p7


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def metrics(y, p, threshold):
    predicted = (p >= threshold).astype(int)
    tn = int(((y == 0) & (predicted == 0)).sum())
    fp = int(((y == 0) & (predicted == 1)).sum())
    fn = int(((y == 1) & (predicted == 0)).sum())
    tp = int(((y == 1) & (predicted == 1)).sum())

    curve_true, curve_pred = calibration_curve(
        y, p, n_bins=10, strategy="quantile"
    )
    errors = np.abs(curve_true - curve_pred)

    return {
        "rows": int(len(y)),
        "prevalence": float(y.mean()),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "mean_absolute_bin_error": float(errors.mean()),
        "maximum_bin_error": float(errors.max()),
        "nonempty_calibration_bins": int(len(errors)),
        "threshold": float(threshold),
        "flagged": int(predicted.sum()),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def main():
    print("=" * 68)
    print("EDUPREDICT — SEPARATE SIGMOID CALIBRATION EXPERIMENT")
    print("=" * 68)

    model_path = p7.CLS_MODEL
    valid_path = p7.VALID

    if not model_path.is_file() or not valid_path.is_file():
        raise FileNotFoundError(
            "Frozen classifier or validation split is missing."
        )

    # This experiment intentionally loads validation data only.
    valid = p7.load_csv(valid_path)

    id_col = next(
        (c for c in ["Student_ID", "student_id"] if c in valid.columns),
        None,
    )
    if id_col is None:
        raise ValueError(
            "No Student_ID/student_id column in validation split. "
            "Stopped without fitting: a student-grouped split is required."
        )

    required = [p7.TARGET_CLS, *p7.FEATURES, id_col]
    missing = sorted(set(required) - set(valid.columns))
    if missing:
        raise ValueError(f"Required validation columns missing: {missing}")

    y = valid[p7.TARGET_CLS].astype(int).to_numpy()
    groups = valid[id_col].astype(str).to_numpy()

    if not set(np.unique(y)).issubset({0, 1}) or len(np.unique(y)) != 2:
        raise ValueError("Target must contain both binary classes, 0 and 1.")

    # Capture frozen model hash before loading/predicting.
    model_hash_before = sha256(model_path)
    frozen_model = joblib.load(model_path)
    p_raw = frozen_model.predict_proba(p7.X(valid))[:, 1]

    if not np.isfinite(p_raw).all():
        raise ValueError("Frozen model returned non-finite probabilities.")

    # Split by student, not by row.
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.40, random_state=20261016
    )
    fit_idx, eval_idx = next(splitter.split(valid, y, groups=groups))

    if len(np.unique(y[fit_idx])) != 2 or len(np.unique(y[eval_idx])) != 2:
        raise ValueError(
            "Both classes must be represented in calibration-fit "
            "and evaluation partitions. No candidate saved."
        )

    overlap = set(groups[fit_idx]) & set(groups[eval_idx])
    if overlap:
        raise RuntimeError("Student-group leakage detected.")

    # Platt-style sigmoid mapping on the log-odds of the frozen model.
    eps = 1e-6
    clipped = np.clip(p_raw, eps, 1 - eps)
    log_odds = np.log(clipped / (1 - clipped)).reshape(-1, 1)

    calibrator = LogisticRegression(
        C=1e6, solver="lbfgs", max_iter=2000
    )
    calibrator.fit(log_odds[fit_idx], y[fit_idx])

    p_cal = calibrator.predict_proba(log_odds[eval_idx])[:, 1]
    p_eval_raw = p_raw[eval_idx]
    y_eval = y[eval_idx]

    raw_metrics = metrics(y_eval, p_eval_raw, threshold=0.40)
    calibrated_metrics = metrics(y_eval, p_cal, threshold=0.40)

    # Reliability bins for the held-out student groups.
    true_rate, pred_rate = calibration_curve(
        y_eval, p_cal, n_bins=10, strategy="quantile"
    )
    bin_table = pd.DataFrame({
        "bin": np.arange(1, len(true_rate) + 1),
        "mean_predicted_probability": pred_rate,
        "observed_positive_rate": true_rate,
        "absolute_error": np.abs(pred_rate - true_rate),
    })

    # Ensure the original frozen classifier was not modified.
    model_hash_after = sha256(model_path)
    if model_hash_before != model_hash_after:
        raise RuntimeError("Frozen classifier hash changed unexpectedly.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_dir = ROOT / "reports" / "nigeria_validation"
    model_dir = ROOT / "models" / "nigeria_validation"
    report_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    report_path = report_dir / f"calibration_experiment_{stamp}.json"
    bins_path = report_dir / f"calibration_experiment_bins_{stamp}.csv"
    predictions_path = report_dir / f"calibration_experiment_predictions_{stamp}.csv"
    candidate_path = model_dir / f"sigmoid_calibrator_experiment_{stamp}.joblib"

    predictions = pd.DataFrame({
        "student_id": groups[eval_idx],
        "actual_target": y_eval,
        "raw_probability": p_eval_raw,
        "calibrated_probability": p_cal,
    })

    # Save a candidate artifact separately; never replace Phase 5 model.
    joblib.dump(calibrator, candidate_path)
    bin_table.to_csv(bins_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    report = {
        "experiment": "student_grouped_sigmoid_calibration",
        "status": "EXPERIMENT_COMPLETE_NOT_PRODUCTION_APPROVAL",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "data_used": "validation_2024_2025.csv only",
        "final_holdout_loaded": False,
        "method": "Logistic regression on clipped frozen-model log-odds",
        "split": {
            "random_state": 20261016,
            "fit_rows": int(len(fit_idx)),
            "evaluation_rows": int(len(eval_idx)),
            "fit_students": int(len(set(groups[fit_idx]))),
            "evaluation_students": int(len(set(groups[eval_idx]))),
            "overlapping_students": int(len(overlap)),
        },
        "raw_evaluation_metrics": raw_metrics,
        "calibrated_evaluation_metrics": calibrated_metrics,
        "brier_improvement": float(
            raw_metrics["brier"] - calibrated_metrics["brier"]
        ),
        "log_loss_improvement": float(
            raw_metrics["log_loss"] - calibrated_metrics["log_loss"]
        ),
        "frozen_classifier_sha256_before": model_hash_before,
        "frozen_classifier_sha256_after": model_hash_after,
        "artifacts": {
            "candidate_calibrator": str(candidate_path),
            "calibration_bins": str(bins_path),
            "evaluation_predictions": str(predictions_path),
        },
        "limitations": [
            "Internal validation experiment, not independent final confirmation.",
            "The validation session has also been used in prior model-policy analysis.",
            "The existing 2025/2026 holdout was not loaded or used for fitting.",
            "This result does not change Phase 7 release status.",
        ],
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nRAW vs CALIBRATED — STUDENT-GROUPED EVALUATION")
    print(f"Evaluation rows: {len(eval_idx):,}")
    print(f"Fit students: {len(set(groups[fit_idx])):,}")
    print(f"Evaluation students: {len(set(groups[eval_idx])):,}")
    print(f"Student overlap: {len(overlap)}")

    for label, result in [
        ("RAW", raw_metrics),
        ("CALIBRATED", calibrated_metrics),
    ]:
        print(f"\n{label}")
        for key in [
            "brier", "log_loss", "roc_auc", "pr_auc",
            "mean_absolute_bin_error", "maximum_bin_error",
            "flagged", "precision",
        ]:
            if key == "precision":
                tp, fp = result["tp"], result["fp"]
                value = tp / (tp + fp) if tp + fp else 0.0
            else:
                value = result[key]
            print(f"  {key}: {value:.6f}")

    print("\nBrier improvement (positive is better):", report["brier_improvement"])
    print("Log-loss improvement (positive is better):", report["log_loss_improvement"])
    print("\nReport:", report_path)
    print("Calibration bins:", bins_path)
    print("Predictions:", predictions_path)
    print("Candidate calibrator:", candidate_path)
    print("\nPhase 5 classifier unchanged. Phase 7 status unchanged.")


if __name__ == "__main__":
    main()
