from pathlib import Path
import json
import hashlib

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
)


# ============================================================
# EDUPREDICT PHASE 7B
# SUBGROUP SAFETY GATE
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "nigeria_validation" / "splits"
MODEL_DIR = ROOT / "models" / "nigeria_validation"
REPORT_DIR = ROOT / "reports" / "nigeria_validation"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

TEST = DATA_DIR / "test_2025_2026.csv"
CLS_MODEL = MODEL_DIR / "phase5_classification_hgb.joblib"

TARGET = "target_below_credit"

FEATURES = [
    "School",
    "Term",
    "Class",
    "Arm",
    "Gender",
    "Subject",
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

FORBIDDEN = {
    "Exam",
    "Total",
    "Grade",
    "Final_Year_Result",
    "Promotion_Status",
    "Intervention",
}

SELECTED_THRESHOLD = 0.40

# ------------------------------------------------------------
# SAFETY POLICY
# ------------------------------------------------------------
#
# These are minimum operating safeguards.
#
# A subgroup must:
#
#   - contain at least 100 rows
#   - have both target classes present
#   - ROC-AUC >= 0.85
#   - F1 >= 0.55
#   - recall >= 0.50
#   - precision >= 0.50
#   - Brier <= 0.15
#
# These are release-gate criteria, not claims about educational
# fairness or causal performance.
# ------------------------------------------------------------

MIN_ROWS = 100
MIN_ROC_AUC = 0.85
MIN_F1 = 0.55
MIN_RECALL = 0.50
MIN_PRECISION = 0.50
MAX_BRIER = 0.15


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def load_test():
    if not TEST.exists():
        raise FileNotFoundError(TEST)

    df = pd.read_csv(TEST)

    required = set(FEATURES + [TARGET])

    missing = sorted(required - set(df.columns))

    if missing:
        raise RuntimeError(
            f"Final-test dataset is missing required columns: {missing}"
        )

    return df


def subgroup_metrics(model, group, threshold):
    y = group[TARGET].astype(int).to_numpy()

    p = model.predict_proba(group[FEATURES])[:, 1]

    pred = (p >= threshold).astype(int)

    positive_count = int(y.sum())
    negative_count = int(len(y) - positive_count)

    if len(np.unique(y)) < 2:
        return {
            "roc_auc": np.nan,
            "pr_auc": np.nan,
            "precision": precision_score(
                y, pred, zero_division=0
            ),
            "recall": recall_score(
                y, pred, zero_division=0
            ),
            "f1": f1_score(
                y, pred, zero_division=0
            ),
            "brier": brier_score_loss(y, p),
            "positive_rate": float(np.mean(y)),
            "positive_count": positive_count,
            "negative_count": negative_count,
            "status": "REVIEW",
            "failure_reasons": "single_target_class",
        }

    roc_auc = roc_auc_score(y, p)
    pr_auc = average_precision_score(y, p)
    precision = precision_score(y, pred, zero_division=0)
    recall = recall_score(y, pred, zero_division=0)
    f1 = f1_score(y, pred, zero_division=0)
    brier = brier_score_loss(y, p)

    failures = []

    if roc_auc < MIN_ROC_AUC:
        failures.append(
            f"roc_auc<{MIN_ROC_AUC}"
        )

    if f1 < MIN_F1:
        failures.append(
            f"f1<{MIN_F1}"
        )

    if recall < MIN_RECALL:
        failures.append(
            f"recall<{MIN_RECALL}"
        )

    if precision < MIN_PRECISION:
        failures.append(
            f"precision<{MIN_PRECISION}"
        )

    if brier > MAX_BRIER:
        failures.append(
            f"brier>{MAX_BRIER}"
        )

    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "brier": float(brier),
        "positive_rate": float(np.mean(y)),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "status": "PASS" if not failures else "FAIL",
        "failure_reasons": ";".join(failures),
    }


def main():

    print("=" * 72)
    print("EDUPREDICT NIGERIA VALIDATION — PHASE 7B")
    print("=" * 72)
    print("Purpose: Subgroup Safety / Stability Gate")
    print()

    # --------------------------------------------------------
    # 1. ARTIFACT CHECK
    # --------------------------------------------------------

    print("Checking final-test dataset...")

    if not TEST.exists():
        raise RuntimeError(
            f"Missing final-test dataset: {TEST}"
        )

    print("PASS:", TEST)

    print()
    print("Checking frozen classification model...")

    if not CLS_MODEL.exists():
        raise RuntimeError(
            f"Missing classification model: {CLS_MODEL}"
        )

    print("PASS:", CLS_MODEL)

    # --------------------------------------------------------
    # 2. LOAD TEST DATA
    # --------------------------------------------------------

    test = load_test()

    print()
    print(f"Final-test rows: {len(test):,}")

    # Session integrity
    sessions = sorted(
        test["Academic_Session"]
        .astype(str)
        .unique()
    )

    if sessions != ["2025/2026"]:
        raise RuntimeError(
            "Final-test session integrity failure: "
            f"{sessions}"
        )

    print("Session integrity: PASS")

    # --------------------------------------------------------
    # 3. FEATURE CONTRACT
    # --------------------------------------------------------

    forbidden_used = sorted(
        set(FEATURES).intersection(FORBIDDEN)
    )

    if forbidden_used:
        raise RuntimeError(
            "Forbidden predictor(s) found in production feature "
            f"contract: {forbidden_used}"
        )

    print("Feature contract: PASS")

    # --------------------------------------------------------
    # 4. LOAD FROZEN MODEL
    # --------------------------------------------------------

    model = joblib.load(CLS_MODEL)

    print("Classification model: LOADED")

    if not hasattr(model, "predict_proba"):
        raise RuntimeError(
            "Classification model does not expose predict_proba."
        )

    print("Classification interface: PASS")

    # --------------------------------------------------------
    # 5. GLOBAL BASELINE
    # --------------------------------------------------------

    y = test[TARGET].astype(int).to_numpy()

    p = model.predict_proba(test[FEATURES])[:, 1]

    pred = (p >= SELECTED_THRESHOLD).astype(int)

    global_metrics = {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "precision": float(
            precision_score(y, pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y, pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y, pred, zero_division=0)
        ),
        "brier": float(
            brier_score_loss(y, p)
        ),
        "threshold": SELECTED_THRESHOLD,
        "rows": int(len(test)),
    }

    # --------------------------------------------------------
    # 6. SUBGROUP ANALYSIS
    # --------------------------------------------------------

    dimensions = [
        "School",
        "Class",
        "Arm",
        "Term",
        "Subject",
    ]

    results = []

    for dimension in dimensions:

        print()
        print(f"Checking {dimension} subgroups...")

        for value, group in test.groupby(
            dimension,
            dropna=False
        ):

            value = str(value)

            rows = len(group)

            if rows < MIN_ROWS:
                results.append(
                    {
                        "dimension": dimension,
                        "group": value,
                        "rows": rows,
                        "roc_auc": np.nan,
                        "pr_auc": np.nan,
                        "precision": np.nan,
                        "recall": np.nan,
                        "f1": np.nan,
                        "brier": np.nan,
                        "positive_rate": np.nan,
                        "positive_count": int(
                            group[TARGET].sum()
                        ),
                        "negative_count": int(
                            len(group) - group[TARGET].sum()
                        ),
                        "status": "REVIEW",
                        "failure_reasons": (
                            f"rows<{MIN_ROWS}"
                        ),
                        "threshold": SELECTED_THRESHOLD,
                    }
                )

                continue

            metrics = subgroup_metrics(
                model,
                group,
                SELECTED_THRESHOLD,
            )

            metrics.update(
                {
                    "dimension": dimension,
                    "group": value,
                    "rows": rows,
                    "threshold": SELECTED_THRESHOLD,
                }
            )

            results.append(metrics)

    subgroup_df = pd.DataFrame(results)

    # --------------------------------------------------------
    # 7. OVERALL GATE
    # --------------------------------------------------------

    failing = subgroup_df[
        subgroup_df["status"] == "FAIL"
    ]

    review = subgroup_df[
        subgroup_df["status"] == "REVIEW"
    ]

    gate_pass = (
        len(failing) == 0
        and len(review) == 0
    )

    # --------------------------------------------------------
    # 8. DIMENSION SUMMARY
    # --------------------------------------------------------

    dimension_summary = []

    for dimension in dimensions:

        part = subgroup_df[
            subgroup_df["dimension"] == dimension
        ]

        dimension_summary.append(
            {
                "dimension": dimension,
                "groups": int(len(part)),
                "pass": int(
                    (part["status"] == "PASS").sum()
                ),
                "fail": int(
                    (part["status"] == "FAIL").sum()
                ),
                "review": int(
                    (part["status"] == "REVIEW").sum()
                ),
                "minimum_f1": (
                    float(part["f1"].min())
                    if part["f1"].notna().any()
                    else np.nan
                ),
                "minimum_recall": (
                    float(part["recall"].min())
                    if part["recall"].notna().any()
                    else np.nan
                ),
                "minimum_precision": (
                    float(part["precision"].min())
                    if part["precision"].notna().any()
                    else np.nan
                ),
                "maximum_brier": (
                    float(part["brier"].max())
                    if part["brier"].notna().any()
                    else np.nan
                ),
            }
        )

    dimension_df = pd.DataFrame(dimension_summary)

    # --------------------------------------------------------
    # 9. WRITE REPORTS
    # --------------------------------------------------------

    subgroup_path = (
        REPORT_DIR / "phase7B_subgroup_safety_gate.csv"
    )

    summary_path = (
        REPORT_DIR / "phase7B_dimension_summary.csv"
    )

    subgroup_df.to_csv(
        subgroup_path,
        index=False,
    )

    dimension_df.to_csv(
        summary_path,
        index=False,
    )

    # --------------------------------------------------------
    # 10. CONTRACT
    # --------------------------------------------------------

    contract = {
        "project": "EduPredict",
        "phase": "7B",
        "gate": "Subgroup Safety Gate",
        "status": "PASS" if gate_pass else "CONDITIONAL_REVIEW",
        "test_session": "2025/2026",
        "threshold": SELECTED_THRESHOLD,
        "minimum_rows": MIN_ROWS,
        "criteria": {
            "roc_auc_min": MIN_ROC_AUC,
            "f1_min": MIN_F1,
            "recall_min": MIN_RECALL,
            "precision_min": MIN_PRECISION,
            "brier_max": MAX_BRIER,
        },
        "dimensions": dimensions,
        "global_metrics": global_metrics,
        "subgroup_count": int(len(subgroup_df)),
        "passing_subgroups": int(
            (subgroup_df["status"] == "PASS").sum()
        ),
        "failing_subgroups": int(
            (subgroup_df["status"] == "FAIL").sum()
        ),
        "review_subgroups": int(
            (subgroup_df["status"] == "REVIEW").sum()
        ),
        "model_sha256": sha256_file(CLS_MODEL),
    }

    contract_path = (
        REPORT_DIR / "phase7B_subgroup_safety_gate.json"
    )

    contract_path.write_text(
        json.dumps(
            contract,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # 11. CONSOLE REPORT
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("PHASE 7B RESULT")
    print("=" * 72)

    print(
        f"Global ROC-AUC:  {global_metrics['roc_auc']:.4f}"
    )

    print(
        f"Global PR-AUC:   {global_metrics['pr_auc']:.4f}"
    )

    print(
        f"Global F1:       {global_metrics['f1']:.4f}"
    )

    print(
        f"Global recall:   {global_metrics['recall']:.4f}"
    )

    print(
        f"Global precision: {global_metrics['precision']:.4f}"
    )

    print()

    print(
        f"Subgroups checked: {len(subgroup_df)}"
    )

    print(
        f"PASS:             "
        f"{(subgroup_df['status'] == 'PASS').sum()}"
    )

    print(
        f"FAIL:             "
        f"{(subgroup_df['status'] == 'FAIL').sum()}"
    )

    print(
        f"REVIEW:           "
        f"{(subgroup_df['status'] == 'REVIEW').sum()}"
    )

    print()

    if len(failing) > 0:

        print("FAILING SUBGROUPS:")
        print()

        for _, row in failing.iterrows():

            print(
                f"  {row['dimension']} = "
                f"{row['group']}: "
                f"{row['failure_reasons']}"
            )

    if len(review) > 0:

        print()
        print("SUBGROUPS REQUIRING REVIEW:")

        for _, row in review.iterrows():

            print(
                f"  {row['dimension']} = "
                f"{row['group']}: "
                f"{row['failure_reasons']}"
            )

    print()
    print(
        "PHASE 7B STATUS: "
        f"{'PASS' if gate_pass else 'CONDITIONAL_REVIEW'}"
    )

    print()
    print(f"Detailed results: {subgroup_path}")
    print(f"Dimension summary: {summary_path}")
    print(f"Contract:         {contract_path}")

    print("=" * 72)


if __name__ == "__main__":
    main()
