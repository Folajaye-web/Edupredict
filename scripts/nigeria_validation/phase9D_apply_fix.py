from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]

REMEDIATION = ROOT / "scripts" / "nigeria_validation" / "phase9D_data_quality_remediation.py"
HEALTH = ROOT / "scripts" / "nigeria_validation" / "phase9D_continuous_validation_gate.py"

# ============================================================
# 1. FIX PHASE 9D REMEDIATION SESSION COLUMN
# ============================================================

text = REMEDIATION.read_text(encoding="utf-8")

old = '''    actual_sessions = sorted(
        df["Session"].dropna().astype(str).unique().tolist()
    ) if "Session" in df.columns else []
'''

new = '''    # Production dataset uses Academic_Session.
    actual_sessions = sorted(
        df["Academic_Session"].dropna().astype(str).unique().tolist()
    ) if "Academic_Session" in df.columns else []
'''

if old in text:
    text = text.replace(old, new, 1)
elif 'df["Session"]' in text:
    text = text.replace(
        'df["Session"].dropna().astype(str).unique().tolist()',
        'df["Academic_Session"].dropna().astype(str).unique().tolist()',
        1
    )
else:
    raise RuntimeError(
        "Could not locate the Phase 9D remediation session block."
    )

REMEDIATION.write_text(text, encoding="utf-8")


# ============================================================
# 2. ADD REMEDIATION REPORT PATH TO HEALTH GATE
# ============================================================

text = HEALTH.read_text(encoding="utf-8")

marker = 'EXPECTED_REG_HASH = ('

if 'REMEDIATION_REPORT = (' not in text:
    insert = '''REMEDIATION_REPORT = (
    ROOT
    / "reports"
    / "nigeria_validation"
    / "phase9"
    / "phase9D_data_quality_remediation.json"
)

'''
    pos = text.find(marker)
    if pos == -1:
        raise RuntimeError(
            "Could not locate EXPECTED_REG_HASH in Phase 9D health gate."
        )
    text = text[:pos] + insert + text[pos:]


# ============================================================
# 3. REPLACE DATA QUALITY HEALTH SECTION
# ============================================================

pattern = re.compile(
    r'    # 5\. DATA QUALITY HEALTH.*?'
    r'    # 6\. MODEL IDENTITY HEALTH',
    re.DOTALL
)

replacement = '''    # 5. DATA QUALITY HEALTH
    # =========================================================

    production_features = df[FEATURES]

    missing_cells = int(
        production_features.isna().sum().sum()
    )

    duplicate_rows = int(
        df.duplicated().sum()
    )

    invalid_attendance = int(
        (
            pd.to_numeric(
                df["Attendance"],
                errors="coerce"
            ).notna()
            &
            ~pd.to_numeric(
                df["Attendance"],
                errors="coerce"
            ).between(0, 100)
        ).sum()
    )

    # ---------------------------------------------------------
    # Read the approved Phase 9D remediation contract.
    # Structural first-term historical missingness is allowed
    # only when the remediation gate has explicitly validated it.
    # ---------------------------------------------------------

    remediation_exists = REMEDIATION_REPORT.exists()

    remediation = {}
    if remediation_exists:
        try:
            remediation = json.loads(
                REMEDIATION_REPORT.read_text(encoding="utf-8")
            )
        except Exception as exc:
            remediation = {
                "status": "INVALID",
                "load_error": str(exc),
            }

    remediation_status_ok = (
        remediation.get("status") == "PASS"
    )

    remediation_session = remediation.get("session", [])
    remediation_session_ok = (
        remediation_session == [EXPECTED_SESSION]
    )

    structural_missing_cells = int(
        remediation.get("structural_missing_cells", 0)
    )

    unexpected_missing_cells = int(
        remediation.get("unexpected_missing_cells", 0)
    )

    non_historical_missing_cells = int(
        remediation.get("non_historical_missing_cells", 0)
    )

    count_consistency = remediation.get(
        "historical_count_consistency",
        {}
    )

    historical_count_consistency_ok = (
        int(
            count_consistency.get(
                "subject_missing_average_with_count",
                -1
            )
        ) == 0
        and
        int(
            count_consistency.get(
                "overall_missing_average_with_count",
                -1
            )
        ) == 0
    )

    remediation_policy = remediation.get(
        "policy",
        {}
    )

    no_dataset_modification = (
        remediation_policy.get(
            "dataset_modified"
        ) is False
    )

    no_imputation = (
        remediation_policy.get(
            "imputation_applied"
        ) is False
    )

    no_retraining = (
        remediation_policy.get(
            "model_retrained"
        ) is False
    )

    remediation_contract_ok = (
        remediation_exists
        and remediation_status_ok
        and remediation_session_ok
        and unexpected_missing_cells == 0
        and non_historical_missing_cells == 0
        and historical_count_consistency_ok
        and no_dataset_modification
        and no_imputation
        and no_retraining
        and structural_missing_cells == missing_cells
    )

    # ---------------------------------------------------------
    # IMPORTANT:
    # The 4,409 historical cells are not treated as defects
    # because the remediation contract has already validated
    # them as legitimate first-term structural missingness.
    #
    # Any unexpected missingness remains a production failure.
    # ---------------------------------------------------------

    data_quality_ok = (
        remediation_contract_ok
        and duplicate_rows == 0
        and invalid_attendance == 0
        and unexpected_missing_cells == 0
        and non_historical_missing_cells == 0
    )

    checks.append(
        make_check(
            "data_quality",
            data_quality_ok,
            (
                f"missing_cells={missing_cells}; "
                f"structural_missing_cells={structural_missing_cells}; "
                f"unexpected_missing_cells={unexpected_missing_cells}; "
                f"non_historical_missing_cells={non_historical_missing_cells}; "
                f"duplicate_rows={duplicate_rows}; "
                f"invalid_attendance={invalid_attendance}; "
                f"remediation_contract="
                f"{'PASS' if remediation_contract_ok else 'FAIL'}"
            ),
        )
    )

    if not data_quality_ok:
        failures.append(
            "Production data quality health failed."
        )

    # 6. MODEL IDENTITY HEALTH
'''

text, count = pattern.subn(replacement, text, count=1)

if count != 1:
    raise RuntimeError(
        "Could not replace the Phase 9D data-quality section."
    )


# ============================================================
# 4. ADD REMEDIATION DETAILS TO HEALTH SNAPSHOT
# ============================================================

needle = '''        "predicted_exam_score_mean": float(
            predicted_scores.mean()
        ),
'''

if needle in text and '"data_quality_remediation"' not in text:
    replacement_snapshot = needle + '''        "data_quality_remediation": {
            "status": remediation.get("status"),
            "session": remediation_session,
            "structural_missing_cells": structural_missing_cells,
            "unexpected_missing_cells": unexpected_missing_cells,
            "non_historical_missing_cells": non_historical_missing_cells,
            "contract": bool(remediation_contract_ok),
        },
'''
    text = text.replace(
        needle,
        replacement_snapshot,
        1
    )


# ============================================================
# 5. MAKE JSON SERIALIZATION ROBUST
# ============================================================

old_json = '''json.dumps(
            report,
            indent=2
        )
'''

if old_json in text:
    text = text.replace(
        old_json,
        '''json.dumps(
            report,
            indent=2,
            default=lambda o: o.item() if hasattr(o, "item") else str(o),
        )
''',
        1
    )

# More general replacement for any report serialization if present.
text = text.replace(
    '''json.dumps(
        report,
        indent=2
    )''',
    '''json.dumps(
        report,
        indent=2,
        default=lambda o: o.item() if hasattr(o, "item") else str(o),
    )'''
)

HEALTH.write_text(text, encoding="utf-8")

print("PHASE 9D PATCH APPLIED")
print(f"Remediation script: {REMEDIATION}")
print(f"Health gate:        {HEALTH}")
