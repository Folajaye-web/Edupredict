# EduPredict Result Handling Hardening — Phase 3 Foundation

## What is fixed

1. **No duplicate current results**
   - Current-result key: `academic session + student + subject`.
   - Manual save is an upsert.
   - CSV import uses the same upsert engine.
   - A SQLite unique index becomes the final database guard after legacy duplicates are repaired.

2. **Result corrections are auditable**
   - Existing values are copied into `result_revisions` before a correction.
   - The current result remains a single row.
   - Duplicate legacy rows are retired with snapshots rather than silently discarded.

3. **Academic eligibility is enforced**
   - Student must have a placement for the selected term.
   - Student must be actively enrolled in the subject for that term.
   - No result can be attached to an unrelated subject or period.

4. **Assessment validation is enforced**
   - CA1, CA2, assignment, project and examination must each be 0–100.
   - A genuine zero is allowed.
   - Assessment weights must total exactly 100%.
   - Final score is calculated from the school's configured policy.
   - Grade and grade point are recalculated from the final score.

5. **Missing is not failure**
   - An enrolled student without a result remains missing, not F9.
   - A result is not manufactured from missing assessment data.

6. **Nigerian-school fit**
   - Term/session is a first-class part of the result key.
   - Subject enrollment is student-specific, supporting Science/Humanities/Business/Trade combinations and school-approved subjects.
   - Internal assessment weights are configurable rather than hard-coded to an external examination body's rules.
   - A1–F9 remains an internal reporting layer; EduPredict does not claim WAEC/NECO/NABTEB certification authority.

## Legacy duplicate repair

The existing database has duplicate Mathematics result rows for PT0001/PT0002. Do not overwrite the database with a clean demo database.

Run:

```powershell
$env:PYTHONPATH = "$PWD\src"
python scripts/repair_results.py
```

The script:

- creates a timestamped backup under `backups/`;
- lists duplicate result keys;
- asks for explicit confirmation;
- keeps the latest current row by update time, then creation time, then row ID;
- stores retired rows in `result_revisions`;
- deletes only retired duplicate current rows;
- enables the database unique guard;
- verifies that no duplicate result keys remain.

For the currently observed demo data, the latest values are the later records. This is a **data-repair policy**, not a claim that the latest mark is academically correct; the audit trail preserves the retired values for review.

## Result integrity UI

The Results page now includes a Result Integrity section showing duplicate groups and duplicate rows. If legacy duplicates are detected, the UI can create a backup and run the same repair operation.

## Files changed

- `src/edupredict/db.py`
- `src/edupredict/results.py` — new result service
- `src/edupredict/ingest.py`
- `src/edupredict/schema.py`
- `app/streamlit_app.py`
- `tests/test_results.py`
- `scripts/repair_results.py`
- `docs/RESULT_HANDLING.md`
- `docs/RESULT_HANDLING_HARDENING.md`

## Verification

The hardening package was tested with:

- score calculation;
- repeated result save/update;
- enrollment eligibility;
- duplicate repair and audit trail;
- existing Phase 3.1 analytics tests;
- Python compile checks.

Current test result: **9 passed**.

The remaining SQLAlchemy messages are deprecation warnings from existing `datetime.utcnow()` usage; they are not test failures and do not change result correctness.
