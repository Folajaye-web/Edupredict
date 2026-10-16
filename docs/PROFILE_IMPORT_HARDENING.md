# EduPredict v3.3.1 — Pre-Exam Profile Import Hardening

This patch makes the Students Data Import path compatible with the cleaned 395-student profile CSV.

## Rules

- CSV delimiter is auto-detected (comma or semicolon).
- `student_code` is required.
- `G1`, `G2`, `G3`, `target_score_pct`, `grade`, `credit_pass`, `exam`, and `exam_score` are blocked from the Students profile import.
- Native Student fields are used when supplied.
- Additional non-examination profile fields are stored in `student_profiles.profile_json`.
- Missing names are left blank and missing class is represented as `Unassigned`; the importer does not invent academic information.
- The AI feature builder reads the stored profile values as pre-exam context. Internal `G1_pct`/`G2_pct` compatibility names are derived from current CA/prior EduPredict results, never from imported legacy G1/G2 columns.

## Clean source dataset

The cleaned source contains 395 rows and no examination/outcome columns. It is suitable for the Students import test.
