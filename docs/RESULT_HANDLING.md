# EduPredict — Result Handling & Integrity

## Purpose

The result subsystem is the authoritative school-internal workflow for entering, importing, correcting and auditing academic results. It is designed for Nigerian school reporting while remaining configurable because schools may use different internal assessment structures.

## Core invariant

For one school academic period:

`session + student + subject = exactly one current result`

The application updates the current result instead of inserting another row. A database unique index is also created after legacy duplicates are repaired, so the rule is enforced below the application layer.

## Eligibility

A result can be recorded only when:

1. the student exists in the school;
2. the student has a placement in the selected academic session/term;
3. the student has an active subject enrollment for that same session/term;
4. all assessment components are valid scores from 0 to 100;
5. the assessment policy weights total exactly 100%.

This prevents results from being attached to the wrong class, term or subject combination.

## Nigerian educational fit

EduPredict separates **school assessment** from **external examination-body certification**. The school can configure CA1, CA2, assignment, project and examination weights rather than hard-coding one national weighting. This is appropriate because internal school assessment arrangements can vary.

The current implementation retains EduPredict's internal reporting bands (A1–F9) as an internal reporting layer only. It does not claim to reproduce WAEC/NECO/NABTEB certification calculations.

For SSS, the current revised national curriculum framework retains 8–9 subjects and identifies five core/compulsory areas; subject enrollment therefore remains term-specific and student-specific rather than assuming every student takes every catalogue subject.

## Corrections and audit trail

When an existing result is changed:

- the current result row is updated;
- the previous values are copied to `result_revisions`;
- the revision records the action and reason;
- the current result remains the only row used by broadsheets and analytics.

This gives schools a traceable correction history instead of accumulating duplicate current results.

## Duplicate repair

Older databases can contain duplicates because SQLite `create_all()` does not add new constraints to an already-existing table. EduPredict therefore provides an explicit repair operation.

The repair process:

1. creates a timestamped database backup;
2. identifies duplicate `(session_id, student_id, subject_id)` groups;
3. keeps the latest current row using update time, then creation time, then row ID;
4. stores every retired duplicate in `result_revisions`;
5. deletes only the retired duplicate current rows;
6. creates the database unique index;
7. verifies that duplicate keys are gone.

Repair is never performed automatically at application startup.

## Import behavior

CSV result imports use the same upsert engine as manual entry. A repeated import updates the existing result key; it does not create another current result.

Invalid rows fail validation rather than being silently converted to zeros or attached to arbitrary students/subjects.

## Assessment rules

Each configured component is validated between 0 and 100. A genuine zero is allowed. A missing component is not silently treated as zero by the result engine.

The final score is calculated from the configured policy and stored to two decimal places. Grade and grade point are recalculated from that final score every time the result is created or corrected.

## Missing result versus failure

A student who is enrolled but has no recorded result is not automatically assigned zero or F9. Analytics treats missing data as missing data.

## Result status and future extensions

The architecture is ready for later extensions such as Draft/Final, Absent, Withheld, Malpractice/Cancelled, approval workflow and electronic sign-off. Those should be added as explicit result states rather than encoded as fake numeric scores.

## Operational safety

Before repairing legacy duplicates, the system creates a database backup in:

`backups/`

The original database should still be backed up externally as part of the school's normal data-protection process.

## Verification standard

The Result Handling Engine must pass:

- unit tests for score calculation;
- validation tests;
- eligibility tests;
- repeated-save/upsert tests;
- duplicate-repair tests;
- compile checks;
- database-integrity checks after migration.
