# Phase 2.1 — Academic Structure

EduPredict now separates the school academic structure from raw result records.

## Structure

`School → Academic Session + Term → Class / Stream → Student Placement → Subject Enrollment → Result`

### Academic session and term

A term is represented as part of an `AcademicSession` record (`2025/2026 + First Term`). This preserves a clean period key for results, attendance, placements and enrollments.

### Class / stream / arm

`ClassGroup` is a school-defined grouping for a specific session/term. The `arm` field is intentionally flexible: a school can use `A/B/C`, `Science/Arts/Commercial`, or another local naming convention.

### Student placement

`StudentPlacement` records where a student belongs for a particular session/term. This allows a student to move from SSS2 A to SSS3 B without rewriting historical records.

### Subject enrollment

`StudentSubjectEnrollment` records the subjects a student actually takes in a term. This is critical: a missing result for an enrolled subject is different from a subject the student does not take.

`SubjectOffering` records subjects offered to a class/stream and can be used for bulk enrollment.

## Backward compatibility

When an older EduPredict database is opened, the application applies additive schema migration for the new `subjects.active` column, creates the new tables, and backfills placements from existing student class/arm information and enrollments from existing result records. Existing result/attendance data are not deleted.

## Demo data

The supplied development data currently contains Mathematics result records. The other catalogue subjects are intentionally not assumed to have results. To analyze another subject, configure/enroll it for the class/stream and import or enter its results.

## Important reporting rule

The broadsheet must never interpret an unenrolled subject as a failure. For a selected term it displays enrolled subjects and distinguishes:

- **Score** — result recorded
- **Blank** — student is enrolled but result is not yet recorded
- **Not listed** — student is not enrolled in that subject
