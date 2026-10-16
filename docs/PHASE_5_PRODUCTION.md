# EduPredict Phase 5 — Production Operational Layer

## Scope
Phase 5 makes the CRUD-first system operational for a Nigerian school workflow:
student/profile ingestion, multi-subject enrollment, pre-exam evidence, engagement,
batch early-warning predictions, intervention tracking, examination capture, final
results, prediction evaluation, and exportable analytics.

## Pre-exam boundary
The native model target is the actual examination score. Native features include
CA components, attendance, engagement, student profile variables, and prior-period
performance. The current examination score is never a feature.

## Native model lifecycle
1. Import students and profiles.
2. Configure academic period/classes/subjects.
3. Enroll students in subjects.
4. Capture CA, attendance and engagement.
5. Train native model once enough historical examination records exist.
6. Generate predictions before the current examination.
7. Create interventions for high-risk students.
8. Capture actual examination scores after the exam.
9. Finalize results from separated CA + examination records.
10. Evaluate predictions against actual examination scores.

## Data integrity
Student imports are idempotent by school + Student ID. Subject enrollment is
explicit. Results remain uniquely keyed by period + student + subject. Missing
results are not treated as zero. Final examination records are separate from
pre-exam evidence.

## Deployment
The application remains Streamlit + SQLAlchemy/SQLite for local deployment. The
same service contracts can later be moved to PostgreSQL without changing the
UI workflow.
