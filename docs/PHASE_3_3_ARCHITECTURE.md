# EduPredict v3.3 — CRUD-First Pre-Examination Intelligence Architecture

## Principle
EduPredict is CRUD-first, analytics-driven and AI-powered. Persistent data is created, read, updated and safely deactivated through service boundaries; analytics and AI consume validated data.

## Data boundary
### Pre-examination inputs
- Student and academic context
- Active subject enrollment
- Current continuous assessment: CA1, CA2, assignment, project
- Attendance
- Engagement where available
- Historical observed results and trends

### Prohibited pre-examination inputs
- Current examination score
- Current final score
- Any value derived from the current examination

### Post-examination inputs
- Actual examination score
- Final result derived from CA + actual examination
- Prediction evaluation: predicted vs actual

## Persistent domains
School, Student, AcademicSession, ClassGroup, StudentPlacement, Subject, SubjectOffering, StudentSubjectEnrollment, AssessmentPolicy, ContinuousAssessment, Attendance, Engagement, Prediction, Intervention, FinalExamination, Result, ResultRevision, PredictionEvaluation.

## Service boundary
UI -> Application workflow -> CRUD service -> validation -> database/audit -> analytics/AI.

UI code should not contain business-critical database mutation logic.

## Prediction history
Predictions are append-only records. A new prediction creates a new record rather than overwriting an earlier prediction.

## Legacy compatibility
The existing trained model expects legacy behavioral features. v3.3 removes those fields from the user prediction form and builds compatibility values from pre-exam evidence/defaults. This is an interim bridge; the model should be retrained on EduPredict's native pre-exam feature set before making high-stakes automated decisions.

## Migration
Run `python scripts/migrate_v3_3_data.py` once against an existing database. The script creates a backup and copies legacy Result CA/exam components into the separated v3.3 tables without deleting or rewriting existing Result rows.
