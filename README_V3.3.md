# EduPredict v3.3

## CRUD-first, pre-examination AI architecture
This release restructures the application around a controlled CRUD service layer and a strict pre-/post-examination boundary.

### Core flow
School → Session/Term → Class/Stream → Student Placement → Subject Enrollment → CA/Attendance/Engagement → Analytics → AI Prediction → Early Warning → Intervention → Final Examination → Final Result → Prediction Evaluation.

### Important safety rule
The current examination score is never an input to a pre-examination prediction. The actual examination is entered only through **Final Examination** after the exam, and predictions remain historically immutable.

### Existing database
The application is additive. It creates the new v3.3 tables and does not replace `edupredict.db`. For an existing database, make a backup and run:

```powershell
$env:PYTHONPATH="$PWD\src"
python scripts/migrate_v3_3_data.py
```

### Run
```powershell
$env:PYTHONPATH="$PWD\src"
python run.py
```

### Tests
```powershell
pytest -q
```

## v3.3.1 profile import hardening

The Students Data Import path now accepts the cleaned pre-exam profile CSV, auto-detects CSV delimiters, stores extra profile fields separately, and blocks examination/outcome columns from student profile imports. See `docs/PROFILE_IMPORT_HARDENING.md`.
