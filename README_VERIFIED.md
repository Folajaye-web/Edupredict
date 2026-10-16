# EduPredict Phase 3.2 - Verified Final

This package contains the corrected Phase 3.2 Intelligent Broadsheet application.

## Important

Do not copy `edupredict.db` from this package over an existing school database.
Keep the school's current database in the project root.

## Clean start on Windows

From PowerShell in this folder:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\START_PHASE_3_2_8501.ps1
```

The launcher stops only an EduPredict/Streamlit process currently occupying port 8501, then starts this exact project's `app\streamlit_app.py`.

## Verification

The Phase 3.2 source was compiled and tested successfully:

- 11 tests passed
- No test failures
- The class/stream rendering uses explicit `if/else` blocks rather than conditional Streamlit expressions.
- The package contains `src\edupredict\broadsheet.py`.

## Database model

The broadsheet is derived from placements, subject enrolments, results and attendance. It is not stored as a separate collated-results table.
