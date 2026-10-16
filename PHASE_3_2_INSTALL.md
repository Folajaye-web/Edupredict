# EduPredict Phase 3.2 — Installation

## Active project

Expected project:

`C:\Users\l2e\Desktop\EduPredict_Refined\EduPredict_Refined`

The package intentionally contains **no database**. Keep the existing repaired `edupredict.db`.

## Files added/changed

- `src/edupredict/broadsheet.py`
- `app/streamlit_app.py`
- `src/edupredict/db.py` (result-handling hardening retained)
- `src/edupredict/ingest.py` (result-handling hardening retained)
- `src/edupredict/schema.py` (result-handling hardening retained)
- `src/edupredict/results.py`
- `scripts/repair_results.py`
- `tests/test_results.py`
- `tests/test_broadsheet.py`
- `docs/PHASE_3_2_INTELLIGENT_BROADSHEET.md`

## PowerShell

```powershell
cd "C:\Users\l2e\Desktop\EduPredict_Refined\EduPredict_Refined"
& "C:\Users\l2e\Desktop\EduPredict\.venv\Scripts\Activate.ps1"
$env:PYTHONPATH = "$PWD\src"
```

Extract the Phase 3.2 package to a temporary folder, then copy its project files into the active project. Do not copy any database from the package over the existing database.

## Verification

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m compileall -q app src tests scripts
pytest -q
python run.py
```

Expected test result for this release:

`11 passed`

## New page

Open Streamlit and select:

**Intelligent Broadsheet**

The page is separate from the existing Results entry workflow so reporting does not interfere with result entry/correction.
