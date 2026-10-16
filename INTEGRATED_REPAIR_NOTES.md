# EduPredict integrated repair — October 2026

## What this repair changes

1. **Corrects historical feature chronology** in `scripts/nigeria_validation/build_features.py`.
   - Groups history by `School + Student_ID` so reused student codes do not cross-contaminate schools.
   - Calculates overall history from completed earlier terms only; results from another subject in the same term cannot become historical inputs.
   - Adds `feature_pipeline_version = 2.0-period-safe-school-aware` and scalable temporal checks.
2. **Requires revalidation from the corrected feature pipeline** before Phase 5 will train. Phase 5 records the feature version, source CSV SHA-256, and scikit-learn version.
3. **Aligns Phase 7 qualification and Phase 7C.** Phase 7 now checks the frozen Phase 6 holdout calibration diagnostic; Phase 7C will not claim a technical release when Phase 7 qualification is still `CONDITIONAL_REVIEW`. A technical release is not a school-pilot approval.
4. **Uses one Phase 5 prediction path in the app.** The silent legacy/native model fallback and in-app native retraining control are removed. The app now shows why a prediction is blocked when current CA, attendance, model provenance, runtime compatibility, or release-gate evidence is missing.
5. **Corrects score interpretation.** The Phase 5 regressor predicts the examination component on a 0–60 scale. The app displays that separately and computes the estimated overall total as current CA sum (0–40) plus predicted exam score (0–60). It no longer labels the exam component as a percentage out of 100.
6. **Adds database-to-model feature construction** with chronological history, same-school validation, assessment-scale validation, required current CA and attendance, and tests that prove current-term exam results are excluded.
7. **Aligns downstream technical gates** with `TECHNICAL_RELEASE_READY` and makes Phase 9 monitoring consume the current Phase 7C manifest. Phase 10B/10C retain their separate fixed model-approval allowlist; a new model is not silently approved for a pilot.
8. **Tightens onboarding score ranges** to CA components 0–10, Exam 0–60, Total 0–100.
9. **Pins scikit-learn to 1.9.0**, the version recorded for the bundled model artifacts.
10. Adds `phase10E_integrated_revalidation.py`, a single backup-first runner that rebuilds features and reruns dependent validation gates in order.

## Run once from PowerShell

Use the project root and the existing Desktop virtual environment. Copy only the commands inside the code block.

```powershell
$root = "C:\Users\l2e\OneDrive\Desktop\EduPredict_MASTER_FINAL"
Set-Location $root
& "C:\Users\l2e\OneDrive\Desktop\.venv\Scripts\python.exe" -m pip install -r .\requirements.txt
& "C:\Users\l2e\OneDrive\Desktop\.venv\Scripts\python.exe" .\scripts\nigeria_validation\phase10E_integrated_revalidation.py --confirm-rebuild
```

The runner makes a timestamped backup under `backups\` before replacing derived feature files, candidate models, or validation reports. It does **not** write to `edupredict.db`, import the CSV, alter `models\production`, automatically promote models, or authorise a school pilot.

## Expected gate behaviour

- The runner rebuilds the features and candidate models, then runs Phase 6 and Phase 7 gates.
- If Phase 7C is still `CONDITIONAL_REVIEW`, the runner stops there and does not reuse old Phase 8/9 PASS reports as fresh evidence. Inspect `reports\nigeria_validation\phase10\phase10E_integrated_revalidation.json` and the saved logs.
- If Phase 7C passes technically, Phase 8/9 integration checks are rerun. Phase 10B/10C still require separate approval of the new model hashes and the 13 governance controls.
- **Pilot authorisation remains false.** A software test cannot provide school approval, lawful data-processing evidence, staff training, or a human governance decision.

## Live database status remains a separate dependency

The database audit supplied in the conversation showed four students but no results, CA, attendance, final examinations, or subject enrollments. This repair deliberately does not import the 12-school CSV into the current single-school app. A safe multi-school history import needs an explicit school mapping and a reviewed transaction plan. Until current-period CA, attendance, and enrollment exist, the app will explain why it cannot generate a real-student prediction.

## Validation performed on this repair copy

- Python syntax compilation: passed.
- Unit tests: 27 passed.
- Model retraining, CSV import, live-database writes, and school-pilot approval were **not** performed in this repair environment. Those steps must run in the user's project environment with the actual CSV and virtual environment.

## Cross-phase reconciliation findings

| Phase | Prior evidence | Reconciliation result |
|---|---|---|
| Dataset audit / Phase 2 | 156,930 rows, 12 schools, 3 academic sessions; prior feature report said PASS | Structural CSV checks passed, but the old overall-history feature ordering was not period-safe. Corrected builder now forces strict prior-term history and school isolation. |
| Phase 3 | One session per train/validation/test split | Split design remains chronological, but all splits must be regenerated from corrected features. |
| Phase 5 | Holdout MAE 5.1816; R² 0.6093; classification ROC-AUC 0.9560; PR-AUC 0.7861 | Those metrics describe the previous feature build and must not be carried forward as current evidence. The runner retrains candidate models only after the corrected feature report passes. |
| Phase 6 | Mean absolute calibration-bin error 0.0242; maximum 0.1129 | The maximum is above the configured 0.10 review trigger. The corrected workflow keeps Phase 7 in review unless fresh evidence clears that gate. |
| Phase 7 / 7C | Phase 7 contract `CONDITIONAL_REVIEW`, while old Phase 7C claimed `PRODUCTION_RELEASE_READY` | Status contradiction fixed: Phase 7C now requires `TECHNICAL_QUALIFICATION_PASS` and uses `TECHNICAL_RELEASE_READY` only for technical readiness. It never authorises a school pilot. |
| Phase 8 | Old reports were PASS; the stored Phase 8C smoke report recorded zero dataframes and zero buttons | Those reports are not a substitute for a full interactive prediction test. The integrated runner regenerates Phase 8 evidence only after the technical release gate passes. |
| Phase 9 | Old orchestrator PASS and scheduled validation evidence | Old hashes/reports become stale when models are rebuilt. Active Phase 9 gates now read the current Phase 7C technical manifest; they still stop on failed gates. |
| Phase 10A / 10B | Prior failure-recovery checks passed; 23 readiness checks passed | These are historical results against prior artifacts. Phase 10B's fixed model-hash allowlist intentionally requires explicit approval of a new model. |
| Phase 10C | `BLOCKED`; 0 of 13 governance controls verified | Correctly remains a separate human governance gate. The repair cannot mark any control verified or authorise a pilot. |
| Phase 10D | CSV structure checks passed; import not performed | Onboarding remains validation-only. The patched scale checks enforce the model's actual CA 0–10, Exam 0–60, Total 0–100 scales. |

The original reports are preserved in the timestamped backup made by the runner. New reports should be treated as the current evidence only after the integrated workflow has run to completion or a documented gate stop.
