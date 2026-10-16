# EduPredict — Phase 3.2 Intelligent Broadsheet

## Purpose

Phase 3.2 turns the Phase 3.1 derived broadsheet into a complete teacher-facing class/stream reporting layer. It remains read-derived: no broadsheet or collated-results table is stored.

## Data flow

`Student → Session Placement → Subject Enrollment → Current Result → Attendance → Broadsheet`

The population is determined by active term-specific placement and active subject enrollment. Results are joined only to enrolled student-subject pairs.

## Result-state rules

| State | Display | Included in aggregates |
|---|---|---|
| Enrolled + result | Numeric score or grade | Yes |
| Enrolled + missing result | `—` | No |
| Not enrolled | `N/E` | No |

Missing results therefore do not become zero, F9, failure counts, or artificial low averages.

## Canonical broadsheet fields

- S/N
- Student ID
- Student Name
- Class
- Arm / Stream
- One dynamic column per subject represented in the selected cohort
- Average
- Performance Band
- Rank
- Ranking Status
- Subjects Enrolled
- Subjects Recorded
- Missing Results
- Completion %
- Below Credit
- Credit Pass %
- Attendance %

## Ranking

Ranking uses the observed subject average only. Students with no recorded score are not ranked. Students with partial results may be ranked, but the row is explicitly marked `Ranked — incomplete`. Ties receive the same rank.

Ranking is descriptive decision support and is not a sole basis for promotion, discipline, placement or other high-stakes decisions.

## Aggregation rules

- Average uses recorded subject totals only.
- Completion = recorded results / enrolled subjects × 100.
- Below Credit counts observed scores below the configured school credit threshold.
- Credit Pass % uses observed scores only.
- Attendance is contextual and is not mixed into academic marks.
- The stored Result.total remains the source of truth; the broadsheet does not silently recalculate assessment weights.

## Teacher-facing features

The Streamlit `Intelligent Broadsheet` page provides:

1. Academic period selection.
2. Whole-school or class/stream selection.
3. Dynamic subject selection.
4. Score or grade matrix display.
5. Student search.
6. Sorting by rank, student ID, name or average.
7. Missing-result-only filtering.
8. Class KPIs.
9. Subject intelligence table.
10. Missing-result worklist.
11. Ranking and data-quality view.
12. CSV export.
13. Multi-sheet Excel export containing the broadsheet, subject summary and missing-result worklist.

## Data-quality handling

The page surfaces:

- enrolled subject records
- recorded results
- missing results
- completion rate
- attendance coverage
- duplicate analytical result keys

Duplicate result integrity is handled by the Phase 2/Result Handling Hardening layer. Phase 3.2 does not silently repair or mutate source data.

## API

`src/edupredict/broadsheet.py` exposes:

- `build_broadsheet`
- `build_broadsheet_display`
- `broadsheet_missing_results`
- `broadsheet_subject_summary`
- `broadsheet_kpis`
- `export_broadsheet_xlsx`

## Testing

Phase 3.2 tests cover:

- enrolled/result state
- enrolled/missing state
- not-enrolled state
- missing-result worklist
- completion calculations
- class KPIs
- ranking status
- existing Phase 3.1 analytics
- result handling hardening

## Database safety

The existing `edupredict.db` must be preserved. Phase 3.2 does not reset, rebuild, import or recreate the database and does not add a collated broadsheet table.
