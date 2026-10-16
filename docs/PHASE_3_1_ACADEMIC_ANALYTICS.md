# EduPredict — Phase 3.1 Academic Analytics Engine

## 1. Purpose

Phase 3.1 converts EduPredict from an academic record system into an academic intelligence layer. It derives analytics directly from students, term-specific placements, subject enrolments, results, attendance and engagement. No duplicate broadsheet table is stored.

The engine is designed around one rule: **missing data is missing data, not failure**. An enrolled subject with no result contributes to enrolment/completion metrics but does not receive a zero, F9 or failure count.

## 2. Scope completed

The engine now provides:

- Student-period performance profiles
- Class/stream cohort analytics
- Subject analytics
- Academic-period trends
- Student subject-by-subject trends
- Grade distributions
- Class ranking and percentile calculations
- Attendance integration
- Engagement coverage integration
- Result completion/data-quality metrics
- Assessment-component averages
- Low/high outlier detection using IQR
- Derived intelligent broadsheet
- Configurable credit and pass thresholds
- Chronological session/term ordering

## 3. Data inputs

### Student identity and placement

- Student ID/code
- Student name
- Active status
- Session-specific class/stream placement

### Subject structure

- Subject code
- Subject name
- Active subject
- Session-specific student enrolment

### Assessment results

- CA1
- CA2
- Assignment
- Project
- Examination
- Final total
- Grade
- Grade point

The final total remains the value produced by the configured assessment policy. Analytics do not silently recalculate an alternative final mark.

### Attendance

- School days
- Days present
- Days absent
- Attendance percentage

### Engagement

- Homework rate
- Participation
- Behaviour
- Study hours per week

Engagement is currently used for coverage/context in Phase 3.1. It is not mixed into academic marks.

## 4. Core parameters

### Credit threshold

Default: **50%**.

This is configurable at function/UI level and is the threshold used for credit-rate and below-credit analysis. It is not presented as an official WAEC calculator.

### Pass threshold

Default: **40%**.

Used to separate pass/fail statistics from credit statistics.

### Grade bands

The existing EduPredict reporting bands are retained:

| Score | Grade | Interpretation |
|---:|---|---|
| 75–100 | A1 | Excellent |
| 70–74 | B2 | Very Good |
| 65–69 | B3 | Good |
| 60–64 | C4 | Credit |
| 55–59 | C5 | Credit |
| 50–54 | C6 | Credit |
| 45–49 | D7 | Pass |
| 40–44 | E8 | Pass |
| 0–39 | F9 | Fail |

These bands are an internal reporting layer and must not be represented as an official examination-body grade calculator.

## 5. Student metrics

For a selected student and academic period the engine calculates:

1. Subjects enrolled
2. Results recorded
3. Result completion rate
4. Average score
5. Median score
6. Lowest score
7. Highest score
8. Standard deviation
9. Credit-pass count
10. Below-credit count
11. Pass count
12. Fail count
13. Grade-point average
14. Attendance percentage
15. Missing-result subjects
16. Weak subjects below the credit threshold
17. Strong subjects at 75% or above

### Interpretation

- **Average** describes central performance.
- **Median** reduces the effect of unusually high/low marks.
- **Standard deviation** describes consistency across subjects.
- **Completion rate** tells whether the profile is sufficiently observed.
- **Weak/strong subject lists** identify actionable subject-level differences.

## 6. Class/stream metrics

The class engine calculates:

- Active student count
- Distinct subjects
- Enrolled student-subject records
- Recorded results
- Completion rate
- Cohort mean
- Cohort median
- Cohort standard deviation
- Credit pass rate
- Pass rate
- Average attendance
- Student ranking table
- Subject-level performance table

Student averages are calculated from recorded subject results. Missing results are excluded from the mean rather than treated as zero.

## 7. Subject metrics

For every enrolled subject:

- Enrolled count
- Result count
- Completion percentage
- Average
- Median
- Minimum
- Maximum
- Standard deviation
- Credit pass percentage
- Pass percentage
- A1 percentage
- F9 percentage
- CA1 average
- CA2 average
- Assignment average
- Project average
- Examination average
- Grade distribution

This allows EduPredict to identify both **weak students in a subject** and **subjects that are broadly weak across a class**.

## 8. Trend engine

The trend engine orders periods chronologically using academic session and term. For each observed period it reports:

- Session
- Term
- Average
- Results
- Enrolments
- Completion
- Credit-pass percentage
- Attendance
- Change versus previous period
- Direction

Direction rules:

- Change > +0.5 percentage points: **Improving**
- Change < −0.5 percentage points: **Declining**
- Otherwise: **Stable**

The 0.5-point movement rule is an analytics display threshold, not a prediction rule.

## 9. Subject-specific student trends

For each student and subject, the engine retains the sequence of observed scores by period and calculates period-to-period change and direction.

This is essential for early warning because a student can have an acceptable overall average while one important subject is deteriorating rapidly.

## 10. Ranking

The class ranking function:

- Ranks higher averages first.
- Uses `min` ranking for ties, so equal averages receive equal ranks.
- Excludes students without an observed average from meaningful ranking.
- Calculates a percentile-style position for comparison.

Ranking is descriptive only. It should not be used as the sole basis for promotion, discipline, placement or other high-stakes decisions.

## 11. Outlier detection

The engine uses the interquartile range (IQR) separately within each subject:

- Q1 = 25th percentile
- Q3 = 75th percentile
- IQR = Q3 − Q1
- Low outlier boundary = Q1 − 1.5 × IQR
- High outlier boundary = Q3 + 1.5 × IQR

At least four observed scores are required before an IQR-based subject outlier check is performed.

Outlier detection is a review aid. It does not label a student as weak or strong by itself.

## 12. Data-quality layer

Analytics quality is measured separately from academic performance.

The engine reports:

- Enrolled subject records
- Recorded results
- Missing results
- Result completion rate
- Attendance coverage
- Engagement coverage
- Duplicate result keys detected in the analytical frame

A low completion rate means confidence in the observed academic profile should be reduced. This prevents the system from treating sparse data as a complete academic history.

## 13. Derived broadsheet

The broadsheet is dynamically generated from:

`Student → Placement → Subject Enrollment → Result → Attendance`

It provides:

- Student identity
- Class/stream
- One column per enrolled subject
- Average
- Performance band
- Subjects enrolled
- Subjects recorded
- Completion percentage
- Below-credit count
- Credit-pass percentage
- Attendance percentage

A missing result remains blank.

## 14. What the engine deliberately does not do

Phase 3.1 does **not**:

- Treat missing results as zero
- Treat missing results as F9
- Change stored marks
- Change the assessment policy
- Mix attendance into marks
- Mix engagement into marks
- Claim official WAEC/NECO calculation authority
- Make an AI prediction
- Automatically label a student as a future failure
- Store a duplicate broadsheet table

Those boundaries keep the analytics layer auditable and prepare it for the Phase 3.2/3.3/3.4 intelligence layers.

## 15. Recommended interpretation hierarchy

When reading an academic profile, EduPredict should use this order:

1. **Data completeness** — Is enough information present?
2. **Current performance** — What is the current observed level?
3. **Subject pattern** — Where are strengths and weaknesses?
4. **Consistency** — Is performance stable or highly variable?
5. **Attendance/context** — Is attendance available and concerning?
6. **Trend** — Is performance improving, stable or declining?
7. **Risk** — Phase 3.4 will combine validated indicators.
8. **Prediction** — Phase 3.5 will estimate future difficulty only after the rule-based layer is validated.

## 16. Database safety

The analytics engine is read-derived. It does not require replacement of the existing `edupredict.db` and does not create a separate collated-results table.

## 17. API surface

Main functions in `src/edupredict/analytics.py`:

- `attendance_pct`
- `period_history`
- `result_records`
- `analytics_dataset`
- `student_period_metrics`
- `student_subject_report`
- `subject_analytics`
- `class_analytics`
- `class_overview`
- `trend_analytics`
- `subject_trends`
- `student_trend_by_subject`
- `performance_distribution`
- `ranking_table`
- `identify_outliers`
- `broadsheet`
- `data_quality_summary`

## 18. Phase 3.1 completion criterion

Phase 3.1 is complete when the application can display or export these derived views without changing source records:

- Cohort summary
- Student ranking
- Subject performance table
- Student profile
- Period trend
- Student subject trend
- Grade distribution
- Data-quality/completion summary
- Derived broadsheet

The next phase can therefore build the UI intelligence and early-warning logic on a stable analytical foundation.
