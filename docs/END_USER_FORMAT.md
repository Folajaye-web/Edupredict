# EduPredict — Refined End-User Format

## 1. School setup
- School profile: name, state, LGA, school type.
- Academic periods: session + First/Second/Third Term.
- One current period can be designated for day-to-day work.

## 2. Class organisation
A class level is separate from its school-defined arm/stream.

Examples:
- JSS2 A / JSS2 B
- SSS2 Science / SSS2 Arts / SSS2 Commercial

The system does not assume that every school uses A/B/C. The school defines the label.

Students are placed into a class/stream per academic period, so promotion or movement does not rewrite historical records.

## 3. Subject catalogue and enrolment
The school maintains its own active subject catalogue.

A subject can be offered to a class/stream and then enrolled for students. Individual exceptions are supported.

This creates three distinct states:
1. Enrolled + result recorded = score shown.
2. Enrolled + result missing = blank/no result; not a failure.
3. Not enrolled = excluded from that student's subject performance calculation.

## 4. Results
Results are stored in normalized long format:
`student + period + subject + assessment components`.

The Results area provides:
- Class Broadsheet
- Individual Student Report
- Enter / Update Result

Result entry requires academic period, class/stream, student and an enrolled subject. Existing results are updated rather than duplicated.

## 5. Attendance
Attendance is tied to the same academic period and student. Class/stream and student selection are period-aware. Days absent are calculated from school days minus days present.

## 6. Dashboard
Dashboard metrics are student-level rather than raw-result-level:
- active students
- students with results
- average student performance
- students below the configured credit threshold
- credit rate

## 7. AI readiness
The structure is deliberately designed for Phase 3:
`Academic data → performance trends → attendance/engagement → risk factors → prediction → explainable intervention support`.

AI outputs remain decision-support only. Gender and similar demographic fields should be used for monitoring/audit, not as automatic exclusion criteria.

## 8. Nigerian alignment
The system supports school-configured subjects, terms, assessment weights and reporting bands. It is not an official NERDC, FME, WAEC, NECO or NABTEB system and does not claim to calculate official examination certification grades.
