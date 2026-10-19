# EduPredict 🇳🇬
### AI-Powered Student Academic Intelligence and Early-Warning System

EduPredict is an AI-powered educational analytics platform designed to help Nigerianss
 schools monitor student performance, identify learners at risk of academic underperformance, and support timely, data-driven educational interventions.

By combining academic results, continuous assessment, attendance, engagement indicators, and historical performance, EduPredict aims to help educators move from reactive assessment to proactive student support.

> **Project status:** Under development and undergoing model-quality validation. Predictive performance has been evaluated on historical Nigerian school records, but production qualification and pilot authorisation remain pending.

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Project Objectives](#project-objectives)
- [Key Features](#key-features)
- [How EduPredict Works](#how-edupredict-works)
- [Technology Stack](#technology-stack)
- [Data Model and Academic Structure](#data-model-and-academic-structure)
- [AI and Machine Learning](#ai-and-machine-learning)
- [Model Validation](#model-validation)
- [Installation and Setup](#installation-and-setup)
- [Running the Application](#running-the-application)
- [Project Structure](#project-structure)
- [Data Privacy and Responsible AI](#data-privacy-and-responsible-ai)
- [Current Limitations](#current-limitations)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview

Educational institutions generate valuable information about student achievement, attendance, engagement, and learning progress. However, these records are not always used systematically to identify early signs of academic difficulty.

EduPredict is being developed to address this challenge through a combination of academic record management, educational analytics, machine learning, and early-warning capabilities.

The platform is designed around the Nigerian secondary-school context, including:

- Junior Secondary School (JSS1–JSS3) and Senior Secondary School (SS1–SS3).
- Academic sessions and terms.
- Classes, arms or streams, subjects, and student enrolment.
- Continuous Assessment (CA) and examination results.
- Attendance and historical academic performance.
- Academic-risk prediction and intervention recommendations.

The long-term objective is to provide schools with an integrated tool for understanding academic progress and identifying students who may benefit from additional support.

## Problem Statement

Schools often identify academic difficulties after examination results have been released. By that stage, students may already have experienced sustained learning gaps.

Common challenges include:

- Fragmented student academic records.
- Limited visibility into performance trends across terms.
- Difficulty identifying students whose performance is declining.
- Inconsistent use of attendance and historical results in academic decisions.
- Limited tools for evaluating whether interventions are helping students improve.
- Manual analysis of large volumes of student assessment records.

EduPredict aims to make these records more useful by presenting understandable academic insights and developing predictive tools that can support earlier intervention.

## Project Objectives

1. **Centralise academic information:** Organise student profiles, academic structures, assessments, attendance, and results.
2. **Monitor performance:** Track results by student, subject, class, term, and academic session.
3. **Identify academic risk:** Develop machine-learning models that estimate the likelihood of a student falling below a defined performance threshold.
4. **Support early intervention:** Generate recommendations that help educators decide which students may need additional attention.
5. **Evaluate model reliability:** Validate predictive performance, probability calibration, data integrity, and reproducibility.
6. **Support evidence-based decisions:** Help schools use historical and current academic evidence to plan appropriate student support.

## Key Features

### 1. Academic Dashboard

Provides a central view of academic information and school performance indicators.

Planned and implemented capabilities depend on the current application build and include summaries of student records, academic performance, and other available school indicators.

### 2. Student Management

Supports the organisation of student records and access to student information.

The broader application workflow includes student listing, record editing, and deactivation, subject to the stability of the current build.

### 3. Academic Structure Management

Models the academic organisation of participating schools, including:

- Academic sessions and terms.
- Class levels.
- Classes and arms or streams.
- Subject catalogues.
- Student enrolment.

### 4. Continuous Assessment and Results

Supports the academic workflow for recording and analysing assessment components.

The project’s data model includes:

- CA1
- CA2
- Assignment
- Project
- Examination
- Total score
- Grade

Assessment rules and grading configurations must be checked against the relevant school’s approved grading policy before operational use.

### 5. Attendance Analytics

Uses attendance information as part of the broader academic-monitoring and predictive framework.

Attendance analysis can help educators investigate possible relationships between school participation and academic outcomes. It should not be treated as proof that absence alone causes poor performance.

### 6. AI Early Warning

Develops predictive models to identify students who may be at risk of underperforming.

The classification task uses a defined academic-performance threshold to estimate the likelihood of a student falling below the target. The regression task estimates academic outcomes numerically.

Predictions are intended to support professional judgement, not replace teachers or school administrators.

### 7. Intervention Recommendations

EduPredict is designed to support follow-up actions for students who may need additional assistance.

Possible interventions include remedial classes, extra lessons, targeted subject support, attendance follow-up, and progress monitoring.

Recommendations should be appropriate to the student’s circumstances and reviewed by authorised educators.

### 8. Model-Quality Validation

The project includes a structured validation pipeline intended to evaluate:

- Dataset integrity.
- Temporal training and evaluation splits.
- Predictive performance.
- Calibration of predicted probabilities.
- Subject and subgroup performance.
- Model artifact integrity.
- Reproducibility.
- Technical release readiness.

Passing an individual validation check does not automatically mean that the complete system is ready for production.

## How EduPredict Works

The intended workflow is:

1. **Collect:** Import or record academic, attendance, and student information.
2. **Validate:** Check data quality, scoring consistency, missing information, and academic structure.
3. **Process:** Prepare historical features and organise data by academic session.
4. **Analyse:** Generate descriptive academic statistics and performance trends.
5. **Predict:** Apply validated machine-learning models to eligible student records.
6. **Flag:** Identify students whose estimated academic risk warrants review.
7. **Recommend:** Present suitable intervention options for educator consideration.
8. **Monitor:** Track subsequent academic performance and evaluate outcomes over time.

The predictive workflow must use only information that would genuinely be available at the time a prediction is made. Examination outcomes or other information revealed after the prediction point must not leak into the model’s input features.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application and model-development language |
| Streamlit | Interactive application interface |
| Pandas | Tabular data processing |
| NumPy | Numerical operations |
| Scikit-learn | Machine-learning pipelines and model evaluation |
| Matplotlib | Data visualisation |
| Seaborn | Statistical visualisation where used |
| Jupyter Notebook | Exploratory analysis and experimentation |
| SQLite | Local relational database used by the application |
| SQLAlchemy | Database interaction where used |

The final dependency list should be taken from the project's actual environment and dependency files rather than inferred from this overview.

## Data Model and Academic Structure

EduPredict is being developed around Nigerian school records and a longitudinal academic data structure.

Core academic entities in the application model include:

- **School:** The institution to which student and academic records belong.
- **Academic session:** The school year and associated academic period.
- **Class group:** Class level and arm or stream.
- **Student:** A student’s school-linked record.
- **Subject:** A subject offered within the school’s academic structure.
- **Result:** Assessment and examination information for a student, subject, and academic period.
- **Attendance:** Attendance information for the relevant student and period.
- **Engagement:** Additional engagement information where available.
- **Intervention:** Records of support actions.
- **Prediction:** Model-generated academic-risk or outcome estimates.

The application uses relational data structures to support the organisation and retrieval of these records. Exact database fields, constraints, and supported operations should be verified against the current schema.

### Historical validation dataset

The project has used a combined historical dataset named `Nigeria_Student_Records_2023_2026_Combined.csv`.

The recorded validation dataset contains:

- 156,930 student-subject-term records.
- 12 secondary schools.
- Academic sessions spanning 2023/2024 to 2025/2026.
- Three terms per academic session.
- Classes from JSS1 to SS3.
- Arms A and B.
- Academic scores, grades, attendance, and selected historical performance features.

These figures describe the dataset used in the validation work; they do not establish that all records are independently verified administrative records.

**Data note:** Do not upload student-level records, identifiable school data, credentials, local database files, or confidential model artifacts to a public repository without appropriate authorisation and privacy review.

## AI and Machine Learning

EduPredict’s predictive work includes two main modelling tasks.

### Academic outcome regression

Regression estimates a numerical academic outcome, such as a student’s expected score.

Evaluation metrics include:

- **Mean Absolute Error (MAE):** Average absolute prediction error.
- **Root Mean Squared Error (RMSE):** Error measure that penalises larger mistakes more heavily.
- **R²:** A measure of explained variation relative to a baseline.

### Academic-risk classification

Classification estimates the probability of falling below a defined academic-performance threshold.

Evaluation metrics include:

- **ROC-AUC:** Ability to distinguish between the defined outcome classes.
- **PR-AUC:** Precision-recall performance, particularly relevant when outcomes are imbalanced.
- **Precision:** The proportion of flagged cases that meet the target risk definition.
- **Recall:** The proportion of target risk cases successfully identified.
- **F1-score:** A combined measure of precision and recall.
- **Brier score and log loss:** Measures of probabilistic prediction quality.

### Feature design

The documented validation pipeline includes predictors such as:

- School, class, arm, subject, term, and gender.
- Continuous assessment components.
- Attendance.
- Previous-term totals.
- Previous-year averages.
- Historical subject and overall performance indicators.

Feature availability depends on the prediction point. A production pipeline must enforce a strict feature contract to prevent future information from entering a prediction.

### Risk thresholds

A probability threshold determines when a student is flagged for review. Changing the threshold changes the balance between precision and recall and affects how many students are flagged.

The threshold must be selected using appropriate development data, documented, and reviewed for its operational consequences. It should not be changed simply to make a validation gate pass.

## Model Validation

EduPredict has undergone staged historical model validation, including temporal holdout evaluation and subsequent technical release checks.

The recorded Phase 5 final-test results were:

| Metric | Recorded result |
|---|---:|
| Regression MAE | 5.18 |
| Regression RMSE | 6.47 |
| Regression R² | 0.610 |
| Classification ROC-AUC | 0.956 |
| Classification PR-AUC | 0.787 |
| Classification F1 at threshold 0.50 | 0.699 |
| Classification Brier score | 0.055 |

These are historical evaluation results, not a guarantee of performance on new schools, students, or future academic sessions. The classification threshold used for the Phase 7 evaluation was subsequently selected as approximately 0.40; metrics at different thresholds should not be treated as directly interchangeable.

### Current qualification status

The recorded Phase 7 qualification report has status `CONDITIONAL_REVIEW`.

The subsequent Phase 7C technical release gate confirms that:

- Required model artifacts are present.
- Classification and regression prediction smoke tests pass.
- Reproducibility checks pass.
- Phase 7A and Phase 7B gates pass.
- The production contract does not pass because technical qualification has not yet been granted.

A calibration review remains required. The recorded Phase 6 holdout maximum calibration error was approximately 0.1124, above the 0.10 review threshold used in this pipeline.

Accordingly, **EduPredict is not yet technically qualified for production release, and pilot authorisation remains pending.**

The correct next step is to resolve the outstanding qualification evidence without weakening the release criteria. A genuinely new, untouched cohort should be used for independent confirmation where possible, because repeated examination of an existing holdout reduces its value as pristine confirmatory evidence.

## Installation and Setup

The following setup is intended for local development on Windows. It assumes the project contains the referenced application entry point and a `requirements.txt` file. Adjust the paths or entry-point filename if the repository differs.

### Prerequisites

- Python compatible with the project’s pinned dependencies.
- Git.
- A terminal such as PowerShell.
- The EduPredict source code.
- An authorised dataset for local testing.

### 1. Clone the repository

Replace the example URL with the actual GitHub repository URL.

```powershell
git clone https://github.com/YOUR-USERNAME/EduPredict.git
cd EduPredict
```

### 2. Create a virtual environment

```powershell
py -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Using a virtual environment keeps project dependencies separate from the system-wide Python installation.

If PowerShell blocks activation, follow your organisation’s security policy or activate the environment using an alternative supported terminal method.

### 3. Install dependencies

If `requirements.txt` is present:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Use the dependency versions specified by the project. Do not arbitrarily upgrade scikit-learn or other model-related libraries, as this may affect compatibility or reproducibility.

### 4. Configure the dataset and database

Place authorised development data in the location expected by the application or its configuration.

Before running an import:

- Verify the dataset schema.
- Check score ranges and grading consistency.
- Confirm academic session, term, class, and subject values.
- Confirm how student identifiers are generated and maintained.
- Back up the local database before any schema migration or bulk import.

Do not assume that the historical validation dataset is automatically imported into the live application database.

## Running the Application

From the project root, activate the virtual environment and run the configured Streamlit entry point.

For example, if the entry point is `app/streamlit_app.py`:

```powershell
python -m streamlit run .\app\streamlit_app.py
```

Open the local URL printed in the terminal, commonly:

```text
http://localhost:8501
```

If the entry-point filename differs, use the actual file in the repository.

For development and stability verification, test the application in a controlled order:

1. Dashboard loading.
2. Student listing and record management.
3. Academic sessions and terms.
4. Classes, arms, subjects, and student enrolment.
5. Continuous Assessment save and update.
6. Attendance save, update, and percentage calculations.
7. Results processing.
8. AI Early Warning and prediction behaviour.

Check database persistence after each write operation. Do not treat a page loading successfully as proof that its data operations or predictions are correct.

## Project Structure

The following is a conceptual overview of the project areas. Actual paths and filenames may differ between repository versions.

```text
EduPredict/
├── app/                         # Streamlit application
├── src/                         # Application and modelling logic
│   └── edupredict/              # Core Python package
├── scripts/
│   └── nigeria_validation/      # Historical validation scripts
├── data/                        # Local data; exclude private data
├── models/                      # Model artifacts; review before sharing
├── reports/                     # Validation reports
├── db/                          # Local database files; exclude private data
├── requirements.txt             # Python dependencies, if provided
├── .gitignore
└── README.md
```

This is an illustrative structure, not a guarantee that every directory is present in every version.

## Data Privacy and Responsible AI

EduPredict deals with educational information that may be sensitive. Responsible handling of student data is essential.

The project should follow these principles:

- **Data minimisation:** Collect only the information required for defined educational purposes.
- **Access control:** Restrict student records and risk predictions to authorised users.
- **Privacy:** Remove direct identifiers from datasets used for public examples and development where appropriate.
- **Secure storage:** Protect database files, model artifacts, backups, and credentials.
- **Human oversight:** Require educators to review risk flags and recommendations.
- **Fairness evaluation:** Examine model performance across relevant schools, classes, subjects, and student groups.
- **Transparency:** Explain the purpose and limitations of predictions.
- **Non-punitive use:** Use risk estimates to offer support, not to label, stigmatise, or automatically penalise students.
- **Monitoring:** Reassess performance when school populations, grading practices, or data collection procedures change.

A model prediction is an estimate, not a definitive statement about a student’s ability or future.

## Current Limitations

EduPredict is an evolving project. The following limitations should be recognised:

1. Production qualification has not been completed.
2. The calibration review remains outstanding.
3. Historical evaluation results may not generalise to new schools or future sessions.
4. The recorded dataset’s representativeness and provenance require appropriate verification before external claims are made.
5. Changes in grading, data collection, or student populations may affect model performance.
6. Application functionality must be verified against the current build.
7. Intervention recommendations require review and outcome monitoring.
8. Security, access control, deployment, and privacy requirements must be assessed before real student data is used in a live deployment.

## Roadmap

### Application stability
- Complete stability checks for core pages and database operations.
- Verify academic structure and student enrolment workflows.
- Validate assessment, attendance, and results persistence.
- Improve input validation and error handling.

### Data quality
- Strengthen ingestion and schema validation.
- Verify student identity consistency across academic sessions.
- Audit missing fields, grading rules, and temporal ordering.
- Document dataset provenance and permitted uses.

### Model quality
- Resolve calibration and qualification requirements.
- Preserve strict separation between training, validation, and final evaluation data.
- Evaluate subgroup and subject-level performance.
- Conduct prospective evaluation on new records.
- Document threshold selection and operational consequences.

### Responsible deployment
- Complete security and privacy reviews.
- Establish role-based access where required.
- Add prediction monitoring and model-version tracking.
- Evaluate intervention outcomes with school partners.
- Obtain the required technical and governance approvals before a pilot or production release.

## Contributing

Contributions that improve educational usefulness, data quality, software reliability, model evaluation, and responsible AI are welcome.

Suggested contribution process:

1. Open an issue describing the problem or proposed improvement.
2. Create a focused branch for the change.
3. Implement the change with suitable tests.
4. Run the relevant validation checks.
5. Document changes to the data schema, model features, or application workflow.
6. Submit a pull request explaining the change and its validation evidence.

Do not include identifiable student records, private school information, credentials, or unapproved model artifacts in issues or pull requests.

## License

No license has been specified in this README. Add a license only after the project owner has selected the appropriate terms. Until then, do not assume that the repository grants permission to reuse or redistribute the code.

## Contact

**Project:** EduPredict  
**Focus:** AI-powered academic intelligence and early warning for Nigerian schools  
**Repository:** https://github.com/Folajaye-web/Edupredict.git
**Maintainer:** Alfred Nathaniel, +234-806-6948177, zealousgoldmind37@gmail.com
---

*EduPredict aims to help schools identify academic difficulties earlier, understand student progress more clearly, and make better-informed decisions about educational support.*
