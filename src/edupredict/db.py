from pathlib import Path
from sqlalchemy import create_engine, String, Integer, Float, Date, DateTime, Boolean, Text, ForeignKey, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from datetime import datetime, date, timezone

DB_PATH = Path(__file__).resolve().parents[2] / "edupredict.db"
engine = create_engine(f"sqlite:///{DB_PATH}", future=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

class Base(DeclarativeBase):
    pass

class School(Base):
    __tablename__='schools'
    id: Mapped[int]=mapped_column(primary_key=True)
    name: Mapped[str]=mapped_column(String(200), unique=True)
    state: Mapped[str]=mapped_column(String(80))
    lga: Mapped[str|None]=mapped_column(String(120), nullable=True)
    school_type: Mapped[str]=mapped_column(String(30), default='Private')
    created_at: Mapped[datetime]=mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

class AcademicSession(Base):
    __tablename__='academic_sessions'
    id: Mapped[int]=mapped_column(primary_key=True)
    school_id: Mapped[int]=mapped_column(ForeignKey('schools.id'))
    session_name: Mapped[str]=mapped_column(String(20))
    term: Mapped[str]=mapped_column(String(20))
    is_current: Mapped[bool]=mapped_column(Boolean, default=True)
    __table_args__=(UniqueConstraint('school_id','session_name','term'),)

class ClassGroup(Base):
    """A school-defined class/stream for one academic term."""
    __tablename__='class_groups'
    id: Mapped[int]=mapped_column(primary_key=True)
    school_id: Mapped[int]=mapped_column(ForeignKey('schools.id'))
    session_id: Mapped[int]=mapped_column(ForeignKey('academic_sessions.id'))
    class_level: Mapped[str]=mapped_column(String(30))
    arm: Mapped[str|None]=mapped_column(String(50), nullable=True)
    display_name: Mapped[str]=mapped_column(String(100))
    active: Mapped[bool]=mapped_column(Boolean, default=True)
    __table_args__=(UniqueConstraint('session_id','class_level','arm'),)

class Student(Base):
    __tablename__='students'
    id: Mapped[int]=mapped_column(primary_key=True)
    school_id: Mapped[int]=mapped_column(ForeignKey('schools.id'))
    student_code: Mapped[str]=mapped_column(String(40))
    first_name: Mapped[str]=mapped_column(String(80))
    last_name: Mapped[str]=mapped_column(String(80))
    gender: Mapped[str|None]=mapped_column(String(20), nullable=True)
    class_level: Mapped[str]=mapped_column(String(20))
    arm: Mapped[str|None]=mapped_column(String(50), nullable=True)
    admission_year: Mapped[int|None]=mapped_column(Integer, nullable=True)
    dob: Mapped[date|None]=mapped_column(Date, nullable=True)
    active: Mapped[bool]=mapped_column(Boolean, default=True)
    __table_args__=(UniqueConstraint('school_id','student_code'),)

class StudentProfile(Base):
    """Non-examination student profile/features imported from legacy or school source data.

    Stored separately from Student so demographic/behavioural source fields do not
    become fake class, subject, CA, or examination records.
    """
    __tablename__='student_profiles'
    id: Mapped[int]=mapped_column(primary_key=True)
    student_id: Mapped[int]=mapped_column(ForeignKey('students.id'), unique=True)
    profile_json: Mapped[str]=mapped_column(Text)
    source_format: Mapped[str|None]=mapped_column(String(80), nullable=True)
    updated_at: Mapped[datetime]=mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

class StudentPlacement(Base):
    """Term-specific placement; allows promotion/reassignment without rewriting history."""
    __tablename__='student_placements'
    id: Mapped[int]=mapped_column(primary_key=True)
    session_id: Mapped[int]=mapped_column(ForeignKey('academic_sessions.id'))
    student_id: Mapped[int]=mapped_column(ForeignKey('students.id'))
    class_group_id: Mapped[int]=mapped_column(ForeignKey('class_groups.id'))
    __table_args__=(UniqueConstraint('session_id','student_id'),)

class Subject(Base):
    __tablename__='subjects'
    id: Mapped[int]=mapped_column(primary_key=True)
    code: Mapped[str]=mapped_column(String(30), unique=True)
    name: Mapped[str]=mapped_column(String(120))
    stage: Mapped[str]=mapped_column(String(20))
    compulsory: Mapped[bool]=mapped_column(Boolean, default=False)
    active: Mapped[bool]=mapped_column(Boolean, default=True)

class SubjectOffering(Base):
    """Subjects offered to a class/stream in a specific term."""
    __tablename__='subject_offerings'
    id: Mapped[int]=mapped_column(primary_key=True)
    class_group_id: Mapped[int]=mapped_column(ForeignKey('class_groups.id'))
    subject_id: Mapped[int]=mapped_column(ForeignKey('subjects.id'))
    compulsory: Mapped[bool]=mapped_column(Boolean, default=False)
    __table_args__=(UniqueConstraint('class_group_id','subject_id'),)

class StudentSubjectEnrollment(Base):
    """Actual subject enrollment; absence of enrollment is different from a missing result."""
    __tablename__='student_subject_enrollments'
    id: Mapped[int]=mapped_column(primary_key=True)
    session_id: Mapped[int]=mapped_column(ForeignKey('academic_sessions.id'))
    student_id: Mapped[int]=mapped_column(ForeignKey('students.id'))
    subject_id: Mapped[int]=mapped_column(ForeignKey('subjects.id'))
    active: Mapped[bool]=mapped_column(Boolean, default=True)
    __table_args__=(UniqueConstraint('session_id','student_id','subject_id'),)

class AssessmentPolicy(Base):
    __tablename__='assessment_policies'
    id: Mapped[int]=mapped_column(primary_key=True)
    school_id: Mapped[int]=mapped_column(ForeignKey('schools.id'))
    name: Mapped[str]=mapped_column(String(100))
    ca1_weight: Mapped[float]=mapped_column(Float, default=10)
    ca2_weight: Mapped[float]=mapped_column(Float, default=10)
    assignment_weight: Mapped[float]=mapped_column(Float, default=10)
    project_weight: Mapped[float]=mapped_column(Float, default=10)
    exam_weight: Mapped[float]=mapped_column(Float, default=60)
    credit_threshold: Mapped[float]=mapped_column(Float, default=50)

class Result(Base):
    __tablename__='results'
    id: Mapped[int]=mapped_column(primary_key=True)
    session_id: Mapped[int]=mapped_column(ForeignKey('academic_sessions.id'))
    student_id: Mapped[int]=mapped_column(ForeignKey('students.id'))
    subject_id: Mapped[int]=mapped_column(ForeignKey('subjects.id'))
    ca1: Mapped[float|None]=mapped_column(Float, nullable=True)
    ca2: Mapped[float|None]=mapped_column(Float, nullable=True)
    assignment: Mapped[float|None]=mapped_column(Float, nullable=True)
    project: Mapped[float|None]=mapped_column(Float, nullable=True)
    exam: Mapped[float|None]=mapped_column(Float, nullable=True)
    total: Mapped[float]=mapped_column(Float)
    grade: Mapped[str]=mapped_column(String(3))
    grade_point: Mapped[float]=mapped_column(Float)
    teacher_comment: Mapped[str|None]=mapped_column(Text, nullable=True)
    created_at: Mapped[datetime]=mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime]=mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    __table_args__=(UniqueConstraint('session_id','student_id','subject_id'),)

class ResultRevision(Base):
    """Immutable audit snapshot for result corrections/replacements."""
    __tablename__='result_revisions'
    id: Mapped[int]=mapped_column(primary_key=True)
    result_id: Mapped[int|None]=mapped_column(ForeignKey('results.id'), nullable=True)
    session_id: Mapped[int]=mapped_column(ForeignKey('academic_sessions.id'))
    student_id: Mapped[int]=mapped_column(ForeignKey('students.id'))
    subject_id: Mapped[int]=mapped_column(ForeignKey('subjects.id'))
    ca1: Mapped[float|None]=mapped_column(Float, nullable=True)
    ca2: Mapped[float|None]=mapped_column(Float, nullable=True)
    assignment: Mapped[float|None]=mapped_column(Float, nullable=True)
    project: Mapped[float|None]=mapped_column(Float, nullable=True)
    exam: Mapped[float|None]=mapped_column(Float, nullable=True)
    total: Mapped[float]=mapped_column(Float)
    grade: Mapped[str]=mapped_column(String(3))
    grade_point: Mapped[float]=mapped_column(Float)
    teacher_comment: Mapped[str|None]=mapped_column(Text, nullable=True)
    action: Mapped[str]=mapped_column(String(30))
    reason: Mapped[str|None]=mapped_column(Text, nullable=True)
    recorded_at: Mapped[datetime]=mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

class Attendance(Base):
    __tablename__='attendance'
    id: Mapped[int]=mapped_column(primary_key=True)
    session_id: Mapped[int]=mapped_column(ForeignKey('academic_sessions.id'))
    student_id: Mapped[int]=mapped_column(ForeignKey('students.id'))
    school_days: Mapped[int]=mapped_column(Integer)
    days_present: Mapped[int]=mapped_column(Integer)
    days_absent: Mapped[int]=mapped_column(Integer)
    __table_args__=(UniqueConstraint('session_id','student_id'),)

class Engagement(Base):
    __tablename__='engagement'
    id: Mapped[int]=mapped_column(primary_key=True)
    session_id: Mapped[int]=mapped_column(ForeignKey('academic_sessions.id'))
    student_id: Mapped[int]=mapped_column(ForeignKey('students.id'))
    homework_rate: Mapped[float|None]=mapped_column(Float, nullable=True)
    participation: Mapped[float|None]=mapped_column(Float, nullable=True)
    behaviour: Mapped[float|None]=mapped_column(Float, nullable=True)
    study_hours_week: Mapped[float|None]=mapped_column(Float, nullable=True)
    __table_args__=(UniqueConstraint('session_id','student_id'),)

class ContinuousAssessment(Base):
    """Pre-examination assessment evidence. Never contains the current exam score."""
    __tablename__='continuous_assessments'
    id: Mapped[int]=mapped_column(primary_key=True)
    session_id: Mapped[int]=mapped_column(ForeignKey('academic_sessions.id'))
    student_id: Mapped[int]=mapped_column(ForeignKey('students.id'))
    subject_id: Mapped[int]=mapped_column(ForeignKey('subjects.id'))
    ca1: Mapped[float|None]=mapped_column(Float, nullable=True)
    ca2: Mapped[float|None]=mapped_column(Float, nullable=True)
    assignment: Mapped[float|None]=mapped_column(Float, nullable=True)
    project: Mapped[float|None]=mapped_column(Float, nullable=True)
    teacher_comment: Mapped[str|None]=mapped_column(Text, nullable=True)
    created_at: Mapped[datetime]=mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime]=mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    __table_args__=(UniqueConstraint('session_id','student_id','subject_id'),)

class FinalExamination(Base):
    """Actual current examination result; created only after the examination."""
    __tablename__='final_examinations'
    id: Mapped[int]=mapped_column(primary_key=True)
    session_id: Mapped[int]=mapped_column(ForeignKey('academic_sessions.id'))
    student_id: Mapped[int]=mapped_column(ForeignKey('students.id'))
    subject_id: Mapped[int]=mapped_column(ForeignKey('subjects.id'))
    exam_score: Mapped[float]=mapped_column(Float)
    exam_date: Mapped[date|None]=mapped_column(Date, nullable=True)
    comment: Mapped[str|None]=mapped_column(Text, nullable=True)
    created_at: Mapped[datetime]=mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime]=mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    __table_args__=(UniqueConstraint('session_id','student_id','subject_id'),)

class Intervention(Base):
    __tablename__='interventions'
    id: Mapped[int]=mapped_column(primary_key=True)
    session_id: Mapped[int]=mapped_column(ForeignKey('academic_sessions.id'))
    student_id: Mapped[int]=mapped_column(ForeignKey('students.id'))
    subject_id: Mapped[int|None]=mapped_column(ForeignKey('subjects.id'), nullable=True)
    prediction_id: Mapped[int|None]=mapped_column(ForeignKey('predictions.id'), nullable=True)
    intervention_type: Mapped[str]=mapped_column(String(80))
    description: Mapped[str]=mapped_column(Text)
    assigned_to: Mapped[str|None]=mapped_column(String(120), nullable=True)
    status: Mapped[str]=mapped_column(String(30), default='OPEN')
    start_date: Mapped[date|None]=mapped_column(Date, nullable=True)
    review_date: Mapped[date|None]=mapped_column(Date, nullable=True)
    completion_date: Mapped[date|None]=mapped_column(Date, nullable=True)
    outcome: Mapped[str|None]=mapped_column(Text, nullable=True)
    created_at: Mapped[datetime]=mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime]=mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

class Prediction(Base):
    __tablename__='predictions'
    id: Mapped[int]=mapped_column(primary_key=True)
    session_id: Mapped[int]=mapped_column(ForeignKey('academic_sessions.id'))
    student_id: Mapped[int]=mapped_column(ForeignKey('students.id'))
    subject_id: Mapped[int|None]=mapped_column(ForeignKey('subjects.id'), nullable=True)
    predicted_score: Mapped[float]=mapped_column(Float)
    risk_level: Mapped[str]=mapped_column(String(20))
    below_credit_probability: Mapped[float]=mapped_column(Float)
    rationale: Mapped[str]=mapped_column(Text)
    model_version: Mapped[str]=mapped_column(String(40), default='unknown')
    feature_snapshot: Mapped[str|None]=mapped_column(Text, nullable=True)
    created_at: Mapped[datetime]=mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

class PredictionEvaluation(Base):
    __tablename__='prediction_evaluations'
    id: Mapped[int]=mapped_column(primary_key=True)
    prediction_id: Mapped[int]=mapped_column(ForeignKey('predictions.id'))
    session_id: Mapped[int]=mapped_column(ForeignKey('academic_sessions.id'))
    student_id: Mapped[int]=mapped_column(ForeignKey('students.id'))
    subject_id: Mapped[int]=mapped_column(ForeignKey('subjects.id'))
    actual_exam_score: Mapped[float]=mapped_column(Float)
    error: Mapped[float]=mapped_column(Float)
    absolute_error: Mapped[float]=mapped_column(Float)
    correct_risk: Mapped[bool|None]=mapped_column(Boolean, nullable=True)
    evaluated_at: Mapped[datetime]=mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    __table_args__=(UniqueConstraint('prediction_id'),)

def _migrate_legacy_schema():
    """Apply additive SQLite changes needed when opening an older EduPredict database.

    Result uniqueness is enforced separately after duplicate repair; existing duplicates
    must never be silently deleted during application startup.
    """
    with engine.begin() as conn:
        tables=conn.exec_driver_sql("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        names={row[0] for row in tables}
        if 'subjects' in names:
            cols={row[1] for row in conn.exec_driver_sql('PRAGMA table_info(subjects)').fetchall()}
            if 'active' not in cols:
                conn.exec_driver_sql('ALTER TABLE subjects ADD COLUMN active BOOLEAN NOT NULL DEFAULT 1')
        if 'results' in names:
            cols={row[1] for row in conn.exec_driver_sql('PRAGMA table_info(results)').fetchall()}
            if 'updated_at' not in cols:
                conn.exec_driver_sql('ALTER TABLE results ADD COLUMN updated_at DATETIME')
                conn.exec_driver_sql("UPDATE results SET updated_at=COALESCE(created_at, CURRENT_TIMESTAMP) WHERE updated_at IS NULL")
        if 'predictions' in names:
            cols={row[1] for row in conn.exec_driver_sql('PRAGMA table_info(predictions)').fetchall()}
            if 'model_version' not in cols:
                conn.exec_driver_sql("ALTER TABLE predictions ADD COLUMN model_version VARCHAR(40) NOT NULL DEFAULT 'unknown'")
            if 'feature_snapshot' not in cols:
                conn.exec_driver_sql('ALTER TABLE predictions ADD COLUMN feature_snapshot TEXT')

def ensure_result_unique_guard():
    """Create the hard database guard only when the existing data is clean."""
    with engine.begin() as conn:
        dup=conn.exec_driver_sql("""
            SELECT 1 FROM results
            GROUP BY session_id, student_id, subject_id
            HAVING COUNT(*) > 1 LIMIT 1
        """).fetchone()
        if dup:
            return False
        conn.exec_driver_sql("""
            CREATE UNIQUE INDEX IF NOT EXISTS ux_results_session_student_subject
            ON results(session_id, student_id, subject_id)
        """)
        return True


def init_db():
    _migrate_legacy_schema()
    Base.metadata.create_all(engine)
    ensure_result_unique_guard()
    with SessionLocal() as s:
        school = s.query(School).first()
        if not school:
            school=School(name='Demo Nigerian School', state='Lagos', lga='Lagos Mainland', school_type='Private')
            s.add(school); s.commit()
        # Backfill term-specific class placement for existing demo records without changing student master data.
        sessions=s.query(AcademicSession).filter_by(school_id=school.id).all()
        for sess in sessions:
            groups={}
            for st in s.query(Student).filter_by(school_id=school.id, active=True).all():
                key=(st.class_level, st.arm or '')
                cg=s.query(ClassGroup).filter_by(session_id=sess.id,class_level=key[0],arm=key[1] or None).first()
                if not cg:
                    cg=ClassGroup(school_id=school.id,session_id=sess.id,class_level=key[0],arm=key[1] or None,display_name=f"{key[0]} {key[1]}".strip())
                    s.add(cg); s.flush()
                if not s.query(StudentPlacement).filter_by(session_id=sess.id,student_id=st.id).first():
                    s.add(StudentPlacement(session_id=sess.id,student_id=st.id,class_group_id=cg.id))
        s.commit()

def reset_db():
    Base.metadata.drop_all(engine); Base.metadata.create_all(engine); init_db()
