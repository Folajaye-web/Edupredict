from pydantic import BaseModel, Field, field_validator
from typing import Optional

class StudentRow(BaseModel):
    student_code: str
    first_name: str
    last_name: str
    gender: Optional[str]=None
    class_level: str
    arm: Optional[str]=None
    admission_year: Optional[int]=None

class ResultRow(BaseModel):
    student_code: str
    session: str
    term: str
    subject_code: str
    ca1: float=Field(default=0, ge=0, le=100)
    ca2: float=Field(default=0, ge=0, le=100)
    assignment: float=Field(default=0, ge=0, le=100)
    project: float=Field(default=0, ge=0, le=100)
    exam: float=Field(default=0, ge=0, le=100)
    teacher_comment: Optional[str]=None

class AttendanceRow(BaseModel):
    student_code: str
    session: str
    term: str
    school_days: int=Field(ge=0)
    days_present: int=Field(ge=0)
    days_absent: int=Field(ge=0)

    @field_validator('days_present','days_absent')
    @classmethod
    def nonnegative(cls,v):
        if v<0: raise ValueError('must be non-negative')
        return v
