"""One-time, explicit migration from legacy Result components into v3.3 boundaries.

Safe behavior:
- Creates a database backup first.
- Never deletes or rewrites existing Result rows.
- Copies CA components into continuous_assessments when missing.
- Copies non-null exam components into final_examinations when missing.
- Re-running is idempotent.
"""
import shutil
from datetime import datetime, timezone
from pathlib import Path

from edupredict.db import DB_PATH, SessionLocal, Result, ContinuousAssessment, FinalExamination, init_db


def main():
    init_db()
    backup=DB_PATH.with_name(f"edupredict_before_v33_migration_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.db")
    shutil.copy2(DB_PATH,backup)
    ca_count=exam_count=0
    with SessionLocal() as s:
        for r in s.query(Result).all():
            ca=s.query(ContinuousAssessment).filter_by(session_id=r.session_id,student_id=r.student_id,subject_id=r.subject_id).first()
            if ca is None:
                s.add(ContinuousAssessment(session_id=r.session_id,student_id=r.student_id,subject_id=r.subject_id,ca1=r.ca1,ca2=r.ca2,assignment=r.assignment,project=r.project,teacher_comment=r.teacher_comment)); ca_count+=1
            if r.exam is not None:
                ex=s.query(FinalExamination).filter_by(session_id=r.session_id,student_id=r.student_id,subject_id=r.subject_id).first()
                if ex is None:
                    s.add(FinalExamination(session_id=r.session_id,student_id=r.student_id,subject_id=r.subject_id,exam_score=r.exam,comment=r.teacher_comment)); exam_count+=1
        s.commit()
    print(f'Backup: {backup}')
    print(f'CA records created: {ca_count}')
    print(f'Examination records created: {exam_count}')

if __name__=='__main__': main()
