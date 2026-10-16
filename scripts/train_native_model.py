from edupredict.db import SessionLocal, School
from edupredict.ai.native import train_native

if __name__ == '__main__':
    with SessionLocal() as s:
        school=s.query(School).first()
        if not school: raise SystemExit('No school configured.')
        metrics=train_native(s,school.id)
        print(metrics)
