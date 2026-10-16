from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from edupredict.db import Base, School, Subject
from edupredict.services.subject_service import deactivate_subject, restore_subject


def test_subject_archive_and_restore_preserve_record():
    e=create_engine('sqlite:///:memory:')
    Base.metadata.create_all(e)
    S=sessionmaker(bind=e, expire_on_commit=False)
    s=S(); sub=Subject(code='OLD',name='Old Subject',stage='SSS',active=True); s.add(sub); s.commit()
    deactivate_subject(s, sub.id)
    assert s.get(Subject, sub.id).active is False
    restore_subject(s, sub.id)
    assert s.get(Subject, sub.id).active is True
