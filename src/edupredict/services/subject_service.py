from ..db import Subject


def deactivate_subject(session, subject_id: int) -> Subject:
    subject = session.get(Subject, subject_id)
    if subject is None:
        raise ValueError('Subject not found.')
    subject.active = False
    session.commit()
    return subject


def restore_subject(session, subject_id: int) -> Subject:
    subject = session.get(Subject, subject_id)
    if subject is None:
        raise ValueError('Subject not found.')
    subject.active = True
    session.commit()
    return subject
