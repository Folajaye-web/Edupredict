from edupredict.db import Base


def test_structure_tables():
    assert 'class_groups' in Base.metadata.tables
    assert 'student_placements' in Base.metadata.tables
    assert 'subject_offerings' in Base.metadata.tables
    assert 'student_subject_enrollments' in Base.metadata.tables
