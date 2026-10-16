from edupredict.db import Base
def test_tables():
 assert "students" in Base.metadata.tables
 assert "results" in Base.metadata.tables
 assert "attendance" in Base.metadata.tables

def test_analytics_module_imports():
 from edupredict.analytics import broadsheet, result_records
 assert callable(broadsheet)
 assert callable(result_records)
