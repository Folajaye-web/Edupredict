from pathlib import Path


def test_period_selection_uses_stable_database_id_everywhere():
    text = Path(__file__).parents[1].joinpath("app", "streamlit_app.py").read_text(encoding="utf-8")
    assert "options = {plabel(x): int(x.id) for x in clean}" in text
    assert "s.query(AcademicSession).filter(AcademicSession.id == period_id).first()" in text
    assert "ps[[plabel(x) for x in ps].index" not in text
