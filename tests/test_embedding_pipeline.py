"""Unit tests for embedding_pipeline text-building helpers."""

from app.services.cellar_retrieve_pipeline.embedding_pipeline import (
    _work_text,
    _concept_text,
    _agent_text,
)


def test_work_text_all_fields():
    props = {"title": "AI Regulation", "celex": "32021R0001", "date": "2021-04-21", "text": "Article 1..."}
    result = _work_text(props)
    assert "AI Regulation" in result
    assert "32021R0001" in result
    assert "2021-04-21" in result
    assert "Article 1..." in result


def test_work_text_skips_none():
    props = {"title": "AI Regulation", "celex": None, "date": None, "text": None}
    result = _work_text(props)
    assert result == "AI Regulation"
    assert "None" not in result


def test_work_text_missing_keys():
    result = _work_text({})
    assert result == ""


def test_concept_text():
    assert _concept_text({"label": "artificial intelligence"}) == "artificial intelligence"
    assert _concept_text({}) == ""


def test_agent_text():
    assert _agent_text({"label": "European Commission"}) == "European Commission"
    assert _agent_text({}) == ""
