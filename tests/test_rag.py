"""Unit tests for the RAG search helper."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services import rag


@pytest.mark.asyncio
async def test_search_returns_ranked_results():
    fake_records = [
        {"props": {"title": "AI Act", "celex": "32021R0001"}, "score": 0.95},
        {"props": {"title": "Data Act", "celex": "32022R0001"}, "score": 0.72},
    ]

    mock_result = AsyncMock()
    mock_result.data = AsyncMock(return_value=fake_records)

    mock_session = AsyncMock()
    mock_session.run = AsyncMock(return_value=mock_result)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    mock_driver = MagicMock()
    mock_driver.session = MagicMock(return_value=mock_session)
    mock_driver.__aenter__ = AsyncMock(return_value=mock_driver)
    mock_driver.__aexit__ = AsyncMock(return_value=False)

    fake_embedding = [0.1] * 384

    with (
        patch("app.services.rag.AsyncGraphDatabase.driver", return_value=mock_driver),
        patch("app.services.rag.get_model") as mock_get_model,
    ):
        mock_model = MagicMock()
        mock_model.encode = MagicMock(return_value=MagicMock(tolist=lambda: fake_embedding))
        mock_get_model.return_value = mock_model

        results = await rag.search("artificial intelligence regulation", k=2)

    assert len(results) == 2
    assert results[0]["title"] == "AI Act"
    assert results[0]["score"] == pytest.approx(0.95)
    assert results[1]["score"] < results[0]["score"]


@pytest.mark.asyncio
async def test_search_unknown_label_raises():
    with pytest.raises(KeyError):
        await rag.search("test", label="UnknownLabel")
