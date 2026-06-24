"""Agent-facing tools.

This is the model-facing layer. The docstrings and argument names below are
what the model reads to decide when and how to call each tool, so they are
written for that audience — narrow and intention-revealing — rather than
mirroring the service signatures one-to-one. The actual work is delegated to
``app.services.cellar``.
"""

from __future__ import annotations

from langchain_core.tools import tool

from app.services import cellar


@tool
async def search_eu_legislation(topic: str, limit: int = 5) -> list[dict]:
    """Search EU legislation by topic (e.g. "data protection", "copyright").

    Use this to find relevant EU legal acts. Returns a list of records, each
    with the CELEX identifier, title, act type, and adoption date.
    """
    return await cellar.search_legislation(topic, limit)


@tool
async def get_eu_document(celex: str) -> dict | None:
    """Fetch a single EU legislation record by its CELEX identifier.

    Use this when you already know the CELEX id (e.g. "32016R0679") and want
    that specific act. Returns the record, or null if no document matches.
    """
    return await cellar.get_document_by_celex(celex)


# Exposed as a list so the graph can bind the whole toolset in one place.
TOOLS = [search_eu_legislation, get_eu_document]
