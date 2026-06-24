from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services import rag

router = APIRouter()


class RagRequest(BaseModel):
    query: str
    label: Literal["Work", "Concept", "Agent"] = "Work"
    k: int = 5


class RagResult(BaseModel):
    results: list[dict[str, Any]]


class RagExpandedRequest(BaseModel):
    query: str
    k: int = 5


@router.post("/rag", response_model=RagResult)
async def rag_search(body: RagRequest) -> RagResult:
    # curl -X POST http://localhost:8000/rag -H "Content-Type: application/json" \
    #   -d '{"query": "artificial intelligence regulation", "label": "Work", "k": 5}'
    """Return the top-k Neo4j nodes most similar to the query text."""
    try:
        results = await rag.search(body.query, k=body.k, label=body.label)
    except KeyError:
        raise HTTPException(status_code=400, detail=f"No vector index for label '{body.label}'")
    return RagResult(results=results)


@router.post("/rag/expanded", response_model=RagResult)
async def rag_search_expanded(body: RagExpandedRequest) -> RagResult:
    # curl -X POST http://localhost:8000/rag/expanded -H "Content-Type: application/json" \
    #   -d '{"query": "artificial intelligence regulation", "k": 5}'
    """Return the top-k Works most similar to the query, each expanded with its Concepts, Citations, and Agents."""
    results = await rag.search_expanded(body.query, k=body.k)
    return RagResult(results=results)
