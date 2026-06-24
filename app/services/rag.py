"""RAG helper: embed a query and retrieve similar nodes from Neo4j by vector similarity."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from neo4j import AsyncGraphDatabase

from app.core.config import NEO4J_AUTH, NEO4J_URI
from app.core.models import Agent, Concept, ExpandedWork, Work
from app.services.cellar_retrieve_pipeline.embedding_pipeline import get_model

_VECTOR_SEARCH = """
CALL db.index.vector.queryNodes($index_name, $k, $embedding)
YIELD node, score
RETURN properties(node) AS props, score
"""

_EXPAND_WORKS = """
UNWIND $uris AS uri
MATCH (w:Work {uri: uri})
OPTIONAL MATCH (w)-[:ABOUT]->(c:Concept)
OPTIONAL MATCH (w)-[:CITES]->(cited:Work)
OPTIONAL MATCH (w)-[:AUTHORED_BY]->(a:Agent)
RETURN
    uri,
    [c IN collect(DISTINCT c) | {uri: c.uri, label: c.label}] AS concepts,
    [cited IN collect(DISTINCT cited) | {uri: cited.uri, celex: cited.celex, title: cited.title, date: cited.date}] AS citations,
    [a IN collect(DISTINCT a) | {uri: a.uri, label: a.label}] AS agents
"""

_INDEX_NAMES = {
    "Work": "work_embedding_idx",
    "Concept": "concept_embedding_idx",
    "Agent": "agent_embedding_idx",
}


async def search(query: str, k: int = 5, label: str = "Work") -> list[dict[str, Any]]:
    """Embed query and return top-k similar nodes from Neo4j by cosine similarity.

    Returns a list of dicts — each node's properties plus a `score` key (0–1).
    """
    if label not in _INDEX_NAMES:
        raise ValueError(f"label must be one of {list(_INDEX_NAMES)}")
    embedding = get_model().encode(query, convert_to_numpy=True).tolist()

    async with AsyncGraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH) as driver:
        async with driver.session() as session:
            result = await session.run(
                _VECTOR_SEARCH,
                index_name=_INDEX_NAMES[label],
                k=k,
                embedding=embedding,
            )
            records = await result.data()

    return [
        {k: v for k, v in r["props"].items() if k != "embedding"} | {"score": r["score"]}
        for r in records
    ]


async def search_expanded(query: str, k: int = 5) -> list[dict[str, Any]]:
    """RAG search over Works, each result expanded with its Concepts, Citations, and Agents.

    Returns a JSON-serializable list of ExpandedWork dicts, preserving RAG ranking order.
    """
    base_results = await search(query, k=k, label="Work")
    if not base_results:
        return []

    uris = [r["uri"] for r in base_results]
    score_by_uri = {r["uri"]: r["score"] for r in base_results}
    props_by_uri = {r["uri"]: r for r in base_results}

    async with AsyncGraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH) as driver:
        async with driver.session() as session:
            result = await session.run(_EXPAND_WORKS, uris=uris)
            expansion = {row["uri"]: row for row in await result.data()}

    out = []
    for uri in uris:
        p = props_by_uri[uri]
        exp = expansion.get(uri, {})
        expanded = ExpandedWork(
            uri=uri,
            celex=p.get("celex"),
            title=p.get("title"),
            date=p.get("date"),
            abstract=p.get("abstract"),
            text=p.get("text"),
            score=score_by_uri[uri],
            concepts=[Concept(**c) for c in exp.get("concepts", [])],
            citations=[Work(uri=c["uri"], celex=c.get("celex"), title=c.get("title"), date=c.get("date")) for c in exp.get("citations", [])],
            agents=[Agent(**a) for a in exp.get("agents", [])],
        )
        out.append(asdict(expanded))
    return out
