"""Retrieve pipeline: pull a domain of EU documents from CELLAR into Neo4j.

Mirrors the CDM graph structure:
  (:Work)-[:ABOUT]->(:Concept)
  (:Work)-[:CITES]->(:Work)
  (:Work)-[:AUTHORED_BY]->(:Agent)

A "domain" is defined by a SPARQL query that must project ?work (CELLAR URI)
and optionally ?celex, ?title, ?date, ?abstract. Works are keyed by their CELLAR URI.

Pipeline order:
  1. Fetch domain works from SPARQL
  2. Fetch concepts, citations, agents sequentially (each internally batched)
  3. Store everything to Neo4j (creates cited Work stubs)
  4. Fetch text concurrently for all Work nodes (original + cited stubs)
  5. Write text back to Neo4j
"""

from __future__ import annotations

import asyncio
import logging

from neo4j import AsyncGraphDatabase
from tqdm import tqdm

from app.core.config import NEO4J_AUTH, NEO4J_URI
from app.services import cellar

logger = logging.getLogger(__name__)

_BATCH_SIZE = 100
_TEXT_CONCURRENCY = 10

_STORE_WORKS = """
UNWIND $rows AS row
MERGE (w:Work {uri: row.work})
SET w.celex    = coalesce(w.celex,    row.celex),
    w.title    = coalesce(w.title,    row.title),
    w.date     = coalesce(w.date,     row.date),
    w.abstract = coalesce(w.abstract, row.abstract)
"""

_STORE_CONCEPTS = """
UNWIND $rows AS row
MERGE (c:Concept {uri: row.concept})
SET c.label = coalesce(c.label, row.label)
WITH c, row
MATCH (w:Work {uri: row.work})
MERGE (w)-[:ABOUT]->(c)
"""

_STORE_CITATIONS = """
UNWIND $rows AS row
MERGE (cited:Work {uri: row.cited_work})
SET cited.celex = coalesce(cited.celex, row.cited_celex)
WITH cited, row
MATCH (w:Work {uri: row.work})
MERGE (w)-[:CITES]->(cited)
"""

_STORE_AGENTS = """
UNWIND $rows AS row
MERGE (a:Agent {uri: row.agent})
SET a.label = coalesce(a.label, row.label)
WITH a, row
MATCH (w:Work {uri: row.work})
MERGE (w)-[:AUTHORED_BY]->(a)
"""

_STORE_TEXT = """
UNWIND $rows AS row
MATCH (w:Work {uri: row.uri})
SET w.text = coalesce(w.text, row.text)
"""


def _batches(rows: list) -> list:
    return [rows[i : i + _BATCH_SIZE] for i in range(0, len(rows), _BATCH_SIZE)]


async def _fetch_texts_concurrent(uris: list[str]) -> dict[str, str | None]:
    sem = asyncio.Semaphore(_TEXT_CONCURRENCY)

    async def _one(uri: str) -> tuple[str, str | None]:
        async with sem:
            return uri, await cellar.fetch_document_text(uri)

    results: dict[str, str | None] = {}
    tasks = [asyncio.create_task(_one(uri)) for uri in uris]
    pbar = tqdm(total=len(tasks), desc="Fetching text", unit="doc")
    for coro in asyncio.as_completed(tasks):
        uri, text = await coro
        results[uri] = text
        pbar.update(1)
    pbar.close()
    return results


async def clear_database() -> None:
    """Delete every node and relationship in Neo4j."""
    async with AsyncGraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH) as driver:
        async with driver.session() as session:
            await session.run("MATCH (n) DETACH DELETE n")
    logger.info("Database cleared")


async def pipeline(sparql: str) -> int:
    """Fetch all works matched by ``sparql`` and store the full CDM graph in Neo4j.

    ``sparql`` must project ``?work`` (CELLAR URI). Returns the number of domain works ingested.
    """
    # Step 1: fetch domain works
    documents = await cellar.run_query(sparql)
    logger.info("Retrieved %d works from CELLAR", len(documents))
    documents = [doc for doc in documents if doc.get("work")]
    if not documents:
        logger.info("No works to store")
        return 0

    work_uris = [doc["work"] for doc in documents]

    # Step 2: fetch related nodes concurrently, each with its own tqdm bar
    logger.info("Fetching concepts, citations, agents for %d works...", len(work_uris))
    concepts, citations, agents = await asyncio.gather(
        cellar.fetch_concepts(work_uris, position=0),
        cellar.fetch_citations(work_uris, position=1),
        cellar.fetch_agents(work_uris, position=2),
    )

    # Step 3: store everything — cited Works are created as stubs here
    async with AsyncGraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH) as driver:
        async with driver.session() as session:
            for batch in tqdm(_batches(documents), desc="Storing works", unit="batch"):
                await session.run(_STORE_WORKS, rows=batch)
            for batch in tqdm(_batches(concepts), desc="Storing concepts", unit="batch"):
                await session.run(_STORE_CONCEPTS, rows=batch)
            for batch in tqdm(_batches(citations), desc="Storing citations", unit="batch"):
                await session.run(_STORE_CITATIONS, rows=batch)
            for batch in tqdm(_batches(agents), desc="Storing agents", unit="batch"):
                await session.run(_STORE_AGENTS, rows=batch)

    logger.info(
        "Stored %d works, %d concept links, %d citations, %d agent links",
        len(documents), len(concepts), len(citations), len(agents),
    )

    # Step 4: collect all Work URIs now in DB (original + cited stubs without text yet)
    async with AsyncGraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH) as driver:
        async with driver.session() as session:
            result = await session.run(
                "MATCH (w:Work) WHERE w.text IS NULL RETURN w.uri AS uri"
            )
            uris_without_text = [row["uri"] for row in await result.data()]

    logger.info("Fetching text for %d works...", len(uris_without_text))
    text_map = await _fetch_texts_concurrent(uris_without_text)

    # Step 5: write text back, skipping nulls (no content available)
    rows = [{"uri": uri, "text": text} for uri, text in text_map.items() if text is not None]
    logger.info("%d/%d works returned text", len(rows), len(uris_without_text))
    async with AsyncGraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH) as driver:
        async with driver.session() as session:
            for batch in tqdm(_batches(rows), desc="Storing text", unit="batch"):
                await session.run(_STORE_TEXT, rows=batch)

    return len(documents)
