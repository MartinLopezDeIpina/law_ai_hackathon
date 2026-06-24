"""Retrieve pipeline: pull a domain of EU documents from CELLAR into Neo4j.

Mirrors the CDM graph structure:
  (:Work)-[:ABOUT]->(:Concept)
  (:Work)-[:CITES]->(:Work)
  (:Work)-[:AUTHORED_BY]->(:Agent)

A "domain" is defined by a SPARQL query that must project ?work (CELLAR URI)
and optionally ?celex, ?title, ?date. Works are keyed by their CELLAR URI.
"""

from __future__ import annotations

import logging

from neo4j import AsyncGraphDatabase
from tqdm import tqdm

from app.core.config import NEO4J_AUTH, NEO4J_URI
from app.services import cellar

logger = logging.getLogger(__name__)

_BATCH_SIZE = 100

_STORE_WORKS = """
UNWIND $rows AS row
MERGE (w:Work {uri: row.work})
SET w.celex    = coalesce(w.celex,    row.celex),
    w.title    = coalesce(w.title,    row.title),
    w.date     = coalesce(w.date,     row.date),
    w.text     = coalesce(w.text,     row.text),
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


def _batches(rows: list) -> list:
    return [rows[i : i + _BATCH_SIZE] for i in range(0, len(rows), _BATCH_SIZE)]


async def clear_database() -> None:
    """Delete every node and relationship in Neo4j."""
    async with AsyncGraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH) as driver:
        async with driver.session() as session:
            await session.run("MATCH (n) DETACH DELETE n")
    logger.info("Database cleared")


async def pipeline(sparql: str) -> int:
    """Fetch all works matched by ``sparql`` and store the full CDM graph in Neo4j.

    ``sparql`` must project ``?work`` (CELLAR URI). Returns the number of works ingested.
    """
    documents = await cellar.run_query(sparql)
    logger.info("Retrieved %d works from CELLAR", len(documents))

    documents = [doc for doc in documents if doc.get("work")]
    if not documents:
        logger.info("No works to store")
        return 0

    for doc in tqdm(documents, desc="Fetching text", unit="doc"):
        doc["text"] = await cellar.fetch_document_text(doc["work"])

    work_uris = [doc["work"] for doc in documents]

    logger.info("Fetching EuroVoc concepts (%d works)...", len(work_uris))
    concepts = await cellar.fetch_concepts(work_uris)
    logger.info("Fetching citations...")
    citations = await cellar.fetch_citations(work_uris)
    logger.info("Fetching agents...")
    agents = await cellar.fetch_agents(work_uris)

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
    return len(documents)
