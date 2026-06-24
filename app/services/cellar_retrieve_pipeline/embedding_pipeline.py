"""Embedding pipeline: generate vector embeddings for all Neo4j nodes.

Reads every :Work, :Concept, and :Agent node, concatenates their text fields,
embeds them with the configured LangChain embedder, and writes the vector back
as an `embedding` property. Also creates the Neo4j vector indexes needed by
the RAG search in app/services/rag.py.

Change provider in app/core/config.py (EMBEDDING_PROVIDER / EMBEDDING_MODEL / EMBEDDING_DIM).
"""

from __future__ import annotations

import logging

from neo4j import AsyncGraphDatabase
from tqdm import tqdm

from app.core.config import EMBEDDING_DIM, NEO4J_AUTH, NEO4J_URI
from app.core.embeddings import get_embedder

logger = logging.getLogger(__name__)

_EMBED_BATCH_SIZE = 64
_WRITE_BATCH_SIZE = 100

_LABEL_CONFIG = [
    ("Work", "work_embedding_idx"),
    ("Concept", "concept_embedding_idx"),
    ("Agent", "agent_embedding_idx"),
]

_CREATE_VECTOR_INDEX = (
    "CREATE VECTOR INDEX {index_name} IF NOT EXISTS "
    "FOR (n:{label}) ON (n.embedding) "
    "OPTIONS {{indexConfig: {{`vector.dimensions`: {dim}, `vector.similarity_function`: 'cosine'}}}}"
)

_FETCH_NODES = "MATCH (n:{label}) RETURN elementId(n) AS eid, properties(n) AS props"

_STORE_EMBEDDINGS = """
UNWIND $rows AS row
MATCH (n) WHERE elementId(n) = row.eid
SET n.embedding = row.embedding
"""


def _node_text(label: str, props: dict) -> str:
    if label == "Work":
        parts = [
            props.get("title", ""),
            props.get("celex", ""),
            props.get("date", ""),
            props.get("abstract", ""),
            props.get("text", ""),
        ]
    else:
        parts = [props.get("label", "")]
    return " ".join(p for p in parts if p)


async def pipeline() -> None:
    """Embed all nodes and write vectors back to Neo4j. Creates vector indexes if absent."""
    embedder = get_embedder()

    async with AsyncGraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH) as driver:
        async with driver.session() as session:
            for label, index_name in _LABEL_CONFIG:
                await session.run(
                    _CREATE_VECTOR_INDEX.format(
                        index_name=index_name, label=label, dim=EMBEDDING_DIM
                    )
                )
            logger.info("Vector indexes ensured")

            for label, _ in _LABEL_CONFIG:
                result = await session.run(_FETCH_NODES.format(label=label))
                records = await result.data()
                if not records:
                    logger.info("No %s nodes found, skipping", label)
                    continue

                logger.info("Embedding %d %s nodes...", len(records), label)
                texts = [_node_text(label, r["props"]) for r in records]
                eids = [r["eid"] for r in records]

                all_embeddings: list[list[float]] = []
                for i in tqdm(
                    range(0, len(texts), _EMBED_BATCH_SIZE),
                    desc=f"Embedding {label}",
                    unit="batch",
                ):
                    all_embeddings.extend(
                        embedder.embed_documents(texts[i : i + _EMBED_BATCH_SIZE])
                    )

                rows = [{"eid": eid, "embedding": emb} for eid, emb in zip(eids, all_embeddings)]
                for i in tqdm(
                    range(0, len(rows), _WRITE_BATCH_SIZE),
                    desc=f"Writing {label} embeddings",
                    unit="batch",
                ):
                    await session.run(_STORE_EMBEDDINGS, rows=rows[i : i + _WRITE_BATCH_SIZE])

                logger.info("Done: %d %s nodes embedded", len(records), label)
