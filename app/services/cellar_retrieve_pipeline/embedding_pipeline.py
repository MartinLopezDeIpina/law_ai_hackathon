"""Embedding pipeline: generate vector embeddings for all Neo4j nodes.

Reads every :Work, :Concept, and :Agent node, concatenates their text fields,
embeds them with a local sentence-transformer (GPU if available), and writes
the vector back as an `embedding` property. Also creates the Neo4j vector
indexes needed by the RAG search in app/services/rag.py.
"""

from __future__ import annotations

import logging

from neo4j import AsyncGraphDatabase
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from app.core.config import NEO4J_AUTH, NEO4J_URI

logger = logging.getLogger(__name__)

_MODEL_NAME = "all-MiniLM-L6-v2"
_EMBEDDING_DIM = 384
_EMBED_BATCH_SIZE = 64
_WRITE_BATCH_SIZE = 100

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info("Loading embedding model %s ...", _MODEL_NAME)
        _model = SentenceTransformer(_MODEL_NAME)
        logger.info("Model loaded on device: %s", _model.device)
    return _model


def _work_text(props: dict) -> str:
    parts = [
        props.get("title", ""),
        props.get("celex", ""),
        props.get("date", ""),
        props.get("abstract", ""),
        props.get("text", ""),
    ]
    return " ".join(p for p in parts if p)


def _concept_text(props: dict) -> str:
    return props.get("label", "")


def _agent_text(props: dict) -> str:
    return props.get("label", "")


_LABEL_CONFIG = [
    ("Work", _work_text, "work_embedding_idx"),
    ("Concept", _concept_text, "concept_embedding_idx"),
    ("Agent", _agent_text, "agent_embedding_idx"),
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


async def pipeline() -> None:
    """Embed all nodes and write vectors back to Neo4j. Creates vector indexes if absent."""
    model = get_model()

    async with AsyncGraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH) as driver:
        async with driver.session() as session:
            for label, _, index_name in _LABEL_CONFIG:
                await session.run(
                    _CREATE_VECTOR_INDEX.format(
                        index_name=index_name, label=label, dim=_EMBEDDING_DIM
                    )
                )
            logger.info("Vector indexes ensured")

            for label, text_fn, _ in _LABEL_CONFIG:
                result = await session.run(_FETCH_NODES.format(label=label))
                records = await result.data()
                if not records:
                    logger.info("No %s nodes found, skipping", label)
                    continue

                logger.info("Embedding %d %s nodes...", len(records), label)
                texts = [text_fn(r["props"]) for r in records]
                eids = [r["eid"] for r in records]

                embeddings = model.encode(
                    texts,
                    batch_size=_EMBED_BATCH_SIZE,
                    show_progress_bar=True,
                    convert_to_numpy=True,
                )

                rows = [
                    {"eid": eid, "embedding": emb.tolist()}
                    for eid, emb in zip(eids, embeddings)
                ]
                for i in tqdm(
                    range(0, len(rows), _WRITE_BATCH_SIZE),
                    desc=f"Writing {label} embeddings",
                    unit="batch",
                ):
                    await session.run(_STORE_EMBEDDINGS, rows=rows[i : i + _WRITE_BATCH_SIZE])

                logger.info("Done: %d %s nodes embedded", len(records), label)
