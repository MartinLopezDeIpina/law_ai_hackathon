"""Client for the EU Publications Office CELLAR repository.

This is the transport layer: it knows how to talk to CELLAR and how to shape
the response into clean Python dicts. It has no knowledge of LangChain or the
agent — that keeps it independently testable and reusable from plain routes.

It queries CELLAR's public SPARQL endpoint and returns flattened records with
a stable shape: ``celex``, ``title``, ``date``.
"""

from __future__ import annotations

import httpx
import logging

from tqdm import tqdm

from app.core.config import SPARQL_ENDPOINT

logger = logging.getLogger(__name__)

# Ask the endpoint for JSON-shaped results rather than RDF/XML.
_HEADERS = {"Accept": "application/sparql-results+json"}
_CONTENT_TIMEOUT = 30.0
_SPARQL_TIMEOUT = 60.0

# English expression, so titles come back in English.
_ENGLISH = "<http://publications.europa.eu/resource/authority/language/ENG>"


def _escape(value: str) -> str:
    """Escape a user string for safe inlining inside a SPARQL literal."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _flatten(bindings: list[dict]) -> list[dict]:
    """Turn SPARQL JSON bindings into plain ``{field: value}`` dicts."""
    return [
        {key: cell["value"] for key, cell in row.items()}
        for row in bindings
    ]


async def fetch_document_text(work_uri: str) -> str | None:
    """Download the rendered body of an act using its CELLAR work URI.

    Uses the work URI returned directly from SPARQL rather than reconstructing
    a URL from the CELEX id, which fails for sub-document identifiers.
    Returns the body as text, or ``None`` if no content is available.
    """
    async with httpx.AsyncClient(
        timeout=_CONTENT_TIMEOUT, follow_redirects=True
    ) as client:
        try:
            response = await client.get(
                work_uri,
                headers={
                    "Accept": "application/xhtml+xml",
                    "Accept-Language": "eng",
                },
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.debug("%s has no content in CELLAR", work_uri)
            else:
                logger.warning("%s fetch failed (%s)", work_uri, e.response.status_code)
            return None
        except httpx.HTTPError as e:
            logger.warning("%s fetch error: %s", work_uri, e)
            return None
    return response.text


_SPARQL_BATCH = 20


def _values_clause(uris: list[str]) -> str:
    return " ".join(f"<{uri}>" for uri in uris)


async def _batch_query(make_query, work_uris: list[str], desc: str = "", position: int = 0) -> list[dict]:
    """Run make_query for each 50-URI chunk of work_uris and concatenate results."""
    results = []
    batches = [work_uris[i : i + _SPARQL_BATCH] for i in range(0, len(work_uris), _SPARQL_BATCH)]
    for batch in tqdm(batches, desc=desc, unit="batch", disable=not desc, position=position, leave=True):
        try:
            results.extend(await _run_query(make_query(batch)))
        except httpx.HTTPError as e:
            logger.warning("SPARQL batch failed: %s", e)
    return results


async def fetch_concepts(work_uris: list[str], position: int = 0) -> list[dict]:
    """Return all English EuroVoc concept labels for the given work URIs.

    Each row: ``work``, ``concept`` (URI), ``label``.

    Split into two queries to avoid the expensive 3-way join on every batch:
      1. work → concept URI  (fast single-hop predicate lookup)
      2. concept URI → label (simple VALUES lookup on the unique concepts found)
    """
    def make_work_concept_query(batch: list[str]) -> str:
        return f"""
            PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>
            SELECT DISTINCT ?work ?concept
            WHERE {{
                VALUES ?work {{ {_values_clause(batch)} }}
                ?work cdm:work_is_about_concept_eurovoc ?concept .
            }}
        """

    def make_label_query(batch: list[str]) -> str:
        return f"""
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
            SELECT DISTINCT ?concept ?label
            WHERE {{
                VALUES ?concept {{ {_values_clause(batch)} }}
                ?concept skos:prefLabel ?label .
                FILTER(LANG(?label) = "en")
            }}
        """

    work_concepts = await _batch_query(
        make_work_concept_query, work_uris, desc="Fetching concepts", position=position
    )

    concept_uris = list({row["concept"] for row in work_concepts})
    concept_labels = await _batch_query(
        make_label_query, concept_uris, desc="Fetching concept labels", position=position
    )
    label_by_uri = {row["concept"]: row["label"] for row in concept_labels}

    return [
        {"work": row["work"], "concept": row["concept"], "label": label_by_uri.get(row["concept"])}
        for row in work_concepts
    ]


async def fetch_citations(work_uris: list[str], position: int = 1) -> list[dict]:
    """Return works cited by the given work URIs.

    Each row: ``work``, ``cited_work`` (URI), ``cited_celex`` (optional).
    """
    def make_query(batch: list[str]) -> str:
        return f"""
            PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>
            SELECT DISTINCT ?work ?cited_work ?cited_celex
            WHERE {{
                VALUES ?work {{ {_values_clause(batch)} }}
                ?work cdm:work_cites ?cited_work .
                OPTIONAL {{ ?cited_work cdm:resource_legal_id_celex ?cited_celex }}
            }}
        """
    return await _batch_query(make_query, work_uris, desc="Fetching citations", position=position)


async def fetch_agents(work_uris: list[str], position: int = 2) -> list[dict]:
    """Return agents (institutions/authors) responsible for the given work URIs.

    Each row: ``work``, ``agent`` (URI), ``label`` (optional English label).
    """
    def make_query(batch: list[str]) -> str:
        return f"""
            PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
            SELECT DISTINCT ?work ?agent ?label
            WHERE {{
                VALUES ?work {{ {_values_clause(batch)} }}
                ?work cdm:work_created_by_agent ?agent .
                OPTIONAL {{ ?agent skos:prefLabel ?label FILTER(LANG(?label) = "en") }}
            }}
        """
    return await _batch_query(make_query, work_uris, desc="Fetching agents", position=position)


async def run_query(query: str) -> list[dict]:
    """Run an arbitrary SPARQL query against CELLAR, return flattened rows.

    Use this when the query is supplied by the caller (e.g. a domain-specific
    query driving the ingestion pipeline) rather than built by the helpers
    below.
    """
    return await _run_query(query)


async def _run_query(query: str) -> list[dict]:
    """POST a SPARQL query to CELLAR and return the flattened result rows."""
    async with httpx.AsyncClient(timeout=_SPARQL_TIMEOUT) as client:
        response = await client.post(
            SPARQL_ENDPOINT,
            data={"query": query, "format": "application/sparql-results+json"},
            headers=_HEADERS,
        )
        response.raise_for_status()
        bindings = response.json()["results"]["bindings"]
    return _flatten(bindings)


async def search_legislation(query: str, limit: int = 5) -> list[dict]:
    """Search EU legislation in CELLAR by free-text topic.

    Matches the topic against the English title of each act. Returns a list of
    records, each with ``celex``, ``title``, and ``date``.
    """
    sparql = f"""
        PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>
        SELECT DISTINCT ?work ?celex ?title ?date
        WHERE {{
            ?work cdm:resource_legal_id_celex ?celex .
            ?expr cdm:expression_belongs_to_work ?work ;
                  cdm:expression_uses_language {_ENGLISH} ;
                  cdm:expression_title ?title .
            OPTIONAL {{ ?work cdm:work_date_document ?date . }}
            FILTER(CONTAINS(LCASE(STR(?title)), LCASE("{_escape(query)}")))
        }}
        LIMIT {int(limit)}
    """
    return await _run_query(sparql)


async def get_document_by_celex(celex: str) -> dict | None:
    """Fetch a single legislation record by its CELEX identifier.

    Returns the record dict, or ``None`` if no document matches.
    """
    sparql = f"""
        PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>
        SELECT DISTINCT ?work ?celex ?title ?date
        WHERE {{
            ?work cdm:resource_legal_id_celex ?celex .
            ?expr cdm:expression_belongs_to_work ?work ;
                  cdm:expression_uses_language {_ENGLISH} ;
                  cdm:expression_title ?title .
            OPTIONAL {{ ?work cdm:work_date_document ?date . }}
            FILTER(STR(?celex) = "{_escape(celex)}")
        }}
        LIMIT 1
    """
    rows = await _run_query(sparql)
    return rows[0] if rows else None
