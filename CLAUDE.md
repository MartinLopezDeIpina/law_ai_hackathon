# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the app

```bash
# Activate the virtual environment
source .venv/bin/activate

# Start the FastAPI server
uvicorn main:app --reload

# Run the ingestion pipeline (one-shot batch job)
python -m app.services.cellar_retrieve_pipeline
```

## Architecture

This is a FastAPI application that queries the EU Publications Office **CELLAR** repository — the public SPARQL endpoint for EU legislation — and exposes an AI agent interface over it.

**Layers (strict dependency direction: top → bottom)**

- `app/api/routes.py` — FastAPI route handlers (currently empty stub)
- `app/agents/graph.py` — LangGraph agent graph (currently empty stub)
- `app/agents/tools.py` — LangChain `@tool` wrappers; the docstrings/arg names here are what the model reads to decide how to call them, so they are written for the model audience
- `app/services/cellar.py` — transport layer: SPARQL queries against CELLAR and content-negotiation fetches of full document text; knows nothing about LangChain
- `app/core/config.py` — hardcoded SPARQL endpoint URL, CELLAR content URL template, and Neo4j connection details

**Ingestion pipeline** (`app/services/cellar_retrieve_pipeline/`)

A standalone batch job (not part of the FastAPI app) that ingests a domain of EU documents into Neo4j. The domain is expressed as a SPARQL query projecting a `?celex` column. Pipeline upserts `:Document` nodes keyed by CELEX id, fetching document body text via `cellar.fetch_document_text`. Edit `DOMAIN_QUERY` in `__main__.py` to target a different slice of CELLAR.

**Key external dependencies**

- CELLAR SPARQL endpoint: `http://publications.europa.eu/webapi/rdf/sparql`
- CELLAR content endpoint: `http://publications.europa.eu/resource/celex/{celex}` (content-negotiation, returns HTML)
- Neo4j at `bolt://localhost:7687` (credentials in `app/core/config.py`)
- LangGraph + `langchain-anthropic` for the agent graph
