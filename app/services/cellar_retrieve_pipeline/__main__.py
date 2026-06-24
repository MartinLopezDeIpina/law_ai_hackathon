"""Run the ingestion and/or embedding pipeline as a standalone job.

    python -m app.services.cellar_retrieve_pipeline              # run both
    python -m app.services.cellar_retrieve_pipeline --step retrieve
    python -m app.services.cellar_retrieve_pipeline --step embed
    python -m app.services.cellar_retrieve_pipeline --clear      # wipe DB then run both

Edit ``DOMAIN_QUERY`` below to ingest a different domain.
"""

from __future__ import annotations

import asyncio
import logging
import sys

from tqdm import tqdm

from app.services.cellar_retrieve_pipeline.retrieve_pipeline import clear_database, pipeline as retrieve_pipeline
from app.services.cellar_retrieve_pipeline.embedding_pipeline import pipeline as embedding_pipeline


class _TqdmHandler(logging.StreamHandler):
    def emit(self, record: logging.LogRecord) -> None:
        tqdm.write(self.format(record), file=sys.stderr)


# Fetch all English works tagged with the EuroVoc "artificial intelligence" concept.
DOMAIN_QUERY = """
    PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    SELECT DISTINCT ?work ?celex ?title ?date ?abstract
    WHERE {
        ?concept skos:prefLabel "artificial intelligence"@en .
        ?work cdm:work_is_about_concept_eurovoc ?concept .
        ?expr cdm:expression_belongs_to_work ?work ;
              cdm:expression_uses_language <http://publications.europa.eu/resource/authority/language/ENG> ;
              cdm:expression_title ?title .
        OPTIONAL { ?work cdm:resource_legal_id_celex ?celex . }
        OPTIONAL { ?work cdm:work_date_document ?date . }
        OPTIONAL { ?expr cdm:expression_abstract ?abstract . }
    }
"""


async def main(step: str, clear: bool) -> None:
    if clear:
        await clear_database()

    if step in ("retrieve", "all"):
        count = await retrieve_pipeline(DOMAIN_QUERY)
        logging.info("Retrieve done: %d documents ingested", count)

    if step in ("embed", "all"):
        await embedding_pipeline()
        logging.info("Embedding done")


"""
python3 -m app.services.cellar_retrieve_pipeline --step all --clear
"""
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step",
        choices=["retrieve", "embed", "all"],
        default="all",
        help="Which pipeline stage to run (default: all)",
    )
    parser.add_argument("--clear", action="store_true", help="Wipe the database before running")
    args = parser.parse_args()

    handler = _TqdmHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    logging.getLogger("httpx").setLevel(logging.WARNING)

    asyncio.run(main(step=args.step, clear=args.clear))
