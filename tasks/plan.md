# Feature: RAG Embedding Pipeline + Route

## What it does
1. **Refactor** `cellar_retrieve_pipeline` into two named sub-pipelines (`retrieve_pipeline.py`, `embedding_pipeline.py`), each exposing a `pipeline()` function, dispatched from `__main__.py`.
2. **`embedding_pipeline.py`** — reads every node from Neo4j (`:Work`, `:Concept`, `:Agent`), concatenates text fields, embeds with `all-MiniLM-L6-v2` on CUDA (384 dims), writes vector back on the node, ensures a Neo4j vector index per label.
3. **`app/services/rag.py`** — `search(query, label, top_k)`: embeds query, queries Neo4j vector index, returns ranked results.
4. **`app/api/routes.py`** — POST `/rag` endpoint wrapping `rag.search`.

## Files

| Path | Status |
|------|--------|
| `app/services/cellar_retrieve_pipeline/retrieve_pipeline.py` | NEW |
| `app/services/cellar_retrieve_pipeline/embedding_pipeline.py` | NEW |
| `app/services/cellar_retrieve_pipeline/__main__.py` | MODIFY |
| `app/services/cellar_retrieve_pipeline/pipeline.py` | DELETE |
| `app/services/cellar_retrieve_pipeline/embedding.py` | DELETE |
| `app/services/rag.py` | NEW |
| `app/api/routes.py` | MODIFY |
| `main.py` | MODIFY |
| `requirements.txt` | MODIFY |
| `tests/test_embedding_pipeline.py` | NEW |
| `tests/test_rag.py` | NEW |
| `scripts/validate_plan.sh` | NEW |

## Checklist

- [x] 1. Write `scripts/validate_plan.sh`
- [x] 2. Create `retrieve_pipeline.py` (extracted from `pipeline.py`)
- [x] 3. Create `embedding_pipeline.py`
- [x] 4. Update `__main__.py`
- [x] 5. Delete `pipeline.py` and `embedding.py`
- [x] 6. Write `app/services/rag.py`
- [x] 7. Update `app/api/routes.py` and `main.py`
- [x] 8. Add `sentence-transformers` to `requirements.txt`
- [x] 9. Write tests
- [x] 10. Run tests

## Tests
- `test_build_embedding_text_work` — all fields concatenated, Nones skipped
- `test_build_embedding_text_concept` — label only
- `test_search_returns_ranked_results` — mock Neo4j; returns `(props, score)` list
