SPARQL_ENDPOINT = "http://publications.europa.eu/webapi/rdf/sparql"

# Content-negotiation endpoint for fetching the rendered body of an act by
# CELEX id (SPARQL only returns metadata, not the document text).
CELLAR_CONTENT_URL = "http://publications.europa.eu/resource/celex/{celex}"

NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "neo4jneo4j")

# Embedding provider: "local" | "google" | "nvidia"
EMBEDDING_PROVIDER = "local"

# Model name — must match the chosen provider:
#   local:  "all-MiniLM-L6-v2"
#   google: "models/text-embedding-004"
#   nvidia: "nvidia/nv-embedqa-e5-v5"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Vector dimension — must match EMBEDDING_MODEL.
# Used when creating Neo4j vector indexes.
#   384  → all-MiniLM-L6-v2 (local)
#   768  → text-embedding-004 (google)
#   1024 → nv-embedqa-e5-v5 (nvidia)
EMBEDDING_DIM = 384

# LLM provider: "ollama" | "anthropic"
LLM_PROVIDER = "ollama"
# Model name — must match provider:
#   ollama:    "qwen3.5:9b"
#   anthropic: "claude-sonnet-4-6"
LLM_MODEL = "qwen3.5:9b"
