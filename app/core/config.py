SPARQL_ENDPOINT = "http://publications.europa.eu/webapi/rdf/sparql"

# Content-negotiation endpoint for fetching the rendered body of an act by
# CELEX id (SPARQL only returns metadata, not the document text).
CELLAR_CONTENT_URL = "http://publications.europa.eu/resource/celex/{celex}"

NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "neo4jneo4j")
