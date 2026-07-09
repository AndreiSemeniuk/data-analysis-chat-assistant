from src.db.base import DatabaseClient, QueryResult


def make_db_client(backend: str) -> DatabaseClient:
    """Factory keeping the rest of the app backend-agnostic (extensibility:
    a new data source = one new class implementing DatabaseClient)."""
    if backend == "bigquery":
        from src.db.bigquery_client import BigQueryClient
        return BigQueryClient()
    if backend == "demo":
        from src.db.duckdb_client import DuckDBDemoClient
        return DuckDBDemoClient()
    raise ValueError(f"Unknown DB_BACKEND: {backend!r} (expected 'bigquery' or 'demo')")
