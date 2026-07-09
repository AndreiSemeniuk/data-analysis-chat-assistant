"""Live BigQuery backend against bigquery-public-data.thelook_ecommerce.

Cost / safety controls:
  * maximum_bytes_billed hard cap on every query (BQ kills the job server-side)
  * dry-run first: syntax errors and the scan estimate are caught for free,
    which makes the self-heal loop cheap (bad SQL never bills anything)
"""

import logging

from google.api_core import retry as gcp_retry
from google.cloud import bigquery

from src.db.base import DatabaseClient, DatabaseError, QueryResult, TableSchema
from src.settings import settings

log = logging.getLogger(__name__)

TABLES = ["orders", "order_items", "products", "users"]


class BigQueryClient(DatabaseClient):
    def __init__(self) -> None:
        self.client = bigquery.Client(project=settings.gcp_project_id or None)
        self.dataset_id = settings.bq_dataset
        self._schemas: list[TableSchema] | None = None

    def _dataset_ref(self) -> bigquery.DatasetReference:
        """Bare table names in the SQL resolve against this default dataset -
        BigQuery's native mechanism, no string rewriting."""
        return bigquery.DatasetReference.from_string(self.dataset_id)

    def execute(self, bigquery_sql: str) -> QueryResult:
        sql = bigquery_sql
        try:
            # Dry run: free syntax/semantic check + scan estimate.
            dry = self.client.query(
                sql, job_config=bigquery.QueryJobConfig(
                    dry_run=True, default_dataset=self._dataset_ref()
                )
            )
            if dry.total_bytes_processed and dry.total_bytes_processed > settings.bq_max_bytes_billed:
                raise DatabaseError(
                    f"Query would scan {dry.total_bytes_processed:,} bytes, over the "
                    f"{settings.bq_max_bytes_billed:,} byte budget. Narrow the date range "
                    f"or select fewer columns."
                )
            job_config = bigquery.QueryJobConfig(
                maximum_bytes_billed=settings.bq_max_bytes_billed,
                default_dataset=self._dataset_ref(),
            )
            job = self.client.query(sql, job_config=job_config, retry=gcp_retry.Retry(deadline=60))
            df = job.result().to_dataframe()
            return QueryResult(df=df, bytes_processed=job.total_bytes_processed)
        except DatabaseError:
            raise
        except Exception as e:  # normalize BQ errors for the heal loop
            raise DatabaseError(str(e)) from e

    def get_schemas(self) -> list[TableSchema]:
        if self._schemas is None:
            schemas = []
            for name in TABLES:
                table = self.client.get_table(f"{self.dataset_id}.{name}")
                schemas.append(TableSchema(
                    name=name,
                    columns=[{"name": f.name, "type": f.field_type,
                              "description": f.description or ""} for f in table.schema],
                ))
            self._schemas = schemas
        return self._schemas
