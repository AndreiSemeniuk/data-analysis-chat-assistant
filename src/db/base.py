"""Backend-agnostic database interface.

The agent always *thinks* in BigQuery SQL (that is what the golden bucket and
prompts use). Each backend receives BigQuery SQL and is responsible for making
it run - the DuckDB demo backend transpiles it via sqlglot.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class QueryResult:
    df: pd.DataFrame
    bytes_processed: int | None = None  # cost signal, when the backend reports it


@dataclass
class TableSchema:
    name: str
    columns: list[dict] = field(default_factory=list)  # {name, type, description}


class DatabaseError(Exception):
    """Normalized execution error; message is fed back to the self-heal loop."""


class DatabaseClient(ABC):
    @abstractmethod
    def execute(self, bigquery_sql: str) -> QueryResult: ...

    @abstractmethod
    def get_schemas(self) -> list[TableSchema]: ...

    def schema_prompt(self) -> str:
        """Schema rendered for the LLM context (cached by callers)."""
        lines = []
        for table in self.get_schemas():
            lines.append(f"Table `{table.name}`:")
            for col in table.columns:
                desc = f" -- {col['description']}" if col.get("description") else ""
                lines.append(f"  - {col['name']} ({col['type']}){desc}")
        return "\n".join(lines)
