"""Static SQL validation - runs BEFORE any query touches the database.

The LLM proposes SQL; this module is the deterministic gatekeeper:
  * single statement only
  * read-only: SELECT / WITH ... SELECT - everything else rejected at AST level
  * only whitelisted tables of the analytics dataset
  * a LIMIT is enforced (added or lowered) to bound result size and cost

Because validation happens on the parsed AST (sqlglot), comment tricks,
casing, or stacked statements ("SELECT 1; DROP TABLE x") do not get through.
"""

from dataclasses import dataclass

import sqlglot
from sqlglot import exp

ALLOWED_TABLES = {"orders", "order_items", "products", "users"}


class SQLGuardError(Exception):
    """Raised when a query violates the safety policy (not self-healable)."""


@dataclass
class ValidatedSQL:
    sql: str          # normalized, LIMIT-enforced BigQuery SQL
    tables: list[str]


def validate_sql(raw_sql: str, max_rows: int = 200, dialect: str = "bigquery",
                 dataset: str | None = None) -> ValidatedSQL:
    """`dataset` is the only project.dataset tables may be qualified with;
    bare names are always fine (the execution layer supplies the default
    dataset). Without this check, `other-project.other_dataset.orders` would
    slip past a name-only whitelist."""
    allowed_qualifiers = {""}
    if dataset:
        allowed_qualifiers.add(dataset)              # project.dataset
        allowed_qualifiers.add(dataset.split(".")[-1])  # bare dataset
    statements = [s for s in sqlglot.parse(raw_sql, read=dialect) if s is not None]
    if len(statements) != 1:
        raise SQLGuardError("Exactly one SQL statement is allowed.")
    tree = statements[0]

    # Read-only enforcement: the root must be a SELECT (possibly under WITH/UNION).
    root = tree
    if isinstance(root, exp.With):
        root = root.this
    if not isinstance(root, (exp.Select, exp.Union)):
        raise SQLGuardError(f"Only SELECT queries are allowed, got: {type(tree).__name__}.")

    # No DML/DDL anywhere in the tree (also catches sub-expressions).
    forbidden = (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create,
                 exp.Alter, exp.Merge, exp.TruncateTable, exp.Grant)
    for node in tree.walk():
        if isinstance(node, forbidden):
            raise SQLGuardError(f"Forbidden operation: {type(node).__name__}.")

    # Table whitelist (ignore CTE aliases).
    cte_names = {cte.alias_or_name for cte in tree.find_all(exp.CTE)}
    tables = []
    for table in tree.find_all(exp.Table):
        name = table.name
        if name in cte_names:
            continue
        if name not in ALLOWED_TABLES:
            raise SQLGuardError(
                f"Table '{name}' is not allowed. Allowed tables: {sorted(ALLOWED_TABLES)}."
            )
        qualifier = ".".join(p for p in (table.catalog, table.db) if p)
        if qualifier not in allowed_qualifiers:
            raise SQLGuardError(
                f"Table '{qualifier}.{name}' is outside the allowed dataset."
            )
        tables.append(name)
    if not tables:
        raise SQLGuardError("Query must reference at least one dataset table.")

    # Enforce a row cap on the outermost query.
    limit_expr = tree.args.get("limit")
    if limit_expr is None or int(limit_expr.expression.this) > max_rows:
        tree = tree.limit(max_rows)

    return ValidatedSQL(sql=tree.sql(dialect=dialect), tables=sorted(set(tables)))
