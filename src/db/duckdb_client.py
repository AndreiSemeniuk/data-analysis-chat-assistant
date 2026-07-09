"""Local demo backend: DuckDB seeded with synthetic thelook-like data.

Purpose: lets anyone run the full agent end-to-end (including PII masking -
the synthetic users table contains fake emails and phone numbers) without a
GCP account. The agent still generates BigQuery SQL; we transpile it to the
DuckDB dialect with sqlglot, which also demonstrates how a new data source
plugs in behind the same interface.
"""

import random
from datetime import datetime, timedelta

import duckdb
import pandas as pd
import sqlglot
from sqlglot import exp

from src.db.base import DatabaseClient, DatabaseError, QueryResult, TableSchema
from src.settings import settings

FIRST_NAMES = ["Alex", "Sam", "Dana", "Lee", "Noa", "Omer", "Maya", "Yuri", "Tal", "Rina",
               "Ben", "Gil", "Adi", "Roni", "Shay", "Lior", "Eden", "Amit", "Noam", "Ella"]
LAST_NAMES = ["Levi", "Cohen", "Mizrahi", "Peretz", "Biton", "Avraham", "Friedman",
              "Katz", "Shapiro", "Goldberg", "Rosen", "Weiss", "Klein", "Berger"]
CATEGORIES = ["Jeans", "Sweaters", "Tops & Tees", "Accessories", "Outerwear & Coats",
              "Dresses", "Shorts", "Swim", "Active", "Sleep & Lounge"]
BRANDS = ["Levi's", "Calvin Klein", "Carhartt", "Columbia", "Wrangler", "Dockers",
          "Volcom", "Quiksilver", "Hurley", "Diesel"]
STATUSES = ["Complete", "Shipped", "Processing", "Cancelled", "Returned"]
COUNTRIES = ["United States", "China", "Brasil", "United Kingdom", "Germany", "France", "Israel"]


def _seed(conn: duckdb.DuckDBPyConnection, n_users: int = 400, n_products: int = 300,
          n_orders: int = 2500) -> None:
    rng = random.Random(42)
    now = datetime(2026, 7, 1)

    users = pd.DataFrame({
        "id": range(1, n_users + 1),
        "first_name": [rng.choice(FIRST_NAMES) for _ in range(n_users)],
        "last_name": [rng.choice(LAST_NAMES) for _ in range(n_users)],
        "email": [f"user{i}@example-mail.com" for i in range(1, n_users + 1)],
        "phone_number": [f"+1-555-{rng.randint(100, 999)}-{rng.randint(1000, 9999)}"
                         for _ in range(n_users)],
        "age": [rng.randint(18, 70) for _ in range(n_users)],
        "gender": [rng.choice(["M", "F"]) for _ in range(n_users)],
        "country": [rng.choice(COUNTRIES) for _ in range(n_users)],
        "city": [f"City {rng.randint(1, 50)}" for _ in range(n_users)],
        "traffic_source": [rng.choice(["Search", "Organic", "Email", "Display", "Facebook"])
                           for _ in range(n_users)],
        "created_at": [now - timedelta(days=rng.randint(30, 1200)) for _ in range(n_users)],
    })

    products = pd.DataFrame({
        "id": range(1, n_products + 1),
        "name": [f"{rng.choice(BRANDS)} {rng.choice(CATEGORIES)} #{i}"
                 for i in range(1, n_products + 1)],
        "brand": [rng.choice(BRANDS) for _ in range(n_products)],
        "category": [rng.choice(CATEGORIES) for _ in range(n_products)],
        "department": [rng.choice(["Men", "Women"]) for _ in range(n_products)],
        "cost": [round(rng.uniform(5, 120), 2) for _ in range(n_products)],
        "retail_price": [round(rng.uniform(10, 300), 2) for _ in range(n_products)],
    })

    orders_rows, item_rows = [], []
    item_id = 1
    for order_id in range(1, n_orders + 1):
        user_id = rng.randint(1, n_users)
        created = now - timedelta(days=rng.randint(0, 900), hours=rng.randint(0, 23))
        status = rng.choices(STATUSES, weights=[55, 20, 10, 8, 7])[0]
        n_items = rng.randint(1, 4)
        orders_rows.append({
            "order_id": order_id, "user_id": user_id, "status": status,
            "created_at": created,
            "shipped_at": created + timedelta(days=2) if status in ("Complete", "Shipped") else None,
            "delivered_at": created + timedelta(days=5) if status == "Complete" else None,
            "num_of_item": n_items,
        })
        for _ in range(n_items):
            product_id = rng.randint(1, n_products)
            price = float(products.loc[product_id - 1, "retail_price"])
            item_rows.append({
                "id": item_id, "order_id": order_id, "user_id": user_id,
                "product_id": product_id, "status": status, "created_at": created,
                "sale_price": round(price * rng.uniform(0.6, 1.0), 2),
            })
            item_id += 1

    conn.register("users_df", users)
    conn.register("products_df", products)
    conn.register("orders_df", pd.DataFrame(orders_rows))
    conn.register("order_items_df", pd.DataFrame(item_rows))
    for table in ["users", "products", "orders", "order_items"]:
        conn.execute(f"CREATE TABLE {table} AS SELECT * FROM {table}_df")


class DuckDBDemoClient(DatabaseClient):
    def __init__(self, db_path=None) -> None:
        path = db_path or settings.demo_db_path
        path.parent.mkdir(parents=True, exist_ok=True)
        fresh = not path.exists()
        self.conn = duckdb.connect(str(path))
        if fresh:
            _seed(self.conn)

    @staticmethod
    def _to_duckdb(bigquery_sql: str) -> str:
        """BigQuery -> DuckDB, with one binding fix: BigQuery allows GROUP BY
        <select alias> even when a real column shares the name; in DuckDB the
        column wins, which can break the query. Rewriting such GROUP BY items
        to ordinal positions keeps the semantics."""
        tree = sqlglot.parse_one(bigquery_sql, read="bigquery")
        # Strip dataset qualifiers (`project.dataset.orders` -> `orders`);
        # the demo DB has the tables at top level.
        for table in tree.find_all(exp.Table):
            table.set("db", None)
            table.set("catalog", None)
        for select in tree.find_all(exp.Select):
            group = select.args.get("group")
            if not group:
                continue
            aliases = {s.alias: i + 1 for i, s in enumerate(select.selects) if s.alias}
            group.set("expressions", [
                exp.Literal.number(aliases[e.name])
                if isinstance(e, exp.Column) and not e.table and e.name in aliases
                else e
                for e in group.expressions
            ])
        return tree.sql(dialect="duckdb")

    def execute(self, bigquery_sql: str) -> QueryResult:
        try:
            duck_sql = self._to_duckdb(bigquery_sql)
            df = self.conn.execute(duck_sql).df()
            return QueryResult(df=df, bytes_processed=None)
        except Exception as e:
            raise DatabaseError(str(e)) from e

    def get_schemas(self) -> list[TableSchema]:
        schemas = []
        for name in ["orders", "order_items", "products", "users"]:
            rows = self.conn.execute(f"DESCRIBE {name}").fetchall()
            schemas.append(TableSchema(
                name=name,
                columns=[{"name": r[0], "type": r[1], "description": ""} for r in rows],
            ))
        return schemas
