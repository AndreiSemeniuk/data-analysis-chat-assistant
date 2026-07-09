import pytest

from src.safety.sql_guard import SQLGuardError, validate_sql


def test_accepts_plain_select():
    v = validate_sql("SELECT status, COUNT(*) AS c FROM orders GROUP BY status")
    assert "LIMIT" in v.sql.upper()
    assert v.tables == ["orders"]


def test_accepts_cte_and_joins():
    v = validate_sql("""
        WITH rev AS (
          SELECT user_id, SUM(sale_price) AS s FROM order_items GROUP BY user_id
        )
        SELECT u.first_name, r.s FROM rev r JOIN users u ON u.id = r.user_id
        ORDER BY r.s DESC LIMIT 10
    """)
    assert set(v.tables) == {"order_items", "users"}


@pytest.mark.parametrize("sql", [
    "DELETE FROM orders WHERE 1=1",
    "DROP TABLE users",
    "UPDATE orders SET status = 'x'",
    "INSERT INTO orders VALUES (1)",
    "CREATE TABLE t AS SELECT 1",
])
def test_rejects_writes(sql):
    with pytest.raises(SQLGuardError):
        validate_sql(sql)


def test_rejects_stacked_statements():
    with pytest.raises(SQLGuardError):
        validate_sql("SELECT 1 FROM orders; DROP TABLE orders")


def test_rejects_unknown_tables():
    with pytest.raises(SQLGuardError):
        validate_sql("SELECT * FROM secret_admin_table")


def test_lowers_excessive_limit():
    v = validate_sql("SELECT * FROM products LIMIT 999999", max_rows=200)
    assert "LIMIT 200" in v.sql.upper()


DATASET = "bigquery-public-data.thelook_ecommerce"


def test_accepts_own_dataset_qualification():
    v = validate_sql(f"SELECT status FROM `{DATASET}.orders`", dataset=DATASET)
    assert v.tables == ["orders"]


def test_rejects_foreign_dataset_with_whitelisted_name():
    # Same table NAME, different project/dataset - must not pass the whitelist.
    with pytest.raises(SQLGuardError, match="outside the allowed dataset"):
        validate_sql("SELECT * FROM `evil-project.other_dataset.orders`", dataset=DATASET)
