from pathlib import Path

from src.knowledge.golden_bucket import GoldenBucket
from src.settings import PROJECT_ROOT
from src.stores.app_store import AppStore


def make_store(tmp_path: Path) -> AppStore:
    return AppStore(tmp_path / "app.db")


def test_reports_are_owner_scoped(tmp_path):
    store = make_store(tmp_path)
    a = store.save_report("manager_a", "Client X revenue", "q", "SELECT 1", "report about Client X")
    store.save_report("manager_b", "Client X churn", "q", "SELECT 1", "report about Client X")

    # manager_a only sees / deletes their own reports
    assert [r.id for r in store.find_reports("manager_a", text_query="Client X")] == [a]
    deleted = store.delete_reports("manager_a", [1, 2])  # tries both ids
    assert deleted == 1
    assert store.list_reports("manager_b")  # b's report survived


def test_find_reports_by_date(tmp_path):
    store = make_store(tmp_path)
    store.save_report("m", "today's report", "q", "SELECT 1", "body")
    today = store.list_reports("m")[0].created_at[:10]
    assert store.find_reports("m", created_on=today)
    assert not store.find_reports("m", created_on="1999-01-01")


def test_preferences_upsert(tmp_path):
    store = make_store(tmp_path)
    store.set_preference("m", "format", "tables")
    store.set_preference("m", "format", "bullet points")
    assert store.get_preferences("m") == {"format": "bullet points"}


def test_golden_bucket_retrieval_is_relevant():
    bucket = GoldenBucket(PROJECT_ROOT / "golden_bucket" / "trios")
    top = bucket.retrieve("who are the best customers by spend", k=2)
    assert top and top[0].source_file == "001_top_customers.yaml"
    monthly = bucket.retrieve("monthly revenue trend", k=2)
    assert monthly[0].source_file == "002_monthly_revenue.yaml"


def test_golden_bucket_candidate_saved(tmp_path):
    bucket = GoldenBucket(PROJECT_ROOT / "golden_bucket" / "trios", tmp_path / "candidates")
    path = bucket.save_candidate("q", "SELECT 1", "report", "manager_a")
    assert path.exists() and "pending_review" in path.read_text()
