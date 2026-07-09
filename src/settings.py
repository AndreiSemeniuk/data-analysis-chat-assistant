"""Central configuration. Everything tunable lives in .env / environment."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    # LLM
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    model_name: str = os.getenv("MODEL_NAME", "gemini-2.5-flash")
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_model: str = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")

    # Database
    db_backend: str = os.getenv("DB_BACKEND", "bigquery")  # "bigquery" | "demo"
    gcp_project_id: str = os.getenv("GCP_PROJECT_ID", "")
    bq_dataset: str = os.getenv("BQ_DATASET", "bigquery-public-data.thelook_ecommerce")
    bq_max_bytes_billed: int = int(os.getenv("BQ_MAX_BYTES_BILLED", str(1 << 30)))

    # Agent behaviour
    default_user: str = os.getenv("DEFAULT_USER", "manager_a")
    max_sql_attempts: int = int(os.getenv("MAX_SQL_ATTEMPTS", "3"))
    max_llm_calls_per_turn: int = int(os.getenv("MAX_LLM_CALLS_PER_TURN", "8"))
    max_result_rows: int = int(os.getenv("MAX_RESULT_ROWS", "200"))

    # Paths
    prompts_dir: Path = PROJECT_ROOT / "prompts"
    persona_file: Path = PROJECT_ROOT / "config" / "persona.yaml"
    golden_bucket_dir: Path = PROJECT_ROOT / "golden_bucket" / "trios"
    candidates_dir: Path = PROJECT_ROOT / "golden_bucket" / "candidates"
    data_dir: Path = PROJECT_ROOT / "data"
    app_db_path: Path = PROJECT_ROOT / "data" / "app.db"
    demo_db_path: Path = PROJECT_ROOT / "data" / "demo_thelook.duckdb"
    trace_file: Path = PROJECT_ROOT / os.getenv("TRACE_FILE", "logs/traces.jsonl")


settings = Settings()
