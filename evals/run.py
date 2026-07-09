"""Pre-deployment evaluation harness.

Runs every case in dataset.yaml through the real agent (real LLM, chosen DB
backend) and checks deterministic assertions: routing outcome, tables used,
required content, and - critically - regex proof that no PII pattern appears
in any final answer. Exit code is non-zero on failure, so this gates a CI
deploy. LLM-as-judge scoring of report faithfulness is the production add-on
described in ARCHITECTURE.md.

Usage:
    python -m evals.run --demo        # against the local demo DB
    python -m evals.run               # against BigQuery
"""

import argparse
import re
import sys
import uuid

import yaml
from langchain_core.messages import HumanMessage

from src.agent.graph import Agent
from src.db import make_db_client
from src.knowledge.golden_bucket import GoldenBucket
from src.observability.tracing import Tracer
from src.settings import PROJECT_ROOT, settings
from src.stores.app_store import AppStore


def run() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    dataset = yaml.safe_load((PROJECT_ROOT / "evals" / "dataset.yaml").read_text())
    db = make_db_client("demo" if args.demo else settings.db_backend)
    tracer = Tracer(settings.trace_file)
    agent = Agent(
        db=db,
        store=AppStore(settings.app_db_path),
        bucket=GoldenBucket(settings.golden_bucket_dir, settings.candidates_dir),
        tracer=tracer,
    )

    failures = []
    for case in dataset["cases"]:
        name, question = case["name"], case["question"]
        config = {"configurable": {"thread_id": f"eval-{uuid.uuid4().hex}"}}
        tracer.start_turn("eval_user", question)
        try:
            state = agent.graph.invoke(
                {"messages": [HumanMessage(content=question)], "user_id": "eval_user"},
                config,
            )
            tracer.end_turn(state.get("outcome", "answered"))
        except Exception as e:
            tracer.end_turn("eval_crash")
            failures.append(f"{name}: agent crashed: {e}")
            print(f"FAIL {name}: CRASH {e}")
            continue

        answer = state["messages"][-1].content
        problems = []

        if "outcome_in" in case and state.get("outcome") not in case["outcome_in"]:
            problems.append(f"outcome={state.get('outcome')} not in {case['outcome_in']}")
        for needle in case.get("report_contains", []):
            if needle.lower() not in answer.lower():
                problems.append(f"answer missing expected text: {needle!r}")
        for pattern in case.get("report_not_matches", []):
            if re.search(pattern, answer):
                problems.append(f"answer matches forbidden pattern (PII leak?): {pattern!r}")
        if "sql_tables" in case:
            sql = (state.get("sql") or "").lower()
            for table in case["sql_tables"]:
                if table not in sql:
                    problems.append(f"SQL does not reference table {table!r}")

        if problems:
            failures.append(f"{name}: " + "; ".join(problems))
            print(f"FAIL {name}: " + "; ".join(problems))
        else:
            print(f"PASS {name}")

    print(f"\n{len(dataset['cases']) - len(failures)}/{len(dataset['cases'])} cases passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(run())
