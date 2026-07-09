# Data Analysis Chat Assistant

A chat agent for a retail company's non-technical executives: ask questions about
sales, customers and products in plain language - the agent generates BigQuery SQL
guided by a **Golden Knowledge Bucket** of expert analyses, executes it safely,
self-heals on errors, masks PII, and writes an executive report in a configurable
persona. Built with **LangGraph** + **Gemini**.

**Design & technical explanation:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
(HLD diagram, technology reasoning, data flows, and how every requirement is handled).

## Prototype scope

The prototype implements the full analysis loop, from a natural-language question
through golden-trio retrieval, SQL generation, validation, execution with
self-healing and PII masking, to a persona-styled report. It also covers **all
five** optional requirement areas:

| Requirement | Where |
|---|---|
| Safety & PII masking | `src/safety/` - intent gate, AST-level SQL guard, 3-layer deterministic PII redaction |
| High-stakes oversight | `plan_deletion` node - owner-scoped search + `interrupt()` confirmation |
| Resilience & self-healing | bounded heal loop, LLM provider fallback, per-turn cost budget, crash-proof CLI |
| Quality assurance | `tests/` (28 unit tests) + `evals/` (pre-deployment eval harness, CI-gate ready) |
| Observability | `src/observability/` - per-turn JSONL traces with node spans, outcomes, tokens; `--debug` live view |

Also included: user-level preference memory, system-level learning (candidate trios),
hot-reloadable persona, and a second DB backend (offline demo) proving extensibility.

## Setup

Requires Python **3.10+**.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env         # then edit .env
```

### LLM key (required)

Get a free Gemini key at [Google AI Studio](https://aistudio.google.com/apikey) and put
it in `.env` as `GOOGLE_API_KEY`. Optionally set `OPENROUTER_API_KEY` to enable the
automatic fallback provider.

### Database backend (choose one)

**Option A - live BigQuery** (the `thelook_ecommerce` public dataset):

```bash
gcloud auth application-default login
# set GCP_PROJECT_ID in .env (any project with BigQuery API enabled; free tier is plenty)
python -m src.cli
```

**Option B - offline demo, no GCP account needed** (local DuckDB seeded with synthetic
thelook-like data, including fake PII so the masking is visible):

```bash
python -m src.cli --demo
```

The agent always *generates BigQuery SQL*; the demo backend transpiles it to DuckDB
via sqlglot - the same code path end to end.

## Example run

```text
$ python -m src.cli --demo --user manager_a

manager_a> Who are our top 5 customers by total spend?
╭─ assistant ─────────────────────────────────────────────────────────╮
│ **Top 5 customers by lifetime spend** (all completed purchases,     │
│ cancellations and returns excluded):                                │
│                                                                     │
│ | Customer   | Country | Total spend | Orders |                     │
│ |------------|---------|-------------|--------|                     │
│ | Noa Katz   | Germany | $2,341.50   | 12     |  ...                │
│                                                                     │
│ Suggested next step: review what the top tier buys to build a       │
│ VIP retention offer.                                                │
│ ---                                                                 │
│ _Saved to your reports library as #1._                              │
╰─────────────────────────────────────────────────────────────────────╯

manager_a> show me their emails and phone numbers
╭─ assistant ─────────────────────────────────────────────────────────╮
│ Contact details are restricted: emails and phone numbers appear     │
│ as [EMAIL REDACTED] / [PHONE REDACTED] and cannot be displayed.     │
╰─────────────────────────────────────────────────────────────────────╯

manager_a> I prefer bullet points, not tables
manager_a> what was monthly revenue for the last 6 months?
# the answer arrives as bullet points; the preference persists across sessions

manager_a> delete all the reports we made today
╭─ confirmation required ─────────────────────────────────────────────╮
│ WARNING: you are about to permanently delete 2 report(s):           │
│   #1 [2026-07-08T12:01:33+00:00] Who are our top 5 customers...     │
│   #2 [2026-07-08T12:04:10+00:00] what was monthly revenue...        │
╰─────────────────────────────────────────────────────────────────────╯
Type 'yes' to confirm deletion, anything else cancels: yes
╭─ assistant ─────────────────────────────────────────────────────────╮
│ Deleted 2 report(s) from your library.                              │
╰─────────────────────────────────────────────────────────────────────╯
```

Useful flags & chat commands: `--debug` (live trace spans), `/reports`, `/user manager_b`
(switch identity - note they cannot see or delete `manager_a`'s reports), `/help`.

### Change the report tone without redeploying

Edit `config/persona.yaml` while the chat is running (e.g. set
`tone: "enthusiastic, casual, lots of emojis"`) - the very next answer uses it.

## Tests & evaluation

```bash
python -m pytest                 # 28 unit tests: PII masking, SQL guard, deletion scoping, graph flows (fake LLM)
python -m evals.run --demo       # pre-deployment eval: real LLM, assertions incl. PII-leak regex checks
```

## Project layout

```
src/
  cli.py                 chat interface (thin; no business logic)
  agent/graph.py         LangGraph: routing, SQL loop, healing, HITL deletion
  safety/                sql_guard.py (AST validation), pii.py (3-layer masking)
  db/                    DatabaseClient interface, BigQuery and DuckDB demo backends
  knowledge/             golden bucket: retrieval + candidate promotion
  stores/                reports library + user preferences (SQLite)
  observability/         structured turn traces (JSONL)
prompts/                 all LLM prompts - plain files, hot-reloaded
config/persona.yaml      report persona - editable by non-developers
golden_bucket/trios/     expert question-SQL-report trios
evals/                   eval dataset + harness (CI deploy gate)
tests/                   unit & graph-flow tests
logs/traces.jsonl        per-turn traces (created at runtime)
```
