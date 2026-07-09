"""Chat CLI.

    python -m src.cli                     # BigQuery backend (needs GCP auth)
    python -m src.cli --demo              # local DuckDB demo, no GCP needed
    python -m src.cli --user manager_b    # act as a different user
    python -m src.cli --debug             # live trace spans while chatting

Commands inside the chat: /reports, /user <id>, /debug, /help, /quit
"""

import argparse
import sys
import uuid

from langchain_core.messages import HumanMessage
from langgraph.types import Command
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from src.agent.graph import Agent, TurnBudgetExceeded
from src.db import make_db_client
from src.knowledge.golden_bucket import GoldenBucket
from src.observability.tracing import Tracer
from src.settings import settings
from src.stores.app_store import AppStore

console = Console()

HELP = """\
Ask analysis questions in plain language, e.g.:
  - Who are our top 10 customers by total spend?
  - What was monthly revenue for the last 6 months?
  - Which product categories have the best margins?
  - What tables and columns do you have?
Manage saved reports:
  - "show my reports" / "delete all reports we made today"
Commands: /reports  /user <id>  /debug  /help  /quit
"""


def print_answer(text: str) -> None:
    console.print(Panel(Markdown(text), border_style="cyan", title="assistant"))


def handle_interrupt(agent: Agent, payload: dict, config: dict) -> dict:
    """Human-in-the-loop confirmation for destructive operations."""
    console.print(Panel(
        "\n".join([
            f"[bold red]WARNING: you are about to permanently delete {payload['count']} report(s):[/bold red]",
            *[f"  {line}" for line in payload["reports"]],
        ]),
        border_style="red", title="confirmation required",
    ))
    answer = console.input("[bold]Type 'yes' to confirm deletion, anything else cancels: [/bold]")
    result = agent.graph.invoke(Command(resume=answer.strip().lower() == "yes"), config)
    print_answer(result["messages"][-1].content)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Data Analysis Chat Assistant")
    parser.add_argument("--demo", action="store_true",
                        help="use the local DuckDB demo database instead of BigQuery")
    parser.add_argument("--user", default=settings.default_user, help="user id for this session")
    parser.add_argument("--debug", action="store_true", help="print trace spans live")
    args = parser.parse_args()

    backend = "demo" if args.demo else settings.db_backend
    console.print(f"[dim]Starting... backend={backend}, model={settings.model_name}[/dim]")

    tracer = Tracer(settings.trace_file)
    tracer.debug = args.debug
    try:
        db = make_db_client(backend)
        db.get_schemas()  # fail fast with a clear message if auth is missing
    except Exception as e:
        console.print(f"[red]Could not connect to the database backend '{backend}': {e}[/red]")
        console.print("[yellow]Tip: run with --demo to use the local demo database.[/yellow]")
        sys.exit(1)

    store = AppStore(settings.app_db_path)
    bucket = GoldenBucket(settings.golden_bucket_dir, settings.candidates_dir)
    agent = Agent(db=db, store=store, bucket=bucket, tracer=tracer)

    user_id = args.user
    thread_id = uuid.uuid4().hex
    console.print(Panel(
        f"[bold]Retail Data Analysis Assistant[/bold]\nUser: [green]{user_id}[/green] | "
        f"Backend: {backend} | type /help for examples",
        border_style="green",
    ))

    while True:
        try:
            user_input = console.input(f"[bold green]{user_id}>[/bold green] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]bye[/dim]")
            break
        if not user_input:
            continue
        if user_input in ("/quit", "/exit", "exit", "quit"):
            break
        if user_input == "/help":
            console.print(HELP)
            continue
        if user_input == "/debug":
            tracer.debug = not tracer.debug
            console.print(f"[dim]debug tracing: {tracer.debug}[/dim]")
            continue
        if user_input.startswith("/user"):
            parts = user_input.split(maxsplit=1)
            if len(parts) == 2:
                user_id = parts[1].strip()
                thread_id = uuid.uuid4().hex  # new conversation for the new user
                console.print(f"[dim]now acting as {user_id}[/dim]")
            continue
        if user_input == "/reports":
            user_input = "show my saved reports"

        config = {"configurable": {"thread_id": thread_id}}
        tracer.start_turn(user_id, user_input)
        try:
            with console.status("[dim]analyzing...[/dim]"):
                result = agent.graph.invoke(
                    {"messages": [HumanMessage(content=user_input)], "user_id": user_id},
                    config,
                )
            if "__interrupt__" in result:
                result = handle_interrupt(agent, result["__interrupt__"][0].value, config)
            else:
                print_answer(result["messages"][-1].content)
            tracer.end_turn(result.get("outcome", "answered"))
        except TurnBudgetExceeded as e:
            console.print(f"[yellow]Stopped to protect costs: {e}[/yellow]")
            tracer.end_turn("budget_exceeded")
        except Exception as e:
            # The UI must never crash on a failing turn.
            console.print(f"[red]Something went wrong with this request: {e}[/red]")
            console.print("[dim]The error was recorded in the trace log.[/dim]")
            tracer.end_turn("unhandled_error")


if __name__ == "__main__":
    main()
