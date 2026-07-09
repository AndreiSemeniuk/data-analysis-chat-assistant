"""Shared state flowing through the LangGraph."""

from typing import Annotated, Any, Optional

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    # Conversation (checkpointed across turns by LangGraph)
    messages: Annotated[list[AnyMessage], add_messages]
    user_id: str

    # Routing (per turn)
    intent: str
    is_suspicious: bool
    delete_text_query: Optional[str]
    delete_date: Optional[str]

    # SQL pipeline (per turn)
    sql: Optional[str]
    sql_attempt: int
    sql_error: Optional[str]
    result_markdown: Optional[str]   # masked, truncated results as markdown
    result_rows: int
    masked_columns: list[str]

    # Output (per turn)
    report: Optional[str]
    outcome: str                     # for tracing: answered/refused/failed/...
