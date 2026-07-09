"""Lightweight structured tracing.

Every user turn gets a trace_id; every graph node records a span with timing,
metadata (SQL text, attempt number, row counts, masked columns, errors) and
outcome. Traces are appended as JSON lines to logs/traces.jsonl so a turn can
be replayed step by step when debugging ("what did the model see, what did it
answer, where did it fail").

In production the same span structure ships to LangSmith / OpenTelemetry -
enable LangSmith by exporting LANGSMITH_TRACING=true (LangChain picks it up
automatically); this module still writes the local JSONL either way, so the
prototype's observability does not depend on a third-party service.
"""

import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TurnTrace:
    trace_id: str
    user_id: str
    user_message: str
    spans: list[dict] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    llm_calls: int = 0

    def add_span(self, name: str, duration_ms: float, status: str = "ok",
                 **metadata: Any) -> None:
        self.spans.append({
            "name": name,
            "duration_ms": round(duration_ms, 1),
            "status": status,
            **{k: v for k, v in metadata.items() if v is not None},
        })


class Tracer:
    def __init__(self, trace_file: Path) -> None:
        self.trace_file = trace_file
        self.trace_file.parent.mkdir(parents=True, exist_ok=True)
        self.current: TurnTrace | None = None
        self.debug = False

    def start_turn(self, user_id: str, user_message: str) -> TurnTrace:
        self.current = TurnTrace(
            trace_id=uuid.uuid4().hex[:12], user_id=user_id, user_message=user_message
        )
        return self.current

    @contextmanager
    def span(self, name: str, **metadata: Any):
        start = time.time()
        holder: dict[str, Any] = {}
        try:
            yield holder
            status = holder.pop("status", "ok")
        except Exception as e:
            if self.current:
                self.current.add_span(name, (time.time() - start) * 1000,
                                      status="error", error=str(e)[:500], **metadata)
                self._debug_print(name, "error", str(e)[:200])
            raise
        if self.current:
            self.current.add_span(name, (time.time() - start) * 1000,
                                  status=status, **metadata, **holder)
            self._debug_print(name, status, holder)

    def count_llm_call(self) -> None:
        if self.current:
            self.current.llm_calls += 1

    def end_turn(self, outcome: str) -> None:
        if not self.current:
            return
        record = {
            "trace_id": self.current.trace_id,
            "user_id": self.current.user_id,
            "user_message": self.current.user_message,
            "outcome": outcome,   # answered | refused | healed_then_answered | failed | ...
            "total_ms": round((time.time() - self.current.started_at) * 1000, 1),
            "llm_calls": self.current.llm_calls,
            "spans": self.current.spans,
        }
        with self.trace_file.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self.current = None

    def _debug_print(self, name: str, status: str, detail: Any) -> None:
        if self.debug:
            print(f"    [trace] {name}: {status} {detail if detail else ''}")
