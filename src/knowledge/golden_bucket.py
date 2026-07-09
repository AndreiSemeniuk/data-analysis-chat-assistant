"""Golden Knowledge Bucket: retrieval over expert Trios
(Question -> SQL -> Analyst Report).

Prototype retrieval is lexical BM25 - zero external dependencies, fully
offline, deterministic, and plenty for a bucket of dozens of trios. In
production this swaps for a vector store (see ARCHITECTURE.md); the interface
(`retrieve(question, k)`) stays identical, which is the point of keeping it
behind this class.

The bucket also grows: every successful interaction can be promoted to a
*candidate* trio (see `save_candidate`) which a human analyst reviews before
it enters the golden set - the update loop described in the design doc.
"""

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml


@dataclass
class Trio:
    question: str
    sql: str
    report: str
    source_file: str = ""


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class GoldenBucket:
    def __init__(self, trios_dir: Path, candidates_dir: Path | None = None) -> None:
        self.trios_dir = trios_dir
        self.candidates_dir = candidates_dir
        self.trios: list[Trio] = []
        for path in sorted(trios_dir.glob("*.yaml")):
            data = yaml.safe_load(path.read_text())
            self.trios.append(Trio(
                question=data["question"], sql=data["sql"],
                report=data["report"], source_file=path.name,
            ))
        self._docs = [_tokenize(t.question) for t in self.trios]
        self._avgdl = (sum(len(d) for d in self._docs) / len(self._docs)) if self._docs else 0
        self._df: dict[str, int] = {}
        for doc in self._docs:
            for term in set(doc):
                self._df[term] = self._df.get(term, 0) + 1

    def _bm25(self, query: list[str], doc: list[str], k1: float = 1.5, b: float = 0.75) -> float:
        score, n = 0.0, len(self._docs)
        for term in query:
            tf = doc.count(term)
            if tf == 0:
                continue
            idf = math.log(1 + (n - self._df.get(term, 0) + 0.5) / (self._df.get(term, 0) + 0.5))
            score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(doc) / (self._avgdl or 1)))
        return score

    def retrieve(self, question: str, k: int = 3) -> list[Trio]:
        query = _tokenize(question)
        scored = [(self._bm25(query, doc), trio) for doc, trio in zip(self._docs, self.trios)]
        scored = [(s, t) for s, t in scored if s > 0]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [t for _, t in scored[:k]]

    def save_candidate(self, question: str, sql: str, report: str, user_id: str) -> Path:
        """System-level learning loop: promote a successful interaction to a
        candidate trio awaiting human review."""
        assert self.candidates_dir is not None
        self.candidates_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        path = self.candidates_dir / f"candidate_{stamp}.yaml"
        path.write_text(yaml.safe_dump(
            {"question": question, "sql": sql, "report": report,
             "submitted_by": user_id, "status": "pending_review"},
            sort_keys=False, allow_unicode=True,
        ))
        return path

    def render_for_prompt(self, trios: list[Trio]) -> str:
        if not trios:
            return "(no similar past analyses found)"
        blocks = []
        for i, t in enumerate(trios, 1):
            blocks.append(
                f"### Example {i}\nQuestion: {t.question}\nSQL:\n```sql\n{t.sql}\n```\n"
                f"Analyst report:\n{t.report}"
            )
        return "\n\n".join(blocks)
