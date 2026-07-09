"""SQLite-backed application state: saved reports library + user preferences.

One small store, two concerns:
  * ReportStore     - the "Saved Reports" library (with owner-scoped deletion,
                      used by the high-stakes confirmation flow)
  * PreferenceStore - user-level learning loop ("Manager A prefers tables")

SQLite keeps the prototype dependency-free; in production both map to a
managed Postgres (see ARCHITECTURE.md).
"""

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    title TEXT NOT NULL,
    question TEXT NOT NULL,
    sql TEXT NOT NULL,
    report_md TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS preferences (
    user_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (user_id, key)
);
"""


@dataclass
class SavedReport:
    id: int
    user_id: str
    title: str
    question: str
    sql: str
    report_md: str
    created_at: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class AppStore:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)

    # ---- reports library -----------------------------------------------
    def save_report(self, user_id: str, title: str, question: str, sql: str,
                    report_md: str) -> int:
        cur = self.conn.execute(
            "INSERT INTO reports (user_id, title, question, sql, report_md, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, title, question, sql, report_md, _now()),
        )
        self.conn.commit()
        return cur.lastrowid

    def list_reports(self, user_id: str) -> list[SavedReport]:
        rows = self.conn.execute(
            "SELECT * FROM reports WHERE user_id = ? ORDER BY created_at DESC", (user_id,)
        ).fetchall()
        return [SavedReport(**dict(r)) for r in rows]

    def find_reports(self, user_id: str, text_query: str | None = None,
                     created_on: str | None = None) -> list[SavedReport]:
        """Owner-scoped search used by the deletion flow. `user_id` is taken
        from the authenticated session, never from model output - users can
        only ever see/delete their own reports."""
        sql = "SELECT * FROM reports WHERE user_id = ?"
        params: list = [user_id]
        if text_query:
            sql += " AND (title LIKE ? OR question LIKE ? OR report_md LIKE ?)"
            like = f"%{text_query}%"
            params += [like, like, like]
        if created_on:  # ISO date, e.g. "2026-07-08"
            sql += " AND created_at LIKE ?"
            params.append(f"{created_on}%")
        rows = self.conn.execute(sql + " ORDER BY created_at DESC", params).fetchall()
        return [SavedReport(**dict(r)) for r in rows]

    def delete_reports(self, user_id: str, report_ids: list[int]) -> int:
        """Deletes only rows that match BOTH the id list and the owner."""
        if not report_ids:
            return 0
        marks = ",".join("?" * len(report_ids))
        cur = self.conn.execute(
            f"DELETE FROM reports WHERE user_id = ? AND id IN ({marks})",
            [user_id, *report_ids],
        )
        self.conn.commit()
        return cur.rowcount

    # ---- user preferences ------------------------------------------------
    def set_preference(self, user_id: str, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO preferences (user_id, key, value, updated_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(user_id, key) DO UPDATE SET value = excluded.value, "
            "updated_at = excluded.updated_at",
            (user_id, key, value, _now()),
        )
        self.conn.commit()

    def get_preferences(self, user_id: str) -> dict[str, str]:
        rows = self.conn.execute(
            "SELECT key, value FROM preferences WHERE user_id = ?", (user_id,)
        ).fetchall()
        return {r["key"]: r["value"] for r in rows}
