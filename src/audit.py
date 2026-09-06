"""SQLite audit log: every tool call, what it was given, and what happened.

Two-phase by design. begin() writes the INTENT row before the tool runs and
complete() fills in the outcome afterwards. If the app crashes or is killed
mid-operation, the intent row survives with outcome='pending' -- which is
exactly the case you most want a record of. A single write-after-the-fact
would lose it.

Denied calls are logged too. A rejected path-traversal attempt is more
security-interesting than a successful read.
"""
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS tool_calls (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc      TEXT    NOT NULL,
    tool        TEXT    NOT NULL,
    arguments   TEXT    NOT NULL,          -- JSON
    tier        TEXT    NOT NULL,          -- read_only | destructive
    confirmed   INTEGER NOT NULL,          -- 0 = auto-run, 1 = user confirmed
    outcome     TEXT    NOT NULL,          -- pending | ok | denied | error
    result      TEXT,                      -- truncated result or error message
    duration_ms INTEGER
);
CREATE INDEX IF NOT EXISTS idx_tool_calls_ts   ON tool_calls(ts_utc);
CREATE INDEX IF NOT EXISTS idx_tool_calls_tool ON tool_calls(tool);
"""

MAX_RESULT_CHARS = 4000


class AuditLog:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # WAL: the background indexer and the UI both touch this DB.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=FULL")  # durability > speed here
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def begin(self, tool: str, arguments: dict, tier: str, confirmed: bool) -> tuple[int, float]:
        """Record intent BEFORE execution. Returns (row_id, start_time)."""
        cur = self._conn.execute(
            "INSERT INTO tool_calls (ts_utc, tool, arguments, tier, confirmed, outcome)"
            " VALUES (?,?,?,?,?,'pending')",
            (
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                tool,
                json.dumps(arguments, default=str, ensure_ascii=False),
                tier,
                int(confirmed),
            ),
        )
        self._conn.commit()
        return cur.lastrowid, time.time()

    def complete(self, row_id: int, started: float, outcome: str, result: str = "") -> None:
        """Fill in the outcome after execution."""
        text = str(result)
        if len(text) > MAX_RESULT_CHARS:
            text = text[:MAX_RESULT_CHARS] + f"... [truncated, {len(text)} chars total]"
        self._conn.execute(
            "UPDATE tool_calls SET outcome=?, result=?, duration_ms=? WHERE id=?",
            (outcome, text, int((time.time() - started) * 1000), row_id),
        )
        self._conn.commit()

    def log_denied(self, tool: str, arguments: dict, reason: str) -> int:
        """One-shot record for a call rejected before it ever ran."""
        row_id, started = self.begin(tool, arguments, tier="denied", confirmed=False)
        self.complete(row_id, started, "denied", reason)
        return row_id

    def recent(self, limit: int = 50) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM tool_calls ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()

    def close(self) -> None:
        self._conn.close()
