from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class HistoryDB:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        self.conn.close()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                stage TEXT NOT NULL,
                config_hash TEXT NOT NULL,
                config_path TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                status TEXT NOT NULL,
                note TEXT
            );

            CREATE TABLE IF NOT EXISTS batches (
                run_id TEXT NOT NULL,
                batch_id INTEGER NOT NULL,
                status TEXT NOT NULL,
                total_items INTEGER NOT NULL,
                done_items INTEGER NOT NULL DEFAULT 0,
                failed_items INTEGER NOT NULL DEFAULT 0,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                PRIMARY KEY (run_id, batch_id)
            );

            CREATE TABLE IF NOT EXISTS items (
                run_id TEXT NOT NULL,
                item_id TEXT NOT NULL,
                sample_id TEXT NOT NULL,
                source_id TEXT NOT NULL,
                split TEXT NOT NULL,
                raw_path TEXT NOT NULL,
                color_path TEXT NOT NULL,
                gray_path TEXT NOT NULL,
                status TEXT NOT NULL,
                last_error TEXT,
                retries INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (run_id, item_id)
            );

            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                item_id TEXT,
                batch_id INTEGER,
                event_type TEXT NOT NULL,
                message TEXT,
                created_at TEXT NOT NULL
            );
            """
        )
        self.conn.commit()

    def create_or_update_run(
        self,
        run_id: str,
        stage: str,
        config_hash: str,
        config_path: str,
        status: str = "running",
        note: str = "",
    ) -> None:
        now = now_utc_iso()
        self.conn.execute(
            """
            INSERT INTO runs (run_id, stage, config_hash, config_path, started_at, status, note)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                stage = excluded.stage,
                config_hash = excluded.config_hash,
                config_path = excluded.config_path,
                status = excluded.status,
                note = excluded.note
            """,
            (run_id, stage, config_hash, config_path, now, status, note),
        )
        self.conn.commit()

    def finish_run(self, run_id: str, status: str, note: str = "") -> None:
        self.conn.execute(
            "UPDATE runs SET finished_at = ?, status = ?, note = ? WHERE run_id = ?",
            (now_utc_iso(), status, note, run_id),
        )
        self.conn.commit()

    def upsert_items(self, run_id: str, items: Iterable[dict[str, str]]) -> None:
        rows = []
        now = now_utc_iso()
        for item in items:
            rows.append(
                (
                    run_id,
                    item["item_id"],
                    item["sample_id"],
                    item["source_id"],
                    item["split"],
                    item["raw_path"],
                    item["color_path"],
                    item["gray_path"],
                    item.get("status", "pending"),
                    item.get("last_error", ""),
                    int(item.get("retries", 0)),
                    now,
                )
            )
        self.conn.executemany(
            """
            INSERT INTO items (
                run_id, item_id, sample_id, source_id, split,
                raw_path, color_path, gray_path, status, last_error, retries, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, item_id) DO UPDATE SET
                sample_id = excluded.sample_id,
                source_id = excluded.source_id,
                split = excluded.split,
                raw_path = excluded.raw_path,
                color_path = excluded.color_path,
                gray_path = excluded.gray_path
            """,
            rows,
        )
        self.conn.commit()

    def mark_item_status(
        self,
        run_id: str,
        item_id: str,
        status: str,
        last_error: str = "",
        increment_retry: bool = False,
    ) -> None:
        if increment_retry:
            self.conn.execute(
                """
                UPDATE items
                SET status = ?, last_error = ?, retries = retries + 1, updated_at = ?
                WHERE run_id = ? AND item_id = ?
                """,
                (status, last_error, now_utc_iso(), run_id, item_id),
            )
        else:
            self.conn.execute(
                """
                UPDATE items
                SET status = ?, last_error = ?, updated_at = ?
                WHERE run_id = ? AND item_id = ?
                """,
                (status, last_error, now_utc_iso(), run_id, item_id),
            )
        self.conn.commit()

    def get_items_by_status(self, run_id: str, statuses: list[str]) -> list[sqlite3.Row]:
        if not statuses:
            return []
        placeholders = ",".join("?" for _ in statuses)
        query = (
            "SELECT * FROM items WHERE run_id = ? "
            f"AND status IN ({placeholders}) ORDER BY source_id, item_id"
        )
        cursor = self.conn.execute(query, [run_id, *statuses])
        return cursor.fetchall()

    def get_all_items(self, run_id: str) -> list[sqlite3.Row]:
        cursor = self.conn.execute(
            "SELECT * FROM items WHERE run_id = ? ORDER BY source_id, item_id", (run_id,)
        )
        return cursor.fetchall()

    def record_batch_start(self, run_id: str, batch_id: int, total_items: int) -> None:
        self.conn.execute(
            """
            INSERT INTO batches (run_id, batch_id, status, total_items, started_at)
            VALUES (?, ?, 'running', ?, ?)
            ON CONFLICT(run_id, batch_id) DO UPDATE SET
                status = 'running',
                total_items = excluded.total_items,
                started_at = excluded.started_at,
                finished_at = NULL
            """,
            (run_id, batch_id, total_items, now_utc_iso()),
        )
        self.conn.commit()

    def record_batch_finish(
        self, run_id: str, batch_id: int, done_items: int, failed_items: int
    ) -> None:
        status = "done" if failed_items == 0 else "partial_failed"
        self.conn.execute(
            """
            UPDATE batches
            SET status = ?, done_items = ?, failed_items = ?, finished_at = ?
            WHERE run_id = ? AND batch_id = ?
            """,
            (status, done_items, failed_items, now_utc_iso(), run_id, batch_id),
        )
        self.conn.commit()

    def record_event(
        self,
        run_id: str,
        event_type: str,
        message: str,
        item_id: str | None = None,
        batch_id: int | None = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO events (run_id, item_id, batch_id, event_type, message, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (run_id, item_id, batch_id, event_type, message, now_utc_iso()),
        )
        self.conn.commit()

