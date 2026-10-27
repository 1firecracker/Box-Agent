"""Plugin-owned local decisions, with atomic turn and reminder deduplication."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path


class Journal:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS decisions (
                    session_id TEXT NOT NULL, turn_id TEXT NOT NULL,
                    updated REAL NOT NULL, payload TEXT NOT NULL,
                    PRIMARY KEY (session_id, turn_id)
                );
                CREATE TABLE IF NOT EXISTS reminders (
                    fingerprint TEXT PRIMARY KEY, created REAL NOT NULL,
                    payload TEXT NOT NULL
                );
            """)
        path.chmod(0o600)

    def connect(self):
        return sqlite3.connect(self.path, timeout=0.2)

    def claim(self, session_id: str, turn_id: str) -> bool:
        with self.connect() as db:
            return db.execute("INSERT OR IGNORE INTO decisions VALUES (?, ?, ?, ?)",
                              (session_id, turn_id, time.time(), '{"status":"pending"}')).rowcount == 1

    def recent(self) -> list[dict]:
        with self.connect() as db:
            rows = db.execute("SELECT payload FROM reminders ORDER BY created DESC LIMIT 20").fetchall()
        return [json.loads(row[0]) for row in rows]

    def finish(self, session_id: str, turn_id: str, record: dict) -> dict:
        record = dict(record)
        with self.connect() as db:
            if record["status"] == "suggest":
                decision = record["decision"]
                digests = sorted(item["digest"] for item in record["evidence"]
                                 if item["id"] in decision["evidence_ids"])
                scope = [decision["kind"], decision["target"].strip().casefold(), digests]
                action = " ".join(decision["suggestion"].split()).casefold()
                if decision["kind"] == "watch":
                    # A proposal is scoped to its original session, source and dates,
                    # not the model's wording. A retry must not reopen the window.
                    watch = decision["watch"]
                    scope = ["watch", session_id, watch["source_url"],
                             watch.get("planned_visit_date"), watch["expires_on"]]
                    if watch.get("check_condition"):
                        scope.extend([watch["subject"].strip().casefold(),
                                      " ".join(watch["check_condition"].split()).casefold()])
                    action = "watch_proposal"
                # Historical memory can remain unchanged while this turn corrects a fact.
                # The judge handles semantic repetition; persistence deduplicates the action,
                # not every future update to the same document.
                fingerprint = hashlib.sha256(json.dumps(
                    [*scope, action], ensure_ascii=False,
                ).encode()).hexdigest()
                legacy_key = hashlib.sha256(json.dumps(scope, ensure_ascii=False).encode()).hexdigest()
                legacy = db.execute("SELECT payload FROM reminders WHERE fingerprint=?",
                                    (legacy_key,)).fetchone()
                legacy_duplicate = bool(legacy) and (
                    " ".join(json.loads(legacy[0])["suggestion"].split()).casefold() == action
                )
                inserted = 0 if legacy_duplicate else db.execute(
                    "INSERT OR IGNORE INTO reminders VALUES (?, ?, ?)",
                    (fingerprint, time.time(), json.dumps(decision, ensure_ascii=False)),
                ).rowcount
                if not inserted:
                    record.update(status="silent", reason="duplicate_suggestion")
            db.execute("UPDATE decisions SET updated=?, payload=? WHERE session_id=? AND turn_id=?",
                       (time.time(), json.dumps(record, ensure_ascii=False), session_id, turn_id))
        return record

    def pending_deliveries(self, session_id: str) -> list[tuple[str, dict]]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT turn_id, payload FROM decisions WHERE session_id=? "
                "AND json_extract(payload, '$.status')='suggest' "
                "AND json_extract(payload, '$.delivery')='pending' ORDER BY updated LIMIT 20",
                (session_id,),
            ).fetchall()
        return [(turn_id, json.loads(payload)) for turn_id, payload in rows]

    def mark_delivery(self, session_id: str, turn_id: str, status: str, error: str = "") -> None:
        with self.connect() as db:
            db.execute(
                "UPDATE decisions SET payload=json_set(payload, '$.delivery', ?, '$.delivery_error', ?) "
                "WHERE session_id=? AND turn_id=? AND json_extract(payload, '$.status')='suggest'",
                (status, error, session_id, turn_id),
            )
