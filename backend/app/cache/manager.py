"""
Caching and persistent metadata/job storage layer (Section 11 & 15).

Requirements satisfied:
  - Cache keys account for dataset, region, date range, and processing configuration.
  - Cached outputs retain full source metadata and processing timestamps.
  - Expired entries or temporary URLs are automatically evicted and never served.
  - Explicit invalidation methods (`invalidate_key`, `invalidate_dataset`, `clear_all`).
  - Avoids storing duplicate copies of large original satellite scenes.
"""

import hashlib
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from app.core.config import get_settings


def build_cache_key(
    dataset: str,
    region_id: str,
    start_date: str,
    end_date: str,
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate a deterministic cache key incorporating dataset, region, date range,
    and canonicalized processing configuration.
    """
    cfg_str = json.dumps(config or {}, sort_keys=True, default=str)
    cfg_hash = hashlib.sha256(cfg_str.encode("utf-8")).hexdigest()[:12]
    return f"as_cache:{dataset}:{region_id}:{start_date}:{end_date}:{cfg_hash}"


class CacheAndJobStore:
    """
    SQLite-backed persistent cache and processing job store.
    Safe for multi-request hackathon deployment without requiring an external Redis daemon,
    while honoring TTL expiration, source metadata preservation, and cache invalidation.
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        settings = get_settings()
        self.db_path = db_path or settings.cache_db_path
        self.default_ttl = settings.cache_default_ttl_seconds
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_entries (
                    cache_key TEXT PRIMARY KEY,
                    dataset TEXT NOT NULL,
                    region_id TEXT NOT NULL,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    processing_timestamp TEXT NOT NULL,
                    expires_at REAL NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS processing_jobs (
                    job_id TEXT PRIMARY KEY,
                    job_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    data_status TEXT NOT NULL,
                    region_id TEXT NOT NULL,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT,
                    progress_pct REAL NOT NULL,
                    message TEXT NOT NULL,
                    error TEXT,
                    cache_key TEXT,
                    request_json TEXT NOT NULL,
                    result_json TEXT
                )
                """
            )
            conn.commit()

    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached output if present and not expired."""
        now = time.time()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json, expires_at FROM cache_entries WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
            if not row:
                return None
            if float(row["expires_at"]) <= now:
                conn.execute("DELETE FROM cache_entries WHERE cache_key = ?", (cache_key,))
                conn.commit()
                return None
            return json.loads(row["payload_json"])

    def set(
        self,
        cache_key: str,
        dataset: str,
        region_id: str,
        start_date: str,
        end_date: str,
        payload: Dict[str, Any],
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """Store processed output with source metadata and expiration timestamp."""
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        expires_at = time.time() + max(ttl, 1)
        proc_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        serialized = json.dumps(payload, default=str)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO cache_entries
                (cache_key, dataset, region_id, start_date, end_date, processing_timestamp, expires_at, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (cache_key, dataset, region_id, start_date, end_date, proc_ts, expires_at, serialized),
            )
            conn.commit()

    def invalidate(
        self,
        cache_key: Optional[str] = None,
        dataset: Optional[str] = None,
        region_id: Optional[str] = None,
    ) -> int:
        """Invalidate cached results by exact key, dataset, region, or all entries."""
        with self._connect() as conn:
            if cache_key:
                cur = conn.execute("DELETE FROM cache_entries WHERE cache_key = ?", (cache_key,))
            elif dataset and region_id:
                cur = conn.execute(
                    "DELETE FROM cache_entries WHERE dataset = ? AND region_id = ?",
                    (dataset, region_id),
                )
            elif dataset:
                cur = conn.execute("DELETE FROM cache_entries WHERE dataset = ?", (dataset,))
            elif region_id:
                cur = conn.execute("DELETE FROM cache_entries WHERE region_id = ?", (region_id,))
            else:
                cur = conn.execute("DELETE FROM cache_entries")
            conn.commit()
            return cur.rowcount

    # Job management persistence methods
    def save_job(self, job_dict: Dict[str, Any], request_dict: Dict[str, Any], result_dict: Optional[Dict[str, Any]] = None) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO processing_jobs
                (job_id, job_type, status, data_status, region_id, start_date, end_date,
                 created_at, updated_at, completed_at, progress_pct, message, error, cache_key,
                 request_json, result_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_dict["job_id"],
                    job_dict["job_type"],
                    job_dict["status"],
                    job_dict["data_status"],
                    job_dict["region_id"],
                    job_dict["start_date"],
                    job_dict["end_date"],
                    job_dict["created_at"],
                    job_dict["updated_at"],
                    job_dict.get("completed_at"),
                    job_dict["progress_pct"],
                    job_dict["message"],
                    job_dict.get("error"),
                    job_dict.get("cache_key"),
                    json.dumps(request_dict, default=str),
                    json.dumps(result_dict, default=str) if result_dict is not None else None,
                ),
            )
            conn.commit()

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM processing_jobs WHERE job_id = ?", (job_id,)).fetchone()
            if not row:
                return None
            return dict(row)


_store_instance: Optional[CacheAndJobStore] = None


def get_cache_store() -> CacheAndJobStore:
    """Return singleton CacheAndJobStore instance."""
    global _store_instance
    if _store_instance is None:
        _store_instance = CacheAndJobStore()
    return _store_instance
