"""Append-only JSONL logs: what was asked, what happened, what it cost.

The spec asks for missing recipes to be logged, and that log is the most useful
artifact this system produces. Every miss is a request a real user made that
neither the dataset nor the web could answer, which makes the file a ranked,
evidence-backed list of what to add to the dataset next.

JSONL rather than a database: appends are atomic enough for a single-process
Streamlit app, the file survives a crash mid-write with at most one bad line,
and it can be read with pandas in one call. Logs live beside the index in
ARTIFACT_DIR, off the synced drive, because OneDrive locking a file that is
being appended to on every query is a failure this project has already hit once.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path

from . import config

QUERY_LOG = config.ARTIFACT_DIR / "query_log.jsonl"
MISS_LOG = config.ARTIFACT_DIR / "missing_recipes.jsonl"


def _write(path: Path, record: dict) -> None:
    """Append one record. Logging must never take the request down with it."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **record}
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=_fallback) + "\n")
    except OSError:
        pass


def _fallback(value):
    if is_dataclass(value):
        return asdict(value)
    return str(value)


def log_query(record: dict) -> None:
    """Every request, answered or not. The denominator for every rate we quote."""
    _write(QUERY_LOG, record)


def log_miss(record: dict) -> None:
    """Requests that ended in an apology.

    Written *in addition to* the query log, not instead of it, so that "how
    often do we fail" and "what do we fail at" are separate questions with
    separate files.
    """
    _write(MISS_LOG, record)


def read(path: Path) -> list[dict]:
    """Load a log, skipping any line a crash left half-written."""
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def miss_summary(top: int = 20) -> list[tuple[str, int]]:
    """Most-requested unanswerable dishes, commonest first."""
    counts: dict[str, int] = {}
    for record in read(MISS_LOG):
        key = (record.get("cleaned") or record.get("query") or "").strip().lower()
        if key:
            counts[key] = counts.get(key, 0) + 1
    return sorted(counts.items(), key=lambda pair: pair[1], reverse=True)[:top]
