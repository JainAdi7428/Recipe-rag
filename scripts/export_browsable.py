"""Export a small, flat, human-readable SQLite copy of the index.

Chroma's own chroma.sqlite3 is a working format, not a readable one: metadata is
stored one row per field (536,080 rows for 67,010 chunks), the full-text index
adds five internal shadow tables, and the vectors are not in SQLite at all. At
238 MB most GUI viewers hang before they show you anything.

This writes .artifacts/browse.sqlite3 with two ordinary tables that open
instantly in any viewer:

    recipes   one row per recipe   (18,249)
    chunks    one row per chunk    (67,010)

It is a read-only convenience copy. Nothing reads it back; delete it freely.

    python scripts/export_browsable.py
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recipe_rag import config, store

OUTPUT = config.ARTIFACT_DIR / "browse.sqlite3"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-chunks", action="store_true",
                        help="export recipes only (smaller, faster)")
    args = parser.parse_args()

    store.require_index()
    parents = store.load_parents()

    if OUTPUT.exists():
        OUTPUT.unlink()
    con = sqlite3.connect(OUTPUT)

    con.execute("""
        CREATE TABLE recipes (
            parent_id TEXT PRIMARY KEY,
            title TEXT,
            diet TEXT,
            diet_confidence REAL,
            diet_conflict INTEGER,
            diet_reason TEXT,
            nutrition_status TEXT,
            calories REAL, protein REAL, fat REAL, sodium REAL,
            raw_calories REAL,
            rating REAL,
            n_ingredients INTEGER,
            n_steps INTEGER,
            n_chunks INTEGER,
            dup_group TEXT,
            dup_rank INTEGER,
            categories TEXT,
            ingredients TEXT,
            directions TEXT,
            description TEXT
        )
    """)
    con.executemany(
        "INSERT INTO recipes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            (
                r.parent_id, r.title, r.diet, r.diet_confidence,
                int(bool(r.diet_conflict)), r.diet_reason, r.nutrition_status,
                r.calories, r.protein, r.fat, r.sodium, r.raw_calories, r.rating,
                len(r.ingredients), len(r.directions), r.n_chunks,
                r.dup_group, r.dup_rank,
                " | ".join(r.categories),
                "\n".join(f"- {i}" for i in r.ingredients),
                "\n".join(f"{n}. {d}" for n, d in enumerate(r.directions, 1)),
                r.desc,
            )
            for r in parents.itertuples()
        ],
    )
    # Indexes on the columns you actually filter by while exploring.
    for column in ("title", "diet", "nutrition_status", "dup_group"):
        con.execute(f"CREATE INDEX idx_recipes_{column} ON recipes({column})")
    print(f"recipes : {len(parents):,} rows")

    if not args.no_chunks:
        collection = store.get_collection(store.get_client(), create=False)
        got = store.scan(collection, include=["documents", "metadatas"])
        con.execute("""
            CREATE TABLE chunks (
                chunk_id TEXT PRIMARY KEY,
                parent_id TEXT,
                title TEXT,
                section TEXT,
                part INTEGER,
                n_parts INTEGER,
                diet TEXT,
                n_chars INTEGER,
                text TEXT
            )
        """)
        title_of = dict(zip(parents.parent_id, parents.title))
        con.executemany(
            "INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?)",
            [
                (
                    cid, m["parent_id"], title_of.get(m["parent_id"], ""),
                    m["section"], m["part"], m["n_parts"], m["diet"],
                    len(doc), doc,
                )
                for cid, m, doc in zip(got["ids"], got["metadatas"], got["documents"])
            ],
        )
        for column in ("parent_id", "section", "diet"):
            con.execute(f"CREATE INDEX idx_chunks_{column} ON chunks({column})")
        print(f"chunks  : {len(got['ids']):,} rows")

    con.commit()
    con.execute("VACUUM")
    con.close()
    print(f"\nwrote {OUTPUT}  ({OUTPUT.stat().st_size / 1024 / 1024:.1f} MB)")
    print("Open this file in any SQLite viewer -- it is a copy, so nothing can")
    print("corrupt the real index. Delete it any time.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
