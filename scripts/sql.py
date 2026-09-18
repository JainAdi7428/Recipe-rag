"""Run a SQL query against the browsable export and print it as a table.

    python scripts/sql.py "SELECT COUNT(*) FROM recipes"
    python scripts/sql.py --tables
    python scripts/sql.py --file myquery.sql

Opens read-only, so it can never modify or lock anything. Build the file it
reads with `python scripts/export_browsable.py` first.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recipe_rag import config

DATABASE = config.ARTIFACT_DIR / "browse.sqlite3"
_MAX_CELL = 60


def render(columns: list[str], rows: list[tuple]) -> str:
    if not rows:
        return "(no rows)"

    def cell(value) -> str:
        text = "" if value is None else str(value).replace("\n", " ")
        return text[: _MAX_CELL - 3] + "..." if len(text) > _MAX_CELL else text

    table = [list(columns)] + [[cell(v) for v in row] for row in rows]
    widths = [max(len(r[i]) for r in table) for i in range(len(columns))]
    line = "-+-".join("-" * w for w in widths)
    out = [" | ".join(h.ljust(w) for h, w in zip(table[0], widths)), line]
    out += [" | ".join(c.ljust(w) for c, w in zip(r, widths)) for r in table[1:]]
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", help="SQL to run")
    parser.add_argument("--file", help="read the SQL from a file instead")
    parser.add_argument("--tables", action="store_true",
                        help="list tables and their row counts")
    parser.add_argument("--schema", metavar="TABLE", help="show a table's columns")
    parser.add_argument("--limit", type=int, default=50,
                        help="max rows to print (default 50)")
    args = parser.parse_args()

    if not DATABASE.exists():
        raise SystemExit(
            f"\n{DATABASE} not found.\n"
            f"Build it first:  python scripts/export_browsable.py\n"
        )

    con = sqlite3.connect(f"file:{DATABASE}?mode=ro", uri=True)

    if args.tables:
        for (name,) in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ):
            count = con.execute(f"SELECT COUNT(*) FROM '{name}'").fetchone()[0]
            print(f"{count:>10,}  {name}")
        return 0

    if args.schema:
        rows = con.execute(f"PRAGMA table_info('{args.schema}')").fetchall()
        if not rows:
            raise SystemExit(f"no such table: {args.schema}")
        for row in rows:
            print(f"  {row[1]:<20} {row[2]}")
        return 0

    sql = Path(args.file).read_text(encoding="utf-8") if args.file else args.query
    if not sql:
        parser.print_help()
        return 1

    cursor = con.execute(sql)
    rows = cursor.fetchmany(args.limit)
    columns = [d[0] for d in cursor.description] if cursor.description else []
    print(render(columns, rows))
    if len(rows) == args.limit:
        print(f"\n(stopped at {args.limit} rows; raise it with --limit)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
