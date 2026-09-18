"""Dedup invariants: exact duplicates die, genuine variants survive ranked.

Run standalone:  python tests/test_dedup.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recipe_rag.dedup import content_hash, deduplicate
from recipe_rag.loader import Recipe


def make(index: int, title: str, ingredients=None, directions=None, **kwargs) -> Recipe:
    return Recipe(
        source_index=index,
        title=title,
        ingredients=ingredients or ["flour", "water"],
        directions=directions or ["Mix.", "Bake."],
        **kwargs,
    )


def check(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(f"  {message}")


def run() -> int:
    failures: list[str] = []

    # --- exact duplicates collapse to one -------------------------------
    rows = [make(0, "Pastry Dough"), make(1, "Pastry Dough"), make(2, "Pastry Dough")]
    kept, stats = deduplicate(rows)
    check(len(kept) == 1, f"exact duplicates should collapse, kept {len(kept)}", failures)
    check(stats.exact_duplicates_removed == 2, "should report 2 removed", failures)
    check(kept[0].dup_group == "", "a collapsed single has no variant group", failures)

    # --- title differing only by case and trailing space is the same -----
    kept, _ = deduplicate([make(0, "Pastry Dough"), make(1, "pastry dough ")])
    check(len(kept) == 1, "title normalisation should collapse these", failures)

    # --- genuinely different recipes sharing a title all survive ---------
    variants = [
        make(0, "Chicken Stock", ingredients=["chicken", "carrot"]),
        make(1, "Chicken Stock", ingredients=["chicken", "leek", "thyme"]),
        make(2, "Chicken Stock", ingredients=["chicken wings", "onion"]),
    ]
    kept, stats = deduplicate(variants)
    check(len(kept) == 3, f"variants must survive, kept {len(kept)}", failures)
    check(stats.variant_groups == 1, "should be one variant group", failures)
    check(stats.rows_in_variant_groups == 3, "group should cover 3 rows", failures)
    check(
        {r.dup_group for r in kept} == {"chicken stock"},
        "all variants share the normalised title as dup_group",
        failures,
    )
    check(
        sorted(r.dup_rank for r in kept) == [0, 1, 2],
        "ranks must be contiguous from 0",
        failures,
    )

    # --- completeness beats rating when ranking variants -----------------
    rich = make(0, "Soup", ingredients=["a", "b", "c"], calories=100.0, protein=5.0,
                fat=2.0, sodium=50.0, desc="A good soup.", rating=3.0)
    thin = make(1, "Soup", ingredients=["x"], rating=5.0)
    kept, _ = deduplicate([thin, rich])
    best = min(kept, key=lambda r: r.dup_rank)
    check(
        best.desc == "A good soup.",
        "the more complete variant should rank first despite a lower rating",
        failures,
    )

    # --- parent ids are content-addressed and stable ---------------------
    a, _ = deduplicate([make(0, "Bread")])
    b, _ = deduplicate([make(99, "Bread")])
    check(
        a[0].parent_id == b[0].parent_id,
        "parent_id must depend on content, not row position",
        failures,
    )
    check(len(a[0].parent_id) == 16, "parent_id should be 16 hex chars", failures)

    # --- differing categories alone make a distinct recipe ---------------
    one = make(0, "Salad", categories=["Summer"])
    two = make(1, "Salad", categories=["Winter"])
    check(
        content_hash(one) != content_hash(two),
        "categories must participate in the content hash",
        failures,
    )

    total = 13
    if failures:
        print(f"FAILED {len(failures)}/{total}")
        print("\n".join(failures))
        return 1
    print(f"ok  {total}/{total} dedup invariants held")
    return 0


def test_dedup_invariants():
    assert run() == 0


if __name__ == "__main__":
    raise SystemExit(run())
