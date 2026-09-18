"""Two-stage duplicate handling.

The dataset has 2,314 rows sharing a title with another row. They split into two
very different problems, so they get two different treatments:

Stage 1 -- 1,794 of them are byte-identical recipes (same title, ingredients,
directions and categories). These are pure noise and are dropped outright.

Stage 2 -- the remaining 886 rows across 366 titles are genuinely different
recipes that happen to share a generic name ("chicken stock" x16, "pastry
dough" x15). Dropping them would lose real recipes, so they are all kept and
tagged with a `dup_group` plus a `dup_rank`; retrieval collapses a result set
to rank 0 per group so the top-k is not flooded by near-identical variants.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass

from .loader import Recipe


def content_hash(recipe: Recipe) -> str:
    """Stable content fingerprint. Also the source of `parent_id`."""
    payload = "|".join(
        (
            recipe.title_norm,
            "\x1f".join(item.lower() for item in recipe.ingredients),
            "\x1f".join(item.lower() for item in recipe.directions),
            "\x1f".join(sorted(item.lower() for item in recipe.categories)),
        )
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _variant_sort_key(recipe: Recipe) -> tuple:
    """Rank variants within a duplicate title group; best first.

    Completeness beats popularity: a recipe with full nutrition and a
    description is more useful to answer with than a slightly higher-rated one
    that is missing half its fields.
    """
    return (
        -recipe.nutrition_fields_present(),
        0 if recipe.desc else 1,
        -(recipe.rating if recipe.rating is not None else -1.0),
        -len(recipe.ingredients),
        recipe.source_index,
    )


@dataclass
class DedupStats:
    input_rows: int = 0
    exact_duplicates_removed: int = 0
    parents: int = 0
    variant_groups: int = 0
    rows_in_variant_groups: int = 0


def deduplicate(recipes: list[Recipe]) -> tuple[list[Recipe], DedupStats]:
    """Drop exact duplicates, then tag remaining same-title variants."""
    stats = DedupStats(input_rows=len(recipes))

    # Stage 1: keep the first occurrence of each distinct content hash.
    by_hash: dict[str, Recipe] = {}
    for recipe in recipes:
        digest = content_hash(recipe)
        if digest in by_hash:
            stats.exact_duplicates_removed += 1
            continue
        # Content-addressed id: stable across rebuilds, unlike a row counter.
        recipe.parent_id = digest[:16]
        by_hash[digest] = recipe

    survivors = list(by_hash.values())
    stats.parents = len(survivors)

    # Stage 2: rank the genuine variants that still share a title.
    by_title: dict[str, list[Recipe]] = defaultdict(list)
    for recipe in survivors:
        by_title[recipe.title_norm].append(recipe)

    for title_norm, group in by_title.items():
        if len(group) == 1:
            group[0].dup_group = ""
            group[0].dup_rank = 0
            continue

        stats.variant_groups += 1
        stats.rows_in_variant_groups += len(group)
        for rank, recipe in enumerate(sorted(group, key=_variant_sort_key)):
            recipe.dup_group = title_norm
            recipe.dup_rank = rank

    return survivors, stats
