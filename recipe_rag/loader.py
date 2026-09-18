"""Load and normalise the Epicurious recipe JSON into clean records."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

_WHITESPACE = re.compile(r"\s+")


def clean_text(value: object) -> str:
    """Collapse whitespace and strip. Non-strings become ''."""
    if not isinstance(value, str):
        return ""
    return _WHITESPACE.sub(" ", value).strip()


def clean_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [text for text in (clean_text(item) for item in value) if text]


def as_float(value: object) -> float | None:
    """Coerce to a finite float, or None. Booleans are not numbers here."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    # NaN and infinities would poison both Parquet and Chroma metadata.
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def normalise_title(title: str) -> str:
    """Comparison key for titles: lowercased, whitespace-collapsed."""
    return clean_text(title).lower()


@dataclass
class Recipe:
    """One normalised recipe. Downstream stages fill the trailing fields."""

    source_index: int
    title: str
    ingredients: list[str]
    directions: list[str]
    desc: str = ""
    categories: list[str] = field(default_factory=list)
    calories: float | None = None
    protein: float | None = None
    fat: float | None = None
    sodium: float | None = None
    rating: float | None = None
    date: str = ""

    # set by dedup.py
    parent_id: str = ""
    dup_group: str = ""
    dup_rank: int = 0

    @property
    def title_norm(self) -> str:
        return normalise_title(self.title)

    @property
    def has_nutrition(self) -> bool:
        return any(
            value is not None
            for value in (self.calories, self.protein, self.fat, self.sodium)
        )

    def nutrition_fields_present(self) -> int:
        return sum(
            1
            for value in (self.calories, self.protein, self.fat, self.sodium)
            if value is not None
        )


@dataclass
class LoadStats:
    total_records: int = 0
    dropped_no_title: int = 0
    dropped_no_ingredients: int = 0
    dropped_no_directions: int = 0
    kept: int = 0

    @property
    def dropped(self) -> int:
        return (
            self.dropped_no_title
            + self.dropped_no_ingredients
            + self.dropped_no_directions
        )


def load_recipes(
    path: Path, limit: int | None = None
) -> tuple[list[Recipe], LoadStats]:
    """Read the source JSON, dropping records that cannot form a recipe.

    A record is unusable without a title, ingredients and directions -- there is
    nothing to retrieve or to answer with. ~87 of 20,130 records fail this.
    """
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)

    stats = LoadStats(total_records=len(raw))
    recipes: list[Recipe] = []

    for index, record in enumerate(raw):
        if not isinstance(record, dict):
            stats.dropped_no_title += 1
            continue

        # The raw titles carry trailing spaces, e.g. "Turkey Wrap ".
        title = clean_text(record.get("title"))
        if not title:
            stats.dropped_no_title += 1
            continue

        ingredients = clean_list(record.get("ingredients"))
        if not ingredients:
            stats.dropped_no_ingredients += 1
            continue

        directions = clean_list(record.get("directions"))
        if not directions:
            stats.dropped_no_directions += 1
            continue

        recipes.append(
            Recipe(
                source_index=index,
                title=title,
                ingredients=ingredients,
                directions=directions,
                desc=clean_text(record.get("desc")),
                categories=clean_list(record.get("categories")),
                calories=as_float(record.get("calories")),
                protein=as_float(record.get("protein")),
                fat=as_float(record.get("fat")),
                sodium=as_float(record.get("sodium")),
                rating=as_float(record.get("rating")),
                date=clean_text(record.get("date")),
            )
        )

        if limit is not None and len(recipes) >= limit:
            break

    stats.kept = len(recipes)
    return recipes, stats
