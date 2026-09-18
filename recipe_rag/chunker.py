"""Turn one recipe (the parent) into several embeddable child chunks.

Every child's text begins with the recipe title. This is the single most
important detail in the whole module: the dominant query shape is a title
lookup ("Give me the recipe for Chicken Alfredo"), but a raw directions chunk
is procedural prose ("Combine first 9 ingredients in a heavy saucepan...") with
no title signal at all. Without the prefix, which chunk of a recipe wins for a
title query is close to arbitrary.

Sections rather than a fixed child count, because the sections are wildly
uneven: the ingredient block exceeds the cap for only 202 of 18,249 recipes,
while the directions block exceeds it for 36.8% of them.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import (
    CHUNK_CHAR_CAP,
    MAX_MODEL_TOKENS,
    OVERSIZE_STEP_OVERLAP,
    SECTION_DIRECTIONS,
    SECTION_INGREDIENTS,
    SECTION_SUMMARY,
)
from .loader import Recipe

# Worst-case width of " - Directions (part 10 of 10): " style suffixes, so the
# body budget is reserved before the real part count is known.
_PART_SUFFIX_RESERVE = len(" - Directions (part 99 of 99): ")
_MAX_CATEGORY_CHARS = 220


@dataclass
class Chunk:
    chunk_id: str
    parent_id: str
    section: str
    part: int
    n_parts: int
    text: str


def _hard_wrap(text: str, cap: int, overlap: int = OVERSIZE_STEP_OVERLAP) -> list[str]:
    """Split a single oversized item, preferring word boundaries.

    Only 315 of ~69,000 direction steps need this, but a step of 2,320
    characters would otherwise be silently truncated by the model.
    """
    parts: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + cap, length)
        if end < length:
            space = text.rfind(" ", start, end)
            if space > start + cap // 2:
                end = space
        piece = text[start:end].strip()
        if piece:
            parts.append(piece)
        if end >= length:
            break
        # max(..., start + 1) guarantees forward progress even if overlap >= cap.
        start = max(end - overlap, start + 1)
    return parts


def _pack(items: list[str], cap: int, sep: str = " ") -> list[str]:
    """Greedily pack items into groups no longer than `cap` characters."""
    parts: list[str] = []
    current: list[str] = []
    current_len = 0

    for item in items:
        if len(item) > cap:
            if current:
                parts.append(sep.join(current))
                current, current_len = [], 0
            parts.extend(_hard_wrap(item, cap))
            continue

        addition = len(item) + (len(sep) if current else 0)
        if current and current_len + addition > cap:
            parts.append(sep.join(current))
            current, current_len = [item], len(item)
        else:
            current.append(item)
            current_len += addition

    if current:
        parts.append(sep.join(current))
    return parts or [""]


def _truncate(text: str, budget: int) -> str:
    """Trim to `budget` characters *including* the ellipsis."""
    if budget <= 0:
        return ""
    if len(text) <= budget:
        return text
    keep = budget - 3
    if keep <= 0:
        return ""
    cut = text.rfind(" ", 0, keep)
    return text[: cut if cut > keep // 2 else keep].rstrip(" ,;") + "..."


def nutrition_line(nutrition) -> str:
    """Human-readable nutrition, omitting missing and rejected values.

    Takes a validated NutritionResult, never the raw record: implausible values
    must not reach the embedded text or the user.
    """
    pieces = []
    if nutrition.calories is not None:
        pieces.append(f"{nutrition.calories:g} calories")
    if nutrition.protein is not None:
        pieces.append(f"{nutrition.protein:g} g protein")
    if nutrition.fat is not None:
        pieces.append(f"{nutrition.fat:g} g fat")
    if nutrition.sodium is not None:
        pieces.append(f"{nutrition.sodium:g} mg sodium")
    return "Nutrition per serving: " + ", ".join(pieces) if pieces else ""


def _summary_text(recipe: Recipe, diet: str, cap: int, nutrition=None) -> str:
    """The primary title-lookup target.

    Diet and categories go into the embedded text (not just metadata) so that
    queries like "vegan banana bread" have something to match on.
    """
    head = f"{recipe.title}."
    tail = [f"Diet: {diet}"]
    nutrition_text = nutrition_line(nutrition) if nutrition is not None else ""
    if nutrition_text:
        tail.append(nutrition_text)

    fixed_len = len(head) + sum(len(line) + 1 for line in tail) + 1
    budget = cap - fixed_len

    category_line = ""
    if recipe.categories:
        category_line = "Categories: " + _truncate(
            ", ".join(recipe.categories), min(_MAX_CATEGORY_CHARS, max(budget, 0))
        )
        budget -= len(category_line) + 1

    body = [head]
    if recipe.desc and budget > 0:
        description = _truncate(recipe.desc, budget)
        if description:
            body.append(description)
    if category_line:
        body.append(category_line)
    body.extend(tail)
    text = "\n".join(body)
    # Belt and braces: the budget arithmetic above must never let the summary
    # past the cap, but a silent overrun here would mean silent truncation by
    # the model, which is exactly what this module exists to prevent.
    return text if len(text) <= cap else text[:cap]


def build_chunks(
    recipe: Recipe, diet: str, cap: int = CHUNK_CHAR_CAP, nutrition=None
) -> list[Chunk]:
    """Build every child chunk for one recipe, in stable order."""
    chunks: list[Chunk] = []

    def emit(section: str, texts: list[str]) -> None:
        total = len(texts)
        for index, text in enumerate(texts):
            chunks.append(
                Chunk(
                    chunk_id=f"{recipe.parent_id}:{section}:{index}",
                    parent_id=recipe.parent_id,
                    section=section,
                    part=index,
                    n_parts=total,
                    text=text,
                )
            )

    emit(SECTION_SUMMARY, [_summary_text(recipe, diet, cap, nutrition)])

    # The title prefix is charged against the cap, not added on top of it.
    body_cap = max(cap - len(recipe.title) - _PART_SUFFIX_RESERVE, 200)

    ingredient_parts = _pack(recipe.ingredients, body_cap, sep="; ")
    emit(
        SECTION_INGREDIENTS,
        [
            f"{recipe.title} - Ingredients"
            + (f" (part {i + 1} of {len(ingredient_parts)})" if len(ingredient_parts) > 1 else "")
            + f": {body}"
            for i, body in enumerate(ingredient_parts)
        ],
    )

    direction_parts = _pack(recipe.directions, body_cap, sep=" ")
    emit(
        SECTION_DIRECTIONS,
        [
            f"{recipe.title} - Directions"
            + (f" (part {i + 1} of {len(direction_parts)})" if len(direction_parts) > 1 else "")
            + f": {body}"
            for i, body in enumerate(direction_parts)
        ],
    )

    return chunks


# Characters per token vary a lot by section: flowing prose in directions runs
# near 4.5, but an ingredient line ("1 1/2 cups all-purpose flour, sifted") is
# closer to 3.2 because numbers, fractions and units all fragment. A single
# char cap therefore cannot guarantee the token cap. Rather than lower the cap
# for everything -- which would inflate the index for the ~0.2% of recipes that
# actually need it -- the offenders are rebuilt at a tighter cap.
_RETRY_FACTORS = (0.75, 0.55, 0.40)
_MIN_CAP = 120


def build_all(
    pairs: list[tuple],
    measure=None,
    cap: int = CHUNK_CHAR_CAP,
    token_cap: int = MAX_MODEL_TOKENS,
) -> list[list[Chunk]]:
    """Chunk many recipes, guaranteeing every chunk fits the model's token cap.

    `pairs` holds (recipe, diet_label) or (recipe, diet_label, nutrition).
    `measure` takes a list of strings and returns their token lengths. Without
    it this is plain character-capped chunking (used by the unit tests); with
    it, chunks are re-packed until nothing would be silently truncated.
    """

    def build(entry, chunk_cap: int) -> list[Chunk]:
        recipe, diet = entry[0], entry[1]
        nutrition = entry[2] if len(entry) > 2 else None
        return build_chunks(recipe, diet, chunk_cap, nutrition)

    per_recipe = [build(entry, cap) for entry in pairs]
    if measure is None:
        return per_recipe

    def over_cap_indices() -> set[int]:
        flat = [chunk.text for chunks in per_recipe for chunk in chunks]
        lengths = measure(flat)
        offenders: set[int] = set()
        position = 0
        for index, chunks in enumerate(per_recipe):
            for _ in chunks:
                if lengths[position] > token_cap:
                    offenders.add(index)
                position += 1
        return offenders

    for factor in _RETRY_FACTORS:
        offenders = over_cap_indices()
        if not offenders:
            return per_recipe
        tighter = max(int(cap * factor), _MIN_CAP)
        for index in offenders:
            per_recipe[index] = build(pairs[index], tighter)

    # Last resort for a pathological recipe: trim the text itself so the
    # invariant holds unconditionally.
    for index in over_cap_indices():
        for chunk in per_recipe[index]:
            while measure([chunk.text])[0] > token_cap and len(chunk.text) > 40:
                chunk.text = chunk.text[: int(len(chunk.text) * 0.9)].rstrip()
    return per_recipe
