"""Chunker invariants: title prefixing, size caps, and no content loss.

Run standalone:  python tests/test_chunker.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recipe_rag.chunker import build_all, build_chunks
from recipe_rag.config import (
    CHUNK_CHAR_CAP,
    SECTION_DIRECTIONS,
    SECTION_INGREDIENTS,
    SECTION_SUMMARY,
)
from recipe_rag.loader import Recipe
from recipe_rag.nutrition import validate

TITLE = "Chicken Alfredo"


def make(ingredients=None, directions=None, **kwargs) -> Recipe:
    recipe = Recipe(
        source_index=0,
        title=kwargs.pop("title", TITLE),
        ingredients=ingredients or ["1 lb fettuccine", "2 cups heavy cream"],
        directions=directions or ["Boil the pasta.", "Add the sauce."],
        **kwargs,
    )
    recipe.parent_id = "deadbeefdeadbeef"
    return recipe


def check(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(f"  {message}")


def run() -> int:
    failures: list[str] = []

    # --- every chunk carries the title -----------------------------------
    chunks = build_chunks(make(), "non-veg")
    check(
        all(TITLE in chunk.text for chunk in chunks),
        "every chunk must contain the recipe title",
        failures,
    )
    sections = [chunk.section for chunk in chunks]
    check(sections[0] == SECTION_SUMMARY, "first chunk must be the summary", failures)
    check(SECTION_INGREDIENTS in sections, "ingredients section missing", failures)
    check(SECTION_DIRECTIONS in sections, "directions section missing", failures)

    # --- the summary carries diet and nutrition into the embedded text ----
    recipe = make(calories=620.0, protein=28.0, categories=["Pasta", "Dinner"])
    good = validate({"calories": 620.0, "protein": 28.0, "fat": 41.0, "sodium": 800.0})
    summary = build_chunks(recipe, "non-veg", nutrition=good)[0].text
    check("Diet: non-veg" in summary, "summary must state the diet", failures)
    check("620 calories" in summary, "summary must state nutrition", failures)
    check("Pasta" in summary, "summary must list categories", failures)

    # --- nutrition is omitted, not faked, when absent ---------------------
    bare = build_chunks(make(), "veg")[0].text
    check("Nutrition" not in bare, "absent nutrition must not appear", failures)

    # --- rejected nutrition must never reach the embedded text -----------
    corrupt = validate(
        {"calories": 30111218.0, "protein": 200968.0, "fat": 5000.0, "sodium": 27675110.0}
    )
    poisoned = build_chunks(
        make(calories=30111218.0), "veg", nutrition=corrupt
    )[0].text
    check(
        "Nutrition" not in poisoned and "30111218" not in poisoned,
        "implausible nutrition must not appear in a chunk",
        failures,
    )

    # --- character cap holds across pathological inputs -------------------
    long_ingredients = [f"{i} cups of ingredient number {i} finely chopped" for i in range(60)]
    long_directions = ["Stir constantly and watch carefully. " * 40] * 5
    stress = build_chunks(
        make(
            ingredients=long_ingredients,
            directions=long_directions,
            desc="A very long description. " * 60,
            categories=[f"Category{i}" for i in range(80)],
        ),
        "vegan",
    )
    worst = max(len(chunk.text) for chunk in stress)
    check(worst <= CHUNK_CHAR_CAP, f"cap exceeded: {worst} > {CHUNK_CHAR_CAP}", failures)
    check(
        all(TITLE in chunk.text for chunk in stress),
        "title prefix must survive splitting",
        failures,
    )

    # --- multi-part chunks are labelled and numbered ----------------------
    parts = [c for c in stress if c.section == SECTION_DIRECTIONS]
    check(len(parts) > 1, "long directions should split into parts", failures)
    check(
        all(c.n_parts == len(parts) for c in parts),
        "n_parts must match the actual part count",
        failures,
    )
    check(
        [c.part for c in parts] == list(range(len(parts))),
        "part indices must be contiguous from 0",
        failures,
    )
    check(
        len({c.chunk_id for c in stress}) == len(stress),
        "chunk ids must be unique",
        failures,
    )

    # --- no ingredient is dropped by packing ------------------------------
    ingredient_text = " ".join(
        c.text for c in stress if c.section == SECTION_INGREDIENTS
    )
    missing = [ing for ing in long_ingredients if ing not in ingredient_text]
    check(not missing, f"packing dropped {len(missing)} ingredients", failures)

    # --- token cap enforcement re-packs offenders -------------------------
    # A fake measure that calls anything over 300 chars "too many tokens"
    # should drive build_all to produce only short chunks.
    def fake_measure(texts: list[str]) -> list[int]:
        return [len(t) // 2 for t in texts]  # 2 chars per "token"

    rebuilt = build_all(
        [(make(ingredients=long_ingredients, directions=long_directions), "vegan")],
        measure=fake_measure,
        token_cap=200,  # => 400 chars
    )[0]
    worst_tokens = max(fake_measure([c.text for c in rebuilt]))
    check(
        worst_tokens <= 200,
        f"build_all must honour the token cap, got {worst_tokens}",
        failures,
    )

    # --- deterministic ----------------------------------------------------
    a = [c.text for c in build_chunks(make(), "veg")]
    b = [c.text for c in build_chunks(make(), "veg")]
    check(a == b, "chunking must be deterministic", failures)

    total = 17
    if failures:
        print(f"FAILED {len(failures)}/{total}")
        print("\n".join(failures))
        return 1
    print(f"ok  {total}/{total} chunker invariants held")
    return 0


def test_chunker_invariants():
    assert run() == 0


if __name__ == "__main__":
    raise SystemExit(run())
