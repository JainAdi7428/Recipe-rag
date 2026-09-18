"""schema.org Recipe extraction from HTML. No network, no index.

The fixtures below are trimmed versions of markup shapes that actually appear on
mainstream recipe sites. Each one exists because it broke something, or would
have: the four instruction encodings, sodium in grams rather than milligrams,
@graph nesting, and a page carrying Recipe markup that is not a recipe.

Run standalone:  python tests/test_extract.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recipe_rag.extract import from_html, _quantity


def page(payload) -> str:
    return (
        "<html><head><title>x</title>"
        '<script type="application/ld+json">'
        + json.dumps(payload)
        + "</script></head><body>ignored</body></html>"
    )


BASE = {
    "@context": "https://schema.org",
    "@type": "Recipe",
    "name": "Palak Paneer",
    "recipeIngredient": [
        "500 g spinach", "200 g paneer", "2 tbsp butter", "1 tsp cumin seeds",
    ],
    "recipeInstructions": "Blanch the spinach.\nBlend to a puree.\nFry the paneer.",
    "recipeCategory": "Main Course",
    "recipeCuisine": "Indian",
    "keywords": "vegetarian, north indian, curry",
    "author": {"@type": "Person", "name": "A Cook"},
    "recipeYield": "4 servings",
    "nutrition": {
        "@type": "NutritionInformation",
        "calories": "320 kcal",
        "proteinContent": "14 g",
        "fatContent": "22 g",
        "sodiumContent": "480 mg",
    },
}


_TALLY = [0]


def check(condition: bool, message: str, failures: list[str]) -> None:
    _TALLY[0] += 1
    if not condition:
        failures.append(f"  {message}")


def run() -> int:
    failures: list[str] = []

    # --- the ordinary case ------------------------------------------------
    recipe = from_html(page(BASE), "https://example.com/palak-paneer")
    check(recipe is not None, "a well-formed Recipe page must extract", failures)
    if recipe:
        check(recipe.title == "Palak Paneer", f"title -> {recipe.title!r}", failures)
        check(len(recipe.ingredients) == 4, f"4 ingredients, got {len(recipe.ingredients)}", failures)
        check(len(recipe.directions) == 3,
              f"newline-split instructions -> {len(recipe.directions)} steps", failures)
        check(recipe.calories == 320, f"calories -> {recipe.calories}", failures)
        check(recipe.sodium == 480, f"sodium in mg -> {recipe.sodium}", failures)
        check(recipe.domain == "example.com", f"domain -> {recipe.domain}", failures)
        check(recipe.author == "A Cook", f"author -> {recipe.author}", failures)
        # Butter and paneer are dairy, no meat/egg -> veg, via the SAME
        # classifier the dataset used. This is the cross-source consistency
        # requirement, tested rather than asserted.
        check(recipe.diet == "veg", f"butter + paneer -> veg, got {recipe.diet!r}", failures)

    # --- the four instruction encodings -----------------------------------
    shapes = {
        "list of strings": ["Step one here.", "Step two here."],
        "HowToStep": [
            {"@type": "HowToStep", "text": "Step one here."},
            {"@type": "HowToStep", "text": "Step two here."},
        ],
        "HowToSection": [
            {"@type": "HowToSection", "name": "Prep", "itemListElement": [
                {"@type": "HowToStep", "text": "Step one here."},
                {"@type": "HowToStep", "text": "Step two here."},
            ]},
        ],
        "single string": "Step one here.\nStep two here.",
    }
    for label, value in shapes.items():
        node = dict(BASE, recipeInstructions=value)
        got = from_html(page(node), "https://example.com/x")
        check(got is not None and len(got.directions) == 2,
              f"{label}: expected 2 steps, got "
              f"{len(got.directions) if got else 'no recipe'}", failures)

    # --- sodium published in grams must not read as milligrams ------------
    # A 1000x error in the field most likely to be extreme already.
    node = dict(BASE, nutrition=dict(BASE["nutrition"], sodiumContent="1.2 g"))
    got = from_html(page(node), "https://example.com/x")
    check(got is not None and got.sodium == 1200,
          f"1.2 g sodium -> 1200 mg, got {got.sodium if got else None}", failures)

    # --- @graph nesting, the commonest real-world layout -------------------
    graph = {
        "@context": "https://schema.org",
        "@graph": [
            {"@type": "WebSite", "name": "A Blog"},
            {"@type": "BreadcrumbList"},
            BASE,
        ],
    }
    got = from_html(page(graph), "https://example.com/x")
    check(got is not None and got.title == "Palak Paneer",
          "Recipe nested in @graph must still be found", failures)

    # --- @type as a list ---------------------------------------------------
    node = dict(BASE, **{"@type": ["Recipe", "NewsArticle"]})
    got = from_html(page(node), "https://example.com/x")
    check(got is not None, "@type as a list must still match Recipe", failures)

    # --- pages that must be REFUSED ---------------------------------------
    check(from_html("<html><body>no markup at all</body></html>", "u") is None,
          "a page with no JSON-LD must extract nothing, not guess", failures)
    check(from_html(page({"@type": "Article", "name": "About paneer"}), "u") is None,
          "an Article must not be treated as a Recipe", failures)
    stub = dict(BASE, recipeIngredient=["salt"], recipeInstructions=[])
    check(from_html(page(stub), "u") is None,
          "a markup stub with no real content must be rejected", failures)
    check(from_html('<script type="application/ld+json">{broken</script>', "u") is None,
          "unparseable JSON-LD must not raise", failures)

    # --- HTML entities and tags inside markup ------------------------------
    node = dict(BASE, name="Ma&#239;s &amp; Beans",
                recipeInstructions=["<p>Stir <b>well</b>.</p>", "Serve hot."])
    got = from_html(page(node), "u")
    check(got is not None and "&" in got.title and "&amp;" not in got.title,
          f"entities must be unescaped, got {got.title if got else None!r}", failures)
    check(got is not None and "<b>" not in got.directions[0],
          "tags inside instruction text must be stripped", failures)

    # --- quantity parsing --------------------------------------------------
    for value, kwargs, expected in [
        ("553 calories", {}, 553.0),
        ("48 g", {}, 48.0),
        (38, {}, 38.0),
        ("1,5 g", {"to_mg": True}, 1500.0),
        ("97225 mg", {"to_mg": True}, 97225.0),
        ("no number here", {}, None),
        (None, {}, None),
    ]:
        got = _quantity(value, **kwargs)
        check(got == expected, f"_quantity({value!r}) -> {got}, want {expected}", failures)

    if failures:
        print(f"FAIL {len(failures)}/{_TALLY[0]} schema.org extraction cases")
        for line in failures:
            print(line)
        return 1
    print(f"ok  {_TALLY[0]}/{_TALLY[0]} schema.org extraction cases passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
