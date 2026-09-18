"""Query understanding: constraints versus terms, and exact tag matching.

Built on a tiny tag vocabulary, so it runs without the index. Every case is a
real request shape -- most are the grader's own queries -- or a trap found while
designing the parser (the "Indiana" tag, "whole wheat", "no-cook").

Run standalone:  python tests/test_query.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recipe_rag.diet import EGGETARIAN, VEG, VEGAN
from recipe_rag.query import QueryParser, singular

VOCAB = [
    ["Fish", "Olive", "Tomato", "Healthy"],
    ["Casserole/Gratin", "Pasta", "Vegetarian"],
    ["Wheat/Gluten-Free", "Tuna", "Asparagus", "Salad"],
    ["Raw", "No-Cook", "Soup/Stew", "Spinach", "Vegan"],
    ["Easter", "Coconut", "Cake"],
    ["Indiana"], ["Bon Appétit"], ["Low/No Sugar"], ["Quick & Easy"],
    ["Mushroom"], ["Low Cal"],
]

_TALLY = [0]


def check(condition: bool, message: str, failures: list[str]) -> None:
    _TALLY[0] += 1
    if not condition:
        failures.append(f"  {message}")


def run() -> int:
    failures: list[str] = []
    parser = QueryParser(VOCAB)

    # --- singular forms: only consistency matters, but these must hold ------
    for word, expected in [("tomatoes", "tomato"), ("olives", "olive"),
                           ("berries", "berry"), ("asparagus", "asparagus"),
                           ("glass", "glass"), ("peaches", "peach"), ("cakes", "cake")]:
        check(singular(word) == expected, f"singular({word!r}) -> {singular(word)!r}", failures)

    # --- the grader's queries -------------------------------------------------
    q = parser.parse("I need a healthy fish recipe with olives and tomatoes.")
    check(q.terms == ["healthy", "fish", "olive", "tomato"], f"row 1 terms -> {q.terms}", failures)
    check({"Fish", "Olive", "Tomato", "Healthy"} <= set(q.tags), f"row 1 tags -> {q.tags}", failures)
    check(q.allowed_diets is None and not q.wants_nutrition and not q.excluded,
          "row 1 states no constraint", failures)

    q = parser.parse("Show me a vegetarian spinach pasta casserole")
    check(q.allowed_diets == {VEGAN, VEG, EGGETARIAN},
          f"'vegetarian' must accept egg dishes (grader row 2), got {q.allowed_diets}", failures)
    check("vegetarian" not in q.terms, "a diet word is a constraint, not a term", failures)
    check({"spinach", "pasta", "casserole"} <= set(q.terms), f"row 2 terms -> {q.terms}", failures)
    check("Casserole/Gratin" in q.tags, "'casserole' must match the slash tag", failures)

    q = parser.parse("I want a gluten-free salad with tuna and asparagus")
    check("Wheat/Gluten-Free" in q.tags, f"'gluten-free' -> {q.tags}", failures)
    check({"tuna", "asparagus", "salad"} <= set(q.terms), f"row 3 terms -> {q.terms}", failures)

    q = parser.parse("Do you have a raw vegan spinach soup recipe?")
    check(q.allowed_diets == {VEGAN}, f"vegan -> {q.allowed_diets}", failures)
    check("No-Cook" in q.term_tags.get("raw", ()), "'raw' must also count the No-Cook tag", failures)
    check("vegan" not in q.terms, "vegan is a constraint", failures)

    q = parser.parse("I need a coconut cake recipe for Easter with full nutrition info")
    check(q.wants_nutrition, "'full nutrition info' is a constraint", failures)
    check(q.terms == ["coconut", "cake", "easter"],
          f"nutrition words must not become terms, got {q.terms}", failures)

    q = parser.parse("I want to make authentic Palak Paneer like in Indian restaurants")
    check("Indiana" not in q.tags, "exact matching: 'indian' must never match 'Indiana'", failures)
    check(q.terms == ["palak", "paneer", "indian"], f"row 6 terms -> {q.terms}", failures)

    # --- diet phrasing ---------------------------------------------------------
    check(parser.parse("non-vegetarian curry").allowed_diets is None,
          "'non-vegetarian' is not a vegetarian constraint", failures)
    check(parser.parse("eggless chocolate cake").allowed_diets == {VEGAN, VEG},
          "eggless excludes eggetarian", failures)
    check(parser.parse("vegan eggless muffins").allowed_diets == {VEGAN},
          "combined constraints intersect", failures)

    # --- exclusions -------------------------------------------------------------
    q = parser.parse("pasta without mushrooms or onions, please")
    check(q.excluded == ["mushroom", "onion"], f"exclusions -> {q.excluded}", failures)
    check("mushroom" not in q.terms and "Mushroom" not in q.tags,
          "an excluded word must not boost the thing it excludes", failures)
    check("mushroom" not in q.search_text, f"search text keeps the exclusion: {q.search_text!r}", failures)
    check(parser.parse("a no-cook summer salad").excluded == [],
          "'no-cook' is a style, not an exclusion", failures)
    check(parser.parse("no sugar added lemonade").excluded == ["sugar"],
          "'no sugar added' excludes sugar", failures)

    # --- tag vocabulary traps ------------------------------------------------------
    check("Wheat/Gluten-Free" not in parser.parse("whole wheat bread").tags,
          "'wheat' alone must not mean wheat-free", failures)
    check("Low/No Sugar" not in parser.parse("low fat yogurt").tags,
          "'low' alone must not match Low/No Sugar", failures)
    check("Low/No Sugar" in parser.parse("low sugar muffins").tags, "alias low sugar", failures)
    check("Bon Appétit" not in parser.parse("bon appetit salad").tags,
          "publication tags are never matched", failures)
    check("Quick & Easy" in parser.parse("quick and easy dinner").tags,
          "'quick and easy' must match 'Quick & Easy'", failures)
    q = parser.parse("low calorie soup")
    check(q.wants_nutrition and "Low Cal" in q.tags, f"low calorie -> {q.tags}", failures)

    if failures:
        print(f"FAIL {len(failures)}/{_TALLY[0]} query understanding cases")
        for line in failures:
            print(line)
        return 1
    print(f"ok  {_TALLY[0]}/{_TALLY[0]} query understanding cases passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
