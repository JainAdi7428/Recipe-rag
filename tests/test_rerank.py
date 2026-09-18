"""Re-ranking: coverage, rarity weighting, constraints, tag candidates.

A six-recipe collection modelled on the grader's two hardest rows -- the fish
with olives and tomatoes, and the Easter coconut cake -- so each behaviour can
be checked in isolation and without the index.

Run standalone:  python tests/test_rerank.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from recipe_rag.query import QueryParser
from recipe_rag.rerank import Reranker

PARENTS = pd.DataFrame({
    "parent_id": ["mahi", "bass", "fillet", "choco", "easter", "tofu"],
    "title": [
        "Mahi-Mahi in Tomato Olive Sauce",
        "Roasted Black Sea Bass with Tomato and Olive Salad",
        "Fish Fillets With Tomatoes and Olives",
        "Coconut Cake with Chocolate Chunks",
        "Coconut-Key Lime Sheet Cake",
        "Tofu Scramble",
    ],
    "categories": [
        ["Fish", "Olive", "Tomato", "Healthy"],
        ["Olive", "Tomato", "Bass", "Healthy"],
        ["Fish", "Olive", "Tomato"],
        ["Cake", "Coconut", "Chocolate"],
        ["Cake", "Coconut", "Easter"],
        ["Vegan", "Breakfast"],
    ],
    "ingredients": [
        ["4 mahi-mahi fillets", "1/2 cup olives", "2 cans tomatoes"],
        ["2 lb sea bass", "olives", "tomatoes"],
        ["4 white fish fillets", "olives", "tomatoes"],
        ["1 cup shredded coconut", "6 oz chocolate", "3 eggs"],
        ["1 cup shredded coconut", "3 eggs", "key lime juice"],
        ["1 lb firm tofu", "turmeric"],
    ],
    "diet": ["non-veg", "non-veg", "non-veg", "eggetarian", "eggetarian", "vegan"],
    "nutrition_status": ["missing", "ok", "missing", "missing", "ok", "ok"],
    "dup_rank": [0] * 6,
})


@dataclass
class Hit:
    parent_id: str
    score: float


_TALLY = [0]


def check(condition: bool, message: str, failures: list[str]) -> None:
    _TALLY[0] += 1
    if not condition:
        failures.append(f"  {message}")


def order(result) -> list[str]:
    return [r.parent_id for r in result.ranked]


def run() -> int:
    failures: list[str] = []
    parser = QueryParser(PARENTS.categories)
    reranker = Reranker(PARENTS)

    # --- the fish row: vector alone picks the wrong recipe -----------------
    fish = parser.parse("I need a healthy fish recipe with olives and tomatoes")
    hits = [Hit("bass", 0.686), Hit("fillet", 0.667), Hit("mahi", 0.640)]
    plain = reranker.rank(fish, hits, weight=0.0)
    check(order(plain)[0] == "bass", f"weight 0 must reproduce vector order, got {order(plain)}", failures)
    boosted = reranker.rank(fish, hits, weight=0.3)
    check(order(boosted)[0] == "mahi",
          f"the recipe covering every term must win, got {order(boosted)}", failures)
    top = boosted.top
    check(top.coverage == 1.0, f"mahi covers all four terms, coverage {top.coverage}", failures)
    check(top.matched.get("fish") == "tag" and top.matched.get("olive") == "title",
          f"field attribution -> {top.matched}", failures)
    check([r.parent_id for r in plain.reordered(0.3)] == order(boosted),
          "reordered(w) must equal ranking at w", failures)

    # --- rarity: the Easter cake beats the chocolate one on one rare word ---
    cake = parser.parse("coconut cake for Easter")
    hits = [Hit("choco", 0.726), Hit("easter", 0.705)]
    check(order(reranker.rank(cake, hits, weight=0.3))[0] == "easter",
          "'easter' is the rarest term and must decide it", failures)
    check(reranker.idf("easter") > reranker.idf("coconut"),
          "a rarer term must weigh more", failures)

    # --- hard constraints remove, they do not merely demote ------------------
    nutrition = parser.parse("coconut cake with full nutrition info")
    result = reranker.rank(nutrition, [Hit("choco", 0.9), Hit("easter", 0.5)])
    check(order(result) == ["easter"], f"no-nutrition recipe must be removed, got {order(result)}", failures)
    check(result.removed.get("choco") == "no nutrition data", f"reason -> {result.removed}", failures)

    vegan = parser.parse("vegan breakfast")
    result = reranker.rank(vegan, [Hit("choco", 0.8), Hit("tofu", 0.4)])
    check(order(result) == ["tofu"], f"diet filter -> {order(result)}", failures)

    without = parser.parse("cake without chocolate")
    result = reranker.rank(without, [Hit("choco", 0.8), Hit("easter", 0.7)])
    check(order(result) == ["easter"], f"exclusion filter -> {order(result)}", failures)
    check("contains chocolate" in result.removed.values(), f"reason -> {result.removed}", failures)

    impossible = parser.parse("vegan coconut cake")
    result = reranker.rank(impossible, [Hit("choco", 0.8), Hit("easter", 0.7)])
    check(result.ranked == [] and result.filtered == {"diet is eggetarian": 2},
          f"nothing complies -> empty with reasons, got {result.filtered}", failures)

    # --- rare tags pull in what the vector search missed ---------------------
    missed = reranker.rank(cake, [Hit("choco", 0.726)])
    check("easter" in order(missed) and missed.added_by_tags == 1,
          f"tag intersection must add the Easter cake, got {order(missed)}", failures)
    found = reranker.rank(cake, [Hit("choco", 0.726)], vector_lookup=lambda ids: {"easter": 0.705})
    check(order(found)[0] == "easter", "with its real vector score it must win", failures)
    check(found.ranked[0].origin == "tags", "origin must record how it entered", failures)

    if failures:
        print(f"FAIL {len(failures)}/{_TALLY[0]} re-ranking cases")
        for line in failures:
            print(line)
        return 1
    print(f"ok  {_TALLY[0]}/{_TALLY[0]} re-ranking cases passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
