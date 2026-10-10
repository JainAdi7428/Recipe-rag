"""Web search filtering and the wrong-dish gate. No network, no index.

The gate is the piece that keeps the web path honest. Serper always returns
something: ask for a dish that does not exist and the top results are a
different dish entirely, carrying perfectly valid recipe markup. These cases
pin down that an unknown word in the request has to be found on the page.

Run standalone:  python tests/test_web.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from recipe_rag.pipeline import WEB_FUZZY_MATCH, WEB_MIN_COVERAGE
from recipe_rag.query import QueryParser
from recipe_rag.rerank import Reranker, fields_from
from recipe_rag.websearch import SearchResult, build_query, page_encoding, rank_results

PARENTS = pd.DataFrame({
    "parent_id": ["t1", "t2", "t3", "tikka", "paneer", "span"],
    "title": [
        "Tandoori Chicken",
        "Tandoori Shrimp and Mango Salad",
        "Grilled Tandoori Lamb",
        "Chicken Tikka Masala",
        "Fresh Indian Cheese (Paneer)",
        "Spinach and Feta Phyllo Triangles",
    ],
    "categories": [
        ["Chicken", "Grill/Barbecue"], ["Shrimp", "Salad"], ["Lamb", "Grill/Barbecue"],
        ["Chicken", "Curry"], ["Cheese", "Milk/Cream"], ["Cheese", "Appetizer"],
    ],
    "ingredients": [
        ["1 chicken", "1 cup yogurt", "2 tbsp tandoori masala"],
        ["1 lb shrimp", "1 mango"],
        ["2 lb lamb", "yogurt"],
        ["chicken thighs", "yogurt", "garam masala"],
        ["2 quarts whole milk", "lemon juice"],
        ["spinach", "feta", "phyllo"],
    ],
    "diet": ["non-veg", "non-veg", "non-veg", "non-veg", "veg", "veg"],
    "nutrition_status": ["ok"] * 6,
    "dup_rank": [0] * 6,
})

_TALLY = [0]


def check(condition: bool, message: str, failures: list[str]) -> None:
    _TALLY[0] += 1
    if not condition:
        failures.append(f"  {message}")


def covered(reranker, parsed, title, categories, ingredients, unknown_counts=True,
            fuzzy=WEB_FUZZY_MATCH):
    score, matched = reranker.coverage_of(
        parsed, *fields_from(title, categories, ingredients),
        unknown_counts=unknown_counts, fuzzy=fuzzy,
    )
    return score, matched


def run() -> int:
    failures: list[str] = []
    parser = QueryParser(PARENTS.categories)
    reranker = Reranker(PARENTS)

    # --- search query building ---------------------------------------------
    check(build_query("Palak Paneer") == "Palak Paneer recipe ingredients instructions",
          f"build_query -> {build_query('Palak Paneer')!r}", failures)
    check(build_query("chicken curry recipe").count("recipe") == 1,
          "the word 'recipe' must not be added twice", failures)

    # --- result filtering and ordering --------------------------------------
    organic = [
        {"link": "https://www.youtube.com/watch?v=1", "title": "video"},
        {"link": "https://someblog.example/dish", "title": "a blog"},
        {"link": "https://www.allrecipes.com/recipe/1", "title": "allrecipes"},
        {"link": "", "title": "no link"},
        {"link": "https://www.pinterest.com/pin/2", "title": "pin"},
    ]
    ranked = rank_results(organic)
    urls = [r.domain for r in ranked]
    check("youtube.com" not in urls and "pinterest.com" not in urls,
          f"aggregators must be dropped, got {urls}", failures)
    check(urls[0] == "allrecipes.com",
          f"a known-good publisher must be tried first, got {urls}", failures)
    check("someblog.example" in urls,
          "a site not on the list is still allowed -- markup decides", failures)
    check(SearchResult(title="", url="https://www.food.com/x").domain == "food.com",
          "www. must be stripped from the domain", failures)
    check(rank_results([]) == [] and rank_results(None) == [],
          "no results must not raise", failures)

    # --- character set of a fetched page -----------------------------------
    # requests defaults text/html with no declared charset to ISO-8859-1, which
    # the web stopped honouring long ago; the document usually knows better.
    body = b"<html><head><meta charset=utf-8></head><body>x</body></html>"
    check(page_encoding({"Content-Type": "text/html; charset=UTF-8"}, body) == "utf-8",
          "an explicit header charset wins", failures)
    check(page_encoding({"Content-Type": "text/html"}, body) == "utf-8",
          "with no header charset, the document's meta tag decides", failures)
    check(page_encoding({}, b"<html><head><meta charset='windows-1252'>") == "windows-1252",
          "a quoted meta charset must be read too", failures)
    check(page_encoding({"Content-Type": "text/html"}, b"<html>no declaration") == "utf-8",
          "with nothing declared, assume UTF-8 rather than ISO-8859-1", failures)

    # --- the wrong-dish gate ------------------------------------------------
    # "Tandoori Jilaji" does not exist. Google returns Tandoori Chicken.
    jilaji = parser.parse("I need a recipe for Tandoori Jilaji?")
    check(set(jilaji.terms) == {"tandoori", "jilaji"}, f"terms -> {jilaji.terms}", failures)
    score, matched = covered(reranker, jilaji, "Tandoori Chicken",
                             ["Chicken", "Indian"], ["chicken", "yogurt", "tandoori masala"])
    check(score < WEB_MIN_COVERAGE,
          f"a page missing the dish's own name must be refused, scored {score:.2f}", failures)
    check("jilaji" not in matched, "the unknown word is what is missing", failures)

    # The same page scores perfectly when unknown words are ignored -- which is
    # why judging a web page uses the opposite rule to ranking dataset rows.
    dataset_score, _ = covered(reranker, jilaji, "Tandoori Chicken",
                               ["Chicken"], ["chicken"], unknown_counts=False)
    check(dataset_score == 1.0,
          f"dataset-mode scoring ignores unknown words, got {dataset_score}", failures)

    # A page that really is the dish must pass.
    palak = parser.parse("I want to make authentic Palak Paneer like in Indian restaurants")
    score, _ = covered(reranker, palak, "Palak Paneer (Indian Spinach and Cheese Curry)",
                       ["Indian", "Curry"], ["spinach", "paneer cheese", "ghee"])
    check(score >= WEB_MIN_COVERAGE, f"the right dish must pass, scored {score:.2f}", failures)

    # A near-miss from the same cuisine must not.
    score, _ = covered(reranker, palak, "Saag Aloo (Indian Spinach and Potato)",
                       ["Indian"], ["spinach", "potato", "cumin"])
    check(score < WEB_MIN_COVERAGE,
          f"a different Indian spinach dish must be refused, scored {score:.2f}", failures)

    # --- near-spellings (fuzzy, cutoff WEB_FUZZY_MATCH) -----------------------
    # Transliterated names have no one spelling; the page is the same dish.
    dal_page = ("Dal Makhani Recipe (Restaurant Style)", ["Indian"],
                ["urad dal", "rajma", "butter", "cream"])
    for request in ("dal makhni", "dalmakhani", "dal Makhani"):
        score, matched = covered(reranker, parser.parse(request), *dal_page)
        check(score >= WEB_MIN_COVERAGE,
              f"{request!r} must accept a Dal Makhani page, scored {score:.2f} {matched}",
              failures)
    score, matched = covered(reranker, parser.parse("dal makhni"), *dal_page)
    check(matched.get("makhni") == "title~makhani",
          f"a fuzzy hit must say what it matched, got {matched}", failures)

    # Below the cutoff on purpose: one letter in a short word is too much.
    score, matched = covered(reranker, parser.parse("daal makhani"), *dal_page)
    check("daal" not in matched,
          f"'daal' vs 'dal' scores 86 and must stay unmatched at 90, got {matched}",
          failures)
    score, matched = covered(reranker, parser.parse("pea soup"), "Pear Soup", [], ["pear"])
    check("pea" not in matched, f"'pea' must not match 'pear', got {matched}", failures)

    # The guards the coverage gate exists for still hold with fuzzy on.
    score, _ = covered(reranker, jilaji, "Tandoori Chicken",
                       ["Chicken", "Indian"], ["chicken", "yogurt", "tandoori masala"])
    check(score < WEB_MIN_COVERAGE,
          f"fuzzy must not let Tandoori Chicken answer Jilaji, scored {score:.2f}", failures)
    # (Whether Moon Pies passes on exact words depends on corpus rarity weights,
    # which this six-recipe fixture cannot reproduce; what fuzzy must not do is
    # find "rocks" anywhere or raise the score.)
    rocks = parser.parse("Deep Fried Moon Rocks")
    moon_pies = ("Deep Fried Moon Pies", [], ["moon pies", "oil"])
    score, matched = covered(reranker, rocks, *moon_pies)
    exact, _ = covered(reranker, rocks, *moon_pies, fuzzy=None)
    check("rock" not in matched and score == exact,
          f"fuzzy must not match 'rocks' to Moon Pies, got {matched}", failures)

    # Without the fuzzy argument, behaviour is exactly as before.
    score, _ = covered(reranker, parser.parse("dal makhni"), *dal_page, fuzzy=None)
    check(score < WEB_MIN_COVERAGE,
          f"exact-only scoring must still refuse 'makhni', scored {score:.2f}", failures)

    # A request with no distinctive words cannot be judged this way, and must
    # not be blocked by it.
    vague = parser.parse("give me something to cook")
    score, _ = covered(reranker, vague, "Tandoori Chicken", [], ["chicken"])
    check(score == 0.0 and not vague.terms,
          f"a request with no terms yields no evidence, got {vague.terms}", failures)

    if failures:
        print(f"FAIL {len(failures)}/{_TALLY[0]} web search and gate cases")
        for line in failures:
            print(line)
        return 1
    print(f"ok  {_TALLY[0]}/{_TALLY[0]} web search and gate cases passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
