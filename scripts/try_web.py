"""Exercise one piece of the web path at a time.

    python scripts/try_web.py --search "Palak Paneer"       # Serper only: which URLs come back
    python scripts/try_web.py --url https://site/recipe      # one page: fetch + extract
    python scripts/try_web.py --ask "Palak Paneer recipe"    # the whole pipeline, web enabled

Built for working bottom-up, so a failure names its own layer: a search that
returns nothing is a different problem from a page that blocks us, which is
different again from a page carrying no recipe markup.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Recipe titles arrive in any script: Korean, Japanese, accented French. The
# Windows console defaults to cp1252 and raises rather than printing them.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


from recipe_rag import extract, websearch


def show_search(query: str, n: int) -> int:
    outcome = websearch.search(query, n=n)
    print(f"sent to Serper: {outcome.query!r}")
    print(f"{len(outcome.results)} results in {outcome.seconds:.1f}s "
          f"(after dropping aggregators)")
    for result in outcome.results:
        flag = "preferred" if result.preferred else "         "
        print(f"  {result.rank:>2}. [{flag}] {result.domain}")
        print(f"      {result.title[:88]}")
        print(f"      {result.url[:100]}")
    return 0


def show_url(url: str) -> int:
    print(f"fetching {url}")
    page = websearch.fetch(url)
    if page is None:
        print("  FETCH FAILED -- blocked, not HTML, or timed out")
        return 1
    print(f"  fetched {len(page):,} characters")
    node = extract.find_recipe_node(page)
    if node is None:
        blocks = len(list(extract.iter_json_ld(page)))
        print(f"  NO RECIPE MARKUP -- {blocks} JSON-LD block(s), none of @type Recipe")
        return 1
    recipe = extract.from_html(page, url)
    if recipe is None:
        print("  markup found but too thin to use (under 2 ingredients, or no steps)")
        return 1
    print(f"  TITLE      {recipe.title}")
    print(f"  domain     {recipe.domain}   author {recipe.author or '-'}")
    print(f"  diet       {recipe.diet} ({recipe.diet_confidence:.2f}) -- "
          f"{recipe.diet_reason[:60]}")
    print(f"  nutrition  [{recipe.nutrition_status}] cal={recipe.calories} "
          f"protein={recipe.protein} fat={recipe.fat} sodium={recipe.sodium}")
    print(f"  {len(recipe.ingredients)} ingredients, {len(recipe.directions)} steps, "
          f"yield {recipe.servings or '-'}")
    for line in recipe.ingredients[:6]:
        print(f"    - {line[:80]}")
    for index, step in enumerate(recipe.directions[:2], 1):
        print(f"    {index}. {step[:100]}")
    return 0


def show_ask(query: str) -> int:
    from recipe_rag.pipeline import Pipeline

    pipeline = Pipeline.load(use_judge=False, use_web=True)
    answer = pipeline.answer(query, log=False)
    print(f"Q: {query}")
    print(f"  band {answer.band}  vector {answer.vector_score:.3f}  "
          f"title {answer.title_score:.1f}")
    for reason in answer.reasons:
        print(f"    - {reason}")
    print(f"  source={answer.source}  stage={answer.failure_stage or '-'}  "
          f"{answer.seconds:.1f}s")
    if answer.recipe:
        print(f"  ANSWER: {answer.recipe.title}")
        print(f"          {answer.attribution}")
        print(f"          diet {answer.recipe.diet}, "
              f"nutrition [{answer.recipe.nutrition_status}] {answer.recipe.nutrition_line}")
    else:
        print(f"  {answer.message}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--search", help="run a Serper search and list the results")
    group.add_argument("--url", help="fetch and extract one page")
    group.add_argument("--ask", help="run the full pipeline with web enabled")
    parser.add_argument("-n", type=int, default=8, help="results to show for --search")
    args = parser.parse_args()

    try:
        if args.search:
            return show_search(args.search, args.n)
        if args.url:
            return show_url(args.url)
        return show_ask(args.ask)
    except websearch.WebUnavailable as exc:
        print(f"web unavailable: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
