"""Write a held-out set of descriptive queries, one per sampled recipe.

    python scripts/make_tuning_set.py --n 150        # ~$0.03 with gpt-4o-mini

Why this exists: the grader's ground truth has six dataset queries, and any
ranking weight can be fitted to six queries. Weights are chosen on THIS set
instead, and the ground truth is kept for the final check.

Why an LLM writes them: building queries from category tags ourselves would
make a tag-aware re-ranker look perfect by construction. The model is shown the
whole recipe -- title, tags, ingredients -- and asked for the kind of request a
person would type, so the words it picks are its own. Ground-truth recipes are
excluded from sampling so nothing leaks between the two sets.

Each row records the recipe it was written from. Other recipes may also fit the
request, so R@1 on this set is a lower bound; R@5 and MRR matter as much.

Resumable: rows already in the file are skipped. Output:
.artifacts/tuning_queries.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recipe_rag import config, groundtruth, store
from recipe_rag import llm as llm_module
from recipe_rag.groundtruth import DIET_DISPLAY, normalise_title
from recipe_rag.llm import LLM, LLMUnavailable
from recipe_rag.query import NON_DISH_TAGS

OUT = config.ARTIFACT_DIR / "tuning_queries.jsonl"

SYSTEM = """\
You write realistic requests that people type into a recipe app.
You are shown one recipe. Write ONE request (8 to 16 words) that this recipe
would be a good answer to, in the style of these real examples:
  "I need a healthy fish recipe with olives and tomatoes."
  "Show me a vegetarian spinach pasta casserole"
  "I want a gluten-free salad with tuna and asparagus"
Mention 2 to 4 features that would help find THIS recipe among thousands: main
ingredients, dish type, diet, occasion, or cooking method. Do not copy the
recipe title, and do not name the recipe. Vary the phrasing.
Reply with JSON only: {"query": "..."}"""


def recipe_card(row) -> str:
    tags = [t for t in row.categories if t.lower() not in NON_DISH_TAGS][:12]
    return (
        f"Title: {row.title}\n"
        f"Diet: {DIET_DISPLAY.get(row.diet, row.diet)}\n"
        f"Tags: {', '.join(tags)}\n"
        f"Ingredients: {'; '.join(i[:60] for i in list(row.ingredients)[:8])}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    parents = store.load_parents()
    pool = parents[(parents.dup_group == "") & (parents.ingredients.map(len) >= 3)]
    try:
        held_out = {normalise_title(c.expected_title) for c in groundtruth.read_cases()
                    if c.expected_title}
    except (OSError, PermissionError) as exc:
        raise SystemExit(f"cannot read the ground truth to exclude it ({exc}); close Excel and retry")
    pool = pool[~pool.title.map(normalise_title).isin(held_out)]
    sample = pool.sample(min(args.n, len(pool)), random_state=args.seed)

    done = set()
    if OUT.exists():
        done = {json.loads(l)["parent_id"] for l in OUT.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = [row for row in sample.itertuples() if row.parent_id not in done]
    print(f"{len(done)} already written, {len(todo)} to go (excluded {len(held_out)} ground-truth recipes)")

    llm = LLM()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with OUT.open("a", encoding="utf-8") as handle:
        for row in todo:
            try:
                parsed, _ = llm.complete_json(SYSTEM, recipe_card(row), purpose="tuning_set", max_tokens=60)
            except LLMUnavailable as exc:
                print(f"\nstopped after {written}: {str(exc).splitlines()[0]}")
                break
            query = str(parsed.get("query") or "").strip()
            if len(query.split()) < 4 or normalise_title(row.title) in normalise_title(query):
                continue    # too short, or it named the recipe outright
            handle.write(json.dumps({"query": query, "parent_id": row.parent_id,
                                     "title": row.title}, ensure_ascii=False) + "\n")
            written += 1
            if written % 25 == 0:
                print(f"  {written}  ${llm_module.LEDGER.total.cost_usd:.4f}")

    print(f"\nwrote {written} queries to {OUT}")
    print(llm_module.LEDGER.report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
