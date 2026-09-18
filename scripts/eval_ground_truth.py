"""Score the system against the grader's ground truth (UC13 validation data).

    python scripts/eval_ground_truth.py                    # judge + web if keys exist
    python scripts/eval_ground_truth.py --no-judge --no-web
    python scripts/eval_ground_truth.py --show 3           # one row in full

Grades every query the way the problem statement says it will be graded --
source, recipe, dietary class, nutrition -- and then runs two checks that do
not depend on retrieval at all:

  * the diet label we store for the *expected* recipe
  * the nutrition we store for the *expected* recipe

That split is the point. A wrong answer can mean we ranked the wrong recipe
first, or that we found the right one but labelled it wrongly, and those need
completely different fixes. Mixing them into one accuracy number hides which.

**Ten queries, six of them from the dataset.** Every rate here has a very wide
error bar, and nothing should be tuned against these rows directly: a weight
chosen to fix row 9 is a weight fitted to row 9. They are the acceptance test;
tuning happens on a separate, larger set.

Writes .artifacts/eval_ground_truth.json.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recipe_rag import config, groundtruth
from recipe_rag import llm as llm_module
from recipe_rag.groundtruth import (
    DIET_DISPLAY, NUTRIENTS, SOURCE_DATASET, SOURCE_NOT_FOUND,
    compare_nutrition, normalise_title,
)
from recipe_rag.pipeline import Pipeline

REPORT = config.ARTIFACT_DIR / "eval_ground_truth.json"


def mark(value) -> str:
    return "-" if value is None else ("ok" if value else "MISS")


def nutrition_of(recipe) -> dict:
    return {name: getattr(recipe, name) for name in NUTRIENTS}


def grade(case, answer, web_on: bool) -> dict:
    expected = case.expected_source
    got = answer.source

    # With the web leg switched off, "not found" only tells us the query was
    # escalated past the dataset -- which is the right call for both a web row
    # and an apology row. Scored as routing, not as a final outcome.
    if got == SOURCE_NOT_FOUND and not web_on:
        route_label, route_ok = "escalated", expected != SOURCE_DATASET
    else:
        route_label, route_ok = got, got == expected

    recipe_ok = None
    if expected == SOURCE_DATASET:
        recipe_ok = (
            got == SOURCE_DATASET
            and answer.recipe is not None
            and normalise_title(answer.recipe.title) == normalise_title(case.expected_title)
        )
    elif expected == SOURCE_NOT_FOUND and web_on:
        # The spec's hardest rule: no recipe at all when none exists.
        recipe_ok = answer.recipe is None

    diet_ok = None
    if answer.recipe is not None and case.expected_diet:
        diet_ok = answer.recipe.diet == case.expected_diet

    nutrition_ok, nutrition_rows = None, []
    if recipe_ok and expected == SOURCE_DATASET and case.expected_nutrition:
        nutrition_rows = compare_nutrition(case.expected_nutrition, nutrition_of(answer.recipe))
        nutrition_ok = all(ok for *_, ok in nutrition_rows)

    return {
        "n": case.number,
        "query": case.query,
        "expected_source": expected,
        "got_source": route_label,
        "route_ok": route_ok,
        "expected_title": case.expected_title,
        "got_title": answer.recipe.title if answer.recipe else None,
        "recipe_ok": recipe_ok,
        "expected_diet": case.expected_diet,
        "got_diet": answer.recipe.diet if answer.recipe else None,
        "diet_ok": diet_ok,
        "nutrition_ok": nutrition_ok,
        "nutrition": [list(r) for r in nutrition_rows],
        "band": answer.band,
        "vector": round(answer.vector_score, 4),
        "title": round(answer.title_score, 2),
        "judged": answer.verdict is not None,
        "reasons": answer.reasons,
        "seconds": round(answer.seconds, 2),
        "cost_usd": answer.usage.cost_usd,
    }


def label_checks(cases, pipeline) -> list[dict]:
    """Diet and nutrition for the EXPECTED recipe, whatever retrieval returned."""
    titles = pipeline.parents.title.map(normalise_title)
    out = []
    for case in cases:
        if case.expected_source != SOURCE_DATASET or not case.expected_title:
            continue
        matches = pipeline.parents[titles == normalise_title(case.expected_title)]
        if matches.empty:
            out.append({"n": case.number, "in_index": False})
            continue
        recipe = pipeline.dataset_answer(matches.iloc[0].parent_id)
        nutrition_rows = (
            compare_nutrition(case.expected_nutrition, nutrition_of(recipe))
            if case.expected_nutrition else []
        )
        out.append({
            "n": case.number,
            "in_index": True,
            "our_diet": recipe.diet,
            "expected_diet": case.expected_diet,
            "diet_ok": recipe.diet == case.expected_diet,
            "diet_reason": recipe.diet_reason,
            "nutrition_ok": all(ok for *_, ok in nutrition_rows) if nutrition_rows else None,
            "nutrition": [list(r) for r in nutrition_rows],
        })
    return out


def tally(rows, key) -> tuple[int, int]:
    scored = [r[key] for r in rows if r.get(key) is not None]
    return sum(1 for v in scored if v), len(scored)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--no-web", action="store_true")
    parser.add_argument("--show", type=int, help="print one row in full")
    parser.add_argument("--path", type=Path, default=config.GROUND_TRUTH_XLSX)
    args = parser.parse_args()

    cases = groundtruth.read_cases(args.path)
    pipeline = Pipeline.load(use_judge=not args.no_judge, use_web=not args.no_web)
    web_on = not args.no_web
    judge_live = not args.no_judge and pipeline.llm.available

    print(f"{len(cases)} ground-truth queries   judge: "
          f"{'on' if judge_live else 'OFF (rule gate)'}   web: {'on' if web_on else 'OFF'}")
    if not args.no_judge and not judge_live:
        print("  no usable OPENAI_API_KEY -- run scripts/check_llm.py")

    rows = []
    for case in cases:
        answer = pipeline.answer(case.query, log=False)
        rows.append(grade(case, answer, web_on))
        if args.show == case.number:
            print(f"\n#{case.number} {case.query}")
            print(f"  expected: {case.expected_source} / {case.expected_title} / "
                  f"{DIET_DISPLAY.get(case.expected_diet or '', case.diet_text)}")
            print(f"  got     : {answer.source} / "
                  f"{answer.recipe.title if answer.recipe else None}")
            print(f"  band {answer.band}  vector {answer.vector_score:.3f}  "
                  f"title {answer.title_score:.1f}")
            for reason in answer.reasons:
                print(f"    - {reason}")

    print(f"\n{'#':>3}  {'route':<22} {'recipe':<6} {'diet':<5} {'nutr':<5} {'band':<10} query")
    print("-" * 100)
    for r in rows:
        route = f"{r['expected_source'][:7]}->{r['got_source'][:9]}"
        print(f"{r['n']:>3}  {route:<17} {mark(r['route_ok']):<4} "
              f"{mark(r['recipe_ok']):<6} {mark(r['diet_ok']):<5} "
              f"{mark(r['nutrition_ok']):<5} {r['band']:<10} {r['query'][:44]}")
        if r["recipe_ok"] is False and r["got_title"]:
            print(f"{'':>24}got {r['got_title'][:60]!r}")

    checks = label_checks(cases, pipeline)
    print("\nindependent of retrieval -- what we store for the EXPECTED recipe:")
    for c in checks:
        if not c["in_index"]:
            print(f"  #{c['n']:<3} expected recipe NOT IN INDEX")
            continue
        diet = (f"diet {mark(c['diet_ok']):<4} ours={DIET_DISPLAY.get(c['our_diet'], c['our_diet'])!r}"
                f" expected={DIET_DISPLAY.get(c['expected_diet'] or '', c['expected_diet'])!r}")
        print(f"  #{c['n']:<3} {diet:<75} nutrition {mark(c['nutrition_ok'])}")
        if not c["diet_ok"]:
            print(f"        because: {c['diet_reason'][:80]}")

    route = tally(rows, "route_ok")
    recipe = tally([r for r in rows if r["expected_source"] == SOURCE_DATASET], "recipe_ok")
    apology = tally([r for r in rows if r["expected_source"] == SOURCE_NOT_FOUND], "recipe_ok")
    diet = tally(rows, "diet_ok")
    nutrition = tally(rows, "nutrition_ok")
    label_diet = tally(checks, "diet_ok")
    label_nutrition = tally(checks, "nutrition_ok")
    seconds = [r["seconds"] for r in rows]

    print("\nsummary")
    print(f"  source routing          {route[0]}/{route[1]}")
    print(f"  right recipe (dataset)  {recipe[0]}/{recipe[1]}")
    if apology[1]:
        print(f"  no recipe when none     {apology[0]}/{apology[1]}")
    print(f"  diet on answers         {diet[0]}/{diet[1]}")
    print(f"  nutrition on answers    {nutrition[0]}/{nutrition[1]}")
    print(f"  -- for the expected recipe, regardless of retrieval --")
    print(f"  diet label              {label_diet[0]}/{label_diet[1]}")
    print(f"  nutrition values        {label_nutrition[0]}/{label_nutrition[1]}")
    print(f"  latency  median {statistics.median(seconds):.2f}s  max {max(seconds):.2f}s")
    if llm_module.LEDGER.total.calls:
        print(f"  {llm_module.LEDGER.report()}")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(
        {"judge": judge_live, "web": web_on, "rows": rows, "label_checks": checks},
        indent=2, default=str,
    ), encoding="utf-8")
    print(f"\nwrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
