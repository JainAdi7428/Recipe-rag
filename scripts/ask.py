"""Ask the system a question and watch it decide.

    python scripts/ask.py "Give me the recipe for Chicken Alfredo"
    python scripts/ask.py "How do I make Paneer Butter Masala?"
    python scripts/ask.py --demo
    python scripts/ask.py --demo --no-web        # gate + judge only, no network
    python scripts/ask.py --demo --no-judge      # phase 1 behaviour, for contrast

Prints both retrieval signals, which band they land in, the judge's verdict if
one was needed, the web pages tried if it got that far, and then the recipe --
with its source attributed and its cost stated.
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


from recipe_rag import llm as llm_module
from recipe_rag.pipeline import SOURCE_DATASET, SOURCE_NOT_FOUND, Pipeline
from recipe_rag.present import present

DEMO = [
    "Give me the recipe for Chicken Alfredo",     # confident, title
    "Spanakopita",                                 # confident by title, weak vector
    "Recipe for Vegan Banana Bread",               # the "pecan" trap
    "something creamy with pasta and chicken",     # descriptive, ambiguous band
    "How do I make Paneer Butter Masala?",         # absent -> web
    "Masala Dosa",                                 # absent, lexically tempting
    "Quantum Blockchain Smoothie",                 # absent everywhere -> apology
]

BANDS = {"confident": "CONFIDENT", "ambiguous": "AMBIGUOUS", "hopeless": "HOPELESS"}


def show(answer) -> None:
    print(f"\n{'=' * 74}")
    print(f"Q: {answer.query}")
    if answer.cleaned.lower() != answer.query.strip().lower():
        print(f"   cleaned -> {answer.cleaned!r}")
    print(f"{'-' * 74}")
    print(f"  vector {answer.vector_score:.3f}   title {answer.title_score:>5.1f}"
          f"   band {BANDS.get(answer.band, answer.band)}")
    for reason in answer.reasons:
        print(f"    - {reason}")

    if answer.source == SOURCE_NOT_FOUND:
        print(f"\n  NOT FOUND  ({answer.seconds:.1f}s)")
        print(f"    {answer.message}")
        return

    recipe = answer.recipe
    label = "DATASET" if answer.source == SOURCE_DATASET else "WEB"
    print(f"\n  ANSWER [{label}]  ({answer.seconds:.1f}s, ${answer.usage.cost_usd:.6f})")
    print(f"    {recipe.title}")
    print(f"    source: {answer.attribution}")
    if recipe.url:
        print(f"            {recipe.url}")
    print(f"    diet: {recipe.diet} ({recipe.diet_confidence:.2f})"
          f"{'  [CONFLICT]' if recipe.diet_conflict else ''}")
    print(f"    reason: {recipe.diet_reason[:76]}")
    print(f"    nutrition [{recipe.nutrition_status}]: {recipe.nutrition_line}")
    print(f"    {len(recipe.ingredients)} ingredients, {len(recipe.directions)} steps")
    for item in recipe.ingredients[:4]:
        print(f"      - {item[:66]}")
    if len(recipe.ingredients) > 4:
        print(f"      ... {len(recipe.ingredients) - 4} more")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="*")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--no-judge", action="store_true",
                        help="skip the LLM judge; phase-1 rule gate only")
    parser.add_argument("--no-web", action="store_true",
                        help="skip the web fallback; no network calls")
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--grader", action="store_true",
                        help="print the four columns the validation file compares")
    args = parser.parse_args()

    pipeline = Pipeline.load(
        use_judge=not args.no_judge,
        use_web=not args.no_web,
    )
    if not args.no_judge and not pipeline.llm.available:
        print("note: no usable OPENAI_API_KEY -- the judge will be skipped and the\n"
              "      rule-based gate used instead. Run scripts/check_llm.py.\n")

    queries = DEMO if args.demo or not args.query else [" ".join(args.query)]
    for query in queries:
        answer = pipeline.answer(query, log=not args.no_log)
        if args.grader:
            print(f"{'=' * 74}")
            print(f"Q: {query}")
            print(present(answer).as_text())
        else:
            show(answer)

    if llm_module.LEDGER.total.calls:
        print(f"\n{'=' * 74}")
        print(llm_module.LEDGER.report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
