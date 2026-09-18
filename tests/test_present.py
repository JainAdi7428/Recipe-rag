"""The submission's output format: the grader compares against these strings.

Each case mirrors a row of the validation file, so a change in wording that
would cost marks fails here first.

Run standalone:  python tests/test_present.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recipe_rag.pipeline import (
    APOLOGY, SOURCE_DATASET, SOURCE_NOT_FOUND, SOURCE_WEB, Answer, AnswerRecipe,
)
from recipe_rag.present import as_markdown, present

CASSEROLE = AnswerRecipe(
    title="Spinach Noodle Casserole",
    ingredients=["1 12-ounce package frozen spinach souffle, thawed",
                 "1/2 pound extra-wide egg noodles, freshly cooked",
                 "1 cup sour cream"],
    directions=["Preheat oven to 350 F.", "Bake until set, about 45 minutes."],
    diet="eggetarian",
    calories=547.0, protein=20.0, fat=32.0, sodium=452.0,
    nutrition_status="ok",
)
MAHI = AnswerRecipe(
    title="Mahi-Mahi in Tomato Olive Sauce",
    ingredients=["2 tablespoons extra-virgin olive oil", "1 cup chopped onion"],
    directions=["Heat oil in heavy large skillet."],
    diet="non-veg",
    nutrition_status="missing",
)

_TALLY = [0]


def check(condition: bool, message: str, failures: list[str]) -> None:
    _TALLY[0] += 1
    if not condition:
        failures.append(f"  {message}")


def answer_for(recipe, source) -> Answer:
    return Answer(query="q", cleaned="q", source=source, recipe=recipe,
                  message=APOLOGY if recipe is None else "")


def run() -> int:
    failures: list[str] = []

    # --- a dataset answer, as the grader writes it -------------------------
    shown = present(answer_for(CASSEROLE, SOURCE_DATASET))
    check(shown.source == "Internal Database", f"source -> {shown.source!r}", failures)
    check(shown.diet == "Vegetarian, Contains Egg",
          f"our 'eggetarian' must print in their words, got {shown.diet!r}", failures)
    check(shown.nutrition == "Calories: 547 | Protein: 20g | Fat: 32g | Sodium: 452mg",
          f"nutrition -> {shown.nutrition!r}", failures)
    lines = shown.answer.splitlines()
    check(lines[0] == "Recipe Title: Spinach Noodle Casserole", f"first line -> {lines[0]!r}", failures)
    check(lines[1] == "Ingredients:", f"second line -> {lines[1]!r}", failures)
    check(lines[2] == "1 12-ounce package frozen spinach souffle, thawed",
          "each ingredient goes on its own line", failures)
    check("Cooking Instructions:" in lines, "the steps need their own heading", failures)
    check(lines.index("Cooking Instructions:") > lines.index("Ingredients:"),
          "ingredients come before the method", failures)

    # --- a recipe with no nutrition, in the grader's phrasing ---------------
    shown = present(answer_for(MAHI, SOURCE_DATASET))
    check(shown.nutrition == ("Calories: Not specified | Protein: Not specified | "
                              "Fat: Not specified | Sodium: Not specified"),
          f"missing nutrition -> {shown.nutrition!r}", failures)
    check(shown.diet == "Non-Vegetarian", f"diet -> {shown.diet!r}", failures)

    # Carbohydrates are never reported: the dataset has no such field.
    check("Carb" not in shown.nutrition and "carb" not in shown.nutrition,
          "carbs must never appear", failures)

    # --- a web answer -------------------------------------------------------
    web = AnswerRecipe(title="Palak Paneer Recipe", ingredients=["spinach", "paneer"],
                       directions=["Blanch the spinach."], diet="veg",
                       url="https://www.vegrecipesofindia.com/palak-paneer/",
                       calories=265.0, protein=10.0, fat=23.0, sodium=419.0,
                       nutrition_status="ok")
    shown = present(answer_for(web, SOURCE_WEB))
    check(shown.source == "Web Search (vegrecipesofindia.com)",
          f"web source -> {shown.source!r}", failures)
    check(shown.diet == "Vegetarian", f"web diet -> {shown.diet!r}", failures)

    # --- nothing found ------------------------------------------------------
    shown = present(answer_for(None, SOURCE_NOT_FOUND))
    check(shown.source == "NA" and shown.diet == "NA" and shown.nutrition == "NA",
          f"an apology carries NA in every column, got {shown}", failures)
    check(shown.answer == APOLOGY, "the apology text is fixed, never generated", failures)

    # --- number formatting --------------------------------------------------
    odd = AnswerRecipe(title="x", ingredients=["a"], directions=["b"], diet="vegan",
                       calories=483.0, protein=5.5, fat=None, sodium=100.0)
    shown = present(answer_for(odd, SOURCE_DATASET))
    check(shown.nutrition == "Calories: 483 | Protein: 5.5g | Fat: Not specified | Sodium: 100mg",
          f"whole numbers must not print as 483.0, got {shown.nutrition!r}", failures)

    # --- markdown rendering for the app ------------------------------------
    text = as_markdown(answer_for(CASSEROLE, SOURCE_DATASET))
    check(text.startswith("### Spinach Noodle Casserole"), "markdown heading", failures)
    check("- 1 cup sour cream" in text, "ingredients as a list", failures)
    check("1. Preheat oven to 350 F." in text, "steps numbered", failures)
    check(as_markdown(answer_for(None, SOURCE_NOT_FOUND)) == APOLOGY,
          "markdown apology", failures)

    if failures:
        print(f"FAIL {len(failures)}/{_TALLY[0]} output format cases")
        for line in failures:
            print(line)
        return 1
    print(f"ok  {_TALLY[0]}/{_TALLY[0]} output format cases passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
