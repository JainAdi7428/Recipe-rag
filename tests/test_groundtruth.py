"""Parsing and comparison rules for the grader's ground truth.

These rules decide what counts as a match, so they are pinned down here rather
than trusted: a scorer that silently mis-parses "452mg" or "Vegetarian, Contains
Egg" reports a number that means nothing.

Run standalone:  python tests/test_groundtruth.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recipe_rag import config
from recipe_rag.groundtruth import (
    DIET_DISPLAY, SOURCE_DATASET, SOURCE_NOT_FOUND, SOURCE_WEB,
    compare_nutrition, diet_from_display, normalise_title, parse_nutrition,
    read_cases, source_kind,
)

_TALLY = [0]


def check(condition: bool, message: str, failures: list[str]) -> None:
    _TALLY[0] += 1
    if not condition:
        failures.append(f"  {message}")


def run() -> int:
    failures: list[str] = []

    # --- nutrition lines, in every shape the file uses --------------------
    for text, expected in [
        ("Calories: 547 | Protein: 20g | Fat: 32g | Sodium: 452mg",
         {"calories": 547.0, "protein": 20.0, "fat": 32.0, "sodium": 452.0}),
        ("Calories: Not specified | Protein: Not specified | Fat: Not specified | Sodium: Not specified",
         {"calories": None, "protein": None, "fat": None, "sodium": None}),
        # Extra fields we do not report are ignored, not mis-assigned.
        ("Calories: 483 per serving | Protein: 5g | Fat: 35g | Sodium: 100mg | Carbs: ~42g (estimated)",
         {"calories": 483.0, "protein": 5.0, "fat": 35.0, "sodium": 100.0}),
        ("Calories: ~280 per serving | Protein: ~14g | Fat: ~22g | Fiber: ~4g | Iron: High",
         {"calories": 280.0, "protein": 14.0, "fat": 22.0}),
        ("NA", None),
        ("", None),
    ]:
        got = parse_nutrition(text)
        check(got == expected, f"parse_nutrition({text[:40]!r}) -> {got}, want {expected}", failures)

    # sodium given in grams must come back in milligrams
    got = parse_nutrition("Calories: 10 | Sodium: 1.2g")
    check(got and got["sodium"] == 1200.0, f"1.2g sodium -> {got}", failures)

    # --- source ------------------------------------------------------------
    for text, expected in [
        ("Internal Database", SOURCE_DATASET),
        ("Web Search (sudachirecipes.com, allrecipes.com)", SOURCE_WEB),
        ("NA", SOURCE_NOT_FOUND),
        ("", SOURCE_NOT_FOUND),
    ]:
        check(source_kind(text) == expected, f"source_kind({text!r}) -> {source_kind(text)}", failures)

    # --- diet vocabulary ---------------------------------------------------
    check(set(DIET_DISPLAY) == set(config.DIET_LABELS),
          "every internal diet label needs a display name", failures)
    for text, expected in [
        ("Vegetarian, Contains Egg", "eggetarian"),
        ("Non-Vegetarian", "non-veg"),
        ("Vegetarian", "veg"),
        ("Vegan", "vegan"),
        ("  vegetarian,contains egg ", "eggetarian"),
        ("NA", None),
    ]:
        got = diet_from_display(text)
        check(got == expected, f"diet_from_display({text!r}) -> {got}, want {expected}", failures)
    for diet, label in DIET_DISPLAY.items():
        check(diet_from_display(label) == diet, f"round trip failed for {diet}", failures)

    # --- comparison --------------------------------------------------------
    rows = compare_nutrition({"calories": 547.0, "sodium": None},
                             {"calories": 547.0, "sodium": None})
    check(all(ok for *_, ok in rows), "equal values and matching Nones must pass", failures)
    rows = compare_nutrition({"calories": None}, {"calories": 300.0})
    check(not rows[0][3], "a value where the grader has none must fail", failures)
    rows = compare_nutrition({"fat": 32.0}, {"fat": 32.4})
    check(rows[0][3], "within tolerance must pass", failures)

    check(normalise_title("  Spinach  Noodle Casserole ") == "spinach noodle casserole",
          "title normalisation", failures)

    # --- the real file, if present -----------------------------------------
    if config.GROUND_TRUTH_XLSX.exists():
        cases = read_cases()
        check(len(cases) >= 10, f"expected at least 10 cases, got {len(cases)}", failures)
        first = cases[0]
        check(first.expected_title == "Mahi-Mahi in Tomato Olive Sauce",
              f"row 1 title -> {first.expected_title!r}", failures)
        check(first.expected_source == SOURCE_DATASET, "row 1 is internal", failures)
        check(first.expected_nutrition == {"calories": None, "protein": None, "fat": None, "sodium": None},
              "row 1 nutrition is 'Not specified' throughout", failures)

    if failures:
        print(f"FAIL {len(failures)}/{_TALLY[0]} ground-truth parsing cases")
        for line in failures:
            print(line)
        return 1
    print(f"ok  {_TALLY[0]}/{_TALLY[0]} ground-truth parsing cases passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
