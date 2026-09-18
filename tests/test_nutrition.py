"""Nutrition validation: implausible rows are dropped whole, not clamped.

Run standalone:  python tests/test_nutrition.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recipe_rag.nutrition import (
    STATUS_FIELD_REJECTED,
    STATUS_MISSING,
    STATUS_OK,
    STATUS_PARTIAL,
    STATUS_REJECTED,
    validate,
)


def check(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(f"  {message}")


def run() -> int:
    failures: list[str] = []

    # --- ordinary rows pass through untouched ----------------------------
    ok = validate({"calories": 426.0, "protein": 30.0, "fat": 7.0, "sodium": 559.0})
    check(ok.status == STATUS_OK, f"complete row should be ok, got {ok.status}", failures)
    check(ok.calories == 426.0, "values must pass through unchanged", failures)

    # --- a rich but real recipe survives ---------------------------------
    pie = validate({"calories": 4898.0, "protein": 50.0, "fat": 319.0, "sodium": 1200.0})
    check(
        pie.status == STATUS_OK,
        "a 4,898 kcal pecan pie is high but real and must survive",
        failures,
    )

    # --- the corrupt rows are rejected -----------------------------------
    corrupt = validate(
        {"calories": 30111218.0, "protein": 200968.0, "fat": 5000.0, "sodium": 27675110.0}
    )
    check(
        corrupt.status == STATUS_REJECTED,
        f"30M calories must be rejected, got {corrupt.status}",
        failures,
    )
    check(
        all(v is None for v in corrupt.as_dict().values()),
        "a rejected row must expose no values at all",
        failures,
    )
    check(not corrupt.has_any, "rejected row has_any must be False", failures)

    # --- two impossible fields means the parse failed: drop everything ---
    two_bad = validate(
        {"calories": 4157357.0, "protein": 236489.0, "fat": 200.0, "sodium": 900.0}
    )
    check(
        two_bad.status == STATUS_REJECTED and two_bad.fat is None,
        "2+ impossible fields must drop the whole row, including plausible ones",
        failures,
    )

    # --- one impossible field is dropped alone, the rest survive ---------
    # "Whole Fish Baked in Sea Salt": the salt crust is not eaten, so the
    # sodium is genuinely enormous while everything else is ordinary.
    brine = validate(
        {"calories": 553.0, "protein": 48.0, "fat": 38.0, "sodium": 97225.0}
    )
    check(
        brine.status == STATUS_FIELD_REJECTED,
        f"one bad field should be field_rejected, got {brine.status}",
        failures,
    )
    check(brine.sodium is None, "the impossible field must be dropped", failures)
    check(
        brine.calories == 553.0 and brine.protein == 48.0 and brine.fat == 38.0,
        "the plausible fields beside it must be retained",
        failures,
    )
    check(
        brine.rejected_fields == ["sodium"],
        f"the dropped field should be recorded, got {brine.rejected_fields}",
        failures,
    )
    check(bool(brine.notes), "a rejection must explain itself", failures)

    # --- zero calories is a missing value wearing a number ---------------
    zero = validate({"calories": 0.0, "protein": 2.0, "fat": 1.0, "sodium": 10.0})
    check(
        zero.status == STATUS_FIELD_REJECTED and zero.calories is None,
        f"zero calories should not be trusted, got {zero.status}",
        failures,
    )
    check(zero.protein == 2.0, "zero calories must not void the other fields", failures)
    # ...but zero protein or fat is perfectly ordinary (simple syrup).
    syrup = validate({"calories": 90.0, "protein": 0.0, "fat": 0.0, "sodium": 1.0})
    check(
        syrup.status == STATUS_OK and syrup.protein == 0.0,
        "zero protein/fat is legitimate and must be kept",
        failures,
    )

    # --- partial and missing ---------------------------------------------
    partial = validate({"calories": 300.0, "protein": None, "fat": None, "sodium": None})
    check(partial.status == STATUS_PARTIAL, "partial rows are flagged", failures)
    check(partial.has_any, "partial rows still have data", failures)

    nothing = validate({"calories": None, "protein": None, "fat": None, "sodium": None})
    check(nothing.status == STATUS_MISSING, "empty rows are missing", failures)
    check(not nothing.has_any, "missing rows have nothing", failures)

    negative = validate({"calories": -10.0, "protein": 5.0, "fat": 1.0, "sodium": 20.0})
    check(
        negative.status == STATUS_FIELD_REJECTED and negative.calories is None,
        "a single negative is dropped on its own",
        failures,
    )

    # --- nothing is invented ---------------------------------------------
    check(
        validate({"calories": 99999.0, "protein": None, "fat": None, "sodium": None}).calories
        is None,
        "rejected values must become None, never clamped to the limit",
        failures,
    )

    total = 18
    if failures:
        print(f"FAILED {len(failures)}/{total}")
        print("\n".join(failures))
        return 1
    print(f"ok  {total}/{total} nutrition validation cases passed")
    return 0


def test_nutrition_validation():
    assert run() == 0


if __name__ == "__main__":
    raise SystemExit(run())
