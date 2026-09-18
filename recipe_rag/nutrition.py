"""Nutrition plausibility validation.

The source data cannot be taken at face value. The worst row claims 30,111,218
calories and 27.6 kg of sodium for a cranberry pie -- a unit or parsing error in
the original scrape. Indexing that as fact would let the system report it to a
user as nutrition guidance.

The key finding that shapes this module: an internal-consistency check does not
catch these. Atwater says calories must be at least 4*protein + 9*fat, and
*zero* rows violate that beyond a 15% tolerance -- because the corrupt rows are
scaled consistently across every field at once. The 30M-calorie pie also claims
200,968 g of protein, so the ratios still look fine.

So detection has to come from absolute physiological ceilings, not from
cross-field ratios.

**How much of a row to discard depends on how many fields broke.** Measuring the
111 affected rows showed two distinct populations:

    1 field out of range :  68 rows   (61%)
    2 fields             :  25
    3 fields             :  11
    4 fields             :   7

The multi-field rows are scaling errors that wrecked everything at once:

    30,111,218 kcal | 200,968 g protein | 27,675,110 mg sodium   (a pie)

The single-field rows are usually a *real* measurement that is extreme for a
legitimate reason, sitting beside three perfectly sensible values:

      553 kcal |  48 g protein |  38 g fat |  97,225 mg sodium
      "Whole Fish Baked in Sea Salt" -- the salt crust is not eaten

    180 kcal | 4 g protein | 2 g fat | 45,166 mg sodium
      "Citrus and Peppercorn Dry Brine" -- it is mostly salt, by design

Discarding calories, protein and fat for those would throw away good data to
punish one field. So:

    2 or more fields impossible  ->  reject the whole row (systematic corruption)
    exactly 1 field impossible   ->  null that field, keep the rest

Nulling rather than clamping or imputing is deliberate throughout: for a system
whose stated purpose is dietary compliance, "we don't know" is honest and
useful, while a plausible-looking invented number is neither.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Ceilings above which a value cannot describe a single dish, read generously
# as an entire recipe rather than one serving. Chosen well above the legitimate
# tail (p99 calories is 3,338) so that genuinely rich recipes survive: a whole
# pecan pie at 4,898 kcal is kept, chicken wings at 12,824 kcal are not.
NUTRITION_LIMITS: dict[str, float] = {
    "calories": 5_000.0,   # kcal
    "protein": 500.0,      # g
    "fat": 500.0,          # g
    "sodium": 20_000.0,    # mg
}

NUTRITION_FIELDS = tuple(NUTRITION_LIMITS)

# Status values recorded on the parent row and in chunk metadata.
STATUS_OK = "ok"
STATUS_PARTIAL = "partial"        # some fields present, others absent at source
STATUS_MISSING = "missing"        # nothing at source
STATUS_REJECTED = "rejected"      # whole row dropped: 2+ impossible fields
STATUS_FIELD_REJECTED = "field_rejected"   # one field dropped, rest kept

# Two or more impossible values means the row was mis-parsed as a whole; one
# means a single extreme-but-real measurement (a brine's sodium, say).
ROW_REJECT_THRESHOLD = 2


@dataclass
class NutritionResult:
    calories: float | None = None
    protein: float | None = None
    fat: float | None = None
    sodium: float | None = None
    status: str = STATUS_MISSING
    rejected_fields: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def has_any(self) -> bool:
        return any(
            getattr(self, name) is not None for name in NUTRITION_FIELDS
        )

    def as_dict(self) -> dict[str, float | None]:
        return {name: getattr(self, name) for name in NUTRITION_FIELDS}


def validate(values: dict[str, float | None]) -> NutritionResult:
    """Screen one recipe's nutrition against physiological ceilings.

    One impossible field is dropped on its own; two or more take the whole row
    down, because that pattern indicates the parse itself failed. Zero calories
    counts as impossible: a dish with ingredients cannot have none, so a 0 there
    is a missing value wearing a number.
    """
    result = NutritionResult()

    present = {
        name: value
        for name, value in values.items()
        if name in NUTRITION_LIMITS and value is not None
    }
    if not present:
        result.status = STATUS_MISSING
        return result

    impossible = {
        name
        for name, value in present.items()
        if value < 0 or value > NUTRITION_LIMITS[name]
    }
    if present.get("calories") == 0:
        impossible.add("calories")

    result.rejected_fields = sorted(impossible)

    if len(impossible) >= ROW_REJECT_THRESHOLD:
        # Several fields impossible at once: the row was mis-parsed as a whole,
        # so even its plausible-looking values are untrustworthy.
        result.status = STATUS_REJECTED
        result.notes.append(
            f"nutrition dropped: {len(impossible)} fields "
            f"({', '.join(result.rejected_fields)}) are outside the plausible "
            f"range, so the whole row's parse is untrusted"
        )
        return result

    for name, value in present.items():
        if name not in impossible:
            setattr(result, name, value)

    if impossible:
        offender = result.rejected_fields[0]
        result.status = STATUS_FIELD_REJECTED
        result.notes.append(
            f"{offender}={present.get(offender)!r} is outside the plausible "
            f"range and was dropped; the other values are retained"
        )
        return result

    result.status = (
        STATUS_OK if len(present) == len(NUTRITION_LIMITS) else STATUS_PARTIAL
    )
    return result
