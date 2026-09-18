"""Query cleaning, title coverage scoring, and the dataset/web gate.

Uses a tiny in-memory title table, so it runs without the built index.

Run standalone:  python tests/test_gate.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from recipe_rag.gate import SOURCE_DATASET, SOURCE_WEB, GateThresholds, decide
from recipe_rag.lexical import TitleMatcher, _coverage, _tokens, clean_query

TITLES = [
    "Chicken Alfredo with Zucchini Ribbons",
    "Spanakopita",
    "Chicken Tikka Masala",
    "Korean Marinated Beef",
    "Banana-Chocolate Chip Cake With Peanut Butter Frosting",
    "Boudin Blanc Terrine with Red Onion Confit",
    "Simple Syrup",
    "Grilled Salmon with Basil and Mint",
]


def make_matcher() -> TitleMatcher:
    frame = pd.DataFrame(
        {
            "parent_id": [f"id{i}" for i in range(len(TITLES))],
            "title": TITLES,
            "dup_rank": [0] * len(TITLES),
        }
    )
    return TitleMatcher(frame)


@dataclass
class FakeHit:
    score: float
    parent_id: str = "vec-parent"


@dataclass
class FakeTitle:
    score: float
    parent_id: str = "title-parent"
    title: str = "Some Recipe"


def check(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(f"  {message}")


def run() -> int:
    failures: list[str] = []

    # --- query cleaning strips request boilerplate ------------------------
    for raw, expected in [
        ("Give me the recipe for Chicken Alfredo", "chicken alfredo"),
        ("How do I make Paneer Butter Masala?", "paneer butter masala"),
        ("Recipe for Vegan Banana Bread", "vegan banana bread"),
        ("what is the recipe for spanakopita", "spanakopita"),
        ("I want to make Simple Syrup", "simple syrup"),
        ("Spanakopita", "spanakopita"),
    ]:
        got = clean_query(raw)
        check(got == expected, f"clean_query({raw!r}) -> {got!r}, want {expected!r}", failures)

    # cleaning must never empty the query out
    check(clean_query("recipe for").strip() != "", "cleaning must not return empty", failures)

    # --- coverage: the four real leak cases must score low ----------------
    # Each shares exactly one word with the title it wrongly matched under
    # WRatio, which scored all of them 85.5.
    leaks = [
        ("paneer butter masala", "Banana-Chocolate Chip Cake With Peanut Butter Frosting"),
        ("masala dosa", "Chicken Tikka Masala"),
        ("beef rendang", "Korean Marinated Beef"),
        ("recursive onion soup", "Boudin Blanc Terrine with Red Onion Confit"),
    ]
    for query, title in leaks:
        score = _coverage(_tokens(query), _tokens(title))
        check(
            score < 80.0,
            f"one shared word must not read as a match: {query!r} ~ {title[:30]!r} = {score:.0f}",
            failures,
        )

    # --- coverage: a query fully contained in a longer title scores high --
    full = _coverage(_tokens("chicken alfredo"), _tokens("Chicken Alfredo with Zucchini Ribbons"))
    check(full >= 95.0, f"a fully covered query should score high, got {full:.0f}", failures)

    exact = _coverage(_tokens("spanakopita"), _tokens("Spanakopita"))
    check(exact == 100.0, f"an exact match must be 100, got {exact:.0f}", failures)

    # --- the matcher end to end -------------------------------------------
    matcher = make_matcher()
    hit = matcher.best("Give me the recipe for Spanakopita")
    check(
        hit is not None and hit.title == "Spanakopita" and hit.score >= 95,
        f"conversational exact lookup failed: {hit}",
        failures,
    )
    hit = matcher.best("chiken alfredo")  # typo
    check(
        hit is not None and hit.title.startswith("Chicken Alfredo"),
        f"typo lookup failed: {hit}",
        failures,
    )
    hit = matcher.best("Paneer Butter Masala")
    check(
        hit is not None and hit.score < 80,
        f"an absent dish must not score as present: {hit}",
        failures,
    )

    # Several titles can cover the whole query; the closest one -- fewest extra
    # words -- must win the tie. Listed longest-first so row order cannot help.
    tie = TitleMatcher(pd.DataFrame({
        "parent_id": ["long", "exact"],
        "title": ["Chicken Stock with Ginger", "Chicken Stock"],
        "dup_rank": [0, 0],
    }))
    hit = tie.best("Chicken Stock")
    check(
        hit is not None and hit.parent_id == "exact",
        f"an exact title must win a coverage tie, got {hit}",
        failures,
    )
    # ...but a truncated name is the start of the LONGER title, and must get it.
    hit = tie.best("Chicken Stock with")
    check(
        hit is not None and hit.parent_id == "long",
        f"a typed prefix must win over a shorter title, got {hit}",
        failures,
    )

    # --- the gate ----------------------------------------------------------
    th = GateThresholds()

    strong = decide(FakeHit(0.487), FakeTitle(100.0), th)
    check(
        strong.source == SOURCE_DATASET and strong.parent_id == "title-parent",
        "a perfect title match must win even when the vector is weak (Spanakopita)",
        failures,
    )

    semantic = decide(FakeHit(0.80), FakeTitle(20.0), th)
    check(
        semantic.source == SOURCE_DATASET and semantic.parent_id == "vec-parent",
        "a strong vector must carry a paraphrase with no title match",
        failures,
    )

    both_weak = decide(FakeHit(0.596), FakeTitle(41.0), th)
    check(
        both_weak.source == SOURCE_WEB,
        "both signals weak must escalate, not guess (Paneer Butter Masala)",
        failures,
    )
    check(bool(both_weak.reasons), "every decision must explain itself", failures)

    supported = decide(FakeHit(0.50), FakeTitle(82.0), th)
    check(
        supported.source == SOURCE_DATASET,
        "a good title corroborated by a fair vector is a dataset hit",
        failures,
    )

    unsupported = decide(FakeHit(0.20), FakeTitle(82.0), th)
    check(
        unsupported.source == SOURCE_WEB,
        "a good title with no vector support must not be trusted alone",
        failures,
    )

    nothing = decide(None, None, th)
    check(nothing.source == SOURCE_WEB, "no hits at all must escalate", failures)

    total = 24
    if failures:
        print(f"FAILED {len(failures)}/{total}")
        print("\n".join(failures))
        return 1
    print(f"ok  {total}/{total} gate and lexical cases passed")
    return 0


def test_gate():
    assert run() == 0


if __name__ == "__main__":
    raise SystemExit(run())
