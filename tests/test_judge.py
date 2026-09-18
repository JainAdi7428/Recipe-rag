"""Bands, candidate assembly, and the judge's contract with a stubbed model.

No network and no index: the LLM is replaced by a fake that returns whatever
JSON the test wants. That is the point -- these tests pin down what the code
does with a verdict, including a hostile or malformed one. Whether the real
model gives *good* verdicts is a different question, measured with real calls by
scripts/eval_judge.py.

Run standalone:  python tests/test_judge.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from recipe_rag.gate import (
    BAND_AMBIGUOUS, BAND_CONFIDENT, BAND_HOPELESS, GateThresholds, band_of,
)
from recipe_rag.judge import Candidate, candidates_from, judge
from recipe_rag.llm import Usage

TH = GateThresholds()

PARENTS = pd.DataFrame({
    "parent_id": ["p0", "p1", "p2", "p3"],
    "title": [
        "Spanakopita",
        "Chicken Tikka Masala",
        "Chicken Alfredo with Zucchini Ribbons",
        "Simple Syrup",
    ],
    "diet": ["veg", "non-veg", "non-veg", "vegan"],
    "categories": ["Cheese, Appetizer", "Curry", "Pasta", "Syrup"],
    "ingredients": [
        ["spinach", "feta", "phyllo"],
        ["chicken", "yoghurt", "tomato"],
        ["chicken", "cream", "fettuccine"],
        ["sugar", "water"],
    ],
    "dup_rank": [0, 0, 0, 0],
})


@dataclass
class Hit:
    parent_id: str
    score: float = 0.0
    title: str = ""


class FakeLLM:
    """Returns a canned payload and records what it was asked."""

    def __init__(self, payload):
        self.payload = payload
        self.last_user = ""
        self.calls = 0

    def complete_json(self, system, user, purpose="judge", max_tokens=120):
        self.calls += 1
        self.last_user = user
        return self.payload, Usage(calls=1, input_tokens=400, output_tokens=25)


_TALLY = [0]


def check(condition: bool, message: str, failures: list[str]) -> None:
    _TALLY[0] += 1
    if not condition:
        failures.append(f"  {message}")


def run() -> int:
    failures: list[str] = []

    # --- bands: the thresholds the build was signed off with ---------------
    for vector, title, expected, note in [
        (0.20, 100.0, BAND_CONFIDENT, "exact title alone is decisive"),
        (0.487, 100.0, BAND_CONFIDENT, "the Spanakopita case"),
        (0.75, 10.0, BAND_CONFIDENT, "strong vector alone is decisive"),
        (0.662, 66.0, BAND_AMBIGUOUS, "creamy pasta: was going to web, now judged"),
        (0.50, 20.0, BAND_AMBIGUOUS, "inside the present/absent overlap"),
        (0.36, 0.0, BAND_AMBIGUOUS, "just above the vector floor"),
        (0.0, 56.0, BAND_AMBIGUOUS, "just above the title floor"),
        (0.30, 40.0, BAND_HOPELESS, "below both floors"),
        (0.0, 0.0, BAND_HOPELESS, "nothing retrieved"),
    ]:
        got = band_of(vector, title, TH)
        check(got == expected,
              f"band_of({vector}, {title}) -> {got}, want {expected} ({note})", failures)

    # A hopeless query must never reach the judge, and a confident one must
    # never pay for it. Those two facts are what keep the cost near zero.
    check(band_of(0.0, 0.0, TH) == BAND_HOPELESS, "empty result -> hopeless", failures)

    # --- candidate assembly: the union, not just the vector list -----------
    ranked = [Hit("p1", 0.55), Hit("p2", 0.51)]
    titles = [Hit("p0", 100.0, "Spanakopita")]
    cands = candidates_from(ranked, titles, PARENTS, vector_n=8, title_n=3)
    ids = [c.parent_id for c in cands]
    check(ids == ["p1", "p2", "p0"],
          f"title hits must be appended to the vector pool, got {ids}", failures)
    check(cands[0].vector_score == 0.55, "vector scores must carry through", failures)
    check(cands[2].vector_score == 0.0,
          "a title-only candidate has no vector score", failures)
    check(cands[0].title == "Chicken Tikka Masala", "titles resolved from parents", failures)

    # deduplication: the same parent from both retrievers appears once
    cands = candidates_from([Hit("p0", 0.6)], [Hit("p0", 90.0)], PARENTS)
    check(len(cands) == 1, f"a parent found by both must appear once, got {len(cands)}", failures)

    # a parent id missing from the table must be skipped, not crash
    cands = candidates_from([Hit("ghost", 0.9), Hit("p0", 0.6)], [], PARENTS)
    check([c.parent_id for c in cands] == ["p0"],
          "unknown parent ids must be dropped silently", failures)

    # --- the prompt actually contains what we claim ------------------------
    llm = FakeLLM({"type": "named", "choice": 1, "dish": "Spanakopita", "reason": "same dish"})
    pool = candidates_from([Hit("p0", 0.487)], [], PARENTS)
    judge("Spanakopita", pool, llm, TH)
    check("Spanakopita" in llm.last_user, "the prompt must list candidate titles", failures)
    check("spinach" in llm.last_user, "the prompt must list ingredients", failures)
    check("[veg]" in llm.last_user, "the prompt must carry the diet label", failures)

    # --- accepting ---------------------------------------------------------
    verdict = judge("Spanakopita", pool, llm, TH)
    check(verdict.accepted, "a valid choice must be accepted", failures)
    check(verdict.parent_id == "p0", f"parent_id -> {verdict.parent_id}", failures)
    check(verdict.usage.calls == 1, "usage must be reported", failures)

    # --- refusing ----------------------------------------------------------
    for payload, note in [
        ({"type": "named", "choice": None, "dish": "", "reason": "not here"}, "null choice"),
        ({"type": "named", "choice": 0, "dish": "", "reason": ""}, "index 0 is out of range"),
        ({"type": "named", "choice": 99, "dish": "", "reason": ""}, "index past the list"),
        ({"type": "named", "choice": "1", "dish": "", "reason": ""}, "a string is not an index"),
        ({"reason": "no fields at all"}, "missing keys"),
        ({}, "empty object"),
    ]:
        verdict = judge("Masala Dosa", pool, FakeLLM(payload), TH)
        check(not verdict.accepted, f"must refuse: {note}", failures)
        check(verdict.parent_id is None, f"refusal must carry no parent: {note}", failures)

    # An unrecognised type falls back to the stricter bar, never the looser one.
    llm = FakeLLM({"type": "nonsense", "choice": 1, "dish": "x", "reason": ""})
    verdict = judge("Spanakopita", pool, llm, TH)
    check(verdict.query_type == "named",
          f"unknown type must default to 'named', got {verdict.query_type!r}", failures)

    # --- the two-signal veto ----------------------------------------------
    # A named request, the chosen title shares nothing with the query, AND the
    # vector is weak: two independent signals against one, so it is vetoed.
    weak = candidates_from([Hit("p3", 0.20)], [], PARENTS)   # "Simple Syrup"
    llm = FakeLLM({"type": "named", "choice": 1, "dish": "Simple Syrup",
                   "reason": "seems related"})
    verdict = judge("Beef Rendang", weak, llm, TH)
    check(verdict.vetoed and not verdict.accepted,
          "a named pick with no lexical and no vector support must be vetoed", failures)

    # The veto must NOT fire when the vector agrees -- the LLM is then the only
    # thing objecting to, and it is entitled to see a synonym we cannot.
    strong = candidates_from([Hit("p3", 0.72)], [], PARENTS)
    verdict = judge("Beef Rendang", strong, FakeLLM(
        {"type": "named", "choice": 1, "dish": "Simple Syrup", "reason": "x"}), TH)
    check(verdict.accepted and not verdict.vetoed,
          "a strong vector must protect a lexically-distant pick", failures)

    # This is the case the veto must never break: same dish, different name.
    span = candidates_from([Hit("p0", 0.30)], [], PARENTS)
    verdict = judge("something creamy with pasta", span, FakeLLM(
        {"type": "descriptive", "choice": 1, "dish": "Spanakopita", "reason": "x"}), TH)
    check(verdict.accepted, "the veto must not apply to descriptive requests", failures)

    # --- no candidates: refuse without spending a call ---------------------
    llm = FakeLLM({"type": "named", "choice": 1, "dish": "x", "reason": ""})
    verdict = judge("anything", [], llm, TH)
    check(not verdict.accepted and llm.calls == 0,
          "an empty candidate list must refuse without calling the model", failures)

    # --- rendering doesn't explode on missing fields -----------------------
    bare = Candidate(parent_id="x", title="Untitled")
    check("1. Untitled" in bare.render(1), "a bare candidate must still render", failures)

    if failures:
        print(f"FAIL {len(failures)}/{_TALLY[0]} judge and band cases")
        for line in failures:
            print(line)
        return 1
    print(f"ok  {_TALLY[0]}/{_TALLY[0]} judge and band cases passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
