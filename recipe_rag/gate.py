"""The existence gate: does this recipe exist in the dataset, or go to the web?

This is the decision the spec grades most directly -- "correctly attribute the
source (internal, web, or not found), and avoid unnecessary web fallbacks" -- and
it cannot be made from vector similarity alone. Measured on the built index:

    present recipes : min 0.487   mean 0.724   max 0.845
    absent  recipes : min 0.314   mean 0.490   max 0.596
                                  OVERLAP of 0.108

Any single cosine threshold is wrong somewhere. Cut at 0.55 and Spanakopita --
which IS in the dataset and retrieves itself perfectly -- gets sent to the web.
Cut at 0.45 and "Paneer Butter Masala", which is absent, gets answered from a
panna cotta and labelled source=dataset. The second failure is the serious one:
it is a fabricated source attribution.

The fix is a second, independent signal. Dense vectors and fuzzy title matching
fail in different places, so their union covers what neither does alone:

    query                      vector   title    truth
    "Spanakopita"               0.487     100    present  <- title saves it
    "creamy pasta with chicken" 0.70+      ~30    present  <- vector saves it
    "Paneer Butter Masala"      0.596      ~41    absent   <- both weak, escalate

Policy: **prefer the web when uncertain.** The system claims a dataset source
only when one of the two signals is clearly strong. An unnecessary web call
costs a little latency; a wrong source attribution is the failure the spec
penalises and the one that erodes user trust.

Thresholds are calibrated by scripts/eval_gate.py, not guessed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

SOURCE_DATASET = "dataset"
SOURCE_WEB = "web"


@dataclass(frozen=True)
class GateThresholds:
    """Tuned by `scripts/eval_gate.py --sweep` over 108 configurations.

    Selected by maximising (identity accuracy - 4 x false-attribution rate),
    the weighting reflecting that a fabricated source is far worse than a
    wasted web call. The winning configuration beat the hand-picked starting
    point on every metric at once:

                        identity   missed   fabricated
        hand-picked        92.9%     2.2%        4.8%
        swept              94.9%     0.3%        0.0%

    Note `title_strong` came *down* from 90 to 85. That is only safe because
    the scorer changed: coverage-based matching makes a score of 85 mean "most
    of the query is accounted for", where WRatio's 85 could mean "one word out
    of three matched". `title_support` went the other way, 75 -> 80, which is
    what closes the remaining leaks.
    """

    title_strong: float = 85.0     # an exact/near-exact name match on its own
    title_support: float = 80.0    # a decent name match, needs vector agreement
    vector_support: float = 0.45   # the vector level that corroborates a name
    vector_strong: float = 0.70    # semantic match strong enough without a name

    # Below both floors there is nothing worth showing a judge. Kept separate
    # from the thresholds above because they answer a different question:
    # `*_strong` asks "good enough to answer with", the floors ask "good enough
    # to be worth looking at". Phase 1 conflated the two, which is why present
    # recipes with weak embeddings were being discarded outright.
    vector_floor: float = 0.35
    title_floor: float = 55.0


DEFAULT = GateThresholds()

BAND_CONFIDENT = "confident"    # answer from the dataset, no LLM call
BAND_AMBIGUOUS = "ambiguous"    # ask the judge
BAND_HOPELESS = "hopeless"      # go to the web, no LLM call


def band_of(
    vector_score: float,
    title_score: float,
    thresholds: GateThresholds = DEFAULT,
) -> str:
    """Which of the three bands a query's scores fall into.

    The bands exist because a single cutoff has to serve two incompatible
    purposes at once. Measured on this index, present and absent recipes overlap
    between roughly 0.49 and 0.60 cosine -- no threshold placed inside that
    overlap is right, it only chooses which way to be wrong.

    So the overlap is not resolved by a number. It is routed to a judge that can
    read the candidate titles and say whether any of them is actually the dish
    that was asked for. The confident and hopeless bands are the regions where
    the scores *are* decisive, and those cost nothing.
    """
    if title_score >= thresholds.title_strong or vector_score >= thresholds.vector_strong:
        return BAND_CONFIDENT
    if title_score >= thresholds.title_floor or vector_score >= thresholds.vector_floor:
        return BAND_AMBIGUOUS
    return BAND_HOPELESS


@dataclass
class GateDecision:
    source: str                     # SOURCE_DATASET | SOURCE_WEB
    confidence: float               # 0-1, how sure we are of `source`
    parent_id: str | None
    title: str | None
    vector_score: float
    title_score: float
    reasons: list[str] = field(default_factory=list)
    band: str = ""

    @property
    def is_dataset(self) -> bool:
        return self.source == SOURCE_DATASET


def decide(
    vector_hit,
    title_hit,
    thresholds: GateThresholds = DEFAULT,
) -> GateDecision:
    """Rule-based dataset/web decision, with the band stamped on it.

    This is phase 1's gate, unchanged in behaviour. It remains the fallback the
    pipeline drops back to whenever the judge cannot run -- no key, no credit,
    API timeout -- so that losing the LLM degrades the system to something that
    still works rather than breaking it.
    """
    decision = _decide_rules(vector_hit, title_hit, thresholds)
    decision.band = band_of(decision.vector_score, decision.title_score, thresholds)
    return decision


def _decide_rules(
    vector_hit,
    title_hit,
    thresholds: GateThresholds = DEFAULT,
) -> GateDecision:
    """Choose dataset or web from the best vector hit and the best title hit.

    `vector_hit` is a ParentHit from retrieve.aggregate (or None); `title_hit` is
    a TitleHit from lexical.TitleMatcher (or None). They may point at different
    recipes -- when a strong title match disagrees with the vector's pick, the
    title wins, because that is precisely the case vectors handle badly.
    """
    vector_score = vector_hit.score if vector_hit else 0.0
    title_score = title_hit.score if title_hit else 0.0
    reasons: list[str] = []

    # 1. A near-exact name match is decisive on its own. This is the Spanakopita
    #    case: 0.487 by vector, 100 by title.
    if title_score >= thresholds.title_strong:
        reasons.append(f"title match {title_score:.0f} >= {thresholds.title_strong:.0f}")
        return GateDecision(
            source=SOURCE_DATASET,
            confidence=min(0.99, 0.70 + title_score / 400),
            parent_id=title_hit.parent_id,
            title=title_hit.title,
            vector_score=vector_score,
            title_score=title_score,
            reasons=reasons,
        )

    # 2. A good-but-not-certain name match, corroborated by the vector.
    if (
        title_score >= thresholds.title_support
        and vector_score >= thresholds.vector_support
    ):
        reasons.append(
            f"title {title_score:.0f} supported by vector {vector_score:.3f}"
        )
        return GateDecision(
            source=SOURCE_DATASET,
            confidence=0.75,
            parent_id=title_hit.parent_id,
            title=title_hit.title,
            vector_score=vector_score,
            title_score=title_score,
            reasons=reasons,
        )

    # 3. No usable name, but the meaning matches strongly. This is the
    #    "creamy pasta with chicken" case, where no title contains those words.
    if vector_score >= thresholds.vector_strong:
        reasons.append(f"vector {vector_score:.3f} >= {thresholds.vector_strong:.2f}")
        return GateDecision(
            source=SOURCE_DATASET,
            confidence=0.70,
            parent_id=vector_hit.parent_id if vector_hit else None,
            title=None,
            vector_score=vector_score,
            title_score=title_score,
            reasons=reasons,
        )

    # 4. Neither signal is convincing -> escalate rather than guess.
    reasons.append(
        f"title {title_score:.0f} and vector {vector_score:.3f} both below "
        f"threshold; escalating rather than claiming a dataset source"
    )
    return GateDecision(
        source=SOURCE_WEB,
        confidence=0.60,
        parent_id=None,
        title=None,
        vector_score=vector_score,
        title_score=title_score,
        reasons=reasons,
    )
