"""Child-to-parent score aggregation.

A recipe is split across ~3.67 child chunks, so a query matches *chunks* but a
user wants *recipes*. Several children of the same parent routinely appear in
one result set, and their scores have to be folded into a single parent score.

Phase 1 did this implicitly: results came back in distance order and the first
chunk seen for each parent won, which is max-pooling arrived at by accident
rather than by choice. This module makes the choice explicit and measurable.

The strategies differ in what they assume a good match looks like:

  max       one strongly matching chunk is enough. Robust to a recipe having
            many chunks that are irrelevant to the query -- which is the norm,
            since a title query has no reason to match the directions.
  mean      the whole recipe should be on-topic. Penalises long recipes, whose
            extra direction chunks drag the average down for no good reason.
  sum_topk  a compromise: reward corroboration across a couple of chunks
            without letting chunk count dominate.
  section   max, but weighted so that a hit on the summary card (which is the
            purpose-built title-lookup target) outranks an incidental hit deep
            in a directions chunk at the same raw similarity.

Measured by `scripts/eval_retrieval.py` over 1,500 exact-title queries, two
seeds (Recall@1):

    max       96.6 / 96.1     <- default
    sum_topk  96.1 / 95.9
    section   96.0 / 95.8
    mean      94.9 / 95.0

The correct recipe contributes more than one chunk to the candidate set in 99%
of queries, so this really is being exercised. `mean` is reliably worst, by
about 1.2 points, exactly as its chunk-count penalty predicts. The other three
sit within roughly half a point of each other, which is inside the noise at this
sample size -- so `max` is the default on the strength of being consistently at
or near the top, not because it is significantly better than `sum_topk`.

Worth re-measuring once queries are natural language rather than exact titles;
a discursive query spread across ingredients and directions is the case where
corroboration-based aggregation could start to pay off.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .config import SECTION_DIRECTIONS, SECTION_INGREDIENTS, SECTION_SUMMARY

# Applied by the "section" strategy. The summary card carries the title, the
# description and the categories, so it is the chunk a title query should
# match; directions text is the weakest evidence of what a recipe *is*.
SECTION_WEIGHTS = {
    SECTION_SUMMARY: 1.00,
    SECTION_INGREDIENTS: 0.97,
    SECTION_DIRECTIONS: 0.94,
}

STRATEGIES = ("max", "mean", "sum_topk", "section")


@dataclass
class ParentHit:
    parent_id: str
    score: float
    n_chunks_matched: int
    best_section: str
    chunk_scores: list[tuple[str, float]] = field(default_factory=list)


def aggregate(
    hits: list[tuple[str, str, float]],
    strategy: str = "max",
    topk: int = 2,
) -> list[ParentHit]:
    """Fold (parent_id, section, similarity) rows into ranked parents.

    Similarity is 1 - cosine distance, so higher is better.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}, expected one of {STRATEGIES}")

    grouped: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for parent_id, section, score in hits:
        grouped[parent_id].append((section, score))

    parents: list[ParentHit] = []
    for parent_id, entries in grouped.items():
        ordered = sorted(entries, key=lambda pair: pair[1], reverse=True)
        scores = [score for _, score in ordered]

        if strategy == "max":
            value = scores[0]
        elif strategy == "mean":
            value = sum(scores) / len(scores)
        elif strategy == "sum_topk":
            top = scores[:topk]
            # Averaged, not summed, so a recipe is not rewarded merely for
            # having more chunks than its competitors.
            value = sum(top) / len(top)
        else:  # section
            value = max(
                score * SECTION_WEIGHTS.get(section, 0.94) for section, score in ordered
            )

        parents.append(
            ParentHit(
                parent_id=parent_id,
                score=value,
                n_chunks_matched=len(entries),
                best_section=ordered[0][0],
                chunk_scores=ordered,
            )
        )

    parents.sort(key=lambda hit: hit.score, reverse=True)
    return parents


def collapse_variants(hits: list[ParentHit], dup_group_of: dict[str, str]) -> list[ParentHit]:
    """Keep only the best-scoring parent per duplicate-title group.

    Without this, a query for "chicken stock" returns up to 16 near-identical
    variants and crowds out everything else.
    """
    seen: set[str] = set()
    kept: list[ParentHit] = []
    for hit in hits:
        group = dup_group_of.get(hit.parent_id, "")
        if group:
            if group in seen:
                continue
            seen.add(group)
        kept.append(hit)
    return kept
