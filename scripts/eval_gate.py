"""Grade the existence gate: dataset vs web, on a labelled query set.

Builds its own ground truth, free and reproducibly:

  PRESENT  sampled recipe titles, in four shapes -- verbatim, with typos
           injected, wrapped in conversational boilerplate, and truncated to
           the first few words. The correct answer is known by construction.
  ABSENT   a curated list of dishes an Epicurious corpus genuinely lacks
           (largely Indian, East and South-East Asian), plus fabricated names
           that cannot exist anywhere. Any entry that turns out to have a close
           title in the dataset is dropped as contaminated, and reported.

The two error types are not equally bad:

  present -> web       a wasted web call. Costs latency and an API quota.
  absent  -> dataset   a FABRICATED SOURCE. The system answers "Paneer Butter
                       Masala" with a panna cotta and calls it internal data.

The second is what the spec penalises and what destroys trust, so the sweep
reports both and ranks configurations with that asymmetry in mind.

    python scripts/eval_gate.py --sample 250
    python scripts/eval_gate.py --sample 250 --sweep
    python scripts/eval_gate.py --sample 250 --bands
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recipe_rag import config, store
from recipe_rag.gate import GateThresholds, decide
from recipe_rag.lexical import TitleMatcher
from recipe_rag.retrieve import aggregate, collapse_variants

# Dishes this corpus does not cover. Verified at runtime, not assumed.
ABSENT_DISHES = [
    "Paneer Butter Masala", "Masala Dosa", "Hyderabadi Biryani", "Pani Puri",
    "Chole Bhature", "Rajma Chawal", "Dal Makhani", "Idli Sambar", "Vada Pav",
    "Misal Pav", "Pav Bhaji", "Baingan Bharta", "Malai Kofta", "Palak Paneer",
    "Chicken 65", "Gulab Jamun", "Rasmalai", "Jalebi", "Dhokla", "Upma",
    "Bisi Bele Bath", "Puran Poli", "Undhiyu", "Kadai Paneer", "Shahi Paneer",
    "Bibimbap", "Tteokbokki", "Japchae", "Okonomiyaki", "Takoyaki",
    "Xiao Long Bao", "Char Siu Bao", "Nasi Goreng", "Beef Rendang", "Laksa",
    "Bun Cha", "Banh Xeo", "Khao Soi", "Som Tam Thai", "Adobo sa Gata",
    # Fabricated: these cannot exist in any corpus.
    "Quantum Blockchain Smoothie", "Deep Fried Moon Rocks",
    "Unicorn Tears Consomme", "Gravitational Lasagna", "Photosynthesis Pie",
    "Recursive Onion Soup", "Tungsten Carbide Brownies",
]

CONVERSATIONAL = [
    "Give me the recipe for {}",
    "How do I make {}?",
    "Recipe for {}",
    "I want to make {} tonight",
]


def inject_typos(text: str, rng: random.Random, n: int = 2) -> str:
    """Realistic keyboard-ish damage: drop, swap or duplicate a character."""
    chars = list(text)
    for _ in range(n):
        if len(chars) < 4:
            break
        i = rng.randrange(1, len(chars) - 1)
        action = rng.choice(("drop", "swap", "double"))
        if action == "drop":
            chars.pop(i)
        elif action == "swap":
            chars[i], chars[i - 1] = chars[i - 1], chars[i]
        else:
            chars.insert(i, chars[i])
    return "".join(chars)


def build_queries(parents, sample: int, seed: int):
    """Return (query, expected_parent_id or None) pairs."""
    rng = random.Random(seed)
    unique = parents[parents.dup_group == ""]
    picks = unique.sample(min(sample, len(unique)), random_state=seed)

    cases: list[tuple[str, str, str | None]] = []  # (kind, query, truth)
    for row in picks.itertuples():
        title = row.title
        cases.append(("exact", title, row.parent_id))
        cases.append(("typo", inject_typos(title, rng), row.parent_id))
        cases.append(("chatty", rng.choice(CONVERSATIONAL).format(title), row.parent_id))
        words = title.split()
        if len(words) >= 3:
            cases.append((("partial"), " ".join(words[:3]), row.parent_id))
    return cases


def verify_absent(dishes: list[str], matcher: TitleMatcher, limit: float = 88.0):
    """Drop 'absent' dishes that actually have a close title in the dataset."""
    clean, contaminated = [], []
    for dish in dishes:
        hit = matcher.best(dish)
        if hit and hit.score >= limit:
            contaminated.append((dish, hit.title, hit.score))
        else:
            clean.append(dish)
    return clean, contaminated


def run_queries(collection, matcher, queries, fetch, dup_group_of,
                keep: int = 10, title_keep: int = 5):
    """Score every query once; thresholds are applied afterwards.

    Keeps the top `keep` vector parents and top `title_keep` title parents, not
    just the winner. The extra rows cost nothing here and are what the band
    report needs: an LLM judge can only rescue a recipe that retrieval actually
    put in front of it, so "is the truth anywhere in the candidate pool" is a
    separate and more important number than "is the truth at rank 1".
    """
    scored = []
    batch = max(1, 8000 // max(fetch, 1))
    texts = [q for _, q, _ in queries]
    metas, dists = [], []
    for start in range(0, len(texts), batch):
        page = collection.query(
            query_texts=texts[start : start + batch],
            n_results=fetch,
            include=["metadatas", "distances"],
        )
        metas.extend(page["metadatas"])
        dists.extend(page["distances"])

    for (kind, query, truth), meta_row, dist_row in zip(queries, metas, dists):
        rows = [
            (m["parent_id"], m["section"], 1.0 - d)
            for m, d in zip(meta_row, dist_row)
        ]
        ranked = collapse_variants(aggregate(rows, "max"), dup_group_of)
        title_hits = matcher.match(query, limit=title_keep)
        scored.append(
            {
                "kind": kind,
                "query": query,
                "truth": truth,
                "vector_hit": ranked[0] if ranked else None,
                "title_hit": title_hits[0] if title_hits else None,
                "ranked": ranked[:keep],
                "title_hits": title_hits,
            }
        )
    return scored


def evaluate(scored, thresholds):
    """Confusion matrix plus recipe-identity accuracy."""
    stats = {
        "present_dataset_right": 0, "present_dataset_wrong": 0,
        "present_web": 0, "absent_dataset": 0, "absent_web": 0,
    }
    per_kind: dict[str, list[int]] = {}
    for item in scored:
        decision = decide(item["vector_hit"], item["title_hit"], thresholds)
        truth = item["truth"]
        if truth is None:
            stats["absent_dataset" if decision.is_dataset else "absent_web"] += 1
            continue
        bucket = per_kind.setdefault(item["kind"], [0, 0])
        bucket[1] += 1
        if not decision.is_dataset:
            stats["present_web"] += 1
        elif decision.parent_id == truth:
            stats["present_dataset_right"] += 1
            bucket[0] += 1
        else:
            stats["present_dataset_wrong"] += 1
    return stats, per_kind


def summarise(stats):
    present = (
        stats["present_dataset_right"] + stats["present_dataset_wrong"]
        + stats["present_web"]
    )
    absent = stats["absent_dataset"] + stats["absent_web"]
    return {
        "recall": 100 * (stats["present_dataset_right"] + stats["present_dataset_wrong"]) / max(present, 1),
        "identity": 100 * stats["present_dataset_right"] / max(present, 1),
        "false_attribution": 100 * stats["absent_dataset"] / max(absent, 1),
        "present": present,
        "absent": absent,
    }


def _percentiles(values, points=(1, 5, 10, 25, 50, 75, 90, 99)):
    ordered = sorted(values)
    if not ordered:
        return {p: float("nan") for p in points}
    return {
        p: ordered[min(len(ordered) - 1, int(p / 100 * len(ordered)))]
        for p in points
    }


def candidate_pool(item, vector_n: int, title_n: int) -> list[str]:
    """Parent ids the judge would be shown: vector top-N union title top-N."""
    ids = [hit.parent_id for hit in item["ranked"][:vector_n]]
    for hit in item["title_hits"][:title_n]:
        if hit.parent_id not in ids:
            ids.append(hit.parent_id)
    return ids


def band_report(scored, t_strong, v_strong, t_floor, v_floor,
                vector_n, title_n) -> None:
    """How traffic splits across confident / ambiguous / hopeless.

    This is what decides whether an LLM judge is an edge case or the main code
    path, and what it costs. It needs no LLM to compute -- the bands are just a
    partition of scores the gate evaluation already produced.
    """
    present = [i for i in scored if i["truth"] is not None]
    absent = [i for i in scored if i["truth"] is None]

    print("vector top-1 score distribution")
    print(f"  {'pct':>5}" + "".join(f"{p:>7}" for p in (1, 5, 10, 25, 50, 75, 90, 99)))
    for label, group in (("present", present), ("absent", absent)):
        values = [i["vector_hit"].score if i["vector_hit"] else 0.0 for i in group]
        pcts = _percentiles(values)
        print(f"  {label:>5}" + "".join(f"{pcts[p]:>7.3f}" for p in sorted(pcts)))
    print("  ^ where these two overlap is exactly the region a single threshold"
          "\n    cannot separate, and the judge has to.\n")

    def band_of(item) -> str:
        vector = item["vector_hit"].score if item["vector_hit"] else 0.0
        title = item["title_hit"].score if item["title_hit"] else 0.0
        if title >= t_strong or vector >= v_strong:
            return "confident"
        if title >= t_floor or vector >= v_floor:
            return "ambiguous"
        return "hopeless"

    print(f"bands   confident: title>={t_strong:.0f} or vector>={v_strong:.2f}")
    print(f"        hopeless : title< {t_floor:.0f} and vector< {v_floor:.2f}")
    print(f"        judge pool: vector top-{vector_n} + title top-{title_n}\n")
    print(f"  {'band':<11}{'total':>7}{'%':>7}{'present':>9}{'absent':>8}"
          f"{'rank1 ok':>10}{'in pool':>9}")
    print("  " + "-" * 61)

    for band in ("confident", "ambiguous", "hopeless"):
        rows = [i for i in scored if band_of(i) == band]
        rows_present = [i for i in rows if i["truth"] is not None]
        rank1 = sum(
            1 for i in rows_present
            if i["vector_hit"] and i["vector_hit"].parent_id == i["truth"]
        )
        pooled = sum(
            1 for i in rows_present
            if i["truth"] in candidate_pool(i, vector_n, title_n)
        )
        n_present = len(rows_present)
        print(
            f"  {band:<11}{len(rows):>7}{100 * len(rows) / max(len(scored), 1):>6.1f}%"
            f"{n_present:>9}{len(rows) - n_present:>8}"
            f"{100 * rank1 / max(n_present, 1):>9.1f}%"
            f"{100 * pooled / max(n_present, 1):>8.1f}%"
        )

    ambiguous = [i for i in scored if band_of(i) == "ambiguous"]
    amb_present = sum(1 for i in ambiguous if i["truth"] is not None)
    hopeless_present = sum(
        1 for i in scored if band_of(i) == "hopeless" and i["truth"] is not None
    )
    print(f"\n  'in pool' is the judge's ceiling: it can never rescue a recipe"
          f"\n  retrieval did not surface. 'rank1 ok' is roughly what we get today.")
    print(f"\n  judge would run on {len(ambiguous)} of {len(scored)} queries "
          f"({100 * len(ambiguous) / max(len(scored), 1):.1f}%)")
    print(f"    of those, {amb_present} are answerable from the dataset and "
          f"{len(ambiguous) - amb_present} are genuinely absent")
    per_call = config.llm_cost(120 + 55 * (vector_n + title_n), 30)
    print(f"    ~${per_call:.6f}/call  ->  ${per_call * len(ambiguous):.4f} for this run,"
          f" ${per_call * 100:.4f} per 100 judged queries")
    if hopeless_present:
        print(f"\n  WARNING: {hopeless_present} answerable queries fall in 'hopeless'"
              f" and would never reach the judge. Lower the floors.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=250)
    parser.add_argument("--fetch", type=int, default=40)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--sweep", action="store_true",
                        help="grid-search thresholds instead of testing defaults")
    parser.add_argument("--bands", action="store_true",
                        help="report confident/ambiguous/hopeless traffic split")
    parser.add_argument("--v-floor", type=float, default=0.35)
    parser.add_argument("--t-floor", type=float, default=55.0)
    parser.add_argument("--vector-n", type=int, default=8,
                        help="vector candidates shown to the judge")
    parser.add_argument("--title-n", type=int, default=3,
                        help="title candidates shown to the judge")
    args = parser.parse_args()

    store.require_index()
    collection = store.get_collection(store.get_client(), create=False)
    parents = store.load_parents()
    dup_group_of = dict(zip(parents.parent_id, parents.dup_group))
    matcher = TitleMatcher(parents)

    absent, contaminated = verify_absent(ABSENT_DISHES, matcher)
    if contaminated:
        print("dropped as contaminated (a close title exists in the dataset):")
        for dish, title, score in contaminated:
            print(f"   {dish:<28} ~ {title[:44]:<44} {score:.0f}")
        print()

    queries = build_queries(parents, args.sample, args.seed)
    queries += [("absent", dish, None) for dish in absent]
    print(
        f"{len(queries)} queries: "
        f"{sum(1 for k, _, t in queries if t is not None)} present, "
        f"{len(absent)} absent\n"
    )

    scored = run_queries(
        collection, matcher, queries, args.fetch, dup_group_of,
        keep=max(args.vector_n, 10), title_keep=max(args.title_n, 5),
    )

    if args.bands:
        defaults = GateThresholds()
        band_report(
            scored, defaults.title_strong, defaults.vector_strong,
            args.t_floor, args.v_floor, args.vector_n, args.title_n,
        )
        return 0

    if not args.sweep:
        thresholds = GateThresholds()
        stats, per_kind = evaluate(scored, thresholds)
        s = summarise(stats)
        print(f"thresholds: {thresholds}\n")
        print(f"  present -> dataset, right recipe : {stats['present_dataset_right']:>5}")
        print(f"  present -> dataset, WRONG recipe : {stats['present_dataset_wrong']:>5}")
        print(f"  present -> web (missed)          : {stats['present_web']:>5}")
        print(f"  absent  -> web (correct)         : {stats['absent_web']:>5}")
        print(f"  absent  -> dataset (FABRICATED)  : {stats['absent_dataset']:>5}")
        print(f"\n  dataset recall     {s['recall']:.1f}%")
        print(f"  correct identity   {s['identity']:.1f}%")
        print(f"  false attribution  {s['false_attribution']:.1f}%   <- keep near zero")
        print("\n  by query shape:")
        for kind, (right, total) in sorted(per_kind.items()):
            print(f"    {kind:<9} {100 * right / max(total, 1):>5.1f}%  ({right}/{total})")
        return 0

    print(f"{'t_strong':>8} {'t_supp':>7} {'v_supp':>7} {'v_strong':>9}"
          f" {'identity':>9} {'missed':>7} {'FABRIC':>7}")
    print("-" * 60)
    best = None
    for t_strong in (85.0, 88.0, 90.0, 93.0):
        for t_support in (70.0, 75.0, 80.0):
            for v_support in (0.45, 0.55, 0.65):
                for v_strong in (0.70, 0.75, 0.80):
                    th = GateThresholds(t_strong, t_support, v_support, v_strong)
                    stats, _ = evaluate(scored, th)
                    s = summarise(stats)
                    # Rank by identity accuracy, but treat any fabricated source
                    # as several times more costly than a wasted web call.
                    score = s["identity"] - 4.0 * s["false_attribution"]
                    if best is None or score > best[0]:
                        best = (score, th, s)
                    print(
                        f"{t_strong:>8.0f} {t_support:>7.0f} {v_support:>7.2f}"
                        f" {v_strong:>9.2f} {s['identity']:>8.1f}%"
                        f" {100 - s['recall']:>6.1f}% {s['false_attribution']:>6.1f}%"
                    )
    print("\nbest by (identity - 4 x false_attribution):")
    print(f"  {best[1]}")
    print(f"  identity {best[2]['identity']:.1f}%  "
          f"missed {100 - best[2]['recall']:.1f}%  "
          f"fabricated {best[2]['false_attribution']:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
