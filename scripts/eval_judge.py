"""Grade the LLM judge against the rule gate, on the same labelled queries.

    python scripts/eval_judge.py --max-calls 120            # ~$0.02
    python scripts/eval_judge.py --max-calls 200 --no-veto  # ablate the veto

The judge's whole purpose is to rescue recipes the dataset holds but the scores
could not confirm. The danger is that it rescues things that were never there.
Both are measured here, because arguing about which is more likely is useless
and counting is cheap.

Three rules make the numbers honest:

  **Only ambiguous-band queries are judged.** That is the only band the judge
  ever runs in. Including confident queries would pad every rate with cases the
  scores already answered, and would hide exactly the failures we are hunting.

  **Absent dishes are split into hard and easy.** "Quantum Blockchain Smoothie"
  is a free win that no judge fails; "Masala Dosa" against "Chicken Tikka
  Masala" is the real test. Over-acceptance on the hard set is the number that
  decides whether this is safe to ship.

  **The rule gate is scored on the identical queries**, so every figure has a
  baseline. "The judge is 96% accurate" means nothing on its own.

The four outcomes, and what each costs:

    present -> right recipe   correct
    present -> wrong recipe   MISATTRIBUTION, the worst outcome: a confident
                              answer to a question it did not answer
    present -> none           under-accept; a wasted web call, recoverable
    absent  -> a recipe       OVER-ACCEPT; a fabricated source
    absent  -> none           correct
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recipe_rag import config, gate, store
from recipe_rag import llm as llm_module
from recipe_rag.gate import BAND_AMBIGUOUS, GateThresholds
from recipe_rag.judge import candidates_from, judge
from recipe_rag.lexical import TitleMatcher
from recipe_rag.llm import LLM, LLMUnavailable

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_gate import ABSENT_DISHES, build_queries, run_queries, verify_absent

# Absent dishes chosen to collide with this corpus rather than to be obviously
# fictional. Every one shares a word, a cuisine or a technique with something
# Epicurious does publish, which is what makes them worth spending calls on.
HARD_NEGATIVES = [
    "Paneer Tikka Masala", "Butter Chicken Masala", "Egg Curry Masala",
    "Aloo Gobi", "Gobi Manchurian", "Bombay Potato Masala", "Chicken Chettinad",
    "Mutton Rogan Josh", "Prawn Balchao", "Fish Moilee", "Kadhi Pakora",
    "Sarson da Saag", "Litti Chokha", "Thalipeeth", "Pesarattu",
    "Beef Bulgogi Bibimbap", "Pork Sisig", "Chicken Inasal", "Kare Kare",
    "Lumpia Shanghai", "Pad Kee Mao", "Massaman Beef Curry", "Gaeng Som",
    "Cha Ca La Vong", "Bun Bo Hue", "Mie Goreng", "Ayam Penyet",
    "Soto Ayam", "Rendang Padang", "Sambal Goreng Tempeh",
]


def confusion(rows) -> dict:
    stats = {
        "present_right": 0, "present_wrong": 0, "present_none": 0,
        "absent_hard_accept": 0, "absent_hard_none": 0,
        "absent_easy_accept": 0, "absent_easy_none": 0,
    }
    for row in rows:
        chosen, truth, hard = row["chosen"], row["truth"], row["hard"]
        if truth is not None:
            if chosen is None:
                stats["present_none"] += 1
            elif chosen == truth:
                stats["present_right"] += 1
            else:
                stats["present_wrong"] += 1
        else:
            bucket = "hard" if hard else "easy"
            key = f"absent_{bucket}_" + ("accept" if chosen else "none")
            stats[key] += 1
    return stats


def rates(stats) -> dict:
    present = stats["present_right"] + stats["present_wrong"] + stats["present_none"]
    hard = stats["absent_hard_accept"] + stats["absent_hard_none"]
    easy = stats["absent_easy_accept"] + stats["absent_easy_none"]
    absent = hard + easy
    return {
        "identity": 100 * stats["present_right"] / max(present, 1),
        "misattribute": 100 * stats["present_wrong"] / max(present, 1),
        "under_accept": 100 * stats["present_none"] / max(present, 1),
        "over_accept": 100 * (stats["absent_hard_accept"] + stats["absent_easy_accept"]) / max(absent, 1),
        "over_accept_hard": 100 * stats["absent_hard_accept"] / max(hard, 1),
        "n_present": present, "n_hard": hard, "n_easy": easy,
    }


def report(name: str, stats: dict) -> dict:
    r = rates(stats)
    print(f"\n  {name}")
    print(f"    present -> right recipe    {stats['present_right']:>4}   "
          f"{r['identity']:>5.1f}%")
    print(f"    present -> WRONG recipe    {stats['present_wrong']:>4}   "
          f"{r['misattribute']:>5.1f}%   <- misattribution")
    print(f"    present -> none (to web)   {stats['present_none']:>4}   "
          f"{r['under_accept']:>5.1f}%   <- under-accept")
    print(f"    absent  -> accepted        "
          f"{stats['absent_hard_accept'] + stats['absent_easy_accept']:>4}   "
          f"{r['over_accept']:>5.1f}%   <- OVER-ACCEPT")
    print(f"      of which hard negatives  {stats['absent_hard_accept']:>4}   "
          f"{r['over_accept_hard']:>5.1f}%   of {r['n_hard']} hard")
    return r


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=150)
    parser.add_argument("--fetch", type=int, default=40)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-calls", type=int, default=120,
                        help="cap on LLM calls, so a run cannot surprise the budget")
    parser.add_argument("--vector-n", type=int, default=8)
    parser.add_argument("--title-n", type=int, default=3)
    parser.add_argument("--no-veto", action="store_true",
                        help="disable the two-signal veto, to measure its effect")
    parser.add_argument("--hard-title", type=float, default=50.0,
                        help="an absent dish scoring above this counts as a hard negative")
    args = parser.parse_args()

    thresholds = GateThresholds()
    store.require_index()
    collection = store.get_collection(store.get_client(), create=False)
    parents = store.load_parents()
    dup_group_of = dict(zip(parents.parent_id, parents.dup_group))
    matcher = TitleMatcher(parents)

    absent, contaminated = verify_absent(ABSENT_DISHES + HARD_NEGATIVES, matcher)
    if contaminated:
        print(f"dropped {len(contaminated)} 'absent' dishes that do exist here:")
        for dish, title, score in contaminated[:8]:
            print(f"   {dish:<26} ~ {title[:40]:<40} {score:.0f}")
        print()

    queries = build_queries(parents, args.sample, args.seed)
    queries += [("absent", dish, None) for dish in absent]
    scored = run_queries(
        collection, matcher, queries, args.fetch, dup_group_of,
        keep=max(args.vector_n, 10), title_keep=max(args.title_n, 5),
    )

    # The judge only ever runs in the ambiguous band, so that is the only place
    # it can be fairly measured.
    ambiguous = [
        item for item in scored
        if gate.band_of(
            item["vector_hit"].score if item["vector_hit"] else 0.0,
            item["title_hit"].score if item["title_hit"] else 0.0,
            thresholds,
        ) == BAND_AMBIGUOUS
    ]
    print(f"{len(scored)} queries scored, {len(ambiguous)} in the ambiguous band "
          f"({100 * len(ambiguous) / max(len(scored), 1):.1f}%)")

    rng = random.Random(args.seed)
    rng.shuffle(ambiguous)
    # Absent queries are the scarce, expensive-to-get-wrong half, so they are
    # kept in full and the present half is trimmed to fit the call budget.
    absent_rows = [i for i in ambiguous if i["truth"] is None]
    present_rows = [i for i in ambiguous if i["truth"] is not None]
    room = max(args.max_calls - len(absent_rows), 0)
    selected = absent_rows + present_rows[:room]
    if not selected:
        print("nothing landed in the ambiguous band; nothing to judge.")
        return 0
    print(f"judging {len(selected)} of them "
          f"({len(absent_rows)} absent, {min(room, len(present_rows))} present)\n")

    llm = LLM()
    if not llm.available:
        print("no usable OPENAI_API_KEY -- run scripts/check_llm.py first.")
        return 1

    baseline_rows, judged_rows = [], []
    for index, item in enumerate(selected, 1):
        title_score = item["title_hit"].score if item["title_hit"] else 0.0
        hard = item["truth"] is None and title_score >= args.hard_title

        decision = gate.decide(item["vector_hit"], item["title_hit"], thresholds)
        baseline_rows.append({
            "chosen": decision.parent_id if decision.is_dataset else None,
            "truth": item["truth"], "hard": hard,
        })

        candidates = candidates_from(
            item["ranked"], item["title_hits"], parents, args.vector_n, args.title_n
        )
        try:
            verdict = judge(item["query"], candidates, llm, thresholds,
                            veto=not args.no_veto)
        except LLMUnavailable as exc:
            print(f"\nstopped at {index}: {exc}")
            break
        judged_rows.append({
            "chosen": verdict.parent_id if verdict.accepted else None,
            "truth": item["truth"], "hard": hard,
            "vetoed": verdict.vetoed, "type": verdict.query_type,
            "query": item["query"], "title": verdict.title,
        })
        if index % 10 == 0 or index == len(selected):
            print(f"  {index}/{len(selected)}  "
                  f"${llm_module.LEDGER.total.cost_usd:.5f}", end="\r", flush=True)

    baseline_rows = baseline_rows[:len(judged_rows)]
    print(" " * 40, end="\r")
    print(f"\n{'=' * 68}")
    base = report("rule gate (baseline)", confusion(baseline_rows))
    new = report("LLM judge", confusion(judged_rows))

    print(f"\n  {'':<22}{'baseline':>10}{'judge':>10}{'change':>10}")
    for label, key, good in (
        ("correct identity", "identity", +1),
        ("misattribution", "misattribute", -1),
        ("under-accept", "under_accept", -1),
        ("over-accept (all)", "over_accept", -1),
        ("over-accept (hard)", "over_accept_hard", -1),
    ):
        delta = new[key] - base[key]
        mark = "  better" if delta * good > 0.05 else ("  WORSE" if delta * good < -0.05 else "")
        print(f"  {label:<22}{base[key]:>9.1f}%{new[key]:>9.1f}%{delta:>+9.1f}{mark}")

    vetoed = sum(1 for row in judged_rows if row["vetoed"])
    named = sum(1 for row in judged_rows if row["type"] == "named")
    print(f"\n  veto fired on {vetoed} verdict(s); "
          f"{named}/{len(judged_rows)} classified as named requests")

    wrong = [r for r in judged_rows if r["truth"] is None and r["chosen"]]
    if wrong:
        print(f"\n  over-accepted (these are the ones to read):")
        for row in wrong[:10]:
            print(f"    {row['query'][:34]:<34} -> {str(row['title'])[:32]}")

    print(f"\n{llm_module.LEDGER.report()}")
    print(f"  per judged query  ${llm_module.LEDGER.total.cost_usd / max(len(judged_rows), 1):.6f}")
    print(f"  100 queries       ${100 * llm_module.LEDGER.total.cost_usd / max(len(judged_rows), 1):.4f}"
          f"  of ${config.LLM_BUDGET_USD:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
