"""Measure the re-ranker: does the right recipe come first?

    python scripts/eval_rerank.py --set title --sample 200     # named-dish regression check
    python scripts/eval_rerank.py --set tuning                 # LLM-written descriptive queries
    python scripts/eval_rerank.py --set ground-truth           # the grader's 6 dataset rows
    python scripts/eval_rerank.py --set tuning --weights 0,0.1,0.2,0.3,0.5

Ranking only: no bands, no judge, no web. Each query is retrieved and scored
once, then re-ordered at every weight, so a sweep costs one retrieval per query.

  weight 0      = pure vector order, i.e. phase 1's behaviour (with constraints)
  R@1 / R@5     the expected recipe is first / in the top five of the re-ranked list
  MRR           mean reciprocal rank -- rewards "2nd" over "20th"
  pick          what the pipeline would actually answer with: an exact title
                match first, else the re-ranked top
  removed       the expected recipe was deleted by a constraint filter. Anything
                above zero here is a bug in the filters or in our labels.

Which set tunes what:
  title         must not get WORSE -- the re-ranker must not break name lookups
  tuning        chooses the weight; never seen by anything else
  ground-truth  reported, never tuned on. Six queries fit any weight you like.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from recipe_rag import config, groundtruth
from recipe_rag.groundtruth import normalise_title
from recipe_rag.pipeline import Pipeline

TUNING_SET = config.ARTIFACT_DIR / "tuning_queries.jsonl"


def load_queries(name: str, pipeline: Pipeline, sample: int, seed: int):
    """(kind, query, expected parent_id) triples."""
    parents = pipeline.parents
    if name == "title":
        from eval_gate import build_queries
        return build_queries(parents, sample, seed)
    if name == "ground-truth":
        titles = parents.title.map(normalise_title)
        out = []
        for case in groundtruth.read_cases():
            if case.expected_source != groundtruth.SOURCE_DATASET or not case.expected_title:
                continue
            match = parents[titles == normalise_title(case.expected_title)]
            if not match.empty:
                out.append((f"row{case.number}", case.query, match.iloc[0].parent_id))
        return out
    if name == "tuning":
        if not TUNING_SET.exists():
            raise SystemExit(f"{TUNING_SET} not found -- run scripts/make_tuning_set.py first")
        rows = [json.loads(line) for line in TUNING_SET.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [("tuning", r["query"], r["parent_id"]) for r in rows][: sample or None]
    raise SystemExit(f"unknown set {name!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", default="title", choices=("title", "tuning", "ground-truth"))
    parser.add_argument("--sample", type=int, default=200)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--weights", default="0,0.1,0.2,0.3,0.4,0.5")
    parser.add_argument("--verbose", action="store_true", help="print every miss at the default weight")
    args = parser.parse_args()
    weights = [float(w) for w in args.weights.split(",")]

    pipeline = Pipeline.load(use_judge=False, use_web=False)
    queries = load_queries(args.set, pipeline, args.sample, args.seed)
    print(f"{len(queries)} queries from the '{args.set}' set\n")

    stats = {w: defaultdict(lambda: defaultdict(float)) for w in weights}
    removed = 0
    title_of = dict(zip(pipeline.parents.parent_id, pipeline.parents.title))
    misses = []
    for kind, query, truth in queries:
        parsed, _, title_hits, reranked = pipeline.candidates(query)
        if truth in reranked.removed:
            removed += 1
            misses.append((query, f"REMOVED: {reranked.removed[truth]}"))
        for w in weights:
            order = reranked.reordered(w)
            ids = [r.parent_id for r in order]
            rank = ids.index(truth) + 1 if truth in ids else None
            pick = pipeline.pick(reranked, title_hits, order)
            for bucket in (kind, "ALL"):
                s = stats[w][bucket]
                s["n"] += 1
                s["r1"] += rank == 1
                s["r5"] += bool(rank and rank <= 5)
                s["mrr"] += 1.0 / rank if rank else 0.0
                s["pick"] += pick == truth
                s["pool"] += rank is not None
            if args.verbose and w == pipeline.reranker.weight and pick != truth:
                misses.append((query, f"got {title_of.get(pick, pick)!r} want "
                               f"{title_of.get(truth, truth)!r} (rank {rank})"))

    kinds = sorted({k for k, _, _ in queries}) + ["ALL"]
    for bucket in kinds if len(kinds) > 2 else ["ALL"]:
        n = int(stats[weights[0]][bucket]["n"])
        print(f"{bucket}  (n={n})")
        print(f"  {'weight':>6} {'R@1':>7} {'R@5':>7} {'MRR':>6} {'pick':>7} {'in pool':>8}")
        for w in weights:
            s = stats[w][bucket]
            print(f"  {w:>6.2f} {100 * s['r1'] / n:>6.1f}% {100 * s['r5'] / n:>6.1f}% "
                  f"{s['mrr'] / n:>6.3f} {100 * s['pick'] / n:>6.1f}% {100 * s['pool'] / n:>7.1f}%")
        print()
    print(f"expected recipe removed by a constraint: {removed}  <- must be 0")
    if misses and (args.verbose or removed):
        print("\nmisses:")
        for query, why in misses[:40]:
            print(f"  {query[:60]!r}\n      {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
