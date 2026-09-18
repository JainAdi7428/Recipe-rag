"""Measure child-to-parent aggregation strategies against known-correct answers.

Builds its own ground truth: sample N recipes from the parent store, use each
one's exact title as the query, and check whether that recipe comes back top-1.
Because the query is the title verbatim, the correct answer is known without
hand-labelling.

    python scripts/eval_retrieval.py --sample 400

Reports Recall@1, Recall@5 and MRR per strategy. This is a lower bound on real
quality (a verbatim title is an easy query) but it is a fair *relative*
comparison between aggregation strategies, which is what it is for.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recipe_rag import store
from recipe_rag.retrieve import STRATEGIES, aggregate, collapse_variants


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=300)
    parser.add_argument("--fetch", type=int, default=40,
                        help="child chunks to fetch before aggregating")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    store.require_index()
    client = store.get_client()
    collection = store.get_collection(client, create=False)
    parents = store.load_parents()
    dup_group_of = dict(zip(parents.parent_id, parents.dup_group))

    # Sample only from unambiguous titles: if several distinct recipes share a
    # title, "top-1 is the right one" has no single correct answer.
    unique = parents[parents.dup_group == ""]
    random.seed(args.seed)
    picks = unique.sample(min(args.sample, len(unique)), random_state=args.seed)
    print(f"evaluating {len(picks)} title queries against {collection.count():,} chunks")

    queries = list(picks.title)
    truth = list(picks.parent_id)

    # Batched: Chroma binds one SQL variable per returned row, and a large
    # query_texts x n_results product trips SQLite's variable limit.
    batch = max(1, 8000 // max(args.fetch, 1))
    result = {"metadatas": [], "distances": []}
    for start in range(0, len(queries), batch):
        page = collection.query(
            query_texts=queries[start : start + batch],
            n_results=args.fetch,
            include=["metadatas", "distances"],
        )
        result["metadatas"].extend(page["metadatas"])
        result["distances"].extend(page["distances"])

    stats = {s: {"r1": 0, "r5": 0, "mrr": 0.0} for s in STRATEGIES}
    chunk_counts = []

    for index, expected in enumerate(truth):
        rows = [
            (meta["parent_id"], meta["section"], 1.0 - distance)
            for meta, distance in zip(
                result["metadatas"][index], result["distances"][index]
            )
        ]
        by_parent: dict[str, int] = defaultdict(int)
        for parent_id, _, _ in rows:
            by_parent[parent_id] += 1
        chunk_counts.append(by_parent.get(expected, 0))

        for strategy in STRATEGIES:
            ranked = collapse_variants(aggregate(rows, strategy), dup_group_of)
            ids = [hit.parent_id for hit in ranked]
            if expected in ids:
                rank = ids.index(expected) + 1
                stats[strategy]["mrr"] += 1.0 / rank
                if rank == 1:
                    stats[strategy]["r1"] += 1
                if rank <= 5:
                    stats[strategy]["r5"] += 1

    total = len(truth)
    multi = sum(1 for n in chunk_counts if n > 1)
    print(
        f"\nthe correct recipe contributed >1 chunk to the candidate set in "
        f"{multi}/{total} queries ({100 * multi / total:.0f}%)"
        f"  -- aggregation only matters for these\n"
    )
    print(f"{'strategy':<10} {'Recall@1':>9} {'Recall@5':>9} {'MRR':>7}")
    print("-" * 38)
    for strategy in STRATEGIES:
        s = stats[strategy]
        print(
            f"{strategy:<10} {100 * s['r1'] / total:>8.1f}% "
            f"{100 * s['r5'] / total:>8.1f}% {s['mrr'] / total:>7.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
