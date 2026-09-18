"""Query the built index by hand, and calibrate the "does it exist?" threshold.

    python scripts/inspect_index.py                      # spec example queries
    python scripts/inspect_index.py -q "chicken alfredo" # one query
    python scripts/inspect_index.py --calibrate          # score distributions
    python scripts/inspect_index.py --diet vegan -q "banana bread"
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recipe_rag import config, store

# The spec's own examples. Epicurious is a largely Western corpus, so the two
# Indian dishes are expected misses -- they are the negative controls that the
# web-fallback path exists to serve.
DEMO_QUERIES = [
    "Give me the recipe for Chicken Alfredo",
    "Recipe for Vegan Banana Bread",
    "How do I make Paneer Butter Masala?",
]

# Titles that are definitely in the dataset, for calibrating the threshold.
KNOWN_PRESENT = [
    "Lentil, Apple, and Turkey Wrap",
    "Boudin Blanc Terrine with Red Onion Confit",
    "Pastry Dough",
    "Chicken Stock",
    "Simple Syrup",
    "Spanakopita",
    "Grilled Salmon",
    "Chocolate Chip Cookies",
]

# Dishes this corpus does not cover, or that do not exist at all.
KNOWN_ABSENT = [
    "Paneer Butter Masala",
    "Masala Dosa",
    "Hyderabadi Biryani",
    "Pani Puri",
    "Chicken 65",
    "Pad Thai Gai Yang Som Tam Platter",
    "Deep Fried Moon Rocks",
    "Quantum Blockchain Smoothie",
]


def _similarity(distance: float) -> float:
    """Chroma returns cosine distance; similarity is 1 - distance."""
    return 1.0 - distance


def search(collection, parents, query: str, k: int = 5, diet: str | None = None):
    where = {"diet": diet} if diet else None
    result = collection.query(
        query_texts=[query],
        n_results=k * 4,  # over-fetch so dedup by parent still leaves k
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    seen: dict[str, dict] = {}
    for document, meta, distance in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        parent_id = meta["parent_id"]
        if parent_id in seen:
            continue
        row = parents.loc[parents.parent_id == parent_id]
        seen[parent_id] = {
            "score": _similarity(distance),
            "section": meta["section"],
            "diet": meta["diet"],
            "title": row.iloc[0]["title"] if len(row) else "<missing parent>",
            "dup_group": row.iloc[0]["dup_group"] if len(row) else "",
            "text": document,
        }
        if len(seen) >= k:
            break

    # Collapse same-title variants to the best-ranked one.
    collapsed, groups = [], set()
    for hit in seen.values():
        group = hit["dup_group"]
        if group and group in groups:
            continue
        if group:
            groups.add(group)
        collapsed.append(hit)
    return collapsed


def show(collection, parents, query: str, k: int, diet: str | None) -> None:
    print(f"\n=== {query!r}" + (f"  [diet={diet}]" if diet else ""))
    hits = search(collection, parents, query, k, diet)
    if not hits:
        print("  no results")
        return
    for rank, hit in enumerate(hits, 1):
        print(
            f"  {rank}. {hit['score']:.3f}  {hit['title'][:58]:<58} "
            f"{hit['diet']:<10} via {hit['section']}"
        )


def calibrate(collection, parents) -> None:
    """Compare top-1 scores for present vs absent recipes.

    This is the evidence for the spec's question about what similarity value
    should decide that a recipe "exists" in the dataset. The gap between the
    two distributions is where the threshold belongs.
    """
    print("\n--- present (should score high) " + "-" * 34)
    present = []
    for title in KNOWN_PRESENT:
        hits = search(collection, parents, title, k=1)
        score = hits[0]["score"] if hits else 0.0
        present.append(score)
        print(f"  {score:.3f}  {title[:48]:<48} -> {hits[0]['title'][:40] if hits else '-'}")

    print("\n--- absent (should score low) " + "-" * 36)
    absent = []
    for title in KNOWN_ABSENT:
        hits = search(collection, parents, title, k=1)
        score = hits[0]["score"] if hits else 0.0
        absent.append(score)
        print(f"  {score:.3f}  {title[:48]:<48} -> {hits[0]['title'][:40] if hits else '-'}")

    lo = min(present)
    hi = max(absent)
    print("\n--- summary " + "-" * 54)
    print(f"  present: min {lo:.3f}  mean {statistics.mean(present):.3f}  max {max(present):.3f}")
    print(f"  absent : min {min(absent):.3f}  mean {statistics.mean(absent):.3f}  max {hi:.3f}")
    if lo > hi:
        print(f"  SEPARABLE: gap {hi:.3f} .. {lo:.3f}  -> threshold ~{(lo + hi) / 2:.3f}")
    else:
        print(
            f"  OVERLAP of {hi - lo:.3f} (absent max {hi:.3f} >= present min {lo:.3f}).\n"
            f"  A similarity cutoff alone will not separate these; phase 2 should\n"
            f"  combine it with a title-match signal."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-q", "--query", action="append", help="query (repeatable)")
    parser.add_argument("-k", type=int, default=5, help="results per query")
    parser.add_argument("--diet", choices=config.DIET_LABELS, help="filter by diet")
    parser.add_argument("--calibrate", action="store_true",
                        help="print present/absent score distributions")
    args = parser.parse_args()

    store.require_index()
    client = store.get_client()
    collection = store.get_collection(client, create=False)
    parents = store.load_parents()
    print(
        f"collection '{config.COLLECTION_NAME}': {collection.count():,} chunks, "
        f"{len(parents):,} parents"
    )

    if args.calibrate:
        calibrate(collection, parents)
        return 0

    for query in args.query or DEMO_QUERIES:
        show(collection, parents, query, args.k, args.diet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
