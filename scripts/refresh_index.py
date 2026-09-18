"""Re-run the pipeline and update only the chunks whose text actually changed.

A full --rebuild re-embeds all 67,010 chunks and costs ~35 minutes. Most changes
touch far less than that: adjusting the nutrition rule alters the summary card of
a hundred recipes and nothing else. This recomputes everything cheaply (no
embedding), diffs it against what is stored, and re-embeds only the difference.

    python scripts/refresh_index.py --dry-run    # report what would change
    python scripts/refresh_index.py

Use --rebuild instead when the *embedding model* changes: then every vector is
stale even though the text is identical, and a diff would see no work to do.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recipe_rag import config, store
from recipe_rag.chunker import build_all
from recipe_rag.dedup import deduplicate
from recipe_rag.diet import classify_diet
from recipe_rag.embedder import FastMiniLMEmbedder
from recipe_rag.loader import load_recipes
from recipe_rag.nutrition import validate as validate_nutrition


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="report the diff and exit without writing")
    args = parser.parse_args()

    store.require_index()
    started = time.perf_counter()

    # ---- recompute the pipeline (everything except embedding) -------------
    recipes, _ = load_recipes(config.SOURCE_JSON)
    recipes, _ = deduplicate(recipes)
    diets = [
        classify_diet(r.ingredients, r.categories, r.title) for r in recipes
    ]
    nutritions = [
        validate_nutrition(
            {"calories": r.calories, "protein": r.protein,
             "fat": r.fat, "sodium": r.sodium}
        )
        for r in recipes
    ]
    print(f"nutrition {dict(Counter(n.status for n in nutritions))}")

    encoder = FastMiniLMEmbedder()
    per_recipe = build_all(
        [(r, d.diet, n) for r, d, n in zip(recipes, diets, nutritions)],
        measure=encoder.token_lengths,
    )

    fresh: dict[str, tuple] = {}
    for recipe, diet, nutrition, chunks in zip(recipes, diets, nutritions, per_recipe):
        for chunk in chunks:
            fresh[chunk.chunk_id] = (
                chunk,
                store.chunk_metadata(chunk, recipe, diet, nutrition),
            )
    print(f"recomputed {len(fresh):,} chunks  [{time.perf_counter() - started:.1f}s]")

    # ---- diff against what is stored --------------------------------------
    client = store.get_client()
    collection = store.get_collection(client, create=False)
    stored = store.scan(collection, include=["documents", "metadatas"])
    stored_text = dict(zip(stored["ids"], stored["documents"]))
    stored_meta = dict(zip(stored["ids"], stored["metadatas"]))

    changed = [
        cid for cid, (chunk, _) in fresh.items()
        if stored_text.get(cid) != chunk.text
    ]
    changed_set = set(changed)
    # A diet relabel rewrites the summary card's text ("Diet: vegan") but only
    # the *metadata* of the recipe's ingredient and direction chunks. Those need
    # their metadata updated, not their vectors recomputed.
    retagged = [
        cid for cid, (_, meta) in fresh.items()
        if cid not in changed_set and cid in stored_meta and stored_meta[cid] != meta
    ]
    removed = [cid for cid in stored_text if cid not in fresh]

    print(f"stored    {len(stored_text):,} chunks")
    print(f"changed   {len(changed):,}   new/edited text -> re-embed")
    print(f"retagged  {len(retagged):,}   metadata only -> update, no re-embed")
    print(f"removed   {len(removed):,}   no longer produced -> delete")

    if args.dry_run:
        for cid in changed[:5]:
            print(f"\n--- {cid} ---")
            print(f"  was: {(stored_text.get(cid) or '<new>')[:150]}")
            print(f"  now: {fresh[cid][0].text[:150]}")
        print("\ndry run: nothing written")
        return 0

    if not changed and not removed and not retagged:
        print("\nindex already matches the current pipeline; nothing to do")
        return 0

    if retagged:
        for start in range(0, len(retagged), config.UPSERT_BATCH):
            batch = retagged[start:start + config.UPSERT_BATCH]
            collection.update(ids=batch, metadatas=[fresh[cid][1] for cid in batch])
        print(f"retagged  {len(retagged):,}  metadata updated")

    # ---- apply -------------------------------------------------------------
    if removed:
        collection.delete(ids=removed)
        print(f"deleted   {len(removed):,}")

    if changed:
        step = time.perf_counter()
        chunks = [fresh[cid][0] for cid in changed]
        metadatas = [fresh[cid][1] for cid in changed]
        vectors = encoder.encode([c.text for c in chunks])
        store.upsert_chunks(client, collection, chunks, metadatas, embeddings=vectors)
        print(f"upserted  {len(changed):,}  [{time.perf_counter() - step:.1f}s]")

    # The parent store is rewritten wholesale: it is 9 MB and takes a moment,
    # and a partial update risks it disagreeing with the collection.
    rows = [
        store.parent_row(recipe, diet, len(chunks), nutrition)
        for recipe, diet, nutrition, chunks in zip(
            recipes, diets, nutritions, per_recipe
        )
    ]
    store.write_parents(rows)
    print(f"parents   {len(rows):,} rows rewritten")

    print(f"verify    collection.count() = {collection.count():,}")
    print(f"total     {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
