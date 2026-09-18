"""Build the recipe vector index end to end.

    python scripts/build_index.py --rebuild            # full build
    python scripts/build_index.py --limit 500 --rebuild  # fast smoke build
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recipe_rag import config, store
from recipe_rag.chunker import build_all
from recipe_rag.dedup import deduplicate
from recipe_rag.diet import classify_diet
from recipe_rag.loader import load_recipes
from recipe_rag.nutrition import validate as validate_nutrition


def _warn_if_synced() -> None:
    """Note the sync risk once, without being obnoxious about it.

    The project deliberately keeps artifacts in-tree. If a build ever dies on a
    locked database, this is the first thing to check.
    """
    if config.is_synced_path(config.ARTIFACT_DIR):
        print(
            f"  note: artifacts are in a cloud-synced folder ({config.ARTIFACT_DIR.name}). "
            "If a build fails on a locked database, exclude it from sync via\n"
            "        OneDrive Settings > Account > Choose folders.",
            file=sys.stderr,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="only ingest the first N usable recipes")
    parser.add_argument("--rebuild", action="store_true",
                        help="delete any existing index first")
    parser.add_argument("--skip-embed", action="store_true",
                        help="run the pipeline and write parents, but do not embed")
    args = parser.parse_args()

    _warn_if_synced()
    started = time.perf_counter()

    if args.rebuild:
        store.reset_store()
        print("reset: removed existing index")

    # ---- load -------------------------------------------------------------
    step = time.perf_counter()
    recipes, load_stats = load_recipes(config.SOURCE_JSON, limit=args.limit)
    print(
        f"load     {load_stats.kept:>6,} usable of {load_stats.total_records:,} records "
        f"({load_stats.dropped} dropped: {load_stats.dropped_no_title} no title, "
        f"{load_stats.dropped_no_ingredients} no ingredients, "
        f"{load_stats.dropped_no_directions} no directions)  "
        f"[{time.perf_counter() - step:.1f}s]"
    )

    # ---- dedup ------------------------------------------------------------
    step = time.perf_counter()
    recipes, dedup_stats = deduplicate(recipes)
    print(
        f"dedup    {dedup_stats.parents:>6,} parents  "
        f"(-{dedup_stats.exact_duplicates_removed:,} exact duplicates; "
        f"{dedup_stats.variant_groups} variant groups covering "
        f"{dedup_stats.rows_in_variant_groups} rows)  "
        f"[{time.perf_counter() - step:.1f}s]"
    )

    # ---- classify ---------------------------------------------------------
    step = time.perf_counter()
    diets = []
    diet_counts: Counter[str] = Counter()
    conflicts = 0
    for recipe in recipes:
        diet = classify_diet(recipe.ingredients, recipe.categories, recipe.title)
        diets.append(diet)
        diet_counts[diet.diet] += 1
        conflicts += bool(diet.conflict)
    print(
        f"classify {dict(diet_counts)}  ({conflicts} category conflicts)  "
        f"[{time.perf_counter() - step:.1f}s]"
    )

    # ---- validate nutrition ----------------------------------------------
    # The source contains impossible values (one pie claims 30,111,218 kcal).
    # Screening happens before chunking so rejected numbers never reach the
    # embedded text, the metadata, or a user.
    nutritions = [
        validate_nutrition(
            {
                "calories": recipe.calories,
                "protein": recipe.protein,
                "fat": recipe.fat,
                "sodium": recipe.sodium,
            }
        )
        for recipe in recipes
    ]
    nutrition_counts = Counter(n.status for n in nutritions)
    rejected_fields: Counter[str] = Counter()
    for entry in nutritions:
        rejected_fields.update(entry.rejected_fields)
    print(
        f"nutrition {dict(nutrition_counts)}  "
        f"rejected rows: {nutrition_counts.get('rejected', 0)} "
        f"(triggers: {dict(rejected_fields)})"
    )

    # ---- chunk ------------------------------------------------------------
    # The encoder is loaded before chunking so the chunker can measure real
    # token lengths; a character cap alone cannot guarantee the token cap.
    encoder = None
    parity = None
    if not args.skip_embed:
        from recipe_rag.embedder import FastMiniLMEmbedder

        encoder = FastMiniLMEmbedder()

    step = time.perf_counter()
    per_recipe = build_all(
        [
            (recipe, diet.diet, nutrition)
            for recipe, diet, nutrition in zip(recipes, diets, nutritions)
        ],
        measure=encoder.token_lengths if encoder else None,
    )

    all_chunks, all_metadata, parent_rows = [], [], []
    for recipe, diet, nutrition, chunks in zip(recipes, diets, nutritions, per_recipe):
        all_chunks.extend(chunks)
        all_metadata.extend(
            store.chunk_metadata(chunk, recipe, diet, nutrition) for chunk in chunks
        )
        parent_rows.append(store.parent_row(recipe, diet, len(chunks), nutrition))

    section_counts = Counter(chunk.section for chunk in all_chunks)
    longest = max(len(chunk.text) for chunk in all_chunks)
    print(
        f"chunk    {len(all_chunks):>6,} children "
        f"({len(all_chunks) / max(len(recipes), 1):.2f}/recipe)  "
        f"{dict(section_counts)}  longest {longest} chars  "
        f"[{time.perf_counter() - step:.1f}s]"
    )

    # ---- parents ----------------------------------------------------------
    written = store.write_parents(parent_rows)
    size_mb = config.PARENTS_PARQUET.stat().st_size / 1024 / 1024
    print(f"parents  {written:>6,} rows -> {config.PARENTS_PARQUET.name} ({size_mb:.1f} MB)")

    # ---- embed ------------------------------------------------------------
    embedded = 0
    over_cap = 0
    if encoder is not None:
        # The build encoder and the query encoder must share a vector space.
        parity = encoder.verify_parity([chunk.text for chunk in all_chunks[:16]])
        print(f"parity   build vs query encoder max delta {parity:.2e}  (ok)")

        # The entire point of the chunking scheme is that nothing gets
        # truncated, so assert it rather than assume it.
        texts = [chunk.text for chunk in all_chunks]
        lengths = encoder.token_lengths(texts)
        over_cap = sum(1 for n in lengths if n > config.MAX_MODEL_TOKENS)
        print(
            f"tokens   max {max(lengths)}  mean {sum(lengths) / len(lengths):.0f}  "
            f"over {config.MAX_MODEL_TOKENS}-token cap: {over_cap}"
        )
        if over_cap:
            worst = max(range(len(lengths)), key=lambda i: lengths[i])
            raise SystemExit(
                f"ERROR: {over_cap} chunks exceed the model's {config.MAX_MODEL_TOKENS}"
                f"-token limit and would be silently truncated.\n"
                f"  worst: {lengths[worst]} tokens, id={all_chunks[worst].chunk_id}"
            )

        step = time.perf_counter()

        def progress(done: int, total: int) -> None:
            elapsed = time.perf_counter() - step
            rate = done / elapsed if elapsed else 0
            eta = (total - done) / rate if rate else 0
            print(
                f"\r  embedding {done:,}/{total:,} "
                f"({100 * done / total:.1f}%)  {rate:.0f} chunks/s  eta {eta / 60:.1f}m",
                end="",
                flush=True,
            )

        vectors = encoder.encode(texts, progress=progress)
        print(
            f"\rembed    {len(vectors):,} vectors {vectors.shape}"
            f"  [{time.perf_counter() - step:.1f}s]{' ' * 30}"
        )

        step = time.perf_counter()
        client = store.get_client()
        collection = store.get_collection(client)
        embedded = store.upsert_chunks(
            client, collection, all_chunks, all_metadata, embeddings=vectors
        )
        print(
            f"index    {embedded:>6,} chunks into '{config.COLLECTION_NAME}'"
            f"  [{time.perf_counter() - step:.1f}s]"
        )
        print(f"verify   collection.count() = {collection.count():,}")

        del client, collection  # release the SQLite handle before VACUUM
        before_mb, after_mb = store.compact()
        print(
            f"compact  {before_mb:.1f} -> {after_mb:.1f} MB sqlite "
            f"(artifact total {store.artifact_size_mb():.1f} MB)"
        )

    report = {
        "source": str(config.SOURCE_JSON),
        "artifact_dir": str(config.ARTIFACT_DIR),
        "embed_model": config.EMBED_MODEL_NAME,
        "embed_backend": config.EMBED_BACKEND,
        "limit": args.limit,
        "load": load_stats.__dict__,
        "dedup": dedup_stats.__dict__,
        "diet": dict(diet_counts),
        "diet_conflicts": conflicts,
        "nutrition_status": dict(nutrition_counts),
        "nutrition_rejected_triggers": dict(rejected_fields),
        "sections": dict(section_counts),
        "parents": written,
        "chunks": len(all_chunks),
        "embedded": embedded,
        "longest_chunk_chars": longest,
        "encoder_parity_delta": parity,
        "chunks_over_token_cap": over_cap,
        "seconds": round(time.perf_counter() - started, 1),
    }
    config.BUILD_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report   {config.BUILD_REPORT}")
    print(f"total    {report['seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
