"""Persistence: Chroma for the child vectors, Parquet for the parent records.

Parents deliberately live outside Chroma. Chroma stores metadata as one SQLite
row per key-value pair, so putting ~15 fields on 64,569 chunks would mean close
to a million rows -- a large cost paid on an artifact that has to be shipped to
Streamlit Cloud. Child metadata is therefore kept to what retrieval actually
filters on, and the full record is fetched from Parquet by `parent_id`.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Iterable

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from .chunker import Chunk
from .config import (
    ARTIFACT_DIR,
    CHROMA_DIR,
    COLLECTION_NAME,
    PARENTS_PARQUET,
    UPSERT_BATCH,
)
from .diet import DietResult
from .loader import Recipe

_LIST_COLUMNS = ("ingredients", "directions", "categories")


def get_embedding_function():
    """all-MiniLM-L6-v2 as an ONNX export -- the spec's model, without torch.

    This must be identical at build and query time: swapping it invalidates
    every vector in the index.
    """
    return embedding_functions.ONNXMiniLM_L6_V2()


def require_index() -> None:
    """Fail fast, and legibly, if there is no built index to read.

    Call this *before* get_client(). Chroma's PersistentClient creates its
    directory on construction, so by the time a collection lookup fails the
    evidence is already gone -- an empty database sits where the missing one
    was, and the error reads like corruption rather than a wrong path.
    """
    if CHROMA_DIR.exists() and PARENTS_PARQUET.exists():
        return

    override = os.environ.get("RECIPE_RAG_ARTIFACTS")
    source = (
        f"RECIPE_RAG_ARTIFACTS={override!r}"
        if override
        else "the default .artifacts/ folder (no RECIPE_RAG_ARTIFACTS set)"
    )
    hint = (
        "That environment variable is stale -- clear it with "
        "`Remove-Item Env:RECIPE_RAG_ARTIFACTS` and try again."
        if override
        else "Build it with `python scripts/build_index.py --rebuild`."
    )
    raise SystemExit(
        f"\nNo index found.\n"
        f"  looked in : {ARTIFACT_DIR}\n"
        f"  chosen by : {source}\n"
        f"  chroma_db : {'found' if CHROMA_DIR.exists() else 'MISSING'}\n"
        f"  parents   : {'found' if PARENTS_PARQUET.exists() else 'MISSING'}\n"
        f"{hint}\n"
    )


def get_client() -> chromadb.ClientAPI:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )


def reset_store() -> None:
    """Delete the whole artifact directory so a rebuild starts clean."""
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    if PARENTS_PARQUET.exists():
        PARENTS_PARQUET.unlink()


def get_collection(client: chromadb.ClientAPI, create: bool = True):
    """Fetch (or create) the child collection.

    Cosine, not Chroma's L2 default. MiniLM is trained for cosine similarity;
    leaving this at L2 degrades ranking and makes any similarity threshold we
    later calibrate meaningless.
    """
    embedder = get_embedding_function()
    if not create:
        try:
            return client.get_collection(
                name=COLLECTION_NAME, embedding_function=embedder
            )
        except Exception as error:
            # Chroma's own message names the collection but never says which
            # directory it searched, which makes a mis-pointed config look like
            # a corrupt index. Say where we looked and how we decided.
            source = (
                f"RECIPE_RAG_ARTIFACTS={os.environ['RECIPE_RAG_ARTIFACTS']!r}"
                if os.environ.get("RECIPE_RAG_ARTIFACTS")
                else "the default .artifacts/ folder (no RECIPE_RAG_ARTIFACTS set)"
            )
            raise RuntimeError(
                f"No index found.\n"
                f"  looked in : {ARTIFACT_DIR}\n"
                f"  chosen by : {source}\n"
                f"  exists    : chroma_db={CHROMA_DIR.exists()} "
                f"parents.parquet={PARENTS_PARQUET.exists()}\n"
                f"  original  : {error}\n"
                f"Either RECIPE_RAG_ARTIFACTS points somewhere stale (clear it with "
                f"`Remove-Item Env:RECIPE_RAG_ARTIFACTS`), or the index has not been "
                f"built yet (`python scripts/build_index.py --rebuild`)."
            ) from error
    try:
        return client.get_collection(name=COLLECTION_NAME, embedding_function=embedder)
    except Exception:
        return client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedder,
            configuration={"hnsw": {"space": "cosine"}},
        )


def chunk_metadata(
    chunk: Chunk, recipe: Recipe, diet: DietResult, nutrition=None
) -> dict:
    """Only what retrieval filters or ranks on. Everything else is on the parent."""
    return {
        "parent_id": chunk.parent_id,
        "section": chunk.section,
        "part": chunk.part,
        "n_parts": chunk.n_parts,
        "diet": diet.diet,
        "dup_rank": recipe.dup_rank,
        # Reflects validated nutrition, so a row rejected as implausible reads
        # as having none rather than as having numbers we refuse to show.
        "has_nutrition": bool(nutrition.has_any) if nutrition else False,
    }


def upsert_chunks(client, collection, chunks: list[Chunk], metadatas: list[dict],
                  embeddings=None, batch_size: int = UPSERT_BATCH,
                  progress=None) -> int:
    """Store chunks in batches; returns the number written.

    `embeddings` are precomputed by FastMiniLMEmbedder. Passing them means
    Chroma never calls its own (much slower) embedding function during the
    build; it still owns the identical one for query time.
    """
    written = 0
    try:
        limit = min(batch_size, client.get_max_batch_size())
    except Exception:
        limit = batch_size

    total = len(chunks)
    for start in range(0, total, limit):
        stop = min(start + limit, total)
        payload = {
            "ids": [chunk.chunk_id for chunk in chunks[start:stop]],
            "documents": [chunk.text for chunk in chunks[start:stop]],
            "metadatas": metadatas[start:stop],
        }
        if embeddings is not None:
            payload["embeddings"] = embeddings[start:stop].tolist()
        collection.upsert(**payload)
        written += stop - start
        if progress is not None:
            progress(written, total)
    return written


def compact() -> tuple[float, float]:
    """VACUUM the Chroma SQLite file after a build; returns (before, after) MB.

    Bulk inserts leave free pages behind. This matters more than usual here
    because the whole artifact gets uploaded to Hugging Face and pulled down
    again on every cold start of the Streamlit app.
    """
    import sqlite3

    database = CHROMA_DIR / "chroma.sqlite3"
    if not database.exists():
        return 0.0, 0.0
    before = database.stat().st_size / 1024 / 1024
    connection = sqlite3.connect(database)
    try:
        connection.execute("VACUUM")
    finally:
        connection.close()
    return before, database.stat().st_size / 1024 / 1024


def artifact_size_mb() -> float:
    return sum(f.stat().st_size for f in ARTIFACT_DIR.rglob("*") if f.is_file()) / 1024 / 1024


def iter_all(collection, include=("documents", "metadatas"), page: int = 5000):
    """Yield every record in the collection, one page at a time.

    A bare collection.get() asks SQLite for every row in a single statement,
    and SQLite caps how many parameters one statement may bind (~32,766). At
    67,010 chunks that raises "too many SQL variables", which reads like
    corruption but is only a batch-size limit. Always page a full scan.
    """
    include = list(include)
    offset = 0
    while True:
        batch = collection.get(include=include, limit=page, offset=offset)
        ids = batch.get("ids") or []
        if not ids:
            return
        yield batch
        offset += len(ids)


def scan(collection, include=("documents", "metadatas")) -> dict:
    """Full scan, paged, flattened back into one dict of lists."""
    include = list(include)
    out: dict[str, list] = {"ids": []}
    for key in include:
        out[key] = []
    for batch in iter_all(collection, include, page=5000):
        out["ids"].extend(batch.get("ids") or [])
        for key in include:
            out[key].extend(batch.get(key) or [])
    return out


def parent_row(
    recipe: Recipe, diet: DietResult, n_chunks: int, nutrition=None
) -> dict:
    """One Parquet row. Lists are JSON-encoded to stay schema-stable.

    Nutrition columns hold *validated* values. The raw source values are kept
    alongside under `raw_*` so a rejection can be audited without re-reading
    the original JSON, but only the validated ones should ever be shown.
    """
    values = nutrition.as_dict() if nutrition else {}
    return {
        "parent_id": recipe.parent_id,
        "title": recipe.title,
        "title_norm": recipe.title_norm,
        "desc": recipe.desc,
        "ingredients": json.dumps(recipe.ingredients, ensure_ascii=False),
        "directions": json.dumps(recipe.directions, ensure_ascii=False),
        "categories": json.dumps(recipe.categories, ensure_ascii=False),
        "calories": values.get("calories"),
        "protein": values.get("protein"),
        "fat": values.get("fat"),
        "sodium": values.get("sodium"),
        "nutrition_status": nutrition.status if nutrition else "missing",
        "nutrition_rejected": ",".join(nutrition.rejected_fields) if nutrition else "",
        "raw_calories": recipe.calories,
        "raw_protein": recipe.protein,
        "raw_fat": recipe.fat,
        "raw_sodium": recipe.sodium,
        "rating": recipe.rating,
        "date": recipe.date,
        "diet": diet.diet,
        "diet_confidence": diet.confidence,
        "diet_conflict": diet.conflict,
        "diet_reason": diet.reason,
        "dup_group": recipe.dup_group,
        "dup_rank": recipe.dup_rank,
        "n_chunks": n_chunks,
        "source_index": recipe.source_index,
    }


def write_parents(rows: Iterable[dict], path: Path = PARENTS_PARQUET) -> int:
    import pandas as pd

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(list(rows))
    frame.to_parquet(path, index=False, compression="zstd")
    return len(frame)


def load_parents(path: Path = PARENTS_PARQUET):
    """Load the parent store, decoding the JSON-encoded list columns."""
    import pandas as pd

    frame = pd.read_parquet(path)
    for column in _LIST_COLUMNS:
        frame[column] = frame[column].map(json.loads)
    return frame
