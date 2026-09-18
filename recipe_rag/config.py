"""Central configuration for the recipe RAG index build."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_JSON = PROJECT_ROOT / "full_format_recipes.json"

# Build artifacts (Chroma index + parent store). Keep these OFF a synced drive:
# OneDrive will lock or half-upload the Chroma SQLite file mid-build. Override
# with the RECIPE_RAG_ARTIFACTS environment variable.
ARTIFACT_DIR = Path(
    os.environ.get("RECIPE_RAG_ARTIFACTS", PROJECT_ROOT / ".artifacts")
).resolve()
CHROMA_DIR = ARTIFACT_DIR / "chroma_db"
PARENTS_PARQUET = ARTIFACT_DIR / "parents.parquet"
BUILD_REPORT = ARTIFACT_DIR / "build_report.json"

# The grader's validation queries and expected answers (UC13 "Validation Data").
GROUND_TRUTH_XLSX = PROJECT_ROOT / "Ground Truth.xlsx"

COLLECTION_NAME = "recipe_children"

# chromadb ships all-MiniLM-L6-v2 as an ONNX export, so we get the model the
# spec asks for without pulling torch in. 384 dims, already L2-normalised.
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_BACKEND = "chromadb.ONNXMiniLM_L6_V2"
EMBED_DIM = 384

# MiniLM truncates at 256 wordpieces. 900 characters lands around 225 tokens,
# which leaves headroom for the title prefix carried by every chunk.
CHUNK_CHAR_CAP = 900
MAX_MODEL_TOKENS = 256
OVERSIZE_STEP_OVERLAP = 100

# Chroma caps how many records go in one add(); we stay well under it.
UPSERT_BATCH = 1000

SECTION_SUMMARY = "summary"
SECTION_INGREDIENTS = "ingredients"
SECTION_DIRECTIONS = "directions"

DIET_LABELS = ("vegan", "veg", "eggetarian", "non-veg")

_SYNC_MARKERS = ("onedrive", "dropbox", "google drive", "icloud")


def is_synced_path(path: Path) -> bool:
    """True if `path` looks like it sits inside a cloud-sync folder."""
    lowered = str(path).lower()
    return any(marker in lowered for marker in _SYNC_MARKERS)


# --------------------------------------------------------------------------
# Phase 2: the LLM layer.
# --------------------------------------------------------------------------

# Secrets live in a gitignored .env, never in code. Point RECIPE_RAG_ENV at a
# file outside this folder to keep the key off OneDrive.
ENV_FILE = Path(os.environ.get("RECIPE_RAG_ENV", PROJECT_ROOT / ".env"))


def load_env(path: Path | None = None) -> dict[str, str]:
    """Read a .env file into os.environ. Returns the keys it set.

    Deliberately not python-dotenv: the format we need is `KEY=value` lines
    with `#` comments, which is fifteen lines of parsing, and this project has
    kept its dependency list short on purpose (no torch, no sentence-
    transformers). Values already present in the real environment win, so a
    shell export can always override the file.
    """
    path = path or ENV_FILE
    loaded: dict[str, str] = {}
    if not path.exists():
        return loaded
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        # Tolerate quotes; people paste them in without meaning to.
        value = value.strip().strip('"').strip("'")
        if not key or not value:
            continue
        os.environ.setdefault(key, value)
        loaded[key] = value
    return loaded


class MissingKeyError(RuntimeError):
    """Raised with instructions rather than a bare KeyError."""


def azure_settings() -> dict[str, str] | None:
    """Azure OpenAI connection details, or None when talking to api.openai.com.

    An Azure key is useless on its own: Azure routes by *your* resource URL and
    by the name you gave the model deployment, neither of which can be derived
    from the key. So the presence of AZURE_OPENAI_ENDPOINT is what selects Azure.
    """
    load_env()
    endpoint = (os.environ.get("AZURE_OPENAI_ENDPOINT") or "").strip()
    if not endpoint:
        return None
    return {
        "endpoint": endpoint,
        "api_version": (os.environ.get("AZURE_OPENAI_API_VERSION") or "2024-10-21").strip(),
        # On Azure the `model` argument is the deployment name, not "gpt-4o-mini".
        "deployment": (os.environ.get("AZURE_OPENAI_DEPLOYMENT") or LLM_MODEL).strip(),
    }


def openai_key(required: bool = True) -> str | None:
    """Fetch the API key (OPENAI_API_KEY or AZURE_OPENAI_API_KEY), loading .env.

    Fails with the fix in the message. A wrong key otherwise surfaces much later
    as an opaque 401 from inside the openai client. Never echoes any part of the
    key -- error text ends up in terminals, logs and screenshots.
    """
    load_env()
    key = (
        os.environ.get("OPENAI_API_KEY") or os.environ.get("AZURE_OPENAI_API_KEY") or ""
    ).strip()
    if key and not key.startswith("sk-") and azure_settings() is None:
        raise MissingKeyError(
            "The key in .env is not an api.openai.com key (those start with 'sk-').\n"
            "It has the shape of an Azure OpenAI key, which needs three more lines\n"
            f"in {ENV_FILE}:\n"
            "  AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/\n"
            "  AZURE_OPENAI_DEPLOYMENT=<your deployment name, often gpt-4o-mini>\n"
            "  AZURE_OPENAI_API_VERSION=2024-10-21\n"
            "Endpoint: Azure portal -> your OpenAI resource -> Keys and Endpoint.\n"
            "Deployment name: Azure AI Foundry -> Deployments."
        )
    if not key and required:
        raise MissingKeyError(
            f"No OPENAI_API_KEY found.\n"
            f"  1. Open {ENV_FILE}\n"
            f"  2. Set OPENAI_API_KEY=sk-... (no quotes, no spaces)\n"
            f"  3. Re-run. No shell restart needed -- the file is read at call time."
        )
    return key or None


# The spec mandates this model; do not silently upgrade it.
LLM_MODEL = os.environ.get("RECIPE_RAG_LLM_MODEL", "gpt-4o-mini")
# Zero temperature so evaluation runs are reproducible and a judge verdict can
# be regression-tested. Creativity is not wanted anywhere in this pipeline.
LLM_TEMPERATURE = 0.0
LLM_TIMEOUT_SECONDS = 20.0

# USD per million tokens, (input, output). A build-time snapshot, not a live
# feed: verify against current pricing before quoting a figure to anyone.
#
# Keyed by model family and matched as a prefix, because an Azure deployment is
# named by whoever created it ("gpt-4.1-mini-4") rather than by the model. A
# deployment nobody here recognises falls back to the mandated gpt-4o-mini rate,
# which is the cheapest of these and so the least alarming way to be wrong --
# hence the warning in the cost reports that the model is unrecognised.
LLM_PRICES_USD_PER_MTOK = {
    "gpt-4o-mini": (0.150, 0.600),
    "gpt-4.1-nano": (0.100, 0.400),
    "gpt-4.1-mini": (0.400, 1.600),
    "gpt-4.1": (2.000, 8.000),
    "gpt-4o": (2.500, 10.000),
}
LLM_BUDGET_USD = 2.00


def llm_prices(model: str | None = None) -> tuple[float, float]:
    """(input, output) price per million tokens for a model or deployment name."""
    name = (model or LLM_MODEL).lower()
    for family in sorted(LLM_PRICES_USD_PER_MTOK, key=len, reverse=True):
        if name.startswith(family) or family in name:
            return LLM_PRICES_USD_PER_MTOK[family]
    return LLM_PRICES_USD_PER_MTOK["gpt-4o-mini"]


def llm_cost(input_tokens: int, output_tokens: int, model: str | None = None) -> float:
    """USD for one call, given its reported token usage."""
    price_in, price_out = llm_prices(model)
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000
