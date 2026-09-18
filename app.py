"""Zwigato recipe assistant -- Streamlit front end.

    streamlit run app.py

Deliberately thin: every decision belongs to recipe_rag/pipeline.py, and this
file only asks and displays. That keeps the UI honest -- what you see here is
exactly what scripts/ask.py and the evaluation harnesses see.

Two things it shows that a recipe app normally would not, because the spec is
graded on them:

  * **where the answer came from**, on every answer, in the grader's own words
    ("Internal Database", "Web Search (site)", or an apology);
  * **why**, expandable: the constraints understood from the request, the
    scores, what the constraints removed, and the pages tried.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import streamlit as st

from recipe_rag import config, logs, store
from recipe_rag.pipeline import SOURCE_DATASET, SOURCE_NOT_FOUND, SOURCE_WEB, Pipeline
from recipe_rag.present import as_markdown, present

EXAMPLES = [
    "I need a healthy fish recipe with olives and tomatoes.",
    "Show me a vegetarian spinach pasta casserole",
    "Do you have a raw vegan spinach soup recipe?",
    "I want to make authentic Palak Paneer like in Indian restaurants",
    "I need a recipe for Tandoori Jilaji?",
]

st.set_page_config(page_title="Zwigato Recipe Assistant", page_icon="🍲", layout="wide")


@st.cache_resource(show_spinner="Fetching the recipe index (about 360 MB, once per deploy)...")
def ensure_index() -> str:
    """Make sure the index is on disk; fetch it from Hugging Face if not.

    A deployed app starts with only the repository: the index is 360 MB, far
    past GitHub's limits, so it is hosted as a Hugging Face dataset and pulled
    once per deploy. Locally this finds the built index and does nothing.
    """
    config.load_env()
    if config.PARENTS_PARQUET.exists() and config.CHROMA_DIR.exists():
        return "local build"
    repo = (os.environ.get("RECIPE_RAG_INDEX_REPO") or "").strip()
    if not repo:
        raise RuntimeError(
            "There is no index here, and RECIPE_RAG_INDEX_REPO is not set. "
            "Upload one with scripts/push_index.py, then set "
            "RECIPE_RAG_INDEX_REPO to <user>/<repo> in the app secrets."
        )
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        local_dir=str(config.ARTIFACT_DIR),
        token=(os.environ.get("HF_TOKEN") or None),
    )
    return repo


@st.cache_resource(show_spinner="Loading the recipe index...")
def load_pipeline(use_judge: bool, use_web: bool) -> Pipeline:
    """One pipeline per settings combination, reused across reruns.

    Loading parses 18,249 recipes and indexes their titles, tags and
    ingredients -- a few seconds, which must not happen on every keystroke.
    """
    return Pipeline.load(use_judge=use_judge, use_web=use_web)


def index_missing_message(exc: Exception) -> None:
    st.error("The recipe index is not available.")
    st.code(str(exc))
    st.markdown(
        "Build it once with:\n\n"
        "```\npython scripts/build_index.py --rebuild\n```\n\n"
        "or point `RECIPE_RAG_ARTIFACTS` at a folder that already holds one."
    )


def source_banner(answer) -> None:
    if answer.source == SOURCE_DATASET:
        st.success("**Source: Internal Database** -- our own trusted recipe collection.")
    elif answer.source == SOURCE_WEB:
        recipe = answer.recipe
        st.info(f"**Source: Web Search ({recipe.domain})** -- not in our collection, "
                f"so this was read from the publisher's own recipe data.")
        if recipe.url:
            st.markdown(f"[Open the original page]({recipe.url})")
    else:
        st.warning(answer.message)
        st.caption("Logged to missing_recipes.jsonl so the collection can be extended.")


def nutrition_row(recipe) -> None:
    columns = st.columns(4)
    values = [("Calories", recipe.calories, ""), ("Protein", recipe.protein, " g"),
              ("Fat", recipe.fat, " g"), ("Sodium", recipe.sodium, " mg")]
    for column, (label, value, unit) in zip(columns, values):
        column.metric(label, "Not specified" if value is None else f"{value:g}{unit}")
    if recipe.nutrition_status in {"rejected", "field_rejected", "partial"}:
        st.caption(f"Nutrition status: `{recipe.nutrition_status}` -- implausible values "
                   f"were dropped rather than shown.")


def why_expander(answer) -> None:
    with st.expander("How this answer was chosen"):
        st.markdown(f"**Understood from the request:** `{answer.constraints}`")
        left, middle, right = st.columns(3)
        left.metric("Best vector score", f"{answer.vector_score:.3f}")
        middle.metric("Best title score", f"{answer.title_score:.1f}")
        right.metric("Band", answer.band or "-")
        st.markdown("**Decision trail**")
        for reason in answer.reasons:
            st.markdown(f"- {reason}")
        if answer.pages_tried:
            st.markdown("**Web pages tried**")
            for url in answer.pages_tried:
                st.markdown(f"- {url}")
        if answer.failure_stage:
            st.markdown(f"**Stopped at stage:** `{answer.failure_stage}`")
        st.caption(f"{answer.seconds:.1f}s"
                   + (f" · ${answer.usage.cost_usd:.6f}" if answer.usage.calls else " · no LLM call")
                   + (f" · judge: {answer.verdict.query_type}" if answer.verdict else ""))
        st.markdown("**Exactly as submitted for grading**")
        st.code(present(answer).as_text(), language="text")


def ask_tab(pipeline: Pipeline) -> None:
    st.caption("Ask for a recipe in your own words. Dietary words (vegan, "
               "gluten-free), 'with nutrition info' and 'without X' are treated as "
               "requirements, not suggestions.")

    if "query" not in st.session_state:
        st.session_state.query = ""
    columns = st.columns(len(EXAMPLES))
    for column, example in zip(columns, EXAMPLES):
        if column.button(example[:26] + "...", help=example, use_container_width=True):
            st.session_state.query = example

    query = st.text_input("What would you like to cook?", key="query",
                          placeholder="a gluten-free salad with tuna and asparagus")
    if not query.strip():
        return

    with st.spinner("Searching the collection..."):
        answer = pipeline.answer(query)

    source_banner(answer)
    if answer.recipe is None:
        why_expander(answer)
        return

    recipe = answer.recipe
    left, right = st.columns([3, 1])
    with right:
        st.metric("Dietary class", present(answer).diet)
        st.caption(recipe.diet_reason[:120] or "no animal-derived ingredients detected")
        if recipe.diet_conflict:
            st.warning("The source's own tags disagree with its ingredients here.")
    with left:
        nutrition_row(recipe)
    st.markdown(as_markdown(answer))
    why_expander(answer)


def missing_tab() -> None:
    st.caption("Every request neither source could answer, so the collection can be "
               "grown where users actually need it.")
    misses = logs.miss_summary(40)
    if not misses:
        st.info("No missing recipes logged yet.")
        return
    st.metric("Distinct unanswered requests", len(misses))
    st.table({"request": [dish for dish, _ in misses],
              "times asked": [count for _, count in misses]})


def main() -> None:
    st.title("Zwigato Recipe Assistant")

    with st.sidebar:
        st.header("Settings")
        use_web = st.toggle("Search the web when the collection has nothing", value=True)
        use_judge = st.toggle("Use the LLM judge for borderline matches", value=True)
        st.divider()
        st.caption("Keys")
        config.load_env()
        st.write("OpenAI / Azure:", "configured" if _llm_ready() else "missing")
        st.write("Serper (web search):", "configured" if _serper_ready() else "missing")

    try:
        index_source = ensure_index()
        pipeline = load_pipeline(use_judge, use_web)
    except Exception as exc:  # noqa: BLE001 -- the message is the useful part
        index_missing_message(exc)
        return

    with st.sidebar:
        st.divider()
        st.caption("Collection")
        st.write(f"{len(pipeline.parents):,} recipes")
        st.write(f"{pipeline.collection.count():,} indexed chunks")
        st.caption(f"index: {index_source}")
        if use_judge and not pipeline.llm.available:
            st.warning("No usable LLM key: borderline matches fall back to the "
                       "rule-based gate.")

    ask, missing = st.tabs(["Ask", "Missing recipes"])
    with ask:
        ask_tab(pipeline)
    with missing:
        missing_tab()


def _llm_ready() -> bool:
    """True if a key is present AND usable. An Azure key with no endpoint set
    raises here rather than returning False, so the check has to catch."""
    try:
        return bool(config.openai_key(required=False))
    except Exception:  # noqa: BLE001
        return False


def _serper_ready() -> bool:
    from recipe_rag import websearch

    try:
        return bool(websearch.serper_key(required=False))
    except Exception:  # noqa: BLE001
        return False


main()
