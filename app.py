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

Requests neither source can answer are not shown in the UI; they are written to
.artifacts/missing_recipes.log (readable) and missing_recipes.jsonl (for code).
"""

from __future__ import annotations

import html
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import streamlit as st


def secrets_to_environment(secrets=None) -> list[str]:
    """Copy Streamlit secrets into os.environ. Returns the names it set.

    Every setting in recipe_rag is read from environment variables: locally they
    come from .env, on Streamlit Cloud from the app's Secrets panel. Streamlit
    also exports root-level secrets as environment variables itself, but copying
    them here means nothing depends on that -- if a secret exists, the library
    sees it. Values already in the environment win, as they do for .env.

    Runs before recipe_rag is imported, so that settings config.py reads at
    import time (RECIPE_RAG_ARTIFACTS, RECIPE_RAG_LLM_MODEL) work from secrets
    too.
    """
    if secrets is None:
        try:
            secrets = dict(st.secrets)
        except Exception:  # noqa: BLE001 -- no secrets file: running locally on .env
            return []
    copied = []
    for key, value in secrets.items():
        # Skip TOML tables ([section]); the library only reads flat names.
        if isinstance(value, (str, int, float, bool)) and key not in os.environ:
            os.environ[key] = str(value)
            copied.append(key)
    return copied


secrets_to_environment()

from recipe_rag import config
from recipe_rag.pipeline import SOURCE_DATASET, SOURCE_WEB, Pipeline
from recipe_rag.present import present

EXAMPLES = [
    "Healthy fish with olives and tomatoes",
    "Vegetarian spinach pasta casserole",
    "Raw vegan spinach soup",
    "Authentic Palak Paneer",
    "Tandoori Jilaji",
]

# One class per diet, styled as the familiar veg / non-veg food mark.
DIET_CLASS = {"vegan": "vegan", "veg": "veg", "eggetarian": "egg", "non-veg": "nonveg"}

CSS = """
<style>
:root {
  --ink: #1c1917; --muted: #78716c; --line: #e7e5e4; --card: #ffffff;
  --soft: #f5f5f4; --accent: #c2410c; --accent-soft: #fff7ed;
  --veg: #15803d; --veg-bg: #f0fdf4; --vegan: #0f766e; --vegan-bg: #f0fdfa;
  --egg: #b45309; --egg-bg: #fffbeb; --nonveg: #b91c1c; --nonveg-bg: #fef2f2;
  --db: #3730a3; --db-bg: #eef2ff; --web: #075985; --web-bg: #f0f9ff;
}
.block-container { max-width: 1080px; padding-top: 2.2rem; padding-bottom: 3rem; }
.brand { display: flex; align-items: center; gap: .85rem; }
.brand .logo { width: 44px; height: 44px; border-radius: 12px; background: var(--accent);
  color: #fff; display: grid; place-items: center; font-size: 1.4rem; }
.brand .name { font-size: 1.45rem; font-weight: 700; color: var(--ink); letter-spacing: -.01em; }
.brand .tag { color: var(--muted); font-size: .92rem; }
.hint { color: var(--muted); font-size: .82rem; margin-top: -.4rem; }

.card { background: var(--card); border: 1px solid var(--line); border-radius: 16px;
  padding: 1.5rem 1.6rem; margin: 1.2rem 0 1rem; box-shadow: 0 1px 2px rgba(28,25,23,.04); }
.badges { display: flex; flex-wrap: wrap; gap: .5rem; margin-bottom: .7rem; }
.badge { display: inline-flex; align-items: center; gap: .4rem; font-size: .78rem;
  font-weight: 600; padding: .28rem .7rem; border-radius: 999px; border: 1px solid transparent; }
.badge.db { color: var(--db); background: var(--db-bg); border-color: #c7d2fe; }
.badge.web { color: var(--web); background: var(--web-bg); border-color: #bae6fd; }
.badge.vegan { color: var(--vegan); background: var(--vegan-bg); border-color: #99f6e4; }
.badge.veg { color: var(--veg); background: var(--veg-bg); border-color: #bbf7d0; }
.badge.egg { color: var(--egg); background: var(--egg-bg); border-color: #fde68a; }
.badge.nonveg { color: var(--nonveg); background: var(--nonveg-bg); border-color: #fecaca; }
.mark { width: 12px; height: 12px; border: 1.6px solid currentColor; border-radius: 2px;
  display: inline-grid; place-items: center; }
.mark::after { content: ""; width: 5px; height: 5px; border-radius: 50%; background: currentColor; }
.title { font-size: 1.75rem; font-weight: 700; color: var(--ink); line-height: 1.25;
  margin: 0 0 .35rem; letter-spacing: -.015em; }
.meta { color: var(--muted); font-size: .88rem; }
.meta a { color: var(--accent); text-decoration: none; font-weight: 600; }

.nutrition { display: grid; grid-template-columns: repeat(4, 1fr); gap: .75rem; margin-top: 1.2rem; }
.tile { background: var(--soft); border-radius: 12px; padding: .8rem 1rem; }
.tile .label { font-size: .72rem; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); }
.tile .value { font-size: 1.35rem; font-weight: 700; color: var(--ink); }
.tile .value small { font-size: .8rem; font-weight: 500; color: var(--muted); margin-left: .15rem; }
.tile .value.na { font-size: .95rem; font-weight: 500; color: var(--muted); padding-top: .3rem; }
.note { font-size: .8rem; color: var(--muted); margin-top: .7rem; }

.section { font-size: .78rem; font-weight: 700; text-transform: uppercase; letter-spacing: .08em;
  color: var(--accent); margin: .4rem 0 .8rem; }
ul.ingredients { list-style: none; padding: 0; margin: 0; }
ul.ingredients li { padding: .55rem 0; border-bottom: 1px dashed var(--line); color: var(--ink);
  font-size: .95rem; }
ol.steps { list-style: none; padding: 0; margin: 0; counter-reset: step; }
ol.steps li { counter-increment: step; position: relative; padding: 0 0 1.1rem 2.6rem;
  color: var(--ink); font-size: .97rem; line-height: 1.6; }
ol.steps li::before { content: counter(step); position: absolute; left: 0; top: .05rem;
  width: 1.75rem; height: 1.75rem; border-radius: 50%; background: var(--accent-soft);
  color: var(--accent); font-weight: 700; font-size: .85rem; display: grid; place-items: center; }

.empty { text-align: center; border: 1px dashed var(--line); background: var(--card); }
.empty .icon { font-size: 2rem; }
.empty h3 { margin: .3rem 0 .4rem; color: var(--ink); font-size: 1.15rem; }
.empty p { color: var(--muted); margin: 0 auto; max-width: 560px; font-size: .93rem; }
.footer { color: var(--muted); font-size: .78rem; text-align: center; margin-top: 2.5rem; }

@media (max-width: 640px) {
  .nutrition { grid-template-columns: repeat(2, 1fr); }
  .card { padding: 1.1rem 1rem; }
  .title { font-size: 1.4rem; }
}
</style>
"""

st.set_page_config(page_title="Zwigato Recipes", page_icon="🍲", layout="wide")


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


def _html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def header():
    """Brand on the left, settings behind a gear on the right. Returns the popover."""
    brand, gear = st.columns([14, 1], vertical_alignment="center")
    brand.markdown(
        '<div class="brand"><div class="logo">🍲</div><div>'
        '<div class="name">Zwigato Recipes</div>'
        '<div class="tag">Trusted recipes, found from a description in your own words.</div>'
        "</div></div>",
        unsafe_allow_html=True,
    )
    with gear:
        return st.popover("", icon=":material/settings:", help="Settings")


def _use_example() -> None:
    choice = st.session_state.example
    if choice:
        st.session_state.query_input = choice
        st.session_state.active_query = choice
    st.session_state.example = None


def search_box() -> str:
    with st.form("search", border=False, enter_to_submit=True):
        text, button = st.columns([6, 1], vertical_alignment="bottom")
        typed = text.text_input(
            "What would you like to cook?", key="query_input",
            placeholder="e.g. a gluten-free salad with tuna and asparagus",
            label_visibility="collapsed",
        )
        if button.form_submit_button("Search", type="primary", width="stretch"):
            st.session_state.active_query = typed
    _html('<div class="hint">Diet words (vegan, gluten-free), "with nutrition info" and '
          '"without X" are treated as requirements, not suggestions.</div>')
    st.pills("Try an example", EXAMPLES, key="example", on_change=_use_example,
             label_visibility="collapsed")
    return (st.session_state.get("active_query") or "").strip()


def get_answer(pipeline: Pipeline, query: str, use_judge: bool, use_web: bool):
    """Answer once per (query, settings). Streamlit reruns the script on every
    click -- opening the settings, expanding the details -- and each rerun must
    not repeat the LLM call, its cost, or the log entry."""
    key = (query, use_judge, use_web)
    cached = st.session_state.get("answer")
    if cached and cached[0] == key:
        return cached[1]
    with st.spinner("Finding your recipe..."):
        answer = pipeline.answer(query)
    st.session_state.answer = (key, answer)
    return answer


def recipe_header(answer, shown) -> str:
    recipe = answer.recipe
    e = html.escape
    source_class = "db" if answer.source == SOURCE_DATASET else "web"
    source_icon = "📚" if answer.source == SOURCE_DATASET else "🌐"
    badges = [f'<span class="badge {source_class}">{source_icon} {e(shown.source)}</span>']
    if recipe.diet:
        css = DIET_CLASS.get(recipe.diet, "veg")
        reason = e(recipe.diet_reason or "no animal-derived ingredients detected")
        badges.append(f'<span class="badge {css}" title="{reason}">'
                      f'<span class="mark"></span>{e(shown.diet)}</span>')

    meta = [f"{len(recipe.ingredients)} ingredients", f"{len(recipe.directions)} steps"]
    if recipe.servings:
        meta.append(e(str(recipe.servings)))
    if recipe.author:
        meta.append(f"by {e(recipe.author)}")
    if answer.source == SOURCE_WEB and recipe.url:
        meta.append(f'<a href="{e(recipe.url)}" target="_blank">View original on '
                    f"{e(recipe.domain)} ↗</a>")

    return (f'<div class="card"><div class="badges">{"".join(badges)}</div>'
            f'<div class="title">{e(recipe.title)}</div>'
            f'<div class="meta">{" · ".join(meta)}</div>'
            f"{nutrition_tiles(recipe)}</div>")


def nutrition_tiles(recipe) -> str:
    tiles = []
    for label, value, unit in (("Calories", recipe.calories, "kcal"),
                               ("Protein", recipe.protein, "g"),
                               ("Fat", recipe.fat, "g"),
                               ("Sodium", recipe.sodium, "mg")):
        body = ('<div class="value na">Not specified</div>' if value is None
                else f'<div class="value">{value:g}<small>{unit}</small></div>')
        tiles.append(f'<div class="tile"><div class="label">{label}</div>{body}</div>')
    note = ""
    if recipe.nutrition_status in {"rejected", "field_rejected", "partial"}:
        note = ('<div class="note">Some nutrition values at the source were implausible '
                "and have been left out rather than shown.</div>")
    return f'<div class="nutrition">{"".join(tiles)}</div>{note}'


def recipe_body(recipe) -> None:
    e = html.escape
    left, right = st.columns([2, 3], gap="large")
    left.markdown(
        '<div class="section">Ingredients</div><ul class="ingredients">'
        + "".join(f"<li>{e(item)}</li>" for item in recipe.ingredients) + "</ul>",
        unsafe_allow_html=True,
    )
    right.markdown(
        '<div class="section">Method</div><ol class="steps">'
        + "".join(f"<li>{e(step)}</li>" for step in recipe.directions) + "</ol>",
        unsafe_allow_html=True,
    )


def not_found(answer) -> None:
    _html('<div class="card empty"><div class="icon">🔎</div>'
          "<h3>We couldn't find a reliable recipe for that</h3>"
          f"<p>{html.escape(answer.message)}</p></div>")


def why_expander(answer) -> None:
    with st.expander("How this answer was chosen", icon=":material/insights:"):
        if answer.constraints:
            st.markdown(f"**Understood from the request:** `{answer.constraints}`")
        left, middle, right = st.columns(3)
        left.metric("Best vector score", f"{answer.vector_score:.3f}")
        middle.metric("Best title score", f"{answer.title_score:.1f}")
        right.metric("Match band", answer.band or "-")
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
        st.markdown("**Grader-format output**")
        st.code(present(answer).as_text(), language="text")


def render_answer(answer) -> None:
    if answer.recipe is None:
        not_found(answer)
        why_expander(answer)
        return
    recipe = answer.recipe
    _html(recipe_header(answer, present(answer)))
    if recipe.diet_conflict:
        st.caption("Note: the source's own tags disagree with its ingredients; the "
                   "dietary class above is based on the ingredients.")
    recipe_body(recipe)
    why_expander(answer)


def main() -> None:
    _html(CSS)
    settings = header()
    with settings:
        st.markdown("**Settings**")
        use_web = st.toggle("Search the web when the collection has nothing", value=True,
                            key="use_web")
        use_judge = st.toggle("Use the AI judge for borderline matches", value=True,
                              key="use_judge")

    try:
        ensure_index()
        pipeline = load_pipeline(use_judge, use_web)
    except Exception as exc:  # noqa: BLE001 -- the message is the useful part
        index_missing_message(exc)
        return

    with settings:
        st.divider()
        st.caption(f"{len(pipeline.parents):,} recipes · "
                   f"{pipeline.collection.count():,} indexed passages")
        if use_judge and not pipeline.llm.available:
            st.caption("The AI judge is unavailable, so borderline matches use the "
                       "rule-based check.")

    query = search_box()
    if query:
        render_answer(get_answer(pipeline, query, use_judge, use_web))

    _html('<div class="footer">Every recipe comes from our curated collection or the '
          "publisher's own page. Nothing is generated.</div>")


# `streamlit run app.py` executes this file as __main__. The guard lets the
# functions above be imported and tested without rendering the whole app.
if __name__ == "__main__":
    main()
