# Zwigato Recipe Assistant (UC13)

A recipe question-answering system that **answers from a trusted internal
collection first**, falls back to **web search** when the collection has nothing,
and **says so plainly** when neither can help — attributing every answer,
classifying every recipe as veg / non-veg / vegan / eggetarian, and never
inventing a recipe.

Built on the Epicurious dataset: 20,130 raw records become **18,249 recipes** and
**67,010 embedded chunks**.

## Quick start

```powershell
# 1. Environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Build the index once (~35 min), into .artifacts\
python scripts\build_index.py --rebuild

# 3. The app
streamlit run app.py

# 4. Or ask from the terminal
python scripts\ask.py "I want a gluten-free salad with tuna and asparagus"
python scripts\ask.py "I need a recipe for Tandoori Jilaji" --grader
```

Secrets go in `.env` (gitignored; copy `.env.example`). Everything except the
LLM judge works without any key; the web fallback needs a free
[serper.dev](https://serper.dev) key.

## How it works

```
request -> understand constraints -> collect candidates -> apply constraints
        -> re-rank -> choose source -> [dataset | web | apology] -> attributed answer
```

The system answers two separate questions, and keeping them apart is the central
design decision:

- **Which source?** Retrieval scores put a request in one of three bands.
  Confident and hopeless are settled for free; the uncertain middle goes to an
  LLM judge, because present and absent recipes overlap between roughly 0.49 and
  0.60 cosine and no single threshold can separate them.
- **Which recipe?** A re-ranker scores candidates on the request's own words —
  found in a recipe's title, category tags or ingredients — weighted by how rare
  each word is, with stated requirements (diet, "with nutrition info",
  "without X") applied as hard filters.

Full detail in [docs/architecture.md](docs/architecture.md); the reasoning
behind each choice in [docs/decisions.md](docs/decisions.md).

## Where it stands

Full system: judge and web enabled.

| | |
|---|---|
| Grader's ground truth: right recipe | 4 / 6 dataset rows |
| Grader's ground truth: diet label | **6 / 6** |
| Grader's ground truth: nutrition | 6 / 6 |
| Grader's ground truth: no recipe invented when none exists | **2 / 2** |
| Dataset-vs-web decision (1,006 queries) | 95.6% right recipe, 98.8% recall |
| **Absent dishes wrongly answered from the dataset** | **0.0%** |
| Descriptive queries (145 held out): right recipe | 42.8% vector alone, **65.5%** re-ranked |
| Web fallback (47 dishes) | **35/35** answered, **0** fabricated, 3.0 s median |
| Response time | 0.3 s median, 8.8 s slowest, against a 60 s target |
| **Cost per 100 queries** | **$0.059** of a $2.00 budget |
| Tests | **351**, all passing |

Numbers, methods and their limits: [docs/evaluation.md](docs/evaluation.md) and
[docs/web-evaluation.md](docs/web-evaluation.md).

**Two things to know.** The reachable Azure deployment is `gpt-4.1-mini`, not
the `gpt-4o-mini` the spec mandates. And the LLM judge is a measured trade: in
the 10.5% of queries that reach it, it lifts correct identity from 36% to 91%
and cuts wasted web calls from 64% to 9%, while over-acceptance rises from 1.8%
to 12.3% -- most of which turn out to be dishes this corpus really does hold
under an English name. See [docs/open-questions.md](docs/open-questions.md).

## Documentation

| file | contents |
|---|---|
| [architecture.md](docs/architecture.md) | the flow, the modules, where things are stored |
| [decisions.md](docs/decisions.md) | every non-obvious choice and its evidence |
| [evaluation.md](docs/evaluation.md) | what is measured, current numbers, their limits |
| [web-evaluation.md](docs/web-evaluation.md) | the web fallback, measured (generated) |
| [web-fallback.md](docs/web-fallback.md) | how the web path works and why |
| [streamlit-deployment.md](docs/streamlit-deployment.md) | running locally and deploying |
| [open-questions.md](docs/open-questions.md) | what is unfinished, and what it needs |

## Layout

```
app.py                 Streamlit UI
recipe_rag/            the library: ingestion, retrieval, decision, web, output
scripts/               build, inspect, evaluate, ask
tests/                 python tests/run_all.py -- no network, no index needed
docs/                  the files above
.artifacts/            index, logs, evaluation output (gitignored, rebuildable)
```

> **On OneDrive.** This folder is synced and `.artifacts/` holds a live SQLite
> database. Pause syncing during long builds. Never exclude a folder through
> OneDrive's *Choose folders* — that deletes the local copy.
