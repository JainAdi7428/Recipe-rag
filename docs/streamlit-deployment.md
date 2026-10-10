# Running and deploying the app

## Locally

```
.venv\Scripts\Activate.ps1
streamlit run app.py
```

It opens on <http://localhost:8501> and uses the index already in `.artifacts/`.

The sidebar has two switches, both on by default:

- **Search the web when the collection has nothing** — turn off to work offline.
- **Use the LLM judge for borderline matches** — turn off to see the
  rule-based behaviour. With no usable key the app falls back to the rule gate
  and says so, rather than erroring.

Every answer shows where it came from. "How this answer was chosen" expands to
the constraints understood from the request, the scores, the decision trail, any
web pages tried, and the answer exactly as it would be submitted for grading.

## Deploying to Streamlit Cloud

The app code needs no changes. Three things stand between a working local app
and a deployed one, and none of them is hard.

### 1. The code has to be on GitHub

The project is not a git repository yet.

```
git init
git add .
git commit -m "Recipe RAG system"
```

`.gitignore` already excludes `.env`, `.venv/`, `.artifacts/` and the source
data, so no key and no 360 MB artifact can be committed by accident. Create an
empty repository on GitHub (the spec asks for a Sigmoid account), then:

```
git remote add origin https://github.com/<you>/<repo>.git
git branch -M main
git push -u origin main
```

### 2. The index has to live somewhere else

`.artifacts/` is 361 MB of index — past GitHub's 100 MB per-file limit, and
not something to keep in git anyway. It goes to a Hugging Face dataset repo,
which is free and needs only an account:

```
huggingface-cli login
python scripts/push_index.py --repo <you>/recipe-rag-index --dry-run
python scripts/push_index.py --repo <you>/recipe-rag-index
```

The dry run prints the destination, the visibility and the exact file list
before anything leaves the machine: 7 files, 361 MB — the Chroma collection and
the parent table. Everything else stays local, including your query log, the
missing-recipe log, the 87 MB browsable copy, and `build_report.json`, which
records the absolute paths of the machine that built the index.

**The dataset is created private.** Pass `--public` to publish it, which is a
deliberate act rather than a default: it is a redistribution of the Epicurious
corpus, and it is your account's name on it.

Because it is private, the app needs a read token to fetch it: create one at
<https://huggingface.co/settings/tokens> and add it to the app's secrets as
`HF_TOKEN`.

The app pulls it on first start (`ensure_index()` in `app.py`) and caches it for
the life of the container. Set `RECIPE_RAG_INDEX_REPO` to the repo name; add
`HF_TOKEN` too if the dataset is private.

### 3. Secrets

Streamlit Cloud has its own secrets store. Add the same names the `.env` file
uses, plus the index repo:

```toml
OPENAI_API_KEY = "..."
SERPER_API_KEY = "..."
RECIPE_RAG_INDEX_REPO = "<you>/recipe-rag-index"
HF_TOKEN = "hf_..."          # a read token, since the dataset is private

# only for an Azure OpenAI key:
AZURE_OPENAI_ENDPOINT = "https://<resource>.openai.azure.com/"
AZURE_OPENAI_DEPLOYMENT = "gpt-4.1-mini-4"
AZURE_OPENAI_API_VERSION = "2025-01-01-preview"
```

Every setting in `recipe_rag` is read from environment variables. `app.py`
copies the secrets into the environment itself (`secrets_to_environment()`),
before the library is imported, so nothing depends on Streamlit's own habit of
exporting root-level secrets. Verified by simulating a fresh container: empty
index folder, the repo name and token supplied *only* as secrets, the machine's
stored Hugging Face login disabled — the app fetched the private index in 31
seconds and answered correctly.

### 4. Deploy

On <https://share.streamlit.io>: **New app**, pick the repository and branch,
set the main file to `app.py`, paste the secrets, deploy. The first start
downloads the index, which takes a few minutes; later restarts reuse it.

## Memory: the one real risk

Measured locally, with `psutil`:

| stage | resident memory |
|---|---|
| imports | 83 MB |
| pipeline loaded | 387 MB |
| after the first query | 674 MB |
| after five queries | 680 MB |

The jump on the first query is the ONNX embedding model, loaded lazily by Chroma
to encode the query.

680 MB plus Streamlit's own overhead is uncomfortably close to the free tier's
limit. If the app is killed and restarted on its first query, the usual causes
and fixes, cheapest first:

1. **Drop columns the app never reads** when loading the parent table
   (`raw_*`, `title_norm`, `date`, `rating`) — the ingredient and direction text
   is most of the 9 MB Parquet expanded in memory.
2. **Turn the judge off** in the sidebar; it does not change memory, but it
   removes the openai client from the import path.
3. Ask for more resources, or host the app somewhere with more RAM.

Nothing here is fixed by shrinking the index on disk: the memory goes on the
HNSW vectors (107 MB), the parent table and the encoder.

## What the app deliberately does not do

- It never writes recipe text. Every line shown is the stored record or the
  publisher's own structured data.
- It does not hide the fallback. A web answer is labelled with the site it came
  from and links to the original page.
- When nothing can be found it apologises and records the request in
  `.artifacts/missing_recipes.log` (one readable line per miss) and
  `missing_recipes.jsonl` (full record) — the dataset-expansion loop the spec
  asks for. The log is kept out of the user-facing UI.
