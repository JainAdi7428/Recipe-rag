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

The dry run lists exactly what would be uploaded: the Chroma collection, the
parent table and the build report — 8 files, 361 MB. The browsable SQLite copy,
the logs and the evaluation output stay local.

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

# only for an Azure OpenAI key:
AZURE_OPENAI_ENDPOINT = "https://<resource>.openai.azure.com/"
AZURE_OPENAI_DEPLOYMENT = "gpt-4.1-mini-4"
AZURE_OPENAI_API_VERSION = "2025-01-01-preview"
```

Streamlit exposes top-level secrets as environment variables, which is exactly
where `config.load_env()` looks, so no code change is needed.

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
- When nothing can be found it apologises and records the request, which the
  **Missing recipes** tab reads back — the dataset-expansion loop the spec asks
  for, visible in the product rather than buried in a log file.
