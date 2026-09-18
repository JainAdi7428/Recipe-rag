# Architecture

The UC13 "Zwigato" recipe assistant: answer from a trusted internal collection
first, fall back to the web, and say so plainly when neither can help.

## The flow

```
 USER QUERY   "I need a coconut cake recipe for Easter with full nutrition info"
     |
     v
 1. UNDERSTAND THE REQUEST                                    recipe_rag/query.py
     strict constraints  ->  diet | "needs nutrition" | "without X"
     descriptive words   ->  coconut, cake, easter
     category tags       ->  Cake, Coconut, Easter      (exact matching only)
     |
     v
 2. COLLECT CANDIDATES  (three sources, one pool of ~55 recipes)
     a. vector search over 67,010 child chunks            recipe_rag/store.py
     b. fuzzy title match over all 18,249 titles          recipe_rag/lexical.py
     c. recipes carrying the request's two rarest tags    recipe_rag/rerank.py
     |
     v
 3. APPLY THE STRICT CONSTRAINTS       a violation is removed, never demoted
     |
     v
 4. RE-RANK       score = vector similarity + 0.30 x coverage
     coverage = rarity-weighted share of the request's words found in the
                recipe's title / tags (full credit) or ingredients (half)
     |
     v
 5. CHOOSE THE SOURCE        bands, computed on the survivors of step 3
     CONFIDENT  title >= 85 or vector >= 0.70   ->  exact title, else re-ranked #1
     AMBIGUOUS  title >= 55 or vector >= 0.35   ->  LLM judge reads the top 8
     HOPELESS   below both                      ->  straight to the web
     |
     v
 6. WEB FALLBACK                                        recipe_rag/websearch.py
     Serper search -> skip aggregators -> fetch up to 4 pages
     -> require schema.org Recipe markup                recipe_rag/extract.py
     -> require the page to be the dish that was asked for
     -> run OUR diet classifier and nutrition validator on it
     |
     v
 7. APOLOGISE AND LOG                                     recipe_rag/logs.py
```

Two questions are answered separately, and keeping them apart is the central
design decision:

| question | decided by |
|---|---|
| **which source** — dataset, web or nothing | the bands, plus the LLM judge in the middle band |
| **which recipe** — among the dataset's candidates | the re-ranker, on the request's own words |

Phase 1 let the vector score answer both at once. On descriptive requests it got
the source right and the recipe wrong, because the top five candidates sat within
0.02 of each other, making "rank 1" close to arbitrary.

## Modules

| file | responsibility |
|---|---|
| `loader.py` | read `full_format_recipes.json`, normalise, drop unusable rows |
| `dedup.py` | exact-duplicate removal, then variant grouping |
| `diet.py` | veg / non-veg / vegan / eggetarian from the ingredient list |
| `nutrition.py` | reject impossible values, per field or per row |
| `chunker.py` | one recipe -> title-prefixed child chunks under the token cap |
| `embedder.py` | ONNX all-MiniLM-L6-v2, no torch |
| `store.py` | Chroma collection (children) + Parquet (parents) |
| `retrieve.py` | fold child-chunk scores into one score per recipe |
| `lexical.py` | fuzzy title matching over every title |
| `query.py` | split a request into constraints and descriptive terms |
| `rerank.py` | rarity-weighted coverage, constraint filters, tag candidates |
| `gate.py` | the three bands |
| `judge.py` | the LLM relevance judge for the ambiguous band |
| `websearch.py` | Serper search, result filtering, page fetching |
| `extract.py` | schema.org Recipe markup -> our recipe shape |
| `pipeline.py` | the flow above, end to end |
| `present.py` | the grader's output format |
| `logs.py` | query log and missing-recipe log |
| `groundtruth.py` | read the validation file and score against it |

## Where things are stored

```
.artifacts/
  chroma_db/                 67,010 child chunks: ids, text, metadata, vectors
  parents.parquet            18,249 recipes, 27 columns (the full record)
  query_log.jsonl            one line per request
  missing_recipes.jsonl      requests neither source could answer
  web_eval.json              raw rows behind docs/web-evaluation.md
  browse.sqlite3             optional readable copy for a SQLite viewer
```

Children are embedded and searched; parents hold the full record. Chroma stores
metadata as one row per field per chunk, so keeping the 27-column record out of
it avoids about 1.8 million rows in the file that has to ship to Streamlit Cloud.

## How the spec's requirements are met

| requirement | where |
|---|---|
| dataset-first retrieval | steps 2–5; the web is only reached when the bands or the judge rule the dataset out |
| web fallback | step 6, and only with the publisher's own structured recipe data |
| explicit apology + logging | step 7, with a `failure_stage` on every miss |
| dietary classification | `diet.py`, run at ingest **and** on every web result |
| clear source attribution | `present.py`: "Internal Database" / "Web Search (site)" / "NA" |
| no hallucinated recipes | nothing generates recipe text; the LLM only ever *chooses* between candidates it was shown |
| under 60 s | measured: median 0.2 s from the dataset, 3.1 s via the web |
