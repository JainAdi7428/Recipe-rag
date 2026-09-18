# Decisions, and why

Every non-obvious choice in this project, with the evidence behind it. Where a
decision was made by the user rather than by measurement, it says so.

## Data

**Parent/child chunking, split by section.** Each recipe is a parent; children
are a summary card, the ingredients, and the directions, each prefixed with the
recipe title. Measured before choosing: the median recipe is 1,084 characters
and 36.8% of direction blocks exceed 900, so a fixed "2–3 children per recipe"
would have truncated about 7,400 recipes. Result: 18,249 parents, 67,010
children, zero truncation.

**Exact-duplicate removal, variant grouping for the rest.** 1,794 rows were
byte-identical after normalising and were dropped. 366 titles had genuinely
different versions; those are kept and tagged so retrieval can collapse them to
the best one. *Chosen by the user* over both "keep only the best copy" and
"merge into one".

**all-MiniLM-L6-v2 through chromadb's bundled ONNX export.** The model the spec
names, without torch — which matters because Streamlit Cloud's free tier has
about 1 GB of RAM and Python 3.14 wheels for torch are a risk. *Chosen by the
user.* Changing the encoder means re-embedding everything, so this is expensive
to revisit.

**Parents in Parquet, children in Chroma.** Chroma stores metadata as one row
per field per chunk. Putting the 27-column record in it would mean ~1.8 million
rows in the artifact shipped to Streamlit Cloud.

## Dietary classification

**Rules over an LLM, run at ingest.** Deterministic, free, explainable, and
reusable: the identical function runs on web results, which is what makes
classification consistent across sources rather than merely similar.

**Ingredients decide; category tags may only make the label stricter.** The
dataset's own tags are unreliable — its "Miso Soup" carries a `Vegan` tag while
listing bonito flakes. Tags can escalate a label (a `Pork` tag on a rub whose
ingredients hide the meat) but never relax one, because wrongly warning a
vegetarian is a nuisance while wrongly calling something vegan is a real harm.

**The title may add meat or fish, never dairy or egg.** A title often names the
protein an ingredient list leaves vague ("Chicken Alfredo" over "2 cups cooked,
shredded"). But for dairy and egg a title names a *style*: "Raw Cream of Spinach
Soup" is avocado and olive oil, and the word "cream" was making it veg when the
grader calls it vegan. Fixing this corrected **50 labels**, each reviewed by
hand before being written.

**Sub-recipes hide ingredients.** "Pastry Cream" and "lemon curd" are made of
eggs but say so nowhere; bare "yolks" appears without the word "egg". All are in
the egg vocabulary.

## Nutrition

**Impossible values are dropped, never corrected.** Two or more impossible
fields means the row was mis-parsed and the whole row goes; exactly one means a
single extreme-but-real measurement and only that field goes. Evidence: 68 of
111 rejected rows had just one bad field, and were mostly brines and salt
crusts whose sodium is genuinely enormous.

**Carbohydrates are never shown, for any source.** The dataset has no
carbohydrate field at all. *Chosen by the user*: not shown, rather than marked
unavailable or estimated. It applies to web recipes too, even when their markup
carries carbs, so that the two sources present identical fields. The grader's
own answer for one row estimates carbs and labels the estimate — we do not,
because estimating a health-adjacent number is the hallucination the spec
forbids.

## Retrieval and ranking

**Two signals, because they fail in different places.** Dense vectors are poor
at proper nouns: "Spanakopita" is in the dataset, retrieves itself, and still
scores 0.487 — below three dishes that are genuinely absent. Fuzzy title
matching is the opposite: perfect on names and typos, useless on paraphrase.

**Query coverage, not `WRatio`.** The off-the-shelf scorer gave four different
wrong matches an identical 85.5, each sharing a single word ("Masala Dosa" vs
"Chicken Tikka Masala"). Coverage — how much of the *query* the title accounts
for, weighted by word length — drops them to 54–69.

**Every title is scored, with two tie-breaks.** An early version shortlisted 60
titles with `WRatio` and re-scored those; the grader's queries proved the
shortlist drops correct answers. Scoring all 18,249 exactly, ties are broken
first by "the title begins with what was typed" (a truncated name), then by
"fewest extra words" (an exact name). Both tie-breaks were added because
removing them measurably hurt: 93.8% -> 87.1% without the second, and truncated
names fell to 69% without the first.

**Constraints filter; tags only boost.** A stated diet, "with nutrition info"
and "without X" remove candidates outright. Category tags do not, because they
are incomplete: the grader's "Spinach Noodle Casserole" carries neither a
`Spinach` nor a `Casserole` tag.

**Words are weighted by rarity.** `Vegetarian` is on 6,167 recipes and narrows
almost nothing; `Easter` is on 210 and decides the Easter-cake query by itself.

**The re-ranking weight is 0.30, chosen against three sets that disagree.** On
145 held-out descriptive queries, re-ranking lifts R@1 from 42.8% (vector alone)
to 65.5% at 0.30 and 73.1% at 0.80; named-dish lookups are flat throughout; the
grader's six rows score 66.7% at 0.30 and 50% above it. The held-out set is by
far the largest, but its queries were written by a model shown each recipe's own
title, tags and ingredients, which biases it towards word coverage. 0.30 is the
largest weight that costs nothing on the graded rows while taking most of the
gain over pure vector ranking. The full table is in evaluation.md.

## Deciding the source

**Three bands, not one threshold.** Present and absent recipes overlap between
roughly 0.49 and 0.60 cosine, so no single cutoff is right — it only chooses
which way to be wrong. The confident and hopeless ends are free; the middle goes
to an LLM judge. *The judge was the user's idea*, over escalating to the web on
a number alone.

**The bands are computed on candidates that survive the constraints.** A strong
match that is not vegan says nothing about whether a vegan answer exists.

**Prefer the web when genuinely uncertain.** A wasted web call costs a few
seconds; a fabricated source attribution is the failure the spec penalises.

## The web fallback

**Structured markup or nothing.** A page must publish schema.org Recipe data or
it is skipped. The alternative — handing scraped HTML to an LLM and asking it to
write out the recipe — is exactly how quantities get invented.

**A preference list of publishers, not an allowlist.** Sites known to publish
clean markup are tried first, but any site is accepted if its markup is valid.
An allowlist would have made most Indian and South-East Asian recipes
unreachable, which is precisely this dataset's gap.

**The page must be the dish that was asked for.** Search always returns
something: ask for "Tandoori Jilaji" and four Tandoori Chicken pages come back,
each with perfect markup for a dish nobody asked for. A page must account for at
least 75% of the request -- measured: across 47 dishes every correct web
answer scored 100%, while the one wrong page ("Deep Fried Moon Pies" for
"Deep Fried Moon Rocks") scored 68% -- and — unlike dataset ranking — an unknown word counts
against it, because a word appearing nowhere is the whole reason to refuse.

**Time is bounded twice.** A page download has a 15-second wall-clock deadline
and the whole web leg has 30 seconds, because `requests`' own timeout bounds the
gap between bytes rather than the download: one measured page trickled for 350
seconds inside a 12-second timeout.

**Nothing generates recipe text.** Dataset answers are the stored record; web
answers are the publisher's own markup. The LLM only ever chooses among
candidates it was shown, so it cannot invent a quantity.

**The judge is kept even though it raises over-acceptance.** Measured on the
ambiguous band only: correct identity 36% -> 91%, wasted web calls 64% -> 9%,
over-acceptance 1.8% -> 12.3%. The trade is accepted because six of the seven
over-accepts are dishes the corpus genuinely holds under an English name, and
because the spec grades unnecessary web fallbacks. Every verdict is logged so
the rate can be re-read rather than trusted.

**The model is whatever the key can reach.** The spec mandates `gpt-4o-mini`;
the supplied Azure key reaches a `gpt-4.1-mini` deployment. The model name is
configuration, so nothing in the code changes, but cost figures are quoted for
the model actually used.

## Known disagreement with the ground truth

**Two rows expect a web answer where the dataset actually has the dish under an
English name.** Row 6 asks for Palak Paneer; the judge answers with "Indian
Cheese and Red Peppers in Fragrant Spinach Sauce", which is palak paneer. And
row 5, "authentic Japanese miso soup from scratch": we
answer from the dataset, which contains a Miso Soup that makes its stock from
kombu and bonito flakes — literally from scratch. The spec says to avoid
unnecessary web fallbacks, and a rule like "the word *authentic* means go to the
web" would be fitted to a single row. Documented rather than chased.

## Working practices

- Build artifacts live in `.artifacts/`, the environment in `.venv/`, both
  inside the project folder. *Chosen by the user.* Pause OneDrive syncing during
  long builds; never exclude the folders through "Choose folders", which deletes
  the local copy.
- Secrets live in a gitignored `.env`. `.env.example` is the committed template
  and must never contain a real key.
- Every measurement is reproducible from a script in `scripts/`, and the numbers
  live in `docs/evaluation.md` rather than in a chat transcript.
