# Open questions and remaining work

What is genuinely unfinished, why, and what it would take. Kept honest on
purpose: this is the list to work from, not a wish list.

## Everything that was blocked is now done

The supplied key is an Azure OpenAI key; with the endpoint, deployment and API
version in `.env` it works, and the four pieces that were waiting on it have all
run: the 145-query held-out tuning set, the weight sweep on it, the judge
evaluation, and the cost estimate. See [evaluation.md](evaluation.md).

**One deviation from the spec.** The reachable deployment is `gpt-4.1-mini-4`,
not the `gpt-4o-mini` the spec mandates. The code does not care -- the model name
is configuration -- but the submission should say so, and the cost figures are
for the model actually used (gpt-4o-mini would be about 2.7x cheaper).

## Known weaknesses

**The two evaluation sets disagree about the re-ranking weight.** 145 held-out
queries prefer 0.6-0.8; the grader's six rows prefer 0.30 and drop sharply above
it. 0.30 is kept. The held-out set's queries were written by a model that had
seen each recipe's title, tags and ingredients, which biases it towards exactly
the kind of word matching the weight controls -- so a query set built from
something else (a photograph, a cuisine description, a human) would settle it.

**The judge trades recall for over-acceptance.** It lifts correct identity in
the ambiguous band from 36% to 91% and cuts wasted web calls from 64% to 9%, but
over-acceptance rises from 1.8% to 12.3%. Six of its seven over-accepts are
dishes this corpus really does hold under an English name; one ("Butter Chicken
Masala" answered with Chicken Tikka Masala) is a genuine error. Worth a stricter
prompt, or a second opinion, if that number has to come down.

**Two grader rows return a different valid recipe.** For rows 1 and 9 several
recipes satisfy every part of the request and vector similarity breaks the tie
in favour of one the grader did not pick. Tuning this against six rows would be
fitting to the test.

**The web page bar is fitted to 47 dishes.** Raising it from 50% to 75% removed
the one wrong answer ("Deep Fried Moon Pies" for "Deep Fried Moon Rocks") without
costing a real one, but 47 dishes is a small sample to place a threshold on. An
LLM judge over web candidates would settle these cases on meaning instead of word
overlap; it is the same fix the dataset path already has.

**Partial-name lookups sit at ~87%.** Typing the first few words of a long
recipe title is inherently ambiguous; several titles genuinely start the same way.

**No caching anywhere.** A repeated question re-runs the search and re-fetches
the page. Needed before any public deployment.

**`robots.txt` is not read.** We set a real User-Agent, cap page size and pause
between requests, but do not check the file.

## Decisions still to make

**Miso Soup (ground-truth row 5).** The grader expects a web answer; we answer
from the dataset, which does contain a from-scratch kombu-and-bonito miso soup.
Keep our answer and document the disagreement, or match the grader? Currently
ours, documented in [decisions.md](decisions.md).

**Serving several options for descriptive requests.** "Something creamy with
pasta" has many right answers, and we return exactly one. Showing three would
suit the product but complicates grading against a single expected answer.

## Deliverables not yet produced

- **Index upload to Hugging Face** (`scripts/push_index.py` is written, never
  run — it needs a token) and the matching download step in the app.
- **Deployment to Streamlit Cloud** itself; see
  [streamlit-deployment.md](streamlit-deployment.md).
- **Architecture diagram** as an image, for the submission. The flow in
  [architecture.md](architecture.md) is the content for it.
- **Answers to the spec's six design questions**, which the docs here cover in
  substance but not in that format.
