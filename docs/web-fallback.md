# The web fallback

Stage 2 of the spec's decision flow: what happens when the internal collection
has been ruled out. Measured results live in
[web-evaluation.md](web-evaluation.md); this file is the design.

## The path

```
  Serper search        one API call; Google's results as JSON
       |               query rewritten: "Pav Bhaji" -> "Pav Bhaji recipe ingredients instructions"
       v
  filter + rank        drop YouTube, Pinterest, Instagram, Reddit and the like:
       |               they rank well and carry nothing usable
       |               known-good recipe publishers are tried first
       v
  fetch (max 4)        15 s per page, 30 s for the leg, 2 MB cap, real User-Agent
       |
       v
  require markup       the page must publish schema.org Recipe JSON-LD
       |
       v
  require the dish     the page must account for >= 75% of the request
       |
       v
  our own checks       the same diet classifier and nutrition validator the
       |               dataset went through
       v
  answer, attributed   "Web Search (site.com)" plus a link to the page
```

Anything that fails a step moves to the next search result. If all four pages
fail, the request ends in the apology and is logged with the stage it reached.

## Why each rule exists

**Serper, not scraping Google.** Google has no general free search API;
scraping it directly breaks within minutes and violates its terms. Serper
returns clean JSON for a flat price, with a free tier of about 2,500 queries.

**A search result is not a recipe.** It is a title, a URL and a ~150-character
snippet. Answering from a snippet is how a system invents quantities, so nothing
is used on the strength of a search result alone.

**Structured markup or nothing.** Most recipe sites publish schema.org Recipe
data — the same block that drives Google's recipe cards — giving title,
ingredients, instructions, yield and often nutrition, already parsed. A page
without it is skipped rather than guessed at. This is a property of the page,
which makes it a better quality filter than any guess about the publisher.

**Preferred publishers, not an allowlist.** 29 domains known to publish clean
markup are tried first, which usually means one fetch instead of four. Half of
them are Indian, Korean, Japanese, Chinese or South-East Asian, because that is
exactly where this Epicurious-derived dataset is empty. A site not on the list
is still accepted if its markup is valid; an allowlist would have made most of
the world's recipes unreachable.

**The page must be the dish that was asked for.** Search always returns
something. Asked for "Tandoori Jilaji" — a dish that does not exist — the top
four results are Tandoori Chicken pages with perfect markup. Each is refused at
42% coverage, and the request correctly ends in an apology.

The coverage rule is the dataset re-ranker's, with one deliberate inversion: a
word that appears in no recipe **counts against** a web page, where in dataset
ranking it is ignored. A word appearing nowhere on the page is the whole reason
to refuse it.

The bar is 75%, chosen from measurement rather than taste. Because a rejected
page sends the pipeline to the next search result instead of ending the request,
raising the bar mostly changes *which* page answers: every dish answered from the
web scored 100%, while the one wrong answer -- "Deep Fried Moon Pies" for "Deep
Fried Moon Rocks" -- scored 68%, having matched three of four words.

**The same diet and nutrition code.** A web recipe is classified by our own
reading of its ingredients, not by the page's claims. A publisher's "vegan" tag
over a fish-sauce ingredient list does not survive. This is what the spec's
"consistency across dataset and web-sourced recipes" means in practice: not a
policy, but literally the same function.

## Failure stages

Every miss records how far it got, because the fixes are completely different:

| stage | meaning | what it would take to fix |
|---|---|---|
| `search_failed` | Serper unreachable or out of quota | credentials, quota, retry |
| `timeout` | the whole web leg ran out of its 30 s budget | faster publishers, caching |
| `no_results` | nothing came back | better query rewriting |
| `fetch_failed` | every page blocked us or timed out | different publishers, or a real browser |
| `no_markup` | pages carried no Recipe data | broaden extraction, carefully |
| `wrong_dish` | the pages were a different dish | nothing — this is the system working |
| `diet_mismatch` | pages did not meet the stated diet | nothing — also working |

`wrong_dish` and `diet_mismatch` are successes wearing the word "failure": they
are the system declining to fabricate.

## Not built

- **Caching.** A repeated question re-runs the search and re-fetches the page.
  Worth adding before any public deployment, both for latency and for quota.
- **`robots.txt` checks.** We set a real User-Agent, cap page size, and wait
  between requests, but do not read `robots.txt`. Worth adding.
- **The judge on web candidates.** The dataset path gets an LLM judge for
  borderline cases; the web path uses coverage only. The one measured failure
  (below) is a case a judge would have caught.
