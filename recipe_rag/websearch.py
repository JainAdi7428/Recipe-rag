"""Stage 2 of the spec's decision flow: look on the web.

Reached only when the dataset has been ruled out -- by the scores in the
hopeless band, or by the judge in the ambiguous one. Serper.dev is the search
provider the spec names; it returns Google results as JSON for a flat per-query
price with no scraping of Google itself.

**What this module will and will not accept.** A search result is a URL and a
snippet, and neither is a recipe. Answering from a snippet is how a system ends
up inventing quantities, so nothing here is passed to the user on the strength
of a search result alone: a page must yield real schema.org Recipe markup (see
extract.py) before its contents are usable. That structural requirement is a
much better quality filter than a domain list, because it is a property of the
page rather than a guess about the publisher.

The domain preference below is therefore a *ranking* signal, not a gate. Sites
known to publish clean markup are tried first, which usually means one fetch
instead of four; a site not on the list is still accepted if its markup is
valid. Hard-coding an allowlist would have quietly made most of the world's
Indian and South-East Asian recipes unreachable, which is precisely the gap
this dataset has.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from urllib.parse import urlparse

from . import config

SERPER_URL = "https://google.serper.dev/search"

# Publishers observed to ship complete schema.org Recipe JSON-LD. Deliberately
# spans the cuisines this Epicurious-derived dataset is weakest at, since those
# are the queries that actually reach the web path.
PREFERRED_DOMAINS = (
    "allrecipes.com", "seriouseats.com", "simplyrecipes.com", "food.com",
    "foodnetwork.com", "bonappetit.com", "epicurious.com", "delish.com",
    "tasteofhome.com", "bbcgoodfood.com", "kingarthurbaking.com",
    "budgetbytes.com", "thekitchn.com", "recipetineats.com",
    "minimalistbaker.com", "loveandlemons.com", "sallysbakingaddiction.com",
    "indianhealthyrecipes.com", "vegrecipesofindia.com", "hebbarskitchen.com",
    "cookwithmanali.com", "spiceupthecurry.com", "cubesnjuliennes.com",
    "maangchi.com", "justonecookbook.com", "chinasichuanfood.com",
    "thewoksoflife.com", "hungryhuy.com", "rasamalaysia.com",
)

# Aggregators, video pages and stores: they rank well and carry no usable
# markup, so fetching them is wasted latency.
SKIP_DOMAINS = (
    "youtube.com", "youtu.be", "pinterest.com", "facebook.com", "instagram.com",
    "tiktok.com", "reddit.com", "amazon.com", "twitter.com", "x.com",
)

USER_AGENT = (
    "Mozilla/5.0 (compatible; RecipeRAG/1.0; +educational coursework project)"
)
MAX_PAGE_BYTES = 2_000_000
FETCH_TIMEOUT = 12.0
# requests' `timeout` bounds the connection and the gap between bytes, not the
# whole download: one measured page trickled for 350 seconds inside a 12-second
# timeout. The body is therefore read in chunks against a wall-clock deadline.
MAX_FETCH_SECONDS = 15.0
POLITE_DELAY = 0.4


class WebUnavailable(RuntimeError):
    """Search could not run. Callers fall through to 'not found', never invent."""


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str = ""
    rank: int = 0

    @property
    def domain(self) -> str:
        return urlparse(self.url).netloc.lower().removeprefix("www.")

    @property
    def preferred(self) -> bool:
        return any(self.domain.endswith(d) for d in PREFERRED_DOMAINS)


@dataclass
class SearchOutcome:
    results: list[SearchResult] = field(default_factory=list)
    query: str = ""
    seconds: float = 0.0
    calls: int = 0


def serper_key(required: bool = True) -> str | None:
    config.load_env()
    import os

    key = (os.environ.get("SERPER_API_KEY") or "").strip()
    if not key and required:
        raise WebUnavailable(
            f"No SERPER_API_KEY found.\n"
            f"  1. Get a free key at https://serper.dev (2,500 free queries)\n"
            f"  2. Add SERPER_API_KEY=... to {config.ENV_FILE}\n"
            f"  3. Re-run."
        )
    return key or None


def build_query(user_query: str) -> str:
    """Bias the search toward pages that carry a full recipe.

    Bare dish names return encyclopaedia entries, restaurant listings and
    shopping results. The two extra words push Google toward pages structured
    as recipes, which are the only ones the extractor can use anyway.
    """
    cleaned = user_query.strip()
    lowered = cleaned.lower()
    if "recipe" not in lowered:
        cleaned = f"{cleaned} recipe"
    return f"{cleaned} ingredients instructions"


def rank_results(organic) -> list[SearchResult]:
    """Filter and order Serper's organic results.

    Split out from `search` so it can be exercised without the network.
    """
    results: list[SearchResult] = []
    for position, item in enumerate(organic or [], 1):
        url = item.get("link") or ""
        if not url:
            continue
        domain = urlparse(url).netloc.lower().removeprefix("www.")
        if any(domain.endswith(skip) for skip in SKIP_DOMAINS):
            continue
        results.append(
            SearchResult(
                title=item.get("title", ""),
                url=url,
                snippet=item.get("snippet", ""),
                rank=position,
            )
        )
    # Stable sort: preferred domains first, original Google order preserved
    # within each group. Google's ranking is a genuine relevance signal and is
    # not worth discarding, only reordering.
    results.sort(key=lambda r: not r.preferred)
    return results


def search(user_query: str, key: str | None = None, n: int = 8) -> SearchOutcome:
    """Run one Serper query and return ranked, filtered results."""
    try:
        import requests
    except ImportError as exc:  # pragma: no cover
        raise WebUnavailable("the requests package is not installed") from exc

    key = key or serper_key()
    query = build_query(user_query)
    started = time.perf_counter()
    try:
        response = requests.post(
            SERPER_URL,
            headers={"X-API-KEY": key, "Content-Type": "application/json"},
            json={"q": query, "num": max(n, 10)},
            timeout=FETCH_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:  # noqa: BLE001 - network failures vary widely
        raise WebUnavailable(f"{type(exc).__name__}: {exc}") from exc

    results = rank_results(payload.get("organic", []))
    return SearchOutcome(
        results=results[:n],
        query=query,
        seconds=time.perf_counter() - started,
        calls=1,
    )


def page_encoding(headers, content: bytes) -> str:
    """Work out a page's character set from the header, then the document.

    requests defaults text/html with no declared charset to ISO-8859-1, per an
    HTTP/1.1 rule the web stopped following long ago. Trusting it turned "Add
    1/2 teaspoon salt" on a real recipe page into a replacement character, so
    the header is used only when it actually names a charset, then the document
    meta tag, and UTF-8 otherwise.
    """
    content_type = (headers.get("Content-Type") or "").lower()
    if "charset=" in content_type:
        return content_type.split("charset=")[1].split(";")[0].strip() or "utf-8"
    head = content[:2048].decode("ascii", errors="ignore").lower()
    if "charset=" in head:
        declared = head.split("charset=")[1][:40]
        for stop in (">", ";", " ", "/"):
            declared = declared.split(stop)[0]
        declared = declared.strip().strip('"').strip("'")
        if declared:
            return declared
    return "utf-8"


def fetch(url: str, timeout: float = FETCH_TIMEOUT) -> str | None:
    """Download one page as text. Returns None rather than raising.

    A single unreachable page must not fail the whole request -- the caller is
    walking a ranked list and can simply try the next one.
    """
    try:
        import requests
    except ImportError:  # pragma: no cover
        return None
    try:
        response = requests.get(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,*/*"},
            timeout=timeout,
            stream=True,
        )
        response.raise_for_status()
        if "html" not in response.headers.get("Content-Type", "").lower():
            return None
        # Capped by size, so a pathological page cannot exhaust memory on the
        # Streamlit free tier, and by time, so a slow one cannot hang a request.
        deadline = time.perf_counter() + MAX_FETCH_SECONDS
        parts, total = [], 0
        for chunk in response.iter_content(65_536):
            parts.append(chunk)
            total += len(chunk)
            if total >= MAX_PAGE_BYTES or time.perf_counter() > deadline:
                break
        content = b"".join(parts)
        return content.decode(page_encoding(response.headers, content), errors="replace")
    except Exception:  # noqa: BLE001
        return None
    finally:
        time.sleep(POLITE_DELAY)
