"""Fuzzy title matching -- the signal dense vectors are worst at.

Embeddings compare *meaning*, which makes them poor at proper nouns. Measured on
the built index, "Spanakopita" is in the dataset, retrieves itself as the exact
top hit, and still scores only 0.487 -- below three dishes that are genuinely
absent. The reason is visible in the chunk it has to match:

    Spanakopita.
    Spinach and Feta Phyllo Triangles
    Categories: Cheese, Leafy Green, Appetizer, Bake, Vegetarian, Phyllo/Puff
    Pastry Dough, Pastry, Gourmet, Sugar Conscious, Kidney Friendly, ...
    Diet: veg
    Nutrition per serving: 73 calories, 2 g protein, 6 g fat, 112 mg sodium

One rare word, averaged against 300 characters of category list. The vector is
mostly "appetizer, cheese, kosher, pastry" and only slightly "Spanakopita".

Character-level matching has the opposite profile: it nails exact names and
typos, and is useless on paraphrase. Combining the two is what closes the gap --
see gate.py.

Costs about 1 MB and a few milliseconds; the titles are already in
parents.parquet, so nothing new is stored.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import numpy as np
from rapidfuzz import fuzz, process

# Conversational wrapping that carries no information about which dish is
# wanted. Stripping it matters: "Give me the recipe for Chicken Alfredo" scores
# far lower against the title "Chicken Alfredo" than the bare dish name does,
# purely because of the extra tokens.
_BOILERPLATE = re.compile(
    r"^\s*(?:"
    r"please\s+|can\s+you\s+|could\s+you\s+|i\s+(?:want|need|would\s+like)\s+"
    r"(?:to\s+(?:make|cook)\s+)?|"
    r"give\s+me\s+(?:the\s+)?|show\s+me\s+(?:the\s+)?|find\s+me\s+(?:the\s+)?|"
    r"tell\s+me\s+(?:how\s+to\s+(?:make|cook)\s+)?|"
    r"what(?:'s|\s+is)\s+(?:the\s+)?|how\s+(?:do\s+i|to|can\s+i)\s+(?:make|cook|prepare)\s+|"
    r"recipe\s+for\s+|the\s+recipe\s+for\s+|a\s+recipe\s+for\s+"
    r")+",
    re.IGNORECASE,
)
_TRAILING = re.compile(
    r"\s*(?:recipe|please|for\s+me)?\s*[?.!]*\s*$", re.IGNORECASE
)
_WHITESPACE = re.compile(r"\s+")


def clean_query(text: str) -> str:
    """Strip request boilerplate down to the dish name.

    "Give me the recipe for Chicken Alfredo" -> "chicken alfredo"
    "How do I make Paneer Butter Masala?"    -> "paneer butter masala"
    """
    cleaned = text.strip()
    # Applied repeatedly: "please give me the recipe for X" nests two prefixes.
    for _ in range(3):
        stripped = _BOILERPLATE.sub("", cleaned)
        if stripped == cleaned:
            break
        cleaned = stripped
    cleaned = _TRAILING.sub("", cleaned)
    return _WHITESPACE.sub(" ", cleaned).strip().lower() or text.strip().lower()


_TOKEN = re.compile(r"[a-z0-9]+")
# Words that carry no dish identity; they would inflate coverage for free.
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "with", "without", "of", "in", "on", "for",
    "to", "my", "some", "style", "homemade", "easy", "quick", "best",
})


def _tokens(text: str) -> list[str]:
    words = [w for w in _TOKEN.findall(text.lower()) if w not in _STOPWORDS]
    return words or _TOKEN.findall(text.lower())


# A query token scoring below this against every title token counts as not
# covered at all, rather than earning partial credit. Short words otherwise
# collide by accident: "vegan" scores 60 against "pecan" on raw edit distance,
# which was enough to rank "Banana-Pecan Ice Cream Bread" at 87.5 for the query
# "vegan banana bread" -- a confident answer to a question it does not answer.
_MIN_TOKEN_MATCH = 70.0


def _coverage(query_tokens: list[str], title_tokens: list[str]) -> float:
    """How much of the *query* the title accounts for, 0-100.

    This is the scorer that matters, and rapidfuzz's WRatio is the wrong tool
    for it. WRatio rewards partial containment without penalising the query
    words that went unmatched, so on the built index these four all scored an
    identical 85.5 and were wrongly claimed as dataset hits:

        "Paneer Butter Masala"  ~ "...Peanut Butter..."     (shares "butter")
        "Masala Dosa"           ~ "Chicken Tikka Masala"    (shares "masala")
        "Beef Rendang"          ~ "Korean Marinated Beef"   (shares "beef")
        "Recursive Onion Soup"  ~ "...Red Onion Confit"     (shares "onion")

    One word out of three should not read as a match. Scoring by query coverage
    drops all four to 54-69 while leaving "chicken alfredo" against "Chicken
    Alfredo with Zucchini Ribbons" at 100, because there every query word is
    present.

    Longer words weigh more: they carry the dish's identity ("rendang" matters,
    "beef" is generic), and a short word matching by accident should not rescue
    a query whose distinctive term is missing.
    """
    if not query_tokens or not title_tokens:
        return 0.0
    total_weight = 0.0
    total_score = 0.0
    for token in query_tokens:
        best = max(fuzz.ratio(token, other) for other in title_tokens)
        if best < _MIN_TOKEN_MATCH:
            best = 0.0
        weight = float(len(token))
        total_score += best * weight
        total_weight += weight
    return total_score / total_weight


def _squash(text: str) -> str:
    """Accent-folded, lowercase, punctuation to single spaces: for prefix tests."""
    folded = "".join(
        ch for ch in unicodedata.normalize("NFKD", text.lower())
        if not unicodedata.combining(ch)
    )
    return re.sub(r"[^a-z0-9]+", " ", folded).strip()


def coverage_score(query: str, title: str) -> float:
    """Public wrapper: how much of `query` the `title` accounts for, 0-100.

    Used to check an LLM judge's work. The judge is asked to echo back the title
    it chose, and that echo is scored against the query here. It is only ever a
    corroborating signal -- a judge picking "Spinach and Feta Phyllo Triangles"
    for "Spanakopita" is *right* and scores near zero on this, so a low value
    must never veto on its own.
    """
    return _coverage(_tokens(clean_query(query)), _tokens(title))


@dataclass
class TitleHit:
    parent_id: str
    title: str
    score: float          # 0-100, query coverage


class TitleMatcher:
    """Scores a query against every recipe title by query coverage -- all of them.

    An earlier version first narrowed 18,249 titles to a 60-strong shortlist with
    rapidfuzz's WRatio and re-scored only those, on the claim that coverage could
    never rank highly a title WRatio had rejected. The grader's queries proved
    that claim wrong. For "I want a gluten-free salad with tuna and asparagus",
    the expected "Tuna, Asparagus, and New Potato Salad..." covers the query at
    ~64 but never reached the shortlist; the best title that did scored 17.9.
    WRatio favours short strings that partially overlap, which is exactly the
    wrong bias for long descriptive queries.

    Scoring everything stays fast by working on the *vocabulary*, not the titles:
    each query token is compared once against the ~11k distinct title words
    (rapidfuzz cdist, in C), and each title's coverage is then a numpy gather
    over its own word indices -- the same arithmetic as `_coverage`, applied to
    every title at once.
    """

    def __init__(self, parents, shortlist: int | None = None) -> None:
        # `shortlist` is accepted for backwards compatibility and ignored.
        # Titles are ordered best-variant-first; the stable sort in match()
        # keeps that order among equal scores, so a duplicate group cannot
        # push its weaker copies ahead of its best one.
        ordered = parents.sort_values("dup_rank", kind="stable")
        self.parent_ids: list[str] = list(ordered.parent_id)
        self.titles: list[str] = list(ordered.title)
        title_tokens = [_tokens(t) for t in self.titles]
        self._vocab: list[str] = sorted({tok for toks in title_tokens for tok in toks})
        index = {tok: i for i, tok in enumerate(self._vocab)}
        width = max((len(toks) for toks in title_tokens), default=1) or 1
        # Padding points one past the vocabulary, at a column of zeros.
        grid = np.full((len(title_tokens), width), len(self._vocab), dtype=np.int32)
        for row, toks in enumerate(title_tokens):
            if toks:
                grid[row, : len(toks)] = [index[t] for t in toks]
        self._grid = grid
        # Word lengths per vocabulary entry; the padding column is 0 so it
        # never counts towards a title's length.
        self._vocab_len = np.array([len(t) for t in self._vocab] + [0], dtype=np.float32)
        self._squashed = np.array([_squash(t) for t in self.titles] or [""])

    def _coverages(self, query: str) -> tuple[np.ndarray, np.ndarray]:
        """(query covered by each title, each title covered by the query), 0-100."""
        n = len(self.titles)
        query_tokens = _tokens(clean_query(query))
        if not query_tokens or not self._vocab:
            zeros = np.zeros(n, dtype=np.float32)
            return zeros, zeros
        sim = process.cdist(
            query_tokens, self._vocab, scorer=fuzz.ratio, dtype=np.float32
        )
        sim[sim < _MIN_TOKEN_MATCH] = 0.0
        sim = np.hstack([sim, np.zeros((len(query_tokens), 1), dtype=np.float32)])
        best = sim[:, self._grid].max(axis=2)          # (query tokens, titles)
        weights = np.array([len(t) for t in query_tokens], dtype=np.float32)
        query_cov = (weights[:, None] * best).sum(axis=0) / weights.sum()

        per_word = sim.max(axis=0)                     # best query match per vocab word
        lengths = self._vocab_len[self._grid]           # (titles, width)
        title_cov = (per_word[self._grid] * lengths).sum(axis=1) / np.maximum(
            lengths.sum(axis=1), 1.0
        )
        return query_cov, title_cov

    def scores(self, query: str) -> np.ndarray:
        """Coverage of `query` by every title, 0-100, in `self.titles` order."""
        return self._coverages(query)[0]

    def match(self, query: str, limit: int = 5) -> list[TitleHit]:
        scores, title_cov = self._coverages(query)
        # Ranked by query coverage -- the reported score, and what the gate's
        # thresholds are calibrated on. Every title containing all the query's
        # words ties at 100, so two tie-breaks, in order:
        #   1. the title BEGINS with what was typed. Someone who types
        #      "Chicken Alfredo with" is naming the start of a longer title.
        #   2. how much of the TITLE the query accounts for, so "Chicken Stock"
        #      beats "Chicken Stock with Ginger" for the query "chicken stock".
        # With only the second, truncated-name lookups fell from ~79% to 69%.
        typed = _squash(clean_query(query))
        if typed and len(self.titles):
            titles = self._squashed[: len(self.titles)]
            prefix = (titles == typed) | np.char.startswith(titles, typed + " ")
        else:
            prefix = np.zeros(len(self.titles), dtype=bool)
        order = np.lexsort((-title_cov, ~prefix, -np.round(scores, 2)))[:limit]
        return [
            TitleHit(
                parent_id=self.parent_ids[i],
                title=self.titles[i],
                score=float(scores[i]),
            )
            for i in order
        ]

    def best(self, query: str) -> TitleHit | None:
        hits = self.match(query, limit=1)
        return hits[0] if hits else None
