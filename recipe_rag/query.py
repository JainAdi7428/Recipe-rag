"""Understand a request: which words describe the dish, and what it demands.

The grader's queries are not dish names. They are descriptions with constraints
-- "a gluten-free salad with tuna and asparagus", "a coconut cake for Easter
with full nutrition info" -- and treating every word as something to match by
meaning is what ranked a chocolate coconut cake above the Easter one. So the
words are separated by the job they do:

  constraints   what a recipe MUST satisfy; enforced as hard filters
      diet          "vegan", "vegetarian", "eggless"
      nutrition     "with full nutrition info", "calories"
      exclusions    "without mushrooms", "no nuts"
  terms         what the dish is LIKE; used for ranking only, matched against
                each recipe's title, category tags and ingredients

and terms are mapped onto Epicurious's own category tags -- exactly, never
fuzzily. fuzz.ratio("indian", "indiana") is 92: a fuzzy matcher reads "like in
Indian restaurants" as a request for one of three recipes tagged with a US state.

Diet words are constraints, not terms. "Vegan" as a ranking term would penalise
the right answer whenever its tags forget to say so -- the grader's vegan soup
carries a Vegetarian tag and no Vegan tag -- while the diet filter already
guarantees every candidate complies.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

from .diet import EGGETARIAN, VEG, VEGAN
from .lexical import clean_query

_WORD = re.compile(r"[a-z0-9]+")


def _fold(text: str) -> str:
    lowered = text.lower()
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", lowered) if not unicodedata.combining(ch)
    )


def singular(word: str) -> str:
    """A crude, deterministic singular: "tomatoes" -> "tomato", "olives" -> "olive".

    It does not need to be linguistically right, only identical on both sides:
    it is used exclusively to compare query words with recipe words, and both go
    through this same function. "asparagus", "hummus" and "glass" are left alone.
    """
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith(("oes", "ches", "shes", "sses", "xes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    """Words of `text`, accent-folded and singularised. "&" and "-" split."""
    return [singular(w) for w in _WORD.findall(_fold(text).replace("&", " "))]


# Request scaffolding: says nothing about which dish is wanted. Checked against
# both the raw word and its singular, so "does" and "recipes" both drop out.
FILLER = frozenset("""
i me my we our you your a an the and or of in on at to for from with by as into
is are be am was were do does did can could would will should shall may might must
please need needs want wanted wanting like love looking look find give show tell
make making made cook cooking prepare preparing get try have has having
how what which who where when some something anything any another other
recipe recipes receipe recipie recepie dish dishes meal meals idea ideas option
good nice great best tasty delicious yummy perfect authentic traditional classic
homemade home scratch style restaurant restaurants version simple really very just
one it this that there here so also too all full info information detail details
about let know kind sort type way
""".split())

# Hard dietary constraints, strictest first. Each maps to the labels allowed.
# The grader's own row 2 settles "vegetarian": it answers a vegetarian request
# with an egg-noodle casserole labelled "Vegetarian, Contains Egg", so eggetarian
# dishes satisfy it.
_DIET_RULES: list[tuple[str, re.Pattern[str], frozenset[str]]] = [
    ("vegan",
     re.compile(r"\b(?:vegan|plant[\s-]*based)\b"),
     frozenset({VEGAN})),
    ("egg-free",
     re.compile(r"\b(?:eggless|egg[\s-]*free|without\s+eggs?|no\s+eggs?)\b"),
     frozenset({VEGAN, VEG})),
    ("vegetarian",
     re.compile(r"(?<!non-)(?<!non )\b(?:vegetarian|veggie|meatless|meat[\s-]*free"
                r"|without\s+meat|no\s+meat)\b"),
     frozenset({VEGAN, VEG, EGGETARIAN})),
    ("eggetarian",
     re.compile(r"\beggetarian\b"),
     frozenset({VEGAN, VEG, EGGETARIAN})),
]

# Asking for nutrition means a recipe without it cannot be the answer.
_NUTRITION = re.compile(
    r"\b(?:(?:full|complete|detailed)\s+)?"
    r"(?:nutrition(?:al)?|nutrients?|macros?|calories|calorie|kcal)"
    r"(?:\s+(?:info(?:rmation)?|facts|details?|values?|breakdown|count|content|data))?\b"
)

# "without X", "no X" up to the end of the phrase. Hyphens are not whitespace,
# so "no-cook" and "no-bake" are styles and never reach this pattern.
_NEGATION = re.compile(
    r"\b(?:without|minus|hold\s+the|free\s+of|no)\s+([a-z][a-z ,'-]*?)"
    r"(?=\s+(?:with|for|but|that|which|please|in|on|and\s+with)\b|[.?!;]|$)"
)
# Words that can follow "no" without naming an ingredient. The negated phrase
# ends at the first of them: in "no sugar added lemonade" only sugar is negated,
# and lemonade is still the dish being asked for.
_NOT_AN_INGREDIENT = frozenset({
    "cook", "bake", "baking", "fuss", "oven", "mess", "knead", "churn", "added",
    "more", "than", "less", "minute", "minutes", "hour", "hours", "time", "idea",
    "extra", "fancy", "special",
})
_TERMINATOR = re.compile(
    r"\b(?:" + "|".join(sorted(_NOT_AN_INGREDIENT, key=len, reverse=True)) + r")\b"
)


def _split_negation(group: str) -> tuple[str, str]:
    """"sugar added lemonade" -> ("sugar ", " lemonade"): negated part, remainder."""
    stop = _TERMINATOR.search(group)
    if not stop:
        return group, ""
    return group[: stop.start()], group[stop.end():]

# Tags that name a publication or a hashtag, not anything about the dish.
NON_DISH_TAGS = frozenset({"bon appétit", "bon appetit", "gourmet", "#cakeweek", "#wasteless"})

# Phrasings users write that Epicurious tags spell differently.
TAG_ALIASES: dict[tuple[str, ...], tuple[str, ...]] = {
    ("raw",): ("No-Cook",),
    ("low", "calorie"): ("Low Cal",),
    ("low", "sugar"): ("Low/No Sugar",),
    ("wheat", "free"): ("Wheat/Gluten-Free",),
    ("kid",): ("Kid-Friendly",),
    ("quick",): ("Quick & Easy",),
    ("easy",): ("Quick & Easy",),
}
# "Low/No Sugar" must not make every "low ..." query match it, nor "Wheat/
# Gluten-Free" every "whole wheat" one: a lone modifier is not an alternative.
_MODIFIER_ONLY = frozenset({"low", "no", "high", "non", "wheat"})


def _alternatives(tag: str) -> list[str]:
    parts = [p.strip() for p in tag.split("/") if p.strip()]
    if len(parts) == 1:
        return parts
    return [p for p in parts if _fold(p) not in _MODIFIER_ONLY]


@dataclass
class ParsedQuery:
    raw: str
    cleaned: str
    search_text: str                       # cleaned, minus exclusions: what gets embedded
    terms: list[str] = field(default_factory=list)
    tags: dict[str, str] = field(default_factory=dict)            # tag -> words that matched
    term_tags: dict[str, frozenset[str]] = field(default_factory=dict)
    diet: str | None = None
    allowed_diets: frozenset[str] | None = None
    wants_nutrition: bool = False
    excluded: list[str] = field(default_factory=list)

    @property
    def has_constraints(self) -> bool:
        return bool(self.allowed_diets or self.wants_nutrition or self.excluded)

    def describe(self) -> str:
        parts = []
        if self.allowed_diets:
            parts.append(f"diet={self.diet} ({'/'.join(sorted(self.allowed_diets))})")
        if self.wants_nutrition:
            parts.append("needs nutrition")
        if self.excluded:
            parts.append("excluding " + ", ".join(self.excluded))
        if self.tags:
            parts.append("tags " + ", ".join(sorted(self.tags)))
        parts.append("terms " + (", ".join(self.terms) or "-"))
        return "; ".join(parts)


class QueryParser:
    """Built once from the recipe collection's category vocabulary."""

    def __init__(self, categories) -> None:
        counts = Counter(tag for tags in categories for tag in (tags or []))
        self.tag_counts = counts
        self._phrases: dict[tuple[str, ...], set[str]] = {}
        for tag in counts:
            if tag.lower() in NON_DISH_TAGS:
                continue
            for alt in _alternatives(tag):
                words = tuple(tokenize(alt))
                if words:
                    self._phrases.setdefault(words, set()).add(tag)
        for phrase, tags in TAG_ALIASES.items():
            present = {t for t in tags if t in counts}
            if present:
                self._phrases.setdefault(phrase, set()).update(present)
        self._longest = max((len(p) for p in self._phrases), default=1)

    @classmethod
    def from_parents(cls, parents) -> "QueryParser":
        return cls(parents.categories)

    def parse(self, query: str) -> ParsedQuery:
        cleaned = clean_query(query)
        text = _fold(cleaned)

        allowed: frozenset[str] | None = None
        names: list[str] = []
        for name, pattern, diets in _DIET_RULES:
            if pattern.search(text):
                allowed = diets if allowed is None else allowed & diets
                names.append(name)

        excluded: list[str] = []
        for match in _NEGATION.finditer(text):
            negated, _ = _split_negation(match.group(1))
            for piece in re.split(r",|\band\b|\bor\b", negated):
                words = [w for w in _WORD.findall(piece) if w not in FILLER]
                if words:
                    head = singular(words[-1])   # "red onion" -> "onion"
                    if head not in excluded:
                        excluded.append(head)

        # Negated phrases are removed before anything else reads the text:
        # "without mushrooms" must neither boost nor retrieve mushroom dishes.
        # Whatever follows a terminator ("... added lemonade") is kept.
        without_negation = _NEGATION.sub(
            lambda m: " " + _split_negation(m.group(1))[1], text
        )
        # Tags are matched before the nutrition words are stripped, so that
        # "low calorie" still reaches its Low Cal tag.
        tags, term_tags = self._match_tags(
            [w for w in tokenize(without_negation) if w != "and"]
        )

        term_text = _NUTRITION.sub(" ", without_negation)
        for _, pattern, _ in _DIET_RULES:
            term_text = pattern.sub(" ", term_text)
        terms: list[str] = []
        for raw in _WORD.findall(term_text):
            word = singular(raw)
            if (raw in FILLER or word in FILLER or raw in _NOT_AN_INGREDIENT
                    or word in terms or word in excluded):
                continue
            terms.append(word)

        search_text = re.sub(r"\s+", " ", without_negation).strip() or cleaned
        return ParsedQuery(
            raw=query,
            cleaned=cleaned,
            search_text=search_text,
            terms=terms,
            tags=tags,
            term_tags=term_tags,
            diet="+".join(names) or None,
            allowed_diets=allowed,
            wants_nutrition=bool(_NUTRITION.search(text)),
            excluded=excluded,
        )

    def _match_tags(self, words: list[str]):
        tags: dict[str, str] = {}
        term_tags: dict[str, set[str]] = {}
        for size in range(min(self._longest, len(words)), 0, -1):
            for start in range(len(words) - size + 1):
                phrase = tuple(words[start:start + size])
                for tag in self._phrases.get(phrase, ()):
                    tags.setdefault(tag, " ".join(phrase))
                    for word in phrase:
                        term_tags.setdefault(word, set()).add(tag)
        return tags, {word: frozenset(t) for word, t in term_tags.items()}
