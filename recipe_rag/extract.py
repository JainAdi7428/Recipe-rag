"""Turn a fetched web page into the same shape as a dataset recipe.

This is the module that makes web answers trustworthy. It reads schema.org
Recipe markup -- the JSON-LD block publishers embed for Google's recipe cards --
and refuses anything that does not have one. A page without markup is skipped,
not guessed at.

That refusal is the point. The alternative is to hand a search snippet or a
scraped blob of HTML to an LLM and ask it to write out the recipe, which is
exactly the hallucination the spec forbids: the model would happily supply
plausible quantities for a dish whose page never listed any. Here, every
ingredient and every step is a string the publisher wrote.

Once extracted, the recipe goes through **the identical diet classifier and
nutrition validator the dataset went through**. That is the spec's cross-source
consistency requirement, and it is not a nice-to-have: without it a web recipe
could be labelled vegan by the publisher's own keywords while listing fish
sauce, and the two halves of the system would disagree about the same food.

schema.org defines NutritionInformation as per-serving, which is the basis the
dataset uses, so the two are directly comparable with no rescaling.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

from .diet import classify_diet
from .nutrition import validate as validate_nutrition

_LD_BLOCK = re.compile(
    r"<script[^>]*type\s*=\s*[\"']application/ld\+json[\"'][^>]*>(.*?)</script>",
    re.IGNORECASE | re.DOTALL,
)
_TAGS = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")
_INLINE_SPACE = re.compile(r"[^\S\n]+")   # spaces and tabs, but not newlines
_NUMBER = re.compile(r"(\d+(?:[.,]\d+)?)")


@dataclass
class WebRecipe:
    """A recipe extracted from a page, in the dataset's own vocabulary."""

    title: str
    url: str
    ingredients: list[str] = field(default_factory=list)
    directions: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    diet: str = ""
    diet_confidence: float = 0.0
    diet_reason: str = ""
    diet_conflict: bool = False
    calories: float | None = None
    protein: float | None = None
    fat: float | None = None
    sodium: float | None = None
    nutrition_status: str = "missing"
    author: str = ""
    servings: str = ""

    @property
    def domain(self) -> str:
        return urlparse(self.url).netloc.lower().removeprefix("www.")

    @property
    def usable(self) -> bool:
        """Enough of a recipe to show someone.

        Two ingredients and one step is a low bar deliberately -- it exists to
        reject navigation stubs and category pages that carry Recipe markup
        without a recipe, not to judge quality.
        """
        return bool(self.title) and len(self.ingredients) >= 2 and len(self.directions) >= 1


def _clean(value) -> str:
    if not isinstance(value, str):
        value = "" if value is None else str(value)
    return _WHITESPACE.sub(" ", html.unescape(_TAGS.sub(" ", value))).strip()


def _split_block(value) -> list[str]:
    """Split a run-together instruction string into steps.

    Publishers who put the whole method in one string separate the steps either
    with newlines or with block-level markup. Both have to survive long enough
    to be split on, which means the usual `_clean` -- which flattens every run
    of whitespace to a single space -- cannot be applied first. It was, and
    every such page collapsed to one giant step.
    """
    text = html.unescape(value if isinstance(value, str) else str(value))
    text = re.sub(r"(?i)</(?:p|li|div|h[1-6])\s*>|<br\s*/?>", "\n", text)
    text = _TAGS.sub(" ", text)
    parts = [_INLINE_SPACE.sub(" ", part).strip() for part in text.split("\n")]
    return [part for part in parts if len(part) > 3]


def _as_list(value) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _types_of(node) -> set[str]:
    return {
        str(t).lower().removeprefix("http://schema.org/").removeprefix("https://schema.org/")
        for t in _as_list(node.get("@type"))
    }


def iter_json_ld(page_html: str):
    """Yield every JSON-LD object on the page, flattening @graph and arrays.

    Publishers nest these inconsistently: some emit a bare Recipe, some an
    array, most a @graph containing WebSite, Organization, BreadcrumbList and
    the Recipe together. Walking everything is simpler and more robust than
    guessing the layout per publisher.
    """
    for match in _LD_BLOCK.finditer(page_html):
        raw = match.group(1).strip()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            # Some CMSs emit trailing commas or wrap in CDATA. One cheap repair
            # attempt, then give up -- this is not worth a JSON5 dependency.
            try:
                payload = json.loads(re.sub(r",\s*([}\]])", r"\1", raw))
            except json.JSONDecodeError:
                continue
        stack = _as_list(payload)
        while stack:
            node = stack.pop()
            if isinstance(node, list):
                stack.extend(node)
            elif isinstance(node, dict):
                stack.extend(_as_list(node.get("@graph")))
                yield node


def find_recipe_node(page_html: str) -> dict | None:
    for node in iter_json_ld(page_html):
        if "recipe" in _types_of(node):
            return node
    return None


def _instructions(value) -> list[str]:
    """Flatten recipeInstructions, which has four shapes in the wild.

    A plain string, a list of strings, a list of HowToStep objects, or a list of
    HowToSection objects each wrapping their own steps. All four appear on
    mainstream sites.
    """
    steps: list[str] = []
    for item in _as_list(value):
        if isinstance(item, str):
            # A single string sometimes holds the whole method, with the steps
            # separated by newlines or by block markup rather than by the list.
            steps.extend(_split_block(item))
        elif isinstance(item, dict):
            types = _types_of(item)
            if "howtosection" in types or item.get("itemListElement"):
                steps.extend(_instructions(item.get("itemListElement")))
            else:
                text = _clean(item.get("text") or item.get("name") or "")
                if len(text) > 3:
                    steps.append(text)
    return steps


def _quantity(value, to_mg: bool = False) -> float | None:
    """Parse "553 kcal", "48 g", "1.2 grams" into a number.

    `to_mg` converts to milligrams, because the dataset stores sodium in mg
    while publishers use either mg or g. Getting this wrong is a 1000x error in
    the one field most likely to be extreme already, so the unit is read rather
    than assumed.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        unit = ""
    else:
        text = str(value).strip().lower()
        match = _NUMBER.search(text)
        if not match:
            return None
        number = float(match.group(1).replace(",", "."))
        unit = text[match.end():].strip()
    if to_mg:
        if unit.startswith("g") or unit.startswith("gram"):
            number *= 1000
        elif unit.startswith("mcg") or unit.startswith("µg"):
            number /= 1000
    return number


def _categories(node: dict) -> list[str]:
    out: list[str] = []
    for key in ("recipeCategory", "recipeCuisine", "keywords", "suitableForDiet"):
        for item in _as_list(node.get(key)):
            text = _clean(item).removeprefix("https://schema.org/").removeprefix("Diet")
            # keywords is frequently one comma-joined string.
            out.extend(part.strip() for part in text.split(",") if part.strip())
    seen: set[str] = set()
    unique = []
    for item in out:
        if item.lower() not in seen:
            seen.add(item.lower())
            unique.append(item)
    return unique[:24]


def _first(value) -> str:
    """First entry of a field publishers emit as either a scalar or a list."""
    items = _as_list(value)
    return _clean(items[0])[:60] if items else ""


def _author(node: dict) -> str:
    for item in _as_list(node.get("author")):
        if isinstance(item, dict):
            name = _clean(item.get("name", ""))
        else:
            name = _clean(item)
        if name:
            return name[:80]
    return ""


def from_html(page_html: str, url: str) -> WebRecipe | None:
    """Extract a recipe from a page, or None if it has no usable markup."""
    node = find_recipe_node(page_html)
    if node is None:
        return None

    title = _clean(node.get("name", ""))
    ingredients = [
        _clean(item)
        for item in _as_list(node.get("recipeIngredient") or node.get("ingredients"))
        if _clean(item)
    ]
    directions = _instructions(node.get("recipeInstructions"))
    categories = _categories(node)

    nutrition_node = node.get("nutrition")
    nutrition_node = nutrition_node if isinstance(nutrition_node, dict) else {}
    checked = validate_nutrition(
        {
            "calories": _quantity(nutrition_node.get("calories")),
            "protein": _quantity(nutrition_node.get("proteinContent")),
            "fat": _quantity(nutrition_node.get("fatContent")),
            "sodium": _quantity(nutrition_node.get("sodiumContent"), to_mg=True),
        }
    )

    # The same classifier the 18,249 dataset recipes went through. The page's
    # own "vegan" keyword is passed in as a category prior, exactly as the
    # dataset's categories were -- it informs the verdict but cannot override
    # an ingredient list that says otherwise.
    diet = classify_diet(ingredients, categories, title)

    recipe = WebRecipe(
        title=title,
        url=url,
        ingredients=ingredients,
        directions=directions,
        categories=categories,
        diet=diet.diet,
        diet_confidence=diet.confidence,
        diet_reason=diet.reason,
        diet_conflict=diet.conflict,
        calories=checked.calories,
        protein=checked.protein,
        fat=checked.fat,
        sodium=checked.sodium,
        nutrition_status=checked.status,
        author=_author(node),
        servings=_first(node.get("recipeYield")),
    )
    return recipe if recipe.usable else None
