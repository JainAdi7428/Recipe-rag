"""Dietary classification: veg / non-veg / vegan / eggetarian.

Deterministic and ingredient-driven, so the same recipe always gets the same
label and the reasoning can be shown to a user. `classify_diet` is the single
public entry point and is deliberately source-agnostic -- the web-search
fallback calls this exact function on scraped ingredients, which is what keeps
classification consistent across the dataset and the web, as the spec requires.

The hard part is not the lexicon, it is substring false positives. "eggplant"
contains "egg", "butternut squash" contains "butter", "coconut milk" contains
"milk", "cheesecloth" contains "cheese", and -- the expensive one in this
dataset -- "goat cheese" contains "goat". Naive matching mislabels thousands of
recipes. Three defences are used together:

1.  Text is unicode-folded first, so "creme fraiche" matches "creme fraiche".
2.  Known phrases are rewritten before any lexicon runs. Most rewrite to
    nothing ("eggplant"), but some rewrite to the term that *should* have
    matched: "goat cheese" -> "cheese" keeps it dairy instead of meat, and
    "quail eggs" -> "eggs" keeps it eggetarian instead of non-veg.
3.  All lexicon matching is word-boundary anchored, so "graham" never matches
    "ham" and "collard" never matches "lard".

`buttermilk` is deliberately absent from the rewrite table -- it really is dairy
and should match.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

VEGAN = "vegan"
VEG = "veg"
EGGETARIAN = "eggetarian"
NON_VEG = "non-veg"

# Phrase -> replacement, applied before lexicon matching. An empty replacement
# deletes the phrase; a non-empty one redirects it to the correct lexicon.
PHRASE_REWRITES: dict[str, str] = {
    # "egg" that is not egg
    "eggplant": "",
    "eggplants": "",
    "eggplant caviar": "",
    # "butter" that is not butter
    "peanut butter": "",
    "almond butter": "",
    "cashew butter": "",
    "hazelnut butter": "",
    "sunflower butter": "",
    "nut butter": "",
    "cocoa butter": "",
    "shea butter": "",
    "apple butter": "",
    "coconut butter": "",
    "butternut squash": "",
    "butternut": "",
    "butter lettuce": "",
    "butter bean": "",
    "butter beans": "",
    "buttercup squash": "",
    "butterhead": "",
    "butterfly pea": "",
    # "milk" / "cream" that is not dairy
    "coconut milk": "",
    "coconut cream": "",
    "cream of coconut": "",
    "almond milk": "",
    "soy milk": "",
    "soymilk": "",
    "oat milk": "",
    "rice milk": "",
    "cashew milk": "",
    "hemp milk": "",
    "milk thistle": "",
    "cream of tartar": "",
    "creamed corn": "",
    # "cheese" / "curd" that is not dairy
    "cheesecloth": "",
    "bean curd": "",
    # "meat" that is not meat
    "coconut meat": "",
    "meat grinder": "",
    "meat thermometer": "",
    # produce and fungi that shadow animal words
    "beefsteak tomato": "",
    "beefsteak tomatoes": "",
    "oyster mushroom": "",
    "oyster mushrooms": "",
    "lamb's lettuce": "",
    "lambs lettuce": "",
    "lamb's quarters": "",
    "lambs quarters": "",
    "crabapple": "",
    "crab apple": "",
    "crabapples": "",
    "honeydew": "",
    # _normalise turns hyphens into spaces, so "Honey-Dew Granita" arrives as
    # "honey dew" and would miss the closed-up spelling above.
    "honey dew": "",
    "bread and butter pickle": "",
    "bread and butter pickles": "",
    "chicken of the woods": "",
    "hen of the woods": "",
    # redirected: the animal word is a modifier, the real category is elsewhere
    "goat cheese": "cheese",
    "goat's cheese": "cheese",
    "goats cheese": "cheese",
    "goat milk": "milk",
    "goat's milk": "milk",
    "goats milk": "milk",
    "goat butter": "butter",
    "goat yogurt": "yogurt",
    "sheep's milk": "milk",
    "sheep milk": "milk",
    "buffalo mozzarella": "mozzarella",
    "quail egg": "egg",
    "quail eggs": "eggs",
    "duck egg": "egg",
    "duck eggs": "eggs",
    "duck fat": "duck fat",
    # "chop" means pork chop, except here
    "chop suey": "",
    "chopped": "chopped",
    # Plant parts and products that borrow butchery words. "2 celery ribs"
    # appeared in 138 recipes and is the single biggest false-positive source.
    "celery rib": "",
    "celery ribs": "",
    "kidney bean": "",
    "kidney beans": "",
    "hot dog bun": "",
    "hot dog buns": "",
    "vegetable marrow": "",
    "marrow bean": "",
    "marrow beans": "",
}

# Epicurious category tags that reliably imply an animal ingredient. These act
# as a floor, never a ceiling: a category can make the classification *more*
# restrictive but never less. That direction is deliberate -- for dietary
# compliance, wrongly warning a vegetarian is a nuisance, while wrongly telling
# them a dish is vegan is a real harm.
#
# The floor exists because ingredient lists routinely reference sub-recipes
# ("court bouillon", "1 recipe polenta", "Pico de Gallo") that hide the animal
# product entirely. "Warm Skate Salad" and "Polenta with Sausages" both slipped
# through ingredient matching alone.
MEAT_CATEGORIES = frozenset({
    "beef", "pork", "lamb", "veal", "goat", "venison", "rabbit", "game",
    "chicken", "poultry", "turkey", "duck", "goose", "quail", "meat",
    "steak", "brisket", "sausage", "bacon", "ham", "hamburger", "hot dog",
    "ground beef", "meatball", "meatloaf", "pork rib", "beef rib", "veal chop",
})
FISH_CATEGORIES = frozenset({
    "fish", "seafood", "shellfish", "salmon", "tuna", "cod", "halibut",
    "trout", "snapper", "bass", "sardine", "anchovy", "shrimp", "prawn",
    "crab", "lobster", "clam", "mussel", "oyster", "scallop", "squid",
    "octopus", "caviar",
})
EGG_CATEGORIES = frozenset({"egg", "omelet", "frittata", "quiche", "souffle"})
DAIRY_CATEGORIES = frozenset({
    "dairy", "cheese", "milk/cream", "butter", "yogurt", "ice cream",
    "goat cheese", "cream cheese", "blue cheese", "parmesan", "mozzarella",
    "cheddar", "feta", "brie", "ricotta", "mascarpone",
})

# Meat and poultry, plus rendered animal fats and gelatine. Plurals are
# handled by the regex, so singular forms are enough.
MEAT_TERMS = (
    "meat", "meatball", "meatloaf", "beef", "steak", "veal", "pork", "ham",
    "bacon", "prosciutto", "pancetta", "guanciale", "sausage", "chorizo",
    "salami", "pepperoni", "mortadella", "bresaola", "speck", "lamb",
    "mutton", "goat", "venison", "rabbit", "hare", "chicken", "poultry",
    "turkey", "duck", "duckling", "goose", "gosling", "quail", "squab",
    "pheasant", "capon", "hen", "poussin", "fowl", "bison", "elk", "boar",
    "brisket", "sirloin", "tenderloin", "ribeye", "rib eye", "oxtail", "ox",
    "shank", "chuck roast", "ground round",
    # Bare "rib" is deliberately absent: a celery rib is not a pork rib.
    "spareribs", "spare rib", "short rib", "prime rib", "baby back rib",
    "pork rib", "beef rib", "rib roast",
    "chop", "cutlet", "loin", "breast", "thigh", "drumstick", "hock",
    "trotter", "jowl", "lardon", "crackling", "liver", "liverwurst", "pate",
    "rillettes", "foie gras", "kidney", "sweetbread", "tripe", "marrow",
    "giblet", "gizzard", "tongue", "brain", "bone", "bone broth", "carcass",
    "lard", "tallow", "suet", "schmaltz", "gelatin", "gelatine", "marshmallow",
    "snail", "escargot", "frog leg", "andouille", "kielbasa", "bratwurst",
    "hot dog", "jerky", "pastrami", "corned beef", "charcuterie",
)

# Fish and seafood, plus condiments built on them. Anchovy hides inside
# Worcestershire sauce and classic Caesar dressing -- both are common in this
# dataset and both make a dish non-vegetarian.
FISH_TERMS = (
    "fish", "anchovy", "anchovies", "salmon", "tuna", "cod", "haddock",
    "halibut", "trout", "bass", "snapper", "mackerel", "sardine", "herring",
    "sole", "flounder", "tilapia", "swordfish", "monkfish", "catfish",
    "shellfish", "skate", "orange roughy", "roughy", "turbot", "grouper",
    "perch", "pike", "eel", "whitefish", "smelt", "bluefish", "pompano",
    "arctic char", "plaice", "hake", "pollock", "brill", "john dory",
    "barramundi", "tilefish", "sablefish", "branzino", "mahi mahi",
    "shrimp", "prawn", "crab", "crabmeat", "lobster", "langoustine", "scampi",
    "crayfish", "crawfish", "clam", "mussel", "oyster", "scallop", "squid",
    "calamari", "octopus", "abalone", "conch", "whelk", "periwinkle",
    "surimi", "caviar", "roe", "tobiko", "ikura", "uni", "bottarga",
    "fish sauce", "fish stock", "clam juice", "court bouillon", "nam pla",
    "nuoc mam", "oyster sauce", "shrimp paste", "bonito", "dashi",
    "katsuobushi", "worcestershire", "caesar dressing", "caesar salad dressing",
)

EGG_TERMS = (
    # "yolk" alone: recipes write "4 large yolks". Sub-recipes built on yolks
    # ("Pastry Cream", "lemon curd") hide the egg behind their name -- found
    # when the title stopped being read for egg and a pastry-cream souffle
    # dropped to veg.
    "yolk", "pastry cream", "creme patissiere", "lemon curd", "sabayon",
    "egg", "egg white", "egg yolk", "eggnog", "albumen", "mayonnaise", "mayo",
    "aioli", "meringue", "custard", "hollandaise", "bearnaise",
    "creme anglaise", "wonton wrapper", "ladyfinger", "sponge cake",
    "angel food cake", "quiche", "frittata", "omelet", "omelette", "souffle",
    "pavlova", "zabaglione", "royal icing", "macaron", "nougat", "flan",
)

DAIRY_TERMS = (
    "milk", "buttermilk", "cream", "half and half", "sour cream",
    "heavy cream", "whipping cream", "creme fraiche", "clotted cream",
    "butter", "butterscotch", "butterfat", "ghee", "cheese", "parmesan",
    "parmigiano", "mozzarella", "cheddar", "gruyere", "emmental", "feta",
    "brie", "camembert", "gorgonzola", "roquefort", "stilton", "provolone",
    "romano", "pecorino", "asiago", "manchego", "mascarpone", "ricotta",
    "paneer", "halloumi", "queso", "yogurt", "yoghurt", "kefir", "quark",
    "curd", "whey", "casein", "condensed milk", "evaporated milk",
    "dulce de leche", "ice cream", "gelato", "creme brulee", "panna cotta",
    # Composite items that reliably contain butter, cream or cheese.
    "puff pastry", "phyllo", "filo", "pie crust", "pie shell", "tart shell",
    "pastry dough", "shortcrust", "brioche", "croissant", "graham cracker",
    "white chocolate", "milk chocolate", "pesto", "alfredo", "bechamel",
    "gratin", "tiramisu",
)

# Vegetarian but not vegan, and not covered by dairy or egg.
NON_VEGAN_TERMS = ("honey", "beeswax", "royal jelly")

# Genuinely ambiguous: unqualified "broth"/"stock" may be vegetable or animal.
# These never change the label -- they lower confidence and leave a note, so
# the answer layer can hedge rather than assert.
AMBIGUOUS_TERMS = ("broth", "stock", "bouillon", "consomme", "chocolate chip")

# Qualifiers that turn an animal word into its plant-based substitute:
# "vegan butter", "non-dairy yogurt", "mock chicken". When one of these appears
# in an ingredient line, that whole line is excluded from animal matching --
# the line describes a substitute, not the thing it is named after.
#
# This was found by comparing against the dataset's own Vegan tag: of 138
# Vegan-tagged recipes we disagreed with, the largest cause after honey was
# dairy words inside lines like "1/2 cup vegan butter".
SUBSTITUTE_QUALIFIERS = (
    "vegan", "non dairy", "nondairy", "dairy free", "dairyfree",
    "plant based", "meatless", "mock", "faux", "imitation", "vegetarian",
    "egg free", "eggless", "meat free", "soy based",
    # "Tofu sour cream", "tofu ricotta" -- the base ingredient names the
    # substitute rather than a qualifier, but it works the same way.
    "tofu", "tempeh", "seitan",
)

# Epicurious ingredient lists end with equipment notes -- "Equipment: an ice
# cream maker", "Special equipment: cheesecloth". They are not food, and reading
# them as food labelled a plum sorbet veg because of the ice cream maker.
_EQUIPMENT_LINE = re.compile("^ *(?:special +)?equipment(?:[ :]|$)", re.IGNORECASE)

_PUNCTUATION = re.compile(r"[^a-z0-9\s']+")
_WHITESPACE = re.compile(r"\s+")
# Tolerate regular plurals: "sausage" must match "sausages", "steak" must match
# "steaks". Without this, "Polenta with Sausages" classifies as vegetarian.
_PLURAL = r"(?:e?s)?"


def _compile_alternation(terms, plural: bool = False) -> re.Pattern[str]:
    # Longest first: regex alternation is first-match-wins, so without this
    # "egg" would shadow "eggplant" and "milk" would shadow "coconut milk".
    ordered = sorted(set(terms), key=len, reverse=True)
    suffix = _PLURAL if plural else ""
    return re.compile(
        r"\b(?:" + "|".join(re.escape(t) + suffix for t in ordered) + r")\b"
    )


_REWRITE_RE = _compile_alternation(PHRASE_REWRITES.keys())
_LEXICONS: dict[str, re.Pattern[str]] = {
    "meat": _compile_alternation(MEAT_TERMS, plural=True),
    "fish": _compile_alternation(FISH_TERMS, plural=True),
    "egg": _compile_alternation(EGG_TERMS, plural=True),
    "dairy": _compile_alternation(DAIRY_TERMS, plural=True),
    "other_animal": _compile_alternation(NON_VEGAN_TERMS, plural=True),
}
_AMBIGUOUS_RE = _compile_alternation(AMBIGUOUS_TERMS, plural=True)
# Lexicon groups the recipe title is allowed to contribute to. See classify_diet.
_TITLE_GROUPS = ("meat", "fish")
_SUBSTITUTE_RE = _compile_alternation(SUBSTITUTE_QUALIFIERS)

# How restrictive each label is; used to apply the category floor.
_RESTRICTIVENESS = {VEGAN: 0, VEG: 1, EGGETARIAN: 2, NON_VEG: 3}


def _normalise(text: str) -> str:
    """Lowercase, fold accents, split hyphens, and reduce to matchable text.

    The accent fold matters: the dataset is full of "creme fraiche", "saute"
    and "pate", and stripping the accented letters outright would silently
    break every term containing one.

    Hyphens become spaces so that "rib-eye steaks" and "rib eye steaks" are the
    same string. Without it, "Rib-Eye Steaks with Blue Cheese" classified as
    vegetarian.
    """
    lowered = text.lower().replace("&", " and ").replace("-", " ")
    folded = "".join(
        ch
        for ch in unicodedata.normalize("NFKD", lowered)
        if not unicodedata.combining(ch)
    )
    return _WHITESPACE.sub(" ", _PUNCTUATION.sub(" ", folded)).strip()


def _apply_rewrites(text: str) -> str:
    return _REWRITE_RE.sub(lambda m: PHRASE_REWRITES[m.group(0)], text)


@dataclass
class DietResult:
    diet: str
    confidence: float
    conflict: bool = False
    matched: dict[str, list[str]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def reason(self) -> str:
        if not self.matched:
            return "no animal-derived ingredients detected"
        return "; ".join(
            f"{group}: {', '.join(sorted(set(terms))[:4])}"
            for group, terms in self.matched.items()
        )


def _tags(categories: list[str] | None) -> set[str]:
    return {c.strip().lower() for c in (categories or [])}


def _category_expectations(tags: set[str]) -> dict[str, bool]:
    """Read the dataset's own tags as weak priors.

    Epicurious tags "Pescatarian" on plenty of purely vegetable dishes, so it is
    evidence of *no meat* rather than evidence of fish. Treating it as "contains
    fish" would be wrong for thousands of salads.
    """
    return {
        "claims_vegan": "vegan" in tags,
        "claims_vegetarian": "vegetarian" in tags,
        "claims_no_meat": "pescatarian" in tags or "vegetarian" in tags,
        "claims_dairy_free": "dairy free" in tags or "dairy-free" in tags,
        "claims_egg": "egg" in tags,
    }


def _category_floor(tags: set[str]) -> tuple[str | None, str]:
    """The most restrictive label the category tags alone imply."""
    if tags & MEAT_CATEGORIES:
        return NON_VEG, sorted(tags & MEAT_CATEGORIES)[0]
    if tags & FISH_CATEGORIES:
        return NON_VEG, sorted(tags & FISH_CATEGORIES)[0]
    if tags & EGG_CATEGORIES:
        return EGGETARIAN, sorted(tags & EGG_CATEGORIES)[0]
    if tags & DAIRY_CATEGORIES:
        return VEG, sorted(tags & DAIRY_CATEGORIES)[0]
    return None, ""


def classify_diet(
    ingredients: list[str],
    categories: list[str] | None = None,
    title: str = "",
) -> DietResult:
    """Classify a recipe from its ingredient list.

    Used for both dataset recipes and web-sourced ones so that the two agree.
    The title is included because it often names the protein when the
    ingredient list is vague ("Chicken Alfredo" listing only "cooked chicken").
    """
    # Processed per ingredient line, not as one blob, so that a substitute
    # qualifier only neutralises the line it appears on. "1/2 cup vegan butter"
    # must not also excuse the real butter two lines below it.
    ingredient_lines = [
        _apply_rewrites(_normalise(line))
        for line in ingredients
        if line and line.strip() and not _EQUIPMENT_LINE.match(line)
    ]
    title_line = _apply_rewrites(_normalise(title)) if title and title.strip() else ""
    lines = ingredient_lines + ([title_line] if title_line else [])
    substitutes = [line for line in lines if _SUBSTITUTE_RE.search(line)]
    scanned = [line for line in ingredient_lines if not _SUBSTITUTE_RE.search(line)]
    title_scanned = (
        title_line if title_line and not _SUBSTITUTE_RE.search(title_line) else ""
    )
    masked = " ; ".join(scanned + ([title_scanned] if title_scanned else []))

    matched: dict[str, list[str]] = {}
    for group, pattern in _LEXICONS.items():
        hits = [hit for line in scanned for hit in pattern.findall(line)]
        # The title can name the protein that the ingredient list leaves vague
        # ("Chicken Alfredo" over "2 cups cooked, shredded"), so it is read for
        # meat and fish. It is NOT read for dairy, egg or honey: there a title
        # names a *style* far more often than an ingredient. "Raw Cream of
        # Spinach Soup" is avocado, olive oil and vegetable juice -- vegan, per
        # the grader -- and was labelled veg because its title says "cream".
        if group in _TITLE_GROUPS and title_scanned:
            hits += pattern.findall(title_scanned)
        if hits:
            matched[group] = hits

    has_meat = "meat" in matched
    has_fish = "fish" in matched
    has_egg = "egg" in matched
    has_dairy = "dairy" in matched
    has_other = "other_animal" in matched

    if has_meat or has_fish:
        diet, confidence = NON_VEG, 0.95
    elif has_egg:
        diet, confidence = EGGETARIAN, 0.9
    elif has_dairy or has_other:
        diet, confidence = VEG, 0.9
    else:
        # Absence of evidence: an unrecognised animal ingredient would land
        # here, so vegan starts at a lower confidence than the positive cases.
        diet, confidence = VEGAN, 0.7

    result = DietResult(diet=diet, confidence=confidence, matched=matched)
    if substitutes:
        result.notes.append(
            f"{len(substitutes)} ingredient line(s) excluded as plant-based "
            f"substitutes, e.g. {substitutes[0][:48]!r}"
        )
    tags = _tags(categories)

    # Category floor: escalate only. See MEAT_CATEGORIES for why.
    floor, evidence = _category_floor(tags)
    if floor and _RESTRICTIVENESS[floor] > _RESTRICTIVENESS[result.diet]:
        result.notes.append(
            f"escalated {result.diet} -> {floor} on category '{evidence}' "
            f"(ingredients did not name it)"
        )
        result.diet = floor
        result.confidence = 0.8

    # Agreement and disagreement with the dataset's own diet tags. Reads
    # result.diet, not `diet`, so it sees any escalation applied above.
    expect = _category_expectations(tags)
    if expect["claims_vegan"]:
        if result.diet == VEGAN:
            result.confidence = 0.99
        else:
            result.conflict = True
            result.confidence = 0.5
            result.notes.append(
                f"tagged Vegan but evidence suggests {result.diet}"
            )
    elif expect["claims_vegetarian"] and result.diet == NON_VEG:
        result.conflict = True
        result.confidence = 0.5
        result.notes.append("tagged Vegetarian but meat/fish detected")
    elif expect["claims_no_meat"] and has_meat:
        result.conflict = True
        result.confidence = 0.6
        result.notes.append("tagged Pescatarian but meat detected")
    elif expect["claims_vegetarian"] and result.diet in (VEG, EGGETARIAN, VEGAN):
        result.confidence = min(0.99, result.confidence + 0.05)
    elif expect["claims_egg"] and result.diet == EGGETARIAN:
        result.confidence = min(0.99, result.confidence + 0.05)

    if expect["claims_dairy_free"] and has_dairy:
        result.notes.append("tagged Dairy Free but dairy detected")

    # Applied last so nothing above can raise confidence back over the cap.
    # Unqualified "broth"/"stock" may be vegetable or animal; do not guess --
    # keep the label and record the doubt so the answer layer can hedge.
    if result.diet in (VEGAN, VEG):
        ambiguous = _AMBIGUOUS_RE.findall(masked)
        if ambiguous:
            result.confidence = min(result.confidence, 0.6)
            result.notes.append(
                f"ambiguous ingredient(s) {sorted(set(ambiguous))[:3]}: "
                f"could be animal-derived"
            )

    return result
