"""Fixtures for the dietary classifier, weighted towards the failure modes.

Run standalone:  python tests/test_diet.py
Or under pytest: pytest tests/test_diet.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recipe_rag.diet import EGGETARIAN, NON_VEG, VEG, VEGAN, classify_diet

# (label, expected_diet, ingredients, categories)
CASES: list[tuple[str, str, list[str], list[str]]] = [
    # --- plain positives -------------------------------------------------
    ("chicken", NON_VEG, ["2 boneless chicken breasts", "1 tbsp olive oil"], []),
    ("salmon", NON_VEG, ["1 lb salmon fillet", "lemon"], []),
    ("plain vegan", VEGAN, ["2 cups rolled oats", "1 ripe banana", "cinnamon"], []),
    ("dairy makes it veg", VEG, ["1 cup whole milk", "2 cups flour"], []),
    ("egg makes it eggetarian", EGGETARIAN, ["3 large eggs", "2 cups flour"], []),
    ("honey is not vegan", VEG, ["2 tbsp honey", "1 cup oats"], []),

    # --- substring traps: must NOT trigger the obvious lexicon ------------
    ("eggplant is not egg", VEGAN, ["1 large eggplant", "olive oil", "salt"], []),
    ("butternut is not butter", VEGAN, ["1 butternut squash", "sage", "olive oil"], []),
    ("peanut butter is vegan", VEGAN, ["2 tbsp peanut butter", "1 banana"], []),
    ("cocoa butter is vegan", VEGAN, ["50 g cocoa butter", "cocoa powder"], []),
    ("coconut milk is vegan", VEGAN, ["1 can coconut milk", "2 cups rice"], []),
    ("almond milk is vegan", VEGAN, ["1 cup almond milk", "oats"], []),
    ("cream of tartar is vegan", VEGAN, ["1 tsp cream of tartar", "2 cups flour"], []),
    ("cheesecloth is not cheese", VEGAN, ["cheesecloth for straining", "2 cups rice"], []),
    ("graham does not match ham", VEGAN, ["2 cups graham flour", "water", "salt"], []),
    ("graham crackers do contain dairy", VEG, ["1 cup graham cracker crumbs", "sugar"], []),
    ("collard does not match lard", VEGAN, ["1 bunch collard greens", "garlic"], []),
    ("beefsteak tomato is vegan", VEGAN, ["2 beefsteak tomatoes", "basil"], []),
    ("oyster mushroom is vegan", VEGAN, ["8 oz oyster mushrooms", "thyme"], []),
    ("bean curd is vegan", VEGAN, ["1 block bean curd", "soy sauce"], []),
    ("honeydew is vegan", VEGAN, ["1 honeydew melon", "mint"], []),
    ("butter lettuce is vegan", VEGAN, ["1 head butter lettuce", "vinaigrette"], []),
    ("crab apple is vegan", VEGAN, ["2 lb crabapples", "sugar"], []),
    ("lamb's lettuce is vegan", VEGAN, ["4 oz lamb's lettuce", "walnut oil"], []),

    # --- redirected phrases: right lexicon, not the modifier --------------
    ("goat cheese is veg not meat", VEG, ["4 oz goat cheese", "beet"], []),
    ("goat's milk is veg", VEG, ["2 cups goat's milk", "flour"], []),
    ("quail eggs are eggetarian", EGGETARIAN, ["6 quail eggs", "salt"], []),
    ("duck eggs are eggetarian", EGGETARIAN, ["2 duck eggs", "flour"], []),

    # --- hidden animal products ------------------------------------------
    ("worcestershire hides anchovy", NON_VEG, ["1 tsp Worcestershire sauce", "tomato juice"], []),
    ("fish sauce", NON_VEG, ["1 tbsp fish sauce", "lime", "rice noodles"], []),
    ("gelatin is not vegetarian", NON_VEG, ["1 packet gelatin", "2 cups juice"], []),
    ("oyster sauce", NON_VEG, ["2 tbsp oyster sauce", "broccoli"], []),
    ("lard", NON_VEG, ["1/2 cup lard", "3 cups flour"], []),
    ("mayonnaise is egg", EGGETARIAN, ["1/2 cup mayonnaise", "2 cups cabbage"], []),
    ("duck fat is meat", NON_VEG, ["3 tbsp duck fat", "potatoes"], []),

    # --- accents must survive normalisation -------------------------------
    ("creme fraiche accented", VEG, ["1/2 cup crème fraîche", "chives"], []),
    ("saute accented is harmless", VEGAN, ["sauté the shallots", "olive oil"], []),
    ("pate accented is meat", NON_VEG, ["4 oz pâté", "baguette"], []),

    # --- precedence -------------------------------------------------------
    ("meat beats dairy", NON_VEG, ["1 lb ground beef", "1 cup milk", "2 eggs"], []),
    ("egg beats dairy", EGGETARIAN, ["3 eggs", "1 cup milk", "flour"], []),
    ("buttermilk is still dairy", VEG, ["1 cup buttermilk", "2 cups flour"], []),

    # --- title carries the protein ---------------------------------------
    ("title supplies the meat", NON_VEG, ["1 lb pasta", "cream", "parmesan"], []),

    # --- regressions found auditing the full corpus ----------------------
    # Plurals: "Polenta with Sausages" classified as vegetarian.
    ("plural sausages", NON_VEG, ["1 1/2 pounds hot Italian sausages", "polenta"], []),
    ("plural steaks", NON_VEG, ["4 rib eye steaks", "olive oil"], []),
    ("plural chops", NON_VEG, ["4 pork chops", "sage"], []),
    ("chopped is not a chop", VEGAN, ["2 cups chopped tomatoes", "basil"], []),
    # Hyphens: "Rib-Eye Steaks with Blue Cheese" classified as vegetarian.
    ("hyphenated rib-eye", NON_VEG, ["2 8-ounce rib-eye steaks", "pepper"], []),
    ("hyphenated half-and-half", VEG, ["1 cup half-and-half", "flour"], []),
    # Vocabulary gaps found in the corpus.
    ("skate is fish", NON_VEG, ["10 ounces boneless skate wing", "mesclun"], []),
    ("orange roughy is fish", NON_VEG, ["4 orange roughy fillets", "leeks"], []),
    ("duckling is meat", NON_VEG, ["4 pound Long Island duckling", "cognac"], []),
    ("game hens are meat", NON_VEG, ["2 Cornish game hens, halved", "sage"], []),
    ("spareribs are meat", NON_VEG, ["3 lb spareribs", "paprika", "sugar"], []),
    ("court bouillon is fish", NON_VEG, ["court bouillon", "sea salt"], []),
    # Composite items that reliably contain butter or cheese.
    ("puff pastry is not vegan", VEG, ["1 sheet puff pastry", "2 apples"], []),
    ("phyllo is not vegan", VEG, ["8 sheets phyllo", "spinach"], []),
    ("pesto is not vegan", VEG, ["1/2 cup pesto", "1 lb pasta"], []),
    ("white chocolate is dairy", VEG, ["6 oz white chocolate", "raspberries"], []),
    ("marshmallows hide gelatin", NON_VEG, ["2 cups mini marshmallows", "cocoa"], []),
    ("chop suey is not a pork chop", VEGAN, ["vegetable chop suey", "rice"], []),
    # Plant parts that borrow butchery words.
    ("celery ribs are vegan", VEGAN, ["2 celery ribs, finely chopped", "tofu"], []),
    ("kidney beans are vegan", VEGAN, ["2 cans kidney beans, rinsed", "chili powder"], []),
    ("hot dog buns are not hot dogs", VEGAN, ["6 hot dog buns", "lettuce"], []),
    ("vegetable marrow is vegan", VEGAN, ["1 vegetable marrow, sliced", "olive oil"], []),
    ("short ribs are still meat", NON_VEG, ["3 lb bone-in short ribs", "red wine"], []),
    # Plant-based substitutes: the qualifier neutralises its own line.
    ("vegan butter stays vegan", VEGAN, ["1/2 cup vegan butter", "2 cups flour"], []),
    ("non-dairy yogurt stays vegan", VEGAN, ["1 cup non-dairy yogurt", "granola"], []),
    ("mock chicken stays vegan", VEGAN, ["8 oz mock chicken strips", "soy sauce"], []),
    ("vegan mayonnaise stays vegan", VEGAN, ["1/4 cup vegan mayonnaise", "cabbage"], []),
    # ...but only its own line: real butter elsewhere must still register.
    ("substitute does not excuse other lines", VEG,
     ["1/2 cup vegan butter", "1 cup heavy cream"], []),
    ("substitute does not excuse meat", NON_VEG,
     ["8 oz mock duck", "1 lb ground beef"], []),
    ("tofu sour cream stays vegan", VEGAN, ["1 cup tofu sour cream", "tortillas"], []),
    # Hyphens normalise to spaces, so these must match in their split form too.
    ("honey-dew melon is vegan", VEGAN, ["1 honey-dew melon, cubed", "mint"], []),
    ("bread-and-butter pickles are vegan", VEGAN,
     ["1 tbsp chopped bread-and-butter pickles", "cider vinegar"], []),
    # A title names a style, not an ingredient, for dairy and egg. The grader
    # labels "Raw Cream of Spinach Soup" Vegan; its title's "cream" made it veg.
    ("cream-style title over vegan ingredients", VEGAN,
     ["2 cups fresh tomato juice", "2 cups chopped spinach leaves",
      "1/2 medium avocado", "1/2 cup olive oil"], []),
    ("egg-named title over eggless ingredients", VEGAN,
     ["1 lb firm tofu", "1/2 tsp turmeric", "black salt"], []),
    # ...while a real cream-of soup in the ingredient list is still dairy.
    ("canned cream of mushroom soup is dairy", VEG,
     ["1 can cream of mushroom soup", "2 cups rice"], []),
    # Egg hidden in a sub-recipe name, or written without the word "egg".
    ("pastry cream sub-recipe is egg", EGGETARIAN,
     ["1 sheet puff pastry", "Pastry Cream", "2 tbsp sugar"], []),
    ("bare yolks are egg", EGGETARIAN, ["4 large yolks", "1 cup sugar", "2 cups milk"], []),
    ("lemon curd is egg", EGGETARIAN, ["1 cup lemon curd", "shortbread"], []),
    # Equipment notes are not food. A plum sorbet was labelled veg because its
    # ingredient list ends "Equipment: an ice cream maker".
    ("equipment is not an ingredient", VEGAN,
     ["1 pound ripe plums", "3/4 cup dry red wine", "3/4 cup sugar",
      "Equipment: an ice cream maker"], []),
    ("special equipment is not an ingredient", VEGAN,
     ["2 cups water", "1 cup rice", "Special equipment: cheesecloth"], []),
    # ...but a real ingredient naming the same thing still counts.
    ("ice cream as an ingredient is dairy", VEG,
     ["2 cups vanilla ice cream", "1 cup root beer"], []),
]

TITLE_CASES = {
    "title supplies the meat": "Chicken Alfredo",
    "cream-style title over vegan ingredients": "Raw Cream of Spinach Soup",
    "egg-named title over eggless ingredients": "Scrambled Eggs, Vegan Style",
}


def run() -> int:
    failures = []
    for label, expected, ingredients, categories in CASES:
        title = TITLE_CASES.get(label, "")
        result = classify_diet(ingredients, categories, title)
        if result.diet != expected:
            failures.append(
                f"  {label!r}: expected {expected}, got {result.diet} "
                f"({result.reason})"
            )

    # Category cross-check should flag, not silently agree.
    conflict = classify_diet(["1 lb chicken breast"], ["Vegetarian"], "")
    if not conflict.conflict:
        failures.append("  category conflict not flagged for chicken tagged Vegetarian")

    agreement = classify_diet(["2 cups oats", "1 banana"], ["Vegan"], "")
    if agreement.diet != VEGAN or agreement.confidence < 0.95:
        failures.append(
            f"  vegan agreement should raise confidence, got {agreement.confidence}"
        )

    # Pescatarian is a no-meat claim, not a contains-fish claim.
    salad = classify_diet(["romaine", "olive oil"], ["Pescatarian", "Vegetarian"], "")
    if salad.diet != VEGAN:
        failures.append(f"  pescatarian salad should stay vegan, got {salad.diet}")

    # The category floor escalates when ingredients hide the animal product,
    # e.g. a rub whose meat only appears in the category tags.
    rub = classify_diet(
        ["1/4 cup sugar", "2 tbsp lemon pepper", "1 tbsp paprika"],
        ["Pork", "Backyard BBQ"],
        "",
    )
    if rub.diet != NON_VEG:
        failures.append(f"  Pork category should escalate to non-veg, got {rub.diet}")

    # ...but it must never relax a label in the other direction.
    beef = classify_diet(["1 lb ground beef"], ["Dairy"], "")
    if beef.diet != NON_VEG:
        failures.append(f"  Dairy category must not relax non-veg, got {beef.diet}")

    # Unqualified stock is ambiguous: keep the label, lower the confidence.
    broth = classify_diet(["4 cups broth", "1 cup rice"], [], "")
    if broth.diet != VEGAN or broth.confidence > 0.6 or not broth.notes:
        failures.append(
            f"  unqualified broth should stay {VEGAN} at low confidence, "
            f"got {broth.diet}/{broth.confidence} notes={broth.notes}"
        )

    total = len(CASES) + 6
    if failures:
        print(f"FAILED {len(failures)}/{total}")
        print("\n".join(failures))
        return 1
    print(f"ok  {total}/{total} diet cases passed")
    return 0


def test_diet_cases():
    assert run() == 0


if __name__ == "__main__":
    raise SystemExit(run())
