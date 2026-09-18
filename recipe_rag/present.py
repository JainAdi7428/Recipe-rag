"""Render an answer in the shape the grader compares against.

The validation file has four columns per query -- Answer, Nutrition
Information, Dietry_Class, Source -- and the submission is judged on whether
ours matches theirs. So the wording lives here, in one place, rather than being
improvised by whatever happens to be printing:

    Recipe Title: Spinach Noodle Casserole
    Ingredients:
    1 12-ounce package frozen spinach souffle, thawed
    ...
    Cooking Instructions:
    Preheat oven to 350 F. Lightly grease ...

    Nutrition Information : Calories: 547 | Protein: 20g | Fat: 32g | Sodium: 452mg
    Dietry_Class          : Vegetarian, Contains Egg
    Source                : Internal Database

Three rules this file exists to keep:

  * **Their vocabulary, not ours.** We store "eggetarian"; the grader writes
    "Vegetarian, Contains Egg". The mapping is shared with the scorer
    (groundtruth.DIET_DISPLAY) so what we print and what we are graded on can
    never drift apart.
  * **"Not specified" rather than a guess.** A missing nutrient is reported as
    missing, in the grader's own phrasing. Carbohydrates are never shown: the
    dataset has no such field, and inventing one is the hallucination the spec
    forbids.
  * **Nothing is generated.** Every line is the stored record or the
    publisher's own markup. The apology is a fixed sentence, not a model's
    improvisation.
"""

from __future__ import annotations

from dataclasses import dataclass

from .groundtruth import DIET_DISPLAY
from .pipeline import SOURCE_DATASET, SOURCE_WEB, Answer

NOT_SPECIFIED = "Not specified"
NA = "NA"

# (attribute, label, unit) in the order the grader lists them.
NUTRIENTS = (
    ("calories", "Calories", ""),
    ("protein", "Protein", "g"),
    ("fat", "Fat", "g"),
    ("sodium", "Sodium", "mg"),
)


@dataclass
class Presented:
    """One row of the submission, matching the validation file's columns."""

    answer: str
    nutrition: str
    diet: str
    source: str

    def as_text(self) -> str:
        return (
            f"{self.answer}\n\n"
            f"Nutrition Information: {self.nutrition}\n"
            f"Dietry_Class: {self.diet}\n"
            f"Source: {self.source}"
        )


def nutrition_line(recipe) -> str:
    """"Calories: 547 | Protein: 20g | Fat: 32g | Sodium: 452mg"."""
    if recipe is None:
        return NA
    parts = []
    for attribute, label, unit in NUTRIENTS:
        value = getattr(recipe, attribute, None)
        if value is None:
            parts.append(f"{label}: {NOT_SPECIFIED}")
        else:
            parts.append(f"{label}: {value:g}{unit}")
    return " | ".join(parts)


def diet_label(recipe) -> str:
    if recipe is None or not recipe.diet:
        return NA
    return DIET_DISPLAY.get(recipe.diet, recipe.diet)


def source_label(answer: Answer) -> str:
    if answer.source == SOURCE_DATASET:
        return "Internal Database"
    if answer.source == SOURCE_WEB:
        domain = answer.recipe.domain if answer.recipe else ""
        return f"Web Search ({domain})" if domain else "Web Search"
    return NA


def recipe_text(recipe) -> str:
    lines = [f"Recipe Title: {recipe.title}", "Ingredients:"]
    lines += list(recipe.ingredients)
    lines.append("Cooking Instructions:")
    lines += list(recipe.directions)
    return "\n".join(lines)


def present(answer: Answer) -> Presented:
    """Turn a pipeline answer into the grader's four columns."""
    if answer.recipe is None:
        return Presented(answer=answer.message, nutrition=NA, diet=NA, source=NA)
    return Presented(
        answer=recipe_text(answer.recipe),
        nutrition=nutrition_line(answer.recipe),
        diet=diet_label(answer.recipe),
        source=source_label(answer),
    )


def as_markdown(answer: Answer) -> str:
    """The same content as markdown, for the Streamlit app."""
    if answer.recipe is None:
        return answer.message
    recipe = answer.recipe
    out = [f"### {recipe.title}", ""]
    if recipe.url:
        out.append(f"[{recipe.domain}]({recipe.url})"
                   + (f" -- by {recipe.author}" if recipe.author else ""))
        out.append("")
    out += ["**Ingredients**", ""]
    out += [f"- {item}" for item in recipe.ingredients]
    out += ["", "**Cooking Instructions**", ""]
    out += [f"{index}. {step}" for index, step in enumerate(recipe.directions, 1)]
    return "\n".join(out)
