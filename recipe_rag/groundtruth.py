"""The grader's ground truth, and the rules for scoring our answers against it.

UC13 is graded by running a fixed set of queries and comparing each answer with
an expected one on four things: **source, recipe, dietary class, nutrition**.
This module reads that file and turns each comparison into an explicit rule, so
a score is reproducible and every miss can be traced to exactly one of the four.

What the file taught us, which shaped the rules below:

  * Queries are descriptive with constraints ("a gluten-free salad with tuna and
    asparagus"), not bare dish names. Our own title-derived test set measured
    the wrong thing.
  * Diet is written in the grader's vocabulary -- "Vegetarian, Contains Egg"
    where we say "eggetarian" -- so labels are compared through a fixed mapping,
    never by string equality.
  * Nutrition is a " | "-separated line, "Not specified" for a missing value,
    and sometimes carries fields we do not report (carbs, fibre). Only the four
    nutrients we report are compared.
  * The file is not infallible: row 3's cooking instructions belong to a
    chocolate tart, not the tuna salad it is filed under. So instructions are
    never scored by text, and the dataset record is treated as authoritative
    when the two disagree.

Read with the standard library (zipfile + ElementTree) rather than openpyxl:
ten rows do not justify a new dependency.
"""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

from . import config
from .extract import _quantity

SOURCE_DATASET = "dataset"
SOURCE_WEB = "web"
SOURCE_NOT_FOUND = "not_found"

NUTRIENTS = ("calories", "protein", "fat", "sodium")

# Our four labels spelled the way the grader spells them. The answer layer uses
# this same table for its output, so what we print and what we are graded on
# can never drift apart.
DIET_DISPLAY = {
    "vegan": "Vegan",
    "veg": "Vegetarian",
    "eggetarian": "Vegetarian, Contains Egg",
    "non-veg": "Non-Vegetarian",
}

_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def _fold(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


_DISPLAY_TO_DIET = {_fold(label): diet for diet, label in DIET_DISPLAY.items()}


def diet_from_display(text: str) -> str | None:
    """"Vegetarian, Contains Egg" -> "eggetarian". Unknown or "NA" -> None."""
    return _DISPLAY_TO_DIET.get(_fold(text or ""))


def source_kind(text: str) -> str:
    """"Internal Database" / "Web Search (a.com, b.com)" / "NA" -> our three."""
    lowered = (text or "").strip().lower()
    if lowered.startswith("internal"):
        return SOURCE_DATASET
    if lowered.startswith("web"):
        return SOURCE_WEB
    return SOURCE_NOT_FOUND


def normalise_title(title: str) -> str:
    return re.sub(r"\s+", " ", title or "").strip().lower()


def parse_nutrition(text: str) -> dict[str, float | None] | None:
    """"Calories: 547 | Protein: 20g | Fat: 32g | Sodium: 452mg" -> numbers.

    "Not specified" becomes None, which is how the dataset's own missing values
    are represented, so a correctly-missing field compares equal. Sodium goes
    through the same mg/g handling as web extraction. Returns None for "NA".
    """
    stripped = (text or "").strip()
    if not stripped or stripped.upper() == "NA":
        return None
    out: dict[str, float | None] = {}
    for part in stripped.split("|"):
        name, _, value = part.partition(":")
        name = name.strip().lower()
        if name in NUTRIENTS:
            out[name] = _quantity(value, to_mg=(name == "sodium"))
    return out or None


def compare_nutrition(
    expected: dict[str, float | None],
    actual: dict[str, float | None],
    tolerance: float = 0.5,
) -> list[tuple[str, float | None, float | None, bool]]:
    """Field-by-field comparison of the nutrients the grader listed."""
    rows = []
    for name, want in expected.items():
        got = actual.get(name)
        if want is None or got is None:
            ok = want is None and got is None
        else:
            ok = abs(float(want) - float(got)) <= tolerance
        rows.append((name, want, got, ok))
    return rows


@dataclass
class Case:
    number: int
    query: str
    answer: str
    nutrition_text: str
    diet_text: str
    source_text: str

    @property
    def expected_source(self) -> str:
        return source_kind(self.source_text)

    @property
    def expected_title(self) -> str | None:
        match = re.search(r"Recipe Title:\s*(.+)", self.answer)
        return match.group(1).strip() if match else None

    @property
    def expected_diet(self) -> str | None:
        return diet_from_display(self.diet_text)

    @property
    def expected_nutrition(self) -> dict[str, float | None] | None:
        return parse_nutrition(self.nutrition_text)


def _column(ref: str) -> int:
    n = 0
    for ch in re.match(r"[A-Z]+", ref).group(0):
        n = n * 26 + ord(ch) - 64
    return n - 1


def read_xlsx_rows(path: Path) -> list[list[str]]:
    """First worksheet as rows of strings, blank rows dropped."""
    with zipfile.ZipFile(path) as book:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in book.namelist():
            for item in ET.fromstring(book.read("xl/sharedStrings.xml")).iter(f"{_NS}si"):
                shared.append("".join(t.text or "" for t in item.iter(f"{_NS}t")))
        workbook = ET.fromstring(book.read("xl/workbook.xml"))
        first = workbook.find(f"{_NS}sheets")[0]
        rels = ET.fromstring(book.read("xl/_rels/workbook.xml.rels"))
        target = {r.get("Id"): r.get("Target") for r in rels}[first.get(f"{_REL}id")]
        sheet = ET.fromstring(book.read("xl/" + target.lstrip("/").removeprefix("xl/")))

    rows: list[list[str]] = []
    for row in sheet.iter(f"{_NS}row"):
        cells: dict[int, str] = {}
        for cell in row.findall(f"{_NS}c"):
            kind = cell.get("t")
            value = cell.find(f"{_NS}v")
            if kind == "inlineStr":
                text = "".join(t.text or "" for t in cell.iter(f"{_NS}t"))
            elif value is None:
                text = ""
            elif kind == "s":
                text = shared[int(value.text)]
            else:
                text = value.text or ""
            cells[_column(cell.get("r"))] = text
        # The grader's sheet is formatted down to row 1000; only real rows count.
        if any(v.strip() for v in cells.values()):
            rows.append([cells.get(i, "") for i in range(max(cells) + 1)])
    return rows


def read_cases(path: Path | None = None) -> list[Case]:
    """Load the ground-truth queries. Column names are matched loosely, since
    the file spells one of them "Dietry_Class"."""
    path = path or config.GROUND_TRUTH_XLSX
    rows = read_xlsx_rows(path)
    header = [h.strip().lower() for h in rows[0]]

    def column(*names: str) -> int:
        for name in names:
            if name in header:
                return header.index(name)
        raise ValueError(f"ground truth has none of {names}; columns are {header}")

    i_no = column("s. no.", "s.no.", "no", "#")
    i_query = column("input query", "query")
    i_answer = column("answer")
    i_nutrition = column("nutrition information", "nutrition")
    i_diet = column("dietry_class", "dietary_class", "diet")
    i_source = column("source")

    cases: list[Case] = []
    for row in rows[1:]:
        def cell(i: int) -> str:
            return row[i].strip() if i < len(row) else ""

        if not cell(i_query):
            continue
        try:
            number = int(float(cell(i_no)))
        except ValueError:
            number = len(cases) + 1
        cases.append(
            Case(number, cell(i_query), cell(i_answer), cell(i_nutrition),
                 cell(i_diet), cell(i_source))
        )
    return cases
