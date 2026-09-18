"""The relevance judge: an LLM adjudicating the band no threshold can split.

Retrieval hands back a ranked list. In the confident and hopeless bands the
scores already answer the question. In between they do not, and the reason is
visible in the numbers: present and absent recipes overlap between roughly 0.49
and 0.60 cosine on this index. A threshold placed in that overlap does not
resolve it, it only picks which way to be wrong.

What the judge adds is the ability to *read the candidates*. "Chicken Tikka
Masala" and "Masala Dosa" are close in embedding space and share a word, and no
amount of threshold tuning separates them; a model that has seen both dishes
answers instantly. Equally, "Spinach and Feta Phyllo Triangles" is the right
answer to "Spanakopita" while scoring near zero on every lexical measure we have.

**The risk this buys is over-acceptance.** A small model shown eight candidates
and asked "is any of these relevant?" will say yes. That failure is worse than
the one it replaces, because a fabricated source now arrives wrapped in a fluent
justification instead of a suspiciously low number. Four things push back:

  1. The question is "does this recipe MAKE the dish asked for", never "is this
     relevant". Chicken Tikka Masala is genuinely relevant to Masala Dosa, and
     completely wrong.
  2. Named and descriptive requests get explicitly different bars. "Something
     creamy with pasta and chicken" has many correct answers; "Beef Rendang"
     has exactly one, or none.
  3. A worked refusal is shown in the prompt, so `null` is a demonstrated
     option rather than a theoretical one.
  4. The judge must echo the title it chose, and that echo is scored against
     the query lexically. Where the echo AND the vector both disagree with the
     verdict, the verdict is vetoed -- two independent signals overruling one.

Whether that is enough is a measured question, not an argued one: see
`scripts/eval_judge.py`, which reports over-acceptance on hard negatives.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .gate import DEFAULT, GateThresholds
from .lexical import coverage_score
from .llm import LLM, LLMUnavailable, Usage

TYPE_NAMED = "named"
TYPE_DESCRIPTIVE = "descriptive"

SYSTEM_PROMPT = """\
You decide whether a recipe database already contains what a user asked for.

You get the user's request and a numbered list of candidate recipes retrieved \
from the database. Pick the candidate that makes the dish the user asked for, \
or report that none of them do.

First classify the request:

  named        The user named a specific dish ("Spanakopita", "Beef Rendang", \
"Pav Bhaji"). A candidate counts ONLY if it is that same dish, possibly under a \
descriptive or translated name. A recipe that is merely similar, from the same \
cuisine, or shares one ingredient is NOT a match.
  descriptive  The user described what they want ("something creamy with pasta \
and chicken", "a light summer dessert"). Any candidate that genuinely satisfies \
the description is correct. Pick the best one.

Answering with the wrong recipe is far worse than answering "none". On a named \
request, if you are not sure the candidate is that exact dish, choose null.

Examples of the distinction:
  "Masala Dosa" vs candidate "Chicken Tikka Masala" -> null. Shares a word, \
different dish entirely.
  "Spanakopita" vs candidate "Spinach and Feta Phyllo Triangles" -> match. \
Same dish, described rather than named.
  "something creamy with pasta and chicken" vs candidate "Chicken Alfredo with \
Zucchini Ribbons" -> match. Satisfies the description.

Reply with JSON only, no other text:
{"type": "named" or "descriptive",
 "choice": <candidate number, or null>,
 "dish": "<the exact title you chose, or empty string>",
 "reason": "<one short sentence>"}"""


@dataclass
class Candidate:
    """One retrieved recipe as the judge will see it."""

    parent_id: str
    title: str
    diet: str = ""
    categories: str = ""
    ingredients: list[str] = field(default_factory=list)
    vector_score: float = 0.0

    def render(self, number: int, n_ingredients: int = 5) -> str:
        parts = [f"{number}. {self.title}"]
        if self.diet:
            parts.append(f"[{self.diet}]")
        line = " ".join(parts)
        if self.categories:
            line += f"\n   categories: {self.categories[:110]}"
        if self.ingredients:
            # Ingredients are what make a descriptive request answerable at all;
            # a title alone cannot tell you whether a dish is creamy.
            shown = "; ".join(item[:52] for item in self.ingredients[:n_ingredients])
            line += f"\n   ingredients: {shown[:260]}"
        return line


@dataclass
class JudgeVerdict:
    accepted: bool
    parent_id: str | None = None
    title: str | None = None
    query_type: str = ""
    reason: str = ""
    echo_coverage: float = 0.0
    vetoed: bool = False
    usage: Usage = field(default_factory=Usage)
    raw: dict = field(default_factory=dict)

    def describe(self) -> str:
        if self.vetoed:
            return (
                f"judge chose {self.title!r} but both the name check "
                f"({self.echo_coverage:.0f}) and the vector disagreed -- vetoed"
            )
        if self.accepted:
            return f"judge ({self.query_type}): {self.reason}"
        return f"judge ({self.query_type}): none of the candidates match -- {self.reason}"


def build_prompt(query: str, candidates: list[Candidate]) -> str:
    listing = "\n".join(
        candidate.render(index) for index, candidate in enumerate(candidates, 1)
    )
    return f"User request: {query}\n\nCandidates:\n{listing}"


# A named-request echo scoring below this is treated as the judge disagreeing
# with the lexical evidence. On its own that means nothing -- the Spanakopita
# case scores 0 here and is correct -- so it only bites when the vector agrees
# that the candidate is weak.
ECHO_VETO_BELOW = 40.0


def judge(
    query: str,
    candidates: list[Candidate],
    llm: LLM | None = None,
    thresholds: GateThresholds = DEFAULT,
    veto: bool = True,
) -> JudgeVerdict:
    """Ask the model which candidate, if any, answers `query`.

    Raises `LLMUnavailable` if the model cannot be reached, so the caller can
    fall back to the rule-based gate rather than failing the request.
    """
    if not candidates:
        return JudgeVerdict(accepted=False, reason="no candidates retrieved")

    llm = llm or LLM()
    parsed, usage = llm.complete_json(
        system=SYSTEM_PROMPT,
        user=build_prompt(query, candidates),
        purpose="judge",
        max_tokens=120,
    )

    query_type = str(parsed.get("type") or "").strip().lower()
    if query_type not in (TYPE_NAMED, TYPE_DESCRIPTIVE):
        query_type = TYPE_NAMED  # the stricter bar when the model is unclear
    reason = str(parsed.get("reason") or "").strip()[:200]
    choice = parsed.get("choice")

    if not isinstance(choice, int) or not 1 <= choice <= len(candidates):
        return JudgeVerdict(
            accepted=False,
            query_type=query_type,
            reason=reason or "no candidate selected",
            usage=usage,
            raw=parsed,
        )

    chosen = candidates[choice - 1]
    echo = str(parsed.get("dish") or "").strip()
    # Scored against the title we actually hold, not the model's paraphrase of
    # it, so a sloppy echo cannot fabricate agreement.
    coverage = coverage_score(query, chosen.title)

    vetoed = (
        veto
        and query_type == TYPE_NAMED
        and coverage < ECHO_VETO_BELOW
        and chosen.vector_score < thresholds.vector_support
    )
    if vetoed:
        return JudgeVerdict(
            accepted=False,
            parent_id=chosen.parent_id,
            title=chosen.title,
            query_type=query_type,
            reason=reason,
            echo_coverage=coverage,
            vetoed=True,
            usage=usage,
            raw=parsed,
        )

    return JudgeVerdict(
        accepted=True,
        parent_id=chosen.parent_id,
        title=chosen.title,
        query_type=query_type,
        reason=reason or f"selected candidate {choice}",
        echo_coverage=coverage,
        usage=usage,
        raw=parsed,
    )


def candidates_from(
    ranked,
    title_hits,
    parents,
    vector_n: int = 8,
    title_n: int = 3,
) -> list[Candidate]:
    """Assemble the candidate pool: vector top-N union title top-N.

    The union matters. A vector-only pool silently caps the judge below the
    thing it exists to fix -- "Spanakopita" is found by title at 100 and by
    vector at 0.487, so on a busy result set it can fall out of the vector top-8
    entirely. Feeding both retrievers' opinions in costs a few hundred tokens
    and removes that ceiling.
    """
    by_id: dict[str, float] = {}
    order: list[str] = []
    for hit in ranked[:vector_n]:
        if hit.parent_id not in by_id:
            order.append(hit.parent_id)
        by_id[hit.parent_id] = max(by_id.get(hit.parent_id, 0.0), hit.score)
    for hit in title_hits[:title_n]:
        if hit.parent_id not in by_id:
            order.append(hit.parent_id)
            by_id[hit.parent_id] = 0.0

    indexed = parents.set_index("parent_id")
    out: list[Candidate] = []
    for parent_id in order:
        if parent_id not in indexed.index:
            continue
        row = indexed.loc[parent_id]
        categories = row.get("categories", "")
        if isinstance(categories, (list, tuple)):
            categories = ", ".join(str(item) for item in categories)
        ingredients = row.get("ingredients", [])
        out.append(
            Candidate(
                parent_id=parent_id,
                title=str(row["title"]),
                diet=str(row.get("diet", "")),
                categories=str(categories or ""),
                ingredients=list(ingredients) if ingredients is not None else [],
                vector_score=by_id[parent_id],
            )
        )
    return out
