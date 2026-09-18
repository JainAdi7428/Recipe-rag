"""One place for the model, the retries and the money.

Every LLM call in this project goes through here, for three reasons:

  1. **The budget is a hard requirement.** The spec caps total spend at $2, so
     cost cannot be something we estimate afterwards -- each call reports its own
     price from the provider's own token counts, and the ledger accumulates.
  2. **The model is mandated.** gpt-4o-mini, set once in config, never inline.
  3. **Failure must degrade, not crash.** The judge and the web fallback are
     enhancements over a working rule-based system. If the API is down, out of
     credit, or slow, callers get `LLMUnavailable` and fall back to the phase-1
     gate rather than showing the user a stack trace.

Temperature is pinned at 0 so that an evaluation run is reproducible and a
verdict can be regression-tested. Nothing in this pipeline wants creativity.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field

from . import config


class LLMUnavailable(RuntimeError):
    """The model could not be reached, or returned something unusable.

    Always catchable: no caller should treat the LLM as load-bearing.
    """


@dataclass
class Usage:
    """Token and cost accounting for one call, or a summed set of them."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    seconds: float = 0.0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            calls=self.calls + other.calls,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
            seconds=self.seconds + other.seconds,
        )

    def __str__(self) -> str:
        return (
            f"{self.calls} call(s), {self.input_tokens} in / "
            f"{self.output_tokens} out, ${self.cost_usd:.6f}, {self.seconds:.2f}s"
        )


@dataclass
class Ledger:
    """Running total for a session, with the budget check the spec asks for."""

    total: Usage = field(default_factory=Usage)
    budget_usd: float = config.LLM_BUDGET_USD
    by_purpose: dict[str, Usage] = field(default_factory=dict)

    def record(self, purpose: str, usage: Usage) -> None:
        self.total = self.total + usage
        self.by_purpose[purpose] = self.by_purpose.get(purpose, Usage()) + usage

    @property
    def remaining_usd(self) -> float:
        return self.budget_usd - self.total.cost_usd

    @property
    def exhausted(self) -> bool:
        return self.remaining_usd <= 0

    def report(self) -> str:
        lines = [f"spend  ${self.total.cost_usd:.6f} of ${self.budget_usd:.2f} "
                 f"({100 * self.total.cost_usd / max(self.budget_usd, 1e-9):.3f}%)"]
        for purpose, usage in sorted(self.by_purpose.items()):
            lines.append(f"  {purpose:<12} {usage}")
        return "\n".join(lines)


LEDGER = Ledger()


class LLM:
    """Lazily-constructed OpenAI client.

    The import and the key lookup happen on first use, not at module import, so
    that `import recipe_rag` keeps working on a machine with no key and no
    openai package -- the phase-1 index tooling must not gain a dependency on
    the phase-2 answer layer.
    """

    def __init__(self, model: str | None = None, ledger: Ledger | None = None) -> None:
        self.model = model or config.LLM_MODEL
        self.ledger = ledger if ledger is not None else LEDGER
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        try:
            from openai import AzureOpenAI, OpenAI
        except ImportError as exc:  # pragma: no cover - environment issue
            raise LLMUnavailable(
                "the openai package is not installed; "
                "run: pip install -r requirements.txt"
            ) from exc
        try:
            key = config.openai_key()
        except config.MissingKeyError as exc:
            raise LLMUnavailable(str(exc)) from exc

        azure = config.azure_settings()
        if azure:
            self._client = AzureOpenAI(
                api_key=key,
                azure_endpoint=azure["endpoint"],
                api_version=azure["api_version"],
                timeout=config.LLM_TIMEOUT_SECONDS,
            )
            # Azure addresses the model by deployment name.
            self.model = azure["deployment"]
        else:
            self._client = OpenAI(api_key=key, timeout=config.LLM_TIMEOUT_SECONDS)
        return self._client

    @property
    def available(self) -> bool:
        """True if a call could plausibly be made. Does not hit the network."""
        try:
            self._ensure_client()
        except LLMUnavailable:
            return False
        return True

    def complete_json(
        self,
        system: str,
        user: str,
        purpose: str = "llm",
        max_tokens: int = 200,
        attempts: int = 3,
    ) -> tuple[dict, Usage]:
        """Call the model in JSON mode and return (parsed, usage).

        JSON mode is used rather than prose parsing because the verdicts this
        project needs are structured -- a choice index and a reason -- and a
        free-text answer that has to be regex-scraped is exactly where a
        judge silently starts "accepting" things it did not mean to.
        """
        if self.ledger.exhausted:
            raise LLMUnavailable(
                f"token budget exhausted: ${self.ledger.total.cost_usd:.4f} of "
                f"${self.ledger.budget_usd:.2f} spent"
            )
        client = self._ensure_client()

        last_error: Exception | None = None
        for attempt in range(attempts):
            started = time.perf_counter()
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    temperature=config.LLM_TEMPERATURE,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
            except Exception as exc:  # noqa: BLE001 - provider exception types vary
                last_error = exc
                if not _retryable(exc) or attempt == attempts - 1:
                    break
                # Jittered backoff: a burst of eval queries hitting a rate limit
                # should not all retry on the same tick.
                time.sleep((2**attempt) * 0.5 + random.random() * 0.3)
                continue

            elapsed = time.perf_counter() - started
            counts = response.usage
            usage = Usage(
                calls=1,
                input_tokens=counts.prompt_tokens,
                output_tokens=counts.completion_tokens,
                cost_usd=config.llm_cost(
                    counts.prompt_tokens, counts.completion_tokens, self.model
                ),
                seconds=elapsed,
            )
            # Recorded even when parsing fails below: the tokens were spent.
            self.ledger.record(purpose, usage)

            content = response.choices[0].message.content or ""
            try:
                return json.loads(content), usage
            except json.JSONDecodeError as exc:
                raise LLMUnavailable(
                    f"model returned unparseable JSON: {content[:200]!r}"
                ) from exc

        raise LLMUnavailable(
            f"{type(last_error).__name__}: {last_error}"
        ) from last_error


def _retryable(exc: Exception) -> bool:
    """Rate limits and 5xx are worth retrying; 401 and 404 never are."""
    status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
    if status in (429, 500, 502, 503, 504):
        return True
    if status is not None:
        return False
    # Connection and timeout errors carry no status code.
    return "timeout" in type(exc).__name__.lower() or "connection" in type(exc).__name__.lower()
