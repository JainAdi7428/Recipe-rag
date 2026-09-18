"""Verify the LLM connection works, and price one real call.

    python scripts/check_llm.py

Checks, in order, so a failure names its own cause:

  1. the .env file is found and parsed
  2. a key is present, and the provider (OpenAI or Azure OpenAI) is configured
  3. the mandated model answers
  4. the reported token usage prices out where we expect

Prints no part of the key except a 4-character fingerprint. Costs ~$0.00001.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recipe_rag import config
from recipe_rag.llm import LLM, LLMUnavailable


def main() -> int:
    print(f"env file   {config.ENV_FILE}")
    if not config.ENV_FILE.exists():
        print("           NOT FOUND -- copy .env.example to .env")
        return 1
    loaded = config.load_env()
    print(f"           found, keys set: {', '.join(sorted(loaded)) or 'none'}")

    llm = LLM()
    try:
        client = llm._ensure_client()
    except LLMUnavailable as exc:
        print(f"\nsetup      FAILED\n{exc}")
        return 1

    key = config.openai_key()
    print(f"key        ok, {len(key)} chars, ends ...{key[-4:]}")
    azure = config.azure_settings()
    if azure:
        print(f"provider   Azure OpenAI  host={urlparse(azure['endpoint']).netloc}")
        print(f"           deployment={azure['deployment']}  api-version={azure['api_version']}")
    else:
        print("provider   api.openai.com")

    print(f"model      {llm.model}, calling...")
    started = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=llm.model,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=10,
            messages=[{"role": "user", "content": "Reply with exactly: READY"}],
        )
    except Exception as exc:  # noqa: BLE001 -- surface the provider's own words
        print(f"           FAILED after {time.perf_counter() - started:.1f}s")
        print(f"           {type(exc).__name__}: {str(exc)[:300]}")
        print("\n  401 -> key wrong or revoked, or the key belongs to a different endpoint")
        print("  404 -> Azure: deployment name or endpoint wrong; OpenAI: model not enabled")
        print("  429 -> no credit / quota, or rate limited")
        return 1

    elapsed = time.perf_counter() - started
    usage = response.usage
    cost = config.llm_cost(usage.prompt_tokens, usage.completion_tokens)
    print(f"           replied {response.choices[0].message.content.strip()!r} in {elapsed:.2f}s")
    print(f"tokens     {usage.prompt_tokens} in / {usage.completion_tokens} out")
    print(f"cost       ${cost:.8f}   (budget ${config.LLM_BUDGET_USD:.2f} = "
          f"~{int(config.LLM_BUDGET_USD / max(cost, 1e-9)):,} calls this size)")
    print("\nOK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
