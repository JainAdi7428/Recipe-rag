# Superseded

This was the plan for the web fallback, written before it was built. It is kept
only so older references do not dangle.

What was actually built, and why it differs from the plan, is in
[web-fallback.md](web-fallback.md). The measured results are in
[web-evaluation.md](web-evaluation.md).

Three things changed during the build:

- **No domain allowlist.** Publishers known to publish clean markup are tried
  first, but any site is accepted if its markup is valid. An allowlist would have
  made most Indian and South-East Asian recipes unreachable.
- **No `recipe-scrapers` dependency.** schema.org recipes arrive as JSON-LD
  inside a script tag, so the standard library is enough.
- **No LLM answer synthesis.** The model never writes recipe text; it only
  chooses between candidates it was shown. That is a stronger guarantee against
  invented ingredients than any prompt.
