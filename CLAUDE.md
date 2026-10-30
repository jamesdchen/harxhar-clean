# harxhar-0dte-professor: project rules

Moved here from the global `~/.claude/CLAUDE.md` on 2026-10-29 (the author: project-specific rules belong in the
repo, not global). The Overleaf push rule was dropped: the post-commit hook (`writeup/overleaf_sync/install_hook.sh`)
pushes the paper after every commit that changes `writeup/`.

## Pre-commit checks
Before any `git commit`, run these **in parallel** on all staged `.py` files (`C:\Users\james\miniconda3\Scripts`):
1. `ruff check --fix`
2. `ruff format --line-length 120`
3. `mypy --ignore-missing-imports`

Fix any issues, then commit.

## The paper
- **Writing-process reference** (user, 2026-10-28): when designing or building the paper-writing system (layers,
  results hookup, prose scrub, review loop, citations, disclosure), read `writeup/reports/AI layered paper writing
  system.md` first and cite its sections (e.g. §8.3 C18, Rule NT §4.3.7); its sources are in
  `writeup/research_notes/AI layered paper writing system/`. Section 8 is the recommended design and phased build
  order.
- **Question escalation** (user, 2026-10-29: "only the important stuff reaches me"): never put a question to the
  author directly. Enqueue it in `writeup/deliberation/queue.jsonl` and run the `deliberate` skill (rules:
  `writeup/deliberation/PROTOCOL.md`); the author sees only the round's `digest.md` (T3 questions and high-impact T2
  one-liners).
