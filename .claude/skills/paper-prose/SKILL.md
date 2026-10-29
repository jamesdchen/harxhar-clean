---
name: paper-prose
description: "Style patterns for the harxhar JFEc paper. Use it before you write or rewrite any text of this paper: sections, beats, bullets, body sentences, section titles, captions, table notes, note strings in writeup/make_*.py generators, and the wording you apply from an author's edit or a deliberation outcome. It routes each task to the few pattern files it needs in writeup/style/patterns/ (index: writeup/style/voice.md) and ends with a self-check. Never use it to change the author's own words or text collaborators typed on Overleaf (those are copied or kept as written), and never to polish a layer the author has not locked. Coordinators name it in every drafting subagent's prompt."
---

# paper-prose: which style patterns to read, by task

The author, 29 Oct 2026: "the style guide seems like it should be part of a skill or something. Maybe it needs to be
invoked like a Gang of 4 style skill tree". The rules are pattern files, one per rule, in the GoF template (intent,
applicability, rewrite, detection, source). This skill picks the ones a task needs: read only those, in the order given.

- **Index:** `writeup/style/voice.md` (V1–V16, AP-V, AP-S).
- **Patterns:** `writeup/style/patterns/<family>/<id>-<slug>.md`; families `claims/`, `numbers/`, `sentences/`,
  `scope/`, `anti-patterns/`. The branches below name files by id, and each id is one file:
  `writeup/style/patterns/*/<id>-*.md` (V6 → `claims/V6-no-after-the-fact-tails.md`, AP-S →
  `anti-patterns/AP-S-structural-markers.md`).
- **Design:** the report's S0 drafting guide and its limits A2 and A4 (`writeup/reports/AI layered paper writing
  system.md` §8.4, Author decisions).
- **Questions:** never ask the author. Queue them with the `deliberate` skill.

## Step 0: whose text is it? (every task)

1. **Words the author supplied** for the paper → copy them byte for byte; no pattern changes them. Read V13.
2. **Text pulled from Overleaf** → do not restyle it. Read V15.
3. **Existing agent text outside your edit's target** → leave its style alone until its layer is locked. Read V14.
4. **Unsure** whether words are the author's to keep as written → keep them as written and queue the question
   (deliberate skill).

## Step 1: pick the branch and read its patterns in order

**A. Drafting a claim: a sentence, beat or bullet (L0–L3).**
V1 → V4 → V6 → V5 → V7 → V2 → V3 → V10 → AP-V. For a beat label, add V8.

**B. Reporting a number or a test.**
V2 → V4 → V6 → V7 → V3. Add V5 when the sentence has a universal, superlative or booster.

**C. A caption, table note or generator note string (`writeup/make_*.py`).**
V2 (its generator bullet) → V3 → V4 → V6 → V1 → V5.

**D. A section, subsection or appendix title.**
V9 → V3.

**E. Applying an author's edit or a deliberation outcome.**
V13 → V14 → V15 (when the target came from Overleaf). Then the branch (A–D) for the words you write yourself, and V16
when you reword existing agent text without changing its meaning.
- A deliberation outcome's prescribed wording is agent text: check it against V6, V5 and V4 before applying it, and
  keep the meaning it records. If a pattern would change that meaning, queue the item instead.
- Park the replaced text in a dated `%` comment (notation guide, house rules).

**F. Edit-stage pass on a locked layer (scrub, register, typography).**
V14 first: if `writeup/layer_state.json` does not record the layer as locked, stop. (On 29 Oct 2026 the file does not
exist, so no layer is locked.) Then V13 → V15 (what to skip) → AP-V → AP-S → V6 (review flags only) → V5 → V10 → V12 →
V16 → V11.

**G. Reviewing a draft (critic).**
Run the checklist below on the draft. Report each finding with its pattern id and, for a style finding, a named
operation (V12). Do not rewrite. A reviewer's request for a new number, test or citation is a proposal, never a task
for the writer (§8.6).

Scope comes first because it stops work that should not happen; claims come before numbers and sentences because a
wrong claim matters more than its wording; anti-patterns come last.

## Step 2: check your own output before you return it

One line per pattern. Skip a line only when its pattern does not apply to your task.

- [ ] V13: the author's words are byte-identical and recorded with their date.
- [ ] V15: no sentence pulled from Overleaf was restyled.
- [ ] V14: on an unlocked layer, nothing outside the edit's target changed.
- [ ] V1: one claim a sentence; subject and verb early.
- [ ] V2: no typed digit or number word outside the whitelist; generator strings format computed values.
- [ ] V3: each estimator, set, sample and test has the notation guide's name.
- [ ] V4: each comparison, verdict and magnitude word matches the sign, p-value or threshold of its macro.
- [ ] V5: each universal, completeness adverb, superlative and booster passed NEEDED?, then TRUE? with its backing named.
- [ ] V6: no clause-final tail of a few words after `,` `;` `:` or a dash that carries a quantifier or a verdict.
- [ ] V7: findings carry their sample, period and test; hedges sit on mechanism or interpretation only.
- [ ] V8: beat labels state claims; sibling beats do not overlap.
- [ ] V9: titles are conventional noun phrases, with no colon, question or clause; labels unchanged.
- [ ] V10: no inanimate subject with an evaluative verb.
- [ ] V11: no sentence copied or closely paraphrased from a paper.
- [ ] V12: (edit stage) each style change is one named operation, logged with its rule id.
- [ ] V16: (rewrites) the same macros, cites, refs, negations, modals, direction words and agents.
- [ ] AP-V: no hard-flag word, field whitelist excepted.
- [ ] AP-S: no em dash; no participial tail, signposting or summary opener outside the conclusion where a plain form
  exists.

The Detection section of V2, V5, V6, V10, AP-V and AP-S gives a grep for candidates. Most automatic checks (C2, C5,
C18, C25, C26, `scrub.py`) are not built yet, so this checklist is the check.

## Subagents

- **Coordinators:** in every prompt that has a subagent write or rewrite paper text, name this skill and the branch
  ("Use the paper-prose skill, branch B"), or list that branch's pattern files in the prompt's input manifest (§8.6).
  Give no global style instruction such as "make it read like a journal" or "vary sentence length" (V12).
- **A subagent without the Skill tool** reads this file, then its branch's pattern files.
- **Subagents return findings, not questions.** Anything for the author goes through the deliberate skill; a subagent
  without the Agent tool runs its step 1 only and returns the queued ids.
