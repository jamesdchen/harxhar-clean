---
name: paper-prose
description: "Style and claim rules for the text of the harxhar JFEc paper. Use it whenever the conversation turns to writing the paper up: the author talking about the write-up, drafting, wording, phrasing, prose, how to say or frame a result, a sentence, paragraph, section, title, caption or table note, editing or rewriting text, putting a result or table into the paper, or edits to the PDF or Overleaf. Use it also before any agent writes or rewrites text of this paper: sections, beats, bullets, body sentences, titles, captions, table notes, note strings in writeup/make_*.py generators, and the wording of an author's edit or a deliberation outcome. Never use it to change the author's own words or text collaborators typed on Overleaf (those are kept as written), and never to polish a layer the author has not locked."
---

# paper-prose: draft the paper's text, then audit it

The rules are pattern files, one per rule (intent, applicability, rewrite, detection, source). Step 0 settles whose
text it is, Step 1 picks the patterns a task needs (read only those, in the order given), Step 2 is the drafter's
quick pass, and Step 3 audits the result with code and three read-only auditors.

- **Index:** `writeup/style/voice.md` (V1–V16, AP-V, AP-S).
- **Patterns:** `writeup/style/patterns/*/<id>-*.md`, one file per id (V6 → `claims/V6-no-after-the-fact-tails.md`,
  AP-S → `anti-patterns/AP-S-structural-markers.md`). Where a pattern's `Severity` or `Checked by` line differs from
  Step 2, the pattern file wins.
- **Code checks:** `writeup/style/prose_check.py` (Step 3a).
- **Design:** S0, A2, A4, §8.3 and §8.6 of the report `writeup/reports/AI layered paper writing system.md`.
- **Questions:** never ask the author. Queue them with the `deliberate` skill.

## Step 0: whose text is it? (every task)

1. **Words the author supplied** for the paper → copy them byte for byte; no pattern changes them. Read V13.
2. **Text pulled from Overleaf** → do not restyle it. Read V15.
3. **Existing agent text outside your edit's target** → leave its style alone until its layer is locked. Read V14.
4. **Unsure** whether words are the author's to keep as written → keep them as written and queue the question
   (deliberate skill).

## Step 1: pick the branch and read its patterns in order

**A. Drafting a claim: a sentence, beat or bullet (L0–L3).** After Step 0:
V4 → V5 → V6 → V7 → V2 → V3 → V1 → V10 → AP-V. For a beat label, add V8.

**B. Reporting a number or a test.** V2 → V4 → V6 → V7 → V3; add V5 for a universal, superlative or booster.

**C. Caption, table note or generator string (`writeup/make_*.py`).** V2 (generator bullet) → V3 → V4 → V6 → V1 → V5.

**D. A section, subsection or appendix title.** V9 → V3.

**E. Applying an author's edit or a deliberation outcome.** V13 → V14 → V15 (target from Overleaf); then the branch
(A–D) for the words you write yourself, and V16 when you reword existing agent text without changing its meaning.
- A deliberation outcome's prescribed wording is agent text: check it against V6, V5 and V4 before applying it, and
  keep the meaning it records. If a pattern would change that meaning, queue the item instead.
- Park the replaced text in a dated `%` comment (notation guide, house rules).

**F. Edit-stage pass on a locked layer (scrub, register, typography).** V14 first: if `writeup/layer_state.json` does
not record the layer as locked, stop; if the file is absent, no layer is locked. Then V13 → V15 (what to skip) → AP-V →
AP-S → V6 (review flags only) → V5 → V10 → V12 → V16 → V11.

**G. Reviewing a draft (critic).** Run Step 3 (a) and (b) on it and report the merged findings. Do not rewrite. A
reviewer's request for a new number, test or citation is a proposal, never a task for the writer (§8.6).

Scope comes first because it stops work that should not happen; claims come before numbers and sentences because a
wrong claim matters more than its wording; anti-patterns come last.

## Step 2: pre-audit pass (the drafter, on its own output)

A quick pass before Step 3, not the gate. Run `python writeup/style/prose_check.py lint FILE:START-END --json` on your
changed ranges, fix its non-candidate hits, weigh its candidates, and read the [auditor] lines yourself. Skip a line
only when its pattern does not apply. **[code]**: the script decides. **[code + auditor]**: the script finds candidates
or part of the rule; the slice auditor confirms or dismisses them and judges the rest. **[auditor]**: judgment only.
Severity: **gate** (a confirmed hit blocks the return), **review** (the slice auditor confirms or dismisses), **rate**
(counted only).

- [ ] V13 [code] gate: the author's words are byte-identical and recorded with their date.
- [ ] V15 [code + auditor] gate: no sentence pulled from Overleaf was restyled.
- [ ] V14 [code + auditor] gate: on an unlocked layer, nothing outside the edit's target changed.
- [ ] V16 [code + auditor] gate: (rewrites) same macros, cites, refs, negations, modals, direction words and agents.
- [ ] V12 [code] gate: (edit stage) each style change is one named operation, logged with its rule id.
- [ ] V4 [auditor] gate: each comparison, verdict and magnitude word fits the sign, p-value or threshold of its macro.
- [ ] V5 [code + auditor] review (an unsupported universal, a TRUE? failure, is gate): each universal, completeness
  adverb, superlative and booster passed NEEDED?, then TRUE? with its backing named.
- [ ] V6 [code + auditor] gate: no clause-final tail of a few words after `,` `;` `:` or a dash that carries a
  quantifier or a verdict.
- [ ] V7 [auditor] review: findings carry their sample, period and test; hedges sit on mechanism or interpretation only.
- [ ] V2 [code] gate: no typed digit outside the whitelist; generator strings format computed values (number words:
  the claims auditor).
- [ ] V3 [code + auditor] gate: each estimator, set, sample and test has the notation guide's name.
- [ ] V1 [auditor] review: one claim a sentence; subject and verb early.
- [ ] V10 [code + auditor] review: no inanimate subject with an evaluative verb.
- [ ] V8 [auditor] review: beat labels state claims; sibling beats do not overlap.
- [ ] V9 [code + auditor] review: titles are conventional noun phrases (no colon, question or clause); labels unchanged.
- [ ] V11 [auditor] review: no sentence copied or closely paraphrased from a paper.
- [ ] AP-V [code + auditor]: hard-flag words gate (field whitelist excepted), AP-V3 review, density words [code] rate.
- [ ] AP-S [code + auditor], per row as the file's Severity column gives it: participial tails, summary openers outside
  the conclusion and importance-only sentences review; em-dash rate, triads, signposting and length [code] rate.

Where a pattern file's `**Severity:**` or `**Checked by:**` line differs from a line above, the pattern file wins.

**Do not flag:** the field whitelist (the leverage effect, robust standard errors, significant in its statistical
sense, causal, empirical); quotations, and a word cited as a word; a genuine list of three, including the
contributions paragraph; a long sentence that carries one argument (V1 counts claims, not words); formal register;
section roadmaps and exhibit pointers ("Table~\ref{…} reports …"). The em-dash rule governs agent drafting and is not
a scrub ban: an em dash outside new agent text is not flagged.

## Step 3: audit (the gate)

Every task that writes or rewrites paper text ends here; its text is not returned as final or committed before this
step. Whoever has the Agent tool runs it: the main session or a coordinator. A drafting subagent without the Agent
tool returns its draft, its target line ranges and its edit log to the coordinator, which runs Step 3. **Small edits:**
if the edit changes at most two sentences and no claim, number, caption or note, run (a) only.

**(a) Code checks** (before any auditor, §8.3), on the changed ranges; for edits and rewrites also the diff:

    python writeup/style/prose_check.py lint FILE:START-END ... --json
    python writeup/style/prose_check.py diff --base <rev before the edit> --target FILE:START-END ... \
        [--log EDITLOG.jsonl] [--author-record FILE] --json FILE ...

Output: a JSON list of `{mode, id, file, line, severity, candidate, match, message[, per_1000]}`; exit 1 on a
non-candidate gate hit. Code covers eleven lines (V2, V3, V6, V9, V10, AP-V, AP-S, V12, V13, V14, V16) and part of
V5 and V15. A non-candidate hit is confirmed by code; a candidate goes to its slice's auditor.

**(b) Auditors:** three Agent calls in ONE message, so they run in parallel: `subagent_type` `prose-auditor-claims`
(V2, V4, V5, V6, V7), `prose-auditor-sentences` (V1, V3, V8, V9, V10, V11, AP-V, AP-S) and `prose-auditor-scope`
(V12, V13, V14, V15, V16). Each prompt gives the files and changed line ranges, the code hits of its slice only
(filtered from the JSON by id), and the macro files the ranges use (`writeup/generated/*.tex`). The scope prompt also
gives the base revision and the base text of each changed range (`git show <rev>:<file>`, pasted or saved to a scratch
file, since auditors cannot run git), the edit log, and the author's dated record if there is one. No prompt carries
anything from another auditor: they judge independently.

**(c) Merge.** Each auditor returns `{"slice", "clean": bool, "violations": [{"id", "file", "line", "quote",
"severity", "confirmed": bool, "why", "fix"}], "dismissed": [...]}`: `fix` is one named operation (V12), and
`dismissed` lists the code hits it dismissed, each with its reason.
1. The drafter (the coordinator, if the drafting subagent has returned) fixes every confirmed gate and review
   violation with its named operation, and logs each operation with its rule id, one JSON line per operation:
   `{"file": ..., "line": ..., "op": "<named operation>", "rule_id": "<pattern id>"}` (the `--log` file of (a)).
   A hit in existing text outside the edit's target is not fixed now: it waits for that layer's edit stage (V14), or
   goes to the deliberate queue if it states something false.
2. Rerun (a). Re-audit only the slices that had confirmed gate violations. At most two audit rounds (§8.6).
3. Anything still open or disputed, and any `confirmed: false` finding, goes to the deliberate skill's queue, never
   as a question to the author.
4. Record the audit in the commit message: rounds, and counts by slice and severity (fixed, dismissed, queued).

## Subagents

- **Coordinators** name this skill and the branch in every drafting prompt ("Use the paper-prose skill, branch B"), or
  list that branch's pattern files in the prompt's input manifest (§8.6). Give no global style instruction such as
  "make it read like a journal" or "vary sentence length" (V12).
- **A subagent without the Skill tool** reads this file, then its branch's pattern files.
- **Auditors** (`.claude/agents/prose-auditor-*.md`) are spawned only by Step 3. They are read-only and never draft.
- **Subagents return findings, not questions.** Anything for the author goes through the deliberate skill; a subagent
  without the Agent tool runs its step 1 only and returns the queued ids.

## Sources

- The author, 29 Oct 2026: "the style guide seems like it should be part of a skill or something. Maybe it needs to be
  invoked like a Gang of 4 style skill tree".
- **How it is triggered** (author, 29 Oct 2026, answering Q-2026-10-29-25: "no. We shouldn't need the CLAUDE.md line
  because it should be a skill that I invoke in prose whenever we're talking about writing it up. Make sure the
  paper-prose skill triggers from that, not from CLAUDE.md."): by the description above, when the author's messages
  turn to writing the paper up, and before any agent writes paper text. There is no CLAUDE.md line for it. A
  coordinator still names it in drafting subagents' prompts, because subagents do not receive the author's messages.
- **The audit stage** (author, 29 Oct 2026): "Then the skill should have an audit swarm stage, no? Instead of the
  drafter ticking its own checklist, add a stage where separate read-only auditor agents (parallel, each owning a
  slice of the rules, plus the code checks for the 11 mechanizable lines) audit the draft and feed violations back.
  Fold that into your plan, then apply the first tier plus this stage to paper-prose."
- **Design:** `writeup/reports/Academic writing skills comparison.md` §2 and §6 (B1, B2, B3, B5, B10); report §8.3
  (checks run before any LLM critic) and §8.6 (critics in fresh contexts; oracle-backed loops stop after two).
