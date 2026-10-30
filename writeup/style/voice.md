# Voice guide for drafting the paper: catalog index

The drafting guide (stage S0) of `writeup/reports/AI layered paper writing system.md` (§8.2, §8.4), kept as a catalog
of patterns. Each rule is one file under `writeup/style/patterns/<family>/`, written in the Gang-of-Four template
adapted to prose: intent, also known as, motivation, applicability, structure and rewrite, consequences, detection,
related patterns, source. This file only indexes them; the pattern files hold the rules.

**Entry point.** Agents reach the catalog through the `paper-prose` skill (`.claude/skills/paper-prose/SKILL.md`). It
sends each writing task (a claim, a number, a caption, a title, an applied edit, an edit-stage pass) to the patterns it
needs, in order, and ends with the audit stage (its Step 3). Read patterns directly only when a coordinator named them.

**Audit stage.** The drafter's self-check is no longer the gate; an audit of the returned text is. Code checks run
first: `writeup/style/prose_check.py` runs the Detection checks of every pattern whose Checked-by line names code (if
the script is absent, the auditors run those greps by hand). Then three read-only auditors
(`.claude/agents/prose-auditor-*.md`) run in parallel, each owning one slice of the catalog: claims (V2, V4, V5, V6,
V7), sentences (V1, V3, V8, V9, V10, V11, AP-V, AP-S) and scope (V12–V16). Each auditor confirms or dismisses the
code's candidates in its slice and judges its auditor-only patterns; none edits the text. Confirmed violations go back
to the drafter with pattern id and line, and each pattern's Severity line, under its title, says what follows: gate, a
confirmed hit blocks the return until the drafter fixes it; review, the auditor confirms or dismisses each hit and the
drafter fixes the confirmed ones; rate, counted per 1,000 words against a band, never forcing an edit by itself. The
loop runs at most two rounds. Violations left after the second round go to the deliberate queue
(`writeup/deliberation/queue.jsonl`), never to the author directly. A hit in existing text outside the edit's target
is not sent back: it waits for the edit stage (V14), or goes to the queue as a flipped claim if it is false.

**Scope.** The catalog governs agent-written text: sections, beats, bullets, captions, table notes, generator note
strings and wording applied from a deliberation outcome. Text the author supplied in an edit message is protected
verbatim and never changed to fit these rules (A4, §4.3.7 "Scope"; V13). Text people typed on Overleaf is not
restyled (V15). The drafting patterns apply before lock; the edit-stage patterns (V12, V16) and the scrub run only on
layers the author has locked (A2; V14).

**Ids.** V1–V6 keep the numbers they had in the flat guide of 29 Oct 2026 (commit 2f61ba3), so references in commits,
deliberation records and the report's S3 row still resolve. V7–V16 were added when the guide became a catalog. AP-V and
AP-S are the anti-pattern catalogs of scrub stages S2 and S3, with row ids AP-V1… and AP-S1…. An id is never reused.
When `style/rules/` (§8.2) is built for `scrub.py`, its rules should carry these ids.

**Sources and grades.** Every pattern cites the report, the notation guide or the author's dated words. Its basis uses
the report's grades: D (tested or mandated), T-code and T-genre (transferred), P (prescriptive, no measured effect),
I (inferred, untested); "author" marks the author's own rule. In rewrite examples numbers appear as macro names, and
`\rhX…` marks a macro a generator would have to emit.

| Id | Name | Family | Stage | Severity | Checked by | Intent | File |
|---|---|---|---|---|---|---|---|
| V1 | Plain declarative sentences | sentences | S0 | review | auditor (sentences) | One plain claim a sentence, subject and verb early | `patterns/sentences/V1-plain-declarative.md` |
| V2 | Numbers come from macros | numbers | S0; C2 | gate | code (claims) | Every number is a generated macro, never typed | `patterns/numbers/V2-numbers-from-macros.md` |
| V3 | One name per estimator | numbers | S0 | gate | code + auditor (sentences) | The notation guide's single name for each estimator, set, sample and test | `patterns/numbers/V3-one-name-per-estimator.md` |
| V4 | Verdict and magnitude words come from predicates | claims | S0; C5 | gate | auditor (claims) | Comparison, verdict and magnitude words only where a predicate licenses them | `patterns/claims/V4-verdict-words-from-predicates.md` |
| V5 | Rule NT: NEEDED? first, then TRUE? | claims | S0; S4b; C18 | review; a TRUE? failure is gate | code + auditor (claims) | Universals, superlatives and boosters only if needed, then only if true | `patterns/claims/V5-rule-nt.md` |
| V6 | No punchy qualification after the fact | claims | S0; S3 | gate | code + auditor (claims) | No clipped verdict tail after a finished clause | `patterns/claims/V6-no-after-the-fact-tails.md` |
| V7 | Scope and hedges match the evidence | claims | S0; C19 | review | auditor (claims) | Findings carry sample, period and test; hedges only on interpretation | `patterns/claims/V7-scope-and-hedges-match-evidence.md` |
| V8 | Beat labels state a claim | sentences | S0; C8 | review | auditor (sentences) | A beat label is a claim, not a topic | `patterns/sentences/V8-beat-labels-state-claims.md` |
| V9 | Conventional section titles | sentences | S0 | review | code + auditor (sentences) | Titles are the field's noun phrases, not claims, questions or colons | `patterns/sentences/V9-conventional-titles.md` |
| V10 | Inanimate subjects do not evaluate | sentences | S0; S4a | review | code + auditor (sentences) | No "the results underscore"; state the number | `patterns/sentences/V10-inanimate-subjects-do-not-evaluate.md` |
| V11 | Take the literature's register, not its sentences | sentences | S0; S5, S6, S8 | review | auditor (sentences) | Phrase frames and corpus statistics, never copied sentences | `patterns/sentences/V11-register-not-sentences.md` |
| V12 | Named sentence operations, not style targets | sentences | S7 (locked layers) | gate (edit stage) | code (scope) | One named operation per flagged sentence; stop inside the band | `patterns/sentences/V12-named-sentence-operations.md` |
| V13 | The author's text is copied verbatim | scope | all | gate | code (scope) | Author-supplied text byte for byte, never restyled (A4) | `patterns/scope/V13-author-text-verbatim.md` |
| V14 | No polish before lock | scope | all; C25 | gate | code + auditor (scope) | No style edit of existing text until its layer is locked (A2) | `patterns/scope/V14-no-polish-before-lock.md` |
| V15 | Text typed on Overleaf is not restyled | scope | all | gate | code + auditor (scope) | Human text pulled from Overleaf is exempt from the scrub (provisional) | `patterns/scope/V15-overleaf-text-not-restyled.md` |
| V16 | A rewrite preserves content | scope | S8 | gate | code + auditor (scope) | Same macros, cites, refs, negations, modals, direction words and agents | `patterns/scope/V16-rewrite-preserves-content.md` |
| AP-V | Vocabulary markers | anti-patterns | S2 | per row | code + auditor (sentences) | Hard-flag and density-flag words, with the field whitelist | `patterns/anti-patterns/AP-V-vocabulary-markers.md` |
| AP-S | Structural markers | anti-patterns | S3 | per row | code + auditor (sentences) | Em dashes, participial tails, signposting, summary openers and the rest | `patterns/anti-patterns/AP-S-structural-markers.md` |

**Not rules yet.** Per-section voice (passive share, "we" by verb class) waits for the corpus bands (§4.3.2; S0; §8.9
Q7), and phrase frames wait for the corpus (S6; §8.9 Q4). Both enter the catalog only when built and confirmed.
