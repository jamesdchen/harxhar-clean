# V4. Verdict and magnitude words come from predicates

**Family:** claims. **Stage:** drafting (S0); check C5 on L2–L4. **Basis:** D for the risk (Alizadeh et al.: the
prompt flipped verdicts; §8.6); I for the predicate check (§3.3, C5).

**Severity:** gate. A comparison, verdict, magnitude or rank word with no predicate, stated threshold or recorded hand check behind it blocks the return; the drafter backs it with the macro and its test, or removes it, before the text is accepted.
**Checked by:** auditor (claims slice). Judgment until the predicates and C5 exist: the auditor looks up each such word's macro in `writeup/generated/` and checks its sign, rank, p-value and the stated level.

## Intent

Write a comparison, verdict or magnitude word only where a generated predicate or a stated threshold licenses it.

## Also known as

Direction words from predicates (§3.3); C5; generator-emitted magnitude words (§4.3.7).

## Motivation

No tool checks direction words, and LLM checkers do poorly at it (SPOT, SciTab; §3.3). The paper already makes the
check by hand: a parked comment records that the abstract's "all ... each significant" is backed by
`\rhorAfLower = \rhorAfLowerSig = \rhorAfPairs` (`writeup/paper_rehaul.tex:241`). The 29 Oct round shows a verdict with
no predicate behind it: Section 8 said the trees' attribution "ranks the parts the wrong way round", but across the
eight groups the attribution ranks correlate positively with the add-alone gains, and the reversal is the FOMC
calendar's alone. The claim was narrowed (Q-2026-10-29-26, commit 136794b).

## Applicability

- Comparison words: lower, higher, beats, improves.
- Verdict words: significant, concentrated, sparse, dense, redundant.
- Magnitude words: substantially, materially, modestly, small, large.
- Rank words: largest, smallest, first (V5 also governs them).

"Significant" means statistically significant at the stated level; economic size takes another word ("economically
large"), as the notation guide's §6 asks. The guide's preferred magnitude words ("substantially", "materially",
"modestly") survive only as generator-emitted words tied to stated thresholds (§4.3.7; the switch is open as §8.9 Q7).
All agent text; not author-supplied text (V13).

## Structure / Rewrite

Give the result, then its test (notation guide §6): the macro, a direction word that matches its sign, and the test
statistic or p-value. `\rhX…` marks a macro the generator would have to emit.

| Instead of | Write |
|---|---|
| "The full set substantially improves accuracy." | "The full set lowers QLIKE by `\rhX…Pct`\% ($\DM = $ `\rhX…Dm`, $p = $ `\rhX…P`)." |
| "The gain is concentrated in a few series." | Use "concentrated" only where the ladder generator's classification says so for that model and span, and give the measure with it (`\ladderEBTopK` series and the share of the gain they keep). |

Until predicates exist, check by hand and record the backing in a `%` comment beside the sentence, as
`paper_rehaul.tex:241` does.

## Consequences

Costs: a sentence that needs a new verdict waits for the generator to emit it, or for a recorded hand check. Buys:
verdicts change with the numbers instead of going stale silently, and C24 can tell a flipped claim from an unchanged one.

## Detection

C5, `tests/check_claims.py` over `generated/*_predicates.tex` (gate on L2–L4, once both exist; Phases 2 and 5). By
hand: for each sentence with a numeric macro and a comparison, verdict or magnitude word, look the macro up in
`writeup/generated/` and check its sign, rank, p-value and the stated level. S4b reuses the predicates for TRUE? (V5),
and S8 keeps direction words unchanged in rewrites (V16).

## Related patterns

V2, V5, V6, V7, V16.

## Source

`writeup/style/voice.md` V4; report §3.3, §4.3.7 (last paragraphs), §8.3 C5, §8.6 ("Verdict words come only from
predicates derived from the pre-registered `RULE_*.txt`"); `writeup/NOTATION_GUIDE_2026-10-15.md` §6.
