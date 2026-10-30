# V5. Rule NT: NEEDED? first, then TRUE?

**Family:** claims. **Stage:** drafting (S0); S4b on locked layers; check C18. **Basis:** P + I (Grice's Quantity
maxim and Cochrane for NEEDED?; the predicate mechanism of §3.3 for TRUE?); the need for the rule is D (Peters &
Chin-Yee; Zhou et al.).

**Severity:** review. A confirmed NEEDED? failure makes the drafter delete the word, and a TRUE? failure, an unsupported universal or superlative, is gate: it blocks the return until the drafter backs the word with a count, argmax or predicate macro, or deletes it.
**Checked by:** code + auditor (claims slice). Candidates only: the code greps the lexicon below; NEEDED? and TRUE? stay the auditor's judgment.

## Intent

Keep a universal, completeness adverb, superlative or booster only if the sentence needs it, and then only if a
computed fact makes it true.

## Also known as

Rule NT; C18; S4b (stance).

## Motivation

The author, 28 Oct 2026: "they also use a bunch of superlatives like "always" "everyone" "never" "no one"", and "but
the point is that sometimes superlatives are sometimes distracting and not necessary if the context necessitates their
implicit validity". In the 28 Oct main text "every" was the most frequent universal (§4.3.6). Many instances name the
evaluation scope, "every regular-hours bar" as against the closing half hour, which the reader needs; others restate a
scope the sentence has already fixed. Only a per-instance test separates them.

## Applicability

The lexicon of §4.3.7: universal quantifiers (all, every, each, any, no, none, always, never, uniformly, without
exception), completeness adverbs (entirely, completely, fully), superlatives (the most/least, -est forms, best, only,
first as a rank, unique) and boosters (clearly, crucially, fundamentally, importantly, indeed, very, remarkably,
strikingly, substantially, dramatically). At drafting, write nothing that fails it (S0). It never applies to
author-supplied text (A4; V13) or to text pulled from Overleaf (V15).

## Structure / Rewrite

- **NEEDED?** Keep the word only if it (a) contrasts scope against a plausible expectation of exceptions, (b) states a
  scope the reader cannot recover from the definitions, the table or the previous two sentences, or (c) is the paper's
  claim itself, stated once in the abstract, introduction or conclusion, with a number. Otherwise delete it, even if
  it is true.
- **TRUE?** A kept literal universal or superlative cites a computed fact that holds: a count equal to the total, an
  argmax over the cited table, or a predicate macro (V4). A rhetorical absolute with no finite set behind it is deleted
  or replaced with a number.

| Instead of (illustrative) | Write | Reason code |
|---|---|---|
| "We score all forecasts over the entire evaluation span." (span defined in the previous sentence) | "We score the forecasts over the evaluation span." | REDUNDANT-SCOPE |
| "on every regular-hours bar, not only the closing half hour" | kept: (a), a scope contrast | none |
| "Crucially, the FOMC calendar carries the statement-day gain." | "The FOMC calendar carries the statement-day gain." | BOOSTER |
| "The reading holds in all `\rhchLinOrdNRedTest` linear classes." | kept only with a predicate that the count of classes where it holds equals `\rhchLinOrdNRedTest` | UNSUPPORTED-UNIVERSAL if none |

Until predicates exist, record the backing beside the sentence, as `writeup/paper_rehaul.tex:270` does for
"significantly in all" (`\rhorAfLowerSig = \rhorAfPairs`).

## Consequences

Costs: a judgment per hit, and a count or argmax macro for each kept literal universal. Buys: no word that says nothing
or makes the reader wonder what the exceptions were. The case rests on information, not on a measured loss of
credibility (§4.3.7).

## Detection

S4b (§8.4), once built. Stage A marks a REDUNDANT-CANDIDATE when the scope noun is in the set registry
(`writeup/style/sets.json`) and no contrast marker ("including", "even", "although", "except", "despite") is near.
Stage B, a deletion-equivalence judge, must name the exception the word excludes, or the word goes. Each deletion is
logged with its reason code; a kept but unbacked universal goes to the agent queue and fails the L4 gate (C18). By
hand: grep your new text for the lexicon, then apply NEEDED? before TRUE?.

## Related patterns

V4, V6, V7, V13, V15.

## Source

Report §4.3.7 (the rule box, including "Scope (A4)"), §4.3.6, §8.3 C18, §8.4 S4b, §8.6; the author's words of 28 Oct
2026 (22:52 and 22:54 PDT), transcribed in `writeup/deliberation/2026-10-29c/Q-2026-10-29-25.evidence.md`, item 6;
`writeup/style/voice.md` V5.
