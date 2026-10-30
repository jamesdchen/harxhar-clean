# V2. Numbers come from macros

**Family:** numbers. **Stage:** drafting (S0); gate C2 on L0–L4. **Basis:** D (data-to-paper; Jr. AI Scientist; Beel;
§8.3 C2, §8.6).

**Severity:** gate. A number typed into live text blocks the return; the drafter replaces it with a generated macro, or `\rhPending{<macro name>}`, before the text is accepted.
**Checked by:** code (claims slice). The code flags digits in live text outside the exemptions and the C2 whitelist; number words ("six classes"), which it misses, are read by the claims auditor.

## Intent

Every number in the paper's text is a generated macro, so it moves with the results and traces back to the code that
made it.

## Also known as

Never type a digit (§8.6); C2; P0.1; the notation guide's house rule "every number stays a generated macro".

## Motivation

data-to-paper ties every numeric value to a label in the analysis output, and typed numbers were not among its reported
errors (§3.1). This paper has the same layer: `writeup/generated/*.tex`, each macro with a `%` comment naming what it
measures. `\rhchOrdLoOtherTreeMaxAbs`, for example, is the "largest |full w/o g vs full| % over lgbm, xgb"
(`writeup/generated/rehaul_channels_numbers.tex:929`), and `08_feature_importance.tex:224` uses it. The report's first
build step (P0.1, §8.7) is a check that fails on any digit typed into live text.

## Applicability

- Live text in every layer. Exempt: `\ref`, `\cite`, labels, `%` comments, and the C2 whitelist (years, "HAR(3)",
  "Section 3"; each entry needs a reason, §8.7).
- A number written as a word ("six classes") is still a number under the notation guide's house rule, and C2 does not
  catch it. Take it from a count macro.
- Derived numbers (a ratio, a difference, a share) are computed in Python and emitted as a macro with a formula comment
  (§8.6).
- Generator captions and note strings (`writeup/make_*.py`): format the number from the computed value, or from
  `locks.py` for a locked parameter once that exists (C20). A literal typed into the string is a typed digit.
- Digits inside author-supplied text are exempt from C2 but must agree with the decision they state (C26; V13).

## Structure / Rewrite

`\rhX…` marks a macro the generator would have to emit.

| Instead of | Write |
|---|---|
| "lowers QLIKE by 4.2\%" | "lowers QLIKE by `\rhX…Pct`\%" |
| "in all six model classes" | "in all `\rhorAfPairs` model classes" |
| "($\DM = 2.31$, $p = 0.021$)" | "($\DM = $ `\rhX…Dm`, $p = $ `\rhX…P`)" |
| `f"... by at most 0.2\\%"` in a generator | `f"... by at most {max_abs:.1f}\\%"`, with `max_abs` computed from the results file |

Until the macro exists, write `\rhPending{<macro name>}`, which prints the missing name in the PDF
(`writeup/paper_rehaul.tex:73`).

## Consequences

Costs: a new number needs a generator change and a rerun before its sentence is final. Buys: no transcription errors,
and a chain from each number to its code. It does not catch a stale macro file (C3, §3.2) or a wrong word about a
right number (V4).

## Detection

C2, `tests/check_numbers.py` (gate in the build and in `push --check`, once built; Phase 0). By hand:
`grep -nE '[0-9]' <file> | grep -v ':\s*%'`, then inspect each hit outside math, `\ref`, `\cite` and labels, and read
for number words. In the scrub, S1 turns macros into placeholder tokens, and S8 rejects a rewrite that changes the
multiset of macros (V16).

## Related patterns

V3, V4, V5, V13, V16.

## Source

`writeup/style/voice.md` V2; report §3.1, §8.3 C2 and C20, §8.4 S0, §8.6 ("Never type a digit. Derived numbers are
computed in Python and emitted as macros with a formula comment."), §8.7 P0.1; `writeup/NOTATION_GUIDE_2026-10-15.md`,
"House rules" ("every number stays a generated macro ... Never type a number in place of a macro, and never rename a
macro").
