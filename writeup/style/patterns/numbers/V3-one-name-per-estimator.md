# V3. One name per estimator

**Family:** numbers. **Stage:** drafting (S0); also a target of the S7 register pass (§4.5). **Basis:** T-genre (the
slop and LAMP taxonomies, §4.5); the term list is the repo's notation guide.

## Intent

Call each estimator, predictor set, sample and test by the one name the notation guide gives it, everywhere in the
paper.

## Also known as

The term list; the notation pass (15 Oct 2026).

## Motivation

The notation guide's §5 replaces code jargon with standard terms: "all features" becomes "full predictor set",
"per-bar model" becomes "closing-interval model", "baseline" becomes "HAR benchmark", "test window" becomes "evaluation
span". Its addenda found "benchmark" naming three things at once, the predictor set, the reference OLS and the
short-straddle strategy, and gave each its own name (`writeup/NOTATION_GUIDE_ADDENDA_2026-10-15.md` §B.3, row 32). On
28 Oct the author pruned "pooled" and kept it only where it carries the design (commit ae0f526).

## Applicability

All agent text, including titles, captions, table notes and generator strings (addenda §A.4 is a drop-in dictionary
for the generators). Not author-supplied text (V13). Where the guide has no name, it says to "choose the most standard
usage in financial econometrics" and record the choice as a new decision in your final reply.

## Structure / Rewrite

| Instead of | Write |
|---|---|
| "the all_features arm" | "the full predictor set" |
| "the per-bar OLS" | "the closing-interval OLS" |
| "the benchmark" (meaning the OLS forecast) | "the reference forecast" |
| "the test window" | "the evaluation span $\mathcal T_{\mathrm{oos}}$" |
| a synonym chosen for variety ("the richer specification") | the guide's name again ("the full predictor set") |

## Consequences

Costs: repetition, and a lookup for each new term. Buys: the reader never has to ask whether two names are two
objects. S8 checks macros, citations and agents but not names, so a rewrite keeps the names only if its writer does.

## Detection

No check exists. By hand: grep live text for the left column of the guide's §5 table ("all features", "per bar",
"arm", "baseline", "walk-forward", "shipped", "test window") and for the clashes listed in addenda §B.1 and §B.3.

## Related patterns

V2, V9, V11, V16.

## Source

`writeup/style/voice.md` V3; `writeup/NOTATION_GUIDE_2026-10-15.md` (preamble, §5) and
`writeup/NOTATION_GUIDE_ADDENDA_2026-10-15.md` (§A.4, §B.1, §B.3); report §4.5 (inference: "keep one name per
estimator, using the existing NOTATION_GUIDE_2026-10-15.md as the term list"), §8.4 S0.
