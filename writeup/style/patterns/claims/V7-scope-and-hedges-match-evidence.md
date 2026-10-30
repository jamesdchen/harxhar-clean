# V7. Scope and hedges match the evidence

**Family:** claims. **Stage:** drafting (S0); advisory check C19 on L0 and L2–L4. **Basis:** D for the failure mode
(Peters & Chin-Yee); I for the check.

**Severity:** review. A confirmed hit makes the drafter add the sample, period or test the finding rests on, or move a hedge off a computed fact onto the interpretation.
**Checked by:** auditor (claims slice). Judgment until C19's qualifier diff exists: the auditor compares each summary claim with its results sentence and checks what each hedge hedges.

## Intent

State each finding with the sample, period and test it rests on, and put modal hedges on mechanism and interpretation
only, never on a computed fact.

## Also known as

Hedge calibration; "hedging should run in both directions" (§4.3.6); C19.

## Motivation

Across 10 LLMs, Claude 3.7 Sonnet among them, summaries were nearly five times as likely as human ones to contain broad
generalizations, because they omit "details that limit the scope of research conclusions", and newer models did worse
(§4.3.6). The 29 Oct round caught one in this paper: Section 8 said the trees' attribution "ranks the parts the wrong
way round", while the reversal holds for the FOMC calendar only. The bullet was narrowed to "The trees' attribution
understates the FOMC calendar" (Q-2026-10-29-26, commit 136794b).

## Applicability

Every finding an agent writes, above all in the abstract, introduction and conclusion, where sample qualifiers that
the results carry get dropped (C19 compares the sections). Modal hedges include may, might, could, suggests, appears and
seems. Not author-supplied text (V13).

## Structure / Rewrite

- Scope a finding to its sample, period, frequency and test.
- Hedge a mechanism or an interpretation. State a computed fact plainly.

| Instead of | Write |
|---|---|
| "Alternative data do not improve intraday volatility forecasts." | "On every bar of the evaluation span, adding the other series to HAR, the realized moments and the FOMC calendar does not significantly change the linear models' loss ($p \geq$ `\rhabLinFullVsAddRmFomcAllPMin`)." |
| "The loss appears to be lower with the full set." (the table shows that it is) | "The full set lowers the loss by `\rhX…Pct`\% (…)." |
| "The calendar captures the announcement effect." (a mechanism, stated as fact) | "The statement-day gain suggests that the calendar captures the announcement effect." |

`\rhX…` marks a macro the generator would have to emit.

## Consequences

Costs: longer claims in the abstract and conclusion, with qualifiers repeated from the results. Buys: the summary
sections claim no more than the results show, and no computed fact is understated by a hedge.

## Detection

C19 (advisory, agent queue), once built: a qualifier diff across sections and a hedge lexicon. By hand: for each claim
in the abstract, introduction and conclusion, find its sentence in the results and compare sample, period and test;
grep the hedge words above and check what each one hedges.

## Related patterns

V4, V5, V6, V13.

## Source

Report §4.3.6 (inference: "Findings are scoped to sample, period, frequency and test. Modal hedges ("may", "suggests")
are reserved for mechanism and interpretation, and a hedged statement of a computed fact ... is flagged too"), §4.5
("calibrate hedges to the test statistics"), §8.3 C19.
