# V8. Beat labels state a claim

**Family:** sentences. **Stage:** drafting (S0) of L1; advisory check C8 before the layer is served. **Basis:** P
(McCloskey; Minto).

**Severity:** review. A confirmed hit makes the drafter rewrite the label as the claim the beat's bullets support, distinct from its siblings' claims.
**Checked by:** auditor (sentences slice). Judgment only: is the label a claim or a topic, and does it overlap a sibling?

## Intent

Make each beat's label state the beat's claim, not its topic, and keep sibling beats from stating the same claim.

## Also known as

C8; "claims, not topics".

## Motivation

The beats are the paper's top-level claims. Under A2 the author reads L1 once and each corrected beat becomes a locked
claim (§8.1), so a topic label gives him nothing to lock. In the final layer the label is dropped and the beat's first
sentence carries its claim as the topic sentence (§8.1; C10). On 29 Oct the paper had both kinds side by side:
`\item \textbf{Small, and only in the boosted trees}` (`writeup/rehaul/08_feature_importance.tex:258`) states a claim;
`\item \textbf{In short.}` (`08_feature_importance.tex:374`) names a function and is also a summary opener (AP-S6).

## Applicability

Beat labels: `\item \textbf{…}`, or `\beat{<id>}{<label>}` once the macros of §8.1 exist. Not section titles, which
follow V9. Not labels the author supplied (V13). Existing labels change only through an edit or at the edit stage (V14).

## Structure / Rewrite

- The label is a short claim the beat's bullets support.
- Two beats in one section do not state the same claim.

| Instead of | Write |
|---|---|
| `\textbf{In short.}` | the claim the beat states, e.g. `\textbf{Two groups carry the gain.}`, the first sentence after it at `08_feature_importance.tex:374` |
| a label naming a function ("Evidence", "Robustness") | the finding that part establishes, in the beat's own words |

## Consequences

Costs: a label must change when its claim changes. Buys: the author can correct and lock a claim by its label, and the
final topic sentence exists before any prose is written.

## Detection

C8: an LLM judge, one round, advisory to the agent queue, run before L1 is served, once built. By hand: ask the judge's
question of each label (claim or topic? does it overlap a sibling?).

## Related patterns

V1, V9, AP-S (AP-S6, AP-S9).

## Source

Report §8.3 C8 ("Beat label is a claim, not a topic; siblings non-overlapping"; basis P: McCloskey, Minto), §8.1 (the
L1 row; the final form of `\beat`), A2.
