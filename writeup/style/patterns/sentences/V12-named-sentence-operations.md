# V12. Named sentence operations, not style targets

**Family:** sentences. **Stage:** edit stage only (S7), on locked layers. **Basis:** D against global targets
(Reinhart et al.'s exemplar result; Zhou et al.'s overshoot; Peters & Chin-Yee); T-genre (LAMP; Patchwork rule
citation); I for the operation list.

## Intent

Change a sentence's style only by one named operation aimed at a flagged sentence, re-measure, and stop once the
section is inside the corpus band.

## Also known as

The S7 register pass; the five-step correction of §4.3.5.

## Motivation

Global instructions are weak tools for grammar. Reinhart's models had the human exemplar and a "same style" instruction
and still diverged; epistemic markers jumped when requested, overshooting; accuracy prompts did not stop
overgeneralization (§4.3.5). §8.6 therefore bans "use more passives", "vary sentence length" and "be accurate about
scope" as instructions.

## Applicability

The edit stage, on locked layers only (V14), and only on sentences a stage (S2–S5) or a reviewer flagged. Also any
prompt a coordinator writes for a subagent: name the patterns or the operation, never a global style target. Never
inside author-supplied text (V13) or text pulled from Overleaf (V15). Never touch macros, citations, references or
comparison words (S7).

## Structure / Rewrite

The operations (S7): convert to agentless passive; replace an inanimate agent with "we" or a passive; merge with the
next sentence; split at the conjunction; delete the participial tail; delete the quantifier; replace an evaluative verb
with the number; cut an importance-assertion. One operation per flagged sentence, sent with its paragraph and rule id.

| Instead of | Write |
|---|---|
| "Make Section 8 read more like a finance journal and vary the sentence length." | "`08_feature_importance.tex:<line>`: delete the participial tail (AP-S2)." |
| a pass that rewrites until the section matches the corpus median | stop once the section is inside the p10–p90 band (§4.3.5, step 5) |

A section is flagged only when it sits outside the band by more than its own sampling error (§4.3.4).

## Consequences

Costs: one call per sentence, and no band-based edits until the corpus exists, so only rule-based flags (AP-V hard
flags, V5, V6 review flags, V10) drive edits now. Buys: each edit is auditable by rule id, S8 can check it, and the text
is not pulled toward the corpus median.

## Detection

S7 logs each edit with its rule id; S8 rejects one that fails its checks (V16). A reviewer treats a style edit with no
named operation or rule id as a violation.

## Related patterns

V10, V14, V16, AP-V, AP-S.

## Source

Report §4.3.4, §4.3.5, §8.4 S7, §8.6 ("No global style instruction ... Style edits are named sentence operations,
re-measured after each pass").
