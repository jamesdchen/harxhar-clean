# V10. Inanimate subjects do not evaluate

**Family:** sentences. **Stage:** drafting (S0); flagged outright at S4a (Tier C). **Basis:** D for the LLM profile
(Reinhart et al.; Jiang & Hyland); I for the flag.

**Severity:** review. A confirmed hit makes the drafter replace the evaluative verb with the number or finding it stood for.
**Checked by:** code + auditor (sentences slice). The code's grep (Detection) finds an inanimate subject with an evaluative verb; the auditor confirms each hit.

## Intent

Do not give an inanimate subject ("the results", "this analysis", "these findings") an evaluative verb (underscore,
highlight, showcase, reveal); state the number or the finding instead.

## Also known as

Evaluative verbs with inanimate subjects (S4a); the "The analysis reveals…" profile (§4.3.1).

## Motivation

Instruction-tuned models avoid both the agentless passive and "we" and favor inanimate or nominal subjects with active
verbs, as in "The analysis reveals…" or "This decomposition shows…" (§4.3.1, inference from Reinhart et al. and the
stance studies). The author, 28 Oct 2026 (22:50 PDT): "there's something else about AI prose. A certain cadence and
insistence on using active or passive voice in a specific way".

## Applicability

Evaluative verbs only. An exhibit with a reporting verb ("Table~\ref{tab:rh_ablation} reports the removals") is the form
the report expects in JFEc prose (§4.3.2, an inference still to be tested) and is not flagged. This pattern sets no
passive share and no "we" rate: per-section voice rules wait for the corpus bands (S0; §4.3.2). The notation guide's §6
asks for "We" and active voice; the report expects data and methods sections to keep agentless passives and leaves the
question open (§8.9 Q7). Not author-supplied text (V13) or text pulled from Overleaf (V15).

## Structure / Rewrite

The S7 operations are "replace the evaluative verb with the number" and "replace the inanimate agent with "we" or a
passive" (V12). `\rhX…` marks a macro the generator would have to emit.

| Instead of | Write |
|---|---|
| "The results underscore the role of the FOMC calendar." | "On statement days the FOMC calendar lowers the loss by `\rhX…Pct`\% (…)." |
| "This decomposition highlights the importance of the realized moments." | "Removing the realized moments raises the ordinary-day loss by `\rhX…Pct`\% (…)." |

## Consequences

Costs: the writer has to find the number the verb stood in for. Buys: the sentence states its finding, and a marker
measured in LLM text is gone.

## Detection

S4a, once built, flags these outright, not against a band. By hand:
`grep -nEi "(results|analysis|findings|evidence|this|these)( [a-z]+)? (underscore|highlight|showcase|reveal)" writeup/rehaul/*.tex`.

## Related patterns

V1, V4, V12, AP-V (underscore and showcase are hard-flag words).

## Source

Report §4.3.1, §4.3.5 (step 2: "inanimate subjects with evaluative verbs ("the results underscore")"), §8.4 S4a
("Evaluative verbs with inanimate subjects ... are flagged outright"); the author's words of 28 Oct 2026, transcribed
in `writeup/deliberation/2026-10-29c/Q-2026-10-29-25.evidence.md`, item 6.
