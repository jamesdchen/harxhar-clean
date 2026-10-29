# V15. Text typed on Overleaf is not restyled

**Family:** scope. **Stage:** every stage. **Basis:** a provisional deliberation outcome (Q-2026-10-29-11: T2,
confidence 0.65, importance HIGH, vetoable), extending A4 (I).

## Intent

Do not scrub, restyle or cut under Rule NT any text a person typed on Overleaf (a collaborator, or the author) that came
back through the pull.

## Also known as

A4 for collaborators (§8.9 Q11).

## Motivation

A5 (28 Oct 2026): "but other people will collaborate on the overleaf. so edits do still need to have the option of
flowing back from overleaf as well". The 29 Oct round answered whether A4's protection covers that text: "All human
text pulled from Overleaf is exempt from the scrub and Rule NT; no byte-lock" (`writeup/deliberation/log.jsonl`,
Q-2026-10-29-11). For the author: "Text anyone types on Overleaf is never scrubbed or cut by Rule NT; your chat edits
and decisions can still change it."

## Applicability

Text that entered the sources through `overleaf_git.sh pull` and `unflatten.py`. It differs from V13 in three ways:
there is no byte-lock; the author's chat edits and decisions can still change it; and a collaborator's edit locks only
when the author adopts it (§8.9 Q10, the default kept). The outcome is provisional: if the author vetoes it, this
pattern changes with it.

## Structure / Rewrite

| Instead of | Write |
|---|---|
| a scrub pass deleting "clearly" from a sentence a coauthor typed on Overleaf | leave the sentence; the scrub skips it |
| restyling a pulled sentence while applying an unrelated edit next to it | apply the edit only |
| (the author's chat edit rewrites that pulled sentence) | apply his edit: his edits govern (V13 for any words he supplied) |

## Consequences

Costs: pulled text may keep markers the scrub would remove, and the register can differ between paragraphs. Buys: no
agent pass rewrites what a person typed; only a person's edit or a decision changes it.

## Detection

The provenance sidecar that `unflatten.py` is to write at each pull (file, sentence hash, pull SHA, editor) is pending
(Q-2026-10-29-11's action). Until then: the pull's `writeup/overleaf_sync/REPORT.md` and the commit that applied it;
`git blame` as an audit cross-check.

## Related patterns

V13, V14, V16.

## Source

A5 in the report's "Author decisions" box; `writeup/deliberation/log.jsonl` (Q-2026-10-29-11; adjudication
`writeup/deliberation/2026-10-29/Q-2026-10-29-11.adjudication.json`); report §8.9 Q10 and Q11.
