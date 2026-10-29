# V16. A rewrite preserves content

**Family:** scope. **Stage:** edit stage (S8), and any agent rewording meant to keep meaning. **Basis:** T-genre (Lee et
al. on copying); I.

## Intent

A style rewrite keeps the macros, citations, references, negations, modals, direction words and agents of the sentence
it replaces.

## Also known as

S8 post-checks; the content-preservation diff.

## Motivation

No study measures how often voice conversion changes meaning or agency in academic prose (§4.3.5), so the checks are
mechanical. The report's example: "Data are provided by OptionMetrics" must not become "We provide data". Under A2,
locked content is untouchable, and S8 together with the lock checks rejects any edit that changes it (§8.4).

## Applicability

Every style rewrite (S7; V12), and any agent rewording of agent text that should keep its meaning, such as bringing a
deliberation outcome's wording in line with V6. Not a content edit the author asked for, which changes content by
design (V13, V14). The operation's own target is exempt: "delete the quantifier" removes that quantifier.

## Structure / Rewrite

The checks (S8):

- the same multiset of macros, `\cite` keys and `\ref` labels;
- negations, modals and direction words unchanged unless the operation targeted them;
- no invented agent: a passive-to-active rewrite must not make "we" do what a data vendor or a cited paper did.

| Rewrite | Verdict |
|---|---|
| "Data are provided by OptionMetrics." → "We provide data." | rejected: invented agent |
| "lowers the loss by `\rhX…Pct`\% ($p = $ `\rhX…P`)" → "lowers the loss by `\rhX…Pct`\%" | rejected: a macro dropped |
| "does not lower the loss" → "barely lowers the loss" | rejected: negation changed |

## Consequences

Costs: some useful rewrites are rejected, and a change of meaning has to go through an edit or a deliberation. Buys: a
style pass cannot move a claim.

## Detection

S8 rejects the rewrite automatically on any failure, reruns S2–S5 (rewrites can add markers) and checks the longest
word n-gram shared with the corpus (V11); not built. By hand: compare the old and new sentence's lists of macros, cites
and refs, for example with
`grep -oE '\\(rh|cm|fqs|ladder)[A-Za-z]+|\\cite[tp]?\{[^}]*\}|\\ref\{[^}]*\}' | sort` on each; reread negations, modals
and direction words; ask who performs each verb.

## Related patterns

V2, V4, V11, V12, V13.

## Source

Report §4.3.5 (the meaning checks), §8.4 (bullet "Locked content is untouchable"; S8 row).
