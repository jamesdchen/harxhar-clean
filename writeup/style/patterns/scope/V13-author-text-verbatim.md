# V13. The author's text is copied verbatim

**Family:** scope. **Stage:** every stage and layer, the final build included. **Basis:** I (A4).

## Intent

Put text the author supplied in an edit message into the source exactly as he wrote it, and never restyle, scrub or
correct it afterwards.

## Also known as

A4; protected author text; `\auth{<decision ID>}{…}`; C26.

## Motivation

A4 (28 Oct 2026): asked whether Rule NT applies to text the author typed, the author answered "no". The report's
effect: Rule NT and the rest of the scrub never alter text the author supplied in an edit message; it is protected
verbatim, and the protection travels with it into lower layers and the final build. Under A2 no layer is served twice
(§8.1), so a correction a later pass undid would cost him a reading the design does not plan for.

## Applicability

Words the author wrote to appear in the paper: a replacement sentence, a title, a phrase in quotes. Not an instruction
the agent must put into words (for example 'prune "pooled"', commit ae0f526): the agent's wording is agent text, and
the catalog applies to it. Where it is unclear whether he meant his words to be used as written, use them as written
and queue the question with the deliberate skill (an ambiguous edit); do not ask him directly. Digits in his text are
exempt from C2 but must agree with the decision they state; a mismatch is a conflict for the deliberate queue (C26).

## Structure / Rewrite

- Insert the words byte for byte, his spelling and punctuation included.
- Record them with the date and his message words, as the source already does in dated comments
  (`% 2026-10-28 (author: ...)`), and name the edit in the commit message.
- Once `\auth{<decision ID>}{…}` exists (§8.2; Phase 2), wrap the words in it.

| Instead of | Write |
|---|---|
| the author's sentence with "always" cut under Rule NT | the author's sentence as he wrote it |
| the author's phrase rephrased to match V1 or V3 | the phrase as he wrote it; the surrounding agent text adapts to it |

## Consequences

Costs: his text may keep a word the scrub would remove or a digit outside a macro. Buys: his corrections survive every
later pass, and he never corrects his own words twice.

## Detection

C26 (gate; `check_locks.py` against `decisions.yaml`; not built): every `\auth` span is byte-identical to the text its
decision recorded. In the scrub, S1 removes `\auth` spans from what S2–S7 may touch, and S8 rejects any edit inside one.
Until then, before changing a sentence, check its dated `% (author ...)` comments and `git log -L` for the edit that
wrote it.

## Related patterns

V14, V15, V16, V5.

## Source

A4 in the report's "Author decisions" box; report §4.3.7 ("Scope (A4)"), §8.3 C26, §8.4 (third bullet), §8.6 ("Text
the author supplied is copied verbatim inside `\auth{ID}{…}`. No agent pass edits it").
