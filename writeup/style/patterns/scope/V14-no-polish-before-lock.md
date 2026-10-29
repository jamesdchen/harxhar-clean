# V14. No polish before lock

**Family:** scope. **Stage:** every agent edit; gate C25. **Basis:** I (A2).

## Intent

Draft plain and leave the style of existing text alone until the author has read and locked its layer; the scrub,
register passes and stylistic rewrites run only on locked layers.

## Also known as

A2's order and polish; C25.

## Motivation

A2 (28 Oct 2026): "thus, the paper needs to start from a simple unedited state and, through my reading and correction
of the simple version, end in a finished edited state." So "the scrub is editing, so under A2 it runs last" (§8.4). On
29 Oct the coordinator declined to rewrite the live V6 tails for this reason ("No stylistic sweep of the existing text
before lock": `writeup/deliberation/2026-10-29c/Q-2026-10-29-25.author_record.md`); they remain at
`08_feature_importance.tex:224` and `:261` and in the appendix.

## Applicability

- Allowed before lock: drafting new text under the drafting patterns (S0 "is a drafting rule, not a rewrite, so it
  applies before lock"), and applying an author's edit or a deliberation outcome, with the words you write following
  the catalog.
- Not allowed before lock: a style sweep of existing text, a scrub pass, a register rewrite (V12), and restyling
  sentences next to the one an edit targets.
- A tail or universal that makes a false claim is a content error, not style: it goes to the deliberate queue as a
  flipped claim, whatever the lock state.
- Lock state lives in `writeup/layer_state.json` (§8.2). The file does not exist (29 Oct 2026), so no layer is
  recorded as locked and edit-stage passes do not run.

## Structure / Rewrite

| Instead of | Write |
|---|---|
| applying the author's edit to one bullet and also rewriting the next bullet's "none significantly" | apply the edit only; leave the tail for the edit stage |
| a "light polish" of a section while regenerating its numbers | regenerate; the prose of unflipped claims stays as it was (§8.6) |

## Consequences

Costs: known style flaws stay in the text the author reads. Buys: he reads content first, no polish is spent on text he
then changes, and no polish moves content he has locked.

## Detection

C25 (gate; `layer_state.json`, `check_locks.py`, S8; not built). A reviewer reads the diff: on an unlocked layer, a
changed sentence outside the edit's target, with no content reason, is a violation.

## Related patterns

V1, V12, V13, V16.

## Source

A2 in the report's "Author decisions" box; report §8.1 (the sequence), §8.3 C25, §8.4 (first paragraph), §8.6 ("A
layer is generated only from a locked parent, and drafted plain. No scrub, register pass or stylistic rewrite runs
before lock."; "Regenerate only the beats whose governing decision changed"), §8.7.
