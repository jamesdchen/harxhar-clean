# V6. No punchy qualification after the fact

**Family:** claims. **Stage:** drafting (S0); found at S3 (Tier S) and sent to review. **Basis:** the author's rule
(29 Oct 2026); I for the detector.

## Intent

Do not end a finished clause with a clipped verdict or qualification: state it as a claim of its own, with its number
and test, or delete it.

## Also known as

Clause-final verdict tail; "none significantly"; the catalogued "punch" of §4.3.3.

## Motivation

Section 8 said the gradient-boosted trees' removals of the other groups change the ordinary-day loss "by at most
`\rhchOrdLoOtherTreeMaxAbs`\%, none significantly" (`writeup/rehaul/08_feature_importance.tex:224`). Under the numbers
before ablation addendum J that was false (LightGBM's sentiment removal had p = 0.035), and the tail drew no check. It
became true only when the patched rerun moved that p-value to 0.066 (claim check, 29 Oct 2026;
`writeup/deliberation/2026-10-29b`). The tail hides a universal ("none", "all", "not one") or a verdict with no number,
test or count behind it, so Rule NT's TRUE? step (V5) has nothing to check. The same tail is live at
`08_feature_importance.tex:261`, `A_results.tex:180` and `:447` and `A_closing.tex:216`, and a T0 deliberation outcome
prescribed one (Q-2026-10-29-15; `writeup/deliberation/2026-10-29c/Q-2026-10-29-25.evidence.md`, item 10).

## Applicability

Agent-written text, at drafting and at every rewrite, including wording taken from a deliberation outcome. A tail of a
few words after a comma, semicolon, colon or dash that restricts, overturns or grades the claim just made:

- ", none significantly" / ", none of them significant" / ", all significant"
- ", and none survive" / ", not one" / "; neither does X" / ", if at all"
- "-- and only barely" / ": none." / ", and it is not close"

Not author-supplied text (V13) or text pulled from Overleaf (V15). The live tails above wait for the edit stage (V14).

## Structure / Rewrite

If the qualification matters, state it as a claim of its own, inside the sentence's grammar, with its number and test.
If it does not matter, delete it. Back a kept universal with a macro (a count of significant tests against the total,
or the smallest p), as V4 and V5 require. `\rhX…` marks a macro the generator would have to emit.

| Instead of | Write |
|---|---|
| "... by at most `\rhchOrdLoOtherTreeMaxAbs`\%, none significantly." | "... by at most `\rhchOrdLoOtherTreeMaxAbs`\%. No removal is significant at the `\rhX…Level`\% level (smallest $p = $ `\rhX…PMin`)." |
| "... and `\rhabEnetFullVsAddRmFomcAllPct`\%, none significantly ($p \geq$ `\rhabLinFullVsAddRmFomcAllPMin`)" | "... and `\rhabEnetFullVsAddRmFomcAllPct`\%. The smallest of the three $p$-values is `\rhabLinFullVsAddRmFomcAllPMin`." |
| "The other groups add little, if at all." | "Each other group lowers the loss by at most `\rhX…MaxPct`\% (Table~\ref{…})." |
| "Three classes beat HAR -- and only barely." | "`\rhX…NBeat` classes beat HAR, by `\rhX…Min`\% to `\rhX…Max`\% (DM $p$ from `\rhX…PMin` to `\rhX…PMax`)." |

## Consequences

Costs: the qualification takes its own sentence, and often a macro the generator does not yet emit. Buys: the verdict
becomes checkable by the predicate check (V4) and by TRUE? (V5). The main clause no longer sets up a reading for a tail
to overturn or seal, which is rhetoric rather than reporting.

## Detection

S3 (§8.4): flag a clause-final tail of at most five words, after `,`, `;`, `:` or a dash, holding a universal or
negative quantifier (none, no, not, all, every, only, neither, nor) or a verdict word (significant(ly), barely, at all,
close); the flag goes to review, never to an automatic rewrite. Until `scrub.py` exists, a narrower grep (without all,
every, only, no and not, which match too often; it misses tails split across lines):

```bash
grep -nE "(,|;|:| --) *([A-Za-z']+ ){0,3}(none|neither|nor|not one|significant|significantly|barely|if at all|at all)( [A-Za-z']+){0,3} *(\(|[.;:)]|$)" writeup/rehaul/*.tex | grep -v ':\s*%'
```

## Related patterns

V4, V5, V7, V14, AP-S.

## Source

The author, 29 Oct 2026: "add a rule to the style guide about "none significantly" or any punchy qualification after
the fact". `writeup/style/voice.md` V6 as first written (commit 2f61ba3); report §4.3.3 (the "punch", catalogued, not
measured), §8.4 S3 ("clipped verdict tails after the fact ... flagged for review, never auto-rewritten").
