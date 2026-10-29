# Voice guide for drafting the paper

The drafting guide (stage S0) of `writeup/reports/AI layered paper writing system.md` (§8.2, §8.4). Every agent that writes or
rewrites paper text, beats, bullets or table notes follows it. It governs agent-written text only: text the author supplied
in an edit message is protected verbatim and never changed to fit these rules (A4, §4.3.7 "Scope").

Each rule gives its source. Rules V1-V5 restate what the report already specifies for S0; later rules are the author's.

## V1. Plain declarative sentences

Draft the simple version plainly: one claim a sentence, subject and verb early, no polish before the layer is locked (A2,
§8.4 S0).

## V2. Numbers come from macros

Every number in the text is a generated macro (`writeup/generated/*.tex`); none is typed (§3, §8.4 S0).

## V3. One name per estimator

Use the notation guide's single name for each estimator, set and test (`writeup/NOTATION_GUIDE_2026-10-15.md` and its addenda).

## V4. Verdict and magnitude words come from predicates

Comparison and verdict words ("lower", "significant", "concentrated", "redundant") and magnitude words ("substantially",
"modestly") appear only where a generated predicate or threshold backs them (§3.3, §4.3.7).

## V5. Rule NT: NEEDED? first, then TRUE?

Every universal (all, every, each, any, no, none, always, never), completeness adverb, superlative and booster is kept only
if it is needed, and then only if a computed fact makes it true. The full rule and its detection are in §4.3.7.

## V6. No punchy qualification after the fact

**Source.** Author, 29 Oct 2026: "add a rule to the style guide about "none significantly" or any punchy qualification
after the fact".

**Rule.** Do not append a clipped verdict or qualification to the end of a finished clause. A tail of a few words after a
comma, semicolon, dash or colon that restricts, overturns or grades the claim just made is banned, for example:

- ", none significantly" / ", none of them significant" / ", all significant"
- ", and none survive" / ", not one" / "; neither does X" / ", if at all"
- "-- and only barely" / ": none." / ", and it is not close"

If the qualification matters, state it as a claim of its own, inside the sentence's grammar, with its number and test.
If it does not matter, delete it.

**Why.**

1. It is a cadence device (the catalogued "punch" of §4.3.3). The main clause sets up a reading and the tail overturns or
   seals it, which is rhetoric rather than reporting.
2. It hides a universal claim from checking. The tail carries a universal ("none", "all", "not one") or a verdict with no
   number, test or count behind it, so Rule NT's TRUE? step (V5) has nothing to check. This paper's own case: Section 8
   said the gradient-boosted trees' removals of the other groups change the ordinary-day loss "by at most
   \rhchOrdLoOtherTreeMaxAbs\%, none significantly". Under the numbers before ablation addendum J that was false (LightGBM's
   sentiment removal had p = 0.035), and the tail drew no check. It became true only when the patched rerun moved that
   p-value to 0.066 (claim check, 29 Oct 2026; `writeup/deliberation/2026-10-29b`).

**Rewrite** (illustrative numbers).

| Instead of | Write |
|---|---|
| "... changes the loss by at most 0.5%, none significantly." | "... changes the loss by at most 0.5%. No removal is significant at 5% (smallest p = 0.066)." |
| "The other groups add little, if at all." | "Each other group lowers the loss by at most 0.3% (Table 4)." |
| "Three classes beat HAR -- and only barely." | "Three classes beat HAR, by 0.4% to 0.9% (DM p from 0.03 to 0.05)." |

Back the kept universal with a macro (a count of significant tests against the total, or the smallest p), as V4 and V5
require.

**Detection** (stage S3, structure tier, §8.4). Flag a clause-final tail of at most five words, after `,`, `;`, `:` or a
dash, that contains a universal or negative quantifier (none, no, not, all, every, only, neither, nor) or a verdict word
(significant(ly), barely, at all, close). A flag goes to review; it is not rewritten automatically.
