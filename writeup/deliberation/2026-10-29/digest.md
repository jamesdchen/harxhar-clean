# Digest, 29 October 2026

Five solvers and one adjudicator worked through the fresh version's conflicts (C1-C7), the starting state (Q8) and the
Overleaf process questions (Q10-Q13). One question needs you. Everything else is applied, provisionally where marked.
Full record: `adjudication.md` (same folder).

## (a) Question for you

**Q8. What do you read first?**
- **Problem.** The fresh version you asked for (89c359e) breaks six of your later instructions:
  - it splits Sections 6 and 7;
  - it uses "pooled" 20 times;
  - its titles are prose;
  - it calls the trees' remaining gain "dense";
  - it has no Related work and no Maisch comparison;
  - it states "no linear gain" without scope.
- **Recommendation.**
  - First, a one-page top layer built from the fresh version's claim sentences, with those six fixed and the items
    below applied.
  - Once you lock it, the bullet layer is the current bullets brought in line with it, so your ~970 dated edits
    survive.
- **Alternative.** Regenerate the bullets from the locked top layer and re-apply your edits. That is more work and may
  lose some edits.
- **Default if no reply:** the recommendation.
- **Answer:** `Q8 keep` or `Q8 regenerate`.

## (b) Applied provisionally unless you veto (one line: "veto C1", etc.)

- **C1.** The research question asks two things: does the full set beat OLS-HAR on every bar, and what do the 36
  series beyond HAR, realized moments and FOMC add. Results keep their order: gain plus MSE caveat, then the two
  channels, then the 36 series. Three numbered contributions.
- **C1b.** "Alternative data" keeps its narrow meaning (StockTwits, vega demand). The 36 series are "the other
  exogenous series", as in your 28 Oct edit.
- **C2 layout.** Table 1 gets two columns, % and DM against OLS-HAR, plus a note that the linear HAR rows equal the
  OLS.
- **C4.** When the every-bar ladder rerun lands, the ladder figure shows every bar only, and its 16:00 panel moves to
  the closing-interval appendix.
- **C5.** The attribution-vs-ablation reconciliation table you asked for on 28 Oct goes in 6.3.5: one row per group,
  no new compute.
- **X8.** Trading strategies signed by the HAR + RM + FOMC forecasts. Local, from existing forecasts, appendix only,
  rule file first. It answers whether the 36 series add trading value. Runs after this digest unless vetoed.
- **Q10.** Coauthors' Overleaf edits are applied and recorded with the editor's name, and only you lock. An edit that
  breaks a lock shows in the draft PDF as a held note until you decide; the final build fails while any remain. Your
  own Overleaf account is not a locking channel.
- **Q11.** Which sentences came from Overleaf is tracked in a sidecar file written at each pull.
- **Q12.** From a coauthor's first Overleaf comment on, open comments are listed (read-only) in the digest. Agents
  never accept, reject or resolve anything there without your word.

## (c) Settled (T1: evidence decided; T0: mechanical)

- **C2 (T1).** Every main-text forecast is tested against OLS-HAR on every bar, plus full vs own HAR set (D2). The
  16:00 columns go to the appendix. A Giacomini-White sentence covers nested comparisons.
- **C3 (T1).** "No linear gain beyond HAR + RM + FOMC" compares each model with its own base. It is scoped to every
  bar and ordinary days, because the lasso gains on statement days.
- **C3, cont.** One sentence discloses that OLS on HAR + RM + FOMC is as accurate as the penalized full sets (-9.2%).
  The main-text factor test uses your penalized version: k ≤ 3 recovers "little", not "none".
- **C4 (T0).** The ladder figure stays in the appendix (D16 and D20 beat the budget row).
- **C5 (T1).** SHAP shares get a D18 grouping, labelled post hoc: the six groups hold 20-23% of the trees' attribution
  but give 4-11% of their gain. The registered split stays in the appendix.
- **C6 (T1).** The application is unchanged. One sentence, with no numbers, points to the 16:00 accuracy in the
  appendix.
- **C7 (T0).** The closing-interval models stay in the appendix (D16 came after D3).
- **X1 (T1).** The abstract's "beats augmented HAR models" is too broad: OLS on HAR + RM + FOMC beats the ridge and
  elastic-net full sets. Drop or scope the clause, and add that OLS row to the ablation table.
- **X2 (T0).** "Under squared error the comparison reverses" becomes "is higher in all six classes, in none
  significantly".
- **X3 (T1), X4 (T0).** The vs-OLS tests get a table (via C2). A generator is added for the 28 Oct penalized tests, which no macro
  reads yet.
- **X5 (T0).** Two sync fixes:
  - after a coauthor edits `main2.tex`, pull now records what it took in, so push no longer refuses forever (that
    refusal also froze the review layers);
  - `map.json` is written only once a push has landed.
- **X6 (T0).** Report §8.5 corrected: Overleaf does record who edited, as the last editor before each pull.
- **Q11 (T1).** Text typed on Overleaf by anyone is never scrubbed or cut. Your chat edits and decisions can still
  change it.
- **Q12 (T1).** Git carries no comments, and pending suggestions arrive as accepted text, handled like edits. A
  read-only check of the track-changes setting comes next; you are asked only if it is on for everyone.
- **Q13 (T1).** Agents keep writing through git. No claudeleaf, no bot account.

## (d) CLAUDE.md change that needs your OK

Your Overleaf rule says: "If push refuses (Overleaf edits not pulled back), run `pull --dry-run`, report, and ask." It
predates A5/A6 and would turn every coauthor edit into a question.

**Proposed:**

> If push refuses (Overleaf edits not pulled back): pull, apply the edits under the A5 rules (applied as text and
> recorded; any edit that breaks a lock is held as a visible note), push, and list them in the digest; ask only about
> T3 items.

It takes effect only after the X5 deadlock fix. Until you say "OK CLAUDE.md", the current rule stands.
