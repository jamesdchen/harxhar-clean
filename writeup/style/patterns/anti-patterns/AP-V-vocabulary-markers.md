# AP-V. Vocabulary markers (scrub stage S2, Tier V)

**Family:** anti-patterns. **Stage:** avoided at drafting (S0); flagged by S2 and fixed by S7 at the edit stage, on
locked layers only. **Basis:** D for the marker frequencies (Kobak et al.; Liang et al.; Reinhart et al.; Wikipedia's
catalog); P for AP-V3.

**Severity:** per row (see the Severity column).
**Checked by:** code + auditor (sentences slice). The code matches every row's words in the detexed text, minus the field whitelist; the auditor confirms review rows and reads for adjectives on one's own results, which no list catches.

## What S2 flags

S2 matches words in the detexed text (S1). The report sorts them into hard flags and density flags, and exempts a
field whitelist.

**Field whitelist: never flagged, at any severity.** leverage (the leverage effect), robust (robust standard errors),
significant (statistically significant), causal, empirical. All three "leverage" hits in the 28 Oct main text were the
HAR literature's leverage effect, so a generic list would flag correct domain vocabulary (§4.1).

| Id | Words | Flag | Severity | Source |
|---|---|---|---|---|
| AP-V1 | delve, underscore, showcase, intricate, meticulous, pivotal, realm, tapestry, testament, multifaceted, garner (the report's list is open-ended) | **Hard flag**: fixed by S7 and logged with rule id and source | gate (field whitelist excepted) | Kobak et al.: excess ratios of 28.0 for "delves", 13.8 for "underscores" and 10.7 for "showcasing" in 2024 PubMed abstracts; Liang et al.: "meticulous" and "intricate" in ICLR 2024 reviews; Wikipedia's mid-2026 list includes "multifaceted" (§4.1) |
| AP-V2 | additionally, notably, crucial, comprehensive, insights, enhance, highlighting | **Density flag** only: a rate per 1,000 words against the corpus band, not a hit-by-hit fix | rate | §8.4 S2 row |
| AP-V3 | "utilize", "huge", exclamation marks, adjectives on one's own results | Not in S2's lists; prescriptive | review | Cochrane, via §4.5 ("use", not "utilize"; no adjectives on one's own results); notation guide §6 ("No exclamation marks, no "huge", no "clearly"") |

The Flag column says how S2 treats a word at the edit stage; the Severity column says what a hit in text a drafter
returns does at the audit stage (`writeup/style/voice.md`, "Audit stage").

**Location.** Hits weigh more in the abstract and introduction, which a reader meets first, than elsewhere (location
weighting; `writeup/reports/Academic writing skills comparison.md` B4).

"Clearly" and the other boosters are governed by Rule NT (V5); "It should be noted that" is signposting (AP-S5).

## At drafting (S0)

Do not write an AP-V1 or AP-V3 word: S0 asks for "nothing the later tiers would remove (rare markers, ...)". AP-V2 words
are not banned at drafting; they matter only as a rate.

## At the edit stage (S2, S7)

Only on locked layers (V14), and never inside author-supplied text or text pulled from Overleaf (V13, V15). S7 replaces
each hard flag through one named operation (V12), usually the plain verb or the number (V10). S8 reruns S2 after every
rewrite, since rewrites can add markers.

## Why the list is not fixed

The fingerprint moves: "delve" fell after it was publicized in spring 2024, and Wikipedia's catalog dates its lists by
model era (§4.1). Words are scored as rates against a field baseline, not general English, and the over-use list is to
be regenerated from S5's corpus z-scores (`style/rules/corpus_overuse.yml`, once built).

## State of the paper

On 28 Oct the main text held none of the rare markers (§4.1). While the main text holds none, S2's value is stopping
regressions as agents redraft.

## Detection

```bash
grep -nEi "delv|underscor|showcas|intricate|meticulous|pivotal|realm|tapestr|testament|multifaceted|garner|utiliz" writeup/rehaul/*.tex | grep -v ':\s*%'
```

## Related patterns

V5, V10, V12, AP-S.

## Source

Report §4.1, §4.5, §8.4 S0 and S2 rows; `writeup/NOTATION_GUIDE_2026-10-15.md` §6.
