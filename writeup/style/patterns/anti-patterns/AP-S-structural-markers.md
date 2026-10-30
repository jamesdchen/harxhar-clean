# AP-S. Structural markers (scrub stage S3, Tier S)

**Family:** anti-patterns. **Stage:** avoided at drafting where a plain form exists (S0); counted by S3 at the edit
stage, on locked layers only. **Basis:** D for the grammar ratios (Reinhart et al.); catalog-level for the rest (§4.2;
§8.4 S3 row).

**Severity:** per row (see the Severity column).
**Checked by:** code + auditor (sentences slice). The code counts the rate rows per 1,000 words and lists the hits of the gate and review rows; the auditor confirms review rows and finds AP-S10, which no grep catches.

## How S3 flags

S3 counts constructions as rates per 1,000 words against the corpus p10–p90 band, per draft and per section; outliers
go to S7 with a named operation (V12). The exception is AP-S3 (V6), flagged one by one for review and never rewritten
automatically. Until the corpus exists (§8.9 Q4), there is no band to count against.

| Id | Marker | Flag | Severity | Fix | Source (grade) |
|---|---|---|---|---|---|
| AP-S1 | Em dashes (`---`, "—") | Rate; also a per-paper percentile in S5 | rate | Split at the conjunction (S7), or reviewer | Freeburg 2026 (preprint): em dashes persisted under "no markdown" instructions; Wikipedia, secondhand: among current models only Claude used them more than professional writers (§4.2). None in the 28 Oct main text (§4.1) |
| AP-S2 | Participial tails: ", highlighting …", ", ensuring …" | Rate | review | Delete the participial tail | Reinhart et al.: present-participial clauses at about 5.3 times the human rate (D; §4.2, §4.3.1). None on 28 Oct (§4.1) |
| AP-S3 | Clipped verdict tails after the fact (", none significantly") | Review flag, never auto-rewritten | review (at drafting, V6 is gate) | See V6 | The author, 29 Oct 2026 (V6) |
| AP-S4 | "Not X but Y", "It's not X, it's Y"; "rather than" | Rate | rate | No named S7 operation; reviewer | Antislop: up to 6.3 times the human rate in some models' creative writing (D, another genre; §4.2). Six "rather than" on 28 Oct (§4.1) |
| AP-S5 | Signposting: "it is worth noting", "it is important to note", "It should be noted that", "In other words" | Rate | rate | Cut an importance-assertion | Wikipedia's catalog (§4.2); Cochrane (P): "It should be noted that" is "particularly obnoxious", "In other words" is "a sign of trouble" (§4.5, §4.3.7) |
| AP-S6 | Summary openers outside the conclusion ("In short", "In summary", "Overall") | Rate | review | No named S7 operation; reviewer | S3 row; summary closers are catalogued, not measured (§4.2). On 29 Oct: `08_feature_importance.tex:374`, `\textbf{In short.}` |
| AP-S7 | "serves as", "stands as" | Rate | rate | No named S7 operation; reviewer | S3 row; catalog-level |
| AP-S8 | Triads (lists of three) per paragraph | Rate | rate | No named S7 operation; reviewer | Catalogued, not measured (§4.2) |
| AP-S9 | Bold inline headers | Present in L4 only; in L1–L3 bold beat labels are the design (V8) | gate in L4; none in L1–L3 | L4 strips them (§8.1) | "Inline-header vertical lists", a catalogued AI formatting sign (§4.2) |
| AP-S10 | Sentences that only assert importance; a paragraph-final "This matters because" | Not an S3 count; the S7 judge or a reviewer finds them | review | Cut an importance-assertion | Catalogued, not measured (§4.3.3); the slop taxonomy's information utility (§4.5) |

The Flag column says how S3 counts a marker at the edit stage; the Severity column says what a hit in text a drafter
returns does at the audit stage (`writeup/style/voice.md`, "Audit stage").

**Location.** Hits weigh more in the abstract and introduction, which a reader meets first, than elsewhere (location
weighting; `writeup/reports/Academic writing skills comparison.md` B4).

**Not flagged.**

- AP-S1 governs agent drafting: an agent avoids em dashes where a plain form exists (S0). It is not a scrub ban: at the
  edit stage S3 counts em dashes as a rate, and an em dash changes only in a section outside the band, through a named
  operation (V12).
- AP-S5: a section roadmap ("Section 3 describes the data") and an exhibit pointer ("Table 3 reports ...") are not
  signposting.
- AP-S8: a genuine three-item list, including a contributions paragraph that lists three contributions, is not a flag.

## Measured or catalogued, with no rule

- Nominalizations (about 2.1 times the human rate), phrasal coordination (about 1.9 times) and agentless passives (about
  half): S4a shows them through main-clause subject types and passive share. No drafting rule follows until
  per-section corpus bands exist (§4.2, §4.3.2, §8.9 Q7).
- Cadence devices (alternating long and short "punch" sentences, colon reveals, sentence-initial repetition) are
  catalogued, not measured (§4.3.3). S4a reports opening entropy and same-opener runs against the band; lag-1 length
  autocorrelation is dashboard only. V6 covers the clipped-tail case.
- Uniform sentence length ("low burstiness") is contested and confounded with length: rate, dashboard only, never a gate
  (§4.2, §4.3.4).

## Detection

```bash
grep -nE -- "---|—|, (highlighting|ensuring|underscoring|showcasing)|worth noting|important to note|should be noted|In other words|In short|In summary|serves as|stands as|rather than" writeup/rehaul/*.tex | grep -v ':\s*%'
```

## Related patterns

V6, V8, V12, V16, AP-V.

## Source

Report §4.1, §4.2, §4.3.3, §4.3.4, §4.5, §8.4 S3 and S4a rows.
