# Claims ledger — alpha-manifestation study, as of 2026-08-06

The graded index for writing up. Every number cites its section in
`alpha_manifestation_findings_2026-08-04.md` (§) unless noted. Grading: **Tier 1** survived the
§24 Romano–Wolf FWER sweep; **Tier 2** passed a single pre-registered gate and awaits the AM-10
family sweep; **Tier 3** carried in the model with no evidence claim; **Casualty** tested and
killed at a pre-registered gate (the meta-law's ledger — these are results, not failures to
report).

## The deliverable

**FINAL (2026-08-06, §34.11):** 699 columns (§22's 679 + the 20-column transmission block,
confirmed per-bar +8.54) + the conditional smear (means + leverage + probe, §34.8 FWER-graded):
**QLIKE 0.12331** (679-const baseline 0.12579 on the same span; smear increment on the final
model DM +3.95). ≈ −7% vs HAR, from −5.6% at session start. First moment dense-and-fast;
second moment sparse-and-named. Original §22 record: 679 @ per-bar, QLIKE 0.12526 (§22 span),
R² 0.612; 250-day window (§25: longer LOSES); edge GROWS by era: 65 → 79 → 82 ×1e-4 (§24).

## Tier 1 — proven (Romano–Wolf survivors, §24)

| Claim | Number | § |
|---|---|---|
| Refit cadence (monthly → per-bar), linear-channel-driven | rw p ≈ 0.000; §21.1 attribution | §18.1, §21.1, §24 |
| Products at at-least-daily refit | rw p = 0.001 (+3.56 DM) | §18.1, §24 |

## Tier 1½ — survived the AM-10 family sweep (§36; FWER-controlled within the new family)

| Claim | Number | rw-adj p | § |
|---|---|---|---|
| Open-decision edge under dedicated-horizon training | +49.3e-4, t +3.32 | 0.005 | §29.2, §36 |
| Per-horizon α-law at H = 16 | +31.7e-4, t +2.27 | 0.034 | §29.3, §36 |
| Per-bar cadence at H = 8 | +31.2e-4, t +3.49 | 0.005 | §29 verdicts, §36 |
| Conditional smear (means-only) | +11.7e-4, t +3.86 | 0.005 | §34.2, §36 |
| Transmission columns at H = 8 | +1.2e-4, t +2.58 | 0.026 | §30.2–30.3, §36 |

**Died in the sweep:** the §28.2 phase pair (rw-adj p 0.105) — dropped from the stack.

## Second moment (smear family, §34.8 sweep — FWER-graded)

| Claim | dm_t | rw-adj p | Status |
|---|---|---|---|
| Leverage asymmetry (vs means) | +4.03 | 0.002 | **production** |
| Probe-5 nonlinear terms (vs means+lev) | +2.41 | 0.042 | **production** |
| Regime labels | +1.93 | 0.072 | dead |
| Event decay | +1.04 | 0.268 | dead |
| Dense-679 | −1.87 | 0.956 | dead (control) |
| Transmission-smear | −1.99 | 0.965 | dead (control) |

Production smear: **means + leverage + probe (10 columns)**. Plus §35c/§36-era additions to the
first moment: transmission block CONFIRMED per-bar (+8.54) — final model = 699 cols per-bar.

## Tier 2 — single-gate passes, post-sweep (provisional until a follow-up accounting)

| Claim | Number | Caveat | § |
|---|---|---|---|
| Daily-cycle phase pair as columns (h = 1 bar) | DM +2.09 (gate 2.0) | footnote-sized; 2020+ +1.38 | §28.2 |
| Per-bar cadence at H = 8 | DM +3.48 (2020+ +2.43) | channel attribution pending (§29.5) | §29 verdicts |
| Open-decision edge under dedicated-horizon training | 10:00-slice DM +3.24 (was −0.18 mixed-horizon) | pooled H=13 only +0.40 — edge concentrates at the open | §29.2 |
| α-law at H = 16 (α = 3e3·H) | DM +2.26 (0.07034 → 0.06717) | H=8 +1.95 just under; 2020+ weaker | §29.3 |
| Lagged-transmission columns at H = 8 | DM +2.58 | **raw-lag control pending (§30.3)**; 2020+ +1.58 | §30.2 |

## Tier 3 — carried, no claim

| Block | Evidence | § |
|---|---|---|
| 36 cross-section ratios | rw p = 0.569 (failed FWER) | §20, §24 |
| Announcement organ (28 cols) | DM +0.99 (gate fail; rich base proxies it) | §29.4 |
| Smear variant | rw p = 0.181 (claim dead; baseline Duan smearing is machinery, not claim) | §16, §24 |

## The casualty ledger (the meta-law: nothing cleverer than uniform-generous-shrunk-fast)

1. Sparse/selected features (multiple ladders) — §§4–7
2. Trees, pure seat — §19 / msweep
3. Trees, residual seat (+0.0096 WORSE) — msweep
4. Graph-diffusion features (Laplacian heat) — §12
5. Intensity-modulation gating — §13
6. Spectral hard/soft thresholds on the product pool — §19, pc_quadratics
7. WLS / vol-regime reweighting — §19.5
8. Concave zero-free-parameter response — §19.2
9. Dial drivers / timing (10 drivers, all changes-test null) — §18.3–18.4
10. Phase-conditioned dial (amplitude confound; profile corr +0.96 looked spectacular) — §28.1(b)
11. Flow-predicted products, daily lag (map anticorrelated −0.18) — §27 C6
12. SyncRank ladder + SPONGE blocks (no ladder, no blocks to find) — §27 C2–C3
13. Intraday flow-state features {cos φ, sin φ, log r} at H = 8 (DM −0.72) — §30.1
14. **The re-draw itself**: refreshed-2022 product selection LOSES to frozen-2005 on the 0DTE
    era (−2.79 / −2.41); rot detector demoted to alarm-only — §32

## Structural facts (not tradable, load-bearing)

- Dense-but-weak at every level; mid-spectrum directions gauge-degenerate (dependence-adjusted
  MP edge keeps 5 of 106) → the product subspace's basis is unpinnable, which explains both
  "representatives don't matter" and the re-draw failure — §27 C5, §32.
- Three replicating geometric objects on the frozen 2005 frame: coupling map (+0.62), daily flow
  (+0.79, 92% curl, 35-day rotor, phase forecastable ~1 month), intraday flow (+0.992, 96.5%
  curl, no clock, position coherent 0.89 a session ahead; survives the MA-arithmetic kill test
  at 6%) — §§26, 27, 28, 30.
- **Amplitude forecastable (R² 0.485 next-day), gain not (R² ≈ 0.02): the edge loads on
  amplitude innovations** — §31.
- Horizon curve: composite-over-backbone +5.42 (2h) / +2.77 (4h) / +1.23 (8h); the dense edge is
  fast; at 13 bars it concentrates at the open under dedicated training — §29, §29.2.
- Breaks: 2013 (taper), 2017 (pre-Volmageddon), 2024 (0DTE) via map CPD; COVID did NOT break the
  map — §26, §27 C4.

## Instruments (operational, not alpha)

- Rot detector (`interaction_map_health`, warn < 0.20 / fail < 0.00): ALARM-ONLY after §32 —
  detect, de-risk, investigate; do NOT re-select.
- Intraday flow monitor (+0.99 baseline): one-bar-deep microstructure alarm — §30.
- Era table: edge-growth tracking — §24.

## Trading translation (intraday straddles; scope H < 48 bars)

- Open-of-day 0DTE: has model edge under dedicated-horizon training (+3.24) — §29.2 amendment.
- Midday-entry hold-to-close: remaining-session horizon in the model's power zone; one spread
  crossing — §29.1.
- Veto overlay on the short-VRP harvest (implied/realized ≈ 1.37; prior friction-net t = +6.5).
- VRP measurement: 3-tier spec (`om_0dte_atopen_export_spec.md`); Tier-1 consumer built
  (`analysis/vrp_eod.py`), blocked on chain parquets; overnight decomposition cached.
- Sizing overlay: trailing amplitude concentration (§28.1(b) legitimate use).

## Pending / open

- In flight: §29.5 cadence channel decomposition; §33 P1 kNN 3-view head-to-head; §34 regime
  atlas; §30.3 raw-lag control. Then AM-10 multiplicity sweep.
- Data-blocked: AM-01 panel refresh 2024-03 → present (the biggest item); AM-02 chain parquets;
  AM-08 second market; AM-11 minute bars. Full backlog: `future_work.md` AM-01…AM-12.
