# 04 — The remaining-session reading at every entry clock

Read-only study; nothing wired into a notebook. Scripts, log and CSVs:
`scratchpad/remaining_session/{study.py, study.log, per_clock_htc.csv, costs_by_clock.csv}`.
Block-diagonal ridge forecast throughout.

## Idea

The diagnosis (00) found that the intraday per-bar implied "slice"
$\mathrm{IV}^2_{\mathrm{hr}}\,h\,w$ is an allocation, not a price, and that the
mispricing the sign rule exploits lives in the remaining session, rich only at
the close. The paper's to-close reading compares like with like at every clock:
the forecast of the variance remaining to the close against the market's price
of that same quantity, with the package held to the close and cash-settled. If
the intraday clocks have an edge, this is where it should appear.

## Exact construction

At bar $t$ with $h_{\mathrm{rem}}$ hours to 16:00 and $w_t$ the causal per-clock
share of the remaining realized variance (expanding per-clock mean of the bar's
realized variance over prior days, minimum 63, lagged one day — the intraday
notebook's own share):

$$F_{\mathrm{rem},t}=\frac{\widehat{RV}_t}{w_t},\qquad
\mathrm{IV}_{\mathrm{rem},t}=\mathrm{IV}^2_{\mathrm{hr},t}\,h_{\mathrm{rem}},\qquad
s_{\mathrm{rem},t}=F_{\mathrm{rem},t}-\mathrm{IV}_{\mathrm{rem},t}.$$

Position: the nearest-OTM package picked at $t$, entered at the midpoint, held
to the close, exit = intrinsic at the official close; return per premium
$R^{\mathrm{htc}}_t=\mathrm{exit}/P_t-1$. Rules: sign($s_{\mathrm{rem}}$) and
always short. Frame: the days on which $w_t$ exists (803 days, 2020-04 to
2024-04; the 63-day warm-up per clock removes the first quarter of 2020, which
is why the 15:30 row scores 1.89 here and 1.63 on the RV–IV notebook's 871 days).

**Two facts established before any result was read.**

1. $\mathrm{IV}_{\mathrm{rem}}$ is a price: pricing the package with
   Black–Scholes–Merton over the remaining window at the vendor's hourly
   volatility reproduces the quoted midpoint at every clock (median ratio 1.0000;
   5th–95th percentile within ±0.06% at 10:00 and ±0.6% at 15:30).
2. **The remaining-session signal has exactly the same sign as the intraday
   notebook's bar signal.** Algebraically
   $s_{\mathrm{rem}}=(\widehat{RV}_t-\mathrm{IV}^2_{\mathrm{hr}}h\,w)/w=s_{\mathrm{bar}}/w$
   with $w>0$; on the data the sign agreement is 1.0000 and
   $|s_{\mathrm{rem}}w-s_{\mathrm{bar}}|<10^{-19}$. Scaling the forecast up by the
   share and scaling the implied down by the same share is the same comparison.
   The only thing this study changes is therefore the **exit** (hold to the close
   instead of one bar). A genuinely different remaining-session reading needs a
   remaining-session forecast that is not the one-bar forecast divided by the
   profile share — the direct multi-horizon forecasts of the main repository —
   and those per-bar files are not in this repository.

Reconciliation at 15:30 (the gate): on the 803 common days the signal equals
the RV–IV notebook's signal to $10^{-11}$, the return equals it exactly, and
sign(s) scores 1.891 on both sides.

## Results

### Per entry clock, one trade per day, held to the close (803 days)

| clock | sign(s_rem) Sharpe | t | worst | maxDD | buy share | always short Sharpe | t | sign − AS: t | ΔSharpe 95% CI | placebo pct |
|---|---|---|---|---|---|---|---|---|---|---|
| 10:00 | 0.19 | 0.35 | −4.3 | −24.6 | 0.61 | −0.49 | −1.06 | 0.78 | [−1.06, +2.27] | 22 |
| 10:30 | 0.52 | 0.99 | −7.8 | −28.5 | 0.59 | −0.08 | −0.18 | 0.79 | [−0.91, +2.09] | 47 |
| 11:00 | 0.48 | 0.92 | −8.7 | −21.1 | 0.60 | −0.22 | −0.47 | 0.93 | [−0.81, +2.16] | 44 |
| 11:30 | 0.34 | 0.65 | −9.4 | −32.9 | 0.50 | −0.10 | −0.20 | 0.62 | [−1.00, +1.70] | 32 |
| 12:00 | 0.71 | 1.42 | −10.8 | −19.8 | 0.61 | −0.11 | −0.24 | 1.21 | [−0.66, +2.12] | 63 |
| 12:30 | 0.51 | 0.93 | −10.7 | −29.3 | 0.57 | −0.25 | −0.53 | 1.12 | [−0.70, +2.04] | 47 |
| 13:00 | 0.31 | 0.61 | −11.8 | −19.4 | 0.67 | −0.55 | −1.17 | 1.10 | [−0.67, +2.28] | 30 |
| 13:30 | 0.10 | 0.21 | −11.3 | −22.3 | 0.46 | −0.25 | −0.50 | 0.48 | [−1.02, +1.70] | 17 |
| 14:00 | −0.05 | −0.09 | −7.6 | −28.0 | 0.67 | −0.11 | −0.21 | 0.07 | [−1.88, +1.67] | 11 |
| 14:30 | 0.29 | 0.54 | −9.5 | −28.9 | 0.52 | 0.40 | 0.79 | −0.14 | [−1.66, +1.41] | 29 |
| 15:00 | 1.29 | 2.24 | −5.2 | −22.0 | 0.50 | 0.75 | 1.41 | 0.58 | [−1.46, +2.16] | 94 |
| **15:30** | **1.89** | **3.62** | −5.4 | −12.5 | 0.37 | 0.31 | 0.64 | **2.22** | **[+0.08, +2.78]** | **99.8** |

"placebo pct" = the percentile of the clock's sign(s_rem) Sharpe among 2,000
random entry-clock-per-day series (placebo median 0.55, 5th–95th −0.28 to
1.32). Because the sign is identical to the bar signal's, the bar-signal
hold-to-close columns are these same columns.

Reading: before 15:00 the sign rule held to the close is indistinguishable from
a random entry clock; always short held to the close from a morning entry
loses money (−0.5 to −0.1) because the morning premium (19.8 points at 10:00 vs
5.4 at 15:30) has to survive the whole day's realized variance; the forecast's
increment over always short is positive at most clocks but never significant
before 15:30 (t ≤ 1.2). The edge appears at 15:00 (94th percentile) and is
decisive at 15:30 (99.8th).

### Aggregates that do not stack positions (one trade per day)

| rule | Sharpe | t | worst | maxDD | traded days | vs 15:30-only sign(s) 1.89 |
|---|---|---|---|---|---|---|
| first clock with s_rem < 0: sell, hold to close | −0.41 | −0.96 | −7.8 | −28.0 | 93% | ΔSharpe −2.30, CI [−3.66, −0.99], t −3.44 |
| first clock with s_rem > 0: buy, hold to close | 0.32 | 0.68 | −1.0 | −18.3 | 97% | ΔSharpe −1.57, CI [−2.87, −0.31], t −2.78 |
| random entry clock per day (placebo median) | 0.55 | — | — | — | 100% | — |

The first sell signal fires at 10:00 or 10:30 on 59% of days, so the
"first-fire" rules are mostly morning entries; both are significantly worse
than waiting for the close.

### Costs: one crossing per day at entry (settlement is cash)

| clock | half-spread, % of premium | sign(s_rem) mid | crossed | always short mid | crossed |
|---|---|---|---|---|---|
| 10:00 | 1.45 | 0.19 | −0.17 | −0.49 | −0.85 |
| 12:00 | 1.32 | 0.71 | 0.49 | −0.11 | −0.34 |
| 14:00 | 1.75 | −0.05 | −0.42 | −0.11 | −0.47 |
| 15:00 | 2.22 | 1.29 | 0.95 | 0.75 | 0.38 |
| 15:30 | 2.80 | 1.89 | 1.46 | 0.31 | −0.13 |

One crossing per day (against 21–23 for the intraday re-pick) keeps the
15:30 and 15:00 entries positive after costs; no earlier entry survives
above 0.5.

### Per year, sign(s_rem) held to the close

| entry | 2020 | 2021 | 2022 | 2023 | 2024 | warm-up matched | all |
|---|---|---|---|---|---|---|---|
| 10:00 | −1.38 | −0.38 | 0.84 | 0.53 | 0.55 | 0.41 | 0.19 |
| 12:00 | 3.14 | 1.15 | 0.47 | 0.40 | −0.73 | 0.57 | 0.71 |
| 14:00 | −0.17 | 1.69 | −2.06 | 1.17 | −2.31 | −0.05 | −0.05 |
| 15:00 | 4.11 | 2.60 | 0.00 | 0.14 | 0.92 | 1.15 | 1.29 |
| 15:30 | 4.54 | 1.01 | 0.36 | 2.30 | 3.14 | 1.61 | 1.89 |

## Gates

- Reconciliation at 15:30: exact (signal to $10^{-11}$, return to 0, Sharpe 1.891
  both sides).
- Implied term is a price: BSM over the remaining window reproduces the quoted
  mid at every clock (ratio 1.0000).
- Paired tests: Newey–West lag $\lfloor 1.5\,n^{1/3}\rfloor$; block bootstrap
  B = 2000, rng(0), block $\lceil n^{1/3}\rceil$.
- Placebo: 2,000 random entry-clock-per-day series.
- Causality: 10 future-perturbation cuts (realized variance ×3 and returns
  negated after the cut), 0 violations — $w$, $F_{\mathrm{rem}}$ and
  $s_{\mathrm{rem}}$ before the cut are unchanged.
- No thresholds, clips or magic numbers; the only constants are the notebook's
  standing 63-day minimum and the standard test conventions.

## Verdict

**Reject as an intraday generalization:** the remaining-session signal is the
intraday bar signal with the same sign by algebra (both sides scaled by the same
profile share), holding to the close from any entry before 15:00 is
indistinguishable from a random entry clock (placebo percentiles 11–63), and the
one-trade-per-day first-fire rules are significantly worse than the 15:30 trade
(ΔSharpe −2.3 and −1.6); the edge lives in the last hour (15:00 at the 94th
percentile, 15:30 at the 99.8th), exactly where the remaining window is the
traded bar.

## Open questions

- A remaining-session forecast that is not $\widehat{RV}_t/w_t$: the main
  repository's direct multi-horizon per-bar forecasts would give
  $F_{\mathrm{rem}}$ a different sign from the bar signal and are the only
  construction that could still change this verdict; they are not in this
  repository.
- The 15:00 entry (Sharpe 1.29, 94th placebo percentile, 0.95 after one
  crossing) is the one clock besides 15:30 with a hint of edge; a two-clock
  rule (15:00 or 15:30) would need its own placebo.
- Always short held to the close from the morning loses; the morning premium is
  large in points but not rich per unit of realized variance — consistent with
  the diagnosis that the day's variance premium is concentrated at the close.
