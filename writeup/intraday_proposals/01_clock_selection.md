# Proposal 1 — Choosing *when* to trade: clock selection for the intraday trade

**Idea.** The intraday trade re-picks the nearest out-of-the-money package every
30 minutes from 10:00 and holds each bar to the next midpoint (the 15:30 bar is
cash-settled). Its per-clock table shows the edge is uneven across the day, so a
natural lever is to trade only some bars — but any such choice must be made from
information available at the time, and it must beat a random choice of the same
number of bars. Four candidates are stated exactly and gated below. All numbers
are daily aggregates (the sum of the traded bars' returns on each day, 866 days,
block-diagonal ridge forecast); the frame reproduces the notebook's own daily
Sharpes to three decimals before any candidate is applied.

**Construction (shared).** Per-bar return of the position taken, $R'_{t,c}$ for
day $t$ and clock $c$; always short $q=-1$; sign(s) $q=\mathrm{sign}(s^{\mathrm m}_{t,c})$
on the window-matched signal (zero in its 63-day warm-up). "Trailing edge" of a
clock is the expanding mean of $R'_{\cdot,c}$ over prior days, lagged one day,
at least 63 days — no thresholds other than zero. Daily aggregate
$D_t=\sum_{c\in\text{traded}} R'_{t,c}$. Sharpe $=\sqrt{252}\,\bar D/\hat\sigma_D$.
Paired test: $D_t^{\text{cand}}-D_t^{\text{base}}$ with lag-robust standard
errors (lag $\lfloor 1.5\,n^{1/3}\rfloor=14$) and a block bootstrap of the
Sharpe difference (2000 draws, block $\lceil n^{1/3}\rceil$, seed 0). Placebo:
2000 random clock subsets of the same per-day size, drawn among that day's
available clocks; the realized Sharpe's percentile must be $\ge 95$.
Causality: returns on days $\ge T$ are perturbed for ten cut dates $T$ and every
decision before $T$ must be unchanged.

| candidate | Sharpe | mean/day | worst day | maxDD | vs baseline (Sharpe) | ΔSharpe [boot 95%] | paired t | placebo pct | causality |
|---|---|---|---|---|---|---|---|---|---|
| all bars, always short (baseline) | 1.90 | 0.160 | −11.40 | −16.8 | — | — | — | — | — |
| all bars, sign(s) (baseline) | 1.92 | 0.153 | −6.26 | −14.6 | — | — | — | — | — |
| 15:30 only, sign(s) (the close trade) | 1.82 | 0.122 | −5.42 | −12.5 | — | — | — | — | — |
| (c) always short all day, **unit-median** at close (existing hybrid) | 3.05 | 0.272 | −9.65 | −22.0 | AS 1.90 | +1.15 [−0.30, +2.25] | +2.53 | n/a (fixed structure) | pass |
| (c′) always short all day, **sign(s)** at close | **3.37** | 0.268 | −5.72 | −10.9 | AS 1.90 | **+1.47 [+0.14, +2.54]** | +2.27 | n/a (fixed structure) | pass |
| (a) always short, only clocks with positive trailing edge | 1.68 | 0.117 | −11.40 | −14.3 | AS 1.90 | −0.22 [−0.91, +0.41] | −1.82 | 41 | pass |
| (a) sign(s), only clocks with positive trailing edge | 1.60 | 0.123 | −6.28 | −15.9 | sign(s) 1.92 | −0.32 [−0.56, −0.09] | −2.74 | 32 | pass |
| (b) always short, from the causally chosen start clock | 1.80 | 0.146 | −11.40 | −16.8 | AS 1.90 | −0.11 [−0.57, +0.17] | −0.85 | 37 | pass |
| (b) sign(s), from the causally chosen start clock | 1.80 | 0.142 | −6.22 | −14.7 | sign(s) 1.92 | −0.12 [−0.23, −0.01] | −2.40 | 61 | pass |
| (d) per clock: sign(s) where its trailing edge beats always short, else always short | 2.81 | 0.229 | −7.09 | −12.6 | AS 1.90 / sign(s) 1.92 | +0.91 [−0.33, +2.10] / +0.89 [+0.13, +1.86] | +1.45 / +2.16 | **99.4** | pass |

Per-year Sharpe: all-bars always short {2020 1.64, 2021 1.33, 2022 2.95, 2023 2.81, 2024 −0.73};
all-bars sign(s) {4.14, 2.43, 0.88, 1.79, 0.74}; (c) hybrid with unit-median close
{4.87, 3.34, 1.81, 3.17, 4.73}; (c′) hybrid with sign(s) close {5.58, 2.88, 1.74, 3.91, 3.51};
(d) switch {3.59, 2.48, 1.59, 3.50, 2.73}.

**Reading.**
- **(a) Trailing-edge filter — reject.** Dropping clocks whose past mean is
  negative removes about two bars a day and *lowers* the Sharpe for both rules;
  for sign(s) the loss is significant (t −2.7, bootstrap interval below zero).
  Both sit at the 30–40th percentile of random subsets of the same size: the
  filter is no better than picking bars at random. Per-clock means are too noisy
  at 63+ days to identify the weak clocks in time.
- **(b) Causal late-session start — reject.** The argmax of the trailing tail
  sums almost always chooses 10:00 (always short 746 of 803 decided days; sign(s)
  382), i.e. it re-selects the all-bars trade and adds noise on the days it does
  not. Both variants trail the baseline; sign(s) significantly (t −2.4). The
  late-session edge in the per-clock table is not large enough, relative to its
  noise, to be found causally.
- **(c′) Hybrid with sign(s) at the close — the one adoptable construction.**
  Always short on every bar before the close and sign(s) on the 15:30 bar gives
  Sharpe 3.37 against 1.90 for all-bars always short, with a bootstrap interval
  on the Sharpe difference that excludes zero (+0.14 to +2.54; paired t 2.3),
  the smallest drawdown of any variant (−10.9) and a worst day of −5.7, positive
  in every calendar year. It dominates the existing unit-median-close hybrid
  (3.05, drawdown −22.0) and uses no unit-median sizing, which the deck has
  removed. It is a fixed structure, not a per-day selection, so the placebo does
  not apply; its justification is the deck's own finding that the signal's sign
  content lives at the close and that the earlier bars carry only the short
  premium.
- **(d) Per-clock switch — passes its gates but is dominated.** Letting the
  trailing evidence choose sign(s) or always short clock by clock reaches 2.81,
  at the 99.4th percentile of random assignments of the same size, significant
  against all-bars sign(s) (t 2.2, interval above zero) but not against all-bars
  always short (t 1.5). It puts 32% of clock-days on sign(s), mostly the close
  and the late bars — it is learning (c′) slowly and paying for the learning.
  Nothing it finds is outside what (c′) states in advance.

**Verdict.** Adopt (c′) — always short on every bar before the close and sign(s)
on the 15:30 bar — as the intraday headline; reject (a) and (b); retire (d) as
a confirmation that the causal search converges on (c′).

**Open questions.**
- The (c′) gain over the close-only sign(s) trade (1.82 → 3.37) is the short
  premium collected on the 11 earlier bars; the paired tests above are against
  the all-bars rules, and a test against the close-only trade would complete
  the picture.
- Fills: the earlier bars are one-bar holds marked at the next midpoint; the
  per-bar spread cost at eleven crossings a day is the first thing that should
  be charged before (c′) is quoted outside the notebook.
- Whether (d)'s late-bar sign(s) selections (14:30–15:00) carry any edge of
  their own, once (c′) is the baseline, is the one residual question worth a
  paired test.

Scripts and logs: `scratchpad\intraday_prop_clock\{build_frame.py, study.py, study.log, results.csv}`
(session scratch area). Nothing in any notebook was changed.
