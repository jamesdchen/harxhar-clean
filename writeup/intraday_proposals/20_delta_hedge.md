# 20. One-bar delta-hedge of the intraday re-pick

Read-only. Script `writeup/intraday_proposals/20_delta_hedge.py`; tables and
figure in `results/atm_straddle_intraday/proposals/20/`. Same 10,387 bars,
same \(q\), as the notebook. Nothing is wired in.

The hold is already one 30-minute bar. Hedge once at entry with the Black-76
package delta, flatten at the same exit.

\[
\Delta_t = N(d1_c)+N(d1_p)-1,\quad
n_t=-q_t\Delta_t,\quad
\Pi_t=q_t(C_{t+1}-C_t)+n_t(S_{t+1}-S_t).
\]

\(r=0\), \(F=S_t\), total vol \(\sqrt{V_M}=\mathrm{IV}_{\mathrm{hr}}\sqrt{h_t}\).
\(S_{t+1}\) is the next trade bar's vendor underlying (10:00–15:00; `nxt_ts`
matches the next row to 0 s) and the official GSPC close at 15:30. \(\Delta\)
is frozen at entry. Per-premium \(R^{\mathrm{dh}}=\Pi/C_t\). Underlying cost,
when charged: 0.5 bp of \(S|n|\) at entry and at flatten (DH-legs convention).

## Gate

Unhedged rule tables vs the notebook. Worst gap \(8.63\mathrm{e}{-8}\).

Causality of \(\Delta\): 0/10 on later \(S\), on other clocks, on \(S_{\mathrm{exit}}\);
10/10 teeth on this bar's IV. Frozen at entry.

## Headline (daily-sum Sharpe, 866 days)

| rule | unhedged mid | DH mid 0 bp | DH mid 0.5 bp | unhedged crossed | DH crossed 0 bp |
|---|---|---|---|---|---|
| always short | 1.90 | **2.00** | 1.55 | −1.92 | −1.86 |
| always short, flat at 15:30 | 3.43 | 3.58 | 2.93 | −3.62 | −3.55 |
| sign(s) | 2.01 | **2.02** | 1.57 | −2.30 | −2.34 |
| hybrid | 2.97 | 3.01 | 2.55 | −1.10 | −1.10 |

Paired dSharpe DH minus unhedged (mid, 0 bp), circular block bootstrap:

| rule | dSharpe | percentile 95% | NW \(t\) | CI excludes 0 |
|---|---|---|---|---|
| always short | +0.10 | \([−0.14,+0.33]\) | +0.81 | no |
| sign(s) | +0.00 | \([−0.18,+0.20]\) | −0.14 | no |
| hybrid | +0.04 | \([−0.18,+0.26]\) | +0.11 | no |

DH \(\mathrm{sign}(s)\) minus DH always short: dSharpe **+0.015**, percentile
\([−1.53,+1.48]\). Rate-matched sign placebo on DH: real rule at the **99.95th**
of 2,000 (same as unhedged: the sign is the content, the hedge is not).

## The hedge is small

Always short, mid, 0 bp, index points of the package:

| | mean option | mean hedge | sd option | sd hedge | \(\mathrm{corr}\) | \(\mathrm{var}(\mathrm{hedge})/\mathrm{var}(\mathrm{DH})\) |
|---|---|---|---|---|---|---|
| pooled | +0.153 | +0.004 | 3.40 | 0.51 | −0.086 | **0.023** |
| 15:30 | +0.020 | +0.020 | 7.91 | 1.03 | −0.075 | 0.017 |

Two percent of DH variance is the stock hedge. Residual \(|\Delta|\) is 0.06
pooled and 0.09 at 15:30, but a 0.09 delta on a 30-minute \(\Delta S\) does not
move the straddle P&L. The unhedged package was already a variance bet.

Per-clock Sharpe (DH minus unhedged) wanders around zero; 15:00 always-short
picks up +0.30, 15:30 only +0.03. No late-day long-bias of the size the
single-leg DH book was built to remove.

## Verdict

- The note is right that an unhedged option is stock + variance. On *this*
  nearest-OTM 30-minute re-pick the stock term is noise.
- Delta-hedging does not change \(\mathrm{sign}(s)\) vs always short, does not
  rescue the crossed spread, and costs 0.5 bp of a ~0.06 notional each way
  (mid Sharpe 2.00 → 1.55).
- **Do not replace the notebook's unhedged table.** Report DH as a diagnostic
  that the scored object is already the variance term.

Files: `20_delta_hedge.{py,md}`, `results/atm_straddle_intraday/proposals/20/`.
