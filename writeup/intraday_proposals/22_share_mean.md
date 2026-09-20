# 22. \(w\) as mean of daily remaining shares, not ratio of clock means

Read-only. Script `writeup/intraday_proposals/22_share_mean.py`.
10,387 bars, 866 days. Same in-fit panel back to 2001, 63-day min, `shift(1)`.

Notebook **V0** (ratio of means):

\[
\hat m_{d,c}=\overline{\mathrm{RV}}_{\cdot,c},\qquad
w_{d,c}=\hat m_{d,c}\Big/\sum_{s\ge c}\hat m_{d,s}.
\]

High-vol days dominate each clock mean.

**Mean of shares** (this file):

\[
\pi_{d',c}=\mathrm{RV}_{d',c}\Big/\sum_{s\ge c}\mathrm{RV}_{d',s},\qquad
w_{d,c}=\overline{\pi}_{\cdot,c}\ \text{over }d'<d.
\]

Each prior day votes equally on the remaining pie. At 15:30 both are 1.
Causality: 0/10 on the cut bar.

## Gate

V0 vs the notebook. Worst gap \(8.63\mathrm{e}{-8}\).

## The two \(w\)'s

\(\mathrm{corr}=0.9999\) (same U). Median \(|w_{\mathrm{share}}/w_{\mathrm{V0}}-1|=0.14\).
Mean-of-shares is **fatter in the morning**:

| clock | median \(w\) V0 | median \(w\) mean-of-shares |
|---|---|---|
| 10:00 | 0.130 | **0.150** |
| 12:00 | 0.091 | 0.109 |
| 15:00 | 0.449 | 0.455 |
| 15:30 | 1 | 1 |

## Calibration \(\mathrm{mean}(RV)/\mathrm{mean}(w V_M)\)

Mean-of-shares is a **richer** slice (further below 1), especially before noon.

| clock | V0 | mean of shares |
|---|---|---|
| 10:00 | 0.91 | 0.79 |
| 12:00 | 1.01 | 0.84 |
| 15:30 | 0.82 | 0.82 |

## Sign rule

| \(w\) | sign(s) mid | hybrid mid | sign(s) crossed | % long |
|---|---|---|---|---|
| V0 | 2.01 | 2.97 | −2.30 | 51.7 |
| mean of shares | **2.34** | 2.97 | −1.90 | 38.7 |

Hybrid is identical: it only uses \(\mathrm{sign}(s)\) at 15:30, where \(w=1\).

Paired dSharpe sign(s) mean-of-shares minus V0: **+0.33**, percentile \([+0.08,+0.58]\), basic \([+0.08,+0.58]\), both exclude 0.

That is the 11 caveat: Sharpe moved because the rule **shorts more** (morning \(w\) up → slice up → \(s\) down), not because the slice is a better price. Calibration got worse.

## Verdict

The algebra you want is right: remaining share is a *day-level* fraction, so average the fractions, don't divide averages. It is F_t. It does not improve the implied term. Do not replace V0 in the notebook.

Files: `22_share_mean.{py,md}`, `results/atm_straddle_intraday/proposals/22/`.
