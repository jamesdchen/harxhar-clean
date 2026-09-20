# 21. Per-period implied as a variance drop, non-peeking

Read-only. Script `writeup/intraday_proposals/21_variance_drop.py`.
10,387 bars, 866 days. Nothing wired in.

The 30-minute implied from one expiry, in **variance**:

\[
\Delta V_t=V_M(t)-V_M(t+30\mathrm{m})
=\bigl(\mathrm{IV}^{\mathrm{hr}}_t\bigr)^2 h_t
-\bigl(\mathrm{IV}^{\mathrm{hr}}_{t+30\mathrm{m}}\bigr)^2 h_{t+30\mathrm{m}}.
\]

At 15:30, \(V_M(16{:}00)=0\) so \(\Delta V=V_M=\mathrm{IV}^2/2\).

Today's \(\Delta V\) needs the next stamp. This file uses only dates \(<d\):

| slice | formula |
|---|---|
| abs | lagged expanding mean of \(\Delta V^+\) at this clock |
| scaled | \((\mu_{\Delta}/\mu_V)\cdot V_{d,t}\) (ratio of means) |
| ratio | \(\overline{\Delta V^+/V}\cdot V_{d,t}\) (mean of ratios) |
| V0 | notebook \(w V_M\) (realized U) |

\(\Delta V\) floored at 0 for the means. Ex post \(\Delta V<0\) on **8.8%** of scored bars (remaining implied rose). Peeking \(\Delta V\) is calibration only, not a rule. Causality: 0/10 on the cut bar, teeth on later days.

## Gate

Notebook V0 rule table, mid and crossed. Worst gap \(8.63\mathrm{e}{-8}\).

## Calibration: \(\mathrm{mean}(RV)/\mathrm{mean}(\mathrm{slice})\)

1 = fair. Peeking \(\Delta V\) is **not** a better 30-min price than \(w V_M\): revision contaminates the drop.

| clock | V0 | scaled | ratio | PEEK \(\Delta V\) |
|---|---|---|---|---|
| 10:00 | 0.91 | 0.79 | 0.85 | 0.67 |
| 12:00 | 1.01 | 1.19 | 0.83 | 0.83 |
| 15:00 | 0.74 | 1.02 | 1.03 | 0.82 |
| 15:30 | 0.82 | 0.79 | 0.79 | 0.82 |

QLIKE (lower better): V0 **−11.55**, scaled −11.61, ratio −11.61, abs −11.00, peek **−10.32**.
\(\mathrm{corr}(RV,\mathrm{slice})\): V0 **0.86**, scaled 0.67, ratio 0.68, peek 0.71, abs 0.14.

The ex post drop matches bar RV *worse* than the realized-U share. That is the revision term.

## Sign rule (do not adopt)

| slice | sign(s) mid | hybrid mid | sign(s) crossed |
|---|---|---|---|
| V0 | **2.01** | 2.97 | −2.30 |
| abs | 1.80 | 2.03 | −1.69 |
| scaled | 1.77 | 3.16 | −2.22 |
| ratio | 1.96 | 3.16 | −1.89 |

Paired dSharpe vs V0 sign(s), mid: abs −0.21, scaled −0.24, ratio −0.05; all percentile intervals cover 0. Scaled sign(s) placebo 99.8th — it is a signal, not a better implied.

## Verdict

Work in variance: already the book. Non-peeking \(V_t-V_{t+1}\) is 11 V2 / 10 B in simpler clothes. It does not beat \(w V_M\). Even **today's** drop is a worse bar-variance price than \(w V_M\) (peek QLIKE, corr). Do not adopt.

Files: `21_variance_drop.{py,md}`, `results/atm_straddle_intraday/proposals/21/`.
