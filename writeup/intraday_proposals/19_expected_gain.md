# 19. Expected-gain identity \(G=C^{\mathrm{BS}}(V_H)-C^{\mathrm{BS}}(V_M)\)

Read-only diagnostic. Script `writeup/intraday_proposals/19_expected_gain.py`;
tables and figure in `results/atm_straddle_intraday/proposals/19/`. Nothing is
wired into a trading rule. The notebook's §5b now *displays* the identity;
§6 still trades \(\mathrm{sign}(s^{\mathrm{m}})\).

The professor note (deterministic variance disagreement, \(\mu=r-q\), market
curve held fixed) maps onto objects the notebook already computes. Hold from
clock \(t\) to the next bar \(u\), sell at the market, expire at the close:

\[
V_{M,t}=\mathrm{IV}^{2}_{\mathrm{hr},t}\,h_t,\qquad
V_{H,t}=\widehat{RV}_t+(1-w_t)V_{M,t},
\]
\[
s^{\mathrm{m}}_t=V_{H,t}-V_{M,t},\qquad
G_t=C^{\mathrm{BS}}(S_t,K_c,K_p;\sqrt{V_{H,t}})-C^{\mathrm{BS}}(S_t,K_c,K_p;\sqrt{V_{M,t}}).
\]

\(C^{\mathrm{BS}}\) is Black-76 on the nearest-OTM package, \(r=0\), \(F=S\)
(the vendor convention). \(s^{\mathrm{m}}\) is the variance gap; \(G\) is the
expected change in the package mid, in index points. Vega from a *moving*
implied curve is printed in section 7 and is not in the signal.

Frame: the notebook's own 10,387 bars, 866 days, twelve clocks, block-diagonal
ridge. Vectorized \(C^{\mathrm{BS}}\) matches `asl._bsm_package_price` to
\(1\mathrm{e}{-9}\) on a sample of rows.

## Gate

The notebook's two rule tables, rebuilt from \(s^{\mathrm{m}}=V_H-V_M\).
Midpoint and crossed. Worst gap \(8.63\mathrm{e}{-8}\).

| rule | Sharpe mid | target | Sharpe crossed | target |
|---|---|---|---|---|
| always short | 1.9020 | 1.9020 | −1.9181 | −1.9181 |
| always short, flat at 15:30 | 3.4300 | 3.4300 | −3.6207 | −3.6207 |
| sign(s) | 2.0127 | 2.0127 | −2.2998 | −2.2998 |
| always short, sign(s) close | 2.9723 | 2.9723 | −1.0955 | −1.0955 |

\(\mathrm{sign}(G)=\mathrm{sign}(s)\) on 10,374 / 10,374 finite bars. The sign
rule is a free match. \(C^{\mathrm{BS}}(V_M)/\mathrm{entry}\) has median
**1.0000** at every clock (p5–p95 inside \([0.9972, 1.0063]\)): the remaining
implied is a price.

## Pre-registered scores

**1. Does \(G/|s|\) rise through the day like ATM gamma \(1/\sigma\sqrt{T}\)?**
Yes. Spearman(clock order, median \(G/s\)) \(=+1.000\); Spearman(median \(G/s\),
median \(1/\sqrt{V_M}\)) \(=+1.000\). Median \(G/s\) goes \(2.37\mathrm{e}{5}\)
at 10:00 to \(6.81\mathrm{e}{5}\) at 15:30, a factor of 2.9, and sits on top of
the Black-76 \(\partial C/\partial V\). This is the 15:30-dominance story in
dollars: the same variance gap is worth more as \(T_{\mathrm{rem}}\to 0\).

**2. Is \(R\sim a+bG\) tighter than \(R\sim a+bs\)?** Mildly, and only as a
return. Bar-level \(R^2\) of the unmarked long-package return:

| spec | all bars \(R^2\) (t) | 15:30 \(R^2\) (t) | 10:00–15:00 \(R^2\) (t) |
|---|---|---|---|
| \(R\sim s\) | 0.0001 (−1.08) | 0.0024 (−2.90) | 0.0001 (+1.17) |
| \(R\sim G\) (points) | 0.0005 (+1.59) | 0.0003 (+0.47) | 0.0012 (+3.22) |
| \(R\sim G/\mathrm{entry}\) | **0.0035 (+2.81)** | **0.0046 (+2.08)** | **0.0016 (+2.93)** |

The map is the right *units* (expected return, not a variance gap). It is not a
usable predictor: \(R^2\) is still 0.0035. The 30-minute mark is mostly not
\(G\).

**3. Unit-median sized by \(|G|\) vs \(|s|\).** Same signs, different scale.
Warmup 756 bars (\(63\times 12\)). Daily-sum Sharpe, 866 days:

| rule | Sharpe mid | mean/day | Sharpe crossed |
|---|---|---|---|
| sign(s) | 2.01 | 0.169 | −2.30 |
| UM \(\lvert s\rvert\) | **2.36** | 0.249 | −1.02 |
| UM \(\lvert G\rvert\) | 2.34 | **0.355** | −0.31 |

Paired dSharpe UM-\(\lvert G\rvert\) minus UM-\(\lvert s\rvert\): **−0.012**,
percentile \([-0.42,+0.38]\), basic \([-0.40,+0.40]\), both cover zero. Newey–West
\(t\) of the daily *dollar* difference is \(+2.81\): UM-\(\lvert G\rvert\) makes
more points per day (0.355 vs 0.249) by putting more size on the close, and
pays for it in variance, so Sharpe does not move.

Sizing placebo (2,000 within-day permutations of \(\lvert G\rvert\), signs of
\(s\) held): real UM-\(\lvert G\rvert\) is at the **100th** percentile against
a shuffle of its own magnitudes. The clock pattern of \(G/s\) is real. It does
not beat UM-\(\lvert s\rvert\) on Sharpe.

**Adopt bar:** causality 0/10 on \(w\), on \(G\) across clocks, on future IV,
and on the lagged UM median; paired dSharpe CI must exclude zero; placebo
\(\ge 95\)th. Placebo clears, CI does not. **Do not adopt.**

## Residual delta (section 3 of the note)

The book is nearest-OTM, not a synthetic ATM. Black-76 residual package delta
\(N(d1_c)+N(d1_p)-1\), evaluated at \(\sqrt{V_M}\):

| | mean | mean \(\lvert\Delta\rvert\) | p90 \(\lvert\Delta\rvert\) | pct \(\lvert\Delta\rvert>0.05\) | pct \(\lvert\Delta\rvert>0.10\) |
|---|---|---|---|---|---|
| pooled (10,374) | +0.0013 | 0.057 | 0.120 | 46.5% | 15.9% |
| 10:00 | +0.0018 | 0.038 | 0.077 | 28.7% | 3.5% |
| 15:30 | +0.0019 | 0.093 | 0.204 | 67.0% | 38.3% |

Mean delta is ~0. Mean *absolute* delta is not: 0.04 in the morning, 0.09 at
the close. An unhedged nearest-OTM package leaks a directional term that the
note's \(\mu=r-q\) closed form sets to zero. The leak grows as \(T_{\mathrm{rem}}\to 0\)
because a 5-point OTM gap is a larger \(\lvert d1\rvert\) when total vol is small.

## Curve revision (not in the signal)

If \(v_M\) were the note's fixed curve, leftover \((1-w_t)V_{M,t}\) would equal
the next stamp's remaining implied. The gap is a curve revision:

| clock | mean revision / \(V_M\) | median | mean \(\lvert\mathrm{revision}\rvert\) / \(V_M\) |
|---|---|---|---|
| 10:00 | −0.009 | −0.028 | 0.095 |
| 12:00 | −0.012 | −0.025 | 0.081 |
| 14:00 | +0.029 | +0.010 | 0.102 |
| 15:00 | +0.062 | +0.043 | 0.110 |

Typical move is ~9–11% of remaining implied per bar. Morning revisions are
slightly down; the last hour is up. That is diagnosis mechanism 2 (vega on the
30-minute mark). It is **not** a forecast we currently have, and it is not in
\(G\).

## Verdict

- Write the identity in §5b: **done.** \(s^{\mathrm{m}}=V_H-V_M\); \(G\) is the
  dollar map. Rules still use \(\mathrm{sign}(s^{\mathrm{m}})\).
- \(G/\lvert s\rvert\) *is* ATM gamma through the day. That is why the close
  bar dominates a linear variance signal.
- \(G/\mathrm{entry}\) is a slightly tighter predictor of \(R\) than \(s\),
  and still \(R^2=0.0035\).
- UM sized by \(\lvert G\rvert\) does not clear the Sharpe gate against UM
  sized by \(\lvert s\rvert\) (dSharpe −0.012, CI covers 0). **Not adopted.**
- Residual \(\lvert\Delta\rvert\) is 0.06 pooled, 0.09 at 15:30. The unhedged
  closed form is an approximation.
- The implied curve does move (~10% of \(V_M\) per bar). Vega stays out of the
  signal.

Files: `19_expected_gain.{py,md}`, `results/atm_straddle_intraday/proposals/19/`.
