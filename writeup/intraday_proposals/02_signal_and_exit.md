# Intraday proposals 02 — the signal and the exit

*Shadow study, 2026-09-04. Read-only on the notebooks. Every construction is
causal; every comparison is against the intraday notebook's current
construction (block-diagonal ridge forecast; window-matched signal
$s^{m}_t=\widehat{RV}_t-\mathrm{IV}^2_{\mathrm{hr}}\,h_t\,w_t$; sign(s) at
every bar 10:00–15:00 held one bar to the next midpoint, the 15:30 bar
cash-settled; daily sums, $\sqrt{252}$). Reproduction gate: all four rows
of `rule_table_intraday_blk2.csv` reproduce to $10^{-6}$ (always short
1.902, sign(s) 1.916, unit-median 1.881, always short + unit-median close
3.053; 866 days, 10,387 bars). Paired tests: Newey–West with lag
$\lfloor 1.5\,n^{1/3}\rfloor=14$, block bootstrap $B=2000$, block
$\lceil n^{1/3}\rceil$, seed 0. Scripts and logs:
`scratchpad/intraday_prop_signal/{common.py, a, b, c, d}`.*

**Summary.** None of the four levers improves the intraday sign(s)
portfolio at conventional significance. Two facts organise the results:
(i) the vendor implied volatility *is* the Black–Scholes–Merton implied
volatility of the quoted package to within 0.1% at every clock, so there
is no better implied term to be had from the mid; (ii) the 15:30
settlement leg alone earns Sharpe 1.82 and is statistically
indistinguishable from the whole twelve-bar portfolio (1.92) — the eleven
intraday bars add almost nothing, and their per-clock sign decisions carry
no information beyond random slicings of the implied variance. The exit
question therefore answers itself (the paper's exit is the right one), and
the one sizing simplification worth carrying forward is *always short
before 15:30, sign(s) on the settlement leg* (3.37 vs 3.05 for the current
unit-median hybrid; not significant, but one fewer construction).

---

### (a) The implied term: Black–Scholes–Merton inversion, and how the remaining variance is sliced

**Idea.** The matched signal takes the vendor's hourly standard deviation,
scales it to the remaining session, and hands the next bar its share $w_t$
of that variance from a causal diurnal profile. Two things could be wrong
with it: the vendor number (replace it by inverting the quoted package
mid), or the slicing (replace the profile share by the flat share
$0.5/h_t$).

**Construction.** At every bar invert the package mid (call at $K_c$ plus
put at $K_p$, $r=0$) for the total volatility over the remaining window;
$V^{\mathrm{bsm}}$ is its square, $V^{\mathrm{vendor}}=\mathrm{IV}^2_{\mathrm{hr}}h_t$
the vendor route. Three alternative signals, each traded as sign(s):
a1 $=\widehat{RV}_t-V^{\mathrm{bsm}}w_t$; a2 $=\widehat{RV}_t-V^{\mathrm{vendor}}\cdot 0.5/h_t$;
a3 $=\widehat{RV}_t-V^{\mathrm{bsm}}\cdot 0.5/h_t$.

**Results.** $V^{\mathrm{bsm}}/V^{\mathrm{vendor}}$: median 1.0000 at
every clock, 5th–95th percentile within $[0.991, 1.004]$; zero inversion
failures on 10,387 bars.

| signal (sign(s), daily) | Sharpe | mean/day | worst day | maxDD | ΔSharpe vs current | bootstrap 95% | NW $t$ | sign agreement | % buy |
|---|---|---|---|---|---|---|---|---|---|
| current: vendor × profile share | 1.916 | 0.153 | −6.26 | −14.6 | — | — | — | 1.000 | 51.5 |
| a1: BSM × profile share | 1.834 | 0.147 | −6.26 | −14.2 | −0.083 | [−0.21, +0.02] | −1.42 | 0.999 | 51.5 |
| a2: vendor × flat share | 2.256 | 0.190 | −6.27 | −16.0 | +0.340 | [−0.23, +0.95] | +1.43 | 0.743 | 38.2 |
| a3: BSM × flat share | 2.147 | 0.181 | −6.27 | −16.6 | +0.231 | [−0.37, +0.90] | +1.03 | 0.742 | 38.3 |

The flat share changes *which* bars are buys: 91% of 10:00 bars and 14%
of 12:30 bars are buys under a2, against 56% and 52% under the profile
share — it is a bet on the shape of the day, not on mispricing.

**Gates.** Reproduction: pass. Placebo (2,000 random per-clock share
vectors, positive weights normalised over the remaining clocks): Sharpe
median 2.00, 5th–95th percentile $[0.94, 2.68]$; the profile share sits at
the **44th** percentile and the flat share at the 70th. Causality: the
inversion uses only the bar's own quotes; the profile share is lagged one
day (gated in `common.py`).

**Verdict.** *Reject.* The vendor field already equals the market's own
implied volatility, and no slicing beats a random one: the profile share
is at the median of random shares, the flat share's +0.34 is inside the
bootstrap interval and at the 70th percentile of random slicings.

**Open questions.** Why do random slicings do as well as the fitted
profile? Because the per-clock sign decisions are close to noise (see (c)
and (d)); the slicing only matters at clocks where the signal has content,
and there is essentially one such clock.

---

### (b) The forecast term: one Mincer–Zarnowitz recalibration per clock

**Idea.** The recalibration is fitted on all session bars of the prior 250
days and applied to every bar; the 10:00 bar is over-forecast by 22% and
the 13:30 bar under-forecast by 1% under that pooled map. A map fitted per
clock could remove the clock-level bias.

**Construction.** For each trade clock $c$, fit $m=a+b\hat y$ and
$\hat\sigma^2$ on the prior 250 days' rows of that clock only (same
weighted fit, same window; a mirror of the library routine, checked
identical to it on one clock), $\widehat{RV}^{(c)}_t=(m^2+\hat\sigma^2)B_t$;
$s_b=\widehat{RV}^{(c)}_t-\mathrm{IV}^2_{\mathrm{hr}}h_tw_t$; trade
sign($s_b$). The panel's history starts in 1998, so every per-clock window
is full from the first traded day; the 200-row and 63-row thresholds give
identical results (reported as a knob check only).

**Results.**

| clock | pooled $\widehat{RV}/RV$ | per-clock $\widehat{RV}/RV$ | clock | pooled | per-clock |
|---|---|---|---|---|---|
| 10:00 | 1.22 | 1.22 | 13:00 | 1.20 | 1.24 |
| 10:30 | 1.07 | 1.03 | 13:30 | 0.99 | 1.01 |
| 11:00 | 1.12 | 1.09 | 14:00 | 1.02 | 1.09 |
| 11:30 | 1.10 | 1.05 | 14:30 | 1.02 | 1.06 |
| 12:00 | 1.10 | 1.07 | 15:00 | 1.06 | 1.11 |
| 12:30 | 1.12 | 1.10 | 15:30 | 1.11 | 1.10 |

sign(s) with the per-clock map: Sharpe **1.543** (mean 0.126, worst −6.26,
maxDD −20.7) against 1.916 for the pooled map on the same 866 days;
ΔSharpe −0.373, bootstrap $[-0.98, +0.22]$, NW $t=-1.17$; sign agreement
0.868. The per-clock map does not even improve calibration: the 10:00
overshoot stays at 1.22 and five clocks get worse.

**Gates.** Reproduction: pass; mirror of the library fit: identical.
Placebo (2,000 permutations of which clock's map is applied to which
clock's rows): median 1.69, 5th–95th $[1.41, 2.00]$; the true assignment
sits at the **19th** percentile — random assignments do better. Causality:
coefficients for days $\le D$ unchanged under perturbation of later days'
realized variance (5 cuts): pass.

**Verdict.** *Reject.* Splitting the fit by clock trades 250 days × 12
bars of information for 250 rows and buys noise; the clock-level bias is
not what limits the intraday signal.

**Open questions.** The 10:00 bar is over-forecast by 22% under both maps
— a property of the first traded bar (the vendor spot is not live at
9:30), worth a data note rather than a model change.

---

### (c) The exit: hold to the close, one entry per day, the paper's exit

**Idea.** The intraday trade closes every position at the next midpoint.
If the mispricing the signal detects resolves at settlement rather than
over the next half hour, holding to the close should earn more.

**Construction.** $R^{\mathrm{close}}_t=\mathrm{exit}^{\mathrm{settle}}_t/P_t-1$
with the bar's own strikes settled at the official close (available for
every bar). c1: every entry held to the close, sign($s^m_t$) at entry
(twelve overlapping positions a day; also always short). c2: one entry per
day at the first clock with a matched signal, sign(s), held to the close.
c3: long to the close at the first clock where $s^m_t>0$, else short from
the first available clock. c4: the 15:30 entry only, sign(s), settled —
the paper's trade.

**Results** (daily; ΔSharpe against the current sign(s) portfolio, 1.916).

| variant | Sharpe | mean/day | worst day | maxDD | ΔSharpe | bootstrap 95% | NW $t$ |
|---|---|---|---|---|---|---|---|
| c1: every entry to the close, sign(s) | 1.004 | 0.417 | −74.0 | −134 | −0.91 | [−1.98, +0.12] | +1.34 |
| c1: every entry to the close, always short | 0.059 | 0.038 | −87.6 | −263 | −1.86 | [−3.24, −0.50] | −0.39 |
| c2: first clock with a signal, to the close | 0.190 | 0.012 | −4.30 | −24.6 | −1.80 | [−3.23, −0.44] | −3.00 |
| c3: first clock with $s>0$ long, else short from the first | 0.342 | 0.021 | −1.50 | −19.3 | −1.65 | [−3.11, −0.41] | −2.90 |
| c4: 15:30 only (the paper's trade) | 1.820 | 0.122 | −5.42 | −12.5 | −0.10 | [−0.75, +0.56] | −1.21 |

Per clock, holding to the close helps sign(s) only at 15:00 (0.08 → 1.24)
and the two mid-morning bars (10:30 0.10 → 0.50, 11:00 −0.03 → 0.47), and
it destroys always short at ten of twelve clocks (the short premium is
earned bar by bar; a position held to the close gives it back). c4
reproduces the 15:30 leg of the intraday portfolio exactly.

**Gates.** Reproduction: pass. Placebo (2,000 draws of a random entry
clock per day, sign(s) at that clock, held to the close): median 0.55,
5th–95th $[-0.26, 1.36]$; c2 sits at the 23rd percentile, c4 (15:30) at the
**99.7th**. Placebo for c3 (random long-entry clock on buy days): c3 at the
79th percentile. Causality: the entry choice for day $d$ is unchanged when
later bars of day $d$ are perturbed: pass.

**Verdict.** *Reject* every alternative exit; the paper's exit is the right
one, and the settlement leg is where the signal's content lives — the
15:30 entry alone is within 0.10 Sharpe of the whole intraday portfolio
and beats 99.7% of random entry clocks.

**Open questions.** The 15:00 bar held to the close (1.24) is the one
intraday entry that looks like the settlement leg; it is one clock, chosen
after seeing twelve, and would need a pre-registered retest.

---

### (d) Sizing: sign only against the unit-median rule, by clock

**Idea.** The unit-median rule levers each clock by $|s|$ relative to that
clock's own trailing median (cap 3). Which sizing do the intraday data
prefer, clock by clock, and does the settlement leg's sizing add anything
beyond its sign?

**Construction.** Per clock, the daily series of that clock's bar under
always short, sign(s), and unit-median; paired unit-median − sign per
clock and pooled; the hybrid "always short before 15:30" with either the
unit-median position or the sign only on the 15:30 leg; a
best-rule-per-clock composite chosen on one half of the sample and
evaluated on the other.

**Results** (daily Sharpe of that clock's bar).

| clock | always short | sign(s) | unit-median | Δ(UM − sign) | NW $t$ | mean $|q|$ (UM) | % at cap |
|---|---|---|---|---|---|---|---|
| 10:00 | 0.26 | 0.14 | −0.02 | −0.16 | −0.34 | 1.31 | 18 |
| 10:30 | 0.64 | 0.10 | 0.24 | +0.14 | +0.49 | 1.13 | 12 |
| 11:00 | 0.46 | −0.03 | 0.44 | +0.47 | +1.25 | 1.20 | 15 |
| 11:30 | 2.19 | 1.01 | 0.87 | −0.15 | +0.70 | 1.27 | 16 |
| 12:00 | 1.12 | 0.82 | 0.54 | −0.28 | −0.09 | 1.32 | 21 |
| 12:30 | 1.78 | −0.01 | −0.28 | −0.27 | −0.73 | 1.21 | 14 |
| 13:00 | 0.23 | 0.79 | 0.59 | −0.20 | +0.25 | 1.44 | 21 |
| 13:30 | 0.79 | 0.63 | −0.03 | −0.66 | −1.17 | 1.19 | 16 |
| 14:00 | 0.50 | −0.27 | 0.20 | +0.46 | +1.24 | 1.32 | 18 |
| 14:30 | 1.00 | 0.20 | 0.85 | +0.64 | +2.01 | 1.30 | 19 |
| 15:00 | 2.51 | 0.08 | 0.14 | +0.05 | +0.20 | 1.15 | 11 |
| 15:30 | 0.20 | 1.82 | 1.59 | −0.23 | +0.12 | 1.13 | 13 |

Always short beats both signal rules at **10 of 12** clocks; sign(s) beats
it only at 13:00 and 15:30. Unit-median beats sign at 5 of 12 clocks, none
significant; pooled Δ(UM − sign) = −0.035, bootstrap $[-0.83, +0.66]$.
Hybrids: always short + unit-median close **3.053** (the notebook's
headline) against always short + **sign** close **3.372**; ΔSharpe −0.32,
bootstrap $[-1.12, +0.41]$, NW $t=+0.12$ — the unit-median sizing of the
settlement leg does not beat its sign. Best-rule-per-clock composite,
held out: 1.19 vs sign-only 1.20 (chosen on the first half), 2.82 vs 2.77
(chosen on the second) — no transfer; in sample it scores 2.24.

**Gates.** Reproduction: pass. Placebo (2,000 random per-clock assignments
of {sign, unit-median}): median 1.90, 5th–95th $[1.66, 2.16]$; the
in-sample composite at the 99.8th percentile is the selection effect the
split-half test just refuted. The hybrid's 100th percentile is not a fair
test (the placebo family never contains always short). Causality: the
unit-median scale is lagged; positions for days $\le D$ unchanged under
perturbation of later $|s|$ (5 cuts): pass.

**Verdict.** *Adopt the simplification, not a Sharpe claim:* the intraday
data prefer always short at ten of twelve clocks and the sign of the
signal on the settlement leg; "always short before 15:30, sign(s) at
15:30" (3.37) is at least as good as the unit-median hybrid (3.05) with
one construction fewer — a parsimony argument.

**Open questions.** Whether the two clocks where sign(s) beats always
short (13:00, 15:30) are one fact or two: 15:30 is the paper's trade;
13:00 is one clock out of twelve.

---

### Side findings for the coordinator (outside this study's scope)

1. **The intraday 15:30 leg is the RV–IV trade.** On the 866 common days
   the strikes, entry, return, forecast and implied variance are identical
   to the RV–IV daily series; positions differ only on the 63-day warm-up,
   where the matched rule sits flat and the RV–IV rule trades. That is why
   the intraday 15:30 leg scores 1.89 against the RV–IV series' 1.56 on the
   same days (the RV–IV deck's own warm-up-matched figure is 1.94).
2. **Why the intraday Sharpe is "low".** It is not low relative to its
   parts: the settlement leg alone is 1.82, and the eleven other bars add
   +0.10 while contributing eleven bars of noise (per-clock sign(s)
   Sharpes of −0.3 to 1.0 at every clock but 15:30, and a placebo in which
   random slicings of the implied variance match the fitted profile). The
   intraday portfolio is the paper's trade plus dilution.
