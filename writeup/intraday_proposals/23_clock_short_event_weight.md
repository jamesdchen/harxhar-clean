# 23. High-Sharpe daytime shorts + sign(s) at 15:30; event days as \(q=0\)

Read-only. Script `writeup/intraday_proposals/23_clock_short_event_weight.py`.
Gate: notebook hybrid 2.97, sign(s) 2.34, always short 1.90.

## Event days — already weight 0 in the notebook

The §7 calendar test sets \(q_{15:30}=0\) on 84 FOMC/month-end days and **keeps all 866 rows**. Zeros stay in the mean and the sd.

| 15:30 sign(s) | n | Sharpe |
|---|---|---|
| unfiltered | 866 | 1.37 |
| \(q=0\) on events, days kept | 866 (89 zeros: 84 events + 5 censored IV) | **1.81** |
| **drop** event days | 782 | **1.90** ← cherry-pick |

The notebook's 1.39 → 1.83 is the middle row, not the drop. Hybrid with the same weight-0 close: **2.97 → 3.42**, dSharpe \(+0.45\), CI \([+0.11,+0.85]\) excludes 0. Still a forward test, not adopted (proposal 18: IR vs always-short sitting out the same days falls).

§7 markdown now says weight 0, not “non-event days.”

## Only short the high-Sharpe daytime clocks

Always-short Sharpe by clock (one bar/day): 15:00 **2.51**, 11:30 **2.19**, 12:30 **1.78**, 12:00 **1.13**, 14:30 **1.00**; open and 13:00–14:00 are 0.2–0.8; 15:30 always-short is 0.20.

Picking those clocks from this table is in-sample. Causal: trailing expanding Sharpe of always-short \(R'\) at that clock, `shift(1)`, min 63; short iff \(>H\). Then sign(s) at 15:30. That is proposal 01(a) on the body plus 01(c′) at the close. 01(a) already lost without the close overlay.

| rule | Sharpe mid | crossed | vs hybrid dSharpe |
|---|---|---|---|
| hybrid (all 11 daytime shorts + sign close) | **2.97** | −1.10 | — |
| IN-SAMPLE AS\(>1\) (5 clocks) + sign close | 2.79 | +0.33 | −0.18, CI covers 0 |
| IN-SAMPLE AS\(>1.5\) (3 clocks) + sign close | 2.51 | +0.55 | −0.46, covers 0 |
| causal trail AS Sharpe\(>0\) + sign close | 2.64 | −0.80 | **−0.34, CI below 0** |
| causal trail AS Sharpe\(>1\) + sign close | 2.29 | −0.03 | **−0.68, CI below 0** |

Causal clock filters **lose** to shorting every daytime bar. Same as 01(a). The weak clocks still pay decay; trailing Sharpe cannot drop them in time. Do not adopt a subset of daytime shorts.

Crossed: fewer shorts means fewer crossings, so IS AS>1.5 is **+0.55** crossed vs hybrid **−1.10**. That is a cost story, not a better variance bet, and the clock list is in-sample.

## Verdict

- Event flat: keep \(q=0\), \(n=866\). Do not quote the 782-day Sharpe.
- Daytime clock subset: in-sample list is peeking; causal trailing loses to the hybrid. Leave the 11 always-shorts in place.
