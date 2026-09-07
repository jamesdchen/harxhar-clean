# 16 - Asymmetric body: a strangle on buy days, the straddle on sell days

**The fact.** On the forecast's buy days a long 15:30 straddle is right 41% of the time with an average win of +128% of premium and an average loss of -71%; on sell days a short is right 63% of the time with wins of +67% and losses of -91%. The edge is in the size of the moves on the selected days, not in how often the call is right, and every symmetric rule that leans on magnitude (sizing, thresholds, wings, bodies) has failed against a fixed-size sign(s).

**The one cell, fixed before any number was seen.** Sell days unchanged: the deck's short nearest-OTM straddle. Buy days: a long strangle with strikes 25 points outside the deck's strikes (call at K_c + 25, put at K_p - 25), both legs quoted two-sided at 15:30, held to cash settlement; when a wing has no two-sided quote the deck's straddle is used for that day and counted. Frame of record: the strangle position is scaled to the same dollars as one straddle premium. Comparator: the deck's rule. Script: `16_asymmetric_body.py`; the gate reproduces the deck's row (sign(s) 1.338322 on 866 days).

## Fill diagnostics

- Buy days: 346. Both 25-point wings quoted two-sided on 131 of them (38%); the other 215 fall back to the straddle.
- Median strangle premium on those days: 0.33 points, 5.0% of the straddle's 7.55 points.
- The strangle expires worthless on 97% of the buy days on which it is held.

## Results, 866 days, block-diagonal ridge

| | midpoint | crossed spread |
|---|---|---|
| deck: sign(s), straddle both sides | 1.338 | 0.870 |
| cell: strangle on buy days, same dollars | -0.102 (t -0.19) | -0.533 (t -0.99) |

Buy days only, per unit of the deck's straddle premium: the straddle earns +0.100 a day (hit 41%, win +1.28, loss -0.71); the strangle earns -0.160 a day (hit 26%, win +1.87, loss -0.87).

Paired cell minus deck: Sharpe difference -1.44 at the midpoint with percentile interval [-2.42, -0.68]; -1.40 at the crossed spread with [-2.39, -0.61]; daily mean difference -0.107, t -3.90. Both intervals exclude zero against the cell.

## Verdict

**Kill.** Twenty-five points is about two and a half standard deviations of a half-hour move: the strangle is quoted on only 38% of buy days, costs 5% of the straddle's premium, and expires worthless on 97% of the days it is held, so scaling it to the straddle's dollars turns those days into total losses. Its convexity is real when it pays (+187% against +128%) and it pays a quarter as often. The buy-day edge lives in moves of five to fifteen points, which the straddle already covers; the nearest-out strangle five points out was measured in the experimental notebook's body lab at 1.49 against 1.34 per unit of time value (t 0.3) and bounds any tighter asymmetric version. This closes the magnitude line: sizing, thresholds, wings, bodies and now an asymmetric body have all been measured against a fixed-size sign(s) on the straddle, and sign(s) stands.

Note on the distance: 25 points was taken from the condor lab's wing ladder, where it is a wing width, and is the wrong distance for a body with thirty minutes to live. That is the coordinator's specification error, recorded here; a five-point asymmetric version would be a new pre-registration and is not proposed.
