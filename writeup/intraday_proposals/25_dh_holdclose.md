# 25. Delta-hedged \(t\to T\)

Hold the entry \(K\) to official close. Rebalance Black-76 package \(\Delta\)
every 30 minutes on vendor \(S\), remaining vol = stamp IV \(\times\sqrt{h}\).
Sign of \(s\) is the same as the re-pick book. 866 days.

| rule | Sharpe mid | mean/day |
|---|---|---|
| sign(s) unhedged \(t\to T\) | 0.71 | 0.304 |
| **sign(s) DH 0 bp** | **2.37** | 0.559 |
| sign(s) DH 0.5 bp | 1.42 | 0.335 |
| always short unhedged \(t\to T\) | 0.06 | 0.037 |
| **always short DH 0 bp** | **2.90** | 1.007 |
| 15:30 unhedged (paper) | 1.37 | — |
| 15:30 DH 0 bp | 1.33 | — |

Unhedged morning \(t\to T\) was a terminal-move bet. Flattening \(\Delta\)
along the path is the variance object. The 15:30 row is unchanged (same as
proposal 20: a 30-minute hold does not accumulate \(\Delta\)).

Files: `25_dh_holdclose.{py,md}`, `results/atm_straddle_intraday_holdclose/proposals/25/`.
