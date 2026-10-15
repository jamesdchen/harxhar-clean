# Cold session handoff — 2026-09-01 (0DTE professor notes / intraday notebook)

Paste this to the next session: *Continue from `writeup/COLD_SESSION_HANDOFF_2026-09-01_0dte_intraday.md`. Work in `harxhar-0dte-professor` on `grok/0dte-professor-notes`, not `paper2`. Do not hunt for a better book. Diagnostics only.*

## Repo

- Path: `C:\Users\james\CC Allowed\harxhar-0dte-professor` (git worktree; **not** `harxhar-clean` / `paper2`)
- Branch: **`grok/0dte-professor-notes`** tracking `origin/grok/0dte-professor-notes`
- HTTPS: `https://github.com/jamesdchen/harxhar-clean.git` (same remote, this branch)
- HEAD at handoff: **`f038792`** *Fix pick_wings merge and land local diagnostic 0DTE tables.*
- Lots of **uncommitted** notebook/writer/lib work below. Data (`data/spxw_chain.parquet`) and yhats are untracked; do not commit them.
- Native Windows, PowerShell. Kernel: `C:\Users\james\miniconda3\envs\285J\python.exe`. Never `/mnt/c`.

## What this session is

Meeting homework around the **15:30 ATM nearest-OTM** book, then **intraday re-pick**. Experiments are **diagnostics**, not a search for a better trading rule.

Two notebooks:

| notebook | what it is |
|---|---|
| `notebooks/atm_straddle_rv_iv.ipynb` | paper 15:30→official close book. IR, \(R\sim a+bs\), buy-day plot, condors, non-compounded P&L (replaced old compound-wealth §10). |
| `notebooks/atm_straddle_experimental.ipynb` | ensembles / extra weights. Did **not** beat blk2 UM. |
| `notebooks/atm_straddle_intraday.ipynb` | **this session's focus.** Re-pick nearest-OTM every 30-min bar; next-mid 30-min hold; 16:00 cash-settle. |

Writers: `notebooks/_write_0dte_nb.py`, `_write_0dte_experimental_nb.py`, `_write_0dte_intraday_nb.py`. Lib: `notebooks/atm_straddle_lib.py`. Runner: `notebooks/_run_nb.py`. Output-carry: `notebooks/_nb_io.py`.

Regen (from `notebooks/`):

```
C:\Users\james\miniconda3\envs\285J\python.exe _write_0dte_intraday_nb.py
C:\Users\james\miniconda3\envs\285J\python.exe _run_nb.py atm_straddle_intraday.ipynb
```

`--force` on `_run_nb.py` re-runs even if every code cell already has outputs. Markdown-only writer runs **carry outputs** from the existing ipynb so you should **not** need a kernel.

## Intraday book (locked decisions)

**This notebook re-picks.** It does not mark one 15:30 straddle through the day. Hold-to-close-by-entry-hour ("when do I enter?") is **not built**.

- Strikes: new nearest-OTM call+put at every 30-min bar \(t\).
- \(R_t\): next mid of **those** \(K\), over entry, minus 1. 16:00 has no next mid → cash-settle vs `^GSPC` close (different \(R\); park or score separately).
- Paper 15:30→**official close** vs this book's 15:30→**16:00 mid**: not the same. Score both. Last executed numbers: always-short Sharpe **−0.016** (next-mid) vs **+0.346** (paper close), n=1279.
- Signal \(s=\widehat{RV}-\mathrm{IV}_{30}^{2}\). \(\widehat{RV}\) is **next-bar** smeared variance (same smear as the close book, every row kept). \(\mathrm{IV}_{30}^{2}=(\mathrm{IV}^{\mathrm{hr}})^{2}/2\).
- Vendor `impl_volatility` is **Chris hourly SD** (~0.002 at ATM), **not** OM-annualized (~0.20). A $12 ATM straddle cannot be 0.2% vol. \(\mathrm{Var}(30\mathrm{min})=0.002^{2}/2\sim 2\times 10^{-6}\).
- Default scored panel starts **10:00** (vendor \(S\) live). 9:30 vendor `underlying_price` is all NaN. Live 9:30 mids on ~40% of days (519/1279); GSPC-Open ATM straddles ~518. **Not in the default book.** Bid/ask do not add rows (`frac_mid_from_ba=0`); usable bid/ask ≡ live mid.
- Spot: `stamp_spot` = dropna then `.first()`, not median.
- Missing next-mid legs: **0** dropped.

**Three rules** (only \(q\) changes; \(R'_t=q_t R_t\)):

- Always short: \(q\equiv -1\).
- Long-short: \(q=\mathrm{sign}(s)\), \(s=0\mapsto -1\).
- Unit-median: same sign, \(\ell=\mathrm{clip}(|s_t|/\mathrm{med}_t,3)\), \(\mathrm{med}_t\) = expanding median of \(|s|\) on **all prior 30-min bars** (not 15:30-only), `.shift(1)`, warmup \(63\times\) (median bars/day). Implementation: `asl.lagged_expanding_median` + `asl.um_leverage_vs_lagged_scale`.

## Pooled Sharpe (do not regress)

Old `rule_row` `Sharpe_ann` = \(\overline{R'}/\mathrm{sd}(R')\times\sqrt{252}\) on the stacked ~11,254 bars. That treats each 30-min row as a day. **Wrong** for the pooled table. Also not \(\times\sqrt{13}\).

**Pooled `Sharpe_ann` in §6 is now:**

1. \(R'_t=q_t R_t\)
2. \(R^{day}_d=\sum_{t\in d} R'_t\) (non-compounded)
3. \(\mathrm{Sharpe}_{ann}=\overline{R}^{day}/\mathrm{sd}(R^{day})\times\sqrt{252}\)

`t_mean` on that table is the **daily** \(t\), so \(t=\mathrm{Sharpe}_{ann}\times\sqrt{n_{\mathrm{days}}/252}\). `Sharpe_bar` is unannualized 30-min \(\mu/\sigma\).

Per-clock tables in §7 **are** daily series (one bar per expiration day at that clock). \(\sqrt{252}\) is correct there. Use those Sharpes for "if I only entered at 11:30".

Last executed pooled table (`results/atm_straddle_intraday/rule_table_intraday_blk2.csv`; n≈11,253 bars, 866 days, blk2):

| rule | mean (30m) | mean_daily | Sharpe_bar | Sharpe_ann (daily) | pct_buy |
|---|---|---|---|---|---|
| always short | 0.078 | 1.017 | 0.118 | **7.31** | 0 |
| long-short | 0.068 | 0.887 | 0.103 | **6.34** | 45 |
| unit-median | 0.199 | 2.582 | 0.118 | **7.35** | 45 |

16:00 cash-settle **contaminates** pooled always-short / UM (always-short 16:00 mean R' ~0.89). Per-clock always-short Sharpe_ann (use these): 11:30 **2.21**, 15:00 **2.51**, 15:30 **−0.29**, 16:00 **7.98** (different \(R\)).

## Copy rules the human already enforced

User-facing prose in the **intraday** notebook:

- **straddle** / 30-min bar / two legs — not "package"
- **clock time** / 30-min bar — not "stamp" (keep column `timestamp`, `pd.Timestamp`, `asl.stamp_spot`)
- No "hold the 15:30 package", no "(10:00 or 15:30)", no "That is a half, not times twelve."
- §6 must be as descriptive as the RV–IV **Rule table** section (rules, column glossary, how Sharpe is computed). That markdown is in `_write_0dte_intraday_nb.py` §6.
- §7 should **not** recap §6. Plot caption: 16:00 is cash-settle vs close; other bars are next-mid 30-min holds.
- RV–IV / experimental notebooks still say "package". Only change those if asked.

## Speed / regen (in the writer, verify execute)

Bottleneck was 8.2M 0DTE rows + Jupyter + loading **seven** yhat panels when only **blk2** is scored.

In `_write_0dte_intraday_nb.py` (may not be in the last *finished* execute):

- Book cache: `results/atm_straddle_intraday/cache/book_{size}_{mtime_ns}.parquet`. On hit, skip pick + 8M next-mid join.
- Pick: merge \(S\) (not `index.map`, 18s→0.8s), ATM band ±100 then `idxmin` (full OTM sort was the slow path; band missing-vs-full pick = 0).
- Next-mid: merge the 16k keys into quotes, not rename-copy the 4M-row legs first.
- Forecast cell loads **blk2 only**.
- `_nb_io.carry_outputs`: writer copies outputs onto unchanged code cells.
- `_run_nb.py`: skip execute if every code cell already has outputs; `--force` to rerun. Cell timings print to stdout.

**In-flight at handoff:** `_run_nb.py atm_straddle_intraday.ipynb` started after the speed + UM-plot writer changes (task still running ~5 min when this file was written). Check:

- `notebooks/atm_straddle_intraday.run.log` ends with §7 tables and `status=0` in the terminal log
- `results/atm_straddle_intraday/cache/book_*.parquet` exists
- §7 figure has **two** series (always short + unit-median VRP)

If that process died, from `notebooks/`:

```
C:\Users\james\miniconda3\envs\285J\python.exe _write_0dte_intraday_nb.py
C:\Users\james\miniconda3\envs\285J\python.exe _run_nb.py atm_straddle_intraday.ipynb
```

Second run should hit the book cache and be much faster. First run still reads the 8.2M chain (~2s) and the 9:30 diagnostic.

Further speed if still slow: skip chain load when book cache hits (9:30 cell still wants `chain` — cache those prints or skip). Do not load a0/lgbm/xgb/lasso/enet in this notebook.

## §7 plot (requested, in writer)

Always-short **and** unit-median mean \(R'\) vs clock time on the same axes. Filename still `mean_by_entry_hhmm_as.png`. Confirm the executed notebook actually shows both lines.

## Not done (do not pretend otherwise)

- **Hold-to-close by entry hour** ("when do I want to enter?"). Re-pick + next-mid is what exists.
- `/sync` of this branch. Local HEAD `f038792` is already on origin; **notebook edits are uncommitted**.
- RV–IV "package" wording.
- Committing `data/spxw_*.parquet` or `results/spxw_pnl/yhat_*.parquet`.

## Other lab results (do not re-run unless asked)

- Ensembles do not beat blk2 UM Sharpe ~0.90 on the **15:30 close** book.
- \(R\sim a+bs\): \(b<0\).
- Buy days 13–16%, mean \(R\) on buy days < 0. Buy-day plot: calendar dates on x, **no** FOMC/ME overlay.
- Iron condor UM Sharpe ~0.43.
- IR: \(R^a=R^p-R^{AS}\); `te_daily`=`std(R^a,ddof=1)`; `IR_ann`=`mean/std*sqrt(252)`; `t_active`=`mean/std*sqrt(n)`. Lives in the RV–IV notebook.

## Data

- Chain: `data/spxw_chain.parquet` (0DTE filter cached at `results/atm_straddle_intraday/cache/chain_0dte_*.parquet`, ~8.2M rows, 1279 full sessions after half-session drop).
- Yhats: `results/spxw_pnl/yhat_blk2_fomc1.parquet` (intraday scores this one).
- GSPC OHLC cache: `results/atm_straddle_intraday/cache/gspc_ohlc.parquet`.
