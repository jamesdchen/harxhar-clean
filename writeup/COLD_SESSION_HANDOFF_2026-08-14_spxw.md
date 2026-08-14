# Cold session handoff — 2026-08-14 (SPXW / QLIKE increment / 0DTE swap)

Paste this to the next session: *Continue from `writeup/COLD_SESSION_HANDOFF_2026-08-14_spxw.md`. Do not re-trade the smile as the exog result. Do not submit hedge.*

## Repo

- Path: `C:\Users\james\CC Allowed\harxhar-clean`
- Branch: **`paper2`** (tracks `origin/main` in status text; **`origin/paper2` exists** and is the paper remote)
- HTTPS: `https://github.com/jamesdchen/harxhar-clean.git`
- HEAD at handoff: **`cd4d3cd`** *Add SPXW 0DTE variance-swap experiments and harvest ridge/lasso table.*
- **`/sync` did not finish.** Commit is local only (`paper2` ahead of `origin/main` by 2). Push `origin paper2` still needed. Do **not** `stash -u` the whole tree (data/results will hang).
- Authors / paper: James Chen (UCLA Math), Austin Pollok, Chris Jones (USC Marshall), Mihai Cucuringu (UCLA Math). “Working draft — not for citation.” Overleaf zip, not GitHub, for the paper. Native Windows PowerShell, not WSL. Paths `C:\Users\james\...` never `/mnt/c`.

## What the paper result is

Exog (blk2 = two-block ridge) is a **better one-bar / remaining-RV forecast than a0** (OLS HAR+calendar). It is **not** a better options-vs-IV signal.

Scored unification (`results/unification_scores.csv`, n=273,554, 100/100):

| arm | QLIKE | Duan QLIKE | DM vs a0 |
|---|---|---|---|
| a0_ols_har | 0.23353 | 0.23165 | — |
| **blk2_user** | **0.22941** | **0.22489** | **−9.0** |
| blk2_doc | 0.22775 | **0.22383** | −9.6 |

The “0.22350 vs 0.22519” line from an older harvest is **stale**. 0.22383 is blk2_doc Duan. Do not quote 0.219 on 91/100-chunk arms as comparable.

## Three prices (same 10:00→16:00 wiggle)

Typical day: RV ≈ 1.7e-5, **a0 ≈ 1.7e-5**, smile/MFIV ≈ **7.5e-5** (~6×). Figure: `results/spxw_pnl/explain_one_day.png`.

- **Smile** = VRP. Always-short paper +11.7, short **actual strip** +6.0. Not the exog contribution. Professor will say “selling vol.”
- **a0** = fair HAR strike. That is the incumbent.
- **blk2** = a0 + revision. Only used to **size** a swap struck at a0.

ATM weekly `sign(f−IV)` : a0 and blk2 agree ~98%, 3.6% long, Sharpe/trade ~+0.04–0.06 vs always-long ~−0.03. H-invariant algebra (`sign` independent of h). Wrong product for this edge.

## Best paper frame (apples-to-apples)

**Product:** 0DTE variance swap, 10:00 ET → 16:00 PM settle. Both models forecast remaining RV. Score = **QLIKE**. Trading analogue: swap **struck at a0**, size \(N=\mathrm{blk2}-a_0\) (linear) or **\(N=(\mathrm{blk2}-a_0)/a_0^2\)** (QLIKE Hessian, 1% clip).

| book | all | drop 2020 | daily 0DTE (May 2022+) |
|---|---|---|---|
| ΔQLIKE 10:00→close | +1.11 | **+1.11** | **+1.08** |
| linear \(N(\mathrm{RV}-a_0)\) + D10 | +1.05 | +0.32 | **+0.09** |
| **Hessian** \(N/a_0^2\) | +3.17 | **+3.09** | **+2.79** |
| one-bar ΔQLIKE, RTH 10–16 | +1.16 | +1.26 | **+6.47** (2022+) |

**2020 is 12% of QLIKE gain and 95% of linear-increment dollars.** D10 helped COVID and **zeros** the linear swap after daily 0DTE. Do not lead the paper with linear+D10. Lead with QLIKE; Hessian size is the swap that survives eras.

May 2022 (every-weekday 0DTE) is the structural break, not 2020. Pre-register when more tape arrives: all 0DTE days / drop 2020 / weekly-0DTE vs daily-0DTE. Ask the professor for **every historical expiration-day chain** (Friday weeklies back as far as they go). Yhats already start ~2004.

## Options replication

Log-strip (BJN/VIX): OTM puts \(K<F\), OTM calls \(K>F\), \(w=2\Delta K/K^2\), hold to 16:00, **cash-adjust** \(N(\mathrm{MFIV}-a_0)\) so strike is a0. Keep 0-delta rows if mid is live. **Not** an ATM straddle.

Identity: \(\mathrm{RV}-a_0=(\mathrm{RV}-\mathrm{MFIV})+(\mathrm{MFIV}-a_0)\).

- Short the **actual strip** = sell those OTM contracts at 10:00, pay intrinsic at 16:00 (VRP, +6).
- Increment book = hold \(N\) units of the strip + cash so strike is a0.

**Every-bar MTM** (just scored, `experiments/spxw_mfiv_everybar.py`, n=10,425 RTH bars):

| book | all | drop 2020 | daily-0DTE |
|---|---|---|---|
| paper \(N(\mathrm{RV}-a_0)\) | **+10.8** | +10.6 | +10.9 |
| listed MTM \(N(\mathrm{RV}+C_{t+1}-C_t)\) | +1.3 | **−0.6** | **−7.2** |
| last-bar expire (day) | paper +1.8 / MTM +0.7 | | |

Every-bar options are mostly **vega** (smile mark). Only **expiry** (10:00→close or last bar) can look like paper. Paper every-bar Hessian is the unused harvest; it is **not** listed except at 16:00.

## On disk (do not rebuild unless broken)

**Data:** `data/spxw_chain.parquet` (8.2M rows, **all expiration-day / 0DTE**, 18,074 stamps, 1,291 days, 2020-01-03–2025-12-31). `data/SPXW.csv` source. Do not drop 0-delta.

**Yhats:** `results/spxw_pnl/yhat_{a0,blk2,tree00,tree16}.parquet`

**Scripts (committed in cd4d3cd except everybar):**
- `src/data/spxw.py` — load, ATM, exit/settle
- `experiments/spxw_horizon_pnl.py` / `spxw_fast_remaining.py` — ATM straddle h=1–13
- `experiments/spxw_complete_table.py` — mid-IV resign table
- `experiments/spxw_qlike_harvest.py` — one-bar QLIKE location
- `experiments/spxw_mfiv_toclose.py` — 10:00 strip, 4 shards
- `experiments/spxw_mfiv_everybar.py` — **uncommitted** every-bar MTM
- `experiments/spxw_a0_vs_blk2.py`, `spxw_two_sleeve.py`, `spxw_toclose_varswap.py`

**Tables/figs:** `results/spxw_pnl/{complete_table,qlike_harvest,mfiv_toclose,two_sleeve,a0_vs_blk2_strategy,paper_frame,everybar_mtm}.csv` and matching pngs. `explain_one_day.png` is the one-panel explainer.

**Scored unification:** `results/unification_scores.csv`

## Cluster (as of 2026-08-14 evening)

Refill of **incomplete unification arms only**. No new campaign. **Do not submit hedge** (`probe_hedg` write path was unsafe; local `probe_hedge_linear.py` is now arm-conditional).

- CARC `jc_905@discovery2.usc.edu` (`id_carc`): submitted 33 incomplete arrays (`11055746+`). Last seen ~30 R + 55 PD. `unif_tree_expert_*` leftover + refill. QOS ~100 jobs.
- Hoffman2 `jamesdc1@hoffman2.idre.ucla.edu` (`id_hoffman2`, `ssh -o ProxyJump=none`, **`bash -lc` for qstat**): 91 arrays `14343842`–`14343982`. Also old `probe_alli` 2/3/5/6 and `probe_hedg` task 2 — leave hedge alone.
- Journal `running_where=[]` is **not** live state (raw qsub, not hpc-agent).
- Incomplete npz: many blk\* at 91/100; trees almost done except expert_19.

## Constraints / do not

- Do not invent new cluster campaigns.
- Do not submit 6-way hedge until write path is arm-conditional **on the cluster PIDs** (local script is fixed; those jobs are old).
- Do not claim blk2 “trades options” or beats HAR on ATM weeklies.
- Do not put mid-IV in the increment strike.
- Do not lead with linear+D10 or 2020 dollar PnL.
- Native Windows; no `/mnt/c`; HTTPS remotes; no passwords in repo.

## Next session — pick up here

1. Push `paper2` (`git push origin paper2`) without stashing `data/` / `results/`.
2. Commit `experiments/spxw_mfiv_everybar.py` if keeping the every-bar result.
3. If professor sends pre-2020 expiration-day chains: same 10:00 strip + remaining yhats; three-sample table (all / drop 2020 / weekly vs daily 0DTE); **QLIKE + Hessian**, not linear+D10.
4. Optional: bid-ask on the strip (mids → +6 short-VRP will shrink).
5. Paper tex: QLIKE a0 vs blk2 on the 10:00–close claim; one sentence on the a0-struck increment; smile/VRP in a footnote at most.

## Environment notes (not the science)

- PowerShell function `qwen` in `OneDrive\Documents\WindowsPowerShell\Microsoft.PowerShell_profile.ps1` launches **Claude Code** against Qwen Cloud Token Plan, **not** npm Qwen CLI. Models set to `qwen3.8-max` (not preview). Restart the CC window.
- `DASHSCOPE_API_KEY` is in user env + `~\.qwen\.env` (Coding Plan `sk-sp-…`). Do not commit it.
