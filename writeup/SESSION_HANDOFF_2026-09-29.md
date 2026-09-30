# Session handoff — 2026-09-29 (read this first after /clear)

Worktree `C:\Users\james\CC Allowed\harxhar-0dte-professor` (Git Bash `/c/Users/james/CC Allowed/harxhar-0dte-professor`),
branch `insurance-selling-2026-09-19`, remote `https://github.com/jamesdchen/harxhar-clean.git`.
Last sync `a8c957c` (pushed). Another Claude session also commits card/live studies (84-90) on this branch — stage only your own files.

---

## 1. The user's instructions for this session (verbatim)

> I want all of the following done by conducting opus 5.5 agents in as much parallelism as possible. You have access to usc carc. Please do not use any hpc-agent tools (only use the templates if they are helpful), instead do as much of the hpc work as you can by yourself (again, you can use hpc-agent code to make things easier, but avoid all of the restrictions and permissions that was designed to harness a dumber model). If any of the work is done already, I want you to note it and aggregate the completion all of the finished products in one list that you check off as the agent team chugs along. For feature importance, my other professor suggested the following measures:
>
> [Image #3] [Image #4]   <-- NOT carried over by /clear: ask the user to re-attach these two images (the professor's feature-importance measures) before starting the feature-importance work.
>
> Here are the tasks to be completed.
>
> "Finish analysis of the close (3:30-4pm)
> building a model that is specific for 3:30-4 forecasting; you did most of this, just need the trees
> weights that are a function of the VRP (e.g. w_t = f(VRP)); this could be proportional VRP, Kelly criterion, etc.
> later, we will need to rerun all of the forecast models on the closing strategy (i.e. a master table for the closing strategy); this will allow us to compare directly apples to apples with QLIKEs, and will likely need to do this for 3:30-4pm specific models, as well
>
> Explain the return sequence
> P&L analysis (sort of in-line with what you have done, but maybe a little more); we can think this through
> correlation with other markets / asset classes (I didn't get what you initially showed today with SPX, but it grew on me and the correlations and P&L are both interesting; we might not use it for this paper, but it is something to keep in mind)
>
> Explain the models and features of the models
> feature importance (we described some methods today, but we can expand and think more on this as well)
> trees vs linear models (maybe think about the weak but dense vs sparse signals explanations)
> link model performance and features with return profitability; i.e. can we link model performance and feature importance with the returns P&L
> etc. (we will need to think more)
>
> WRITE UP
> we need to make the story, evidence, comparisons, and more much tighter and clear (again, as apples-to-apples as possible moving one thing at a time)
>
> Create a running appendix
> this is where you should stick all your side and tangent analysis of additional empirical facts, either from models or option returns
> this can include (briefly from my memory)
> every 30 min analysis
> any time-till-close analysis
> the time-varying 30min Sharpes
> whatever you're finding with the 2025, 2026 analysis
> anything else we've found interesting but detracts from the main story
>
> I think you might consider still running, or at least writing them down, the different variations to the strategy that I delineated by writing on the screen a week or so ago (including different combos; e.g. strangles, butterfly spreads, iron condors, etc.)
>
> I still think you should run an LSTM just to have a NN in the story, and see if there is anything interesting there"

Added by the user (2026-09-29): **use Hoffman2 as well — treat CARC as the main workhorse and Hoffman2 for faster, lighter jobs** (canaries, scorers, small sweeps, one-off re-runs; big fleets on CARC).

---

## 2. Environment and ground rules learned this session

- **Python**: `/c/Users/james/miniconda3/envs/285J/python.exe` (the msys `python` on PATH has no pandas). pdflatex/pdftoppm/pdftotext on PATH.
- **CARC**: native OpenSSH `/c/Windows/System32/OpenSSH/ssh.exe -o BatchMode=yes usc-discovery '<cmd>'` (Git Bash ssh can't reach the agent; VPN required). Project root `/scratch1/jc_905/harxhar-subsection`. Partition QOS `normal`: 100 running jobs / 5000 submitted per user, 2000 CPUs per user, MaxArraySize 5001.
- **Hoffman2** (lighter/faster jobs): same native ssh, alias `hoffman2` (user `jamesdc1`; data-transfer node alias `h2dtn`) — key login works non-interactively (`-o BatchMode=yes`; checked 2026-09-29). Use the ALIASES: `permissions.deny` blocks `ssh`/`scp` to the full `*hoffman2.idre.ucla.edu` hostnames. Project root `/u/scratch/j/jamesdc1/harxhar-subsection` (exists; scratch is purged periodically — check files before relying on them, re-ship what's missing). Scheduler is SGE/UGE, not Slurm: `qsub -terse -t 1-N` arrays, `-hold_jid` dependencies, `-l h_rt=HH:MM:SS,h_data=16G`; env `source /u/local/apps/anaconda3/2023.03/etc/profile.d/conda.sh && conda activate hpc-pi`. Templates: `cluster/submit_subsection.sh`, `cluster/subsection_array.sh`, `cluster/subsection_score.sh`, `cluster/submit_lassofix.sh`, `cluster/lassofix_*.sh` (the 2026-09-20/21 per-bar campaign ran there; results under `results/linear_subsection/*_hoffman2/`). Submit through a `cluster/submit_*.sh` script over `ssh.exe -o BatchMode=yes hoffman2 "cd /u/scratch/j/jamesdc1/harxhar-subsection && bash cluster/submit_<x>.sh"` (pre-approved, see next bullet). Ship by tar over native ssh + md5 check, as the CARC ship scripts do. Hoffman2's CPU/queue limits are tighter than CARC's — keep its jobs small.
- **Cluster ssh discipline (learned 2026-09-30)**: last night's "VPN drops" were not the VPN (AnyConnect stayed connected; its 30-min "idled due to an error … Rekey reconnect complete" lines are the normal rekey). CARC's login node stopped answering SSH banner exchanges while ~10 agents polled in parallel (sshd connection-rate throttling / load). Rules: ONE poller per cluster writing a status file the agents read; poll every >= 10 min with backoff; never a burst of parallel ssh sessions. `~/.ssh/config` now has `ConnectionAttempts 3`, `ConnectTimeout 30`, keep-alives on `usc-discovery`, and a fallback alias `usc-discovery1` (discovery1.usc.edu, same project root, same allow-rule prefix). A `10.72.*` route is NOT a valid VPN test (Split-Exclude mode installs none); test port 22 with a TCP connect instead. The USC profile forces DisconnectOnSuspend (headend-locked): keep the laptop awake during runs (sleep is already "never" on AC).
- **Submitting**: the hpc-agent hooks were REMOVED from `~/.claude/settings.json` (2026-09-29; only the ruff/mypy `py-lint-guard.py` PostToolUse hook remains). The user PRE-APPROVED cluster submissions via the allow rules `Bash(/c/Windows/System32/OpenSSH/ssh.exe *usc-discovery*submit_*.sh*)` and `Bash(/c/Windows/System32/OpenSSH/ssh.exe *hoffman2*submit_*.sh*)` — so submit through a `submit_*.sh` script run over that ssh form (a raw `sbatch` over ssh is not covered and the auto-mode classifier blocks it). `qdel`/`scancel` are still in `permissions.deny` (cancelling jobs needs the user).
- **Do not use hpc-agent MCP tools/skills** (user order). Reuse its code/templates only if helpful. The working cluster pattern this session = `cluster/slurm/ship_<x>_carc.sh` (tar over native ssh + md5 check of shipped files and dependencies) → `submit_<x>.sh` (one-chunk canary → fleet arrays held `afterok` + a CANARY_OK flag → scorer `afterany`). Mirror `ship_trees_carc.sh` / `submit_trees.sh` / `*treestuned*` / `*restofday*`.
- **Local memory guard**: ~3 chain-sized pandas jobs at once (`data/spxw_chain.parquet`) get killed — at most one chain-sized job at a time across all agents.
- **Git Bash gotchas**: `cmd & cmd & wait` loses shell variables in the background subshells (write a script file instead); a stray `cat > file` with no heredoc hangs; git pathspec globs (`'dir/*.csv'`) match recursively — check the staged list.
- **Commits**: pre-commit = ruff check --fix, ruff format, mypy on staged .py (in parallel). Do NOT reformat files already shipped to CARC (specs, `score_*`/`reduce_*`/`check_*` scripts in the ship manifests) — keep committed == shipped. Never commit `*.npz`, logs, per-arm result folders, LaTeX aux, `writeup/main.pdf`, `results/spxw_pnl/smile_rv_daily.parquet` (not ours). Commit trailer lines per the system reminder.
- **Auto memory is OFF** (`autoMemoryEnabled: false`); the user dislikes memory bookkeeping — put state in handoff files / the repo instead.
- **Professor-facing wording**: the rule is **sign(s)** (never "long-short volatility"); **straddle** (defined once: nearest out-of-the-money call + nearest out-of-the-money put, same-day expiry, one position) — never "package"; **last-30-min trade** — never "paper trade"; no lab/cluster names (no CARC/Hoffman2/blk2/tw2000) in prose; features in figures by design column names (`har_ma_*`, `adj_vix_ma_*`, …) with a key; park (comment out, dated) rather than delete; every number in prose from executed output; no magic numbers (named, justified constants; no ad-hoc clips/floors). Panel stamps are naive ET, bar-END labelled (row 16:00 = the 15:30–16:00 bar, forecast issued at 15:30); chain/spot stamps are true UTC. Nothing issued after the entry time may enter a forecast (summing within-window per-bar forecasts looks ahead — the 2026-09-01 audit took a DH book from Sharpe 9.1 to an honest 0.8).
- **Two scorers, two Sharpes**: the rv_iv notebook recalibrates on all 13 session bars; the research scorers (`compare_mfiv_harlag.py`, `score_trees_subsection.py`) recalibrate the 16:00 bar alone — e.g. per-bar ridge live-feasible 1.68 (notebook) vs 1.90 (research), all features 1.77 vs 1.67. Never mix them in one comparison; the master table must pick ONE.

---

## 3. Live cluster state (at handoff)

| Campaign | Jobs | State | Next step |
|---|---|---|---|
| Causally tuned per-bar trees (widened grids) | canary 12464029, GB 12464030 (510), RF 12464031 (255), score 12464032 | canary RUNNING, fleets held | when `results/linear_subsection_trees_tuned/SCORED` exists: pull CSVs (not per-arm/npz), run `experiments/score_trees_tuned.py` locally vs the linear arms (`--linear-root results/linear_subsection/arms_hoffman2` for all_features), read the grid-edge table |
| Direct rest-of-day model | 12344061-63 | DONE | trading test (see 4.1) |
| Untuned per-bar trees | 12339870-73 | DONE, scored locally | superseded by tuned run for the headline |

**Uncommitted at handoff**: `specs/causal_tune_trees_tuned.py` (grid widening; this exact file is what CARC is running, md5 `5c8ac99b0a28`) and this handoff file — commit both.

---

## 4. Master checklist (the professor's tasks) — status at handoff

Legend: [x] done · [~] partial / first pass · [ ] not started. Paths are repo-relative.

### A. Finish analysis of the close (15:30–16:00)
- [x] 15:30-specific linear models: per-bar ridge / lasso / elastic net, 2000-session window, causal penalty tuning; buckets HAR + calendar, all features, live-feasible (16), plus VIX-only family. Defined in the rv_iv notebook ("What per-bar ridge means"); tables in `notebooks/atm_straddle_rv_iv.executed.ipynb` §10/§17 and `writeup/rule_by_strategy_standalone.pdf` (per-bar rows under their own heading; the paper's table untouched).
- [~] 15:30-specific TREES: untuned per-bar LightGBM/XGBoost/RF DONE (they lose to linear: QLIKE vs ridge, all features +1.7 % [−0.5, +7.2] / +4.3 % / +7.9 %; live-feasible +1.5 / +4.6 / +6.8 %; 15:30 sign(s) Sharpe all features 1.35 / 1.26 / 1.33 vs ridge 1.67, live-feasible 1.69 / 1.70 / 1.58 vs 1.90 — research scorer). Forecast tables `results/spxw_pnl/yhat_subtree_*.parquet`; scores `results/linear_subsection_trees/rescore_local/`. Causally TUNED trees RUNNING (§3) — spec `specs/causal_tune_trees_tuned.py` (tuning every 250 sessions on a 125-session tail after a 25-session embargo, MSE-selected + QLIKE-selected path saved; 32 candidates of grids 700 / 2800 / 112 incl. the widened regularization edges; gates: identity 0, chunked = unchunked, pool = serial, ×50 perturbation).
- [~] Weights w_t = f(VRP): proportional-to-s and rank sizing DONE in rv_iv §10 "Sizing by the size of the forecast gap" (`results/atm_straddle_0dte_1530/vrp_sized_rules.csv`): proportional loses to sign(s) for all 15 forecasts, sign-kept rank ties (median +0.03), centred rank loses; no interval above zero (agrees with the earlier finding that the sign carries the signal). NOT done: Kelly-type w_t = f(VRP) (e.g. fraction from the forecast's implied edge / payoff distribution, causal), and the "fixed fraction" §15 is not VRP-dependent. Prior: `writeup/intraday_proposals/71_conviction_sizing.py`.
- [ ] MASTER TABLE for the closing strategy: every forecast (paper's 8, per-bar linear 7+, tuned trees, rest-of-day direct where relevant, VIX-only family, LSTM) on the SAME 866 days / same rows, one scorer, with 16:00-bar QLIKE (+ DM vs a fixed reference) and sign(s) Sharpe mid/crossed with paired intervals. Pieces exist but are spread over the notebook, the standalone PDF and the research scorers.

### B. Explain the return sequence
- [~] P&L analysis: rv_iv §13 (P&L without compounding, mid/crossed/points/dollars/margin), §14 (IR vs always short), §15 (fixed fraction of wealth); month-end concentration known (the 15:30 sign(s) is a tail trade — ~20 of 866 days carry most of the P&L; the ridge shorts most month-end closes; the long straddle on month-end closes pays, study 83: R +0.432, t 2.62). Needs: a proper decomposition of the return sequence (tail days, month-ends, FOMC, regimes, drawdowns, hit-rate vs payoff), apples-to-apples across forecasts.
- [x] Correlation with other markets (first pass): rv_iv §18 — trade vs S&P same window 15:30→close −0.11..+0.08, vs |that move| −0.23..−0.08, close-to-close −0.04..+0.03; "What the trade's return moves with" table (VIX, NDX, RUT, 10y, TLT, HYG, DXY, S&P RV−IV; `results/atm_straddle_0dte_1530/trade_return_comoves.csv`, closes cached in `cross_asset_daily_closes.parquet`): 1 of 26 intervals excludes zero (|S&P 15:30→close| −0.11). Note: that cell drops exact-zero changes as repeated prints (17 of 812 VIX days). Portfolio weights §18 (one table, weight from past days only, six-step recipe): block-diagonal ridge risk share 0.73, median $4.32 premium per $100 S&P, S&P 0.40 → 1.44 on 614 days, gain 1.04 [−0.46, +2.55]; every interval includes zero. Plot has a trade-alone risk-unit line.

### C. Explain the models and features
- [~] Feature importance (first pass) — `writeup/model_diagnostics_1530.pdf` (12 pp; `experiments/model_diagnostics_1530*.py`, `writeup/make_model_diagnostics_1530_tex.py`): standardized weights β·sd over every refit; contributions β(x−x̄) (= linear SHAP, checked against the shap library); rolling contribution shares (63/126 sessions); regime breaks (every weight step is the penalty switch; target-on-contributions break sup-Wald 29-40, p ≤ 0.008, date not pinned; Feb–Apr 2020 differs); per-bar LASSO VIX path (VIX enters 2021-06-21 at the penalty re-choice, +0.14, vs negative VIX3M long means; held penalty keeps it all along); tree TreeSHAP vs linear at 16:00 (har_ma leads all: ridge 30 %, lasso 51 %, trees 57–66 %; VIX/VIX3M 5–7 % linear vs ≤ 2 % trees; tree–linear rank corr 0.56–0.72 live-feasible, 0.26–0.51 all features). TODO: the professor's measures (Images #3/#4 — re-attach), tuned-tree importance, permutation / drop-column importance, stability across regimes.
- [~] Trees vs linear / weak-dense vs sparse: evidence exists (ridge spreads weight over correlated return-size series — absolute return 15 % of ridge vs 3–6 % of trees; lasso concentrates on har_ma; trees lose on QLIKE and trade; ridge ≥ lasso ≥ enet on the trade). TODO: a clean argument + tests (e.g. signal density: number of inputs needed for x % of the forecast variance; ridge vs lasso vs trees as sparsity varies; synthetic-free, on the real design).
- [~] Link model performance / features to P&L: diagnostics pp 1–4: weights do not line up with P&L (1 of 138 weight intervals excludes zero vs ~7 by chance); contributions barely (14/138 all days, chance on non-month-ends); the forecast level correlates −0.014 with P&L; the buy/sell decision is driven by implied variance (−0.25) and the diurnal scale (+0.33), not the regression part (+0.04). QLIKE gains don't imply trade gains (VIX-only: best 16:00 QLIKE, worse trade; multi-horizon M3: best QLIKE, worst trade). TODO: formalize (QLIKE-vs-Sharpe across all models in the master table; which loss predicts P&L — e.g. sign accuracy on tail days).

### D. Write-up
- [ ] Tighten story / evidence / comparisons, one change at a time. The paper is `writeup/main.tex` (sections under `writeup/sections/`, the close-option results in `sections/results_close_option.tex`, which reads `writeup/generated/table_rule_by_strategy_paper.tex` — currently the 8 paper forecasts only). Decide the headline forecast and ONE scorer first.

### E. Running appendix (collect side analyses; none assembled yet)
- [ ] Every-30-min analysis: `notebooks/atm_straddle_intraday.executed.ipynb` (30-min re-pick), `writeup/rule_by_strategy_intraday_index.pdf` (non-hedged; NOT rebuilt with 09:35).
- [ ] Time-till-close: `notebooks/atm_straddle_intraday_holdclose.executed.ipynb` (delta-hedged hold-to-close, §4b hedge note; §6 pooled always short 2.93 / sign(s) 2.40 mid; §8 per-entry-time; §8c confusion matrices; §8d rest-of-day fitted maps M1–M3 trade WORSE than the current signal: pooled mid 1.70 / 1.15 / 0.86 vs 2.30), `writeup/rule_by_strategy_dh_holdclose_index.pdf` (17 pp incl. 09:35: DH always short 3.11 / 2.44, sign(s) blk2 0.68 / 0.02).
- [ ] Time-varying 30-min Sharpes: DH per-clock always-short (10:30–11:00 ≈ 4.2 → 15:30 0.24) and sign(s) per clock (15:30 1.33 vs always short 0.24) — tables in the holdclose notebook §8 and the DH PDF.
- [ ] Direct rest-of-day model (`results/linear_subsection_restofday/score_rest_of_day.csv`): like-for-like QLIKE vs the one-step construction RV̂/w: median −31..−37 % live-feasible, −36..−44 % all features, −18 % HAR + calendar, significantly better at 6–11 of 12 entry times, never worse; 15:30 identical by construction. Use the "plain" columns (the scorer's "causal" column adds a correction to the direct side only). TRADING TEST NOT DONE: `experiments/build_restofday_yhat.py` → holdclose notebook s = F_rem,direct − IV²·h.
- [ ] 2025–2026: the other session's studies — 85 (the daily card has no edge after the research span: SPX 2024-05..2025-12 −0.013/day, t −0.31), 86 (no decision time 15:00–15:50 beats 15:30), 87 (post-2024 input changes don't explain the loss), 88/90 (walk-forward refits / regime filters don't restore it), 89 (recent windows / trade-aligned target don't restore it). Month-end leg holds. See `git log` and `writeup/AUDIT_2026-09-18.md` (ledger).
- [ ] Other: VIX-only bucket study (rv_iv §17: the VIX alone wins the 16:00 QLIKE −5.9 % but trades worse 1.38 vs 1.68; VVIX/VIX3M add nothing; the free feed matches), per-bar vs pooled QLIKE (gain concentrated in the 14:00–14:30 bar on FOMC days), multi-horizon, month-end long, 15:30 decision-time studies (80–86), portfolio weights.

### F. Strategy variations (strangles, butterflies, iron condors, …)
- [ ] Not run this session. Prior work: iron flies / credit verticals / vol-target PARKED earlier (cost at every width); `results/atm_straddle_0dte_1530/condor_lab_strangle_rules_{straddle,strangle}_w{25,50}.csv` exist; the rv_iv writer has a parked iron-fly section. At minimum write them down (the user's on-screen list from ~a week ago — ask the user for it if not in the repo), then run on the same 866 days / same fills.

### G. LSTM
- [ ] Not started. Per-bar (16:00) and/or sequence model on the same design and window discipline (causal refits, 2000-session window or expanding), same target / recalibration, into the master table. CPU-feasible on CARC; check torch in the CARC env.

---

## 5. Other open items / known defects
- Multi-horizon writeup tables (`experiments/multihorizon_tex.py` from the committed `results/multihorizon/ttc_*.csv`) carry two defects: the rest-of-day target omits the 15:30–16:00 bar, and its "blk2" is the non-FOMC run. The fixed variant is `--through-close --blk2-file … --tag close_fomc1` (`results/multihorizon/ttc_close_fomc1_*`). The 14:00→14:30 QLIKE drop is the FOMC statement bar, not a bug.
- `data/releases.parquet` FOMC flags: Sept 2013 dated 09-19 (should be 09-18); the feed stops at 2023-11-01.
- Stored per-bar ALL-FEATURES forecasts have rare extreme values in the plain back-transform (16:00 ridge QLIKE 2.65); the deck's recalibration hides it; the master table must handle it explicitly.
- Holdclose notebook §8b QLIKE text ("daytime Sharpes are small") written before hedging — recheck. §6 printout label "30-min R'" is really the hold-to-close return.
- `notebooks/_run_nb.py` PNG fallback searches `results/atm_straddle_intraday/` for holdclose figures (latent: showed the wrong notebook's figure once).
- `notebooks/atm_straddle_lib.py` differs on CARC (unpinned; scorers only use PERIODS_PER_YEAR = 252).
- Non-hedged intraday index PDF not rebuilt with 09:35; `writeup/generated/*` for the DH PDF is current.
- Rest-of-day all_features lasso at 09:30 sat at the top of its penalty grid; winsor bounds for RV_rem reuse the per-bar window (bind less at early clocks).

## 6. Where things are
- Notebooks (writers → executed): `notebooks/_write_0dte_nb.py` → `atm_straddle_rv_iv.executed.ipynb` (15:30 deck, ~3 min via `notebooks/_run_nb.py <nb> --force`); `notebooks/_write_0dte_intraday_T_nb.py` → `atm_straddle_intraday_holdclose.executed.ipynb` (DH hold-to-close, ~10 min, chain-sized); lib `notebooks/atm_straddle_lib.py`.
- PDFs: `writeup/rule_by_strategy_standalone.pdf` (15:30, `make_rule_by_strategy_tex.py`), `writeup/rule_by_strategy_dh_holdclose_index.pdf` (`make_rule_by_strategy_intraday_tex.py --dh-holdclose`), `writeup/model_diagnostics_1530.pdf`.
- Specs: `specs/causal_tune_linear.py` (per-bar linear), `specs/causal_tune_trees.py` (untuned trees), `specs/causal_tune_trees_tuned*.py`, `specs/causal_tune_rest_of_day.py`. Stacking: `experiments/build_subsection_yhat.py`, `build_subsection_tree_yhat.py`, `build_restofday_yhat.py`, `build_vixonly_yhat.py`. Scoring: `experiments/score_linear_subsection*.py`, `compare_mfiv_harlag.py`, `score_trees_subsection.py`, `score_trees_tuned.py`, `score_rest_of_day.py`, `trees_vs_linear_pooled.py`.
- Ledger of studies: `writeup/AUDIT_2026-09-18.md`; study scripts `writeup/intraday_proposals/NN_*.py`.

---

## 7. Kickoff prompt to paste after /clear

> Read `writeup/SESSION_HANDOFF_2026-09-29.md` in full, then carry out section 1 with Opus 5.5 agents in maximum parallelism (subagents allowed and encouraged; process-level parallelism inside scripts too). I'm re-attaching the professor's two feature-importance images here: [attach Images #3 and #4]. First, build the single checklist from section 4 (keep it in a file you update as agents finish, e.g. `writeup/PROGRESS_2026-09-29.md`), mark what's already done, check on the tuned-tree campaign on CARC, commit the uncommitted spec + handoff, then dispatch the independent workstreams at once. Don't use hpc-agent tools; submit through `submit_*.sh` scripts over native ssh (pre-approved) — CARC (`usc-discovery`, Slurm) is the main workhorse, Hoffman2 (`hoffman2`, SGE) takes the faster, lighter jobs.
