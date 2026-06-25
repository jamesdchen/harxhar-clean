# Session notes — feature-engineering investigation (2026-06-23)

A long exploration of candidate features for the next-30-min `adj_RV` forecast.
The honest one-line summary: **HAR is hard to beat, the one "win" we chased was an
imputation artifact, and there is a real pipeline bug worth fixing.** Several
claims were made and then retracted on closer checking — those are flagged below
so they don't get re-believed.

---

## How to read the target/feature space (so the numbers below make sense)

- **Target** = `winsorize( sqrt( RV / rolling-per-slot-diurnal-MEAN ) )`, predicted
  one bar ahead (the `.shift(1)` lives in the feature construction; `horizon=1`).
- **Features** = HAR rolling means at lags `[1,5,25,125,625,3125]` of `adj_RV`,
  plus calendar (`DOW`, `hour`, `is_overnight`), plus optional exog.
- Calendar + diurnal are **on for every model, Ridge included** (`ridge.py:90-91`).
  The executor/target docstrings that say "Ridge: False" are **stale** — fix them.
- Plain models (Ridge/XGB/kNN) use `IdentityResidualizer` — **no residualization**;
  that machinery only runs on the spectral/stacking path.
- **MSE/MAE are reported in sqrt space; QLIKE in raw space** (Duan smearing,
  `metrics.py:48`). They are not on the same footing — a model can move one
  without the other.
- Grid is ~24h: 48 bars/day, ~34 overnight + ~14 RTH.
- All the ΔR² numbers below are **OLS-R² in sqrt space** — a fast proxy for the
  real Ridge/QLIKE pipeline. Directions hold; magnitudes will differ.

---

## What holds

- **HAR is a multi-timescale regime model and is very hard to beat.** Slow/daily
  regime = the long lags (`har_ma_625/3125`); intraday regime = the short lags
  (`har_ma_1/5/25`). Tested additions that add ~nothing: session-so-far state
  (+0.0005), regime×time-of-day interaction (+0.0001), overnight RV magnitude
  (single/multi-night/innovation, ~0), staleness clocks (~0), order-flow
  imbalance (redundant with index leverage).
- **Leverage (signed returns → vol) is real but already in the feature set.**
  Well-grounded in the literature (Black 1976; EGARCH/GJR; Corsi–Renò 2012 LHAR;
  Barndorff-Nielsen et al. 2010 + Patton–Sheppard 2015 semivariance), and it's
  orthogonal to the magnitude-only HAR (which has no sign info). BUT the main
  asset's `sumret` (signed) and `sumabsret` (magnitude) are already in the
  `moments` subgroup, and `r⁻ ≈ (sumret − sumabsret)/2`, so Ridge can already
  form it and trees get `min(sumret,0)` from `sumret` via splits. Nothing to
  build. (Bar-level semivariance underperforms the net signed return at 30-min
  granularity — a proper RS⁻ needs tick data.)
- **Genuine cross-section moments carry a real, broad signal.** Clean `market_ew`
  (ffill, no artifact) adds **+0.0088 OOS, 15/17 expanding-window years**. This
  is the only validated lever — see the HPC ablation below for the real-QLIKE
  confirmation.
- **The calendar-cadence exog are not stepwise-constant** — they change every bar
  when present (`held% ≈ 0`); the missingness is *overnight market-closure*
  (~96% fresh in RTH). **Imputation choice is immaterial at the margin** (ffill /
  zero-fill / impute-and-flag all within noise; for per-bar flow stats ffill is
  mildly *worse* than zero-fill in the morning).

## What was RETRACTED (read this before re-using anything about "skew")

- The **"cross-section overnight skew" lead** — billed mid-session as the best
  find (+0.012–0.025 morning OOS, "robust 15/16 years") — is **fully retracted.**
  Every skew test reflexively copied a line calling
  `apply_overnight_fills(df, ["sumret3_ewstock","sumret3_vwstock"])`. That function
  substring-matches `ewstock`/`vwstock` and fills the column with the constant
  **1.0** in the overnight window — a fill meant for *ratio* columns, applied to a
  3rd moment (real values ≈ ±0.07; `cbrt(1.0)=1.0` is ~2.5× a real value). So the
  rolling mean of the "skew" was dominated by *how many overnight bars were
  missing-and-filled-with-1.0* → an **overnight-missingness-density** feature
  (which correlates with era/regime), not skewness.
  - Head-to-head, same feature, only the fill differs:
    `apply_overnight_fills(1.0)+ffill` = **+0.0252** (16/16 yrs);
    `ffill only` = **+0.0003**; `zero-fill` = **−0.0001**.
  - Everything downstream (the "6.5× time-of-day dilution", "trees get it for
    free via `hour`", the hardcoded 1-column feature) was correctly *describing
    an artifact* and is void.
- Four over-calls total, each caught by pushback: log-vs-sqrt (test in the real
  transform space), "we already have `sumret`", "stepwise-constant", and the skew
  artifact. The durable lesson is in the methodology section.

---

## The structural-missingness problem (still open — proposal here)

This is the piece that's been bugging you, and it's genuinely the hardest one.

**The two kinds of missingness are different and need different handling:**
1. *Between-update / overnight* gaps (market closed) — handled fine by ffill; the
   imputation choice barely matters (tested).
2. *Structural absence* — the series **didn't exist yet** (VIX before ~2007, the
   equity cross-section in the early sample). ffill cannot fill this (there is no
   prior value), so it's the genuinely unsolved case.

**Why it hurts today:** the Ridge path does `dropna_with_exog` — it drops any row
with a NaN exog. So adding a partially-available feature like VIX drops **all
pre-availability rows for the entire model**, which is why `implied_vol`/
`all_features` collapse to ~11–14k rows (vs ~200k). You're paying for one late
feature with the whole early history of *every* feature.

**Proposal — impute-and-indicate, don't drop:**
1. Fill structurally-absent values with a **neutral constant** — 0 *after
   centering* the feature (so the absent era contributes ~nothing to the linear
   predictor), or the feature's causal expanding mean.
2. Add a binary `<feat>_available` indicator (1 = real value, 0 = filled). The
   Ridge coefficient on the indicator absorbs any level shift between eras.
3. Best version: include `feature × available` (an interaction), so the feature's
   effect is **exactly gated to its availability window** — zero contribution when
   the series didn't exist. This is the linear way to say "use this feature only
   when it exists," and it keeps all ~200k rows.
4. The rolling walk-forward already helps: in the pre-availability era the *train*
   window also lacks the feature, so the model can't and shouldn't lean on it;
   post-availability it's present in train and test. Fill+indicator just stops the
   solver from choking on NaN and makes the feature self-gating.

**For trees:** simpler — keep the NaN (they handle it natively) or use 0+indicator;
they split on availability themselves.

**Do NOT** reconstruct the pre-existence period with a proxy (e.g. a fake VIX from
RV) — that injects model assumptions and risks leakage.

Net: this converts "adding VIX costs you 190k rows" into "adding VIX costs you
nothing; it just contributes only where it exists." Worth implementing as a small
extension to `load_and_transform` (a structural-fill + indicator path, distinct
from the overnight ffill).

---

## The pipeline bug to fix

`apply_overnight_fills` (`src/data/loading.py`) matches **any** column containing
`ewstock`/`vwstock`/`voldemand` — including the moment columns `sumret3_*`,
`sumret4_*`, `sumbipow_*`, `sumret2_*` — and fills them with 1.0. The executor
calls it on `exog_cols` in production, so **your real `market_ew`/`all_features`
subgroup runs are contaminated** by this artifact wherever those columns appear.
In OLS it doesn't help — it *hurts* (clean `market_ew` +0.0088 → with-fill −0.0019).

Fix: restrict the 1.0 fill to genuinely ratio-type columns (or drop it). A
non-destructive **toggle** has been added this session (`overnight_fill` param,
default `True` = legacy) so the ablation can measure the artifact in real QLIKE
before deciding the permanent fix.

---

## The HPC experiment that's running (3 arms, Hoffman2)

The real Ridge walk-forward times out interactively (>2 min even HAR-only), so it
goes to hpc-agent. Three configs under `configs/`:
- `ridge_har_baseline.yaml` — HAR only (should reproduce QLIKE ~0.13804).
- `ridge_market_ew_prod.yaml` — `market_ew`, `overnight_fill: true` (production /
  artifact; should reproduce ~0.13375).
- `ridge_market_ew_clean.yaml` — `market_ew`, `overnight_fill: false` (clean).

Reading the result: if clean ≈ prod in QLIKE, Ridge's regularization was already
absorbing the artifact; if clean < prod (lower QLIKE = better), removing the bad
fill genuinely helps and the fix is a free win.

---

## Methodology lessons (the durable part)

1. **Test in the real transform space** (sqrt + diurnal-mean + the actual 6-lag
   HAR), not raw/log proxies. Raw-space testing inflated the overnight-RV signal
   ~30×.
2. **Fill-swap-verify.** Imputation can *manufacture* signal — a wrong-type fill
   produced a robust-looking (16/16-year) predictor out of pure missingness
   structure. Always confirm a feature survives swapping the fill before believing
   it.
3. **"Correlates with era/regime" is a red flag, not an encoding target.**
   Calendar-era proxies don't generalize forward. The generalizing regime signal
   is already in the slow HAR lags; the legitimate forward-looking exogenous add is
   the VIX/implied-vol *level* (with its missingness caveat).
4. OLS-R²-in-sqrt-space is a proxy for the real Ridge/QLIKE pipeline — trust the
   direction, re-measure the magnitude.

Tooling note: working sklearn/tree env is conda `285J`; miniconda base is fine for
pure-numpy OLS (its scipy ABI is broken, so sklearn won't import there).

---

# Part 2 — Running the 3-arm ablation on CARC (execution log, 2026-06-24)

Recording this because the failure modes are reusable, not because it went smoothly.
It didn't.

## CARC bring-up
- Added `carc` to `~/.hpc-agent/clusters.yaml` (SLURM, `discovery2.usc.edu`, user
  `jc_905`, scratch `/scratch1/jc_905`, default account `pollok_1603`, partition
  `main` = 48 h). Conda is module-provided: `module load conda` +
  `source /apps/conda/miniforge3/25.3.0/etc/profile.d/conda.sh`.
- Created env `harxhar` at `/home1/jc_905/.conda/envs/harxhar` (the system
  `/apps/conda/envs` is **read-only**, so `-n` failed → use an explicit `-p` prefix;
  `mamba` also tripped a login-node process limit, `conda --solver=libmamba` worked).
  Deps resolved to numpy 2.3.5 / pandas 3.0.1 / sklearn 1.9.0 / **numba 0.65.1** —
  `@njit` verified on numpy 2.3.5 (the Ridge `prescale` path hard-imports numba).

## SSH transport: off WSL, onto native OpenSSH
- hpc-agent's `HPC_SSH_BINARY` pointed at `wssh.cmd`, which was
  `wsl -d Ubuntu -e ssh`. WSL's ssh config matched carc only by the **alias**
  `usc-discovery`, not the **hostname** `discovery2.usc.edu` that hpc-agent connects
  by → every connection hung the full 60 s. Rewrote `wssh.cmd` to native
  `C:\Windows\System32\OpenSSH\ssh.exe` and fixed the Windows `~/.ssh/config` carc
  block to match the hostname (+ `id_carc`, `IdentitiesOnly`). Preflight went
  60 s-timeout → **3.4 s green**.
- **WSL multiplexing is not viable here.** `ControlMaster` (which only WSL ssh
  supports, not native Windows OpenSSH) would collapse hpc-agent's many calls onto
  one connection — but WSL2's NAT does **not** carry the VPN route to CARC's private
  IP (`10.72.0.14`), so WSL ssh times out even with the config fixed. It'd need WSL2
  **mirrored networking** (`.wslconfig` + `wsl --shutdown`). Native ssh reaches CARC
  (uses the host routing table incl. the VPN) but can't multiplex.

## The fail2ban "banhammer"
- A ~30-minute burst of **failed/hung** SSH connections — the WSL-misrouting 60 s
  hangs (×2–3), env-creation retries, the spawned `--bare` worker, and a sandboxed
  subagent each retrying for ~20 min — tripped CARC's fail2ban. Symptom: all login
  nodes + ping to the private IP go dark at once while internet and UCLA stay up
  (silent DROP on a routable host = firewall ban, not a VPN drop). It auto-expired
  in ~an hour. **Lesson:** inspect the SSH transport *before* hammering preflight;
  batch cluster setup into one or two sessions; never let workers/subagents
  retry-storm a cluster.

## The hpc-agent dispatch bug (the real blocker)
- Canary FAILED in 0.5 s, ×3: `run.py: error: the following arguments are required:
  --config`. Root cause: hpc-agent 0.10.63's submit-flow recorded
  `"executor": "python run.py"` in the run sidecar — it **dropped `--config {config}`**
  from the entry-point argv, and ignored the correct materialized `executor_cmd`
  (which loads the wrapper and reads kwargs from `HPC_KW_*` env).
- **Fix (self-contained in harxhar):** `dispatch.py:815` *always* sets
  `env[f"HPC_KW_{key.upper()}"]` per task, so `$HPC_KW_CONFIG` is reliably present in
  the job env. Made `run.py`'s `--config` optional with a fallback to
  `os.environ["HPC_KW_CONFIG"]`. Now the jobs run whether or not the executor-string
  bug is present. (Also upgraded hpc-agent 0.10.63 → 0.10.65 from the local source
  repo; the run.py fix is robust either way.)

## Journal cleanup + resubmit
- The crashed submits left a stale `in_flight` **canary** record in the global
  journal (`~/.claude/hpc/<repo-hash>/index.json` + `runs/`) that blocked resubmit;
  `reconcile` couldn't settle it (`unable_to_verify` on the already-failed-and-purged
  SLURM job `9567759`). Deleted the record per hpc-agent's own remediation
  (index → `{}`, removed the run json + lock).
- Resubmitted **native + `no_canary`**: the canary task is itself a full ~10–30 min
  ridge walk-forward, so the canary gate is what made the bare worker `ScheduleWakeup`
  and return no report. `no_canary` submits all 3 jobs directly.

## Worker-mode notes (relevant to hpc-agent itself)
- A `--bare` spawned worker **can't hold a synchronous canary wait** — it schedules a
  wake-up and `hpc-agent run` sees "no JSON report."
- Inline mode falls to a **subagent that shares this session's sandbox**, which is
  SSH-blocked to CARC → fails. The path that works on this machine is driving the
  submit from the **main loop** directly (it has working native SSH).

## Uncommitted at session end (for `/sync`)
`run.py` (config env-fallback), `src/backtest/executor.py` + `src/models/ridge.py`
(`overnight_fill` toggle), `configs/ridge_{har_baseline,market_ew_prod,market_ew_clean}.yaml`,
`.hpc/` (onboarding), and this writeup. Cluster config + ssh config + `wssh.cmd` live
outside the repo (not synced). Plus the `apply_overnight_fills` fix in
`src/data/loading.py` (skip `sum*` moments from the 1.0 fill) and
`results/cluster_market_ew_ablation.csv`.

## Results — the 3-arm ablation in real Ridge/QLIKE (CARC job 9574648)

All three ran clean on CARC (array job 9574648, ~5 min HAR / ~14 min market_ew).
QLIKE is raw-space (Duan smearing); **lower is better**.

| arm | QLIKE | w_QLIKE | MSE | MAE | ΔQLIKE vs HAR |
|---|---|---|---|---|---|
| HAR baseline | 0.13460 | 0.11770 | 0.06203 | 0.18649 | — |
| `market_ew` prod (1.0-fill artifact) | 0.13416 | 0.11704 | 0.06196 | 0.18612 | −0.00044 |
| `market_ew` clean (no 1.0 fill) | **0.13332** | **0.11656** | 0.06169 | 0.18586 | **−0.00128** |

**Confirms the OLS prediction in real walk-forward QLIKE, both halves:**
1. Genuine cross-section moments help — clean `market_ew` beats HAR by −0.00128 QLIKE (~0.95%).
2. The `apply_overnight_fills` 1.0-fill artifact **degrades** it — prod is +0.00084 worse
   than clean, eating ~66% of the genuine gain (improvement drops 0.95% → 0.33%).

→ The fix (skip `sum*` moments from the 1.0 fill) is now in `loading.py`. (Note: the
HAR baseline here is 0.1346 vs the published ~0.138 — newer numpy/pandas/sklearn — but
all three arms share the same env, so the *relative* comparison is clean.)

## Subgroup sweep (CARC job 9574902) + the two fixes

Ran HAR + all 8 exog subgroups clean on CARC (`results/cluster_subgroup_sweep.csv`).
**Comparable arms (full ~219k sample), vs HAR 0.13460, lower=better:**

| subgroup | QLIKE | ΔvsHAR |
|---|---|---|
| **moments** | 0.13108 | −0.00352 (best) |
| **liquidity** | 0.13167 | −0.00293 |
| market_vw | 0.13303 | −0.00156 |
| market_ew | 0.13332 | −0.00128 |
| HAR | 0.13460 | — |

All five comparable subgroups beat HAR; **`moments` (main-asset leverage + magnitude
+ higher moments) and `liquidity` are the genuine ~2.5% winners** — the session's core
finding validated in real walk-forward QLIKE, cross-section secondary.

Two arms exposed problems (both predicted):
- **Missingness** — `sentiment` (n=150k) and `implied_vol` (n=127k) score on shrunk
  samples (Ridge `dropna_with_exog`), so not comparable.
- **`vol_demand` catastrophe** — QLIKE **0.614** on the full sample: residual
  `apply_overnight_fills` 1.0-fill on `voldemand` (range ±millions). `all_features`
  TIMEOUT at 1h (41-col transforms too slow) + doubly compromised.

**Two fixes built (lint-clean, locally smoke-tested; NOT yet cluster-validated):**
1. `apply_overnight_fills` — 1.0 fill gated off via empty `RATIO_FILL_COLS`
   (`loading.py`); no current column is a 1.0-neutral ratio.
2. **Impute-and-indicate** — `impute_indicate` flag (`ridge.run → run_executor →
   load_and_transform`): keep rows, fill adj-exog with 0, add `<col>_avail`
   indicators. Smoke test: `implied_vol` keeps **246k rows** (vs naive 14k), no NaN.

**Re-run (job 9579151) FAILED at dispatch — an hpc-agent bug, not the fixes.** The
executor materialized as a bare `module:function` ref `hpc_wrappers.ridge_imp:ridge_imp`
and the cluster dispatch ran it as a *shell command* → `command not found` (exit 127),
before Python started. Earlier runs used a working `python3 -c "…import wrapper…"` form.
So the user's wheel's newer executor materialization (`module:function`) isn't resolved
by `_hpc_dispatch.py` on the cluster (wrapper at `.hpc/wrappers/ridge_imp.py` but
referenced as the `hpc_wrappers` package — a PYTHONPATH/path mismatch). **The two
harxhar fixes are sound; only the cluster validation is blocked, on hpc-agent's side.**

## Two more bugs found driving the impute-indicate validation to ground

The "hpc-agent bug" above was diagnosed wrong by me, then corrected by the user, then
the validation hit a *second* (real, mine) bug. Both now fixed.

**Bug A — exit-127 dispatch (NOT the framework; my local CLI).** The `module:function`
ref `hpc_wrappers.ridge_imp:ridge_imp` did not come from hpc-agent — the framework only
ever emits file-path `.hpc/wrappers/<run>.py` executors, and the `python_module` schema
carries no `executor_cmd`. It came from **my locally-installed `hpc-agent` CLI being a
divergent build** (a `uv tool` build from source, not the user's canonical wheel). The
*submit* ran through that local CLI and wrote `hpc_wrappers.<run>:<fn>` into the run
sidecar; the cluster runs the user's wheel (file-path), so it couldn't resolve it.
**Fix:** `uv tool install <user's wheel> --force` so the local CLI == the cluster wheel,
then re-onboard + resubmit. Verified: the fresh sidecar executor is now
`python3 -c "…exec_module('.hpc/wrappers/ridge_imp.py')…"`, and the arms ran past dispatch
(RUNNING, not FAILED@9s). Lesson for memory: **only ever use the user's hpc-agent wheel,
for the local CLI too — not just cluster installs.**

**Bug B — impute-indicate blew QLIKE to ~2.4 (mine).** First clean run (job 9579644):
n_samples = 218,934 ✅ (impute-indicate kept the full sample), but QLIKE 2.43 / 2.24 for
sentiment / implied_vol — 18× HAR. My first guess (zero-IQR division in
`rolling_robust_scale`) was **wrong** — `_get_robust_stats` already floors IQR to 1.0
(`scaling.py:60`). Real cause: I filled `adj_<col>` NaN with **0 in the *transformed*
space**, but `adj_vix = log(vix)`, so `fillna(0)` imputes `vix = e⁰ = 1` (absurd for a
10–80 index) → wild features → blow-up. **Fix:** fill with the feature's own **median**
(in-distribution neutral, prescales to ~0). Smoke: `adj_vix` fills at 2.85 (log-vix
median), range 2.2–4.42, no NaN, 246k rows. Re-running (job pending) for the comparable
QLIKE of sentiment / implied_vol / vol_demand vs HAR 0.1346.

**Net state of the two harxhar fixes:** (1) `apply_overnight_fills` 1.0-fill gated off
via empty `RATIO_FILL_COLS` — done, validated indirectly (vol_demand no longer in the
1.0-fill path). (2) impute-and-indicate — code now correct (median fill, not 0), locally
smoke-tested; cluster QLIKE pending the re-run. The comparable subgroup ranking
(moments > liquidity > market_vw > market_ew > HAR) is unaffected and stands.

---

# Part 3 — Resolution: three stacked scale bugs + the incremental solver (2026-06-24, cont.)

"Cluster QLIKE pending" resolved — and it went deeper than the two anticipated fixes.
The impute-indicate re-run did NOT just work; it exposed a *class* of bug,
**divide-by-a-transiently-degenerate-scale**, living in three different scale
estimators. Each produced the same signature: winsorized metrics normal (bulk
forecast healthy) but raw QLIKE/MSE blown by a handful of rows whose runaway
predictions inflate the *series-wide* Duan smear (`metrics.py:48`), which is then
added to every raw prediction → uniform QLIKE blow-up. So winsorizing can't save it.

## The three bugs — same disease, one cure

Cure each time: **estimate the scale from the observations that actually carry
dispersion; never divide by a transiently-degenerate scale.**

1. **Rolling-scaler IQR floor.** `_get_robust_stats` (`scaling.py`) floors IQR at
   **1e-12, NOT 1.0** — Part 1's "floors IQR to 1.0" claim was WRONG. So an
   imputed-constant window collapses IQR to ~1e-7 and the incoming real point scales
   to ~1e5 (`adj_vix3m_ma_3125` → 2.6e5 at the vix3m availability transition). Fix:
   floor at the causal expanding IQR over *available* values + pass `*_avail_ma_*`
   indicators through raw (`executor._build_scale_guards`).
2. **Diurnal std baseline.** `diurnal_adjust` (`target.py`) divides a signed exog by
   its per-slot rolling **std**; `replace(0, 1.0)` caught only *exactly* zero, so a
   ~1e-7 std (flat/ffilled patch) on a ±1e6 feature (voldemand) blew adj to ±1e13.
   Fix: floor the std at 10% of its typical per-slot level (`DIURNAL_STD_FLOOR_FRAC`).
3. **Zero-inflation.** voldemand is a *mixture* (point mass at 0 + continuous demand),
   so median/IQR is the wrong summary — IQR ≈ 0 by construction, any non-zero value
   explodes. Fix: the **hurdle (two-part) encoding** — an occurrence indicator
   (`<col>_active`) + the magnitude scaled over its *active (non-zero)* values
   (`ZERO_INFLATED_FRAC`). `apply_semantic_transform`'s rule-5 identity for signed
   features is the deeper mis-design (signed gets no variance-stabilizer); asinh would
   fit, but the hurdle sufficed — the residual COVID fat tail (|z|≈1892) was absorbed
   by Ridge's regularized coefficient (`mse` stayed at HAR's 0.062).

## Incremental rolling least-squares (orthogonal, exact)

Profiled: the walk-forward is 96% per-bar Ridge refit (994s loop vs 44s transform).
`RollingLeastSquares` (new) maintains `(XᵀX, Xᵀy)` + intercept sums and rolls them
rank-1 (O(p²)) as the window slides, vs an O(W·p²) sklearn refit per bar.
`MultiStageBacktest` detects the `roll` capability by duck-typing → incremental fast
path; every other model falls through unchanged. **Exact to 1.8e-11 vs sklearn,
108× faster** (4.589 → 0.042 ms/bar). On by default for Ridge. (Note: for large p the
O(p³) solve becomes the new bottleneck — Sherman-Morrison would help.)

## Final validated ranking (real walk-forward QLIKE, full 218,934, vs HAR 0.13460)

| subgroup | QLIKE | ΔvsHAR | job |
|---|---|---|---|
| **all_buckets (41 exog)** | **0.12807** | **−0.00653** | 9585181 |
| moments | 0.13108 | −0.00352 | 9574902_2 |
| liquidity | 0.13167 | −0.00293 | 9574902_3 |
| implied_vol | 0.13265 | −0.00195 | 9583387_3 |
| market_vw | 0.13303 | −0.00156 | 9574902_5 |
| market_ew | 0.13332 | −0.00128 | 9574902_4 |
| vol_demand | 0.13367 | −0.00093 | 9585155_1 |
| sentiment | 0.13374 | −0.00086 | 9585155_3 |
| HAR | 0.13460 | — | 9574902_1 |

**Every subgroup beats HAR.** The three arms Part 2 flagged as non-comparable losers
(sentiment, implied_vol, vol_demand) were ALL scaling artifacts — properly scaled,
each is a genuine win. New durable lesson: **a "feature that hurts" is often a feature
the pipeline mis-scales, not a feature without signal.** HAR control reproduced
0.13460 exactly (guards are no-ops on non-impute runs); moments/sentiment controls
held to 7e-6 (zero collateral). **All-buckets (41 exog combined): QLIKE 0.12807
(−0.00653 vs HAR), 22 min** (job 9585181) — beats *every* single subgroup,
moments included, by a further −0.00301. Stacking all buckets wins: the diverse
signals are complementary and α=1 Ridge soaks up the ~526-feature redundancy
without overfit collapse. Feasible ONLY via the incremental solver (sklearn
full-refit timed out at 1h; at p≈526 the cost is the O(p³) per-bar solve, not the
108× small-p win — but it ran).

## Files changed (all ruff + mypy clean)
`scaling.py` · `executor.py` · `target.py` · `residualizer.py` · `multi_stage.py` ·
`ridge.py` · `rolling_least_squares.py` (new) · `results/cluster_subgroup_sweep.csv`.

---

# Part 4 — Training-window ablation (Ridge), all buckets (2026-06-24, Hoffman2)

The incremental solver made a `train_window` sweep cheap, so we ran it across every
feature bucket. **Method:** vary `train_window` (which ties BOTH the Ridge rolling fit
window AND the robust-scaler window — the pipeline couples them); evaluate every
window on a **common OOS region** (start at the largest window's warm-up) so the
comparison is apples-to-apples, not a different test period; refit-every-bar via the
incremental solver. Run on Hoffman2 base anaconda (numpy 1.23 / pandas 1.5) — so the
*absolute* QLIKE differs from CARC, but the ablation is **relative** (same env across
windows), so the argmin is valid.

**Result — the optimum is ~250 days, universal across all 8 buckets** (60–1000-day
grid, common-OOS ≈194,934 bars, lower=better):

| bucket | best window | QLIKE@250 | Δ vs 500-day default |
|---|---|---|---|
| HAR | 250 (≈225 on a finer grid, flat 200–250) | — | −0.0014 |
| moments | 250 | 0.13449 | −0.00104 |
| sentiment | 250 | 0.13025 | −0.00092 |
| market_ew | 250 | 0.13042 | −0.00065 |
| market_vw | 250 | 0.13023 | −0.00050 |
| implied_vol | 250 | 0.12961 | −0.00044 |
| liquidity | 250 | 0.12895 | −0.00039 |
| vol_demand | 250 | 0.13080 | −0.00029 |

(all_buckets re-running after a long-filename crash; expected to match.)

**Takeaways:**
- **Every bucket's argmin is 250 days** — the optimal window is a property of the
  vol process's adaptivity, not the feature set. The current **500-day default is
  suboptimal everywhere** (gain −0.0003 to −0.0014 QLIKE, biggest for HAR/sentiment).
- The minimum is **broad and shallow** (250→1000 is nearly flat for most buckets);
  the finer HAR grid pins it to ~225 in a 200–250 plateau.
- **QLIKE/MSE divergence is universal**: MSE keeps improving out to ≥1000 days while
  QLIKE peaks at ~250. QLIKE (raw-space, penalizes under-forecasting spikes) rewards a
  more *adaptive* short window; MSE (sqrt-space) rewards a stable long one. Optimizing
  QLIKE → ~250.
- **Production change** to capture it: `ridge.run` `train_window` default 500 → ~250.
  ⚠️ This shifts every published baseline (HAR 0.13460 etc. were at 500), so it's a
  re-baseline, not a free swap — rerun the sweep at 250 before adopting.
