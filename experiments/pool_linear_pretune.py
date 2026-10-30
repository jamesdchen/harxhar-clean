"""The POOLED linear models (one coefficient vector over all 48 half-hour bars) pre-tuned, backbone 1/100.

User (2026-10-06): "use the pre-tuned version for all linear models."  The 16:00-bar version is
results/linear_pretune_2026-10-06/ (experiments/close_linear_pretune.py).  This is its pooled twin,
the runner the clusters execute (the local scoring and SUMMARY.md: experiments/pool_linear_pretune_report.py).

Model and walk: the spec's own, specs/causal_tune_linear.py with segment ``none`` (the stored pooled arms
pool_<ridge|lasso|enet>_<bucket>): one model over all bars, window 2000 sessions = 96,000 bars, refit
every bar (ridge: Sherman-Morrison rank-one add / drop; lasso / elastic net: two warm Garrigues-El Ghaoui
online-homotopy updates a bar), the identifiability mask and a cold seed at every 250th solve, the penalty
re-chosen there on the window's any-clock validation tail (fit = the first 95,850 bars, a 25-bar embargo,
the last 125 bars), intercept unpenalized, the spec's grids (ridge 1e-2 .. 1e3, lasso / elastic net
1e-6 .. 1e-2, elastic net l1_ratio 0.5).  The spec's estimator section is EXECUTED here read-only
(spec_section) and run through the spec's own run_executor invocation (same data prep, same prescale,
same chunk seam); the only new code is a subclass of the spec's RollingTunedLinear that

  * carries glmnet-style penalty factors pf (penalty on column j = alpha pf_j (l1 |b_j| + (1 - l1)/2 b_j^2)):
    pf = r on the HAR + calendar backbone (the 34 columns of the pooled baseline design: the target's HAR
    ladder har_ma_*, its session-edge interactions har_ma_*_x_open / _x_close, the calendar and expiry
    columns), pf = 1 on every exogenous column; r = 1 executes exactly the spec's floating-point
    operations (a * 1.0 == a), so arm (a) is the spec; the batch elastic net with pf != 1 is solved by
    column scaling and certified by the unscaled problem's KKT conditions (repaired by support changes,
    else coordinate descent + an exact KKT solve: the C port of experiments/close_exogpen_kernel.c);
  * records, at every re-choice, the validation-tail residuals of every grid alpha (the light-retune
    margin rule of arm f is decided from them, offline);
  * or, given a schedule {block: alpha}, skips the re-choice and seeds the scheduled alpha (arm f's walk).

A block's forecasts depend only on its alpha and the window (cold seed at the block start), so a chunk
that starts at a re-choice boundary with a 96,000-bar halo reproduces the whole-series run (the chunk seam
the stored pooled campaigns gated bit for bit).

Modes  (python experiments/pool_linear_pretune.py <mode> ...; root = the deployment root, data = ./data)
  walk     --bucket B --est E --r R [--sched causal|<json {block: alpha}>] --start S --end E --halo H --out D
           D/causal_tune_linear/<est>/<bucket>/results.csv (the spec's table), results_rth.csv (10:00-16:00),
           trace.npz (alpha, mask count, validation residuals of every grid alpha at each re-choice),
           run.json (seconds, host, numpy).
  pretune  --bucket B --est E --r R --start S --halo H --out D
           4 folds x 12,000 bars (250 sessions) tiling the 48,000 bars before row S (the first bar of the
           first trade day), each fit on the 96,000 bars before a 1,200-bar (25-session) embargo -- the
           16:00-bar study's fold scheme in bars; every grid alpha by the spec's batch solution on the fit
           block (identifiability mask from that block); validation residuals of every row stored (fold.npz).
  dates    --out D: the row dates of the pooled design (baseline bucket; every bucket shares the index
           space: impute_indicate), dates.csv.

--rerun (2026-10-19, user: the pooled linear rerun, results/pooled_linear_rerun_2026-10-19_48/; default off,
every path above unchanged) switches every mode to the rerun design:
  * HAR ladder base 5, k = 1, 5, 25, 125, 625, 3125 (HPC_KW_HAR_BASE=5, asserted; author 2026-10-20: "base 5
    and FOMC included", base 5 for every model, the OLS included);
    [2026-10-20 (user): parked -- was "HAR ladder base 2, k = 1, 2, 4, ..., 2048 (HPC_KW_HAR_BASE=2; the printed
    run never set it: base 5)"]
  * src/backtest/executor.py DROP_ACTIVITY_MA (E6) and CALENDAR_THROUGH_OVERNIGHT (E9) on: no *_active_ma_*
    columns, calendar DOW_0 .. DOW_4, hour, is_overnight, no HAR x open/close products;
  * all_features WITH the four FOMC series (author 2026-10-20; src/data/loading.py FOMC_FEATURES, FOMC_CALENDAR_FIX
    on; E6 still drops their activity averages; asserted);
    [2026-10-20 (user): parked -- was "all_features without the FOMC series (the paper's text: not in the pooled
    model; src/data/loading.py:136 appends them to ALL_FEATURES)"]
  * the backbone (penalty factor r) is the HAR benchmark set: har_ma_1 .. har_ma_3125 + the 7 calendar columns;
  * --est ols: the spec's ridge path at alpha 0 (the pooled OLS on the HAR benchmark set, a0's successor);
  * --sched may hold one number: that penalty at every re-choice (the frozen walk, no margin re-tune).
  design --part K/N builds the design of the K-th of N contiguous parts of the bucket's raw columns (memory: the
  48-interval all_features design, ~1,050 columns, does not fit 15 GB in one pass); merge --parts D1,D2,.. --out D
  stacks the parts' columns in the one-pass design's order (target HAR, every value ladder, every availability
  ladder, calendar).  Every step of the data prep and of the scaling acts column by column, so the merged matrix is
  the one-pass matrix (gated bit for bit on a small bucket, results/pooled_linear_rerun_2026-10-19_48/SUMMARY.md).

2026-10-19 (the pooled linear rerun on the 13 regular-hours bars, note E15; results/pooled_linear_rerun_2026-10-19_13/)
options, every one defaulting to the behaviour above:
  --sample rth13      rows = the 13 bars with end stamps 10:00 .. 16:00 (segment rth13, lag scope global: the HAR
                      components and every trailing average on the full 48-interval grid, then the rows restricted);
                      PERIODS_PER_DAY = 13, so the 2,000-day window is 26,000 rows and the pre-tune folds, embargo
                      and halo are counted in 13-bar sessions
  --har-base 2        the spec's HAR_BASE (ladder 1, 2, 4, ..., 2048)
  --drop-fomc         the FOMC series out of the bucket (not in the pooled model, the paper's text)
  --ex-flag NAME=0|1  a module flag of src/backtest/executor.py set for this run (repeatable)
  --sched const:A     one penalty A over the whole walk (the frozen rule): no re-choice, the identifiability
                      mask and the cold seed still at every 250th solve
  --est ols           min-norm least squares on the centred window (the a0 estimator, src/unification.py
                      _walk_ols: eigh pseudo-inverse at numpy's rcond 1e-15), rank-one sums, exact rebuild of
                      the sums every 250 solves

2026-10-20 (user: one runner for both reruns).  The 48-interval rerun ran with --rerun; the 13-bar rerun ran with
--sample rth13 --har-base 5 --ex-flag ... (experiments/pooled_linear_rerun_2026_10_19_13.py), not --rerun.  Where
the two added the same thing differently, both stay selectable:
  --ols auto|ridge0|minnorm   the estimator of --est ols.  ridge0: the spec's ridge path at alpha 0 (the
                      48-interval rerun); minnorm: min-norm least squares, eigh pseudo-inverse (the 13-bar rerun).
                      auto (default): ridge0 under --rerun, minnorm otherwise, so both reruns' commands are unchanged
  --sched             causal (default), a JSON file of {block: alpha}, a JSON file holding one number (the
                      48-interval rerun) or const:A (the 13-bar rerun); the last two are the same schedule
  backbone check      --rerun with none of --sample/--har-base/--drop-fomc/--ex-flag: the rerun's HAR benchmark set
                      exactly (BACKBONE_RERUN, the 48-interval rerun); none of those options and no --rerun: the
                      printed run's BACKBONE; any of those options: har_ma_* and calendar columns only (the 13-bar
                      rerun)
  --rerun fixes the ladder at base 5 and keeps the FOMC series, so with it --har-base may only be 5 and --drop-fomc
  is refused; --ex-flag is applied after --rerun's own flags.

2026-10-22 (user: "Fix the late start series problems."): --keep-fomc-release puts the FOMC statement-bar dummy
back into all_features at its old place (src/data/loading.py EXCLUDE_FOMC_RELEASE is on since 2026-10-22), so a
rerun can be set against the published 576-column runs; recorded in the design cache's config only when set.  The
executor's late-start guard is a module flag like the others (--ex-flag LATE_START_WARMUP=0|1; on by default).

2026-10-22 (James: "add the KKT check flag"): src/models/reclasso_har.py WALK_KKT_RESEED (on by default).  After every
warm update of the lasso / elastic net the walk checks the KKT conditions of the current window's problem
(reclasso_har.walk_kkt on the warm Gr, c, mu_vec; the batch certification's tolerance) and, where they fail, takes the
cold seed on the current window (_seed) and goes on warm from it (PFTuned.roll / solve).  --walk-kkt-reseed 0 turns it
off: the walk before the flag, bit for bit.  Recorded with the flag on: trace.npz kkt_* (every lasso / elastic-net solve
and every KKT-triggered reseed) and run.json kkt_*.  A chunk that starts at a re-choice boundary still reproduces the
whole walk (a reseed depends only on the state since the block's cold seed).  Ridge and the OLS are not touched.

2026-10-22 (James: "yes, adopt standardization and hold the pretune"): STANDARDIZE_PENALTY (on by default; this module
only, so no closing-interval path sees it).  Ridge, lasso and elastic net are fit as if every predictor column were
scaled to unit variance in its training window (glmnet standardize=TRUE), as per-column penalty weights: with s_j the
column's np.std (ddof 0) over the fit's rows, ridge a sum pf_j s_j^2 b_j^2, lasso / elastic net
a sum pf_j [l1 s_j |b_j| + (1 - l1)/2 s_j^2 b_j^2] (mu_j = n a l1 pf_j s_j, lam2_j = n a (1 - l1) pf_j s_j^2), i.e.
the existing penalty on gamma_j = s_j b_j of z_j = x_j / s_j; intercept free, nothing centred; a non-locked column with
s_j < 1e-12 masked for the fit.  Ridge: weights from every bar's own window (X'X, X'y and the window's column moments
kept rank-one since the seed, one solve a bar).  Lasso / elastic net: weights recomputed at every cold seed (tune
boundary, mask-addition and KKT reseeds) and held between seeds.  Pretune: weights from each fold's fit block.  The OLS
(alpha 0) is never weighted.  --standardize-penalty 0 turns it off: the runner before the flag, bit for bit.
--grid-alphas replaces the spec's penalty grid (the orchestrator's grid extensions).  Recorded with the flag on:
trace.npz std_* (the s.d. at every seed; ridge: the within-block drift), run.json std_*; --std-check-blockends
(diagnostic) also refits the lasso / elastic net at each block's last solve with the current window's weights and
records the forecast difference.  Rule: results/standardize_penalty_2026-10-22/RULE_std.txt (written 2026-10-22 16:06
PDT, before any run).  Its addenda (each written before its code): 1-2 the evidence mask (a column enters only if its
variable's k = 1 column has >= 250 rows off its window mode) and the centring; 3 (19:25 PDT) the scaled coordinates
(the lasso / elastic net warm path, KKT check and batch seed in z = (x - m) / s with m, s held between seeds and the
penalty factors only; ridge with exact per-bar scaling -- the same estimator) and the binomial availability rule (a
k > 1 availability average is masked when its window s.d. <= sqrt(p (1 - p) / k), or when it differs from an earlier
kept one of the same k by an s.d. <= sqrt(2) times that resolution at the pair's mean level; a data-cleaning rule).

2026-10-30 (the author's question "can we investigate if the variance reducing transforms are actually helping?";
results/transform_ablation_2026-10-30/RULE.txt): the transform switches are executor flags, so --ex-flag sets them and
the design cache's config records them; no code here changed.  TARGET_DIURNAL, TARGET_WINSOR, TARGET_WINSOR_5_95,
TARGET_LOG, TARGET_LEVEL, EXOG_DIURNAL, EXOG_SEMANTIC, FEATURE_WINSOR, PRESCALE (src/backtest/executor.py; defaults:
every design as before, bit for bit).  run_spec_executor's literals (target_use_diurnal=True, target_winsor_window=240,
prescale=True) stay; TARGET_DIURNAL=0, TARGET_WINSOR=0 and PRESCALE=0 override them inside the executor.
"""

from __future__ import annotations

import os

for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(
    os.environ.get("PLP_ROOT", str(Path(__file__).resolve().parents[1]))
).resolve()
TW_DAYS = 2000
PERIODS_PER_DAY = 48
W = TW_DAYS * PERIODS_PER_DAY  # 96,000 bars
PRE_FOLDS, PRE_FOLD_SESS, PRE_EMBARGO_SESS = (
    4,
    250,
    25,
)  # the 16:00-bar study's pre-tune, in sessions
BACKBONE = frozenset(
    [f"har_ma_{k}" for k in (1, 5, 25, 125, 625, 3125)]
    + [
        f"har_ma_{k}_x_{e}"
        for k in (1, 5, 25, 125, 625, 3125)
        for e in ("open", "close")
    ]
    + [f"DOW_{d}" for d in range(5)]
    + ["hour", "is_overnight", "is_open", "is_close"]
    + [
        "is_opex",
        "is_opex_week",
        "is_quad_witch",
        "is_rebalance_close",
        "is_month_end",
        "is_quarter_end",
        "days_to_opex",
    ]
)
# 2026-10-19 (user): the rerun's backbone, the HAR benchmark set of the rerun design (--rerun)
# 2026-10-20 (user): base 5.  Parked -- was base 2:
# HAR_LADDER_RERUN = [2**j for j in range(12)]  # 1 .. 2048 (resolve_har_lags(base=2), cap 3125)
HAR_LADDER_RERUN = [5**j for j in range(6)]  # 1 .. 3125 (resolve_har_lags(base=5))
BACKBONE_RERUN = frozenset(
    [f"har_ma_{k}" for k in HAR_LADDER_RERUN]
    + [f"DOW_{d}" for d in range(5)]
    + ["hour", "is_overnight"]
)
RERUN = False  # set by main from --rerun
PART: tuple[int, int] | None = None  # design --part K/N (1-based)
_EXOG = re.compile(r"_ma_\d+$")
# 2026-10-19: the run's configuration (configure(), from the command line); the defaults are the 6 Oct runner
SAMPLE = "48"
SEGMENT_OF = {"48": "none", "rth13": "rth13"}
PPD_OF = {"48": 48, "rth13": 13}
HAR_BASE = ""
DROP_FOMC = False
EX_FLAGS: dict = {}
# 2026-10-22 (user, the late-start rerun): --keep-fomc-release puts the FOMC statement-bar dummy back into
# all_features (and the fomc bucket) at its place before fomc_day, as before src/data/loading.py
# EXCLUDE_FOMC_RELEASE went on, so a run can be compared with the published 576-column runs; default off
KEEP_FOMC_RELEASE = False
# 2026-10-22 (James: "yes, adopt standardization and hold the pretune"; module docstring): the penalty of the pooled
# ridge / lasso / elastic net on the window-standardized coefficients s_j b_j; read at call time; False = the runner
# before the flag, bit for bit
STANDARDIZE_PENALTY: bool = True
# 2026-10-22 (owner: no floor on s, "learn from what we've done in the past"; RULE_std.txt addendum 1): the EVIDENCE
# MASK of STANDARDIZE_PENALTY -- a non-locked column enters a standardized fit only if at least this many rows of the
# fit's window differ from its modal value there; the 250-row minimum-evidence rule of src/backtest/executor.py
# MODE_MIN_PRINTS (and src/features/transforms/target.py ASINH_ROUTE_MIN_ROWS), asserted equal in spec_section
STD_MIN_EVIDENCE = 250
# STD_MIN: a non-locked column whose window s.d. is below it is masked for the fit; STD_CHECK_BLOCKENDS: walk
# --std-check-blockends (the diagnostic of the held lasso / elastic-net weights)
STD_MIN = 1e-12
STD_CHECK_BLOCKENDS = False
CALENDAR_ALL = frozenset(
    [f"DOW_{d}" for d in range(5)]
    + ["hour", "is_overnight", "is_open", "is_close"]
    + [
        "is_opex",
        "is_opex_week",
        "is_quad_witch",
        "is_rebalance_close",
        "is_month_end",
        "is_quarter_end",
        "days_to_opex",
    ]
)


def configure(a) -> None:
    """Set the sample, the ladder, the bucket filter and the executor flags of this run."""
    global SAMPLE, PERIODS_PER_DAY, W, HAR_BASE, DROP_FOMC, EX_FLAGS, TW_DAYS
    global KEEP_FOMC_RELEASE
    SAMPLE, HAR_BASE, DROP_FOMC = a.sample, a.har_base, bool(a.drop_fomc)
    KEEP_FOMC_RELEASE = bool(a.keep_fomc_release)  # 2026-10-22 (--keep-fomc-release)
    TW_DAYS = a.window_days  # 2026-10-20 (user): --window-days, default 2000
    PERIODS_PER_DAY = PPD_OF[SAMPLE]
    W = TW_DAYS * PERIODS_PER_DAY
    EX_FLAGS = {}
    for kv in a.ex_flag:
        k, v = kv.split("=")
        assert v in ("0", "1"), kv
        EX_FLAGS[k] = v == "1"


def default_config() -> bool:
    # 2026-10-20 (user): parked -- before --window-days. Was:
    #     return SAMPLE == "48" and not HAR_BASE and not DROP_FOMC and not EX_FLAGS
    # 2026-10-22 (user): parked -- before --keep-fomc-release (the condition below adds it). Was the same
    # expression without "and not KEEP_FOMC_RELEASE".
    return (
        SAMPLE == "48"
        and not HAR_BASE
        and not DROP_FOMC
        and not EX_FLAGS
        and TW_DAYS == 2000
        and not KEEP_FOMC_RELEASE
    )


def config_dict() -> dict:
    # 2026-10-22 (user): keep_fomc_release is recorded only when set, so the design caches built before the option
    # still pass check_meta
    return dict(
        sample=SAMPLE,
        periods_per_day=PERIODS_PER_DAY,
        window=W,
        har_base=HAR_BASE,
        drop_fomc=DROP_FOMC,
        ex_flags=EX_FLAGS,
        **({"keep_fomc_release": True} if KEEP_FOMC_RELEASE else {}),
    )


def seg_csv(path: str) -> str:
    """The table run_executor writes for output_file ``path``: a segment appends _<segment>."""
    seg = SEGMENT_OF[SAMPLE]
    if seg == "none":
        return path
    stem, ext = os.path.splitext(path)
    return f"{stem}_{seg}{ext}"


RTH = {f"{h:02d}:{m:02d}" for h in range(10, 17) for m in (0, 30)} - {
    "16:30"
}  # 13 bars, 10:00 .. 16:00


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def backbone_mask(names: list[str]) -> np.ndarray:
    """True on the HAR + calendar backbone (the pooled baseline design's 34 columns)."""
    bb = np.array(
        [not (_EXOG.search(n) and not n.startswith("har_ma_")) for n in names]
    )
    got = {n for n, b in zip(names, bb) if b}
    if RERUN and default_config():
        # 2026-10-19 (the 48-interval rerun, --rerun): the rerun's HAR benchmark set exactly
        assert got == BACKBONE_RERUN, sorted(got ^ BACKBONE_RERUN)
    elif default_config():
        assert got == BACKBONE, sorted(got ^ BACKBONE)
    else:  # 2026-10-19 (the 13-bar rerun): the target's HAR ladder and whatever calendar columns the flags leave
        assert all(n.startswith("har_ma_") or n in CALENDAR_ALL for n in got), sorted(
            got - CALENDAR_ALL
        )
    # 2026-10-20 (user): parked -- the 48-interval rerun's check, now the first branch above.  Was:
    # want = BACKBONE_RERUN if RERUN else BACKBONE
    # assert got == want, sorted(got ^ want)
    return bb


# ============================================================================ the spec, read-only
def spec_section(est: str, bucket: str, start: int, end: int, halo: int) -> dict:
    """Execute specs/causal_tune_linear.py from its estimator section's first import to just before the
    arm loop (constants, RollingTunedLinear, fit_predict_lin_tuned), with the pooled arm's environment."""
    if HAR_BASE:  # 2026-10-19 (--har-base)
        os.environ["HPC_KW_HAR_BASE"] = HAR_BASE
    os.environ.update(
        {
            "HPC_KW_SEGMENT": SEGMENT_OF[SAMPLE],
            "HPC_KW_LAG_SCOPE": "global",
            "HPC_KW_ESTIMATOR": est,
            "HPC_KW_EXOG_BUCKET": bucket,
            "HPC_KW_TRAIN_WIN": str(TW_DAYS),
            "HPC_KW_START": str(start),
            "HPC_KW_END": str(end),
            "HPC_KW_HALO": str(halo),
        }
    )
    if RERUN:
        # 2026-10-20 (user): base 5; parked -- was os.environ["HPC_KW_HAR_BASE"] = "2"
        os.environ["HPC_KW_HAR_BASE"] = "5"
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    src = (ROOT / "specs" / "causal_tune_linear.py").read_text(encoding="utf-8")
    a = src.index(
        "import numpy as np\nimport pandas as pd\n\nfrom src.models.reclasso_har"
    )
    b = src.index("arm_results: dict = {}")
    ns: dict = {"__name__": "causal_tune_linear_section", "os": os}
    exec(compile(src[a:b], str(ROOT / "specs" / "causal_tune_linear.py"), "exec"), ns)
    # the pooled OLS (--est ols): the spec's ridge path at alpha 0 (G = X'X, intercept free)
    ns["ESTIMATOR_GRIDS"]["ols"] = [("ridge", 0.0, 0.0)]
    if KEEP_FOMC_RELEASE:  # 2026-10-22 (--keep-fomc-release): fomc_release back at its place before fomc_day
        import src.data.loading as LD

        gb_k = ns["get_bucket"]

        def get_bucket_keep(name: str) -> list[str]:
            cols = list(gb_k(name))
            if (
                LD.EXCLUDE_FOMC_RELEASE
                and name in ("all_features", "fomc")
                and "fomc_release" not in cols
            ):
                cols.insert(cols.index("fomc_day"), "fomc_release")
            return cols

        ns["get_bucket"] = get_bucket_keep
    if RERUN:
        import src.backtest.executor as EX
        import src.data.loading as LD

        EX.DROP_ACTIVITY_MA = True
        EX.CALENDAR_THROUGH_OVERNIGHT = True
        assert ns["HAR_LAGS"] == HAR_LADDER_RERUN == [1, 5, 25, 125, 625, 3125], ns[
            "HAR_LAGS"
        ]
        # 2026-10-20 (user): the FOMC series stay in all_features, with the calendar fix
        # 2026-10-22 (user): parked -- every FOMC series, before EXCLUDE_FOMC_RELEASE went on. Was:
        # assert set(LD.FOMC_FEATURES) <= set(ns["get_bucket"]("all_features")), "FOMC"
        fomc_in = [
            f
            for f in LD.FOMC_FEATURES
            if f != "fomc_release" or KEEP_FOMC_RELEASE or not LD.EXCLUDE_FOMC_RELEASE
        ]
        assert set(fomc_in) <= set(ns["get_bucket"]("all_features")), "FOMC"
        assert LD.FOMC_CALENDAR_FIX, "FOMC_CALENDAR_FIX off"
        # 2026-10-20 (user): parked -- the FOMC series left all_features:
        # gb = ns["get_bucket"]
        #
        # def get_bucket_rerun(name: str) -> list[str]:
        #     cols = gb(name)
        #     return (
        #         [c for c in cols if c not in LD.FOMC_FEATURES]
        #         if name == "all_features"
        #         else cols
        #     )
        #
        # ns["get_bucket"] = get_bucket_rerun
    if PART is not None:
        gb_p = ns["get_bucket"]

        def get_bucket_part(name: str) -> list[str]:
            cols = gb_p(name)
            k, n = PART
            return [c for i, c in enumerate(cols) if i * n // len(cols) == k - 1]

        ns["get_bucket"] = get_bucket_part
    assert (
        ns["SEGMENT"] == (None if SAMPLE == "48" else SEGMENT_OF[SAMPLE])
        and ns["LAG_SCOPE"] == "global"
        and ns["TRAIN_WIN"] == TW_DAYS
    )
    if HAR_BASE:
        assert ns["HAR_LAGS"] == ns["resolve_har_lags"](base=int(HAR_BASE)), ns[
            "HAR_LAGS"
        ]
    import src.backtest.executor as EX

    # 2026-10-22: the evidence mask's 250 is the executor's minimum-evidence count
    assert STD_MIN_EVIDENCE == EX.MODE_MIN_PRINTS, (
        STD_MIN_EVIDENCE,
        EX.MODE_MIN_PRINTS,
    )
    import src.data.loading as LD_
    import src.features.transforms.target as TG_

    for k, v in EX_FLAGS.items():  # 2026-10-19 (--ex-flag)
        # 2026-10-25 (results/fomcfix_2026-10-26/RULE.txt): a flag that src/backtest/executor.py does not have is
        # looked up in src/data/loading.py, then src/features/transforms/target.py (FOMC_SCHEDULE_2020_FIX,
        # FOMC_PROXIMITY_NO_WINSOR); the executor's flags are set as before.  Was:
        # assert isinstance(getattr(EX, k), bool), k
        # setattr(EX, k, v)
        mod = next((m for m in (EX, LD_, TG_) if hasattr(m, k)), EX)
        assert isinstance(getattr(mod, k), bool), k
        setattr(mod, k, v)
    if DROP_FOMC:  # 2026-10-19 (--drop-fomc)
        from src.data.loading import FOMC_FEATURES

        gb = ns["get_bucket"]
        ns["get_bucket"] = lambda b: [c for c in gb(b) if c not in FOMC_FEATURES]
    assert (ns["TUNE_PER"], ns["VAL_TAIL"], ns["EMBARGO"], ns["REFIT_FREQUENCY"]) == (
        250,
        125,
        25,
        1,
    )
    return ns


def run_spec_executor(
    ns: dict,
    est: str,
    bucket: str,
    fit_predict,
    out_csv: str,
    start: int,
    end: int,
    halo: int,
    data_path: str,
) -> None:
    """The spec's run_executor invocation (its arm loop), line for line, with another fit_predict."""
    ns["run_executor"](
        method_name=f"lin_tuned_{est}",
        fit_predict=fit_predict,
        hyperparams={"_refit_frequency": ns["REFIT_FREQUENCY"]},
        data_path=data_path,
        output_file=out_csv,
        horizon=ns["HORIZON"],
        train_window=ns["TRAIN_WIN"],
        start=start,
        end=end,
        halo=halo,
        exog_cols=ns["get_bucket"](bucket),
        segment=ns["SEGMENT"],
        lag_scope=ns["LAG_SCOPE"],
        har_lags=ns["HAR_LAGS"],
        add_calendar=True,
        target_use_diurnal=True,
        target_winsor_window=240,
        dropna_with_exog=False,
        overnight_fill=True,
        impute_indicate=True,
        diurnal_mode="divide",
        prescale=True,
        seed=ns["SEED"],
    )


# ============================================================================ penalty factors
REPAIR = {"kkt_ok": 0, "active_repair": 0, "cd_polish": 0, "fail": 0}


def _kkt_ok(G, c, mu, lam2, b, tol) -> bool:
    r = c - G @ b - lam2 * b
    on = b != 0.0
    if np.any(np.abs(r[on] - mu[on] * np.sign(b[on])) > tol):
        return False
    return not np.any(np.abs(r[~on]) > mu[~on] + tol)


def _active_repair(G, c, mu, lam2, b, tol):
    """Support changes: solve the KKT system on the support with its signs, drop sign flips, add the
    worst violator, until every condition holds (close_exogpen_kernel.c active_repair)."""
    act = list(np.flatnonzero(b != 0.0))
    sg = list(np.sign(b[act]))
    m = len(c)
    for _ in range(200):
        if act:
            A = np.array(act)
            M = G[np.ix_(A, A)] + np.diag(lam2[A])
            try:
                x = np.linalg.solve(M, c[A] - mu[A] * np.array(sg))
            except np.linalg.LinAlgError:
                return None
            keep = np.sign(x) == np.array(sg)
            if not keep.all():
                act = [a for a, k in zip(act, keep) if k]
                sg = [s for s, k in zip(sg, keep) if k]
                continue
        bt = np.zeros(m)
        if act:
            bt[np.array(act)] = x
        r = c - G @ bt
        v = np.abs(r) - mu
        v[bt != 0.0] = -np.inf
        jw = int(np.argmax(v))
        if v[jw] <= tol:
            return bt if _kkt_ok(G, c, mu, lam2, bt, tol) else None
        act.append(jw)
        sg.append(1.0 if r[jw] > 0 else -1.0)
    return None


def _cd_polish(G, c, mu, lam2, b, tol):
    """Covariance-form coordinate descent from b, then the exact KKT solve on its support."""
    b = b.copy()
    g = c - G @ b
    ctol = 1e-5 * max(float(np.linalg.norm(c)), 1.0)
    dg = np.diag(G).copy()
    for _ in range(8):
        for _sweep in range(20000):
            big = 0.0
            for j in range(len(c)):
                den = dg[j] + lam2[j]
                if den <= 0.0:
                    continue
                z = g[j] + dg[j] * b[j]
                nb = (
                    (z - mu[j]) / den
                    if z > mu[j]
                    else (z + mu[j]) / den
                    if z < -mu[j]
                    else 0.0
                )
                d = nb - b[j]
                if d != 0.0:
                    b[j] = nb
                    g -= G[j] * d
                    big = max(big, abs(d) * np.sqrt(dg[j]))
            if big < ctol:
                break
        A = np.flatnonzero((b != 0.0) | (mu == 0.0))
        if len(A):
            M = G[np.ix_(A, A)] + np.diag(lam2[A])
            try:
                x = np.linalg.solve(M, c[A] - mu[A] * np.sign(b[A]))
            except np.linalg.LinAlgError:
                x = None
            if x is not None and np.all((np.sign(x) == np.sign(b[A])) | (mu[A] == 0.0)):
                bt = np.zeros(len(c))
                bt[A] = x
                if _kkt_ok(G, c, mu, lam2, bt, tol):
                    return bt
        ctol *= 1e-2
    return None


def enet_coef_pf(enet_coef, lasso_homotopy, g, c, n, alpha, l1, pf) -> np.ndarray:
    """reclasso_har.enet_coef with penalty factors; pf all 1 -> exactly enet_coef."""
    if np.all(pf == 1.0):
        return enet_coef(g, c, n, alpha, l1)
    lam2 = n * alpha * (1.0 - l1)
    mu = n * alpha * l1
    d = 1.0 / pf
    gr = g * np.outer(d, d)
    gr[np.diag_indices_from(gr)] += lam2 * d
    th, _, _ = lasso_homotopy(gr, c * d, mu)
    b = th * d
    muv, l2v = mu * pf, lam2 * pf
    tol = 1e-9 * max(float(np.abs(c).max()), 1.0)
    if _kkt_ok(g, c, muv, l2v, b, tol):
        REPAIR["kkt_ok"] += 1
        return b
    bt = _active_repair(g, c, muv, l2v, b, tol)
    if bt is not None:
        REPAIR["active_repair"] += 1
        return bt
    bt = _cd_polish(g, c, muv, l2v, b, tol)
    if bt is not None:
        REPAIR["cd_polish"] += 1
        return bt
    REPAIR["fail"] += 1
    return b


def batch_theta_pf(ns: dict, Xa, yr, locked, a, l1, pf) -> np.ndarray:
    """The spec's _batch_theta (FWL on the locked block + the batch elastic net on the rest) with
    penalty factors; pf all 1 -> the spec's floating-point operations exactly."""
    exog = ~locked
    Hh, Ee = Xa[:, locked], Xa[:, exog]
    Hp = np.linalg.pinv(Hh)
    bHy, bHE = Hp @ yr, Hp @ Ee
    Eres, yres = Ee - Hh @ bHE, yr - Hh @ bHy
    pfE = pf[exog]
    if np.all(pfE == 1.0):
        bE = ns["enet_coef"](Eres.T @ Eres, Eres.T @ yres, len(yr), a, l1)
    else:
        import src.models.reclasso_har as RH

        bE = enet_coef_pf(
            ns["enet_coef"],
            RH.lasso_homotopy,
            Eres.T @ Eres,
            Eres.T @ yres,
            len(yr),
            a,
            l1,
            pfE,
        )
    th = np.zeros(Xa.shape[1])
    th[locked] = bHy - bHE @ bE
    th[exog] = bE
    return th


# ============================================================================ STANDARDIZE_PENALTY (2026-10-22)
def col_mode(Xa: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(modal value, its count) of every column of Xa (exact equality; a tie returns one of the tied values, with the
    same count)."""
    n, p = Xa.shape
    S = np.sort(Xa, axis=0)
    new = np.empty((n, p), dtype=bool)
    new[0] = True
    np.not_equal(S[1:], S[:-1], out=new[1:])
    idx = np.arange(n)[:, None]
    run = idx - np.maximum.accumulate(np.where(new, idx, 0), axis=0) + 1
    k = run.argmax(axis=0)
    cols = np.arange(p)
    return S[k, cols], run[k, cols]


# evid_ref codes: no evidence rule; the k = 1 column is not in the design
EV_EXEMPT, EV_ABSENT = -1, -2
_EV_VALUE = re.compile(r"^adj_(.+)_ma_(\d+)$")
_EV_AVAIL = re.compile(r"^(.+)_avail_ma_(\d+)$")


def evid_ref(names: list[str], bb: np.ndarray) -> np.ndarray:
    """2026-10-22 (RULE_std.txt addendum 2): for every column of [X, 1] the index of the column whose off-mode row count
    is its evidence -- v's own k = 1 column of the same kind (adj_<v>_ma_1 for adj_<v>_ma_<k>, <v>_avail_ma_1 for
    <v>_avail_ma_<k>); EV_ABSENT when that column is not in the design; EV_EXEMPT for the backbone (HAR + calendar) and
    the intercept; a non-backbone column of neither form keeps its own count."""
    idx = {n: j for j, n in enumerate(names)}
    ref = np.full(len(names) + 1, EV_EXEMPT, dtype=np.int64)
    for j, n in enumerate(names):
        if bb[j]:
            continue
        m = _EV_VALUE.match(n)
        k1 = f"adj_{m.group(1)}_ma_1" if m else None
        if m is None:
            m = _EV_AVAIL.match(n)
            k1 = f"{m.group(1)}_avail_ma_1" if m else None
        ref[j] = j if k1 is None else idx.get(k1, EV_ABSENT)
    return ref


def group_evidence(ev_own: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Each column's evidence (addendum 2) from the per-column off-mode counts: the count of its reference column; 0 if
    that column is absent; no limit (exempt) for the backbone and the intercept."""
    big = np.iinfo(np.int64).max
    return np.where(
        ref >= 0, ev_own[np.maximum(ref, 0)], np.where(ref == EV_ABSENT, 0, big)
    )


def avail_k(names: list[str], bb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """2026-10-22 (RULE_std.txt addendum 3): the availability averages <v>_avail_ma_<k> with k > 1 (column indices in
    design order) and their k -- the columns the binomial availability rule applies to (the backbone exempt)."""
    idx, ks = [], []
    for j, n in enumerate(names):
        m = _EV_AVAIL.match(n)
        if m and not bb[j] and int(m.group(2)) > 1:
            idx.append(j)
            ks.append(int(m.group(2)))
    return np.array(idx, dtype=np.int64), np.array(ks, dtype=np.int64)


def av_stats(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(m, s1, C) of the window rows A of the availability averages: the means, the sums of A - m and its cross-products
    (the between-seed tracker's state; mean = m + s1 / n, cov = C / n - (s1 / n)(s1 / n)')."""
    m = A.mean(axis=0)
    D = A - m
    return m, D.sum(axis=0), D.T @ D


def av_mean_cov(st: tuple, n: int) -> tuple[np.ndarray, np.ndarray]:
    """The window means and covariance matrix of the availability averages from av_stats-type sums over n rows."""
    m, s1, C = st
    return m + s1 / n, C / n - np.outer(s1 / n, s1 / n)


def binom_mask(
    mean: np.ndarray, cov: np.ndarray, ks: np.ndarray, eligible: np.ndarray
) -> tuple[np.ndarray, list]:
    """Addendum 3, the binomial availability rule on the availability averages (k > 1) eligible after the existing
    masks: se = sqrt(p (1 - p) / k), p = the window mean; (i) near-constant: masked if the window s.d. <= se; (ii)
    near-duplicate: in design order, masked if against an earlier kept one of the same k the s.d. of the difference
    <= sqrt(2) sqrt(pbar (1 - pbar) / k), pbar the pair's mean level.  Returns (masked over the positions, records
    (position, "near-constant" | "near-duplicate", partner position or -1))."""
    q = len(ks)
    masked = np.zeros(q, dtype=bool)
    recs: list = []
    kept: dict = {}
    var = np.diag(cov)
    for i in range(q):
        if not eligible[i]:
            continue
        k = int(ks[i])
        p = float(mean[i])
        if np.sqrt(max(float(var[i]), 0.0)) <= np.sqrt(max(p * (1.0 - p), 0.0) / k):
            masked[i] = True
            recs.append((i, "near-constant", -1))
            continue
        kk = kept.setdefault(k, [])
        if kk:
            J = np.array(kk)
            pbar = 0.5 * (p + mean[J])
            sep = np.sqrt(np.maximum(pbar * (1.0 - pbar), 0.0) / k)
            sdd = np.sqrt(np.maximum(var[i] + var[J] - 2.0 * cov[i, J], 0.0))
            hit = np.flatnonzero(sdd <= np.sqrt(2.0) * sep)
            if hit.size:
                masked[i] = True
                recs.append((i, "near-duplicate", int(J[hit[0]])))
                continue
        kk.append(i)
    return masked, recs


def std_weights(
    Xa: np.ndarray,
    locked: np.ndarray,
    ref: np.ndarray,
    av: tuple,
    full: bool = False,
) -> tuple:
    """(s, out) of the fit's rows Xa (masked columns already 0): s_j = np.std(Xa[:, j]) (ddof 0); out = the non-locked
    columns with s_j < STD_MIN or (the evidence mask, 2026-10-22, addenda 1 and 2) an evidence below STD_MIN_EVIDENCE:
    the rows that differ from the modal value of the column's reference column (evid_ref: the variable's k = 1 column),
    and then (addendum 3) the availability averages av = avail_k(...) the binomial rule masks (masked for the fit,
    every masked column among them); s_j := 1 on out and on the locked intercept (a scale never used: the column is 0,
    resp. unscaled).  full: also (modal value, own off-mode count) of every column, the availability averages' (m, s1,
    C) and the binomial rule's records (column, kind, partner column)."""
    s = np.std(Xa, axis=0)
    v, cnt = col_mode(Xa)
    ev_own = Xa.shape[0] - cnt
    ev = group_evidence(ev_own, ref)
    out = ~locked & ((s < STD_MIN) | (ev < STD_MIN_EVIDENCE))
    ai, ak = av
    st = av_stats(Xa[:, ai])
    mean, cov = av_mean_cov(st, Xa.shape[0])
    bm, recs = binom_mask(mean, cov, ak, ~out[ai])
    out[ai[bm]] = True
    sw = np.where(out | locked, 1.0, s)
    if not full:
        return sw, out
    rr = [(int(ai[i]), kind, int(ai[j]) if j >= 0 else -1) for i, kind, j in recs]
    return sw, out, v, ev_own, st, rr


def ridge_keep(
    G0: np.ndarray, c0: np.ndarray, D: np.ndarray, keep: np.ndarray
) -> np.ndarray:
    """(G0 + diag(D)) b = c0 on the kept columns, b = 0 on the others (G0 = X'X, c0 = X'y, D = a pf s^2, 0 locked)."""
    b = np.zeros(len(c0))
    if keep.all():
        G = G0.copy()
        G[np.diag_indices_from(G)] += D
        b[:] = np.linalg.solve(G, c0)
    else:
        k = np.flatnonzero(keep)
        G = G0[np.ix_(k, k)]
        G[np.diag_indices_from(G)] += D[k]
        b[k] = np.linalg.solve(G, c0[k])
    return b


def enet_coef_w(g, c, n, alpha, l1, w1, w2) -> np.ndarray:
    """The batch elastic net  min 0.5 b'g b - c'b + sum_j (mu w1_j |b_j| + 0.5 lam2 w2_j b_j^2),  mu = n alpha l1,
    lam2 = n alpha (1 - l1) (STANDARDIZE_PENALTY: w1 = pf s, w2 = pf s^2): the lasso in th = w1 b (Gram g / w1 w1',
    L2 term lam2 w2 / w1^2 on its diagonal), then certified on the unscaled problem's KKT conditions and repaired
    as enet_coef_pf (support changes, else coordinate descent + the exact KKT solve)."""
    import src.models.reclasso_har as RH

    lam2 = n * alpha * (1.0 - l1)
    mu = n * alpha * l1
    d = 1.0 / w1
    gr = g * np.outer(d, d)
    gr[np.diag_indices_from(gr)] += lam2 * w2 * d * d
    th, _, _ = RH.lasso_homotopy(gr, c * d, mu)
    b = th * d
    muv, l2v = mu * w1, lam2 * w2
    tol = 1e-9 * max(float(np.abs(c).max()), 1.0)
    if _kkt_ok(g, c, muv, l2v, b, tol):
        REPAIR["kkt_ok"] += 1
        return b
    bt = _active_repair(g, c, muv, l2v, b, tol)
    if bt is not None:
        REPAIR["active_repair"] += 1
        return bt
    bt = _cd_polish(g, c, muv, l2v, b, tol)
    if bt is not None:
        REPAIR["cd_polish"] += 1
        return bt
    REPAIR["fail"] += 1
    return b


def batch_theta_w(Xa, yr, locked, a, l1, w1, w2) -> np.ndarray:
    """The spec's _batch_theta (FWL on the locked block, the batch elastic net on the rest) with the L1 weights w1 and
    the L2 weights w2 of STANDARDIZE_PENALTY (enet_coef_w)."""
    exog = ~locked
    Hh, Ee = Xa[:, locked], Xa[:, exog]
    Hp = np.linalg.pinv(Hh)
    bHy, bHE = Hp @ yr, Hp @ Ee
    Eres, yres = Ee - Hh @ bHE, yr - Hh @ bHy
    bE = enet_coef_w(Eres.T @ Eres, Eres.T @ yres, len(yr), a, l1, w1[exog], w2[exog])
    th = np.zeros(Xa.shape[1])
    th[locked] = bHy - bHE @ bE
    th[exog] = bE
    return th


def batch_z(Z, yr, locked, a, l1, pf) -> np.ndarray:
    """The KKT-certified batch fit in the scaled coordinates (addendum 3): the spec's FWL + batch elastic net with the
    penalty factors pf only (batch_theta_pf with the spec's own enet_coef)."""
    import src.models.reclasso_har as RH

    return batch_theta_pf({"enet_coef": RH.enet_coef}, Z, yr, locked, a, l1, pf)


def fit_std(
    Xa, yr, locked, kind, a, l1, pf, ref, av, sw=None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One standardized batch fit on the rows (Xa, yr) (masked columns 0): (theta, s, out) with s, out of std_weights
    (evidence references ref = evid_ref, availability averages av = avail_k; or sw, std_weights of these rows computed
    by the caller); the out columns masked for the fit (theta_j = 0).  Addendum 3: solved in the scaled coordinates
    z = (x - m) / s (m, s of these rows) with the penalty factors pf only, intercept free -- ridge (Z'Z + diag(a pf))
    gamma = Z'y on the kept columns, lasso / enet batch_z -- and back-transformed (beta = gamma / s, intercept
    gamma_0 - m' beta): the weighted + centred problem of addenda 0-2."""
    s, out = std_weights(Xa, locked, ref, av) if sw is None else sw
    m = np.where(locked | out, 0.0, Xa.mean(axis=0))
    Z = (Xa - m) / s
    Z[:, out] = 0.0
    if kind == "ridge":
        D = np.where(locked, 0.0, a * pf)
        g = ridge_keep(Z.T @ Z, Z.T @ yr, D, ~out)
    else:
        g = batch_z(Z, yr, locked, a, l1, pf)
    th = g / s
    th[locked] = g[locked] - m @ th
    return th, s, out


def _trailing_const(Xw: np.ndarray) -> np.ndarray:
    """Rows since each column of the chronological window Xw last changed value, capped at its length (the spec's
    _init_runs, columns only)."""
    n = Xw.shape[0]
    bad = ~(Xw == Xw[-1])[::-1]
    return np.where(bad.any(axis=0), bad.argmax(axis=0), n).astype(np.int64)


# ============================================================================ the walk
class ConstSched(dict):
    """A schedule with one penalty for every block (--sched holding one number)."""

    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha

    def __missing__(self, key):
        return self.alpha


def make_class(ns: dict, r: float, sched: dict | None, g0: int, rec: list):
    """The spec's RollingTunedLinear with penalty factors, the residual record, and an optional schedule."""
    Base = ns["RollingTunedLinear"]
    fws = ns["forward_window_split"]
    VAL_TAIL, EMBARGO, TUNE_PER = ns["VAL_TAIL"], ns["EMBARGO"], ns["TUNE_PER"]
    import src.models.reclasso_har as RH

    class PFTuned(Base):
        bb = None  # bool over the feature columns, set by fit_predict from the column names
        ev_ref = None  # 2026-10-22: evid_ref of the column names (the evidence mask's k = 1 columns), set with bb
        av = None  # 2026-10-22 (addendum 3): avail_k of the column names (the binomial rule's columns), set with bb
        # 2026-10-22 (src/models/reclasso_har.py WALK_KKT_RESEED, read at every solve): with the flag on, every
        # lasso / elastic-net solve records (solve, violation / max(max|c|, 1), ok); every KKT-triggered reseed
        # (solve, relative violation before, after, ok after); seconds in the check and in the reseeds
        kkt_rows: list = []
        kkt_reseeds: list = []
        kkt_sec: list = [0.0, 0.0]
        # 2026-10-22 (STANDARDIZE_PENALTY, read at every call): every cold seed's (solve, s with NaN on the masked /
        # left-out / locked columns, columns newly masked by s < STD_MIN); ridge: [seed solve, max_j |s_j / s_j(seed)
        # - 1| over the block's bars, max columns left out beyond the mask]; --std-check-blockends: (solve, forecast
        # with the held weights, forecast with the current window's weights, max |s_now / s_held - 1|)
        # evidence mask (addendum 1): std_seeds entries also carry the columns the rule leaves out beyond the
        # identifiability mask at that seed; std_evid_reseeds: (solve, columns added) of every evidence-triggered
        # lasso / enet reseed
        std_seeds: list = []
        std_ridge: list = []
        std_blockend: list = []
        std_blockend_sec: list = [0.0]
        std_evid_reseeds: list = []
        # addendum 3: (solve, [(column, kind, partner)]) of the binomial availability rule at every cold seed;
        # (solve, columns) of every binomial-rule-triggered lasso / enet reseed
        std_binom: list = []
        std_binom_reseeds: list = []

        def _std_on(self) -> bool:
            """STANDARDIZE_PENALTY applies (a penalized fit: the OLS at alpha 0 is never weighted)."""
            return STANDARDIZE_PENALTY and self.alpha_ > 0.0

        def _recompute_mask(self, Xraw):
            """The spec's identifiability mask; 2026-10-22: the evidence-only mask restarts with it (tune boundary)."""
            super()._recompute_mask(Xraw)
            self._evid_only = np.zeros(len(self._maskout), dtype=bool)

        def _evid_update(self, ua, ur) -> None:
            """The evidence tracker (addendum 1): rows of the window that differ from the seed-time modal value, per
            column, moved by the entering and leaving rows (masked design values)."""
            self._ev_d += (ua != self._ev_v).astype(np.int64) - (
                ur != self._ev_v
            ).astype(np.int64)

        def init_window(self, X_win, y_win):
            pf = np.ones(np.asarray(X_win).shape[1] + 1)
            pf[:-1][self.bb] = r
            self._pf = pf
            self._warm = False  # 2026-10-22: the state came from a warm update (WALK_KKT_RESEED)
            return super().init_window(X_win, y_win)

        def roll(self, x_in, y_in, x_out, y_out):
            """The spec's roll; 2026-10-22: notes whether it was a warm update (the two enet_online calls), not
            ridge and not the between-tune mask-addition reseed (a cold seed).  STANDARDIZE_PENALTY: ridge, the
            rank-one update of the centred X'X, X'y and the window's column moments (_roll_std_ridge) in place of
            Sherman-Morrison; lasso / elastic net, the spec's roll on the centred state (_roll_std_enet)."""
            if self.kind_ == "ridge" and self._std_on():
                self._roll_std_ridge(x_in, y_in, x_out, y_out)
                self._warm = False
                return
            if self._std_on():
                self._warm = self._roll_std_enet(x_in, y_in, x_out, y_out)
                return
            n0 = len(Base.reseed_trace)
            super().roll(x_in, y_in, x_out, y_out)
            self._warm = self.kind_ != "ridge" and len(Base.reseed_trace) == n0

        def _roll_std_enet(self, x_in, y_in, x_out, y_out):
            """STANDARDIZE_PENALTY, lasso / elastic net: the spec's roll (the RAW ring; the between-tune
            mask-addition reseed; the masked update rows) with (addendum 1-2) the evidence tracker on the k = 1
            counts -- a live column whose evidence drops below STD_MIN_EVIDENCE joins the mask and the walk reseeds
            cold -- and the two enet_online updates on rows shifted by the seed-time column means (the centred
            state, addendum 2).  Returns whether the state came from the warm update."""
            ua = np.append(np.asarray(x_in, dtype=np.float64), 1.0)
            ur = np.append(np.asarray(x_out, dtype=np.float64), 1.0)
            prev = self._X[self._ptr - 1].copy()  # newest row in the ring (RAW)
            self._X[self._ptr] = ua
            self._y[self._ptr] = float(y_in)
            self._ptr = (self._ptr + 1) % len(self._y)
            if self._track:
                gone = self._degenerate_live(ua, prev)
                if (
                    gone.any() and self.l1_ == 1.0
                ):  # the spec's between-tune mask addition
                    self._maskout = self._maskout | gone
                    Base.reseed_trace.append(self._n_solve)
                    self._seed()
                    return False
            ua = ua.copy()
            ur = ur.copy()
            ua[self._maskout] = 0.0
            ur[self._maskout] = 0.0
            self._evid_update(ua, ur)
            ev = group_evidence(self._ev_d, self.ev_ref)  # addendum 2: the k = 1 counts
            low = (ev < STD_MIN_EVIDENCE) & ~self._maskout & ~self._locked
            if low.any():  # masked like the identifiability mask; cold reseed (weights, offsets, modes refreshed)
                self._maskout = self._maskout | low
                self._evid_only = self._evid_only | low
                PFTuned.std_evid_reseeds.append(
                    (g0 + self._n_solve, [int(j) for j in np.flatnonzero(low)])
                )
                self._seed()
                return False
            # addendum 3: the binomial rule on the live availability averages, from the tracked window moments
            ai, ak = self.av
            self._av_update(ua[ai], ur[ai])
            mean, cov = av_mean_cov(self._av_st, len(self._y))
            bm, _ = binom_mask(mean, cov, ak, ~self._maskout[ai])
            if bm.any():  # masked like the identifiability mask; cold reseed
                low = np.zeros(len(self._maskout), dtype=bool)
                low[ai[bm]] = True
                self._maskout = self._maskout | low
                self._evid_only = self._evid_only | low
                PFTuned.std_binom_reseeds.append(
                    (g0 + self._n_solve, [int(j) for j in ai[bm]])
                )
                self._seed()
                return False
            # the scaled rows (addendum 3; masked entries stay 0: offset 0, scale 1)
            uc, rc = (ua - self._ctr) / self._sc, (ur - self._ctr) / self._sc
            self._th, self._A, self._s = RH.enet_online(
                self._Gr,
                self._c,
                self._mu_vec,
                uc,
                float(y_in),
                +1.0,
                self._th,
                self._A,
                self._s,
                self._locked,
            )
            self._th, self._A, self._s = RH.enet_online(
                self._Gr,
                self._c,
                self._mu_vec,
                rc,
                float(y_out),
                -1.0,
                self._th,
                self._A,
                self._s,
                self._locked,
            )
            return True

        def _av_update(self, a_in, a_out) -> None:
            """The binomial rule's tracker (addendum 3): the availability averages' sums and cross-products about the
            seed-time means, moved by the entering and leaving rows (masked design values)."""
            m, s1, C = self._av_st
            di, do = a_in - m, a_out - m
            s1 += di - do
            C += np.outer(di, di)
            C -= np.outer(do, do)

        def coef_original(self) -> np.ndarray:
            """The walk's coefficients in the original coordinates (beta = theta / s, intercept theta_0 - m' beta)."""
            b = self._th / self._sc
            b[-1] = self._th[-1] - self._ctr @ b
            return b

        def _roll_std_ridge(self, x_in, y_in, x_out, y_out):
            """The spec's ridge roll (the RAW ring, the masked update rows) with STANDARDIZE_PENALTY's state: X'X and
            X'y of the rows shifted by the seed-time column means (the centred state, addendum 2), rank-one, the
            column sums and sums of squares about the same means, the run length of every column's newest value
            (exact constancy in the window) and the evidence counts."""
            ua = np.append(np.asarray(x_in, dtype=np.float64), 1.0)
            ur = np.append(np.asarray(x_out, dtype=np.float64), 1.0)
            prev = self._X[self._ptr - 1].copy()  # newest row in the ring (RAW)
            self._X[self._ptr] = ua
            self._y[self._ptr] = float(y_in)
            self._ptr = (self._ptr + 1) % len(self._y)
            n = len(self._y)
            self._sd_run = np.where(ua == prev, np.minimum(self._sd_run + 1, n), 1)
            if self._maskout.any():  # masked columns stay zero in the update rows
                ua = ua.copy()
                ur = ur.copy()
                ua[self._maskout] = 0.0
                ur[self._maskout] = 0.0
            di, do = ua - self._ctr, ur - self._ctr
            self._G += np.outer(di, di)
            self._G -= np.outer(do, do)
            self._c += di * float(y_in)
            self._c -= do * float(y_out)
            self._sd_s1 += di - do
            self._sd_s2 += di * di - do * do
            self._evid_update(ua, ur)  # 2026-10-22: the evidence mask, every bar

        def _std_ridge_solve(self):
            """STANDARDIZE_PENALTY, ridge: s of the current window (the moments; 0 on a column constant over the
            window), the columns with s < STD_MIN or (evidence mask) fewer than STD_MIN_EVIDENCE rows off the modal
            value left out of this solve, (X'X + diag(a pf s^2)) b = X'y on the centred state (theta_c)."""
            n = len(self._y)
            var = self._sd_s2 / n - (self._sd_s1 / n) ** 2
            s = np.sqrt(np.maximum(var, 0.0))
            s[self._sd_run >= n] = 0.0
            ev = group_evidence(self._ev_d, self.ev_ref)  # addendum 2: the k = 1 counts
            out = ~self._locked & ((s < STD_MIN) | (ev < STD_MIN_EVIDENCE))
            # addendum 3: the binomial rule per bar, on the window moments of the centred state
            ai, ak = self.av
            s1a = self._sd_s1[ai] / n
            mean = self._ctr[ai] + s1a
            cov = self._G[np.ix_(ai, ai)] / n - np.outer(s1a, s1a)
            bm, _ = binom_mask(mean, cov, ak, ~out[ai])
            out[ai[bm]] = True
            rec_b = PFTuned.std_ridge[-1]
            rec_b[3] = max(rec_b[3], int(bm.sum()))
            s = np.where(out | self._locked, 1.0, s)
            # exact per-bar scaling (addendum 3): z = (x - m0) / s_t, (Z'Z + diag(a pf)) gamma = Z'y, theta = gamma / s_t
            D = np.where(self._locked, 0.0, self.alpha_ * self._pf)
            g = ridge_keep(self._G / np.outer(s, s), self._c / s, D, ~out)
            self._th = g / s
            rec_r = PFTuned.std_ridge[-1]
            ok = ~out & ~self._locked & np.isfinite(self._sd_seed)
            if ok.any():
                rec_r[1] = max(
                    rec_r[1], float(np.abs(s[ok] / self._sd_seed[ok] - 1.0).max())
                )
            rec_r[2] = max(rec_r[2], int((out & ~self._maskout).sum()))

        def predict_one(self, x):
            """The spec's predict_one; --std-check-blockends: also the forecast with the block-end refit."""
            if self._std_on():  # the scaled state (addendum 3): ((x - m) / s)' theta
                xa = (
                    np.append(np.asarray(x, dtype=np.float64), 1.0) - self._ctr
                ) / self._sc
                p = float(xa @ self._th)
            else:
                p = super().predict_one(x)
            alt = getattr(self, "_std_alt", None)
            if alt is not None:
                th_alt, s_rel = alt
                q = float(np.append(np.asarray(x, dtype=np.float64), 1.0) @ th_alt)
                PFTuned.std_blockend.append((g0 + self._n_solve - 1, p, q, s_rel))
                self._std_alt = None
            return p

        def _std_blockend(self):
            """--std-check-blockends: at a block's last solve, the batch fit of the current window with the current
            window's weights (the held-weight approximation's forecast effect); the walk's state is not touched."""
            t0 = time.perf_counter()
            Xa, yr = self._masked_window()
            rep = dict(REPAIR)
            th, s, out = fit_std(
                Xa,
                yr,
                self._locked,
                self.kind_,
                self.alpha_,
                self.l1_,
                self._pf,
                self.ev_ref,
                self.av,
            )
            REPAIR.update(rep)
            ok = ~out & ~self._locked & np.isfinite(self._sd_seed)
            s_rel = (
                float(np.abs(s[ok] / self._sd_seed[ok] - 1.0).max())
                if ok.any()
                else 0.0
            )
            self._std_alt = (th, s_rel)
            PFTuned.std_blockend_sec[0] += time.perf_counter() - t0

        def solve(self):
            """The spec's solve; 2026-10-22, WALK_KKT_RESEED on: after a warm update (not at a tune solve, whose
            cold seed replaces it), the KKT conditions of the current window's problem on the warm state (Gr, c,
            mu_vec; reclasso_har.walk_kkt, the batch certification's tolerance); if they fail, the cold seed on the
            current window at the penalty and mask in force (_seed), and the walk goes on warm from it.
            STANDARDIZE_PENALTY, ridge: the tune solve's cold seed, else one solve on this bar's window
            (_std_ridge_solve)."""
            tune = self._n_solve > 0 and self._n_solve % TUNE_PER == 0
            warm = self._warm and not tune
            self._warm = False
            if self.kind_ == "ridge" and self._std_on():
                if tune:
                    self._tune()  # its cold seed solves
                else:
                    self._std_ridge_solve()
                self._n_solve += 1
                return
            super().solve()
            self._kkt_after(warm)
            if (
                STD_CHECK_BLOCKENDS
                and self.kind_ != "ridge"
                and self._std_on()
                and self._n_solve % TUNE_PER == 0
            ):
                self._std_blockend()  # the next solve is a tune: this one is the block's last

        def _kkt_after(self, warm: bool) -> None:
            """WALK_KKT_RESEED (2026-10-22; solve's docstring), after the spec's solve."""
            if self.kind_ == "ridge" or not RH.WALK_KKT_RESEED:
                return
            # this solve's global index = the OOS row it forecasts
            s = g0 + self._n_solve - 1
            t0 = time.perf_counter()
            ok, v, sc = RH.walk_kkt(self._Gr, self._c, self._mu_vec, self._th)
            PFTuned.kkt_sec[0] += time.perf_counter() - t0
            if warm and not ok:
                t1 = time.perf_counter()
                # (STANDARDIZE_PENALTY's seed record: this solve, already counted in _n_solve)
                self._seed_off = 1
                self._seed()
                self._seed_off = 0
                ok2, v2, sc2 = RH.walk_kkt(self._Gr, self._c, self._mu_vec, self._th)
                PFTuned.kkt_sec[1] += time.perf_counter() - t1
                PFTuned.kkt_reseeds.append((s, v / sc, v2 / sc2, ok2))
                ok, v, sc = ok2, v2, sc2
            PFTuned.kkt_rows.append((s, v / sc, ok))

        def _seed(self):
            """The spec's _seed with alpha * pf_j in place of alpha.  STANDARDIZE_PENALTY: _seed_std."""
            if self._std_on():
                self._seed_std()
                return
            Xa, yr = self._masked_window()
            tw = len(yr)
            kind, a, l1 = self.kind_, self.alpha_, self.l1_
            pf = self._pf
            if kind == "ridge":
                D = np.where(self._locked, 0.0, a * pf)
                G = Xa.T @ Xa
                G[np.diag_indices_from(G)] += D
                self._K = np.linalg.inv(G)
                self._c = Xa.T @ yr
                self._th = self._K @ self._c
            else:
                mu, lam2 = tw * a * l1, tw * a * (1.0 - l1)
                self._mu_vec = np.where(self._locked, 0.0, mu * pf)
                Gr = Xa.T @ Xa
                ei = np.where(~self._locked)[0]
                Gr[ei, ei] += lam2 * pf[ei]
                self._Gr = Gr
                self._c = Xa.T @ yr
                th = batch_theta_pf(ns, Xa, yr, self._locked, a, l1, pf)
                self._A = list(np.where((np.abs(th) > 1e-9) | self._locked)[0])
                self._s = [
                    0.0 if self._locked[j] else float(np.sign(th[j])) for j in self._A
                ]
                self._th = th

        def _seed_std(self):
            """STANDARDIZE_PENALTY's cold seed on the current (masked) window: s = np.std of the window (std_weights).
            Ridge: X'X, X'y, the column moments about the window means and the constancy run lengths, then the
            bar's solve (_std_ridge_solve, exact per-bar scaling).  Lasso / elastic net: a column with s < STD_MIN, or
            (the evidence mask, addenda 1-2) whose variable's k = 1 column has fewer than STD_MIN_EVIDENCE rows off its
            modal value, or (addendum 3) that the binomial availability rule masks, joins the mask (the columns masked
            by these rules alone are re-evaluated, i.e. may come back, at every seed); the scaled state (addendum 3):
            z = (x - m) / s with m, s of this window (self._ctr, self._sc; 0 / 1 on the masked columns and on the
            intercept) held until the next seed, Gr = Z'Z + n a (1 - l1) pf on the diagonal, mu_j = n a l1 pf_j, the
            KKT-certified batch solution in z (batch_z) and its active set; theta = gamma in z (forecast z(x)' theta).
            Both: the seed-time modal values and evidence counts the between-seed evidence tracker starts from (and,
            lasso / enet, the availability averages' moments the binomial tracker starts from)."""
            kind, a, l1 = self.kind_, self.alpha_, self.l1_
            evo = getattr(self, "_evid_only", None)
            if kind != "ridge" and evo is not None and evo.any():
                self._maskout = self._maskout & ~evo  # un-masking: a seed-time event
            self._evid_only = np.zeros(len(self._maskout), dtype=bool)
            Xa, yr = self._masked_window()
            tw = len(yr)
            # the tracker is exact between seeds <= TUNE_PER solves apart iff the window holds > 2 (250 + TUNE_PER)
            assert tw > 2 * (STD_MIN_EVIDENCE + TUNE_PER), tw
            pf = self._pf
            s, out, v, ev, _, brec = std_weights(
                Xa, self._locked, self.ev_ref, self.av, full=True
            )
            new = out & ~self._maskout
            nan_s = np.where(out | self._locked, np.nan, s)
            # the solve (OOS row) whose forecast first uses this seed
            at = g0 + self._n_solve - getattr(self, "_seed_off", 0)
            PFTuned.std_seeds.append((at, nan_s, int(new.sum()), np.flatnonzero(new)))
            PFTuned.std_binom.append((at, brec))
            self._sd_seed = nan_s
            self._ev_v, self._ev_d = v, ev.astype(np.int64)
            if kind == "ridge":
                self._sd_run = _trailing_const(Xa)
                self._ctr = np.where(self._locked, 0.0, Xa.mean(axis=0))
                self._sc = np.ones(
                    len(self._ctr)
                )  # (the per-bar scale is applied in the solve)
                Xa -= self._ctr  # the centred state (addendum 2); Xa is a fresh array
                self._G = Xa.T @ Xa
                self._c = Xa.T @ yr
                self._sd_s1 = Xa.sum(axis=0)
                self._sd_s2 = (Xa * Xa).sum(axis=0)
                PFTuned.std_ridge.append([g0 + self._n_solve, 0.0, 0, 0])
                self._std_ridge_solve()
                return
            if new.any():  # masked for the fit, and (scales held) until the next seed re-evaluates them
                self._maskout = self._maskout | new
                self._evid_only = new
                Xa[:, new] = 0.0
            # addendum 3: the binomial rule's between-seed tracker (masked design, after this seed's masks)
            self._av_st = av_stats(Xa[:, self.av[0]])
            # the scaled state (addendum 3): z = (x - m) / s, m and s of this window, held until the next seed
            self._ctr = np.where(self._locked, 0.0, Xa.mean(axis=0))
            self._sc = s  # (1 on the masked columns and the intercept)
            Xa -= self._ctr
            Xa /= self._sc
            mu, lam2 = tw * a * l1, tw * a * (1.0 - l1)
            self._mu_vec = np.where(self._locked, 0.0, mu * pf)
            Gr = Xa.T @ Xa
            ei = np.where(~self._locked)[0]
            Gr[ei, ei] += lam2 * pf[ei]
            self._Gr = Gr
            self._c = Xa.T @ yr
            th = batch_z(Xa, yr, self._locked, a, l1, pf)
            self._A = list(np.where((np.abs(th) > 1e-9) | self._locked)[0])
            self._s = [
                0.0 if self._locked[j] else float(np.sign(th[j])) for j in self._A
            ]
            self._th = th

        def _tune(self):
            """The spec's _tune (mask, forward split, grid argmin of the tail MSE, cold seed) with
            penalty factors and the tail residuals of every grid alpha recorded; with a schedule, the
            scheduled alpha is seeded instead of the argmin (no grid fits)."""
            Xraw, _ = self._window()
            self._recompute_mask(Xraw)
            Base.mask_trace.append(int(self._maskout.sum()))
            blk = (g0 + self._n_solve) // TUNE_PER
            t0 = time.process_time()
            if sched is None:
                Xa, yr = self._masked_window()
                n = len(yr)
                fit_lo, fit_hi, val_lo, val_hi = fws(n, n, VAL_TAIL, EMBARGO)
                Xf, yf = Xa[fit_lo:fit_hi], yr[fit_lo:fit_hi]
                Xv, yv = Xa[val_lo:val_hi], yr[val_lo:val_hi]
                best = None
                res = []
                pf = self._pf
                for kind, a, l1 in Base.grid:
                    if STANDARDIZE_PENALTY and a > 0.0:
                        # 2026-10-22: the weights from the fit block's s
                        th = fit_std(
                            Xf, yf, self._locked, kind, a, l1, pf, self.ev_ref, self.av
                        )[0]
                    elif kind == "ridge":
                        D = np.where(self._locked, 0.0, a * pf)
                        G = Xf.T @ Xf
                        G[np.diag_indices_from(G)] += D
                        th = np.linalg.solve(G, Xf.T @ yf)
                    else:
                        th = batch_theta_pf(ns, Xf, yf, self._locked, a, l1, pf)
                    res.append(Xv @ th - yv)
                    mse = float(np.mean((Xv @ th - yv) ** 2))
                    if best is None or mse < best[0]:
                        best = (mse, kind, a, l1)
                _, self.kind_, self.alpha_, self.l1_ = best
                resid = np.array(res)
            else:
                a_s = float(sched[blk])
                hit = [g for g in Base.grid if g[1] == a_s]
                assert len(hit) == 1, (blk, a_s)
                self.kind_, self.alpha_, self.l1_ = hit[0]
                resid = None
            Base.trace.append((self.kind_, self.alpha_, self.l1_))
            self._seed()
            rec.append(
                dict(
                    block=int(blk),
                    solve=int(g0 + self._n_solve),
                    alpha=float(self.alpha_),
                    masked=int(self._maskout.sum()),
                    resid=resid,
                    cpu=time.process_time() - t0,
                )
            )

    return PFTuned


# 2026-10-20 (user): parked -- the 13-bar rerun's copy of ConstSched (above; the same schedule).  Was:
# class _ConstSched(dict):
#     """2026-10-19 (--sched const:A): the schedule that maps every block to one penalty."""
#
#     def __init__(self, alpha: float) -> None:
#         super().__init__()
#         self.alpha = float(alpha)
#
#     def __missing__(self, blk):
#         return self.alpha


def parse_sched(spec: str):
    """--sched: causal (None), const:A (the 13-bar rerun), or a JSON file holding one number (the 48-interval
    rerun) or {block: alpha}."""
    if spec == "causal":
        return None
    if spec.startswith("const:"):
        return ConstSched(float(spec.split(":", 1)[1]))
    js = json.loads(Path(spec).read_text())
    if isinstance(js, (int, float)):  # one penalty at every re-choice (the frozen walk)
        return ConstSched(float(js))
    return {int(k): float(v) for k, v in js.items()}


def ols_path(a) -> str:
    """2026-10-20 (user): --est ols as ridge0 (the 48-interval rerun) or minnorm (the 13-bar rerun); auto picks
    the one each rerun ran: ridge0 under --rerun, minnorm otherwise."""
    if a.ols != "auto":
        return a.ols
    return "ridge0" if RERUN else "minnorm"


PINV_RCOND = 1e-15  # numpy.linalg.pinv's default (src/unification.py PINV_RCOND)


def make_ols_class(tune_per: int):
    """2026-10-19: the pooled OLS of the HAR benchmark set (replaces a0_ols_har, src/unification.py): min-norm
    least squares on the centred trailing window, coefficients = eigh pseudo-inverse of the centred Gram
    (eigenvalues below PINV_RCOND x the largest treated as zero) times the centred cross-product, the
    intercept free (from the means); the window sums roll rank-one every bar and are rebuilt exactly every
    ``tune_per`` solves (MultiStageBacktest's reinit_every)."""

    class RollingMinNormOLS:
        reinit_every = tune_per
        rank_trace: list = []

        def init_window(self, X_win, y_win):
            X = np.asarray(X_win, dtype=np.float64)
            y = np.asarray(y_win, dtype=np.float64)
            self._n = len(y)
            self._S = X.T @ X
            self._c = X.T @ y
            self._sx = X.sum(axis=0)
            self._sy = float(y.sum())
            return self

        def roll(self, x_in, y_in, x_out, y_out):
            u = np.asarray(x_in, dtype=np.float64)
            v = np.asarray(x_out, dtype=np.float64)
            self._S += np.outer(u, u) - np.outer(v, v)
            self._c += u * float(y_in) - v * float(y_out)
            self._sx += u - v
            self._sy += float(y_in) - float(y_out)

        def solve(self):
            n = self._n
            m, ybar = self._sx / n, self._sy / n
            G = self._S - n * np.outer(m, m)
            c = self._c - n * m * ybar
            lam, V = np.linalg.eigh(0.5 * (G + G.T))
            keep = lam > PINV_RCOND * max(float(lam.max()), 0.0)
            Vk = V[:, keep]
            self._b = Vk @ ((Vk.T @ c) / lam[keep])
            self._b0 = ybar - float(m @ self._b)
            RollingMinNormOLS.rank_trace.append(int(keep.sum()))

        def predict_one(self, x):
            return float(np.asarray(x, dtype=np.float64) @ self._b + self._b0)

    return RollingMinNormOLS


def design(a) -> None:
    """Design cache: the spec's run_executor on the whole series with a no-op walk, keeping what its
    _backtest_and_save hands the walk and the smearing step (the prescaled X, y, the column names, the OOS
    rows' baselines and date labels).  The all_features data prep (load, transforms, whole-series rolling
    robust scaling of ~640 columns) costs most of an hour; every walk and pre-tune then reads the cache."""
    ns = spec_section("ridge", a.bucket, 0, -1, 0)
    import src.backtest.executor as EX

    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cap: dict = {}
    orig = EX.apply_duan_smearing

    def spy(preds, y_oos, b_oos):
        cap["b_oos"] = np.array(b_oos, dtype=np.float64)
        return orig(preds, y_oos, b_oos)

    def fit_predict(X_chunk, y_chunk, train_win_periods, hyperparams):
        cap["X"], cap["y"], cap["names"] = (
            X_chunk,
            y_chunk,
            list(hyperparams["_feature_names"]),
        )
        cap["hp"] = {k: v for k, v in hyperparams.items() if k != "_feature_names"}
        return np.zeros(len(X_chunk) - train_win_periods)

    EX.apply_duan_smearing = spy
    try:
        run_spec_executor(
            ns,
            "ridge",
            a.bucket,
            fit_predict,
            str(out / "dummy" / "results.csv"),
            0,
            -1,
            0,
            a.data,
        )
    finally:
        EX.apply_duan_smearing = orig
    dummy = Path(seg_csv(str(out / "dummy" / "results.csv")))
    d = pd.read_csv(dummy, usecols=["date"], dtype=str)
    # the matrix as the pipeline holds it: pandas' .values of a float block is column-major, and the walk's
    # BLAS calls on F-ordered operands sum in another order than on a C-ordered copy (last-bit differences
    # that the lasso / elastic net warm path can amplify), so the layout is kept (np.save records it)
    X = cap["X"]
    assert X.dtype == np.float64 and (X.flags.f_contiguous or X.flags.c_contiguous), (
        X.flags
    )
    assert X.shape[0] == W + len(d) and len(cap["b_oos"]) == len(d), (X.shape, len(d))
    np.save(out / "X.npy", X)
    np.save(out / "y.npy", np.ascontiguousarray(cap["y"], dtype=np.float64))
    np.save(out / "b_oos.npy", cap["b_oos"])
    d.to_csv(out / "dates_oos.csv", index=False)
    (out / "design.json").write_text(
        json.dumps(
            dict(
                bucket=a.bucket,
                rerun=RERUN,
                part=list(PART) if PART else None,
                raw_cols=ns["get_bucket"](a.bucket),
                names=cap["names"],
                hp=cap["hp"],
                shape=list(X.shape),
                f_order=bool(X.flags.f_contiguous),
                host=platform.node(),
                numpy=np.__version__,
                **({} if default_config() else {"config": config_dict()}),
            )
        )
    )
    dummy.unlink()
    say(f"design done: {a.bucket} X {X.shape}")


def merge(a) -> None:
    """Stack the column parts of design --part K/N into the one-pass design (see the module docstring)."""
    parts = [Path(d).resolve() for d in a.parts.split(",")]
    metas = [json.loads((d / "design.json").read_text()) for d in parts]
    assert [m["part"] for m in metas] == [
        [k + 1, len(parts)] for k in range(len(parts))
    ], [m["part"] for m in metas]
    assert len({m["bucket"] for m in metas}) == 1 and all(
        m["rerun"] == RERUN for m in metas
    )
    ref = parts[0]
    y, b = np.load(ref / "y.npy"), np.load(ref / "b_oos.npy")
    d0 = (ref / "dates_oos.csv").read_bytes()
    for d in parts[1:]:  # rows and target are the parts' common columns
        assert np.array_equal(np.load(d / "y.npy"), y) and np.array_equal(
            np.load(d / "b_oos.npy"), b
        )
        assert (d / "dates_oos.csv").read_bytes() == d0
    raw = [c for m in metas for c in m["raw_cols"]]
    common = [
        n for n in metas[0]["names"] if not _EXOG.search(n) or n.startswith("har_ma_")
    ]
    har = [n for n in common if n.startswith("har_ma_")]
    cal = [n for n in common if not n.startswith("har_ma_")]
    for m in metas:
        assert [
            n for n in m["names"] if not _EXOG.search(n) or n.startswith("har_ma_")
        ] == common
    lad = [int(n.rsplit("_", 1)[1]) for n in har]
    exog = {n for m in metas for n in m["names"] if n not in common}
    # load_and_transform's order: every adj_<col>, then every <col>_avail, then every <col>_active
    order = har
    for suf in ("adj_{}", "{}_avail", "{}_active"):
        order += [
            f"{suf.format(c)}_ma_{k}"
            for c in raw
            for k in lad
            if f"{suf.format(c)}_ma_{k}" in exog
        ]
    order += cal
    assert sorted(order) == sorted(set(order)) and set(order) == exog | set(common), (
        "column order"
    )
    src = {}
    for d, m in zip(parts, metas):
        for j, n in enumerate(m["names"]):
            src.setdefault(n, (d, j))
    X = np.empty(
        (len(y), len(order)),
        dtype=np.float64,
        order="F" if metas[0]["f_order"] else "C",
    )
    for d in parts:
        Xp = np.load(d / "X.npy", mmap_mode="r")
        for j, n in enumerate(order):
            if src[n][0] == d:
                X[:, j] = Xp[:, src[n][1]]
        del Xp
    for d, m in zip(parts[1:], metas[1:]):  # the common columns agree across parts
        Xp = np.load(d / "X.npy", mmap_mode="r")
        for n in common:
            assert np.array_equal(Xp[:, m["names"].index(n)], X[:, order.index(n)]), n
        del Xp
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "X.npy", X)
    np.save(out / "y.npy", y)
    np.save(out / "b_oos.npy", b)
    (out / "dates_oos.csv").write_bytes(d0)
    meta = dict(
        metas[0],
        part=None,
        raw_cols=raw,
        names=order,
        shape=list(X.shape),
        merged_from=[str(d) for d in parts],
    )
    (out / "design.json").write_text(json.dumps(meta))
    say(f"merge done: {meta['bucket']} X {X.shape} from {len(parts)} parts")


def check_meta(meta: dict, bucket: str) -> None:
    """The design cache was built for this bucket and this run's configuration (--rerun and the 13-bar rerun's
    options)."""
    assert meta["bucket"] == bucket, (meta["bucket"], bucket)
    assert meta.get("rerun", False) == RERUN, (meta.get("rerun"), RERUN)
    want = None if default_config() else config_dict()
    got = meta.get("config")
    if got is not None:
        got = dict(got, window=int(got["window"]))
    assert got == want, (got, want)


def cached_backtest(a, fit_predict, out_csv: Path) -> None:
    """_backtest_and_save's slice, walk and table on the design cache (the same arrays it would build)."""
    import src.backtest.executor as EX

    dz = Path(a.design)
    meta = json.loads((dz / "design.json").read_text())
    check_meta(meta, a.bucket)
    # 2026-10-20 (user): parked -- both checks are in check_meta now.  Was:
    # assert meta["bucket"] == a.bucket, (meta["bucket"], a.bucket)
    # assert meta.get("rerun", False) == RERUN, (meta.get("rerun"), RERUN)
    # the whole matrix in memory, in the pipeline's layout, and the chunk a VIEW of it, as _backtest_and_save
    # slices it (a C-ordered copy is not bit-identical to the spec's path)
    # --rerun: memory-mapped (the same bytes and layout; the concurrent walks share one page-cache copy)
    X = np.load(dz / "X.npy", mmap_mode="r" if RERUN else None)
    assert bool(X.flags.f_contiguous) == meta["f_order"], X.flags
    y = np.load(dz / "y.npy")
    b_oos = np.load(dz / "b_oos.npy")
    dts = pd.read_csv(dz / "dates_oos.csv", dtype=str)["date"].to_numpy()
    load_start = max(0, a.start - a.halo)
    actual_end = len(X) if a.end < 0 else a.end
    X_chunk = X[load_start:actual_end]
    y_chunk = y[load_start:actual_end]
    preds = fit_predict(
        X_chunk, y_chunk, W, dict(meta["hp"], _feature_names=list(meta["names"]))
    )
    y_oos = y_chunk[W:]
    sl = slice(load_start, actual_end - W)  # OOS row i <-> design row W + i
    pred_raw, true_raw = EX.apply_duan_smearing(preds, y_oos, b_oos[sl])
    res = pd.DataFrame(
        {
            "date": dts[sl],
            "horizon": 1,
            "true_adj": y_oos,
            "pred_adj": preds,
            "true_raw": true_raw,
            "pred_raw": pred_raw,
        }
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_csv, index=False)


def grid_of(ns: dict, a) -> list:
    """The estimator's penalty grid: the spec's, or (--grid-alphas, 2026-10-22: the orchestrator's grid extensions)
    the given penalties with the spec's kind and l1_ratio."""
    grid = ns["ESTIMATOR_GRIDS"][a.est]
    if not a.grid_alphas:
        return grid
    kind, _, l1 = grid[0]
    return [(kind, float(v), l1) for v in a.grid_alphas.split(",")]


def std_record(Cls, penalized: bool, names: list | None = None) -> tuple[dict, dict]:
    """STANDARDIZE_PENALTY's walk record: trace.npz arrays (flag on only) and run.json fields (the flag always; False
    for the OLS, which it never touches).  Evidence mask: per seed the columns left out beyond the identifiability
    mask (count median / max, the most frequent columns), the evidence-triggered reseeds."""
    info: dict = dict(standardize_penalty=bool(STANDARDIZE_PENALTY and penalized))
    seeds = getattr(Cls, "std_seeds", [])
    if not (STANDARDIZE_PENALTY and seeds):
        return {}, info
    S = np.array([x[1] for x in seeds])
    with np.errstate(invalid="ignore"):
        ratio = S[1:] / S[:-1]  # NaN where a column is masked / left out at either seed
    ok = np.isfinite(ratio)
    rel = np.where(ok, np.abs(ratio - 1.0), 0.0)  # the L1 weight pf s
    rel2 = np.where(ok, np.abs(ratio * ratio - 1.0), 0.0)  # the L2 weight pf s^2
    per = rel.max(axis=1) if len(rel) else np.zeros(0)
    sd = S[np.isfinite(S)]
    npz = dict(
        std_seed_solve=np.array([x[0] for x in seeds], dtype=np.int64),
        std_seed_s=S,
        std_seed_new_masked=np.array([x[2] for x in seeds], dtype=np.int64),
    )
    info |= dict(
        std_seeds=len(seeds),
        std_seed_new_masked=int(sum(x[2] for x in seeds)),
        std_s_quantiles_5_50_95=[float(v) for v in np.quantile(sd, [0.05, 0.5, 0.95])]
        if sd.size
        else [],
        std_max_rel_change_consecutive_seeds=float(per.max()) if len(per) else 0.0,
        std_median_of_per_seed_max_rel_change=float(np.median(per))
        if len(per)
        else 0.0,
        std_max_rel_change_l2_weight=float(rel2.max()) if len(rel2) else 0.0,
    )
    # 2026-10-22 (evidence mask)
    cnt = np.array([x[2] for x in seeds])
    freq: dict = {}
    for x in seeds:
        for j in x[3]:
            freq[int(j)] = freq.get(int(j), 0) + 1
    nm = list(names or []) + ["intercept"]
    top = sorted(freq.items(), key=lambda kv: -kv[1])[:25]
    er = getattr(Cls, "std_evid_reseeds", [])
    info |= dict(
        std_evid_out_per_seed_median=float(np.median(cnt)),
        std_evid_out_per_seed_max=int(cnt.max()),
        std_evid_out_per_seed_min=int(cnt.min()),
        std_evid_out_columns_ever=len(freq),
        std_evid_out_top_columns_seeds=[
            [nm[j] if j < len(nm) else j, c] for j, c in top
        ],
        std_evid_reseeds=len(er),
        std_evid_reseed_columns=sorted(
            {nm[j] if j < len(nm) else j for _, js in er for j in js}
        ),
    )
    npz |= dict(std_seed_evid_out=cnt)
    # addendum 3: the binomial availability rule at the seeds and its between-seed reseeds
    bs = getattr(Cls, "std_binom", [])
    bc = np.array([len(x[1]) for x in bs]) if bs else np.zeros(1)
    bf: dict = {}
    for _, recs in bs:
        for c, kind, q in recs:
            key = f"{nm[c]} ({kind}{', ' + nm[q] if q >= 0 else ''})"
            bf[key] = bf.get(key, 0) + 1
    br = getattr(Cls, "std_binom_reseeds", [])
    info |= dict(
        std_binom_per_seed_median=float(np.median(bc)),
        std_binom_per_seed_max=int(bc.max()),
        std_binom_columns_seeds=sorted(bf.items(), key=lambda kv: -kv[1])[:30],
        std_binom_reseeds=len(br),
        std_binom_reseed_columns=sorted({nm[j] for _, js in br for j in js}),
    )
    rr = getattr(Cls, "std_ridge", [])
    if rr:
        npz |= dict(std_ridge=np.array(rr, dtype=np.float64))
        info |= dict(
            std_ridge_blocks=len(rr),
            std_ridge_max_within_block_rel_drift=float(max(x[1] for x in rr)),
            std_ridge_max_cols_left_out_beyond_mask=int(max(x[2] for x in rr)),
            std_ridge_max_binom_per_bar=int(max(x[3] for x in rr)),
        )
    be = getattr(Cls, "std_blockend", [])
    if be:
        B = np.array(be, dtype=np.float64)
        d = np.abs(B[:, 2] - B[:, 1])
        npz |= dict(std_blockend=B)
        info |= dict(
            std_blockend_checks=len(be),
            std_blockend_max_abs_dpred=float(d.max()),
            std_blockend_max_rel_dpred=float((d / np.abs(B[:, 1])).max()),
            std_blockend_median_rel_dpred=float(np.median(d / np.abs(B[:, 1]))),
            std_blockend_max_rel_s_drift=float(B[:, 3].max()),
            std_blockend_sec=float(Cls.std_blockend_sec[0]),
        )
    return npz, info


def walk(a) -> None:
    t_wall = time.time()
    ns = spec_section(a.est, a.bucket, a.start, a.end, a.halo)
    import src.models.reclasso_har as RH

    # 2026-10-22 (--walk-kkt-reseed 0|1); empty: the module's default (on)
    if a.walk_kkt_reseed:
        RH.WALK_KKT_RESEED = a.walk_kkt_reseed == "1"
    global STD_CHECK_BLOCKENDS
    STD_CHECK_BLOCKENDS = bool(a.std_check_blockends)  # 2026-10-22 (diagnostic)
    g0 = max(0, a.start - a.halo)
    assert g0 % ns["TUNE_PER"] == 0, (
        f"chunk start {a.start} - halo {a.halo} is not a re-choice boundary"
    )
    assert a.halo in (0, W), a.halo
    sched = parse_sched(a.sched)
    # 2026-10-20 (user): parked -- the 48-interval rerun's reading, now in parse_sched.  Was:
    # sched = None
    # if a.sched != "causal":
    #     js = json.loads(Path(a.sched).read_text())
    #     if isinstance(
    #         js, (int, float)
    #     ):  # one penalty at every re-choice (the frozen walk)
    #         sched = ConstSched(float(js))
    #     else:
    #         sched = {int(k): float(v) for k, v in js.items()}
    rec: list = []
    Base = ns["RollingTunedLinear"]
    # 2026-10-19: min-norm OLS, no penalty (--est ols with --ols minnorm, the 13-bar rerun)
    if a.est == "ols" and ols_path(a) == "minnorm":
        assert sched is None and a.r == 1.0, (a.sched, a.r)
        Cls = make_ols_class(ns["TUNE_PER"])
        Base.grid = []
    else:  # (--est ols with --ols ridge0: the spec's ridge path at alpha 0, the 48-interval rerun)
        Cls = make_class(ns, a.r, sched, g0, rec)
        Base.grid = grid_of(ns, a)
    names_seen: list = []

    def fit_predict(X_chunk, y_chunk, train_win_periods, hyperparams):
        assert train_win_periods == W, train_win_periods
        names = list(hyperparams["_feature_names"])
        names_seen.append(names)
        Cls.bb = backbone_mask(names)
        if hasattr(Cls, "ev_ref"):  # 2026-10-22 (evidence mask, addendum 2)
            Cls.ev_ref = evid_ref(names, Cls.bb)
        if hasattr(Cls, "av"):  # 2026-10-22 (binomial availability rule, addendum 3)
            Cls.av = avail_k(names, Cls.bb)
        Base.trace, Base.mask_trace, Base.reseed_trace = [], [], []
        if hasattr(Cls, "kkt_rows"):  # 2026-10-22 (WALK_KKT_RESEED)
            Cls.kkt_rows, Cls.kkt_reseeds, Cls.kkt_sec = [], [], [0.0, 0.0]
        if hasattr(Cls, "std_seeds"):  # 2026-10-22 (STANDARDIZE_PENALTY)
            Cls.std_seeds, Cls.std_ridge, Cls.std_blockend = [], [], []
            Cls.std_blockend_sec, Cls.std_evid_reseeds = [0.0], []
            Cls.std_binom, Cls.std_binom_reseeds = [], []
        bt = ns["MultiStageBacktest"](
            residualizer=ns["IdentityResidualizer"](),
            regressor_factory=Cls,
            refit_frequency=int(hyperparams.get("_refit_frequency", 1)),
        )
        return bt.run(X_chunk, y_chunk, train_win_periods, desc="lin_tuned")

    out = Path(a.out).resolve()
    out_csv = out / "causal_tune_linear" / a.est / a.bucket / "results.csv"
    t_cpu = time.process_time()
    if a.design:
        cached_backtest(a, fit_predict, out_csv)
    else:
        out_csv_req = out_csv
        out_csv = Path(seg_csv(str(out_csv)))  # the table run_executor writes
        run_spec_executor(
            ns,
            a.est,
            a.bucket,
            fit_predict,
            str(out_csv_req),
            a.start,
            a.end,
            a.halo,
            a.data,
        )
    cpu = time.process_time() - t_cpu
    res = pd.read_csv(out_csv, usecols=["date"])
    # the regular-hours rows (bar-end labels 10:00 .. 16:00), byte for byte (no float re-parse)
    nl = bytes([10])
    lines = out_csv.read_bytes().split(nl)
    keep = [lines[0]] + [ln for ln in lines[1:] if ln and ln[11:16].decode() in RTH]
    (out / "results_rth.csv").write_bytes(nl.join(keep) + nl)
    G = len(Base.grid)
    blocks = np.array([x["block"] for x in rec], dtype=np.int64)
    resid = (
        np.stack([x["resid"] for x in rec])
        if sched is None and rec
        else np.zeros((len(rec), G, 0))
    )
    # 2026-10-22 (WALK_KKT_RESEED): the KKT record of the lasso / elastic-net solves, written only with the flag on
    kk = getattr(Cls, "kkt_rows", [])
    ks = getattr(Cls, "kkt_reseeds", [])
    kkt_npz: dict = {}
    kkt_info: dict = dict(walk_kkt_reseed=bool(RH.WALK_KKT_RESEED))
    if RH.WALK_KKT_RESEED and kk:
        kkt_npz = dict(
            kkt_solve=np.array([x[0] for x in kk], dtype=np.int64),
            kkt_rel=np.array([x[1] for x in kk]),
            kkt_ok=np.array([x[2] for x in kk], dtype=bool),
            kkt_reseed_solve=np.array([x[0] for x in ks], dtype=np.int64),
            kkt_reseed_rel_before=np.array([x[1] for x in ks]),
            kkt_reseed_rel_after=np.array([x[2] for x in ks]),
            kkt_reseed_ok_after=np.array([x[3] for x in ks], dtype=bool),
        )
        kkt_info |= dict(
            kkt_checks=len(kk),
            kkt_reseeds=len(ks),
            kkt_reseed_solves=[int(x[0]) for x in ks],
            kkt_not_ok_after_check=int(sum(not x[2] for x in kk)),
            kkt_max_rel_after_check=float(max(x[1] for x in kk)),
            kkt_max_rel_before_reseed=float(max((x[1] for x in ks), default=0.0)),
            kkt_check_sec=float(Cls.kkt_sec[0]),
            kkt_reseed_sec=float(Cls.kkt_sec[1]),
        )
        say(
            f"WALK_KKT_RESEED: {len(ks)} KKT-triggered reseeds in {len(kk)} solves; "
            f"{kkt_info['kkt_not_ok_after_check']} not KKT-clean after the check; "
            f"check {Cls.kkt_sec[0]:.1f} s, reseeds {Cls.kkt_sec[1]:.1f} s"
        )
    # 2026-10-22 (STANDARDIZE_PENALTY): the s.d. at every cold seed, written only with the flag on
    std_npz, std_info = std_record(
        Cls, a.est != "ols", names_seen[0] if names_seen else None
    )
    np.savez_compressed(
        out / "trace.npz",
        **kkt_npz,
        **std_npz,
        block=blocks,
        solve=np.array([x["solve"] for x in rec]),
        alpha=np.array([x["alpha"] for x in rec]),
        masked=np.array([x["masked"] for x in rec]),
        tune_cpu=np.array([x["cpu"] for x in rec]),
        resid=resid,
        grid=np.array([g[1] for g in Base.grid]),
        reseed=np.array(Base.reseed_trace, dtype=np.int64) + g0,
        names=np.array(names_seen[0]),
        backbone=backbone_mask(names_seen[0]),
        ols_rank=np.array(getattr(Cls, "rank_trace", []), dtype=np.int64),
    )
    import numpy as _np

    info = dict(
        mode="walk",
        rerun=RERUN,
        bucket=a.bucket,
        est=a.est,
        r=a.r,
        sched=a.sched,
        start=a.start,
        end=a.end,
        halo=a.halo,
        g0=g0,
        rows=int(len(res)),
        first=str(res["date"].iloc[0]),
        last=str(res["date"].iloc[-1]),
        blocks=[int(blocks.min()), int(blocks.max())] if len(blocks) else [],
        n_tunes=len(rec),
        n_cols=len(names_seen[0]),
        n_backbone=int(backbone_mask(names_seen[0]).sum()),
        reseeds=len(Base.reseed_trace),
        repair=dict(REPAIR),
        **kkt_info,
        **std_info,
        cpu_sec=cpu,
        wall_sec=time.time() - t_wall,
        host=platform.node(),
        numpy=_np.__version__,
        python=platform.python_version(),
        **({} if default_config() else {"config": config_dict()}),
    )
    (out / "run.json").write_text(json.dumps(info, indent=1))
    say(f"walk done: {json.dumps(info)}")


# ============================================================================ the pre-tune
def pretune(a) -> None:
    t_wall = time.time()
    ns = spec_section(a.est, a.bucket, a.start, a.start + 1, a.halo)
    fold_n, emb = PRE_FOLD_SESS * PERIODS_PER_DAY, PRE_EMBARGO_SESS * PERIODS_PER_DAY
    # 2026-10-22 (user, window sensitivity at 1,500 .. 3,000 days): --first-fold F fits only folds F .. PRE_FOLDS-1
    # (fold f's validation rows stay start - (PRE_FOLDS - f) * fold_n .. start - (PRE_FOLDS - 1 - f) * fold_n, as
    # with F = 0); the halo covers those folds only.  F = 0 (default) is the code above unchanged.  Was:
    # assert a.halo == PRE_FOLDS * fold_n + emb + W, a.halo
    F = a.first_fold
    assert 0 <= F < PRE_FOLDS, F
    assert a.halo == (PRE_FOLDS - F) * fold_n + emb + W, a.halo
    # 2026-10-22: --grid-alphas (the orchestrator's grid extensions)
    grid = grid_of(ns, a)
    Base = ns["RollingTunedLinear"]
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    store: dict = {}

    def fit_predict(X_chunk, y_chunk, train_win_periods, hyperparams):
        names = list(hyperparams["_feature_names"])
        bb = backbone_mask(names)
        X = np.asarray(X_chunk, dtype=np.float64)
        y = np.asarray(y_chunk, dtype=np.float64)
        n = len(X) - 1  # the rows before the start row
        Xall = np.hstack([X, np.ones((len(X), 1))])
        locked = np.zeros(Xall.shape[1], dtype=bool)
        locked[-1] = True
        pf = np.ones(Xall.shape[1])
        pf[:-1][bb] = a.r
        e_all, info = [], []
        # 2026-10-22 (--first-fold); was: for f in range(PRE_FOLDS):
        for f in range(F, PRE_FOLDS):
            v1 = n - (PRE_FOLDS - 1 - f) * fold_n
            v0 = v1 - fold_n
            f0, f1 = v0 - emb - W, v0 - emb
            assert f0 >= 0
            probe = Base()
            probe._locked = locked
            probe._recompute_mask(
                Xall[f0:f1]
            )  # the spec's identifiability mask, from the fit block
            mk = probe._maskout
            Xf = Xall[f0:f1].copy()
            Xf[:, mk] = 0.0
            Xv = Xall[v0:v1].copy()
            Xv[:, mk] = 0.0
            yf, yv = y[f0:f1], y[v0:v1]
            t0 = time.process_time()
            res = []
            std_fold: dict = {}
            if STANDARDIZE_PENALTY:
                # 2026-10-22: the weights from this fold's fit block (fit_std: s of Xf, the s < STD_MIN and evidence-
                # mask columns out)
                ref = evid_ref(names, bb)  # addendum 2
                av = avail_k(names, bb)  # addendum 3
                swf = std_weights(Xf, locked, ref, av, full=True)
                sw_f = swf[:2]
                for kind, al, l1 in grid:
                    th, s_f, out_f = fit_std(
                        Xf, yf, locked, kind, al, l1, pf, ref, av, sw_f
                    )
                    res.append(Xv @ th - yv)
                sd = s_f[~out_f & ~locked]
                std_fold = dict(
                    std_left_out_beyond_mask=int((out_f & ~mk).sum()),
                    std_left_out_columns=[
                        (names + ["intercept"])[j] for j in np.flatnonzero(out_f & ~mk)
                    ],
                    # addendum 3: the binomial availability rule's masks (column, kind, partner)
                    std_binom=[
                        (names[c], kind, names[q] if q >= 0 else None)
                        for c, kind, q in swf[5]
                    ],
                    std_s_quantiles_5_50_95=[
                        float(v) for v in np.quantile(sd, [0.05, 0.5, 0.95])
                    ],
                    std_s_min_max=[float(sd.min()), float(sd.max())],
                )
            elif a.est == "ridge":
                G0 = Xf.T @ Xf
                cf = Xf.T @ yf
                for _, al, _ in grid:
                    G = G0.copy()
                    G[np.diag_indices_from(G)] += np.where(locked, 0.0, al * pf)
                    res.append(Xv @ np.linalg.solve(G, cf) - yv)
            else:
                for _, al, l1 in grid:
                    res.append(Xv @ batch_theta_pf(ns, Xf, yf, locked, al, l1, pf) - yv)
            e_all.append(np.array(res))
            info.append(
                dict(
                    fold=f,
                    fit=[f0, f1],
                    val=[v0, v1],
                    masked=int(mk.sum()),
                    cpu=time.process_time() - t0,
                    **std_fold,
                )
            )
            say(
                f"fold {f}: fit {f0}..{f1} val {v0}..{v1} masked {int(mk.sum())} "
                f"mse {np.round((np.array(res) ** 2).mean(axis=1), 5).tolist()}"
            )
        store.update(resid=np.array(e_all), info=info, names=names, bb=bb)
        return np.zeros(len(X_chunk) - train_win_periods)

    t_cpu = time.process_time()
    if a.design:  # the same rows of the design cache
        dz = Path(a.design)
        meta = json.loads((dz / "design.json").read_text())
        check_meta(meta, a.bucket)
        # 2026-10-20 (user): parked -- both checks are in check_meta now.  Was:
        # assert meta["bucket"] == a.bucket, (meta["bucket"], a.bucket)
        # assert meta.get("rerun", False) == RERUN, (meta.get("rerun"), RERUN)
        sl = slice(a.start - a.halo, a.start + 1)
        fit_predict(
            np.load(dz / "X.npy", mmap_mode="r" if RERUN else None)[
                sl
            ],  # a view, as in the pipeline (--rerun: memory-mapped)
            np.load(dz / "y.npy")[sl],
            W,
            dict(meta["hp"], _feature_names=list(meta["names"])),
        )
    else:
        run_spec_executor(
            ns,
            a.est,
            a.bucket,
            fit_predict,
            str(out / "dummy" / "results.csv"),  # (written as seg_csv of it)
            a.start,
            a.start + 1,
            a.halo,
            a.data,
        )
    # global row of local row i: a.start - a.halo + i
    lo = a.start - a.halo
    np.savez_compressed(
        out / "fold.npz",
        resid=store["resid"],
        grid=np.array([g[1] for g in grid]),
        fit=np.array([[x["fit"][0] + lo, x["fit"][1] + lo] for x in store["info"]]),
        val=np.array([[x["val"][0] + lo, x["val"][1] + lo] for x in store["info"]]),
        masked=np.array([x["masked"] for x in store["info"]]),
        names=np.array(store["names"]),
        backbone=store["bb"],
    )
    import numpy as _np

    info = dict(
        mode="pretune",
        rerun=RERUN,
        bucket=a.bucket,
        est=a.est,
        r=a.r,
        start=a.start,
        halo=a.halo,
        grid=[g[1] for g in grid],
        standardize_penalty=bool(STANDARDIZE_PENALTY),  # 2026-10-22
        folds=store["info"],
        repair=dict(REPAIR),
        cpu_sec=time.process_time() - t_cpu,
        wall_sec=time.time() - t_wall,
        host=platform.node(),
        numpy=_np.__version__,
        **({} if default_config() else {"config": config_dict()}),
    )
    (out / "run.json").write_text(json.dumps(info, indent=1))
    say(f"pretune done: {json.dumps(info)}")


def dates(a) -> None:
    """Row dates of the pooled design: the spec's run_executor on baseline with a no-op walk."""
    ns = spec_section("ridge", "baseline", 0, -1, 0)
    out = Path(a.out).resolve()

    def fit_predict(X_chunk, y_chunk, train_win_periods, hyperparams):
        return np.zeros(len(X_chunk) - train_win_periods)

    run_spec_executor(
        ns,
        "ridge",
        "baseline",
        fit_predict,
        str(out / "dates_raw" / "results.csv"),
        0,
        -1,
        0,
        a.data,
    )
    d = pd.read_csv(seg_csv(str(out / "dates_raw" / "results.csv")), usecols=["date"])
    d.insert(0, "row", np.arange(W, W + len(d)))
    d.to_csv(out / "dates.csv", index=False)
    say(f"dates: {len(d)} OOS rows, {d['date'].iloc[0]} .. {d['date'].iloc[-1]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("walk", "pretune", "dates", "design", "merge"))
    ap.add_argument("--bucket", default="baseline")
    ap.add_argument(
        "--est", default="ridge", choices=("ridge", "reclasso", "reclasticnet", "ols")
    )
    ap.add_argument("--r", type=float, default=1.0)
    ap.add_argument("--sched", default="causal")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    ap.add_argument("--halo", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument(
        "--design", default="", help="a design cache directory (mode design's output)"
    )
    ap.add_argument(
        "--rerun",
        action="store_true",
        help="the 2026-10-19 rerun design (module docstring); default: the printed run's",
    )
    ap.add_argument(
        "--part", default="", help="design: K/N, the K-th of N column parts"
    )
    ap.add_argument(
        "--parts",
        default="",
        help="merge: the part design directories, K = 1 .. N in order",
    )
    # 2026-10-19 (the 13-bar rerun); every default is the 6 Oct runner
    ap.add_argument("--sample", default="48", choices=tuple(SEGMENT_OF))
    ap.add_argument("--har-base", default="")
    ap.add_argument("--drop-fomc", action="store_true")
    ap.add_argument("--ex-flag", action="append", default=[])
    # 2026-10-20 (user): which OLS --est ols runs (module docstring)
    ap.add_argument("--ols", default="auto", choices=("auto", "ridge0", "minnorm"))
    # 2026-10-20 (user): the rolling window in trading days (the design cache records window = days x bars a day)
    ap.add_argument("--window-days", type=int, default=TW_DAYS)
    # 2026-10-22 (user): fomc_release back in all_features although EXCLUDE_FOMC_RELEASE is on (the late-start rerun
    # against the published 576-column run); default off
    ap.add_argument("--keep-fomc-release", action="store_true")
    # 2026-10-22 (user, window sensitivity at 1,500 .. 3,000 days): pretune: the first pre-sample fold fitted (the
    # folds before it have a fit window that starts before the design's first row); default 0, every fold
    ap.add_argument("--first-fold", type=int, default=0)
    # 2026-10-22 (James: "add the KKT check flag"): src/models/reclasso_har.py WALK_KKT_RESEED for the walk (0 = off,
    # the walk before the flag bit for bit; 1 = on); empty = the module's default (on).  The walk's numerics only: not
    # part of the design cache's config
    ap.add_argument("--walk-kkt-reseed", default="", choices=("", "0", "1"))
    # 2026-10-22 (James: "yes, adopt standardization and hold the pretune"): STANDARDIZE_PENALTY for the pretune and
    # the walk (0 = off, the runner before the flag bit for bit; 1 = on); empty = the module's default (on).  The fits'
    # numerics only: not part of the design cache's config
    ap.add_argument("--standardize-penalty", default="", choices=("", "0", "1"))
    # 2026-10-22: the penalty grid (comma-separated alphas, the spec's kind and l1_ratio); empty = the spec's grid
    ap.add_argument("--grid-alphas", default="")
    # 2026-10-22: walk diagnostic of the held lasso / elastic-net weights (module docstring); default off
    ap.add_argument("--std-check-blockends", action="store_true")
    a = ap.parse_args()
    global RERUN, PART, STANDARDIZE_PENALTY
    if a.standardize_penalty:
        STANDARDIZE_PENALTY = a.standardize_penalty == "1"
    RERUN = a.rerun
    configure(a)
    if RERUN:  # --rerun fixes the ladder (base 5) and keeps the FOMC series
        assert HAR_BASE in ("", "5") and not DROP_FOMC, (HAR_BASE, DROP_FOMC)
    if a.part:
        PART = tuple(int(v) for v in a.part.split("/"))
    a.data = (
        str(Path(a.data).resolve()) if a.data != "data" else "data"
    )  # "data": the root's, after chdir
    a.out = str(Path(a.out).resolve())
    if a.sched != "causal" and not a.sched.startswith("const:"):
        a.sched = str(Path(a.sched).resolve())
    if a.design:
        a.design = str(Path(a.design).resolve())
    {
        "walk": walk,
        "pretune": pretune,
        "dates": dates,
        "design": design,
        "merge": merge,
    }[a.mode](a)


if __name__ == "__main__":
    main()
