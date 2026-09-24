"""Study 85 -- the earliest card clock for the 15:30 limit: how early can P* be computed?

QUESTION (the operator's).  The trade stays at 15:30 ET: the SPXW 0DTE
nearest-OTM strangle (call at/above spot, put at/below), a limit order at
P* = ``live.ibkr.pricing.package_price(sqrt(rv_hat), S, Kc, Kp)``, cancelled if
not filled within a minute, held to the 16:00 settlement; month-ends buy at any
price (excluded here).  Today rv_hat uses data through the 15:30 stamp and
reaches the operator at ~15:31-15:33: no time to stage the order.  If the card
is posted at wall clock w in {15:00, 15:05, ..., 15:25} instead, does the limit
keep the edge?

INFORMATION SET AT w (live/close_signal/state/latency.parquet, 2026-09-24
15:32:01: the ES=F 1-minute feed's last bar 15:22, ^VIX 15:16, ^GSPC real time):
  es_only    ES 1-minute bars ENDING at or before w - 10 min, Cboe prints
             stamped at or before w - 15 min.  The repository has no ^GSPC
             1-minute history, so the ^GSPC substitute the current card uses
             for the missing ES minutes is NOT modelled: this is the ES-only card.
  idealized  ES and Cboe through w -- an UPPER BOUND, labelled as such.
The 16:00 forecast reads the 15:30-stamp row (the 15:00-15:30 bar), which at w
holds k = min(30, ES cutoff - 15:00) minutes:
  es_only    w  15:00 15:05 15:10 15:15 15:20 15:25 15:30
             k    0     0     0     5    10    15    20
             (w 15:00 / 15:05 also see only 20 / 25 minutes of the 14:30-15:00 bar)
  idealized  k    0     5    10    15    20    25    30   (15:30 = the current card)
k = 0: the bar has not started, so the forecast is study 80's h = 2 forecast
(every HAR regressor .shift(2): the 16:00 row reads through the 15:00 stamp),
with a partial 15:00 bar where that one is incomplete too.
Cboe: from 2024-02-13 the panel's VIX cells are Yahoo HOURLY prints
(live/close_signal/ingest: the 15:00 stamp holds the 15:00 print, 15:30 carries
it).  At that resolution the 15:00 print is known at w >= 15:15; earlier the
latest known print is the 14:00 one (the 14:30 cell), which then replaces the
15:00 cell.  The live card at w sees fresher 1-minute prints than this history
holds, so the Cboe side is conservative.

PARTIAL BAR -> FULL BAR (fixed before any P&L was computed):
  profile (primary)  intensity moments (sumret2, sumabsret, sumret4, sumpret2,
                     sumbipow, sumvolume) of the k minutes x the causal
                     within-bar profile: the sum over PRIOR sessions' full bars /
                     the sum of their first-k-minute bars (expanding over the ES
                     minute span, >= 20 prior sessions, else the uniform factor);
                     signed moments (sumret, sumret3, sumautocov) kept as
                     observed -- unobserved minutes add zero in expectation.
  uniform30k         every moment x 30/k (the lag-1 pair moments sumbipow and
                     sumautocov x 29/(k-1)): the naive extrapolation, a sensitivity.

ENGINE -- the card's own forecast, from a FROZEN copy of the committed code
(``git archive HEAD``; the working tree's live/close_signal is being edited):
the 13 per-bar ridge arms of specs/causal_tune_linear.py (ridge, bucket
free_vix_only, LAG_SCOPE global, TRAIN_WIN 2000) through
``src.backtest.executor.run_executor`` on the EXTENDED panel exactly as
``live.close_signal.forecast.build_ext_root`` builds it (vendor panel +
panel_free, whose Databento ES rows start 2024-03-31), the 13-bar yhat table
(``forecast.assemble_yhat_table``) and the deck loader's causal second-order MZ
map (atm_straddle_lib).  In process, as study 80: one executor call per horizon.

The card at w on day d is that pipeline with ONLY day d's last bar (and Cboe
cell) holding what is knowable at w -- the history rows are complete, as they
are for the live card at w.  Every piece of the pipeline at a row is a function
of STRICTLY EARLIER rows and the row's own regressors (per-slot diurnal
baselines and winsor bounds are shift(1) statistics, the robust scaler uses the
trailing window [t - W, t), the ridge at the 16:00 row of d is fit on earlier
sessions' 16:00 rows, the MZ map on earlier sessions).  So the w forecast is an
EXACT substitution into the captured state of one full run per horizon: the
partial raw value through the row's own captured transform (baseline, semantic
transform, winsor bounds), the HAR means from the window's other rows plus the
substituted row, the row's scaler median / IQR, the ridge vector in force at
the row, then the day's MZ coefficients.  G3 checks it against a brute-force
re-run with the partial row written into the panel.  (A panel variant with
EVERY session's 15:30 row truncated -- the literal alternative -- would also
retrain the ridge on partial rows from 2024-05 on, not before (no minute data),
move the 15:30 bar's own target and the MZ fit's rv_raw, and show yesterday's
complete bar as partial to today's HAR ladders: a different, mixed model.)

GATES (the run stops before any P&L on a failed hard gate):
  G0  the in-process arm == the card's own subprocess (forecast.run_arm) on bar1600.
  G1  study 82's 1-minute -> 30-minute moments == the extended panel's 15:00 and
      15:30 rows on every test session (all nine ES columns, <= 1e-12 rel).
  G1b capture fidelity: the re-derived transforms equal the executor's adj
      columns bit for bit; the scaler replica reproduces the scaled matrix bit
      for bit; the captured ridge vectors reproduce pred_adj; substituting the
      panel's own row reproduces pred_adj.
  G2  w = 15:30 idealized (k = 30 from the minutes) == the unmodified
      pipeline's rv_hat to 1e-9 relative on every test session; and w = 15:00
      idealized == the h = 2 pipeline's own rv_hat.
  G3  brute force on the leak day: the partial row written into the panel, the
      bar1600 arm re-run -> the same 16:00 forecast (h = 1 at es_only 15:20;
      h = 2 at es_only 15:05), 1e-9 relative, and the same regressor row.
  G4  LEAK TEST: on the leak day every 1-minute ES return from 15:10 (panel A)
      or 15:20 (panel B) to 16:00 is multiplied by 50; the arms are re-run; the
      rv_hat_w of that day and of every earlier test session must be
      BIT-IDENTICAL for every w whose ES cutoff is <= the perturbation start,
      and must move for later cutoffs (positive control).  The pipeline has two
      FULL-SAMPLE scalars (the signed-diurnal std floor ``typical`` and the
      impute median, the 2026-09-05 audit's inherited defect; the live card
      computes them on data to today, so they are a backtest artifact) through
      which ANY perturbation moves every session, earlier ones included.  So the
      arms are re-run twice: with those two scalars pinned to the unperturbed
      panel's values (the hard gate) and as is (reported: the size of the
      inherited channel).
  G5  the 15:30 books equal study 79's daily table (pair and ask) on common days.

SCORING at the 15:30 ask, on the sessions 2024-05-01 .. 2026-09-22 (the 15:30
rows before May 2024 are the vendor's, not rebuildable from the minutes to G1),
month-ends and early closes excluded:
  (i)  strikes and spot re-read at 15:30: the 15:30 pair, limit P*(rv_hat_w) on it;
  (ii) fully staged at w: spot S_w, the nearest-OTM pair around S_w on the listed
       grid, limit P*(rv_hat_w; S_w, that pair); at 15:30 that pair is bought iff
       its ask <= the limit.  S_w = S_1530 + (ES_w - ES_1530), ES_t the last ES
       1-minute close before t: the index level the operator reads at w, in the
       ES move to 15:30 (no ^GSPC minutes in the repository); checked against
       the chain's 15:00 spot and the XSP parity spot at 15:20 / 15:25.
  Venues: XSP from data/archive/xsp_opra (study 79's loaders, XSP = SPX / 10);
  SPX from data/spxw_chain.parquet (true UTC -> ET) through 2025-12, and
  data/archive/spxw_opra for 2026 where a non-month-end day exists (none yet).
  Per w: days, decision agreement with the current card (idealized 15:30,
  variant (i)), median |P*_w - P*_1530| in SPX points (XSP x 10), per-day P&L
  (R on a buy day, 0 otherwise), t, buys, mean R on buy days, t, the paired
  circular block bootstrap (20 sessions, 10,000 draws) of per-day P&L w minus
  15:30, and for (ii) how often the w pair is not the 15:30 pair.
  KEEPS THE EDGE: per-day P&L >= the 15:30 one minus 10 % of it AND agreement
  >= 90 %.

OUTPUTS (results/atm_straddle_intraday_holdclose/proposals/85/):
  gate.json, gate_parity_1min.csv, leak_test.csv, bruteforce_check.csv,
  spot_proxy_check.csv, forecasts.parquet, forecast_diagnostics.csv,
  books.parquet, per_day_pnl.parquet/.csv, summary.csv, earliest.csv,
  summary.txt, run.log (stdout)

REPRODUCE (about 30 minutes; engine runs two processes at a time, ~1.5 GB each):
    C:/Users/james/miniconda3/envs/285J/python.exe writeup/intraday_proposals/85_earliest_limit.py
    (--work DIR keeps the frozen code, ext roots and engine captures; --reuse
     skips engine runs whose captures exist there; --workers 2; --no-g0)
"""

from __future__ import annotations

import __future__
import argparse
import ast
import importlib.util
import inspect
import io
import json
import os
import pickle
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numba import njit, prange

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "atm_straddle_intraday_holdclose" / "proposals" / "85"
DATA = REPO / "data"
ES_CSV = DATA / "archive" / "es_v0_ohlcv1m_databento.csv"
XSP_DIR = DATA / "archive" / "xsp_opra"
SPXW_OPRA_DIR = DATA / "archive" / "spxw_opra"
CHAIN = DATA / "spxw_chain.parquet"
CHAIN_SPOT = DATA / "spxw_spot.parquet"
PROPOSALS = REPO / "results" / "atm_straddle_intraday_holdclose" / "proposals"
S79_TABLE = PROPOSALS / "79" / "xsp_close_book_daily.csv"
YF_CACHES = (
    PROPOSALS / "79" / "spx_close_yf.csv",
    PROPOSALS / "82" / "spx_close_yf.csv",
)
FROZEN_PATHS = (
    "live",
    "src",
    "specs",
    "notebooks/atm_straddle_lib.py",
    "writeup/intraday_proposals/79_xsp_close_book.py",
    "writeup/intraday_proposals/82_decision_granularity.py",
)
_FROZEN_ENV = "P85_FROZEN_ROOT"
_env_root = os.environ.get(_FROZEN_ENV)
FROZEN: Path | None = Path(_env_root) if _env_root else None
if FROZEN is not None and str(FROZEN) not in sys.path:
    sys.path.insert(0, str(FROZEN))

ET = "America/New_York"
BARS = tuple(
    f"bar{h:02d}{m:02d}" for h in range(10, 17) for m in (0, 30) if (h, m) <= (16, 0)
)
BUCKET = "free_vix_only"  # forecast.BUCKET, the live card's
TRAIN_WIN = 2000
FIRST_DAY = pd.Timestamp("2024-05-01")
PANEL_END = pd.Timestamp(
    "2026-09-24"
)  # panel_free rows from here on are today's live run
W_CLOCKS = ("15:00", "15:05", "15:10", "15:15", "15:20", "15:25", "15:30")
ISETS = ("es_only", "idealized")
SCHEMES = ("profile", "uniform30k")
PRIMARY = "profile"
ES_LAG_MIN = 10
CBOE_LAG_MIN = 15
ES_COLS = (
    "sumret",
    "sumabsret",
    "sumret2",
    "sumret3",
    "sumret4",
    "sumpret2",
    "sumbipow",
    "sumautocov",
    "sumvolume",
)
INTENSITY = ("sumabsret", "sumret2", "sumret4", "sumpret2", "sumbipow", "sumvolume")
SIGNED = ("sumret", "sumret3", "sumautocov")
PAIR_MOMENTS = ("sumbipow", "sumautocov")
PROFILE_MIN_SESSIONS = 20
LEAK_FACTOR = 50.0
LEAK_STARTS = {"A": "15:10", "B": "15:20"}
BF_CASES = {"bf1": ("es_only", "15:20"), "bf2": ("es_only", "15:05")}
GATE_REL_TOL = 1e-9
PARITY_TOL = 1e-12
EDGE_PNL_FRAC = 0.10
EDGE_AGREE = 0.90
BLOCK = 20
N_BOOT = 10_000
SEED = 85
XSP_SCALE = 0.1
# study 79's table is written at 6 significant digits; the chain's prices are float32
ASK_TOL = 1e-4

_LOG: list[str] = []


def log(msg: str = "") -> None:
    _LOG.append(msg)
    print(msg, flush=True)


def _hm(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def info_spec(iset: str, w: str) -> dict:
    """What the card posted at wall clock w knows (see the module docstring)."""
    wm = _hm(w)
    es_cut = wm - ES_LAG_MIN if iset == "es_only" else wm
    cb_cut = wm - CBOE_LAG_MIN if iset == "es_only" else wm
    k = min(30, es_cut - _hm("15:00"))
    if k >= 1:
        h, last = 1, "15:30"
    else:
        h, last, k = 2, "15:00", min(30, es_cut - _hm("14:30"))
    assert k >= 1 and cb_cut >= _hm("14:00")
    vix_sub = cb_cut < _hm("15:00")
    assert not (vix_sub and h == 1)
    return {
        "iset": iset,
        "w": w,
        "h": h,
        "last": last,
        "k": int(k),
        "es_cut": es_cut,
        "cboe_cut": cb_cut,
        "vix_sub": bool(vix_sub),
    }


# --------------------------------------------------------------------------- frozen code
def freeze(work: Path) -> tuple[Path, str]:
    """The committed code (git archive HEAD) under work/frozen, reused if the SHA matches."""
    dst = work / "frozen"
    sha = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    tag = dst / "FROZEN_SHA"
    if tag.exists() and tag.read_text().strip() == sha:
        return dst, sha
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    tar = subprocess.run(
        ["git", "-C", str(REPO), "archive", "--format=tar", "HEAD", *FROZEN_PATHS],
        check=True,
        capture_output=True,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(tar)) as tf:
        tf.extractall(dst, filter="data")
    tag.write_text(sha)
    return dst, sha


def _import_frozen(rel: str, name: str):
    assert FROZEN is not None
    spec = importlib.util.spec_from_file_location(name, FROZEN / rel)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _asl():
    assert FROZEN is not None
    if str(FROZEN / "notebooks") not in sys.path:
        sys.path.insert(0, str(FROZEN / "notebooks"))
    import atm_straddle_lib as asl  # type: ignore

    assert Path(asl.__file__).resolve().is_relative_to(FROZEN.resolve())
    return asl


# --------------------------------------------------------------------------- engine
def _spec_estimator() -> dict:
    """The spec's estimator lifted by AST (study 80's lift): imports, defs, tuning constants."""
    assert FROZEN is not None
    spec = FROZEN / "specs" / "causal_tune_linear.py"
    tree = ast.parse(spec.read_text(encoding="utf-8"))
    keep = {"HORIZON", "REFIT_FREQUENCY", "TUNE_PER", "VAL_TAIL", "EMBARGO", "SEED"}
    keep.add("ESTIMATOR_GRIDS")
    body: list[ast.stmt] = []
    for n in tree.body:
        if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)):
            body.append(n)
        elif isinstance(n, (ast.Assign, ast.AnnAssign)):
            tg = n.targets if isinstance(n, ast.Assign) else [n.target]
            if all(isinstance(t, ast.Name) and t.id in keep for t in tg):
                body.append(n)
    ns: dict = {"__name__": "causal_tune_linear_lifted", "__file__": str(spec)}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(spec), "exec"), ns)
    assert ns["HORIZON"] == 1 and ns["REFIT_FREQUENCY"] == 1
    assert (ns["TUNE_PER"], ns["VAL_TAIL"], ns["EMBARGO"]) == (250, 125, 25)
    return ns


def _har_shifted(h: int):
    """generate_har_features with its one ``.shift(1)`` replaced by ``.shift(h)`` (study 80)."""
    from src.features.extractors import har

    src = inspect.getsource(har.generate_har_features)
    token = ".mean().shift(1)"
    assert src.count(token) == 1, "generate_har_features changed"
    src = src.replace(token, ".mean().shift(_H_SHIFT)")
    ns = dict(vars(har))
    ns["_H_SHIFT"] = int(h)
    exec(compile(src, har.__file__, "exec"), ns)
    return ns["generate_har_features"]


#: Module attributes an engine run replaces; restored to the pristine objects at
#: the start of every run, so a pool worker that runs several jobs never chains
#: one job's wrappers into the next.
_PATCHED = {
    "src.backtest.executor": (
        "load_and_transform",
        "_build_har_and_calendar",
        "_backtest_and_save",
        "rolling_robust_scale",
        "generate_har_features",
        "SEGMENT_DEFINITIONS",
    ),
    "src.features.transforms.target": ("diurnal_adjust",),
}


def _restore_pristine() -> None:
    import importlib

    for modname, names in _PATCHED.items():
        mod = importlib.import_module(modname)
        store = mod.__dict__.setdefault("_P85_PRISTINE", {})
        for n in names:
            if n not in store:
                store[n] = getattr(mod, n)
            setattr(mod, n, store[n])


def _patch_full_sample_scalars(pin: dict | None) -> dict:
    """Route the pipeline's two full-sample scalars through a recorder / pin.

    ``diurnal_adjust``'s signed-branch std floor (``typical``, a median over
    every row) and ``load_and_transform``'s impute fill (``adj.median()`` over
    every observed row) are the inherited full-sample statistics (the
    2026-09-05 audit).  Their sources are re-compiled with each value passed
    through a hook that records it and, when ``pin`` holds a value for the
    column, returns the pinned one -- with no pin the functions are the
    originals, value for value.  Returns the record.
    """
    import src.backtest.executor as E
    from src.features.transforms import target as T

    rec: dict = {"typical": {}, "impute_median": {}}

    def typ(name, v):
        rec["typical"][name] = v
        return pin["typical"][name] if pin and name in pin["typical"] else v

    def med(name, v):
        rec["impute_median"][name] = v
        return pin["impute_median"][name] if pin and name in pin["impute_median"] else v

    for mod, fn, tok, new, hook, hname in (
        (
            T,
            "diurnal_adjust",
            "typical = baseline.replace(0, np.nan).abs().median()",
            "typical = _P85_TYPICAL(series.name, baseline.replace(0, np.nan).abs().median())",
            typ,
            "_P85_TYPICAL",
        ),
        (
            E,
            "load_and_transform",
            'df[f"adj_{col}"] = adj.fillna(adj.median())',
            'df[f"adj_{col}"] = adj.fillna(_P85_MED(col, adj.median()))',
            med,
            "_P85_MED",
        ),
    ):
        src = inspect.getsource(mod._P85_PRISTINE[fn])
        assert src.count(tok) == 1, f"{fn} changed: re-derive the scalar hook"
        setattr(mod, hname, hook)
        code = compile(
            src.replace(tok, new),
            str(mod.__file__),
            "exec",
            flags=__future__.annotations.compiler_flag,
            dont_inherit=True,
        )
        exec(code, vars(mod))
    return rec


def record_pins(job: dict) -> str:
    """The full-sample scalars of the unperturbed panel (load_and_transform as the arms call it)."""
    assert FROZEN is not None
    os.chdir(FROZEN)
    import src.backtest.executor as E
    from src.data.loading import get_bucket

    _restore_pristine()
    rec = _patch_full_sample_scalars(None)
    E.load_and_transform(
        str(job["data_dir"]),
        get_bucket(BUCKET),
        target_use_diurnal=True,
        target_winsor_window=240,
        dropna_with_exog=False,
        overnight_fill=True,
        impute_indicate=True,
        diurnal_mode="divide",
    )
    with open(job["capture"], "wb") as fh:
        pickle.dump(rec, fh)
    print(f"[pins] {json.dumps(rec, default=float)}", flush=True)
    return str(job["capture"])


@njit(cache=False, parallel=True)
def _scale_params(
    X, W, ref_iqr, use_ref, floor_frac, med_out, eff_out
):  # pragma: no cover
    """src.features.transforms.scaling._rolling_scale_cols, returning each row's median and
    effective IQR instead of the scaled value (same arithmetic, same order)."""
    n, p = X.shape
    idx_25 = (W - 1) * 0.25
    idx_50 = (W - 1) * 0.50
    idx_75 = (W - 1) * 0.75
    i25, r25 = int(idx_25), idx_25 - int(idx_25)
    i50, r50 = int(idx_50), idx_50 - int(idx_50)
    i75, r75 = int(idx_75), idx_75 - int(idx_75)
    for j in prange(p):
        s = np.sort(X[:W, j].copy())
        b = X[:W, j].copy()
        pos = 0
        q25 = s[i25] * (1.0 - r25) + s[min(i25 + 1, W - 1)] * r25
        med = s[i50] * (1.0 - r50) + s[min(i50 + 1, W - 1)] * r50
        q75 = s[i75] * (1.0 - r75) + s[min(i75 + 1, W - 1)] * r75
        iq = q75 - q25
        iq_max = iq if iq > 0.0 else 0.0
        iqr = iq
        if floor_frac > 0.0 and iq_max > 0.0:
            fl = floor_frac * iq_max
            if iqr < fl:
                iqr = fl
        if iqr < 1e-12:
            iqr = 1.0
        for t in range(W):
            eff = iqr
            if use_ref and ref_iqr[t, j] > eff:
                eff = ref_iqr[t, j]
            med_out[t, j] = med
            eff_out[t, j] = eff
        for t in range(W, n):
            q25 = s[i25] * (1.0 - r25) + s[min(i25 + 1, W - 1)] * r25
            med = s[i50] * (1.0 - r50) + s[min(i50 + 1, W - 1)] * r50
            q75 = s[i75] * (1.0 - r75) + s[min(i75 + 1, W - 1)] * r75
            iq = q75 - q25
            if iq > iq_max:
                iq_max = iq
            iqr = iq
            if floor_frac > 0.0 and iq_max > 0.0:
                fl = floor_frac * iq_max
                if iqr < fl:
                    iqr = fl
            if iqr < 1e-12:
                iqr = 1.0
            eff = iqr
            if use_ref and ref_iqr[t, j] > eff:
                eff = ref_iqr[t, j]
            med_out[t, j] = med
            eff_out[t, j] = eff
            v_old = b[pos]
            v_new = X[t, j]
            b[pos] = v_new
            pos = pos + 1
            if pos == W:
                pos = 0
            io_ = np.searchsorted(s, v_old)
            inw = np.searchsorted(s, v_new)
            if io_ < inw:
                inw -= 1
                for kk in range(io_, inw):
                    s[kk] = s[kk + 1]
            elif io_ > inw:
                for kk in range(io_, inw, -1):
                    s[kk] = s[kk - 1]
            s[inw] = v_new


def _replica(df: pd.DataFrame, col: str, is_target: bool) -> dict:
    """robust_transform's chain for one column, keeping the per-row pieces it discards."""
    from src.features.transforms import target as T

    series = df[col].copy()
    has_neg = bool((series.dropna() < 0).any())
    slot_band = 0.0 if is_target else T.DIURNAL_SLOT_BAND_FRAC
    base = pd.Series(1.0, index=df.index)
    stab, s_asinh = False, None
    excluded = bool(T.is_diurnal_excluded(col))
    if not excluded:
        adjusted, base = T.diurnal_adjust(
            series, df["time_of_day"], has_neg, slot_band=slot_band, soft=None
        )
        if (
            has_neg
            and not T.has_name_stabilizer(col)
            and float(np.isclose(base, base.min(), rtol=1e-9).mean())
            >= T.DIURNAL_PINNED_MAX
        ):
            series, s_asinh = T.asinh_stabilize_pair(series)
            base = pd.Series(1.0, index=df.index)
            stab = True
        else:
            series = adjusted
    if stab:
        pre = series.fillna(0.0)
    else:
        pre = T.apply_semantic_transform(series, col, has_neg, allow_missing=False)
    lo = pre.rolling(240, min_periods=240).quantile(T.WINSOR_LOWER_Q).shift(1)
    hi = pre.rolling(240, min_periods=240).quantile(T.WINSOR_UPPER_Q).shift(1)
    out = pre.clip(lower=lo, upper=hi)
    return {
        "out": out.to_numpy(float),
        "base": base.to_numpy(float),
        "lo": lo.to_numpy(float),
        "hi": hi.to_numpy(float),
        "s": None if s_asinh is None else s_asinh.to_numpy(float),
        "has_neg": has_neg,
        "excluded": excluded,
        "stab": stab,
    }


def engine_run(job: dict) -> str:
    """One executor call (horizon h, the listed bars) with the bar1600 state captured.

    Writes the arms' CSVs under job['arms_dir'] and pickles the substitution
    ingredients of job['days'] to job['capture']; returns the capture path.
    """
    t0 = time.time()
    assert FROZEN is not None
    os.chdir(FROZEN)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    import src.backtest.executor as E
    from src.backtest.segmentation import BAR_SEGMENTS
    from src.data.loading import get_bucket
    from src.features.extractors.har import resolve_har_lags
    from src.features.transforms import scaling

    h = int(job["h"])
    bars = tuple(job["bars"])
    pin = None
    if job.get("pin_file"):
        with open(job["pin_file"], "rb") as pfh:
            pin = pickle.load(pfh)
    _restore_pristine()
    scalars = _patch_full_sample_scalars(pin)
    ns = _spec_estimator()
    RTL = ns["RollingTunedLinear"]
    RTL.grid = ns["ESTIMATOR_GRIDS"]["ridge"]
    E.generate_har_features = _har_shifted(h)
    E.SEGMENT_DEFINITIONS = {b: BAR_SEGMENTS[b] for b in bars}
    cap: dict = {"seg": None, "thetas": []}
    orig = {
        n: getattr(E, n)
        for n in (
            "load_and_transform",
            "_build_har_and_calendar",
            "_backtest_and_save",
            "rolling_robust_scale",
        )
    }

    def lt(*a, **k):
        df, cols = orig["load_and_transform"](*a, **k)
        cap["df_lt"], cap["adj_exog"] = df, list(cols)
        return df, cols

    def bh(df, exog_cols, add_calendar, har_lags=None):
        out = orig["_build_har_and_calendar"](df, exog_cols, add_calendar, har_lags)
        cap["df_har"] = out[0]
        return out

    def bs(job_df, feature_names, fit_predict, hyperparams, train_win, *rest):
        seg = Path(rest[4]).stem.split("_")[-1]
        cap["seg"] = seg
        if seg == "bar1600":
            cap["seg_t"] = pd.DatetimeIndex(job_df["t"])
            cap["features"] = list(feature_names)
            cap["train_win"] = int(train_win)
            cap["out_file"] = rest[4]
        try:
            return orig["_backtest_and_save"](
                job_df, feature_names, fit_predict, hyperparams, train_win, *rest
            )
        finally:
            cap["seg"] = None

    def rs(X, train_win, ref_iqr=None, fixed_cols=None, **kw):
        out = orig["rolling_robust_scale"](
            X, train_win, ref_iqr=ref_iqr, fixed_cols=fixed_cols, **kw
        )
        if cap["seg"] == "bar1600":
            cap["X_raw"] = np.array(X, dtype=np.float64, copy=True)
            cap["ref_iqr"] = None if ref_iqr is None else np.array(ref_iqr, copy=True)
            cap["fixed"] = (
                np.zeros(X.shape[1], bool)
                if fixed_cols is None
                else np.array(fixed_cols, bool)
            )
            cap["X_scaled"] = np.array(out, copy=True)
            cap["floor_frac"] = float(kw.get("iqr_floor_frac", scaling.IQR_FLOOR_FRAC))
            cap["W_scale"] = int(train_win)
        return out

    orig_predict = RTL.predict_one

    def predict_one(self, x):
        if cap["seg"] == "bar1600":
            cap["thetas"].append(np.array(self._th, copy=True))
        return orig_predict(self, x)

    RTL.predict_one = predict_one
    E.load_and_transform = lt
    E._build_har_and_calendar = bh
    E._backtest_and_save = bs
    E.rolling_robust_scale = rs  # type: ignore[assignment]
    out_csv = Path(job["arms_dir"]) / "causal_tune_linear" / "results.csv"
    E.run_executor(
        method_name="lin_tuned_ridge",
        fit_predict=ns["fit_predict_lin_tuned"],
        hyperparams={"_refit_frequency": ns["REFIT_FREQUENCY"]},
        data_path=str(job["data_dir"]),
        output_file=str(out_csv),
        horizon=ns["HORIZON"],
        train_window=TRAIN_WIN,
        start=0,
        end=-1,
        halo=0,
        exog_cols=get_bucket(BUCKET),
        segment="all",
        lag_scope="global",
        har_lags=None,
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
    t_run = time.time() - t0

    # ---- the substitution ingredients of the requested days
    df = cap["df_lt"]
    dfh = cap["df_har"]
    tg = pd.DatetimeIndex(df["t"])
    gpos = pd.Series(np.arange(len(tg)), index=tg)
    lags = resolve_har_lags()
    max_lag = lags[-1]
    seg_t = cap["seg_t"][max_lag:]
    X_raw, X_sc = cap["X_raw"], cap["X_scaled"]
    assert len(seg_t) == len(X_raw)
    W = cap["train_win"]
    th = np.array(cap["thetas"])
    assert len(th) == len(X_raw) - W, (len(th), len(X_raw), W)
    feats = cap["features"]
    fidx = {n: i for i, n in enumerate(feats)}
    ref = cap["ref_iqr"]
    med = np.empty_like(X_raw)
    eff = np.empty_like(X_raw)
    _scale_params(
        X_raw,
        cap["W_scale"],
        ref if ref is not None else np.zeros((1, X_raw.shape[1])),
        ref is not None,
        cap["floor_frac"],
        med,
        eff,
    )
    fixed = cap["fixed"]
    rep = (X_raw - med) / eff
    rep[:, fixed] = X_raw[:, fixed]
    scale_exact = bool(np.array_equal(rep, X_sc))
    scale_maxdiff = float(np.nanmax(np.abs(rep - X_sc)))
    res = pd.read_csv(cap["out_file"], parse_dates=["date"]).set_index("date")

    bucket_raw = get_bucket(BUCKET)
    raw_cols = ["RV"] + [c for c in bucket_raw if c in ES_COLS] + ["vix"]
    har_src = ["adj_RV"] + list(cap["adj_exog"])
    prm_full = {}
    replica_dev = {}
    for col in raw_cols:
        rr = _replica(df, col, is_target=(col == "RV"))
        adj = df["adj_RV" if col == "RV" else f"adj_{col}"].to_numpy(float)
        obs = (
            np.ones(len(df), bool)
            if col == "RV"
            else df[f"{col}__obs"].to_numpy().astype(bool)
        )
        replica_dev[col] = bool(
            np.array_equal(rr["out"][obs], adj[obs], equal_nan=True)
        )
        prm_full[col] = rr
    modes = {}
    for c in har_src:
        if c.endswith("_active"):
            modes[c] = float(
                E.mode_inflated(df[c[: -len("_active")]].to_numpy(float))[1]
            )

    days = [pd.Timestamp(d) for d in job["days"]]
    last_off = pd.Timedelta(job["last"] + ":00")
    rows = []
    for d in days:
        t16 = d + pd.Timedelta(hours=16)
        tl = d + last_off
        if t16 not in gpos.index or tl not in gpos.index:
            continue
        g16 = int(gpos[t16])
        gl = g16 - h
        if tg[gl] != tl:
            continue
        r = int(np.searchsorted(seg_t.values, t16.to_datetime64()))
        if r >= len(seg_t) or seg_t[r] != t16 or r < W:
            continue
        rows.append((d, g16, gl, r))
    n = len(rows)
    nl, ns_ = len(lags), len(har_src)
    S = np.zeros((n, ns_, nl))
    last_adj = np.zeros((n, ns_))
    cols_vals = {c: df[c].to_numpy(float) for c in har_src}
    for i, (_, g16, gl, _) in enumerate(rows):
        for j, c in enumerate(har_src):
            v = cols_vals[c]
            last_adj[i, j] = v[gl]
            for q, L in enumerate(lags):
                S[i, j, q] = v[gl - L + 1 : gl].sum() if L > 1 else 0.0
    gl_idx = np.array([x[2] for x in rows], int)
    g16_idx = np.array([x[1] for x in rows], int)
    r_idx = np.array([x[3] for x in rows], int)
    prm = {}
    for col, rr in prm_full.items():
        prm[col] = {
            "base": rr["base"][gl_idx],
            "lo": rr["lo"][gl_idx],
            "hi": rr["hi"][gl_idx],
            "s": None if rr["s"] is None else rr["s"][gl_idx],
            "has_neg": rr["has_neg"],
            "excluded": rr["excluded"],
            "stab": rr["stab"],
            "raw_last": df[col].to_numpy(float)[gl_idx],
            "raw_prev": df[col].to_numpy(float)[gl_idx - 1],
            "t_prev": tg[gl_idx - 1],
        }
    thetas = th[r_idx - W]
    pred_rep = np.array(
        [np.append(X_sc[r], 1.0) @ thetas[i] for i, r in enumerate(r_idx)]
    )
    d_index = pd.DatetimeIndex([x[0] for x in rows])
    pred_exec = (
        res["pred_adj"].reindex(d_index + pd.Timedelta(hours=16)).to_numpy(float)
    )
    out = {
        "tag": job["tag"],
        "h": h,
        "last": job["last"],
        "days": d_index,
        "features": feats,
        "fidx": fidx,
        "lags": list(lags),
        "har_src": har_src,
        "raw_cols": raw_cols,
        "S": S,
        "last_adj": last_adj,
        "prm": prm,
        "modes": modes,
        "gate_open": dfh["is_open"].to_numpy(float)[g16_idx],
        "gate_close": dfh["is_close"].to_numpy(float)[g16_idx],
        "x_raw": X_raw[r_idx],
        "x_scaled": X_sc[r_idx],
        "med": med[r_idx],
        "eff": eff[r_idx],
        "fixed": fixed,
        "theta": thetas,
        "pred_exec": pred_exec,
        "pred_replay": pred_rep,
        "scale_replica_exact": scale_exact,
        "scale_replica_maxdiff": scale_maxdiff,
        "transform_replica_exact": replica_dev,
        "n_features": len(feats),
        "wall_run_s": round(t_run, 1),
        "wall_total_s": round(time.time() - t0, 1),
        "arms_dir": str(job["arms_dir"]),
        "bars": bars,
        "full_sample_scalars": scalars,
        "pinned": pin is not None,
    }
    with open(job["capture"], "wb") as fh:
        pickle.dump(out, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(
        f"[{job['tag']}] h={h} bars={len(bars)} run {t_run:.0f}s, {n} days captured, "
        f"scale replica exact {scale_exact}, transforms exact {all(replica_dev.values())}",
        flush=True,
    )
    return str(job["capture"])


# --------------------------------------------------------------------------- substitution
def substitute(ing: dict, raw_b: dict[str, np.ndarray], sel: np.ndarray) -> np.ndarray:
    """yhat (pred_adj) of the 16:00 row with the last bar's raw values replaced.

    ``raw_b`` maps a raw column (RV, the ES moments, vix) to the replacement
    values on the ingredient days ``sel`` (an index array); columns not given
    keep the panel's own value.  The arithmetic is the pipeline's: the row's
    own transform, the HAR means, the row's scaler, ``predict_one``'s dot.
    """
    X = substituted_rows(ing, raw_b, sel)
    XS = (X - ing["med"][sel]) / ing["eff"][sel]
    fx = ing["fixed"]
    XS[:, fx] = X[:, fx]
    th = ing["theta"][sel]
    return np.array([np.append(XS[i], 1.0) @ th[i] for i in range(len(XS))])


def substituted_rows(
    ing: dict, raw_b: dict[str, np.ndarray], sel: np.ndarray
) -> np.ndarray:
    """The unscaled 16:00 regressor rows of the days ``sel`` after the substitution."""
    from src.features.transforms import target as T

    X = ing["x_raw"][sel].copy()
    last = ing["last_adj"][sel].copy()
    src = ing["har_src"]
    sidx = {c: j for j, c in enumerate(src)}
    for col, v in raw_b.items():
        p = ing["prm"][col]
        v = np.asarray(v, float)
        if p["stab"]:
            z = np.nan_to_num(np.arcsinh(v / p["s"][sel]), nan=0.0)
        else:
            z = v if p["excluded"] else v / p["base"][sel]
            z = T.apply_semantic_transform(
                pd.Series(z), col, p["has_neg"], allow_missing=False
            ).to_numpy(float)
        lo, hi = p["lo"][sel], p["hi"][sel]
        z = np.where(np.isfinite(lo) & (z < lo), lo, z)
        z = np.where(np.isfinite(hi) & (z > hi), hi, z)
        last[:, sidx["adj_RV" if col == "RV" else f"adj_{col}"]] = z
        if f"{col}_avail" in sidx:
            last[:, sidx[f"{col}_avail"]] = np.where(np.isfinite(v), 1.0, 0.0)
        if f"{col}_active" in sidx:
            mode = ing["modes"][f"{col}_active"]
            last[:, sidx[f"{col}_active"]] = (
                np.nan_to_num(v, nan=mode) != mode
            ).astype(float)
    fidx = ing["fidx"]
    S = ing["S"][sel]
    for j, c in enumerate(src):
        for q, L in enumerate(ing["lags"]):
            name = f"har_ma_{L}" if c == "adj_RV" else f"{c}_ma_{L}"
            val = (S[:, j, q] + last[:, j]) / L
            X[:, fidx[name]] = val
            if c == "adj_RV":
                for g in ("open", "close"):
                    nm = f"{name}_x_{g}"
                    if nm in fidx:
                        X[:, fidx[nm]] = val * ing[f"gate_{g}"][sel]
    return X


# --------------------------------------------------------------------------- minutes
def load_minutes(p82) -> pd.DataFrame:
    df = p82.es_minutes(ES_CSV)
    df["day"] = df["t"].dt.normalize()
    df["mod"] = (df["t"].dt.hour * 60 + df["t"].dt.minute).astype(int)
    return df


def bar_moments(mins: pd.DataFrame, end_hhmm: str, k: int, p82) -> pd.DataFrame:
    """The ES moments of the bar ending ``end_hhmm`` from its first k minutes, by day."""
    e = _hm(end_hhmm)
    sub = mins[(mins["mod"] >= e - 30) & (mins["mod"] < e - 30 + k)]
    m = p82.moments_30(sub.drop(columns=["day", "mod"]))
    idx = pd.DatetimeIndex(m.index)
    m = m[(idx.hour * 60 + idx.minute) == e]
    m.index = pd.DatetimeIndex(m.index).normalize()
    return m


def extrapolate(
    part: pd.DataFrame,
    full: pd.DataFrame,
    k: int,
    scheme: str,
    prior_days: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Full-bar estimate from the first k minutes (module docstring, PARTIAL BAR)."""
    if k == 30:
        return part[list(ES_COLS)].copy()
    out = part[list(ES_COLS)].copy()
    uni = {c: (29.0 / (k - 1) if c in PAIR_MOMENTS else 30.0 / k) for c in ES_COLS}
    if scheme == "uniform30k":
        for c in ES_COLS:
            out[c] = out[c] * uni[c]
        return out
    assert scheme == "profile"
    days = prior_days.intersection(part.index).intersection(full.index).sort_values()
    for c in INTENSITY:
        num = full[c].reindex(days).astype(float)
        den = part[c].reindex(days).astype(float)
        f = (num.cumsum().shift(1) / den.cumsum().shift(1)).to_numpy()
        nprior = np.arange(len(days))
        f = np.where(nprior >= PROFILE_MIN_SESSIONS, f, uni[c])
        fac = pd.Series(f, index=days).reindex(out.index)
        # a day outside the profile's day set (none among the test days) takes the uniform factor
        out[c] = out[c] * fac.fillna(uni[c]).to_numpy()
    return out


# --------------------------------------------------------------------------- books
def xsp_day(job: dict) -> dict:
    """The XSP book at 15:30 for one day: the 15:30 pair, parity spots, the w pairs' 15:30 asks."""
    p79 = _import_frozen(
        "writeup/intraday_proposals/79_xsp_close_book.py", "p79_frozen"
    )
    day = pd.Timestamp(job["day"])
    q = p79.load_quotes(Path(job["cbbo"]))
    defs = pd.read_parquet(job["defs"])
    strikes = np.sort(defs["strike_price"].astype(float).unique())
    b = p79.book_at(q, day, "15:30:00")
    par = {
        hh: p79.parity_spot(p79.book_at(q, day, f"{hh}:00"))
        for hh in ("15:20", "15:25", "15:30")
    }
    s1530 = job["s1530_chain"]
    src = "chain"
    if not np.isfinite(s1530):
        s1530, src = par["15:30"] / XSP_SCALE, "parity"
    xs = s1530 * XSP_SCALE
    xc = job["s_close"] * XSP_SCALE
    out = {
        "day": day,
        "spot_src": src,
        "S_1530": s1530,
        "par_1520": par["15:20"] / XSP_SCALE,
        "par_1525": par["15:25"] / XSP_SCALE,
        "par_1530": par["15:30"] / XSP_SCALE,
        "S_close": job["s_close"],
    }

    def pair(sx: float) -> dict:
        kc, kp = p79.nearest_otm(strikes, sx)
        _, ac, _, _ = p79.quote(b, kc, "C")
        _, ap, _, _ = p79.quote(b, kp, "P")
        return {
            "kc": kc,
            "kp": kp,
            "ask": ac + ap,
            "payoff": max(xc - kc, 0.0) + max(kp - xc, 0.0),
        }

    out["pair_1530"] = pair(xs)
    out["pairs_w"] = {
        w: pair((s1530 + dlt) * XSP_SCALE) for w, dlt in job["es_delta"].items()
    }
    return out


def spx_books(days: pd.DatetimeIndex, es_delta: pd.DataFrame, spots: pd.Series) -> dict:
    """The SPXW 15:30 book from the chain (true UTC stamps): study 79's spxw_book + the w pairs."""
    p79 = _import_frozen(
        "writeup/intraday_proposals/79_xsp_close_book.py", "p79_frozen_m"
    )
    c = pd.read_parquet(
        CHAIN,
        columns=[
            "expiration",
            "strike",
            "cp",
            "timestamp",
            "bid",
            "ask",
            "underlying_price",
        ],
        filters=[("expiration", ">=", days.min()), ("expiration", "<=", days.max())],
    )
    t = pd.to_datetime(c["timestamp"]).dt.tz_convert(ET).dt.tz_localize(None)
    c = c[
        (t.dt.hour == 15) & (t.dt.minute == 30) & (t.dt.normalize() == c["expiration"])
    ]
    c = c[c["ask"] > 0]
    out = {}
    for day, g in c.groupby("expiration"):
        day = pd.Timestamp(day)
        if day not in days or (day, "16:00") not in spots.index:
            continue
        s = float(g["underlying_price"].median())
        g = g.set_index(["strike", "cp"]).sort_index()
        ks = np.sort(g.index.get_level_values(0).unique().to_numpy(dtype=float))
        sc = float(spots.loc[(day, "16:00")])

        def pair(sx: float) -> dict:
            kc, kp = p79.nearest_otm(ks, sx)
            try:
                ask = float(g.at[(kc, "C"), "ask"] + g.at[(kp, "P"), "ask"])
            except KeyError:
                ask = float("nan")
            return {
                "kc": kc,
                "kp": kp,
                "ask": ask,
                "payoff": max(sc - kc, 0.0) + max(kp - sc, 0.0),
            }

        dl = es_delta.loc[day] if day in es_delta.index else None
        out[day] = {
            "day": day,
            "S_1530": s,
            "S_close": sc,
            "S_1500_chain": float(spots.get((day, "15:00"), np.nan)),
            "pair_1530": pair(s),
            "pairs_w": {w: pair(s + float(dl[w])) for w in W_CLOCKS}
            if dl is not None
            else {},
            "strikes": ks,
        }
    return out


def spot_table() -> pd.Series:
    """SPX by (day, HH:MM) from the chain's spot file (true UTC -> ET; study 79's spot_table)."""
    s = pd.read_parquet(CHAIN_SPOT)
    t = pd.to_datetime(s["timestamp"])
    if getattr(t.dt, "tz", None) is not None:
        t = t.dt.tz_convert(ET).dt.tz_localize(None)
    clk = (
        t.dt.hour.astype(str).str.zfill(2) + ":" + t.dt.minute.astype(str).str.zfill(2)
    )
    s = s.assign(day=t.dt.normalize(), clk=clk)
    return s.set_index(["day", "clk"])["spot"].astype(float)


def yf_closes(days: list[pd.Timestamp]) -> pd.Series:
    """^GSPC daily closes: studies 79 / 82's caches, then yfinance into this study's folder."""
    parts = [
        pd.read_csv(p, index_col=0, parse_dates=True)["close"]
        for p in YF_CACHES
        if p.exists()
    ]
    own = OUT / "spx_close_yf.csv"
    if own.exists():
        parts.append(pd.read_csv(own, index_col=0, parse_dates=True)["close"])
    have = pd.concat(parts) if parts else pd.Series(dtype=float)
    have = have[~have.index.duplicated(keep="last")].sort_index()
    need = [d for d in days if d not in have.index]
    if need:
        import yfinance as yf

        hst = yf.Ticker("^GSPC").history(
            start=min(need).strftime("%Y-%m-%d"),
            end=(max(need) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=False,
        )
        got = pd.Series(
            hst["Close"].to_numpy(dtype=float),
            index=pd.DatetimeIndex(hst.index.tz_localize(None).normalize()),
        )
        got = got[got.index.isin(need)]
        if len(got):
            got.rename("close").to_frame().to_csv(own)
        have = pd.concat([have, got]).sort_index()
        have = have[~have.index.duplicated(keep="last")]
    return have


# --------------------------------------------------------------------------- stats
def block_boot_idx(n: int, block: int, n_boot: int, rng) -> np.ndarray:
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_boot, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    return idx.reshape(n_boot, nb * block)[:, :n]


def tstat(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 3 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def qlike(y: np.ndarray, f: np.ndarray) -> np.ndarray:
    r = y / f
    return r - np.log(r) - 1.0


# --------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    global FROZEN
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--no-g0", action="store_true")
    a = ap.parse_args(argv)
    T0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    work = Path(a.work) if a.work else Path(tempfile.mkdtemp(prefix="p85_"))
    work.mkdir(parents=True, exist_ok=True)
    frozen, sha = freeze(work)
    FROZEN = frozen
    os.environ[_FROZEN_ENV] = str(frozen)
    if str(frozen) not in sys.path:
        sys.path.insert(0, str(frozen))
    log(f"frozen code: git archive HEAD {sha} -> {frozen}")
    gate: dict = {"frozen_sha": sha}

    from live.close_signal import forecast as F
    from live.close_signal.state import StateStore
    from live.ibkr.calendar_guard import is_last_session_of_month, is_session
    from live.ibkr.pricing import package_price

    assert Path(F.__file__).resolve().is_relative_to(frozen.resolve())
    asl = _asl()
    p82 = _import_frozen(
        "writeup/intraday_proposals/82_decision_granularity.py", "p82_frozen"
    )
    assert F.BUCKET == BUCKET and F.TRAIN_WIN == TRAIN_WIN

    # ---- 1. the extended panel, as the card builds it (panel_free as committed at HEAD)
    st_dir = work / "state"
    st_dir.mkdir(exist_ok=True)
    pf = pd.read_parquet(
        frozen / "live" / "close_signal" / "state" / "panel_free.parquet"
    )
    pf["endbartime"] = pd.to_datetime(pf["endbartime"])
    n_pf = len(pf)
    pf = pf[(~pf["placeholder"].astype(bool)) & (pf["endbartime"] < PANEL_END)]
    pf.to_parquet(st_dir / "panel_free.parquet", index=False)
    fomc_csv = frozen / "live" / "close_signal" / "state" / "fomc_statement_dates.csv"
    ext = work / "ext_root"
    if not (a.reuse and (ext / "data" / "core_stats.parquet").exists()):
        F.build_ext_root(REPO, StateStore(st_dir), ext, fomc_csv)
        for d in ("src", "specs"):  # the frozen code, not the working tree's
            shutil.rmtree(ext / d)
            shutil.copytree(frozen / d, ext / d)
    log(
        f"panel_free (HEAD) {n_pf} rows -> {len(pf)} (placeholders and stamps from "
        f"{PANEL_END.date()} dropped); ext root {json.loads((ext / 'data' / 'extension.json').read_text())}"
    )
    core = pd.read_parquet(ext / "data" / "core_stats.parquet")
    core["endbartime"] = pd.to_datetime(core["endbartime"])
    cboe = pd.read_parquet(ext / "data" / "vix_and_voldemand.parquet")
    cboe["endbartime"] = pd.to_datetime(cboe["endbartime"])
    core_i = core.set_index("endbartime")

    # ---- 2. ES minutes, the partial bars, G1
    mins = load_minutes(p82)
    early = set(pd.to_datetime(list(asl.EARLY_CLOSE_DATES)))
    full30 = {e: bar_moments(mins, e, 30, p82) for e in ("15:00", "15:30")}
    cand = full30["15:30"].index.intersection(full30["15:00"].index)
    prior_days = pd.DatetimeIndex(
        [d for d in cand if is_session(d.date()) and d not in early]
    ).sort_values()
    last_es_day = pd.Timestamp(mins["day"].max())
    have = set(core_i.index[core_i["sumret2"].notna()])
    days = pd.DatetimeIndex(
        [
            d
            for d in prior_days
            if FIRST_DAY <= d <= last_es_day
            and all(
                d + pd.Timedelta(x) in have
                for x in ("15:00:00", "15:30:00", "16:00:00")
            )
        ]
    )
    log(
        f"ES minutes {mins['t'].min()} .. {mins['t'].max()} ({len(mins):,} rows); "
        f"test sessions {len(days)} ({days.min().date()} .. {days.max().date()})"
    )
    g1 = []
    for e in ("15:00", "15:30"):
        m = full30[e].reindex(days)
        p = core_i.reindex(days + pd.Timedelta(e + ":00"))
        for c in (*ES_COLS, "numobs"):
            x, y = m[c].to_numpy(float), p[c].to_numpy(float)
            den = np.maximum(np.abs(x), np.abs(y))
            rel = np.where(den > 0, np.abs(x - y) / np.where(den > 0, den, 1.0), 0.0)
            g1.append(
                {
                    "stamp": e,
                    "column": c,
                    "rows": int(np.isfinite(x).sum()),
                    "nan_mismatch": int((np.isfinite(x) != np.isfinite(y)).sum()),
                    "max_rel": float(np.nanmax(rel)),
                }
            )
    g1 = pd.DataFrame(g1)
    g1.to_csv(OUT / "gate_parity_1min.csv", index=False)
    gate["G1_parity_max_rel"] = float(g1["max_rel"].max())
    gate["G1_nan_mismatch"] = int(g1["nan_mismatch"].sum())
    gate["G1_passed"] = bool(
        gate["G1_parity_max_rel"] <= PARITY_TOL and gate["G1_nan_mismatch"] == 0
    )
    log(
        f"G1 1-min -> 30-min vs the panel's 15:00/15:30 rows: {g1['max_rel'].max():.2e} max rel"
    )
    if not gate["G1_passed"]:
        log(g1.to_string())
        (OUT / "gate.json").write_text(json.dumps(gate, indent=1, default=str))
        return 2

    # ---- 3. the leak day, perturbed panels, brute-force panels
    me_flag = pd.Series([is_last_session_of_month(d.date()) for d in days], index=days)
    nme = days[~me_flag.to_numpy()]
    leak_day = nme[len(nme) // 2]
    parts: dict = {}  # (stamp, k, scheme, minutes-tag) -> DataFrame

    def partial(
        mtag: str, mm: pd.DataFrame, stamp: str, k: int, scheme: str
    ) -> pd.DataFrame:
        key = (mtag, stamp, k, scheme)
        if key not in parts:
            full = bar_moments(mm, stamp, 30, p82) if mtag != "base" else full30[stamp]
            pk = bar_moments(mm, stamp, k, p82) if k < 30 else full
            parts[key] = extrapolate(pk, full, k, scheme, prior_days)
        return parts[key]

    def write_root(
        name: str, core_new: pd.DataFrame, cboe_new: pd.DataFrame | None = None
    ) -> Path:
        root = work / name
        (root / "data").mkdir(parents=True, exist_ok=True)
        for f in F.VENDOR_FILES:
            if f == "core_stats.parquet":
                core_new.to_parquet(root / "data" / f, index=False)
            elif f == "vix_and_voldemand.parquet" and cboe_new is not None:
                cboe_new.to_parquet(root / "data" / f, index=False)
            else:
                shutil.copy2(ext / "data" / f, root / "data" / f)
        return root

    pert_mins = {}
    for nm, start in LEAK_STARTS.items():
        mp = mins.copy()
        hit = (
            (mp["day"] == leak_day)
            & (mp["mod"] >= _hm(start))
            & (mp["mod"] < _hm("16:00"))
        )
        mp.loc[hit, "r"] = mp.loc[hit, "r"] * LEAK_FACTOR
        pert_mins[nm] = mp
    roots = {"R": ext}
    if not a.reuse or not (work / "leakA" / "data").exists():
        for nm, mp in pert_mins.items():
            cn = core.copy().set_index("endbartime")
            for e in ("15:30", "16:00"):
                mo = bar_moments(mp, e, 30, p82).loc[leak_day]
                t = leak_day + pd.Timedelta(e + ":00")
                for c in (*ES_COLS, "numobs"):
                    cn.at[t, c] = float(mo[c])
            write_root(f"leak{nm}", cn.reset_index())
        # brute force: the day's last bar as the card at w holds it
        for nm, (iset, w) in BF_CASES.items():
            sp = info_spec(iset, w)
            ex = partial("base", mins, sp["last"], sp["k"], PRIMARY).loc[leak_day]
            cn = core.copy().set_index("endbartime")
            t = leak_day + pd.Timedelta(sp["last"] + ":00")
            for c in ES_COLS:
                cn.at[t, c] = float(ex[c])
            cb = None
            if sp["vix_sub"]:
                cb = cboe.copy().set_index("endbartime")
                cb.at[t, "vix"] = float(
                    cb.at[leak_day + pd.Timedelta("14:30:00"), "vix"]
                )
                cb = cb.reset_index()
            write_root(nm, cn.reset_index(), cb)
    for nm in LEAK_STARTS:
        roots[f"leak{nm}"] = work / f"leak{nm}"
    for nm in BF_CASES:
        roots[nm] = work / nm

    # ---- 4. the engine runs (process pool; ~1.5 GB each)
    iso = [str(d.date()) for d in days]
    iso_leak = [str(d.date()) for d in days if d <= leak_day]
    (work / "captures").mkdir(exist_ok=True)
    pin_file = work / "captures" / "pins.pkl"
    jobs: list[dict[str, Any]] = []
    for h in (1, 2):
        last = "15:30" if h == 1 else "15:00"
        jobs.append(dict(tag=f"R_h{h}", h=h, last=last, root="R", bars=BARS, days=iso))
        for nm in LEAK_STARTS:
            if nm == "B" and h == 2:
                continue  # panel B perturbs after 15:20; h = 2 reads through 15:00 (panel A covers it)
            for pinned in (False, True):
                jobs.append(
                    dict(
                        tag=f"leak{nm}{'_pin' if pinned else ''}_h{h}",
                        h=h,
                        last=last,
                        root=f"leak{nm}",
                        bars=BARS,
                        days=iso_leak,
                        pin_file=str(pin_file) if pinned else None,
                    )
                )
    for nm, (iset, w) in BF_CASES.items():
        sp = info_spec(iset, w)
        jobs.append(
            dict(
                tag=nm,
                h=sp["h"],
                last=sp["last"],
                root=nm,
                bars=("bar1600",),
                days=[str(leak_day.date())],
            )
        )
    for j in jobs:
        j["data_dir"] = str(roots[j["root"]] / "data")
        j["arms_dir"] = str(work / "arms" / j["tag"])
        j["capture"] = str(work / "captures" / f"{j['tag']}.pkl")
    todo = [j for j in jobs if not (a.reuse and Path(j["capture"]).exists())]
    t1 = time.time()
    if not (a.reuse and pin_file.exists()):
        # the unperturbed panel's full-sample scalars, in a process of their own
        with ProcessPoolExecutor(max_workers=1) as ex_:
            ex_.submit(
                record_pins, {"data_dir": str(ext / "data"), "capture": str(pin_file)}
            ).result()
    with open(pin_file, "rb") as fh:
        pins = pickle.load(fh)
    gate["full_sample_scalars_unperturbed"] = pins
    if todo:
        log(
            f"engine: {len(todo)} runs, {a.workers} at a time: {[j['tag'] for j in todo]}"
        )
        with ProcessPoolExecutor(max_workers=max(1, min(a.workers, len(todo)))) as ex_:
            for fu in [ex_.submit(engine_run, j) for j in todo]:
                fu.result()
    gate["engine_wall_s"] = round(time.time() - t1, 1)
    ing = {}
    for j in jobs:
        with open(j["capture"], "rb") as fh:
            ing[j["tag"]] = pickle.load(fh)
    log(
        f"engine wall {gate['engine_wall_s']}s; per run: "
        + ", ".join(f"{k} {v['wall_total_s']}s" for k, v in ing.items())
    )

    # ---- 4b. G0: the card's own subprocess arm on bar1600 vs the in-process arm
    if not a.no_g0:
        g0_csv = (
            work
            / "g0"
            / "causal_tune_linear"
            / "ridge"
            / BUCKET
            / "results_bar1600.csv"
        )
        if not (a.reuse and g0_csv.exists()):
            t2 = time.time()
            g0_csv = F.run_arm(
                ext,
                work / "g0",
                sys.executable,
                full=True,
                segment="bar1600",
                bucket=BUCKET,
            )
            log(f"G0 subprocess arm {time.time() - t2:.0f}s")
        sub = pd.read_csv(g0_csv, parse_dates=["date"]).set_index("date")
        inp = pd.read_csv(
            Path(ing["R_h1"]["arms_dir"])
            / "causal_tune_linear"
            / "results_bar1600.csv",
            parse_dates=["date"],
        ).set_index("date")
        jj = sub.join(inp, how="outer", rsuffix="_in")
        gate["G0_rows"] = int(len(jj))
        gate["G0_rows_unmatched"] = int(
            jj["pred_adj"].isna().sum() + jj["pred_adj_in"].isna().sum()
        )
        gate["G0_max_rel_pred_adj"] = float(
            (np.abs(jj["pred_adj"] - jj["pred_adj_in"]) / np.abs(jj["pred_adj"])).max()
        )
        gate["G0_passed"] = bool(
            gate["G0_rows_unmatched"] == 0
            and gate["G0_max_rel_pred_adj"] <= GATE_REL_TOL
        )
        log(
            f"G0 in-process vs subprocess bar1600: {gate['G0_max_rel_pred_adj']:.2e} max rel on {gate['G0_rows']} rows"
        )

    # ---- 5. tables, MZ coefficients
    def table(tag: str) -> pd.DataFrame:
        base = Path(ing[tag]["arms_dir"]) / "causal_tune_linear"
        csvs = {b: base / f"results_{b}.csv" for b in BARS}
        root = roots["R"] if tag.startswith("R_") else roots[tag.split("_")[0]]
        return F.assemble_yhat_table(csvs, root)

    def mz(tag: str, want: pd.DatetimeIndex) -> pd.DataFrame:
        tab = table(tag)
        p = work / f"yhat_{tag}.parquet"
        tab.to_parquet(p, index=False)
        df, rth = asl._panel_frame(p)
        codes, uniq = pd.factorize(df["date"], sort=True)
        need = asl._need_days(want, uniq)
        a_, b_, s2_ = asl._mz_day_coefs(
            df["yhat"].to_numpy(float),
            df["rv_raw"].to_numpy(float),
            df["baseline"].to_numpy(float),
            codes,
            len(uniq),
            need,
            None,
            rth,
            method="mean",
        )
        is_row = (
            (df["et"].dt.hour == 16) & (df["et"].dt.minute == 0) & ~df["early_close"]
        ).to_numpy()
        r = df[is_row].copy()
        cr = codes[is_row]
        r["a"], r["b"], r["s2"] = a_[cr], b_[cr], s2_[cr]
        r = r.drop_duplicates("date").set_index("date")
        r = r.reindex(want)
        r["rv_hat_mz"] = ((r["a"] + r["b"] * r["yhat"]) ** 2 + r["s2"]) * r["baseline"]
        if tag.startswith("R_"):
            ld = F.recalibrate(tab, want, frozen, work, tag=f"p85_{tag}")
            r["rv_hat_loader"] = ld["rv_hat"].reindex(want).to_numpy(float)
        return r

    mzs = {"R_h1": mz("R_h1", days), "R_h2": mz("R_h2", days)}
    lk = days[days <= leak_day]
    for tg in [t for t in ing if t.startswith("leak")]:
        mzs[tg] = mz(tg, lk)
    for h in (1, 2):
        r = mzs[f"R_h{h}"]
        gate[f"mz_replica_vs_loader_h{h}"] = float(
            np.nanmax(np.abs(r["rv_hat_mz"] - r["rv_hat_loader"]) / r["rv_hat_loader"])
        )

    # ---- 6. G1b capture fidelity
    for tg, x in ing.items():
        gate[f"G1b_{tg}"] = {
            "days": int(len(x["days"])),
            "features": x["n_features"],
            "scale_replica_exact": x["scale_replica_exact"],
            "transform_replica_exact": all(x["transform_replica_exact"].values()),
            "theta_replay_max_rel": float(
                np.nanmax(
                    np.abs(x["pred_replay"] - x["pred_exec"]) / np.abs(x["pred_exec"])
                )
            ),
        }
        own = {c: x["prm"][c]["raw_last"] for c in x["raw_cols"]}
        ys = substitute(x, own, np.arange(len(x["days"])))
        gate[f"G1b_{tg}"]["self_substitution_max_rel"] = float(
            np.nanmax(np.abs(ys - x["pred_exec"]) / np.abs(x["pred_exec"]))
        )
    g1b_ok = all(
        v["scale_replica_exact"]
        and v["transform_replica_exact"]
        and v["theta_replay_max_rel"] <= 1e-12
        and v["self_substitution_max_rel"] <= GATE_REL_TOL
        for k, v in gate.items()
        if k.startswith("G1b_")
    )
    gate["G1b_passed"] = bool(g1b_ok)
    log(
        "G1b capture: "
        + json.dumps(
            {k: v for k, v in gate.items() if k.startswith("G1b_")}, default=str
        )
    )

    # ---- 7. rv_hat_w for every (info set, w, scheme)
    def forecast_w(
        run: str, mm: pd.DataFrame, mtag: str, iset: str, w: str, scheme: str
    ) -> pd.DataFrame:
        sp = info_spec(iset, w)
        x = ing[f"{run}_h{sp['h']}"]
        dd = x["days"]
        sel = np.arange(len(dd))
        # the last bar's ES moments always come from the minutes (at k = 30 they
        # equal the panel's row: G1), so every w runs the same code path
        ex = partial(mtag, mm, sp["last"], sp["k"], scheme).reindex(dd)
        raw_b: dict[str, np.ndarray] = {"RV": ex["sumret2"].to_numpy(float)}
        for c in x["raw_cols"]:
            if c in ES_COLS:
                raw_b[c] = ex[c].to_numpy(float)
        if sp["vix_sub"]:
            p = x["prm"]["vix"]
            assert all(
                t == d + pd.Timedelta("14:30:00") for t, d in zip(p["t_prev"], dd)
            ), "the row before 15:00 is not 14:30"
            raw_b["vix"] = p["raw_prev"]
        yh = substitute(x, raw_b, sel)
        mzr = mzs[f"{run}_h{sp['h']}"].reindex(dd)
        rv = (
            (mzr["a"].to_numpy() + mzr["b"].to_numpy() * yh) ** 2 + mzr["s2"].to_numpy()
        ) * mzr["baseline"].to_numpy()
        return pd.DataFrame(
            {
                "iset": iset,
                "w": w,
                "scheme": scheme,
                "h": sp["h"],
                "k": sp["k"],
                "last": sp["last"],
                "vix_sub": sp["vix_sub"],
                "yhat": yh,
                "rv_hat": rv,
            },
            index=dd,
        )

    fc = []
    for iset in ISETS:
        for w in W_CLOCKS:
            for scheme in SCHEMES:
                fc.append(forecast_w("R", mins, "base", iset, w, scheme))
    fc = pd.concat(fc)
    fc.index.name = "date"
    base_rv = mzs["R_h1"]["rv_hat_loader"].reindex(days)
    rv_real = core_i["sumret2"].reindex(days + pd.Timedelta("16:00:00")).to_numpy(float)
    fc["rv_hat_base"] = base_rv.reindex(fc.index).to_numpy()
    fc["rv_16"] = pd.Series(rv_real, index=days).reindex(fc.index).to_numpy()
    fc.reset_index().to_parquet(OUT / "forecasts.parquet", index=False)

    # ---- G2
    g2 = fc[(fc.iset == "idealized") & (fc.w == "15:30")]
    rel2 = np.abs(g2["rv_hat"] - g2["rv_hat_base"]) / g2["rv_hat_base"]
    g2b = fc[(fc.iset == "idealized") & (fc.w == "15:00")]
    h2own = mzs["R_h2"]["rv_hat_loader"].reindex(g2b.index)
    rel2b = np.abs(g2b["rv_hat"].to_numpy() - h2own.to_numpy()) / h2own.to_numpy()
    gate["G2_days"] = int(g2["rv_hat"].notna().sum() // len(SCHEMES))
    gate["G2_max_rel_rv_hat_1530"] = float(np.nanmax(rel2))
    gate["G2_max_rel_rv_hat_1500_h2"] = float(np.nanmax(rel2b))
    gate["G2_passed"] = bool(
        gate["G2_max_rel_rv_hat_1530"] <= GATE_REL_TOL
        and gate["G2_max_rel_rv_hat_1500_h2"] <= GATE_REL_TOL
        and gate["G2_days"] == len(days)
    )
    log(
        f"G2 idealized 15:30 (k=30 from minutes) vs the unmodified pipeline: {gate['G2_max_rel_rv_hat_1530']:.2e} "
        f"max rel on {gate['G2_days']} days; idealized 15:00 vs h=2 pipeline {gate['G2_max_rel_rv_hat_1500_h2']:.2e}"
    )

    # ---- G3 brute force
    g3 = []
    for nm, (iset, w) in BF_CASES.items():
        sp = info_spec(iset, w)
        x = ing[f"R_h{sp['h']}"]
        i = int(np.flatnonzero(x["days"] == leak_day)[0])
        f_sub = fc[(fc.iset == iset) & (fc.w == w) & (fc.scheme == PRIMARY)].loc[
            leak_day, "yhat"
        ]
        bf = ing[nm]
        ex = partial("base", mins, sp["last"], sp["k"], PRIMARY).loc[leak_day]
        raw_b = {"RV": np.array([float(ex["sumret2"])])}
        raw_b.update(
            {c: np.array([float(ex[c])]) for c in x["raw_cols"] if c in ES_COLS}
        )
        if sp["vix_sub"]:
            raw_b["vix"] = np.array([float(x["prm"]["vix"]["raw_prev"][i])])
        xr = substituted_rows(x, raw_b, np.array([i]))[0]
        xb = bf["x_raw"][0]
        den = np.maximum(np.abs(xb), 1e-300)
        g3.append(
            {
                "case": nm,
                "iset": iset,
                "w": w,
                "h": sp["h"],
                "k": sp["k"],
                "day": str(leak_day.date()),
                "yhat_substitution": float(f_sub),
                "yhat_bruteforce": float(bf["pred_exec"][0]),
                "rel_dev_yhat": float(
                    abs(f_sub - bf["pred_exec"][0]) / abs(bf["pred_exec"][0])
                ),
                "max_rel_dev_regressor_row": float(np.max(np.abs(xr - xb) / den)),
                "yhat_full_info": float(x["pred_exec"][i]),
            }
        )
    g3 = pd.DataFrame(g3)
    g3["pass"] = (g3["rel_dev_yhat"] <= GATE_REL_TOL) & (
        g3["max_rel_dev_regressor_row"] <= GATE_REL_TOL
    )
    g3.to_csv(OUT / "bruteforce_check.csv", index=False)
    gate["G3_passed"] = bool(g3["pass"].all())
    log("G3 brute force:\n" + g3.to_string(index=False))

    # ---- G4 leak test
    rows = []
    for nm, start in LEAK_STARTS.items():
        for iset in ISETS:
            for w in W_CLOCKS:
                sp = info_spec(iset, w)
                if nm == "B" and sp["h"] == 2:
                    continue
                for scheme, scalars in (
                    (s, p) for s in SCHEMES for p in ("pinned", "as_is")
                ):
                    ref_ = fc[(fc.iset == iset) & (fc.w == w) & (fc.scheme == scheme)]
                    ref_ = ref_[ref_.index <= leak_day]
                    run = f"leak{nm}_pin" if scalars == "pinned" else f"leak{nm}"
                    per = forecast_w(run, pert_mins[nm], f"leak{nm}", iset, w, scheme)
                    per = per.reindex(ref_.index)
                    expect_same = sp["es_cut"] <= _hm(start)
                    d_day = float(
                        abs(per.at[leak_day, "rv_hat"] - ref_.at[leak_day, "rv_hat"])
                        / ref_.at[leak_day, "rv_hat"]
                    )
                    earlier = ref_.index < leak_day
                    same_earlier = bool(
                        np.array_equal(
                            per.loc[earlier, "rv_hat"].to_numpy(),
                            ref_.loc[earlier, "rv_hat"].to_numpy(),
                        )
                    )
                    same_day = bool(
                        per.at[leak_day, "rv_hat"] == ref_.at[leak_day, "rv_hat"]
                    )
                    rows.append(
                        {
                            "full_sample_scalars": scalars,
                            "panel": nm,
                            "perturbed_from": start,
                            "iset": iset,
                            "w": w,
                            "scheme": scheme,
                            "h": sp["h"],
                            "es_cutoff": f"{sp['es_cut'] // 60:02d}:{sp['es_cut'] % 60:02d}",
                            "expect_invariant": expect_same,
                            "rel_dev_leak_day": d_day,
                            "bit_identical_leak_day": same_day,
                            "earlier_sessions": int(earlier.sum()),
                            "earlier_bit_identical": same_earlier,
                            "max_rel_dev_earlier": float(
                                np.nanmax(
                                    np.abs(
                                        per.loc[earlier, "rv_hat"]
                                        - ref_.loc[earlier, "rv_hat"]
                                    )
                                    / ref_.loc[earlier, "rv_hat"]
                                )
                            ),
                        }
                    )
    lt = pd.DataFrame(rows)
    lt["pass"] = (
        np.where(
            lt["expect_invariant"],
            lt["bit_identical_leak_day"],
            lt["rel_dev_leak_day"] > 1e-6,
        )
        & lt["earlier_bit_identical"]
    )
    lt.to_csv(OUT / "leak_test.csv", index=False)
    gate["G4_leak_day"] = str(leak_day.date())
    for scalars, g in lt.groupby("full_sample_scalars"):
        inv = g[g.expect_invariant]
        gate[f"G4_{scalars}"] = {
            "cases": int(len(g)),
            "invariant_bit_identical_leak_day": f"{int(inv.bit_identical_leak_day.sum())}/{len(inv)}",
            "invariant_max_rel_dev_leak_day": float(inv.rel_dev_leak_day.max()),
            "earlier_sessions_bit_identical": f"{int(g.earlier_bit_identical.sum())}/{len(g)}",
            "earlier_max_rel_dev": float(g.max_rel_dev_earlier.max()),
            "positive_controls_moved": f"{int((g[~g.expect_invariant].rel_dev_leak_day > 1e-6).sum())}/"
            f"{int((~g.expect_invariant).sum())}",
            "positive_control_min_rel_dev": float(
                g[~g.expect_invariant].rel_dev_leak_day.min()
            ),
            "all_pass": bool(g["pass"].all()),
        }
        log(
            f"G4 leak test on {leak_day.date()}, full-sample scalars {scalars}: {gate[f'G4_{scalars}']}"
        )
    # the hard gate: with the pipeline's two full-sample scalars pinned, nothing after
    # the cutoff reaches rv_hat_w; as_is shows the size of the inherited scalar channel
    gate["G4_passed"] = bool(gate["G4_pinned"]["all_pass"])
    hard = ["G1_passed", "G1b_passed", "G2_passed", "G3_passed", "G4_passed"]
    if not a.no_g0:
        hard.append("G0_passed")
    gate["passed"] = bool(all(gate[k] for k in hard))
    (OUT / "gate.json").write_text(json.dumps(gate, indent=1, default=str))
    if not gate["passed"]:
        log(
            "GATE FAILED -- no P&L number is reported: "
            + ", ".join(k for k in hard if not gate[k])
        )
        (OUT / "run.log").write_text("\n".join(_LOG), encoding="utf-8")
        return 3

    # ---- 8. forecast diagnostics (16:00-bar realized variance)
    diag = []
    for (iset, w, scheme), g in fc.groupby(["iset", "w", "scheme"], sort=False):
        ok = np.isfinite(g["rv_hat"]) & np.isfinite(g["rv_16"]) & (g["rv_16"] > 0)
        gg = g[ok]
        diag.append(
            {
                "iset": iset,
                "w": w,
                "scheme": scheme,
                "h": int(g["h"].iloc[0]),
                "k": int(g["k"].iloc[0]),
                "days": int(ok.sum()),
                "qlike": float(
                    qlike(gg["rv_16"].to_numpy(), gg["rv_hat"].to_numpy()).mean()
                ),
                "qlike_1530": float(
                    qlike(gg["rv_16"].to_numpy(), gg["rv_hat_base"].to_numpy()).mean()
                ),
                "median_ratio_to_1530": float(
                    np.median(gg["rv_hat"] / gg["rv_hat_base"])
                ),
                "corr_log_with_1530": float(
                    np.corrcoef(np.log(gg["rv_hat"]), np.log(gg["rv_hat_base"]))[0, 1]
                ),
            }
        )
    diag = pd.DataFrame(diag)
    diag.to_csv(OUT / "forecast_diagnostics.csv", index=False)
    log(
        "\nFORECAST DIAGNOSTICS (QLIKE vs the 15:30-16:00 ES realized variance, all test sessions):"
    )
    log(diag.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---- 9. books
    nme_days = days[~me_flag.to_numpy()]
    es_close = mins[(mins["mod"] >= _hm("14:50")) & (mins["mod"] < _hm("15:30"))]
    es_close = es_close.pivot_table(
        index="day", columns="mod", values="close", aggfunc="last"
    )

    def es_at(hhmm: str) -> pd.Series:
        m = _hm(hhmm)
        cols = [c for c in es_close.columns if m - 10 <= c < m]
        return es_close[cols].ffill(axis=1).iloc[:, -1]

    es_lvl = pd.DataFrame({w: es_at(w) for w in W_CLOCKS})
    es_delta = es_lvl.sub(es_lvl["15:30"], axis=0).reindex(nme_days)
    spots = spot_table()
    xsp_jobs: list[dict[str, Any]] = []
    closes_needed = []
    for d in nme_days:
        cb = XSP_DIR / f"cbbo1m_{d.date()}.parquet"
        df_ = XSP_DIR / f"defs_{d.date()}.parquet"
        if (
            not (cb.exists() and df_.exists())
            or d not in es_delta.index
            or es_delta.loc[d].isna().any()
        ):
            continue
        if (d, "15:30") in spots.index and (d, "16:00") in spots.index:
            s1530, sc = float(spots.loc[(d, "15:30")]), float(spots.loc[(d, "16:00")])
        else:
            s1530, sc = float("nan"), float("nan")
            closes_needed.append(d)
        xsp_jobs.append(
            {
                "day": str(d.date()),
                "cbbo": str(cb),
                "defs": str(df_),
                "s1530_chain": s1530,
                "s_close": sc,
                "es_delta": {w: float(es_delta.at[d, w]) for w in W_CLOCKS},
            }
        )
    yfc = yf_closes(closes_needed) if closes_needed else pd.Series(dtype=float)
    for j in xsp_jobs:
        if not np.isfinite(j["s_close"]):
            d = pd.Timestamp(j["day"])
            j["s_close"] = float(yfc.get(d, np.nan))
    xsp_jobs = [j for j in xsp_jobs if np.isfinite(j["s_close"])]
    t3 = time.time()
    with ProcessPoolExecutor(max_workers=max(1, a.workers * 2)) as ex_:
        xsp = list(ex_.map(xsp_day, xsp_jobs, chunksize=8))
    log(f"XSP books: {len(xsp)} days in {time.time() - t3:.0f}s")
    spx = spx_books(nme_days[nme_days <= pd.Timestamp("2025-12-31")], es_delta, spots)
    opra_2026 = [
        d
        for d in nme_days
        if d.year == 2026 and (SPXW_OPRA_DIR / f"cbbo1m_{d.date()}.parquet").exists()
    ]
    log(
        f"SPX books (chain): {len(spx)} days; data/archive/spxw_opra non-month-end 2026 days: {len(opra_2026)}"
    )
    gate["spxw_opra_2026_non_month_end_days"] = len(opra_2026)

    # G5: the 15:30 books vs study 79's table
    s79 = pd.read_csv(S79_TABLE, index_col=0, parse_dates=True)
    xs_df = pd.DataFrame(
        {
            "kc": [b["pair_1530"]["kc"] for b in xsp],
            "kp": [b["pair_1530"]["kp"] for b in xsp],
            "ask": [b["pair_1530"]["ask"] for b in xsp],
        },
        index=pd.DatetimeIndex([b["day"] for b in xsp]),
    )
    cx = xs_df.join(s79[["kc", "kp", "ask"]], rsuffix="_79", how="inner")
    sp_df = pd.DataFrame(
        {
            "kc": [b["pair_1530"]["kc"] for b in spx.values()],
            "kp": [b["pair_1530"]["kp"] for b in spx.values()],
            "ask": [b["pair_1530"]["ask"] for b in spx.values()],
        },
        index=pd.DatetimeIndex(list(spx.keys())),
    )
    cs = sp_df.join(s79[["spx_kc", "spx_kp", "spx_ask"]], how="inner")
    gate["G5_xsp_days"] = int(len(cx))
    gate["G5_xsp_pair_ask_equal"] = int(
        (
            (cx.kc == cx.kc_79)
            & (cx.kp == cx.kp_79)
            & (
                (np.abs(cx.ask - cx.ask_79) <= ASK_TOL)
                | (cx.ask.isna() & cx.ask_79.isna())
            )
        ).sum()
    )
    gate["G5_spx_days"] = int(len(cs))
    gate["G5_spx_pair_ask_equal"] = int(
        (
            (cs.kc == cs.spx_kc)
            & (cs.kp == cs.spx_kp)
            & (
                (np.abs(cs.ask - cs.spx_ask) <= ASK_TOL)
                | (cs.ask.isna() & cs.spx_ask.isna())
            )
        ).sum()
    )
    log(
        f"G5 books vs study 79: XSP {gate['G5_xsp_pair_ask_equal']}/{gate['G5_xsp_days']}, "
        f"SPX {gate['G5_spx_pair_ask_equal']}/{gate['G5_spx_days']} days with the same pair and ask"
    )

    # spot proxy check
    spc = []
    for b in xsp:
        d = b["day"]
        for hh in ("15:20", "15:25"):
            tru = b[f"par_{hh.replace(':', '')}"]
            prox = b["S_1530"] + es_delta.at[d, hh]
            if np.isfinite(tru):
                spc.append(
                    {
                        "venue": "XSP parity",
                        "w": hh,
                        "day": d,
                        "err_pts": prox - tru,
                        "grid": 10.0,
                    }
                )
    for d, b in spx.items():
        if np.isfinite(b["S_1500_chain"]):
            prox = b["S_1530"] + es_delta.at[d, "15:00"]
            spc.append(
                {
                    "venue": "SPX chain",
                    "w": "15:00",
                    "day": d,
                    "err_pts": prox - b["S_1500_chain"],
                    "grid": 5.0,
                }
            )
    spc = pd.DataFrame(spc)
    sps = (
        spc.assign(abs_err=spc["err_pts"].abs())
        .groupby(["venue", "w"])
        .agg(
            days=("abs_err", "size"),
            median_abs_err_pts=("abs_err", "median"),
            p95_abs_err_pts=("abs_err", lambda v: float(np.quantile(v, 0.95))),
            mean_err_pts=("err_pts", "mean"),
        )
        .reset_index()
    )
    sps.to_csv(OUT / "spot_proxy_check.csv", index=False)
    log(
        "S_w proxy (S_1530 + ES move) vs the true index level, SPX points:\n"
        + sps.to_string(index=False)
    )

    # ---- 10. per-day P&L
    fcp = fc[fc.scheme.isin(SCHEMES)]
    pdl = []
    books = []
    for venue, blist in (("SPX", list(spx.values())), ("XSP", xsp)):
        scale = 1.0 if venue == "SPX" else XSP_SCALE
        for b in blist:
            d = b["day"]
            if d not in base_rv.index or not np.isfinite(base_rv.loc[d]):
                continue
            p0 = b["pair_1530"]
            s0 = b["S_1530"] * scale
            pstar0 = package_price(np.sqrt(base_rv.loc[d]), s0, p0["kc"], p0["kp"])
            buy0 = bool(np.isfinite(p0["ask"]) and p0["ask"] <= pstar0)
            books.append(
                {
                    "venue": venue,
                    "date": d,
                    "S_1530": b["S_1530"],
                    "S_close": b["S_close"],
                    "kc_1530": p0["kc"],
                    "kp_1530": p0["kp"],
                    "ask_1530": p0["ask"],
                    "payoff_1530": p0["payoff"],
                    **{f"kc_{w}": b["pairs_w"][w]["kc"] for w in W_CLOCKS},
                    **{f"kp_{w}": b["pairs_w"][w]["kp"] for w in W_CLOCKS},
                    **{f"ask_{w}_pair": b["pairs_w"][w]["ask"] for w in W_CLOCKS},
                }
            )
            g = fcp.loc[[d]] if d in fcp.index else None
            if g is None:
                continue
            for r in g.itertuples():
                sw = (b["S_1530"] + es_delta.at[d, r.w]) * scale
                for variant in ("i", "ii"):
                    if variant == "i":
                        pp, s_used = p0, s0
                    else:
                        pp, s_used = b["pairs_w"][r.w], sw
                    ps = package_price(np.sqrt(r.rv_hat), s_used, pp["kc"], pp["kp"])
                    ok = np.isfinite(pp["ask"]) and pp["ask"] > 0
                    buy = bool(ok and pp["ask"] <= ps)
                    R = pp["payoff"] / pp["ask"] - 1.0 if ok else np.nan
                    pdl.append(
                        {
                            "venue": venue,
                            "date": d,
                            "iset": r.iset,
                            "w": r.w,
                            "scheme": r.scheme,
                            "variant": variant,
                            "h": r.h,
                            "k": r.k,
                            "rv_hat": r.rv_hat,
                            "S": s_used,
                            "kc": pp["kc"],
                            "kp": pp["kp"],
                            "P_star": ps,
                            "ask": pp["ask"],
                            "buy": buy,
                            "R": R if buy else np.nan,
                            "pnl": (R if buy else 0.0) if ok else np.nan,
                            "pair_changed": bool(
                                (pp["kc"], pp["kp"]) != (p0["kc"], p0["kp"])
                            ),
                            "P_star_1530": pstar0,
                            "buy_1530": buy0,
                            "pnl_1530": (
                                (p0["payoff"] / p0["ask"] - 1.0) if buy0 else 0.0
                            )
                            if np.isfinite(p0["ask"]) and p0["ask"] > 0
                            else np.nan,
                            "idx_pts_scale": 1.0 / scale,
                        }
                    )
    pdl = pd.DataFrame(pdl)
    pdl.to_parquet(OUT / "per_day_pnl.parquet", index=False)
    pdl.to_csv(OUT / "per_day_pnl.csv", index=False)
    pd.DataFrame(books).to_parquet(OUT / "books.parquet", index=False)

    # ---- 11. the table + bootstrap
    summ = []
    for (venue, scheme, variant, iset, w), g in pdl.groupby(
        ["venue", "scheme", "variant", "iset", "w"], sort=False
    ):
        g = g[np.isfinite(g["pnl"]) & np.isfinite(g["pnl_1530"])].sort_values("date")
        n = len(g)
        dl = (g["pnl"] - g["pnl_1530"]).to_numpy(float)
        rng = np.random.default_rng(SEED)
        idx = block_boot_idx(n, BLOCK, N_BOOT, rng)
        bm = dl[idx].mean(1)
        base_m = float(g["pnl_1530"].mean())
        pnl_m = float(g["pnl"].mean())
        agree = float((g["buy"] == g["buy_1530"]).mean())
        rb = g.loc[g["buy"], "R"].to_numpy(float)
        summ.append(
            {
                "venue": venue,
                "scheme": scheme,
                "variant": variant,
                "iset": iset,
                "w": w,
                "h": int(g["h"].iloc[0]),
                "k": int(g["k"].iloc[0]),
                "days": n,
                "agree": agree,
                "median_abs_dPstar_spx_pts": float(
                    np.median(
                        np.abs(g["P_star"] - g["P_star_1530"]) * g["idx_pts_scale"]
                    )
                ),
                "pnl_per_day": pnl_m,
                "t_pnl": tstat(g["pnl"].to_numpy(float)),
                "buys": int(g["buy"].sum()),
                "mean_R_buy": float(rb.mean()) if len(rb) else np.nan,
                "t_R_buy": tstat(rb),
                "pnl_per_day_1530": base_m,
                "d_pnl": float(dl.mean()),
                "d_pnl_ci_lo": float(np.quantile(bm, 0.025)),
                "d_pnl_ci_hi": float(np.quantile(bm, 0.975)),
                "p_boot_d_lt_0": float((bm < 0).mean()),
                "pair_changed": float(g["pair_changed"].mean()),
                "keeps_edge": bool(
                    base_m > 0
                    and pnl_m >= base_m - EDGE_PNL_FRAC * base_m
                    and agree >= EDGE_AGREE
                ),
            }
        )
    summ = pd.DataFrame(summ)
    summ.to_csv(OUT / "summary.csv", index=False)
    earliest = []
    for (venue, scheme, variant, iset), g in summ.groupby(
        ["venue", "scheme", "variant", "iset"], sort=False
    ):
        g = g.set_index("w").reindex(list(W_CLOCKS))
        ok = g["keeps_edge"].fillna(False).astype(bool)
        first = next((w for w in W_CLOCKS if ok[w]), None)
        ag = g["agree"] >= EDGE_AGREE
        first_agree = next((w for w in W_CLOCKS if ag[w]), None)
        all_after = next(
            (w for w in W_CLOCKS if ok.iloc[W_CLOCKS.index(w) :].all()), None
        )
        earliest.append(
            {
                "venue": venue,
                "scheme": scheme,
                "variant": variant,
                "iset": iset,
                "earliest_w_meeting": first,
                "earliest_w_from_which_all_later_meet": all_after,
                "earliest_w_agreement_ge_90pct": first_agree,
                "pnl_per_day_1530": float(g["pnl_per_day_1530"].iloc[-1]),
            }
        )
    earliest = pd.DataFrame(earliest)
    earliest.to_csv(OUT / "earliest.csv", index=False)
    for venue in ("SPX", "XSP"):
        b0 = summ[
            (summ.venue == venue)
            & (summ.scheme == PRIMARY)
            & (summ.variant == "i")
            & (summ.iset == "idealized")
            & (summ.w == "15:30")
        ].iloc[0]
        log(
            f"the current 15:30 card on {venue}: {int(b0['days'])} days, {int(b0['buys'])} buys, "
            f"per-day P&L {b0['pnl_per_day']:+.4f} (t {b0['t_pnl']:.2f}), mean R on buys "
            f"{b0['mean_R_buy']:+.4f}"
            + (
                "  -- no positive edge at 15:30 to keep: 'keeps the edge' cannot hold"
                if b0["pnl_per_day"] <= 0
                else ""
            )
        )

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    cols = [
        "venue",
        "variant",
        "iset",
        "w",
        "h",
        "k",
        "days",
        "agree",
        "median_abs_dPstar_spx_pts",
        "pnl_per_day",
        "t_pnl",
        "buys",
        "mean_R_buy",
        "t_R_buy",
        "d_pnl",
        "d_pnl_ci_lo",
        "d_pnl_ci_hi",
        "pair_changed",
        "keeps_edge",
    ]
    for scheme in SCHEMES:
        log(
            f"\nTABLE ({scheme} extrapolation{' -- PRIMARY' if scheme == PRIMARY else ' -- sensitivity'}):"
        )
        s = summ[summ.scheme == scheme].copy()
        s["variant"] = pd.Categorical(s["variant"], ["ii", "i"])
        s = s.sort_values(
            ["venue", "variant", "iset", "w"],
            key=lambda c: c if c.name != "venue" else c.map({"SPX": 0, "XSP": 1}),
        )
        log(s[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    log("\nEARLIEST w KEEPING THE EDGE (P&L/day >= 0.9 x 15:30 AND agreement >= 90 %):")
    log(earliest.to_string(index=False))
    gate["total_wall_s"] = round(time.time() - T0, 1)
    (OUT / "gate.json").write_text(json.dumps(gate, indent=1, default=str))
    (OUT / "summary.txt").write_text("\n".join(_LOG), encoding="utf-8")
    (OUT / "run.log").write_text("\n".join(_LOG), encoding="utf-8")
    log(f"\nwall clock {gate['total_wall_s']}s; work dir {work}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
