"""Proposal 15 - baseline attribution: why the baseline beats a daily HAR.

A PRE-REGISTERED ATTRIBUTION STUDY, not a search. The eight designs below were
fixed before any number was seen. Nothing is selected from a grid; no design was
added, dropped or re-specified after a result.

THE QUESTION. On the 15:30 trade the baseline (HAR + calendar OLS) beats a
traditional daily HAR. Why? Is it overnight persistence, the intraday rungs, the
calendar block, or the diurnal profile?

HOW THE BASELINE IS BUILT (arm a0_ols_har of the main repository). Minimum-norm
ordinary least squares on 52 columns of a 24-hour 30-minute futures panel: the
base-2 HAR lag ladder har_ma_{1,2,4,...,2048} on adj_RV, shifted one bar; the
ladder's session-edge interactions _x_open and _x_close; and a pure-calendar
block (DOW_0..DOW_4, hour, is_overnight, is_open, is_close, is_opex,
is_opex_week, is_quad_witch, is_rebalance_close, is_month_end, is_quarter_end,
days_to_opex). The window is 24,000 bars ending strictly before the target bar,
refit at every bar. The target is the winsorized sqrt(RV / B) with B the
per-slot trailing-20 diurnal profile. Bar-end stamps, naive ET: the 16:00-stamp
row is the 15:30->close bar the trade prices.

THE EIGHT DESIGNS. All are minimum-norm OLS on the same window, the same target,
the same profile, refit at every stamp that is scored or that the recalibration
reads. They differ only in the regressors.

  N0  daily HAR only - trailing 1-day, 5-day and 22-day means of RV, that is
      48, 240 and 1,056 bars of the 24-hour grid (48 bars = 1 day), shifted one
      bar, plus an intercept. No calendar, no session-edge interactions. The
      rungs are means of the RAW realized variance: the diurnal profile enters
      N0 only through the target, so N0's forecast is profile-scaled like every
      other design's - the only difference between designs is the regressors.
  N1  the daily-and-above rungs the forecaster actually uses:
      har_ma_{32,64,128,256,512,1024,2048} on the profile-adjusted target
      series (32 bars = 16 hours, 64 = 1.3 days, 1024 = 21 days). N0 -> N1 is
      the diurnal profile moving from the target alone into the regressors, on
      the panel's own base-2 ladder.
  N2  N1 plus the session rungs har_ma_{4,8,16} - two, four and eight hours.
  N3  N2 plus the last-hour rungs har_ma_{1,2} - the last half hour and hour.
  N4  N3 plus the 24 session-edge interactions.
  N5  N4 plus the 16-column calendar block = the full baseline. Must equal the
      gate.
  N6  the overnight question: N5 with the ladder recomputed on SESSION BARS
      ONLY (stamps 10:00..16:00, 13 bars on the modal session). Rung r is the
      mean of the last r session bars, so each rung spans the same number of
      session bars as before and no overnight bar enters. Otherwise identical
      to N5.
  N7  the complement: N5 with the ladder computed on OVERNIGHT BARS ONLY
      (every stamp outside 10:00..16:00). Otherwise identical to N5. The script
      prints the measured bars a session on both sides.

SCORING. Each design's forecasts are written to a parquet in the deck's format
and read back through the library's own recalibration (asl.load_yhat_1530), so
the Mincer-Zarnowitz map is identical for every design. Then, on the deck's 866
days: forecast QLIKE of rv_hat against rv_raw at the 16:00 stamps with a paired
Diebold-Mariano t against N5 (autocorrelation-robust, Bartlett lag
floor(1.5 n^(1/3))); the trade q = sign(rv_hat - iv_var) at the midpoint and at
the crossed spread (long pays ask_c + ask_p, short receives bid_c + bid_p, cash
settlement, no exit spread); the paired Sharpe difference against N5 with a
circular block bootstrap (block 21, B 2000, rng(0), draws shared across designs
and fills), percentile and basic intervals; sign agreement with N5; and the rank
correlation of each design's 16:00 forecast with the 15:00-15:30 realized
variance (the last bar it can see) and with the prior session's realized
variance (yesterday's daily RV).

CAUSALITY. Every regressor is shifted one bar against its bar-end label. The
assertion is tested, not assumed: on ten cut days everything at or after the
day's 16:00 stamp is multiplied (RV by 9, the target by 3), every ladder is
rebuilt through the same pipeline and every design is refit. Day d's own 16:00
forecast must not move; a later day's must.

Outputs: CSVs and one figure under results/atm_straddle_intraday/proposals/15/.
Every number in 15_baseline_attribution.md is printed by this script.

Run:  python writeup/intraday_proposals/15_baseline_attribution.py

Environment:
  P15_MAIN   the main checkout holding the panel (default the path below)
  P15_CACHE  npz cache of the panel extract (default under the system temp dir)
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _bootstrap_repo(start: Path) -> Path:
    for q in [start.resolve(), *start.resolve().parents]:
        if (q / "notebooks" / "atm_straddle_lib.py").exists():
            return q
    raise FileNotFoundError("repo root not found from " + str(start))


sys.path.insert(0, str(_bootstrap_repo(Path(__file__)) / "notebooks"))

import atm_straddle_lib as asl  # noqa: E402

REPO = asl.find_repo(Path(__file__).resolve().parent)
DECK_DIR = REPO / "results" / "atm_straddle_0dte_1530"
OUT = REPO / "results" / "atm_straddle_intraday" / "proposals" / "15"

# The main checkout: the forecasting pipeline and the b2 panel a0 is fitted on.
MAIN = Path(os.environ.get("P15_MAIN", r"C:\Users\james\CC Allowed\harxhar-clean"))
CACHE = Path(
    os.environ.get("P15_CACHE", str(Path(tempfile.gettempdir()) / "p15_panel.npz"))
)

TAG = "a0"  # the forecast under study: baseline (HAR + calendar OLS)
SEED = 0
BOOT_B = 2000
BOOT_BLOCK = 21
ANN = float(np.sqrt(asl.PERIODS_PER_YEAR))
FILLS = ("mid", "crossed")

PPD = 48  # bars a day on the 24-hour 30-minute grid
N0_RUNGS = (1 * PPD, 5 * PPD, 22 * PPD)  # 48, 240, 1056 bars = 1, 5, 22 days
LADDER = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
SESSION_MINUTES = (10 * 60, 16 * 60)  # stamps 10:00..16:00 = the 13 RTH bars
FIT_MINUTES = asl.FIT_MASK_MINUTES  # stamps 10:30..16:00, the library's fit mask
CLOSE_MINS = 16 * 60
LAST_BAR_MINS = 15 * 60 + 30  # the 15:00-15:30 bar's stamp
MARGIN_SESSIONS = (
    260  # sessions fitted before the first scored day (>= the 250 the map needs)
)
N_CUTS = 10
TEETH_SESSIONS = 20  # sessions after a cut day for the teeth check

# The deck's own a0 rule table on its 866 days.
GATE = {
    "yhat_rel": 1e-9,
    "sign_sharpe": 0.967310,
    "as_sharpe": 0.203779,
    "sharpe_tol": 1e-6,
}

DESIGNS = [
    ("N0", "daily HAR only (1, 5, 22 day means of RV)"),
    ("N1", "N0 -> the profile-adjusted daily-and-above rungs"),
    ("N2", "N1 + session rungs (4, 8, 16 bars)"),
    ("N3", "N2 + last-hour rungs (1, 2 bars)"),
    ("N4", "N3 + session-edge interactions"),
    ("N5", "N4 + calendar = the full baseline"),
    ("N6", "N5, ladder on session bars only"),
    ("N7", "N5, ladder on overnight bars only"),
]


def tick(t0: float, msg: str) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def hdr(s: str) -> None:
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78, flush=True)


# ------------------------------------------------------------------ the panel


def load_panel() -> dict[str, Any]:
    """The b2 panel's backbone design plus the tape, cached to one npz.

    The panel is the main repository's own build (unification._load_panel,
    which calls run_geometry_local.prepare_full on the cached prep matrix).
    FEATURE_SET_TAG is cleared first: a0 is fitted on the incumbent panel, not
    the FOMC one (results/spxw_pnl/MANIFEST.md).
    """
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        return {
            "F": z["F"],
            "kept": [str(v) for v in z["kept"]],
            "y": z["y"],
            "baseline": z["baseline"],
            "rv_raw": z["rv_raw"],
            "t": z["t"],
            "window": int(z["window"][0]),
            "rcond": float(z["rcond"][0]),
        }
    cwd = Path.cwd()
    os.chdir(MAIN)
    sys.path.insert(0, str(MAIN))
    sys.path.insert(0, str(MAIN / "experiments"))
    os.environ.setdefault("UNIFY_CACHE_DIR", "results")
    import run_geometry_local as rgl
    import src.unification as uni

    rgl.FEATURE_SET_TAG = ""  # the incumbent panel: a0's panel of record
    rgl.CACHE_DIR = "results"
    p = uni._load_panel()
    spec = uni.ARMS["a0_ols_har"]
    idx, nm = uni._design_cols(p, spec)
    f, kept, dropped = uni._dedup_ols_design(p.X[:, idx], nm)
    if dropped:
        raise ValueError(f"the a0 design lost columns to dedup: {dropped}")
    mat = np.ascontiguousarray(f, dtype=np.float64)
    names = list(kept)
    yv = np.asarray(p.y, dtype=np.float64)
    bv = np.asarray(p.baseline, dtype=np.float64)
    rv = np.asarray(p.rv_raw, dtype=np.float64)
    tv = np.asarray(p.t)
    win = int(uni.DEFAULT_WINDOW_BARS)
    rcond = float(uni.PINV_RCOND)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        CACHE,
        F=mat,
        kept=np.array(names, dtype=object),
        y=yv,
        baseline=bv,
        rv_raw=rv,
        t=tv,
        window=np.array([win]),
        rcond=np.array([rcond]),
    )
    os.chdir(cwd)
    return {
        "F": mat,
        "kept": names,
        "y": yv,
        "baseline": bv,
        "rv_raw": rv,
        "t": tv,
        "window": win,
        "rcond": rcond,
    }


def robust_scale(raw: np.ndarray, window: int) -> np.ndarray:
    """The pipeline's own whole-series rolling robust scaler.

    HAR and calendar columns take no reference-IQR floor and are not passed
    through unscaled (executor._build_scale_guards), so ref_iqr=None and
    fixed_cols=None reproduce the panel's treatment of a ladder column exactly.
    """
    if str(MAIN) not in sys.path:
        sys.path.insert(0, str(MAIN))
    from src.features.transforms.scaling import rolling_robust_scale

    return rolling_robust_scale(np.nan_to_num(raw, nan=0.0), window)


# ------------------------------------------------------- ladders and designs


def trailing_means(
    series: np.ndarray, rungs, take: np.ndarray | None = None
) -> np.ndarray:
    """Trailing means over `rungs` bars, shifted one bar (the ladder's kernel).

    `take` selects the bars the mean runs over: None is every panel bar (the
    forecaster's own ladder), a session mask gives N6's ladder, an overnight
    mask N7's. In every case rung r averages the last r TAKEN bars strictly
    before the row, so the rung spans the same number of taken bars as the
    panel ladder spans panel bars. Partial windows are allowed at the very
    start (pandas min_periods=1), which is the pipeline's convention.
    """
    if take is None:
        s = pd.Series(series)
        return np.column_stack(
            [s.rolling(int(r), min_periods=1).mean().shift(1).to_numpy() for r in rungs]
        )
    m = np.asarray(take, dtype=bool)
    sub = np.asarray(series, dtype=float)[m]
    pre = np.concatenate([[0.0], np.cumsum(sub)])
    k = np.cumsum(m) - m  # taken bars strictly before each row
    cols = []
    for r in rungs:
        lo = np.maximum(k - int(r), 0)
        cnt = (k - lo).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            cols.append(
                np.where(cnt > 0, (pre[k] - pre[lo]) / np.maximum(cnt, 1), np.nan)
            )
    return np.column_stack(cols)


def edge_block(rungs_raw: np.ndarray, is_open: np.ndarray, is_close: np.ndarray):
    """A ladder with its session-edge interactions, in the pipeline's order."""
    return np.column_stack(
        [rungs_raw, rungs_raw * is_open[:, None], rungs_raw * is_close[:, None]]
    )


def build_matrices(
    pan: dict,
) -> tuple[dict[str, np.ndarray], dict[str, tuple[str, np.ndarray]], dict]:
    """One column block per design family, plus the design column index sets."""
    kept = pan["kept"]
    f = pan["F"]
    win = pan["window"]
    t = pd.DatetimeIndex(pan["t"])
    mins = (t.hour * 60 + t.minute).to_numpy()
    sess = (mins >= SESSION_MINUTES[0]) & (mins <= SESSION_MINUTES[1])
    over = ~sess
    is_open = f[:, kept.index("is_open")]
    is_close = f[:, kept.index("is_close")]
    for nm, col in (("is_open", is_open), ("is_close", is_close)):
        if not np.array_equal(np.unique(col), np.array([0.0, 1.0])):
            raise ValueError(f"{nm} is not a 0/1 column in the scaled panel")

    har = [kept.index(f"har_ma_{r}") for r in LADDER]
    edge = [kept.index(f"har_ma_{r}_x_{s}") for r in LADDER for s in ("open", "close")]
    cal = [j for j, nm in enumerate(kept) if not nm.startswith("har_ma_")]
    if len(har) + len(edge) + len(cal) != len(kept):
        raise ValueError("the backbone did not partition into ladder/edge/calendar")

    # N0: the traditional daily HAR, raw RV, same scaler as every other column
    m0 = robust_scale(trailing_means(pan["rv_raw"], N0_RUNGS), win)
    # N6 / N7: the same 36-column ladder block, restricted to session / overnight bars
    m6 = robust_scale(
        edge_block(trailing_means(pan["y"], LADDER, sess), is_open, is_close), win
    )
    m7 = robust_scale(
        edge_block(trailing_means(pan["y"], LADDER, over), is_open, is_close), win
    )
    blocks = {
        "panel": f,
        "N0": m0,
        "N6": np.hstack([m6, f[:, cal]]),
        "N7": np.hstack([m7, f[:, cal]]),
    }

    slow = [kept.index(f"har_ma_{r}") for r in (32, 64, 128, 256, 512, 1024, 2048)]
    sessr = [kept.index(f"har_ma_{r}") for r in (4, 8, 16)]
    lasth = [kept.index(f"har_ma_{r}") for r in (1, 2)]
    cols = {
        "N0": ("N0", np.arange(3)),
        "N1": ("panel", np.array(slow)),
        "N2": ("panel", np.array(slow + sessr)),
        "N3": ("panel", np.array(slow + sessr + lasth)),
        "N4": ("panel", np.array(har + edge)),
        "N5": ("panel", np.array(har + edge + cal)),
        "N6": ("N6", np.arange(36 + len(cal))),
        "N7": ("N7", np.arange(36 + len(cal))),
    }
    per_day = pd.Series(sess).groupby(t.normalize()).sum()
    full = per_day[per_day > 0]
    meta = {
        "mins": mins,
        "sess": sess,
        "over": over,
        "n_cal": len(cal),
        "sess_bars_modal": int(full.mode().iloc[0]),
        "sess_bars_mean": float(full.mean()),
        "over_bars_mean": float(
            pd.Series(over).groupby(t.normalize()).sum().reindex(full.index).mean()
        ),
    }
    return blocks, cols, meta


# ------------------------------------------------------------- walk-forward


def walk(
    mat: np.ndarray,
    subsets: dict,
    rows: np.ndarray,
    y: np.ndarray,
    win: int,
    rcond: float,
):
    """Minimum-norm OLS forecasts for several column subsets of one matrix.

    The window's centred cross-product is formed once per row; each subset then
    reads its own principal submatrix, so nested designs cost one pass. The
    solve is the arm's own: eigendecomposition, modes below rcond * lambda_max
    discarded, unpenalized intercept.
    """
    out = {k: np.empty(len(rows)) for k in subsets}
    for i, r in enumerate(rows):
        w = mat[r - win : r]
        ys = y[r - win : r]
        mu = w.mean(0)
        my = float(ys.mean())
        wc = w - mu
        gram = wc.T @ wc
        rhs = wc.T @ (ys - my)
        xr = mat[r]
        for k, idx in subsets.items():
            g = gram[np.ix_(idx, idx)]
            lam, v = np.linalg.eigh(g)
            keep = lam > rcond * float(lam[-1])
            vk = v[:, keep]
            coef = vk @ ((vk.T @ rhs[idx]) / lam[keep])
            out[k][i] = float(xr[idx] @ coef + (my - float(mu[idx] @ coef)))
    return out


def forecast_all(blocks, cols, rows, pan, t0) -> dict[str, np.ndarray]:
    """Every design's forecast at every fit row, one pass per column block."""
    by_block: dict[str, dict] = {}
    for name, (blk, idx) in cols.items():
        by_block.setdefault(blk, {})[name] = idx
    yh: dict[str, np.ndarray] = {}
    for blk, subsets in by_block.items():
        res = walk(blocks[blk], subsets, rows, pan["y"], pan["window"], pan["rcond"])
        yh.update(res)
        tick(
            t0, f"block {blk}: {', '.join(sorted(subsets))} fitted at {len(rows)} rows"
        )
    return yh


# -------------------------------------------------------------- the library


def recalibrated(yhat: np.ndarray, rows, pan, need_dates, path: Path) -> pd.DataFrame:
    """Write one design's table in the deck's format, read it back through asl."""
    t = pd.DatetimeIndex(pan["t"][rows])
    pd.DataFrame(
        {
            "t": t.tz_localize("America/New_York").tz_convert("UTC"),
            "yhat": np.asarray(yhat, dtype=float),
            "baseline": pan["baseline"][rows],
            "rv_raw": pan["rv_raw"][rows],
        }
    ).to_parquet(path, index=False)
    return asl.load_yhat_1530(path, need_dates=need_dates)


# -------------------------------------------------------------- statistics


def sharpe(x) -> float:
    v = pd.Series(x).astype(float).dropna().to_numpy()
    sd = float(v.std(ddof=1)) if len(v) >= 2 else 0.0
    return float(v.mean() / sd * ANN) if sd > 0 else float("nan")


def qlike(f, y) -> pd.Series:
    """QLIKE loss y/f - log(y/f) - 1; rows with y <= 0 or f <= 0 are NaN."""
    f = pd.Series(f).astype(float)
    y = pd.Series(y).astype(float)
    y.index = f.index
    ok = (f > 0) & (y > 0)
    r = y.where(ok) / f.where(ok)
    return r - np.log(r) - 1.0


_BOOT_IDX: dict[int, np.ndarray] = {}


def boot_idx(n: int) -> np.ndarray:
    """One circular-block index array per sample size, shared across designs."""
    if n not in _BOOT_IDX:
        _BOOT_IDX[n] = asl.circular_block_bootstrap_idx(
            np.random.default_rng(SEED), n, BOOT_BLOCK, BOOT_B
        )
    return _BOOT_IDX[n]


def paired_stats(base: pd.Series, cell: pd.Series) -> dict[str, float]:
    """cell - base on the common days: Sharpe difference and its intervals."""
    j = pd.concat([base.rename("a"), cell.rename("b")], axis=1).dropna()
    x = j["b"].to_numpy(float)
    y = j["a"].to_numpy(float)
    d = x - y
    n = len(d)
    hac, lag = asl.newey_west_t(pd.Series(d))
    hat = (x.mean() / x.std(ddof=1) - y.mean() / y.std(ddof=1)) * ANN
    idx = boot_idx(n)
    xa, ya = x[idx], y[idx]
    ds = (
        xa.mean(axis=1) / xa.std(axis=1, ddof=1)
        - ya.mean(axis=1) / ya.std(axis=1, ddof=1)
    ) * ANN
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return {
        "n": float(n),
        "mean_diff": float(d.mean()),
        "hac_t_diff": float(hac),
        "hac_lag": float(lag),
        "dSharpe": float(hat),
        "pctile_lo": float(lo),
        "pctile_hi": float(hi),
        "basic_lo": float(2.0 * hat - hi),
        "basic_hi": float(2.0 * hat - lo),
    }


def returns_for(px: pd.DataFrame, q: pd.Series, fill: str) -> pd.Series:
    if fill == "mid":
        return q.astype(float) * px["R"].astype(float)
    return asl.crossed_premium_return(
        q.astype(float), px["exit"], px["bid_entry"], px["ask_entry"]
    )


# ==================================================================== main


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 60)
    print(f"proposal 15 - baseline attribution.  repo {REPO}")
    print(
        "PRE-REGISTERED: eight fixed designs, one scoring protocol. Nothing was "
        "added, dropped or re-specified after a number was seen."
    )

    pan = load_panel()
    tick(t0, f"panel: {len(pan['y'])} rows, {pan['F'].shape[1]} backbone columns")
    t = pd.DatetimeIndex(pan["t"])
    blocks, cols, meta = build_matrices(pan)
    tick(t0, "ladders rebuilt (N0 raw-RV, N6 session-only, N7 overnight-only)")

    mins = meta["mins"]
    dates = t.normalize()
    sess_dates = pd.DatetimeIndex(sorted(set(dates[mins == CLOSE_MINS])))
    deck = pd.read_parquet(DECK_DIR / f"daily_{TAG}.parquet")
    days = pd.DatetimeIndex(deck.index)
    k0 = int(np.searchsorted(sess_dates, days[0]))
    start = sess_dates[k0 - MARGIN_SESSIONS]
    lo, hi = FIT_MINUTES
    rows = np.flatnonzero(
        (mins >= lo) & (mins <= hi) & (dates >= start) & (dates <= days[-1])
    )
    print(
        f"scored days {len(days)} ({days[0].date()}..{days[-1].date()}); "
        f"fit stamps {lo // 60}:{lo % 60:02d}..{hi // 60}:{hi % 60:02d} from "
        f"{start.date()} ({MARGIN_SESSIONS} sessions of margin) = {len(rows)} rows"
    )
    print(
        f"grid: {PPD} bars a day. Session bars (stamps 10:00..16:00) "
        f"{meta['sess_bars_modal']} a session on the modal day, "
        f"{meta['sess_bars_mean']:.2f} on average (half sessions carry fewer); "
        f"overnight bars {meta['over_bars_mean']:.2f} on average. "
        f"N0 rungs {N0_RUNGS} bars = 1, 5, 22 days."
    )

    # ------------------------------------------------- 0. the reconstruction gate
    hdr("0. GATE - the panel ladder rebuilt from the target series")
    rebuilt = robust_scale(
        edge_block(
            trailing_means(pan["y"], LADDER),
            blocks["panel"][:, pan["kept"].index("is_open")],
            blocks["panel"][:, pan["kept"].index("is_close")],
        ),
        pan["window"],
    )
    names36 = [f"har_ma_{r}" for r in LADDER]
    names36 += [f"har_ma_{r}_x_open" for r in LADDER]
    names36 += [f"har_ma_{r}_x_close" for r in LADDER]
    gap = max(
        float(
            np.abs(rebuilt[100000:, k] - pan["F"][100000:, pan["kept"].index(nm)]).max()
        )
        for k, nm in enumerate(names36)
    )
    print(
        f"the 36 ladder + edge columns rebuilt from the panel's own target and "
        f"rescaled by the pipeline's scaler match the panel to {gap:.3e} "
        f"(rows 100,000 on). N0, N6 and N7 are built by this same path."
    )
    if gap > 1e-9:
        raise SystemExit("ladder reconstruction gate failed")

    # -------------------------------------------------------- 1. the forecasts
    hdr("1. THE EIGHT DESIGNS - walk-forward forecasts")
    for name, desc in DESIGNS:
        blk, idx = cols[name]
        print(f"  {name}  {len(idx):3d} columns  {desc}")
    yh = forecast_all(blocks, cols, rows, pan, t0)

    # ------------------------------------------------------------- 2. the gate
    hdr("2. GATE - N5 is the deck's baseline (HAR + calendar OLS)")
    shipped = pd.read_parquet(REPO / "results" / "spxw_pnl" / f"yhat_{TAG}.parquet")
    shipped["et"] = shipped["t"].dt.tz_convert("America/New_York").dt.tz_localize(None)
    ref = shipped.set_index("et")["yhat"].reindex(t[rows]).to_numpy()
    rel = np.abs(yh["N5"] - ref) / np.maximum(np.abs(ref), 1e-12)
    print(
        f"N5 against the shipped a0 table at all {len(rows)} fit rows: "
        f"max relative error {np.nanmax(rel):.3e} (gate {GATE['yhat_rel']:.0e}), "
        f"{int(np.isnan(rel).sum())} unmatched stamps"
    )
    if not (np.nanmax(rel) < GATE["yhat_rel"]) or int(np.isnan(rel).sum()):
        raise SystemExit("forecast gate failed")

    tmp = OUT / "_tmp_yhat.parquet"
    rec = {k: recalibrated(v, rows, pan, days, tmp) for k, v in yh.items()}
    tmp.unlink(missing_ok=True)
    tick(t0, "every design recalibrated through asl.load_yhat_1530")
    dgap = float((rec["N5"]["rv_hat"].reindex(days) - deck["rv_hat"]).abs().max())
    print(f"N5's recalibrated rv_hat against the deck's: max |difference| {dgap:.3e}")

    px = deck.copy()
    px["bid_entry"] = px["bid_c"].astype(float) + px["bid_p"].astype(float)
    px["ask_entry"] = px["ask_c"].astype(float) + px["ask_p"].astype(float)
    px["signal"] = rec["N5"]["rv_hat"].reindex(days) - px["iv_var"]
    sizes = asl.rule_sizes(px)
    gate_tab = pd.DataFrame(
        {
            nm: asl.rule_row(sizes[nm] * px["R"], sizes[nm])
            for nm in ("always short", "sign(s)")
        }
    ).T
    print(gate_tab.to_string())
    g_sign = float(gate_tab.loc["sign(s)", "Sharpe_ann"])
    g_as = float(gate_tab.loc["always short", "Sharpe_ann"])
    print(
        f"deck sign(s) {GATE['sign_sharpe']:.6f} vs {g_sign:.6f} "
        f"(|d| {abs(g_sign - GATE['sign_sharpe']):.2e}); "
        f"always short {GATE['as_sharpe']:.6f} vs {g_as:.6f} "
        f"(|d| {abs(g_as - GATE['as_sharpe']):.2e})"
    )
    if (
        abs(g_sign - GATE["sign_sharpe"]) > GATE["sharpe_tol"]
        or abs(g_as - GATE["as_sharpe"]) > GATE["sharpe_tol"]
    ):
        raise SystemExit("rule-row gate failed")
    print(
        f"crossed fills unpriceable on "
        f"{asl.crossed_untradeable_count(sizes['sign(s)'], px['bid_entry'], px['ask_entry'])} "
        f"of the {len(days)} days"
    )

    # ------------------------------------------------------ 3. what each leans on
    hdr("3. WHAT EACH DESIGN LEANS ON")
    close_row = mins == CLOSE_MINS
    last_row = mins == LAST_BAR_MINS
    rv_close = pd.Series(pan["rv_raw"][close_row], index=dates[close_row])
    rv_last = pd.Series(pan["rv_raw"][last_row], index=dates[last_row])
    rv_close = rv_close[~rv_close.index.duplicated()]
    rv_last = rv_last[~rv_last.index.duplicated()]
    sess_rv = (
        pd.Series(pan["rv_raw"][meta["sess"]], index=dates[meta["sess"]])
        .groupby(level=0)
        .sum()
        .reindex(sess_dates)
    )
    yday_rv = sess_rv.shift(1)

    daily = pd.DataFrame(index=days)
    for name, _ in DESIGNS:
        daily[name] = rec[name]["rv_hat"].reindex(days)
    daily["rv_raw_1600"] = rv_close.reindex(days)
    daily["rv_1500_1530"] = rv_last.reindex(days)
    daily["rv_yesterday"] = yday_rv.reindex(days)
    daily["iv_var"] = px["iv_var"]
    daily.to_csv(OUT / "15_daily_forecasts.csv")
    if int(
        daily[[n for n, _ in DESIGNS] + ["rv_raw_1600", "rv_1500_1530", "rv_yesterday"]]
        .isna()
        .sum()
        .sum()
    ):
        raise SystemExit("a design or a tape column is missing on some scored day")
    print(
        "the two horizons each forecast is scored against: the last bar the "
        "forecast can see (the 15:00-15:30 realized variance, stamp 15:30) and "
        "the prior session's realized variance (the sum of its "
        f"{meta['sess_bars_modal']} session bars)."
    )
    print(
        daily[["rv_raw_1600", "rv_1500_1530", "rv_yesterday", "iv_var"]]
        .describe()
        .loc[["mean", "50%", "max"]]
        .to_string()
    )
    print(
        "rank correlation between the two horizons themselves: "
        f"{daily['rv_1500_1530'].corr(daily['rv_yesterday'], method='spearman'):.4f}"
    )

    # ----------------------------------------------------------- 4. the table
    hdr("4. THE TABLE")
    loss = {n: qlike(daily[n], daily["rv_raw_1600"]) for n, _ in DESIGNS}
    ret: dict[tuple[str, str], pd.Series] = {}
    pos: dict[str, pd.Series] = {}
    for name, _ in DESIGNS:
        s = daily[name] - daily["iv_var"]
        pos[name] = pd.Series(np.where(s.to_numpy() > 0, 1.0, -1.0), index=days)
    rows_out = []
    for name, desc in DESIGNS:
        q = pos[name]
        row = {"design": name, "what": desc, "n_cols": len(cols[name][1])}
        row["QLIKE"] = float(loss[name].mean())
        dm, lag = asl.newey_west_t(loss[name] - loss["N5"])
        row["DM_t_vs_N5"] = float(dm) if name != "N5" else float("nan")
        row["DM_lag"] = float(lag)
        for fill in FILLS:
            r = returns_for(px, q, fill)
            ret[(name, fill)] = r
            rr = asl.rule_row(r, q)
            row[f"mean_{fill}"] = float(rr["mean"])
            row[f"t_{fill}"] = float(rr["t_mean"])
            row[f"Sharpe_{fill}"] = float(rr["Sharpe_ann"])
            row[f"pct_long_{fill}"] = float(rr["pct_buy"])
        row["agree_N5_pct"] = 100.0 * float((q == pos["N5"]).mean())
        row["rho_last_bar"] = float(
            daily[name].corr(daily["rv_1500_1530"], method="spearman")
        )
        row["rho_yesterday"] = float(
            daily[name].corr(daily["rv_yesterday"], method="spearman")
        )
        rows_out.append(row)

    paired = {}
    for name, _ in DESIGNS:
        if name == "N5":
            continue
        for fill in FILLS:
            paired[(name, fill)] = paired_stats(ret[("N5", fill)], ret[(name, fill)])
    for row in rows_out:
        for fill in FILLS:
            p = paired.get((str(row["design"]), fill))
            row[f"dSharpe_{fill}"] = p["dSharpe"] if p else float("nan")
            row[f"ci_lo_{fill}"] = p["pctile_lo"] if p else float("nan")
            row[f"ci_hi_{fill}"] = p["pctile_hi"] if p else float("nan")
            row[f"basic_lo_{fill}"] = p["basic_lo"] if p else float("nan")
            row[f"basic_hi_{fill}"] = p["basic_hi"] if p else float("nan")

    as_r = {f: returns_for(px, pd.Series(-1.0, index=days), f) for f in FILLS}
    tab = pd.DataFrame(rows_out).set_index("design")
    tab.to_csv(OUT / "15_main_table.csv")
    show = tab[
        [
            "n_cols",
            "QLIKE",
            "DM_t_vs_N5",
            "Sharpe_mid",
            "Sharpe_crossed",
            "dSharpe_mid",
            "ci_lo_mid",
            "ci_hi_mid",
            "dSharpe_crossed",
            "ci_lo_crossed",
            "ci_hi_crossed",
            "pct_long_mid",
            "agree_N5_pct",
            "rho_last_bar",
            "rho_yesterday",
        ]
    ]
    print(show.round(4).to_string())
    print(
        f"\nalways short: Sharpe {sharpe(as_r['mid']):.4f} mid, "
        f"{sharpe(as_r['crossed']):.4f} crossed on the same {len(days)} days"
    )
    print("\nmean, t and pct long at both fills:")
    print(
        tab[
            [
                "mean_mid",
                "t_mid",
                "pct_long_mid",
                "mean_crossed",
                "t_crossed",
                "pct_long_crossed",
            ]
        ]
        .round(6)
        .to_string()
    )
    pf = pd.DataFrame([{"design": k[0], "fill": k[1], **v} for k, v in paired.items()])
    pf.to_csv(OUT / "15_paired_vs_n5.csv", index=False)
    print(
        f"\npaired Sharpe difference against N5 (design - N5), circular block "
        f"bootstrap, block {BOOT_BLOCK}, B {BOOT_B}, rng({SEED}), draws shared:"
    )
    print(
        pf[
            [
                "design",
                "fill",
                "dSharpe",
                "pctile_lo",
                "pctile_hi",
                "basic_lo",
                "basic_hi",
                "hac_t_diff",
                "hac_lag",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )
    print(f"\nQLIKE and every trade statistic above are on all {len(days)} days.")

    # ------------------------------------------------- 5. the attribution steps
    hdr("5. THE LADDER, STEP BY STEP")
    steps = []
    ladder = ["N0", "N1", "N2", "N3", "N4", "N5"]
    for a, b in zip(ladder[:-1], ladder[1:]):
        row = {"step": f"{a} -> {b}", "adds": dict(DESIGNS)[b]}
        for fill in FILLS:
            row[f"dSharpe_{fill}"] = sharpe(ret[(b, fill)]) - sharpe(ret[(a, fill)])
        row["dQLIKE"] = float(loss[b].mean() - loss[a].mean())
        steps.append(row)
    for a, b in (("N5", "N6"), ("N5", "N7")):
        row = {"step": f"{a} -> {b}", "adds": dict(DESIGNS)[b]}
        for fill in FILLS:
            row[f"dSharpe_{fill}"] = sharpe(ret[(b, fill)]) - sharpe(ret[(a, fill)])
        row["dQLIKE"] = float(loss[b].mean() - loss[a].mean())
        steps.append(row)
    st = pd.DataFrame(steps)
    st.to_csv(OUT / "15_steps.csv", index=False)
    print(st.round(4).to_string(index=False))

    # ------------------------------------------------------------ 6. the figure
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    order = ["N0", "N1", "N2", "N3", "N4", "N5", "N6", "N7"]
    xs = np.array([0, 1, 2, 3, 4, 5, 6.7, 7.7])
    w = 0.38
    for k, (fill, c) in enumerate((("mid", "#2b6cb0"), ("crossed", "#c05621"))):
        vals = [tab.loc[n, f"Sharpe_{fill}"] for n in order]
        ax.bar(xs + (k - 0.5) * w, vals, width=w, color=c, label=f"{fill} fill")
        for x, v in zip(xs + (k - 0.5) * w, vals):
            ax.text(
                x, v + (0.03 if v >= 0 else -0.09), f"{v:.2f}", ha="center", fontsize=8
            )
    ax.margins(y=0.14)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axhline(
        sharpe(as_r["mid"]), color="#2b6cb0", lw=0.8, ls=":", label="always short, mid"
    )
    ax.axhline(
        sharpe(as_r["crossed"]),
        color="#c05621",
        lw=0.8,
        ls=":",
        label="always short, crossed",
    )
    ax.axvline(6.05, color="0.7", lw=1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [
            "N0\ndaily HAR",
            "N1\n+profile",
            "N2\n+session",
            "N3\n+last hour",
            "N4\n+edges",
            "N5\n+calendar\n= baseline",
            "N6\nsession\nbars only",
            "N7\novernight\nbars only",
        ],
        fontsize=8,
    )
    ax.set_ylabel(f"annualized Sharpe (sqrt({asl.PERIODS_PER_YEAR:.0f}))")
    ax.set_title(
        f"sign(s) on the 15:30 trade, {len(days)} days: the nested ladder to the "
        "baseline (HAR + calendar OLS)",
        fontsize=10,
    )
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "15_sharpe_ladder.png", dpi=150)
    plt.close(fig)
    tick(t0, f"figure written to {OUT / '15_sharpe_ladder.png'}")

    # --------------------------------------------------------- 7. causality
    hdr("7. CAUSALITY - nothing at 15:30 may read the bar it trades")
    close_rows = {
        pd.Timestamp(d): int(i)
        for i, d in zip(np.flatnonzero(close_row), dates[close_row])
    }
    # cut days must leave TEETH_SESSIONS sessions of panel after them for the
    # teeth check, which excludes the last few scored days
    room = np.searchsorted(sess_dates, days) + TEETH_SESSIONS < len(sess_dates)
    eligible = pd.Series(days[room])
    cut_days = list(
        eligible.iloc[np.linspace(0, len(eligible) - 1, N_CUTS).round().astype(int)]
    )
    is_open = blocks["panel"][:, pan["kept"].index("is_open")]
    is_close = blocks["panel"][:, pan["kept"].index("is_close")]
    cal_idx = [j for j, nm in enumerate(pan["kept"]) if not nm.startswith("har_ma_")]
    viol = 0
    teeth = 0
    caus = []
    for d in cut_days:
        r = close_rows[pd.Timestamp(d)]
        k = int(np.searchsorted(sess_dates, d))
        r2 = close_rows[pd.Timestamp(sess_dates[k + TEETH_SESSIONS])]
        s0 = max(0, r - 60000)
        s1 = r2 + 1
        sl = slice(s0, s1)
        yv = pan["y"][sl].copy()
        rvv = pan["rv_raw"][sl].copy()
        after = np.arange(s0, s1) >= r
        moved = {}
        for label, (yy, rr) in (
            ("plain", (yv, rvv)),
            (
                "perturbed",
                (np.where(after, yv * 3.0, yv), np.where(after, rvv * 9.0, rvv)),
            ),
        ):
            m0 = robust_scale(trailing_means(rr, N0_RUNGS), pan["window"])
            har36 = robust_scale(
                edge_block(trailing_means(yy, LADDER), is_open[sl], is_close[sl]),
                pan["window"],
            )
            m5 = np.hstack([har36, pan["F"][sl][:, cal_idx]])
            m6 = np.hstack(
                [
                    robust_scale(
                        edge_block(
                            trailing_means(yy, LADDER, meta["sess"][sl]),
                            is_open[sl],
                            is_close[sl],
                        ),
                        pan["window"],
                    ),
                    pan["F"][sl][:, cal_idx],
                ]
            )
            m7 = np.hstack(
                [
                    robust_scale(
                        edge_block(
                            trailing_means(yy, LADDER, meta["over"][sl]),
                            is_open[sl],
                            is_close[sl],
                        ),
                        pan["window"],
                    ),
                    pan["F"][sl][:, cal_idx],
                ]
            )
            loc = np.array([r - s0, r2 - s0])
            got = {}
            got.update(
                walk(m0, {"N0": np.arange(3)}, loc, yy, pan["window"], pan["rcond"])
            )
            n36 = np.arange(36)
            got.update(
                walk(
                    m5,
                    {
                        "N4": n36,
                        "N5": np.arange(36 + len(cal_idx)),
                    },
                    loc,
                    yy,
                    pan["window"],
                    pan["rcond"],
                )
            )
            got.update(
                walk(
                    m6,
                    {"N6": np.arange(36 + len(cal_idx))},
                    loc,
                    yy,
                    pan["window"],
                    pan["rcond"],
                )
            )
            got.update(
                walk(
                    m7,
                    {"N7": np.arange(36 + len(cal_idx))},
                    loc,
                    yy,
                    pan["window"],
                    pan["rcond"],
                )
            )
            moved[label] = got
        row = {"cut": str(pd.Timestamp(d).date())}
        for nm in ("N0", "N4", "N5", "N6", "N7"):
            d_self = abs(moved["perturbed"][nm][0] - moved["plain"][nm][0])
            d_late = abs(moved["perturbed"][nm][1] - moved["plain"][nm][1])
            row[f"d_{nm}"] = d_self
            row[f"later_{nm}"] = d_late
            viol += int(d_self > 0.0)
            teeth += int(d_late > 0.0)
        caus.append(row)
    cf = pd.DataFrame(caus)
    cf.to_csv(OUT / "15_causality.csv", index=False)
    print(cf.to_string(index=False))
    n_checks = N_CUTS * 5
    print(
        f"day d's own 16:00 forecast moved on {viol} of {n_checks} design-days; "
        f"a session {TEETH_SESSIONS} days later moved on {teeth} of {n_checks} (teeth)."
    )
    if viol:
        raise SystemExit("causality violated")

    tick(t0, "done")


if __name__ == "__main__":
    main()
