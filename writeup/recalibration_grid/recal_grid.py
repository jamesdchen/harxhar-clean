"""Recalibration grid for the 0DTE 15:30 straddle deck.

The deck trades sign(s) with s = rv_hat - iv_var, where rv_hat is the causally
recalibrated 15:30->16:00 variance forecast built by notebooks/atm_straddle_lib.py.
That map has three arbitrary choices: the trailing window, the weight exponent in
the weighted least squares, and whether the variance term added back on the square
scale is flat across the window or scaled row by row. This script re-runs the whole
deck over the 6 x 3 x 2 = 36 combinations of those choices and reports every cell.
Nothing is chosen here: the point is the spread, not a winner.

Everything the report states is printed by this script.

The study FOLLOWS THE LIBRARY. Every convention that changes the fit is imported
from notebooks/atm_straddle_lib.py, never restated here: the forecast tables and
their labels and order, the stamp window, the early-close calendar, the session
rule, the window length, the start session, the truncation-lag rule, the block
bootstrap and the annualization. Only the fits themselves are written here, and
they are vectorized rather than looped. Gate 1 is what proves the two agree: the
current cell has to reproduce the deck's own rv_hat bit for bit on every tag. If
the library or the deck moves, gate 1 fails loudly and this study is re-run
rather than silently reinterpreted.

Run:
    python writeup/recalibration_grid/recal_grid.py
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "notebooks"))

import atm_straddle_lib as asl  # noqa: E402

OUT = REPO / "results" / "atm_straddle_0dte_1530" / "recal_grid"
DECK = REPO / "results" / "atm_straddle_0dte_1530"

SCORE_START = pd.Timestamp("2020-01-03")
SCORE_END = pd.Timestamp("2024-04-30")

# --- every fit-changing convention comes from the library -----------------
TAGS = list(asl.MODEL_ORDER)
YHAT_LABEL = dict(asl.YHAT_LABEL)
WINDOW_DAYS = int(asl.WINDOW_DAYS)  # the current cell's flat window, in sessions
MZ_START_DAY = int(asl.MZ_START_DAY)  # first session rank that gets coefficients
FIT_MASK_MINUTES = asl.FIT_MASK_MINUTES  # stamp window, bar-end labels
EARLY_CLOSE_DATES = asl.EARLY_CLOSE_DATES  # 13:00 ET closes, dropped from the fit
PERIODS_PER_YEAR = float(asl.PERIODS_PER_YEAR)

WINDOW_LEVELS: list[tuple[str, str, int]] = [
    ("flat125", "flat", 125),
    ("flat250", "flat", 250),
    ("flat500", "flat", 500),
    ("ewma125", "ewma", 125),
    ("ewma250", "ewma", 250),
    ("ewma500", "ewma", 500),
]
EXPONENTS = (0, 1, 2)
VARIANTS = ("FLAT", "PER-ROW")
CURRENT = "flat250|p2|FLAT"

MIN_ROWS = 200  # the library's flat-path row gate
MIN_NEFF = 200  # the library's Kish effective-sample gate, EWMA path
BOOT_BLOCK = 21
BOOT_B = 2000
BOOT_SEED = 0


def cell_id(win: str, p: int, variant: str) -> str:
    return f"{win}|p{p}|{variant}"


CELLS = [
    cell_id(w[0], p, v) for w in WINDOW_LEVELS for p in EXPONENTS for v in VARIANTS
]


# --------------------------------------------------------------------------
# panel + per-session coefficients
# --------------------------------------------------------------------------
@dataclass
class Panel:
    """Fit-mask rows of one forecast table, in stamp order."""

    u: np.ndarray  # yhat
    y: np.ndarray  # sqrt(rv_raw / baseline)
    b: np.ndarray  # baseline
    rv: np.ndarray  # rv_raw
    rank: np.ndarray  # session rank of each row
    date: np.ndarray  # session date of each row
    is1600: np.ndarray  # the traded stamp
    sess_dates: pd.DatetimeIndex
    start: np.ndarray  # row index where session k starts
    stop: np.ndarray  # row index where session k stops


def load_panel(path: Path) -> Panel:
    df, rth = asl._panel_frame(path)
    yhat = df["yhat"].to_numpy(float)
    base = df["baseline"].to_numpy(float)
    rv_raw = df["rv_raw"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.sqrt(np.maximum(rv_raw, 0.0) / np.maximum(base, 1e-18))
    finite = np.isfinite(yhat) & np.isfinite(y) & (base > 0) & rth
    et = df["et"]
    is1600 = ((et.dt.hour == 16) & (et.dt.minute == 0)).to_numpy() & ~df[
        "early_close"
    ].to_numpy()
    dates = df["date"].to_numpy()
    d_fit = dates[finite]
    sess_dates = pd.DatetimeIndex(np.unique(d_fit))
    rank = np.searchsorted(sess_dates.to_numpy(), d_fit)
    start = np.searchsorted(rank, np.arange(len(sess_dates)), side="left")
    stop = np.searchsorted(rank, np.arange(len(sess_dates)), side="right")
    return Panel(
        u=yhat[finite],
        y=y[finite],
        b=base[finite],
        rv=rv_raw[finite],
        rank=rank,
        date=d_fit,
        is1600=is1600[finite],
        sess_dates=sess_dates,
        start=start,
        stop=stop,
    )


def _wls_from_sums(
    w: np.ndarray, u: np.ndarray, y: np.ndarray, v_n: float
) -> tuple[float, float, float, float] | None:
    """Weighted line fit; returns (a, b, s2_flat, s2_perrow_base).

    s2_flat divides the weighted residual sum of squares by the weight sum (the
    library's ddof-0 convention). s2_perrow_base divides it by the sum of the
    decay weights only, so that under Var(e_row) = s2_perrow_base * g_row^p it is
    the estimate of the common factor.
    """
    if not np.isfinite(w).all():
        return None
    w_n = float(w.sum())
    wx = float((w * u).sum())
    wxx = float((w * u * u).sum())
    wy = float((w * y).sum())
    wxy = float((w * u * y).sum())
    wyy = float((w * y * y).sum())
    den = w_n * wxx - wx * wx
    if not (den > 0):
        return None
    b = (w_n * wxy - wx * wy) / den
    a = (wy - b * wx) / w_n
    if not (np.isfinite(a) and np.isfinite(b)):
        return None
    num = wyy - a * wy - b * wxy
    return a, b, num / w_n, num / v_n


def _weighted_q10(u: np.ndarray, v: np.ndarray, order: np.ndarray) -> float:
    """Decay-weighted 10th percentile of u (inverse-CDF, no interpolation)."""
    uu = u[order]
    cw = np.cumsum(v[order])
    total = float(cw[-1])
    if not (total > 0):
        return float("nan")
    j = int(np.searchsorted(cw, 0.10 * total, side="left"))
    j = min(j, len(uu) - 1)
    return float(uu[j])


def _q10_floor(q10: float, u: np.ndarray) -> float:
    if q10 > 0:
        return q10
    pos = u[u > 0]
    if len(pos) == 0:
        return float("nan")
    return float(pos.min())


def fit_flat(pan: Panel, w_sess: int, need: np.ndarray) -> dict[int, dict]:
    """Per-session coefficients on a flat trailing window of w_sess sessions."""
    out: dict[int, dict] = {}
    for r in need:
        lo = int(pan.start[max(0, r - w_sess)])
        hi = int(pan.start[r])
        if hi - lo < MIN_ROWS:
            continue
        u_sl, y_sl = pan.u[lo:hi], pan.y[lo:hi]
        q10 = _q10_floor(float(np.quantile(u_sl, 0.10)), u_sl)
        if not np.isfinite(q10):
            continue
        g = np.maximum(u_sl, q10)
        n_rows = float(hi - lo)
        rec: dict = {"q10": q10}
        for p in EXPONENTS:
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                w = np.ones_like(g) if p == 0 else 1.0 / g**p
                fit = _wls_from_sums(w, u_sl, y_sl, n_rows)
            if fit is None:
                continue
            rec[p] = fit
        out[int(r)] = rec
    return out


def fit_ewma(pan: Panel, halflife: int, need: np.ndarray) -> dict[int, dict]:
    """Per-session coefficients with exponential session decay.

    Weights on a session of rank i, seen from session rank r, are
    lam^(r-1-i) with lam = 0.5^(1/halflife) — the library's EWMA convention.
    Every quantity used here (a, b, both variance terms, the Kish effective
    sample size, the weighted 10th percentile) is invariant to a common
    rescaling of those weights, so one reference array lam^(R0-i) serves every
    session and no per-session decay pass is needed.
    """
    lam = 0.5 ** (1.0 / float(halflife))
    r0 = int(need.max())
    dec = lam ** (r0 - pan.rank).astype(float)
    order_all = np.argsort(pan.u, kind="stable")
    out: dict[int, dict] = {}
    for r in need:
        hi = int(pan.start[r])
        if hi < MIN_ROWS:
            continue
        v = dec[:hi]
        v_n = float(v.sum())
        v2 = float((v * v).sum())
        if not (v_n > 0 and v2 > 0):
            continue
        if v_n * v_n / v2 < MIN_NEFF:
            continue
        u_sl, y_sl = pan.u[:hi], pan.y[:hi]
        ordr = order_all[order_all < hi]
        q10 = _q10_floor(_weighted_q10(u_sl, v, ordr), u_sl)
        if not np.isfinite(q10):
            continue
        g = np.maximum(u_sl, q10)
        rec: dict = {"q10": q10}
        for p in EXPONENTS:
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                w = v if p == 0 else v / g**p
                fit = _wls_from_sums(w, u_sl, y_sl, v_n)
            if fit is None:
                continue
            rec[p] = fit
        out[int(r)] = rec
    return out


def apply_cells(pan: Panel, coefs: dict[int, dict], rows: np.ndarray) -> dict:
    """rv_hat for every (p, variant) on the given row indices."""
    ranks = pan.rank[rows]
    u = pan.u[rows]
    b = pan.b[rows]
    n = len(rows)
    res: dict[tuple[int, str], np.ndarray] = {}
    mmap: dict[int, np.ndarray] = {}
    for p in EXPONENTS:
        for v in VARIANTS:
            res[(p, v)] = np.full(n, np.nan)
        mmap[p] = np.full(n, np.nan)
    a_arr = {p: np.full(n, np.nan) for p in EXPONENTS}
    b_arr = {p: np.full(n, np.nan) for p in EXPONENTS}
    s2f = {p: np.full(n, np.nan) for p in EXPONENTS}
    s2w = {p: np.full(n, np.nan) for p in EXPONENTS}
    q10 = np.full(n, np.nan)
    for i, r in enumerate(ranks):
        rec = coefs.get(int(r))
        if rec is None:
            continue
        q10[i] = rec["q10"]
        for p in EXPONENTS:
            f = rec.get(p)
            if f is None:
                continue
            a_arr[p][i], b_arr[p][i], s2f[p][i], s2w[p][i] = f
    for p in EXPONENTS:
        m = a_arr[p] + b_arr[p] * u
        mmap[p] = m
        g = np.maximum(u, q10)
        res[(p, "FLAT")] = (m**2 + s2f[p]) * b
        res[(p, "PER-ROW")] = (m**2 + s2w[p] * g**p) * b
    return {"rv_hat": res, "m": mmap}


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def qlike(rv_hat: np.ndarray, rv: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        z = rv / rv_hat
        return z - np.log(z) - 1.0


def sharpe_of(r: np.ndarray) -> float:
    sd = float(np.std(r, ddof=1))
    if not (sd > 0):
        return float("nan")
    return float(np.mean(r)) / sd * np.sqrt(PERIODS_PER_YEAR)


def boot_sharpe_matrix(mat: np.ndarray, idx: np.ndarray, chunk: int = 100):
    """Sharpe of every column of mat under every resampling row of idx."""
    b_draws = idx.shape[0]
    out = np.empty((b_draws, mat.shape[1]))
    for s in range(0, b_draws, chunk):
        e = min(s + chunk, b_draws)
        x = mat[idx[s:e]]  # (chunk, n, ncell)
        mu = x.mean(axis=1)
        sd = x.std(axis=1, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[s:e] = mu / sd * np.sqrt(PERIODS_PER_YEAR)
    return out


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    paths = asl.yhat_paths(REPO)
    print("=" * 78)
    print("RECALIBRATION GRID -- 0DTE 15:30 straddle deck")
    print("=" * 78)
    print(f"repo            {REPO}")
    print(f"outputs         {OUT}")
    print(f"scored period   {SCORE_START.date()} .. {SCORE_END.date()}")
    print(
        f"grid            {len(WINDOW_LEVELS)} windows x {len(EXPONENTS)} exponents"
        f" x {len(VARIANTS)} variance terms = {len(CELLS)} cells"
    )
    print(f"current cell    {CURRENT}")
    print(
        f"from the library  WINDOW_DAYS={WINDOW_DAYS} MZ_START_DAY={MZ_START_DAY}"
        f" FIT_MASK_MINUTES={FIT_MASK_MINUTES}"
        f" EARLY_CLOSE_DATES={len(EARLY_CLOSE_DATES)}"
    )
    for t in TAGS:
        print(f"  {t:9s} {YHAT_LABEL[t]:44s} {paths[t].name}")
    print()

    # ---------------- per tag: fit every cell -------------------------------
    store: dict[str, dict] = {}
    for tag in TAGS:
        ts = time.time()
        pan = load_panel(paths[tag])
        in_score = (pan.sess_dates >= SCORE_START) & (pan.sess_dates <= SCORE_END)
        need = np.flatnonzero(
            in_score & (np.arange(len(pan.sess_dates)) >= MZ_START_DAY)
        )
        rows = np.flatnonzero(np.isin(pan.rank, need))
        cells: dict[str, np.ndarray] = {}
        mvals: dict[str, np.ndarray] = {}
        for wname, kind, size in WINDOW_LEVELS:
            coefs = (
                fit_flat(pan, size, need)
                if kind == "flat"
                else fit_ewma(pan, size, need)
            )
            got = apply_cells(pan, coefs, rows)
            for p in EXPONENTS:
                mvals[f"{wname}|p{p}"] = got["m"][p]
                for v in VARIANTS:
                    cells[cell_id(wname, p, v)] = got["rv_hat"][(p, v)]
        store[tag] = {
            "pan": pan,
            "rows": rows,
            "cells": cells,
            "m": mvals,
            "rv": pan.rv[rows],
            "y": pan.y[rows],
            "date": pan.date[rows],
            "is1600": pan.is1600[rows],
        }
        print(
            f"  fitted {tag:8s} sessions={len(need):5d} rows={len(rows):6d}"
            f"  [{time.time() - ts:5.1f}s]"
        )
    print()

    # ---------------- GATE 1 ------------------------------------------------
    print("-" * 78)
    print("GATE 1  reproduce the deck's rv_hat at the 16:00 stamp (rel err < 1e-9)")
    print("-" * 78)
    gate_rows = []
    for tag in TAGS:
        st = store[tag]
        deck = pd.read_parquet(DECK / f"daily_{tag}.parquet")
        mine = pd.Series(
            st["cells"][CURRENT][st["is1600"]],
            index=pd.DatetimeIndex(st["date"][st["is1600"]]),
        )
        mine = mine[~mine.index.duplicated()]
        al = mine.reindex(deck.index)
        rel = np.abs(al.to_numpy() / deck["rv_hat"].to_numpy() - 1.0)
        mx = float(np.nanmax(rel))
        n_missing = int(np.isnan(al.to_numpy()).sum())
        ok = bool((mx < 1e-9) and n_missing == 0 and len(deck) == 866)
        gate_rows.append(
            {
                "tag": tag,
                "n_deck_days": len(deck),
                "n_matched": len(deck) - n_missing,
                "max_rel_err": mx,
                "pass": ok,
            }
        )
        print(
            f"  {tag:8s} n={len(deck)}  matched={len(deck) - n_missing}"
            f"  max_rel_err={mx:.3e}  {'PASS' if ok else 'FAIL'}"
        )
    gate = pd.DataFrame(gate_rows)
    gate.to_csv(OUT / "gate1.csv", index=False)
    if not bool(gate["pass"].all()):
        raise SystemExit("GATE 1 FAILED -- nothing further is reported.")
    print(f"  GATE 1: PASS on all {len(TAGS)} tags.")
    print()

    # ---------------- valid row sets ---------------------------------------
    print("-" * 78)
    print("SCORED ROW SETS")
    print("-" * 78)
    for tag in TAGS:
        st = store[tag]
        good = np.isfinite(st["rv"]) & (st["rv"] > 0)
        for c in CELLS:
            good &= np.isfinite(st["cells"][c]) & (st["cells"][c] > 0)
        st["good"] = good
        deck = pd.read_parquet(DECK / f"daily_{tag}.parquet")
        st["deck"] = deck
        d1600 = pd.DatetimeIndex(st["date"][st["is1600"]])
        pos_map = {d: i for i, d in enumerate(d1600)}
        st["traded_pos"] = np.array([pos_map[d] for d in deck.index])
        idx1600 = np.flatnonzero(st["is1600"])
        st["traded_row"] = idx1600[st["traded_pos"]]
        print(
            f"  {tag:8s} all fit-mask rows in period={len(st['rv']):6d}"
            f"  usable on every cell={int(good.sum()):6d}"
            f"  traded 16:00 rows={len(deck)}"
        )
    n_traded = len(store[TAGS[0]]["traded_row"])
    dup = 0.0
    for tag in TAGS:
        for w, _k, _s in WINDOW_LEVELS:
            a = store[tag]["cells"][cell_id(w, 0, "FLAT")]
            b = store[tag]["cells"][cell_id(w, 0, "PER-ROW")]
            dup = max(dup, float(np.nanmax(np.abs(a - b))))
    n_distinct = len(CELLS) - len(WINDOW_LEVELS)
    print(
        f"  p = 0 makes the two variance terms identical: max |FLAT - PER-ROW|"
        f" over all tags and windows = {dup:.3e}"
    )
    print(
        f"  so the {len(CELLS)} cells are {n_distinct} distinct maps"
        f" ({len(WINDOW_LEVELS)} duplicated pairs)"
    )
    print()

    # ---------------- 1. forecast loss --------------------------------------
    print("-" * 78)
    print("1. FORECAST LOSS (QLIKE)")
    print("-" * 78)
    detail = []
    q_all = pd.DataFrame(index=CELLS, columns=TAGS, dtype=float)
    q_trd = pd.DataFrame(index=CELLS, columns=TAGS, dtype=float)
    for tag in TAGS:
        st = store[tag]
        good = st["good"]
        rv_a = st["rv"][good]
        rv_t = st["rv"][st["traded_row"]]
        y_a = st["y"][good]
        loss_a = {c: qlike(st["cells"][c][good], rv_a) for c in CELLS}
        loss_t = {c: qlike(st["cells"][c][st["traded_row"]], rv_t) for c in CELLS}
        for c in CELLS:
            wname, ps, variant = c.split("|")
            p = int(ps[1:])
            m = st["m"][f"{wname}|{ps}"][good]
            ok = np.isfinite(m) & np.isfinite(y_a)
            slope, icept = np.polyfit(m[ok], y_a[ok], 1)
            da = loss_a[c] - loss_a[CURRENT]
            dt = loss_t[c] - loss_t[CURRENT]
            t_a, lag_a = asl.newey_west_t(da) if c != CURRENT else (np.nan, 0)
            t_t, lag_t = asl.newey_west_t(dt) if c != CURRENT else (np.nan, 0)
            q_all.loc[c, tag] = float(loss_a[c].mean())
            q_trd.loc[c, tag] = float(loss_t[c].mean())
            detail.append(
                {
                    "cell": c,
                    "window": wname,
                    "p": p,
                    "variance": variant,
                    "tag": tag,
                    "qlike_all_rows": float(loss_a[c].mean()),
                    "d_qlike_all_rows": float(da.mean()),
                    "dm_t_all_rows": t_a,
                    "dm_lag_all_rows": lag_a,
                    "qlike_traded": float(loss_t[c].mean()),
                    "d_qlike_traded": float(dt.mean()),
                    "dm_t_traded": t_t,
                    "dm_lag_traded": lag_t,
                    "calib_slope": float(slope),
                    "calib_intercept": float(icept),
                }
            )
    q_all.to_csv(OUT / "qlike_all_rows.csv")
    q_trd.to_csv(OUT / "qlike_traded_days.csv")
    print(
        f"QLIKE, all fit-mask rows in {SCORE_START.date()}..{SCORE_END.date()}"
        f" ({len(CELLS)} cells x {len(TAGS)} tags)"
    )
    print(q_all.to_string(float_format=lambda v: f"{v:.5f}"))
    print()
    print(
        f"QLIKE, the {n_traded} traded 16:00 rows ({len(CELLS)} cells x {len(TAGS)} tags)"
    )
    print(q_trd.to_string(float_format=lambda v: f"{v:.5f}"))
    print()

    det = pd.DataFrame(detail)

    dm_a = det.pivot(index="cell", columns="tag", values="dm_t_all_rows").reindex(
        index=CELLS, columns=TAGS
    )
    dm_t = det.pivot(index="cell", columns="tag", values="dm_t_traded").reindex(
        index=CELLS, columns=TAGS
    )
    dm_a.to_csv(OUT / "dm_t_all_rows.csv")
    dm_t.to_csv(OUT / "dm_t_traded_days.csv")
    n_rows_a = int(store[TAGS[0]]["good"].sum())
    n_rows_t = len(store[TAGS[0]]["traded_row"])
    lag_a = asl.newey_west_lag(n_rows_a)
    lag_t = asl.newey_west_lag(n_rows_t)
    assert lag_a == int(det["dm_lag_all_rows"].max())
    assert lag_t == int(det["dm_lag_traded"].max())
    print("Diebold-Mariano t of (cell QLIKE - current-cell QLIKE), all fit-mask rows")
    print(
        f"  negative = the cell loses less than the current map."
        f"  Autocorrelation-robust, Bartlett lag {lag_a}."
    )
    print(dm_a.to_string(float_format=lambda v: f"{v:.2f}"))
    print()
    print("Diebold-Mariano t of (cell QLIKE - current-cell QLIKE), 866 traded days")
    print(f"  Autocorrelation-robust, Bartlett lag {lag_t}.")
    print(dm_t.to_string(float_format=lambda v: f"{v:.2f}"))
    print()
    sig = det[det["cell"] != CURRENT]
    print(
        f"  cells x tags with |t| > 2 on all fit-mask rows: "
        f"{int((sig['dm_t_all_rows'].abs() > 2).sum())} of {len(sig)}"
        f"  (of which better: {int((sig['dm_t_all_rows'] < -2).sum())},"
        f" worse: {int((sig['dm_t_all_rows'] > 2).sum())})"
    )
    print(
        f"  cells x tags with |t| > 2 on the 866 traded days: "
        f"{int((sig['dm_t_traded'].abs() > 2).sum())} of {len(sig)}"
        f"  (of which better: {int((sig['dm_t_traded'] < -2).sum())},"
        f" worse: {int((sig['dm_t_traded'] > 2).sum())})"
    )
    p0 = sig[sig["p"] == 0]
    print(
        f"  the {len(p0)} p = 0 comparisons: DM t on all rows from"
        f" {p0['dm_t_all_rows'].min():.2f} to {p0['dm_t_all_rows'].max():.2f};"
        f" on traded days from {p0['dm_t_traded'].min():.2f} to"
        f" {p0['dm_t_traded'].max():.2f}"
    )
    print(
        f"  of the {int((sig['dm_t_all_rows'] > 2).sum())} comparisons worse at"
        f" |t| > 2 on all rows, {int((p0['dm_t_all_rows'] > 2).sum())} are p = 0 cells"
    )
    print(
        f"  largest improvement over the current map:"
        f" t = {sig['dm_t_all_rows'].min():.2f} on all rows,"
        f" t = {sig['dm_t_traded'].min():.2f} on traded days"
    )
    print()

    cal = det[det["variance"] == "FLAT"].copy()
    cal["map"] = cal["window"] + "|p" + cal["p"].astype(str)
    map_order = [f"{w[0]}|p{p}" for w in WINDOW_LEVELS for p in EXPONENTS]
    cs = cal.pivot(index="map", columns="tag", values="calib_slope").reindex(
        index=map_order, columns=TAGS
    )
    ci = cal.pivot(index="map", columns="tag", values="calib_intercept").reindex(
        index=map_order, columns=TAGS
    )
    cs.to_csv(OUT / "calibration_slope.csv")
    ci.to_csv(OUT / "calibration_intercept.csv")
    print("Mean-map calibration: OLS of y on m over the scored fit-mask rows.")
    print(
        "  slope (1 = calibrated); the mean map m does not depend on the"
        " variance term, so there are 18 distinct maps"
    )
    print(cs.to_string(float_format=lambda v: f"{v:.4f}"))
    print("  intercept (0 = calibrated)")
    print(ci.to_string(float_format=lambda v: f"{v:.5f}"))
    print(
        f"  slope range {cs.to_numpy().min():.4f} .. {cs.to_numpy().max():.4f};"
        f"  intercept range {ci.to_numpy().min():.5f} .. {ci.to_numpy().max():.5f}"
    )
    print()

    # ---------------- 2. the trade ------------------------------------------
    print("-" * 78)
    print("2. THE TRADE (sign(s) on the 866 days)")
    print("-" * 78)
    sh = pd.DataFrame(index=CELLS, columns=TAGS, dtype=float)
    trade_rows = []
    qsign: dict[str, dict[str, np.ndarray]] = {}
    for tag in TAGS:
        st = store[tag]
        deck = st["deck"]
        iv = deck["iv_var"].to_numpy(float)
        R = deck["R"].to_numpy(float)
        qsign[tag] = {
            c: np.where(st["cells"][c][st["traded_row"]] <= iv, -1.0, 1.0)
            for c in CELLS
        }
        for c in CELLS:
            q = qsign[tag][c]
            r = q * R
            n = len(r)
            mu = float(np.mean(r))
            sd = float(np.std(r, ddof=1))
            trade_rows.append(
                {
                    "cell": c,
                    "tag": tag,
                    "n_days": n,
                    "mean_R": mu,
                    "t": np.sqrt(n) * mu / sd,
                    "sharpe": sharpe_of(r),
                    "n_buy": int((q > 0).sum()),
                    "n_sign_diff_vs_current": int((q != qsign[tag][CURRENT]).sum()),
                }
            )
            sh.loc[c, tag] = sharpe_of(r)
        rs = -deck["R"].to_numpy(float)
        print(
            f"  {tag:8s} always short Sharpe = {sharpe_of(rs):.4f}"
            f"   deck sign(s) Sharpe (current cell) = {sh.loc[CURRENT, tag]:.4f}"
        )
    sh.to_csv(OUT / "sharpe.csv")
    trade = pd.DataFrame(trade_rows)
    print()
    print(f"sign(s) Sharpe, {n_traded} days ({len(CELLS)} cells x {len(TAGS)} tags)")
    print(sh.to_string(float_format=lambda v: f"{v:.4f}"))
    print()

    nb = trade.pivot(index="cell", columns="tag", values="n_buy").reindex(
        index=CELLS, columns=TAGS
    )
    nd = trade.pivot(
        index="cell", columns="tag", values="n_sign_diff_vs_current"
    ).reindex(index=CELLS, columns=TAGS)
    nb.to_csv(OUT / "n_buy.csv")
    nd.to_csv(OUT / "n_sign_diff_vs_current.csv")
    print(
        f"days the rule buys the straddle, out of {n_traded}"
        f" ({len(CELLS)} cells x {len(TAGS)} tags)"
    )
    print(nb.to_string())
    print()
    print(
        f"days whose sign differs from the current map, out of {n_traded}"
        f" ({len(CELLS)} cells x {len(TAGS)} tags)"
    )
    print(nd.to_string())
    print()

    full = det.merge(trade, on=["cell", "tag"])
    full.to_csv(OUT / "cells_detail.csv", index=False)
    print(f"  wrote {OUT / 'cells_detail.csv'}")
    print()
    b2d = full[full["tag"] == "blk2"].set_index("cell").reindex(CELLS)
    print("block-diagonal ridge, per cell")
    print(
        b2d[
            [
                "qlike_all_rows",
                "dm_t_all_rows",
                "qlike_traded",
                "dm_t_traded",
                "calib_slope",
                "calib_intercept",
                "mean_R",
                "t",
                "sharpe",
                "n_buy",
                "n_sign_diff_vs_current",
            ]
        ].to_string(float_format=lambda v: f"{v:.4f}")
    )
    print()

    # ---------------- 3. multiplicity ---------------------------------------
    print("-" * 78)
    print("3. MULTIPLICITY (block-diagonal ridge only)")
    print("-" * 78)
    tag = "blk2"
    st = store[tag]
    R = st["deck"]["R"].to_numpy(float)
    mat = np.column_stack([qsign[tag][c] * R for c in CELLS])
    n_days = mat.shape[0]
    rng = np.random.default_rng(BOOT_SEED)
    idx = asl.circular_block_bootstrap_idx(rng, n_days, BOOT_BLOCK, BOOT_B)
    bs = boot_sharpe_matrix(mat, idx)
    cur_col = CELLS.index(CURRENT)
    cur_boot = bs[:, cur_col]
    obs = sh[tag].to_numpy(float)
    obs_cur = float(obs[cur_col])
    obs_max = float(np.nanmax(obs))
    obs_argmax = CELLS[int(np.nanargmax(obs))]
    obs_mean = float(np.nanmean(obs))
    bmax = np.nanmax(bs, axis=1)
    bmean = np.nanmean(bs, axis=1)
    prem = bmax - bmean
    pct_obs_max = float((bmax <= obs_max).mean())
    print(
        f"  circular block bootstrap: block={BOOT_BLOCK} days, B={BOOT_B},"
        f" rng seed {BOOT_SEED}, n={n_days}"
    )
    print(f"  current cell Sharpe                 {obs_cur:.4f}")
    print(f"  current cell bootstrap SE           {float(cur_boot.std(ddof=1)):.4f}")
    print(
        f"  current cell bootstrap 2.5/97.5%    "
        f"{float(np.percentile(cur_boot, 2.5)):.4f} .. "
        f"{float(np.percentile(cur_boot, 97.5)):.4f}"
    )
    print(f"  observed max over the 36 cells      {obs_max:.4f}  ({obs_argmax})")
    print(f"  observed mean over the 36 cells     {obs_mean:.4f}")
    print(f"  observed selection premium max-mean {obs_max - obs_mean:.4f}")
    print(
        f"  bootstrap max-Sharpe distribution   mean {float(bmax.mean()):.4f}"
        f"  sd {float(bmax.std(ddof=1)):.4f}"
    )
    print(
        "    quantiles 5/25/50/75/95%          "
        + "  ".join(f"{float(np.percentile(bmax, q)):.4f}" for q in (5, 25, 50, 75, 95))
    )
    print(f"  fraction of draws with max <= obs   {pct_obs_max:.4f}")
    print(
        f"  bootstrap selection premium max-mean  mean {float(prem.mean()):.4f}"
        f"  sd {float(prem.std(ddof=1)):.4f}"
        f"  95% {float(np.percentile(prem, 95)):.4f}"
    )
    boot_out = pd.DataFrame(
        {
            "stat": [
                "obs_sharpe_current",
                "boot_se_current",
                "boot_lo95_current",
                "boot_hi95_current",
                "obs_max",
                "obs_mean",
                "obs_premium",
                "boot_max_mean",
                "boot_max_sd",
                "boot_max_q05",
                "boot_max_q25",
                "boot_max_q50",
                "boot_max_q75",
                "boot_max_q95",
                "frac_draws_max_le_obs_max",
                "boot_premium_mean",
                "boot_premium_sd",
                "boot_premium_q95",
            ],
            "value": [
                obs_cur,
                float(cur_boot.std(ddof=1)),
                float(np.percentile(cur_boot, 2.5)),
                float(np.percentile(cur_boot, 97.5)),
                obs_max,
                obs_mean,
                obs_max - obs_mean,
                float(bmax.mean()),
                float(bmax.std(ddof=1)),
                *[float(np.percentile(bmax, q)) for q in (5, 25, 50, 75, 95)],
                pct_obs_max,
                float(prem.mean()),
                float(prem.std(ddof=1)),
                float(np.percentile(prem, 95)),
            ],
        }
    )
    boot_out.to_csv(OUT / "bootstrap_blk2.csv", index=False)

    # sign agreement structure
    qm = np.column_stack([qsign[tag][c] for c in CELLS])
    agree_cur = (qm == qm[:, [cur_col]]).mean(axis=0)
    n_cells_95 = int((agree_cur > 0.95).sum())
    pair_ct = 0
    pair_hi = 0
    for i in range(len(CELLS)):
        for j in range(i + 1, len(CELLS)):
            pair_ct += 1
            if (qm[:, i] == qm[:, j]).mean() > 0.95:
                pair_hi += 1
    print(
        f"  cells agreeing with the current cell on >95% of days: "
        f"{n_cells_95} of {len(CELLS)}"
    )
    print(f"  cell pairs agreeing on >95% of days: {pair_hi} of {pair_ct}")
    print(
        f"  min agreement with the current cell: {float(agree_cur.min()):.4f}"
        f"  ({CELLS[int(np.argmin(agree_cur))]})"
    )
    pd.DataFrame({"cell": CELLS, "agree_with_current": agree_cur}).to_csv(
        OUT / "sign_agreement_blk2.csv", index=False
    )
    print()

    # ---------------- 4. sensitivity ----------------------------------------
    print("-" * 78)
    print("4. SENSITIVITY (block-diagonal ridge)")
    print("-" * 78)
    b2 = full[full["tag"] == tag].copy()
    sens_rows = []
    for dim, col in (
        ("window", "window"),
        ("weight exponent p", "p"),
        ("variance term", "variance"),
    ):
        for lev, grp in b2.groupby(col, sort=False):
            sens_rows.append(
                {
                    "dimension": dim,
                    "level": lev,
                    "n_cells": len(grp),
                    "sharpe_min": grp["sharpe"].min(),
                    "sharpe_mean": grp["sharpe"].mean(),
                    "sharpe_max": grp["sharpe"].max(),
                    "qlike_traded_min": grp["qlike_traded"].min(),
                    "qlike_traded_mean": grp["qlike_traded"].mean(),
                    "qlike_traded_max": grp["qlike_traded"].max(),
                    "qlike_all_min": grp["qlike_all_rows"].min(),
                    "qlike_all_mean": grp["qlike_all_rows"].mean(),
                    "qlike_all_max": grp["qlike_all_rows"].max(),
                }
            )
    sens = pd.DataFrame(sens_rows)
    sens.to_csv(OUT / "sensitivity_blk2.csv", index=False)
    print(sens.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    for dim, col in (
        ("window", "window"),
        ("weight exponent p", "p"),
        ("variance term", "variance"),
    ):
        s = sens[sens["dimension"] == dim]
        print(
            f"  {dim:18s} spread of level means:"
            f"  Sharpe {float(s['sharpe_mean'].max() - s['sharpe_mean'].min()):.4f}"
            f"   QLIKE(traded) {float(s['qlike_traded_mean'].max() - s['qlike_traded_mean'].min()):.5f}"
            f"   QLIKE(all) {float(s['qlike_all_mean'].max() - s['qlike_all_mean'].min()):.5f}"
        )
    print()

    # ---------------- ranks of the current cell ------------------------------
    print("-" * 78)
    print("RANK OF THE CURRENT CELL AMONG THE 36")
    print("-" * 78)
    rk_rows = []
    for t in TAGS:
        sub = full[full["tag"] == t]
        r_q_all = int(
            sub["qlike_all_rows"].rank(method="min").loc[sub["cell"] == CURRENT].iloc[0]
        )
        r_q_trd = int(
            sub["qlike_traded"].rank(method="min").loc[sub["cell"] == CURRENT].iloc[0]
        )
        r_sh = int(
            sub["sharpe"]
            .rank(method="min", ascending=False)
            .loc[sub["cell"] == CURRENT]
            .iloc[0]
        )
        rk_rows.append(
            {
                "tag": t,
                "rank_qlike_all_rows": r_q_all,
                "rank_qlike_traded": r_q_trd,
                "rank_sharpe": r_sh,
                "sharpe_min": sub["sharpe"].min(),
                "sharpe_max": sub["sharpe"].max(),
                "sharpe_current": float(sh.loc[CURRENT, t]),
                "qlike_all_min": sub["qlike_all_rows"].min(),
                "qlike_all_max": sub["qlike_all_rows"].max(),
                "qlike_all_current": float(q_all.loc[CURRENT, t]),
            }
        )
    rk = pd.DataFrame(rk_rows)
    rk.to_csv(OUT / "current_cell_rank.csv", index=False)
    print(rk.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("  (rank 1 = lowest QLIKE / highest Sharpe; ties take the lower rank)")
    print()

    # ---------------- figure -------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    marks = {"flat": "o", "ewma": "s"}
    cols = {0: "#4477aa", 1: "#228833", 2: "#cc6677"}
    for _, row in b2.iterrows():
        kind = "flat" if row["window"].startswith("flat") else "ewma"
        ax.scatter(
            row["qlike_traded"],
            row["sharpe"],
            marker=marks[kind],
            facecolor="none" if row["variance"] == "PER-ROW" else cols[row["p"]],
            edgecolor=cols[row["p"]],
            s=46,
            linewidths=1.4,
            zorder=3,
        )
    cur = b2[b2["cell"] == CURRENT].iloc[0]
    ax.scatter(
        [cur["qlike_traded"]],
        [cur["sharpe"]],
        marker="*",
        s=340,
        facecolor="#000000",
        edgecolor="#000000",
        zorder=4,
    )
    ax.annotate(
        "current map\n(flat 250, p = 2, FLAT)",
        (cur["qlike_traded"], cur["sharpe"]),
        textcoords="offset points",
        xytext=(14, 26),
        fontsize=9,
    )
    ax.set_xlabel(f"QLIKE on the {n_traded} traded 16:00 rows (lower is better)")
    ax.set_ylabel(f"sign(s) Sharpe, {n_traded} days")
    ax.set_title(f"Recalibration grid, {YHAT_LABEL[tag]}: {len(CELLS)} cells")
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [], [], ls="", marker="o", mfc="none", mec="#444444", label="flat window"
        ),
        Line2D(
            [], [], ls="", marker="s", mfc="none", mec="#444444", label="EWMA window"
        ),
        Line2D([], [], ls="", marker="o", color=cols[0], label="p = 0"),
        Line2D([], [], ls="", marker="o", color=cols[1], label="p = 1"),
        Line2D([], [], ls="", marker="o", color=cols[2], label="p = 2"),
        Line2D(
            [],
            [],
            ls="",
            marker="o",
            mfc="none",
            mec="#444444",
            label="open face = PER-ROW variance",
        ),
        Line2D([], [], ls="", marker="*", color="k", label="current map"),
        Line2D([], [], ls="", marker="", label="(p = 0: the two variance"),
        Line2D([], [], ls="", marker="", label=" terms coincide, so those"),
        Line2D([], [], ls="", marker="", label=" cells plot on top of each other)"),
    ]
    ax.legend(
        handles=handles,
        fontsize=8,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "sharpe_vs_qlike_blk2.png", dpi=160)
    plt.close(fig)
    print(f"  wrote {OUT / 'sharpe_vs_qlike_blk2.png'}")
    print()
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
