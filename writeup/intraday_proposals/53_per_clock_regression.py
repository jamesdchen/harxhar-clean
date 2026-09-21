"""53 - does regressing by time of day improve the 30-minute variance forecast?

Every forecast the deck trades (block ridge, baseline OLS, the trees) is fitted
with ONE coefficient vector pooled over all 48 bars of the day; time of day
enters only through the multiplicative diurnal baseline and a recalibration
pooled over the session.  This proposal asks what that pooling costs at the
twelve traded clocks (forecasts issued 10:00 .. 15:30 ET for the next
30-minute bar).

PART A - first stage, 1998-2024, forecast-free of the production pipeline.
A heterogeneous-autoregressive proxy on the same 30-minute realized variance
the production panel uses (``data/core_stats.parquet``, ``sumret2``; gated
equal to the panel's ``rv_raw``).  Variance is divided by a causal per-clock
baseline (expanding mean over strictly prior sessions, minimum 63) and the
regressions are log-linear, refitted every session on strictly prior sessions.

  features   log trailing means of normalized variance over the last 1, 2, 4,
             8, 13 (one session), 65 and 286 regular-hours bars, and the log
             normalized overnight variance (previous 16:00 to 09:30).
  P          pooled over the twelve clocks: one coefficient vector.
  D          P plus a clock intercept.
  C          a separate regression per clock.
  S(lam)     per-clock coefficients shrunk toward P: beta_c = beta_P + delta_c
             with penalty lam |delta_c|^2; lam = 0 is C, lam = inf is P.
  Ssel       S with lam picked causally per clock (best trailing-250-session
             QLIKE, lagged one session, minimum 63, P before that: the null is
             in the grid).
  CP(lam)    "times of day as regressors": a per-clock ridge on each of TODAY'S
             earlier regular-hours bars separately, the overnight variance,
             the previous 1/5/22-session means, and the same clock's own bar
             yesterday and over the last five sessions.
  CPsel      CP with lam picked causally the same way, P in the grid.

Three estimation windows: the production's 250 sessions, 1000 sessions, and
expanding.  Every model gets the same back-transform: exp(prediction) times a
causal smearing factor (mean of exp of its own out-of-sample log errors over
the previous 250 sessions, minimum 63), pooled over clocks, and for C and Ssel
also per clock.  Loss is Patton's QLIKE on the variance level.

PART B - second stage, on the PRODUCTION forecasts, deck days only.  For the
ridge, the baseline OLS, XGBoost and LightGBM: a causal log Mincer-Zarnowitz
map log y = a + b log f fitted pooled over the clocks, with a clock intercept,
and per clock (pooled and per-clock smear), expanding and 250-session windows.

PART C - what it does to the trade's inputs on the deck days: the hit rate of
sign(forecast - implied slice) against sign(realized - implied slice) at every
clock, and the mid Sharpe of sign(s) at 15:30.

A difference "improves" only if the circular-block-bootstrap interval (block
21, B = 2000, seed 0) of the paired daily QLIKE difference against the pooled
model excludes zero on the negative side.

Gates:
  GATE R   this script's realized variance equals the production panel's on
           every deck bar.
  GATE Q   the ridge's QLIKE at 15:30 on the deck days is proposal 35's
           0.1100 and its 15:30 sign(s) mid Sharpe is the deck's 1.338322.

Run:  python writeup/intraday_proposals/53_per_clock_regression.py
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "notebooks", ROOT / "writeup"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import atm_straddle_lib as asl  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = ROOT / "results" / "atm_straddle_0dte_1530" / "proposals" / "53"
CORE = ROOT / "data" / "core_stats.parquet"
REF35 = (
    ROOT
    / "results"
    / "atm_straddle_intraday_holdclose"
    / "proposals"
    / "35"
    / "a_qlike_by_clock.csv"
)

#: bar-END labels of the 13 regular-hours bars; bar k is issued at label k-1
RTH_LABELS: tuple[str, ...] = tuple(
    f"{h:02d}:{m:02d}" for h in range(10, 17) for m in (0, 30) if (h, m) <= (16, 0)
)
ISSUE: tuple[str, ...] = RTH_LABELS[:-1]  # 10:00 .. 15:30, target bars k = 1..12
N_BAR = len(RTH_LABELS)
N_K = len(ISSUE)
PANEL_START = "1998-01-05"  # returns are empty before this date
DECK_START, DECK_END = "2020-01-03", "2024-04-30"

LADDER: tuple[int, ...] = (1, 2, 4, 8, 13, 65, 286)  # regular-hours bars
BASE_MIN = 63  # sessions before a clock's baseline exists
FIT_MIN = 250  # rows per clock before any model predicts (one production window)
WINDOWS: dict[str, int | None] = {"250": 250, "1000": 1000, "expanding": None}
SMEAR_W, SMEAR_MIN = 250, 63
PICK_W, PICK_MIN = 250, 63
LAMS: tuple[float, ...] = (0.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
MZ_MIN = 63
MZ_WINDOWS: dict[str, int | None] = {"250": 250, "expanding": None}
TAGS: tuple[str, ...] = ("blk2", "a0", "xgb", "lgbm")

ANN = float(np.sqrt(asl.PERIODS_PER_YEAR))
BOOT_B, BOOT_BLOCK, BOOT_SEED = 2000, 21, 0
GATE_QLIKE_1530, GATE_QLIKE_TOL = 0.1100, 5e-5
GATE_SHARPE_1530, GATE_SHARPE_TOL = 1.338322, 1e-6
GATE_RV_TOL = 1e-9


def _load_module(path: Path, name: str) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, path
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ helpers --
def wsum(a: np.ndarray, window: int | None) -> np.ndarray:
    """Sum of the per-session contributions over the W sessions strictly before d."""
    c = np.cumsum(a, axis=0)
    z = np.concatenate([np.zeros_like(c[:1]), c], axis=0)
    d = np.arange(a.shape[0])
    lo = np.zeros_like(d) if window is None else np.maximum(d - window, 0)
    return z[d] - z[lo]


def bsolve(a: np.ndarray, b: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """Batched solve on the rows flagged ok; NaN elsewhere."""
    out = np.full(b.shape, np.nan)
    if ok.any():
        try:
            out[ok] = np.linalg.solve(a[ok], b[ok][..., None])[..., 0]
        except np.linalg.LinAlgError:
            out[ok] = (np.linalg.pinv(a[ok]) @ b[ok][..., None])[..., 0]
    return out


def outer_rows(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    xv = np.where(valid[..., None], x, 0.0)
    return np.einsum("...i,...j->...ij", xv, xv)


def smear(err: np.ndarray, per_clock: bool) -> np.ndarray:
    """Causal smearing factor from the model's own out-of-sample log errors."""
    es = pd.DataFrame(np.exp(err))
    if not per_clock:
        es = pd.DataFrame({0: es.mean(axis=1, skipna=True)})
    s = es.rolling(SMEAR_W, min_periods=SMEAR_MIN).mean().shift(1).to_numpy()
    return s if per_clock else np.repeat(s, err.shape[1], axis=1)


def qlike(y: np.ndarray, f: np.ndarray) -> np.ndarray:
    r = y / f
    return r - np.log(r) - 1.0


def causal_pick(losses: list[np.ndarray], null: int) -> np.ndarray:
    """Per (session, clock): the candidate with the best trailing QLIKE, lagged."""
    tr = np.stack(
        [
            pd.DataFrame(v)
            .rolling(PICK_W, min_periods=PICK_MIN)
            .mean()
            .shift(1)
            .to_numpy()
            for v in losses
        ]
    )
    ok = np.isfinite(tr).all(axis=0)
    pick = np.where(ok, np.argmin(np.where(np.isfinite(tr), tr, np.inf), axis=0), null)
    return pick.astype(int)


def take(cands: list[np.ndarray], pick: np.ndarray) -> np.ndarray:
    return np.take_along_axis(np.stack(cands), pick[None, ...], axis=0)[0]


_IDX: dict[int, np.ndarray] = {}


def boot_ci(d: np.ndarray) -> tuple[float, float]:
    v = np.asarray(d, float)
    n = v.size
    if n not in _IDX:
        _IDX[n] = asl.circular_block_bootstrap_idx(
            np.random.default_rng([BOOT_SEED, n]), n, BOOT_BLOCK, BOOT_B
        )
    m = v[_IDX[n]].mean(axis=1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return float(lo), float(hi)


def write(df: pd.DataFrame, name: str, title: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / name, index=False)
    print(f"\nwrote {OUT / name}  ({len(df)} rows) - {title}")


# ---------------------------------------------------------------- the panel --
def build_panel() -> dict[str, Any]:
    c = pd.read_parquet(CORE, columns=["endbartime", "sumret2"])
    c["t"] = pd.to_datetime(c["endbartime"])  # naive ET, bar-END labelled
    c = c[c["t"] >= PANEL_START].sort_values("t").reset_index(drop=True)
    c["date"] = c["t"].dt.normalize()
    c["hhmm"] = c["t"].dt.strftime("%H:%M")
    rth = (
        c[c["hhmm"].isin(RTH_LABELS)]
        .pivot_table(index="date", columns="hhmm", values="sumret2", aggfunc="first")
        .reindex(columns=list(RTH_LABELS))
    )
    # overnight: everything after the previous session's 16:00 up to today's 09:30
    times = c["t"].to_numpy()
    cs = np.concatenate([[0.0], np.nancumsum(c["sumret2"].to_numpy(float))])
    has_close = rth.index[rth["16:00"].notna()]
    prev_close = (
        pd.Series(has_close, index=has_close).shift(1).reindex(rth.index).ffill()
    )
    hi = np.searchsorted(
        times, (rth.index + pd.Timedelta("9h30min")).to_numpy(), "right"
    )
    lo = np.searchsorted(
        times, (prev_close + pd.Timedelta("16h")).to_numpy(), side="right"
    )
    on = pd.Series(cs[hi] - cs[lo], index=rth.index).where(prev_close.notna())
    full = rth.notna().all(axis=1) & (rth > 0).all(axis=1) & (on > 0)
    rth, on = rth[full], on[full]
    print(
        f"sessions with all {N_BAR} regular-hours bars and an overnight leg: "
        f"{len(rth)} ({rth.index.min().date()} .. {rth.index.max().date()}); "
        f"dropped {int((~full).sum())}"
    )
    rv = rth.to_numpy(float)
    base = (
        rth.expanding(min_periods=BASE_MIN).mean().shift(1).to_numpy(float)
    )  # strictly prior sessions
    base_on = on.expanding(min_periods=BASE_MIN).mean().shift(1).to_numpy(float)
    keep = np.isfinite(base).all(axis=1) & np.isfinite(base_on)
    d0 = int(np.argmax(keep))
    assert keep[d0:].all()
    dates = pd.DatetimeIndex(rth.index[d0:])
    rv, base = rv[d0:], base[d0:]
    x = rv / base
    x_on = (on.to_numpy(float) / base_on)[d0:]
    return {"dates": dates, "rv": rv, "base": base, "x": x, "x_on": x_on}


def features(pan: dict[str, Any]) -> dict[str, Any]:
    x, x_on = pan["x"], pan["x_on"]
    n = x.shape[0]
    z = x.reshape(-1)
    cz = np.concatenate([[0.0], np.cumsum(z)])

    def trail(w: int) -> np.ndarray:
        out = np.full(z.size, np.nan)
        i = np.arange(w, z.size)
        out[i] = (cz[i] - cz[i - w]) / w  # the w bars strictly before i
        return out.reshape(n, N_BAR)[:, 1:]

    lad = [np.log(trail(w)) for w in LADDER]
    lad.append(np.repeat(np.log(x_on)[:, None], N_K, axis=1))
    x_lad = np.stack([np.ones((n, N_K)), *lad], axis=-1)  # (n, 12, 9)
    y = np.log(x[:, 1:])
    valid = np.isfinite(x_lad).all(axis=-1) & np.isfinite(y)

    # clock intercepts (clock 1 is the reference)
    dum = np.zeros((n, N_K, N_K - 1))
    for k in range(1, N_K):
        dum[:, k, k - 1] = 1.0
    x_dum = np.concatenate([x_lad, dum], axis=-1)

    # "times of day as regressors", one design per clock
    day = pd.Series(x.mean(axis=1))
    yb = {w: np.log(day.rolling(w).mean().shift(1).to_numpy()) for w in (1, 5, 22)}
    xs = pd.DataFrame(x)
    same1 = np.log(xs.shift(1).to_numpy())
    same5 = np.log(xs.rolling(5).mean().shift(1).to_numpy())
    x_cp = []
    for k in range(1, N_BAR):
        cols = [np.ones(n)]
        cols += [np.log(x[:, j]) for j in range(k)]
        cols += [np.log(x_on), yb[1], yb[5], yb[22], same1[:, k], same5[:, k]]
        x_cp.append(np.stack(cols, axis=-1))
    valid_cp = np.stack(
        [np.isfinite(x_cp[k]).all(axis=-1) for k in range(N_K)], axis=1
    ) & np.isfinite(y)
    return {
        "x_lad": x_lad,
        "x_dum": x_dum,
        "x_cp": x_cp,
        "y": y,
        "valid": valid & valid_cp,  # one row set for every model
    }


# ------------------------------------------------------------- first stage ---
def first_stage(ft: dict[str, Any], window: int | None) -> dict[str, np.ndarray]:
    """Out-of-sample log predictions of every model for one estimation window."""
    x, y, valid = ft["x_lad"], ft["y"], ft["valid"]
    n, _, p = x.shape
    yv = np.where(valid, y, 0.0)
    sa = wsum(outer_rows(x, valid), window)  # (n, 12, p, p)
    sb = wsum(np.where(valid[..., None], x, 0.0) * yv[..., None], window)
    cnt = wsum(valid.astype(float), window)
    ok_k = cnt >= FIT_MIN
    ok = np.asarray(ok_k.all(axis=1), dtype=bool)  # every model starts together
    pred: dict[str, np.ndarray] = {}

    beta_p = bsolve(sa.sum(axis=1), sb.sum(axis=1), ok)  # (n, p)
    pred["P"] = np.einsum("nkp,np->nk", x, beta_p)

    xd = ft["x_dum"]
    sad = wsum(outer_rows(xd, valid).sum(axis=1), window)
    sbd = wsum(
        (np.where(valid[..., None], xd, 0.0) * yv[..., None]).sum(axis=1), window
    )
    pred["D"] = np.einsum("nkp,np->nk", xd, bsolve(sad, sbd, ok))

    resid_b = sb - np.einsum("nkij,nj->nki", sa, np.nan_to_num(beta_p))
    eye = np.eye(p)
    okk = ok[:, None] & ok_k
    for lam in LAMS:
        delta = bsolve(sa + lam * eye, resid_b, okk)
        beta = beta_p[:, None, :] + delta
        name = "C" if lam == 0.0 else f"S{lam:g}"
        pred[name] = np.einsum("nkp,nkp->nk", x, beta)

    for k in range(N_K):
        xk = ft["x_cp"][k]
        pk = xk.shape[1]
        vk = valid[:, k]
        sak = wsum(outer_rows(xk, vk), window)
        sbk = wsum(np.where(vk[:, None], xk, 0.0) * yv[:, k][:, None], window)
        pen = np.eye(pk)
        pen[0, 0] = 0.0  # the intercept is not shrunk
        for lam in LAMS:
            b = bsolve(sak + lam * pen, sbk, okk[:, k])
            pred.setdefault(f"CP{lam:g}", np.full((n, N_K), np.nan))[:, k] = np.einsum(
                "np,np->n", xk, b
            )
    for v in pred.values():
        v[~valid] = np.nan
    return pred


def to_level(
    pred: np.ndarray, y_log: np.ndarray, base: np.ndarray, per_clock: bool = False
) -> np.ndarray:
    return np.exp(pred) * smear(y_log - pred, per_clock) * base


def part_a(pan: dict[str, Any], ft: dict[str, Any]) -> dict[str, Any]:
    rv, base, y = pan["rv"][:, 1:], pan["base"][:, 1:], ft["y"]
    fc: dict[tuple[str, str], np.ndarray] = {}
    picks: list[dict[str, Any]] = []
    for wname, window in WINDOWS.items():
        t0 = time.time()
        pred = first_stage(ft, window)
        lv = {m: to_level(v, y, base) for m, v in pred.items()}
        for m in ("C", "D"):
            lv[f"{m}+clock smear"] = to_level(pred[m], y, base, per_clock=True)
        ls = {m: qlike(rv, f) for m, f in lv.items()}
        fams = {
            "Ssel": ["P", *[f"S{lam:g}" for lam in LAMS if lam > 0], "C"],
            "CPsel": ["P", *[f"CP{lam:g}" for lam in LAMS]],
        }
        for sel, names in fams.items():
            pick = causal_pick([ls[m] for m in names], null=0)
            lv[sel] = take([lv[m] for m in names], pick)
            for k in range(N_K):
                share = np.bincount(pick[:, k], minlength=len(names)) / pick.shape[0]
                picks.append(
                    {"window": wname, "family": sel, "clock": ISSUE[k]}
                    | {names[i]: float(share[i]) for i in range(len(names))}
                )
        pick = causal_pick([qlike(rv, lv[m]) for m in ("P", "C+clock smear")], null=0)
        lv["Csel+clock smear"] = take([lv["P"], lv["C+clock smear"]], pick)
        for m, f in lv.items():
            fc[(wname, m)] = f
        print(f"window {wname}: {len(lv)} models in {time.time() - t0:.1f} s")
    return {"fc": fc, "picks": pd.DataFrame(picks)}


def score_a(
    pan: dict[str, Any], fc: dict[tuple[str, str], np.ndarray]
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    rv, dates = pan["rv"][:, 1:], pan["dates"]
    loss = {key: qlike(rv, f) for key, f in fc.items()}
    common = np.all([np.isfinite(v).all(axis=1) for v in loss.values()], axis=0)
    samples = {
        "long": common,
        "deck": common & np.asarray((dates >= DECK_START) & (dates <= DECK_END)),
    }
    pooled, by_clock = [], []
    for sname, m in samples.items():
        for (wname, model), v in loss.items():
            ref = loss[(wname, "P")]
            d = (v[m] - ref[m]).mean(axis=1)
            lo, hi = boot_ci(d) if model != "P" else (0.0, 0.0)
            t_hac, _ = asl.newey_west_t(d) if model != "P" else (float("nan"), 0)
            f = fc[(wname, model)]
            pooled.append(
                {
                    "sample": sname,
                    "window": wname,
                    "model": model,
                    "n_days": int(m.sum()),
                    "QLIKE": float(v[m].mean()),
                    "d_vs_P": float(d.mean()),
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "t_hac": float(t_hac),
                    "improves": bool(hi < 0.0),
                    "bias_mean_y_over_mean_f": float(rv[m].mean() / f[m].mean()),
                }
            )
            for k in range(N_K):
                dk = v[m][:, k] - ref[m][:, k]
                lo_k, hi_k = boot_ci(dk) if model != "P" else (0.0, 0.0)
                by_clock.append(
                    {
                        "sample": sname,
                        "window": wname,
                        "model": model,
                        "clock": ISSUE[k],
                        "QLIKE": float(v[m][:, k].mean()),
                        "d_vs_P": float(dk.mean()),
                        "ci_lo": lo_k,
                        "ci_hi": hi_k,
                        "bias": float(rv[m][:, k].mean() / f[m][:, k].mean()),
                    }
                )
    return pd.DataFrame(pooled), pd.DataFrame(by_clock), samples["deck"]


# ------------------------------------------------------------ second stage ---
def mz_maps(f: np.ndarray, y: np.ndarray, window: int | None) -> dict[str, np.ndarray]:
    """Causal log Mincer-Zarnowitz maps of a (days, 12) forecast panel."""
    lf, ly = np.log(f), np.log(y)
    valid = np.isfinite(lf) & np.isfinite(ly)
    n = f.shape[0]
    yv = np.where(valid, ly, 0.0)
    out: dict[str, np.ndarray] = {}

    x2 = np.stack([np.ones_like(lf), np.nan_to_num(lf)], axis=-1)
    sa, sb = (
        wsum(outer_rows(x2, valid), window),
        wsum(np.where(valid[..., None], x2, 0.0) * yv[..., None], window),
    )
    cnt = wsum(valid.astype(float), window)
    ok = np.asarray((cnt >= MZ_MIN).all(axis=1), dtype=bool)
    out["pooled"] = np.einsum("nkp,np->nk", x2, bsolve(sa.sum(1), sb.sum(1), ok))
    out["per clock"] = np.einsum(
        "nkp,nkp->nk", x2, bsolve(sa, sb, np.repeat(ok[:, None], N_K, axis=1))
    )
    dum = np.zeros((n, N_K, N_K))
    dum[:, np.arange(N_K), np.arange(N_K)] = 1.0
    x13 = np.concatenate([dum, np.nan_to_num(lf)[..., None]], axis=-1)
    sa13 = wsum(outer_rows(x13, valid).sum(axis=1), window)
    sb13 = wsum((np.where(valid[..., None], x13, 0.0) * yv[..., None]).sum(1), window)
    out["clock intercept"] = np.einsum("nkp,np->nk", x13, bsolve(sa13, sb13, ok))
    for v in out.values():
        v[~valid] = np.nan
    return out


def part_b(
    deck: dict[str, Any],
) -> tuple[pd.DataFrame, dict[tuple[str, str, str], np.ndarray]]:
    y = deck["y"]
    rows, fcs = [], {}
    for tag in TAGS:
        f0 = deck["f"][tag]
        cand: dict[tuple[str, str], np.ndarray] = {("-", "production"): f0}
        for wname, window in MZ_WINDOWS.items():
            maps = mz_maps(f0, y, window)
            ly = np.log(y)
            for m, pred in maps.items():
                cand[(wname, m)] = np.exp(pred) * smear(ly - pred, False)
            pc = maps["per clock"]
            cand[(wname, "per clock+clock smear")] = np.exp(pc) * smear(ly - pc, True)
        loss = {key: qlike(y, f) for key, f in cand.items()}
        common = np.all([np.isfinite(v).all(axis=1) for v in loss.values()], axis=0)
        ref = loss[("-", "production")]
        for (wname, m), v in loss.items():
            d = (v[common] - ref[common]).mean(axis=1)
            lo, hi = boot_ci(d) if m != "production" else (0.0, 0.0)
            rec: dict[str, Any] = {
                "forecast": tag,
                "window": wname,
                "map": m,
                "n_days": int(common.sum()),
                "QLIKE": float(v[common].mean()),
                "d_vs_production": float(d.mean()),
                "ci_lo": lo,
                "ci_hi": hi,
                "improves": bool(hi < 0.0),
            }
            for k in (8, 9, 10, 11):
                rec[f"QLIKE_{ISSUE[k]}"] = float(v[common][:, k].mean())
            rows.append(rec)
            fcs[(tag, wname, m)] = np.where(common[:, None], cand[(wname, m)], np.nan)
    return pd.DataFrame(rows), fcs


# ------------------------------------------------------------- trade inputs --
def trade_row(f: np.ndarray, deck: dict[str, Any], m: np.ndarray) -> dict[str, float]:
    y, sl, r = deck["y"][m], deck["slice"][m], deck["R1530"][m]
    ff = f[m]
    hit = ((ff > sl) == (y > sl)).astype(float)
    hit[~(np.isfinite(ff) & np.isfinite(sl) & np.isfinite(y))] = np.nan
    q = np.where(ff[:, -1] > sl[:, -1], 1.0, -1.0)
    pnl = q * r
    rec = {f"hit_{ISSUE[k]}": float(np.nanmean(hit[:, k])) for k in (0, 6, 10, 11)}
    rec["hit_all_clocks"] = float(np.nanmean(hit))
    rec["pct_buy_1530"] = float(100.0 * (q > 0).mean())
    rec["Sharpe_mid_1530"] = float(pnl.mean() / pnl.std(ddof=1) * ANN)
    return rec


def hac_ols(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS coefficients and Newey-West t statistics (asl's lag rule)."""
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    u = y - x @ beta
    xu = x * u[:, None]
    lag = asl.newey_west_lag(len(y))
    s = xu.T @ xu
    for j in range(1, lag + 1):
        g = xu[j:].T @ xu[:-j]
        s += (1.0 - j / (lag + 1.0)) * (g + g.T)
    xtx_inv = np.linalg.inv(x.T @ x)
    cov = xtx_inv @ s @ xtx_inv
    return beta, beta / np.sqrt(np.diag(cov))


def part_d(
    a_deck: dict[tuple[str, str], np.ndarray], deck: dict[str, Any], m: np.ndarray
) -> pd.DataFrame:
    """Does the per-clock proxy carry information the production ridge lacks?

    Two readings on the same deck days, per clock: (i) the geometric average of
    the production ridge and a proxy, which fits nothing, scored against the
    ridge alone; (ii) the encompassing regression log y = a + b1 log ridge +
    b2 log proxy over the whole sample, b2's Newey-West t.  The pooled proxy is
    the control: whatever it adds is diversification, not time-of-day structure.
    """
    y, prod = deck["y"][m], deck["f"]["blk2"][m]
    l_prod = qlike(y, prod)
    rows = []
    for (wname, model), f in a_deck.items():
        if wname == "250" or model not in ("P", "C", "C+clock smear", "Ssel", "CPsel"):
            continue
        fa = f[m]
        l_avg = qlike(y, np.sqrt(prod * fa))
        for k in range(N_K):
            d = l_avg[:, k] - l_prod[:, k]
            lo, hi = boot_ci(d)
            xk = np.column_stack(
                [np.ones(m.sum()), np.log(prod[:, k]), np.log(fa[:, k])]
            )
            beta, tval = hac_ols(xk, np.log(y[:, k]))
            rows.append(
                {
                    "window": wname,
                    "proxy": model,
                    "clock": ISSUE[k],
                    "n_days": int(m.sum()),
                    "QLIKE_ridge": float(l_prod[:, k].mean()),
                    "QLIKE_proxy": float(qlike(y, fa)[:, k].mean()),
                    "QLIKE_average": float(l_avg[:, k].mean()),
                    "d_average_vs_ridge": float(d.mean()),
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "improves": bool(hi < 0.0),
                    "b_ridge": float(beta[1]),
                    "t_ridge": float(tval[1]),
                    "b_proxy": float(beta[2]),
                    "t_proxy": float(tval[2]),
                }
            )
    return pd.DataFrame(rows)


def load_deck(pan: dict[str, Any]) -> dict[str, Any]:
    p51 = _load_module(HERE / "51_accuracy_ladder_1530.py", "p51_accuracy_ladder")
    fr = p51.build_intraday_frames()
    w = fr["work"]["intraday"].copy()
    w["date"] = pd.to_datetime(w["date"])
    days = pd.DatetimeIndex(sorted(w["date"].unique()))

    def piv(col: str) -> np.ndarray:
        return (
            w.pivot_table(index="date", columns="hhmm", values=col, aggfunc="first")
            .reindex(index=days, columns=list(ISSUE))
            .to_numpy(float)
        )

    deck = {
        "dates": days,
        "y": piv("rv_next"),
        "slice": piv("slice"),
        "R1530": piv("R")[:, -1],
        "f": {tag: piv(f"rv_hat_{tag}") for tag in asl.MODEL_ORDER},
    }
    pos = pan["dates"].get_indexer(days)
    have = pos >= 0
    mine = np.full(deck["y"].shape, np.nan)
    mine[have] = pan["rv"][pos[have], 1:]
    both = np.isfinite(mine) & np.isfinite(deck["y"])
    rel = float(np.max(np.abs(mine[both] / deck["y"][both] - 1.0)))
    assert rel < GATE_RV_TOL, rel
    print(
        f"GATE R  realized variance equals the production panel's on {int(both.sum())} "
        f"deck bars (max relative difference {rel:.1e}); deck days outside this "
        f"script's full-session panel: {int((~have).sum())}"
    )
    ql = qlike(deck["y"][:, -1], deck["f"]["blk2"][:, -1])
    q = np.where(deck["f"]["blk2"][:, -1] > deck["slice"][:, -1], 1.0, -1.0)
    pnl = q * deck["R1530"]
    sh = float(pnl.mean() / pnl.std(ddof=1) * ANN)
    assert abs(float(ql.mean()) - GATE_QLIKE_1530) < GATE_QLIKE_TOL, ql.mean()
    assert abs(sh - GATE_SHARPE_1530) < GATE_SHARPE_TOL, sh
    ref = pd.read_csv(REF35)
    ref = ref[(ref["sample"] == "whole") & (ref["cell_set"] == "own")]
    ref = ref[(ref["forecast"] == "ridge") & (ref["hhmm"] == "15:30")]
    print(
        f"GATE Q  ridge QLIKE at 15:30 {float(ql.mean()):.4f} (proposal 35: "
        f"{float(ref['QLIKE'].iloc[0]):.4f}); 15:30 sign(s) mid Sharpe {sh:.6f}"
    )
    deck["pos"] = pos
    return deck


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    pd.set_option("display.max_rows", 400)
    t_start = time.time()

    pan = build_panel()
    ft = features(pan)
    res = part_a(pan, ft)
    pooled, by_clock, _ = score_a(pan, res["fc"])
    write(
        pooled, "a_pooled.csv", "first stage: QLIKE over the twelve clocks, vs pooled"
    )
    write(by_clock, "a_by_clock.csv", "first stage: QLIKE by clock, vs pooled")
    write(res["picks"], "a_lambda_picks.csv", "causal shrinkage picks by clock")
    show = ["P", "D", "C", "C+clock smear", "Csel+clock smear", "Ssel", "CPsel"]
    cols = [
        "sample",
        "window",
        "model",
        "n_days",
        "QLIKE",
        "d_vs_P",
        "ci_lo",
        "ci_hi",
        "t_hac",
        "improves",
        "bias_mean_y_over_mean_f",
    ]
    print("\n--- A1. first stage, QLIKE averaged over the twelve clocks")
    print(pooled[pooled["model"].isin(show)][cols].round(5).to_string(index=False))
    for sname in ("long", "deck"):
        tab = by_clock[
            (by_clock["sample"] == sname)
            & (by_clock["window"] == "expanding")
            & by_clock["model"].isin(show)
        ].pivot(index="clock", columns="model", values="QLIKE")[show]
        print(f"\n--- A2. QLIKE by clock, expanding window, sample {sname}")
        print(tab.round(4).to_string())
    sig = by_clock[(by_clock["model"] != "P") & (by_clock["ci_hi"] < 0)]
    print(
        f"\nclock-level cells whose interval excludes zero on the improving side: "
        f"{len(sig)} of {int((by_clock['model'] != 'P').sum())}"
    )

    deck = load_deck(pan)
    b_tab, b_fc = part_b(deck)
    write(b_tab, "b_second_stage.csv", "second stage on the production forecasts")
    print("\n--- B. log Mincer-Zarnowitz maps of the production forecasts (deck days)")
    print(b_tab.round(5).to_string(index=False))

    # ---------------------------------------------------- trade inputs (C) --
    rows = []
    pos = deck["pos"]
    have = pos >= 0
    a_deck = {key: np.full(deck["y"].shape, np.nan) for key in res["fc"]}
    for key, f in res["fc"].items():
        a_deck[key][have] = f[pos[have]]
    m_a = (
        have
        & np.all([np.isfinite(v).all(axis=1) for v in a_deck.values()], axis=0)
        & np.isfinite(deck["slice"]).all(axis=1)
    )
    for (wname, model), f in a_deck.items():
        if model in show:
            rows.append(
                {
                    "stage": "first (proxy)",
                    "forecast": model,
                    "window": wname,
                    "n_days": int(m_a.sum()),
                }
                | trade_row(f, deck, m_a)
            )
    rows.append(
        {
            "stage": "production",
            "forecast": "blk2",
            "window": "-",
            "n_days": int(m_a.sum()),
        }
        | trade_row(deck["f"]["blk2"], deck, m_a)
    )
    for (tag, wname, m), f in b_fc.items():
        mm = np.isfinite(f).all(axis=1) & np.isfinite(deck["slice"]).all(axis=1)
        rows.append(
            {
                "stage": "second",
                "forecast": f"{tag} | {m}",
                "window": wname,
                "n_days": int(mm.sum()),
            }
            | trade_row(f, deck, mm)
        )
    c_tab = pd.DataFrame(rows)
    write(c_tab, "c_trade_inputs.csv", "sign hit rates against the implied slice")
    print("\n--- C. hit rate of sign(f - slice) and the 15:30 sign(s) mid Sharpe")
    print(c_tab.round(4).to_string(index=False))

    d_tab = part_d(a_deck, deck, m_a)
    write(d_tab, "d_encompassing.csv", "what the per-clock proxy adds to the ridge")
    dcols = [
        "window",
        "proxy",
        "clock",
        "QLIKE_ridge",
        "QLIKE_proxy",
        "QLIKE_average",
        "d_average_vs_ridge",
        "ci_lo",
        "ci_hi",
        "improves",
        "b_ridge",
        "b_proxy",
        "t_proxy",
    ]
    print("\n--- D. production ridge averaged with a proxy, and the encompassing t")
    print(
        d_tab[d_tab["clock"].isin(["10:00", "14:00", "14:30", "15:00", "15:30"])][dcols]
        .round(4)
        .to_string(index=False)
    )
    agg = (
        d_tab.groupby(["window", "proxy"])
        .agg(
            QLIKE_ridge=("QLIKE_ridge", "mean"),
            QLIKE_average=("QLIKE_average", "mean"),
            clocks_improving=("improves", "sum"),
            clocks_t_proxy_gt2=("t_proxy", lambda v: int((v > 2.0).sum())),
        )
        .reset_index()
    )
    trade = []
    for (wname, model), f in a_deck.items():
        if ((d_tab["window"] == wname) & (d_tab["proxy"] == model)).any():
            avg = np.sqrt(deck["f"]["blk2"] * f)
            trade.append({"window": wname, "proxy": model} | trade_row(avg, deck, m_a))
    agg = agg.merge(pd.DataFrame(trade), on=["window", "proxy"], how="left")
    write(
        agg, "d_average_summary.csv", "ridge-proxy average: accuracy and trade inputs"
    )
    print("\n--- D2. over the twelve clocks (hit rates and Sharpe are the AVERAGE's)")
    print(agg.round(4).to_string(index=False))
    ridge_row = trade_row(deck["f"]["blk2"], deck, m_a)
    print(
        "production ridge alone on the same days: "
        + ", ".join(f"{k} {v:.4f}" for k, v in ridge_row.items())
    )
    print(f"\ntotal runtime {time.time() - t_start:.1f} s")


if __name__ == "__main__":
    main()
