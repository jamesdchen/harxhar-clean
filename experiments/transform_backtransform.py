"""Causal back-transforms of the paper's final forecasts against plain (results/transform_ablation_2026-10-30/RULE.txt,
Part B, section 3; proposed after related results were seen; not pre-registered).  Laptop only, no new forecast.

The paper scores plain = pred_adj^2 x B (author decision 2026-10-06: recalibration removed); this script MEASURES what
causal corrections of that back-transform would do on the final 2014-2024 forecasts, class by class.  Plain stays the
paper's forecast unless the author decides otherwise.

Forecasts (writeup/final_run.py FOMCFIX_RUNS, Table 1's 13 rows):
  results/window_sweep_2014start_fomcfix/w3000/forecasts.parquet         ols / ridge / lasso / enet
  results/pooled_trees_final_2014start_fomcfix_w3000/forecasts.parquet   lgbm / xgb
  results/pooled_rf_skx_2014start_fomcfix_w3000/forecasts.parquet        rf_skx (-> rf)
every row from 2014-01-02 10:00 (33,970 bars, 2,643 sessions; the linear rows of 27-31 Dec 2013 are dropped so that
every class has the same history).

Corrections (bar t of session d; L = 250 sessions, lagged one session; identity with fewer than MIN_HIST sessions):
  c         c_d x plain, c_d = mean(RV / plain) over the window
  c_slot    the same within the bar's time-of-day slot (identity where the slot has < MIN_HIST usable bars)
  duan_fit  mean_i (pred + e_i)^2 x B, e = true_adj - pred (the winsorized fit target)
  duan_raw  the same with e = sqrt(RV / B) - pred
  contract  (m^2 + s2) x B, m = a + b pred with (a, b) the window's OLS of sqrt(RV / B) on [1, pred]; s2 the EWMA(0.6)
            of the squared residuals of the previous usable bars (each once), seeded with the window's mean squared MZ
            residual at the first corrected session (experiments/score_unification.py _score_chunk_causal, made causal
            on a rolling session window)
  oracle_c  c* x plain with c* = mean(RV / plain) on the span's scored bars: EX POST, the bound, not a competitor
  mz_window (ADDENDUM A of RULE.txt, 2026-10-30, added after the first results were seen: the contract's two parts
            apart) (m^2 + s2_w) x B with the contract's m and s2_w = the window's mean squared MZ residual, lagged one
            session (no bar-level EWMA)
Scored as experiments/ablate_score.py: QLIKE r - ln r - 1 (r = RV / forecast) and squared error on the bar's actual RV
(data/core_stats.parquet sumret2), Table 1's 33,527 bars (asserted); Diebold-Mariano (src/evaluation/diebold_mariano
dm_test) of each correction against plain, on the daily average loss (every-bar spans) or the bar losses (16:00 spans).

usage: python experiments/transform_backtransform.py [--out results/transform_ablation_2026-10-30/backtransform]
       -> metrics.csv, dm.csv, summary.csv, checks.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(HERE), str(ROOT), str(ROOT / "notebooks")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ablate_score as AS  # noqa: E402
from src.evaluation.diebold_mariano import dm_test  # noqa: E402

RES = ROOT / "results"
FILES = (
    RES / "window_sweep_2014start_fomcfix" / "w3000" / "forecasts.parquet",
    RES / "pooled_trees_final_2014start_fomcfix_w3000" / "forecasts.parquet",
    RES / "pooled_rf_skx_2014start_fomcfix_w3000" / "forecasts.parquet",
)
TABLE1 = RES / "oos_final_2014start" / "table1_w3000" / "metrics.csv"
OUT = RES / "transform_ablation_2026-10-30" / "backtransform"
FIRST = pd.Timestamp("2014-01-02")
N_BARS_FILE = 33_970
N_SCORED = 33_527
L_SESS = 250
MIN_HIST = 20
EWMA_LAMBDA = 0.6
SET_ARM = {"all_features": "full", "baseline": "har"}
CORR = ("c", "c_slot", "duan_fit", "duan_raw", "contract", "mz_window")  # mz_window: RULE.txt addendum A
EVERY_BAR = ("all", "ord", "ev", "pb")


def say(msg: str) -> None:
    print(msg, flush=True)


def load() -> dict[tuple[str, str], pd.DataFrame]:
    """(class, set) -> frame indexed by bar end: pred, true_adj, B, plain (2014-01-02 on)."""
    out: dict[tuple[str, str], pd.DataFrame] = {}
    for p in FILES:
        f = pd.read_parquet(p, columns=["model", "predictor_set", "date", "pred_adj", "true_adj", "B", "plain"])
        for (m, s), g in f.groupby(["model", "predictor_set"]):
            g = g[g["date"] >= FIRST].set_index(pd.DatetimeIndex(g["date"][g["date"] >= FIRST])).sort_index()
            k = ({"rf_skx": "rf"}.get(str(m), str(m)), SET_ARM[str(s)])
            out[k] = pd.DataFrame(
                {"pred": g["pred_adj"].to_numpy(float), "true_adj": g["true_adj"].to_numpy(float)}
                | {"B": g["B"].to_numpy(float), "plain": g["plain"].to_numpy(float)},
                index=g.index,
            )
    return out


def _window(cs: np.ndarray, i: np.ndarray, L: int) -> np.ndarray:
    """Sum over sessions [max(0, i - L), i) from the cumulative sums cs (cs[0] = 0, cs[k] = sum of the first k)."""
    return cs[i] - cs[np.maximum(0, i - L)]


def corrections(
    pred: np.ndarray,
    true_adj: np.ndarray,
    B: np.ndarray,
    rv: np.ndarray,
    sess: np.ndarray,
    slot: np.ndarray,
    n_sess: int,
    n_slot: int,
    L: int = L_SESS,
    min_hist: int = MIN_HIST,
    lam: float = EWMA_LAMBDA,
) -> tuple[dict[str, np.ndarray], dict]:
    """The causal corrections of the module docstring for one forecast series in time order.  sess: 0-based session
    code of each bar (non-decreasing), slot: 0-based slot code.  Returns ({name: forecast}, diagnostics)."""
    n = len(pred)
    assert np.all(np.diff(sess) >= 0)
    plain = pred**2 * B
    y_raw = np.sqrt(np.where(B > 0, rv, np.nan) / B)
    u = np.isfinite(plain) & (plain > 0) & np.isfinite(B) & (B > 0) & np.isfinite(rv) & (rv > 0)
    uf = u & np.isfinite(true_adj)
    hist = np.minimum(np.arange(n_sess), L)  # sessions in the window of each session
    on = hist[sess] >= min_hist  # a correction exists for the bar

    def ssum(w: np.ndarray, mask: np.ndarray) -> np.ndarray:  # per-session sums -> cumulative (len n_sess + 1)
        s = np.bincount(sess[mask], weights=w[mask], minlength=n_sess)
        return np.concatenate([[0.0], np.cumsum(s)])

    one = np.ones(n)
    out: dict[str, np.ndarray] = {}
    diag: dict = {}
    # c: the trailing mean of RV / plain
    r = np.where(u, rv / np.where(u, plain, 1.0), np.nan)
    nW = _window(ssum(one, u), sess, L)
    cW = _window(ssum(r, u), sess, L) / np.where(nW > 0, nW, np.nan)
    c = np.where(on & (nW > 0), cW, 1.0)
    out["c"] = c * plain
    diag["c"] = dict(mean=float(np.mean(c[on])), min=float(np.min(c[on])), max=float(np.max(c[on])))
    # c_slot: per slot
    code = sess * n_slot + slot
    cnt = np.bincount(code[u], minlength=n_sess * n_slot).reshape(n_sess, n_slot)
    sr = np.bincount(code[u], weights=r[u], minlength=n_sess * n_slot).reshape(n_sess, n_slot)
    ccnt = np.vstack([np.zeros((1, n_slot)), np.cumsum(cnt, axis=0)])
    csr = np.vstack([np.zeros((1, n_slot)), np.cumsum(sr, axis=0)])
    lo = np.maximum(0, sess - L)
    nS = ccnt[sess, slot] - ccnt[lo, slot]
    cS = (csr[sess, slot] - csr[lo, slot]) / np.where(nS > 0, nS, np.nan)
    cs_ = np.where(on & (nS >= min_hist), cS, 1.0)
    out["c_slot"] = cs_ * plain
    diag["c_slot"] = dict(mean=float(np.mean(cs_[on])), identity_bars_after_burnin=int((on & (nS < min_hist)).sum()))
    # Duan's smear, fit scale and raw scale: mean_i (pred + e_i)^2 = pred^2 + 2 pred ebar + mean e^2
    for name, e, m in (("duan_fit", true_adj - pred, uf), ("duan_raw", y_raw - pred, u)):
        nE = _window(ssum(one, m), sess, L)
        s1 = _window(ssum(e, m), sess, L) / np.where(nE > 0, nE, np.nan)
        s2 = _window(ssum(e**2, m), sess, L) / np.where(nE > 0, nE, np.nan)
        f = (pred**2 + 2 * pred * s1 + s2) * B
        out[name] = np.where(on & (nE > 0), f, plain)
        diag[name] = dict(mean_ebar=float(np.nanmean(s1[on])), mean_e2=float(np.nanmean(s2[on])))
    # contract: rolling MZ mean map + EWMA second moment
    x, y = pred, y_raw
    nM = _window(ssum(one, u), sess, L)
    Sx, Sy = _window(ssum(x, u), sess, L), _window(ssum(y, u), sess, L)
    Sxx, Sxy, Syy = _window(ssum(x * x, u), sess, L), _window(ssum(x * y, u), sess, L), _window(ssum(y * y, u), sess, L)
    with np.errstate(invalid="ignore", divide="ignore"):
        mx, my = Sx / nM, Sy / nM
        vx = Sxx / nM - mx**2
        cv = Sxy / nM - mx * my
        ok = on & (nM >= 3) & np.isfinite(vx) & (vx > 1e-12 * np.maximum(Sxx / nM, 1e-300))
        b = np.where(ok, cv / np.where(ok, vx, 1.0), 1.0)
        a = np.where(ok, my - b * mx, 0.0)
        mse_w = Syy / nM - 2 * a * my - 2 * b * Sxy / nM + a**2 + 2 * a * b * mx + b**2 * Sxx / nM
    m_ = a + b * x
    s2 = np.full(n, np.nan)
    first = int(np.argmax(on)) if on.any() else n
    state = float(mse_w[first]) if first < n else np.nan
    for t in range(first, n):
        s2[t] = state
        if u[t]:
            state = lam * state + (1.0 - lam) * (y[t] - m_[t]) ** 2
    fc = (m_**2 + s2) * B
    out["contract"] = np.where(on, fc, plain)
    # RULE.txt addendum A: the contract's mean map with the window's scalar second moment (session-lagged)
    out["mz_window"] = np.where(on, (m_**2 + mse_w) * B, plain)
    diag["contract"] = dict(
        mean_a=float(np.mean(a[on])),
        mean_b=float(np.mean(b[on])),
        mean_s2=float(np.nanmean(s2[on])),
        mz_identity_bars_after_burnin=int((on & ~ok).sum()),
        seed=state if first == n else float(mse_w[first]),
    )
    fb = {}
    for k in list(out):
        bad = ~(np.isfinite(out[k]) & (out[k] > 0))
        fb[k] = int((bad & np.isfinite(plain) & (plain > 0)).sum())
        out[k] = np.where(bad, plain, out[k])
    diag["fallback_to_plain"] = fb
    diag["burnin_bars_identity"] = int((~on).sum())
    return out, diag


def qlike_se(f: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = y / f
    return r - np.log(r) - 1.0, (y - f) ** 2


def causality_check(fr: pd.DataFrame, rv: np.ndarray, sess: np.ndarray, slot: np.ndarray, n_sess: int) -> dict:
    """RULE.txt 8(c): perturb every bar from a session d* on and confirm the forecasts before d* (all corrections) and
    of d* (the session-lagged ones; the contract's first bar of d*) are unchanged."""
    rng = np.random.default_rng(20261030)
    d_star = int(rng.integers(400, n_sess - 10))
    t0 = int(np.argmax(sess == d_star))
    P, T = fr["pred"].to_numpy(), fr["true_adj"].to_numpy()
    B = fr["B"].to_numpy()
    base, _ = corrections(P, T, B, rv, sess, slot, n_sess, 13)
    k = np.ones(len(P))
    k[t0:] = np.exp(rng.normal(0, 0.5, len(P) - t0))
    rv2, T2 = rv * k, T * k
    pert, _ = corrections(P, T2, B, rv2, sess, slot, n_sess, 13)  # outcomes from d* on perturbed
    same_d = sess == d_star
    res: dict = {"d_star_session": d_star, "first_bar": t0}
    okall = True
    for c in CORR:
        before = np.array_equal(base[c][:t0], pert[c][:t0])
        on_d = np.array_equal(base[c][same_d], pert[c][same_d]) if c != "contract" else base[c][t0] == pert[c][t0]
        res[c] = dict(before_unchanged=bool(before), session_d_unchanged=bool(on_d))
        okall &= bool(before and on_d)
    P3 = P.copy()
    P3[t0:] *= k[t0:]  # the model's own forecasts from d* on perturbed too: nothing before d* may move
    pert3, _ = corrections(P3, T2, B, rv2, sess, slot, n_sess, 13)
    for c in CORR:
        b3 = np.array_equal(base[c][:t0], pert3[c][:t0])
        res[c]["before_unchanged_pred_perturbed"] = bool(b3)
        okall &= b3
    res["pass"] = bool(okall)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    out = Path(a.out)
    rv_s = AS.core_rv()
    days = AS.evaluation_days(rv_s)
    ev = AS.event_days(days)
    panel = load()
    checks: dict = {
        "fail": [],
        "label": "RULE.txt Part B: proposed after related results were seen; not pre-registered",
    }
    idx0 = next(iter(panel.values())).index
    for k, f in panel.items():
        if not f.index.equals(idx0):
            checks["fail"].append(f"{k}: bar index differs from the first class's")
        if not np.array_equal(f["plain"].to_numpy(), f["pred"].to_numpy() ** 2 * f["B"].to_numpy()):
            d = np.max(np.abs(f["plain"] / (f["pred"] ** 2 * f["B"]) - 1))
            checks.setdefault("plain_vs_pred2B_maxrel", {})[f"{k[0]}/{k[1]}"] = float(d)
            if d > 1e-12:
                checks["fail"].append(f"{k}: plain != pred_adj^2 x B (max rel {d})")
        if not np.array_equal(f["B"].to_numpy(), panel[("ridge", "full")]["B"].to_numpy()):
            checks["fail"].append(f"{k}: B differs from ridge/full's")
    assert len(idx0) == N_BARS_FILE, len(idx0)
    # sessions and slots of the common bar index
    sd = pd.DatetimeIndex(idx0.normalize())
    sessions = sd.unique()
    sess = sessions.get_indexer(sd)
    tod = idx0.hour * 60 + idx0.minute
    slots = np.sort(np.unique(tod))
    slot = np.searchsorted(slots, tod)
    n_sess, n_slot = len(sessions), len(slots)
    assert n_slot == 13, slots
    rv = rv_s.reindex(idx0).to_numpy(float)
    # Table 1's scored bars (ablate_score.scored on the plain forecasts)
    plain_panel = {k: pd.Series(f["plain"].to_numpy(), index=f.index) for k, f in panel.items()}
    idx = AS.scored(plain_panel, rv_s, days)
    assert len(idx) == N_SCORED, len(idx)
    pos = idx0.get_indexer(idx)
    assert (pos >= 0).all()
    on_ev = pd.DatetimeIndex(idx.normalize()).isin(ev)
    c16 = (idx.hour == 16) & (idx.minute == 0)
    pb_from = sessions[L_SESS]
    spans = {
        "all": np.ones(len(idx), bool),
        "ord": ~on_ev,
        "ev": on_ev,
        "c16": c16,
        "c16_2020": c16 & (idx.normalize() >= AS.SUB_FIRST),
        "pb": idx.normalize() >= pb_from,
    }
    assert spans["c16"].sum() == AS.N_DAYS and spans["c16_2020"].sum() == AS.SUB_DAYS
    assert pd.DatetimeIndex(idx[on_ev].normalize()).nunique() == AS.N_EVENT
    checks["spans"] = {
        k: dict(bars=int(v.sum()), days=int(pd.DatetimeIndex(idx[v].normalize()).nunique())) for k, v in spans.items()
    }
    checks["pb_first_session"] = str(pb_from.date())
    checks["window"] = dict(sessions=L_SESS, min_hist=MIN_HIST, ewma_lambda=EWMA_LAMBDA, lag="one session")
    y = rv[pos]
    met, dms, diag_all = [], [], {}
    for (cls, arm), f in sorted(panel.items()):
        fc, diag = corrections(
            f["pred"].to_numpy(), f["true_adj"].to_numpy(), f["B"].to_numpy(), rv, sess, slot, n_sess, n_slot
        )
        diag_all[f"{cls}/{arm}"] = diag
        fc = {"plain": f["plain"].to_numpy(), **fc}
        L = {k: qlike_se(v[pos], y) for k, v in fc.items()}
        for sp, msk in spans.items():
            ix = idx[msk]
            qp, sp_ = L["plain"][0][msk], L["plain"][1][msk]
            fp = fc["plain"][pos][msk]
            cstar = float(np.mean(y[msk] / fp))
            qo, so = qlike_se(cstar * fp, y[msk])
            by_corr = {**{k: (v[0][msk], v[1][msk]) for k, v in L.items()}, "oracle_c": (qo, so)}
            for name, (q, se) in by_corr.items():
                met.append(
                    dict(
                        cls=cls,
                        set=arm,
                        corr=name,
                        span=sp,
                        n_bars=int(msk.sum()),
                        n_days=int(pd.DatetimeIndex(ix.normalize()).nunique()),
                        qlike=float(q.mean()),
                        mse=float(se.mean()),
                        qlike_pct_vs_plain=100.0 * (q.mean() / qp.mean() - 1.0),
                        mse_pct_vs_plain=100.0 * (se.mean() / sp_.mean() - 1.0),
                        oracle_cstar=cstar if name == "oracle_c" else np.nan,
                    )
                )
                if name in ("plain", "oracle_c"):
                    continue
                for li, (x1, x2) in enumerate(((q, qp), (se, sp_))):
                    r_ = dm_test(AS.daily(x1, ix), AS.daily(x2, ix)) if sp in EVERY_BAR else dm_test(x1, x2)
                    dms.append(
                        dict(
                            cls=cls,
                            set=arm,
                            corr=name,
                            vs="plain",
                            span=sp,
                            loss=("qlike", "mse")[li],
                            dm=float(r_["dm"]),
                            p=float(r_["p"]),
                            mean_diff=float(r_["mean_diff"]),
                            T=int(r_["T"]),
                            hac_lag=int(r_.get("hac_lag", -1)),
                        )
                    )
    M, D = pd.DataFrame(met), pd.DataFrame(dms)
    # gate 8(c): plain reproduces Table 1
    t1 = pd.read_csv(TABLE1)
    n_t1 = 0
    for _, r_ in t1.iterrows():
        k = (str(r_["model"]), SET_ARM[str(r_["set"])])
        mine = M[(M["cls"] == k[0]) & (M["set"] == k[1]) & (M["corr"] == "plain")].set_index("span")
        for col, sp, ref in (
            ("qlike", "all", r_["qlike"]),
            ("mse", "all", r_["mse"]),
            ("qlike", "c16_2020", r_["qlike_1600"]),
        ):
            v = float(mine.loc[sp, col])
            n_t1 += 1
            if not np.isclose(v, ref, rtol=1e-12, atol=0):
                checks["fail"].append(f"Table 1 {k} {col} {sp}: {v!r} vs {ref!r}")
        if int(mine.loc["all", "n_bars"]) != int(r_["n_intervals"]):
            checks["fail"].append(f"Table 1 {k}: bars")
    checks["table1_values_checked"] = n_t1
    checks["causality"] = causality_check(panel[("ridge", "full")], rv, sess, slot, n_sess)
    if not checks["causality"]["pass"]:
        checks["fail"].append("causality check")
    checks["diagnostics"] = diag_all
    checks["pass"] = not checks["fail"]
    # summary: span all / ev / pb, QLIKE % vs plain with DM t and p, one row per (class, set, correction)
    q = D[D["loss"] == "qlike"].set_index(["cls", "set", "corr", "span"])
    srows = []
    for (cls, arm, corr), g in M[M["corr"] != "plain"].groupby(["cls", "set", "corr"], sort=False):
        row = dict(cls=cls, set=arm, corr=corr)
        for sp in ("all", "ev", "pb", "c16"):
            mm = g[g["span"] == sp].iloc[0]
            row[f"pct_{sp}"] = round(float(mm["qlike_pct_vs_plain"]), 3)
            if corr != "oracle_c":
                row[f"dm_{sp}"] = round(float(q.loc[(cls, arm, corr, sp), "dm"]), 2)
                row[f"p_{sp}"] = float(q.loc[(cls, arm, corr, sp), "p"])
        srows.append(row)
    S = pd.DataFrame(srows)
    out.mkdir(parents=True, exist_ok=True)
    M.to_csv(out / "metrics.csv", index=False)
    D.to_csv(out / "dm.csv", index=False)
    S.to_csv(out / "summary.csv", index=False)
    (out / "checks.json").write_text(json.dumps(checks, indent=1, default=float))
    say(
        f"scored {len(panel)} forecasts x {len(CORR) + 2} back-transforms on {len(idx)} bars; checks "
        f"{'PASS' if checks['pass'] else 'FAIL'} {checks['fail'][:5]}"
    )
    return 0 if checks["pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
