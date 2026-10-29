"""Section 8 (variable importance), the tree ensembles across training windows (2026-10-23).

The dense-and-weak test of Section 8 reads the linear models' coefficients across windows (make_rehaul_fi_pooled.py,
tab:rh_fip_window).  The trees' SHAP values exist at the main window only, so for them the section reads the accuracy
of the every-bar walks across windows: if the exogenous signal were dense and weak, the full predictor set would gain
more from a longer window than the HAR benchmark does.  This script recomputes that comparison from the merged walks
and writes its table and macros.

RUNS     finished and merged; the final design (first evaluation day 2 January 2014, refit at every bar, each tree on
         its frozen configuration, HAR benchmark and full predictor set):
           W      LightGBM, XGBoost (lgbm, xgb)                         random forest on oneDAL (final_run.RF_KEY)
           1000   results/pooled_trees_lc_2014start_w1000 (Hoffman2)      results/pooled_rf_skx_2014start_w1000 (CARC)
           2000   results/pooled_trees_lc_2014start_w2000 (Hoffman2)      results/pooled_rf_skx_2014start_w2000
           3000   results/pooled_trees_final_2014start_w3000 (CARC)       results/pooled_rf_skx_2014start_w3000
           3500   results/pooled_trees_final_2014start_w3500 (CARC)       results/pooled_rf_skx_2014start_w3500
         (the Hoffman2 pieces are gated bit for bit against CARC pieces: results/pooled_trees_lc_2014start_w1000/gate/),
         and the linear walks of each window, results/window_sweep_2014start_2026-10-22/w<W>, which fix the scored bars
         as in Table 1.  A run without forecasts.parquet is PENDING (final_run's convention): every cell and macro that
         needs it prints \\rhPending{}, a table row whose cells are all pending ends with final_run.ROW_MARK, and every
         count runs over the finished comparisons only (with ...All twins).  The pooled MinGRU is not read.
SCORING  Table 1's rules, imported from make_rehaul_robust_window.py (core, evaluation_days, scored_set, qlike, daily and
         the gates of _frame): every bar of the 2,579 evaluation days 2014-01-02 .. 2024-04-30 and the 16:00 bar;
         asserted: the trees forecast every bar the linear rows of their window score, and all windows score the same
         bars.
STATISTICS  as results/pooled_trees_lc_2014start_w1000/code/cross_window.py (recomputed here, not copied):
           gap(W)      mean over days of the daily QLIKE of the full set minus the HAR benchmark's (negative: the full
                       set is more accurate), Newey-West t
           extra gain  mean of (full_W2 - full_W1) - (HAR_W2 - HAR_W1), the daily difference-in-differences (negative:
                       the full set gains more from the longer window than the HAR benchmark), Newey-West t
           Newey-West: Bartlett weights, lag floor(4 (T/100)^(2/9)) = 8 on 2,579 days, divisor 1/T (dm_test's), two-sided
                       normal p.  DM between windows: src.evaluation.diebold_mariano.dm_test(loss at W2, loss at W1)
                       per arm (negative: the longer window has the lower loss).
         No rule fixed this comparison before the walks (RULE_rf_skx.txt fixes the forest's windows only), so the text
         calls it descriptive.  Significance counts at SIG = 5%.
SELF-CHECK  every recomputed value is compared with the committed outputs of the two cross-window comparisons
         (results/pooled_trees_lc_2014start_cross_window, results/pooled_rf_skx_2014start_cross_window: qlike_by_w,
         dm_pairs, gap_by_w, gap_change_did) and, at W = 3,000 and 3,500, with the window-robustness table's
         (results/rehaul_robust_window_final: metrics.csv QLIKE, dm_window.csv DM 3,500 vs 3,000); written to
         selfcheck.csv; a mismatch stops the run.  claims() checks the words of rehaul/08_feature_importance.tex.

WRITES   writeup/generated/rehaul_fip_lc_numbers.tex   \\rhlc... macros (loaded in the preamble of paper_rehaul.tex)
         writeup/generated/rehaul_fip_lc_table.tex     tab:rh_fip_lc (rehaul/08b_fi_measures_floats.tex)
         results/rehaul_fip_lc_final/                  qlike_by_w.csv, dm_pairs.csv, gap_by_w.csv, gap_change_did.csv,
                                                       selfcheck.csv
         (final_run.GEN and final_run.res_dir: REHAUL_GEN_DIR / REHAUL_RES_DIR redirect a test run.)

Run:  python writeup/make_rehaul_fip_lc.py     (from the repository root; independent of REHAUL_FINAL_RUN)
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WRITEUP = ROOT / "writeup"
for _p in (ROOT, ROOT / "notebooks", WRITEUP):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import make_rehaul_robust_window as RW  # noqa: E402  (Table 1's scoring rules and the frames' gates)

from src.evaluation.diebold_mariano import _nw_lag, dm_test  # noqa: E402

fr = RW.fr
RESULTS = ROOT / "results"
WINDOWS = (1000, 2000, 3000, 3500)
WTAG = {1000: "One", 2000: "Two", 3000: "Main", 3500: "Long"}
STEPS = ((1000, 2000), (2000, 3000), (3000, 3500))  # the table's adjacent steps
PAIRS = STEPS + ((1000, 3000),)  # the cross-window comparisons' pairs
TREES = ("lgbm", "xgb", "rf")
NAME = {"lgbm": "LightGBM", "xgb": "XGBoost", "rf": "Random forest"}
TEXT = {"lgbm": "LightGBM", "xgb": "XGBoost", "rf": "the random forest"}
MTAG = {"lgbm": "Lgbm", "xgb": "Xgb", "rf": "Rf"}
SETS = RW.SETS
BARS = (("every bar", "Bar"), ("16:00", "Close"))
SCALE = 1e3  # QLIKE differences in the table and the gap / extra-gain macros: units of 10^-3
SIG = RW.SIG
TOL = 1e-12  # self-check: relative tolerance (the same arithmetic on the same files)
HEAD = "% AUTO-GENERATED by writeup/make_rehaul_fip_lc.py -- do not edit; re-run the script."
CROSS = {
    "lgbm": RESULTS / "pooled_trees_lc_2014start_cross_window",
    "xgb": RESULTS / "pooled_trees_lc_2014start_cross_window",
    "rf": RESULTS / "pooled_rf_skx_2014start_cross_window",
}
CROSS_KEY = {
    "lgbm": "lgbm",
    "xgb": "xgb",
    "rf": RW.RF_KEY,
}  # model names in those files
ROBUST = RESULTS / "rehaul_robust_window_final"
PENDING_TEX, PENDING_DEF, ROW_MARK = RW.PENDING_TEX, RW.PENDING_DEF, RW.ROW_MARK


def tree_run(est: str, W: int) -> Path:
    if est == "rf":
        return RW.rf_run(W)
    if W in (1000, 2000):
        return RESULTS / f"pooled_trees_lc_2014start_w{W}"
    return RW.trees_run(W)


def step_tag(a: int, b: int) -> str:
    return WTAG[a] + WTAG[b]


# --------------------------------------------------------------------------- statistics


def nw(d: np.ndarray) -> dict:
    """Mean of a daily series, its Newey-West standard error (dm_test's lag, Bartlett weights, divisor 1/T), t and the
    two-sided normal p (cross_window.nw; finite values asserted instead of dropped)."""
    d = np.asarray(d, float)
    assert np.isfinite(d).all()
    T = d.size
    m = float(d.mean())
    dc = d - m
    lag = _nw_lag(T)
    lrv = float(np.mean(dc * dc))
    for k in range(1, lag + 1):
        lrv += 2.0 * (1.0 - k / (lag + 1.0)) * float(np.dot(dc[k:], dc[:-k]) / T)
    se = math.sqrt(lrv / T)
    t = m / se
    return {
        "mean": m,
        "se": se,
        "t": t,
        "p": math.erfc(abs(t) / math.sqrt(2.0)),
        "T": T,
        "lag": lag,
    }


def who(ests: list[str], n_all: int) -> str:
    """The trees in `ests` as text: "all three trees", "LightGBM and the random forest", "none of the trees"."""
    names = [TEXT[e] for e in TREES if e in ests]
    if not names:
        return "none of the trees"
    if len(names) == n_all and n_all > 1:
        return {2: "both trees", 3: "all three trees"}.get(n_all, f"all {n_all} trees")
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


# --------------------------------------------------------------------------- load and score


def load_window(W: int, rv: pd.Series, days: pd.DatetimeIndex):
    """The finished tree arms of window W ((est, set) -> frame), the scored bars of the window's linear rows, and the
    pending tree estimators."""
    lr = RW.linear_run(W)
    ri = json.loads((lr / "run_info.json").read_text())
    fr.guard_standin(ri, lr)
    assert int(ri["window_days"]) == W and pd.Timestamp(ri["first_day"]) == RW.FIRST, (
        lr,
        ri["window_days"],
        ri["first_day"],
    )
    lin = RW._frame(
        lr / "forecasts.parquet",
        {("ols", "baseline")} | {(m, s) for m in RW.LIN_EST for s in SETS},
        fr.N_LINEAR_ROWS,
        {},
    )
    idx = RW.scored_set(lin, rv, days)
    fc: dict = {}
    pending: list[str] = []
    for keys, rename in ((("lgbm", "xgb"), {}), (("rf",), {RW.RF_KEY: "rf"})):
        run = tree_run(keys[0], W)
        if not (run / "forecasts.parquet").is_file():
            pending += list(keys)
            continue
        meta = json.loads((run / "forecasts.json").read_text())
        fr.guard_standin(meta, run)
        assert (meta["refit"], int(meta["window_days"]), list(meta["span"])) == (
            "bar",
            W,
            [str(RW.FIRST.date()), str(RW.LAST.date())],
        ), (run, meta["refit"], meta["window_days"], meta["span"])
        ft = RW._frame(
            run / "forecasts.parquet",
            {(e, s) for e in keys for s in SETS},
            fr.N_BARS,
            rename,
        )
        for k, x in ft.items():
            assert x.index.min().normalize() == RW.FIRST, (run, k, x.index.min())
        fc.update(ft)
    if fc:
        both = RW.scored_set({**lin, **fc}, rv, days)
        assert both.equals(idx), f"W {W}: the trees miss bars the linear rows score"
    return fc, idx, pending


def score() -> dict:
    """Every finished arm's daily every-bar QLIKE and 16:00 QLIKE on the one common scored set, and the statistics."""
    rv = RW.core()
    days = RW.evaluation_days(rv)
    FT, IDX, PEND = {}, {}, {}
    for W in WINDOWS:
        FT[W], IDX[W], PEND[W] = load_window(W, rv, days)
        print(
            f"W {W:,}: {len(IDX[W]):,} scored bars; trees {sorted(FT[W])}" + (f"; PENDING {PEND[W]}" if PEND[W] else "")
        )
    idx = IDX[WINDOWS[0]]
    for W in WINDOWS[1:]:
        assert IDX[W].equals(idx), f"W {W} scores other bars than W {WINDOWS[0]}"
    on16 = idx[(idx.hour == 16) & (idx.minute == 0)]
    assert len(on16) == len(days) == RW.N_DAYS, (len(on16), len(days))
    L: dict = {}
    rows_q = []
    for W in WINDOWS:
        for (e, s), f in FT[W].items():
            q = RW.qlike(f, idx, rv)
            q16 = RW.qlike(f, on16, rv)
            L[(W, e, s)] = (RW.daily(q, idx), q16)
            rows_q.append(
                {
                    "W": W,
                    "model": e,
                    "set": s,
                    "qlike": float(q.mean()),
                    "qlike_1600": float(q16.mean()),
                }
            )
    have = {(W, e) for W in WINDOWS for e in TREES if (W, e, "baseline") in L}
    rows_dm, rows_g, rows_did = [], [], []
    for e in TREES:
        for i, where in enumerate(b for b, _ in BARS):
            for W in WINDOWS:
                if (W, e) not in have:
                    continue
                g = nw(L[(W, e, "all_features")][i] - L[(W, e, "baseline")][i])
                rows_g.append(
                    {
                        "model": e,
                        "bars": where,
                        "W": W,
                        "gap_full_minus_har": g["mean"],
                        "se": g["se"],
                        "t": g["t"],
                        "p": g["p"],
                    }
                )
            for a, b in PAIRS:
                if (a, e) not in have or (b, e) not in have:
                    continue
                for s in SETS:
                    r = dm_test(L[(b, e, s)][i], L[(a, e, s)][i])
                    rows_dm.append(
                        {
                            "pair": f"{a}->{b}",
                            "model": e,
                            "set": s,
                            "bars": where,
                            "qlike_W1": float(np.mean(L[(a, e, s)][i])),
                            "qlike_W2": float(np.mean(L[(b, e, s)][i])),
                            "diff": float(r["mean_diff"]),
                            "dm": float(r["dm"]),
                            "p": float(r["p"]),
                            "T": int(r["T"]),
                            "hac_lag": int(r["hac_lag"]),
                        }
                    )
                dF = L[(b, e, "all_features")][i] - L[(a, e, "all_features")][i]
                dH = L[(b, e, "baseline")][i] - L[(a, e, "baseline")][i]
                g = nw(dF - dH)
                rows_did.append(
                    {
                        "model": e,
                        "bars": where,
                        "pair": f"{a}->{b}",
                        "full_change": float(np.mean(dF)),
                        "har_change": float(np.mean(dH)),
                        "extra_gain": g["mean"],
                        "se": g["se"],
                        "t": g["t"],
                        "p": g["p"],
                        "T": g["T"],
                        "hac_lag": g["lag"],
                    }
                )
    return {
        "n_bars": len(idx),
        "n_days": len(days),
        "hac_lag": _nw_lag(len(days)),
        "first": RW.FIRST,
        "pending": PEND,
        "have": have,
        "Q": pd.DataFrame(rows_q),
        "DM": pd.DataFrame(rows_dm),
        "G": pd.DataFrame(rows_g),
        "DD": pd.DataFrame(rows_did),
    }


# --------------------------------------------------------------------------- self-check


def selfcheck(S: dict, out: Path) -> None:
    """Every recomputed number against the committed cross-window files and the window-robustness table."""
    rows: list[dict] = []

    def cmp(src: str, key: str, col: str, mine: float, theirs: float) -> None:
        ok = math.isclose(mine, theirs, rel_tol=TOL, abs_tol=1e-15)
        rows.append(
            {
                "source": src,
                "key": key,
                "column": col,
                "mine": mine,
                "theirs": theirs,
                "ok": ok,
            }
        )

    spec = {
        "qlike_by_w.csv": (S["Q"], ["W", "model", "set"], ["qlike", "qlike_1600"]),
        "dm_pairs.csv": (
            S["DM"],
            ["pair", "model", "set", "bars"],
            ["qlike_W1", "qlike_W2", "diff", "dm", "p", "T", "hac_lag"],
        ),
        "gap_by_w.csv": (
            S["G"],
            ["model", "bars", "W"],
            ["gap_full_minus_har", "se", "t", "p"],
        ),
        "gap_change_did.csv": (
            S["DD"],
            ["model", "bars", "pair"],
            ["full_change", "har_change", "extra_gain", "se", "t", "p", "T", "hac_lag"],
        ),
    }
    n_ref = 0
    for e in TREES:
        for fname, (mine, keys, cols) in spec.items():
            ref = pd.read_csv(CROSS[e] / fname)
            ref = ref[ref["model"] == CROSS_KEY[e]].assign(model=e)
            m = mine[mine["model"] == e]
            n_ref += len(ref)
            j = ref.merge(m, on=keys, how="outer", suffixes=("_ref", ""), indicator=True)
            src = f"{fr.rel(CROSS[e]).replace(chr(92), '/')}/{fname}"
            pending_e = any(e in S["pending"][W] for W in WINDOWS)
            for _, r in j.iterrows():
                key = "/".join(str(r[k]) for k in keys)
                if r["_merge"] != "both":
                    # a committed row this run cannot recompute is allowed only for a pending tree
                    ok = bool(r["_merge"] == "left_only" and pending_e)
                    rows.append(
                        {
                            "source": src,
                            "key": key,
                            "column": f"(row {r['_merge']})",
                            "mine": np.nan,
                            "theirs": np.nan,
                            "ok": ok,
                        }
                    )
                    continue
                for c in cols:
                    cmp(src, key, c, float(r[c]), float(r[c + "_ref"]))
    # the window-robustness table at 3,000 / 3,500 (its own scoring code, the same rules)
    met = pd.read_csv(ROBUST / "metrics.csv")
    q = S["Q"]
    for _, r in q[q["W"].isin([3000, 3500])].iterrows():
        x = met[(met["W"] == r["W"]) & (met["model"] == r["model"]) & (met["set"] == r["set"])]
        assert len(x) == 1, (r["W"], r["model"], r["set"])
        for c in ("qlike", "qlike_1600"):
            cmp(
                "results/rehaul_robust_window_final/metrics.csv",
                f"{r['W']}/{r['model']}/{r['set']}",
                c,
                float(r[c]),
                float(x.iloc[0][c]),
            )
    dw = pd.read_csv(ROBUST / "dm_window.csv")
    d = S["DM"]
    for _, r in d[d["pair"] == "3000->3500"].iterrows():
        bars = "bar" if r["bars"] == "every bar" else "close"
        x = dw[
            (dw["model"] == r["model"])
            & (dw["set"] == r["set"])
            & (dw["bars"] == bars)
            & (dw["W_a"] == 3500)
            & (dw["W_b"] == 3000)
        ]
        assert len(x) == 1, (r["model"], r["set"], bars)
        for c in ("dm", "p"):
            cmp(
                "results/rehaul_robust_window_final/dm_window.csv",
                f"{r['model']}/{r['set']}/{bars}",
                c,
                float(r[c]),
                float(x.iloc[0][c]),
            )
    sc = pd.DataFrame(rows)
    sc.to_csv(out / "selfcheck.csv", index=False)
    bad = sc[~sc["ok"].astype(bool)]
    mx = float(np.nanmax(np.abs(sc["mine"] - sc["theirs"]))) if len(sc) else 0.0
    print(
        f"self-check: {len(sc)} values ({n_ref} committed rows of the cross-window files), max |diff| {mx:.1e}, "
        f"{len(bad)} mismatch(es) -> {fr.rel(out / 'selfcheck.csv')}"
    )
    if len(bad):
        print(bad.head(20).to_string())
        raise SystemExit("self-check FAILED: a recomputed value differs from the committed files")


# --------------------------------------------------------------------------- the text's words


def sig_neg(r: pd.Series | None) -> bool:
    """A significant extra gain toward the full set (p < SIG, negative)."""
    return r is not None and bool(r["p"] < SIG and r["extra_gain"] < 0)


def sig_pos(r: pd.Series | None) -> bool:
    """A significant extra gain toward the HAR benchmark (p < SIG, positive)."""
    return r is not None and bool(r["p"] < SIG and r["extra_gain"] > 0)


def need(r: pd.Series | None) -> pd.Series:
    """A row the claims read (claims run only when nothing is pending)."""
    assert r is not None
    return r


def did(S: dict, e: str, a: int, b: int, where: str) -> pd.Series | None:
    x = S["DD"]
    x = x[(x["model"] == e) & (x["pair"] == f"{a}->{b}") & (x["bars"] == where)]
    return None if x.empty else x.iloc[0]


def gap(S: dict, e: str, W: int, where: str) -> pd.Series | None:
    x = S["G"]
    x = x[(x["model"] == e) & (x["W"] == W) & (x["bars"] == where)]
    return None if x.empty else x.iloc[0]


def dmrow(S: dict, e: str, s: str, a: int, b: int, where: str) -> pd.Series | None:
    x = S["DM"]
    x = x[(x["model"] == e) & (x["set"] == s) & (x["pair"] == f"{a}->{b}") & (x["bars"] == where)]
    return None if x.empty else x.iloc[0]


def claims(S: dict) -> None:
    """The words of rehaul/08_feature_importance.tex and 99_author_notes.tex that the macros do not carry, checked so
    a regeneration that changes them fails here instead of printing a wrong sentence (only when nothing is
    pending: a pending arm leaves the sentences to the next run)."""
    if any(S["pending"].values()):
        print("claims: skipped (an arm is pending)")
        return
    print("claims in the text")
    # every bar: the full set's lead widens up to the main window for every tree ("widens ... up to 3,000 days")
    for e in TREES:
        g = [float(need(gap(S, e, W, "every bar"))["gap_full_minus_har"]) for W in WINDOWS]
        check(
            g[0] > g[1] > g[2],
            f"{e}: every-bar gap full - HAR more negative from 1,000 to 2,000 to 3,000 days",
        )
        check(
            all(
                need(gap(S, e, W, "every bar"))["p"] < SIG and need(gap(S, e, W, "every bar"))["gap_full_minus_har"] < 0
                for W in WINDOWS
            ),
            f"{e}: every-bar gap negative and significant at every window",
        )
    # every bar: no step significant toward the benchmark (the text names only the negative ones)
    for a, b in STEPS:
        for e in TREES:
            check(
                not sig_pos(did(S, e, a, b, "every bar")),
                f"{e} {a}->{b} every bar: no significant positive extra gain",
            )
    # the t lists of the text follow these sets ("t = LightGBM, XGBoost, forest"; "LightGBM and the forest; XGBoost")
    for (a, b), want in (((1000, 2000), set(TREES)), ((2000, 3000), {"lgbm", "rf"})):
        got = {e for e in TREES if sig_neg(did(S, e, a, b, "every bar"))}
        check(
            got == want,
            f"{a}->{b} every bar: significant toward the full set = {sorted(want)} ({sorted(got)})",
        )
    # "for LightGBM the later step's extra gain is mostly the benchmark's loss rising"
    r = need(did(S, "lgbm", 2000, 3000, "every bar"))
    check(
        r["har_change"] > 0 and r["har_change"] > 0.5 * -r["extra_gain"],
        "lgbm 2,000->3,000 every bar: the HAR benchmark's rise is more than half of the extra gain",
    )
    # the text prints the forest's DM for the full set's fall from 2,000 to 3,000 days
    falls = [
        e
        for e in TREES
        if need(dmrow(S, e, "all_features", 2000, 3000, "every bar"))["p"] < SIG
        and need(dmrow(S, e, "all_features", 2000, 3000, "every bar"))["dm"] < 0
    ]
    check(
        falls == ["rf"],
        f"full-set QLIKE falls significantly 2,000->3,000 only for the forest ({falls})",
    )
    # the 3,500-day step: nothing significant on every bar ("and not beyond")
    check(
        all(not need(did(S, e, 3000, 3500, "every bar"))["p"] < SIG for e in TREES),
        "3,000->3,500 every bar: no tree's extra gain significant",
    )
    # 16:00: the significant steps are the ones the text names
    neg = {(e, a) for e in TREES for a, b in STEPS if sig_neg(did(S, e, a, b, "16:00"))}
    pos = {(e, a) for e in TREES for a, b in STEPS if sig_pos(did(S, e, a, b, "16:00"))}
    check(
        neg == {("lgbm", 2000), ("xgb", 2000)},
        f"16:00: toward the full set = LightGBM, XGBoost 2,000->3,000 ({sorted(neg)})",
    )
    check(
        pos == {("lgbm", 1000)},
        f"16:00: toward the benchmark = LightGBM 1,000->2,000 ({sorted(pos)})",
    )
    check(all(e != "rf" for e, _ in neg | pos), "16:00: never significant for the forest")


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(f"CLAIM FAILED: {msg}")
    print(f"  ok  {msg}")


# --------------------------------------------------------------------------- macros and table


def macros(S: dict) -> list[tuple[str, str, str]]:
    M: list[tuple[str, str, str]] = []
    n_trees = len(TREES)

    def add(name: str, value: str, note: str) -> None:
        M.append(("rhlc" + name, value, note))

    for W in WINDOWS:
        add(
            f"W{WTAG[W]}",
            RW.thousands(W),
            f"rolling window, trading days ({W * fr.BARS_DAY:,} bars)",
        )
    add(
        "FirstDay",
        f"{S['first'].day} {S['first'].strftime('%B %Y')}",
        "first evaluation day of every walk",
    )
    add("NDays", RW.thousands(S["n_days"]), "evaluation days (Table 1's)")
    add("NBars", RW.thousands(S["n_bars"]), "scored bars, the same at every window")
    add("HacLag", str(S["hac_lag"]), "Newey-West lag (Bartlett, floor(4 (T/100)^(2/9)))")
    done = [e for e in TREES if all((W, e) in S["have"] for W in WINDOWS)]
    add("NTrees", str(len(done)), "tree ensembles with every window finished")
    n_arms_all = len(WINDOWS) * n_trees * len(SETS)
    add(
        "NArmsDone",
        str(len(S["have"]) * len(SETS)),
        "tree walks finished (window x tree x set)",
    )
    add(
        "NArmsAll",
        str(n_arms_all),
        "tree walks of the comparison (window x tree x set)",
    )
    n_steps = {}
    for where, B in BARS:
        for e in TREES:
            E = MTAG[e]
            for W in WINDOWS:
                g = gap(S, e, W, where)
                add(
                    f"Gap{E}{WTAG[W]}{B}",
                    RW.mnum(None if g is None else SCALE * g["gap_full_minus_har"], 2),
                    f"{e} {where} W {W}: gap full - HAR, daily QLIKE x 10^3",
                )
                add(
                    f"GapT{E}{WTAG[W]}{B}",
                    RW.mnum(None if g is None else g["t"], 2),
                    f"{e} {where} W {W}: its NW t",
                )
            for a, b in PAIRS:
                tg = step_tag(a, b)
                r = did(S, e, a, b, where)
                add(
                    f"Eg{E}{tg}{B}",
                    RW.mnum(None if r is None else SCALE * r["extra_gain"], 2),
                    f"{e} {where} {a}->{b}: extra gain x 10^3 (negative: the full set gains more)",
                )
                add(
                    f"T{E}{tg}{B}",
                    RW.mnum(None if r is None else r["t"], 2),
                    f"{e} {where} {a}->{b}: NW t",
                )
                add(
                    f"P{E}{tg}{B}",
                    RW.mpval(None if r is None else r["p"]),
                    f"{e} {where} {a}->{b}: NW p",
                )
                for s, stg in (("all_features", "Full"), ("baseline", "Har")):
                    x = dmrow(S, e, s, a, b, where)
                    add(
                        f"Dm{stg}{E}{tg}{B}",
                        RW.mnum(None if x is None else x["dm"], 2),
                        f"{e} {s} {where}: DM loss at {b} vs {a} days (negative: the longer window lower)",
                    )
        for a, b in PAIRS:
            tg = step_tag(a, b)
            rs = {e: did(S, e, a, b, where) for e in TREES}
            fin = [e for e, r in rs.items() if r is not None]
            neg = [e for e in fin if sig_neg(rs[e])]
            pos = [e for e in fin if sig_pos(rs[e])]
            n_steps[(tg, B)] = len(fin)
            add(
                f"NSigNeg{tg}{B}",
                str(len(neg)),
                f"{where} {a}->{b}: trees with a negative extra gain, p < {SIG}",
            )
            add(
                f"NSigPos{tg}{B}",
                str(len(pos)),
                f"{where} {a}->{b}: trees with a positive extra gain, p < {SIG}",
            )
            add(
                f"SigNeg{tg}{B}Who",
                who(neg, len(fin)),
                f"{where} {a}->{b}: those trees, as text",
            )
            for s, stg in (("all_features", "Full"), ("baseline", "Har")):
                xs = {e: dmrow(S, e, s, a, b, where) for e in fin}
                f_ = [e for e, x in xs.items() if x is not None and x["p"] < SIG and x["dm"] < 0]
                add(
                    f"{stg}Fall{tg}{B}Who",
                    who(f_, len(fin)),
                    f"{where} {a}->{b}: trees whose {s} QLIKE falls significantly (DM, p < {SIG})",
                )
        # the adjacent steps together, and the gap cells
        neg_all = sum(int(sig_neg(did(S, e, a, b, where))) for e in TREES for a, b in STEPS)
        pos_all = sum(int(sig_pos(did(S, e, a, b, where))) for e in TREES for a, b in STEPS)
        n_st = sum(n_steps[(step_tag(a, b), B)] for a, b in STEPS)
        add(
            f"NSigNeg{B}",
            str(neg_all),
            f"{where}: adjacent steps x trees with a negative extra gain, p < {SIG}",
        )
        add(
            f"NSigPos{B}",
            str(pos_all),
            f"{where}: adjacent steps x trees with a positive extra gain, p < {SIG}",
        )
        add(f"NSteps{B}", str(n_st), f"{where}: adjacent steps x trees compared")
        add(
            f"NStepsAll{B}",
            str(len(STEPS) * n_trees),
            f"{where}: adjacent steps x trees of the comparison",
        )
        gs = [g for e in TREES for W in WINDOWS if (g := gap(S, e, W, where)) is not None]
        add(
            f"NGapSig{B}",
            str(sum(int(g["p"] < SIG and g["gap_full_minus_har"] < 0) for g in gs)),
            f"{where}: model-window cells with the full set significantly more accurate, p < {SIG}",
        )
        add(f"NGapCells{B}", str(len(gs)), f"{where}: model-window cells computed")
        if gs:
            add(
                f"GapTMin{B}",
                RW.mnum(min(g["t"] for g in gs), 2),
                f"{where}: smallest gap t",
            )
            add(
                f"GapTMax{B}",
                RW.mnum(max(g["t"] for g in gs), 2),
                f"{where}: largest gap t",
            )
    return M


def table(S: dict) -> tuple[str, int]:
    """tab:rh_fip_lc and the number of table rows marked pending."""
    lines: list[str] = []
    n_marked = 0

    def row(cells: list[str]) -> None:
        nonlocal n_marked
        ln = " & ".join(cells) + r" \\"
        if all(c in (PENDING_TEX, "") for c in cells[1:]):
            ln += f" {ROW_MARK}"
            n_marked += 1
        lines.append(ln)

    stars = lambda x, p: RW.tcell(None if x is None else SCALE * x, 2, p)  # noqa: E731
    paren = lambda t: RW.tcell(t, 2, None, paren=True)  # noqa: E731
    for k, (where, B) in enumerate(BARS):
        if k:
            lines.append(r"\addlinespace[4pt]")
        lines.append(
            rf"\multicolumn{{8}}{{@{{}}l}}{{\textit{{Panel {'AB'[k]}: {'every bar' if B == 'Bar' else 'the 16:00 bar'}}}}} \\"
        )
        for e in TREES:
            gs = [gap(S, e, W, where) for W in WINDOWS]
            ds = [did(S, e, a, b, where) for a, b in STEPS]
            row(
                [NAME[e]]
                + [
                    stars(
                        None if g is None else g["gap_full_minus_har"],
                        None if g is None else g["p"],
                    )
                    for g in gs
                ]
                + [
                    stars(
                        None if r is None else r["extra_gain"],
                        None if r is None else r["p"],
                    )
                    for r in ds
                ]
            )
            row(
                [""]
                + [paren(None if g is None else g["t"]) for g in gs]
                + [paren(None if r is None else r["t"]) for r in ds]
            )
    W1, W2, W3, W4 = (rf"\rhlcW{WTAG[W]}" for W in WINDOWS)
    pend = [f"{e} at W = {W:,}" for W in WINDOWS for e in S["pending"][W]]
    body = [
        HEAD,
        "% tab:rh_fip_lc (Section 8, rehaul/08b_fi_measures_floats.tex): the pooled LightGBM, XGBoost and random forest",
        "% across rolling windows of 1,000 / 2,000 / 3,000 / 3,500 trading days (every-bar walks, final design): the gap",
        "% full - HAR by window and the extra gain between windows, with Newey-West t.  Runs:",
        *[
            f"%   W {W:,}: {fr.rel(tree_run('lgbm', W)).replace(chr(92), '/')}, {fr.rel(tree_run('rf', W)).replace(chr(92), '/')}"
            for W in WINDOWS
        ],
        f"% Pending: {', '.join(pend) or 'none'}.",
        "% The pending macro comes from the numbers file (a second definition here would stop the Overleaf flattener).",
        r"\ifdefined\rhlcNDays\else\input{generated/rehaul_fip_lc_numbers}\fi",
        r"\begin{table}[tbp]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{threeparttable}",
        r"\caption{Does the full predictor set gain more from a longer window? The tree ensembles across training windows}",
        r"\label{tab:rh_fip_lc}",
        r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrrrr@{}}",
        r"\toprule",
        r" & \multicolumn{4}{c}{Gap, full minus HAR, by window $W$} & \multicolumn{3}{c}{Extra gain, from $W_1$ to $W_2$} \\",
        r"\cmidrule(lr){2-5}\cmidrule(l){6-8}",
        rf"Model & ${W1}$ & ${W2}$ & ${W3}$ & ${W4}$ & ${W1}\to{W2}$ & ${W2}\to{W3}$ & ${W3}\to{W4}$ \\",
        r"\midrule",
        *lines,
        r"\bottomrule",
        r"\end{tabular*}",
        r"\begin{tablenotes}[flushleft]",
        r"\footnotesize",
        # 2026-10-28 (author: prune "pooled"). Was: r"\item \textit{Notes:} The pooled LightGBM, XGBoost and random
        # forest on the HAR benchmark and on the full "
        r"\item \textit{Notes:} The LightGBM, XGBoost and random forest on the HAR benchmark and on the full "
        r"predictor set, refit at every bar on rolling windows of $W$ trading days, each on its frozen configuration, "
        r"first evaluation day \rhlcFirstDay; QLIKE on the \rhlcNBars{} bars of the \rhlcNDays{} evaluation days "
        r"(the scored set of Table~\ref{tab:rh_oos}, the same at every window; Panel B: their 16:00 bars), in units of "
        r"$10^{-3}$. Gap: the mean over days of the full set's daily QLIKE minus the HAR benchmark's (negative: the "
        r"full set is more accurate). Extra gain: the mean over days of the change of the full set's daily QLIKE from "
        r"$W_1$ to $W_2$ minus the HAR benchmark's change (negative: the full set gains more from the longer window). "
        r"In parentheses: Newey--West $t$-statistics (Bartlett kernel, lag \rhlcHacLag); stars: two-sided normal $p$ "
        r"below 10\%, 5\% and 1\%. Descriptive: no rule fixed this comparison before the walks.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
    ]
    return "\n".join(body) + "\n", n_marked


def main() -> None:
    t0 = time.time()
    print(fr.tag())
    S = score()
    out = fr.res_dir(RESULTS / "rehaul_fip_lc_final")
    out.mkdir(parents=True, exist_ok=True)
    for name, key in (
        ("qlike_by_w", "Q"),
        ("dm_pairs", "DM"),
        ("gap_by_w", "G"),
        ("gap_change_did", "DD"),
    ):
        S[key].to_csv(out / f"{name}.csv", index=False)
    selfcheck(S, out)
    claims(S)
    gen = fr.GEN
    gen.mkdir(parents=True, exist_ok=True)
    M = macros(S)
    names = [n for n, _, _ in M]
    assert len(set(names)) == len(names), "a macro name twice"
    pend = [f"{e} at W = {W:,}" for W in WINDOWS for e in S["pending"][W]]
    num = [
        HEAD,
        "% \\rhlc... macros of Section 8's tree learning curve (rehaul/08_feature_importance.tex, tab:rh_fip_lc): the pooled",
        "% trees at W = 1,000 / 2,000 / 3,000 / 3,500 trading days; names: Gap / GapT (full - HAR, x 10^3, NW t), Eg / T / P",
        "% (extra gain x 10^3, NW t, p statement), DmFull / DmHar (DM between windows); Model Lgbm Xgb Rf; windows One Two",
        "% Main Long (step tags OneTwo TwoMain MainLong OneMain); Bar (every bar) / Close (16:00). Counts at p < 0.05.",
        f"% Pending: {', '.join(pend) or 'none'}.",
        PENDING_DEF,
    ]
    for name, value, note in M:
        num.append(f"\\providecommand{{\\{name}}}{{}}\\renewcommand{{\\{name}}}{{{value}}}  % {note}")
    (gen / "rehaul_fip_lc_numbers.tex").write_text("\n".join(num) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {fr.rel(gen / 'rehaul_fip_lc_numbers.tex')} ({len(M)} macros)")
    tex, n_marked = table(S)
    (gen / "rehaul_fip_lc_table.tex").write_text(tex, encoding="utf-8", newline="\n")
    print(f"wrote {fr.rel(gen / 'rehaul_fip_lc_table.tex')} ({n_marked} pending rows); {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
