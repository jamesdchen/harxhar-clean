"""Scores of the ablation run (results/ablations_2014start_w3000/RULE.txt section 7), on the laptop.

Reads the merged forecasts the cluster wrote, compacted (results/ablations_2014start_w3000/forecasts/linear_wide.parquet
and trees_wide.parquet: date and one pred_adj column per "<class>|<arm>", from ablate_linear / ablate_trees merge) and
the templates (the final run: linear w3000,
LightGBM / XGBoost, the forest on oneDAL), and scores every (class, arm) the way Table 1 does
(writeup/make_rehaul_oos_rerun.py): plain = pred_adj^2 x B against the bar's actual RV (data/core_stats.parquet sumret2),
QLIKE = r - ln r - 1 (r = RV / plain) and squared error, Diebold-Mariano (src/evaluation/diebold_mariano.dm_test) on
daily losses.

Spans  all       every bar of the 2,579 evaluation days 2014-01-02 .. 2024-04-30 (NYSE days with a positive 16:00 print,
                 early closes dropped; bars with RV > 0 and a finite positive forecast of every (class, arm)); DM on the
                 daily average loss
       c16       the 16:00 bar of those days
       c16_2020  the 16:00 bar on 2020-01-03 .. 2024-04-30 (1,082 days)
Pairs  within class: every arm vs the class's full set and vs its HAR benchmark (the OLS: vs its HAR); across classes:
       each class's full set vs each OLS augmented-HAR benchmark (add_rm, shar, harq, lhar, sharql) and vs the OLS HAR.
Checks the template rows reproduce results/oos_final_2014start/table1_w3000/metrics.csv (QLIKE, MSE, every bar; the
       16:00 QLIKE on 2020-2024) to 1e-12 relative and its full-vs-HAR QLIKE DM to 1e-9; the scored set is Table 1's.

usage: python experiments/ablate_score.py [--res results/ablations_2014start_w3000]
       -> <res>/scores/metrics.csv, dm.csv, checks.json, latestart_groups.csv (when <res>/latestart/ exists)
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

from src.evaluation.diebold_mariano import dm_test  # noqa: E402

FIRST, LAST, N_DAYS = pd.Timestamp("2014-01-02"), pd.Timestamp("2024-04-30"), 2579
SUB_FIRST, SUB_DAYS = pd.Timestamp("2020-01-03"), 1082
LIN_T = ROOT / "results" / "window_sweep_2014start_2026-10-22" / "w3000" / "forecasts.parquet"
TREE_T = ROOT / "results" / "pooled_trees_final_2014start_w3000" / "forecasts.parquet"
RF_T = ROOT / "results" / "pooled_rf_skx_2014start_w3000" / "forecasts.parquet"
TABLE1 = ROOT / "results" / "oos_final_2014start" / "table1_w3000"
SET_ARM = {"all_features": "full", "baseline": "har"}
LINEAR = ("ridge", "lasso", "enet")
TREES = ("lgbm", "xgb", "rf")
CLASSES = ("ols", *LINEAR, *TREES)
EST_OF = {"lgbm": "lgbm", "xgb": "xgb", "rf": "rf_skx"}
OLS_BENCH = ("add_rm", "shar", "harq", "lhar", "sharql", "add_rm_fomc")  # addendum F: + add_rm_fomc
# 2026-10-26 addendum F (requested by the author on 26 October, after the channel results; not pre-registered):
# HAR + realized moments + FOMC, every class, on the patched design; scored on every bar, ORDINARY and EVENT days
COMBO = "add_rm_fomc"
N_EVENT = 81  # results/channels_2026-10-26/RULE.txt section 0: the scheduled FOMC statement days of the span
# 2026-10-26 (coordinator): live-window subsamples of the late-starting groups -- the bars from the date on which the
# listed series' k = 1 value averages are all live (latestart/latestart_w3000.csv "live_from") to the end of the span,
# or (end_at_last) to the last print of the series (volatility demand stops on 2023-08-31); key -> (series, group,
# end_at_last).  Spans live_<key> (every bar; DM on daily averages) and live16_<key> (the 16:00 bar).
LIVE = {
    "sent": (["stocktwits_attention", "stocktwits_sentiment", "stocktwits_sentcount"], "sent", False),
    "iv": (["vix"], "iv", False),
    "ivb": (["vix", "vix3m"], "iv", False),
    "vd": (
        [
            "voldemand_spx_open_and_close",
            "voldemand_spx_open_only",
            "voldemand_all_open_and_close",
            "voldemand_all_open_only",
        ],
        "vd",
        True,
    ),
}


def say(msg: str) -> None:
    print(msg, flush=True)


def core_rv() -> pd.Series:
    c = pd.read_parquet(ROOT / "data" / "core_stats.parquet", columns=["endbartime", "sumret2"])
    s = pd.Series(c["sumret2"].to_numpy(float), index=pd.DatetimeIndex(pd.to_datetime(c["endbartime"])))
    return s[~s.index.duplicated()].sort_index()


def evaluation_days(rv: pd.Series) -> pd.DatetimeIndex:
    import atm_straddle_lib as asl

    early = pd.DatetimeIndex(pd.to_datetime(list(asl.EARLY_CLOSE_DATES)))
    s = rv[(rv.index.hour == 16) & (rv.index.minute == 0)]
    d = pd.DatetimeIndex(s.index.normalize())
    keep = (s.to_numpy() > 0) & np.isfinite(s.to_numpy()) & ~d.isin(early)
    days = d[keep]
    days = days[(days >= FIRST) & (days <= LAST)]
    assert len(days) == N_DAYS and days[0] == FIRST and days[-1] == LAST, (len(days), days[0], days[-1])
    return days


def event_days(days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """EVENT days, results/channels_2026-10-26/RULE.txt section 0 (writeup/make_rehaul_channels.py statement_stamps):
    the bars of data/releases.parquet 'fomc release' > 0, plus loading.FOMC_MISSING_STATEMENTS, minus
    loading.UNSCHEDULED_FOMC_DAYS, on Table 1's scored days -- 81 (asserted)."""
    from src.data import loading

    rel = pd.read_parquet(ROOT / "data" / "releases.parquet", columns=["endbartime", loading.FOMC_RAW_COL])
    t = pd.DatetimeIndex(pd.to_datetime(rel["endbartime"]))
    raw = np.nan_to_num(pd.to_numeric(rel[loading.FOMC_RAW_COL], errors="coerce").to_numpy(float), nan=0.0)
    st = t[raw > 0].append(pd.DatetimeIndex(pd.to_datetime(list(loading.FOMC_MISSING_STATEMENTS))))
    st = st[~pd.DatetimeIndex(st.normalize()).isin(list(loading.UNSCHEDULED_FOMC_DAYS))]
    ev = pd.DatetimeIndex(st.normalize()).unique()
    ev = ev[ev.isin(days)].sort_values()
    assert len(ev) == N_EVENT, len(ev)
    return ev


FIX = ROOT / "results" / "fomcfix_2026-10-26" / "forecasts"
RF_FIX = ROOT / "results" / "pooled_rf_skx_2014start_fomcfix_w3000" / "forecasts.parquet"
# 2026-10-26 addendum D: on the FOMC-patched design, the trees' lo_<g> arms for g != fomc hold the pre-patch FOMC
# columns and stay pre-patch; they are compared with the pre-patch full set (key "full_pre")
PRE_TREE = tuple(f"lo_{g}" for g in ("rm", "liq", "ew", "vw", "sent", "iv", "vd"))


def ref_full(design: str, cls: str, arm: str) -> str:
    return "full_pre" if design == "fix" and cls in TREES and arm in PRE_TREE else "full"


def load_panel(res: Path, design: str = "pre", rf_fix: str = "") -> tuple[dict[tuple[str, str], pd.Series], pd.Series]:
    """(class, arm) -> plain forecast indexed by bar end; and B of the bars (from the templates).  design "fix"
    (addendum D): the full-set rows from the patched walks (fomcfix lin_B, forecasts_gb, rf_fix if given), the linear
    ablations from linear_wide_fix.parquet, the trees' add_fomc from trees_wide_fix.parquet, the pre-patch tree full
    sets kept as "full_pre"."""
    out: dict[tuple[str, str], pd.Series] = {}
    lt = pd.read_parquet(LIN_T, columns=["model", "predictor_set", "date", "pred_adj", "B", "plain"])
    for (m, s), f in lt.groupby(["model", "predictor_set"]):
        out[(m, SET_ARM[s])] = pd.Series(f["plain"].to_numpy(), index=pd.DatetimeIndex(f["date"]))
    tt = pd.concat([pd.read_parquet(TREE_T), pd.read_parquet(RF_T)], ignore_index=True)
    tt["model"] = tt["model"].replace({"rf_skx": "rf"})
    for (m, s), f in tt.groupby(["model", "predictor_set"]):
        out[(m, SET_ARM[s])] = pd.Series(f["plain"].to_numpy(), index=pd.DatetimeIndex(f["date"]))
    B = tt[(tt["model"] == "lgbm") & (tt["predictor_set"] == "all_features")].set_index("date")["B"]
    lb = lt[(lt["model"] == "ridge") & (lt["predictor_set"] == "all_features")].set_index("date")["B"]
    assert np.array_equal(lb.reindex(B.index).to_numpy(), B.to_numpy()), "linear and tree templates' B"
    if design == "fix":
        for c in TREES:
            out[(c, "full_pre")] = out[(c, "full")]
        del out[("rf", "full")]  # the forest's patched full set only when given
        lf = pd.read_parquet(FIX / "lin_B.parquet")
        for m, f in lf.groupby("model"):
            assert (f["predictor_set"] == "all_features").all()
            out[(m, "full")] = pd.Series(
                f["pred_adj"].to_numpy() ** 2 * f["B"].to_numpy(), index=pd.DatetimeIndex(f["date"])
            )
        gb = pd.read_parquet(FIX / "forecasts_gb.parquet")
        for m, f in gb.groupby("model"):
            out[(m, "full")] = pd.Series(f["plain"].to_numpy(), index=pd.DatetimeIndex(f["date"]))
        if rf_fix:
            rf = pd.read_parquet(rf_fix)
            rf = rf[(rf["model"] == "rf_skx") & (rf["predictor_set"] == "all_features")]
            assert len(rf) == 33970, len(rf)
            out[("rf", "full")] = pd.Series(rf["plain"].to_numpy(), index=pd.DatetimeIndex(rf["date"]))
    # addendum E: the forest's lo_fomc (design-invariant, gate F0) in its own file, read in both designs
    names: tuple[str, ...] = ("linear_wide.parquet", "trees_wide.parquet", "trees_wide_rflofomc.parquet")
    if design == "fix":
        names = (
            "linear_wide_fix.parquet",
            "trees_wide.parquet",
            "trees_wide_fix.parquet",
            "trees_wide_rflofomc.parquet",
            "linear_wide_rmfomc.parquet",  # addendum F
            "trees_wide_rmfomc.parquet",
            "linear_wide_beyond.parquet",  # addendum G
            "trees_wide_beyond.parquet",
            "trees_wide_ladder.parquet",  # addendum I
        )
    # the merged ablation forecasts, compact: one column of pred_adj per "<class>|<arm>" (2014-01-02 on)
    for name in names:
        p = res / "forecasts" / name
        if not p.is_file():
            continue
        w = pd.read_parquet(p)
        d = pd.DatetimeIndex(w["date"])
        b = B.reindex(d).to_numpy()
        assert np.isfinite(b).all(), name
        for col in w.columns[1:]:
            cls, arm = col.split("|")
            cls = {"rf_skx": "rf"}.get(cls, cls)
            if design == "fix" and name == "trees_wide.parquet" and arm == "add_fomc":
                continue  # the patched add_fomc comes from trees_wide_fix.parquet
            assert (cls, arm) not in out, (cls, arm)
            out[(cls, arm)] = pd.Series(w[col].to_numpy() ** 2 * b, index=d)
    return out, B


def scored(panel: dict, rv: pd.Series, days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = None
    for f in panel.values():
        ok = f.index[np.isfinite(f.to_numpy()) & (f.to_numpy() > 0)]
        idx = ok if idx is None else idx.intersection(ok)
    assert idx is not None
    idx = idx[pd.DatetimeIndex(idx.normalize()).isin(days)]
    t = rv.reindex(idx)
    return idx[(t.to_numpy() > 0) & np.isfinite(t.to_numpy())].sort_values()


def losses(f: pd.Series, idx: pd.DatetimeIndex, rv: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    p, y = f.reindex(idx).to_numpy(float), rv.reindex(idx).to_numpy(float)
    r = y / p
    return r - np.log(r) - 1.0, (y - p) ** 2


def daily(x: np.ndarray, idx: pd.DatetimeIndex) -> np.ndarray:
    return pd.Series(x, index=pd.DatetimeIndex(idx.normalize())).groupby(level=0).mean().to_numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=str(ROOT / "results" / "ablations_2014start_w3000"))
    ap.add_argument("--design", default="pre", choices=("pre", "fix"), help="fix: the FOMC-patched panel (addendum D)")
    ap.add_argument(
        "--rf-fix",
        default=str(RF_FIX) if RF_FIX.is_file() else "",
        help="the forest's patched full-set forecasts (fomcfix ed43a18; default: that run's file when present)",
    )
    a = ap.parse_args()
    res = Path(a.res)
    rv = core_rv()
    days = evaluation_days(rv)
    panel, _ = load_panel(res, a.design, a.rf_fix)
    templ = {k: v for k, v in panel.items() if k[1] in ("full", "har", "full_pre")}
    idx_t = scored(templ, rv, days)
    idx = scored(panel, rv, days)
    checks: dict = {"n_bars_templates": len(idx_t), "n_bars_all": len(idx), "fail": []}
    if not idx.equals(idx_t):
        checks["fail"].append(f"an ablation arm drops {len(idx_t.difference(idx))} bars of Table 1's scored set")
    spans = {
        "all": idx,
        "c16": idx[(idx.hour == 16) & (idx.minute == 0)],
    }
    spans["c16_2020"] = spans["c16"][spans["c16"].normalize() >= SUB_FIRST]
    if any(k[1] == COMBO for k in panel):  # addendum F: every bar of ORDINARY / EVENT days (DM on daily averages)
        ev = event_days(days)
        on_ev = pd.DatetimeIndex(idx.normalize()).isin(ev)
        spans["ord"], spans["ev"] = idx[~on_ev], idx[on_ev]
        checks["day_types"] = dict(
            event_days=len(ev),
            ordinary_days=int(pd.DatetimeIndex(spans["ord"].normalize()).nunique()),
            event_bars=len(spans["ev"]),
            ordinary_bars=len(spans["ord"]),
            rule="results/channels_2026-10-26/RULE.txt section 0",
            label="add_rm_fomc: requested by the author on 26 October, after the channel results; not pre-registered",
        )
    assert len(spans["c16"]) == N_DAYS and len(spans["c16_2020"]) == SUB_DAYS, (
        len(spans["c16"]),
        len(spans["c16_2020"]),
    )
    ls_p = res / "latestart" / "latestart_w3000.csv"
    if ls_p.is_file():
        LS = pd.read_csv(ls_p)
        k1 = LS[LS["k"] == 1].set_index("series")
        checks["live_windows"] = {}
        for key, (ser, _g, end_last) in LIVE.items():
            start = max(pd.Timestamp(k1.loc[s_, "live_from"]) for s_ in ser)
            end = min(pd.Timestamp(k1.loc[s_, "last_print"]) for s_ in ser) if end_last else LAST + pd.Timedelta(days=1)
            ix = idx[(idx >= start) & (idx <= end)]
            spans[f"live_{key}"] = ix
            spans[f"live16_{key}"] = ix[(ix.hour == 16) & (ix.minute == 0)]
            checks["live_windows"][key] = dict(
                start=str(start), end=str(min(end, idx[-1])), bars=len(ix), days=int(ix.normalize().nunique())
            )
    L: dict = {}
    met = []
    for k, f in sorted(panel.items()):
        for sp, ix in spans.items():
            q, se = losses(f, ix, rv)
            L[(k, sp)] = (q, se, ix)
            met.append(
                dict(
                    cls=k[0],
                    arm=k[1],
                    span=sp,
                    n_bars=len(ix),
                    n_days=int(pd.DatetimeIndex(ix.normalize()).nunique()),
                    qlike=float(q.mean()),
                    mse=float(se.mean()),
                )
            )
    M = pd.DataFrame(met)

    def dm(k1: tuple[str, str], k2: tuple[str, str], sp: str) -> list[dict]:
        out = []
        for li, lname in ((0, "qlike"), (1, "mse")):
            x1, x2, ix = L[(k1, sp)][li], L[(k2, sp)][li], L[(k1, sp)][2]
            every_bar = sp in ("all", "ord", "ev") or sp.startswith("live_")
            r = dm_test(daily(x1, ix), daily(x2, ix)) if every_bar else dm_test(x1, x2)
            out.append(
                dict(
                    cls=k1[0],
                    arm=k1[1],
                    vs_cls=k2[0],
                    vs_arm=k2[1],
                    span=sp,
                    loss=lname,
                    dm=float(r["dm"]),
                    p=float(r["p"]),
                    mean_diff=float(r["mean_diff"]),
                    T=int(r["T"]),
                    hac_lag=int(r.get("hac_lag", -1)),
                )
            )
        return out

    rows = []
    for c, arm in sorted(panel):
        for sp in spans:
            if sp.startswith("live"):  # the late group's own arms only (and the full set against HAR)
                g = LIVE[sp.split("_", 1)[1]][1]
                if c == "ols":
                    continue
                if arm == f"lo_{g}" and (c, ref_full(a.design, c, arm)) in panel:
                    rows += dm((c, arm), (c, ref_full(a.design, c, arm)), sp)
                elif arm in (f"add_{g}", "full", "full_pre"):
                    rows += dm((c, arm), (c, "har"), sp)
                continue
            if c == "ols":
                if arm != "har":
                    rows += dm((c, arm), ("ols", "har"), sp)
                continue
            rf_ = ref_full(a.design, c, arm)
            if arm not in ("full", "full_pre") and (c, rf_) in panel:
                rows += dm((c, arm), (c, rf_), sp)
            if arm != "har":
                rows += dm((c, arm), (c, "har"), sp)
            if arm == COMBO and ("ols", "har") in panel:  # addendum F: against the OLS on the HAR set (Table 1's D2)
                rows += dm((c, arm), ("ols", "har"), sp)
            if (arm.startswith(COMBO + "_") or arm.startswith("brung_")) and (c, COMBO) in panel:
                # addendum G: base + g against the base; addendum I: the ladder's rungs against the base
                rows += dm((c, arm), (c, COMBO), sp)
            if arm == "full":
                for b in (*OLS_BENCH, "har"):
                    if ("ols", b) in panel:
                        rows += dm((c, "full"), ("ols", b), sp)
    D = pd.DataFrame(rows)
    # checks against Table 1
    t1 = pd.read_csv(TABLE1 / "metrics.csv")
    d1 = pd.read_csv(TABLE1 / "dm.csv")
    for _, r in t1.iterrows():
        k = (r["model"], SET_ARM[r["set"]])
        if a.design == "fix" and k[1] == "full":  # the pre-patch full rows: kept as full_pre for the trees only
            if k[0] not in TREES:
                continue
            k = (k[0], "full_pre")
        mine = M[(M.cls == k[0]) & (M.arm == k[1])].set_index("span")
        for col, sp, ref in (
            ("qlike", "all", r["qlike"]),
            ("mse", "all", r["mse"]),
            ("qlike", "c16_2020", r["qlike_1600"]),
        ):
            v = float(mine.loc[sp, col])
            if not np.isclose(v, ref, rtol=1e-12, atol=0):
                checks["fail"].append(f"Table 1 {k} {col} {sp}: {v} vs {ref}")
        if int(mine.loc["all", "n_bars"]) != int(r["n_intervals"]):
            checks["fail"].append(f"Table 1 {k}: bars {int(mine.loc['all', 'n_bars'])} vs {int(r['n_intervals'])}")
    for _, r in d1[(d1["comparison"] == "all features vs HAR benchmark") & (d1["loss"] == "qlike")].iterrows():
        if a.design == "fix" and r["model"] not in TREES:
            continue
        mine = D[
            (D.cls == r["model"])
            & (D.arm == ("full_pre" if a.design == "fix" else "full"))
            & (D.vs_cls == r["model"])
            & (D.vs_arm == "har")
            & (D.span == "all")
            & (D.loss == "qlike")
        ]
        if len(mine) != 1 or not np.isclose(float(mine["dm"].iloc[0]), r["dm"], rtol=1e-9, atol=0):
            checks["fail"].append(f"Table 1 DM {r['model']}: {mine['dm'].tolist()} vs {r['dm']}")
    if a.design == "fix":  # the patched linear full rows reproduce fomcfix's cell-B scores
        fm = pd.read_csv(ROOT / "results" / "fomcfix_2026-10-26" / "scores_linear" / "metrics.csv")
        fm = fm[(fm["cell"] == "B") & (fm["set"] == "all_features")]
        for _, r in fm.iterrows():
            mine = M[(M.cls == r["model"]) & (M.arm == "full")].set_index("span")
            for col, sp, ref in (
                ("qlike", "all", r["qlike_bar"]),
                ("mse", "all", r["mse_bar_e10"] * 1e-10),
                ("qlike", "c16", r["qlike_16"]),
                ("qlike", "c16_2020", r["qlike_16_2020"]),
            ):
                v = float(mine.loc[sp, col])
                if not np.isclose(v, ref, rtol=1e-12, atol=0):
                    checks["fail"].append(f"fomcfix B {r['model']} {col} {sp}: {v} vs {ref}")
        checks["design"] = "fix: FOMC-patched (fomcfix cell B); trees' lo_<g> (g != fomc) pre-patch vs full_pre"
        checks["rf_full_fix"] = bool(a.rf_fix)
    checks["pass"] = not checks["fail"]
    checks["arms"] = sorted(f"{c}/{a_}" for c, a_ in panel)
    out = res / ("scores_fomcfix" if a.design == "fix" else "scores")
    out.mkdir(parents=True, exist_ok=True)
    M.to_csv(out / "metrics.csv", index=False)
    D.to_csv(out / "dm.csv", index=False)
    # late-start: share of the evaluation rows on which each late series' value columns are live
    ls = res / "latestart" / "latestart_w3000.csv"
    if ls.is_file():
        T = pd.read_csv(ls)
        g = T.groupby(["group", "series"], sort=False).agg(
            first_print=("first_print", "first"),
            last_print=("last_print", "first"),
            live_from_k1=("live_from", "first"),
            live_from_k3125=("live_from", "last"),
            share_live_min=("eval_share_nonzero", "min"),
            share_live_max=("eval_share_nonzero", "max"),
            zero_prefix_max=("zero_prefix_rows", "max"),
        )
        g.reset_index().to_csv(out / "latestart_groups.csv", index=False)
        checks["latestart_rule_mismatches"] = int((T.get("rule_match", pd.Series(True, index=T.index)) == False).sum())  # noqa: E712
    (out / "checks.json").write_text(json.dumps(checks, indent=1))
    say(
        f"scored {len(panel)} (class, arm) forecasts on {len(idx)} bars; checks {'PASS' if checks['pass'] else 'FAIL'}: "
        f"{checks['fail'][:6]}"
    )
    return 0 if checks["pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
