"""Scorer of the all-48-interval study (results/all48_2026-10-30/RULE.txt section 3; EXPLORATORY, proposed after the
main results were seen, not pre-registered): every half-hour interval of the 48-interval grid scored as the paper scores
its 13 regular-hours bars (experiments/ablate_score.py): plain = pred_adj^2 x B against the interval's actual RV
(data/core_stats.parquet sumret2), QLIKE = r - ln r - 1 (r = RV / plain) on intervals with RV > 0, squared error, and
Diebold-Mariano (src/evaluation/diebold_mariano.dm_test: HLN, Bartlett) on the daily average loss.

Inputs (any combination; the part's panel = every model found, recorded in checks.json):
  --linear-root R   a linear run directory in the orchestrator's layout (experiments/pooled_linear_rerun_2026_10_19_13.py):
                    R/_work/<sub>/<set>/<model>/[c<k>/]causal_tune_linear/<runner model>/<set>/results.csv, e.g. Part 1's
                    train3way ALL48 arm or Part 2's run (experiments/all48_linear.py).  Every chunk of a model is read
                    (c0, c1, .., then a top-level table if present), stitched in date order, and duplicate rows dropped
                    after asserting they agree (true_adj, pred_adj).  Columns read: date, true_adj, pred_adj, true_raw.
                    B = the design cache's b_oos (--design, default R/_work when it holds design_baseline/b_oos.npy),
                    asserted equal to true_raw / true_adj^2 (1e-9 relative) where true_adj > 0; without a design cache
                    B = true_raw / true_adj^2.
  --linear-parquet F / --trees F (repeatable)
                    long forecast files with columns model, predictor_set, date, pred_adj, B (experiments/all48_linear.py
                    export, experiments/all48_trees.py merge), read one model at a time (column subsets, row filters).

Spans (calendar date of the bar-end stamp, ET; E = the evaluation days: NYSE days in --first .. --last with a positive
16:00 print, early closes dropped -- Table 1's 2,579 days, asserted for the default span):
  all48        every interval of the days in E        rth13      end stamps 10:00 .. 16:00 of those days
  non_rth35    the other 35 end stamps of those days  clk_HHMM   one end stamp (per interval, 48 spans)
  grid, grid_non_rth (secondary)  every row dated --first .. --last (Sundays, holidays, early closes included)
Scored set: the span's intervals with RV > 0 and a finite positive plain of EVERY model of the panel.
DM pairs: full vs HAR within each class; secondary: each full set vs the OLS HAR.  Negative = the first-named lower.

Part 1 reproduction check (--ref-scores D, train3way's scores_all/): the rth13 span reproduces qlike_bar.csv (variant
ALL48: bars, QLIKE, MSE x 1e10; 1e-12 relative) and gap.csv (every bar -> rth13, 16:00 -> clk_1600: the HAR and full
QLIKE to 1e-12, the QLIKE DM statistic to 1e-9).  A failed check (or a coverage / duplicate failure) writes checks.json
and exits 3 before any table is written.

Memory: one model's table at a time (a few columns), the panel as one float Series a model (~130,000 rows each).

usage: python experiments/all48_score.py --out DIR [--part 1|2] [--linear-root R [--design D | --no-design]
       [--linear-sub walk]] [--linear-parquet F] [--trees F ...] [--core data/core_stats.parquet]
       [--first 2014-01-02] [--last 2024-04-30] [--ref-scores D] [--require cls/set ...] [--label TEXT]
       -> DIR/metrics.csv, dm.csv, per_interval.csv, checks.json, tables.txt
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

FIRST, LAST, N_DAYS = "2014-01-02", "2024-04-30", 2579
CLOCKS = tuple(f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 30))
RTH = tuple(c for c in CLOCKS if "10:00" <= c <= "16:00")  # 13 end stamps 10:00 .. 16:00
LIN_RUNNER = {"ols": "ols", "ridge": "ridge", "lasso": "reclasso", "enet": "reclasticnet"}  # paper -> runner name
LIN_MODELS = [
    ("baseline", "ols"),
    ("baseline", "ridge"),
    ("baseline", "lasso"),
    ("baseline", "enet"),
    ("all_features", "ridge"),
    ("all_features", "lasso"),
    ("all_features", "enet"),
]
SET_KEY = {"baseline": "har", "all_features": "full"}
CLS_OF = {
    "ols": "ols",
    "ridge": "ridge",
    "lasso": "lasso",
    "enet": "enet",
    "lgbm": "lgbm",
    "xgb": "xgb",
    "rf_skx": "rf",
}
CLASSES = ("ols", "ridge", "lasso", "enet", "lgbm", "xgb", "rf")
PAIR_CLASSES = ("ridge", "lasso", "enet", "lgbm", "xgb", "rf")
MAIN_SPANS = ("all48", "rth13", "non_rth35", "grid", "grid_non_rth")
TOL_Q, TOL_DM = 1e-12, 1e-9

Key = tuple[str, str]  # (class, "har" | "full")


def say(msg: str) -> None:
    print(msg, flush=True)


# ============================================================================ truth and days
def core_rv(path: Path) -> pd.Series:
    c = pd.read_parquet(path, columns=["endbartime", "sumret2"])
    s = pd.Series(c["sumret2"].to_numpy(float), index=pd.DatetimeIndex(pd.to_datetime(c["endbartime"])))
    return s[~s.index.duplicated()].sort_index()


def evaluation_days(rv: pd.Series, first: str, last: str) -> pd.DatetimeIndex:
    """experiments/ablate_score.evaluation_days on any span: days with a finite positive 16:00 print, NYSE early
    closes (notebooks/atm_straddle_lib.EARLY_CLOSE_DATES) dropped; 2,579 asserted on the default span."""
    import atm_straddle_lib as asl

    early = pd.DatetimeIndex(pd.to_datetime(list(asl.EARLY_CLOSE_DATES)))
    s = rv[(rv.index.hour == 16) & (rv.index.minute == 0)]
    d = pd.DatetimeIndex(s.index.normalize())
    v = s.to_numpy()
    days = d[(v > 0) & np.isfinite(v) & ~d.isin(early)]
    days = days[(days >= pd.Timestamp(first)) & (days <= pd.Timestamp(last))]
    if (first, last) == (FIRST, LAST):
        assert len(days) == N_DAYS, len(days)
    return days


# ============================================================================ forecasts
def walk_files(root: Path, sub: str, s: str, m: str) -> list[Path]:
    """Every results.csv of a linear walk: the chunks c0, c1, .. in order, then a top-level table if present."""
    base = root / "_work" / sub / s / m
    if not base.is_dir():
        base = root / sub / s / m
    chunks = sorted((p for p in base.glob("c*") if p.is_dir() and p.name[1:].isdigit()), key=lambda p: int(p.name[1:]))
    pat = f"causal_tune_linear/{LIN_RUNNER[m]}/{s}/results.csv"
    return [f for c in chunks for f in sorted(c.glob(pat))] + sorted(base.glob(pat))


def read_walk(files: list[Path]) -> tuple[pd.DataFrame, dict]:
    """The stitched walk table (date, true_adj, pred_adj, true_raw) and its record; duplicates must agree."""
    assert files, "no walk table"
    parts = []
    for k, p in enumerate(files):
        r = pd.read_csv(p, usecols=["date", "true_adj", "pred_adj", "true_raw"])
        r["part"] = k
        parts.append(r)
    r = pd.concat(parts, ignore_index=True)
    r["date"] = pd.to_datetime(r["date"])
    r = r.sort_values(["date", "part"], kind="stable", ignore_index=True)
    dup = r["date"].duplicated(keep=False)
    n_dup = int(r["date"].duplicated().sum())
    bad = 0
    if n_dup:
        g = r[dup].groupby("date")
        for col in ("true_adj", "pred_adj"):
            bad += int((g[col].nunique(dropna=False) > 1).sum())
    rec = dict(files=[str(p) for p in files], rows_read=len(r), duplicates_dropped=n_dup, duplicates_disagree=bad)
    if bad:
        raise SystemExit(f"walk tables disagree on {bad} duplicated rows: {rec}")
    r = r[~r["date"].duplicated()].drop(columns="part").reset_index(drop=True)
    rec |= dict(rows=len(r), first=str(r["date"].iloc[0]), last=str(r["date"].iloc[-1]))
    return r, rec


def design_b(design: Path) -> tuple[pd.Series, pd.DatetimeIndex]:
    dz = design / "design_baseline"
    d = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(dz / "dates_oos.csv")["date"]))
    b = np.load(dz / "b_oos.npy")
    assert len(b) == len(d) and d.is_unique, (len(b), len(d))
    return pd.Series(b, index=d), d


def linear_panel(root: Path, sub: str, design: Path | None, first: str, last: str) -> tuple[dict, dict]:
    """plain = pred_adj^2 x B of every linear model found under root, dated first .. last; the reading record."""
    B: pd.Series | None = None
    oos = pd.DatetimeIndex([])
    if design is not None:
        B, oos = design_b(design)
    panel: dict[Key, pd.Series] = {}
    rec: dict = {"root": str(root), "sub": sub, "design": str(design) if design else None, "models": {}}
    t0, t1 = pd.Timestamp(first), pd.Timestamp(last) + pd.Timedelta(days=1)
    for s, m in LIN_MODELS:
        files = walk_files(root, sub, s, m)
        if not files:
            continue
        r, rr = read_walk(files)
        t = pd.DatetimeIndex(r["date"])
        ta, pa, tr = (r[c].to_numpy(float) for c in ("true_adj", "pred_adj", "true_raw"))
        ok = ta > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            b_rec = np.where(ok, tr / ta**2, np.nan)
        if B is not None:
            b = B.reindex(t).to_numpy()
            rr["b_missing"] = int(np.isnan(b).sum())
            rr["b_max_rel_vs_true_raw"] = float(np.nanmax(np.abs(b_rec[ok] / b[ok] - 1.0))) if ok.any() else 0.0
            # coverage: the walk's rows are the design's OOS rows from its first row to the end, each once
            k0 = int(oos.get_indexer([t[0]])[0])
            rr["covers_design_to_end"] = bool(k0 >= 0 and t.equals(oos[k0:]))
            rr["b_source"] = "design b_oos"
        else:
            b = b_rec
            rr["covers_design_to_end"] = None
            rr["b_source"] = "true_raw / true_adj^2"
        rr["rows_by_clock"] = {c: int(n) for c, n in pd.Series(t.strftime("%H:%M")).value_counts().sort_index().items()}
        rr["clocks"] = len(rr["rows_by_clock"])
        keep = (t >= t0) & (t < t1)
        panel[(m, SET_KEY[s])] = pd.Series(pa[keep] ** 2 * b[keep], index=t[keep])
        rec["models"][f"{m}/{SET_KEY[s]}"] = rr
        say(f"linear {s}/{m}: {rr['rows']} rows ({rr['duplicates_dropped']} duplicates dropped), {rr['clocks']} clocks")
    return panel, rec


def parquet_panel(path: Path, first: str, last: str) -> tuple[dict, dict]:
    """plain of every (model, predictor_set) of a long forecast file, one model at a time."""
    import pyarrow.parquet as pq

    tb = pq.read_table(path, columns=["model", "predictor_set"])
    models = sorted(set(tb.column("model").unique().to_pylist()))
    sets = sorted(set(tb.column("predictor_set").unique().to_pylist()))
    del tb
    t0, t1 = pd.Timestamp(first), pd.Timestamp(last) + pd.Timedelta(days=1)
    panel: dict[Key, pd.Series] = {}
    rec: dict = {"file": str(path), "models": {}}
    for m in models:
        for s in sets:
            f = pd.read_parquet(
                path, columns=["date", "pred_adj", "B"], filters=[("model", "==", m), ("predictor_set", "==", s)]
            )
            if not len(f):
                continue
            t = pd.DatetimeIndex(pd.to_datetime(f["date"]))
            assert t.is_unique, (path, m, s)
            keep = (t >= t0) & (t < t1)
            k = (CLS_OF[m], SET_KEY[s])
            assert k not in panel, (path, k)
            v = f["pred_adj"].to_numpy(float) ** 2 * f["B"].to_numpy(float)
            panel[k] = pd.Series(v[keep], index=t[keep]).sort_index()
            rec["models"][f"{k[0]}/{k[1]}"] = dict(
                rows=int(keep.sum()),
                clocks=int(pd.Index(t[keep].strftime("%H:%M")).nunique()),
                first=str(t[keep].min()) if keep.any() else None,
                last=str(t[keep].max()) if keep.any() else None,
            )
            say(f"{path.name} {m}/{s}: {int(keep.sum())} rows")
    return panel, rec


# ============================================================================ scoring
def scored_index(panel: dict, rv: pd.Series) -> pd.DatetimeIndex:
    idx: pd.DatetimeIndex | None = None
    for f in panel.values():
        v = f.to_numpy()
        ok = f.index[np.isfinite(v) & (v > 0)]
        idx = ok if idx is None else idx.intersection(ok)
    assert idx is not None and len(idx), "empty panel"
    t = rv.reindex(idx).to_numpy()
    return idx[np.isfinite(t) & (t > 0)].sort_values()


def span_masks(idx: pd.DatetimeIndex, days: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    clk = np.asarray(idx.strftime("%H:%M"))
    on_e = pd.DatetimeIndex(idx.normalize()).isin(days)
    rth = np.isin(clk, RTH)
    sp = {
        "all48": on_e,
        "rth13": on_e & rth,
        "non_rth35": on_e & ~rth,
        "grid": np.ones(len(idx), dtype=bool),
        "grid_non_rth": ~rth,
    }
    for c in CLOCKS:
        sp[f"clk_{c.replace(':', '')}"] = on_e & (clk == c)
    return sp


def daily(x: np.ndarray, idx: pd.DatetimeIndex) -> np.ndarray:
    """experiments/ablate_score.daily: the mean loss of each calendar date (of the bar-end stamp)."""
    return pd.Series(x, index=pd.DatetimeIndex(idx.normalize())).groupby(level=0).mean().to_numpy()


def pairs(keys: list[Key]) -> list[tuple[Key, Key, str]]:
    out = []
    for c in PAIR_CLASSES:
        if (c, "full") in keys and (c, "har") in keys:
            out.append(((c, "full"), (c, "har"), "full vs HAR"))
    if ("ols", "har") in keys:
        for c in PAIR_CLASSES:
            if (c, "full") in keys:
                out.append(((c, "full"), ("ols", "har"), "full vs OLS HAR"))
    return out


def score(panel: dict, rv: pd.Series, days: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    idx = scored_index(panel, rv)
    y = rv.reindex(idx).to_numpy(float)
    masks = span_masks(idx, days)
    keys = sorted(panel, key=lambda k: (CLASSES.index(k[0]), k[1] != "har"))
    L: dict[Key, tuple[np.ndarray, np.ndarray]] = {}
    for k in keys:
        p = panel[k].reindex(idx).to_numpy(float)
        r = y / p
        L[k] = (r - np.log(r) - 1.0, (y - p) ** 2)
    met = []
    for sp, mk in masks.items():
        ix = idx[mk]
        nd = int(pd.DatetimeIndex(ix.normalize()).nunique())
        for k in keys:
            q, se = L[k][0][mk], L[k][1][mk]
            met.append(
                dict(
                    span=sp,
                    cls=k[0],
                    set=k[1],
                    n_bars=int(mk.sum()),
                    n_days=nd,
                    qlike=float(q.mean()) if len(q) else np.nan,
                    mse_e10=float(se.mean() * 1e10) if len(se) else np.nan,
                )
            )
    M = pd.DataFrame(met)
    rows = []
    for a, b, what in pairs(keys):
        for sp, mk in masks.items():
            if not mk.any():
                continue
            ix = idx[mk]
            qa, qb = L[a][0][mk].mean(), L[b][0][mk].mean()
            rec = dict(span=sp, comparison=what, cls=a[0], set=a[1], vs_cls=b[0], vs_set=b[1], qlike=qa, qlike_vs=qb)
            rec["pct"] = 100.0 * (qa / qb - 1.0)
            for li, lname in ((0, "qlike"), (1, "mse")):
                r = dm_test(daily(L[a][li][mk], ix), daily(L[b][li][mk], ix))
                rec |= {
                    f"dm_{lname}": float(r["dm"]),
                    f"p_{lname}": float(r["p"]),
                    f"mean_diff_{lname}": float(r["mean_diff"]),
                }
                rec["T"], rec["hac_lag"] = int(r["T"]), int(r.get("hac_lag", -1))
            rows.append(rec)
    D = pd.DataFrame(rows)
    info = dict(
        n_bars={sp: int(mk.sum()) for sp, mk in masks.items() if not sp.startswith("clk_")},
        n_bars_by_clock={sp[4:]: int(mk.sum()) for sp, mk in masks.items() if sp.startswith("clk_")},
        n_days_e=len(days),
        panel=[f"{k[0]}/{k[1]}" for k in keys],
        first_scored=str(idx[0]),
        last_scored=str(idx[-1]),
    )
    return M, D, info


def per_interval(M: pd.DataFrame, D: pd.DataFrame) -> pd.DataFrame:
    """One row a (clock, class): bars, QLIKE of HAR and full, % vs HAR, the full-vs-HAR DM (QLIKE, MSE)."""
    m = M[M["span"].str.startswith("clk_")]
    d = D[(D["span"].str.startswith("clk_")) & (D["comparison"] == "full vs HAR")] if len(D) else D
    out = []
    for sp in sorted(m["span"].unique()):
        for c in CLASSES:
            h = m[(m.span == sp) & (m.cls == c) & (m.set == "har")]
            f = m[(m.span == sp) & (m.cls == c) & (m.set == "full")]
            if not len(h) and not len(f):
                continue
            qh = float(h["qlike"].iloc[0]) if len(h) else np.nan
            qf = float(f["qlike"].iloc[0]) if len(f) else np.nan
            rec: dict[str, object] = dict(clock=f"{sp[4:6]}:{sp[6:]}", rth=f"{sp[4:6]}:{sp[6:]}" in RTH, cls=c)
            rec["n_bars"] = int((h if len(h) else f)["n_bars"].iloc[0])
            rec["qlike_har"], rec["qlike_full"] = qh, qf
            rec["mse_e10_har"] = float(h["mse_e10"].iloc[0]) if len(h) else np.nan
            rec["mse_e10_full"] = float(f["mse_e10"].iloc[0]) if len(f) else np.nan
            rec["pct"] = 100.0 * (qf / qh - 1.0)
            dd = d[(d.span == sp) & (d.cls == c)] if len(d) else d
            for col in ("dm_qlike", "p_qlike", "dm_mse", "p_mse", "T"):
                rec[col] = float(dd[col].iloc[0]) if len(dd) else np.nan
            out.append(rec)
    return pd.DataFrame(out)


# ============================================================================ checks and tables
def rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-300)


def ref_check(ref: Path, M: pd.DataFrame, D: pd.DataFrame) -> dict:
    """Part 1: the rth13 span and the 16:00 interval against train3way's scores_all (variant ALL48)."""
    out: dict = {"ref": str(ref), "fail": []}
    qb = pd.read_csv(ref / "qlike_bar.csv")
    qb = qb[qb["variant"] == "ALL48"]
    rows = []
    for r in qb.itertuples():
        mine = M[(M.span == "rth13") & (M.cls == r.model) & (M.set == SET_KEY[r.set])]
        if not len(mine):
            out["fail"].append(f"qlike_bar {r.model}/{r.set}: not in the panel")
            continue
        x = mine.iloc[0]
        nb = [int(x["n_bars"]), int(r.n_bars)]  # type: ignore[call-overload]
        rq = rel(float(x["qlike"]), float(r.qlike))  # type: ignore[arg-type]
        rm = rel(float(x["mse_e10"]), float(r.mse_e10))  # type: ignore[arg-type]
        e: dict[str, object] = dict(model=f"{r.model}/{r.set}", n_bars=nb, rel_qlike=rq, rel_mse=rm)
        e["pass"] = bool(nb[0] == nb[1] and rq <= TOL_Q and rm <= TOL_Q)
        if not e["pass"]:
            out["fail"].append(f"qlike_bar {e}")
        rows.append(e)
    out["qlike_bar"] = rows
    gp = pd.read_csv(ref / "gap.csv")
    gp = gp[gp["variant"] == "ALL48"]
    rows = []
    for r in gp.itertuples():
        sp = {"every bar": "rth13", "16:00": "clk_1600"}[r.target]
        mine = D[(D.span == sp) & (D.comparison == "full vs HAR") & (D.cls == r.model)]
        if not len(mine):
            out["fail"].append(f"gap {r.target} {r.model}: no full-vs-HAR pair")
            continue
        x = mine.iloc[0]
        rh = rel(float(x["qlike_vs"]), float(r.qlike_har))  # type: ignore[arg-type]
        rf = rel(float(x["qlike"]), float(r.qlike_full))  # type: ignore[arg-type]
        dm = [float(x["dm_qlike"]), float(r.dm_qlike)]  # type: ignore[arg-type]
        e = dict(target=r.target, model=r.model, rel_qlike_har=rh, rel_qlike_full=rf, dm_qlike=dm)
        dm_ok = abs(dm[0] - dm[1]) <= TOL_DM * max(1.0, abs(dm[1]))
        e["pass"] = bool(rh <= TOL_Q and rf <= TOL_Q and dm_ok)
        if not e["pass"]:
            out["fail"].append(f"gap {e}")
        rows.append(e)
    out["gap"] = rows
    out["pass"] = not out["fail"]
    return out


def fmt_tables(M: pd.DataFrame, D: pd.DataFrame, P: pd.DataFrame, label: str) -> str:
    lines = [
        f"{label}",
        "EXPLORATORY (results/all48_2026-10-30/RULE.txt): proposed after the main results were seen; not "
        "pre-registered.  No correction for the 48 per-interval tests.",
        "",
    ]
    for sp in MAIN_SPANS:
        m = M[M.span == sp]
        if not len(m):
            continue
        lines.append(f"== {sp}: {int(m['n_bars'].iloc[0])} bars on {int(m['n_days'].iloc[0])} days")
        lines.append(f"{'class':6s} {'QLIKE HAR':>11s} {'QLIKE full':>11s} {'% vs HAR':>9s} {'DM':>8s} {'p':>8s}")
        for c in CLASSES:
            h = m[(m.cls == c) & (m.set == "har")]
            f = m[(m.cls == c) & (m.set == "full")]
            if not len(h) and not len(f):
                continue
            qh = float(h["qlike"].iloc[0]) if len(h) else np.nan
            qf = float(f["qlike"].iloc[0]) if len(f) else np.nan
            d = D[(D.span == sp) & (D.comparison == "full vs HAR") & (D.cls == c)] if len(D) else D
            dm = float(d["dm_qlike"].iloc[0]) if len(d) else np.nan
            p = float(d["p_qlike"].iloc[0]) if len(d) else np.nan
            pct = 100.0 * (qf / qh - 1.0) if len(h) and len(f) else np.nan
            lines.append(f"{c:6s} {qh:11.5f} {qf:11.5f} {pct:9.2f} {dm:8.2f} {p:8.4f}")
        lines.append("")
    if len(P):
        lines.append("== per interval: % vs HAR (full-vs-HAR QLIKE DM statistic)")
        cls = [c for c in PAIR_CLASSES if c in set(P["cls"])]
        lines.append(f"{'clock':6s} {'bars':>6s} " + " ".join(f"{c:>16s}" for c in cls))
        for clock, g in P.groupby("clock", sort=True):
            cells = []
            for c in cls:
                x = g[g.cls == c]
                cells.append(f"{x['pct'].iloc[0]:7.2f} ({x['dm_qlike'].iloc[0]:6.2f})" if len(x) else " " * 16)
            lines.append(f"{clock:6s} {int(g['n_bars'].iloc[0]):6d} " + " ".join(f"{s:>16s}" for s in cells))
    return "\n".join(lines) + "\n"


# ============================================================================ main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", default="", help="1 | 2 (a label in the outputs)")
    ap.add_argument("--label", default="")
    ap.add_argument("--linear-root", default="")
    ap.add_argument("--linear-sub", default="walk")
    ap.add_argument("--design", default="", help="the directory holding design_baseline/ (default <root>/_work)")
    ap.add_argument("--no-design", action="store_true", help="B = true_raw / true_adj^2")
    ap.add_argument("--linear-parquet", default="")
    ap.add_argument("--trees", action="append", default=[])
    ap.add_argument("--core", default=str(ROOT / "data" / "core_stats.parquet"))
    ap.add_argument("--first", default=FIRST)
    ap.add_argument("--last", default=LAST)
    ap.add_argument("--ref-scores", default="", help="Part 1: train3way's scores_all/ (qlike_bar.csv, gap.csv)")
    ap.add_argument("--require", action="append", default=[], help="cls/set that must be in the panel, e.g. rf/full")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    label = a.label or f"all48 part {a.part or '?'}"
    checks: dict = dict(label=label, part=a.part, first=a.first, last=a.last, core=a.core, fail=[], sources={})
    panel: dict[Key, pd.Series] = {}
    if a.linear_root:
        root = Path(a.linear_root)
        design: Path | None = None
        if not a.no_design:
            design = Path(a.design) if a.design else root / "_work"
            if not (design / "design_baseline" / "b_oos.npy").is_file():
                if a.design:
                    raise SystemExit(f"no design_baseline/b_oos.npy under {design}")
                design = None
        p, rec = linear_panel(root, a.linear_sub, design, a.first, a.last)
        panel |= p
        checks["sources"]["linear_root"] = rec
        for k, r in rec["models"].items():
            if r.get("covers_design_to_end") is False:
                checks["fail"].append(f"{k}: the walk does not cover the design's rows from its start to the end once")
            if r.get("b_missing"):
                checks["fail"].append(f"{k}: {r['b_missing']} rows without B in the design")
            if r.get("b_max_rel_vs_true_raw", 0.0) > 1e-9:
                checks["fail"].append(
                    f"{k}: design B differs from true_raw / true_adj^2 by {r['b_max_rel_vs_true_raw']}"
                )
            if r["clocks"] != 48:
                checks["fail"].append(f"{k}: {r['clocks']} end stamps, not 48")
    for f in ([a.linear_parquet] if a.linear_parquet else []) + a.trees:
        p, rec = parquet_panel(Path(f), a.first, a.last)
        clash = set(p) & set(panel)
        assert not clash, f"{f}: models already in the panel {sorted(clash)}"
        panel |= p
        checks["sources"][f] = rec
    have = {f"{k[0]}/{k[1]}" for k in panel}
    for r in a.require:
        if r not in have:
            checks["fail"].append(f"required model {r} not in the panel")
    rv = core_rv(Path(a.core))
    days = evaluation_days(rv, a.first, a.last)
    M, D, info = score(panel, rv, days)
    checks |= info
    if a.ref_scores:
        rc = ref_check(Path(a.ref_scores), M, D)
        checks["ref_check"] = rc
        checks["fail"] += [f"reference: {x}" for x in rc["fail"]]
    checks["pass"] = not checks["fail"]
    (out / "checks.json").write_text(json.dumps(checks, indent=1, default=str))
    if checks["fail"]:
        say(f"CHECKS FAIL ({len(checks['fail'])}): {checks['fail'][:5]} -> {out / 'checks.json'}; no table written")
        return 3
    for df in (M, D):
        df.insert(0, "part", a.part)
    M.to_csv(out / "metrics.csv", index=False)
    D.to_csv(out / "dm.csv", index=False)
    P = per_interval(M, D)
    P.insert(0, "part", a.part)
    P.to_csv(out / "per_interval.csv", index=False)
    (out / "tables.txt").write_text(fmt_tables(M, D, P, label))
    say(f"scored {info['n_bars']} ({len(days)} evaluation days); panel {info['panel']} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
