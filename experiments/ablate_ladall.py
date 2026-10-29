"""Addendum K of results/ablations_2014start_w3000/RULE.txt: the feature-count ladder (results/ladder_2026-10-26) refitted
at EVERY bar on the FOMC-patched design, split across Hoffman2 and CARC.  The walks themselves are those of the ablation
run (experiments/ablate_trees.py walk, experiments/ablate_linear.py walk1); the rungs lrung_<est>_<k> are
experiments/ablate_arms.py LADALL.  This file holds the task files, the gates, T2 and the merge of this addendum.

usage: python experiments/ablate_ladall.py <mode> [options]
  tasks    --dir D [--target-hours 2]   D/trees_ladall_carc.txt (LightGBM rungs, CARC), D/trees_ladall_h2.txt (XGBoost
                                        rungs, Hoffman2): lines "<est> <arm> <i>/<n> <hours at EPYC speed>", the largest
                                        rungs first, pieces of <= --target-hours at the cost model's speed (8 threads);
                                        D/linear_ladall.txt (the 15 linear rung walks, "<arm> <model> 0 1") and
                                        D/linear_ladall_gate.txt (the 3 linear gate walks: lrung_<m>_46, chunk 0 of the
                                        full set's chunks); prints the cost model's CPU-hours
  gate     --out O --est lgbm,xgb [--gb F]
                                        tree gate on one CPU model: O/pieces/lrung_<est>_{0,46}_<est>_first10 against
                                        Table 1's HAR walk (the templates' baseline rows) and its FOMC-patched full walk
                                        (fomcfix forecasts_gb.parquet, all_features): pred_adj text-identical (repr),
                                        true_adj and B equal, one CPU model -> O/passed_<est> (the CPU model) for each
                                        class that passes, O/gate_ladall.json; exit 3 if any class fails
  t2       --gates D1,D2,.. --out J     T2: each class's gate pieces bit-identical across the gate dirs that passed it
                                        (the gate2 format ablate_trees.cmd_merge reads)
  merge    --out R --gates D1,.. --tasks F --to P
                                        T2 over the gate dirs -> R/../gate2_<R name>.json, ablate_trees.cmd_merge for
                                        every (est, arm) of F (complete piece sets; mixed CPU models only with T2's bit
                                        identity), then the compact file P (date + one pred_adj column per "<est>|<arm>")
  lingate  --root R --tasks F --out J [--ref lin_B.parquet]
                                        linear gate: each gate walk's rows (chunk 0 of lrung_<m>_46 through the rung code
                                        path: ablate_design.py subsets cache, ablate_linear.py walk1) = Table 1's patched
                                        walk (fomcfix lin_B.parquet) bit for bit in pred_adj and true_adj on the same
                                        dates; exit 3 on a failure
  check    [--csv rank_primary.csv]     LADALL against the ranking; every rung's columns on the full design's names
  join     --parts F1,F2 --to P         (laptop) the two clusters' compact tree files joined on date (dates asserted
                                        equal, columns disjoint, 33,970 rows, no NaN) -> P (forecasts/trees_wide_ladall)
  score    [--res R] [--forecasts F] [--monthly S]
                                        (laptop) addendum K (h): R/forecasts (or F)/trees_wide_ladall.parquet and
                                        linear_wide_ladall.parquet scored as Table 1 (ablate_score.py --design fix:
                                        load_panel, scored, losses, daily, dm_test), k = 0 / 46 = Table 1's HAR / patched
                                        full walks; spans all / ord / ev; R_k, DM vs k = 46 and k = 0, the monthly R_k (S,
                                        primary, span all) beside it; the ladder RULE section 6 reading per (model, span)
                                        -> R/scores_fomcfix/ladall_scores.csv, ladall_readings.csv, ladall_checks.json;
                                        exit 3 if a check fails
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(HERE), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ablate_arms as AA  # noqa: E402

GB = ROOT / "results" / "fomcfix_2026-10-26" / "forecasts" / "forecasts_gb.parquet"
LIN_B = ROOT / "results" / "fomcfix_2026-10-26" / "forecasts" / "lin_B.parquet"
RANK = ROOT / "results" / "ladder_2026-10-26" / "fomcfix" / "rank_primary.csv"
NAMES = ROOT / "results" / "window_sweep_2014start_2026-10-22" / "w3000" / "frozen_penalties.json"
# which cluster walks which tree class (addendum K (f)): LightGBM on CARC EPYC, XGBoost on Hoffman2
CLUSTER = {"carc": ("lgbm",), "h2": ("xgb",)}
LIN = ("lasso", "enet", "ridge")  # the slowest first
GATE_TAG = "first10"


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================================ tasks
def cmd_tasks(a) -> int:
    import ablate_linear as AL
    import ablate_trees as AT

    d = Path(a.dir)
    d.mkdir(parents=True, exist_ok=True)
    tot: dict[str, float] = {}
    for cl, ests in CLUSTER.items():
        lines = []
        for est in ests:
            arms = sorted(AA.LRUNGS[est], key=lambda x: -AA.n_cols(x))
            for arm in arms:
                cpu_h = AT.cost_cpu_h(est, AA.n_cols(arm))
                tot[est] = tot.get(est, 0.0) + cpu_h
                wall = cpu_h / 8.0
                n = max(1, math.ceil(wall / a.target_hours))
                lines += [f"{est} {arm} {i}/{n} {wall / n:.2f}" for i in range(1, n + 1)]
        (d / f"trees_ladall_{cl}.txt").write_text("".join(ln + "\n" for ln in lines))
        say(f"{len(lines)} tree pieces -> {d / f'trees_ladall_{cl}.txt'}")
    lin = [f"lrung_{m}_{k} {m} 0 1" for k in sorted(AA.LADDER_K, reverse=True) for m in LIN]
    (d / "linear_ladall.txt").write_text("".join(ln + "\n" for ln in lin))
    gate = [f"lrung_{m}_46 {m} 0 {AL.FULL_CHUNKS[m]}" for m in LIN]
    (d / "linear_ladall_gate.txt").write_text("".join(ln + "\n" for ln in gate))
    say(f"{len(lin)} linear walks, {len(gate)} linear gate walks; cost model (EPYC speed, CPU-hours): {tot}")
    return 0


# ============================================================================ tree gate
def _ref(est: str, k: int, gb: Path) -> pd.DataFrame:
    """Table 1's every-bar walk of the gate rung: k = 0 the templates' HAR rows, k = 46 the patched full walk."""
    import ablate_trees as AT

    if k == 0:
        t = AT.template(est)
        t = t[t["predictor_set"] == "baseline"]
    else:
        t = pd.read_parquet(gb)
        t = t[(t["model"] == est) & (t["predictor_set"] == "all_features")]
    t = t.set_index(pd.DatetimeIndex(t["date"]))
    assert t.index.is_unique, (est, k)
    return t


def cmd_gate(a) -> int:
    import ablate_trees as AT

    out = Path(a.out)
    rec: dict = {"dir": str(out), "classes": {}, "fail": []}
    for est in [e for e in a.est.split(",") if e]:
        checks, ok, cpus = [], True, set()
        for k in AA.LADALL_GATE_K:
            arm = f"lrung_{est}_{k}"
            p = AT.piece_path(out, arm, est, GATE_TAG)
            if not (p.is_file() and p.with_suffix(".json").is_file()):
                ok = False
                checks.append(dict(arm=arm, missing=True))
                continue
            f = pd.read_parquet(p)
            m = json.loads(p.with_suffix(".json").read_text())
            cpus |= set(m["cpus"])
            t = _ref(est, k, Path(a.gb)).reindex(pd.DatetimeIndex(f["date"]))
            x, y = f["pred_adj"].to_numpy(), t["pred_adj"].to_numpy()
            text = [repr(float(u)) == repr(float(v)) for u, v in zip(x, y)]
            side = bool(np.array_equal(f["true_adj"].to_numpy(), t["true_adj"].to_numpy())) and bool(
                np.array_equal(f["B"].to_numpy(), t["B"].to_numpy())
            )
            good = len(f) == 10 and all(text) and side and len(m["cpus"]) == 1 and m["n_columns"] == AA.n_cols(arm)
            ok &= good
            checks.append(
                dict(
                    arm=arm,
                    rows=len(f),
                    text_identical=int(sum(text)),
                    max_abs_diff=float(np.nanmax(np.abs(x - y))),
                    side_equal=side,
                    n_columns=m["n_columns"],
                    cpus=m["cpus"],
                    sec_refit=m["wall_sec"] / max(1, m["refits"]),
                    dates=[str(f["date"].iloc[0]), str(f["date"].iloc[-1])],
                    pass_=good,
                )
            )
        ok &= len(cpus) == 1
        rec["classes"][est] = dict(pass_=ok, cpus=sorted(cpus), checks=checks)
        if ok:
            (out / f"passed_{est}").write_text(sorted(cpus)[0] + "\n")
        else:
            rec["fail"].append(est)
    rec["reference"] = {"k0": "templates' baseline rows (Table 1's HAR walk)", "k46": str(a.gb)}
    rec["pass"] = not rec["fail"]
    (out / "gate_ladall.json").write_text(json.dumps(rec, indent=1))
    say(f"gate K {out.name}: {'PASS' if rec['pass'] else 'FAIL ' + str(rec['fail'])}")
    return 0 if rec["pass"] else 3


def t2(gates: list[Path], out: Path) -> dict:
    import ablate_trees as AT

    rec: dict = {"classes": {}, "gates": [str(d) for d in gates]}
    for est in ("lgbm", "xgb"):
        ps = []
        for d in gates:
            if not (d / f"passed_{est}").is_file():
                continue
            v, cp = [], set()
            for k in AA.LADALL_GATE_K:
                p = AT.piece_path(d, f"lrung_{est}_{k}", est, GATE_TAG)
                v.append(pd.read_parquet(p)["pred_adj"].to_numpy())
                cp |= set(json.loads(p.with_suffix(".json").read_text())["cpus"])
            ps.append((d.name, sorted(cp), np.concatenate(v)))
        if len(ps) < 2:
            rec["classes"][est] = dict(models={x[0]: x[1] for x in ps}, identical=None)
            continue
        ref = ps[0][2]
        ident = all(np.array_equal(ref, x[2]) for x in ps[1:])
        rel = max(float(np.max(np.abs(ref - x[2]) / np.abs(ref))) for x in ps[1:])
        rec["classes"][est] = dict(models={x[0]: x[1] for x in ps}, identical=bool(ident), max_rel=rel)
    out.write_text(json.dumps(rec, indent=1))
    say(f"T2 (addendum K): {json.dumps({e: v.get('identical') for e, v in rec['classes'].items()})}")
    return rec


def cmd_t2(a) -> int:
    t2([Path(x) for x in a.gates.split(",") if x], Path(a.out))
    return 0


# ============================================================================ merge
def cmd_merge(a) -> int:
    import ablate_trees as AT

    out = Path(a.out)
    g2 = out.parent / f"gate2_{out.name}.json"
    t2([Path(x) for x in a.gates.split(",") if x], g2)
    pairs = sorted({tuple(ln.split()[:2]) for ln in Path(a.tasks).read_text().splitlines() if ln.strip()})
    merged, not_merged = [], []
    for est, arm in pairs:
        try:
            AT.cmd_merge(argparse.Namespace(arm=arm, est=est, out=str(out), gate2=str(g2)))
            merged.append(f"{est}|{arm}")
        except AssertionError as e:
            not_merged.append(f"{est}|{arm}: {e}")
    print(f"merged {len(merged)}, not merged {len(not_merged)}:", *not_merged, sep="\n  ")
    cols, dates = {}, None
    for est, arm in pairs:
        p = out / "forecasts" / f"{est}_{arm}.parquet"
        if not p.is_file():
            continue
        f = pd.read_parquet(p)
        d = pd.DatetimeIndex(f["date"])
        assert d.is_monotonic_increasing and len(d) == 33970, p.name
        if dates is None:
            dates = d
        assert d.equals(dates), p.name
        cols[f"{est}|{arm}"] = f["pred_adj"].to_numpy()
    assert dates is not None, "no merged forecasts"
    w = pd.DataFrame({"date": dates, **dict(sorted(cols.items()))})
    w.to_parquet(a.to, index=False)
    print(f"{len(cols)} arms x {len(dates)} dates -> {a.to}")
    return 0 if not not_merged else 3


# ============================================================================ linear gate
def cmd_lingate(a) -> int:
    import ablate_linear as AL

    root = Path(a.root)
    ref = pd.read_parquet(a.ref, columns=["model", "predictor_set", "date", "pred_adj", "true_adj"])
    rec: dict = {"root": str(root), "ref": str(a.ref), "checks": [], "fail": []}
    lines = [ln.split() for ln in Path(a.tasks).read_text().splitlines() if ln.strip()]
    for arm, m, c, k in lines:
        c_, k_ = int(c), int(k)
        assert AA.lrung_parts(arm) == (m, 46), (arm, m)
        o = AL.out_dir(root, arm, m, c_, k_)
        ps = list(o.glob("causal_tune_linear/*/*/results.csv"))
        if not ((o / "DONE").is_file() and len(ps) == 1):
            rec["fail"].append(f"{arm}/{m} chunk {c_}: not done ({o})")
            continue
        r = pd.read_csv(ps[0], parse_dates=["date"])
        t = ref[(ref["model"] == m) & (ref["predictor_set"] == "all_features")].set_index("date")
        t = t.reindex(pd.DatetimeIndex(r["date"]))
        have = bool(t["pred_adj"].notna().all())
        eq = have and bool(np.array_equal(r["pred_adj"].to_numpy(), t["pred_adj"].to_numpy()))
        eq_y = have and bool(np.array_equal(r["true_adj"].to_numpy(), t["true_adj"].to_numpy()))
        d = float(np.nanmax(np.abs(r["pred_adj"].to_numpy() - t["pred_adj"].to_numpy()))) if have else float("nan")
        done = json.loads((o / "DONE").read_text())
        rec["checks"].append(
            dict(
                arm=arm,
                model=m,
                chunk=c_,
                chunks=k_,
                rows=len(r),
                first=str(r["date"].iloc[0]),
                last=str(r["date"].iloc[-1]),
                dates_in_ref=have,
                pred_adj_bit_identical=eq,
                true_adj_equal=eq_y,
                max_abs_diff=d,
                sec=done.get("sec"),
            )
        )
        if not (eq and eq_y and len(r) > 0):
            rec["fail"].append(f"{arm}/{m} chunk {c_}: {len(r)} rows, max |diff| {d}")
    import platform

    from pooled_trees_rerun_13 import cpu_model

    rec.update(host=platform.node(), cpu=cpu_model(), numpy=np.__version__)
    rec["pass"] = not rec["fail"] and len(rec["checks"]) == len(lines)
    Path(a.out).write_text(json.dumps(rec, indent=1))
    say(f"linear gate K {'PASS' if rec['pass'] else 'FAIL'} on {rec['cpu']}: {rec['fail']}")
    return 0 if rec["pass"] else 3


# ============================================================================ check
def cmd_check(a) -> int:
    AA.check_ladall(a.csv)
    names = json.loads(NAMES.read_text())["designs"]["all_features"]["columns"]
    AA.check_full_names(names)
    gs = AA.group_series()
    for est in AA.LADALL:
        for k in (*AA.LADALL_GATE_K, *AA.LADDER_K):
            arm = f"lrung_{est}_{k}"
            cols = AA.arm_names(arm, names, gs)
            assert len(cols) == AA.n_cols(arm) == 12 + 12 * k, (arm, len(cols))
            assert cols == [n for n in names if n in set(cols)], arm  # design order
            if k == 46:
                assert cols == names, arm
            if k == 0:
                assert set(cols) == set(AA.HAR12), arm
            if k:
                ser = {AA.series_of(c) for c in cols} - {None}
                assert k == 46 or ser == set(AA.LADALL[est][:k]), arm
    say(f"LADALL = {a.csv}'s primary top 20 for {list(AA.LADALL)}; every rung's columns checked on {len(names)} names")
    return 0


def cmd_join(a) -> int:
    """The CARC (LightGBM) and Hoffman2 (XGBoost) compact files -> one compact file of the 10 tree rungs."""
    parts = [pd.read_parquet(p) for p in a.parts.split(",") if p]
    w = parts[0]
    for f in parts[1:]:
        assert pd.DatetimeIndex(f["date"]).equals(pd.DatetimeIndex(w["date"])), "dates differ"
        both = (set(f.columns) & set(w.columns)) - {"date"}
        assert not both, both
        w = pd.concat([w, f.drop(columns="date")], axis=1)
    w = w[["date", *sorted(c for c in w.columns if c != "date")]]
    assert len(w) == 33970 and w.notna().all().all(), (len(w), int(w.isna().sum().sum()))
    want = {f"{e}|{x}" for e in ("lgbm", "xgb") for x in AA.LRUNGS[e]}
    assert set(w.columns) - {"date"} == want, sorted(set(w.columns) ^ want)
    w.to_parquet(a.to, index=False)
    say(f"{w.shape[1] - 1} tree rungs x {len(w)} bars -> {a.to}")
    return 0


# ============================================================================ score (addendum K (h))
RES = ROOT / "results" / "ablations_2014start_w3000"
MONTHLY = ROOT / "results" / "ladder_2026-10-26" / "fomcfix" / "scores.csv"
LABEL = "every-bar refit, requested by the author after the monthly-vs-every-bar comparison; not pre-registered"
TREES_K = ("lgbm", "xgb")
MODELS = (*TREES_K, "ridge", "lasso", "enet")  # LADALL's classes (no forest), in the monthly ladder's order
KS = (0, *AA.LADDER_K, 46)
WIDE = {"trees_wide_ladall.parquet": TREES_K, "linear_wide_ladall.parquet": ("ridge", "lasso", "enet")}
N_BARS, N_ROWS = 33527, 33970
SCORE_COLS = [
    "model",
    "ranking",
    "k",
    "rung",
    "fitted_as",
    "span",
    "n_bars",
    "qlike",
    "mse",
    "dm_vs_full",
    "p_vs_full",
    "dm_vs_har",
    "p_vs_har",
    "hac_lag",
    "R",
    "R_monthly",
]


def _rung(m: str, k: int) -> str:
    return "har" if k == 0 else "full" if k == 46 else f"lrung_{m}_{k}"


def reading(f: pd.DataFrame) -> dict:
    """Ladder RULE section 6 on one (model, span): f indexed by k with qlike, R, p_vs_full, dm_vs_full.  The definition
    of results/ladder_2026-10-26/code/ladder_score.py (readings), with its retained_pct = R here."""
    lf, lh = float(f.loc[46, "qlike"]), float(f.loc[0, "qlike"])
    ret = f["R"]
    kx = {}
    for thr in (50, 75, 90):
        hit = [k for k in (0, 1, 3, 5, 10, 20, 46) if k < 46 and ret.get(k, np.nan) >= thr]
        kx[thr] = min(hit) if hit else 46
    if lf >= lh:
        verdict = "NO GAIN TO RETAIN"
    elif kx[90] <= 5:
        verdict = "CONCENTRATED"
    elif kx[90] <= 20:
        verdict = "INTERMEDIATE"
    else:
        verdict = "DENSE AND WEAK"
    better = [int(k) for k in (1, 3, 5, 10, 20) if float(f.loc[k, "qlike"]) < lf]
    return dict(
        qlike_full=lf,
        qlike_har=lh,
        gain_full_vs_har=lh - lf,
        k50=kx[50],
        k75=kx[75],
        k90=kx[90],
        verdict=verdict,
        rungs_below_full=" ".join(map(str, better)),
        **{f"R{k}": float(ret.get(k, np.nan)) for k in (1, 3, 5, 10, 20)},
        **{f"p{k}_vs_full": float(f.loc[k, "p_vs_full"]) for k in (20, 10, 5)},
        **{f"dm{k}_vs_full": float(f.loc[k, "dm_vs_full"]) for k in (20, 10, 5)},
    )


def cmd_score(a) -> int:
    import ablate_score as AS

    res = Path(a.res)
    fc = Path(a.forecasts) if a.forecasts else res / "forecasts"
    rf_fix = (str(AS.RF_FIX) if AS.RF_FIX.is_file() else "") if a.rf_fix == "auto" else a.rf_fix
    fail: list[str] = []
    rv = AS.core_rv()
    days = AS.evaluation_days(rv)
    panel, B = AS.load_panel(res, "fix", rf_fix)  # Table 1's walks: HAR = the templates, full = the patched walks
    templ = {key: f for key, f in panel.items() if key[1] in ("full", "har", "full_pre")}
    idx_t = AS.scored(templ, rv, days)
    assert len(idx_t) == N_BARS, len(idx_t)  # Table 1's bars

    # ---------------------------------------------------------------- the rungs: plain = pred_adj^2 x B
    lad: dict[tuple[str, int], pd.Series] = {}
    dates_ok: dict[str, bool] = {}
    for name, ests in WIDE.items():
        p = fc / name
        assert p.is_file(), f"missing {p}"
        w = pd.read_parquet(p)
        want = {f"{m}|lrung_{m}_{k}" for m in ests for k in AA.LADDER_K}
        cols = list(w.columns[1:])
        assert w.columns[0] == "date" and len(cols) == len(want) and set(cols) == want, (name, sorted(set(cols) ^ want))
        d = pd.DatetimeIndex(w["date"])
        assert d.is_unique and d.is_monotonic_increasing and len(d) == N_ROWS, (name, len(d))
        dates_ok[name] = bool(d.equals(pd.DatetimeIndex(B.index)))
        if not dates_ok[name]:
            fail.append(f"{name}: dates differ from the templates' B dates")
        b = B.reindex(d).to_numpy()
        assert np.isfinite(b).all(), name
        for col in cols:
            m, arm = col.split("|")
            est, k = AA.lrung_parts(arm)
            assert est == m and k in AA.LADDER_K, col
            lad[(m, k)] = pd.Series(w[col].to_numpy() ** 2 * b, index=d)  # as load_panel does for every ablation arm
    for m in MODELS:
        lad[(m, 0)], lad[(m, 46)] = panel[(m, "har")], panel[(m, "full")]

    # ---------------------------------------------------------------- B: the rungs' walks carry the templates' B
    gb = pd.read_parquet(AS.FIX / "forecasts_gb.parquet", columns=["model", "date", "B"])
    lb = pd.read_parquet(AS.FIX / "lin_B.parquet", columns=["model", "date", "B"])
    walk_b: dict[str, bool] = {}
    for m in MODELS:
        src = gb if m in TREES_K else lb
        s = src[src["model"] == m].set_index("date")["B"].reindex(B.index)
        walk_b[m] = bool(np.array_equal(s.to_numpy(), B.to_numpy()))
        if not walk_b[m]:
            fail.append(f"B: {m}'s patched full walk B != the templates' B")
    mr = res / "runs" / "tree_merge_ladall"
    records: dict[str, object] = {}
    if mr.is_dir():  # ablate_trees.cmd_merge asserted true_adj and B = the templates' before writing each record
        for m in TREES_K:
            for k in AA.LADDER_K:
                arm = f"lrung_{m}_{k}"
                hits = sorted(mr.glob(f"*/{m}_{arm}.json"))
                ok = len(hits) == 1
                if ok:
                    j = json.loads(hits[0].read_text())
                    ok = j.get("arm") == arm and j.get("est") == m and j.get("rows") == N_ROWS
                records[f"{m}|{arm}"] = ok
                if not ok:
                    fail.append(f"B: merge record of {m}|{arm} missing or not {N_ROWS} rows ({len(hits)} found)")
    b_check = dict(
        rule="plain = pred_adj^2 x B with the templates' B (load_panel's); the rung files' dates = the templates' B "
        "dates; the templates' B = the B of each class's patched full walk (fomcfix forecasts_gb / lin_B), whose design "
        "the rungs subset; tree rungs: ablate_trees merge records (true_adj and B asserted = the templates')",
        rung_file_dates_equal=dates_ok,
        walk_B_equal=walk_b,
        tree_merge_records=records if records else f"not present ({mr})",
    )

    # ---------------------------------------------------------------- the scored bars and spans
    rungs = {(m, _rung(m, k)): f for (m, k), f in lad.items() if k not in (0, 46)}
    idx = AS.scored({**templ, **rungs}, rv, days)
    if not idx.equals(idx_t):
        fail.append(f"a rung drops {len(idx_t.difference(idx))} of Table 1's {len(idx_t)} scored bars")
    bad_cols = {}
    for (m, k), f in lad.items():
        v = f.reindex(idx_t).to_numpy(float)
        nbad = int((~(np.isfinite(v) & (v > 0))).sum())
        if nbad:
            bad_cols[f"{m}|{_rung(m, k)}"] = nbad
            fail.append(f"{m}|{_rung(m, k)}: {nbad} non-finite or non-positive plain forecasts on Table 1's bars")
    ev = AS.event_days(days)
    on_ev = pd.DatetimeIndex(idx.normalize()).isin(ev)
    spans = {"all": idx, "ord": idx[~on_ev], "ev": idx[on_ev]}

    # ---------------------------------------------------------------- scores
    mo = pd.read_csv(a.monthly)
    mo = mo[mo["ranking"].isin(["P", "both"]) & mo["k"].notna() & (mo["span"] == "all")]
    assert not mo.duplicated(["model", "k"]).any(), "monthly scores: duplicate (model, k)"
    r_mon = {(str(r.model), int(r.k)): float(r.retained_pct) for r in mo.itertuples()}

    def dm(x1: np.ndarray, x2: np.ndarray, ix: pd.DatetimeIndex) -> tuple[float, float, int]:
        r = AS.dm_test(AS.daily(x1, ix), AS.daily(x2, ix))  # ablate_score's every-bar DM: daily average loss
        return float(r["dm"]), float(r["p"]), int(r.get("hac_lag", -1))

    nan3 = (np.nan, np.nan, -1)
    rows = []
    for m in MODELS:
        for sp, ix in spans.items():
            L = {k: AS.losses(lad[(m, k)], ix, rv) for k in KS}
            l0, l46 = float(L[0][0].mean()), float(L[46][0].mean())
            for k in KS:
                q, se = L[k]
                vf = dm(q, L[46][0], ix) if k != 46 else nan3
                vh = dm(q, L[0][0], ix) if k != 0 else nan3
                rows.append(
                    dict(
                        model=m,
                        ranking="primary",
                        k=k,
                        rung=_rung(m, k),
                        fitted_as="everybar",
                        span=sp,
                        n_bars=len(ix),
                        qlike=float(q.mean()),
                        mse=float(se.mean()),
                        dm_vs_full=vf[0],
                        p_vs_full=vf[1],
                        dm_vs_har=vh[0],
                        p_vs_har=vh[1],
                        hac_lag=max(vf[2], vh[2]),
                        R=(l0 - float(q.mean())) / (l0 - l46) * 100.0 if l0 > l46 else np.nan,
                        R_monthly=r_mon.get((m, k), np.nan) if sp == "all" else np.nan,
                    )
                )
    S = pd.DataFrame(rows)[SCORE_COLS]

    # ---------------------------------------------------------------- Table 1: k = 0 / 46 on every bar
    t1 = pd.read_csv(AS.TABLE1 / "metrics.csv")
    t1_rows = 0
    for m in MODELS:
        for k, st in ((0, "baseline"), (46, "all_features")):
            r = t1[(t1["model"] == m) & (t1["set"] == st)]
            assert len(r) == 1, (m, st)
            mine = S[(S["model"] == m) & (S["k"] == k) & (S["span"] == "all")].iloc[0]
            for col in ("qlike", "mse"):
                if not np.isclose(float(mine[col]), float(r[col].iloc[0]), rtol=1e-12, atol=0):
                    fail.append(f"Table 1 {m}/{st} {col}: {float(mine[col])} vs {float(r[col].iloc[0])}")
            if int(mine["n_bars"]) != int(r["n_intervals"].iloc[0]):
                fail.append(f"Table 1 {m}/{st}: bars {int(mine['n_bars'])} vs {int(r['n_intervals'].iloc[0])}")
            t1_rows += 1

    # ---------------------------------------------------------------- readings (ladder RULE section 6)
    rd = []
    for m in MODELS:
        for sp in spans:
            f = S[(S["model"] == m) & (S["span"] == sp)].set_index("k")
            rd.append(dict(model=m, ranking="primary", span=sp, **reading(f)))
    R = pd.DataFrame(rd)

    out = res / "scores_fomcfix"
    out.mkdir(parents=True, exist_ok=True)
    S.to_csv(out / "ladall_scores.csv", index=False)
    R.to_csv(out / "ladall_readings.csv", index=False)
    checks = {
        "pass": not fail,
        "fail": fail,
        "label": LABEL,
        "rule": "results/ablations_2014start_w3000/RULE.txt addendum K (h); readings: ladder RULE section 6 "
        "(results/ladder_2026-10-26/code/ladder_score.py's definition of k50 / k75 / k90 and the verdict)",
        "inputs": {name: str(fc / name) for name in WIDE},
        "res": str(res),
        "rf_fix": rf_fix,
        "n_bars_table1": len(idx_t),
        "n_bars_scored": len(idx),
        "day_types": dict(
            event_days=len(ev),
            ordinary_days=int(pd.DatetimeIndex(spans["ord"].normalize()).nunique()),
            event_bars=len(spans["ev"]),
            ordinary_bars=len(spans["ord"]),
            rule="ablate_score.event_days (results/channels_2026-10-26/RULE.txt section 0)",
        ),
        "B": b_check,
        "rungs_not_finite_positive": bad_cols,
        "table1_rows_checked": t1_rows,
        "table1": str(AS.TABLE1 / "metrics.csv"),
        "dm": "ablate_score's every-bar DM: dm_test (HLN) on the daily average QLIKE, auto HAC lag (8 on all / ord)",
        "R": "(L_0 - L_k) / (L_0 - L_46) x 100 on the span's own L_0 (Table 1's HAR walk) and L_46 (patched full "
        "walk); NaN unless L_0 > L_46",
        "R_monthly": f"{a.monthly}: retained_pct, ranking P (k = 0 / 46: both), span all; on span all rows only "
        "(the monthly ladder has no ord / ev span), NaN elsewhere",
    }
    (out / "ladall_checks.json").write_text(json.dumps(checks, indent=1))
    say(
        f"scored {len(lad)} (model, k) forecasts on {len(idx)} bars -> {out}; checks "
        f"{'PASS' if checks['pass'] else 'FAIL'}: {fail[:6]}"
    )
    mon = pd.read_csv(Path(a.monthly).parent / "readings.csv")
    mon = mon[(mon["ranking"] == "primary") & (mon["span"] == "all")].set_index("model")
    for r in R.itertuples():
        mm = (
            f"  monthly(all) {mon.loc[r.model, 'k50']}/{mon.loc[r.model, 'k75']}/{mon.loc[r.model, 'k90']} "
            + str(mon.loc[r.model, "verdict"])
            if r.model in mon.index
            else ""
        )
        print(f"  {r.model:5s} {r.span:3s} k50/k75/k90 {r.k50}/{r.k75}/{r.k90} {r.verdict:18s}{mm}")
    return 0 if checks["pass"] else 3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("tasks", "gate", "t2", "merge", "lingate", "check", "join", "score"))
    ap.add_argument("--parts", default="")
    ap.add_argument("--dir", default="")
    ap.add_argument("--target-hours", type=float, default=2.0)
    ap.add_argument("--out", default="")
    ap.add_argument("--est", default="lgbm,xgb")
    ap.add_argument("--gb", default=str(GB))
    ap.add_argument("--gates", default="")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--to", default="")
    ap.add_argument("--root", default="")
    ap.add_argument("--ref", default=str(LIN_B))
    ap.add_argument("--csv", default=str(RANK))
    ap.add_argument(
        "--res", default=str(RES), help="score: the ablation results dir (load_panel; output scores_fomcfix)"
    )
    ap.add_argument(
        "--forecasts", default="", help="score: dir of the two *_wide_ladall.parquet (default <res>/forecasts)"
    )
    ap.add_argument("--monthly", default=str(MONTHLY), help="score: the monthly ladder's scores.csv (R_monthly)")
    ap.add_argument(
        "--rf-fix", default="auto", help="score: load_panel's rf_fix (auto: ablate_score's RF_FIX if present)"
    )
    a = ap.parse_args()
    return {
        "tasks": cmd_tasks,
        "gate": cmd_gate,
        "t2": cmd_t2,
        "merge": cmd_merge,
        "lingate": cmd_lingate,
        "check": cmd_check,
        "join": cmd_join,
        "score": cmd_score,
    }[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
