"""ALL48 Part 2, the linear models (results/all48_2026-10-30/RULE.txt sections 2.1-2.2; EXPLORATORY, proposed after
the main results were seen, not pre-registered): the main run's orchestrator (experiments/pooled_linear_rerun_2026_10_19_13
.py) and runner (experiments/pool_linear_pretune.py, unchanged) on the 48-interval sample of the current (FOMC-patched,
cell B) design, as train3way's experiments/linear_train3way.py configured its ALL48 arm: --sample 48 (every row of the
48-interval grid), --har-base 5, DROP_ACTIVITY_MA=1, CALENDAR_THROUGH_OVERNIGHT=1, DROP_OVERNIGHT_INDICATOR=0 (is_overnight
kept), LEAN_GLOBAL_LAGS=1, --window-days 3000 (W = 144,000 rows), LATE_START_WARMUP=1 and cell B's
FOMC_SCHEDULE_2020_FIX=1, FOMC_PROXIMITY_NO_WINSOR=1, DROP_HOLIDAY_SESSIONS=0; --first-day 2014-01-02, --first-fold 3;
STANDARDIZE_PENALTY, WALK_KKT_RESEED and the grid rule at their defaults (on).  The orchestrator's globals are set by
configure() (its main() sets them from the command line for the 13-bar sample); nothing in either module changes, and
importing this file changes nothing until configure() runs.

The walk refits at EVERY row of the 48-interval design (every interval origin: overnight, weekend and holiday rows
included) from the 250-solve boundary at or before the first 2014-01-02 row to the last row; every row's forecast is
exported (export: no RTH filter).

usage: python experiments/all48_linear.py <stage> --out DIR [options]
  design      the two design caches (--jobs 2; ~1 h, ~96 GB)
  gate        D1 names / config; D2 y and B = the cell-B 13-bar design's at every common stamp (--cellb RUN DIR);
              D3 against train3way's ALL48 caches (--ref RUN DIR; results/fomcfix_2026-10-26/code/design_gate.py
              --expect fomc), recorded as not run when --ref has no caches -> DIR/gate_design.json; exit 3 on a failure
  pretune     fold 3, every grid penalty, the grid rule's extensions (--jobs 4)
  select      frozen_penalties.json, pretune_mse.csv (info.folds / rule / sample rewritten for the 48-interval sample)
  gate_har    L1 (when D3 passed): the HAR-set penalties = --ref's and the first --gate-rows rows of the four HAR-set
              walks identical (date, true_adj, pred_adj strings) to --ref's walk tables -> DIR/gate_har.json; exit 3
  gate_seam   L2 (train3way's): the full-set ridge / lasso / elastic net and the OLS over [S0, S0 + 500) against a chunk
              from S0 + 250, the chunk identical -> DIR/gate_seam.json; exit 3
  njobs       the number of walk jobs (the array size)
  walk1       walk job --task K (1-based; 0 = SLURM_ARRAY_TASK_ID); chunks ridge 2, lasso 8, elastic net 5 (full set)
  export      DIR/forecasts_all.parquet: every walked row of the seven models (model, predictor_set, alpha, date,
              pred_adj, true_adj, B, plain, rv_actual) + forecasts_all.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for _p in (str(HERE), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pooled_linear_rerun_2026_10_19_13 as O  # noqa: E402

WD, PPD, FIRST_DAY, FIRST_FOLD = 3000, 48, "2014-01-02", 3
W48 = WD * PPD  # 144,000 rows
FLAGS = ["DROP_ACTIVITY_MA=1", "CALENDAR_THROUGH_OVERNIGHT=1", "DROP_OVERNIGHT_INDICATOR=0", "LEAN_GLOBAL_LAGS=1"]
CELL_B = ["LATE_START_WARMUP=1", "FOMC_SCHEDULE_2020_FIX=1", "FOMC_PROXIMITY_NO_WINSOR=1", "DROP_HOLIDAY_SESSIONS=0"]
EX_FLAGS = {kv.split("=")[0]: kv.split("=")[1] == "1" for kv in FLAGS + CELL_B}
CAL = [f"DOW_{d}" for d in range(5)] + ["hour", "is_overnight"]
N_COLS = {"baseline": 13, "all_features": 565}
CHUNKS = {("all_features", "ridge"): 2, ("all_features", "lasso"): 8, ("all_features", "enet"): 5}  # train3way's
HAR_MODELS = ["baseline/ols", "baseline/ridge", "baseline/lasso", "baseline/enet"]
SEAM_MODELS = ["all_features/ridge", "all_features/lasso", "all_features/enet", "baseline/ols"]
DESIGN_GATE = REPO / "results" / "fomcfix_2026-10-26" / "code" / "design_gate.py"
DESC = "all 48 half-hour intervals of the 48-interval grid (overnight, weekend and holiday rows included)"


def runner_cfg() -> list[str]:
    """The runner's configuration (the orchestrator's CFG) of the 48-interval cell-B design."""
    c = ["--sample", "48", "--har-base", "5"]
    for f in FLAGS:
        c += ["--ex-flag", f]
    c += ["--window-days", str(WD)]
    for f in CELL_B:
        c += ["--ex-flag", f]
    return c


def configure(out: Path) -> None:
    """The orchestrator's globals for the 48-interval sample (as its main() sets them for the 13-bar sample)."""
    O.OUT = out.resolve()
    O.WORK = O.OUT / "_work"
    O.CFG[:] = runner_cfg()
    O.PPD, O.TW_DAYS = PPD, WD
    O.W = W48
    O.FIRST_FOLD = FIRST_FOLD
    O.PRE_HALO = (O.PRE_FOLDS - FIRST_FOLD) * O.PRE_FOLD_SESS * PPD + O.PRE_EMBARGO_SESS * PPD + O.W
    O.FIRST_DAY = FIRST_DAY
    O.check_design = check_design  # the orchestrator's stages call it by its module-global name
    O.OUT.mkdir(parents=True, exist_ok=True)


def check_design(s: str) -> list[str]:
    """The orchestrator's check_design with the 48-interval calendar (+ is_overnight) and its column counts."""
    all_features, fomc = O.design_features()
    names = json.loads((O.design_dir(s) / "design.json").read_text())["names"]
    har = [n for n in names if n.startswith("har_ma_")]
    assert har == [f"har_ma_{k}" for k in O.HAR_LADDER], har
    cal = [n for n in names if not re.search(r"_ma_\d+$", n)]
    assert cal == CAL, cal
    assert not any("_active_ma_" in n for n in names)
    if s == "all_features":
        assert set(fomc) <= set(all_features)
        for f in fomc:
            for k in O.HAR_LADDER:
                assert f"adj_{f}_ma_{k}" in names and f"{f}_avail_ma_{k}" in names, (f, k)
        exog = {n for n in names if n.startswith("adj_")}
        assert exog == {f"adj_{f}_ma_{k}" for f in all_features for k in O.HAR_LADDER}, sorted(exog)[:5]
    assert len(names) == N_COLS[s], (s, len(names))
    return names


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def say(msg: str) -> None:
    O.say(msg)


# ============================================================================ stages
def st_design(a) -> None:
    O.st_design(a)


def oos(root: Path, s: str) -> tuple[pd.DatetimeIndex, dict]:
    dz = root / "_work" / f"design_{s}"
    meta = json.loads((dz / "design.json").read_text())
    return pd.DatetimeIndex(pd.read_csv(dz / "dates_oos.csv", parse_dates=["date"])["date"]), meta


def st_gate(a) -> None:
    """D1, D2, D3 of RULE.txt section 2.1 -> gate_design.json; exit 3 on a failure."""
    g: dict[str, Any] = {"out": str(O.OUT), "fail": []}
    # D1: names and configuration
    for s in O.SETS:
        try:
            names = check_design(s)
        except AssertionError as e:
            g["fail"].append(f"D1 {s}: check_design: {e}")
            continue
        meta = json.loads((O.design_dir(s) / "design.json").read_text())
        cf = dict(meta.get("config") or {})
        want = dict(sample="48", periods_per_day=PPD, window=W48, har_base="5", drop_fomc=False, ex_flags=EX_FLAGS)
        got: dict[str, Any] = {k: cf.get(k) for k in want}
        if got["window"] is not None:
            got["window"] = int(got["window"])
        g[f"D1_{s}"] = dict(config=cf, shape=meta["shape"], n_columns=len(names), config_ok=got == want)
        if got != want:
            g["fail"].append(f"D1 {s}: config {got} != {want}")
    # D2: y and B = the cell-B 13-bar design's at every common OOS stamp; its stamps from 2014-01-02 all present
    cb = Path(a.cellb).resolve()
    for s in O.SETS:
        d48, m48 = oos(O.OUT, s)
        d13, m13 = oos(cb, s)
        W13, Wn = int(str(m13["config"]["window"])), int(str(m48["config"]["window"]))
        y48, y13 = np.load(O.design_dir(s) / "y.npy"), np.load(cb / "_work" / f"design_{s}" / "y.npy")
        b48, b13 = np.load(O.design_dir(s) / "b_oos.npy"), np.load(cb / "_work" / f"design_{s}" / "b_oos.npy")
        i48 = d48.get_indexer(d13)
        have = i48 >= 0
        late = (d13.normalize() >= pd.Timestamp(FIRST_DAY)).to_numpy()
        j13, j48 = np.flatnonzero(have), i48[have]
        same_y = bool(np.array_equal(y48[Wn + j48], y13[W13 + j13], equal_nan=True))
        same_b = bool(np.array_equal(b48[j48], b13[j13], equal_nan=True))
        names13 = m13["names"]
        names_ok = [n for n in m48["names"] if n != "is_overnight"] == names13
        g[f"D2_{s}"] = dict(
            cellb=str(cb),
            oos_13=len(d13),
            common=int(have.sum()),
            missing_from_2014=int((~have & late).sum()),
            y_equal=same_y,
            B_equal=same_b,
            names_are_cellb_plus_is_overnight=names_ok,
        )
        if not (same_y and same_b and names_ok and not (~have & late).any()):
            g["fail"].append(f"D2 {s}: {g[f'D2_{s}']}")
    # D3: against train3way's ALL48 caches (the fomcfix G3 rule), when they exist
    ref = Path(a.ref).resolve() if a.ref else None
    if ref is not None and all((ref / "_work" / f"design_{s}" / "X.npy").is_file() for s in O.SETS):
        oj = O.OUT / "gate_D3.json"
        r = subprocess.run(
            [sys.executable, str(DESIGN_GATE), str(ref), str(O.OUT), str(oj), "--expect", "fomc"], cwd=REPO
        )
        d3 = json.loads(oj.read_text()) if oj.is_file() else {"pass": False, "fail": ["design_gate.py wrote no record"]}
        g["D3"] = dict(run=True, ref=str(ref), rc=r.returncode, pass_=bool(d3.get("pass")), fail=d3.get("fail"))
        if not d3.get("pass") or r.returncode:
            g["fail"].append(f"D3: {d3.get('fail')}")
    else:
        g["D3"] = dict(run=False, ref=str(ref), note="train3way's ALL48 design caches not found: D3 not run (RULE 2.1)")
    g["pass"] = not g["fail"]
    (O.OUT / "gate_design.json").write_text(json.dumps(g, indent=1, default=str))
    say(f"DESIGN GATE {'PASS' if g['pass'] else 'FAIL'}: {g['fail']}")
    if not g["pass"]:
        sys.exit(3)


def st_pretune(a) -> None:
    O.st_pretune(a)


def fold_info() -> list[dict]:
    """The fitted fold's fit / validation rows (fold.npz, every model alike) and their date labels (OOS rows only; a
    row before the OOS span is labelled by its design row)."""
    z0 = np.load(O.pre_dir("baseline", "ridge") / "fold.npz")
    for s in O.SETS:
        for m in O.EST:
            for f in O.pre_folds(s, m):
                z = np.load(f)
                assert np.array_equal(z["fit"], z0["fit"]) and np.array_equal(z["val"], z0["val"]), f
    d = O.oos_dates()

    def lab(r: int) -> str:
        return str(d[r - O.W]) if r >= O.W else f"design row {r} (before the OOS rows)"

    out = []
    for k, ((f0, f1), (v0, v1)) in enumerate(zip(z0["fit"], z0["val"])):
        out.append(
            dict(
                fold=FIRST_FOLD + k,
                fit=[int(f0), int(f1)],
                val=[int(v0), int(v1)],
                fit_first=lab(int(f0)),
                fit_last=lab(int(f1) - 1),
                val_first=lab(int(v0)),
                val_last=lab(int(v1) - 1),
                val_rows=int(v1 - v0),
            )
        )
    return out


def st_select(a) -> None:
    """The orchestrator's st_select; its info.folds (13-bar fold constants), rule and sample rewritten for the
    48-interval sample from fold.npz."""
    O.st_select(a)
    p = O.OUT / "frozen_penalties.json"
    fr = json.loads(p.read_text())
    fr["info"]["folds"] = fold_info()
    fr["info"]["first_fold_note"] = (
        f"folds 0 .. {FIRST_FOLD - 1} not fitted (--first-fold {FIRST_FOLD}; the main rule); validation = the "
        f"{O.PRE_FOLD_SESS} x {PPD} rows before the first {FIRST_DAY} row, embargo {O.PRE_EMBARGO_SESS} x {PPD}, fit = "
        f"the {O.W} rows before it"
    )
    fr["info"]["rule"] = (
        f"one penalty a model, the lowest mean validation MSE (adjusted scale, every validation row of the sample: "
        f"{DESC}) over pre-sample fold {FIRST_FOLD}; frozen over the evaluation span"
    )
    fr["info"]["sample"] = DESC + "; HAR components and trailing averages on the full 48-interval grid"
    fr["info"]["study"] = "results/all48_2026-10-30/RULE.txt Part 2 (exploratory)"
    fr["info"]["window_rows"] = O.W
    fr["info"]["window_days"] = WD
    p.write_text(json.dumps(fr, indent=1))
    say(f"folds: {fr['info']['folds']}")


def parse_chunks(specs: list[str]) -> dict:
    ch = dict(CHUNKS)
    for spec in specs:
        sm, k = spec.split("=")
        s, m = sm.split("/")
        ch[(s, m)] = int(k)
    return ch


def st_njobs(a) -> None:
    print(len(O.walk_jobs(parse_chunks(a.chunks), a.only or None)))


def st_walk1(a) -> None:
    for s in O.SETS:
        check_design(s)
    jobs = O.walk_jobs(parse_chunks(a.chunks), a.only or None)
    idx = int(os.environ["SLURM_ARRAY_TASK_ID"]) - 1 if a.task == 0 else a.task - 1
    if idx >= len(jobs):  # an array sized for more jobs
        say(f"walk task {idx + 1}: none ({len(jobs)} jobs)")
        return
    args, log, key = jobs[idx]
    say(f"walk task {idx + 1} of {len(jobs)}: {key}, start row {O.walk_start()}: {' '.join(args)}")
    O.pool([(args, log)], 1)


def ref_table(ref: Path, s: str, m: str) -> Path:
    base = ref / "_work" / "walk" / s / m
    base = base / "c0" if (base / "c0").exists() else base
    return base / "causal_tune_linear" / O.EST.get(m, "ols") / s / "results.csv"


def key3(ln: str) -> str:
    f = ln.split(",")
    return ",".join([f[0], f[2], f[3]])  # date, true_adj, pred_adj


def st_gate_har(a) -> None:
    """L1 (RULE.txt section 2.2): the HAR benchmark design is train3way's byte for byte (D3), so its penalties and the
    first walk rows must be train3way's too."""
    gd = json.loads((O.OUT / "gate_design.json").read_text())
    g: dict[str, Any] = {"rows": a.gate_rows}
    if not gd.get("D3", {}).get("run"):
        g |= dict(run=False, note="D3 not run (train3way's caches absent): L1 not run (RULE 2.2)", pass_=True)
        (O.OUT / "gate_har.json").write_text(json.dumps(g, indent=1))
        say("L1 not run (D3 not run)")
        return
    ref = Path(a.ref).resolve()
    fm, ft = (json.loads((d / "frozen_penalties.json").read_text()) for d in (O.OUT, ref))
    pen = {m: [fm[m]["baseline"], ft[m]["baseline"]] for m in O.EST}
    ok = all(v[0] == v[1] for v in pen.values())
    g["penalties_har"] = pen
    S0 = O.walk_start()
    run = []
    for args, _, (s, m, _c) in O.walk_jobs({}, HAR_MODELS):
        aa = list(args)
        aa[aa.index("--end") + 1] = str(S0 + a.gate_rows)
        o = O.WORK / "gate_har" / s / m
        aa[aa.index("--out") + 1] = str(o)
        run.append((aa, o / "run.log"))
    O.pool(run, a.jobs)
    for s, m in (x.split("/") for x in HAR_MODELS):
        mine = O.WORK / "gate_har" / s / m / "causal_tune_linear" / O.EST.get(m, "ols") / s / "results.csv"
        lm = mine.read_text().splitlines()
        lt = ref_table(ref, s, m).read_text().splitlines()[: len(lm)]
        same = len(lm) == a.gate_rows + 1 and [key3(x) for x in lm] == [key3(x) for x in lt]
        g[f"{s}/{m}"] = dict(rows=len(lm) - 1, identical=same, first=lm[1][:19], last=lm[-1][:19])
        ok &= same
    g["walk_start"] = S0
    g["run"] = True
    g["pass_"] = bool(ok)
    (O.OUT / "gate_har.json").write_text(json.dumps(g, indent=1))
    say(f"L1 {'PASS' if ok else 'FAIL'}: {json.dumps(g)}")
    if not ok:
        sys.exit(3)


def st_gate_seam(a) -> None:
    """L2 = train3way's gate_seam: a chunk from a 250-solve boundary with a W-row halo reproduces the long walk."""
    S0 = O.walk_start()
    T = O.TUNE_PER
    run = []
    jobs = O.walk_jobs({}, SEAM_MODELS)
    for args, _, (s, m, _c) in jobs:
        for lab, lo, hi in (("long", S0, S0 + 2 * T), ("seam", S0 + T, S0 + 2 * T)):
            aa = list(args)
            aa[aa.index("--start") + 1] = str(lo)
            aa[aa.index("--end") + 1] = str(hi)
            o = O.WORK / "gate_seam" / lab / s / m
            aa[aa.index("--out") + 1] = str(o)
            run.append((aa, o / "run.log"))
    O.pool(run, a.jobs)
    g: dict[str, Any] = {"walk_start": S0, "seam_row": S0 + T}
    ok = True
    for _, _, (s, m, _c) in jobs:
        rows = {}
        for lab in ("long", "seam"):
            f = O.WORK / "gate_seam" / lab / s / m / "causal_tune_linear" / O.EST.get(m, "ols") / s / "results.csv"
            rows[lab] = [key3(ln) for ln in f.read_text().splitlines()[1:]]
        same = len(rows["long"]) == 2 * T and rows["long"][T:] == rows["seam"]
        g[f"{s}/{m}"] = dict(long_rows=len(rows["long"]), seam_rows=len(rows["seam"]), identical=same)
        ok &= same
    g["pass_"] = bool(ok)
    (O.OUT / "gate_seam.json").write_text(json.dumps(g, indent=1))
    say(f"L2 {'PASS' if ok else 'FAIL'}: {json.dumps(g)}")
    if not ok:
        sys.exit(3)


def st_export(a) -> None:
    """Every walked row of the seven models (no RTH filter): plain = pred_adj^2 x B (the design's b_oos)."""
    d = O.oos_dates()
    B = pd.Series(np.load(O.design_dir("baseline") / "b_oos.npy"), index=pd.DatetimeIndex(d))
    assert B.index.is_unique
    core = pd.read_parquet(REPO / "data" / "core_stats.parquet", columns=["endbartime", "sumret2"])
    rv = pd.Series(core["sumret2"].to_numpy(float), index=pd.DatetimeIndex(pd.to_datetime(core["endbartime"])))
    rv = rv[~rv.index.duplicated()]
    fr = json.loads((O.OUT / "frozen_penalties.json").read_text())
    models = [(s, m) for s in O.SETS for m in O.EST] + [("baseline", "ols")]
    parts, rec = [], {}
    for s, m in models:
        r = O.walk_table(s, m)
        t = pd.DatetimeIndex(r["date"])
        b = B.reindex(t).to_numpy()
        assert np.isfinite(b).all(), (s, m)
        ok = r["true_adj"].to_numpy() > 0
        assert np.allclose(r["true_raw"][ok] / r["true_adj"][ok] ** 2, b[ok], rtol=1e-9), (s, m)
        k0 = int(d.searchsorted(t[0]))
        assert t.equals(pd.DatetimeIndex(d[k0:])), f"{s}/{m}: the walk does not cover every row to the end"
        alpha = fr[m][s] if m != "ols" else None
        parts.append(
            pd.DataFrame(
                dict(
                    model=m,
                    predictor_set=s,
                    alpha=np.nan if alpha is None else float(alpha),
                    date=t,
                    pred_adj=r["pred_adj"].to_numpy(float),
                    true_adj=r["true_adj"].to_numpy(float),
                    B=b,
                    plain=r["pred_adj"].to_numpy(float) ** 2 * b,
                    rv_actual=rv.reindex(t).to_numpy(),
                )
            )
        )
        rec[f"{s}/{m}"] = dict(rows=len(t), first=str(t[0]), last=str(t[-1]), clocks=int(t.strftime("%H:%M").nunique()))
    fc = pd.concat(parts, ignore_index=True)
    fc.to_parquet(O.OUT / "forecasts_all.parquet", index=False)
    (O.OUT / "forecasts_all.json").write_text(json.dumps(dict(models=rec, rows=len(fc)), indent=1))
    say(f"forecasts_all.parquet: {len(fc)} rows, {rec}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    stages = ("design", "gate", "pretune", "select", "gate_har", "gate_seam", "njobs", "walk1", "export")
    ap.add_argument("stage", choices=stages)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--chunks", action="append", default=[], help="set/model=K (default ridge 2, lasso 8, enet 5)")
    ap.add_argument("--only", action="append", default=[])
    ap.add_argument("--task", type=int, default=0, help="walk1: job number (1-based); 0 = SLURM_ARRAY_TASK_ID")
    ap.add_argument("--cellb", default="/home1/jc_905/fomcfix_lin/runs/B", help="gate: the cell-B 13-bar run dir")
    ap.add_argument("--ref", default="", help="gate / gate_har: train3way's ALL48 run dir (read only)")
    ap.add_argument("--gate-rows", type=int, default=50)
    a = ap.parse_args(argv)
    configure(Path(a.out))
    {
        "design": st_design,
        "gate": st_gate,
        "pretune": st_pretune,
        "select": st_select,
        "gate_har": st_gate_har,
        "gate_seam": st_gate_seam,
        "njobs": st_njobs,
        "walk1": st_walk1,
        "export": st_export,
    }[a.stage](a)


if __name__ == "__main__":
    main()
