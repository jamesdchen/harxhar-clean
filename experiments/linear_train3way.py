"""Three-way training-sample comparison of the pooled linear models, 2014 start, 3,000-day window
(results/linear_train3way_2014start_w3000/RULE.txt, written before any job).

The final 13-bar linear run (results/window_sweep_2014start_2026-10-22/w3000) is the template: its orchestrator
experiments/pooled_linear_rerun_2026_10_19_13.py is imported and its stage functions run per arm, with the arm's sample
set in the orchestrator's globals (sample, rows a day, window rows, pre-tune halo) and the runner
experiments/linear_train3way_runner.py (experiments/pool_linear_pretune.py with the close / close_r samples).  Every
other setting is the template's: --har-base 5, the template's executor flags, --window-days 3000, --first-day
2014-01-02, --first-fold 3, LATE_START_WARMUP=1, STANDARDIZE_PENALTY / WALK_KKT_RESEED / the grid rule at their
defaults (on), penalty factor 0.01 / 1 on the HAR benchmark block, chunks of 250 solves.

Arms (RULE.txt): ALL48 (--sample 48, the overnight indicator kept), RTH13 (the template; rebuilt here only for the gate
and the secondary re-tune, its forecasts reused), CLOSE (the 16:00 rows of the rebuilt RTH13 design: derive_close),
CLOSE_NAT (secondary: the closing half hour built natively, segment bar1600).

usage: python experiments/linear_train3way.py <stage> --arm ARM --out DIR [options]
  design        the arm's two design caches (+ dates_all.csv); not for CLOSE
  derive_close  CLOSE's caches from a rebuilt RTH13 design (--from <RTH13 out dir>)
  pretune       fold 3, every grid penalty, the grid rule's extensions (the orchestrator's st_pretune)
  select        frozen_penalties.json, pretune_mse.csv (st_select; info.folds / rule / sample rewritten for the arm)
  walk1         one walk job (Slurm array task SLURM_ARRAY_TASK_ID, 1-based, in walk_jobs order; --chunks set/model=K)
  gate          RTH13 only: gates (i)-(iii) of RULE.txt against --ref <template run dir> -> gate.json
  select16      secondary: the penalty re-selected on fold 3's 16:00 validation rows (+ its grid-rule extensions)
  walk16        secondary walks (array task) of the models whose 16:00 pick differs from the primary
  export        forecasts_rth.parquet (the arm's forecasts on the regular-hours bar ends 10:00 .. 16:00) and, after
                select16, forecasts16_rth.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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

WD, FIRST_DAY, FIRST_FOLD = 3000, "2014-01-02", 3
FLAGS_RTH = ["DROP_ACTIVITY_MA=1", "CALENDAR_THROUGH_OVERNIGHT=1", "DROP_OVERNIGHT_INDICATOR=1", "LEAN_GLOBAL_LAGS=1"]
CAL = [f"DOW_{d}" for d in range(5)] + ["hour"]
ARMS: dict[str, dict[str, Any]] = {
    "ALL48": dict(
        sample="48",
        ppd=48,
        flags=[
            "DROP_ACTIVITY_MA=1",
            "CALENDAR_THROUGH_OVERNIGHT=1",
            "DROP_OVERNIGHT_INDICATOR=0",
            "LEAN_GLOBAL_LAGS=1",
        ],
        cal=CAL + ["is_overnight"],
        desc="all 48 half-hour intervals of the 48-interval grid (overnight included)",
    ),
    "RTH13": dict(
        sample="rth13",
        ppd=13,
        flags=FLAGS_RTH,
        cal=CAL,
        desc="13 regular-hours bars, end stamps 10:00-16:00 ET",
    ),
    "CLOSE": dict(
        sample="close_r",
        ppd=1,
        flags=FLAGS_RTH,
        cal=CAL,
        desc="the closing half hour (16:00 rows of the 13-bar design)",
    ),
    "CLOSE_NAT": dict(
        sample="close",
        ppd=1,
        flags=FLAGS_RTH,
        cal=CAL,
        desc="the closing half hour, design built on the closing rows (segment bar1600)",
    ),
}
RUNNER = HERE / "linear_train3way_runner.py"
RTH_CLOCKS = {f"{h:02d}:{m:02d}" for h in range(10, 17) for m in (0, 30)} - {"16:30"}
ARM = ""


def configure(arm: str, out: Path) -> None:
    """The orchestrator's globals for this arm (as its main() sets them from the command line)."""
    global ARM
    ARM = arm
    c = ARMS[arm]
    O.OUT = out.resolve()
    O.WORK = O.OUT / "_work"
    O.RUNNER = RUNNER
    cfg = ["--sample", c["sample"], "--har-base", "5"]
    for f in c["flags"]:
        cfg += ["--ex-flag", f]
    cfg += ["--window-days", str(WD), "--ex-flag", "LATE_START_WARMUP=1"]
    O.CFG[:] = cfg
    O.PPD, O.TW_DAYS = c["ppd"], WD
    O.W = WD * c["ppd"]
    O.FIRST_FOLD = FIRST_FOLD
    O.PRE_HALO = (O.PRE_FOLDS - FIRST_FOLD) * O.PRE_FOLD_SESS * O.PPD + O.PRE_EMBARGO_SESS * O.PPD + O.W
    O.FIRST_DAY = FIRST_DAY
    O.check_design = check_design
    O.OUT.mkdir(parents=True, exist_ok=True)


def check_design(s: str) -> list[str]:
    """The orchestrator's check_design with the arm's calendar (ALL48: + is_overnight)."""
    all_features, fomc = O.design_features()
    names = json.loads((O.design_dir(s) / "design.json").read_text())["names"]
    har = [n for n in names if n.startswith("har_ma_")]
    assert har == [f"har_ma_{k}" for k in O.HAR_LADDER], har
    cal = [n for n in names if not re.search(r"_ma_\d+$", n)]
    assert cal == ARMS[ARM]["cal"], cal
    assert not any("_active_ma_" in n for n in names)
    if s == "all_features":
        assert set(fomc) <= set(all_features)
        for f in fomc:
            for k in O.HAR_LADDER:
                assert f"adj_{f}_ma_{k}" in names and f"{f}_avail_ma_{k}" in names, (f, k)
        exog = {n for n in names if n.startswith("adj_")}
        assert exog == {f"adj_{f}_ma_{k}" for f in all_features for k in O.HAR_LADDER}, sorted(exog)[:5]
    else:
        assert len(names) == len(O.HAR_LADDER) + len(ARMS[ARM]["cal"]), names
    return names


def dates_all(s: str = "baseline") -> pd.Series:
    return pd.read_csv(O.design_dir(s) / "dates_all.csv", dtype=str)["date"]


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


# ============================================================================ stages
def st_design(a) -> None:
    assert ARM != "CLOSE", "CLOSE is derived from the RTH13 design (derive_close)"
    O.st_design(a)
    for s in O.SETS:
        assert (O.design_dir(s) / "dates_all.csv").exists(), s
        assert dates_all(s).equals(dates_all("baseline")), s


def st_derive_close(a) -> None:
    """CLOSE's design caches: the 16:00 rows of the rebuilt RTH13 caches (--from), X / y / B / names unchanged."""
    src = Path(a.src).resolve() / "_work"
    W_r = WD * 13
    for s in O.SETS:
        dz = src / f"design_{s}"
        meta = json.loads((dz / "design.json").read_text())
        assert meta["config"]["sample"] == "rth13" and int(meta["config"]["window"]) == W_r, meta["config"]
        da = pd.read_csv(dz / "dates_all.csv", dtype=str)["date"]
        X = np.load(dz / "X.npy", mmap_mode="r")
        y, b = np.load(dz / "y.npy"), np.load(dz / "b_oos.npy")
        assert len(da) == X.shape[0] == len(y) == W_r + len(b)
        r16 = np.flatnonzero(da.str.slice(11, 16).to_numpy() == "16:00")
        k0 = int(np.flatnonzero(r16 >= W_r)[0])  # the first 16:00 row with a stored B
        st = max(0, k0 - WD)
        rows = r16[st:]
        assert rows[WD] >= W_r, (rows[WD], W_r)
        Xc = X[rows]
        Xc = np.asfortranarray(Xc) if meta["f_order"] else np.ascontiguousarray(Xc)
        o = O.design_dir(s)
        o.mkdir(parents=True, exist_ok=True)
        np.save(o / "X.npy", Xc)
        np.save(o / "y.npy", np.ascontiguousarray(y[rows]))
        np.save(o / "b_oos.npy", np.ascontiguousarray(b[rows[WD:] - W_r]))
        np.save(o / "rth13_rows.npy", rows)
        pd.DataFrame({"date": da.iloc[rows[WD:]].to_numpy()}).to_csv(o / "dates_oos.csv", index=False)
        pd.DataFrame({"date": da.iloc[rows].to_numpy()}).to_csv(o / "dates_all.csv", index=False)
        cfg = dict(meta["config"], sample="close_r", periods_per_day=1, window=WD)
        out = dict(
            meta,
            config=cfg,
            shape=list(Xc.shape),
            derived_from=dict(
                design=str(dz),
                rule="the 16:00 rows of the rth13 design, from the WD-th 16:00 row before the first one with a stored B",
                rth13_rows_first_last=[int(rows[0]), int(rows[-1])],
                n_rows=int(len(rows)),
                x_md5=md5(dz / "X.npy"),
            ),
        )
        (o / "design.json").write_text(json.dumps(out))
        O.say(f"derive_close {s}: {Xc.shape} from {X.shape}, rth13 rows {rows[0]} .. {rows[-1]}")
    for s in O.SETS:
        check_design(s)


def st_pretune(a) -> None:
    O.st_pretune(a)


def fold_info() -> list[dict]:
    """The fitted fold's fit and validation rows (fold.npz, every model alike) with their first / last date labels."""
    z0 = np.load(O.pre_dir("baseline", "ridge") / "fold.npz")
    for s in O.SETS:
        for m in O.EST:
            for f in O.pre_folds(s, m):
                z = np.load(f)
                assert np.array_equal(z["fit"], z0["fit"]) and np.array_equal(z["val"], z0["val"]), f
    d = dates_all()
    out = []
    for k, ((f0, f1), (v0, v1)) in enumerate(zip(z0["fit"], z0["val"])):
        out.append(
            dict(
                fold=FIRST_FOLD + k,
                fit=[int(f0), int(f1)],
                val=[int(v0), int(v1)],
                fit_first=d[int(f0)],
                fit_last=d[int(f1) - 1],
                val_first=d[int(v0)],
                val_last=d[int(v1) - 1],
                val_rows=int(v1 - v0),
                val_rows_1600=int((d.iloc[int(v0) : int(v1)].str.slice(11, 16) == "16:00").sum()),
            )
        )
    return out


def st_select(a) -> None:
    """The orchestrator's st_select (penalties, pretune_mse.csv, grid rule record).  Its info.folds labels are computed
    with the 13-bar fold constants (3,250 / 325 rows) and can index past the date list of a 1-row-a-day design, so the
    date list it reads is padded at the end for the call (first_row reads its start only) and info.folds,
    first_fold_note, rule and sample are rewritten from the arm's own fold.npz and dates_all.csv."""
    od = O.oos_dates
    pad = od()
    O.oos_dates = lambda s="baseline": pd.concat([od(s), pd.Series([pad.iloc[-1]] * 4000)], ignore_index=True)
    try:
        O.st_select(a)
    finally:
        O.oos_dates = od
    p = O.OUT / "frozen_penalties.json"
    fr = json.loads(p.read_text())
    fr["info"]["folds"] = fold_info()
    S = O.first_row()
    f2 = S - 2 * O.PRE_FOLD_SESS * O.PPD - O.PRE_EMBARGO_SESS * O.PPD - O.W
    fr["info"]["first_fold_note"] = (
        f"folds 0 .. {FIRST_FOLD - 1} not fitted (--first-fold {FIRST_FOLD}; the template's rule); fold "
        f"{FIRST_FOLD - 1}'s {O.W}-row fit window would start at design row {f2}"
    )
    fr["info"]["rule"] = (
        f"one penalty a model, the lowest mean validation MSE (adjusted scale, every validation row of the arm's sample: "
        f"{ARMS[ARM]['desc']}) over pre-sample fold {FIRST_FOLD}; frozen over the evaluation span"
    )
    fr["info"]["sample"] = ARMS[ARM]["desc"] + "; HAR components and trailing averages on the full 48-interval grid"
    fr["info"]["arm"] = ARM
    fr["info"]["window_rows"] = O.W
    p.write_text(json.dumps(fr, indent=1))
    O.say(f"{ARM} folds: {fr['info']['folds']}")


def parse_chunks(specs: list[str]) -> dict:
    ch = {}
    for spec in specs:
        sm, k = spec.split("=")
        s, m = sm.split("/")
        ch[(s, m)] = int(k)
    return ch


def st_walk1(a) -> None:
    for s in O.SETS:
        check_design(s)
    jobs = O.walk_jobs(parse_chunks(a.chunks), a.only)
    idx = int(os.environ["SLURM_ARRAY_TASK_ID"]) - 1 if a.task == 0 else a.task - 1
    if idx >= len(jobs):  # an array sized for the most chunks
        O.say(f"{ARM} walk task {idx + 1}: none ({len(jobs)} jobs)")
        return
    args, log, key = jobs[idx]
    O.say(f"{ARM} walk task {idx + 1} of {len(jobs)}: {key}, start row {O.walk_start()}: {' '.join(args)}")
    O.pool([(args, log)], 1)


# ---------------------------------------------------------------------------- gate (RTH13)
def st_gate(a) -> None:
    """RULE.txt gates (i) the rebuilt design caches, (ii) the pretune / select, (iii) 50-origin walk slices, against
    the template run (--ref, read only)."""
    assert ARM == "RTH13"
    ref = Path(a.ref).resolve()
    g: dict = {"ref": str(ref)}
    ok = True
    # (i) the design caches, byte for byte
    for s in O.SETS:
        mine, theirs = O.design_dir(s), ref / "_work" / f"design_{s}"
        r: dict[str, Any] = {}
        for fn in ("X.npy", "y.npy", "b_oos.npy", "dates_oos.csv"):
            r[fn] = [md5(mine / fn), md5(theirs / fn)]
        jm, jt = (json.loads((d / "design.json").read_text()) for d in (mine, theirs))
        for k in ("names", "shape", "hp", "config", "raw_cols", "bucket", "f_order"):
            r[k] = jm.get(k) == jt.get(k)
        r["pass"] = all(v[0] == v[1] for k, v in r.items() if k.endswith((".npy", ".csv"))) and all(
            v for k, v in r.items() if not k.endswith((".npy", ".csv"))
        )
        ok &= r["pass"]
        g[f"i_design_{s}"] = r
    # (ii) the pretune and the selection
    fm, ft = (json.loads((d / "frozen_penalties.json").read_text()) for d in (O.OUT, ref))
    pen_ok = all(fm[m] == ft[m] for m in ("ridge", "lasso", "enet", "ols"))
    pm, pt = pd.read_csv(O.OUT / "pretune_mse.csv"), pd.read_csv(ref / "pretune_mse.csv")
    mse_ok = pm.shape == pt.shape and bool((pm.astype(str).to_numpy() == pt.astype(str).to_numpy()).all())
    g["ii_pretune"] = dict(penalties_equal=pen_ok, pretune_mse_equal=mse_ok, mine={m: fm[m] for m in O.EST})
    ok &= pen_ok and mse_ok
    # (iii) walk slices: the first a.gate_rows origins from the walk start, the template's chunking
    only = ["all_features/ridge", "all_features/lasso", "all_features/enet", "baseline/ols"]
    ch = {("all_features", "ridge"): 2, ("all_features", "lasso"): 4, ("all_features", "enet"): 4}
    jobs = [j for j in O.walk_jobs(ch, only) if j[2][2] == 0]
    S0 = O.walk_start()
    run = []
    for args, _, (s, m, _c) in jobs:
        args = list(args)
        args[args.index("--end") + 1] = str(S0 + a.gate_rows)
        o = O.WORK / "gate_walk" / s / m
        args[args.index("--out") + 1] = str(o)
        run.append((args, o / "run.log"))
    O.pool(run, a.jobs)
    for _, _, (s, m, _c) in jobs:
        est = O.EST.get(m, "ols")
        mine = O.WORK / "gate_walk" / s / m / "causal_tune_linear" / est / s / "results.csv"
        base = ref / "_work" / "walk" / s / m
        theirs = (base / "c0" if (base / "c0").exists() else base) / "causal_tune_linear" / est / s / "results.csv"
        lm = mine.read_text().splitlines()
        lt = theirs.read_text().splitlines()[: len(lm)]

        def key3(ln: str) -> str:
            f = ln.split(",")
            return ",".join([f[0], f[2], f[3]])  # date, true_adj, pred_adj

        same = len(lm) == a.gate_rows + 1 and [key3(x) for x in lm] == [key3(x) for x in lt]
        g[f"iii_walk_{s}_{m}"] = dict(rows=len(lm) - 1, identical=same, first=lm[1][:19], last=lm[-1][:19])
        ok &= same
    g["pass"] = bool(ok)
    (O.OUT / "gate.json").write_text(json.dumps(g, indent=1))
    O.say(f"GATE {'PASS' if ok else 'FAIL'}: {json.dumps(g)}")
    if not ok:  # the dependent jobs (afterok) do not start
        sys.exit(3)


# ---------------------------------------------------------------------------- secondary: tuned at 16:00
def pre16_dir(s: str, m: str) -> Path:
    return O.WORK / "pretune16" / s / m


def pick16(s: str, m: str) -> tuple[np.ndarray, np.ndarray, int]:
    """(grid ascending, mean validation MSE over fold 3's 16:00 validation rows, argmin) over every pretuned penalty
    (the spec grid, the primary grid rule's extensions, the secondary's)."""
    files = O.pre_folds(s, m) + sorted(pre16_dir(s, m).glob("ext_*/fold.npz"))
    zs = [np.load(p) for p in files]
    for z in zs[1:]:
        assert np.array_equal(z["val"], zs[0]["val"]), (s, m)
    d = dates_all()
    keep = [(d.iloc[int(v0) : int(v1)].str.slice(11, 16) == "16:00").to_numpy() for v0, v1 in zs[0]["val"]]
    grid = np.concatenate([z["grid"] for z in zs])
    o = np.argsort(grid, kind="stable")
    res = np.concatenate([z["resid"] for z in zs], axis=1)[:, o]
    grid = grid[o]
    assert len(np.unique(grid)) == len(grid), (s, m, grid)
    crit = np.mean([(res[f][:, keep[f]] ** 2).mean(axis=1) for f in range(len(keep))], axis=0)
    return grid, crit, int(np.argmin(crit))


def grid_state16(s: str, m: str) -> dict:
    """The grid rule (RULE_std.txt; the orchestrator's grid_state) on the 16:00 criterion."""
    base = np.load(O.pre_dir(s, m) / "fold.npz")["grid"]
    grid, crit, k = pick16(s, m)
    added = [float(g) for g in grid if g < base.min() or g > base.max()]
    st: dict = dict(added=added, selected=float(grid[k]))
    if 0 < k < len(grid) - 1:
        return st | dict(status="interior")
    down = k == 0
    n_dir = sum((g < base.min()) if down else (g > base.max()) for g in added)
    end, nxt = (0, 1) if down else (len(grid) - 1, len(grid) - 2)
    if n_dir and abs(crit[end] - crit[nxt]) <= O.GRID_EXT_FLAT * crit[nxt]:
        return st | dict(status="flat", direction="down" if down else "up")
    if n_dir >= O.GRID_EXT_CAP:
        return st | dict(status="cap", direction="down" if down else "up")
    e = int(round(float(np.log10(grid[end])))) + (-1 if down else 1)
    return st | dict(status="extend", direction="down" if down else "up", alpha=float(f"1e{e}"))


def st_select16(a) -> None:
    assert ARM in ("ALL48", "RTH13"), ARM
    S = O.first_row()
    for rnd in range(1, 2 * O.GRID_EXT_CAP + 2):
        ext = []
        for s in O.SETS:
            for m in O.EST:
                st = grid_state16(s, m)
                if st["status"] != "extend":
                    continue
                o = pre16_dir(s, m) / f"ext_{st['alpha']!r}"
                ext.append((O.pretune_args(s, m, o, S) + ["--grid-alphas", repr(st["alpha"])], o / "run.log"))
                O.say(f"16:00 grid rule round {rnd}: {s}/{m} selected {st['selected']!r}, adding {st['alpha']!r}")
        if not ext:
            break
        O.pool(ext, a.jobs)
    prim = json.loads((O.OUT / "frozen_penalties.json").read_text())
    pen: dict = {}
    rows: list[dict] = []
    exts: dict = {}
    off: dict = {}
    for s in O.SETS:
        for m in O.EST:
            grid, crit, k = pick16(s, m)
            st = grid_state16(s, m)
            assert st["status"] != "extend", (s, m, st)
            pen.setdefault(m, {})[s] = float(grid[k])
            exts.setdefault(m, {})[s] = st
            base = np.load(O.pre_dir(s, m) / "fold.npz")["grid"]
            if float(grid[k]) not in [float(g) for g in base]:
                off.setdefault(m, {})[s] = float(grid[k])
            for j, al in enumerate(grid):
                rows.append(
                    dict(
                        model=m,
                        predictor_set=s,
                        alpha=float(al),
                        mse16=float(crit[j]),
                        picked16=j == k,
                        primary=np.isclose(al, prim[m][s], rtol=1e-12),
                    )
                )
    pen["ols"] = {"baseline": None}
    info = dict(
        prim["info"],
        rule=f"secondary: one penalty a model, the lowest mean validation MSE (adjusted scale) over the validation rows "
        f"ending 16:00 of pre-sample fold {FIRST_FOLD} ({ARMS[ARM]['desc']} sample); frozen over the evaluation span",
        grid_extensions=exts,
        off_spec_grid=off,
        primary={m: prim[m] for m in O.EST},
        changed={m: {s: pen[m][s] != prim[m][s] for s in O.SETS} for m in O.EST},
    )
    d16 = O.OUT / "sec16"
    d16.mkdir(exist_ok=True)
    (d16 / "frozen_penalties.json").write_text(
        json.dumps(
            dict(**{m: pen[m] for m in ("ridge", "lasso", "enet", "ols")}, designs=prim["designs"], info=info), indent=1
        )
    )
    pd.DataFrame(rows).to_csv(d16 / "pretune16_mse.csv", index=False)
    O.say(f"16:00 penalties: { {m: pen[m] for m in O.EST} }; changed: {info['changed']}")


def sec16_jobs(chunks: dict) -> list:
    """walk_jobs on the secondary penalties, for the models whose 16:00 pick differs (walk dirs _work/walk16/...)."""
    d16 = O.OUT / "sec16"
    fr = json.loads((d16 / "frozen_penalties.json").read_text())
    only = [f"{s}/{m}" for m in O.EST for s in O.SETS if fr["info"]["changed"][m][s]]
    if not only:
        return []
    out0, wd0 = O.OUT, O.walk_dir
    O.OUT = d16
    O.walk_dir = lambda s, m, c=None: (
        (O.WORK / "walk16" / s / m) if c is None else (O.WORK / "walk16" / s / m / f"c{c}")
    )
    try:
        return O.walk_jobs(chunks, only)
    finally:
        O.OUT, O.walk_dir = out0, wd0


def st_walk16(a) -> None:
    jobs = sec16_jobs(parse_chunks(a.chunks))
    if a.task < 0:  # the number of jobs, for the array size
        print(len(jobs))
        return
    idx = int(os.environ["SLURM_ARRAY_TASK_ID"]) - 1 if a.task == 0 else a.task - 1
    if idx >= len(jobs):  # an array sized for the most chunks
        O.say(f"{ARM} walk16 task {idx + 1}: none ({len(jobs)} jobs)")
        return
    args, log, key = jobs[idx]
    O.say(f"{ARM} walk16 task {idx + 1} of {len(jobs)}: {key}: {' '.join(args)}")
    O.pool([(args, log)], 1)


# ---------------------------------------------------------------------------- export
def frame(s: str, m: str, alpha, B: pd.Series, rv_bar: pd.Series, walk16: bool = False) -> pd.DataFrame:
    wd0 = O.walk_dir
    if walk16:
        O.walk_dir = lambda s, m, c=None: O.WORK / "walk16" / s / m
    try:
        r = O.walk_table(s, m)
    finally:
        O.walk_dir = wd0
    t = pd.DatetimeIndex(r["date"])
    b = B.reindex(t).to_numpy()
    assert np.isfinite(b).all()
    ok = r["true_adj"].to_numpy() > 0
    assert np.allclose(r["true_raw"][ok] / r["true_adj"][ok] ** 2, b[ok], rtol=1e-9), (s, m)
    f = pd.DataFrame(
        dict(
            series_id=f"linear_train3way/{ARM}{'@16' if walk16 else ''}/{m}_{s}",
            model=m,
            predictor_set=s,
            alpha=np.nan if alpha is None else float(alpha),
            date=r["date"],
            pred_adj=r["pred_adj"],
            true_adj=r["true_adj"],
            B=b,
            plain=r["pred_adj"].to_numpy() ** 2 * b,
            pred_raw_duan=r["pred_raw"],
            true_raw=r["true_raw"],
            rv_actual=rv_bar.reindex(t).to_numpy(),
        )
    )
    return f[f["date"].dt.strftime("%H:%M").isin(RTH_CLOCKS).to_numpy()]


def st_export(a) -> None:
    d = O.oos_dates()
    B = pd.Series(np.load(O.design_dir("baseline") / "b_oos.npy"), index=pd.DatetimeIndex(d))
    assert B.index.is_unique
    core = pd.read_parquet(REPO / "data" / "core_stats.parquet", columns=["endbartime", "sumret2"])
    rv_bar = pd.Series(core["sumret2"].to_numpy(float), index=pd.DatetimeIndex(pd.to_datetime(core["endbartime"])))
    rv_bar = rv_bar[~rv_bar.index.duplicated()]
    models = [(s, m) for s in O.SETS for m in O.EST] + [("baseline", "ols")]
    fr = json.loads((O.OUT / "frozen_penalties.json").read_text())
    if ARM == "RTH13":  # the template's forecasts, reused (RULE.txt, conditional on the gate)
        assert json.loads((O.OUT / "gate.json").read_text())["pass"]
        fc = pd.read_parquet(Path(a.ref).resolve() / "forecasts.parquet")
        assert fc["date"].dt.strftime("%H:%M").isin(RTH_CLOCKS).all()
    else:
        fc = pd.concat([frame(s, m, fr[m][s], B, rv_bar) for s, m in models], ignore_index=True)
    fc.to_parquet(O.OUT / "forecasts_rth.parquet", index=False)
    O.say(f"{ARM} forecasts_rth.parquet: {len(fc)} rows, {fc.groupby(['model', 'predictor_set']).size().to_dict()}")
    p16 = O.OUT / "sec16" / "frozen_penalties.json"
    if a.with16 and p16.exists():
        f16 = json.loads(p16.read_text())
        parts = []
        for s, m in models:
            ch = m != "ols" and f16["info"]["changed"][m][s]
            parts.append(
                frame(s, m, f16[m][s], B, rv_bar, walk16=ch) if ch else fc[(fc.model == m) & (fc.predictor_set == s)]
            )
        f = pd.concat(parts, ignore_index=True)
        f["changed16"] = [m != "ols" and f16["info"]["changed"][m][s] for m, s in zip(f["model"], f["predictor_set"])]
        f.to_parquet(O.OUT / "forecasts16_rth.parquet", index=False)
        O.say(f"{ARM} forecasts16_rth.parquet: {len(f)} rows")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "stage",
        choices=("design", "derive_close", "pretune", "select", "walk1", "gate", "select16", "walk16", "export"),
    )
    ap.add_argument("--arm", required=True, choices=tuple(ARMS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--chunks", action="append", default=[])
    ap.add_argument("--only", action="append", default=[])
    ap.add_argument("--task", type=int, default=0, help="walk1 / walk16: job number (1-based); 0 = SLURM_ARRAY_TASK_ID")
    ap.add_argument("--from", dest="src", default="", help="derive_close: the RTH13 out dir")
    ap.add_argument("--ref", default="", help="gate: the template run dir")
    ap.add_argument("--gate-rows", type=int, default=50)
    ap.add_argument("--with16", action="store_true")
    a = ap.parse_args()
    configure(a.arm, Path(a.out))
    {
        "design": st_design,
        "derive_close": st_derive_close,
        "pretune": st_pretune,
        "select": st_select,
        "walk1": st_walk1,
        "gate": st_gate,
        "select16": st_select16,
        "walk16": st_walk16,
        "export": st_export,
    }[a.stage](a)


if __name__ == "__main__":
    main()
