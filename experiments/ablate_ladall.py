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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("tasks", "gate", "t2", "merge", "lingate", "check"))
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
    a = ap.parse_args()
    return {
        "tasks": cmd_tasks,
        "gate": cmd_gate,
        "t2": cmd_t2,
        "merge": cmd_merge,
        "lingate": cmd_lingate,
        "check": cmd_check,
    }[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
