"""The pooled ridge-initialised MinGRU, final rerun on the final design (2026-10-22): W = 3,000 trading days, first
evaluation day 2014-01-02, tuned on the single pre-test fold 3 (validation 2013), widened grid, 5 seeds, daily refit.
The same rule at W = 3,500 days (owner, 2026-10-22), tuned separately: PMF_WINDOW_DAYS=3500 (design .../w3500/_work,
output results/pooled_mingru_final_2014start_w3500).

Rule: results/pooled_mingru_final_2014start_w3000/RULE_gru.txt (written before any tuning job).  Everything not listed
there is the previous run's (experiments/pooled_mingru_rerun_13.py, results/pooled_mingru_rerun13_2026-10-22): trainer
native_g9 (binary md5 44ab6fff, P100), network, loss, early stopping, penalty factors, ridge init, selection criterion.

DESIGN  The 2014-start window sweep's W = 3,000 design caches (CARC, read-only): /scratch1/jc_905/wsweep/code/results/
        window_sweep_2014start_2026-10-22/w3000/_work/design_<set>/ (X.npy prescaled, Fortran order; y.npy; b_oos.npy;
        dates_oos.csv; design row r >= W has date dates_oos[r - W]).  `export` writes the trainer's row-major copies.

Stages (python experiments/pooled_mingru_final_2014_w3000.py <stage> ...)
  export      --design DIR --data DATA [--prev-rows CSV]   DATA/pmf_<set>/ (X.f32, X.f64, y.f32, meta.txt, enc_pf.f64,
                                                          columns.txt), DATA/pmf_rows.csv, DATA/pmf_export.json
  plan        --data DATA --inputs IN --lin-run DIR      IN/plan.json (fold 3, the daily walk origins)
  inits-tune  --data DATA --inputs IN                    ridge inits of fold 3, every ladder L x wd_lin
  tune-jobs   --inputs IN --round R                      IN/tune_<set>_r<R>.<i>.txt for the configurations of the
                                                          set's current grid not yet issued (round 0: the widened grid)
  select      --data --inputs --runs --out --round R [--tune-gpu-hours X]
                                                          the pick of every open set; an extension (next round's grid)
                                                          or the frozen configuration; OUT/frozen_config.json when
                                                          both sets are frozen
  walk-inits  --data --inputs --out --part i --parts n   the frozen (L, wd_lin) ridge init of every walk origin
  walk-jobs   --inputs --out                             IN/walk_<set>.p<i>.txt
  canary      --data --inputs                            IN/canary.txt (+ its inits)
  check-run   --inputs --run DIR                         trainer kept columns = init's, forecast rows = [o, next)
  merge       --data --inputs --runs --out               OUT/forecasts.parquet, forecasts_seeds.parquet, forecasts.json
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(os.environ.get("PMF_REPO", Path(__file__).resolve().parents[1]))
for _p in (REPO, REPO / "experiments", REPO / "notebooks", REPO / "writeup"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# 2026-10-22 (owner, "Test mingru 3.5k as well"): the same rule at W = 3,500 days, tuned separately; the window is
# PMF_WINDOW_DAYS (3000 or 3500; default 3000), each window in its own CARC root and output directory
PPD, WINDOW_DAYS = 13, int(os.environ.get("PMF_WINDOW_DAYS", "3000"))
assert WINDOW_DAYS in (3000, 3500), WINDOW_DAYS
W = PPD * WINDOW_DAYS  # 39,000 or 45,500 bars
OUT_DEFAULT = (
    Path(__file__).resolve().parents[1]
    / "results"
    / f"pooled_mingru_final_2014start_w{WINDOW_DAYS}"
)
N_ROWS = 84_747
N_COLS = {"baseline": 12, "all_features": 564}
SETS = ("baseline", "all_features")
SEEDS = (20261012, 20261013, 20261014, 20261015, 20261016)
AXES = (
    "L",
    "H",
    "lr",
    "wd_enc",
    "wd_lin",
    "lr_lin",
)  # the order of the configuration name
LADDER: dict[str, tuple] = {  # RULE_gru.txt EXTEND: the ends are the caps
    "H": (4, 16, 64, 256, 1024),
    "L": (3, 8, 20, 48, 115),
    "lr": (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1),
    "wd_enc": (0, 3, 15, 75, 375, 1875),
    "wd_lin": (0.003, 0.03, 0.3, 3, 30, 300, 3000),
    "lr_lin": (0, 1e-05, 1e-04, 1e-03, 1e-02),
}
NATURAL_LOW = {"wd_enc": "0", "lr_lin": "0"}  # not extensible downwards
GRID0: dict[str, tuple] = {  # RULE_gru.txt GRID round 0
    "H": (16, 64, 256),
    "L": (8, 20, 48),
    "lr": (0.001, 0.003, 0.01),
    "wd_enc": (0, 3, 15, 75),
    "wd_lin": (0.03, 0.3, 3, 30),
    "lr_lin": (0, 1e-05, 1e-04),
}
PREV_PICK = {
    "baseline": "L20_H64_lr0.003_we15_wl0.3_ll1e-05",
    "all_features": "L20_H64_lr0.003_we3_wl3_ll1e-05",
}
COMPUTE_CAP_GPU_H = 80.0
PF_BB, PF_EX = 0.01, 1.0
VAL, EMB = 3250, 325
FOLD: dict[str, Any] = dict(
    k=3, fit=[47202 - W, 47202], val=[47527, 50777]
)  # W = 3,000: fit [8,202, 47,202); W = 3,500: [1,702, 47,202)
FIRST_EVAL_ROW, FIRST_EVAL = 50_777, "2014-01-02 10:00:00"
COMMON = dict(
    layers=1,
    batch=128,
    max_epochs=100,
    patience=10,
    momentum=0.95,
    max_grad_norm=1.0,
    anneal=1,
    min_lr_ratio=0.0,
    mask=1,
    xstd=0,
    link="softplus",
    loss="qlike",
    lin=1,
    dec_zero=1,
)
CFG_RE = re.compile(
    r"^L(\d+)_H(\d+)_lr([0-9.e+-]+)_we([0-9.e+-]+)_wl([0-9.e+-]+)_ll([0-9.e+-]+)$"
)
TUNE_RE = re.compile(
    r"^ftune_(baseline|all_features)_f3_(L\d+_H\d+_lr[0-9.e+-]+_we[0-9.e+-]+_wl[0-9.e+-]+_ll[0-9.e+-]+)_s(\d+)$"
)
WALK_RE = re.compile(r"^fwalk_(baseline|all_features)_r(\d+)_s(\d+)$")
CONC = {"all_features": 6, "baseline": 12}  # the previous run's concurrency


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def kv(**k: Any) -> str:
    return " ".join(f"{a}={b}" for a, b in k.items())


def g(x: float) -> str:
    return f"{x:g}"


def fmt(ax: str, v: Any) -> str:
    return str(int(v)) if ax in ("H", "L") else g(float(v))


def cfg_name(c: dict) -> str:
    return f"L{c['L']}_H{c['H']}_lr{c['lr']}_we{c['wd_enc']}_wl{c['wd_lin']}_ll{c['lr_lin']}"


def parse_cfg(name: str) -> dict:
    m = CFG_RE.match(name)
    assert m is not None, name
    return dict(zip(AXES, m.groups()))


def ladder_str(ax: str) -> list[str]:
    return [fmt(ax, v) for v in LADDER[ax]]


def configs(grid: dict[str, list[str]]) -> list[dict]:
    return [dict(zip(AXES, t)) for t in itertools.product(*(grid[a] for a in AXES))]


def bb_mask(names: list[str]) -> np.ndarray:
    """The HAR-benchmark block (as experiments/pooled_mingru_rerun_13.bb_mask): har_ma_* and the columns that are not
    trailing averages (the calendar columns)."""
    return np.array(
        [n.startswith("har_ma_") or not re.search(r"_ma_\d+$", n) for n in names]
    )


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


# ============================================================================ export
def st_export(a: argparse.Namespace) -> None:
    data = Path(a.data)
    info: dict[str, Any] = {}
    d0 = None
    for s in SETS:
        dz = Path(a.design) / f"design_{s}"
        meta = json.loads((dz / "design.json").read_text())
        cfg = meta["config"]
        assert (
            cfg["window"] == W
            and cfg["periods_per_day"] == PPD
            and str(cfg["har_base"]) == "5"
        ), cfg
        assert cfg["ex_flags"].get("LATE_START_WARMUP") is True, cfg["ex_flags"]
        names = meta["names"]
        assert len(names) == N_COLS[s], (s, len(names))
        assert not any(n.startswith("adj_fomc_release") for n in names), (
            s
        )  # EXCLUDE_FOMC_RELEASE
        X = np.load(dz / "X.npy")
        y = np.load(dz / "y.npy")
        b = np.load(dz / "b_oos.npy")
        d = pd.read_csv(dz / "dates_oos.csv", parse_dates=["date"])["date"]
        assert (
            X.shape == (N_ROWS, N_COLS[s])
            and len(y) == N_ROWS
            and len(b) == len(d) == N_ROWS - W
        )
        assert np.isfinite(X).all(), s
        assert (y > 0).all() and np.isfinite(y).all() and (b > 0).all()
        assert str(d[FIRST_EVAL_ROW - W]) == FIRST_EVAL, d[FIRST_EVAL_ROW - W]
        if d0 is None:
            d0 = (d, y, b)
        else:
            assert (
                d.equals(d0[0])
                and np.array_equal(y, d0[1])
                and np.array_equal(b, d0[2])
            )
        o = data / f"pmf_{s}"
        o.mkdir(parents=True, exist_ok=True)
        Xc = np.ascontiguousarray(X, dtype=np.float64)
        Xc.astype("<f8").tofile(o / "X.f64")
        Xc.astype("<f4").tofile(o / "X.f32")
        del Xc
        y.astype("<f4").tofile(o / "y.f32")
        (o / "meta.txt").write_text(f"{N_ROWS} {N_COLS[s]}\n")
        (o / "columns.txt").write_text("\n".join(names) + "\n")
        bb = bb_mask(names)
        assert int(bb.sum()) == 12, int(bb.sum())
        np.where(bb, PF_BB, PF_EX).astype("<f8").tofile(o / "enc_pf.f64")
        info[s] = dict(
            design=str(dz),
            n=N_ROWS,
            p=N_COLS[s],
            har_block=int(bb.sum()),
            ex_flags=cfg["ex_flags"],
            sha256={
                k: sha256(dz / k)
                for k in ("X.npy", "y.npy", "b_oos.npy", "dates_oos.csv", "design.json")
            },
            exported_sha256={
                k: sha256(o / k)
                for k in ("X.f32", "X.f64", "y.f32", "enc_pf.f64", "columns.txt")
            },
        )
        say(f"export {s}: {N_ROWS} x {N_COLS[s]}, HAR block {int(bb.sum())}")
    assert d0 is not None
    d, y, b = d0
    rows = pd.DataFrame(
        {
            "row": np.arange(N_ROWS),
            "date": [""] * W + d.dt.strftime("%Y-%m-%d %H:%M:%S").tolist(),
            "B": np.r_[np.full(W, np.nan), b],
            "y": y,
        }
    )
    rows.to_csv(data / "pmf_rows.csv", index=False)
    if (
        a.prev_rows
    ):  # the previous (w1000, 2012-start) design's rows: the same bars, row for row
        pr = load_rows(Path(a.prev_rows))
        r = np.arange(W, N_ROWS)
        same = bool(
            (
                pr["date"].to_numpy()[r] == pd.to_datetime(rows["date"][r]).to_numpy()
            ).all()
        )
        info["prev_rows"] = dict(
            dates_equal_rows_ge_W=same,
            y_max_abs_diff=float(np.max(np.abs(pr["y"].to_numpy(float) - y))),
            y_equal=bool(np.array_equal(pr["y"].to_numpy(float), y.astype(float))),
        )
        assert same, "the previous design's dates differ"
        say(f"export: previous rows {info['prev_rows']}")
    (data / "pmf_export.json").write_text(
        json.dumps(dict(info=info, host=platform.node()), indent=1)
    )


def load_rows(path: Path) -> pd.DataFrame:
    r = pd.read_csv(path, keep_default_na=False, na_values={"B": [""]})
    r["date"] = pd.to_datetime(r["date"].replace("", None))
    return r


# ============================================================================ plan
def st_plan(a: argparse.Namespace) -> None:
    data, inp = Path(a.data), Path(a.inputs)
    inp.mkdir(parents=True, exist_ok=True)
    rows = load_rows(data / "pmf_rows.csv")
    fp = json.loads((Path(a.lin_run) / "frozen_penalties.json").read_text())
    fr = fp["info"]
    assert fr["window_rows"] == W and fr["first_fold"] == 3 and len(fr["folds"]) == 1, (
        fr["folds"]
    )
    f = fr["folds"][0]
    assert f["fold"] == 3 and f["fit"] == FOLD["fit"] and f["val"] == FOLD["val"], f
    f0, f1 = f["fit"]
    v0, v1 = f["val"]
    assert (
        f1 - f0 == W and v0 - f1 == EMB and v1 - v0 == VAL and v1 == FIRST_EVAL_ROW
    ), f
    assert (
        str(rows["date"][v0]) == f["val_first"]
        and str(rows["date"][v1 - 1]) == f["val_last"]
    ), f
    for s in SETS:  # the trainer's columns = the linear run's design columns
        names = (data / f"pmf_{s}" / "columns.txt").read_text().split()
        assert names == fp["designs"][s]["columns"], s
    fold = dict(
        k=3,
        w0=f0,
        w1=v1,
        fit=[f0, f1],
        val=[v0, v1],
        fit_last=str(rows["date"][f1 - 1]),
        val_first=f["val_first"],
        val_last=f["val_last"],
    )
    day = rows["date"].dt.normalize()
    r = np.arange(FIRST_EVAL_ROW, N_ROWS)
    sess = day[r].to_numpy()
    starts = np.r_[0, np.flatnonzero(sess[1:] != sess[:-1]) + 1] + FIRST_EVAL_ROW
    origins = [int(x) for x in starts]
    assert origins[0] == FIRST_EVAL_ROW and str(rows["date"][origins[0]]) == FIRST_EVAL
    nxt = origins[1:] + [N_ROWS]
    plan: dict[str, Any] = dict(
        W=W,
        window_days=WINDOW_DAYS,
        n_rows=N_ROWS,
        seeds=list(SEEDS),
        common=COMMON,
        grid0={k: [fmt(k, v) for v in vv] for k, vv in GRID0.items()},
        ladder={k: ladder_str(k) for k in LADDER},
        pf=dict(har_block=PF_BB, exogenous=PF_EX),
        tune=dict(val_tail=VAL, embargo=EMB, es="qlike", fold=fold),
        walk=dict(
            start_row=FIRST_EVAL_ROW,
            start=FIRST_EVAL,
            end_row=N_ROWS,
            end=str(rows["date"].iloc[-1]),
            refit="daily (every session)",
            n_sessions=len(origins),
            origins=origins,
            next=nxt,
            origin_dates=[str(rows["date"][o]) for o in origins],
        ),
        source_lin_fold=str(Path(a.lin_run) / "frozen_penalties.json"),
    )
    (inp / "plan.json").write_text(json.dumps(plan, indent=1))
    fc = np.diff(origins + [N_ROWS])
    say(
        f"plan: fold 3 fit [{f0}, {f1}) val [{v0}, {v1}) ({f['val_first']} .. {f['val_last']}); {len(origins)} daily "
        f"origins ({plan['walk']['origin_dates'][0]} .. {plan['walk']['origin_dates'][-1]}), bars a session "
        f"{dict(zip(*np.unique(fc, return_counts=True)))}"
    )


# ============================================================================ ridge inits
def window_keep64(X64: np.ndarray, w0: int, w1: int) -> np.ndarray:
    from src.models.window_mask import window_keep

    return window_keep(np.asarray(X64[w0:w1])).astype(np.int64)


def gram(X32: np.ndarray, y32: np.ndarray, keep: np.ndarray, r0: int, r1: int) -> tuple:
    """experiments/pooled_mingru_rerun_13.gram (= puffer_forecast/pooled_rinit.gram)."""
    Xk = np.empty((r1 - r0, len(keep) + 1))
    Xk[:, :-1] = np.asarray(X32[r0:r1])[:, keep]
    Xk[:, -1] = 1.0
    yy = np.asarray(y32[r0:r1], np.float64)
    ym = float(yy.mean())
    mu = float(np.float32(ym))
    sd = float(np.float32(np.sqrt(np.mean((yy - ym) ** 2))))
    z = (yy - mu) / sd
    n = r1 - r0
    return Xk.T @ Xk / n, Xk.T @ z / n, mu, sd, n


def solve(A: np.ndarray, b: np.ndarray, pf: np.ndarray, lam: float) -> np.ndarray:
    return np.linalg.solve(A + 0.5 * lam * np.diag(pf), b)


class TData:
    def __init__(self, data: Path, s: str):
        n, p = map(int, (data / f"pmf_{s}" / "meta.txt").read_text().split())
        self.X32 = np.memmap(data / f"pmf_{s}" / "X.f32", "<f4", "r", shape=(n, p))
        self.X64 = np.memmap(data / f"pmf_{s}" / "X.f64", "<f8", "r", shape=(n, p))
        self.y = np.fromfile(data / f"pmf_{s}" / "y.f32", "<f4")
        self.pf = np.fromfile(data / f"pmf_{s}" / "enc_pf.f64", "<f8")
        assert len(self.y) == n and len(self.pf) == p


def st_inits_tune(a: argparse.Namespace) -> None:
    data, inp = Path(a.data), Path(a.inputs)
    plan = json.loads((inp / "plan.json").read_text())
    f = plan["tune"]["fold"]
    idx = []
    for s in SETS:
        D = TData(data, s)
        od = data / "pmf_lin_init" / s
        od.mkdir(parents=True, exist_ok=True)
        keep = window_keep64(
            D.X64, f["w0"], f["w1"]
        )  # the trainer's keep of a tune job: its whole window
        pfk = np.append(D.pf[keep], 0.0)
        for L in LADDER["L"]:
            A, bb, mu, sd, n = gram(D.X32, D.y, keep, f["w0"] + L - 1, f["fit"][1])
            for wl in ladder_str("wd_lin"):
                lam = float(wl) / sd**2
                name = f"tune_f3_L{L}_wl{wl}.f32"
                v = solve(A, bb, pfk, lam)
                np.asarray(v, "<f4").tofile(od / name)
                idx.append(
                    dict(
                        design=s,
                        file=f"{s}/{name}",
                        P=len(v),
                        kind="tune",
                        L=L,
                        wl=wl,
                        mu=mu,
                        sd=sd,
                        n=n,
                        alpha_exog=n * float(wl) / (2 * sd**2),
                        n_keep=len(keep),
                    )
                )
        say(
            f"inits {s}: fold 3, keep {len(keep)} of {D.pf.size}, {len(LADDER['L'])} L x {len(LADDER['wd_lin'])} wd_lin"
        )
    pd.DataFrame(idx).to_csv(inp / "init_index_tune.csv", index=False)


# ============================================================================ tune jobs
def tune_line(s: str, f: dict, c: dict, seed: int) -> str:
    return kv(
        id=f"ftune_{s}_f3_{cfg_name(c)}_s{seed}",
        data=f"data/pmf_{s}",
        mode="tune",
        w0=f["w0"],
        w1=f["w1"],
        L=c["L"],
        H=c["H"],
        lr=c["lr"],
        seed=seed,
        fc_n=1,
        wd=0,
        val_tail=VAL,
        embargo=EMB,
        **COMMON,
        es="qlike",
        wd_enc=c["wd_enc"],
        wd_lin=c["wd_lin"],
        lr_lin=c["lr_lin"],
        lin_init=f"data/pmf_lin_init/{s}/tune_f3_L{c['L']}_wl{c['wd_lin']}.f32",
    )


def grid_path(inp: Path, s: str) -> Path:
    return inp / f"grid_{s}.json"


def load_grid(inp: Path, s: str) -> dict:
    p = grid_path(inp, s)
    if not p.is_file():
        st = dict(
            set=s,
            round=0,
            status="open",
            grid={k: [fmt(k, v) for v in vv] for k, vv in GRID0.items()},
            history=[],
        )
        p.write_text(json.dumps(st, indent=1))
    return json.loads(p.read_text())


def issued(inp: Path, s: str) -> set[str]:
    p = inp / f"issued_{s}.txt"
    return set(p.read_text().split()) if p.is_file() else set()


def heavy(c: dict) -> bool:
    """A (H, L) group split by lr into one file each (the large networks / long sequences)."""
    return int(c["H"]) >= 256 or int(c["L"]) >= 48


def st_tune_jobs(a: argparse.Namespace) -> None:
    inp = Path(a.inputs)
    plan = json.loads((inp / "plan.json").read_text())
    f = plan["tune"]["fold"]
    R = a.round
    lst = []
    for s in SETS:
        st = load_grid(inp, s)
        if st["status"] != "open":
            continue
        assert st["round"] == R, (s, st["round"], R)
        done = issued(inp, s)
        new = [c for c in configs(st["grid"]) if cfg_name(c) not in done]
        groups: dict[tuple, list[dict]] = {}
        for c in sorted(new, key=cfg_name):
            k = (c["H"], c["L"], c["lr"] if heavy(c) else "*")
            groups.setdefault(k, []).append(c)
        for i, (k, cs) in enumerate(
            sorted(groups.items(), key=lambda t: (int(t[0][0]), int(t[0][1]), t[0][2]))
        ):
            lines = [tune_line(s, f, c, seed) for c in cs for seed in SEEDS]
            name = f"tune_{s}_r{R}.{i}.txt"
            (inp / name).write_bytes(("\n".join(lines) + "\n").encode())
            lst.append(
                dict(
                    file=name,
                    set=s,
                    conc=CONC[s],
                    H=k[0],
                    L=k[1],
                    lr=k[2],
                    n_cfg=len(cs),
                    n_jobs=len(lines),
                )
            )
        with open(inp / f"issued_{s}.txt", "a") as fh:
            for c in new:
                fh.write(cfg_name(c) + "\n")
        say(
            f"tune jobs {s} round {R}: {len(new)} new configurations x {len(SEEDS)} seeds = {len(new) * len(SEEDS)} jobs"
        )
    pd.DataFrame(lst).to_csv(inp / f"tune_files_r{R}.csv", index=False)
    say(f"round {R}: {len(lst)} files")


# ============================================================================ select
def round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


def tune_table(data: Path, runs: Path, s: str) -> pd.DataFrame:
    """Per configuration of every tune run of set s: the 5-seed-mean validation QLIKE and each seed's best epoch."""
    y = np.fromfile(data / f"pmf_{s}" / "y.f32", "<f4").astype(np.float64)
    recs: list[dict] = []
    seen: set[str] = set()
    for d in sorted(runs.glob(f"tune_{s}_r*_*")):
        if not (d / "tune" / "jobs_out.csv").is_file():
            continue
        jo = pd.read_csv(d / "tune" / "jobs_out.csv")
        m = jo["id"].str.extract(TUNE_RE)
        assert m.notna().all().all(), jo["id"][m.isna().any(axis=1)].head()
        assert not (set(jo["id"]) & seen), d
        seen |= set(jo["id"])
        jo["cfg"], jo["seed"] = m[1], m[2].astype(int)
        pr = pd.read_csv(
            d / "tune" / "preds.csv",
            usecols=["id", "kind", "row", "f"],
            dtype={"id": "category"},
        )
        pr = pr[pr["kind"] == "val"]
        cats = (
            pd.Series(pr["id"].cat.categories.astype(str))
            .str.extract(TUNE_RE)[1]
            .to_numpy()
        )
        pr = pr.assign(cfg=cats[pr["id"].cat.codes.to_numpy()])
        agg = pr.groupby(["cfg", "row"])["f"].agg(["mean", "count", "min"])
        have = set(agg.index.get_level_values(0))
        for cfg, gg in jo.groupby("cfg"):
            ok_seeds = sorted(gg.loc[gg["status"] == "ok", "seed"])
            c = (
                agg.loc[cfg]
                if cfg in have
                else pd.DataFrame(columns=["mean", "count", "min"])
            )
            full = (
                ok_seeds == list(SEEDS)
                and len(c) == VAL
                and bool((c["count"] == len(SEEDS)).all())
                and int(c.index.min()) == FOLD["val"][0]
                and int(c.index.max()) == FOLD["val"][1] - 1
            )
            ql = float("nan")
            if full:
                fm = c["mean"].to_numpy(float)
                yv = y[c.index.to_numpy()]
                ok = yv > 0
                if np.isfinite(fm).all() and (c["min"].to_numpy(float) > 0).all():
                    q = (yv[ok] / fm[ok]) ** 2
                    ql = float(np.mean(q - np.log(q) - 1.0))
            ep = gg.set_index("seed")["best_epoch"]
            recs.append(
                dict(
                    predictor_set=s,
                    cfg=cfg,
                    val_qlike=ql,
                    n_val=int((y[FOLD["val"][0] : FOLD["val"][1]] > 0).sum()),
                    n_seeds_ok=len(ok_seeds),
                    run=d.name,
                    **{f"best_epoch_s{sd}": int(ep.get(sd, -1)) for sd in SEEDS},
                    **{
                        f"status_s{sd}": str(
                            gg.set_index("seed")["status"].get(sd, "missing")
                        )
                        for sd in SEEDS
                    },
                )
            )
    t = pd.DataFrame(recs)
    assert not t["cfg"].duplicated().any()
    return t.sort_values("cfg").reset_index(drop=True)


def st_select(a: argparse.Namespace) -> None:
    data, inp, runs, out = Path(a.data), Path(a.inputs), Path(a.runs), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    R = a.round
    frozen: dict[str, Any] = {}
    for s in SETS:
        st = load_grid(inp, s)
        t = tune_table(data, runs, s)
        want = issued(inp, s)
        assert set(t["cfg"]) == want, (
            s,
            len(set(t["cfg"])),
            len(want),
            sorted(want - set(t["cfg"]))[:5],
        )
        fin = t[np.isfinite(t["val_qlike"])]
        best = str(
            fin.loc[fin["val_qlike"].idxmin(), "cfg"]
        )  # first of the minimum in sorted name order
        t["picked"] = t["cfg"] == best
        t["rank"] = t["val_qlike"].rank(method="first")
        t.to_csv(out / f"tune_{s}.csv", index=False)
        pc = parse_cfg(best)
        if st["status"] == "open":
            assert st["round"] == R, (s, st["round"], R)
            ext, at_cap = {}, []
            for ax in AXES:
                vals = sorted(st["grid"][ax], key=float)
                lad = ladder_str(ax)
                v = pc[ax]
                i = lad.index(v)
                if v == vals[0] and NATURAL_LOW.get(ax) != v:
                    if i == 0:
                        at_cap.append(f"{ax} {v} (low)")
                    else:
                        ext[ax] = ("low", lad[i - 1])
                if v == vals[-1]:
                    if i == len(lad) - 1:
                        at_cap.append(f"{ax} {v} (high)")
                    else:
                        ext[ax] = ("high", lad[i + 1])
            rec = dict(
                round=R,
                n_configs=int(len(t)),
                n_finite=int(len(fin)),
                pick=best,
                val_qlike=float(t.loc[t["picked"], "val_qlike"].iloc[0]),
                extensions={
                    k: dict(direction=d_, added=v_) for k, (d_, v_) in ext.items()
                },
                at_cap=at_cap,
                tune_gpu_hours=a.tune_gpu_hours,
            )
            if (
                ext
                and a.tune_gpu_hours is not None
                and a.tune_gpu_hours >= COMPUTE_CAP_GPU_H
            ):
                rec["stopped"] = (
                    f"compute cap ({a.tune_gpu_hours:.1f} >= {COMPUTE_CAP_GPU_H} P100-hours)"
                )
                st["status"] = "frozen (compute cap)"
            elif ext:
                for ax, (d_, v_) in ext.items():
                    st["grid"][ax] = sorted(set(st["grid"][ax]) | {v_}, key=float)
                st["round"] = R + 1
            else:
                st["status"] = "frozen"
            st["history"].append(rec)
            grid_path(inp, s).write_text(json.dumps(st, indent=1))
            say(
                f"select {s} round {R}: {best} ({rec['val_qlike']:.6f}); extensions {rec['extensions']}; "
                f"at cap {at_cap}; status {st['status']}"
            )
        if st["status"] != "open":
            row = t[t["picked"]].iloc[0]
            epochs = {}
            for sd in SEEDS:
                e = int(row[f"best_epoch_s{sd}"])
                assert e >= 1, (s, best, sd, e)
                epochs[str(sd)] = dict(
                    by_fold=[e], median=float(e), epochs=round_half_up(float(e))
                )
            frozen[s] = dict(
                cfg=best,
                **{ax: (int(pc[ax]) if ax in ("H", "L") else pc[ax]) for ax in AXES},
                val_qlike=float(row["val_qlike"]),
                epochs=epochs,
                n_configs=int(len(t)),
                n_finite_configs=int(len(fin)),
                grid=st["grid"],
                rounds=st["history"],
                status=st["status"],
                previous_pick=PREV_PICK[s],
            )
    if len(frozen) == len(SETS):
        (out / "frozen_config.json").write_text(
            json.dumps(
                dict(
                    frozen=frozen, seeds=list(SEEDS), common=COMMON, rule="RULE_gru.txt"
                ),
                indent=1,
            )
        )
        say(
            f"frozen: { {s: (v['cfg'], {k: e['epochs'] for k, e in v['epochs'].items()}) for s, v in frozen.items()} }"
        )


# ============================================================================ walk
def st_walk_inits(a: argparse.Namespace) -> None:
    data, inp, out = Path(a.data), Path(a.inputs), Path(a.out)
    plan = json.loads((inp / "plan.json").read_text())
    fz = json.loads((out / "frozen_config.json").read_text())["frozen"]
    origins = plan["walk"]["origins"][a.part :: a.parts]
    idx = []
    for s in SETS:
        D = TData(data, s)
        L, wl = int(fz[s]["L"]), fz[s]["wd_lin"]
        od = data / "pmf_lin_init" / s
        od.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        for i, o in enumerate(origins):
            keep = window_keep64(D.X64, o - W, o)
            pfk = np.append(D.pf[keep], 0.0)
            A, bb, mu, sd, n = gram(D.X32, D.y, keep, o - W + L - 1, o)
            v = solve(A, bb, pfk, float(wl) / sd**2)
            name = f"r{o}_L{L}_wl{wl}.f32"
            np.asarray(v, "<f4").tofile(od / name)
            idx.append(
                dict(
                    design=s,
                    file=f"{s}/{name}",
                    P=len(v),
                    kind="refit",
                    origin=o,
                    L=L,
                    wl=wl,
                    mu=mu,
                    sd=sd,
                    n=n,
                    n_keep=len(keep),
                )
            )
            if i % 100 == 0:
                say(
                    f"walk inits {s} part {a.part}: {i + 1} of {len(origins)} ({time.time() - t0:.0f} s)"
                )
    pd.DataFrame(idx).to_csv(inp / f"walk_init_index.p{a.part}.csv", index=False)


def walk_line(s: str, c: dict, o: int, nx: int, sd: int, first: bool) -> str:
    base: dict[str, Any] = dict(
        id=f"fwalk_{s}_r{o}_s{sd}",
        data=f"data/pmf_{s}",
        mode="refit",
        w0=o - W,
        w1=o,
        L=c["L"],
        H=c["H"],
        lr=c["lr"],
        seed=sd,
        epochs=c["epochs"][str(sd)]["epochs"],
        fc_n=nx - o,
        wd=0,
        **COMMON,
        wd_enc=c["wd_enc"],
        wd_lin=c["wd_lin"],
        lr_lin=c["lr_lin"],
        lin_init=f"data/pmf_lin_init/{s}/r{o}_L{c['L']}_wl{c['wd_lin']}.f32",
    )
    if first:
        base["save_params"] = 1
    return kv(**base)


def st_walk_jobs(a: argparse.Namespace) -> None:
    inp, out = Path(a.inputs), Path(a.out)
    plan = json.loads((inp / "plan.json").read_text())
    fz = json.loads((out / "frozen_config.json").read_text())["frozen"]
    o_all, n_all = plan["walk"]["origins"], plan["walk"]["next"]
    parts = {"all_features": a.parts_full, "baseline": a.parts_har}
    lst = []
    for s in SETS:
        lines = [
            walk_line(s, fz[s], o, nx, sd, i == 0)
            for i, (o, nx) in enumerate(zip(o_all, n_all))
            for sd in SEEDS
        ]
        per = -(-len(lines) // parts[s])
        for p in range(parts[s]):
            name = f"walk_{s}.p{p}.txt"
            (inp / name).write_bytes(
                ("\n".join(lines[p * per : (p + 1) * per]) + "\n").encode()
            )
            lst.append(
                dict(
                    file=name,
                    set=s,
                    conc=CONC[s],
                    n_jobs=len(lines[p * per : (p + 1) * per]),
                )
            )
        say(f"walk jobs {s}: {len(lines)} refits in {parts[s]} files")
    pd.DataFrame(lst).to_csv(inp / "walk_files.csv", index=False)


# ============================================================================ canary
CANARY_CFG = dict(L="20", H="64", lr="0.003", wd_enc="3", wd_lin="0.3", lr_lin="1e-05")


def st_canary(a: argparse.Namespace) -> None:
    """A tune job a set (round-0 configuration, 1 seed) and refits at the first and the last walk origin (2 epochs)."""
    data, inp = Path(a.data), Path(a.inputs)
    plan = json.loads((inp / "plan.json").read_text())
    f = plan["tune"]["fold"]
    o_all, n_all = plan["walk"]["origins"], plan["walk"]["next"]
    lines, idx = [], []
    c = dict(CANARY_CFG)
    for s in SETS:
        D = TData(data, s)
        lines.append(
            tune_line(s, f, c, SEEDS[0]).replace(f"id=ftune_{s}", f"id=canary_tune_{s}")
        )
        for j in (0, len(o_all) - 1):
            o, nx = o_all[j], n_all[j]
            keep = window_keep64(D.X64, o - W, o)
            pfk = np.append(D.pf[keep], 0.0)
            A, bb, mu, sd, n = gram(D.X32, D.y, keep, o - W + int(c["L"]) - 1, o)
            v = solve(A, bb, pfk, float(c["wd_lin"]) / sd**2)
            name = f"canary_r{o}_L{c['L']}_wl{c['wd_lin']}.f32"
            (data / "pmf_lin_init" / s).mkdir(parents=True, exist_ok=True)
            np.asarray(v, "<f4").tofile(data / "pmf_lin_init" / s / name)
            idx.append(dict(design=s, origin=o, next=nx, n_keep=len(keep), file=name))
            cc = dict(c, epochs={str(SEEDS[0]): dict(epochs=2)})
            ln = walk_line(s, cc, o, nx, SEEDS[0], False).replace(
                f"id=fwalk_{s}", f"id=canary_walk_{s}"
            )
            ln = ln.replace(f"r{o}_L{c['L']}_wl{c['wd_lin']}.f32", name)
            lines.append(ln)
    (inp / "canary.txt").write_text("\n".join(lines) + "\n")
    pd.DataFrame(idx).to_csv(inp / "canary_index.csv", index=False)
    say(f"canary: {len(lines)} jobs")


def st_check_run(a: argparse.Namespace) -> None:
    """Every refit of a run: status ok, the trainer's kept columns = the init's, forecast rows exactly [o, next)."""
    inp, run = Path(a.inputs), Path(a.run)
    plan = json.loads((inp / "plan.json").read_text())
    nxt = dict(zip(plan["walk"]["origins"], plan["walk"]["next"]))
    sub = "refit" if (run / "refit").is_dir() else "tune"
    jo = pd.read_csv(run / sub / "jobs_out.csv")
    pr = pd.read_csv(run / sub / "preds.csv")
    keepn: dict[tuple[str, Any], int] = {}
    for p in sorted(inp.glob("walk_init_index.p*.csv")) + [
        inp / "canary_index.csv",
        inp / "init_index_tune.csv",
    ]:
        if p.is_file():
            q = pd.read_csv(p)
            for _, r in q.iterrows():
                if "origin" in q.columns and not pd.isna(r.get("origin", np.nan)):
                    keepn[(r["design"], int(r["origin"]))] = int(r["n_keep"])
                else:
                    keepn[(r["design"], "tune")] = int(r["n_keep"])
    bad: list[tuple] = []
    for _, j in jo.iterrows():
        s = "all_features" if "_all_features_" in j["id"] else "baseline"
        if j["status"] != "ok":
            bad.append((j["id"], "status", j["status"]))
            continue
        if j["mode"] == "refit":
            o = int(j["w1"])
            if keepn.get((s, o)) is not None and int(j["n_inputs"]) != keepn[(s, o)]:
                bad.append((j["id"], "n_inputs", int(j["n_inputs"]), keepn[(s, o)]))
            rows = pr.loc[
                (pr["id"] == j["id"]) & (pr["kind"] == "fc"), "row"
            ].to_numpy()
            if not np.array_equal(rows, np.arange(o, nxt[o])):
                bad.append(
                    (
                        j["id"],
                        "rows",
                        rows.min() if len(rows) else None,
                        rows.max() if len(rows) else None,
                    )
                )
        else:
            if int(j["n_inputs"]) != keepn.get((s, "tune"), -1):
                bad.append(
                    (j["id"], "n_inputs", int(j["n_inputs"]), keepn.get((s, "tune")))
                )
    f = pr["f"].to_numpy(float)
    res = dict(
        run=str(run),
        n_jobs=int(len(jo)),
        n_preds=int(len(pr)),
        finite=bool(np.isfinite(f).all()),
        positive=bool((f > 0).all()),
        bad=[list(map(str, b)) for b in bad[:20]],
        n_bad=len(bad),
    )
    (run / "check_run.json").write_text(json.dumps(res, indent=1))
    say(json.dumps(res))
    if bad or not res["finite"] or not res["positive"]:
        sys.exit(1)


# ============================================================================ merge
def rv_bars() -> pd.Series:
    """data/core_stats.parquet sumret2 by bar end (naive ET), as the linear runner's score stage."""
    core = pd.read_parquet(
        REPO / "data" / "core_stats.parquet", columns=["endbartime", "sumret2"]
    )
    rv = pd.Series(
        core["sumret2"].to_numpy(float),
        index=pd.DatetimeIndex(pd.to_datetime(core["endbartime"])),
    )
    return rv[~rv.index.duplicated()]


def read_runs(runs: Path, pat: str, sub: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    jo, pr = [], []
    for d in sorted(runs.glob(pat)):
        if not (d / sub / "jobs_out.csv").is_file():
            continue
        jo.append(pd.read_csv(d / sub / "jobs_out.csv").assign(run=d.name))
        pr.append(pd.read_csv(d / sub / "preds.csv").assign(run=d.name))
    assert jo, (runs, pat)
    return pd.concat(jo, ignore_index=True), pd.concat(pr, ignore_index=True)


def st_merge(a: argparse.Namespace) -> None:
    runs, out, inp = Path(a.runs), Path(a.out), Path(a.inputs)
    rows = load_rows(Path(a.data_rows))
    plan = json.loads((inp / "plan.json").read_text())
    cfgj = json.loads((out / "frozen_config.json").read_text())
    rv = rv_bars()
    fr, seeds_long, meta = [], [], {}
    r_all = np.arange(FIRST_EVAL_ROW, N_ROWS)
    nxt = dict(zip(plan["walk"]["origins"], plan["walk"]["next"]))
    for s in SETS:
        jo, pr = read_runs(runs, f"walk_{s}_p*_*", "refit")
        assert (jo["status"] == "ok").all()
        assert not jo["id"].duplicated().any()
        m = jo["id"].str.extract(WALK_RE)
        assert m.notna().all().all()
        want = {
            f"fwalk_{s}_r{o}_s{sd}" for o in plan["walk"]["origins"] for sd in SEEDS
        }
        assert set(jo["id"]) == want, (s, len(set(jo["id"])), len(want))
        pr = pr[pr["kind"] == "fc"].copy()
        assert not pr.duplicated(["id", "row"]).any()
        mm = pr["id"].str.extract(WALK_RE)
        pr["origin"], pr["seed"] = mm[1].astype(int), mm[2].astype(int)
        assert ((pr["row"] >= pr["origin"]) & (pr["row"] < pr["origin"].map(nxt))).all()
        piv = pr.pivot(index="row", columns="seed", values="f").sort_index()
        assert list(piv.columns) == list(SEEDS) and np.array_equal(
            piv.index.to_numpy(), r_all
        ), (s, piv.shape)
        F = piv.to_numpy(float)
        assert np.isfinite(F).all() and (F > 0).all()
        f = F.mean(axis=1)
        d = rows.set_index("row").loc[r_all]
        B = d["B"].to_numpy(float)
        assert np.isfinite(B).all()
        dates = pd.DatetimeIndex(d["date"])
        fr.append(
            pd.DataFrame(
                dict(
                    model="mingru",
                    predictor_set=s,
                    date=dates,
                    pred_adj=f,
                    true_adj=d["y"].to_numpy(float),
                    B=B,
                    plain=f**2 * B,
                    rv_actual=rv.reindex(dates).to_numpy(),
                    f_seed_sd=F.std(axis=1, ddof=1),
                )
            )
        )
        for j, sd in enumerate(SEEDS):
            seeds_long.append(
                pd.DataFrame(dict(predictor_set=s, seed=sd, date=dates, f=F[:, j]))
            )
        meta[s] = dict(
            cfg=cfgj["frozen"][s]["cfg"],
            refit_jobs=int(len(jo)),
            rows=int(len(r_all)),
            trainer_hours_t_train=float(jo["t_train"].sum() / 3600),
            runs=sorted(jo["run"].unique()),
        )
    fc = pd.concat(fr, ignore_index=True)
    fc["date"] = pd.to_datetime(fc["date"]).astype("datetime64[ns]")
    fc.to_parquet(out / "forecasts.parquet", index=False)
    pd.concat(seeds_long, ignore_index=True).to_parquet(
        out / "forecasts_seeds.parquet", index=False
    )
    (out / "forecasts.json").write_text(
        json.dumps(
            dict(
                window_days=WINDOW_DAYS,
                refit="daily (every session)",
                span=[FIRST_EVAL, str(fc["date"].max())],
                seeds=list(SEEDS),
                sets=meta,
            ),
            indent=1,
        )
    )
    say(f"merged {len(fc)} rows -> {out / 'forecasts.parquet'}")


# ============================================================================ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "stage",
        choices=(
            "export",
            "plan",
            "inits-tune",
            "tune-jobs",
            "select",
            "walk-inits",
            "walk-jobs",
            "canary",
            "check-run",
            "merge",
        ),
    )
    ap.add_argument("--design", default="")
    ap.add_argument("--prev-rows", default="")
    ap.add_argument("--lin-run", default="")
    ap.add_argument("--data", default="data")
    ap.add_argument("--inputs", default="inputs")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--run", default="")
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--tune-gpu-hours", type=float, default=None)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--parts-full", type=int, default=12)
    ap.add_argument("--parts-har", type=int, default=6)
    ap.add_argument("--data-rows", default="")
    a = ap.parse_args()
    {
        "export": st_export,
        "plan": st_plan,
        "inits-tune": st_inits_tune,
        "tune-jobs": st_tune_jobs,
        "select": st_select,
        "walk-inits": st_walk_inits,
        "walk-jobs": st_walk_jobs,
        "canary": st_canary,
        "check-run": st_check_run,
        "merge": st_merge,
    }[a.stage](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
