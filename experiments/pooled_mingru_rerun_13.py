"""The pooled ridge-initialised MinGRU, trained and selected on QLIKE, rerun on the 13-bar design (2026-10-22).

Request (James, 2026-10-22, relayed by the coordinator): rerun the pooled MinGRU network of the parked content of
writeup/rehaul/06_results.tex / 07_economics.tex (results/puffer_forecast_2026-10-12/pooled_rinit_study: ridge-
initialised linear shortcut, trained and early-stopped on QLIKE, selected on the validation QLIKE, refit monthly, seed
mean) on the new design: the 13 regular-hours bars, base-5 HAR horizons, the HAR benchmark (12 columns) and the full
set (576 columns, every FOMC series), a rolling window of 1,000 trading days (13,000 bars), 2012-01-03 .. 2024-04-30,
hyperparameters chosen once on the 4 pre-2012 folds of the pooled linear rerun and frozen.

DESIGN  The design caches of the pooled linear 13-bar rerun (experiments/pooled_linear_rerun_2026_10_19_13.py design
        --window-days 1000), exactly as the pooled trees use them (CARC build, /scratch1/jc_905/harxhar-trees13/results/
        pooled_linear_rerun_2026-10-19_13_w1000/_work/design_<set>/: X.npy prescaled, y.npy the adjusted target,
        b_oos.npy, dates_oos.csv; design row r >= W has date dates_oos[r - W]).  Nothing is rebuilt: `export` writes
        the trainer's row-major copies (data row = design row).

CARRIED OVER FROM THE ORIGINAL POOLED STUDY (experiments/puffer_forecast/pooled_rinit.py, pooled_walk.py,
pooled_select.py; trainer native_g9, experiments/puffer_forecast/native/forecast_train.cu md5 af1964d0, CARC binary
bin/native_g9/forecast_train md5 44ab6fff, P100):
  * network      one MinGRU layer, highway gates, softplus output link, zero-initialised decoder (dec_zero=1), the
                 linear shortcut (lin=1) started at its window's closed-form ridge (lin_init), Muon (momentum 0.95,
                 global clip 1.0), cosine lr over max_epochs = 100 to 0, batch 128, xstd=0 (the design is prescaled),
                 mask=1 (window_keep on the fit's window), wd = 0
  * loss         loss=qlike (eps 1e-3, gradient cap at the 0.99 quantile, the trainer's defaults), early stopping on the
                 validation QLIKE (es=qlike, patience 10)
  * grid         H {64, 256} x L {20, 48} x lr {0.001, 0.003} x wd_enc {0, 3, 15} x wd_lin {0.03, 0.3, 3} x
                 lr_lin {0, 1e-5} = 144 configurations, seeds 20261012, 20261013, 20261014 (the original's three)
  * penalty pf   0.01 on the HAR-benchmark block (har_ma_* and the calendar columns), 1 on every exogenous column, in
                 both sets (the original's enc_pf.f64; the new linear models' penalty factor on the full set)
  * ridge init   v = solve(X'X / n + lam / 2 diag(pf, 0), X'z / n), z = (y - mu_y) / sd_y, lam = wd_lin / sd_y^2, on the
                 training rows with a full look-back of L rows, kept columns + constant (pooled_rinit.gram / solve)
  * selection    the lowest QLIKE of the seed-averaged validation forecast (r = y^2 / f^2, rows y > 0;
                 pooled_select.py --criterion qlike), configurations in sorted name order, ties: the first
  * refit        monthly: every 21 sessions, on the trailing W bars, each seed for its own number of epochs, each refit
                 forecasting every bar to the next refit; the forecast is the seed mean of f, plain = f^2 B
CHANGED FOR THE NEW DESIGN (the linear models' protocol, fixed before any evaluation-sample number was seen):
  * tuning       once, on the 4 pre-2012 folds of the pooled linear rerun (frozen_penalties.json info.folds): fold k
                 fits on its 13,000 rows, a 325-row embargo, validates on its 3,250 rows (validation spans 2008-01-15 ..
                 2011-12-30).  Trainer: mode=tune, w0 = fit start, w1 = validation end, val_tail 3250, embargo 325.
                 The criterion is the mean over the 4 folds of each fold's validation QLIKE; one configuration a set,
                 frozen over the whole walk (the original re-tuned at each of its 5 blocks).
  * epochs       each seed's refit epochs = the median over the 4 folds of that seed's best epoch in the frozen
                 configuration, rounded half up (the original: the block's single tune)
  * window       13,000 bars (the original 24,000 bars of the 48-interval grid)
  * walk         from design row 44,250 (2011-12-28 14:00, the linear rerun's walk start) to the last row
                 (2024-04-30 16:00); refit origins: the walk start, then the first row of every 21st session after it

Stages (python experiments/pooled_mingru_rerun_13.py <stage> ...)
  export   --design DIR --data DATA         trainer data dirs DATA/pm13_<set>/ (X.f32, X.f64, y.f32, meta.txt,
                                            enc_pf.f64, columns.txt) and DATA/pm13_rows.csv
  plan     --data DATA --inputs IN          IN/plan.json (folds, walk origins), the selection rule
  inits    --data DATA --inputs IN          ridge inits DATA/pm13_lin_init/<set>/ (tune: every fold x L x wd_lin;
                                            walk: every origin x L x wd_lin), IN/init_index.csv
  tune-jobs --inputs IN                     IN/tune_<set>_f<k>.s<i>.txt
  select   --inputs IN --runs RUNS --out O  O/pretune_qlike.csv, O/frozen_config.json, IN/walk_<set>.p<i>.txt
  merge    --runs RUNS --data-rows CSV --out O   O/forecasts.parquet (+ forecasts_seeds.parquet, forecasts.json)
  score    --out O                          O/scores_*.csv, O/econ_*.csv (local, needs the repo's results/)
"""

from __future__ import annotations

import argparse
import hashlib
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

REPO = Path(os.environ.get("PM13_REPO", Path(__file__).resolve().parents[1]))
for _p in (REPO, REPO / "experiments", REPO / "notebooks", REPO / "writeup"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

OUT_DEFAULT = (
    Path(__file__).resolve().parents[1] / "results" / "pooled_mingru_rerun13_2026-10-22"
)
LIN_RUN = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "pooled_linear_rerun_2026-10-20_13_2012_w1000"
)
PPD, WINDOW_DAYS = 13, 1000
W = PPD * WINDOW_DAYS  # 13,000 bars
N_ROWS = 84_747
N_COLS = {"baseline": 12, "all_features": 576}
SETS = ("baseline", "all_features")
SEEDS = (20261012, 20261013, 20261014)
HS, LS, LRS = (64, 256), (20, 48), (0.001, 0.003)
WES, WLS, LLS = (0, 3, 15), (0.03, 0.3, 3), (0, 1e-05)
PF_BB, PF_EX = 0.01, 1.0
VAL, EMB, FOLD_FIT = 3250, 325, 13000
WALK_START_ROW, WALK_START = 44_250, "2011-12-28 14:00:00"
FIRST_EVAL_ROW, FIRST_EVAL = 44_281, "2012-01-03 10:00:00"
REFIT_SESS = 21
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
TUNE_RE = re.compile(
    r"^ptune_(baseline|all_features)_f(\d)_L(\d+)_H(\d+)_lr([0-9.]+)_we([0-9.e+-]+)_wl([0-9.e+-]+)_ll([0-9.e+-]+)"
    r"_s(\d+)$"
)
WALK_RE = re.compile(r"^pwalk_(baseline|all_features)_r(\d+)_s(\d+)$")
SELECTION_RULE = (
    "Per predictor set, one configuration of the 144 (H x L x lr x wd_enc x wd_lin x lr_lin), frozen over the walk: "
    "the lowest mean over the 4 pre-2012 folds of the fold's validation QLIKE (r = y^2 / f^2, r - log r - 1, over "
    "the fold's validation rows with y > 0) of the forecast averaged over the 3 seeds; a configuration with a "
    "non-finite fold value is skipped; configurations in sorted name order, ties: the first.  Refit epochs of seed s: "
    "the median over the 4 folds of seed s's best epoch in the frozen configuration, rounded half up.  Fixed "
    "2026-10-22 before any job ran; nothing from 2012-01-03 on enters."
)


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def kv(**k) -> str:
    return " ".join(f"{a}={b}" for a, b in k.items())


def g(x: float) -> str:
    return f"{x:g}"


def cfg_name(L, H, lr, we, wl, ll) -> str:
    return f"L{L}_H{H}_lr{lr}_we{we}_wl{wl}_ll{ll}"


def grid() -> list[tuple]:
    return [
        (L, H, g(lr), g(we), g(wl), g(ll))
        for H in HS
        for L in LS
        for lr in LRS
        for we in WES
        for wl in WLS
        for ll in LLS
    ]


def bb_mask(names: list[str]) -> np.ndarray:
    """The HAR-benchmark block (experiments/pooled_linear_rerun_2026_10_19_13.st_select): har_ma_* and the columns
    that are not trailing averages (DOW_0 .. DOW_4, hour)."""
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
def st_export(a) -> None:
    data = Path(a.data)
    info = {}
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
        names = meta["names"]
        assert len(names) == N_COLS[s], (s, len(names))
        X = np.load(dz / "X.npy", mmap_mode="r")
        y = np.load(dz / "y.npy")
        b = np.load(dz / "b_oos.npy")
        d = pd.read_csv(dz / "dates_oos.csv", parse_dates=["date"])["date"]
        assert (
            X.shape == (N_ROWS, N_COLS[s])
            and len(y) == N_ROWS
            and len(b) == len(d) == N_ROWS - W
        )
        assert (y > 0).all() and np.isfinite(y).all() and (b > 0).all()
        assert (
            str(d[WALK_START_ROW - W]) == WALK_START
            and str(d[FIRST_EVAL_ROW - W]) == FIRST_EVAL
        )
        if d0 is None:
            d0 = (d, y, b)
        else:
            assert (
                d.equals(d0[0])
                and np.array_equal(y, d0[1])
                and np.array_equal(b, d0[2])
            )
        o = data / f"pm13_{s}"
        o.mkdir(parents=True, exist_ok=True)
        with open(o / "X.f32", "wb") as f32, open(o / "X.f64", "wb") as f64:
            for r in range(0, N_ROWS, 5000):
                c = np.ascontiguousarray(np.asarray(X[r : r + 5000], np.float64))
                assert np.isfinite(c).all()
                c.astype("<f8").tofile(f64)
                c.astype("<f4").tofile(f32)
        y.astype("<f4").tofile(o / "y.f32")
        (o / "meta.txt").write_text(f"{N_ROWS} {N_COLS[s]}\n")
        (o / "columns.txt").write_text("\n".join(names) + "\n")
        bb = bb_mask(names)
        assert int(bb.sum()) == 12, int(bb.sum())
        pf = np.where(bb, PF_BB, PF_EX).astype("<f8")
        pf.tofile(o / "enc_pf.f64")
        info[s] = dict(
            design=str(dz),
            n=N_ROWS,
            p=N_COLS[s],
            har_block=int(bb.sum()),
            sha256={
                k: sha256(dz / k)
                for k in ("X.npy", "y.npy", "b_oos.npy", "dates_oos.csv")
            },
            exported_sha256={
                k: sha256(o / k) for k in ("X.f32", "X.f64", "y.f32", "enc_pf.f64")
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
    rows.to_csv(data / "pm13_rows.csv", index=False)
    (data / "pm13_export.json").write_text(
        json.dumps(dict(info=info, host=platform.node()), indent=1)
    )


def load_rows(path: Path) -> pd.DataFrame:
    r = pd.read_csv(path, keep_default_na=False, na_values={"B": [""]})
    r["date"] = pd.to_datetime(r["date"].replace("", None))
    return r


# ============================================================================ plan
def st_plan(a) -> None:
    data, inp = Path(a.data), Path(a.inputs)
    inp.mkdir(parents=True, exist_ok=True)
    rows = load_rows(data / "pm13_rows.csv")
    fr = json.loads((LIN_RUN / "frozen_penalties.json").read_text())["info"]
    assert fr["window_rows"] == W
    folds = []
    for k, f in enumerate(fr["folds"]):
        f0, f1 = f["fit"]
        v0, v1 = f["val"]
        assert f1 - f0 == FOLD_FIT and v0 - f1 == EMB and v1 - v0 == VAL, f
        assert (
            str(rows["date"][v0]) == f["val_first"]
            and str(rows["date"][v1 - 1]) == f["val_last"]
        ), f
        folds.append(
            dict(
                k=k,
                w0=f0,
                w1=v1,
                fit=[f0, f1],
                val=[v0, v1],
                fit_first=str(rows["date"][f0]),
                fit_last=str(rows["date"][f1 - 1]),
                val_first=f["val_first"],
                val_last=f["val_last"],
            )
        )
    assert folds[-1]["val"][1] == FIRST_EVAL_ROW
    day = rows["date"].dt.normalize()
    r = np.arange(WALK_START_ROW, N_ROWS)
    sess = day[r].to_numpy()
    starts = (
        np.r_[0, np.flatnonzero(sess[1:] != sess[:-1]) + 1] + WALK_START_ROW
    )  # first row of each session
    origins = [int(x) for x in starts[::REFIT_SESS]]
    assert origins[0] == WALK_START_ROW
    nxt = origins[1:] + [N_ROWS]
    plan: dict[str, Any] = dict(
        W=W,
        window_days=WINDOW_DAYS,
        n_rows=N_ROWS,
        seeds=list(SEEDS),
        common=COMMON,
        grid=dict(
            H=HS,
            L=LS,
            lr=LRS,
            wd_enc=WES,
            wd_lin=WLS,
            lr_lin=LLS,
            n_configs=len(grid()),
        ),
        pf=dict(har_block=PF_BB, exogenous=PF_EX),
        tune=dict(val_tail=VAL, embargo=EMB, es="qlike", folds=folds),
        walk=dict(
            start_row=WALK_START_ROW,
            start=WALK_START,
            end_row=N_ROWS,
            end=str(rows["date"].iloc[-1]),
            refit_every_sessions=REFIT_SESS,
            n_sessions=int(len(starts)),
            origins=origins,
            next=nxt,
            origin_dates=[str(rows["date"][o]) for o in origins],
        ),
        selection_rule=SELECTION_RULE,
        source_lin_folds=str(LIN_RUN.name) + "/frozen_penalties.json",
    )
    (inp / "plan.json").write_text(json.dumps(plan, indent=1))
    say(
        f"plan: {len(folds)} folds, {len(origins)} refit origins over {len(starts)} sessions "
        f"({plan['walk']['origin_dates'][0]} .. {plan['walk']['origin_dates'][-1]})"
    )


# ============================================================================ ridge inits
def window_keep64(X64: np.ndarray, w0: int, w1: int) -> np.ndarray:
    from src.models.window_mask import window_keep

    return window_keep(np.asarray(X64[w0:w1])).astype(np.int64)


def gram(X32, y32, keep, r0, r1):
    """experiments/puffer_forecast/pooled_rinit.gram: the float32 design's kept columns + constant, z = (y - mu) / sd
    with the trainer's float32 mu_y, sd_y of the training rows."""
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


def solve(A, b, pf, lam):
    return np.linalg.solve(A + 0.5 * lam * np.diag(pf), b)


class TData:
    def __init__(self, data: Path, s: str):
        n, p = map(int, (data / f"pm13_{s}" / "meta.txt").read_text().split())
        self.X32 = np.memmap(data / f"pm13_{s}" / "X.f32", "<f4", "r", shape=(n, p))
        self.X64 = np.memmap(data / f"pm13_{s}" / "X.f64", "<f8", "r", shape=(n, p))
        self.y = np.fromfile(data / f"pm13_{s}" / "y.f32", "<f4")
        self.pf = np.fromfile(data / f"pm13_{s}" / "enc_pf.f64", "<f8")
        assert len(self.y) == n and len(self.pf) == p


def st_inits(a) -> None:
    data, inp = Path(a.data), Path(a.inputs)
    plan = json.loads((inp / "plan.json").read_text())
    idx = []
    for s in SETS:
        D = TData(data, s)
        od = data / "pm13_lin_init" / s
        od.mkdir(parents=True, exist_ok=True)

        def put(name, v, **m):
            np.asarray(v, "<f4").tofile(od / name)
            idx.append(dict(design=s, file=f"{s}/{name}", P=len(v), **m))

        t0 = time.time()
        for f in plan["tune"]["folds"]:
            keep = window_keep64(
                D.X64, f["w0"], f["w1"]
            )  # the trainer's keep of a tune job: its whole window
            pfk = np.append(D.pf[keep], 0.0)
            for L in LS:
                A, bb, mu, sd, n = gram(D.X32, D.y, keep, f["w0"] + L - 1, f["fit"][1])
                for wl in WLS:
                    lam = wl / sd**2
                    put(
                        f"tune_f{f['k']}_L{L}_wl{g(wl)}.f32",
                        solve(A, bb, pfk, lam),
                        kind="tune",
                        k=f["k"],
                        L=L,
                        wl=g(wl),
                        mu=mu,
                        sd=sd,
                        n=n,
                        alpha_exog=n * wl / (2 * sd**2),
                        n_keep=len(keep),
                    )
        say(f"inits {s}: tune done in {time.time() - t0:.0f} s")
        for i, o in enumerate(plan["walk"]["origins"]):
            keep = window_keep64(D.X64, o - W, o)
            pfk = np.append(D.pf[keep], 0.0)
            for L in LS:
                A, bb, mu, sd, n = gram(D.X32, D.y, keep, o - W + L - 1, o)
                for wl in WLS:
                    lam = wl / sd**2
                    put(
                        f"r{o}_L{L}_wl{g(wl)}.f32",
                        solve(A, bb, pfk, lam),
                        kind="refit",
                        origin=o,
                        L=L,
                        wl=g(wl),
                        mu=mu,
                        sd=sd,
                        n=n,
                        alpha_exog=n * wl / (2 * sd**2),
                        n_keep=len(keep),
                    )
            if i % 25 == 0:
                say(
                    f"inits {s}: origin {i + 1} of {len(plan['walk']['origins'])} ({time.time() - t0:.0f} s)"
                )
    pd.DataFrame(idx).to_csv(inp / "init_index.csv", index=False)
    say(f"inits: {len(idx)} files")


# ============================================================================ tune jobs
def tune_line(s: str, f: dict, cfg: tuple, seed: int) -> str:
    L, H, lr, we, wl, ll = cfg
    return kv(
        id=f"ptune_{s}_f{f['k']}_{cfg_name(*cfg)}_s{seed}",
        data=f"data/pm13_{s}",
        mode="tune",
        w0=f["w0"],
        w1=f["w1"],
        L=L,
        H=H,
        lr=lr,
        seed=seed,
        fc_n=1,
        wd=0,
        val_tail=VAL,
        embargo=EMB,
        **COMMON,
        es="qlike",
        wd_enc=we,
        wd_lin=wl,
        lr_lin=ll,
        lin_init=f"data/pm13_lin_init/{s}/tune_f{f['k']}_L{L}_wl{wl}.f32",
    )


SHARDS: dict[
    str, list[tuple[tuple[int, int], ...]]
] = {  # (H, L) groups of a (set, fold), as the original's shards (pooled_rinit.py shard)
    "all_features": [((64, 20), (64, 48)), ((256, 20),), ((256, 48),)],
    "baseline": [((64, 20), (64, 48), (256, 20), (256, 48))],
}


def st_tune_jobs(a) -> None:
    inp = Path(a.inputs)
    plan = json.loads((inp / "plan.json").read_text())
    n = 0
    lst = []
    for s in SETS:
        for f in plan["tune"]["folds"]:
            for i, grp in enumerate(SHARDS[s]):
                lines = [
                    tune_line(s, f, c, seed)
                    for c in grid()
                    if (c[1], c[0]) in grp
                    for seed in SEEDS
                ]
                name = f"tune_{s}_f{f['k']}.s{i}.txt"
                (inp / name).write_bytes(("\n".join(lines) + "\n").encode())
                lst.append(name)
                n += len(lines)
    (inp / "tune_files.txt").write_text("\n".join(lst) + "\n")
    say(f"tune jobs: {n} lines in {len(lst)} files")


# ============================================================================ select
def read_runs(runs: Path, pat: str, sub: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    jo, pr = [], []
    for d in sorted(runs.glob(pat)):
        if not (d / sub / "jobs_out.csv").is_file():
            continue
        jo.append(pd.read_csv(d / sub / "jobs_out.csv").assign(run=d.name))
        pr.append(pd.read_csv(d / sub / "preds.csv").assign(run=d.name))
    assert jo, (runs, pat)
    return pd.concat(jo, ignore_index=True), pd.concat(pr, ignore_index=True)


def round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


def st_select(a) -> None:
    inp, runs, out = Path(a.inputs), Path(a.runs), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    plan = json.loads((inp / "plan.json").read_text())
    rows_y = None
    recs, frozen = [], {}
    for s in SETS:
        y = np.fromfile(Path(a.data) / f"pm13_{s}" / "y.f32", "<f4").astype(np.float64)
        if rows_y is None:
            rows_y = y
        jo, pr = read_runs(runs, f"tune_{s}_f*_s*_*", "tune")
        m = jo["id"].str.extract(TUNE_RE)
        assert m.notna().all().all(), jo["id"][m.isna().any(axis=1)].head()
        jo["fold"], jo["seed"] = m[1].astype(int), m[8].astype(int)
        jo["cfg"] = (
            jo["id"]
            .str.replace(r"^ptune_(baseline|all_features)_f\d_", "", regex=True)
            .str.replace(r"_s\d+$", "", regex=True)
        )
        dup = jo["id"].duplicated()
        assert not dup.any(), jo["id"][dup].head()
        assert (jo["status"] == "ok").all(), jo[jo["status"] != "ok"][
            ["id", "status"]
        ].head()
        want = {
            f"ptune_{s}_f{f['k']}_{cfg_name(*c)}_s{seed}"
            for f in plan["tune"]["folds"]
            for c in grid()
            for seed in SEEDS
        }
        assert set(jo["id"]) == want, (s, len(set(jo["id"])), len(want))
        pr = pr[pr["kind"] == "val"]
        pr = pr.merge(jo[["id", "fold", "cfg", "seed"]], on="id", how="left")
        assert pr["cfg"].notna().all()
        ep = jo.set_index(["cfg", "fold", "seed"])["best_epoch"]
        for (cfg, k), gg in pr.groupby(["cfg", "fold"]):
            fold = plan["tune"]["folds"][k]
            c = gg.groupby("row")["f"].agg(["mean", "count"])
            assert (c["count"] == len(SEEDS)).all(), (s, cfg, k)
            assert (
                c.index.min() == fold["val"][0]
                and c.index.max() == fold["val"][1] - 1
                and len(c) == VAL
            )
            f = c["mean"].to_numpy(float)
            yv = y[c.index.to_numpy()]
            ok = yv > 0
            if np.all(f[ok] > 0) and np.isfinite(f[ok]).all():
                q = (yv[ok] / f[ok]) ** 2
                ql = float(np.mean(q - np.log(q) - 1.0))
            else:
                ql = float("nan")
            recs.append(
                dict(
                    predictor_set=s,
                    cfg=cfg,
                    fold=k,
                    val_qlike=ql,
                    n_val=int(ok.sum()),
                    **{f"best_epoch_s{sd}": int(ep[(cfg, k, sd)]) for sd in SEEDS},
                )
            )
    t = pd.DataFrame(recs)
    t.to_csv(out / "pretune_qlike_folds.csv", index=False)
    wide = []
    for s in SETS:
        u = t[t["predictor_set"] == s].pivot(
            index="cfg", columns="fold", values="val_qlike"
        )
        u.columns = [f"qlike_fold{k}" for k in u.columns]
        u["qlike_mean"] = u.mean(axis=1, skipna=False)
        u = u.sort_index()
        fin = u[np.isfinite(u["qlike_mean"])]
        best = fin["qlike_mean"].idxmin()  # first of the minimum in sorted order
        u["picked"] = u.index == best
        u["rank"] = u["qlike_mean"].rank(method="first")
        u.insert(0, "predictor_set", s)
        wide.append(u.reset_index())
        mt = re.match(
            r"^L(\d+)_H(\d+)_lr([0-9.]+)_we([0-9.e+-]+)_wl([0-9.e+-]+)_ll([0-9.e+-]+)$",
            best,
        )
        assert mt is not None, best
        L, H, lr, we, wl, ll = mt.groups()
        tb = t[(t["predictor_set"] == s) & (t["cfg"] == best)].sort_values("fold")
        epochs = {}
        for sd in SEEDS:
            e = tb[f"best_epoch_s{sd}"].to_numpy(float)
            assert (e >= 1).all(), (s, best, sd, e)
            epochs[str(sd)] = dict(
                by_fold=[int(v) for v in e],
                median=float(np.median(e)),
                epochs=round_half_up(float(np.median(e))),
            )
        frozen[s] = dict(
            cfg=best,
            L=int(L),
            H=int(H),
            lr=lr,
            wd_enc=we,
            wd_lin=wl,
            lr_lin=ll,
            qlike_mean=float(u.loc[best, "qlike_mean"]),
            qlike_folds=[float(v) for v in tb["val_qlike"]],
            epochs=epochs,
            n_finite_configs=int(len(fin)),
            n_configs=int(len(u)),
        )
        say(
            f"select {s}: {best}, mean val QLIKE {frozen[s]['qlike_mean']:.6f}, epochs "
            f"{ {k: v['epochs'] for k, v in epochs.items()} }"
        )
    pd.concat(wide, ignore_index=True).to_csv(out / "pretune_qlike.csv", index=False)
    (out / "frozen_config.json").write_text(
        json.dumps(
            dict(frozen=frozen, rule=SELECTION_RULE, seeds=list(SEEDS), common=COMMON),
            indent=1,
        )
    )
    walk_jobs(inp, plan, frozen, a.parts)


def walk_jobs(inp: Path, plan: dict, frozen: dict, parts: int) -> None:
    o_all, n_all = plan["walk"]["origins"], plan["walk"]["next"]
    lst = []
    for s in SETS:
        c = frozen[s]
        lines = []
        for i, (o, nx) in enumerate(zip(o_all, n_all)):
            for sd in SEEDS:
                base = dict(
                    id=f"pwalk_{s}_r{o}_s{sd}",
                    data=f"data/pm13_{s}",
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
                    lin_init=f"data/pm13_lin_init/{s}/r{o}_L{c['L']}_wl{c['wd_lin']}.f32",
                )
                if i == 0:
                    base["save_params"] = 1
                lines.append(kv(**base))
        per = -(-len(lines) // parts)
        for p in range(parts):
            name = f"walk_{s}.p{p}.txt"
            (inp / name).write_bytes(
                ("\n".join(lines[p * per : (p + 1) * per]) + "\n").encode()
            )
            lst.append(name)
        say(f"walk jobs {s}: {len(lines)} refits in {parts} files")
    (inp / "walk_files.txt").write_text("\n".join(lst) + "\n")


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


def st_merge(a) -> None:
    runs, out = Path(a.runs), Path(a.out)
    rows = load_rows(Path(a.data_rows))
    cfgj = json.loads((out / "frozen_config.json").read_text())
    rv = rv_bars()
    fr, seeds_long, meta = [], [], {}
    r_all = np.arange(WALK_START_ROW, N_ROWS)
    for s in SETS:
        jo, pr = read_runs(runs, f"walk_{s}_p*_*", "refit")
        assert (jo["status"] == "ok").all()
        assert not jo["id"].duplicated().any()
        m = jo["id"].str.extract(WALK_RE)
        assert m.notna().all().all()
        pr = pr[pr["kind"] == "fc"].copy()
        assert not pr.duplicated(["id", "row"]).any()
        pr["seed"] = pr["id"].str.extract(r"_s(\d+)$")[0].astype(int)
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
            gpu_hours_t_train=float(jo["t_train"].sum() / 3600),
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
                refit="monthly (21 sessions)",
                span=[WALK_START, str(fc["date"].max())],
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
        "stage", choices=("export", "plan", "inits", "tune-jobs", "select", "merge")
    )
    ap.add_argument("--design", default="")
    ap.add_argument("--data", default="data")
    ap.add_argument("--inputs", default="inputs")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--parts", type=int, default=6, help="select: walk job files a set")
    ap.add_argument("--data-rows", default="")
    a = ap.parse_args()
    {
        "export": st_export,
        "plan": st_plan,
        "inits": st_inits,
        "tune-jobs": st_tune_jobs,
        "select": st_select,
        "merge": st_merge,
    }[a.stage](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
