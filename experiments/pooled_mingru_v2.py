"""Pooled ridge-initialised MinGRU v2, "memorize, then regularize" (2026-10-23/24).

Rule: results/pooled_mingru_v2_2014start_w3000/RULE_gru_v2.txt (and the W = 3,500 copy), written before any v2 job.
Window: PMF_WINDOW_DAYS = 3000 (default) or 3500.  Trainer, data, fold-3 ridge inits and environment: the old final
rerun's (experiments/pooled_mingru_final_2014_w3000.py; CARC /home1/jc_905/pmf, /home1/jc_905/pmf35), read-only.

CARC stages (python experiments/pooled_mingru_v2.py <stage> --root ROOT --xdata XDATA ...)
  prep        G0: folds (vs folds14.json), Stage-1 inits (the old fold-3 tune inits, recomputed byte for byte and copied),
              TQ_0 a set and L, Stage-2 per-fold ridge inits a set / fold / L, kept columns, late-start columns
              -> ROOT/inputs/plan_v2.json, ROOT/inputs/init_index_v2.csv, ROOT/inputs/g0.json
  post RUN    the POST step of a GPU job: check (status, kept columns = the init's, forecast rows), the epoch curves
              of steps.csv(.gz), compact forecasts -> RUN/summary/
  bar-inits   the ridge inits of the walk probe's bar origins at --L (G7)
Local stages (on pulled summaries)
  jobs        write job files (--kind g1 | g2 | s1 | s2 | g5 | g7) into an inputs directory + a files list
  s1-table    Stage-1 per-job / per-cell tables, rho, pick (G3 / G4)
  s2-table    Stage-2 criterion C, shortlist, pick, extensions (G6); s2-report: bootstrap / jackknife / MCS
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(os.environ.get("PMF_REPO", Path(__file__).resolve().parents[1]))
for _p in (REPO, REPO / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

PPD = 13
WINDOW_DAYS = int(os.environ.get("PMF_WINDOW_DAYS", "3000"))
assert WINDOW_DAYS in (3000, 3500), WINDOW_DAYS
W = PPD * WINDOW_DAYS  # 39,000 or 45,500 rows
S0 = 50_777  # 2014-01-02 10:00, the first evaluation row
N_ROWS = 84_747
FOLD_N, N_FOLDS, EMB, FC_N = 3250, 4, 325, 3575
FOLDS = (1, 2, 3)
SETS = ("baseline", "all_features")
SEEDS_S1 = (20261012, 20261013, 20261014)
SEEDS_SCREEN = (20261012, 20261013, 20261014, 20261015, 20261016)
SEEDS_FRESH = (20261017, 20261018, 20261019, 20261020, 20261021)
WL = "0.003"  # the frozen shortcut's wd_lin
L_LADDER = (8, 20, 48, 115)
CORE_H, CORE_L = (16, 64, 256), (8, 20, 48)
TUNE: dict[str, Any] = dict(w0=47_202 - W, w1=S0, fit1=47_202, val=[47_527, S0])
EPS = 1e-3  # the trainer's qlike_eps
B = 128
COMMON = dict(
    layers=1,
    batch=B,
    momentum=0.95,
    max_grad_norm=1.0,
    anneal=1,
    min_lr_ratio=0,
    mask=1,
    xstd=0,
    link="softplus",
    loss="qlike",
    lin=1,
    dec_zero=1,
)
# the old run's measured P100-s a tune model-epoch at W = 3,000 (analysis_w3000/summary.json epoch_cost)
EPOCH_COST: dict[str, dict[tuple[int, int], float]] = {
    "baseline": {
        (16, 8): 0.01936,
        (16, 20): 0.02377,
        (64, 8): 0.02644,
        (64, 20): 0.04013,
        (256, 8): 0.14632,
        (16, 48): 0.20579,
        (256, 20): 0.25028,
        (64, 48): 0.26833,
        (16, 115): 0.29278,
        (64, 115): 0.40352,
        (256, 48): 0.60789,
        (256, 115): 1.22725,
        (1024, 20): 5.53006,
    },
    "all_features": {
        (16, 8): 0.03731,
        (64, 8): 0.04750,
        (16, 20): 0.04953,
        (64, 20): 0.07074,
        (256, 8): 0.20014,
        (16, 48): 0.24237,
        (256, 20): 0.32989,
        (64, 48): 0.34620,
        (256, 48): 0.60737,
    },
}
# Stage 2 ladders (RULE_gru_v2.txt STAGE 2)
TAU0 = ("0", "0.3", "1", "3")
TAU_EXT = {"low": ("0.1", "0.03"), "high": ("9", "27")}
WE0 = {"baseline": ("0",), "all_features": ("0", "3", "15")}
WE_EXT = {"high": ("75", "375")}
E_LADDER = (5, 10, 20, 40, 80, 160, 300, 600, 1000, 2000)
E0_300 = (10, 20, 40, 80, 160, 300)
E0_1000 = (10, 20, 40, 80, 160, 300, 600, 1000)


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def kv(**k: Any) -> str:
    return " ".join(f"{a}={b}" for a, b in k.items())


def g(x: float) -> str:
    return f"{x:g}"


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def fold_rows(f: int) -> tuple[list[int], list[int]]:
    """fold_fit14.fold_rows at this window; a fit block that would start before row 0 starts at 0 (W = 3,500, folds
    1-2: truncated, disclosed)."""
    halo = N_FOLDS * FOLD_N + EMB + W
    a = S0 - halo + f * FOLD_N
    return [max(0, a), a + W], [a + W + EMB, a + W + EMB + FOLD_N]


def steps_per_epoch(n_fit: int, L: int) -> int:
    return (n_fit - L + 1 + B - 1) // B


def conc(s: str, H: int, L: int) -> int:
    """The measured safe concurrency (RULE: Stage 1)."""
    if H >= 1024:
        return 7
    if H <= 64 and L <= 20:
        return 24
    return 12 if s == "baseline" else 6


def epoch_cost(s: str, H: int, L: int) -> float:
    """P100-s a model-epoch at this window (the old run's table x W / 39,000; cells outside it: H 1,024 from the
    measured HAR (1024, 20) scaled, L 115 at 2x L 48; full-set cells missing: 1.3 x HAR)."""
    t = EPOCH_COST[s]
    if (H, L) in t:
        c = t[(H, L)]
    elif s == "all_features" and (H, L) in EPOCH_COST["baseline"]:
        c = 1.3 * EPOCH_COST["baseline"][(H, L)]
    elif H == 1024:
        c = 5.53 * (0.7 if L <= 8 else 1.0) * (2.0 if L >= 48 else 1.0)
    else:
        c = 2.0 * t.get((H, 48), 0.6)
    return c * W / 39_000


# ============================================================================ data
class TData:
    def __init__(self, data: Path, s: str):
        n, p = map(int, (data / f"pmf_{s}" / "meta.txt").read_text().split())
        self.X32 = np.memmap(data / f"pmf_{s}" / "X.f32", "<f4", "r", shape=(n, p))
        self.X64 = np.memmap(data / f"pmf_{s}" / "X.f64", "<f8", "r", shape=(n, p))
        self.y = np.fromfile(data / f"pmf_{s}" / "y.f32", "<f4")
        self.pf = np.fromfile(data / f"pmf_{s}" / "enc_pf.f64", "<f8")
        self.names = (data / f"pmf_{s}" / "columns.txt").read_text().split()
        assert len(self.y) == n == N_ROWS and len(self.pf) == p == len(self.names)


def window_keep64(X64: np.ndarray, w0: int, w1: int) -> np.ndarray:
    from src.models.window_mask import window_keep

    return window_keep(np.asarray(X64[w0:w1])).astype(np.int64)


def gram(X32: np.ndarray, y32: np.ndarray, keep: np.ndarray, r0: int, r1: int) -> tuple:
    """= experiments/pooled_mingru_final_2014_w3000.gram (the old run's ridge init)."""
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


def ridge_init(
    D: TData, w0: int, w1: int, fit1: int, L: int
) -> tuple[np.ndarray, np.ndarray, float, float, int]:
    """The window mask on [w0, w1) (a tune job: its whole window; a refit: the fit block), the Gram on
    [w0 + L - 1, fit1), lambda = wl / sd_y^2 (pf: 0.01 HAR / calendar block, 1 exogenous; intercept 0)."""
    keep = window_keep64(D.X64, w0, w1)
    pfk = np.append(D.pf[keep], 0.0)
    A, bb, mu, sd, n = gram(D.X32, D.y, keep, w0 + L - 1, fit1)
    v = solve(A, bb, pfk, float(WL) / sd**2)
    return np.asarray(v, "<f4"), keep, mu, sd, n


def link_consts(mu: float, sd: float) -> tuple[float, float]:
    """The trainer's softplus link constants (trainer_host: c = softplus^-1(mu_y), s = sd_y (1 + exp(-c)), as float)."""
    c = math.log(math.expm1(float(np.float32(mu))))
    return float(np.float32(c)), float(
        np.float32(float(np.float32(sd)) * (1.0 + math.exp(-c)))
    )


def shortcut_f(
    D: TData, keep: np.ndarray, v: np.ndarray, mu: float, sd: float, r0: int, r1: int
) -> np.ndarray:
    """f of the frozen shortcut alone (decoder zero: o = 0) on rows [r0, r1)."""
    Z = np.c_[np.asarray(D.X32[r0:r1])[:, keep], np.ones(r1 - r0)].astype(np.float64)
    c, s = link_consts(mu, sd)
    a = c + s * (Z @ v.astype(np.float64))
    return np.where(a > 20, a, np.log1p(np.exp(np.minimum(a, 20))))


def qlike_trainer(y: np.ndarray, f: np.ndarray) -> np.ndarray:
    """The trainer's per-row QLIKE (fc_loss): r = max(y^2, eps^2) / (f^2 + eps^2), r - log r - 1."""
    y = np.asarray(y, np.float64)
    f = np.asarray(f, np.float64)
    e2 = EPS * EPS
    logr = np.clip(np.log(np.maximum(y * y, e2)) - np.log(f * f + e2), -80, 80)
    return np.exp(logr) - logr - 1.0


def qlike_plain(y: np.ndarray, f: np.ndarray) -> float:
    """The selection criterion's QLIKE (RULE STAGE 2 C): rows y > 0, q = y^2 / f^2, mean of q - log q - 1."""
    ok = y > 0
    q = (y[ok] / f[ok]) ** 2
    return float(np.mean(q - np.log(q) - 1.0))


# ============================================================================ prep (G0)
def late_start_cols(D: TData, keep: np.ndarray, w0: int, w1: int) -> dict:
    """Kept value columns adj_<c>_ma_<k> that are exactly 0 on a leading part of the window and not afterwards
    (LATE_START_WARMUP's zeroed rows): count, share of the kept columns, mean / max zeroed share of the window."""
    shares = []
    for j in keep:
        if not re.match(r"adj_.+_ma_\d+$", D.names[j]):
            continue
        col = np.asarray(D.X64[w0:w1, j])
        nz = np.flatnonzero(col != 0.0)
        if len(nz) == 0 or nz[0] == 0:
            continue
        shares.append(nz[0] / (w1 - w0))
    return dict(
        n_partly_zeroed=len(shares),
        share_of_kept=round(len(shares) / len(keep), 4),
        mean_zeroed_share=round(float(np.mean(shares)), 4) if shares else 0.0,
        max_zeroed_share=round(float(np.max(shares)), 4) if shares else 0.0,
    )


def st_prep(a: argparse.Namespace) -> None:
    root, xdata = Path(a.root), Path(a.xdata)
    inp = root / "inputs"
    inp.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(
        xdata / "pmf_rows.csv", keep_default_na=False, na_values={"B": [""]}
    )
    dates = rows["date"].astype(str).to_numpy()
    if a.alt_rows:  # W = 3,500: rows [39,000, 45,500) are dated in the W = 3,000 export (the same design rows)
        alt = (
            pd.read_csv(a.alt_rows, keep_default_na=False, na_values={"B": [""]})[
                "date"
            ]
            .astype(str)
            .to_numpy()
        )
        both = (dates != "") & (alt != "")
        assert (
            len(alt) == len(dates)
            and (dates[both] == alt[both]).all()
            and both.sum() > 0
        )
        dates = np.where(dates == "", alt, dates)
    f14 = {f["fold"]: f for f in json.loads(Path(a.folds_json).read_text())["folds"]}
    folds: dict[int, dict] = {}
    for k in FOLDS:
        fit, val = fold_rows(k)
        assert val == f14[k]["val"], (k, val, f14[k])
        assert (
            dates[val[0]] == f14[k]["val_first"]
            and dates[val[1] - 1] == f14[k]["val_last"]
        ), k
        assert val[0] - fit[1] == EMB and val[1] - val[0] == FOLD_N
        folds[k] = dict(
            fit=fit,
            val=val,
            n_fit=fit[1] - fit[0],
            truncated=bool(fit[1] - fit[0] < W),
            share_of_W=round((fit[1] - fit[0]) / W, 4),
            fit_first=dates[fit[0]] or f"(row {fit[0]} < {W}: undated in the design)",
            fit_last=dates[fit[1] - 1],
            val_first=dates[val[0]],
            val_last=dates[val[1] - 1],
            fc=[fit[1], fit[1] + FC_N],
        )
    assert folds[3]["fit"][1] == TUNE["fit1"] and folds[3]["val"] == TUNE["val"]
    g0: dict[str, Any] = dict(
        window_days=WINDOW_DAYS, W=W, folds=folds, tune=TUNE, sets={}
    )
    idx: list[dict[str, Any]] = []
    for s in SETS:
        D = TData(xdata, s)
        od = root / "data" / "pmf_lin_init" / s
        od.mkdir(parents=True, exist_ok=True)
        gs: dict[str, Any] = dict(stage1={}, stage2={})
        # Stage 1: the old fold-3 tune inits (window mask on the tune window [w0, w1), Gram on [w0 + L - 1, fit1))
        keep_t = window_keep64(D.X64, TUNE["w0"], TUNE["w1"])
        for L in L_LADDER:
            name = f"tune_f3_L{L}_wl{WL}.f32"
            old = np.fromfile(xdata / "pmf_lin_init" / s / name, "<f4")
            v, keep, mu, sd, n = ridge_init(D, TUNE["w0"], TUNE["w1"], TUNE["fit1"], L)
            assert np.array_equal(keep, keep_t)
            same = old.tobytes() == v.tobytes()
            assert len(old) == len(keep) + 1 == len(v), (s, L, len(old), len(keep))
            v.tofile(od / name)
            r0, r1 = TUNE["w0"] + L - 1, TUNE["fit1"]
            fsc = shortcut_f(D, keep, v, mu, sd, r0, r1)
            tq0 = float(np.mean(qlike_trainer(D.y[r0:r1], fsc)))
            v0, v1 = TUNE["val"]
            vq0 = float(
                np.mean(
                    qlike_trainer(D.y[v0:v1], shortcut_f(D, keep, v, mu, sd, v0, v1))
                )
            )
            gs["stage1"][str(L)] = dict(
                file=name,
                P=len(v),
                n_keep=len(keep),
                recomputed_identical=bool(same),
                md5=md5(od / name),
                mu=mu,
                sd=sd,
                n=n,
                S=steps_per_epoch(TUNE["fit1"] - TUNE["w0"], L),
                TQ0=tq0,
                VQ0_trainer=vq0,
                VQ0_plain=qlike_plain(
                    np.asarray(D.y[v0:v1], np.float64),
                    shortcut_f(D, keep, v, mu, sd, v0, v1),
                ),
            )
            idx.append(
                dict(
                    design=s,
                    file=f"{s}/{name}",
                    P=len(v),
                    n_keep=len(keep),
                    mu=mu,
                    sd=sd,
                    n=n,
                    kind="s1",
                    L=L,
                    fold=3,
                )
            )
            say(
                f"{s} stage-1 init L{L}: keep {len(keep)}, recomputed identical {same}, TQ_0 {tq0:.6f}"
            )
        gs["stage1_late_start"] = (
            late_start_cols(D, keep_t, TUNE["w0"], TUNE["fit1"])
            if s == "all_features"
            else None
        )
        # Stage 2: per-fold refit inits (window mask on the fit block [fit0, fit1), Gram on [fit0 + L - 1, fit1))
        for k in FOLDS:
            f0, f1 = folds[k]["fit"]
            keep_k = window_keep64(D.X64, f0, f1)
            fk: dict[str, Any] = dict(
                n_keep=len(keep_k),
                late_start=late_start_cols(D, keep_k, f0, f1)
                if s == "all_features"
                else None,
                L={},
            )
            for L in L_LADDER:
                name = f"s2_f{k}_L{L}_wl{WL}.f32"
                v, keep, mu, sd, n = ridge_init(D, f0, f1, f1, L)
                assert np.array_equal(keep, keep_k) and len(v) == len(keep) + 1
                v.tofile(od / name)
                fsc = shortcut_f(D, keep, v, mu, sd, f0 + L - 1, f1)
                v0, v1 = folds[k]["val"]
                fk["L"][str(L)] = dict(
                    file=name,
                    P=len(v),
                    md5=md5(od / name),
                    mu=mu,
                    sd=sd,
                    n=n,
                    S=steps_per_epoch(f1 - f0, L),
                    TQ0=float(np.mean(qlike_trainer(D.y[f0 + L - 1 : f1], fsc))),
                    VQ0_plain=qlike_plain(
                        np.asarray(D.y[v0:v1], np.float64),
                        shortcut_f(D, keep, v, mu, sd, v0, v1),
                    ),
                )
                idx.append(
                    dict(
                        design=s,
                        file=f"{s}/{name}",
                        P=len(v),
                        n_keep=len(keep),
                        mu=mu,
                        sd=sd,
                        n=n,
                        kind="s2",
                        L=L,
                        fold=k,
                    )
                )
            gs["stage2"][str(k)] = fk
            say(
                f"{s} fold {k}: fit {folds[k]['fit']} keep {len(keep_k)} late-start {fk['late_start']}"
            )
        g0["sets"][s] = gs
    pd.DataFrame(idx).to_csv(inp / "init_index_v2.csv", index=False)
    for s in (
        SETS
    ):  # the targets, for the POST step's in-sample QLIKE and the local Stage-2 tables
        (inp / f"y_{s}.f32").write_bytes((xdata / f"pmf_{s}" / "y.f32").read_bytes())
    ok = all(r["P"] == r["n_keep"] + 1 for r in idx)
    g0["pass"] = bool(ok)
    (inp / "g0.json").write_text(json.dumps(g0, indent=1))
    say(f"G0 {'PASS' if ok else 'FAIL'}: {len(idx)} inits")
    if not ok:
        sys.exit(1)


def st_bar_inits(a: argparse.Namespace) -> None:
    """G7: the ridge init (L, wl 0.003; mask and Gram on [o - W, o)) of the first --n bar origins from row 50,777."""
    root, xdata = Path(a.root), Path(a.xdata)
    rec = []
    for s in a.sets.split(","):
        D = TData(xdata, s)
        od = root / "data" / "pmf_lin_init" / s / "bar24"
        od.mkdir(parents=True, exist_ok=True)
        L = int(a.L)
        for o in range(S0, S0 + a.n):
            v, keep, mu, sd, n = ridge_init(D, o - W, o, o, L)
            name = f"r{o}_L{L}_wl{WL}.f32"
            v.tofile(od / name)
            rec.append(
                dict(
                    design=s,
                    file=f"{s}/bar24/{name}",
                    P=len(v),
                    n_keep=len(keep),
                    mu=mu,
                    sd=sd,
                    n=n,
                    kind="bar",
                    L=L,
                    fold=0,
                    origin=o,
                )
            )
    p = Path(a.root) / "inputs" / "init_index_v2.csv"
    t = pd.read_csv(p)
    t = pd.concat(
        [t[~t["file"].isin([r["file"] for r in rec])], pd.DataFrame(rec)],
        ignore_index=True,
    )
    t.to_csv(p, index=False)
    say(f"bar inits: {len(rec)} files")


# ============================================================================ post (on the GPU node)
def st_post(a: argparse.Namespace) -> None:
    """Check a GPU run and reduce it to RUN/summary/: checks.json, jobs.csv, epoch_curves.csv, epochs.csv (tune jobs),
    fc.npz (refit jobs' forecast rows, float32 as printed), train_ql.csv (fc_train jobs: the exact in-sample QLIKE).
    A file may mix tune and refit jobs (G1): each job is checked by its own mode."""
    run = Path(a.run).resolve()
    root = run.parents[1]
    sub = "refit" if (run / "refit" / "jobs_out.csv").is_file() else "tune"
    out = run / "summary"
    out.mkdir(exist_ok=True)
    jo = pd.read_csv(run / sub / "jobs_out.csv")
    jo.to_csv(out / "jobs.csv", index=False)
    lin: dict[str, dict] = {}
    if a.jobs and Path(a.jobs).is_file():
        for ln in Path(a.jobs).read_text().splitlines():
            d = dict(t.split("=", 1) for t in ln.split() if "=" in t)
            if "id" in d:
                lin[d["id"]] = d
    ii = pd.read_csv(root / "inputs" / "init_index_v2.csv")
    nkeep = dict(zip("data/pmf_lin_init/" + ii["file"], ii["n_keep"]))
    bad: list[list] = []
    missing = sorted(set(lin) - set(jo["id"]))
    if missing:
        bad.append(["missing jobs", len(missing), missing[0]])
    for _, j in jo.iterrows():
        d = lin.get(j["id"], {})
        if d and int(j["n_inputs"]) != nkeep.get(d.get("lin_init", ""), -1):
            bad.append(
                [
                    j["id"],
                    "n_inputs",
                    int(j["n_inputs"]),
                    nkeep.get(d.get("lin_init", ""), -1),
                ]
            )
        if j["mode"] == "refit" and j["status"] != "ok":
            bad.append([j["id"], "status", j["status"]])
        want_e = (
            int(d.get("epochs" if j["mode"] == "refit" else "max_epochs", -1))
            if d
            else -1
        )
        if d and j["status"] == "ok" and int(j["epochs_run"]) != want_e:
            bad.append([j["id"], "epochs_run", int(j["epochs_run"]), want_e])
    # epoch curves (the epoch_curves_remote.py reduction + capped / non-finite counts and clipped steps)
    sp = run / sub / "steps.csv.gz"
    st = pd.read_csv(sp if sp.is_file() else run / sub / "steps.csv")
    nb = dict(zip(jo["id"], zip(jo["N"].astype(int), jo["batch"].astype(int))))
    Nn = st["id"].map(lambda i: nb[i][0])
    Bb = st["id"].map(lambda i: nb[i][1])
    st["w"] = np.minimum(Bb, Nn - st["step"] * Bb)
    st["lw"] = st["loss"] * st["w"]
    st["clip"] = (st["grad_norm"] > 1.0).astype(int)
    ec = st.groupby(["id", "epoch"], sort=True).agg(
        lw=("lw", "sum"),
        w=("w", "sum"),
        gn_mean=("grad_norm", "mean"),
        gn_max=("grad_norm", "max"),
        n_capped=("n_capped", "sum"),
        n_nonfinite=("n_nonfinite", "sum"),
        n_clipped=("clip", "sum"),
        steps=("step", "size"),
    )
    ec["loss_w"] = ec["lw"] / ec["w"]
    ec = ec.drop(columns=["lw", "w"]).reset_index()
    ec.to_csv(out / "epoch_curves.csv", index=False, float_format="%.9g")
    ep = pd.read_csv(run / sub / "epochs.csv")
    ep[ep["id"].isin(set(jo.loc[jo["mode"] == "tune", "id"]))].to_csv(
        out / "epochs.csv", index=False
    )
    pr = pd.read_csv(run / sub / "preds.csv")
    res: dict[str, Any] = dict(
        run=str(run),
        sub=sub,
        n_jobs=int(len(jo)),
        status=jo["status"].value_counts().to_dict(),
    )
    rf = set(jo.loc[jo["mode"] == "refit", "id"])
    tu = set(jo.loc[jo["mode"] == "tune", "id"])
    if rf:
        fc = pr[(pr["kind"] == "fc") & pr["id"].isin(rf)]
        ids, F, r0 = [], [], []
        for jid, gg in fc.groupby("id", sort=True):
            gg = gg.sort_values("row")
            w1 = int(jo.loc[jo["id"] == jid, "w1"].iloc[0])
            n = int(lin.get(jid, {}).get("fc_n", len(gg)))
            if not np.array_equal(gg["row"].to_numpy(), np.arange(w1, w1 + n)):
                bad.append([jid, "rows", int(gg["row"].min()), int(gg["row"].max())])
            ids.append(jid)
            F.append(gg["f"].to_numpy(np.float32))
            r0.append(w1)
        if len(ids) == len(rf) and len({len(x) for x in F}) == 1:
            Fm = np.vstack(F)
            np.savez_compressed(
                out / "fc.npz", ids=np.array(ids), F=Fm, row0=np.array(r0)
            )
            res["finite"] = bool(np.isfinite(Fm).all())
            res["positive"] = bool((Fm > 0).all())
        else:
            res["finite"] = res["positive"] = False
            bad.append(["fc", "ragged, empty or missing jobs", len(ids), len(rf)])
        tr = pr[(pr["kind"] == "train") & pr["id"].isin(rf)]
        if len(tr):
            rec = []
            for jid, gg in tr.groupby("id", sort=True):
                s = "all_features" if "_all_features_" in jid else "baseline"
                y = np.fromfile(root / "inputs" / f"y_{s}.f32", "<f4")
                rr = gg["row"].to_numpy()
                rec.append(
                    dict(
                        id=jid,
                        n=len(rr),
                        row0=int(rr.min()),
                        row1=int(rr.max()) + 1,
                        train_ql=float(
                            np.mean(qlike_trainer(y[rr], gg["f"].to_numpy()))
                        ),
                    )
                )
                np.save(
                    out / f"train_f_{jid}.npy",
                    gg.sort_values("row")["f"].to_numpy(np.float32),
                )
            pd.DataFrame(rec).to_csv(
                out / "train_ql.csv", index=False, float_format="%.12g"
            )
            tp = pr[pr["kind"] == "fc_trainpath"]
            if len(tp):
                m = fc.merge(tp, on=["id", "row"], suffixes=("", "_tp"))
                res["fc_trainpath_identical"] = bool((m["f"] == m["f_tp"]).all())
    if tu:
        vv = pr[(pr["kind"] == "val") & pr["id"].isin(tu)]
        res["val_finite"] = bool(np.isfinite(vv["f"].to_numpy(float)).all())
    res["bad"] = bad[:30]
    res["n_bad"] = len(bad)
    (out / "checks.json").write_text(json.dumps(res, indent=1, default=str))
    say(json.dumps({k: v for k, v in res.items() if k != "bad"}, default=str))
    if bad or (rf and not (res.get("finite") and res.get("positive"))):
        sys.exit(1)


# ============================================================================ job files (local)
def s1_line(s: str, H: int, L: int, lr: str, seed: int, E: int, tag: str) -> str:
    return kv(
        id=f"{tag}_{s}_L{L}_H{H}_lr{lr}_E{E}_s{seed}",
        data=f"data/pmf_{s}",
        mode="tune",
        w0=TUNE["w0"],
        w1=TUNE["w1"],
        val_tail=FOLD_N,
        embargo=EMB,
        fc_n=1,
        L=L,
        H=H,
        lr=lr,
        seed=seed,
        max_epochs=E,
        patience=E,
        **COMMON,
        es="qlike",
        wd=0,
        wd_enc=0,
        lr_lin=0,
        wd_lin=0,
        lin_init=f"data/pmf_lin_init/{s}/tune_f3_L{L}_wl{WL}.f32",
    )


def wd_of(tau: str, lr: str, S: int, E: int) -> str:
    return (
        "0"
        if float(tau) == 0
        else f"{float(tau) / (0.5 * float(lr) * S * (E + 1)):.9g}"
    )


def s2_line(
    s: str,
    k: int,
    H: int,
    L: int,
    lr: str,
    tau: str,
    we: str,
    E: int,
    seed: int,
    g0: dict,
    tag: str = "v2s2",
    **extra: Any,
) -> str:
    f0, f1 = g0["folds"][str(k)]["fit"]
    S = g0["sets"][s]["stage2"][str(k)]["L"][str(L)]["S"]
    assert S == steps_per_epoch(f1 - f0, L)
    return kv(
        id=f"{tag}_{s}_f{k}_L{L}_H{H}_lr{lr}_tau{tau}_we{we}_E{E}_s{seed}",
        data=f"data/pmf_{s}",
        mode="refit",
        w0=f0,
        w1=f1,
        fc_n=extra.pop("fc_n", FC_N),
        L=L,
        H=H,
        lr=lr,
        seed=seed,
        epochs=E,
        max_epochs=E,
        patience=E,
        **COMMON,
        wd=wd_of(tau, lr, S, E),
        wd_enc=we,
        lr_lin=0,
        wd_lin=0,
        lin_init=f"data/pmf_lin_init/{s}/s2_f{k}_L{L}_wl{WL}.f32",
        **extra,
    )


def pack_files(
    items: list[tuple[str, float, int]], max_s: float
) -> list[list[tuple[str, float, int]]]:
    """Cut (line, P100-s, conc) items, in order, into files of at most max_s estimated P100-s; a file holds one
    concurrency only (2026-10-23: a file runs at its minimum concurrency, so mixing CONC 24 and 6 cells slowed the
    small cells below the estimates the time limits come from)."""
    out: list[list] = []
    cur: list = []
    tot = 0.0
    for it in items:
        if cur and (tot + it[1] > max_s or it[2] != cur[0][2]):
            out.append(cur)
            cur, tot = [], 0.0
        cur.append(it)
        tot += it[1]
    if cur:
        out.append(cur)
    return out


def tlim_min(est_s: float) -> int:
    """A file's Slurm time limit (minutes): twice the estimate + 10 min (the estimates are the old run's P100-s a
    model-epoch at its concurrency)."""
    return int(math.ceil(2.0 * est_s / 60.0 + 10.0))


def write_files(
    inp: Path,
    stem: str,
    s: str,
    groups: list[list[tuple[str, float, int]]],
    lst: list[dict],
) -> None:
    for i, gr in enumerate(groups):
        name = f"{stem}_{s}.{i}.txt"
        (inp / name).write_bytes(("\n".join(x[0] for x in gr) + "\n").encode())
        est = sum(x[1] for x in gr)
        lst.append(
            dict(
                file=name,
                set=s,
                conc=min(x[2] for x in gr),
                n_jobs=len(gr),
                est_gpu_s=round(est, 1),
                tlim_min=tlim_min(est),
            )
        )


def st_jobs(a: argparse.Namespace) -> None:
    inp = Path(a.inputs)
    inp.mkdir(parents=True, exist_ok=True)
    lst: list[dict] = []
    kind = a.kind
    if kind == "g1":
        s, H, L, lr, seed, E = "baseline", 64, 20, "0.003", SEEDS_S1[0], 300
        t = s1_line(s, H, L, lr, seed, E, "v2g1tune")
        twin = kv(
            id=f"v2g1refit_{s}_L{L}_H{H}_lr{lr}_E{E}_s{seed}",
            data=f"data/pmf_{s}",
            mode="refit",
            w0=TUNE["w0"],
            w1=TUNE["fit1"],
            fc_n=FC_N,
            fc_train=1,
            L=L,
            H=H,
            lr=lr,
            seed=seed,
            epochs=E,
            max_epochs=E,
            patience=E,
            **COMMON,
            wd=0,
            wd_enc=0,
            lr_lin=0,
            wd_lin=0,
            lin_init=f"data/pmf_lin_init/{s}/tune_f3_L{L}_wl{WL}.f32",
        )
        # the shortcut alone through the trainer (lr 0, 1 epoch: the weights never move): checks the offline TQ_0
        sc = (
            twin.replace(
                f"id=v2g1refit_{s}_L{L}_H{H}_lr{lr}_E{E}",
                f"id=v2g1short_{s}_L{L}_H{H}_lr0_E1",
            )
            .replace(f" lr={lr} ", " lr=0 ")
            .replace(
                " epochs=300 max_epochs=300 patience=300 ",
                " epochs=1 max_epochs=1 patience=1 ",
            )
        )
        assert " lr=0 " in sc and " epochs=1 " in sc
        (inp / "g1_baseline.txt").write_bytes(
            ("\n".join([t, twin, sc]) + "\n").encode()
        )
        est = 2 * E * epoch_cost(s, H, L) + 30
        lst.append(
            dict(
                file="g1_baseline.txt",
                set=s,
                conc=3,
                n_jobs=3,
                est_gpu_s=round(est, 1),
                tlim_min=tlim_min(est),
            )
        )
    elif kind in ("g2", "s1"):
        cells = [tuple(map(int, c.split(":"))) for c in a.cells.split(",")]
        lrs = a.lrs.split(",")
        seeds = [int(x) for x in a.seeds.split(",")] if a.seeds else list(SEEDS_S1)
        for s in a.sets.split(","):
            E = int(a.epochs[s] if isinstance(a.epochs, dict) else a.epochs)
            items = []
            for H, L in cells:
                for lr in lrs:
                    for seed in seeds:
                        items.append(
                            (
                                s1_line(s, H, L, lr, seed, E, a.tag),
                                E * epoch_cost(s, H, L) + 10.0,
                                conc(s, H, L),
                            )
                        )
            write_files(inp, a.tag, s, pack_files(items, a.max_file_s), lst)
    elif kind in ("s2", "g5"):
        g0 = json.loads(Path(a.g0).read_text())
        spec = json.loads(
            Path(a.spec).read_text()
        )  # {set: {H, L, lr, configs: [[tau, we, E], ...], seeds: [...]}}
        for s, sp in spec.items():
            H, L, lr = int(sp["H"]), int(sp["L"]), str(sp["lr"])
            items = []
            folds = sp.get("folds", list(FOLDS))
            for tau, we, E in sorted(
                sp["configs"], key=lambda c: (-int(c[2]), c[0], c[1])
            ):
                for k in folds:
                    for seed in sp["seeds"]:
                        n_fit = g0["folds"][str(k)]["n_fit"]
                        items.append(
                            (
                                s2_line(
                                    s,
                                    int(k),
                                    H,
                                    L,
                                    lr,
                                    str(tau),
                                    str(we),
                                    int(E),
                                    int(seed),
                                    g0,
                                    tag=a.tag,
                                ),
                                int(E) * epoch_cost(s, H, L) * n_fit / W + 15.0,
                                conc(s, H, L),
                            )
                        )
            write_files(inp, a.tag, s, pack_files(items, a.max_file_s), lst)
    elif kind == "g7":
        spec = json.loads(Path(a.spec).read_text())
        for s, sp in spec.items():
            H, L, lr, tau, we, E = (
                int(sp["H"]),
                int(sp["L"]),
                str(sp["lr"]),
                str(sp["tau"]),
                str(sp["we"]),
                int(sp["E"]),
            )
            S = steps_per_epoch(W, L)
            lines = []
            for o in range(S0, S0 + 24):
                for seed in SEEDS_SCREEN:
                    lines.append(
                        kv(
                            id=f"v2g7_{s}_r{o}_s{seed}",
                            data=f"data/pmf_{s}",
                            mode="refit",
                            w0=o - W,
                            w1=o,
                            fc_n=1,
                            L=L,
                            H=H,
                            lr=lr,
                            seed=seed,
                            epochs=E,
                            max_epochs=E,
                            patience=E,
                            **COMMON,
                            wd=wd_of(tau, lr, S, E),
                            wd_enc=we,
                            lr_lin=0,
                            wd_lin=0,
                            lin_init=f"data/pmf_lin_init/{s}/bar24/r{o}_L{L}_wl{WL}.f32",
                        )
                    )
            for c in sp["concs"]:
                name = f"v2g7_{s}_c{c}.txt"
                (inp / name).write_bytes(("\n".join(lines) + "\n").encode())
                est = len(lines) * E * epoch_cost(s, H, L)
                lst.append(
                    dict(
                        file=name,
                        set=s,
                        conc=int(c),
                        n_jobs=len(lines),
                        est_gpu_s=round(est, 1),
                        tlim_min=tlim_min(est),
                    )
                )
    else:
        raise SystemExit(f"unknown kind {kind}")
    pd.DataFrame(lst).to_csv(
        inp / f"files_{a.tag}.csv", index=False, lineterminator="\n"
    )
    say(
        f"{kind}: {len(lst)} files, {sum(x['n_jobs'] for x in lst)} jobs, est {sum(x['est_gpu_s'] for x in lst) / 3600:.2f} P100-h"
    )


# ============================================================================ Stage 1 tables (local)
S1_RE = re.compile(
    r"^(v2[a-z0-9]+)_(baseline|all_features)_L(\d+)_H(\d+)_lr([0-9.e+-]+)_E(\d+)_s(\d+)$"
)


def run_env(sm: Path) -> tuple[str, float]:
    """The GPU model (nvidia-smi name, the third line of env.txt) and the file's trainer wall seconds (RULE
    ADDENDUM 1: the type of every group is recorded; the time a model-epoch is re-measured on the type used)."""
    p = sm / "env.txt"
    if not p.is_file():
        return "", float("nan")
    ln = p.read_text().splitlines()
    gpu = ln[2].split(",")[0].strip() if len(ln) > 2 else ""
    wall = [float(x.split()[1]) for x in ln if x.startswith("wall_s ")]
    return gpu, wall[0] if wall else float("nan")


def load_summaries(
    runs: Path, pat: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """jobs, epoch curves and epochs.csv of every pulled run directory matching pat (runs/<run>/summary/)."""
    J, C, V = [], [], []
    for d in sorted(runs.glob(pat)):
        sm = d / "summary"
        if not (sm / "jobs.csv").is_file():
            continue
        j = pd.read_csv(sm / "jobs.csv")
        j["run"] = d.name
        j["gpu"], j["run_wall_s"] = run_env(sm)
        J.append(j)
        C.append(pd.read_csv(sm / "epoch_curves.csv"))
        if (sm / "epochs.csv").is_file():
            V.append(pd.read_csv(sm / "epochs.csv"))
    cat = lambda x: pd.concat(x, ignore_index=True) if x else pd.DataFrame()  # noqa: E731
    return cat(J), cat(C), cat(V)


def s1_jobs_table(runs: Path, pat: str, g0: dict) -> pd.DataFrame:
    """One row a Stage-1 (or G2) job: TQ_0, TQ_E, rho_E, epochs to rho <= 0.9 / 0.5 / 0.1, gradient and capped counts,
    validation QLIKE at E and its best epoch, the train-validation gap."""
    J, C, V = load_summaries(runs, pat)
    if J.empty:
        return pd.DataFrame()
    J = J.drop_duplicates("id", keep="first")
    rec = []
    for _, j in J.iterrows():
        m = S1_RE.match(j["id"])
        if m is None:
            continue
        tag, s, L, H, lr, E, seed = m.groups()
        L, H, E = int(L), int(H), int(E)
        tq0 = g0["sets"][s]["stage1"][str(L)]["TQ0"]
        c = C[C["id"] == j["id"]].sort_values("epoch")
        v = (
            V[V["id"] == j["id"]].sort_values("epoch")
            if not V.empty
            else pd.DataFrame()
        )
        rho = c["loss_w"].to_numpy() / tq0
        ep = c["epoch"].to_numpy()

        def first(th: float) -> float:
            k = np.flatnonzero(rho <= th)
            return float(ep[k[0]]) if len(k) else float("nan")

        r = dict(
            tag=tag,
            set=s,
            H=H,
            L=L,
            lr=lr,
            E=E,
            seed=int(seed),
            status=j["status"],
            epochs_run=int(j["epochs_run"]),
            mu_y=float(j["mu_y"]),
            sd_y=float(j["sd_y"]),
            n_inputs=int(j["n_inputs"]),
            TQ0=tq0,
            TQ1=float(c["loss_w"].iloc[0]) if len(c) else float("nan"),
            TQ_E=float(c["loss_w"].iloc[-1]) if len(c) else float("nan"),
            rho_E=float(rho[-1]) if len(rho) else float("nan"),
            rho_min=float(np.nanmin(rho)) if len(rho) else float("nan"),
            ep_rho09=first(0.9),
            ep_rho05=first(0.5),
            ep_rho01=first(0.1),
            n_capped=float(c["n_capped"].sum()),
            n_nonfinite=float(c["n_nonfinite"].sum()) + float(j["nonfinite"]),
            n_clipped_steps=float(c["n_clipped"].sum()),
            gn_mean_last=float(c["gn_mean"].iloc[-1]) if len(c) else float("nan"),
            gn_max=float(c["gn_max"].max()) if len(c) else float("nan"),
            t_train=float(j["t_train"]),
            run=j["run"],
            gpu=str(j["gpu"]),
            run_wall_s=float(j["run_wall_s"]),
        )
        if len(v):
            vq = v["val_qlike"].to_numpy()
            r.update(
                val_q_E=float(vq[-1]),
                val_q_min=float(np.nanmin(vq)),
                val_best_epoch=int(v["epoch"].iloc[int(np.nanargmin(vq))]),
                val_q_1=float(vq[0]),
                gap_E=float(vq[-1] - r["TQ_E"]),
            )
        rec.append(r)
    return pd.DataFrame(rec)


def st_s1_table(a: argparse.Namespace) -> None:
    g0 = json.loads(Path(a.g0).read_text())
    T = s1_jobs_table(Path(a.runs), a.pattern, g0)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    T.to_csv(out / f"{a.name}_jobs.csv", index=False, float_format="%.9g")
    cell_rows = []
    picks: dict[str, Any] = {}
    for s, Ts in T.groupby("set"):
        cells = {}
        for (H, L), Tc in Ts.groupby(["H", "L"]):
            by_lr = []
            for lr, Tl in Tc.groupby("lr"):
                ok = (Tl["status"] == "ok").all() and Tl["n_nonfinite"].sum() == 0
                by_lr.append(
                    dict(
                        lr=lr,
                        n=len(Tl),
                        ok=bool(ok),
                        TQ_E=float(Tl["TQ_E"].mean()),
                        rho_E=float(Tl["rho_E"].mean()),
                        rho_max=float(Tl["rho_E"].max()),
                        mem_seeds=int((Tl["rho_E"] <= 0.5).sum()),
                        mem01_seeds=int((Tl["rho_E"] <= 0.1).sum()),
                    )
                )
            okl = [x for x in by_lr if x["ok"] and x["n"] == a.n_seeds]
            if not okl:
                continue
            best = min(okl, key=lambda x: (x["TQ_E"], float(x["lr"])))
            Tb = Tc[Tc["lr"] == best["lr"]]
            cost = epoch_cost(s, int(H), int(L))
            row = dict(
                set=s,
                H=int(H),
                L=int(L),
                lr=best["lr"],
                n_seeds=best["n"],
                TQ0=float(Tb["TQ0"].iloc[0]),
                TQ_E=best["TQ_E"],
                rho_E=best["rho_E"],
                rho_seeds=";".join(f"{x:.4f}" for x in Tb.sort_values("seed")["rho_E"]),
                memorizes=bool(best["mem_seeds"] == best["n"]),
                seeds_agree=bool(best["mem_seeds"] in (0, best["n"])),
                mem01=bool(best["mem01_seeds"] == best["n"]),
                ep_rho09=float(Tb["ep_rho09"].mean()),
                ep_rho05=float(Tb["ep_rho05"].mean()),
                ep_rho01=float(Tb["ep_rho01"].mean()),
                val_q_E=float(Tb["val_q_E"].mean())
                if "val_q_E" in Tb
                else float("nan"),
                val_q_min=float(Tb["val_q_min"].mean())
                if "val_q_min" in Tb
                else float("nan"),
                val_best_epoch=float(Tb["val_best_epoch"].mean())
                if "val_best_epoch" in Tb
                else float("nan"),
                gap_E=float(Tb["gap_E"].mean()) if "gap_E" in Tb else float("nan"),
                n_capped=float(Tb["n_capped"].sum()),
                n_clipped_steps=float(Tb["n_clipped_steps"].sum()),
                n_nonfinite=float(Tb["n_nonfinite"].sum()),
                cost_epoch_p100s=cost,
                walk_p100h_per_epoch=round(cost * 33_970 * 5 / 3600, 2),
                gpu=";".join(sorted(set(Tb["gpu"].astype(str)))),
                t_train_s_per_epoch=float((Tb["t_train"] / Tb["epochs_run"]).mean()),
                lrs=";".join(
                    f"{x['lr']}:{x['TQ_E']:.6f}:{x['rho_E']:.4f}{'' if x['ok'] else '!'}"
                    for x in by_lr
                ),
            )
            cells[(int(H), int(L))] = row
            cell_rows.append(row)
        mem = [r for r in cells.values() if r["memorizes"]]
        if mem:
            p = min(mem, key=lambda r: (r["cost_epoch_p100s"], r["H"], r["L"]))
            label = "memorizing"
        elif cells:
            p = min(cells.values(), key=lambda r: (r["rho_E"], r["cost_epoch_p100s"]))
            label = "capacity-limited"
        else:
            continue
        core_run = all((H, L) in cells for H in CORE_H for L in CORE_L)
        need_h1024 = (
            core_run
            and not any(cells[(H, L)]["memorizes"] for H in CORE_H for L in CORE_L)
            and not all((1024, L) in cells for L in (8, 20))
        )
        need_l115 = (
            bool(mem) and p["L"] == 48 and not all((H, 115) in cells for H in (64, 256))
        )
        picks[s] = dict(
            pick=dict(H=p["H"], L=p["L"], lr=p["lr"]),
            label=label,
            rho_E=p["rho_E"],
            rho_seeds=p["rho_seeds"],
            seeds_agree=p["seeds_agree"],
            core_complete=core_run,
            next_edge="H1024 x L{8,20}"
            if need_h1024
            else ("L115 x H{64,256}" if need_l115 else None),
            final=bool(core_run and not need_h1024 and not need_l115),
            gpu_types=sorted(set(Ts["gpu"].astype(str))),
        )
    # RULE ADDENDUM 1: the type of every run and its measured GPU-s a model-epoch (the file's trainer wall / its jobs'
    # epochs: the concurrency-shared rate on that type)
    if len(T):
        R = T.groupby("run").agg(
            set=("set", "first"),
            gpu=("gpu", "first"),
            wall_s=("run_wall_s", "first"),
            n_jobs=("seed", "size"),
            model_epochs=("epochs_run", "sum"),
        )
        R["gpu_s_per_model_epoch"] = R["wall_s"] / R["model_epochs"]
        R.to_csv(out / f"{a.name}_runs.csv", float_format="%.6g")
    pd.DataFrame(cell_rows).to_csv(
        out / f"{a.name}_cells.csv", index=False, float_format="%.6g"
    )
    (out / f"{a.name}_pick.json").write_text(json.dumps(picks, indent=1))
    say(json.dumps(picks))


# ============================================================================ Stage 2 tables (local)
S2_RE = re.compile(
    r"^(v2[a-z0-9]+)_(baseline|all_features)_f(\d)_L(\d+)_H(\d+)_lr([0-9.e+-]+)_tau([0-9.e+-]+)_we([0-9.e+-]+)_E(\d+)_s(\d+)$"
)


def load_fc(runs: Path, pat: str) -> dict[str, np.ndarray]:
    """id -> forecast rows (float32 as printed) of every pulled refit run matching pat; first copy wins."""
    out: dict[str, np.ndarray] = {}
    for d in sorted(runs.glob(pat)):
        p = d / "summary" / "fc.npz"
        c = d / "summary" / "checks.json"
        if not p.is_file() or not c.is_file():
            continue
        z = np.load(p)
        for i, jid in enumerate(z["ids"]):
            out.setdefault(str(jid), z["F"][i].astype(np.float64))
    return out


def cfg_key(tau: str, we: str, E: int) -> str:
    return f"tau{tau}_we{we}_E{E}"


def s2_matrix(
    fc: dict[str, np.ndarray], y: np.ndarray, s: str, seeds: tuple, g0: dict
) -> pd.DataFrame:
    """Per configuration of set s with every seed and fold: the per-fold QLIKE of the seed-mean forecast and C."""
    by: dict[tuple, dict[tuple, np.ndarray]] = {}
    for jid, F in fc.items():
        m = S2_RE.match(jid)
        if m is None or m.group(2) != s:
            continue
        _, _, k, L, H, lr, tau, we, E, seed = m.groups()
        by.setdefault((tau, we, int(E), L, H, lr), {})[(int(k), int(seed))] = F
    rec = []
    for (tau, we, E, L, H, lr), d in by.items():
        if not all((k, sd) in d for k in FOLDS for sd in seeds):
            continue
        q, ok = {}, True
        for k in FOLDS:
            Fk = np.vstack([d[(k, sd)][EMB:] for sd in seeds])
            if not (np.isfinite(Fk).all() and (Fk > 0).all()):
                ok = False
                break
            v0, v1 = g0["folds"][str(k)]["val"]
            q[k] = qlike_plain(y[v0:v1], Fk.mean(axis=0))
        rec.append(
            dict(
                cfg=cfg_key(tau, we, E),
                tau=tau,
                we=we,
                E=E,
                H=int(H),
                L=int(L),
                lr=lr,
                ok=ok,
                **{f"q_f{k}": q.get(k, float("nan")) for k in FOLDS},
                C=float(np.mean([q[k] for k in FOLDS])) if ok else float("nan"),
            )
        )
    return pd.DataFrame(rec)


def edges(
    grid: dict[str, list], pick: dict, ext_done: dict[str, dict]
) -> dict[str, str]:
    """The axes whose pick sits at an extensible edge of the current grid -> the value to add."""
    add: dict[str, str] = {}
    taus = sorted(grid["tau"], key=float)
    if pick["tau"] == taus[0]:
        n = ext_done["tau"]["low"]
        if n < 2:
            add["tau"] = TAU_EXT["low"][n]
    if pick["tau"] == taus[-1]:
        n = ext_done["tau"]["high"]
        if n < 2:
            add["tau"] = TAU_EXT["high"][n]
    wes = sorted(grid["we"], key=float)
    if len(wes) > 1 and pick["we"] == wes[-1]:
        n = ext_done["we"]["high"]
        if n < 2:
            add["we"] = WE_EXT["high"][n]
    Es = sorted(grid["E"])
    i = E_LADDER.index(int(pick["E"]))
    if int(pick["E"]) == Es[0] and ext_done["E"]["low"] < 2 and i > 0:
        add["E"] = str(E_LADDER[i - 1])
    if int(pick["E"]) == Es[-1] and ext_done["E"]["high"] < 2 and i + 1 < len(E_LADDER):
        add["E"] = str(E_LADDER[i + 1])
    return add


def st_s2_table(a: argparse.Namespace) -> None:
    """The Stage-2 rule on the configurations run so far: screen C, shortlist (top 5), fresh-seed C, pick, edges.
    State file (per set): grid, extension counts.  Writes the next spec (configs to run) for st_jobs --kind s2."""
    g0 = json.loads(Path(a.g0).read_text())
    st = json.loads(Path(a.state).read_text())
    runs = Path(a.runs)
    fc = load_fc(runs, a.pattern)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    nxt: dict[str, Any] = {}
    rep: dict[str, Any] = {}
    for s, ss in st.items():
        if ss.get("frozen"):
            continue
        y = np.fromfile(Path(a.y_dir) / f"y_{s}.f32", "<f4").astype(np.float64)
        grid = ss["grid"]
        want = [
            (t, w, int(E)) for t in grid["tau"] for w in grid["we"] for E in grid["E"]
        ]
        Ts = s2_matrix(fc, y, s, SEEDS_SCREEN, g0)
        Ts = (
            Ts[(Ts["H"] == ss["H"]) & (Ts["L"] == ss["L"]) & (Ts["lr"] == ss["lr"])]
            if len(Ts)
            else Ts
        )
        have = set(Ts["cfg"]) if len(Ts) else set()
        missing = [c for c in want if cfg_key(str(c[0]), str(c[1]), c[2]) not in have]
        if missing:
            nxt[s] = dict(
                H=ss["H"],
                L=ss["L"],
                lr=ss["lr"],
                seeds=list(SEEDS_SCREEN),
                configs=[list(map(str, c[:2])) + [c[2]] for c in missing],
            )
            rep[s] = dict(state="screen incomplete", missing=len(missing))
            continue
        Ts = Ts[Ts["cfg"].isin({cfg_key(t, w, E) for t, w, E in want})]
        Ts.sort_values(["C", "E", "cfg"]).to_csv(
            out / f"s2_screen_{s}.csv", index=False, float_format="%.9g"
        )
        short = Ts[Ts["ok"]].sort_values(["C", "E", "cfg"]).head(5)
        Tf = s2_matrix(fc, y, s, SEEDS_FRESH, g0)
        Tf = (
            Tf[(Tf["H"] == ss["H"]) & (Tf["L"] == ss["L"]) & (Tf["lr"] == ss["lr"])]
            if len(Tf)
            else Tf
        )
        havef = set(Tf["cfg"]) if len(Tf) else set()
        missf = [r for _, r in short.iterrows() if r["cfg"] not in havef]
        if missf:
            nxt[s] = dict(
                H=ss["H"],
                L=ss["L"],
                lr=ss["lr"],
                seeds=list(SEEDS_FRESH),
                configs=[[str(r["tau"]), str(r["we"]), int(r["E"])] for r in missf],
            )
            rep[s] = dict(
                state="fresh incomplete",
                shortlist=list(short["cfg"]),
                missing=len(missf),
            )
            continue
        F = Tf[Tf["cfg"].isin(set(short["cfg"]))].copy()
        F = F[F["ok"]]
        Cmin = F["C"].min()
        tie = F[F["C"] <= Cmin + 1e-7].sort_values(["E", "cfg"])
        p = tie.iloc[0]
        F.sort_values("C").to_csv(
            out / f"s2_fresh_{s}.csv", index=False, float_format="%.9g"
        )
        pick: dict[str, Any] = dict(tau=str(p["tau"]), we=str(p["we"]), E=int(p["E"]))
        add = edges(grid, pick, ss["ext"])
        rep[s] = dict(
            state="picked",
            pick=pick,
            C_fresh=float(p["C"]),
            shortlist=list(short["cfg"]),
            C_screen=dict(zip(short["cfg"], short["C"])),
            add=add,
        )
        if add:
            ng = {k: list(v) for k, v in grid.items()}
            for ax, val in add.items():
                key = "tau" if ax == "tau" else ("we" if ax == "we" else "E")
                ng[key] = sorted(
                    set(ng[key]) | {val if key != "E" else int(val)}, key=float
                )
                if key == "tau":
                    side = (
                        "low"
                        if float(val) < float(pick["tau"]) or float(pick["tau"]) == 0
                        else "high"
                    )
                elif key == "we":
                    side = "high"
                else:
                    side = "low" if int(val) < pick["E"] else "high"
                ss["ext"][key][side] += 1
            ss["grid"] = ng
            ss.setdefault("history", []).append(
                dict(pick=pick, add=add, C_fresh=float(p["C"]))
            )
            new = [(t, w, int(E)) for t in ng["tau"] for w in ng["we"] for E in ng["E"]]
            newm = [c for c in new if cfg_key(str(c[0]), str(c[1]), c[2]) not in have]
            nxt[s] = dict(
                H=ss["H"],
                L=ss["L"],
                lr=ss["lr"],
                seeds=list(SEEDS_SCREEN),
                configs=[[str(c[0]), str(c[1]), c[2]] for c in newm],
            )
        else:
            ss["frozen"] = dict(
                pick=pick, C_fresh=float(p["C"]), shortlist=list(short["cfg"])
            )
    if a.write_state:
        Path(a.state).write_text(json.dumps(st, indent=1))
    (out / "s2_next_spec.json").write_text(json.dumps(nxt, indent=1))
    (out / "s2_report.json").write_text(json.dumps(rep, indent=1, default=str))
    say(json.dumps(rep, default=str))
    say(f"next: {[(s, len(v['configs']), v['seeds'][0]) for s, v in nxt.items()]}")


def strat_boot_index(
    n_per_fold: dict[int, int], Bn: int, block: int, seed: int
) -> dict[int, np.ndarray]:
    """Moving-block bootstrap within each fold: ceil(n / block) block starts uniform on 0..n-block, truncated to n."""
    rng = np.random.RandomState(seed)
    out = {}
    for k in sorted(n_per_fold):
        n = n_per_fold[k]
        nb = int(math.ceil(n / block))
        st = rng.randint(0, n - block + 1, size=(Bn, nb))
        out[k] = ((st[:, :, None] + np.arange(block)).reshape(Bn, -1))[:, :n]
    return out


def st_s2_report(a: argparse.Namespace) -> None:
    """Reported, not deciding: per set, over the shortlist on the fresh seeds -- the stratified 10-day moving-block
    bootstrap SE of C(cfg) - C(pick), the seed jackknife SE of C, and the 90% MCS (T_max)."""
    g0 = json.loads(Path(a.g0).read_text())
    st = json.loads(Path(a.state).read_text())
    fc = load_fc(Path(a.runs), a.pattern)
    out: dict[str, Any] = {}
    for s, ss in st.items():
        if not ss.get("frozen"):
            continue
        y = np.fromfile(Path(a.y_dir) / f"y_{s}.f32", "<f4").astype(np.float64)
        shortl = ss["frozen"]["shortlist"]
        pick = cfg_key(
            ss["frozen"]["pick"]["tau"],
            ss["frozen"]["pick"]["we"],
            ss["frozen"]["pick"]["E"],
        )
        losses: dict[str, dict[int, np.ndarray]] = {}
        jk: dict[str, float] = {}
        for c in shortl:
            m = re.match(r"tau(.+)_we(.+)_E(\d+)$", c)
            assert m is not None
            tau, we, E = m.group(1), m.group(2), int(m.group(3))
            pre = f"_{s}_f{{k}}_L{ss['L']}_H{ss['H']}_lr{ss['lr']}_tau{tau}_we{we}_E{E}_s{{sd}}"
            Fk = {
                k: np.vstack(
                    [
                        fc[next(i for i in fc if i.endswith(pre.format(k=k, sd=sd)))][
                            EMB:
                        ]
                        for sd in SEEDS_FRESH
                    ]
                )
                for k in FOLDS
            }
            losses[c] = {}
            for k in FOLDS:
                v0, v1 = g0["folds"][str(k)]["val"]
                yy = y[v0:v1]
                q = np.where(yy > 0, (yy / Fk[k].mean(axis=0)) ** 2, np.nan)
                losses[c][k] = q - np.log(q) - 1.0
            # seed jackknife of C
            cj = []
            for i in range(len(SEEDS_FRESH)):
                ks = [j for j in range(len(SEEDS_FRESH)) if j != i]
                cj.append(
                    np.mean(
                        [
                            qlike_plain(
                                y[slice(*g0["folds"][str(k)]["val"])],
                                Fk[k][ks].mean(axis=0),
                            )
                            for k in FOLDS
                        ]
                    )
                )
            cj_a = np.asarray(cj)
            n = len(cj_a)
            jk[c] = float(math.sqrt((n - 1) / n * np.sum((cj_a - cj_a.mean()) ** 2)))
        nfold = {k: len(losses[pick][k]) for k in FOLDS}
        idx = strat_boot_index(nfold, 2000, 130, 20261023)

        def cstar(c: str) -> np.ndarray:
            vals = []
            for k in FOLDS:
                Lk = losses[c][k][idx[k]]
                vals.append(np.nanmean(Lk, axis=1))
            return np.mean(vals, axis=0)

        Cs = {
            c: float(np.mean([np.nanmean(losses[c][k]) for k in FOLDS])) for c in shortl
        }
        boot = {c: cstar(c) for c in shortl}
        se_d = {c: float(np.std(boot[c] - boot[pick], ddof=1)) for c in shortl}
        # MCS (T_max) on C with the same bootstrap replicates
        names = list(shortl)
        Cv = np.array([Cs[c] for c in names])
        Bm = np.column_stack([boot[c] for c in names])
        active = list(range(len(names)))
        pv: dict[str, float] = {}
        run_p = 0.0
        elim = []
        while len(active) > 1:
            aa = np.asarray(active)
            dbar = Cv[aa] - Cv[aa].mean()
            dstar = Bm[:, aa] - Bm[:, aa].mean(axis=1, keepdims=True)
            var = np.maximum(((dstar - dbar) ** 2).mean(axis=0), 1e-300)
            t = dbar / np.sqrt(var)
            tst = ((dstar - dbar) / np.sqrt(var)).max(axis=1)
            p = float(np.mean(tst >= t.max()))
            run_p = max(run_p, p)
            if p >= 0.10:
                break
            w = int(np.argmax(t))
            pv[names[active[w]]] = run_p
            elim.append(names[active[w]])
            del active[w]
        for i in active:
            pv[names[i]] = max(run_p, 0.10)
        out[s] = dict(
            pick=pick,
            C_fresh=Cs,
            boot_se_diff_vs_pick=se_d,
            seed_jackknife_se=jk,
            mcs90=[names[i] for i in active],
            mcs_pvals=pv,
            mcs_eliminated=elim,
            bootstrap=dict(B=2000, block_rows=130, seed=20261023, stratified="by fold"),
        )
    Path(a.out).write_text(json.dumps(out, indent=1))
    say(json.dumps(out, indent=1)[:3000])


def st_g2_decide(a: argparse.Namespace) -> None:
    """G2 (RULE GATES): per set, E1 (300 or 1,000) from (256, 20) at the set's best lr by the 2-seed mean TQ_300, and
    the Stage-1 lr set from the argmin lr of every (set, cell)."""
    g0 = json.loads(Path(a.g0).read_text())
    T = s1_jobs_table(Path(a.runs), a.pattern, g0)
    T.to_csv(Path(a.out).with_suffix(".jobs.csv"), index=False, float_format="%.9g")
    T3 = T[T["E"] == 300]
    T10 = T[T["E"] == 1000]
    argmins: dict[str, str] = {}
    res: dict[str, Any] = dict(sets={})
    for s, Ts in T3.groupby("set"):
        rs: dict[str, Any] = dict(cells={})
        for (H, L), Tc in Ts.groupby(["H", "L"]):
            m = Tc.groupby("lr").agg(
                TQ=("TQ_E", "mean"),
                rho=("rho_E", "mean"),
                n=("seed", "size"),
                nf=("n_nonfinite", "sum"),
            )
            m = m[(m["n"] == 2) & (m["nf"] == 0)]
            best = str(m.sort_values(["TQ"]).index[0])
            argmins[f"{s}_H{H}_L{L}"] = best
            rs["cells"][f"H{H}_L{L}"] = dict(
                best_lr=best,
                by_lr={
                    k: dict(TQ_300=float(v["TQ"]), rho_300=float(v["rho"]))
                    for k, v in m.iterrows()
                },
            )
        b = rs["cells"]["H256_L20"]
        lr = b["best_lr"]
        rho300 = b["by_lr"][lr]["rho_300"]
        t10 = T10[
            (T10["set"] == s) & (T10["H"] == 256) & (T10["L"] == 20) & (T10["lr"] == lr)
        ]
        rho1000 = float(t10["rho_E"].mean()) if len(t10) == 2 else float("nan")
        gain = (rho300 - rho1000) / rho300
        E1 = 300 if (rho300 <= 0.5 or gain < 0.2) else 1000
        rs.update(
            lr_256_20=lr,
            rho_300=rho300,
            rho_1000=rho1000,
            rel_gain=gain,
            E1=E1,
            E1000_by_lr={
                k: float(v["rho_E"].mean())
                for k, v in T10[(T10["set"] == s)].groupby("lr")
            },
        )
        res["sets"][s] = rs
    lrs = {"0.003", "0.01"}
    if "0.001" in argmins.values():
        lrs.add("0.001")
    if "0.01" in argmins.values():
        lrs.add("0.03")
    res["argmins"] = argmins
    res["stage1_lrs"] = sorted(lrs, key=float)
    Path(a.out).write_text(json.dumps(res, indent=1))
    say(json.dumps(res, indent=1))


def st_g1_check(a: argparse.Namespace) -> None:
    """G1 on a pulled run directory (summary/ + tune/steps.csv.gz + tune/epochs.csv): 300 epochs; steps.csv
    text-identical (the id field dropped) between the tune job and its refit twin; the twin's exact in-sample QLIKE
    (fc_train rows) within 1% of the last-epoch running mean; its validation QLIKE (trainer formula on rows
    [47,527, 50,777)) = the tune job's epochs.csv epoch-300 val_qlike within 1e-5 relative; no non-finite; and the
    shortcut alone through the trainer (lr 0, 1 epoch) against the offline TQ_0."""
    run = Path(a.run)
    g0 = json.loads(Path(a.g0).read_text())
    y = np.fromfile(Path(a.y_dir) / "y_baseline.f32", "<f4").astype(np.float64)
    jo = pd.read_csv(run / "summary" / "jobs.csv").set_index("id")
    tid = next(i for i in jo.index if i.startswith("v2g1tune_"))
    rid = next(i for i in jo.index if i.startswith("v2g1refit_"))
    sid = next(i for i in jo.index if i.startswith("v2g1short_"))
    lines: dict[str, list[str]] = {tid: [], rid: [], sid: []}
    with gzip.open(run / "tune" / "steps.csv.gz", "rt") as fh:
        next(fh)
        for ln in fh:
            i, rest = ln.split(",", 1)
            lines[i].append(rest)
    ident = lines[tid] == lines[rid]
    nd = sum(x != y_ for x, y_ in zip(lines[tid], lines[rid]))
    ec = pd.read_csv(run / "summary" / "epoch_curves.csv")
    tq = ec[ec["id"] == rid].sort_values("epoch")["loss_w"].to_numpy()
    tq_t = ec[ec["id"] == tid].sort_values("epoch")["loss_w"].to_numpy()
    trq = pd.read_csv(run / "summary" / "train_ql.csv").set_index("id")
    exact = float(trq.loc[rid, "train_ql"])
    z = np.load(run / "summary" / "fc.npz")
    F = {str(i): z["F"][k].astype(np.float64) for k, i in enumerate(z["ids"])}
    fv = F[rid][EMB:]
    v0, v1 = TUNE["val"]
    vq_twin = float(np.mean(qlike_trainer(y[v0:v1], fv)))
    ep = pd.read_csv(run / "tune" / "epochs.csv")
    vq_tune = float(ep[(ep["id"] == tid) & (ep["epoch"] == 300)]["val_qlike"].iloc[0])
    s1 = g0["sets"]["baseline"]["stage1"]["20"]
    tq0_trainer = float(trq.loc[sid, "train_ql"])
    res: dict[str, Any] = dict(
        epochs_run=dict(
            tune=int(jo.loc[tid, "epochs_run"]), refit=int(jo.loc[rid, "epochs_run"])
        ),
        steps_identical=bool(ident),
        n_steps=len(lines[tid]),
        n_step_lines_differing=int(nd),
        TQ_300_running=float(tq[-1]),
        TQ_300_running_tune=float(tq_t[-1]),
        in_sample_exact=exact,
        in_sample_rel_dev=float(abs(exact - tq[-1]) / tq[-1]),
        val_qlike_twin=vq_twin,
        val_qlike_tune_epoch300=vq_tune,
        val_rel_dev=float(abs(vq_twin - vq_tune) / vq_tune),
        nonfinite=float(
            ec[ec["id"].isin([tid, rid])]["n_nonfinite"].sum()
            + jo.loc[[tid, rid], "nonfinite"].sum()
        ),
        TQ0_offline=float(s1["TQ0"]),
        TQ0_trainer=tq0_trainer,
        TQ0_rel_dev=float(abs(tq0_trainer - s1["TQ0"]) / s1["TQ0"]),
        mu_sd_job=[float(jo.loc[rid, "mu_y"]), float(jo.loc[rid, "sd_y"])],
        mu_sd_init=[s1["mu"], s1["sd"]],
        rho_300=float(tq[-1] / s1["TQ0"]),
        rows_train=[int(trq.loc[rid, "row0"]), int(trq.loc[rid, "row1"])],
    )
    res["pass"] = bool(
        res["epochs_run"] == dict(tune=300, refit=300)
        and ident
        and res["in_sample_rel_dev"] <= 0.01
        and res["val_rel_dev"] <= 1e-5
        and res["nonfinite"] == 0
    )
    Path(a.out).write_text(json.dumps(res, indent=1))
    say(json.dumps(res))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "stage",
        choices=[
            "prep",
            "post",
            "bar-inits",
            "jobs",
            "s1-table",
            "s2-table",
            "s2-report",
            "g1-check",
            "g2-decide",
        ],
    )
    ap.add_argument("--root", default="")
    ap.add_argument("--xdata", default="")
    ap.add_argument("--folds-json", default="")
    ap.add_argument("--alt-rows", default="")
    ap.add_argument("--run", default="")
    ap.add_argument("--jobs", default="")
    ap.add_argument("--sets", default="baseline,all_features")
    ap.add_argument("--L", default="20")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--inputs", default="")
    ap.add_argument("--kind", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--cells", default="")
    ap.add_argument("--lrs", default="")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--epochs", default="300")
    ap.add_argument("--max-file-s", type=float, default=1500.0)
    ap.add_argument("--g0", default="")
    ap.add_argument("--spec", default="")
    ap.add_argument("--runs", default="")
    ap.add_argument("--pattern", default="*")
    ap.add_argument("--out", default="")
    ap.add_argument("--name", default="s1")
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--state", default="")
    ap.add_argument("--y-dir", default="")
    ap.add_argument("--write-state", action="store_true")
    a = ap.parse_args()
    if a.stage == "jobs" and a.epochs.startswith("{"):
        a.epochs = json.loads(a.epochs)
    dict(
        prep=st_prep,
        post=st_post,
        jobs=st_jobs,
        **{
            "bar-inits": st_bar_inits,
            "s1-table": st_s1_table,
            "s2-table": st_s2_table,
            "s2-report": st_s2_report,
            "g1-check": st_g1_check,
            "g2-decide": st_g2_decide,
        },
    )[a.stage](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
