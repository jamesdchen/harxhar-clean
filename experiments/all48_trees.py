"""ALL48 Part 2, the trees (results/all48_2026-10-30/RULE.txt section 2.3; EXPLORATORY, proposed after the main results
were seen, not pre-registered): LightGBM, XGBoost and the random forest on oneDAL (rf_skx), HAR benchmark set and full
set, refitted at EVERY interval origin of the 48-interval design -- every OOS row dated 2014-01-02 .. 2024-04-30,
overnight, weekend and holiday rows included -- each from scratch on the W rows before it (design rows [r - W, r), the
rows the linear walk uses at r; W = the design cache's window, 144,000 rows).

The main trees' model code is used unchanged: experiments/pooled_trees_rerun_13.py model_factory / fit_predict (seed 42,
the frozen configurations), rv_bars, cpu_model, versions, the oneDAL dispatch check and the one-CPU-family rule
(ONE_CPU, cpu_family); LightGBM with deterministic=True and force_col_wise=True (train3way's ALL48 addenda B / D; the
ablations).  The design cache is any cache of experiments/pool_linear_pretune.py design (the 48-interval cell-B cache of
experiments/all48_linear.py, or the cell-B 13-bar cache for gate T1); its label is "a48" (sample 48) or "r13" (rth13).
No state carries between refits, so a piece is any contiguous block of origins and pieces of one class and set are
independent.  Nothing in pooled_trees_rerun_13 changes; importing this file changes nothing there.

usage: python experiments/all48_trees.py <mode> [options]
  check   --design D [--cellb D13]      the 48-interval caches as the trees read them (sample, window 144,000, columns
                                        13 / 565, flags; with --cellb: names = the 13-bar cache's + is_overnight and y /
                                        B equal at every common stamp) -> stdout (+ --json F); exit 3 on a failure
  walk    --design D --est E --set S (--part i/n | --ends K | --first K) --out O [--threads T] [--tag X]
                                        -> O/pieces/<label>_<est>_<set>_<tag>.parquet + .json (checkpoint .ckpt.json
                                        every --ckpt-sec s; resumable; a forest piece never resumes on another family)
  tasks   --design D --out F [--target-hours 1] [--from-gate G0]
                                        the task file, lines "<est> <set> <i>/<n> <hours at EPYC speed>" (HAR sets,
                                        LightGBM, XGBoost, then the forest's full set), every piece <= --target-hours at
                                        the seconds a refit of SEC_REFIT (train3way's ALL48 piece metas, EPYC 7513, 8
                                        threads) or of the reference gate's T2 pieces (--from-gate) -> F + F.projection.csv
  gate    --out G [--ref G0] [--est E1,E2,..]
                                        T1: G/pieces/r13_<est>_<set>_ends10 (walked on the cell-B 13-bar cache, md5 =
                                        CELLB_MD5) against the main forecasts (LightGBM / XGBoost text-identical, the
                                        forest within RF_TOL; true_adj, B equal); T2: G/pieces/a48_<est>_<set>_ends10
                                        against G0's (LightGBM / XGBoost bit-identical, the forest within RF_TOL; without
                                        --ref, G is the reference and its CPU model must be EPYC 7513 / 7542) ->
                                        G/gate_<ests>.json and G/passed_<est> (the CPU model) for each class that passes;
                                        exit 3 if any fails
  merge   --design D --out O --gates G1,G2,.. [--est E ..] [--set S ..] [--require-all]
                                        one complete piece set a class and set (every origin once, finite, plain =
                                        pred_adj^2 x B, CPU models gated for the class, one family a forest piece,
                                        LightGBM deterministic + col-wise; a piece's (cluster, CPU model) gated, the
                                        cluster read off the host) -> O/forecasts.parquet (+ .json); exit 1 under
                                        --require-all if a class / set has no complete set
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(HERE), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pooled_trees_rerun_13 as T  # noqa: E402

FIRST_DAY, LAST_DAY = "2014-01-02", "2024-04-30"
EST = ("lgbm", "xgb", "rf_skx")
SETS = ("baseline", "all_features")
RF_TOL = 1e-14
LABEL = {"48": "a48", "rth13": "r13"}
WINDOW = {"48": 144000, "rth13": 39000}  # RULE.txt sections 2.1 and 2.3 (T1)
N_COLS = {"48": {"baseline": 13, "all_features": 565}, "rth13": {"baseline": 12, "all_features": 564}}
N_RTH13 = 33970  # the main trees' origins on the 13-bar design
# the cell-B 13-bar caches (results/fomcfix_2026-10-26/gates/G3.json md5_new), gate T1's design
CELLB_MD5 = {
    "baseline": {
        "X.npy": "05da4b5b30ec4579df77f1a3b6dd57ed",
        "y.npy": "996504f16da7982d64656d1ba0d9722a",
        "b_oos.npy": "7670f01a3055265dfac28ebac1eca864",
        "dates_oos.csv": "d8a203fe273cd3e8e34987fe5a73048e",
    },
    "all_features": {
        "X.npy": "725eb201044783d3cdc62501307dcc3a",
        "y.npy": "996504f16da7982d64656d1ba0d9722a",
        "b_oos.npy": "7670f01a3055265dfac28ebac1eca864",
        "dates_oos.csv": "d8a203fe273cd3e8e34987fe5a73048e",
    },
}
# the main forecasts on cell B (gate T1): LightGBM / XGBoost full set re-walked on cell B, HAR rows = the templates'
TEMPLATE = {
    "lgbm": ROOT / "results" / "pooled_trees_final_2014start_fomcfix_w3000" / "forecasts.parquet",
    "xgb": ROOT / "results" / "pooled_trees_final_2014start_fomcfix_w3000" / "forecasts.parquet",
    "rf_skx": ROOT / "results" / "pooled_rf_skx_2014start_fomcfix_w3000" / "forecasts.parquet",
}
# seconds a refit at 8 threads, EPYC 7513, W = 144,000 (train3way's ALL48 piece metas, median; RULE.txt section 4);
# the forest's HAR set measured at 20 threads (1.23 s = 24.6 CPU-seconds) and taken at 8 threads as 24.6 / 8
SEC_REFIT = {
    ("lgbm", "all_features"): 9.51,
    ("lgbm", "baseline"): 0.95,
    ("xgb", "all_features"): 12.22,
    ("xgb", "baseline"): 0.70,
    ("rf_skx", "all_features"): 249.9,
    ("rf_skx", "baseline"): 3.08,
}
# task order: cheapest first, results early (RULE.txt section 6)
ORDER = [
    ("lgbm", "baseline"),
    ("xgb", "baseline"),
    ("rf_skx", "baseline"),
    ("lgbm", "all_features"),
    ("xgb", "all_features"),
    ("rf_skx", "all_features"),
]
PIECE_COLS = T.COLS + ["fit_row"]
REF_CPUS = set(T.CPU_FAMILY)  # CARC EPYC 7513 / 7542: the main trees' family, T2's reference


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


# ============================================================================ design
class Design:
    """One design cache: <root>/_work/design_<set>/ (X.npy read through a memory map, y.npy, b_oos.npy, dates_oos.csv,
    design.json).  OOS row i is design row W + i (W = the cache's window)."""

    def __init__(self, root: Path, s: str):
        self.root, self.set = Path(root), s
        self.dir = self.root / "_work" / f"design_{s}"
        self.meta = json.loads((self.dir / "design.json").read_text())
        cf = dict(self.meta.get("config") or {})
        self.config = cf
        self.sample = str(cf.get("sample"))
        assert self.sample in LABEL, f"{self.dir}: sample {self.sample!r}"
        self.label = LABEL[self.sample]
        self.W = int(cf["window"])
        self.names = list(self.meta["names"])
        assert len(self.names) == N_COLS[self.sample][s], (self.dir, len(self.names))
        self.X = np.load(self.dir / "X.npy", mmap_mode="r")
        self.y = np.load(self.dir / "y.npy")
        self.b = np.load(self.dir / "b_oos.npy")
        self.d = pd.DatetimeIndex(pd.read_csv(self.dir / "dates_oos.csv", parse_dates=["date"])["date"])
        assert self.X.shape == (self.W + len(self.d), len(self.names)), (self.X.shape, self.W, len(self.d))
        assert len(self.y) == self.X.shape[0] and len(self.b) == len(self.d) and self.d.is_unique

    def origins(self) -> np.ndarray:
        """OOS rows of every origin: dated FIRST_DAY .. LAST_DAY (the 13-bar cache: the main trees' 33,970)."""
        day = self.d.normalize()
        i = np.flatnonzero(np.asarray((day >= pd.Timestamp(FIRST_DAY)) & (day <= pd.Timestamp(LAST_DAY))))
        if self.sample == "rth13" and self.W == WINDOW["rth13"]:
            assert len(i) == N_RTH13, len(i)
        assert len(i), "no origin"
        return i


def model(est: str, threads: int):
    """pooled_trees_rerun_13.model_factory; LightGBM with deterministic=True and force_col_wise=True added."""
    cfg, make = T.model_factory(est, threads)
    if est != "lgbm":
        return cfg, make
    import lightgbm as lgb

    params = dict(cfg.get("params") or {})
    return cfg, lambda: lgb.LGBMRegressor(
        **params, num_threads=threads, random_state=42, verbosity=-1, deterministic=True, force_col_wise=True
    )


def select(a, D: Design) -> tuple[np.ndarray, str]:
    """The OOS rows this call refits and its piece tag."""
    o = D.origins()
    if a.ends:
        return np.concatenate([o[: a.ends], o[-a.ends :]]), a.tag or f"ends{a.ends}"
    if a.first:
        return o[: a.first], a.tag or f"first{a.first}"
    i, n = (int(v) for v in a.part.split("/"))
    assert 1 <= i <= n <= len(o), (a.part, len(o))
    return np.array_split(o, n)[i - 1], a.tag or f"{i}of{n}"  # contiguous blocks of origins


def piece_path(out: Path, label: str, est: str, s: str, tag: str) -> Path:
    return out / "pieces" / f"{label}_{est}_{s}_{tag}.parquet"


# ============================================================================ walk
def cmd_walk(a) -> int:
    D = Design(Path(a.design), a.set)
    est = a.est
    rows, tag = select(a, D)
    out = Path(a.out)
    path = piece_path(out, D.label, est, a.set, tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta_p, ckpt_p = path.with_suffix(".json"), path.with_suffix(".ckpt.json")
    stamps = D.d[rows]
    done = pd.read_parquet(path) if path.is_file() else pd.DataFrame(columns=PIECE_COLS)
    have = set(pd.DatetimeIndex(done["date"]).tolist())
    if meta_p.is_file() and len(done) == len(rows):
        say(f"complete: {path.name}")
        return 0
    W = D.W
    lo, hi = int(rows.min()), W + int(rows.max()) + 1  # design rows [lo, hi): every window [r - W, r) and row r
    X = np.ascontiguousarray(D.X[lo:hi], dtype=np.float64)
    cfg, make = model(est, a.threads)
    rv = T.rv_bars()
    prior: dict = json.loads(ckpt_p.read_text()) if ckpt_p.is_file() else {"wall_sec": 0.0, "refits": 0}
    cpus: list[str] = [str(x) for x in prior.get("cpus", [])] + [T.cpu_model()]
    if est in T.ONE_CPU and len({T.cpu_family(x) for x in cpus}) > 1:
        raise SystemExit(f"{est}: piece {path.name} was begun on {cpus[:-1]}, this node is {cpus[-1]}")
    say(
        f"walk {D.label} {est} {a.set} {tag}: {len(rows)} refits {stamps[0]} .. {stamps[-1]}, {len(have)} on disk; "
        f"{len(D.names)} columns, W {W}; {cfg['name']} {cfg['sha']}, threads {a.threads}"
        f"{', LightGBM deterministic + force_col_wise' if est == 'lgbm' else ''}; {cpus[-1]}"
    )
    recs = [done] if len(done) else []
    t0, t_ck, k = time.time(), time.time(), 0
    skx_dispatch: list[str] = []

    def save(final: bool) -> dict:
        f = pd.concat(recs, ignore_index=True)[PIECE_COLS].sort_values("date", ignore_index=True)
        f.to_parquet(path.with_suffix(".part"), index=False)
        os.replace(path.with_suffix(".part"), path)
        st = {"wall_sec": prior["wall_sec"] + time.time() - t0, "refits": prior["refits"] + k, "cpus": cpus}
        if not final:
            ckpt_p.write_text(json.dumps(st))
        return st

    for g, t in zip(rows, stamps):
        if t in have:
            continue
        r = W + int(g)  # design row of the origin
        yh = T.fit_predict(make, est, X[r - W - lo : r - lo], D.y[r - W : r], X[r - lo : r - lo + 1])
        b = float(D.b[g])
        recs.append(
            pd.DataFrame(
                {
                    "model": est,
                    "predictor_set": a.set,
                    "date": [t],
                    "pred_adj": yh,
                    "true_adj": [float(D.y[r])],
                    "B": [b],
                    "plain": yh**2 * b,
                    "rv_actual": [float(rv.get(t, np.nan))],
                    "fit_row": [r - W],
                }
            )
        )
        k += 1
        if est == "rf_skx" and k == 1:  # the first fit ran on oneDAL, not stock scikit-learn (as the main trees)
            assert any("fit: running accelerated version" in m for m in T._SkxLog.msgs), T._SkxLog.msgs[:5]
            assert not any("fallback" in m.lower() for m in T._SkxLog.msgs), T._SkxLog.msgs[:5]
            skx_dispatch = sorted(set(T._SkxLog.msgs))
            logging.getLogger("sklearnex").setLevel(logging.WARNING)
        if time.time() - t_ck > a.ckpt_sec:
            save(False)
            t_ck = time.time()
            say(f"  {k} refits, {(time.time() - t0) / k:.2f} s a refit, through {t}")
    st = save(True)
    f = pd.read_parquet(path)
    assert pd.DatetimeIndex(f["date"]).equals(stamps), "piece rows"
    meta: dict[str, Any] = {
        "study": "results/all48_2026-10-30/RULE.txt Part 2 (exploratory)",
        "est": est,
        "predictor_set": a.set,
        "tag": tag,
        "label": D.label,
        "sample": D.sample,
        "design": str(D.dir.resolve()),
        "design_shape": list(D.X.shape),
        "design_config": D.config,
        "window_rows": W,
        "block": [str(stamps[0]), str(stamps[-1])],
        "rows": len(rows),
        "refits": int(st["refits"]),
        "wall_sec": round(st["wall_sec"], 1),
        "config": cfg["name"],
        "sha": cfg["sha"],
        "params": cfg.get("params"),
        "config_file_sha256": T.sha256(T.CONFIGS[est]),
        "threads": a.threads,
        "host": platform.node(),
        "cpu": T.cpu_model(),
        "cpus": sorted(set(cpus)),
        "versions": T.versions(),
        "job": os.environ.get("SLURM_JOB_ID", os.environ.get("JOB_ID", "")),
    }
    if est == "lgbm":
        meta["lgbm_deterministic"] = True
        meta["lgbm_force_col_wise"] = True
    if est == "rf_skx":
        meta["skx_dispatch"] = skx_dispatch
        meta["skx_split"] = {k_: v for k_, v in make().get_params().items() if k_ in ("max_bins", "min_bin_size")}
    meta_p.write_text(json.dumps(meta, indent=1, default=str))
    ckpt_p.unlink(missing_ok=True)
    say(f"DONE {path.name}: {k} refits in {time.time() - t0:.1f} s ({meta['wall_sec']} s in all)")
    return 0


# ============================================================================ check
def cmd_check(a) -> int:
    rec: dict[str, Any] = {"design": str(Path(a.design).resolve()), "fail": []}
    for s in SETS:
        D = Design(Path(a.design), s)
        cf = D.config
        flags = cf.get("ex_flags") or {}
        r = dict(
            sample=D.sample,
            window=D.W,
            n_columns=len(D.names),
            rows=int(D.X.shape[0]),
            origins=len(D.origins()),
            first_origin=str(D.d[D.origins()[0]]),
            last_origin=str(D.d[D.origins()[-1]]),
            ex_flags=flags,
        )
        if D.sample != "48" or D.W != WINDOW["48"]:
            rec["fail"].append(f"{s}: sample {D.sample} window {D.W}")
        want = {
            "DROP_OVERNIGHT_INDICATOR": False,
            "FOMC_SCHEDULE_2020_FIX": True,
            "FOMC_PROXIMITY_NO_WINSOR": True,
            "DROP_HOLIDAY_SESSIONS": False,
            "LATE_START_WARMUP": True,
        }
        if any(flags.get(k_) != v for k_, v in want.items()):
            rec["fail"].append(f"{s}: flags {flags}")
        if a.cellb:
            C = Design(Path(a.cellb), s)
            i = D.d.get_indexer(C.d)
            ok = i >= 0
            r["names_cellb_plus_is_overnight"] = [n for n in D.names if n != "is_overnight"] == C.names
            r["y_B_equal_common"] = bool(
                np.array_equal(D.y[D.W + i[ok]], C.y[C.W + np.flatnonzero(ok)], equal_nan=True)
                and np.array_equal(D.b[i[ok]], C.b[np.flatnonzero(ok)], equal_nan=True)
            )
            r["cellb_origins_missing"] = int((~ok[C.origins()]).sum())
            if not (r["names_cellb_plus_is_overnight"] and r["y_B_equal_common"] and not r["cellb_origins_missing"]):
                rec["fail"].append(f"{s}: against cell B {r}")
        rec[s] = r
    rec["pass"] = not rec["fail"]
    txt = json.dumps(rec, indent=1, default=str)
    if a.json:
        Path(a.json).write_text(txt)
    print(txt)
    return 0 if rec["pass"] else 3


# ============================================================================ tasks
def piece_sizes(n_orig: int, sec: float, target_h: float) -> list[int]:
    """Contiguous blocks of origins (np.array_split), none above target_h hours at sec seconds a refit."""
    k_max = max(1, math.floor(target_h * 3600.0 / sec))
    n = max(1, math.ceil(n_orig / k_max))
    return [len(x) for x in np.array_split(np.arange(n_orig), n)]


def gate_sec(g0: Path) -> dict[tuple[str, str], float]:
    """Seconds a refit of the reference gate's T2 pieces (8 threads, EPYC), where present."""
    out = {}
    for est, s in ORDER:
        p = piece_path(g0, "a48", est, s, "ends10").with_suffix(".json")
        if p.is_file():
            m = json.loads(p.read_text())
            if int(m["threads"]) == 8 and set(m["cpus"]) <= REF_CPUS and m["refits"] > 0:
                out[(est, s)] = float(m["wall_sec"]) / int(m["refits"])
    return out


def cmd_tasks(a) -> int:
    D = Design(Path(a.design), "baseline")
    n_orig = len(D.origins())
    sec = dict(SEC_REFIT)
    src = {k: "train3way ALL48 piece metas (EPYC 7513, 8 threads, median)" for k in sec}
    if a.from_gate:
        for k, v in gate_sec(Path(a.from_gate)).items():
            sec[k], src[k] = v, f"gate T2 pieces of {a.from_gate}"
    lines, proj = [], []
    for est, s in ORDER:
        sizes = piece_sizes(n_orig, sec[(est, s)], a.target_hours)
        n = len(sizes)
        lines += [f"{est} {s} {i}/{n} {sz * sec[(est, s)] / 3600.0:.3f}" for i, sz in enumerate(sizes, 1)]
        wall = n_orig * sec[(est, s)] / 3600.0
        proj.append(
            dict(
                est=est,
                set=s,
                origins=n_orig,
                sec_refit=sec[(est, s)],
                source=src[(est, s)],
                pieces=n,
                max_refits_a_piece=max(sizes),
                max_hours_a_piece=max(sizes) * sec[(est, s)] / 3600.0,
                cpu_hours_8threads=wall * 8,
            )
        )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("".join(ln + "\n" for ln in lines))
    P = pd.DataFrame(proj)
    P.to_csv(a.out + ".projection.csv", index=False)
    print(P.drop(columns=["source"]).to_string(index=False))
    say(f"{len(lines)} lines -> {a.out}; {P['cpu_hours_8threads'].sum():,.0f} CPU-hours at the table's speed")
    return 0


# ============================================================================ gate
def template(est: str) -> pd.DataFrame:
    f = pd.read_parquet(TEMPLATE[est], columns=["model", "predictor_set", "date", "pred_adj", "true_adj", "B"])
    return f[f["model"] == est]


def piece(p: Path) -> tuple[pd.DataFrame, dict]:
    assert p.is_file() and p.with_suffix(".json").is_file() and not p.with_suffix(".ckpt.json").exists(), p.name
    return pd.read_parquet(p), json.loads(p.with_suffix(".json").read_text())


def cmd_gate(a) -> int:
    G = Path(a.out)
    ref = Path(a.ref) if a.ref else None
    ests = [e for e in a.est.split(",") if e] or list(EST)
    rec: dict[str, Any] = {"dir": str(G), "ref": str(ref) if ref else None, "classes": {}, "fail": []}
    md5_seen: dict[str, dict] = {}
    for est in ests:
        tf = template(est)
        res: list[dict] = []
        cpus: set[str] = set()
        ok = True
        for s in SETS:
            # T1: the cell-B 13-bar cache, the first and last 10 RTH origins, against the main forecasts
            try:
                f, m = piece(piece_path(G, "r13", est, s, "ends10"))
            except AssertionError as e:
                res.append(dict(gate="T1", set=s, missing=str(e)))
                ok = False
                continue
            cpus |= set(m["cpus"])
            if m["design"] not in md5_seen:
                dz = Path(m["design"])
                md5_seen[m["design"]] = {fn: md5(dz / fn) for fn in CELLB_MD5[s]}
            md5_ok = md5_seen[m["design"]] == CELLB_MD5[s]
            t = tf[tf["predictor_set"] == s].set_index("date").reindex(pd.DatetimeIndex(f["date"]))
            x, y = f["pred_adj"].to_numpy(float), t["pred_adj"].to_numpy(float)
            text = [repr(float(u)) == repr(float(v)) for u, v in zip(x, y)]
            relmax = float(np.max(np.abs(x - y) / np.abs(y)))
            side = bool(np.array_equal(f["true_adj"].to_numpy(), t["true_adj"].to_numpy())) and bool(
                np.array_equal(f["B"].to_numpy(), t["B"].to_numpy())
            )
            good = (
                md5_ok
                and side
                and len(f) == 20
                and m["window_rows"] == WINDOW["rth13"]
                and (all(text) if est != "rf_skx" else relmax <= RF_TOL)
            )
            ok &= good
            res.append(
                dict(
                    gate="T1",
                    set=s,
                    rows=len(f),
                    design_md5_cellb=md5_ok,
                    text_identical=int(sum(text)),
                    max_rel=relmax,
                    side_equal=side,
                    cpus=m["cpus"],
                    sec_refit=m["wall_sec"] / max(1, m["refits"]),
                    pass_=bool(good),
                )
            )
            # T2: the 48-interval cache, the first and last 10 origins, against the reference model's pieces
            try:
                f2, m2 = piece(piece_path(G, "a48", est, s, "ends10"))
            except AssertionError as e:
                res.append(dict(gate="T2", set=s, missing=str(e)))
                ok = False
                continue
            cpus |= set(m2["cpus"])
            good2 = m2["window_rows"] == WINDOW["48"] and len(f2) == 20 and np.isfinite(f2["pred_adj"]).all()
            r2: dict[str, Any] = dict(gate="T2", set=s, rows=len(f2), sec_refit=m2["wall_sec"] / max(1, m2["refits"]))
            if ref is None:
                r2["reference"] = True
                good2 &= set(m2["cpus"]) <= REF_CPUS
            else:
                f0, m0 = piece(piece_path(ref, "a48", est, s, "ends10"))
                same_rows = bool(pd.DatetimeIndex(f0["date"]).equals(pd.DatetimeIndex(f2["date"])))
                x0, x2 = f0["pred_adj"].to_numpy(float), f2["pred_adj"].to_numpy(float)
                side2 = bool(np.array_equal(f0["true_adj"].to_numpy(), f2["true_adj"].to_numpy())) and bool(
                    np.array_equal(f0["B"].to_numpy(), f2["B"].to_numpy())
                )
                rel2 = float(np.max(np.abs(x2 - x0) / np.abs(x0))) if same_rows else float("inf")
                bit = same_rows and bool(np.array_equal(x0, x2))
                good2 &= same_rows and side2 and (bit if est != "rf_skx" else rel2 <= RF_TOL)
                r2 |= dict(ref_cpus=m0["cpus"], bit_identical=bit, max_rel=rel2, side_equal=side2)
            r2["cpus"] = m2["cpus"]
            r2["pass_"] = bool(good2)
            ok &= bool(good2)
            res.append(r2)
        one = len(cpus) == 1
        ok &= one
        rec["classes"][est] = dict(pass_=bool(ok), cpus=sorted(cpus), checks=res)
        if ok:
            (G / f"passed_{est}").write_text(sorted(cpus)[0] + "\n")
        else:
            rec["fail"].append(est)
    rec["pass"] = not rec["fail"]
    (G / f"gate_{'_'.join(ests)}.json").write_text(json.dumps(rec, indent=1, default=str))
    say(f"gate {G.name}: {'PASS' if rec['pass'] else 'FAIL ' + str(rec['fail'])}")
    return 0 if rec["pass"] else 3


# ============================================================================ merge
def cluster_of(host: str) -> str:
    """CARC compute nodes are <name>.hpc.usc.edu; any other host is Hoffman2's (train3way's rule)."""
    return "carc" if host.endswith("hpc.usc.edu") else "h2"


def gated_models(gates: list[Path]) -> dict[str, set[tuple[str, str]]]:
    """(cluster, CPU model) of every class's passed gate: passed_<est> holds the model, the gate's T2 piece meta the
    host (so a model gated on one cluster's environment does not admit pieces from the other's)."""
    out: dict[str, set[tuple[str, str]]] = {e: set() for e in EST}
    for g in gates:
        for e in EST:
            p = g / f"passed_{e}"
            if not p.is_file():
                continue
            m = json.loads(piece_path(g, "a48", e, "baseline", "ends10").with_suffix(".json").read_text())
            cl = cluster_of(str(m["host"]))
            out[e] |= {(cl, ln.strip()) for ln in p.read_text().splitlines() if ln.strip()}
    return out


def cmd_merge(a) -> int:
    out = Path(a.out)
    gated = gated_models([Path(x) for x in a.gates.split(",") if x])
    frames, used, missing = [], {}, []
    for est in a.est or list(EST):
        for s in a.set or list(SETS):
            D = Design(Path(a.design), s)
            want = D.d[D.origins()]
            pat = re.compile(rf"^{D.label}_{est}_{s}_(\d+)of(\d+)\.parquet$")
            by_n: dict[int, dict[int, Path]] = {}
            for p in sorted((out / "pieces").glob(f"{D.label}_{est}_{s}_*of*.parquet")):
                mm = pat.match(p.name)
                if mm and p.with_suffix(".json").is_file() and not p.with_suffix(".ckpt.json").exists():
                    by_n.setdefault(int(mm.group(2)), {})[int(mm.group(1))] = p
            full = {n: ps for n, ps in by_n.items() if set(ps) == set(range(1, n + 1))}
            if not full:
                say(f"{est} {s}: no complete piece set ({ {n: len(ps) for n, ps in by_n.items()} } of n)")
                missing.append(f"{est}/{s}")
                continue
            assert len(full) == 1, f"{est} {s}: complete splits {sorted(full)}"
            n, ps = next(iter(full.items()))
            parts, cms = [], set()
            for i in range(1, n + 1):
                f = pd.read_parquet(ps[i], columns=T.COLS)
                m = json.loads(ps[i].with_suffix(".json").read_text())
                assert (m["est"], m["predictor_set"], m["label"], m["window_rows"]) == (est, s, D.label, D.W), ps[i]
                if est == "lgbm":
                    assert m.get("lgbm_deterministic") and m.get("lgbm_force_col_wise"), ps[i].name
                pc = set(m["cpus"])
                cl = cluster_of(str(m["host"]))
                bad = sorted(c for c in pc if (cl, c) not in gated[est])
                assert not bad, f"{ps[i].name}: CPU models {bad} not gated for {est} on {cl}"
                if est in T.ONE_CPU:
                    assert len({T.cpu_family(x) for x in pc}) == 1, f"{ps[i].name}: CPU families {sorted(pc)}"
                cms |= pc
                parts.append(f)
            f = pd.concat(parts, ignore_index=True).sort_values("date", ignore_index=True)
            assert f["date"].is_unique and pd.DatetimeIndex(f["date"]).equals(want), f"{est} {s}: rows"
            num = f[["pred_adj", "true_adj", "B", "plain"]].to_numpy(float)
            assert np.isfinite(num).all(), f"{est} {s}: non-finite values"
            assert np.allclose(f["plain"], f["pred_adj"] ** 2 * f["B"], rtol=1e-12, atol=0)
            assert np.array_equal(f["true_adj"].to_numpy(), D.y[D.W + D.origins()]) and np.array_equal(
                f["B"].to_numpy(), D.b[D.origins()]
            ), f"{est} {s}: true_adj / B"
            frames.append(f)
            used[f"{est}/{s}"] = dict(pieces=n, rows=len(f), cpu_models=sorted(cms))
            say(f"{est} {s}: {len(f)} rows from {n} pieces, CPU models {sorted(cms)}")
    if frames:
        fc = pd.concat(frames, ignore_index=True)
        fc["date"] = pd.to_datetime(fc["date"]).astype("datetime64[ns]")
        fc.to_parquet(out / "forecasts.parquet", index=False)
        (out / "forecasts.json").write_text(
            json.dumps(
                dict(merged=used, missing=missing, gated={e: sorted(map(list, v)) for e, v in gated.items()}), indent=1
            )
        )
        say(f"merged {len(fc)} rows -> {out / 'forecasts.parquet'}; missing {missing}")
    if a.require_all and missing:
        return 1
    return 0 if frames else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("check", "walk", "tasks", "gate", "merge"))
    ap.add_argument("--design", default="", help="a run dir holding _work/design_<set>/")
    ap.add_argument("--cellb", default="", help="check: the cell-B 13-bar run dir")
    ap.add_argument("--json", default="", help="check: also write the record here")
    ap.add_argument("--out", default="")
    ap.add_argument("--est", default="", help="walk: one class; gate: a comma-separated list")
    ap.add_argument("--set", default="")
    ap.add_argument("--part", default="1/1")
    ap.add_argument("--ends", type=int, default=0)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    ap.add_argument("--ckpt-sec", type=float, default=300.0)
    ap.add_argument("--target-hours", type=float, default=1.0)
    ap.add_argument("--from-gate", default="")
    ap.add_argument("--ref", default="", help="gate: the reference model's gate dir (T2)")
    ap.add_argument("--gates", default="", help="merge: the gate dirs (passed_<est>)")
    ap.add_argument("--require-all", action="store_true")
    a = ap.parse_args(argv)
    if a.mode == "walk":
        assert a.est in EST and a.set in SETS and a.design and a.out, "walk: --design, --est, --set, --out"
    if a.mode == "merge":
        a.est = [e for e in a.est.split(",") if e]
        a.set = [s for s in a.set.split(",") if s]
    return {"check": cmd_check, "walk": cmd_walk, "tasks": cmd_tasks, "gate": cmd_gate, "merge": cmd_merge}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
