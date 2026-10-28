"""The trees' factor test beyond HAR + realized moments + FOMC (results/beyond_trees_2026-10-27/RULE.txt sections 0, 1
and 3; its CARC addendum).

Arms (LightGBM and XGBoost, frozen configurations, refit at every one of Table 1's 33,970 bars on the W = 39,000 rows
before it, the FOMC-patched design D13):
  base     the ablation's add_rm_fomc arm (HAR set + realized moments + FOMC, 132 columns, experiments/ablate_arms.py
           order); only a gate here -- the BASE forecasts are the ablation's (results/ablations_2014start_w3000/
           forecasts/trees_wide_rmfomc.parquet) and are not rerun
  pca<k>   base + the first k principal components of the remaining block, k in {1, 2, 3, 5, 10}
  pls<k>   base + the first k partial-least-squares components of the remaining block against the target, k in {1, 3}
Remaining block: the value averages adj_<s>_ma_<k> (every horizon) of the series of the groups ew, vw, liq, iv, sent, vd
(216 columns: 36 series x 6 horizons); no availability averages.

Map (once a trading day, from the window ending at the bar before the day's first bar, held for the day's bars): the
block's window means m and standard deviations s (ddof 0); active columns: s > 1e-12 and at least 250 rows off the
window mode in the series' k = 1 column (the linear factor test's entry rule, experiments/factor_har_2026_10_26.py,
pool_linear_pretune.col_mode); the eigenvectors V of the active columns' correlation matrix, descending eigenvalues,
each signed to a non-negative sum (factor_har's convention); the SIMPLS weights R of the standardized active block
against the window's target (factor_har.simpls).  Scores of a row z: f = (z - m) P with P = diag(1/s) [V_10 | R_3] on
the active rows (zero elsewhere): the PCA scores have the window variances of the components, the PLS scores unit
window variance.  Every training row of a bar's window and the bar's own row take the day's map.

The maps are computed once (mode maps, one process) and stored; the walks read them, so every arm and piece uses the
same loadings.  The walk computes the scores column by column with elementwise numpy operations in a fixed order
(no BLAS reduction), so they are the same bits on every CPU model.  Model code: experiments/ablate_trees.py model()
(pooled_trees_rerun_13.model_factory; LightGBM deterministic + force_col_wise) and pooled_trees_rerun_13.fit_predict.

usage: python experiments/beyond_factor_trees.py <mode> [options]
  maps     --full D --out O [--check-days N]   O/maps.npz + O/maps.json; the direct-materialization check (gate GM) on
                                               N spread days -> O/gate_maps.json
  walk     --est E[,E] --arm A[,A] (--part i/n | --gate) --full D --maps M --out O [--threads T]
           -> O/pieces/<arm>_<est>_<tag>.parquet + .json (checkpoint every --ckpt-sec s; resumable)
  gate1    --out O                             BASE reproduction (base pieces vs the ablation's add_rm_fomc, text-
                                               identical) -> O/gate1.json, O/passed_<est>
  gate2    --gates O1,O2,.. --out J            T2: every arm's gate pieces bit-identical across the gate dirs
  project  --gates O1,.. --out J               CPU-hours of the 14 arms at the slowest gated model's speed
  tasks    --project J --out F [--target-hours H]   task lines "<est> <arm> <i>/<n> <hours>", costliest first
  merge    --out O [--gate2 J]                 every complete piece set -> O/forecasts_wide.parquet (date, <est>|<arm>)
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

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(HERE), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ablate_arms as AA  # noqa: E402
import ablate_trees as AT  # noqa: E402

T = AT.T  # experiments/pooled_trees_rerun_13.py
W = AT.W  # 39,000
BASE_ARM = "add_rm_fomc"
REMAIN = ("ew", "vw", "liq", "iv", "sent", "vd")
N_BLOCK = 216
NPC, NPLS = 10, 3
K_PCA = (1, 2, 3, 5, 10)
K_PLS = (1, 3)
ARMS = ["base"] + [f"pca{k}" for k in K_PCA] + [f"pls{k}" for k in K_PLS]
FACTOR_ARMS = ARMS[1:]
EST = ("lgbm", "xgb")
STD_MIN, STD_MIN_EVIDENCE = 1e-12, 250
BASE_FC = ROOT / "results" / "ablations_2014start_w3000" / "forecasts" / "trees_wide_rmfomc.parquet"
FULL_FC = ROOT / "results" / "pooled_trees_final_2014start_fomcfix_w3000" / "forecasts.parquet"
# the gate origins (RULE addendum): six bars spread over the span, each on its own day
GATE_ORIGINS = (
    "2014-01-02 10:00",
    "2016-08-03 12:30",
    "2018-12-20 15:00",
    "2020-03-16 11:00",
    "2022-06-15 14:30",
    "2024-04-30 16:00",
)
PIECE_COLS = AT.PIECE_COLS


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def md5_file(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def block_names(names: list[str]) -> list[str]:
    """The remaining block: value averages of the six remaining groups' series, in the cache's order."""
    gs = AA.group_series()
    ser = {s for g in REMAIN for s in gs[g]}
    out = []
    for n in names:
        m = re.match(r"^adj_(.+)_ma_(\d+)$", n)
        if m and not n.startswith("har_ma_") and m.group(1) in ser:
            out.append(n)
    assert len(out) == N_BLOCK and len(ser) == 36, (len(out), len(ser))
    return out


def arm_sel(arm: str) -> np.ndarray:
    """Score columns of an arm (indices into the 13 stored scores: PCA 0..9, PLS 10..12)."""
    if arm == "base":
        return np.arange(0)
    k = int(arm[3:])
    if arm.startswith("pca"):
        assert k in K_PCA, arm
        return np.arange(k)
    assert arm.startswith("pls") and k in K_PLS, arm
    return NPC + np.arange(k)


def simpls(C: np.ndarray, s: np.ndarray, K: int) -> np.ndarray:
    """factor_har_2026_10_26.simpls (SIMPLS weights, de Jong 1993), unchanged."""
    p = len(s)
    R = np.zeros((p, K))
    V = np.zeros((p, K))
    s = s.copy()
    for a in range(K):
        r = s.copy()
        r /= np.sqrt(float(r @ C @ r))
        pl = C @ r
        v = pl - V[:, :a] @ (V[:, :a].T @ pl)
        v /= np.linalg.norm(v)
        s = s - v * float(v @ s)
        R[:, a], V[:, a] = r, v
    return R


def mode_count(A: np.ndarray) -> np.ndarray:
    """The count of each column's modal value (pool_linear_pretune.col_mode's second output)."""
    n, p = A.shape
    S = np.sort(A, axis=0)
    new = np.empty((n, p), dtype=bool)
    new[0] = True
    np.not_equal(S[1:], S[:-1], out=new[1:])
    idx = np.arange(n)[:, None]
    run = idx - np.maximum.accumulate(np.where(new, idx, 0), axis=0) + 1
    return run.max(axis=0)


# ============================================================================ maps
def day_firsts(F: AT.Full) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Every forecast day and the OOS row of its first bar."""
    o = F.origins()
    day = F.d[o].normalize()
    first = pd.Series(o, index=day).groupby(level=0).min()
    return pd.DatetimeIndex(first.index), first.to_numpy()


def compute_map(Zw: np.ndarray, yw: np.ndarray, k1: np.ndarray, col_ser: np.ndarray) -> dict:
    m = Zw.mean(axis=0)
    sd = Zw.std(axis=0)
    off = W - mode_count(Zw[:, k1])
    act = (sd > STD_MIN) & (off[col_ser] >= STD_MIN_EVIDENCE)
    a = np.flatnonzero(act)
    Zs = (Zw[:, a] - m[a]) / sd[a]
    C = Zs.T @ Zs / W
    C = 0.5 * (C + C.T)
    lam, V = np.linalg.eigh(C)
    lam, V = lam[::-1], V[:, ::-1]
    V = V[:, :NPC] * np.where(V[:, :NPC].sum(axis=0) < 0, -1.0, 1.0)
    s = Zs.T @ (yw - yw.mean()) / W
    R = simpls(C, s, NPLS)
    P = np.zeros((Zw.shape[1], NPC + NPLS))
    P[a, :NPC] = V / sd[a][:, None]
    P[a, NPC:] = R / sd[a][:, None]
    return dict(m=m, P=P, act=act, lam=lam[:NPC].copy(), trace=float(lam.sum()), n_active=len(a))


def scores(Z: np.ndarray, m: np.ndarray, P: np.ndarray, act: np.ndarray, sel: np.ndarray) -> np.ndarray:
    """f = (z - m) P[:, sel], accumulated column by column with elementwise operations in a fixed order (the same bits
    on every CPU model; no BLAS reduction)."""
    f = np.zeros((Z.shape[0], len(sel)))
    if len(sel) == 0:
        return f
    Ps = P[:, sel]
    for j in np.flatnonzero(act):
        f += (Z[:, j] - m[j])[:, None] * Ps[j][None, :]
    return f


class Maps:
    def __init__(self, p: Path):
        z = np.load(p)
        self.path, self.md5 = p, md5_file(p)
        self.days = pd.DatetimeIndex(pd.to_datetime(z["days"].astype(str)))
        self.first = z["first_row"]
        self.m, self.P, self.act = z["m"], z["P"], z["act"]
        self.ix = {d: i for i, d in enumerate(self.days)}

    def get(self, day: pd.Timestamp) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        i = self.ix[day]
        return self.m[i], self.P[i], self.act[i], int(self.first[i])


def cmd_maps(a) -> int:
    t0 = time.time()
    F = AT.Full(Path(a.full))
    Z = block_names(F.names)
    zi = [F.names.index(n) for n in Z]
    zser = [re.match(r"^adj_(.+)_ma_\d+$", n).group(1) for n in Z]  # type: ignore[union-attr]
    ser_list = list(dict.fromkeys(zser))
    k1 = np.array([Z.index(f"adj_{s}_ma_1") for s in ser_list])
    col_ser = np.array([ser_list.index(s) for s in zser])
    days, first = day_firsts(F)
    D = len(days)
    M = np.zeros((D, N_BLOCK))
    P = np.zeros((D, N_BLOCK, NPC + NPLS))
    ACT = np.zeros((D, N_BLOCK), dtype=bool)
    LAM = np.zeros((D, NPC))
    NA = np.zeros(D, dtype=np.int32)
    TR = np.zeros(D)
    say(f"maps: {D} days {days[0].date()} .. {days[-1].date()}; block {N_BLOCK} columns, {len(ser_list)} series")
    for i in range(D):
        t = W + int(first[i])  # design row of the day's first bar
        Zw = np.asarray(F.X[t - W : t][:, zi], dtype=np.float64)
        mp = compute_map(Zw, np.asarray(F.y[t - W : t], dtype=np.float64), k1, col_ser)
        M[i], P[i], ACT[i], LAM[i], NA[i], TR[i] = mp["m"], mp["P"], mp["act"], mp["lam"], mp["n_active"], mp["trace"]
        if i % 250 == 0:
            say(
                f"  day {i} / {D} ({days[i].date()}): {mp['n_active']} active, lam1/trace {mp['lam'][0] / mp['trace']:.3f}"
            )
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / "maps.npz",
        days=np.array([str(d.date()) for d in days]),
        first_row=first,
        m=M,
        P=P,
        act=ACT,
        lam=LAM,
        n_active=NA,
        trace=TR,
    )
    info = dict(
        design=str(F.dir),
        x_md5=md5_file(F.dir / "X.npy"),
        days=D,
        block=Z,
        series=ser_list,
        n_active_min=int(NA.min()),
        n_active_max=int(NA.max()),
        share_pc1_median=float(np.median(LAM[:, 0] / TR)),
        maps_md5=md5_file(out / "maps.npz"),
        wall_sec=round(time.time() - t0, 1),
        host=platform.node(),
        cpu=T.cpu_model(),
        numpy=np.__version__,
    )
    (out / "maps.json").write_text(json.dumps(info, indent=1))
    say(f"maps done: {json.dumps({k: v for k, v in info.items() if k not in ('block', 'series')})}")
    if a.check_days:
        return gate_maps(F, Maps(out / "maps.npz"), Z, zi, k1, col_ser, a.check_days, out / "gate_maps.json")
    return 0


def gate_maps(F, mp: Maps, Z, zi, k1, col_ser, n_days: int, out: Path) -> int:
    """Gate GM: on n_days spread days, the stored map's scores of the window rows and of the day's first bar against a
    direct computation on the materialized window (np.linalg.svd of the standardized active block for PCA; NIPALS PLS1
    for PLS), both normalized to unit window variance and signed alike: max |difference| <= 1e-8."""
    sel = np.linspace(0, len(mp.days) - 1, n_days).round().astype(int)
    rec: dict = {"days": [], "fail": []}
    for i in sel:
        day = mp.days[i]
        m, P, act, first = mp.get(day)
        t = W + first
        Zr = np.asarray(F.X[t - W : t + 1][:, zi], dtype=np.float64)  # window rows + the day's first bar
        yw = np.asarray(F.y[t - W : t], dtype=np.float64)
        f = scores(Zr, m, P, act, np.arange(NPC + NPLS))
        # direct: the entry rule from scratch (col_mode), SVD, NIPALS
        Zw = Zr[:W]
        mu, sd = Zw.mean(axis=0), Zw.std(axis=0)
        cnt = mode_count(Zw[:, k1])
        ser_ok = (W - cnt) >= STD_MIN_EVIDENCE
        a = np.flatnonzero((sd > STD_MIN) & ser_ok[col_ser])
        Zs = (Zr[:, a] - mu[a]) / sd[a]
        _, sv, Vt = np.linalg.svd(Zs[:W], full_matrices=False)
        Fp = Zs @ Vt[:NPC].T
        Xd, yv = Zs[:W].copy(), yw - yw.mean()
        Wl, Pl = [], []
        for _ in range(NPLS):
            w = Xd.T @ yv
            w /= np.linalg.norm(w)
            tt = Xd @ w
            t2 = float(tt @ tt)
            pl = Xd.T @ tt / t2
            yv = yv - float(yv @ tt) / t2 * tt
            Xd -= np.outer(tt, pl)
            Wl.append(w)
            Pl.append(pl)
        Wm, Pm = np.array(Wl).T, np.array(Pl).T
        Fl = Zs @ (Wm @ np.linalg.inv(Pm.T @ Wm))
        D = np.column_stack([Fp, Fl])

        fa = f - f[:W].mean(axis=0)
        da = D - D[:W].mean(axis=0)
        fa = fa / fa[:W].std(axis=0)
        da = da / da[:W].std(axis=0)
        sign = np.sign((fa[:W] * da[:W]).sum(axis=0))
        da = da * sign
        diff = np.abs(fa - da).max(axis=0)
        gap = sv[NPC - 1] ** 2 / sv[NPC] ** 2
        r = dict(
            day=str(day.date()),
            n_active_map=int(act.sum()),
            n_active_direct=len(a),
            same_active=bool(np.array_equal(np.flatnonzero(act), a)),
            max_abs_diff=[float(x) for x in diff],
            lam10_over_lam11=float(gap),
        )
        ok = r["same_active"] and float(diff.max()) <= 1e-8
        r["pass"] = ok
        rec["days"].append(r)
        if not ok:
            rec["fail"].append(r["day"])
        say(f"GM {r['day']}: active {r['n_active_map']}/{r['n_active_direct']}, max diff {diff.max():.2e}")
    rec["pass"] = not rec["fail"]
    out.write_text(json.dumps(rec, indent=1))
    say(f"gate GM: {'PASS' if rec['pass'] else 'FAIL ' + str(rec['fail'])}")
    return 0 if rec["pass"] else 3


# ============================================================================ walk
def piece_rows(a, F: AT.Full) -> tuple[np.ndarray, str]:
    o = F.origins()
    if a.gate:
        st = pd.DatetimeIndex(pd.to_datetime(list(GATE_ORIGINS)))
        pos = pd.Series(np.arange(len(o)), index=F.d[o])
        assert st.isin(pos.index).all(), [str(x) for x in st[~st.isin(pos.index)]]
        return o[pos.reindex(st).to_numpy()], "gate"
    i, n = (int(v) for v in a.part.split("/"))
    assert 1 <= i <= n, a.part
    day = F.d[o].normalize()
    blocks = np.array_split(day.unique(), n)
    return o[day.isin(blocks[i - 1])], f"{i}of{n}"


def walk_one(F: AT.Full, mp: Maps | None, est: str, arm: str, rows: np.ndarray, tag: str, out: Path, a) -> None:
    sel = arm_sel(arm)
    path = out / "pieces" / f"{arm}_{est}_{tag}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    meta_p, ckpt_p = path.with_suffix(".json"), path.with_suffix(".ckpt.json")
    stamps = F.d[rows]
    done = pd.read_parquet(path) if path.is_file() else pd.DataFrame(columns=PIECE_COLS)
    have = set(pd.DatetimeIndex(done["date"]).tolist())
    if meta_p.is_file() and len(done) == len(rows):
        say(f"complete: {path.name}")
        return
    cols = F.cols(BASE_ARM)
    assert len(cols) == 132, len(cols)
    lo, hi = int(rows.min()), W + int(rows.max()) + 1  # design rows [lo, hi)
    Xb = np.ascontiguousarray(F.X[lo:hi][:, cols], dtype=np.float64)  # as ablate_trees.cmd_walk
    Zb = None
    if len(sel):
        assert mp is not None
        zi = [F.names.index(n) for n in block_names(F.names)]
        Zb = np.asarray(F.X[lo:hi][:, zi], dtype=np.float64)
    cfg, make = AT.model(est, a.threads)
    rv = T.rv_bars()
    prior: dict = json.loads(ckpt_p.read_text()) if ckpt_p.is_file() else {"wall_sec": 0.0, "refits": 0}
    cpus: list[str] = [str(x) for x in prior.get("cpus", [])] + [T.cpu_model()]
    say(
        f"walk {arm} {est} {tag}: {len(rows)} refits {stamps[0]} .. {stamps[-1]}, {len(have)} on disk; "
        f"{len(cols)} + {len(sel)} columns; {cfg['name']} {cfg['sha']}, threads {a.threads}; {cpus[-1]}"
    )
    recs = [done] if len(done) else []
    t0, t_ck, k = time.time(), time.time(), 0
    t_score = 0.0

    def save(final: bool) -> dict:
        f = pd.concat(recs, ignore_index=True)[PIECE_COLS].sort_values("date", ignore_index=True)
        f.to_parquet(path.with_suffix(".part"), index=False)
        os.replace(path.with_suffix(".part"), path)
        st = {"wall_sec": prior["wall_sec"] + time.time() - t0, "refits": prior["refits"] + k, "cpus": cpus}
        if not final:
            ckpt_p.write_text(json.dumps(st))
        return st

    day = stamps.normalize()
    for d in day.unique():
        todo = [(int(g), t) for g, t in zip(rows[day == d], stamps[day == d]) if t not in have]
        if not todo:
            continue
        r_first, r_last = W + todo[0][0], W + todo[-1][0]  # design rows of the day's bars still to fit
        if len(sel):
            ts = time.perf_counter()
            m, P, act, first = mp.get(pd.Timestamp(d))  # type: ignore[union-attr]
            sc = scores(Zb[r_first - W - lo : r_last + 1 - lo], m, P, act, sel)  # type: ignore[index]
            Xd = np.ascontiguousarray(np.hstack([Xb[r_first - W - lo : r_last + 1 - lo], sc]))
            off = r_first - W  # design row of Xd's first row
            t_score += time.perf_counter() - ts
        else:
            Xd, off = Xb, lo
        for g, t in todo:
            r = W + g
            yh = T.fit_predict(make, est, Xd[r - W - off : r - off], F.y[r - W : r], Xd[r - off : r - off + 1])
            b = float(F.b[g])
            recs.append(
                pd.DataFrame(
                    {
                        "model": est,
                        "arm": arm,
                        "date": [t],
                        "pred_adj": yh,
                        "true_adj": [float(F.y[r])],
                        "B": [b],
                        "plain": yh**2 * b,
                        "rv_actual": [float(rv.get(t, np.nan))],
                        "fit_row": [r - W],
                    }
                )
            )
            k += 1
            if time.time() - t_ck > a.ckpt_sec:
                save(False)
                t_ck = time.time()
                say(f"  {k} refits, {(time.time() - t0) / k:.2f} s a refit, through {t}")
    st = save(True)
    f = pd.read_parquet(path)
    assert pd.DatetimeIndex(f["date"]).equals(stamps), "piece rows"
    meta = {
        "arm": arm,
        "est": est,
        "tag": tag,
        "n_columns": len(cols) + len(sel),
        "base_columns_md5": hashlib.md5("\n".join(F.names[j] for j in cols).encode()).hexdigest(),
        "scores": [int(x) for x in sel],
        "maps": None if mp is None else str(mp.path),
        "maps_md5": None if mp is None else mp.md5,
        "design": str(F.dir),
        "window_rows": W,
        "block": [str(stamps[0]), str(stamps[-1])],
        "rows": len(rows),
        "refits": int(st["refits"]),
        "wall_sec": round(st["wall_sec"], 1),
        "score_sec": round(t_score, 1),
        "config": cfg["name"],
        "sha": cfg["sha"],
        "params": cfg.get("params"),
        "threads": a.threads,
        "host": platform.node(),
        "cpu": T.cpu_model(),
        "cpus": sorted(set(cpus)),
        "versions": T.versions(),
        "job": os.environ.get("SLURM_JOB_ID", ""),
        "array_task": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
    }
    if est == "lgbm":
        meta["lgbm_deterministic"] = True
        meta["lgbm_force_col_wise"] = True
    meta_p.write_text(json.dumps(meta, indent=1))
    ckpt_p.unlink(missing_ok=True)
    say(f"DONE {path.name}: {k} refits in {time.time() - t0:.1f} s ({meta['wall_sec']} s in all)")


def cmd_walk(a) -> int:
    F = AT.Full(Path(a.full))
    ests = [e for e in a.est.split(",") if e]
    arms = ARMS if a.arm == "all" else [x for x in a.arm.split(",") if x]
    assert set(ests) <= set(EST) and set(arms) <= set(ARMS), (ests, arms)
    mp = Maps(Path(a.maps)) if any(x != "base" for x in arms) else None
    rows, tag = piece_rows(a, F)
    for est in ests:
        for arm in arms:
            walk_one(F, mp, est, arm, rows, tag, Path(a.out), a)
    return 0


# ============================================================================ gates
def base_template() -> pd.DataFrame:
    return pd.read_parquet(BASE_FC)


def full_template() -> pd.DataFrame:
    f = pd.read_parquet(FULL_FC, columns=["model", "predictor_set", "date", "pred_adj", "true_adj", "B"])
    return f[f["predictor_set"] == "all_features"]


def cmd_gate1(a) -> int:
    """BASE reproduction: the base gate pieces against the ablation's add_rm_fomc forecasts, text-identical (repr), and
    true_adj / B against Table 1's patched full-set rows."""
    out = Path(a.out)
    bt = base_template().set_index("date")
    ft = full_template()
    rec: dict = {"dir": str(out), "classes": {}, "fail": []}
    cpu = None
    for est in EST:
        p = out / "pieces" / f"base_{est}_gate.parquet"
        if not (p.is_file() and p.with_suffix(".json").is_file()):
            rec["classes"][est] = dict(missing=True)
            rec["fail"].append(est)
            continue
        f = pd.read_parquet(p)
        m = json.loads(p.with_suffix(".json").read_text())
        cpu = cpu or m["cpus"][0]
        d = pd.DatetimeIndex(f["date"])
        x = f["pred_adj"].to_numpy()
        y = bt[f"{est}|{BASE_ARM}"].reindex(d).to_numpy()
        text = [repr(float(u)) == repr(float(v)) for u, v in zip(x, y)]
        t = ft[ft["model"] == est].set_index("date").reindex(d)
        side = bool(np.array_equal(f["true_adj"].to_numpy(), t["true_adj"].to_numpy())) and bool(
            np.array_equal(f["B"].to_numpy(), t["B"].to_numpy())
        )
        ok = all(text) and side and len(m["cpus"]) == 1 and len(f) >= 5
        rec["classes"][est] = dict(
            origins=[str(x) for x in d],
            text_identical=int(sum(text)),
            rows=len(f),
            max_abs=float(np.max(np.abs(x - y))),
            side_equal=side,
            cpus=m["cpus"],
            pass_=ok,
        )
        if ok:
            (out / f"passed_{est}").write_text(m["cpus"][0] + "\n")
        else:
            rec["fail"].append(est)
    rec["cpu"] = cpu
    rec["pass"] = not rec["fail"]
    (out / "gate1.json").write_text(json.dumps(rec, indent=1))
    say(f"gate BASE {out.name} [{cpu}]: {'PASS' if rec['pass'] else 'FAIL ' + str(rec['fail'])}")
    return 0 if rec["pass"] else 3


def cmd_gate2(a) -> int:
    """T2: each (est, arm)'s gate piece bit-identical across the gate dirs (CPU models)."""
    dirs = [Path(x) for x in a.gates.split(",") if x]
    rec: dict = {"dirs": [str(d) for d in dirs], "pairs": {}, "fail": []}
    for est in EST:
        for arm in ARMS:
            got = []
            for d in dirs:
                p = d / "pieces" / f"{arm}_{est}_gate.parquet"
                if p.is_file() and p.with_suffix(".json").is_file():
                    m = json.loads(p.with_suffix(".json").read_text())
                    got.append((d.name, m["cpus"], pd.read_parquet(p)["pred_adj"].to_numpy()))
            if len(got) < 2:
                rec["pairs"][f"{est}|{arm}"] = dict(models=[g[1] for g in got], identical=None)
                continue
            ident = all(np.array_equal(got[0][2], g[2]) for g in got[1:])
            mx = max(float(np.max(np.abs(got[0][2] - g[2]))) for g in got[1:])
            rec["pairs"][f"{est}|{arm}"] = dict(models={g[0]: g[1] for g in got}, identical=bool(ident), max_abs=mx)
            if not ident:
                rec["fail"].append(f"{est}|{arm}")
    rec["pass"] = not rec["fail"]
    Path(a.out).write_text(json.dumps(rec, indent=1))
    say(f"gate T2: {'PASS' if rec['pass'] else 'FAIL ' + str(rec['fail'])}")
    return 0 if rec["pass"] else 3


def cmd_project(a) -> int:
    """CPU-hours of each (est, arm) over 33,970 refits at 8 threads, at the slowest gated model's seconds a refit
    (the gate pieces; the base arm's speed for the k = 0 columns, the factor arms' own)."""
    dirs = [Path(x) for x in a.gates.split(",") if x]
    sp: dict = {}
    for d in dirs:
        for est in EST:
            if not (d / f"passed_{est}").is_file():
                continue
            for arm in ARMS:
                p = d / "pieces" / f"{arm}_{est}_gate.json"
                if p.is_file():
                    m = json.loads(p.read_text())
                    sec = (m["wall_sec"] - m.get("score_sec", 0.0)) / max(1, m["refits"])
                    sp.setdefault(f"{est}|{arm}", {})[d.name] = sec
    rows = {}
    for key, per in sp.items():
        worst = max(per.values())
        rows[key] = dict(sec_refit=per, worst=worst, cpu_h=worst * 33970 * 8 / 3600.0 * a.margin)
    tot = sum(r["cpu_h"] for k, r in rows.items() if not k.endswith("|base"))
    rec = dict(margin=a.margin, arms=rows, total_cpu_h=tot, cap=a.cap)
    Path(a.out).write_text(json.dumps(rec, indent=1))
    say(f"projection (factor arms): {tot:,.0f} CPU-hours (cap {a.cap:,.0f})")
    return 0 if tot <= a.cap else 3


def cmd_tasks(a) -> int:
    pr = json.loads(Path(a.project).read_text())["arms"]
    items = []
    for est in EST:
        for arm in FACTOR_ARMS:
            ch = pr[f"{est}|{arm}"]["cpu_h"]
            n = max(1, math.ceil(ch / 8.0 / a.target_hours))
            items.append((ch, est, arm, n))
    lines = []
    for ch, est, arm, n in sorted(items, key=lambda x: -x[0]):
        lines += [f"{est} {arm} {i}/{n} {ch / 8.0 / n:.2f}" for i in range(1, n + 1)]
    Path(a.out).write_text("".join(ln + "\n" for ln in lines))
    say(f"{len(lines)} task lines -> {a.out}")
    return 0


# ============================================================================ merge
def cmd_merge(a) -> int:
    out = Path(a.out)
    g2 = json.loads(Path(a.gate2).read_text()) if a.gate2 else {}
    ft = full_template()
    wide = None
    info: dict = {}
    for est in EST:
        tf = ft[ft["model"] == est].set_index("date")
        for arm in FACTOR_ARMS:
            pat = re.compile(rf"^{arm}_{est}_(\d+)of(\d+)\.parquet$")
            by_n: dict[int, dict[int, Path]] = {}
            for p in sorted((out / "pieces").glob(f"{arm}_{est}_*of*.parquet")):
                mm = pat.match(p.name)
                if mm and p.with_suffix(".json").is_file() and not p.with_suffix(".ckpt.json").exists():
                    by_n.setdefault(int(mm.group(2)), {})[int(mm.group(1))] = p
            full = [n for n, ps in by_n.items() if set(ps) == set(range(1, n + 1))]
            assert len(full) == 1, f"{arm}/{est}: complete piece sets {full}"
            n = full[0]
            fr = []
            cpus: set[str] = set()
            md5s = set()
            for i in range(1, n + 1):
                p = by_n[n][i]
                fr.append(pd.read_parquet(p))
                pm = json.loads(p.with_suffix(".json").read_text())
                assert pm["arm"] == arm and pm["est"] == est and pm["window_rows"] == W
                cpus |= set(pm["cpus"])
                md5s.add(pm["maps_md5"])
            assert len(md5s) == 1, (arm, est, md5s)
            f = pd.concat(fr, ignore_index=True).sort_values("date", ignore_index=True)
            if len(cpus) > 1:
                assert g2.get("pass") is True, f"{arm}/{est}: pieces from {sorted(cpus)} without a passed T2"
            assert pd.DatetimeIndex(f["date"]).equals(pd.DatetimeIndex(tf.index)), f"{arm}/{est}: rows"
            assert np.array_equal(f["true_adj"].to_numpy(), tf["true_adj"].to_numpy()), "true_adj"
            assert np.array_equal(f["B"].to_numpy(), tf["B"].to_numpy()), "B"
            col = f[["date", "pred_adj"]].rename(columns={"pred_adj": f"{est}|{arm}"})
            wide = col if wide is None else wide.merge(col, on="date", how="outer")
            info[f"{est}|{arm}"] = dict(pieces=n, rows=len(f), cpus=sorted(cpus), maps_md5=md5s.pop())
            say(f"merged {est}|{arm}: {n} pieces, {len(f)} rows, {sorted(cpus)}")
    assert wide is not None
    wide.to_parquet(out / "forecasts_wide.parquet", index=False)
    (out / "forecasts_wide.json").write_text(json.dumps(info, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("maps", "walk", "gate1", "gate2", "project", "tasks", "merge"))
    ap.add_argument("--full", default="")
    ap.add_argument("--maps", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--est", default="lgbm,xgb")
    ap.add_argument("--arm", default="all")
    ap.add_argument("--part", default="1/1")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    ap.add_argument("--ckpt-sec", type=float, default=300.0)
    ap.add_argument("--check-days", type=int, default=0)
    ap.add_argument("--gates", default="")
    ap.add_argument("--gate2", default="")
    ap.add_argument("--project", default="")
    ap.add_argument("--target-hours", type=float, default=3.0)
    ap.add_argument("--margin", type=float, default=1.15)
    ap.add_argument("--cap", type=float, default=2500.0)
    a = ap.parse_args()
    return {
        "maps": cmd_maps,
        "walk": cmd_walk,
        "gate1": cmd_gate1,
        "gate2": cmd_gate2,
        "project": cmd_project,
        "tasks": cmd_tasks,
        "merge": cmd_merge,
    }[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
