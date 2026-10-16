"""Local half of the pooled pre-tune study: pull the cluster output, plan arm f, score, SUMMARY.md, export.

The cluster half (the walks, the pre-tunes) is experiments/pool_linear_pretune.py; its docstring holds the
model.  Arms (one change at a time), each pooled model scored at the 16:00 bar:
  a_causal            one penalty, re-chosen every 250 solves on the any-clock validation tail (the spec;
                      Hoffman2: the stored forecasts results/spxw_pnl/yhat_pool_<est>_<bucket>, gated)
  b_causal_bb         the HAR + calendar backbone (34 columns incl. the HAR x session-edge interactions) at
                      r = 1/100 of the exogenous penalty, the same causal re-choice
  f_retune_margin_bb  backbone 1/100; the penalty pre-tuned before 2020-01-03 (4 folds x 12,000 bars tiling
                      the 48,000 bars before the first trade day, each fit on the 96,000 bars before a
                      1,200-bar embargo; criterion = mean any-clock validation MSE on the adjusted scale);
                      from the first re-choice boundary on or after 2020-01-03, every 250 solves the two grid
                      neighbours challenge the incumbent on the spec's validation tail and it switches only
                      when the Diebold-Mariano t of the paired squared-error difference is below
                      -z(0.05 / number of challengers) (the 16:00-bar study's margin rule)
On baseline every column is backbone: b does not exist and f runs with one penalty (r = 1).

A block's forecasts depend only on its alpha and its window (cold seed at each re-choice), so arm f is
the causal record walk (b; a on baseline) where its alpha agrees with it and the scheduled walk elsewhere;
each scheduled run starts one agreeing block early, and that block must reproduce the record walk bit for
bit (stitch gate).

Scoring: close_linear_pretune.Scorer (the master-table GUI's plain scorer: plain = pred_adj^2 x B, no
recalibration, the 866 trade days 2020-01-03 .. 2024-04-30); QLIKE; DM (daily QLIKE) vs sub_ols_baseline and
vs arm a; sign(s) Sharpe mid / crossed; Newey-West t (lag 14) of the daily mid P&L difference vs arm a; days
with another position.  Point estimates only.

Stages:  python experiments/pool_linear_pretune_report.py pull <h2|carc> | plan <h2|carc> | report
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "experiments", REPO / "notebooks", REPO / "writeup"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

OUT = REPO / "results" / "pool_linear_pretune_2026-10-06"
WORK = OUT / "_work"
SPXW = REPO / "results" / "spxw_pnl"
MASTER = REPO / "results" / "close_master_table" / "master_table.csv"
SCHED_DIR = REPO / "cluster" / "pool_pretune_sched"
TASK_DIR = REPO / "cluster"
SSH = "C:/Windows/System32/OpenSSH/ssh.exe"  # native OpenSSH (reaches the Windows ssh-agent)
CLUSTERS = {
    "h2": dict(
        host="hoffman2",
        root="/u/scratch/j/jamesdc1/harxhar-pool-pretune",
        login=True,
        label="Hoffman2 (ridge / elastic net intel-6736p, lasso intel-gold-6140)",
    ),
    "carc": dict(
        host="usc-discovery",
        root="/scratch1/jc_905/harxhar-pool-pretune",
        login=False,
        label="CARC (epyc-7513)",
    ),
}
MOMVIX_FLEET = "/u/scratch/j/jamesdc1/harxhar-pool-momvix/results/pool_momvix/fleet"
BUCKETS = ("all_features", "lean_vol_vix", "baseline")
EST = ("ridge", "reclasso", "reclasticnet")
SHORT = {"ridge": "ridge", "reclasso": "lasso", "reclasticnet": "enet"}
LABEL = {"ridge": "ridge", "reclasso": "lasso", "reclasticnet": "elastic net"}
ARCH = {
    "ridge": "intel-6736p",
    "reclasticnet": "intel-6736p",
    "reclasso": "intel-gold-6140",
}
W, TUNE_PER, K0, K_END = 96_000, 250, 600, 818
FIRST_TRADE_DAY, LAST_TRADE_DAY = "2020-01-03", "2024-04-30"
PRE_START, PRE_HALO = 246_112, 145_200
MARGIN_ALPHA = 0.05
MERGE_GAP = 12  # blocks of agreement (about 7 min of the all_features lasso) cheaper to walk than a new data prep
R_BB = {"all_features": "0.01", "lean_vol_vix": "0.01", "baseline": "1.0"}
REC_TAG = {"all_features": "b", "lean_vol_vix": "b", "baseline": "a"}
F_CHUNK = {"all_features": 28, "lean_vol_vix": 109, "baseline": 218}
ARM_NAME = {"a": "a_causal", "b": "b_causal_bb", "f": "f_retune_margin_bb"}
ARM_LABEL = {
    "a_causal": "one penalty, causal re-choice every 250 solves (spec)",
    "b_causal_bb": "backbone 1/100, causal re-choice every 250 solves",
    "f_retune_margin_bb": "backbone 1/100, pre-tuned + margin retune every 250 solves",
    "f_retune_margin": "one penalty, pre-tuned + margin retune every 250 solves",
}


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def f_name(b: str) -> str:
    return "f_retune_margin" if b == "baseline" else "f_retune_margin_bb"


# ============================================================================ pull
def ssh(cl: str, cmd: str, timeout: int = 600, binary: bool = False):
    c = CLUSTERS[cl]
    remote = f"bash -lc '{cmd}'" if c["login"] else cmd
    r = subprocess.run(
        [SSH, "-o", "BatchMode=yes", c["host"], remote],
        capture_output=True,
        timeout=timeout,
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"ssh {cl} failed ({r.returncode}): {r.stderr.decode(errors='replace')[-800:]}"
        )
    return r.stdout if binary else r.stdout.decode()


def pull(cl: str) -> None:
    """results/pool_pretune (without the full-series results.csv and the pre-tunes' dummy tables) and the
    plp logs; on Hoffman2 also the stored lean_vol_vix pooled arms' regular-hours rows (the other campaign's
    fleet, text-filtered on the cluster)."""
    import io
    import tarfile

    dest = WORK / cl
    dest.mkdir(parents=True, exist_ok=True)
    root = CLUSTERS[cl]["root"]
    cmd = (
        f"cd {root} && tar czf - --exclude=results.csv --exclude=dummy --exclude=*.npy --exclude=dates_oos.csv results/pool_pretune "
        f"$(ls logs/plp_* logs/submitted_pool_pretune* 2>/dev/null)"
    )
    blob = ssh(cl, cmd, timeout=900, binary=True)
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
        tf.extractall(dest, filter="data")
    say(f"pulled {len(blob) / 1e6:.1f} MB from {cl}")
    if cl == "h2":
        for est in EST:
            p = f"{MOMVIX_FLEET}/lean_vol_vix/none/{est}/tw2000/causal_tune_linear/{est}/lean_vol_vix/results.csv"
            try:  # header + the bars labelled 10:00 .. 16:00, byte for byte
                txt = ssh(
                    cl,
                    f'test -f {p} && grep -E "^date,|^.{{11}}(1[0-5]:(00|30)|16:00):00," {p}',
                    binary=True,
                )
            except RuntimeError as e:
                say(f"stored lean_vol_vix {est}: not available ({str(e)[:120]})")
                continue
            o = dest / "stored" / "lean_vol_vix" / est / "results_rth.csv"
            o.parent.mkdir(parents=True, exist_ok=True)
            o.write_bytes(txt)
            say(
                f"stored lean_vol_vix {est}: {txt.count(bytes([10])) - 1} regular-hours rows"
            )


# ============================================================================ local data
_DATES: pd.Series | None = None


def dates() -> pd.Series:
    """Date of every OOS solve (solve s = design row W + s), from the runner's dates mode."""
    global _DATES
    if _DATES is None:
        d = pd.read_csv(WORK / "dates.csv", parse_dates=["date"])
        assert (d["row"].to_numpy() == W + np.arange(len(d))).all()
        _DATES = d["date"]
    return _DATES


# a tag's twin through the all_features design cache (Hoffman2: the data prep done once per CPU type)
ALIASES = {"a": ("a", "ac"), "b": ("b", "bc"), "pre": ("pre", "prec"), "f": ("f", "fc")}
DUPLICATES: list[
    dict
] = []  # chunks run both ways: the cached and the uncached output must agree


def task_dirs(cl: str, tag: str, b: str, est: str, r: str) -> list[Path]:
    """DONE task directories of a tag and its cached twin, one per start row (the plain tag first)."""
    by_start: dict[int, Path] = {}
    for t in ALIASES.get(tag, (tag,)):
        base = WORK / cl / "results" / "pool_pretune" / t / b / est / f"r{r}"
        for d in base.glob("s*"):
            if not (d / "DONE").is_file():
                continue
            k = int(d.name[1:])
            if k in by_start:
                f0, f1 = (by_start[k] / "results_rth.csv", d / "results_rth.csv")
                if f0.is_file() and f1.is_file():
                    cr = bytes([13])
                    same = f0.read_bytes().replace(cr, b"") == f1.read_bytes().replace(
                        cr, b""
                    )
                    DUPLICATES.append(
                        dict(
                            cluster=cl,
                            tag=tag,
                            bucket=b,
                            est=est,
                            start=k,
                            identical=same,
                        )
                    )
                continue
            by_start[k] = d
    return [by_start[k] for k in sorted(by_start)]


def load_walk(
    cl: str, tag: str, b: str, est: str, r: str, need_full: bool = True
) -> dict | None:
    """The chunks of one walk: regular-hours rows and the re-choice record, concatenated; None if missing."""
    ds = task_dirs(cl, tag, b, est, r)
    if not ds:
        return None
    parts, recs, infos = [], [], []
    for d in ds:
        parts.append(pd.read_csv(d / "results_rth.csv", float_precision="round_trip"))
        z = np.load(d / "trace.npz")
        recs.append(
            {k: z[k] for k in ("block", "alpha", "masked", "resid", "grid", "tune_cpu")}
        )
        infos.append(json.loads((d / "run.json").read_text()) | dict(seconds=_secs(d)))
    blocks = np.concatenate([x["block"] for x in recs])
    if need_full and not (
        len(blocks) == K_END - K0 and (blocks == np.arange(K0, K_END)).all()
    ):
        return None
    res = pd.concat(parts, ignore_index=True)
    res["date"] = pd.to_datetime(res["date"])
    assert res["date"].is_monotonic_increasing and not res["date"].duplicated().any(), (
        cl,
        tag,
        b,
        est,
    )
    G = recs[0]["grid"]
    resid = (
        np.concatenate([x["resid"] for x in recs])
        if recs[0]["resid"].shape[-1]
        else None
    )
    return dict(
        rth=res,
        block=blocks,
        alpha=np.concatenate([x["alpha"] for x in recs]),
        masked=np.concatenate([x["masked"] for x in recs]),
        resid=resid,
        grid=G,
        info=infos,
        tune_cpu=np.concatenate([x["tune_cpu"] for x in recs]),
    )


def s1600(rth: pd.DataFrame) -> pd.Series:
    m = (rth["date"].dt.hour == 16) & (rth["date"].dt.minute == 0)
    return pd.Series(
        rth.loc[m, "pred_adj"].to_numpy(float),
        index=pd.DatetimeIndex(rth.loc[m, "date"]),
    )


def stored_rth(b: str, est: str) -> pd.DataFrame | None:
    """The stored arm-a regular-hours rows: results/linear_subsection/arms_hoffman2 (all_features, baseline),
    or the lean_vol_vix fleet pulled from Hoffman2."""
    p = (
        REPO
        / "results"
        / "linear_subsection"
        / "arms_hoffman2"
        / b
        / est
        / "tw2000"
        / "results_none_rth.csv"
        if b != "lean_vol_vix"
        else WORK / "h2" / "stored" / "lean_vol_vix" / est / "results_rth.csv"
    )
    if not p.is_file():
        return None
    r = pd.read_csv(p, float_precision="round_trip")
    r["date"] = pd.to_datetime(r["date"])
    return r


def stored_table_1600(b: str, est: str) -> pd.Series | None:
    """16:00 rows of results/spxw_pnl/yhat_pool_<est>_<bucket>.parquet (the master table's input)."""
    p = SPXW / f"yhat_pool_{SHORT[est]}_{b}.parquet"
    if not p.is_file():
        return None
    t = pd.read_parquet(p, columns=["t", "yhat"])
    et = pd.to_datetime(t["t"]).dt.tz_convert("America/New_York").dt.tz_localize(None)
    m = ((et.dt.hour == 16) & (et.dt.minute == 0)).to_numpy()
    return pd.Series(t.loc[m, "yhat"].to_numpy(float), index=pd.DatetimeIndex(et[m]))


# ============================================================================ plan (arm f)
def fold_table(cl: str, b: str, est: str) -> dict | None:
    ds = [x for x in task_dirs(cl, "pre", b, est, R_BB[b]) if x.name == f"s{PRE_START}"]
    if not ds:
        return None
    d = ds[0]
    z = np.load(d / "fold.npz")
    e2 = z["resid"] ** 2  # (folds, G, rows)
    val = z["val"]  # global design rows
    dts = dates()
    m16 = np.array(
        [
            (
                (dts.iloc[np.arange(v0, v1) - W].dt.hour == 16)
                & (dts.iloc[np.arange(v0, v1) - W].dt.minute == 0)
            ).to_numpy()
            for v0, v1 in val
        ]
    )
    mse = e2.mean(axis=2)  # (folds, G)
    mse16 = np.array(
        [
            [e2[f, g][m16[f]].mean() for g in range(e2.shape[1])]
            for f in range(e2.shape[0])
        ]
    )
    return dict(
        grid=z["grid"],
        mse=mse,
        mse16=mse16,
        val=val,
        fit=z["fit"],
        masked=z["masked"],
        val_first=[str(dts.iloc[v0 - W]) for v0, _ in val],
        val_last=[str(dts.iloc[v1 - 1 - W]) for _, v1 in val],
        info=json.loads((d / "run.json").read_text()),
    )


def argmin_first(v) -> int:
    v = np.asarray(v, float)
    return int(np.flatnonzero(v == np.nanmin(v))[0])


def retune_blocks() -> list[int]:
    dts = dates()
    return [
        k
        for k in range(K0, K_END)
        if dts.iloc[TUNE_PER * k] >= pd.Timestamp(FIRST_TRADE_DAY)
    ]


def margin_schedule(T: dict, rec: dict) -> tuple[dict, list[dict]]:
    """Arm f's alpha in every block 600 .. 817 and the retune log (the 16:00-bar study's light_retune with the
    spec's tail as the validation block)."""
    from src.evaluation.diebold_mariano import dm_test

    grid = [float(a) for a in T["grid"]]
    i0 = argmin_first(T["mse"].mean(axis=0))
    rt = set(retune_blocks())
    inc, sched, log = i0, {}, []
    dts = dates()
    for j, k in enumerate(rec["block"]):
        k = int(k)
        if k in rt:
            e2 = rec["resid"][j] ** 2  # (G, 125)
            mse = e2.mean(axis=1)
            cands = [i for i in (inc - 1, inc + 1) if 0 <= i < len(grid)]
            best = min(cands, key=lambda i: (mse[i], i))
            dvec = e2[best] - e2[inc]
            t = float(dm_test(dvec, np.zeros_like(dvec))["dm"])
            z = NormalDist().inv_cdf(1 - MARGIN_ALPHA / len(cands))
            sw = bool(t < -z)
            log.append(
                dict(
                    block=k,
                    start=str(dts.iloc[TUNE_PER * k]),
                    incumbent=grid[inc],
                    challengers=json.dumps([grid[i] for i in cands]),
                    best_challenger=grid[best],
                    incumbent_mse=float(mse[inc]),
                    best_mse=float(mse[best]),
                    rel_gain=float(1 - mse[best] / mse[inc]),
                    dm_t=t,
                    z_crit=z,
                    gain_any=bool(mse[best] < mse[inc]),
                    switched=sw,
                    causal_choice=float(rec["alpha"][j]),
                )
            )
            if sw:
                inc = best
            log[-1]["in_force"] = grid[inc]
        sched[k] = grid[inc]
    return sched, log


def plan(cl: str) -> None:
    SCHED_DIR.mkdir(parents=True, exist_ok=True)
    lines: dict[str, list[str]] = {}
    out = {}
    for b in BUCKETS:
        for est in EST:
            T = fold_table(cl, b, est)
            rec = load_walk(cl, REC_TAG[b], b, est, R_BB[b])
            if T is None or rec is None:
                say(
                    f"plan {cl} {b} {est}: pre-tune {'ok' if T is not None else 'MISSING'}, "
                    f"record walk {'ok' if rec is not None else 'MISSING'}"
                )
                continue
            sched, log = margin_schedule(T, rec)
            fp = SCHED_DIR / f"{cl}_{b}_{est}.json"
            fp.write_bytes(
                json.dumps({str(k): v for k, v in sched.items()}, indent=0).encode()
            )
            ks = np.arange(K0, K_END)
            diff = np.array(
                [sched[int(k)] != rec["alpha"][j] for j, k in enumerate(ks)]
            )
            runs, k = [], 0
            while k < len(ks):
                if diff[k]:
                    k2 = k
                    while k2 < len(ks) and diff[k2]:
                        k2 += 1
                    runs.append((k, k2))
                    k = k2
                else:
                    k += 1
            merged: list[
                list[int]
            ] = []  # runs closer than MERGE_GAP blocks share one walk (a data prep each)
            for k1, k2 in runs:
                if merged and k1 - merged[-1][1] < MERGE_GAP:
                    merged[-1][1] = k2
                else:
                    merged.append([k1, k2])
            runs = [tuple(x) for x in merged]
            tl = []
            for k1, k2 in runs:
                s = (
                    k1 - 1 if k1 > 0 else k1
                )  # one agreeing block first: the stitch gate
                for c0 in range(s, k2, F_CHUNK[b]):
                    c1 = min(c0 + F_CHUNK[b], k2)
                    end = -1 if K0 + c1 == K_END else W + TUNE_PER * (K0 + c1)
                    cache = (
                        ("fc", f" results/pool_pretune/design/{ARCH[est]}/all_features")
                        if (cl == "h2" and b == "all_features")
                        else ("f", "")
                    )
                    tl.append(
                        f"{cache[0]} walk {b} {est} {R_BB[b]} cluster/pool_pretune_sched/{fp.name} "
                        f"{W + TUNE_PER * (K0 + c0)} {end} {W}{cache[1]}"
                    )
            arch = ARCH[est] if cl == "h2" else cl
            lines.setdefault(arch, []).extend(tl)
            out[f"{b}/{est}"] = dict(
                pick=float(T["grid"][argmin_first(T["mse"].mean(axis=0))]),
                n_diff_blocks=int(diff.sum()),
                runs=runs,
                tasks=len(tl),
                switches=int(sum(x["switched"] for x in log)),
                retunes=len(log),
            )
            say(
                f"plan {cl} {b} {est}: pick {out[f'{b}/{est}']['pick']:g}, f differs from the record in "
                f"{int(diff.sum())} of {len(ks)} blocks, {len(tl)} tasks, {out[f'{b}/{est}']['switches']} switches"
            )
    for arch, tl in lines.items():
        p = TASK_DIR / f"pool_pretune_tasks_f_{arch}.txt"
        p.write_bytes(("\n".join(tl) + "\n").encode() if tl else b"")
        say(f"wrote {p}: {len(tl)} tasks")
    (WORK / cl).mkdir(parents=True, exist_ok=True)
    (WORK / cl / "plan.json").write_text(json.dumps(out, indent=1))


# ============================================================================ arms
def arm_f_series(cl: str, b: str, est: str, rec: dict) -> tuple[pd.Series | None, dict]:
    """Arm f's regular-hours rows: the record walk where the alphas agree, the scheduled walks elsewhere;
    the stitch gate on each scheduled walk's leading agreeing block."""
    fp = SCHED_DIR / f"{cl}_{b}_{est}.json"
    if not fp.is_file():
        return None, {}
    sched = {int(k): float(v) for k, v in json.loads(fp.read_text()).items()}
    ds = task_dirs(cl, "f", b, est, R_BB[b])
    dts = dates()
    rr = rec["rth"].set_index("date")["pred_adj"]
    out = rr.copy()
    need = {k for k in range(K0, K_END) if sched[k] != rec["alpha"][k - K0]}
    covered, stitch = set(), []
    for d in ds:
        z = np.load(d / "trace.npz")
        res = pd.read_csv(d / "results_rth.csv", float_precision="round_trip")
        res["date"] = pd.to_datetime(res["date"])
        s = res.set_index("date")["pred_adj"]
        for k, a in zip(z["block"], z["alpha"]):
            k = int(k)
            assert a == sched[k], (cl, b, est, k)
            lo = dts.iloc[TUNE_PER * k]
            hi = dts.iloc[min(TUNE_PER * (k + 1), len(dts)) - 1]
            blk = s[(s.index >= lo) & (s.index <= hi)]
            if k in need:
                out.loc[blk.index] = blk.to_numpy()
                covered.add(k)
            else:
                ref = rr.loc[blk.index]
                stitch.append(
                    dict(
                        block=k,
                        rows=len(blk),
                        bitwise=bool(np.array_equal(blk.to_numpy(), ref.to_numpy())),
                        max_abs=float(np.abs(blk.to_numpy() - ref.to_numpy()).max())
                        if len(blk)
                        else 0.0,
                    )
                )
    if need - covered:
        return None, dict(missing=sorted(need - covered))
    return out, dict(stitch=stitch, n_need=len(need))


def collect(cl: str) -> dict:
    """Every arm's 16:00 series, alpha path and records for one cluster."""
    A: dict = {}
    for b in BUCKETS:
        for est in EST:
            key = (b, est)
            e: dict = {}
            if cl == "h2":
                st = stored_table_1600(b, est)
                if st is None:
                    srth = stored_rth(b, est)
                    st = s1600(srth) if srth is not None else None
                e["a_series"] = st
                e["a_walk"] = load_walk(cl, "a", b, est, "1.0")
            else:
                e["a_walk"] = load_walk(cl, "a", b, est, "1.0")
                e["a_series"] = s1600(e["a_walk"]["rth"]) if e["a_walk"] else None
            if b != "baseline":
                e["b_walk"] = load_walk(cl, "b", b, est, R_BB[b])
                e["b_series"] = s1600(e["b_walk"]["rth"]) if e["b_walk"] else None
            rec = e.get("b_walk") if b != "baseline" else e.get("a_walk")
            e["rec"] = rec
            e["T"] = fold_table(cl, b, est)
            if rec is not None and e["T"] is not None:
                e["sched"], e["rlog"] = margin_schedule(e["T"], rec)
                fr, e["stitch"] = arm_f_series(cl, b, est, rec)
                e["f_series"] = None
                if fr is not None:
                    e["f_series"] = s1600(fr.rename("pred_adj").reset_index())
            A[key] = e
    return A


# ============================================================================ report
def to_targets(sc, s: pd.Series) -> np.ndarray:
    return s.reindex(pd.DatetimeIndex(sc.dates)).to_numpy(float)


def path_compare(mine: pd.Series, ref: pd.Series) -> dict:
    j = pd.concat([mine.rename("m"), ref.rename("r")], axis=1, join="inner")
    j = j[(j.index >= pd.Timestamp(FIRST_TRADE_DAY))]
    rel = np.abs(j["m"] / j["r"] - 1)
    return dict(
        rows=len(j),
        bitwise=int((j["m"] == j["r"]).sum()),
        max_rel=float(rel.max()),
        rows_rel_gt_1e9=int((rel > 1e-9).sum()),
    )


def report() -> None:  # noqa: C901 -- one linear report
    import close_linear_pretune as CLP

    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    sc = CLP.Scorer()
    mt = pd.read_csv(MASTER).set_index("key")
    gates: list[dict] = []

    def gate(name, item, value, ok):
        gates.append(dict(gate=name, item=item, value=value, ok=bool(ok)))

    # scorer gate: the stored pooled tables through this scorer == master_table.csv
    for b in ("all_features", "baseline"):
        for est in EST:
            key = f"pool_{SHORT[est]}_{b}"
            st = stored_table_1600(b, est)
            if st is None or key not in mt.index:
                continue
            s = sc.stats(sc.daily(to_targets(sc, st)))
            want = mt.loc[key, ["qlike_raw", "Sharpe_mid", "Sharpe_crossed"]].to_numpy(
                float
            )
            got = np.array([s["qlike"], s["sharpe_mid"], s["sharpe_crossed"]])
            gate(
                "scorer: stored pooled table -> master_table.csv (QLIKE, Sharpe mid, crossed)",
                key,
                f"{got[0]:.4f} / {got[1]:.2f} / {got[2]:.2f} (max abs diff {np.abs(got - want).max():.1e})",
                np.abs(got - want).max() < 1e-9,
            )
            # the local regular-hours file == the table (the stored arm a source for lean_vol_vix's twin)
            sr = stored_rth(b, est)
            if (
                sr is not None
            ):  # the tables were built with pandas' default CSV parser: within one ulp
                pc = path_compare(s1600(sr), st)
                gate(
                    "stored: arms_hoffman2 regular-hours 16:00 rows (exact parse) == the yhat_pool table "
                    "(default parse) to 1 ulp",
                    key,
                    f"{pc['bitwise']} of {pc['rows']} bit for bit, max rel {pc['max_rel']:.1e}",
                    pc["max_rel"] <= 2.3e-16,
                )

    per_cl: dict = {}
    for cl in CLUSTERS:
        if not (WORK / cl / "results").is_dir():
            continue
        A = collect(cl)
        per_cl[cl] = A
        # canary gate as run on the cluster
        gp = WORK / cl / "results" / "pool_pretune" / "gate.csv"
        if gp.is_file():
            g = pd.read_csv(gp)
            for _, r in g.iterrows():
                v = r.get("verdict")
                extra = ""
                if "max_rel_diff" in g.columns and pd.notna(r.get("max_rel_diff")):
                    extra = f"; max rel {r['max_rel_diff']:.1e}"
                    if "pred_adj_bitwise_rows" in g.columns and pd.notna(
                        r.get("pred_adj_bitwise_rows")
                    ):
                        extra += f", {int(r['pred_adj_bitwise_rows'])} of {int(r['rows'])} rows bit for bit"
                    if "bitwise_rows" in g.columns and pd.notna(r.get("bitwise_rows")):
                        extra += f", {int(r['bitwise_rows'])} of {int(r['rth_rows'])} regular-hours rows bit for bit"
                gate(
                    f"{cl} canary: {r['tag']} {r['mode']} r={r['r']} (slice from row {r['start']})",
                    f"{r['bucket']} {SHORT[r['est']]}",
                    f"{v}{extra}",
                    v in ("bitwise", "ok", "stored pending")
                    or str(v).startswith("ridge within")
                    or (
                        cl == "carc"
                        and r["est"] != "ridge"
                        and r["tag"] == "gate"
                        and bool(r.get("done", True))
                    ),
                )
        for (b, est), e in A.items():
            it = f"{b} {SHORT[est]}"
            for tag in ("a", "b"):
                w = e.get(f"{tag}_walk")
                if w is None:
                    continue
                n = len(w["rth"])
                rep = {
                    k: sum(x["repair"].get(k, 0) for x in w["info"])
                    for k in ("kkt_ok", "active_repair", "cd_polish", "fail")
                }
                gate(
                    f"{cl} walk {tag}: blocks 600 .. 817 contiguous, chunk rows tile, every pf batch solution "
                    "certified",
                    it,
                    f"{len(w['info'])} chunks, {n} regular-hours rows, repair {rep}",
                    rep["fail"] == 0 and n > 0,
                )
            # arm a vs the stored forecasts over the whole trade period (where rerun)
            if e.get("a_walk") is not None:
                sr = stored_rth(
                    b, est
                )  # the stored arm's exact (round-trip) regular-hours rows
                ref = s1600(sr) if sr is not None else None
                if ref is not None:
                    pc = path_compare(s1600(e["a_walk"]["rth"]), ref)
                    ridge_ok = (
                        est == "ridge" and pc["max_rel"] <= 1e-6
                    )  # no branches: float rounding only
                    ok = (
                        (pc["bitwise"] == pc["rows"] or ridge_ok)
                        if cl == "h2"
                        else (est != "ridge" or ridge_ok)
                    )
                    gate(
                        f"{cl} arm a rerun over the trade period == stored Hoffman2 forecasts (16:00 rows; "
                        f"{'bit for bit, ridge to 1e-6 where the stored arm ran on another CPU type' if cl == 'h2' else 'ridge to 1e-6, lasso / elastic net recorded'})",
                        it,
                        f"{pc['bitwise']} of {pc['rows']} bit for bit; max rel {pc['max_rel']:.1e}; "
                        f"rows > 1e-9: {pc['rows_rel_gt_1e9']}",
                        ok,
                    )
            st = e.get("stitch") or {}
            if st.get("stitch"):
                nb = sum(x["bitwise"] for x in st["stitch"])
                gate(
                    f"{cl} stitch: the scheduled walks' blocks whose alpha agrees with the record walk == the record walk",
                    it,
                    f"{nb} of {len(st['stitch'])} blocks bit for bit (max abs "
                    f"{max(x['max_abs'] for x in st['stitch']):.1e})",
                    nb == len(st["stitch"]),
                )
            if st.get("missing"):
                gate(
                    f"{cl} arm f: every scheduled block run",
                    it,
                    f"missing blocks {st['missing'][:8]}",
                    False,
                )

    uniq = {
        (x["cluster"], x["tag"], x["bucket"], x["est"], x["start"]): x
        for x in DUPLICATES
    }
    for x in (
        uniq.values()
    ):  # a chunk run both through the design cache and through the full data prep
        gate(
            f"{x['cluster']} design cache: cached chunk == uncached chunk (regular-hours file, byte for byte)",
            f"{x['bucket']} {SHORT[x['est']]} {x['tag']} s{x['start']}",
            "identical" if x["identical"] else "DIFFERS",
            x["identical"],
        )

    # ---------------------------------------------------------------- scores
    rows, switch_rows, retune_rows, pick_rows = [], [], [], []
    series_export: dict = {}
    for cl, A in per_cl.items():
        for (b, est), e in A.items():
            arms = {"a_causal": e.get("a_series")}
            if b != "baseline":
                arms["b_causal_bb"] = e.get("b_series")
            arms[f_name(b)] = e.get("f_series")
            alph = {
                "a_causal": e["a_walk"]["alpha"] if e.get("a_walk") else None,
                "b_causal_bb": e["b_walk"]["alpha"] if e.get("b_walk") else None,
                f_name(b): np.array([e["sched"][k] for k in range(K0, K_END)])
                if e.get("sched")
                else None,
            }
            if b == "baseline" and alph["a_causal"] is None and e.get("rec"):
                alph["a_causal"] = e["rec"]["alpha"]
            da = (
                sc.daily(to_targets(sc, arms["a_causal"]))
                if arms["a_causal"] is not None
                else None
            )
            for arm, s in arms.items():
                if s is None:
                    continue
                d = sc.daily(to_targets(sc, s))
                if not np.isfinite(d["plain"]).all():
                    say(
                        f"{cl} {b} {est} {arm}: {int((~np.isfinite(d['plain'])).sum())} trade days without a forecast"
                    )
                    continue
                st = sc.stats(d)
                c_ols = sc.contrast(d, sc.ols)
                al = alph.get(arm)
                row = dict(
                    cluster=cl,
                    bucket=b,
                    estimator=LABEL[est],
                    est=est,
                    arm=arm,
                    label=ARM_LABEL[arm],
                    r=0.01 if arm != "a_causal" and b != "baseline" else 1.0,
                    alpha_first=float(al[0]) if al is not None else np.nan,
                    alpha_changes=int((np.diff(al) != 0).sum())
                    if al is not None
                    else np.nan,
                    alpha_set=json.dumps(sorted({float(x) for x in al}))
                    if al is not None
                    else "",
                    **st,
                    dq_pct_vs_ols=c_ols["dq_pct"],
                    dm_vs_ols=c_ols["dm"],
                    dm_p_vs_ols=c_ols["dm_p"],
                )
                if da is not None and arm != "a_causal":
                    c = sc.contrast(d, da)
                    row.update(
                        dq_pct_vs_a=c["dq_pct"],
                        dm_vs_a=c["dm"],
                        dm_p_vs_a=c["dm_p"],
                        hac_t_mid_vs_a=c["hac_t_mid"],
                        hac_lag=c["hac_lag"],
                        n_diff_position_vs_a=c["n_diff_position"],
                        dsharpe_mid_vs_a=st["sharpe_mid"] - sc.stats(da)["sharpe_mid"],
                    )
                if (
                    arm != "a_causal"
                    and b != "baseline"
                    and arm.startswith("f")
                    and arms.get("b_causal_bb") is not None
                ):
                    db = sc.daily(to_targets(sc, arms["b_causal_bb"]))
                    c = sc.contrast(d, db)
                    row.update(
                        dm_vs_b=c["dm"],
                        hac_t_mid_vs_b=c["hac_t_mid"],
                        n_diff_position_vs_b=c["n_diff_position"],
                    )
                rows.append(row)
                sid = f"pool_linear_pretune/{SHORT[est]}_{b}_{arm}" + (
                    "" if cl == "h2" else "_carc"
                )
                series_export[sid] = dict(
                    series=s,
                    bucket=b,
                    est=est,
                    arm=arm,
                    cluster=cl,
                    alphas=[float(x) for x in al] if al is not None else [],
                )
            # causal switch / edge evidence (every causal walk with a record)
            for tag in ("a", "b"):
                w = e.get(f"{tag}_walk")
                if w is None or w["resid"] is None:
                    continue
                g = [float(x) for x in w["grid"]]
                al = w["alpha"]
                z = NormalDist().inv_cdf(1 - MARGIN_ALPHA / (len(g) - 1))
                from src.evaluation.diebold_mariano import dm_test

                for j in range(len(al)):
                    k = int(w["block"][j])
                    ia = g.index(float(al[j]))
                    e2 = w["resid"][j] ** 2
                    mse = e2.mean(axis=1)
                    prev = g.index(float(al[j - 1])) if j else None
                    sw = prev is not None and prev != ia
                    t = (
                        float(dm_test(e2[ia] - e2[prev], np.zeros(e2.shape[1]))["dm"])
                        if sw
                        else np.nan
                    )
                    switch_rows.append(
                        dict(
                            cluster=cl,
                            bucket=b,
                            est=est,
                            arm=ARM_NAME[tag],
                            block=k,
                            start=str(dates().iloc[TUNE_PER * k]),
                            alpha=g[ia],
                            switch=sw,
                            at_edge=ia in (0, len(g) - 1),
                            masked=int(w["masked"][j]),
                            tail_gain_vs_prev=float(1 - mse[ia] / mse[prev])
                            if sw
                            else np.nan,
                            tail_t_vs_prev=t,
                            passes_margin=bool(sw and t < -z),
                            val_mse=json.dumps([round(float(v), 6) for v in mse]),
                        )
                    )
            for x in e.get("rlog", []):
                retune_rows.append(dict(cluster=cl, bucket=b, est=est, **x))
            T = e.get("T")
            if T is not None:
                g = [float(x) for x in T["grid"]]
                m = T["mse"].mean(axis=0)
                i = argmin_first(m)
                pick_rows.append(
                    dict(
                        cluster=cl,
                        bucket=b,
                        est=est,
                        r=R_BB[b],
                        grid=json.dumps(g),
                        pick_mse=g[i],
                        pick_mse_1600=g[argmin_first(T["mse16"].mean(axis=0))],
                        edge="low" if i == 0 else "high" if i == len(g) - 1 else "",
                        fold_picks=json.dumps(
                            [g[argmin_first(T["mse"][f])] for f in range(len(T["mse"]))]
                        ),
                        val_mse_by_alpha=json.dumps([round(float(v), 6) for v in m]),
                        val_mse_1600_by_alpha=json.dumps(
                            [round(float(v), 6) for v in T["mse16"].mean(axis=0)]
                        ),
                        folds=json.dumps(
                            [
                                f"{a[:10]}..{z_[:10]}"
                                for a, z_ in zip(T["val_first"], T["val_last"])
                            ]
                        ),
                        masked=json.dumps([int(v) for v in T["masked"]]),
                    )
                )
    arms_df = pd.DataFrame(rows)
    sw_df, rt_df, pk_df = (
        pd.DataFrame(switch_rows),
        pd.DataFrame(retune_rows),
        pd.DataFrame(pick_rows),
    )
    arms_df.to_csv(OUT / "arms.csv", index=False)
    sw_df.to_csv(OUT / "causal_switch_log.csv", index=False)
    rt_df.to_csv(OUT / "retune_log.csv", index=False)
    pk_df.to_csv(OUT / "penalty_picks.csv", index=False)
    g_df = pd.DataFrame(gates)
    g_df.to_csv(OUT / "gates.csv", index=False)
    export(series_export, sc)
    cpu = cpu_table()
    cpu.to_csv(OUT / "cpu_seconds.csv", index=False)
    write_summary(arms_df, g_df, sw_df, rt_df, pk_df, cpu)
    say(
        f"report done in {time.time() - t0:.0f}s: {len(arms_df)} arm rows, {int(g_df['ok'].sum())} of {len(g_df)} gates"
    )


def export(series: dict, sc) -> None:
    """forecasts.parquet (Hoffman2) and forecasts_carc.parquet in the export format of
    results/close_studies_2026-10-03/forecasts/: series_id, study, row, date (16:00 stamps), pred_adj,
    pred_clock (NaN: these series are scored plain)."""
    fd = pd.DatetimeIndex(sc.dates)
    keep = fd[
        (fd >= pd.Timestamp(FIRST_TRADE_DAY))
        & (fd <= pd.Timestamp(LAST_TRADE_DAY) + pd.Timedelta(hours=16))
    ]
    for cl, name in (("h2", "forecasts.parquet"), ("carc", "forecasts_carc.parquet")):
        parts, cat = [], []
        for sid, s in series.items():
            if s["cluster"] != cl:
                continue
            v = s["series"].reindex(keep).to_numpy(float)
            parts.append(
                pd.DataFrame(
                    dict(
                        series_id=sid,
                        study="pool_linear_pretune",
                        row=np.arange(len(keep), dtype=np.int16),
                        date=keep.astype("datetime64[ns]"),
                        pred_adj=v,
                        pred_clock=np.full(len(keep), np.nan),
                    )
                )
            )
            cat.append(
                dict(
                    series_id=sid,
                    bucket=s["bucket"],
                    estimator=LABEL[s["est"]],
                    arm=s["arm"],
                    label=ARM_LABEL[s["arm"]],
                    cluster=CLUSTERS[cl]["label"],
                    r=1.0
                    if s["arm"] == "a_causal" or s["bucket"] == "baseline"
                    else 0.01,
                    alphas_by_block_600_817=json.dumps(s["alphas"]),
                )
            )
        if parts:
            pd.concat(parts, ignore_index=True).to_parquet(
                OUT / name, index=False, compression="zstd"
            )
            pd.DataFrame(cat).to_csv(
                OUT
                / name.replace("forecasts", "catalogue").replace(".parquet", ".csv"),
                index=False,
            )


def _secs(d: Path) -> float:
    """A task's wall seconds (NaN when SECONDS is missing or caught mid-write)."""
    try:
        return float((d / "SECONDS").read_text())
    except (OSError, ValueError):
        return float("nan")


def cpu_table() -> pd.DataFrame:
    rows = []
    for cl in CLUSTERS:
        for d in sorted((WORK / cl / "results" / "pool_pretune").glob("*/*/*/r*/s*")):
            if not (d / "run.json").is_file():
                continue
            info = json.loads((d / "run.json").read_text())
            host = (
                (d / "host.txt").read_text().splitlines()
                if (d / "host.txt").is_file()
                else [""]
            )
            rows.append(
                dict(
                    cluster=cl,
                    tag=d.parts[-5],
                    bucket=d.parts[-4],
                    est=d.parts[-3],
                    r=d.parts[-2][1:],
                    start=int(d.name[1:]),
                    cpu_sec=info.get("cpu_sec"),
                    wall_sec=_secs(d),
                    host=info.get("host"),
                    cpu_model=next(
                        (h.split(":", 1)[1].strip() for h in host if "Model name" in h),
                        "",
                    ),
                )
            )
        for d in sorted(
            (WORK / cl / "results" / "pool_pretune" / "design").glob("*/*")
        ):
            if (d / "SECONDS").is_file():  # one thread throughout: CPU time = wall time
                sec = _secs(d)
                rows.append(
                    dict(
                        cluster=cl,
                        tag="design",
                        bucket=d.name,
                        est="",
                        r="",
                        start=0,
                        cpu_sec=sec,
                        wall_sec=sec,
                        host="",
                        cpu_model=d.parent.name,
                    )
                )
    return pd.DataFrame(rows)


def _f(v, nd=4, sign=False):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ""
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


def _md(df: pd.DataFrame) -> str:
    if df.empty:
        return "(none)\n"
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out) + "\n"


def write_summary(arms, gates, sw, rt, pk, cpu) -> None:  # noqa: C901
    L: list[str] = []
    L.append("# The pooled linear models pre-tuned, backbone at 1/100\n")
    L.append(
        f"Written by `experiments/pool_linear_pretune_report.py report` on {time.strftime('%Y-%m-%d %H:%M')}; "
        "every number below is read from the CSVs in this folder.\n"
    )
    L.append("## Question and design\n")
    L.append(
        'User (2026-10-06): "use the pre-tuned version for all linear models."  The 16:00-bar version is '
        "`results/linear_pretune_2026-10-06/`; this is the pooled twin: `pool_<ridge|lasso|enet>_<bucket>`, one "
        "coefficient vector over all 48 half-hour bars (`specs/causal_tune_linear.py`, segment `none`), window "
        "96,000 bars (2000 sessions), coefficients re-solved every bar (ridge: Sherman-Morrison; lasso / elastic "
        "net: warm online homotopy), identifiability mask, cold seed and penalty re-choice every 250 solves on the "
        "window's any-clock validation tail (fit 95,850 bars, embargo 25, tail 125), scored at the 16:00 bar. "
        "The spec's estimator section is executed read-only by `experiments/pool_linear_pretune.py`; its only new "
        "code is a subclass with glmnet-style penalty factors (r = 1 executes the spec's floating-point operations "
        "exactly; the batch elastic net with r != 1 by column scaling, certified by the unscaled KKT conditions), "
        "the re-choice record, and an optional fixed schedule.\n"
    )
    L.append(
        "| arm | penalty structure | alpha schedule |\n|---|---|---|\n"
        "| a_causal | one penalty (spec; Hoffman2: the stored forecasts) | re-chosen every 250 solves on the 125-bar tail |\n"
        "| b_causal_bb | backbone (34 columns: HAR ladder, HAR x open / close, calendar, expiry) at r = 1/100 | the same causal re-choice |\n"
        "| f_retune_margin_bb | backbone 1/100 (baseline: one penalty) | pre-tuned before 2020-01-03; from the first boundary on or after it, every 250 solves the grid neighbours challenge the incumbent on the same tail; switch only if DM t < -z(0.05 / challengers) |\n"
    )
    L.append(
        f"Trade period: blocks {K0} .. {K_END - 1} (block {K0} starts {dates().iloc[TUNE_PER * K0]}, the re-choice "
        f"boundary before {FIRST_TRADE_DAY}; arm f's retunes at the {len(retune_blocks())} boundaries from "
        f"{dates().iloc[TUNE_PER * retune_blocks()[0]]}). Pre-tune: 4 folds x 12,000 bars (250 sessions x 48) "
        "tiling the 48,000 bars before the first bar of 2020-01-03, each fit on the 96,000 bars before a 1,200-bar "
        "(25-session) embargo: the 16:00-bar study's fold scheme in the pooled window's units; every grid alpha by "
        "the spec's batch solution; criterion = mean any-clock validation MSE (adjusted scale); the 16:00-row MSE "
        "pick is recorded. Scoring: plain f^2 x B on the 866 trade days (the master-table GUI's scorer), DM on daily "
        "QLIKE, Newey-West t (lag 14) of the daily mid P&L difference, point estimates.\n"
    )
    L.append("## Where it ran\n")
    L.append(
        "Hoffman2 is the run of record: every arm on the CPU type its stored comparator (arm a) ran on, "
        "numpy 2.2.6 / OpenBLAS 0.3.29, one thread: ridge and elastic net on intel-6736p, lasso on "
        "intel-gold-6140. The stored arms' hosts (run logs): all_features ridge n7134 (intel-E5-2670: the one "
        "arm not rerun on its own CPU type; ridge has no branch to take, and the canary agrees to 4e-7), elastic net n1144 "
        "(6736p), lasso n6637 (gold-6140, the lassofix re-run); baseline ridge n1156 / elastic net n1144 "
        "(6736p), lasso n6643 (gold-6140); lean_vol_vix (2026-10-06 fleet) ridge n1152 / elastic net n1144 "
        "(6736p), lasso n6640 (gold-6140). The all_features walks read a design cache (the spec's own data prep "
        "run once per CPU type, the column-major matrix kept as the pipeline holds it), gated through the cache "
        "as well (tag gatec). CARC (epyc-7513, numpy 2.3.5 / OpenBLAS 0.3.30) is the backup: every arm, arm a "
        "included, rerun there; its lasso / elastic net differ from Hoffman2's by the known cross-machine "
        "branching of the warm homotopy.\n"
    )
    L.append("## Gates\n")
    gg = gates.copy()
    gg["ok"] = gg["ok"].map({True: "yes", False: "NO"})
    L.append(_md(gg))
    L.append(f"{int(gates['ok'].sum())} of {len(gates)} gates pass.\n")
    L.append("## Pre-tune picks\n")
    if not pk.empty:
        L.append(
            _md(
                pk[
                    [
                        "cluster",
                        "bucket",
                        "est",
                        "r",
                        "pick_mse",
                        "pick_mse_1600",
                        "edge",
                        "fold_picks",
                        "val_mse_by_alpha",
                        "folds",
                    ]
                ]
            )
        )
    L.append("## Causal re-choices in the trade period (arms a and b)\n")
    if not sw.empty:
        s = (
            sw[sw["block"] >= K0]
            .groupby(["cluster", "bucket", "est", "arm"])
            .agg(
                blocks=("block", "size"),
                switches=("switch", "sum"),
                at_edge=("at_edge", "sum"),
                pass_margin=("passes_margin", "sum"),
                median_gain_pct=(
                    "tail_gain_vs_prev",
                    lambda v: 100 * np.nanmedian(v) if v.notna().any() else np.nan,
                ),
                alphas=(
                    "alpha",
                    lambda v: " ".join(
                        f"{a:g}:{n}" for a, n in v.value_counts().sort_index().items()
                    ),
                ),
            )
            .reset_index()
        )
        s["median_gain_pct"] = s["median_gain_pct"].map(lambda v: _f(v, 1))
        L.append(
            "switches = blocks whose alpha differs from the previous block's; pass_margin = switches whose "
            "tail DM t (new vs previous alpha) clears -z(0.05 / (grid - 1)); alphas = alpha:blocks.\n"
        )
        L.append(_md(s))
    L.append("## Arm f light retunes\n")
    if not rt.empty:
        r = (
            rt.groupby(["cluster", "bucket", "est"])
            .agg(
                retunes=("block", "size"),
                any_gain=("gain_any", "sum"),
                switches=("switched", "sum"),
                min_t=("dm_t", "min"),
                in_force=(
                    "in_force",
                    lambda v: " ".join(
                        f"{a:g}:{n}" for a, n in v.value_counts().sort_index().items()
                    ),
                ),
            )
            .reset_index()
        )
        r["min_t"] = r["min_t"].map(lambda v: _f(v, 2))
        L.append(_md(r))
    L.append("## Scores\n")
    for cl in arms["cluster"].unique() if not arms.empty else []:
        L.append(f"### {CLUSTERS[cl]['label']}\n")
        a = arms[arms["cluster"] == cl]
        t = pd.DataFrame(
            dict(
                bucket=a["bucket"],
                estimator=a["estimator"],
                arm=a["arm"],
                alphas=a["alpha_set"],
                changes=a["alpha_changes"],
                QLIKE=a["qlike"].map(lambda v: _f(v, 4)),
                DM_vs_OLS=a["dm_vs_ols"].map(lambda v: _f(v, 2, True)),
                Sharpe=[
                    f"{m:.2f} / {x:.2f}"
                    for m, x in zip(a["sharpe_mid"], a["sharpe_crossed"])
                ],
                DM_vs_a=a.get("dm_vs_a", pd.Series(dtype=float)).map(
                    lambda v: _f(v, 2, True)
                ),
                HAC_t_vs_a=a.get("hac_t_mid_vs_a", pd.Series(dtype=float)).map(
                    lambda v: _f(v, 2, True)
                ),
                pos_diff_vs_a=a.get("n_diff_position_vs_a", pd.Series(dtype=float)).map(
                    lambda v: "" if pd.isna(v) else str(int(v))
                ),
            )
        )
        L.append(_md(t))
    L.append("## Reading\n")
    for cl in arms["cluster"].unique() if not arms.empty else []:
        a = arms[arms["cluster"] == cl]
        key = ["bucket", "est"]
        A0 = a[a["arm"] == "a_causal"].set_index(key)
        for arm, ref, lab in (
            ("b_causal_bb", "a", "b vs a"),
            ("f", "a", "f vs a"),
            ("f", "b", "f vs b"),
        ):
            X = a[a["arm"].str.startswith("f")] if arm == "f" else a[a["arm"] == arm]
            if X.empty:
                continue
            X = X.set_index(key)
            R = A0 if ref == "a" else a[a["arm"] == "b_causal_bb"].set_index(key)
            idx = X.index.intersection(R.index)
            if not len(idx):
                continue
            dq = 100 * (X.loc[idx, "qlike"] / R.loc[idx, "qlike"] - 1)
            dm = (
                X.loc[idx, f"dm_vs_{ref}"]
                if f"dm_vs_{ref}" in X
                else pd.Series(np.nan, index=idx)
            )
            ds = X.loc[idx, "sharpe_mid"] - R.loc[idx, "sharpe_mid"]
            ht = (
                X.loc[idx, f"hac_t_mid_vs_{ref}"]
                if f"hac_t_mid_vs_{ref}" in X
                else pd.Series(np.nan, index=idx)
            )
            L.append(
                f"- {CLUSTERS[cl]['label']}, {lab} over {len(idx)} estimator x input-set pairs: QLIKE lower in "
                f"{int((dq < 0).sum())} (mean {dq.mean():+.1f} %), DM below -1.96 in {int((dm < -1.96).sum())}, "
                f"above +1.96 in {int((dm > 1.96).sum())}; Sharpe mid higher in {int((ds > 0).sum())} "
                f"(mean {ds.mean():+.2f}), HAC |t| above 1.96 in {int((ht.abs() > 1.96).sum())}."
            )
    L.append("")
    L.append("## Compute\n")
    if not cpu.empty:
        c = (
            cpu.groupby("cluster")
            .agg(
                tasks=("cpu_sec", "size"),
                cpu_hours=("cpu_sec", lambda v: v.sum() / 3600),
                wall_hours_sum=("wall_sec", lambda v: v.sum() / 3600),
                longest_task_h=("wall_sec", lambda v: v.max() / 3600),
            )
            .reset_index()
        )
        L.append(_md(c.round(2)))
    L.append("## Files\n")
    L.append(
        "- `arms.csv`: every arm's scores; `penalty_picks.csv`, `retune_log.csv`, `causal_switch_log.csv`; "
        "`gates.csv`; `cpu_seconds.csv`.\n- `forecasts.parquet` (Hoffman2; series_id "
        "`pool_linear_pretune/<est>_<bucket>_<arm>`, 16:00 rows 2020-01-03 .. 2024-04-30, pred_adj; pred_clock "
        "NaN: scored plain) and `catalogue.csv`; `forecasts_carc.parquet` / `catalogue_carc.csv` (CARC).\n"
        "- Code: `experiments/pool_linear_pretune.py` (cluster runner), `experiments/pool_linear_pretune_report.py` "
        "(pull / plan / report); cluster scripts `cluster/*pool_pretune*`, `cluster/slurm/*pool_pretune*`; the arm-f "
        "schedules `cluster/pool_pretune_sched/`.\n"
    )
    (OUT / "SUMMARY.md").write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    st = sys.argv[1]
    if st == "pull":
        pull(sys.argv[2])
    elif st == "plan":
        plan(sys.argv[2])
    elif st == "report":
        report()
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
