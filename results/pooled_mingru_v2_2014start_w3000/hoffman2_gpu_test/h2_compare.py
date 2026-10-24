"""MinGRU v2 Hoffman2 GPU test: compare the Hoffman2 G1 runs (h2_g1_bench.sh) with each other and with CARC's P100 G1
(job 13353680), and reduce the throughput bench.

Inputs (pulled, then reduced by `pooled_mingru_v2.py post` and `g1-check` per run):
  --carc DIR    CARC run dir (summary/ + tune/epochs.csv + tune/steps.csv.gz + tune.log)
  --h2 DIR      dir holding one sub-dir per Hoffman2 job id (env.txt, runs.txt, gpu_apps.csv, g1_r1/, g1_r2/, b*_c*/)
  --labels      GPU type per job id, e.g. "A100=15260301,RTX2080Ti=15260303"
  --y           inputs/y_baseline.f32 (the trainer's target)
  --out FILE    compare.json
Agreement criterion (as the CARC cross-GPU-type checks): seed-level QLIKE within 0.1% relative.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "experiments"))
from pooled_mingru_v2 import qlike_trainer  # noqa: E402

TOL = 1e-3
VAL = (47_527, 50_777)  # the twin's validation rows (fold 3), forecast rows [47,202, 50,777) minus the 325-row embargo
EMB = 325
# P100-s a tune model-epoch by HAR (H, L), at the old run's concurrency (CONC 24 / 12 / 7; analysis_w3000 epoch_cost)
P100_COST = {(64, 20): 0.04013, (256, 20): 0.25028, (1024, 20): 5.53006}
P100_EPOCH_COST = P100_COST[(64, 20)]
POOL_RE = re.compile(r"pool phase (\d+): (\d+) jobs on (\d+) slots in ([0-9.]+) s")
DEV_RE = re.compile(r"device (.+?) \(sm_(\d+), (\d+) SMs\), driver (\d+), runtime (\d+)")


def rel(a: float, b: float) -> float:
    return float(abs(a - b) / abs(b))


def load_run(d: Path) -> dict[str, Any]:
    """One G1 run; a run cut short by a crash may hold the tune job only (refit / shortcut fields None)."""
    jo = pd.read_csv(d / "summary" / "jobs.csv").set_index("id")
    tid = next(i for i in jo.index if i.startswith("v2g1tune_"))
    rid = next((i for i in jo.index if i.startswith("v2g1refit_")), None)
    sid = next((i for i in jo.index if i.startswith("v2g1short_")), None)
    ep = pd.read_csv(d / "tune" / "epochs.csv")
    vq = ep[ep["id"] == tid].sort_values("epoch")["val_qlike"].to_numpy(float)
    ec = pd.read_csv(d / "summary" / "epoch_curves.csv")
    tq = {
        k: ec[ec["id"] == i].sort_values("epoch")["loss_w"].to_numpy(float) for k, i in (("tune", tid), ("refit", rid))
    }
    tp = d / "summary" / "train_ql.csv"
    trq = pd.read_csv(tp).set_index("id")["train_ql"] if tp.is_file() else pd.Series(dtype=float)
    fc: dict[str, np.ndarray] = {}
    if (d / "summary" / "fc.npz").is_file():
        z = np.load(d / "summary" / "fc.npz")
        fc = {str(i): z["F"][k].astype(np.float64) for k, i in enumerate(z["ids"])}
    steps: dict[str, list[str]] = {tid: [], str(rid): []}
    with gzip.open(d / "tune" / "steps.csv.gz", "rt") as fh:
        next(fh)
        for ln in fh:
            i, rest = ln.split(",", 1)
            if i in steps:
                steps[i].append(rest)
    log = (d / "tune.log").read_text()
    pools = [(int(m.group(1)), int(m.group(2)), float(m.group(4))) for m in POOL_RE.finditer(log)]
    dev = DEV_RE.search(log)
    return dict(
        tid=tid,
        rid=rid,
        sid=sid,
        complete=bool(rid in jo.index and sid in jo.index and len(tq["refit"]) == 300 and sid in trq.index),
        vq=vq,
        best_epoch=int(jo.loc[tid, "best_epoch"]),
        best_val=float(jo.loc[tid, "best_val_mse"]),
        tq=tq,
        train_ql_refit=float(trq.loc[rid]) if rid in trq.index else None,
        train_ql_short=float(trq.loc[sid]) if sid in trq.index else None,
        fc=fc.get(str(rid)),
        steps=steps[tid],
        steps_refit=steps[str(rid)],
        pools=pools,
        device=dev.groups() if dev else None,
        t_train_tune=float(jo.loc[tid, "t_train"]),
        log_errors=[x for x in log.splitlines() if "error" in x.lower()][:3],
    )


def compare(a: dict[str, Any], b: dict[str, Any], y: np.ndarray) -> dict[str, Any]:
    """a against the reference b (the tune job always; the refit twin and the shortcut when both runs have them)."""
    vr = np.abs(a["vq"] - b["vq"]) / b["vq"]
    tr = np.abs(a["tq"]["tune"] - b["tq"]["tune"]) / b["tq"]["tune"]
    first = next((k for k, (x, z) in enumerate(zip(a["steps"], b["steps"])) if x != z), None)
    r: dict[str, Any] = dict(
        complete=[a["complete"], b["complete"]],
        steps_identical=bool(a["steps"] == b["steps"]),
        first_differing_step_line=first,
        first_differing_epoch=None if first is None else int(a["steps"][first].split(",", 1)[0]),
        step0_loss=[a["steps"][0].split(",")[2], b["steps"][0].split(",")[2]],
        val_qlike_epoch_max_rel=float(vr.max()),
        val_qlike_epoch_argmax=int(vr.argmax()) + 1,
        val_qlike_epoch_median_rel=float(np.median(vr)),
        val_qlike_first_epoch_over_tol=int(np.argmax(vr > TOL)) + 1 if (vr > TOL).any() else None,
        val_qlike_e300=[float(a["vq"][-1]), float(b["vq"][-1])],
        val_qlike_e300_rel=rel(a["vq"][-1], b["vq"][-1]),
        best_epoch=[a["best_epoch"], b["best_epoch"]],
        best_val_qlike=[a["best_val"], b["best_val"]],
        best_val_qlike_rel=rel(a["best_val"], b["best_val"]),
        TQ300_tune=[float(a["tq"]["tune"][-1]), float(b["tq"]["tune"][-1])],
        TQ300_tune_rel=rel(a["tq"]["tune"][-1], b["tq"]["tune"][-1]),
        TQ_epoch_max_rel_tune=float(tr.max()),
    )
    ok = r["val_qlike_epoch_max_rel"] <= TOL and r["TQ300_tune_rel"] <= TOL
    if a["complete"] and b["complete"]:
        fa, fb = a["fc"], b["fc"]
        vqa = float(np.mean(qlike_trainer(y[VAL[0] : VAL[1]], fa[EMB:])))
        vqb = float(np.mean(qlike_trainer(y[VAL[0] : VAL[1]], fb[EMB:])))
        fr = np.abs(fa - fb) / np.abs(fb)
        r.update(
            TQ300_refit=[float(a["tq"]["refit"][-1]), float(b["tq"]["refit"][-1])],
            TQ300_refit_rel=rel(a["tq"]["refit"][-1], b["tq"]["refit"][-1]),
            train_ql_refit=[a["train_ql_refit"], b["train_ql_refit"]],
            train_ql_refit_rel=rel(a["train_ql_refit"], b["train_ql_refit"]),
            train_ql_short=[a["train_ql_short"], b["train_ql_short"]],
            train_ql_short_rel=rel(a["train_ql_short"], b["train_ql_short"]),
            twin_val_qlike=[vqa, vqb],
            twin_val_qlike_rel=rel(vqa, vqb),
            twin_fc_max_rel=float(fr.max()),
            twin_fc_median_rel=float(np.median(fr)),
        )
        ok = ok and r["TQ300_refit_rel"] <= TOL and r["train_ql_refit_rel"] <= TOL and r["twin_val_qlike_rel"] <= TOL
    r["within_0.1pct"] = bool(ok)
    r["best_epoch_equal"] = a["best_epoch"] == b["best_epoch"]
    return r


def speed(job: Path, runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Seconds a model-epoch of each bench run (pool-phase seconds / epochs run, the old run's epoch_cost reduction),
    against the P100 cost of its (H, L) cell; peak GPU memory of the trainer (and peak host RSS where sampled)."""
    out: dict[str, Any] = {}
    txt = (job / "runs.txt").read_text().splitlines()
    rc = {ln.split()[0]: ln for ln in txt if " rc=" in ln}
    apps = pd.read_csv(
        job / "gpu_apps.csv", header=None, names=["t", "label", "pid", "uuid", "mib"], skipinitialspace=True
    )
    apps["mib"] = pd.to_numeric(apps["mib"], errors="coerce")
    peak = apps.groupby(["label", "t"])["mib"].sum().groupby("label").max().to_dict()
    rss: dict[str, float] = {}
    if (job / "host_rss_kib.csv").is_file():
        h = pd.read_csv(job / "host_rss_kib.csv", header=None, names=["t", "label", "kib"])
        rss = (pd.to_numeric(h["kib"], errors="coerce").groupby(h["label"]).max() / 2**20).to_dict()
    g1 = runs.get("g1_r1")
    if g1:
        ph = {p[0]: p for p in g1["pools"]}
        out["g1_single_model_s_per_epoch"] = ph[0][2] / 300 if 0 in ph else None
        out["g1_tune_t_train_s"] = g1["t_train_tune"]
    for lab in sorted(rc):
        m = re.search(r"rc=(\d+) wall_s=([0-9.]*) conc=(\d+) jobs=(\d+)", rc[lab])
        if not m:
            continue
        e: dict[str, Any] = dict(
            rc=int(m.group(1)),
            wall_s=float(m.group(2)) if m.group(2) else None,
            conc=int(m.group(3)),
            jobs=int(m.group(4)),
            peak_trainer_gpu_mib=float(peak.get(lab, float("nan"))),
        )
        if lab in rss:
            e["peak_host_rss_gib"] = float(rss[lab])
        jp, lg = job / lab / "tune" / "jobs_out.csv", job / lab / "tune.log"
        if lab.startswith("b") and jp.is_file() and lg.is_file():
            jo = pd.read_csv(jp)
            pools = [float(p.group(4)) for p in POOL_RE.finditer(lg.read_text())]
            me = int(jo.loc[jo["status"] == "ok", "epochs_run"].sum())
            e.update(n_ok=int((jo["status"] == "ok").sum()), model_epochs=me, pool_s=sum(pools))
            if me and len(jo):
                cell = (int(jo["H"].iloc[0]), int(jo["L"].iloc[0]))
                e["cell_H_L"] = cell
                e["s_per_model_epoch"] = sum(pools) / me
                if cell in P100_COST:
                    e["P100_s_per_model_epoch"] = P100_COST[cell]
                    e["speed_vs_P100"] = P100_COST[cell] / e["s_per_model_epoch"]
        out[lab] = e
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--carc", required=True)
    ap.add_argument("--h2", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--y", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    y = np.fromfile(a.y, "<f4").astype(np.float64)
    ref = load_run(Path(a.carc))
    p100_single = next(p[2] for p in ref["pools"] if p[0] == 0) / 300
    res: dict[str, Any] = dict(
        reference=dict(
            run="CARC 13353680 (P100)",
            device=ref["device"],
            best_epoch=ref["best_epoch"],
            best_val_qlike=ref["best_val"],
            val_qlike_e300=float(ref["vq"][-1]),
            TQ300_tune=float(ref["tq"]["tune"][-1]),
            single_model_s_per_epoch=p100_single,
            conc24_s_per_model_epoch=P100_EPOCH_COST,
        ),
        tolerance_rel=TOL,
        types={},
    )
    for kv in a.labels.split(","):
        lab, jid = kv.split("=")
        job = Path(a.h2) / jid
        runs: dict[str, dict[str, Any]] = {}
        t: dict[str, Any] = dict(job_id=jid, runs_txt=(job / "runs.txt").read_text().splitlines())
        for r in ("g1_r1", "g1_r2"):
            jp = job / r / "summary" / "jobs.csv"
            if jp.is_file() and pd.read_csv(jp)["id"].str.startswith("v2g1tune_").any():
                runs[r] = load_run(job / r)
            else:
                t[f"{r}_missing"] = "no tune job finished (crashed before its first pool phase ended)"
        for r, x in runs.items():
            t[f"{r}_info"] = dict(
                device=x["device"],
                complete=x["complete"],
                errors=x["log_errors"],
                twin_steps_identical=bool(x["steps"] == x["steps_refit"]) if x["complete"] else None,
            )
            gc = job / r / "g1_check.json"
            if gc.is_file():
                g = json.loads(gc.read_text())
                t[f"{r}_info"]["g1_check"] = {
                    k: g[k]
                    for k in (
                        "pass",
                        "steps_identical",
                        "in_sample_rel_dev",
                        "val_rel_dev",
                        "TQ0_trainer",
                        "TQ0_rel_dev",
                    )
                }
        if len(runs) == 2:
            t["r1_vs_r2"] = compare(runs["g1_r1"], runs["g1_r2"], y)
        for r in runs:
            t[f"{r}_vs_P100"] = compare(runs[r], ref, y)
        t["speed"] = speed(job, runs)
        if t["speed"].get("g1_single_model_s_per_epoch"):
            t["speed"]["g1_single_model_vs_P100"] = p100_single / t["speed"]["g1_single_model_s_per_epoch"]
        res["types"][lab] = t
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    for lab, t in res["types"].items():
        print(lab, t["job_id"])
        for ln in t["runs_txt"]:
            print("  run", ln)
        for k in ("g1_r1_info", "g1_r2_info"):
            if k in t:
                print(f"  {k}: {t[k]}")
        for k in ("r1_vs_r2", "g1_r1_vs_P100", "g1_r2_vs_P100"):
            if k in t:
                c = t[k]
                print(
                    f"  {k}: steps_ident {c['steps_identical']} first_diff_epoch {c['first_differing_epoch']} "
                    f"step0 {c['step0_loss']} valQ max rel {c['val_qlike_epoch_max_rel']:.2e} "
                    f"(ep {c['val_qlike_epoch_argmax']}; first > tol ep {c['val_qlike_first_epoch_over_tol']}) "
                    f"e300 rel {c['val_qlike_e300_rel']:.2e} best {c['best_epoch']} TQ300 {c['TQ300_tune']} "
                    f"rel {c['TQ300_tune_rel']:.2e} refit TQ300 rel {c.get('TQ300_refit_rel', float('nan')):.2e} "
                    f"twin valQ rel {c.get('twin_val_qlike_rel', float('nan')):.2e} "
                    f"fc max rel {c.get('twin_fc_max_rel', float('nan')):.2e} -> within 0.1% {c['within_0.1pct']}"
                )
        for k, v in t["speed"].items():
            print(f"  speed {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
