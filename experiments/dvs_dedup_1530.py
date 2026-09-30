"""C2 dense vs sparse at the 16:00 bar, re-run on the de-duplicated per-bar design with the
per-window column mask on every tree fit (checklist I8).

The first pass (experiments/dense_vs_sparse_1530.py -> results/dense_vs_sparse/) ran on the
per-bar design that still held the 12 session-edge columns (har_ma_k_x_open, all zero in a
one-bar-per-session series, and har_ma_k_x_close, byte-copies of har_ma_k at bar1600).
Commit 47f7f9c removed them (live_feasible 244 -> 232 columns, all_features 640 -> 628);
commit f9a19b6 (user decision 2026-09-29) gives every per-bar tree fit the linear arms'
identifiability rule: src/models/window_mask.window_keep on the fit's own training window
[t - W, t) (drop the constant columns and the byte-copies of an earlier kept column), with
TreeSHAP mapped back to all columns by scatter (0 for a dropped column).

Same methodology, the first pass's own functions (imported from dense_vs_sparse_1530 and run
unchanged; only the design, the stored references and the tree mask differ):

capture <bucket>   dense_vs_sparse_1530.capture: the design through the linear spec's own
                   executor call.  GATES: the re-run ridge equals the de-duplicated stored
                   ridge arm (results/linear_subsection_dedup/arms_hoffman2); with
                   DVS_OLD_WORK (the first pass's work dir) the new design equals the old one
                   minus exactly the 12 session-edge columns, bit for bit, same target.
refit              every refit, at most DVS_WORKERS (default 4) local processes:
                   * linear (ridge / lasso / elastic net on all columns, the screen and own
                     top-k sweeps, the PC1 collapse): dense_vs_sparse_1530._job unchanged
                     (their identifiability mask already drops constant and duplicated
                     columns, so nothing changes for them but the design);
                   * trees: the tree spec's make_model / contributions (specs/causal_tune_trees.py,
                     shipped configuration, refit every 10 sessions, one thread) in the spec's
                     walk with the per-window mask: LightGBM on all columns with TreeSHAP (the
                     sweep's reference and the density table), LightGBM on every screened
                     top-k subset (the mask applied to the subset's columns on each refit
                     window), XGBoost and random forest on all columns with TreeSHAP (density);
                     plus LightGBM on all columns WITHOUT the mask (the "de-dup, no mask"
                     reference).  Tree runs are split into the six 250-session blocks (every
                     refit is independent).  GATE: one run unchunked is bit-identical.
analyze            the first pass's scorer and measures on the new runs, the tables and
                   figures of results/dense_vs_sparse_dedup/, the before / after tables
                   (first pass vs de-dup + mask; "de-dup, no mask" where it exists) and
                   SUMMARY.md, written from this folder's own CSVs.
summary            SUMMARY.md only.

Environment: DVS_WORK (work dir; never committed), DVS_OLD_WORK (the first pass's work dir;
optional: the design gate and the paired new-minus-old intervals), DVS_WORKERS (default 4),
DVS_T10 (the stored unmasked T10 TreeSHAP, pulled by cluster/slurm/pull_dvs_dedup_carc.sh;
default $DVS_WORK/trees_t10).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "experiments", REPO / "notebooks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
OUT = REPO / "results" / "dense_vs_sparse_dedup"
OLD = REPO / "results" / "dense_vs_sparse"
os.environ.setdefault("DVS_WORK", str(OUT / "_work"))

import dense_vs_sparse_1530 as dvs  # noqa: E402
from src.models.window_mask import scatter, window_keep  # noqa: E402

WORK = dvs.WORK
OLD_WORK = Path(os.environ["DVS_OLD_WORK"]) if os.environ.get("DVS_OLD_WORK") else None
T10 = Path(os.environ.get("DVS_T10", str(WORK / "trees_t10")))
STORED_LIN = REPO / "results" / "linear_subsection_dedup" / "arms_hoffman2"
STORED_T10 = REPO / "results" / "linear_subsection_trees_dedup" / "t10"
TREE_SPEC = REPO / "specs" / "causal_tune_trees.py"
SEG, TW, TUNE_PER = dvs.SEG, dvs.TW, dvs.TUNE_PER
REFIT_EVERY = dvs.TREE_REFIT_EVERY
BUCKETS = dvs.BUCKETS
K_GRID = dvs.K_GRID
LIN_EST = dvs.LIN_EST
TREE_MODELS = dvs.TREE_MODELS
LABEL = dvs.LABEL
MAX_WORKERS = dvs.MAX_WORKERS
# commit 47f7f9c: the per-bar design without the session-edge columns
SESSION_EDGE_SUFFIX = ("_x_open", "_x_close")
N_SESSION_EDGE = 12
P_OLD = {"live_feasible": 244, "all_features": 640}
P_NEW = {"live_feasible": 232, "all_features": 628}
# the analysis reads the de-duplicated stored arms
dvs.STORED_LIN = STORED_LIN


# ============================================================================ capture
def capture(bucket: str) -> None:
    dvs.capture(
        bucket
    )  # GATE inside: re-run ridge == de-duplicated stored ridge (< 1e-8)
    dz = dvs.load_design(bucket)
    names = [str(v) for v in dz["names"]]
    gate: dict = dict(
        bucket=bucket, n_columns=len(names), capture_gate=float(dz["gate_rel"])
    )
    assert len(names) == P_NEW[bucket], (bucket, len(names))
    assert not any(n.endswith(SESSION_EDGE_SUFFIX) for n in names)
    if OLD_WORK is not None and (OLD_WORK / f"design_{bucket}.npz").is_file():
        zo = np.load(OLD_WORK / f"design_{bucket}.npz", allow_pickle=False)
        on = [str(v) for v in zo["names"]]
        keep = set(names)
        dropped = [n for n in on if n not in keep]
        assert len(on) == P_OLD[bucket], (bucket, len(on))
        assert len(dropped) == N_SESSION_EDGE, dropped
        assert all(n.endswith(SESSION_EDGE_SUFFIX) for n in dropped), dropped
        assert [n for n in on if n in keep] == names  # the other columns, same order
        idx = [on.index(n) for n in names]
        Xo = np.ascontiguousarray(zo["X"][:, idx])
        Xn = np.ascontiguousarray(dz["X"])
        gate.update(
            dropped=" ".join(dropped),
            X_bitwise_equal=bool(
                Xo.shape == Xn.shape
                and np.array_equal(Xo.view(np.uint64), Xn.view(np.uint64))
            ),
            y_bitwise_equal=bool(
                np.array_equal(
                    np.ascontiguousarray(zo["y"]).view(np.uint64),
                    np.ascontiguousarray(dz["y"]).view(np.uint64),
                )
            ),
            dates_equal=bool((zo["date"] == dz["date"]).all()),
            # the dropped columns in the old design: all-zero (x_open) / copies of har_ma_k (x_close)
            dropped_all_zero=int(
                sum((zo["X"][:, on.index(n)] == 0).all() for n in dropped)
            ),
            dropped_copy_of_har=int(
                sum(
                    np.array_equal(
                        zo["X"][:, on.index(n)],
                        zo["X"][:, on.index(n.replace("_x_close", ""))],
                    )
                    for n in dropped
                    if n.endswith("_x_close")
                )
            ),
        )
        assert (
            gate["X_bitwise_equal"] and gate["y_bitwise_equal"] and gate["dates_equal"]
        ), gate
    (WORK / f"capture_gate_{bucket}.json").write_text(
        json.dumps(gate, indent=1), encoding="utf-8"
    )
    print(f"GATE design {bucket}: {gate}", flush=True)


# ============================================================================ trees
_TNS: dict = {}


def tree_ns(model: str) -> dict:
    """The tree spec's model section (constants, make_model, contributions), run read-only
    for one model, exactly as dense_vs_sparse_1530.spec_ns does for LightGBM."""
    if model in _TNS:
        return _TNS[model]
    os.chdir(REPO)
    for v in (
        "REFIT_EVERY",
        "IMPORTANCE_EVERY",
        "WINDOW_MASK",
        "START",
        "END",
        "HALO",
        "HAR_LAGS",
        "HAR_BASE",
    ):
        os.environ.pop(f"HPC_KW_{v}", None)
        os.environ.pop(v, None)
    os.environ.update(
        {
            "HPC_KW_MODEL": model,
            "HPC_KW_EXOG_BUCKET": "live_feasible",
            "HPC_KW_SEGMENT": SEG,
            "HPC_KW_TRAIN_WIN": str(TW),
            "HPC_KW_LAG_SCOPE": "global",
            "SLURM_CPUS_PER_TASK": "1",
        }
    )
    src = TREE_SPEC.read_text(encoding="utf-8")
    a = src.index("import json\nimport time\nimport warnings")
    b = src.index("SIDE: dict = {}")
    ns: dict = {"__name__": f"causal_tune_trees_section_{model}", "os": os}
    exec(compile(src[a:b], str(TREE_SPEC), "exec"), ns)
    assert ns["MODEL"] == model and ns["TRAIN_WIN"] == TW
    assert ns["REFIT_EVERY"] == REFIT_EVERY and ns["N_THREADS"] == 1
    _TNS[model] = ns
    return ns


def walk_tree(
    model: str, Xb: np.ndarray, yb: np.ndarray, W: int, mask: bool, want_shap: bool
) -> dict:
    """The tree spec's walk (fit_predict_tree) on one block's rows [t0 - W, t0 + nb): refit every
    REFIT_EVERY rows on the W rows strictly before the anchor, predict the next rows.  With the
    mask, the fit and the prediction use only window_keep(X[t - W : t]) (the columns not
    constant and not a byte-copy of an earlier column on that training window); TreeSHAP is
    scattered back to all columns (0 for a dropped column)."""
    ns = tree_ns(model)
    nb = len(Xb) - W
    p = Xb.shape[1]
    preds = np.empty(nb)
    nkeep: list[int] = []
    shap = np.full((nb, p + 1), np.nan) if want_shap else None
    for j in range(0, nb, REFIT_EVERY):
        t = W + j
        k = min(REFIT_EVERY, nb - j)
        m = ns["make_model"]()
        if mask:
            keep = window_keep(Xb[t - W : t])
            Xf = np.ascontiguousarray(Xb[t - W : t][:, keep])
            Xp = np.ascontiguousarray(Xb[t : t + k][:, keep])
        else:  # exactly dense_vs_sparse_1530.run_lgbm_blocks / the spec's walk
            keep = np.arange(p)
            Xf, Xp = Xb[t - W : t], Xb[t : t + k]
        m.fit(Xf, yb[t - W : t])
        preds[j : j + k] = np.asarray(m.predict(Xp), dtype=np.float64)
        nkeep.append(len(keep))
        if shap is not None:
            c = np.asarray(ns["contributions"](m, Xp), dtype=np.float64)
            shap[j : j + k] = scatter(c, keep, p) if mask else c
    return dict(pred=preds, nkeep=np.array(nkeep, dtype=np.int64), shap=shap)


def tree_jobs(where: str = "local") -> list[dict]:
    """The tree runs, split as the first pass split them: the LightGBM sweep and its
    all-column reference run locally (the first pass's environment, so the before / after
    differs only by the design and the mask); the density table's TreeSHAP comes from the
    cluster (the first pass read the tree campaign's stored cluster TreeSHAP; here the
    masked T10 walk of all three models, run on the cluster)."""
    J: list[dict] = []
    for b in BUCKETS:
        if where == "carc":
            for m in TREE_MODELS:  # all columns, masked, TreeSHAP at every refit
                J.append(
                    dict(bucket=b, model=m, rule="full", k="all", mask=True, shap=True)
                )
            continue
        # all columns, masked (the sweep's reference; TreeSHAP kept as a cross-check)
        J.append(
            dict(bucket=b, model="lgbm", rule="full", k="all", mask=True, shap=True)
        )
        # all columns, NO mask: the "de-dup, no mask" reference
        J.append(
            dict(bucket=b, model="lgbm", rule="full", k="all", mask=False, shap=False)
        )
        for k in K_GRID:  # the screened top-k subsets, masked
            J.append(
                dict(bucket=b, model="lgbm", rule="screen", k=k, mask=True, shap=False)
            )
    return J


def tree_tag(job: dict) -> str:
    m = job["model"] if job["mask"] else f"{job['model']}_nomask"
    return f"{m}_all" if job["rule"] == "full" else f"{m}_screen_k{job['k']}"


def n_blocks(bucket: str) -> int:
    dz = np.load(WORK / f"design_{bucket}.npz", allow_pickle=False)
    return len(dvs.block_starts(len(dz["X"]) - int(dz["W"])))


def chunk_path(job: dict, b: int) -> Path:
    if b < 0:
        return WORK / "gate" / job["bucket"] / f"{tree_tag(job)}_unchunked.npz"
    return WORK / "chunks" / job["bucket"] / tree_tag(job) / f"b{b}.npz"


def _tree_chunk(job: dict, b: int) -> dict:
    """One tree run on one 250-session block (b >= 0), or on every block in one process
    (b = -1: the chunking gate).  Runs in a worker process."""
    t0 = time.time()
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    dz = dvs.load_design(job["bucket"])
    X, y, W = dz["X"], dz["y"], int(dz["W"])
    starts = dvs.block_starts(len(X) - W)
    which = range(len(starts)) if b < 0 else [b]
    preds, nk, sh, cols = [], [], [], []
    for bi in which:
        i0, i1 = starts[bi]
        if (
            job["rule"] == "screen"
        ):  # the first pass's causal screen (dvs._job, rule "screen")
            Xw, yw = X[i0 : i0 + W], y[i0 : i0 + W]
            c = dvs.top_k(
                dvs.abs_corr_with(Xw, yw), dvs.identifiable(Xw), int(job["k"])
            )
            Xb = np.ascontiguousarray(X[i0 : W + i1][:, c])
            cols.append(np.pad(c, (0, int(job["k"]) - len(c)), constant_values=-1))
        else:
            Xb = np.ascontiguousarray(X[i0 : W + i1])
        r = walk_tree(
            job["model"],
            Xb,
            y[i0 : i0 + W + (i1 - i0)],
            W,
            bool(job["mask"]),
            bool(job["shap"]),
        )
        preds.append(r["pred"])
        nk.append(r["nkeep"])
        if r["shap"] is not None:
            sh.append(r["shap"])
    out = dict(pred=np.concatenate(preds), nkeep=np.concatenate(nk))
    if sh:
        out["shap"] = np.concatenate(sh)
    if cols:
        out["cols"] = np.array(cols)
    f = chunk_path(job, b)
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f.stem + ".tmp.npz")
    np.savez_compressed(tmp, **out, sec=time.time() - t0)
    os.replace(tmp, f)
    return dict(
        kind="tree",
        tag=tree_tag(job),
        bucket=job["bucket"],
        block=b,
        sec=time.time() - t0,
    )


def assemble(job: dict) -> None:
    """The six block chunks of one tree run -> the run file the analysis reads (the layout of
    dense_vs_sparse_1530._job: pred, params, cols) + the TreeSHAP file of a density run."""
    bucket, tag = job["bucket"], tree_tag(job)
    nbk = n_blocks(bucket)
    parts = []
    for b in range(nbk):
        z = np.load(chunk_path(job, b), allow_pickle=False)
        parts.append({k: z[k] for k in z.files})
    dz = dvs.load_design(bucket)
    W = int(dz["W"])
    pred = np.concatenate([p["pred"] for p in parts])
    assert len(pred) == len(dz["X"]) - W
    run: dict = dict(
        pred=pred,
        nkeep=np.concatenate([p["nkeep"] for p in parts]),
        params=json.dumps(tree_ns(job["model"])["PARAMS"][job["model"]]),
        mask=bool(job["mask"]),
        sec=float(sum(float(p["sec"]) for p in parts)),
    )
    if "cols" in parts[0]:
        run["cols"] = np.concatenate([p["cols"] for p in parts])
    d = WORK / "runs" / bucket
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / f"{tag}.npz", **run)
    if job["shap"]:
        sh = np.concatenate([p["shap"] for p in parts])
        assert np.isfinite(sh).all()
        dd = WORK / "trees_mask" / bucket
        dd.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            dd / f"{job['model']}.npz",
            shap=sh,
            pred_adj=pred,
            feature_names=dz["names"],
            date=dz["date"],
            nkeep=run["nkeep"],
        )


def run_done(job: dict) -> bool:
    return (WORK / "runs" / job["bucket"] / f"{tree_tag(job)}.npz").is_file()


# chunk cost for the scheduling order (longest first): model factor x columns seen
_COST = {"rf": 12.0, "lgbm": 1.0, "xgb": 0.7}
GATE_JOB = dict(
    bucket="live_feasible", model="lgbm", rule="screen", k=4, mask=True, shap=False
)


def refit() -> None:
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"

    def lin_done(j: dict) -> bool:
        return (WORK / "runs" / j["bucket"] / f"{dvs.job_tag(j)}.npz").is_file()

    lin_full = [
        j for j in dvs.jobs_for("full") if j["model"] != "lgbm" and not lin_done(j)
    ]
    lin_sweep = [
        j for j in dvs.jobs_for("sweep") if j["model"] != "lgbm" and not lin_done(j)
    ]
    own = [j for j in lin_sweep if j["rule"] == "own"]
    rest = [j for j in lin_sweep if j["rule"] != "own"]
    tj = [j for j in tree_jobs() if not run_done(j)]
    chunks = [
        (j, b)
        for j in tj
        for b in range(n_blocks(j["bucket"]))
        if not chunk_path(j, b).is_file()
    ]
    p_of = {b: P_NEW[b] for b in BUCKETS}

    def cost(jb: tuple[dict, int]) -> float:
        j, _ = jb
        width = p_of[j["bucket"]] if j["rule"] == "full" else int(j["k"])
        return _COST[j["model"]] * (width + 20)

    chunks.sort(key=cost, reverse=True)
    gate = [] if chunk_path(GATE_JOB, -1).is_file() else [(GATE_JOB, -1)]
    print(
        f"refit: {len(lin_full)} linear full, {len(rest) + len(own)} linear sweep, "
        f"{len(chunks)} tree chunks of {len(tj)} tree runs, {len(gate)} gate run; "
        f"{MAX_WORKERS} workers",
        flush=True,
    )
    log: list[dict] = []
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs: dict = {}
        for j in lin_full:
            futs[ex.submit(dvs._job, j)] = j
        full_futs = set(futs)
        for jb in gate + chunks:
            futs[ex.submit(_tree_chunk, *jb)] = jb
        for j in rest:
            futs[ex.submit(dvs._job, j)] = j
        own_sent = not own
        pending = set(futs)
        while pending:
            fin, pending = wait(pending, return_when=FIRST_COMPLETED)
            for f in fin:
                try:
                    r = f.result()
                    r = {
                        k: v
                        for k, v in r.items()
                        if k in ("tag", "bucket", "block", "sec", "kind")
                    }
                    log.append(r)
                    print(
                        f"  done {r.get('bucket')} {r.get('tag')} "
                        f"{'' if r.get('block') is None else 'block ' + str(r['block'])} "
                        f"{r['sec']:.0f}s  [{time.time() - t_start:.0f}s]",
                        flush=True,
                    )
                except Exception as e:  # noqa: BLE001 -- report and keep the others running
                    print(f"  FAILED {futs[f]}: {type(e).__name__}: {e}", flush=True)
            if not own_sent and full_futs.isdisjoint(pending):
                for j in own:
                    nf = ex.submit(dvs._job, j)
                    futs[nf] = j
                    pending.add(nf)
                own_sent = True
    for j in tree_jobs():
        if not run_done(j) and all(
            chunk_path(j, b).is_file() for b in range(n_blocks(j["bucket"]))
        ):
            assemble(j)
    wall = time.time() - t_start
    prev = WORK / "refit_log.json"
    old = json.loads(prev.read_text(encoding="utf-8")) if prev.is_file() else []
    old.append(
        dict(
            wall_sec=wall,
            workers=MAX_WORKERS,
            cpu_sec=float(sum(r["sec"] for r in log)),
            n_jobs=len(log),
            jobs=log,
        )
    )
    prev.write_text(json.dumps(old, indent=1), encoding="utf-8")
    print(
        f"refit done: {len(log)} jobs, {sum(r['sec'] for r in log) / 3600:.2f} CPU-h, "
        f"wall {wall / 60:.1f} min",
        flush=True,
    )


def file_md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


# ============================================================================ cluster
# the canary: the last (shortest, 219-session) block of live_feasible for all three models,
# so every model path (incl. random forest's shap import) runs on real data first
CANARY_BLOCK = 5


def _versions() -> dict:
    out = {}
    for mod in ("numpy", "pandas", "sklearn", "lightgbm", "xgboost", "shap"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001 -- record an absent library, do not fail on it
            out[mod] = "absent"
    return out


def carc(mode: str) -> None:
    """On the cluster (cluster/slurm/dvs_dedup.sbatch): the masked T10 walk of LightGBM,
    XGBoost and random forest on all columns with TreeSHAP at every refit, the density
    table's inputs.  mode 'canary': block CANARY_BLOCK of live_feasible for the three
    models, then CANARY_OK; mode 'fleet': every other block chunk in a pool of
    SLURM_CPUS_PER_TASK single-threaded workers, then the runs are assembled and DONE is
    written.  A chunk already on disk is skipped (a resubmitted task resumes)."""
    import platform
    from concurrent.futures import as_completed

    nproc = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    jobs = tree_jobs("carc")
    if mode == "canary":
        todo = [(j, CANARY_BLOCK) for j in jobs if j["bucket"] == "live_feasible"]
    elif mode == "fleet":
        todo = [(j, b) for j in jobs for b in range(n_blocks(j["bucket"]))]
    elif mode == "gate":
        # this walk WITHOUT the mask, on the cluster, vs the stored unmasked T10 run of the
        # tree campaign (same platform): block CANARY_BLOCK of live_feasible, three models
        todo = [
            (dict(j, mask=False, shap=False), CANARY_BLOCK)
            for j in jobs
            if j["bucket"] == "live_feasible"
        ]
    else:
        raise SystemExit(f"carc mode must be canary, fleet or gate, got {mode!r}")
    todo = [jb for jb in todo if not chunk_path(*jb).is_file()]
    todo.sort(
        key=lambda jb: _COST[jb[0]["model"]] * (P_NEW[jb[0]["bucket"]] + 20),
        reverse=True,
    )
    print(f"carc {mode}: {len(todo)} chunks, pool of {nproc}", flush=True)
    t0 = time.time()
    log: list[dict] = []
    bad = 0
    if todo:
        with ProcessPoolExecutor(max_workers=max(1, min(nproc, len(todo)))) as ex:
            futs = {ex.submit(_tree_chunk, *jb): jb for jb in todo}
            for f in as_completed(futs):
                try:
                    r = f.result()
                    log.append(r)
                    print(
                        f"  done {r['bucket']} {r['tag']} block {r['block']} {r['sec']:.0f}s "
                        f"[{time.time() - t0:.0f}s]",
                        flush=True,
                    )
                except Exception as e:  # noqa: BLE001 -- report, fail the task at the end
                    bad += 1
                    print(f"  FAILED {futs[f]}: {type(e).__name__}: {e}", flush=True)
    info: dict = dict(
        mode=mode,
        host=platform.node(),
        pool=nproc,
        wall_sec=time.time() - t0,
        cpu_sec=float(sum(r["sec"] for r in log)),
        n_chunks=len(log),
        failed=bad,
        versions=_versions(),
        tree_spec_md5=file_md5(TREE_SPEC),
        jobs=log,
    )
    (WORK / f"carc_log_{mode}.json").write_text(
        json.dumps(info, indent=1), encoding="utf-8"
    )
    if bad:
        raise SystemExit(f"{bad} chunks failed")
    if mode == "canary":
        (WORK / "CANARY_OK").write_text(json.dumps(info["versions"]), encoding="utf-8")
    elif mode == "gate":
        dz = dvs.load_design("live_feasible")
        i0, i1 = dvs.block_starts(len(dz["X"]) - int(dz["W"]))[CANARY_BLOCK]
        res: dict = {}
        for j, b in todo:
            st = pd.read_csv(
                STORED_T10
                / "live_feasible"
                / SEG
                / j["model"]
                / f"tw{TW}"
                / "causal_tune_trees"
                / j["model"]
                / "live_feasible"
                / f"results_{SEG}.csv"
            )
            z = np.load(chunk_path(j, b))
            ref = st["pred_adj"].to_numpy(float)[i0:i1]
            res[j["model"]] = dict(
                rows=int(i1 - i0),
                first_row=int(i0),
                bitwise_equal=bool(np.array_equal(z["pred"], ref)),
                max_rel=float(np.max(_rel(z["pred"], ref))),
            )
        (WORK / "gate_nomask.json").write_text(
            json.dumps(res, indent=1), encoding="utf-8"
        )
        print(f"gate: {res}", flush=True)
    else:
        for j in jobs:
            assemble(j)
        (WORK / "DONE").write_text(json.dumps(info["versions"]), encoding="utf-8")
    print(
        f"carc {mode} finished: {len(log)} chunks, {info['cpu_sec'] / 3600:.2f} CPU-h, "
        f"wall {info['wall_sec'] / 60:.1f} min",
        flush=True,
    )


# ============================================================================ analyze
V_OLD = "first pass (old design)"
V_NEW = "de-dup + mask"
V_NOMASK = "de-dup, no mask"
TREES_CARC = WORK / "trees_mask_carc"  # the cluster's masked TreeSHAP, pulled back
OLD_TREES = REPO / "results" / "linear_subsection_trees"  # first pass: stored trees
OLD_LIN = (
    REPO / "results" / "linear_subsection" / "arms_hoffman2"
)  # first pass: stored arms


def _model_section(src: str) -> str:
    """The tree spec's model definition: the parameter block and make_model .. contributions."""
    a = src[src.index("LGBM_PARAMS: dict = {") : src.index("PROVENANCE = {")]
    b = src[src.index("def make_model():") : src.index("SIDE: dict = {}")]
    return a + b


def spec_model_section_check(timing: dict) -> dict:
    """The cluster runs record the md5 of the tree spec they executed; if the local spec has
    moved on since (another campaign's commit), check that the model section is unchanged by
    finding that version in git history (by md5) and comparing the sections."""
    import subprocess

    local = TREE_SPEC.read_text(encoding="utf-8")
    out: dict = dict(local_md5=file_md5(TREE_SPEC))
    ran = {
        v.get("tree_spec_md5")
        for k, v in timing.items()
        if k.startswith("cluster_") and "tree_spec_md5" in v
    }
    for md5 in sorted(m for m in ran if m):
        out[f"cluster_md5_{md5[:12]}"] = (
            "same file" if md5 == out["local_md5"] else "not found in git log"
        )
        if md5 == out["local_md5"]:
            continue
        revs = subprocess.run(
            ["git", "log", "--format=%H", "--", "specs/causal_tune_trees.py"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
        for rev in revs:
            txt = subprocess.run(
                ["git", "show", f"{rev}:specs/causal_tune_trees.py"],
                cwd=REPO,
                capture_output=True,
                check=True,
            ).stdout
            # the working tree may carry CRLF line endings (Windows checkout), as shipped
            crlf = txt.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
            if md5 in (hashlib.md5(txt).hexdigest(), hashlib.md5(crlf).hexdigest()):
                same = _model_section(
                    txt.decode("utf-8").replace("\r\n", "\n")
                ) == _model_section(local)
                out[f"cluster_md5_{md5[:12]}"] = (
                    f"commit {rev[:7]}; model section (parameters, make_model, contributions) "
                    + (
                        "identical to the local spec"
                        if same
                        else "DIFFERS from the local spec"
                    )
                )
                break
    return out


def tree_contrib(bucket: str, model: str, dz: dict, variant: str) -> dict | None:
    """Per forecast row and design column, a tree's TreeSHAP contribution: the masked T10
    walk run on the cluster (V_NEW, dropped columns scattered to 0) or the stored unmasked
    T10 run of the tree campaign (V_NOMASK)."""
    if variant == V_NEW:
        f = TREES_CARC / bucket / f"{model}.npz"
    else:
        f = T10 / bucket / model / f"trees_{SEG}.npz"
    if not f.is_file():
        return None
    z = np.load(f, allow_pickle=True)
    names = [str(v) for v in z["feature_names"]]
    assert names == [str(v) for v in dz["names"]], (bucket, model, variant)
    assert (np.asarray(z["date"]).astype(str) == dz["date"]).all()
    sh = np.asarray(z["shap"], dtype=np.float64)
    assert np.isfinite(sh).all(), (bucket, model, variant)
    pred = np.asarray(z["pred_adj"], dtype=np.float64)
    gap = float(np.max(np.abs(sh[:, :-1].sum(1) + sh[:, -1] - pred) / np.abs(pred)))
    out = dict(phi=sh[:, :-1], pred=pred, gap=gap)
    if "nkeep" in z.files:
        out["nkeep"] = np.asarray(z["nkeep"])
    return out


def _rel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(a, float) / np.asarray(b, float) - 1.0)


def analyze() -> None:  # noqa: C901 - one linear report, as the first pass's
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import score_linear_subsection as base

    OUT.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    NUM: dict = {"buckets": {}}

    def say(msg: str) -> None:
        print(msg, flush=True)
        lines.append(msg)

    dk = dvs.deck_frame()
    gate_rows: list[dict] = []
    sweep_rows: list[dict] = []
    gap_rows: list[dict] = []
    dens_rows: list[dict] = []
    group_rows: list[dict] = []
    conc_rows: list[dict] = []
    coll_rows: list[dict] = []
    pick_rows: list[dict] = []
    mask_rows: list[dict] = []
    ref_rows: list[dict] = []
    chg_rows: list[dict] = []
    gchg_rows: list[dict] = []
    curves: dict = {}
    sweep_fig: dict = {}
    old_fig: dict = {}

    def gate(bucket: str, check: str, **kw) -> None:
        gate_rows.append(dict(bucket=bucket, check=check, **kw))

    for bucket in BUCKETS:
        dz = dvs.load_design(bucket)
        names = [str(v) for v in dz["names"]]
        W = int(dz["W"])
        p = len(names)
        NB: dict = {
            "n_columns": p,
            "n_forecasts": int(len(dz["X"]) - W),
            "capture_gate": float(dz["gate_rel"]),
        }
        say(
            f"[{bucket}] {p} design columns; {len(dz['X']) - W} forecast sessions "
            f"{dz['date'][0][:10]} .. {dz['date'][-1][:10]} after a {W}-session window; capture gate "
            f"{float(dz['gate_rel']):.1e}"
        )
        # ---------------- gates: the design
        cg = json.loads(
            (WORK / f"capture_gate_{bucket}.json").read_text(encoding="utf-8")
        )
        gate(
            bucket,
            "design re-run through the executor: ridge vs the de-duplicated stored arm",
            n=NB["n_forecasts"],
            max_rel=cg["capture_gate"],
            passed=cg["capture_gate"] < 1e-8,
        )
        if "X_bitwise_equal" in cg:
            ok = cg["X_bitwise_equal"] and cg["y_bitwise_equal"] and cg["dates_equal"]
            gate(
                bucket,
                f"new design = the first pass's design minus the {N_SESSION_EDGE} session-edge "
                "columns (X, target, dates bit for bit)",
                n=p,
                passed=bool(ok),
                note=f"dropped {cg['dropped_all_zero']} all-zero x_open, {cg['dropped_copy_of_har']} "
                "x_close byte-copies of har_ma_k",
            )
            NB["design_gate"] = cg
        # ---------------- gates: full linear refits vs the stored de-duplicated arms
        for m in LIN_EST:
            r = dvs.load_run(bucket, f"{m}_all")
            if r is None:
                say(f"  missing full run {m}")
                continue
            s = pd.read_csv(STORED_LIN / bucket / m / f"tw{TW}" / f"results_{SEG}.csv")
            assert (s["date"].astype(str).to_numpy() == dz["date"]).all()
            rel = _rel(r["pred"], s["pred_adj"].to_numpy(float))
            bad = np.flatnonzero(rel > 1e-6)
            gate(
                bucket,
                f"full {LABEL[m]} refit vs the de-duplicated stored arm",
                n=len(rel),
                max_rel=float(rel.max()),
                rows_above_1e6=int(len(bad)),
                first_bad=int(bad.min()) if len(bad) else -1,
                last_bad=int(bad.max()) if len(bad) else -1,
                passed=not len(bad),
                note=(
                    "a warm-homotopy float path in one tune period: see the next rows"
                    if len(bad)
                    else np.nan
                ),
            )
            so = OLD_LIN / bucket / m / f"tw{TW}" / f"results_{SEG}.csv"
            if so.is_file():
                rs = _rel(
                    s["pred_adj"].to_numpy(float),
                    pd.read_csv(so)["pred_adj"].to_numpy(float),
                )
                gate(
                    bucket,
                    f"the de-duplicated stored {LABEL[m]} arm vs the first pass's stored arm",
                    n=len(rs),
                    max_rel=float(rs.max()),
                    rows_above_1e6=int((rs > 1e-6).sum()),
                    note="the stored arms were fitted on different machines (I1)",
                )
            if (
                OLD_WORK is not None
                and (OLD_WORK / "runs" / bucket / f"{m}_all.npz").is_file()
            ):
                ro = _rel(
                    r["pred"],
                    np.load(OLD_WORK / "runs" / bucket / f"{m}_all.npz")["pred"],
                )
                gate(
                    bucket,
                    f"this {LABEL[m]} refit vs the first pass's refit on this machine (old design)",
                    n=len(ro),
                    max_rel=float(ro.max()),
                    rows_above_1e6=int((ro > 1e-6).sum()),
                    passed=bool(ro.max() < 1e-8),
                )
            if len(bad):
                # the first pass's check: is it the block restarts, or the spec itself?
                cache = WORK / f"gate_continuous_{bucket}_{m}.json"
                if not cache.is_file():
                    ns = dvs.spec_ns("linear")
                    Base = ns["RollingTunedLinear"]
                    Base.grid = ns["ESTIMATOR_GRIDS"][m]
                    bt = ns["MultiStageBacktest"](
                        residualizer=ns["IdentityResidualizer"](),
                        regressor_factory=Base,
                        refit_frequency=1,
                    )
                    pc = np.asarray(bt.run(dz["X"], dz["y"], W, desc="gate"), float)
                    rc = _rel(pc, s["pred_adj"].to_numpy(float))
                    cache.write_text(
                        json.dumps(
                            dict(
                                max_rel=float(rc.max()),
                                rows=int((rc > 1e-6).sum()),
                                vs_blocks=float(np.max(_rel(r["pred"], pc))),
                            )
                        )
                    )
                cgc = json.loads(cache.read_text())
                gate(
                    bucket,
                    f"spec continuous run of {LABEL[m]} here vs the stored arm",
                    n=len(rel),
                    max_rel=cgc["max_rel"],
                    rows_above_1e6=cgc["rows"],
                    note=f"this script's block run vs that continuous run {cgc['vs_blocks']:.1e}",
                )
            say(
                f"  GATE {LABEL[m]} full refit vs stored: max rel diff {rel.max():.1e}, rows > 1e-6: {len(bad)}"
                + (f" (forecast rows {bad.min()}..{bad.max()})" if len(bad) else "")
            )
        # ---------------- gates: trees
        t10csv = (
            STORED_T10
            / bucket
            / SEG
            / "lgbm"
            / f"tw{TW}"
            / "causal_tune_trees"
            / "lgbm"
            / bucket
            / f"results_{SEG}.csv"
        )
        st10 = pd.read_csv(t10csv)
        assert (st10["date"].astype(str).to_numpy() == dz["date"]).all()
        rn = dvs.load_run(bucket, "lgbm_nomask_all")
        rm = dvs.load_run(bucket, "lgbm_all")
        if rn is not None:
            rel = _rel(rn["pred"], st10["pred_adj"].to_numpy(float))
            gate(
                bucket,
                "LightGBM all columns, no mask, local vs the stored T10 run (cluster)",
                n=len(rel),
                max_rel=float(rel.max()),
                rows_above_1e6=int((rel > 1e-6).sum()),
                note=(
                    "not the platform: this walk gives the same LightGBM forecasts locally and on "
                    "the cluster (next rows)"
                ),
            )
        carc_lgbm = tree_contrib(bucket, "lgbm", dz, V_NEW)
        if rm is not None and carc_lgbm is not None:
            rel = _rel(rm["pred"], carc_lgbm["pred"])
            gate(
                bucket,
                "LightGBM all columns, masked, local vs the cluster's masked T10 walk",
                n=len(rel),
                max_rel=float(rel.max()),
                rows_above_1e6=int((rel > 1e-6).sum()),
                passed=bool(np.array_equal(rm["pred"], carc_lgbm["pred"])),
                note="bit-identical: the sweep's reference and the density table's LightGBM are one forecast",
            )
            km = rm["nkeep"]
            kc = carc_lgbm.get("nkeep")
            gate(
                bucket,
                "per-window mask: kept columns per refit, local = cluster",
                n=len(km),
                passed=bool(kc is not None and np.array_equal(km, kc)),
            )
        gj = TREES_CARC / "gate_nomask.json"
        if bucket == "live_feasible" and gj.is_file():
            gres = json.loads(gj.read_text(encoding="utf-8"))
            for m in TREE_MODELS:
                if m in gres:
                    g_ = gres[m]
                    gate(
                        bucket,
                        f"{LABEL[m]}, this walk without the mask on the cluster vs the stored T10 run "
                        f"(same cluster; forecast rows {g_['first_row']}..{g_['first_row'] + g_['rows'] - 1})",
                        n=g_["rows"],
                        max_rel=g_["max_rel"],
                        passed=g_["max_rel"] < 1e-12,
                        note="the stored run is read from CSV text (15-16 significant digits)",
                    )
            fl = WORK / "chunks" / bucket / "lgbm_nomask_all" / f"b{CANARY_BLOCK}.npz"
            fc = TREES_CARC / "lgbm_nomask_all" / f"b{CANARY_BLOCK}.npz"
            if fl.is_file() and fc.is_file():
                zl, zc = np.load(fl)["pred"], np.load(fc)["pred"]
                gate(
                    bucket,
                    f"LightGBM, this walk without the mask, local vs the cluster (block {CANARY_BLOCK}, bit for bit)",
                    n=len(zl),
                    max_rel=float(np.max(_rel(zl, zc))),
                    passed=bool(np.array_equal(zl, zc)),
                )
        if bucket == GATE_JOB["bucket"]:
            gu = chunk_path(GATE_JOB, -1)
            rc_ = dvs.load_run(bucket, tree_tag(GATE_JOB))
            if gu.is_file() and rc_ is not None:
                zu = np.load(gu)
                gate(
                    bucket,
                    f"tree run in six block chunks = one unchunked run ({tree_tag(GATE_JOB)}, bit for bit)",
                    n=len(zu["pred"]),
                    max_rel=float(np.max(np.abs(zu["pred"] - rc_["pred"]))),
                    passed=bool(
                        np.array_equal(zu["pred"], rc_["pred"])
                        and np.array_equal(zu["cols"], rc_["cols"])
                        and np.array_equal(zu["nkeep"], rc_["nkeep"])
                    ),
                )
        # the per-window mask: kept columns per refit
        mruns = [
            ("local", t) for t in ["lgbm_all", *[f"lgbm_screen_k{k}" for k in K_GRID]]
        ]
        for where, tag in mruns:
            r = dvs.load_run(bucket, tag)
            if r is None or "nkeep" not in r:
                continue
            offered = p if tag.endswith("_all") else int(tag.rsplit("k", 1)[1])
            nk = r["nkeep"]
            mask_rows.append(
                dict(
                    bucket=bucket,
                    run=f"LightGBM {tag.replace('lgbm_', '').replace('_', ' ')}",
                    where=where,
                    n_refits=len(nk),
                    columns_offered=offered,
                    kept_min=int(nk.min()),
                    kept_median=float(np.median(nk)),
                    kept_max=int(nk.max()),
                    refits_with_a_dropped_column=int((nk < offered).sum()),
                )
            )
        for m in TREE_MODELS:
            c = tree_contrib(bucket, m, dz, V_NEW)
            if c is not None and "nkeep" in c:
                nk = c["nkeep"]
                mask_rows.append(
                    dict(
                        bucket=bucket,
                        run=f"{LABEL[m]} all (density, TreeSHAP)",
                        where="cluster",
                        n_refits=len(nk),
                        columns_offered=p,
                        kept_min=int(nk.min()),
                        kept_median=float(np.median(nk)),
                        kept_max=int(nk.max()),
                        refits_with_a_dropped_column=int((nk < p).sum()),
                    )
                )

        # ---------------- A1 signal density
        Mser, gser, _stems = dvs.series_matrix(names)
        NB["n_series"] = len(gser)
        deck_rows = np.asarray(
            pd.DatetimeIndex(pd.to_datetime(dz["date"])).normalize().isin(dk.index)
        )
        curves[bucket] = {}
        contrib: dict = {}
        for m in (*LIN_EST, *TREE_MODELS, *(f"{t}_nomask" for t in TREE_MODELS)):
            if m in LIN_EST:
                c = dvs.contributions(bucket, m, dz)
                variant = V_NEW
            elif m in TREE_MODELS:
                c = tree_contrib(bucket, m, dz, V_NEW)
                variant = V_NEW
            else:
                c = tree_contrib(bucket, m.replace("_nomask", ""), dz, V_NOMASK)
                variant = V_NOMASK
            if c is None:
                say(f"  density: {m} missing")
                continue
            contrib[m] = c
            lab = LABEL[m.replace("_nomask", "")]
            gate(
                bucket,
                f"{lab} contributions sum to the forecast ({variant})",
                n=len(c["pred"]),
                max_rel=c["gap"],
                passed=c["gap"] < 1e-5,
            )
            assert c["gap"] < 1e-5, (bucket, m, c["gap"])
            for level, phi in (("column", c["phi"]), ("series", c["phi"] @ Mser)):
                for smp, rows in (
                    ("all forecasts", np.ones(len(phi), bool)),
                    ("deck days", deck_rows),
                ):
                    d_ = dvs.density(phi[rows])
                    if (
                        level == "column"
                        and smp == "all forecasts"
                        and variant == V_NEW
                    ):
                        curves[bucket][m] = d_["r2_curve"]
                    blk = []
                    if smp == "all forecasts":
                        for i0, i1 in dvs.block_starts(len(phi)):
                            if i1 - i0 >= 100:
                                blk.append(dvs.density(phi[i0:i1])["k80"])
                    dens_rows.append(
                        dict(
                            bucket=bucket,
                            model=lab,
                            variant=variant,
                            level=level,
                            sample=smp,
                            n_rows=int(rows.sum()),
                            **{k: v for k, v in d_.items() if k != "r2_curve"},
                            block_k80_median=float(np.median(blk)) if blk else np.nan,
                            block_k80_min=int(min(blk)) if blk else -1,
                            block_k80_max=int(max(blk)) if blk else -1,
                        )
                    )
            if m in LIN_EST:
                nz = (np.abs(c["th"][:, :-1]) > 0).sum(1)
                NB[f"nonzero_weights_{m}"] = [
                    int(nz.min()),
                    float(np.median(nz)),
                    int(nz.max()),
                ]

        # ---------------- A3 groups of correlated inputs (descriptive partition, forecast rows)
        Xo = dz["X"][W:]
        keep = dvs.identifiable(Xo)
        groups = dvs.corr_groups(Xo, keep)
        gid = np.full(p, -1)
        for g_i, g in enumerate(groups):
            gid[g] = g_i
        Mg = np.zeros((p, len(groups)))
        for j in range(p):
            if gid[j] >= 0:
                Mg[j, gid[j]] = 1.0
        sizes = np.array([len(g) for g in groups])
        NB["groups"] = dict(
            n_groups=len(groups),
            n_multi=int((sizes > 1).sum()),
            largest=int(sizes.max()),
            columns_in_multi=int(sizes[sizes > 1].sum()),
            n_kept=int(keep.sum()),
        )
        say(
            f"  groups (|corr| > {dvs.CORR_GROUP}, complete linkage, the {len(Xo)} forecast rows): "
            f"{len(groups)} groups of the {int(keep.sum())} identifiable columns, "
            f"{int((sizes > 1).sum())} with 2+ columns (largest {sizes.max()}; "
            f"{int(sizes[sizes > 1].sum())} columns sit in multi-column groups)"
        )
        gshare: dict = {}
        for m, c in contrib.items():
            G = c["phi"] @ Mg
            F = c["phi"].sum(1)
            vshare = (
                np.array([np.cov(G[:, g], F, ddof=0)[0, 1] for g in range(G.shape[1])])
                / F.var()
            )
            ashare = np.abs(G).mean(0) / np.abs(G).mean(0).sum()
            gshare[m] = (vshare, ashare)
            mabs = np.abs(c["phi"]).mean(0)
            conc_rows.append(
                dict(
                    bucket=bucket,
                    model=LABEL[m.replace("_nomask", "")],
                    variant=V_NOMASK if m.endswith("_nomask") else V_NEW,
                    n_groups=len(groups),
                    neff_groups=float(1.0 / (ashare**2).sum()),
                    neff_columns=float(1.0 / ((mabs / mabs.sum()) ** 2).sum()),
                    top_group_var_share=float(vshare.max()),
                    top_group=names[groups[int(np.argmax(vshare))][0]],
                )
            )
        for g_i, g in enumerate(groups):
            row = dict(
                bucket=bucket,
                group=g_i,
                n_columns=len(g),
                members=" ".join(names[j] for j in g),
            )
            for m, (vs, as_) in gshare.items():
                row[f"var_share_{m}"] = float(vs[g_i])
                row[f"abs_share_{m}"] = float(as_[g_i])
            group_rows.append(row)

        # ---------------- score every run with the research scorer (one panel, paired draws)
        tags = [f"{m}_all" for m in (*LIN_EST, "lgbm")]
        for rule in ("screen", "own"):
            for m in dvs.SWEEP_MODELS if rule == "screen" else ("ridge", "reclasso"):
                tags += [f"{m}_{rule}_k{k}" for k in K_GRID]
        tags += ["ridge_collapse", "reclasso_collapse"]
        extra = ["lgbm_nomask_all"]
        runs: dict[str, dict] = {}
        for t in tags + extra:
            r = dvs.load_run(bucket, t)
            if r is not None:
                runs[t] = r
        runs["lgbm_t10_stored"] = dict(pred=st10["pred_adj"].to_numpy(float))
        if carc_lgbm is not None:  # the cluster's masked walk (the density table's run)
            runs["lgbm_carc_mask"] = dict(pred=carc_lgbm["pred"])
        refs = ["lgbm_t10_stored", "lgbm_carc_mask"]
        have = [t for t in [*tags, *extra, *refs] if t in runs]
        miss = [t for t in tags if t not in runs]
        if miss:
            say(f"  runs missing (left out): {', '.join(miss)}")
        frames = [dvs.research_frame(dz, runs[t]["pred"]) for t in have]
        # the first pass's runs on the old design, in the same panel (paired draws)
        old_have: list[str] = []
        if OLD_WORK is not None and (OLD_WORK / f"design_{bucket}.npz").is_file():
            zo = np.load(OLD_WORK / f"design_{bucket}.npz", allow_pickle=False)
            dzo = {k: zo[k] for k in zo.files}
            for t in [*tags]:
                fo = OLD_WORK / "runs" / bucket / f"{t}.npz"
                if fo.is_file():
                    old_have.append(t)
                    frames.append(
                        dvs.research_frame(dzo, np.load(fo, allow_pickle=False)["pred"])
                    )
        col = {t: i for i, t in enumerate(have)}
        ocol = {t: len(have) + i for i, t in enumerate(old_have)}
        P = dvs.deck_panel(frames, dk)
        pt = dvs.point(P)
        g_tr = max(
            abs(base.trade_1530(f["pred_clock"])["Sharpe_mid"] - pt["sh"][i])
            for i, f in enumerate(frames)
        )
        gate(
            bucket,
            "per-day trade vs trade_1530 (Sharpe mid), every run (new and first pass)",
            n=len(frames),
            max_rel=float(g_tr),
            passed=g_tr < 1e-9,
        )
        assert g_tr < 1e-9, g_tr
        ql_all = {}
        for t in have:
            f = frames[col[t]].dropna(subset=["pred_clock"])
            ratio = f["true_raw"] / f["pred_clock"]
            ql_all[t] = float((ratio - np.log(ratio) - 1).mean())
        D = dvs.boot_draws(P)
        ivv = dk["iv_var"].to_numpy(float)

        def position(f: pd.DataFrame) -> np.ndarray:
            """The day's sign(s) side (+1 buy, -1 sell), as deck_panel sets it."""
            F_ = (
                f["pred_clock"]
                .set_axis(f.index.normalize())
                .reindex(dk.index)
                .to_numpy(float)
            )
            return np.where(F_ > ivv, 1, -1)

        def stat(t: str, i: int | None = None) -> dict:
            i = col[t] if i is None else i
            return dict(
                QLIKE_deck=float(pt["ql"][i]),
                QLIKE_all=ql_all.get(t, np.nan) if i < len(have) else np.nan,
                Sharpe_mid=float(pt["sh"][i]),
                Sharpe_crossed=float(pt["shx"][i]),
                pct_buy=float(100 * P["buy"][i]),
            )

        def contrast_ij(i: int, j: int) -> dict:
            return dict(
                dQLIKE=float(pt["ql"][i] - pt["ql"][j]),
                dQLIKE_lo=dvs.ci(D["ql"][:, i] - D["ql"][:, j])[0],
                dQLIKE_hi=dvs.ci(D["ql"][:, i] - D["ql"][:, j])[1],
                dSharpe=float(pt["sh"][i] - pt["sh"][j]),
                dSharpe_lo=dvs.ci(D["sh"][:, i] - D["sh"][:, j])[0],
                dSharpe_hi=dvs.ci(D["sh"][:, i] - D["sh"][:, j])[1],
                dSharpe_crossed=float(pt["shx"][i] - pt["shx"][j]),
                dSharpe_crossed_lo=dvs.ci(D["shx"][:, i] - D["shx"][:, j])[0],
                dSharpe_crossed_hi=dvs.ci(D["shx"][:, i] - D["shx"][:, j])[1],
            )

        def contrast(a: str, b: str) -> dict:
            return contrast_ij(col[a], col[b])

        def did_ij(i1: int, j1: int, i2: int, j2: int) -> dict:
            """(i1 - j1) - (i2 - j2): how much a gap changes, paired."""
            dq = (D["ql"][:, i1] - D["ql"][:, j1]) - (D["ql"][:, i2] - D["ql"][:, j2])
            ds = (D["sh"][:, i1] - D["sh"][:, j1]) - (D["sh"][:, i2] - D["sh"][:, j2])
            return dict(
                did_QLIKE=float(
                    (pt["ql"][i1] - pt["ql"][j1]) - (pt["ql"][i2] - pt["ql"][j2])
                ),
                did_QLIKE_lo=dvs.ci(dq)[0],
                did_QLIKE_hi=dvs.ci(dq)[1],
                did_Sharpe=float(
                    (pt["sh"][i1] - pt["sh"][j1]) - (pt["sh"][i2] - pt["sh"][j2])
                ),
                did_Sharpe_lo=dvs.ci(ds)[0],
                did_Sharpe_hi=dvs.ci(ds)[1],
            )

        def did(a1: str, b1: str, a2: str, b2: str) -> dict:
            return did_ij(col[a1], col[b1], col[a2], col[b2])

        NB["n_deck"] = int(P["pnl"].shape[0])
        NB["n_qlike_all"] = int(frames[0]["pred_clock"].notna().sum())
        for m in (*LIN_EST, "lgbm"):
            if f"{m}_all" in col:
                NB[f"full_{m}"] = stat(f"{m}_all")
        # the LightGBM references: masked vs no mask (same environment), local vs cluster
        for a, b_, what in (
            ("lgbm_all", "lgbm_nomask_all", "mask effect (both local)"),
            ("lgbm_carc_mask", "lgbm_t10_stored", "mask effect (both cluster)"),
            ("lgbm_nomask_all", "lgbm_t10_stored", "local vs cluster (both unmasked)"),
            ("lgbm_all", "lgbm_carc_mask", "local vs cluster (both masked)"),
        ):
            if a in col and b_ in col:
                ref_rows.append(
                    dict(bucket=bucket, row=f"{a} - {b_}", what=what, **contrast(a, b_))
                )
        for t, what in (
            (
                "lgbm_all",
                f"LightGBM all columns, {V_NEW} (local; the sweep's reference)",
            ),
            ("lgbm_nomask_all", f"LightGBM all columns, {V_NOMASK} (local)"),
            (
                "lgbm_t10_stored",
                f"LightGBM all columns, {V_NOMASK} (stored T10, cluster)",
            ),
            (
                "lgbm_carc_mask",
                f"LightGBM all columns, {V_NEW} (cluster; the density table's run)",
            ),
        ):
            if t in col:
                ref_rows.append(dict(bucket=bucket, row=t, what=what, **stat(t)))
        # A2 sweep table
        for rule in ("screen", "own"):
            for m in dvs.SWEEP_MODELS if rule == "screen" else ("ridge", "reclasso"):
                for k in (*K_GRID, "all"):
                    t = f"{m}_all" if k == "all" else f"{m}_{rule}_k{k}"
                    if t not in col:
                        continue
                    row = dict(
                        bucket=bucket, model=LABEL[m], rule=rule, k=str(k), **stat(t)
                    )
                    if k != "all" and f"{m}_all" in col:
                        row.update(contrast(t, f"{m}_all"))
                        cols_ = runs[t].get("cols")
                        if cols_ is not None:
                            row["distinct_columns_used"] = int(
                                len(np.unique(cols_[cols_ >= 0]))
                            )
                    sweep_rows.append(row)
                    if t in ocol:  # new vs first pass, paired
                        assert OLD_WORK is not None
                        fo = np.load(OLD_WORK / "runs" / bucket / f"{t}.npz")["pred"]
                        chg = contrast_ij(col[t], ocol[t])
                        chg_rows.append(
                            dict(
                                bucket=bucket,
                                model=LABEL[m],
                                rule=rule,
                                k=str(k),
                                max_rel_forecast_change=float(
                                    np.max(_rel(runs[t]["pred"], fo))
                                ),
                                median_rel_forecast_change=float(
                                    np.median(_rel(runs[t]["pred"], fo))
                                ),
                                trade_days_position_differs=int(
                                    (
                                        position(frames[col[t]])
                                        != position(frames[ocol[t]])
                                    ).sum()
                                ),
                                **{f"chg_{kk}": vv for kk, vv in chg.items()},
                            )
                        )
        # the hypothesis: ridge's edge over the lasso and LightGBM, same inputs, as k varies
        for a, b_ in (("ridge", "reclasso"), ("ridge", "lgbm"), ("reclasso", "lgbm")):
            for k in (*K_GRID, "all"):
                ta = f"{a}_all" if k == "all" else f"{a}_screen_k{k}"
                tb = f"{b_}_all" if k == "all" else f"{b_}_screen_k{k}"
                if ta not in col or tb not in col:
                    continue
                pair = f"{LABEL[a]} - {LABEL[b_]}"
                row = dict(bucket=bucket, pair=pair, k=str(k), **contrast(ta, tb))
                if k != "all" and f"{a}_all" in col and f"{b_}_all" in col:
                    row.update(did(f"{a}_all", f"{b_}_all", ta, tb))
                gap_rows.append(row)
                if (
                    ta in ocol and tb in ocol
                ):  # the change of the gap, new vs first pass
                    gchg_rows.append(
                        dict(
                            bucket=bucket,
                            pair=pair,
                            k=str(k),
                            **{
                                f"chg_{kk}": vv
                                for kk, vv in did_ij(
                                    col[ta], col[tb], ocol[ta], ocol[tb]
                                ).items()
                            },
                        )
                    )
        for k in (1, 2, 4, 8):
            t = f"ridge_screen_k{k}"
            if t in col and "cols" in runs[t]:
                for b_i, cc in enumerate(runs[t]["cols"]):
                    pick_rows.append(
                        dict(
                            bucket=bucket,
                            k=k,
                            block=b_i,
                            first_forecast=str(dz["date"][b_i * TUNE_PER])[:10],
                            columns=" ".join(names[j] for j in cc if j >= 0),
                        )
                    )
        sweep_fig[bucket] = {
            m: [
                (k, stat(f"{m}_all" if k == "all" else f"{m}_screen_k{k}"))
                for k in (*K_GRID, "all")
                if (f"{m}_all" if k == "all" else f"{m}_screen_k{k}") in col
            ]
            for m in dvs.SWEEP_MODELS
        }
        old_fig[bucket] = {
            m: [
                (k, stat(t, ocol[t]))
                for k in (*K_GRID, "all")
                for t in [f"{m}_all" if k == "all" else f"{m}_screen_k{k}"]
                if t in ocol
            ]
            for m in dvs.SWEEP_MODELS
        }
        # A3 collapse
        for m in ("ridge", "reclasso"):
            if f"{m}_collapse" in col:
                coll_rows.append(
                    dict(
                        bucket=bucket,
                        row=f"{LABEL[m]} collapsed vs full",
                        **stat(f"{m}_collapse"),
                        **contrast(f"{m}_collapse", f"{m}_all"),
                    )
                )
        if {"ridge_collapse", "reclasso_collapse", "ridge_all", "reclasso_all"} <= set(
            col
        ):
            coll_rows.append(
                dict(
                    bucket=bucket,
                    row="ridge - lasso, full design",
                    **contrast("ridge_all", "reclasso_all"),
                )
            )
            coll_rows.append(
                dict(
                    bucket=bucket,
                    row="ridge - lasso, groups collapsed to PC1",
                    **contrast("ridge_collapse", "reclasso_collapse"),
                )
            )
            coll_rows.append(
                dict(
                    bucket=bucket,
                    row="(ridge - lasso) full minus collapsed",
                    **did(
                        "ridge_all",
                        "reclasso_all",
                        "ridge_collapse",
                        "reclasso_collapse",
                    ),
                )
            )
            NB["collapse_blocks"] = json.loads(
                str(runs["ridge_collapse"]["collapse_info"])
            )
        NUM["buckets"][bucket] = NB

    # ---------------- tables
    GT = pd.DataFrame(gate_rows)
    GT.to_csv(OUT / "gates.csv", index=False)
    DN = pd.DataFrame(dens_rows)
    DN.to_csv(OUT / "density.csv", index=False)
    SW = pd.DataFrame(sweep_rows)
    SW.to_csv(OUT / "sweep.csv", index=False)
    GP = pd.DataFrame(gap_rows)
    GP.to_csv(OUT / "sweep_gaps.csv", index=False)
    GR = pd.DataFrame(group_rows)
    GR.to_csv(OUT / "groups.csv", index=False)
    CC = pd.DataFrame(conc_rows)
    CC.to_csv(OUT / "group_concentration.csv", index=False)
    CL = pd.DataFrame(coll_rows)
    CL.to_csv(OUT / "collapse.csv", index=False)
    PK = pd.DataFrame(pick_rows)
    PK.to_csv(OUT / "screen_picks.csv", index=False)
    MK = pd.DataFrame(mask_rows)
    MK.to_csv(OUT / "mask_counts.csv", index=False)
    RF = pd.DataFrame(ref_rows)
    RF.to_csv(OUT / "lgbm_references.csv", index=False)
    CH = pd.DataFrame(chg_rows)
    GC = pd.DataFrame(gchg_rows)
    cnt = []
    for (bucket, pair), g in GP.groupby(["bucket", "pair"]):
        g = g[g["k"] != "all"]
        for mm in ("QLIKE", "Sharpe"):
            if f"did_{mm}_lo" not in g or g[f"did_{mm}_lo"].isna().all():
                continue
            gg = g.dropna(subset=[f"did_{mm}_lo"])
            ex_ = (gg[f"did_{mm}_lo"] > 0) | (gg[f"did_{mm}_hi"] < 0)
            cnt.append(
                dict(
                    bucket=bucket,
                    pair=pair,
                    measure=mm,
                    n_k=len(gg),
                    gap_ci_excl0=int(
                        ((gg[f"d{mm}_lo"] > 0) | (gg[f"d{mm}_hi"] < 0)).sum()
                    ),
                    did_ci_excl0=int(ex_.sum()),
                    did_ci_excl0_ks=" ".join(gg.loc[ex_, "k"].astype(str)),
                )
            )
    CN = pd.DataFrame(cnt)
    CN.to_csv(OUT / "sweep_gap_counts.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    say("\nGATES:")
    say(GT.to_string(index=False))
    say("\nmask counts:")
    say(MK.to_string(index=False))
    say("\nA1 signal density (column level, all forecasts):")
    say(
        DN[(DN.level == "column") & (DN["sample"] == "all forecasts")][
            [
                "bucket",
                "model",
                "variant",
                "n_inputs",
                "n_active",
                "neff",
                "top1_share",
                "k50",
                "k80",
                "k95",
                "greedy_k95",
                "block_k80_min",
                "block_k80_max",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )
    say("\nA2 sweep:")
    say(
        SW[
            [
                "bucket",
                "model",
                "rule",
                "k",
                "QLIKE_deck",
                "Sharpe_mid",
                "dQLIKE",
                "dQLIKE_lo",
                "dQLIKE_hi",
                "dSharpe",
                "dSharpe_lo",
                "dSharpe_hi",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )
    say("\nA2 gaps:")
    say(GP.round(4).to_string(index=False))
    say("\nA2 counts:")
    say(CN.to_string(index=False))
    say("\nLightGBM references:")
    say(RF.round(4).to_string(index=False))
    say("\nA3 concentration by group:")
    say(CC.round(3).to_string(index=False))
    say("\nA3 collapse:")
    say(CL.round(4).to_string(index=False))

    # ---------------- figures (the first pass's two, on the new runs, + the before / after sweep)
    ks = [*K_GRID, "all"]
    xpos = {k: i for i, k in enumerate(ks)}
    for fname, with_old in (
        ("sweep_qlike_sharpe_vs_k.png", False),
        ("before_after_sweep_vs_k.png", True),
    ):
        fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), sharex=True)
        for c_i, bucket in enumerate(BUCKETS):
            for r_i, (key, ylab) in enumerate(
                (
                    ("QLIKE_deck", "16:00 QLIKE, 866 deck days"),
                    ("Sharpe_mid", "sign(s) Sharpe, mid"),
                )
            ):
                ax = axes[r_i, c_i]
                for m in dvs.SWEEP_MODELS:
                    pts = sweep_fig.get(bucket, {}).get(m, [])
                    if pts:
                        ax.plot(
                            [xpos[k] for k, _ in pts],
                            [s[key] for _, s in pts],
                            color=dvs.COLOR[m],
                            ls=dvs.DASH[m],
                            lw=2,
                            marker="o",
                            ms=5,
                            label=f"{LABEL[m]}" + (f", {V_NEW}" if with_old else ""),
                        )
                    opts = old_fig.get(bucket, {}).get(m, []) if with_old else []
                    if opts:
                        ax.plot(
                            [xpos[k] for k, _ in opts],
                            [s[key] for _, s in opts],
                            color=dvs.COLOR[m],
                            ls=dvs.DASH[m],
                            lw=1,
                            alpha=0.45,
                            marker="s",
                            ms=3,
                            mfc="none",
                            label=f"{LABEL[m]}, {V_OLD}",
                        )
                ax.set_xticks(range(len(ks)))
                ax.set_xticklabels([str(k) for k in ks], fontsize=7)
                ax.tick_params(axis="y", labelsize=7)
                ax.grid(color="0.9", lw=0.6)
                ax.set_axisbelow(True)
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
                ax.set_ylabel(ylab, fontsize=8)
                if r_i == 0:
                    n_c = NUM["buckets"][bucket]["n_columns"]
                    ax.set_title(
                        f"{bucket} design ({n_c} columns"
                        + (f"; first pass {P_OLD[bucket]}" if with_old else "")
                        + ")",
                        fontsize=9,
                    )
                if r_i == 1:
                    ax.set_xlabel(
                        "k = inputs kept (top-k by |corr with target| over the window before each "
                        "250-session block)",
                        fontsize=7,
                    )
        axes[0, 0].legend(fontsize=6 if with_old else 7, frameon=False)
        fig.tight_layout()
        fig.savefig(OUT / fname, dpi=160)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, bucket in zip(axes, BUCKETS):
        for m, r2 in curves.get(bucket, {}).items():
            kk = np.arange(1, len(r2) + 1)
            ax.plot(
                kk,
                np.clip(r2, 0, 1),
                color=dvs.COLOR[m],
                ls=dvs.DASH[m],
                lw=2,
                label=LABEL[m],
            )
        for x in dvs.DENSITY_LEVELS:
            ax.axhline(x, color="0.75", lw=0.6)
        ax.set_xscale("log")
        ax.set_xlabel(
            "inputs, in order of their share of mean |contribution|", fontsize=8
        )
        ax.set_title(
            f"{bucket} ({NUM['buckets'][bucket]['n_columns']} columns; trees masked)",
            fontsize=9,
        )
        ax.tick_params(labelsize=7)
        ax.grid(color="0.92", lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("share of the forecast's variance reproduced", fontsize=8)
    axes[0].legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "density_curves.png", dpi=160)
    plt.close(fig)

    # ---------------- before / after tables
    before_after(CH, GC)
    rl = WORK / "refit_log.json"
    timing: dict = {}
    if rl.is_file():
        rr = json.loads(rl.read_text(encoding="utf-8"))
        timing["local"] = dict(
            cpu_hours=sum(x["cpu_sec"] for x in rr) / 3600,
            wall_min=sum(x["wall_sec"] for x in rr) / 60,
            peak_processes=max(x["workers"] for x in rr),
            invocations=len(rr),
        )
    for mode in ("canary", "fleet", "gate"):
        f = TREES_CARC / f"carc_log_{mode}.json"
        if f.is_file():
            cl = json.loads(f.read_text(encoding="utf-8"))
            timing[f"cluster_{mode}"] = dict(
                cpu_hours=cl["cpu_sec"] / 3600,
                wall_min=cl["wall_sec"] / 60,
                pool=cl["pool"],
                host=cl["host"],
                versions=cl["versions"],
                tree_spec_md5=cl["tree_spec_md5"],
            )
    sa = TREES_CARC / "sacct.psv"
    if (
        sa.is_file()
    ):  # the cluster's own accounting (cluster/slurm/pull_dvs_dedup_carc.sh WHAT=sacct)
        acct = pd.read_csv(sa, sep="|")
        timing["cluster_sacct"] = dict(
            jobs=" ".join(
                f"{r.JobName} {r.JobID} ({r.Partition}, {r.AllocCPUS} CPUs, {r.Elapsed})"
                for r in acct.itertuples()
            ),
            alloc_cpu_hours=float(acct["CPUTimeRAW"].sum() / 3600),
            peak_cpus=int(acct["AllocCPUS"].max()),
            wall_min_total=float(acct["ElapsedRaw"].sum() / 60),
            states=" ".join(sorted(set(acct["State"].astype(str)))),
        )
    NUM["timing"] = timing
    NUM["tree_spec_md5_local"] = file_md5(TREE_SPEC)
    NUM["tree_spec_model_section"] = spec_model_section_check(timing)
    NUM["versions_local"] = _versions()
    NUM["density"] = DN.to_dict("records")
    NUM["sweep"] = SW.to_dict("records")
    NUM["gaps"] = GP.to_dict("records")
    NUM["gap_counts"] = CN.to_dict("records")
    NUM["concentration"] = CC.to_dict("records")
    NUM["collapse"] = CL.to_dict("records")
    NUM["gates"] = GT.to_dict("records")
    NUM["mask_counts"] = MK.to_dict("records")
    NUM["constants"] = dict(
        K_GRID=list(K_GRID),
        CORR_GROUP=dvs.CORR_GROUP,
        TUNE_PER=TUNE_PER,
        TREE_REFIT_EVERY=REFIT_EVERY,
        BOOT_B=dvs.BOOT_B,
        BOOT_BLOCK=base.BOOT_BLOCK,
        BOOT_SEED=dvs.BOOT_SEED,
    )
    (OUT / "numbers.json").write_text(
        json.dumps(NUM, indent=1, default=float), encoding="utf-8"
    )
    (OUT / "analyze_output.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================================================================ before / after
def _merge(
    old: pd.DataFrame, new: pd.DataFrame, on: list[str], vals: list[str]
) -> pd.DataFrame:
    o = old[on + [v for v in vals if v in old.columns]].rename(
        columns={v: f"{v}__first_pass" for v in vals}
    )
    n = new[on + [v for v in vals if v in new.columns]].rename(
        columns={v: f"{v}__dedup_mask" for v in vals}
    )
    m = o.merge(n, on=on, how="outer", indicator=True)
    m["_merge"] = m["_merge"].map(
        {
            "both": "both",
            "left_only": "first pass only",
            "right_only": "de-dup + mask only",
        }
    )
    order = on + [f"{v}__{s}" for v in vals for s in ("first_pass", "dedup_mask")]
    return m[[c for c in order if c in m.columns] + ["_merge"]].rename(
        columns={"_merge": "present_in"}
    )


def before_after(CH: pd.DataFrame, GC: pd.DataFrame) -> None:
    """First pass (old design, results/dense_vs_sparse/) vs de-dup + mask (this folder), side
    by side; the "de-dup, no mask" trees where they exist; paired new-minus-old intervals where
    the first pass's per-run forecasts are at hand (DVS_OLD_WORK)."""
    # density (+ group-level N_eff)
    do = pd.read_csv(OLD / "density.csv")
    dn = pd.read_csv(OUT / "density.csv")
    dcols = [
        "n_inputs",
        "n_active",
        "neff",
        "top1_share",
        "k50",
        "k80",
        "k95",
        "greedy_k50",
        "greedy_k80",
        "greedy_k95",
        "block_k80_median",
        "block_k80_min",
        "block_k80_max",
    ]
    key = ["bucket", "model", "level", "sample"]
    bd = _merge(do, dn[dn.variant == V_NEW], key, dcols)
    nm = dn[dn.variant == V_NOMASK][key + dcols].rename(
        columns={v: f"{v}__dedup_nomask" for v in dcols}
    )
    bd = bd.merge(nm, on=key, how="left")
    co = pd.read_csv(OLD / "group_concentration.csv")
    cn = pd.read_csv(OUT / "group_concentration.csv")
    bd = bd.merge(
        co[["bucket", "model", "neff_groups"]].rename(
            columns={"neff_groups": "neff_groups__first_pass"}
        ),
        on=["bucket", "model"],
        how="left",
    )
    for v, lab in ((V_NEW, "dedup_mask"), (V_NOMASK, "dedup_nomask")):
        bd = bd.merge(
            cn[cn.variant == v][["bucket", "model", "neff_groups"]].rename(
                columns={"neff_groups": f"neff_groups__{lab}"}
            ),
            on=["bucket", "model"],
            how="left",
        )
    bd.loc[
        (bd.level != "column") | (bd["sample"] != "all forecasts"),
        [c for c in bd.columns if c.startswith("neff_groups")],
    ] = np.nan
    bd.to_csv(OUT / "before_after_density.csv", index=False)
    # sweep (QLIKE / Sharpe vs k; the change with its paired interval)
    so = pd.read_csv(OLD / "sweep.csv")
    sn = pd.read_csv(OUT / "sweep.csv")
    for s_ in (so, sn):
        s_["k"] = s_["k"].astype(str)
    svals = [
        "QLIKE_deck",
        "Sharpe_mid",
        "Sharpe_crossed",
        "pct_buy",
        "dQLIKE",
        "dQLIKE_lo",
        "dQLIKE_hi",
        "dSharpe",
        "dSharpe_lo",
        "dSharpe_hi",
        "distinct_columns_used",
    ]
    bs = _merge(so, sn, ["bucket", "model", "rule", "k"], svals)
    if len(CH):
        CH = CH.copy()
        CH["k"] = CH["k"].astype(str)
        bs = bs.merge(
            CH[
                [
                    "bucket",
                    "model",
                    "rule",
                    "k",
                    "max_rel_forecast_change",
                    "median_rel_forecast_change",
                    "trade_days_position_differs",
                    "chg_dQLIKE",
                    "chg_dQLIKE_lo",
                    "chg_dQLIKE_hi",
                    "chg_dSharpe",
                    "chg_dSharpe_lo",
                    "chg_dSharpe_hi",
                ]
            ].rename(
                columns={
                    "chg_dQLIKE": "QLIKE_new_minus_first_pass",
                    "chg_dQLIKE_lo": "QLIKE_new_minus_first_pass_lo",
                    "chg_dQLIKE_hi": "QLIKE_new_minus_first_pass_hi",
                    "chg_dSharpe": "Sharpe_new_minus_first_pass",
                    "chg_dSharpe_lo": "Sharpe_new_minus_first_pass_lo",
                    "chg_dSharpe_hi": "Sharpe_new_minus_first_pass_hi",
                }
            ),
            on=["bucket", "model", "rule", "k"],
            how="left",
        )
    rf = pd.read_csv(OUT / "lgbm_references.csv")
    nmr = rf[rf.row == "lgbm_nomask_all"]
    for c in ("QLIKE_deck", "Sharpe_mid", "Sharpe_crossed"):
        bs[f"{c}__dedup_nomask"] = np.nan
        for _, r in nmr.iterrows():
            sel = (bs.bucket == r.bucket) & (bs.model == "LightGBM") & (bs.k == "all")
            bs.loc[sel, f"{c}__dedup_nomask"] = r[c]
    bs.to_csv(OUT / "before_after_sweep.csv", index=False)
    # the hypothesis gaps
    go = pd.read_csv(OLD / "sweep_gaps.csv")
    gn = pd.read_csv(OUT / "sweep_gaps.csv")
    for g_ in (go, gn):
        g_["k"] = g_["k"].astype(str)
    gvals = [
        "dQLIKE",
        "dQLIKE_lo",
        "dQLIKE_hi",
        "dSharpe",
        "dSharpe_lo",
        "dSharpe_hi",
        "did_QLIKE",
        "did_QLIKE_lo",
        "did_QLIKE_hi",
        "did_Sharpe",
        "did_Sharpe_lo",
        "did_Sharpe_hi",
    ]
    bg = _merge(go, gn, ["bucket", "pair", "k"], gvals)
    if len(GC):
        GC = GC.copy()
        GC["k"] = GC["k"].astype(str)
        bg = bg.merge(
            GC.rename(
                columns={
                    "chg_did_QLIKE": "gap_QLIKE_new_minus_first_pass",
                    "chg_did_QLIKE_lo": "gap_QLIKE_new_minus_first_pass_lo",
                    "chg_did_QLIKE_hi": "gap_QLIKE_new_minus_first_pass_hi",
                    "chg_did_Sharpe": "gap_Sharpe_new_minus_first_pass",
                    "chg_did_Sharpe_lo": "gap_Sharpe_new_minus_first_pass_lo",
                    "chg_did_Sharpe_hi": "gap_Sharpe_new_minus_first_pass_hi",
                }
            ),
            on=["bucket", "pair", "k"],
            how="left",
        )
    bg.to_csv(OUT / "before_after_gaps.csv", index=False)
    # group shares
    gro = pd.read_csv(OLD / "groups.csv")
    grn = pd.read_csv(OUT / "groups.csv")
    gv = [c for c in grn.columns if c.startswith(("var_share_", "abs_share_"))]
    gv_old = [c for c in gro.columns if c.startswith(("var_share_", "abs_share_"))]
    bgr = _merge(
        gro, grn, ["bucket", "members"], ["n_columns"] + sorted(set(gv_old) | set(gv))
    )
    nomask_cols = [c for c in bgr.columns if "_nomask__first_pass" in c]
    bgr = bgr.drop(columns=nomask_cols).rename(
        columns={
            c: c.replace("_nomask__dedup_mask", "__dedup_nomask")
            for c in bgr.columns
            if "_nomask__dedup_mask" in c
        }
    )
    bgr.to_csv(OUT / "before_after_groups.csv", index=False)
    # the PC1 collapse
    clo = pd.read_csv(OLD / "collapse.csv")
    cln = pd.read_csv(OUT / "collapse.csv")
    cvals = [
        "QLIKE_deck",
        "Sharpe_mid",
        "dQLIKE",
        "dQLIKE_lo",
        "dQLIKE_hi",
        "dSharpe",
        "dSharpe_lo",
        "dSharpe_hi",
        "did_QLIKE",
        "did_QLIKE_lo",
        "did_QLIKE_hi",
        "did_Sharpe",
        "did_Sharpe_lo",
        "did_Sharpe_hi",
    ]
    _merge(clo, cln, ["bucket", "row"], cvals).to_csv(
        OUT / "before_after_collapse.csv", index=False
    )
    # the screen's picks
    po = pd.read_csv(OLD / "screen_picks.csv")
    pn = pd.read_csv(OUT / "screen_picks.csv")
    bp = _merge(po, pn, ["bucket", "k", "block", "first_forecast"], ["columns"])
    bp["identical"] = bp["columns__first_pass"] == bp["columns__dedup_mask"]
    bp.to_csv(OUT / "before_after_picks.csv", index=False)
    # the first pass's trees: how much TreeSHAP sat on the session-edge columns
    rows = []
    for bucket in BUCKETS:
        for m in TREE_MODELS:
            f = (
                OLD_TREES
                / bucket
                / SEG
                / m
                / f"tw{TW}"
                / "causal_tune_trees"
                / m
                / bucket
                / f"trees_{SEG}.npz"
            )
            if not f.is_file():
                continue
            z = np.load(f, allow_pickle=True)
            nm_ = [str(v) for v in z["feature_names"]]
            sh = np.abs(np.asarray(z["shap"], float)[:, :-1]).mean(0)
            share = sh / sh.sum()
            edge = [j for j, n in enumerate(nm_) if n.endswith(SESSION_EDGE_SUFFIX)]
            xclose = [j for j, n in enumerate(nm_) if n.endswith("_x_close")]
            har = [
                j
                for j, n in enumerate(nm_)
                if n.startswith("har_ma_") and "_x_" not in n
            ]
            new = tree_contrib(bucket, m, dvs.load_design(bucket), V_NEW)
            nn = [str(v) for v in dvs.load_design(bucket)["names"]]
            new_har = np.nan
            if new is not None:
                a = np.abs(new["phi"]).mean(0)
                new_har = float(
                    a[[j for j, n in enumerate(nn) if n.startswith("har_ma_")]].sum()
                    / a.sum()
                )
            rows.append(
                dict(
                    bucket=bucket,
                    model=LABEL[m],
                    first_pass_columns=len(nm_),
                    first_pass_share_session_edge=float(share[edge].sum()),
                    first_pass_share_x_close=float(share[xclose].sum()),
                    first_pass_share_har_ma=float(share[har].sum()),
                    first_pass_share_har_ma_plus_x_close=float(
                        share[har].sum() + share[xclose].sum()
                    ),
                    dedup_mask_share_har_ma=new_har,
                )
            )
    pd.DataFrame(rows).to_csv(OUT / "before_after_session_edge.csv", index=False)


# ============================================================================ summary
_iv = dvs._iv
_sig = dvs._sig
_members = dvs._members
LF, AF = BUCKETS
RL, RG, LG = "ridge - lasso", "ridge - LightGBM", "lasso - LightGBM"
DENS_MODELS = [LABEL[m] for m in dvs.DENSITY_MODELS]


def _load(d: Path, new: bool) -> dict:
    """One results folder's tables (the first pass's, or this one's de-dup + mask rows)."""
    T = {
        n: pd.read_csv(d / f"{n}.csv")
        for n in (
            "density",
            "group_concentration",
            "groups",
            "sweep",
            "sweep_gaps",
            "sweep_gap_counts",
            "collapse",
            "screen_picks",
            "gates",
        )
    }
    if new:
        T["density"] = T["density"][T["density"].variant == V_NEW]
        T["group_concentration"] = T["group_concentration"][
            T["group_concentration"].variant == V_NEW
        ]
    for n in ("sweep", "sweep_gaps"):
        T[n]["k"] = T[n]["k"].astype(str)
    T["numbers"] = json.loads((d / "numbers.json").read_text(encoding="utf-8"))
    return T


def claims(T: dict) -> dict:
    """The first pass's findings, evaluated on one folder's tables (no assertion: the values
    and whether each statement holds are recorded, whatever they are)."""
    DN, GR, SW, GP, CN, CL = (
        T["density"],
        T["groups"],
        T["sweep"],
        T["sweep_gaps"],
        T["sweep_gap_counts"],
        T["collapse"],
    )

    def den(b: str, m: str) -> pd.Series:
        r = DN[
            (DN.bucket == b)
            & (DN.model == m)
            & (DN.level == "column")
            & (DN["sample"] == "all forecasts")
        ]
        return r.iloc[0]

    def sw(b: str, m: str, k: str, rule: str = "screen") -> pd.Series:
        return SW[
            (SW.bucket == b) & (SW.model == m) & (SW.k == k) & (SW.rule == rule)
        ].iloc[0]

    def gp(b: str, pair: str, k: str) -> pd.Series:
        return GP[(GP.bucket == b) & (GP.pair == pair) & (GP.k == k)].iloc[0]

    others = [m for m in DENS_MODELS if m != "ridge"]
    ks = [str(k) for k in K_GRID]
    C: dict = {}
    for b in BUCKETS:
        ne = {m: float(den(b, m)["neff"]) for m in DENS_MODELS}
        C[f"neff_ridge_{b}"] = ne["ridge"]
        C[f"neff_others_{b}"] = (min(ne[m] for m in others), max(ne[m] for m in others))
        C[f"ridge_densest_{b}"] = ne["ridge"] > max(ne[m] for m in others)
        C[f"k95_ridge_{b}"] = int(den(b, "ridge")["k95"])
        C[f"k95_others_max_{b}"] = int(max(den(b, m)["k95"] for m in others))
        C[f"ridge_k95_largest_{b}"] = C[f"k95_ridge_{b}"] > C[f"k95_others_max_{b}"]
    C["k80_max"] = int(max(den(b, m)["k80"] for b in BUCKETS for m in DENS_MODELS))
    g_lf = GR[GR.bucket == LF]
    size_g = g_lf[
        g_lf["members"].str.split().apply(lambda v: "adj_sumabsret_ma_1" in v)
    ].iloc[0]
    har_g = g_lf[g_lf["members"].str.split().apply(lambda v: "har_ma_1" in v)].iloc[0]
    tk = ("lgbm", "xgb", "rf")
    C["size_group"] = size_g["members"]
    C["size_share"] = {
        m: float(size_g[f"var_share_{m}"]) for m in ("ridge", "reclasso", *tk)
    }
    C["har_share"] = {
        m: float(har_g[f"var_share_{m}"]) for m in ("ridge", "reclasso", *tk)
    }
    C["ridge_most_on_size"] = C["size_share"]["ridge"] > max(
        C["size_share"][m] for m in ("reclasso", *tk)
    )
    C["ridge_least_on_har"] = C["har_share"]["ridge"] < min(
        C["har_share"][m] for m in ("reclasso", *tk)
    )
    sig_q = {
        (b, m, k): bool(_sig(r["dQLIKE_lo"], r["dQLIKE_hi"]) and r["dQLIKE"] > 0)
        for b in BUCKETS
        for m in ("ridge", "lasso", "LightGBM")
        for k in ks
        for r in [sw(b, m, k)]
    }
    K_all = 0
    for k in K_GRID:
        if all(
            sig_q[(b, m, str(k))]
            for b in BUCKETS
            for m in ("ridge", "lasso", "LightGBM")
        ):
            K_all = k
        else:
            break
    K_lf = 0
    for k in K_GRID:
        if all(sig_q[(LF, m, str(k))] for m in ("ridge", "lasso", "LightGBM")):
            K_lf = k
        else:
            break
    C["K_all"], C["K_lf"] = K_all, K_lf
    C["below_all_k16"] = all(
        sw(b, m, k)["Sharpe_mid"] < sw(b, m, "all")["Sharpe_mid"]
        for b in BUCKETS
        for m in ("ridge", "lasso", "LightGBM")
        for k in ks
        if int(k) <= 16
    )
    C["ridge_beats_lgbm_k12"] = all(
        _sig(gp(b, RG, k)["dQLIKE_lo"], gp(b, RG, k)["dQLIKE_hi"])
        and gp(b, RG, k)["dQLIKE"] < 0
        for b in BUCKETS
        for k in ("1", "2")
    )
    C["all_columns_tie_qlike"] = all(
        not _sig(gp(b, pr, "all")["dQLIKE_lo"], gp(b, pr, "all")["dQLIKE_hi"])
        for b in BUCKETS
        for pr in (RL, RG)
    )
    C["lasso_beats_ridge_k48_af"] = all(
        _sig(gp(AF, RL, k)["dQLIKE_lo"], gp(AF, RL, k)["dQLIKE_hi"])
        and gp(AF, RL, k)["dQLIKE"] > 0
        for k in ("4", "8")
    )
    C["af_rl_did_ks"] = [
        k
        for k in ks
        if _sig(gp(AF, RL, k)["did_QLIKE_lo"], gp(AF, RL, k)["did_QLIKE_hi"])
    ]
    C["lf_rl_did_ks"] = [
        k
        for k in ks
        if _sig(gp(LF, RL, k)["did_QLIKE_lo"], gp(LF, RL, k)["did_QLIKE_hi"])
    ]
    C["lf_rg_did_ks"] = [
        k
        for k in ks
        if _sig(gp(LF, RG, k)["did_QLIKE_lo"], gp(LF, RG, k)["did_QLIKE_hi"])
    ]
    C["af_rg_did_ks"] = [
        k
        for k in ks
        if _sig(gp(AF, RG, k)["did_QLIKE_lo"], gp(AF, RG, k)["did_QLIKE_hi"])
    ]
    rp = CN[CN.pair.isin([RL, RG]) & (CN.measure == "Sharpe")]
    C["n_sh"], C["n_sh_gap"], C["n_sh_did"] = (
        int(rp["n_k"].sum()),
        int(rp["gap_ci_excl0"].sum()),
        int(rp["did_ci_excl0"].sum()),
    )
    a_ = CN[CN.measure == "Sharpe"]
    C["n_sh_all"], C["n_sh_all_did"] = (
        int(a_["n_k"].sum()),
        int(a_["did_ci_excl0"].sum()),
    )
    C["full_gap"] = {
        (b, pr): {
            c: float(gp(b, pr, "all")[c])
            for c in (
                "dQLIKE",
                "dQLIKE_lo",
                "dQLIKE_hi",
                "dSharpe",
                "dSharpe_lo",
                "dSharpe_hi",
            )
        }
        for b in BUCKETS
        for pr in (RL, RG)
    }
    C["full_sharpe_gap_sig"] = [
        f"{b} {pr}"
        for b in BUCKETS
        for pr in (RL, RG)
        if _sig(gp(b, pr, "all")["dSharpe_lo"], gp(b, pr, "all")["dSharpe_hi"])
    ]
    C["collapse_did_sharpe"] = {}
    for b in BUCKETS:
        d = CL[
            (CL.bucket == b) & (CL.row == "(ridge - lasso) full minus collapsed")
        ].iloc[0]
        C["collapse_did_sharpe"][b] = (
            float(d["did_Sharpe"]),
            float(d["did_Sharpe_lo"]),
            float(d["did_Sharpe_hi"]),
        )
    C["collapse_sig"] = [
        b for b, v in C["collapse_did_sharpe"].items() if _sig(v[1], v[2])
    ]
    w = [r["dSharpe_hi"] - r["dSharpe_lo"] for _, r in GP[GP.k != "all"].iterrows()]
    C["sharpe_ci_width"] = (min(w), max(w))
    return C


def _yn(v: bool) -> str:
    return "yes" if v else "no"


def write_summary() -> None:  # noqa: C901 - one linear report
    """SUMMARY.md from this folder's own CSVs (and the first pass's, for the before / after);
    no claim is asserted: each first-pass statement is re-evaluated and its status recorded."""
    TN, TO = _load(OUT, True), _load(OLD, False)
    CN_, CO_ = claims(TN), claims(TO)
    NB = TN["numbers"]["buckets"]
    NBo = TO["numbers"]["buckets"]
    GRn = TN["groups"]
    GT = TN["gates"]
    MK = pd.read_csv(OUT / "mask_counts.csv")
    RF = pd.read_csv(OUT / "lgbm_references.csv")
    BS = pd.read_csv(OUT / "before_after_sweep.csv")
    BS["k"] = BS["k"].astype(str)
    BG = pd.read_csv(OUT / "before_after_gaps.csv")
    BG["k"] = BG["k"].astype(str)
    BP = pd.read_csv(OUT / "before_after_picks.csv")
    SE = pd.read_csv(OUT / "before_after_session_edge.csv")
    NUM = TN["numbers"]
    ks = [str(k) for k in K_GRID]

    # ---- the claims table (also written as a CSV)
    def fmt_rng(t: tuple) -> str:
        return f"{t[0]:.0f}–{t[1]:.0f}"

    rows = []

    def claim(name: str, old: object, new: object) -> None:
        rows.append(
            dict(
                claim=name,
                first_pass=str(old),
                dedup_mask=str(new),
                same=str(old) == str(new),
            )
        )

    for b in BUCKETS:
        claim(
            f"ridge has the largest N_eff ({b})",
            _yn(CO_[f"ridge_densest_{b}"]),
            _yn(CN_[f"ridge_densest_{b}"]),
        )
        claim(
            f"N_eff ridge / range of the other five ({b})",
            f"{CO_[f'neff_ridge_{b}']:.1f} / {fmt_rng(CO_[f'neff_others_{b}'])}",
            f"{CN_[f'neff_ridge_{b}']:.1f} / {fmt_rng(CN_[f'neff_others_{b}'])}",
        )
        claim(
            f"k_95 ridge / largest of the other five ({b})",
            f"{CO_[f'k95_ridge_{b}']} / {CO_[f'k95_others_max_{b}']}",
            f"{CN_[f'k95_ridge_{b}']} / {CN_[f'k95_others_max_{b}']}",
        )
    claim("largest k_80 of any model", CO_["k80_max"], CN_["k80_max"])
    claim(
        "ridge puts the most forecast variance on the recent return-size group (live_feasible)",
        _yn(CO_["ridge_most_on_size"]),
        _yn(CN_["ridge_most_on_size"]),
    )
    claim(
        "ridge puts the least on the har_ma_1, har_ma_5 group (live_feasible)",
        _yn(CO_["ridge_least_on_har"]),
        _yn(CN_["ridge_least_on_har"]),
    )
    claim(
        "QLIKE significantly worse than all columns for every model at every k <= K (both designs)",
        f"K = {CO_['K_all']}",
        f"K = {CN_['K_all']}",
    )
    claim("the same, live_feasible", f"K = {CO_['K_lf']}", f"K = {CN_['K_lf']}")
    claim(
        "every model trades below its all-column Sharpe at every k <= 16",
        _yn(CO_["below_all_k16"]),
        _yn(CN_["below_all_k16"]),
    )
    claim(
        "ridge beats LightGBM on QLIKE at k = 1, 2 (both designs, intervals exclude 0)",
        _yn(CO_["ridge_beats_lgbm_k12"]),
        _yn(CN_["ridge_beats_lgbm_k12"]),
    )
    claim(
        "with all columns ridge ties the lasso and LightGBM on QLIKE (both designs)",
        _yn(CO_["all_columns_tie_qlike"]),
        _yn(CN_["all_columns_tie_qlike"]),
    )
    claim(
        "the lasso beats ridge on QLIKE at k = 4, 8 (all_features)",
        _yn(CO_["lasso_beats_ridge_k48_af"]),
        _yn(CN_["lasso_beats_ridge_k48_af"]),
    )
    claim(
        "ridge - lasso QLIKE gap narrows significantly to all columns from k (all_features)",
        ", ".join(CO_["af_rl_did_ks"]) or "none",
        ", ".join(CN_["af_rl_did_ks"]) or "none",
    )
    claim(
        "the same, live_feasible",
        ", ".join(CO_["lf_rl_did_ks"]) or "none",
        ", ".join(CN_["lf_rl_did_ks"]) or "none",
    )
    claim(
        "ridge - LightGBM QLIKE gap closes significantly to all columns from k (live_feasible)",
        ", ".join(CO_["lf_rg_did_ks"]) or "none",
        ", ".join(CN_["lf_rg_did_ks"]) or "none",
    )
    claim(
        "the same, all_features",
        ", ".join(CO_["af_rg_did_ks"]) or "none",
        ", ".join(CN_["af_rg_did_ks"]) or "none",
    )
    claim(
        "ridge-vs-lasso / ridge-vs-LightGBM Sharpe gaps at k <= 64 whose interval excludes 0",
        f"{CO_['n_sh_gap']} of {CO_['n_sh']}",
        f"{CN_['n_sh_gap']} of {CN_['n_sh']}",
    )
    claim(
        "... whose change between k and all columns excludes 0",
        f"{CO_['n_sh_did']} of {CO_['n_sh']}",
        f"{CN_['n_sh_did']} of {CN_['n_sh']}",
    )
    claim(
        "all-column Sharpe gaps (ridge - lasso, ridge - LightGBM) whose interval excludes 0",
        ", ".join(CO_["full_sharpe_gap_sig"]) or "none",
        ", ".join(CN_["full_sharpe_gap_sig"]) or "none",
    )
    claim(
        "PC1 collapse moves the ridge - lasso Sharpe gap significantly",
        ", ".join(CO_["collapse_sig"]) or "no",
        ", ".join(CN_["collapse_sig"]) or "no",
    )
    CLM = pd.DataFrame(rows)
    CLM.to_csv(OUT / "before_after_claims.csv", index=False)

    L: list[str] = []
    a = L.append
    a(
        "# Trees vs linear at the close, re-run on the de-duplicated per-bar design with the per-window tree mask (checklist I8)"
    )
    a("")
    a(
        "Written by `experiments/dvs_dedup_1530.py` (stages `capture`, `refit`, `carc`, `analyze`, which writes "
        "this file) from this folder's CSVs and `numbers.json` and, for the before / after, the first pass's "
        "(`results/dense_vs_sparse/`, `experiments/dense_vs_sparse_1530.py`, whose functions this re-run imports "
        "and runs unchanged). Nothing below is asserted: every statement of the first pass is re-evaluated on the "
        "new runs and its status recorded (`before_after_claims.csv`)."
    )
    a("")
    dg = NB[LF].get("design_gate", {})
    lin_chg = BS[BS.model.isin(["ridge", "lasso"]) & BS.max_rel_forecast_change.notna()]
    lg_ = BS[
        (BS.model == "LightGBM")
        & (BS.rule == "screen")
        & BS.max_rel_forecast_change.notna()
    ]
    lg_all, lg_sub = lg_[lg_.k == "all"], lg_[lg_.k != "all"]
    en_chg = GT[
        GT.check.str.startswith("this elastic net refit vs the first pass")
        & (GT.rows_above_1e6 > 0)
    ]
    mk_txt = []
    for b in BUCKETS:
        r = MK[(MK.bucket == b) & (MK.run.str.startswith("LightGBM all"))]
        if len(r):
            r0 = r.iloc[0]
            mk_txt.append(
                f"`{b}` {int(r0['kept_min'])}–{int(r0['kept_max'])} of {int(r0['columns_offered'])} "
                f"(median {r0['kept_median']:.0f})"
            )
    sub = MK[MK.run.str.contains("screen")]
    a(
        "**What changed, and nothing else.** (1) The design: the 12 session-edge columns `har_ma_k_x_open` "
        f"(all zero in a one-bar series: {dg.get('dropped_all_zero', 'n/a')} of 6) and `har_ma_k_x_close` "
        f"(byte-copies of `har_ma_k` at the 16:00 bar: {dg.get('dropped_copy_of_har', 'n/a')} of 6) are gone "
        f"(commit 47f7f9c): `live_feasible` {P_OLD[LF]} → {NB[LF]['n_columns']} columns, `all_features` "
        f"{P_OLD[AF]} → {NB[AF]['n_columns']}; gate: the new design is the old one minus exactly those 12 columns, "
        "bit for bit, same target and dates. (2) Every tree fit uses the per-window column mask "
        "(`src/models/window_mask.py`, commit f9a19b6, the user's decision of 2026-09-29): on each refit's own "
        "2000-session training window the constant columns and the byte-copies of an earlier column are dropped "
        "before the fit; TreeSHAP is mapped back to all columns with 0 for a dropped column. Kept columns per "
        f"LightGBM refit: {'; '.join(mk_txt)}; on the screened top-k subsets the mask drops a column in "
        f"{int(sub['refits_with_a_dropped_column'].sum())} of {int(sub['n_refits'].sum())} refits "
        "(`mask_counts.csv`). The linear models are unchanged in method: their identifiability mask already "
        "removed constant and duplicated columns at every tune."
    )
    a("")
    if len(lin_chg) and len(lg_all):
        la = {r.bucket: r for r in lg_all.itertuples()}
        a(
            "Consequently every ridge and lasso forecast is the first pass's: the largest relative change of any "
            f"ridge / lasso run (all columns and both top-k sweeps) is {lin_chg.max_rel_forecast_change.max():.1e} and "
            f"the sign(s) side differs on {int(lin_chg.trade_days_position_differs.max())} trade days"
            + (
                " (the elastic net, which enters only the density table, moves by up to "
                + "; ".join(
                    f"{r.max_rel:.1e} on {int(r.rows_above_1e6)} rows (`{r.bucket}`)"
                    for r in en_chg.itertuples()
                )
                + ": the warm-homotopy float path, see the gates)"
                if len(en_chg)
                else ""
            )
            + ". LightGBM on all "
            "columns moves: median relative change of the forecast "
            + " / ".join(
                f"{la[b].median_rel_forecast_change:.1e} (`{b}`)"
                for b in BUCKETS
                if b in la
            )
            + ", the side differs on "
            + " / ".join(
                f"{int(la[b].trade_days_position_differs)}" for b in BUCKETS if b in la
            )
            + f" of {NB[LF]['n_deck']} days; on the screened subsets (k ≤ {K_GRID[-1]}) the largest relative "
            f"change is {lg_sub.max_rel_forecast_change.max():.1e} and the side differs on at most "
            f"{int(lg_sub.trade_days_position_differs.max())} days (`before_after_sweep.csv`)."
        )
        a("")
    a(
        "**Same methodology.** The per-bar forecast of the bar ending 16:00 (issued at 15:30), 2000-session rolling "
        f"window; ridge / lasso / elastic net refit every session with the penalty re-chosen every {TUNE_PER} "
        f"sessions; LightGBM / XGBoost / random forest in the shipped configuration refit every {REFIT_EVERY} "
        "sessions, one thread per fit; the research scorer (the 16:00 bar recalibrated alone, "
        f"(ŷ² + s)·B); QLIKE and the deck's 15:30 **sign(s)** rule on the {NB[LF]['n_deck']} trade days; 95 % "
        f"circular block bootstrap (block {NUM['constants']['BOOT_BLOCK']}, {NUM['constants']['BOOT_B']:,} draws, "
        "the first pass's seed, one set of draws for every run, the first pass's runs included, so every "
        "difference and every new-minus-first-pass change is paired)."
    )
    a("")
    tm = NUM.get("timing", {})
    parts = []
    if "local" in tm:
        parts.append(
            "the linear refits and the LightGBM sweep with its all-column references locally, in the first "
            f"pass's environment (LightGBM {NUM['versions_local'].get('lightgbm')}; {tm['local']['peak_processes']} "
            f"single-threaded processes on a shared laptop, {tm['local']['cpu_hours']:.2f} process-hours, "
            f"{tm['local']['wall_min']:.0f} min wall)"
        )
    cf = tm.get("cluster_fleet")
    sa = tm.get("cluster_sacct")
    if cf:
        parts.append(
            "the density table's masked TreeSHAP (LightGBM, XGBoost, random forest on all columns) and the "
            f"no-mask gate on the cluster (LightGBM {cf['versions'].get('lightgbm')}, XGBoost "
            f"{cf['versions'].get('xgboost')}, shap {cf['versions'].get('shap')})"
            + (
                f": {sa['jobs']}; {sa['alloc_cpu_hours']:.2f} allocated CPU-hours, peak {sa['peak_cpus']} CPUs, "
                f"{sa['wall_min_total']:.0f} min of job wall time"
                if sa
                else ""
            )
        )
    a(
        "**Where it ran** (as the first pass split it: the sweep locally, the density table's TreeSHAP from "
        "cluster runs): " + "; ".join(parts) + "."
    )
    ms = NUM.get("tree_spec_model_section", {})
    if any(k.startswith("cluster_md5_") for k in ms):
        a("")
        a(
            f"The tree spec changed locally since the cluster runs (md5 {ms['local_md5'][:12]} now; another "
            "campaign's WINDOW_MASK axis): "
            + "; ".join(
                f"the cluster ran md5 {k[12:]}: {v}"
                for k, v in ms.items()
                if k.startswith("cluster_md5_")
            )
            + "."
        )
    a("")
    a(
        "Series names are the design's column stems: `har_ma_*` = the target's own HAR ladder (realized variance, "
        "means of the last 1, 5, 25, 125, 625, 3125 bars), `adj_sumabsret_ma_*` = absolute returns, "
        "`adj_sumret4_ma_*` = 4th-power returns, `adj_sumpret2_ma_*` = upside squared returns, "
        "`adj_sumbipow_ma_*` = bipower variation, `adj_sumvolume_ma_*` = ES volume, `adj_vix_ma_*` / "
        "`adj_vvix_ma_*` / `adj_vix3m_ma_*` = VIX, VVIX, VIX3M, `*_ewstock` / `*_vwstock` = the same statistics on "
        "equal- / value-weighted constituent stocks."
    )
    a("")
    # ---- brief
    a("## Answer in brief: the first pass's statements, first pass → de-dup + mask")
    a("")
    a("| statement | first pass (old design) | de-dup + mask | same |")
    a("|---|---|---|---|")
    for _, r in CLM.iterrows():
        a(
            f"| {r['claim']} | {r['first_pass']} | {r['dedup_mask']} | {'yes' if r['same'] else 'no'} |"
        )
    a("")
    BDh = pd.read_csv(OUT / "before_after_density.csv")
    BDh = BDh[(BDh.level == "column") & (BDh["sample"] == "all forecasts")]
    BGRh = pd.read_csv(OUT / "before_after_groups.csv")
    n_same = int(CLM["same"].sum())
    a(
        f"Headline ({n_same} of {len(CLM)} statements read the same; the rest are numbers that moved):"
    )
    a("")
    for b in BUCKETS:
        tr_ = []
        for m in ("LightGBM", "XGBoost", "random forest"):
            r = BDh[(BDh.bucket == b) & (BDh.model == m)]
            if len(r):
                r = r.iloc[0]
                nm_ = r.get("neff__dedup_nomask")
                tr_.append(
                    f"{m} {r['neff__first_pass']:.1f} → {r['neff__dedup_mask']:.1f}"
                    + (f" (no mask {nm_:.1f})" if pd.notna(nm_) else "")
                )
        rr = BDh[(BDh.bucket == b) & (BDh.model == "ridge")].iloc[0]
        tb = BDh[
            (BDh.bucket == b) & BDh.model.isin(["LightGBM", "XGBoost", "random forest"])
        ]
        falls = bool((tb["neff__dedup_mask"] < tb["neff__first_pass"]).all())
        # the share of each tree's N_eff change already there without the mask
        dd = (tb["neff__first_pass"] - tb["neff__dedup_nomask"]) / (
            tb["neff__first_pass"] - tb["neff__dedup_mask"]
        )
        a(
            f"* `{b}`, density: the trees' N_eff {'falls' if falls else 'changes'}: {', '.join(tr_)}; ridge's "
            f"{rr['neff__first_pass']:.1f} → {rr['neff__dedup_mask']:.1f}. Of each tree's change, the "
            f"de-duplication without the mask already gives {100 * dd.min():.0f}–{100 * dd.max():.0f} %."
        )
    hg = BGRh[BGRh.members.str.split().apply(lambda v: v == ["har_ma_1", "har_ma_5"])]
    if len(hg):
        bits_ = []
        for r in hg.itertuples():
            rd = r._asdict()
            bits_.append(
                f"`{r.bucket}` "
                + ", ".join(
                    f"{LABEL[m]} {100 * rd[f'var_share_{m}__first_pass']:.0f} → {100 * rd[f'var_share_{m}__dedup_mask']:.0f} %"
                    for m in ("lgbm", "xgb", "rf")
                )
            )
        a(
            "* The `har_ma_1`, `har_ma_5` group's share of the trees' forecast variance rises ("
            + "; ".join(bits_)
            + f"): in the first pass the trees also split on the byte-copies `har_ma_k_x_close`, which carried "
            f"{100 * SE['first_pass_share_session_edge'].min():.0f}–{100 * SE['first_pass_share_session_edge'].max():.0f} % "
            "of their mean |TreeSHAP| and sat outside every group (a copy is not an identifiable column)."
            if len(SE)
            else ")."
        )
    for r in lg_all.itertuples():
        a(
            f"* `{r.bucket}`, LightGBM on all columns, first pass → de-dup + mask: QLIKE "
            f"{r.QLIKE_deck__first_pass:.4f} → {r.QLIKE_deck__dedup_mask:.4f} (change {r.QLIKE_new_minus_first_pass:+.4f} "
            f"{_iv(r.QLIKE_new_minus_first_pass_lo, r.QLIKE_new_minus_first_pass_hi)}), sign(s) Sharpe mid "
            f"{r.Sharpe_mid__first_pass:.2f} → {r.Sharpe_mid__dedup_mask:.2f} (change {r.Sharpe_new_minus_first_pass:+.2f} "
            f"{_iv(r.Sharpe_new_minus_first_pass_lo, r.Sharpe_new_minus_first_pass_hi, 2)})."
        )
    for r in RF[RF.row == "lgbm_all - lgbm_nomask_all"].itertuples():
        a(
            f"* `{r.bucket}`, the mask alone (LightGBM on all columns, masked minus unmasked, both local): QLIKE "
            f"{r.dQLIKE:+.4f} {_iv(r.dQLIKE_lo, r.dQLIKE_hi)}, Sharpe {r.dSharpe:+.2f} {_iv(r.dSharpe_lo, r.dSharpe_hi, 2)}."
        )
    i_sw = int(CLM.index[CLM.claim.str.startswith("QLIKE significantly worse")][0])
    sweep_same = bool(CLM.iloc[i_sw:]["same"].all())
    picks_same = bool(BP["identical"].all())
    lin_same = (
        bool(lin_chg.max_rel_forecast_change.max() < 1e-8) if len(lin_chg) else False
    )
    moved = set(
        lg_sub.loc[lg_sub.max_rel_forecast_change > 0, ["bucket", "k"]].itertuples(
            index=False, name=None
        )
    )
    MKs = MK[MK.run.str.contains("screen")].copy()
    MKs["k"] = MKs.run.str.extract(r"k(\d+)$")[0]
    dropped = set(
        MKs.loc[MKs.refits_with_a_dropped_column > 0, ["bucket", "k"]].itertuples(
            index=False, name=None
        )
    )
    a(
        f"* The sweep, hypothesis-test and collapse statements of the first pass (from "
        f"'{CLM.iloc[i_sw]['claim']}' on) read {'the same' if sweep_same else 'differently in part'}; the screen "
        f"picks {'the same' if picks_same else 'different'} columns; the linear runs are "
        f"{'unchanged (≤ 1e-8)' if lin_same else 'changed'}; LightGBM's subset forecasts change in "
        f"{len(moved)} (design, k) cells"
        + (
            f" ({', '.join(f'`{b}` k = {k}' for b, k in sorted(moved))})"
            if moved
            else ""
        )
        + f", the mask drops a column in {len(dropped)}"
        + (
            f" ({', '.join(f'`{b}` k = {k}' for b, k in sorted(dropped))})"
            if dropped
            else ""
        )
        + "."
    )
    a("")
    fg_new, fg_old = CN_["full_gap"], CO_["full_gap"]
    a(
        "All-column gaps, first pass → de-dup + mask (ridge minus the other model; QLIKE, then Sharpe mid): "
        + "; ".join(
            f"`{b}` {pr}: {fg_old[(b, pr)]['dQLIKE']:+.4f} {_iv(fg_old[(b, pr)]['dQLIKE_lo'], fg_old[(b, pr)]['dQLIKE_hi'])}"
            f" → {fg_new[(b, pr)]['dQLIKE']:+.4f} {_iv(fg_new[(b, pr)]['dQLIKE_lo'], fg_new[(b, pr)]['dQLIKE_hi'])}, "
            f"{fg_old[(b, pr)]['dSharpe']:+.2f} {_iv(fg_old[(b, pr)]['dSharpe_lo'], fg_old[(b, pr)]['dSharpe_hi'], 2)}"
            f" → {fg_new[(b, pr)]['dSharpe']:+.2f} {_iv(fg_new[(b, pr)]['dSharpe_lo'], fg_new[(b, pr)]['dSharpe_hi'], 2)}"
            for b in BUCKETS
            for pr in (RL, RG)
        )
        + ". The Sharpe intervals of the k-sweep gaps are "
        f"{CN_['sharpe_ci_width'][0]:.1f}–{CN_['sharpe_ci_width'][1]:.1f} wide (first pass "
        f"{CO_['sharpe_ci_width'][0]:.1f}–{CO_['sharpe_ci_width'][1]:.1f})."
    )
    a("")
    # ---- 1. density
    a("## 1. Signal density: how many inputs does each forecast use?")
    a("")
    mx_lin = GT[
        GT.check.str.contains("contributions")
        & GT.check.str.startswith(("ridge", "lasso", "elastic"))
    ]["max_rel"].max()
    mx_tree = GT[
        GT.check.str.contains("contributions")
        & GT.check.str.startswith(("LightGBM", "XGBoost", "random"))
    ]["max_rel"].max()
    a(
        "Contributions as in the first pass: linear β_j (x_j − window mean_j) (linear SHAP; additivity "
        f"{mx_lin:.1e}); trees: TreeSHAP of the masked T10 walk run on the cluster for this study "
        f"(additivity ≤ {mx_tree:.1e} relative; the first pass read the tree campaign's stored unmasked TreeSHAP; "
        "the stored unmasked T10 runs on the new design are the **de-dup, no mask** rows). share_j = mean "
        "|contribution_j| / sum; N_eff = 1 / Σ share²; k_x = the fewest columns, in share order, reproducing x of "
        "the variance of the input-driven forecast; greedy k_95 = columns added one at a time. All 1,469 forecasts; "
        "`before_after_density.csv` has every column, the trade-day sample and the series level."
    )
    a("")
    a(
        "| design | model | N_eff first pass | N_eff de-dup + mask | (no mask) | k_80 | k_95 | greedy k_95 | columns with any weight | N_eff groups |"
    )
    a("|---|---|---:|---:|---:|---|---|---|---|---|")
    BD = pd.read_csv(OUT / "before_after_density.csv")
    BD = BD[(BD.level == "column") & (BD["sample"] == "all forecasts")]
    for b in BUCKETS:
        for i, m in enumerate(DENS_MODELS):
            r = BD[(BD.bucket == b) & (BD.model == m)]
            if not len(r):
                continue
            r = r.iloc[0]

            def pair(c: str, nd: int = 0) -> str:
                o, n = r.get(f"{c}__first_pass"), r.get(f"{c}__dedup_mask")
                f_ = f"{{:.{nd}f}}"
                return (
                    (f_.format(o) if pd.notna(o) else "–")
                    + " → "
                    + (f_.format(n) if pd.notna(n) else "–")
                )

            nmv = r.get("neff__dedup_nomask")
            a(
                f"| {b if i == 0 else ''} | {m} | {r['neff__first_pass']:.1f} | **{r['neff__dedup_mask']:.1f}** | "
                f"{'' if pd.isna(nmv) else f'{nmv:.1f}'} | {pair('k80')} | {pair('k95')} | {pair('greedy_k95')} | "
                f"{pair('n_active')} | {pair('neff_groups', 1)} |"
            )
    a("")
    if len(SE):
        a(
            "Why the trees' numbers move: in the first pass each `har_ma_k_x_close` column was a byte-copy of "
            "`har_ma_k`, and the trees split credit between the two. Share of mean |TreeSHAP| on the 12 session-edge "
            "columns in the first pass's stored tree runs, and the `har_ma` ladder's share (with its copies) then vs "
            "now (masked):"
        )
        a("")
        a(
            "| design | model | session-edge share, first pass | `har_ma_*` + copies, first pass | `har_ma_*`, de-dup + mask |"
        )
        a("|---|---|---:|---:|---:|")
        for _, r in SE.iterrows():
            a(
                f"| {r['bucket']} | {r['model']} | {100 * r['first_pass_share_session_edge']:.1f} % | "
                f"{100 * r['first_pass_share_har_ma_plus_x_close']:.1f} % | {100 * r['dedup_mask_share_har_ma']:.1f} % |"
            )
        a("")
    a(
        "Share of each model's forecast variance carried by a correlated group (|corr| > 0.8, complete linkage; "
        f"{NB[LF]['groups']['n_groups']} groups of the {NB[LF]['groups']['n_kept']} identifiable `live_feasible` "
        f"columns, {NB[AF]['groups']['n_groups']} of {NB[AF]['groups']['n_kept']} `all_features`; first pass "
        f"{NBo[LF]['groups']['n_groups']} of {NBo[LF]['groups']['n_kept']} and {NBo[AF]['groups']['n_groups']} of "
        f"{NBo[AF]['groups']['n_kept']}), de-dup + mask with the first pass in brackets (`before_after_groups.csv`):"
    )
    a("")
    a(
        "| design | group (design columns) | ridge | lasso | LightGBM | XGBoost | random forest |"
    )
    a("|---|---|---:|---:|---:|---:|---:|")
    GRo = TO["groups"]
    for b in BUCKETS:
        gg = (
            GRn[GRn.bucket == b].sort_values("var_share_ridge", ascending=False).head(4)
        )
        for i, (_, r) in enumerate(gg.iterrows()):
            ro = GRo[(GRo.bucket == b) & (GRo.members == r["members"])]
            cells = []
            for m in ("ridge", "reclasso", "lgbm", "xgb", "rf"):
                o = f" ({100 * ro.iloc[0][f'var_share_{m}']:.1f})" if len(ro) else ""
                cells.append(f"{100 * r[f'var_share_{m}']:.1f} %{o}")
            a(
                f"| {b if i == 0 else ''} | {_members(r['members'])} | "
                + " | ".join(cells)
                + " |"
            )
    a("")
    # ---- 2. sweep
    a("## 2. The sparsity sweep: ridge, lasso and LightGBM on the same top-k inputs")
    a("")
    same_picks = int(BP["identical"].sum())
    a(
        "The first pass's causal rule, unchanged: at each of the six penalty re-choices the k columns with the "
        "largest |correlation with the target| over the 2000 sessions before the block (constant or duplicated "
        "columns never picked), the same columns for all three models; LightGBM's fits also apply the per-window "
        f"mask to those k columns. The screen picks the same columns as in the first pass in {same_picks} of "
        f"{len(BP)} (design, k, block) cells (`before_after_picks.csv`)."
    )
    a("")
    a(
        f"QLIKE ({NB[LF]['n_deck']} trade days) and sign(s) Sharpe (mid) against k, de-dup + mask; * = the paired "
        "interval against the same model on all columns excludes zero; first pass in brackets:"
    )
    a("")
    a(
        "| k | ridge QLIKE | lasso QLIKE | LightGBM QLIKE | ridge Sharpe | lasso Sharpe | LightGBM Sharpe |"
    )
    a("|---|---:|---:|---:|---:|---:|---:|")
    SWn, SWo = TN["sweep"], TO["sweep"]
    for b in BUCKETS:
        a(f"| **{b}** | | | | | | |")
        for kx in (*ks, "all"):
            cells = []
            for key in ("QLIKE", "Sharpe"):
                for m in ("ridge", "lasso", "LightGBM"):
                    r = SWn[
                        (SWn.bucket == b)
                        & (SWn.model == m)
                        & (SWn.k == kx)
                        & (SWn.rule == "screen")
                    ].iloc[0]
                    ro = SWo[
                        (SWo.bucket == b)
                        & (SWo.model == m)
                        & (SWo.k == kx)
                        & (SWo.rule == "screen")
                    ].iloc[0]
                    col_ = "QLIKE_deck" if key == "QLIKE" else "Sharpe_mid"
                    star = (
                        " *"
                        if kx != "all" and _sig(r[f"d{key}_lo"], r[f"d{key}_hi"])
                        else ""
                    )
                    fmt = "{:.4f}" if key == "QLIKE" else "{:.2f}"
                    cells.append(
                        f"{fmt.format(r[col_])}{star} ({fmt.format(ro[col_])})"
                    )
            lab = (
                kx
                if kx != "all"
                else f"all ({NB[b]['n_columns']}; first pass {P_OLD[b]})"
            )
            a(f"| {lab} | " + " | ".join(cells) + " |")
    a("")
    a(
        "LightGBM, the model whose forecasts change: de-dup + mask minus first pass, paired (`before_after_sweep.csv`):"
    )
    a("")
    a(
        "| design | k | QLIKE change | Sharpe change | median rel. forecast change | days the side differs |"
    )
    a("|---|---|---|---|---:|---:|")
    for b in BUCKETS:
        for kx in (*ks, "all"):
            r = BS[
                (BS.bucket == b)
                & (BS.model == "LightGBM")
                & (BS.k == kx)
                & (BS.rule == "screen")
            ]
            if not len(r) or pd.isna(r.iloc[0].get("QLIKE_new_minus_first_pass")):
                continue
            r = r.iloc[0]
            a(
                f"| {b} | {kx} | {r['QLIKE_new_minus_first_pass']:+.4f} "
                f"{_iv(r['QLIKE_new_minus_first_pass_lo'], r['QLIKE_new_minus_first_pass_hi'])} | "
                f"{r['Sharpe_new_minus_first_pass']:+.2f} "
                f"{_iv(r['Sharpe_new_minus_first_pass_lo'], r['Sharpe_new_minus_first_pass_hi'], 2)} | "
                f"{r['median_rel_forecast_change']:.1e} | {int(r['trade_days_position_differs'])} |"
            )
    a("")
    GPn = TN["sweep_gaps"]

    def gp(b: str, pair: str, k: str) -> pd.Series:
        return GPn[(GPn.bucket == b) & (GPn.pair == pair) & (GPn.k == k)].iloc[0]

    a(
        "**The hypothesis test** (`sweep_gaps.csv`; first pass vs now in `before_after_gaps.csv`, with the paired change of each gap):"
    )
    a("")
    for pr in (RL, RG, LG):
        for b in BUCKETS:
            sig_ks = [
                f"k = {k} {gp(b, pr, k)['dQLIKE']:+.4f} {_iv(gp(b, pr, k)['dQLIKE_lo'], gp(b, pr, k)['dQLIKE_hi'])}"
                for k in ks
                if _sig(gp(b, pr, k)["dQLIKE_lo"], gp(b, pr, k)["dQLIKE_hi"])
            ]
            did_ks = [
                f"k = {k} {gp(b, pr, k)['did_QLIKE']:+.4f} {_iv(gp(b, pr, k)['did_QLIKE_lo'], gp(b, pr, k)['did_QLIKE_hi'])}"
                for k in ks
                if _sig(gp(b, pr, k)["did_QLIKE_lo"], gp(b, pr, k)["did_QLIKE_hi"])
            ]
            ga = gp(b, pr, "all")
            a(
                f"* {pr}, QLIKE, `{b}`: gap excludes zero at {'; '.join(sig_ks) or 'no k'}; all columns "
                f"{ga['dQLIKE']:+.4f} {_iv(ga['dQLIKE_lo'], ga['dQLIKE_hi'])}; the gap's change from k to all "
                f"columns excludes zero at {'; '.join(did_ks) or 'no k'}."
            )
    chg_sig = (
        BG[
            BG["gap_QLIKE_new_minus_first_pass_lo"].notna()
            & (
                (BG["gap_QLIKE_new_minus_first_pass_lo"] > 0)
                | (BG["gap_QLIKE_new_minus_first_pass_hi"] < 0)
            )
        ]
        if "gap_QLIKE_new_minus_first_pass_lo" in BG
        else BG.iloc[0:0]
    )
    chg_sig_sh = (
        BG[
            BG["gap_Sharpe_new_minus_first_pass_lo"].notna()
            & (
                (BG["gap_Sharpe_new_minus_first_pass_lo"] > 0)
                | (BG["gap_Sharpe_new_minus_first_pass_hi"] < 0)
            )
        ]
        if "gap_Sharpe_new_minus_first_pass_lo" in BG
        else BG.iloc[0:0]
    )
    n_chg = (
        int(BG["gap_QLIKE_new_minus_first_pass_lo"].notna().sum())
        if "gap_QLIKE_new_minus_first_pass_lo" in BG
        else 0
    )
    a(
        f"* Sharpe: of the {CN_['n_sh']} ridge-vs-lasso and ridge-vs-LightGBM gaps at k = 1 … 64, "
        f"{CN_['n_sh_gap']} {'has' if CN_['n_sh_gap'] == 1 else 'have'} an interval excluding zero (first pass {CO_['n_sh_gap']}); the change of the gap "
        f"between k and all columns excludes zero in {CN_['n_sh_did']} (first pass {CO_['n_sh_did']}); across all "
        f"{CN_['n_sh_all']} Sharpe comparisons {CN_['n_sh_all_did']} changes of gap exclude zero (first pass "
        f"{CO_['n_sh_all_did']})."
    )
    a(
        f"* Did a gap itself move between the designs? Of the {n_chg} (pair, k, design) gaps, the paired change "
        f"new minus first pass excludes zero for QLIKE in {len(chg_sig)}"
        + (
            " ("
            + "; ".join(
                f"`{r.bucket}` {r.pair} k = {r.k} {r.gap_QLIKE_new_minus_first_pass:+.4f} "
                f"{_iv(r.gap_QLIKE_new_minus_first_pass_lo, r.gap_QLIKE_new_minus_first_pass_hi)}"
                for r in chg_sig.itertuples()
            )
            + ")"
            if len(chg_sig)
            else ""
        )
        + f" and for Sharpe in {len(chg_sig_sh)}"
        + (
            " ("
            + "; ".join(
                f"`{r.bucket}` {r.pair} k = {r.k} {r.gap_Sharpe_new_minus_first_pass:+.2f} "
                f"{_iv(r.gap_Sharpe_new_minus_first_pass_lo, r.gap_Sharpe_new_minus_first_pass_hi, 2)}"
                for r in chg_sig_sh.itertuples()
            )
            + ")"
            if len(chg_sig_sh)
            else ""
        )
        + "."
    )
    a("")
    # ---- LightGBM references
    a(
        "**LightGBM's all-column references** (`lgbm_references.csv`; research scorer, 866 days):"
    )
    a("")
    for b in BUCKETS:
        rr = RF[RF.bucket == b]
        vals = {r.row: r for r in rr.itertuples()}
        bits = []
        for t in ("lgbm_all", "lgbm_nomask_all", "lgbm_t10_stored", "lgbm_carc_mask"):
            if t in vals:
                v = vals[t]
                bits.append(
                    f"{v.what}: QLIKE {v.QLIKE_deck:.4f}, Sharpe {v.Sharpe_mid:.2f} mid / {v.Sharpe_crossed:.2f} crossed"
                )
        for t in (
            "lgbm_all - lgbm_nomask_all",
            "lgbm_carc_mask - lgbm_t10_stored",
            "lgbm_nomask_all - lgbm_t10_stored",
            "lgbm_all - lgbm_carc_mask",
        ):
            if t in vals:
                v = vals[t]
                bits.append(
                    f"{v.what}: QLIKE {v.dQLIKE:+.4f} {_iv(v.dQLIKE_lo, v.dQLIKE_hi)}, Sharpe {v.dSharpe:+.2f} "
                    f"{_iv(v.dSharpe_lo, v.dSharpe_hi, 2)}"
                )
        a(f"* `{b}`: " + "; ".join(bits) + ".")
    a("")
    # ---- 3. collapse
    a(
        "## 3. Correlated-input groups: collapse each group to its first principal component"
    )
    a("")
    pcs = [
        blk["pc1_share_median"]
        for b in BUCKETS
        for blk in NB[b].get("collapse_blocks", [])
    ]
    a(
        "The first pass's rule, unchanged (groups by complete linkage at |corr| > 0.8 on the 2000 sessions before "
        "each 250-session block, each multi-column group replaced by its first principal component, ridge and lasso "
        f"refit on the collapsed design). The first component carries a median {100 * min(pcs):.0f}–"
        f"{100 * max(pcs):.0f} % of a group's window variance. De-dup + mask, first pass in brackets "
        "(`before_after_collapse.csv`):"
    )
    a("")
    a("| design | | QLIKE | Sharpe mid | vs the same model on the full design |")
    a("|---|---|---:|---:|---|")
    CLn, CLo = TN["collapse"], TO["collapse"]
    for b in BUCKETS:
        for i, m in enumerate(("ridge", "lasso")):
            r = CLn[(CLn.bucket == b) & (CLn.row == f"{m} collapsed vs full")].iloc[0]
            ro = CLo[(CLo.bucket == b) & (CLo.row == f"{m} collapsed vs full")].iloc[0]
            a(
                f"| {b if i == 0 else ''} | {m}, collapsed | {r['QLIKE_deck']:.4f} ({ro['QLIKE_deck']:.4f}) | "
                f"{r['Sharpe_mid']:.2f} ({ro['Sharpe_mid']:.2f}) | QLIKE {r['dQLIKE']:+.4f} "
                f"{_iv(r['dQLIKE_lo'], r['dQLIKE_hi'])}, Sharpe {r['dSharpe']:+.2f} {_iv(r['dSharpe_lo'], r['dSharpe_hi'], 2)} |"
            )
    a("")
    a(
        "Ridge − lasso Sharpe gap, full design minus collapsed: "
        + "; ".join(
            f"`{b}` {CN_['collapse_did_sharpe'][b][0]:+.2f} {_iv(CN_['collapse_did_sharpe'][b][1], CN_['collapse_did_sharpe'][b][2], 2)} "
            f"(first pass {CO_['collapse_did_sharpe'][b][0]:+.2f} {_iv(CO_['collapse_did_sharpe'][b][1], CO_['collapse_did_sharpe'][b][2], 2)})"
            for b in BUCKETS
        )
        + "."
    )
    a("")
    # ---- gates
    a("## Gates and caveats")
    a("")
    for _, r in GT.iterrows():
        bits = []
        if pd.notna(r.get("max_rel")):
            bits.append(
                f"max {'abs' if 'bit for bit' in r['check'] else 'rel'} diff {r['max_rel']:.1e}"
            )
        if pd.notna(r.get("rows_above_1e6")):
            bits.append(f"rows > 1e-6: {int(r['rows_above_1e6'])}")
            if pd.notna(r.get("first_bad")) and r["first_bad"] >= 0:
                bits.append(f"forecast rows {int(r['first_bad'])}–{int(r['last_bad'])}")
        if pd.notna(r.get("passed")) and str(r.get("passed")) != "nan":
            bits.append("PASS" if str(r["passed"]) == "True" else "differs")
        if isinstance(r.get("note"), str):
            bits.append(r["note"])
        a(f"* `{r['bucket']}` {r['check']}: " + ", ".join(bits) + ".")
    a("")
    a(
        "* Only the linear arms' own identifiability mask and the trees' new per-window mask decide which columns "
        "a fit sees; the screen's eligibility rule (`identifiable` on the block's window) is the first pass's."
    )
    a(
        "* LightGBM without the mask: this study's walk gives bit-identical forecasts locally and on the cluster, "
        "and reproduces the stored T10 XGBoost and random forest forecasts on the cluster to the CSV's rounding, "
        "but not the stored T10 LightGBM forecasts (the gate rows above: same cluster, same spec, same design); "
        "the first pass saw the same LightGBM gap and attributed it to the platform. With the mask, local and "
        "cluster LightGBM are bit-identical on every row. Every comparison inside the sweep uses local runs "
        "only; the density table uses the cluster's runs only, as in the first pass."
    )
    a(
        f"* Sample: {NB[LF]['n_deck']} trade days. Sharpe differences between variants carry intervals "
        f"{CN_['sharpe_ci_width'][0]:.1f}–{CN_['sharpe_ci_width'][1]:.1f} wide; QLIKE differences of about "
        "0.005 are resolvable."
    )
    a("")
    a(
        "Files: `density.csv` (variant `de-dup + mask` / `de-dup, no mask`), `groups.csv`, `group_concentration.csv`, "
        "`sweep.csv`, `sweep_gaps.csv`, `sweep_gap_counts.csv`, `screen_picks.csv`, `collapse.csv`, "
        "`lgbm_references.csv`, `mask_counts.csv`, `gates.csv`, `numbers.json`; before / after: "
        "`before_after_claims.csv`, `before_after_density.csv`, `before_after_groups.csv`, `before_after_sweep.csv`, "
        "`before_after_gaps.csv`, `before_after_collapse.csv`, `before_after_picks.csv`, "
        "`before_after_session_edge.csv`; figures `density_curves.png`, `sweep_qlike_sharpe_vs_k.png`, "
        "`before_after_sweep_vs_k.png`."
    )
    (OUT / "SUMMARY.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {OUT / 'SUMMARY.md'} ({len(L)} lines)")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "analyze"
    if stage == "capture":
        capture(sys.argv[2])
    elif stage == "refit":
        refit()
    elif stage == "carc":
        carc(sys.argv[2])
    elif stage == "analyze":
        analyze()
        write_summary()
    elif stage == "summary":
        write_summary()
    else:
        raise SystemExit(f"unknown stage {stage}")
