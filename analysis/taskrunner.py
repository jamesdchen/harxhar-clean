"""Split an experiment into independent cells so a cluster can run them.

Every analysis script in this session is monolithic: one process loops over all
arms, holds two gigabytes of design matrix and writes one JSON at the end. That
shape cannot be an array job. A 46-minute run is a single task, it cannot use
more than one node, and if it dies at minute 44 the whole thing is lost -- which
happened twice here already.

This is the decomposition. Three modes:

    --manifest            print the cell list and its size, for the array bound
    --task-id N           compute exactly cell N and write its own file
    --harvest             combine the cells, run the inference, write the JSON

Design decisions that matter for a preemptible queue:

  * each task writes PREDICTIONS, not a scalar loss.  1.75 MB per cell, and it
    means every downstream question -- DM at any bandwidth, block bootstrap,
    era splits, model confidence sets -- is answerable at harvest without
    re-running anything.  Storing only the loss would force a re-run for each
    new inference, which is how the four-way robustness became expensive.
  * tasks are IDEMPOTENT: an existing output is skipped, so a requeue after
    preemption costs only the cells that were actually lost.
  * the manifest is content-hashed into each cell file.  If a cell definition
    changes, harvest refuses to mix old and new outputs rather than silently
    averaging two different experiments -- the incompatible-footings failure,
    which this project has committed four times.
  * cells are ordered longest-first, so a heterogeneous array does not end with
    one slow task holding the queue.

The worked example is cadence x design, the experiment currently running
locally as one 15-minute process.  As 12 cells it is 12 concurrent tasks.
"""

from __future__ import annotations
import argparse, hashlib, json, os, re, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT_DIR = os.path.join(ROOT, "writeup", "stats")
TASK_DIR = os.path.join(ROOT, "results", "tasks")
W = 24000
LAMS = [0.0] + [10.0 ** e for e in range(-2, 8)]
NBLOCK = 8


def manifest(run):
    """Cells for a run, longest-first so the array does not tail off."""
    if run == "cadence_x_design":
        cells = [{"design": d, "cadence": c, "cache": "b2_mmap_warm"}
                 for c in (100, 250, 1000)
                 for d in ("full + exog x clk", "full-1077", "ridge-504", "backbone")]
    elif run == "interactions":
        cells = [{"design": d, "cadence": 1000, "cache": "b2_mmap_warm"}
                 for d in ("full + exog x clk", "full-1077", "ridge-504", "backbone")]
    else:
        raise SystemExit(f"unknown run: {run}")
    for i, c in enumerate(cells):
        c["id"] = i
    return cells


def mhash(cells):
    return hashlib.sha256(
        json.dumps(cells, sort_keys=True).encode()).hexdigest()[:12]


def build(cache, design):
    D = os.path.join(ROOT, "results", cache)
    X = np.load(os.path.join(D, "X.npy"), mmap_mode="r")
    y = np.load(os.path.join(D, "y.npy")); b = np.load(os.path.join(D, "b.npy"))
    names = [str(s) for s in np.load(os.path.join(D, "names.npy"), allow_pickle=True)]
    n = len(y)
    har = np.array([j for j, nm in enumerate(names)
                    if re.match(r"^har_ma_\d+$", nm)], int)
    chan = {}
    for j, nm in enumerate(names):
        m = re.match(r"^adj_(.+)_ma_(\d+)$", nm)
        if m: chan.setdefault(m.group(1), []).append((int(m.group(2)), j))
    for c in chan: chan[c].sort()
    labels = [c for c in chan if len(chan[c]) == 12]
    lad = np.array([j for c in labels for _, j in chan[c]], int)
    rest = np.array([j for j in range(len(names)) if j not in set(har)], int)
    clk = np.array([names.index(c) for c in ("is_open", "is_close", "is_overnight")
                    if c in names], int)
    H = np.asarray(X[:, har], np.float64)
    if design == "backbone":
        F = H
    elif design == "ridge-504":
        F = np.hstack([H, np.asarray(X[:, lad], np.float64)])
    elif design == "full-1077":
        F = np.hstack([H, np.asarray(X[:, rest], np.float64)])
    else:
        Z = np.asarray(X[:, lad], np.float64); K = np.asarray(X[:, clk], np.float64)
        F = np.hstack([H, np.asarray(X[:, rest], np.float64)]
                      + [Z * K[:, [i]] for i in range(K.shape[1])])
    del X
    return F, y, b, n, len(har)


def run_cell(cell, out_path):
    from fastwalk import RollingGram, block, predict, rss, solve_path
    t0 = time.time()
    F, y, b, n, nh = build(cell["cache"], cell["design"])
    CAD = cell["cadence"]; npen = max(F.shape[1] - nh, 1)
    T = np.eye(F.shape[1]); rvv = y ** 2 * b
    v0, v1 = int(n * 0.55), int(n * 0.65)
    rg = RollingGram(F, y, W)
    num = {L: 0.0 for L in LAMS}; den = {L: 0 for L in LAMS}
    for s in range(v0 - ((v0 - W) % CAD), v1, CAD):
        e = min(s + CAD, v1)
        if e <= v0: continue
        G, c = rg.advance(s); M, vv = block(G, c, T)
        a, z = max(s, v0), e
        for L, cf in solve_path(M, vv, npen, LAMS).items():
            pv = (predict(F[a:z], T, cf) ** 2 + rss(M, vv, cf, rg.yy) / W) * b[a:z]
            rt = rvv[a:z]; mk = (rt > 0) & np.isfinite(pv) & (pv > 0)
            if mk.sum():
                q = rt[mk] / pv[mk]
                num[L] += float(np.sum(q - np.log(q) - 1.0)); den[L] += int(mk.sum())
    lam = min((num[L] / den[L], L) for L in LAMS if den[L])[1]
    rg = RollingGram(F, y, W)
    out = np.full(n - W, np.nan)
    for s in range(W, n, CAD):
        e = min(s + CAD, n)
        G, c = rg.advance(s); M, vv = block(G, c, T)
        cf = solve_path(M, vv, npen, [lam])[lam]
        out[s - W:e - W] = (predict(F[s:e], T, cf) ** 2
                            + rss(M, vv, cf, rg.yy) / W) * b[s:e]
    np.savez_compressed(out_path, pred=out.astype(np.float32), lam=lam,
                        cols=F.shape[1], secs=time.time() - t0,
                        cell=json.dumps(cell))
    print(f"cell {cell['id']} {cell['design']} cad={CAD} cols={F.shape[1]} "
          f"lambda*={lam:g} ({time.time()-t0:.0f}s) -> {out_path}")


def harvest(run, cells, tag):
    from inference import dm_test
    d = os.path.join(TASK_DIR, f"{run}-{tag}")
    got, miss = {}, []
    for c in cells:
        p = os.path.join(d, f"{c['id']:04d}.npz")
        if not os.path.exists(p): miss.append(c["id"]); continue
        z = np.load(p, allow_pickle=True)
        stored = json.loads(str(z["cell"]))
        if mhash([{k: v for k, v in stored.items()}]) != mhash([{k: v for k, v in c.items()}]):
            raise SystemExit(f"cell {c['id']} was computed from a different "
                             f"definition; refusing to mix footings")
        got[(c["cadence"], c["design"])] = (z["pred"].astype(np.float64),
                                            float(z["lam"]), int(z["cols"]))
    if miss:
        print(f"MISSING {len(miss)} of {len(cells)} cells: {miss[:12]}")
        print("harvest reports what it has and names what it lacks rather than "
              "quietly averaging a partial array")
    D = os.path.join(ROOT, "results", cells[0]["cache"])
    y = np.load(os.path.join(D, "y.npy")); b = np.load(os.path.join(D, "b.npy"))
    rv = (y ** 2 * b)[W:]
    good = rv > 0
    for k in got: good &= np.isfinite(got[k][0]) & (got[k][0] > 0)
    idx = np.flatnonzero(good)
    Q = {}
    for k, (pr, lam, cols) in got.items():
        r = rv[idx] / pr[idx]
        Q[k] = float(np.mean(r - np.log(r) - 1.0))
    cads = sorted({k[0] for k in got}); dsg = sorted({k[1] for k in got})
    print(f"\n  scored {len(idx):,} bars, {len(got)}/{len(cells)} cells\n")
    print(f"  {'design':>20}" + "".join(f"{'cad ' + str(c):>12}" for c in cads))
    for dn in dsg:
        print(f"  {dn:>20}" + "".join(
            f"{Q[(c, dn)]:>12.5f}" if (c, dn) in Q else f"{'--':>12}" for c in cads))
    orders = {c: sorted([d for d in dsg if (c, d) in Q], key=lambda d: Q[(c, d)])
              for c in cads}
    for c in cads:
        print(f"    cad {c:>5}: {' < '.join(orders[c])}")
    stable = len({tuple(v) for v in orders.values()}) == 1
    print(f"\n  ordering stable across cadence: {stable}")
    res = {"run": run, "manifest_hash": tag, "n": int(len(idx)),
           "cells_present": len(got), "cells_total": len(cells),
           "missing": miss,
           "qlike": {f"{c}|{d}": Q[(c, d)] for (c, d) in Q},
           "ordering": {str(c): orders[c] for c in cads},
           "ordering_stable": bool(stable)}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"{run}_harvest.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"\nwrote {run}_harvest.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="cadence_x_design")
    ap.add_argument("--manifest", action="store_true")
    ap.add_argument("--task-id", type=int)
    ap.add_argument("--harvest", action="store_true")
    a = ap.parse_args()
    cells = manifest(a.run); tag = mhash(cells)
    if a.manifest:
        print(f"run={a.run} hash={tag} cells={len(cells)} "
              f"(SLURM: --array=0-{len(cells)-1})")
        for c in cells: print(f"  {c['id']:>4}  {c}")
        return
    if a.harvest:
        harvest(a.run, cells, tag); return
    if a.task_id is None:
        raise SystemExit("need --task-id, --manifest or --harvest")
    d = os.path.join(TASK_DIR, f"{a.run}-{tag}")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"{a.task_id:04d}.npz")
    if os.path.exists(p):
        print(f"cell {a.task_id} already present, skipping"); return
    run_cell(cells[a.task_id], p)


if __name__ == "__main__":
    main()
