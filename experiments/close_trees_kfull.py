"""The 16:00 LightGBM trained on the last k bars ending 16:00, k = 1 .. 13, with the bar-end time as a column
(close study, 2026-10-04)

The experiment: a linear model trained on the 16:00 bar against a family of LightGBM models, each trained on
the pool of the last k half-hour bars ending 16:00 (k = 1 .. 13) and scored only on the 16:00 forecast.  Every
tree with k > 1 carries the column ``bar_end_minute`` (the bar-end time in minutes since midnight, from the
row's own stamp), built exactly as the k = 13 variant of experiments/close_trees_morebars.py builds it
(close_trees_morebars.source with cols = BARMIN); k = 1 needs no such column.  Three seeds (LightGBM
random_state 42, 43, 44) at every k.  At every refit the use of the timing columns (``bar_end_minute`` and
the calendar column ``hour``) is recorded from the booster: split count, gain share and rank among the
columns the fit saw.  Lowest priority: the same three seeds without the bar column at k = 6, 8 .. 12,
which fills in the earlier no-column ladder of close_trees_morebars.

MACHINERY: close_trees_morebars (row builder ``source``, runner ``run``: claim files, seeds, causality,
leaf minimum scaled to the rows, arm written when it finishes) and close_trees_datasize (``run_tree``,
``make_tree``, ``tree_params``, ``tree_anchors``), imported, not edited.  This script only adds arms, a
recorder of the boosters' importances (it wraps close_trees_datasize.make_tree / window_keep / run_tree;
the fits themselves are unchanged, checked bitwise by the gate), the manifest and the report.

SCORER: as close_trees_morebars -- the forecasts of the one-bar design's forecast rows from row
TREE_START = 130, the master table's research convention (experiments/dense_vs_sparse_1530.py
research_frame / deck_frame / deck_panel / point), 866 trade days, the 15:30 sign(s) straddle.

Stages:
  python experiments/close_trees_kfull.py gate       first 3 refits of the stored k = 13 bar_end_minute arm
                                                     redone with this script (bitwise), and the stored k = 4
                                                     no-column arm re-scored (QLIKE 0.0974, Sharpe mid 1.63)
  python experiments/close_trees_kfull.py run        a worker: fits the arms of QUEUE in order, skipping an
                                                     arm already written, claimed, or whose design is not
                                                     captured yet; run several workers at once (claim files)
  python experiments/close_trees_kfull.py manifest   manifest.csv, bar-column CSVs, scores, SUMMARY.md
Env: TKF_WORK (forecast folder), CLOSE_DESIGN_DIR (designs, via close_trees_datasize).
A file STOP in the forecast folder, written after a worker started, makes it exit after its current arm.

Outputs: results/close_studies_2026-10-03/trees_kfull/ (manifest.csv, CSVs, SUMMARY.md written from the
CSVs); _work/<arm>.npz holds each new arm's forecasts and importances (never committed).
"""

from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ["SLURM_CPUS_PER_TASK"] = "1"  # the tree spec's N_THREADS

import fcntl  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "experiments", REPO / "notebooks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
os.chdir(REPO)

import close_trees_morebars as mb  # noqa: E402  (row builder, runner; imports close_trees_datasize as mb.c)
import dense_vs_sparse_1530 as dvs  # noqa: E402

c = mb.c
OUT = REPO / "results" / "close_studies_2026-10-03" / "trees_kfull"
WORK = Path(os.environ.get("TKF_WORK", str(OUT / "_work")))
MB_OUT, MB_WORK, DS_WORK = mb.OUT, mb.WORK, mb.DS_WORK
LINEAR_REF = REPO / "results" / "close_studies_2026-10-03" / "exog_penalty" / "_work" / "runs" / "ridge_bb0.npz"
KS = tuple(range(1, 14))
SEEDS = (42, 43, 44)
SPEC_SEED = 42  # the tree spec's SEED (asserted)
BAR = mb.BARMIN  # "bar_end_minute"
TIMING = (BAR, "hour")  # the columns whose use is recorded
NO_COL_KS = (6, 8, 9, 10, 11, 12)  # the no-column ladder's missing k (lowest priority)
GATE_ANCHORS = mb.GATE_ANCHORS  # 3
GATE_ARM_STORED = MB_WORK / "lgbm_bars13_r26000_barmin.npz"
GATE_K4 = dict(path=MB_WORK / "lgbm_bars4_r8000.npz", qlike=0.0974, sharpe_mid=1.63)  # stored, trees_morebars/SUMMARY.md
REPRO_TOL = mb.REPRO_TOL
POLL_SEC = 30  # a worker with nothing to fit waits this long for a design capture


def rows_of(k: int) -> int:
    return k * mb.SESSIONS  # the design's window: 2000 sessions x k bars (asserted against the design)


def arm_name(k: int, seed: int, col: bool) -> str:
    return f"lgbm_bars{k}_r{rows_of(k)}" + ("_barmin" if col else "") + f"_seed{seed}"


# ============================================================================ arms
# spec keys: close_trees_morebars' (model, bars, rows, design, rows_from, cols, reuse, group, seed) + family
ARMS: dict[str, dict] = {}


def _arm(k: int, seed: int, col: bool, family: str) -> str:
    name = arm_name(k, seed, col)
    ARMS[name] = dict(
        model="lgbm", bars=k, rows=rows_of(k), design=mb.segment_of(k), rows_from="all", cols=BAR if col else "", reuse="",
        group=f"kfull {family}", seed=seed, family=family,
    )
    return name


for _k in KS:
    for _s in SEEDS:
        if _k > 1:
            _arm(_k, _s, True, "main")
        elif _s == 44:  # k = 1 at seeds 42 / 43 are stored (data-size lgbm_bar1600_w2000, morebars seed replicate)
            _arm(1, _s, False, "main")
for _k in NO_COL_KS:
    for _s in SEEDS:
        _arm(_k, _s, False, "no_bar_column")

# fitting order: seeds 42 and 43 at every k before seed 44, large k first; then the no-column arms
QUEUE = (
    [arm_name(k, s, True) for k in range(13, 1, -1) for s in (42, 43)]
    + [arm_name(k, 44, True) for k in range(13, 1, -1)]
    + [arm_name(1, 44, False)]
    + [arm_name(k, s, False) for s in SEEDS for k in sorted(NO_COL_KS, reverse=True)]
)
assert sorted(QUEUE) == sorted(ARMS)


# ============================================================================ importance recorder
# close_trees_datasize.run_tree calls window_keep once and make_tree / fit once at each refit, in that
# order; the wrappers below record the kept columns and the booster's split / gain importance of each fit
# and do nothing else (the model, its rows and its forecasts are untouched).
_KEEP: list[np.ndarray] = []
_IMP: list[tuple[np.ndarray, np.ndarray]] = []
_BASE = dict(keep=c.window_keep, make=c.make_tree, run_tree=c.run_tree)


def _keep_rec(Xw: np.ndarray) -> np.ndarray:
    k = _BASE["keep"](Xw)
    _KEEP.append(np.asarray(k))
    return k


def _make_rec(model: str, n_rows: int):
    assert model == "lgbm", model
    m = _BASE["make"](model, n_rows)
    fit = m.fit

    def fit_rec(X, y, *a, **kw):
        out = fit(X, y, *a, **kw)
        b = m.booster_
        _IMP.append((b.feature_importance(importance_type="split").astype(np.int64), b.feature_importance(importance_type="gain").astype(np.float64)))
        return out

    m.fit = fit_rec
    return m


def _run_tree_rec(model: str, src: dict, win, n_fc: int, anchors: list[int]) -> dict:
    _KEEP.clear()
    _IMP.clear()
    out = _BASE["run_tree"](model, src, win, n_fc, anchors)
    p = int(src["X"].shape[1])
    assert len(_KEEP) == len(_IMP) == len(anchors), (len(_KEEP), len(_IMP), len(anchors))
    split = np.zeros((len(anchors), p), np.int64)
    gain = np.zeros((len(anchors), p))
    kept = np.zeros((len(anchors), p), bool)
    for i, (k, (s, g)) in enumerate(zip(_KEEP, _IMP)):
        assert len(k) == len(s) == len(g) == int(out["kept_n"][i])
        split[i, k], gain[i, k], kept[i, k] = s, g, True
    out.update(imp_split=split, imp_gain=gain, imp_kept=kept)
    return out


def install_recorder() -> None:
    c.window_keep, c.make_tree, c.run_tree = _keep_rec, _make_rec, _run_tree_rec


def columns_of(spec_cols: str) -> list[str]:
    return c.sources()["names"] + ([BAR] if spec_cols == BAR else [])


# ============================================================================ checks
def check() -> None:
    mb.check()
    ns = dvs.spec_ns("tree")
    assert ns["SEED"] == SPEC_SEED
    import lightgbm

    assert lightgbm.__version__ == "4.7.0", lightgbm.__version__  # the stored arms' version


def design_ready(seg: str) -> bool:
    """The design file exists and its capture finished (the capture log's last line is written after the save)."""
    f = c.DESIGN / f"design_{seg}_{c.BUCKET}.npz"
    log = REPO / "results" / "close_design" / f"capture_{seg}.log"
    return f.is_file() and log.is_file() and f"captured {seg} {c.BUCKET}" in log.read_text(errors="replace")


def score(preds: dict[str, np.ndarray]) -> pd.DataFrame:
    """QLIKE / Sharpe of each forecast on the 866 trade days (close_trees_morebars.analyze's scorer)."""
    S = c.sources()
    sub = np.arange(c.TREE_START, S["n_fc"])
    dz = {k: v[sub] for k, v in S["dz"].items()}
    names = list(preds)
    frames = [dvs.research_frame(dz, preds[a][sub]) for a in names]
    P = dvs.deck_panel(frames, dvs.deck_frame())
    pt = dvs.point(P)
    n_days = int(P["ql"].shape[0])
    assert n_days == 866, n_days
    return pd.DataFrame(dict(arm=names, qlike=pt["ql"], sharpe_mid=pt["sh"], sharpe_crossed=pt["shx"], pct_buy=100.0 * P["buy"], n_days=n_days))


# ============================================================================ gate
def gate() -> None:
    check()
    install_recorder()
    S = c.sources()
    n_fc = S["n_fc"]
    OUT.mkdir(parents=True, exist_ok=True)
    G = []
    # 1. the stored k = 13 bar_end_minute arm (random_state 42), first GATE_ANCHORS refits, with this script's arm
    arm = arm_name(13, 42, True)
    spec = ARMS[arm]
    stored = np.load(GATE_ARM_STORED, allow_pickle=False)
    sm = json.loads(str(stored["meta"]))
    for key in ("bars", "rows", "design", "rows_from", "cols"):
        G.append(dict(gate=f"stored {GATE_ARM_STORED.name}: configuration {key} = this script's {arm}", value=float(sm[key] == spec[key]), reference=1.0, passed=bool(sm[key] == spec[key]), detail=f"{sm[key]!r} vs {spec[key]!r}"))
    G.append(dict(gate=f"stored {GATE_ARM_STORED.name}: params = the spec rule at {spec['rows']} rows", value=float(sm["params"] == c.tree_params("lgbm", spec["rows"])), reference=1.0, passed=bool(sm["params"] == c.tree_params("lgbm", spec["rows"])), detail=f"leaf minimum {sm['params']['min_child_samples']}"))
    G.append(dict(gate=f"stored {GATE_ARM_STORED.name}: random_state = the spec's {SPEC_SEED} (no seed field = the spec's)", value=float(sm.get("seed", 0) in (0, SPEC_SEED)), reference=1.0, passed=bool(sm.get("seed", 0) in (0, SPEC_SEED)), detail=str(sm.get("seed", "absent"))))
    src, _ = mb.source(spec)
    assert int(src["X"].shape[1]) == len(columns_of(BAR)) == sm["n_columns"]
    assert int(mb.design(spec["design"])["W"]) == spec["rows"]
    anchors = c.tree_anchors(n_fc)[:GATE_ANCHORS]
    make = c.make_tree

    def seeded(m: str, n_rows: int, _seed: int = spec["seed"]):  # as close_trees_morebars.run builds a seeded arm
        return make(m, n_rows).set_params(random_state=_seed)

    c.make_tree = seeded
    t0 = time.time()
    try:
        out = c.run_tree("lgbm", src, spec["rows"], n_fc, anchors)
    finally:
        c.make_tree = make
    ref = stored["pred"]
    m = np.isfinite(out["pred"])
    d = float(np.max(np.abs(out["pred"][m] - ref[m])))
    bit = bool(np.array_equal(out["pred"][m], ref[m]))
    G.append(dict(gate=f"{arm}: first {GATE_ANCHORS} refits redone here vs stored {GATE_ARM_STORED.name} ({int(m.sum())} forecasts), max |difference|", value=d, reference=0.0, passed=bit, detail=f"bitwise {bit}, {time.time() - t0:.0f}s"))
    j = len(columns_of(BAR)) - 1
    G.append(dict(gate=f"{arm}: importances recorded at each of the {GATE_ANCHORS} refits, bar column kept by the window mask", value=float(out["imp_kept"][:, j].all()), reference=1.0, passed=bool(out["imp_kept"][:, j].all() and out["imp_split"].shape == (GATE_ANCHORS, j + 1)), detail="bar_end_minute splits " + " / ".join(str(int(v)) for v in out["imp_split"][:, j]) + ", gain share " + " / ".join(f"{v:.4f}" for v in out["imp_gain"][:, j] / out["imp_gain"].sum(axis=1))))
    # 2. the stored k = 4 no-column arm (random_state 42) re-scored
    z = np.load(GATE_K4["path"], allow_pickle=False)
    sc = score({"k4": z["pred"]}).iloc[0]
    mba = pd.read_csv(MB_OUT / "arms.csv").set_index("arm").loc["lgbm_bars4_r8000"]
    G.append(dict(gate="lgbm_bars4_r8000 re-scored here: QLIKE = the stored 0.0974 (rounded) and = trees_morebars/arms.csv", value=float(sc["qlike"]), reference=float(mba["qlike"]), passed=bool(round(sc["qlike"], 4) == GATE_K4["qlike"] and abs(sc["qlike"] - mba["qlike"]) < REPRO_TOL), detail=f"{sc['qlike']:.6f}, {int(sc['n_days'])} trade days"))
    G.append(dict(gate="lgbm_bars4_r8000 re-scored here: Sharpe mid = the stored 1.63 (rounded) and = trees_morebars/arms.csv", value=float(sc["sharpe_mid"]), reference=float(mba["sharpe_mid"]), passed=bool(round(sc["sharpe_mid"], 2) == GATE_K4["sharpe_mid"] and abs(sc["sharpe_mid"] - mba["sharpe_mid"]) < REPRO_TOL), detail=f"{sc['sharpe_mid']:.4f}"))
    Gd = pd.DataFrame(G)
    Gd["abs_diff"] = (Gd["value"] - Gd["reference"]).abs()
    Gd.to_csv(OUT / "gate.csv", index=False)
    print(Gd.to_string(index=False), flush=True)
    if not Gd["passed"].all():
        raise SystemExit("GATE FAILED")
    print("GATE PASSED", flush=True)


# ============================================================================ run (a worker)
def run() -> None:
    check()
    install_recorder()
    mb.WORK = WORK  # close_trees_morebars.run writes <arm>.npz and its claim file here
    WORK.mkdir(parents=True, exist_ok=True)
    for a in QUEUE:
        mb.ARMS[a] = ARMS[a]
    pending = list(QUEUE)
    t_start = time.time()
    stop = WORK / "STOP"
    while pending:
        if stop.exists() and stop.stat().st_mtime > t_start:  # a STOP written after this worker started
            print("STOP file found, worker exits", flush=True)
            return
        fitted = False
        for a in list(pending):
            f, claim = WORK / f"{a}.npz", WORK / f"{a}.claim"
            if f.is_file() or claim.exists():
                pending.remove(a)
                continue
            if not design_ready(ARMS[a]["design"]):
                continue
            # the training window is the design's own W = 2000 sessions x k bars (close_trees_morebars.source
            # asserts W = 2000 x the median bars a session; this pins the median at k)
            w = int(np.load(c.DESIGN / f"design_{ARMS[a]['design']}_{c.BUCKET}.npz", allow_pickle=False)["W"])
            assert w == ARMS[a]["rows"], (a, w)
            mb._DES.clear()  # one design in memory at a time
            mb.run([a])
            pending.remove(a)
            if f.is_file():
                try:  # a reporting error never stops the fitting
                    with lock():
                        write_manifest()
                except Exception:  # noqa: BLE001
                    import traceback

                    traceback.print_exc()
            fitted = True
            break  # back to the top of the queue: the order is the priority
        if not fitted and pending:
            print(f"waiting for designs: {sorted({ARMS[a]['design'] for a in pending})}", flush=True)
            time.sleep(POLL_SEC)
    print("worker: queue empty", flush=True)


class lock:
    """An exclusive lock on the report folder while the manifest and the bar-column CSVs are rewritten."""

    def __enter__(self):
        OUT.mkdir(parents=True, exist_ok=True)
        WORK.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(WORK / "manifest.lock", os.O_CREAT | os.O_RDWR)
        fcntl.flock(self.fd, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        fcntl.flock(self.fd, fcntl.LOCK_UN)
        os.close(self.fd)


def _write_csv(df: pd.DataFrame, f: Path) -> None:
    tmp = f.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, f)


# ============================================================================ manifest
def entries() -> list[dict]:
    """Every forecast file of the experiment, stored and new, with its (spec, k, seed)."""
    E = []

    def add(arm, path, family, model, k, seed, col, status, primary=True):
        E.append(dict(arm=arm, path=path, family=family, model=model, k=k, seed=seed, bar_column=col, status=status, primary=primary))

    # main spec: k = 1 without a column, k >= 2 with bar_end_minute
    add("lgbm_bars1_r2000", DS_WORK / "lgbm_bar1600_w2000.npz", "main", "lgbm", 1, 42, False, "stored (trees_datasize lgbm_bar1600_w2000)")
    add("lgbm_bars1_r2000_seed43", MB_WORK / "lgbm_bars1_r2000_seed43.npz", "main", "lgbm", 1, 43, False, "stored (trees_morebars)")
    for k in KS:
        for s in SEEDS:
            if k > 1 or s == 44:
                add(arm_name(k, s, k > 1), WORK / f"{arm_name(k, s, k > 1)}.npz", "main", "lgbm", k, s, k > 1, "new")
    rerun = WORK / f"{arm_name(13, 42, True)}.npz"
    rerun_done = rerun.is_file() and not (WORK / f"{arm_name(13, 42, True)}.claim").exists()
    add("lgbm_bars13_r26000_barmin", GATE_ARM_STORED, "main", "lgbm", 13, 42, True, "stored (trees_morebars); refitted here as " + arm_name(13, 42, True), primary=not rerun_done)
    # the earlier variant without the bar column (k = 1 is the main spec's)
    add("lgbm_bars2_r4000", DS_WORK / "lgbm_pool_w4000.npz", "no_bar_column", "lgbm", 2, 42, False, "stored (trees_datasize lgbm_pool_w4000)")
    for k in (3, 4, 5, 7, 13):
        add(f"lgbm_bars{k}_r{rows_of(k)}", MB_WORK / f"lgbm_bars{k}_r{rows_of(k)}.npz", "no_bar_column", "lgbm", k, 42, False, "stored (trees_morebars)")
    for k in (2, 3, 4, 13):
        add(f"lgbm_bars{k}_r{rows_of(k)}_seed43", MB_WORK / f"lgbm_bars{k}_r{rows_of(k)}_seed43.npz", "no_bar_column", "lgbm", k, 43, False, "stored (trees_morebars)")
    for k in NO_COL_KS:
        for s in SEEDS:
            add(arm_name(k, s, False), WORK / f"{arm_name(k, s, False)}.npz", "no_bar_column", "lgbm", k, s, False, "new")
    # linear models trained on the 16:00 bar (2000 sessions)
    add("ridge_bars1_r2000", DS_WORK / "ridge_bar1600_w2000.npz", "linear_1600", "ridge", 1, None, False, "stored (trees_datasize ridge_bar1600_w2000)")
    add("lasso_bars1_r2000", DS_WORK / "lasso_bar1600_w2000.npz", "linear_1600", "lasso", 1, None, False, "stored (trees_datasize lasso_bar1600_w2000)")
    add("ridge_bb0", LINEAR_REF, "linear_1600", "ridge", 1, None, False, "stored (exog_penalty ridge_bb0: HAR + calendar backbone unpenalized)")
    return E


def config_check(e: dict, z: dict, meta: dict) -> list[str]:
    """Differences between a stored LightGBM arm and this experiment's configuration (empty = matches)."""
    k, rows = e["k"], rows_of(e["k"])
    n_fc = c.sources()["n_fc"]
    bad = []
    seg = meta.get("design", {"bar1600": "bar1600", "pool": "last30"}.get(meta.get("source", ""), "?"))
    win = int(meta.get("rows", meta.get("window", -1)))
    checks = [
        ("bucket", meta["bucket"] == c.BUCKET),
        ("model", meta["model"] == "lgbm"),
        ("design", seg == mb.segment_of(k)),
        ("rows", win == rows),
        ("rows from every bar of the design", meta.get("rows_from", "all") == "all"),
        ("bar column", meta.get("cols", "") == (BAR if e["bar_column"] else "")),
        ("params = the spec rule at these rows", meta["params"] == c.tree_params("lgbm", rows)),
        ("random_state", int(meta.get("seed", 0) or SPEC_SEED) == e["seed"]),
        ("refit every 10 from forecast row 130", meta["refit_every"] == c.REFIT_EVERY and meta["tree_start"] == c.TREE_START),
        ("refit anchors", z["anchors"].tolist() == c.tree_anchors(n_fc)),
        ("training rows at every refit", bool((z["n_train"] == rows).all())),
        ("whole arm (no chunk / slice)", meta.get("chunk", "") == "" and int(meta.get("max_anchors", 0)) == 0),
        ("LightGBM 4.7.0", meta["versions"]["lightgbm"] == "4.7.0"),
        ("forecasts from row 130 on", bool(np.isfinite(z["pred"][c.TREE_START:]).all() and np.isnan(z["pred"][: c.TREE_START]).all())),
    ]
    if "n_columns" in meta:
        checks.append(("columns", meta["n_columns"] == len(columns_of(BAR if e["bar_column"] else ""))))
    for name, ok in checks:
        if not ok:
            bad.append(name)
    return bad


INT_COLS = ("k", "rows", "sessions", "seed", "n_columns", "leaf_min", "num_leaves", "n_estimators", "refit_every", "first_forecast_row", "n_refits", "num_threads", "n_forecasts")


def _rel(f: Path) -> str:
    return str(f.relative_to(REPO)) if f.is_relative_to(REPO) else str(f)


def _load(path: Path) -> tuple[dict, dict]:
    z = np.load(path, allow_pickle=False)
    out = {k: z[k] for k in z.files}
    meta = json.loads(str(out.pop("meta"))) if "meta" in out else {}
    return out, meta


def write_manifest() -> tuple[pd.DataFrame, pd.DataFrame]:
    """manifest.csv and the bar-column CSVs from the files present now (cheap: no scoring)."""
    S = c.sources()
    rows, use = [], []
    fc_dates = pd.DatetimeIndex(pd.to_datetime(S["dz"]["date"]))
    for e in entries():
        f = Path(e["path"])
        # close_trees_morebars.run writes <arm>.npz in place and removes <arm>.claim afterwards: a file with a
        # claim may be half written
        claimed = (f.parent / f"{e['arm']}.claim").exists()
        r = dict(
            arm=e["arm"], family=e["family"], model=e["model"], k=e["k"], rows=rows_of(e["k"]) if e["model"] == "lgbm" else mb.SESSIONS, sessions=mb.SESSIONS,
            bar_column="yes" if e["bar_column"] else "no", seed=e["seed"], primary="yes" if e["primary"] else "no",
            status=e["status"], path=_rel(f), exists="yes" if f.is_file() and not claimed else "no",
        )
        if e["model"] == "lgbm":
            p = c.tree_params("lgbm", rows_of(e["k"]))
            r.update(
                design=f"design_{mb.segment_of(e['k'])}_{c.BUCKET}", n_columns=len(columns_of(BAR if e["bar_column"] else "")), leaf_min=p["min_child_samples"],
                num_leaves=p["num_leaves"], n_estimators=p["n_estimators"], learning_rate=p["learning_rate"], feature_fraction=p["feature_fraction"], bagging_fraction=p["bagging_fraction"],
                refit_every=c.REFIT_EVERY, first_forecast_row=c.TREE_START, n_refits=len(c.tree_anchors(S["n_fc"])), window_mask="yes", num_threads=1, lightgbm="4.7.0",
            )
        else:
            r.update(design=f"design_bar1600_{c.BUCKET}", refit_every=1, first_forecast_row=0)
        if f.is_file() and not claimed:
            z, meta = _load(f)
            r.update(n_forecasts=int(np.isfinite(z["pred"]).sum()), cpu_min=round(float(meta.get("cpu_sec", z.get("cpu_sec", np.nan))) / 60.0, 2), written=time.strftime("%Y-%m-%d %H:%M", time.localtime(f.stat().st_mtime)))
            if e["model"] == "lgbm":
                bad = config_check(e, z, meta)
                r["config_check"] = "pass" if not bad else "FAIL: " + "; ".join(bad)
                r["importances"] = "yes" if "imp_gain" in z else "no"
                if "imp_gain" in z:
                    use.extend(bar_use(e, z, meta, fc_dates))
        rows.append(r)
    M = pd.DataFrame(rows)
    for col in INT_COLS:  # integers stay integers where a row has no value (linear rows, files not written yet)
        M[col] = pd.array([None if (v is None or v == "" or (isinstance(v, float) and np.isnan(v))) else int(v) for v in M[col]], dtype="Int64")
    U = pd.DataFrame(use)
    _write_csv(M, OUT / "manifest.csv")
    if len(U):
        _write_csv(U, OUT / "bar_column_refits.csv")
        _write_csv(bar_use_arms(U), OUT / "bar_column_arms.csv")
    return M, U


def bar_use(e: dict, z: dict, meta: dict, fc_dates: pd.DatetimeIndex) -> list[dict]:
    """At each refit: split count, split share, gain share and rank (1 = most gain / most splits) of each
    timing column among the columns the fit saw."""
    cols = columns_of(meta.get("cols", ""))
    split, gain, kept = z["imp_split"], z["imp_gain"], z["imp_kept"]
    assert split.shape[1] == len(cols)
    out = []
    for name in TIMING:
        if name not in cols:
            continue
        j = cols.index(name)
        for i, a in enumerate(z["anchors"]):
            kk = kept[i]
            g, s = gain[i, kk], split[i, kk]
            out.append(
                dict(
                    arm=e["arm"], family=e["family"], k=e["k"], seed=e["seed"], column=name, refit=i, anchor_row=int(a), anchor_date=str(fc_dates[int(a)].date()),
                    kept=bool(kk[j]), n_columns_kept=int(kk.sum()), n_columns_used=int((s > 0).sum()), splits=int(split[i, j]), split_share=float(split[i, j] / s.sum()),
                    gain_share=float(gain[i, j] / g.sum()), rank_gain=int(1 + (g > gain[i, j]).sum()) if kk[j] else np.nan,
                    rank_splits=int(1 + (s > split[i, j]).sum()) if kk[j] else np.nan,
                )
            )
    return out


def bar_use_arms(U: pd.DataFrame) -> pd.DataFrame:
    g = U.groupby(["arm", "family", "k", "seed", "column"], sort=False)
    A = g.agg(
        refits=("refit", "size"), pct_refits_kept=("kept", lambda v: 100.0 * v.mean()), pct_refits_split=("splits", lambda v: 100.0 * (v > 0).mean()),
        splits_mean=("splits", "mean"), split_share_mean=("split_share", "mean"), gain_share_mean=("gain_share", "mean"), gain_share_min=("gain_share", "min"),
        gain_share_max=("gain_share", "max"), rank_gain_median=("rank_gain", "median"), rank_gain_best=("rank_gain", "min"), rank_gain_worst=("rank_gain", "max"),
        rank_splits_median=("rank_splits", "median"), n_columns_kept_median=("n_columns_kept", "median"), n_columns_used_median=("n_columns_used", "median"),
    ).reset_index()
    return A


# ============================================================================ report
def report() -> None:  # noqa: C901 - one linear report
    with lock():
        M, U = write_manifest()
    have = M[(M["exists"] == "yes")]
    preds = {}
    for _, r in have.iterrows():
        z, _ = _load(REPO / r["path"])
        preds[r["arm"]] = z["pred"]
    Sc = score(preds)
    A = M.merge(Sc, on="arm", how="left")
    _write_csv(A, OUT / "arms.csv")
    # gates added here: the refitted k = 13 bar_end_minute arm against the stored one, whole arm
    G = pd.read_csv(OUT / "gate.csv") if (OUT / "gate.csv").is_file() else pd.DataFrame()
    rr = arm_name(13, 42, True)
    if rr in preds:
        ref = preds["lgbm_bars13_r26000_barmin"]
        m = np.isfinite(ref)
        d = float(np.max(np.abs(preds[rr][m] - ref[m])))
        row = dict(gate=f"{rr}: whole arm refitted here vs stored lgbm_bars13_r26000_barmin ({int(m.sum())} forecasts), max |difference|", value=d, reference=0.0, passed=bool(np.array_equal(preds[rr], ref, equal_nan=True)), detail="", abs_diff=d)
        G = pd.concat([G[G["gate"] != row["gate"]], pd.DataFrame([row])], ignore_index=True) if len(G) else pd.DataFrame([row])
        _write_csv(G, OUT / "gate.csv")
    # seeds: mean +- sd over the seeds at each k, main spec and the no-column variant (k = 1 shared)
    P = A[(A["primary"] == "yes") & A["qlike"].notna() & (A["model"] == "lgbm")].copy()
    P["seed"] = P["seed"].astype(int)
    sd_rows = []
    for fam in ("main", "no_bar_column"):
        for k in KS:
            h = P[(P["k"] == k) & ((P["family"] == fam) | ((k == 1) & (P["family"] == "main")))]
            if not len(h):
                continue
            h = h.sort_values("seed")
            sd_rows.append(
                dict(
                    family=fam, k=k, rows=rows_of(k), n_seeds=len(h), seeds=" ".join(str(s) for s in h["seed"]),
                    qlike_mean=h["qlike"].mean(), qlike_sd=h["qlike"].std(ddof=1) if len(h) > 1 else np.nan, qlike_min=h["qlike"].min(), qlike_max=h["qlike"].max(),
                    sharpe_mid_mean=h["sharpe_mid"].mean(), sharpe_mid_sd=h["sharpe_mid"].std(ddof=1) if len(h) > 1 else np.nan,
                    sharpe_crossed_mean=h["sharpe_crossed"].mean(),
                    **{f"qlike_seed{s}": float(h.loc[h["seed"] == s, "qlike"].iloc[0]) if (h["seed"] == s).any() else np.nan for s in SEEDS},
                    **{f"sharpe_mid_seed{s}": float(h.loc[h["seed"] == s, "sharpe_mid"].iloc[0]) if (h["seed"] == s).any() else np.nan for s in SEEDS},
                )
            )
    Sd = pd.DataFrame(sd_rows)
    _write_csv(Sd, OUT / "seeds.csv")
    # bar-column use for each k (main spec): over the seeds and refits
    Bk = pd.DataFrame()
    if len(U):
        Bk = (
            U.groupby(["family", "k", "column"])
            .agg(arms=("arm", "nunique"), refits=("refit", "size"), pct_refits_split=("splits", lambda v: 100.0 * (v > 0).mean()), splits_mean=("splits", "mean"),
                 split_share_mean=("split_share", "mean"), gain_share_mean=("gain_share", "mean"), gain_share_min=("gain_share", "min"), gain_share_max=("gain_share", "max"),
                 rank_gain_median=("rank_gain", "median"), rank_gain_best=("rank_gain", "min"), rank_gain_worst=("rank_gain", "max"), n_columns_kept_median=("n_columns_kept", "median"))
            .reset_index()
        )
        _write_csv(Bk, OUT / "bar_column_by_k.csv")
    write_summary(A, Sd, Bk, G)
    print(Sd[["family", "k", "n_seeds", "qlike_mean", "qlike_sd", "sharpe_mid_mean", "sharpe_mid_sd"]].to_string(index=False))


def _f(v, nd: int = 4) -> str:
    return "" if pd.isna(v) else f"{v:.{nd}f}"


def write_summary(A: pd.DataFrame, Sd: pd.DataFrame, Bk: pd.DataFrame, G: pd.DataFrame) -> None:  # noqa: C901
    L = ["# LightGBM on the last k bars ending 16:00 with the bar-end time as a column, k = 1 .. 13, three seeds (close study, 2026-10-04)", ""]
    L.append(
        "The experiment: a linear model trained on the 16:00 bar against LightGBM trained on the pool of the last k half-hour bars ending 16:00 "
        "(k = 1 .. 13), every tree with k > 1 carrying the column `bar_end_minute` (bar-end time, minutes since midnight), all scored only on the "
        "16:00 forecast. This folder holds the forecasts' manifest for the statistical tests, the score of each arm and the trees' use of the bar column. "
        "Written by `experiments/close_trees_kfull.py manifest` from the CSVs in this folder; every number below is in them."
    )
    L.append("")
    L.append("## Setup")
    L.append(
        "- Rows: the design `lastbars<k>` of `experiments/capture_design_close.py` (all features; k = 1 is `bar1600`, k = 2 is `last30`), the window "
        "2000 sessions x k bars; the column `bar_end_minute` is added exactly as `close_trees_morebars.source` adds it for its k = 13 arm. "
        "k = 1 has no such column (every row ends 16:00)."
    )
    L.append(
        f"- Trees: `close_trees_morebars.run` unchanged -- the tree spec's shipped LightGBM config, leaf minimum `min_child_samples` = round(98 x rows / 24000), "
        f"refit every {c.REFIT_EVERY} sessions from forecast row {c.TREE_START} (2019-01, {len(c.tree_anchors(c.sources()['n_fc']))} refits), window mask on, single-threaded, "
        "LightGBM 4.7.0; random_state 42 (the spec's), 43, 44, nothing else changed between seeds. The model in force for the 16:00 row of day d was fitted on "
        "the design rows strictly before that row (the earlier bars of day d, realized by 15:30, may enter it)."
    )
    L.append(
        "- Bar-column use: at every refit the booster's split count and gain of each column it saw (`feature_importance`), recorded for `bar_end_minute` and the "
        "calendar column `hour` (which also separates the 16:00 row from the earlier bars, since the bars ending 15:00 and 15:30 share hour 15). "
        "Share = the column's splits (gain) over all splits (gain) of that fit; rank 1 = the column with the most gain (most splits) among the columns the fit saw."
    )
    L.append(
        "- Scorer: the master table's research convention (16:00 recalibration over the previous 250 sessions, the 15:30 sign(s) straddle trade), "
        "866 trade days 2020-01-03 .. 2024-04-30, forecast rows from row 130; point estimates (no bootstrap, no test here: the tests are built separately on these forecasts)."
    )
    L.append(
        "- Forecast files: `pred` holds 1469 values on the one-bar design's forecast rows (2018-06-25 .. 2024-04-30, the dates of `design_bar1600_all_features` from row W = 2000; "
        "`close_trees_datasize.sources()['dz']`), NaN before row 130 for the trees. New tree files also hold `imp_split`, `imp_gain`, `imp_kept` (refits x columns) and `anchors`."
    )
    L.append("")
    L.append("## Gates")
    for _, g in G.iterrows():
        det = f" ({g['detail']})" if isinstance(g.get("detail", ""), str) and g.get("detail", "") else ""
        L.append(f"- {g['gate']}: {g['value']:.6g} vs {g['reference']:.6g}{det} {'PASS' if g['passed'] else 'FAIL'}")
    fails = A[(A["exists"] == "yes") & A["config_check"].fillna("").str.startswith("FAIL")]
    L.append(
        "- Configuration of every LightGBM file present (design, rows, bar column, params at these rows, random_state, refit anchors, training rows, LightGBM version, forecast rows): "
        + (f"{(A['config_check'] == 'pass').sum()} pass, none fail." if not len(fails) else "FAIL for " + ", ".join(f"`{a}`" for a in fails["arm"]))
    )
    L.append("")
    L.append("## QLIKE and Sharpe for each k: seed mean +- sd")
    L.append("")
    L.append("Main spec = k = 1 without a column, k >= 2 with `bar_end_minute`. sd over the seeds present (ddof 1).")
    L.append("")
    L.append("| k | rows | main: seeds | main: QLIKE mean +- sd | main: QLIKE 42 / 43 / 44 | main: Sharpe mid mean +- sd | no column: seeds | no column: QLIKE mean +- sd | no column: Sharpe mid mean +- sd |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for k in KS:
        mn = Sd[(Sd["family"] == "main") & (Sd["k"] == k)]
        nc = Sd[(Sd["family"] == "no_bar_column") & (Sd["k"] == k)]

        def ms(h, a, nd):
            if not len(h):
                return ""
            r = h.iloc[0]
            return f"{r[a + '_mean']:.{nd}f}" + ("" if pd.isna(r[a + "_sd"]) else f" +- {r[a + '_sd']:.{nd}f}")

        seeds_main = mn.iloc[0]["seeds"] if len(mn) else ""
        each = " / ".join(_f(mn.iloc[0][f"qlike_seed{s}"]) or "-" for s in SEEDS) if len(mn) else ""
        L.append(
            f"| {k} | {rows_of(k)} | {seeds_main} | {ms(mn, 'qlike', 4)} | {each} | {ms(mn, 'sharpe_mid', 2)} | {nc.iloc[0]['seeds'] if len(nc) else ''} | {ms(nc, 'qlike', 4)} | {ms(nc, 'sharpe_mid', 2)} |"
        )
    L.append("")
    lin = A[(A["family"] == "linear_1600") & A["qlike"].notna()]
    if len(lin):
        L.append("Linear models trained on the 16:00 bar (2000 sessions, same scorer and rows): " + "; ".join(f"`{r['arm']}` ({r['model']}) QLIKE {r['qlike']:.4f}, Sharpe mid {r['sharpe_mid']:.2f}" for _, r in lin.iterrows()) + ".")
        L.append("")
    if len(Bk):
        L.append("## Bar-column use for each k (over the seeds and refits present)")
        L.append("")
        L.append("| family | k | column | arms | refits | refits with a split % | splits (mean) | split share (mean) | gain share mean (min .. max) | rank by gain: median (best .. worst) | columns seen (median) |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for _, r in Bk.sort_values(["family", "column", "k"]).iterrows():
            L.append(
                f"| {r['family']} | {r['k']} | `{r['column']}` | {r['arms']} | {r['refits']} | {r['pct_refits_split']:.1f} | {r['splits_mean']:.1f} | {r['split_share_mean']:.4f} | "
                f"{r['gain_share_mean']:.4f} ({r['gain_share_min']:.4f} .. {r['gain_share_max']:.4f}) | {r['rank_gain_median']:.0f} ({r['rank_gain_best']:.0f} .. {r['rank_gain_worst']:.0f}) | {r['n_columns_kept_median']:.0f} |"
            )
        L.append("")
        L.append("Each arm and each refit: `bar_column_refits.csv`; each arm: `bar_column_arms.csv`.")
        L.append("")
    L.append("## Each arm")
    L.append("")
    L.append("| arm | family | k | bar column | seed | status | QLIKE | Sharpe mid | Sharpe crossed | buys % | CPU min |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in A[A["exists"] == "yes"].iterrows():
        L.append(
            f"| `{r['arm']}` | {r['family']} | {r['k']} | {r['bar_column']} | {'' if pd.isna(r['seed']) else int(r['seed'])} | {r['status']}{'' if r['primary'] == 'yes' else ' (not primary)'} | {r['qlike']:.4f} | {r['sharpe_mid']:.2f} | "
            f"{r['sharpe_crossed']:.2f} | {r['pct_buy']:.1f} | {_f(r['cpu_min'], 1)} |"
        )
    L.append("")
    miss = A[A["exists"] == "no"]
    L.append("## Not run")
    L.append("- Arms of the plan without a forecast file: " + (", ".join(f"`{a}`" for a in miss["arm"]) if len(miss) else "none") + ".")
    L.append("")
    L.append("## Caveats")
    L.append("- The leaf minimum, the rows and the bars change together along k (the spec's rule); each design has its own rolling scaling (close_trees_morebars SUMMARY).")
    L.append("- Three seeds give a rough sd only; the seed changes the bagging and feature-fraction draws and nothing else.")
    L.append("- `hour` and `bar_end_minute` carry overlapping timing information for k >= 2, so a low share for one of them does not by itself mean the trees ignore timing.")
    L.append("- CPU minutes depend on what else ran on the machine (four single-threaded fits at once here).")
    L.append("")
    L.append("## Files")
    L.append("- `experiments/close_trees_kfull.py` (stages gate / run / manifest; machinery from `experiments/close_trees_morebars.py` and `experiments/close_trees_datasize.py`)")
    L.append("- `manifest.csv` (every forecast file: family, k, rows, bar column, seed, primary, path, configuration, check), `arms.csv` (manifest + scores), `seeds.csv`, "
             "`bar_column_refits.csv`, `bar_column_arms.csv`, `bar_column_by_k.csv`, `gate.csv`; new forecasts in `_work/<arm>.npz` (not committed)")
    new = A[(A["status"] == "new") & (A["exists"] == "yes")]
    L.append(f"- CPU: {new['cpu_min'].sum():.0f} min over the {len(new)} arms fitted here (single-threaded, at most four at once).")
    (OUT / "SUMMARY.md").write_text("\n".join(L) + "\n", encoding="utf-8")


# ============================================================================ main
if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("gate", "run", "manifest"):
        raise SystemExit(__doc__)
    {"gate": gate, "run": run, "manifest": report}[sys.argv[1]]()
