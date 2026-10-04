"""Train the 16:00-bar trees on more bars of each session than the last two  (close study, 2026-10-04)

The user's request: "run ablations on training on more bars than the last 2".  The data-size study
(experiments/close_trees_datasize.py, results/close_studies_2026-10-03/trees_datasize/) found that
LightGBM trained on both last-hour bars (rows ending 15:30 and 16:00, 2000 sessions = 4000 rows) and
scored on the 16:00 forecast only had QLIKE 0.0982 against 0.1007 for the shipped one-bar fit.  This
study extends that ladder: train on the N half-hour bars ending 16:00 (segment lastbars<N> of
experiments/capture_design_close.py; lastbars2 is last30) and score ONLY the 16:00 forecast.

ARMS (all on the all_features design; one model a fit, forecasting the 16:00 rows):
  1. LADDER, 2000 sessions fixed: N = 1, 2 (the data-size study's lgbm_bar1600_w2000 and
     lgbm_pool_w4000, reused after the gates below), 3, 4, 5, 7, 13 bars; the training window is the
     design's own W = 2000 sessions x the median bars a session (6000 .. 26000 rows).
  2. ROWS HELD AT 4000: N = 4 with 1000 sessions, N = 13 with 4000 rows (about 308 sessions), and,
     added here, N = 1 with 4000 sessions (the 16:00 rows of the lastbars13 design, which reach back
     to 2002 -- the one-bar design starts in 2010-07 and cannot hold 4000 sessions).  The design's
     rolling scaling is unchanged (it uses the design's 2000-session window).
  3. XGBoost at the rung where LightGBM has the lowest QLIKE (run on request: ``run xgb_bars<N>_r<rows>``),
     and at N = 13 a DESIGN CHANGE: one added column, the bar-end time in minutes since midnight
     (``bar_end_minute``, from the row's own stamp), since the calendar column ``hour`` is shared by
     the bars ending hh:00 and hh:30.
  4. Linear controls at N = 4 and N = 13: the linear spec's ridge and lasso (RollingTunedLinear), run
     exactly as close_trees_datasize.run_linear drives them (a solve at every row from the first
     forecast row, 15:30 and earlier rows included and not scored; penalty re-chosen every 250 solves).

TREES: close_trees_datasize.run_tree unchanged -- the tree spec's shipped configs (make_model of
specs/causal_tune_trees.py), refit every 10 sessions from forecast row 130 (2019-01), single-threaded,
window mask on (src.models.window_mask.window_keep on each fit's own window), the leaf minimum scaled
to the fit's rows by the spec's rule (LightGBM min_child_samples = max(1, round(98 x rows / 24000)),
XGBoost min_child_weight = 24.86 x rows / 24000).  CAUSALITY: the model in force for the 16:00 row of
day d was fitted on the design rows strictly before that row, so the earlier bars of day d (targets
realized by 15:30, when the 16:00 forecast is issued) may enter it; nothing later does.

SCORER: every arm forecasts the one-bar design's forecast rows (date / true_adj / true_raw of
design_bar1600) and is scored on the 866 trade days by the master table's research convention
(experiments/dense_vs_sparse_1530.py research_frame / deck_frame / deck_panel / point); point
estimates (the block bootstrap is off); DM = Diebold-Mariano on daily QLIKE
(src/evaluation/diebold_mariano.py), HAC t = Newey-West t on the paired daily mid-fill P&L difference.

Stages:
  python experiments/close_trees_morebars.py gate            short-slice refit of the two reused arms
                                                             with this script's row builder (bitwise)
  python experiments/close_trees_morebars.py run [arm ...]   fit arms (default ORDER); an arm already in
                                                             _work/ or claimed by another process is skipped
  python experiments/close_trees_morebars.py analyze         score, CSVs, figure, SUMMARY.md
Env: TMB_WORK (forecast folder), CLOSE_DESIGN_DIR (designs, via close_trees_datasize).

Outputs: results/close_studies_2026-10-03/trees_morebars/ (CSVs, bars_ladder.png, SUMMARY.md written
from the CSVs); _work/<arm>.npz holds each arm's forecasts (never committed).
"""

from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ["SLURM_CPUS_PER_TASK"] = "1"  # the tree spec's N_THREADS

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

import close_trees_datasize as c  # noqa: E402  (machinery: sources, run_tree, run_linear, tree_params)
import dense_vs_sparse_1530 as dvs  # noqa: E402
from src.evaluation.diebold_mariano import dm_test  # noqa: E402

OUT = REPO / "results" / "close_studies_2026-10-03" / "trees_morebars"
WORK = Path(os.environ.get("TMB_WORK", str(OUT / "_work")))
DS_REPORT = c.OUT  # the data-size study's report folder (its arms.csv)
DS_WORK = c.OUT / "_work"  # its forecasts: the reused rungs 1 and 2
SESSIONS = 2000  # the per-bar arms' window in sessions (capture_design_close.TRAIN_WIN)
ROWS_FIXED = 4000  # the rows-held-fixed comparison: the 2-bar arm's rows
LADDER = (1, 2, 3, 4, 5, 7, 13)
GATE_ANCHORS = 3  # refits of the short-slice reproduction (gate stage)
REPRO_TOL = 1e-9
MASTER = REPO / "results" / "close_master_table" / "master_table.csv"
CITED_KEY = "lgbm"  # the paper's pooled 48-bar LightGBM (cited, not rerun)
BARMIN = "bar_end_minute"
TREES = ("lgbm", "xgb")
LINEAR = ("ridge", "lasso")
LABEL = {"lgbm": "LightGBM", "xgb": "XGBoost", "ridge": "ridge", "lasso": "lasso"}


def segment_of(n: int) -> str:
    return "bar1600" if n == 1 else ("last30" if n == 2 else f"lastbars{n}")


# arm -> spec.  design: the captured segment the rows come from; rows_from: "all" = every row of the
# design, "h16" = its 16:00 rows only; cols: "" or BARMIN (the added column); reuse: the data-size
# study's arm of the same configuration (rungs 1 and 2)
ARMS: dict[str, dict] = {}


def _arm(name: str, model: str, bars: int, rows: int, design: str, *, rows_from: str = "all", cols: str = "", reuse: str = "", group: str) -> None:
    ARMS[name] = dict(model=model, bars=bars, rows=rows, design=design, rows_from=rows_from, cols=cols, reuse=reuse, group=group)


for _m in (*TREES, *LINEAR):
    _arm(f"{_m}_bars1_r2000", _m, 1, SESSIONS, "bar1600", reuse=f"{_m}_bar1600_w2000", group="ladder")
    _arm(f"{_m}_bars2_r4000", _m, 2, 2 * SESSIONS, "last30", reuse=f"{_m}_pool_w4000", group="ladder")
    for _n in LADDER[2:]:
        _arm(f"{_m}_bars{_n}_r{_n * SESSIONS}", _m, _n, _n * SESSIONS, segment_of(_n), group="ladder")
_arm("lgbm_bars4_r4000", "lgbm", 4, ROWS_FIXED, "lastbars4", group="rows4000")
_arm("lgbm_bars13_r4000", "lgbm", 13, ROWS_FIXED, "lastbars13", group="rows4000")
_arm("lgbm_bars1_r4000", "lgbm", 1, ROWS_FIXED, "lastbars13", rows_from="h16", group="rows4000")
_arm("lgbm_bars13_r26000_barmin", "lgbm", 13, 13 * SESSIONS, "lastbars13", cols=BARMIN, group="design change")
CTRL = "lgbm_bars1_r2000"  # the 1-bar control (the shipped configuration)
TWO = "lgbm_bars2_r4000"  # the 2-bar arm
ORDER = [
    "lgbm_bars13_r26000",
    "lgbm_bars7_r14000",
    "lgbm_bars5_r10000",
    "lgbm_bars4_r8000",
    "lgbm_bars3_r6000",
    "lgbm_bars13_r26000_barmin",
    "lgbm_bars13_r4000",
    "lgbm_bars4_r4000",
    "lgbm_bars1_r4000",
    "lasso_bars13_r26000",
    "ridge_bars13_r26000",
    "lasso_bars4_r8000",
    "ridge_bars4_r8000",
]


# ============================================================================ rows
_DES: dict = {}


def design(seg: str) -> dict:
    if seg not in _DES:
        z = np.load(c.DESIGN / f"design_{seg}_{c.BUCKET}.npz", allow_pickle=False)
        _DES[seg] = {k: z[k] for k in z.files}
    return _DES[seg]


def source(spec: dict) -> tuple[dict, dict]:
    """(rows for c.run_tree / c.run_linear, design facts).  fc = the series index of each of the
    1469 one-bar forecast rows (2018-06-25 .. 2024-04-30, 16:00)."""
    S = c.sources()
    fc_dates = pd.DatetimeIndex(pd.to_datetime(S["dz"]["date"]))
    z = design(spec["design"])
    assert [str(v) for v in z["names"]] == S["names"]
    d = pd.DatetimeIndex(pd.to_datetime(z["date"]))
    X, y = z["X"], z["y"]
    i16 = np.asarray((d.hour == 16) & (d.minute == 0))
    per_day = pd.Series(1, index=d.normalize()).groupby(level=0).size()
    h = S["names"].index("hour")
    facts = dict(
        segment=spec["design"],
        design_rows=int(len(X)),
        design_W=int(z["W"]),
        design_first=str(d[0]),
        median_bars_a_session=float(per_day.median()),
        share_sessions_median_bars=float((per_day == per_day.median()).mean()),
        # the calendar column `hour` isolates the 16:00 row when its values there never occur elsewhere
        hour_isolates_1600=bool(not np.intersect1d(np.unique(X[i16, h]), np.unique(X[~i16, h])).size) if (~i16).any() else True,
    )
    if spec["rows_from"] == "all":
        # the design's own window is 2000 sessions x the median bars a session
        assert spec["design"] == "bar1600" or int(z["W"]) == SESSIONS * int(per_day.median()), (spec, int(z["W"]))
        if spec["group"] == "ladder":
            assert spec["rows"] == int(z["W"]), (spec, int(z["W"]))
    else:  # the 16:00 rows only
        X, y, d = np.ascontiguousarray(X[i16]), y[i16], d[i16]
    pos = d.get_indexer(fc_dates)
    assert (pos >= 0).all()
    # the same target as the one-bar design on every forecast row
    assert np.array_equal(y[pos], S["dz"]["true_adj"])
    if spec["cols"] == BARMIN:
        X = np.column_stack([X, (d.hour * 60 + d.minute).to_numpy(float)])
    return dict(X=X, y=y, fc=pos, date=d), facts


def sessions_in_windows(src: dict, rows: int, anchors: list[int]) -> np.ndarray:
    """Distinct sessions in the training window ending at each refit anchor's 16:00 row."""
    days = src["date"].normalize()
    out = []
    for j in anchors:
        r = int(src["fc"][j])
        out.append(len(np.unique(days[r - rows : r].asi8)))
    return np.array(out)


# ============================================================================ checks
def check() -> None:
    c.check_spec()
    assert c.REFIT_EVERY == c.SPEC_REFIT_EVERY == 10 and c.MAX_ANCHORS == 0 and not c.CHUNK
    for spec in ARMS.values():
        if spec["model"] == "lgbm" and spec["group"] == "ladder":
            # the spec's leaf rule gives 98 at the pooled bank's 24000 rows
            assert c.tree_params("lgbm", 24000)["min_child_samples"] == 98


def config_matches(arm: str, run: dict) -> list[tuple[str, bool, str]]:
    """The reused data-size arm has this study's configuration."""
    spec = ARMS[arm]
    m = run["meta"]
    n_fc = c.sources()["n_fc"]
    out = [
        ("bucket", m["bucket"] == c.BUCKET, m["bucket"]),
        ("model", m["model"] == spec["model"], m["model"]),
        ("window rows", int(m["window"]) == spec["rows"], str(m["window"])),
        ("rows (source)", m["source"] == ("bar1600" if spec["bars"] == 1 else "pool"), m["source"]),
        # arms written before the chunk / slice axes existed carry neither field: whole arms
        ("whole arm (no chunk / slice)", m.get("chunk", "") == "" and int(m.get("max_anchors", 0)) == 0, f"chunk {m.get('chunk', 'absent')!r}, slice {m.get('max_anchors', 'absent')}"),
    ]
    if spec["model"] in TREES:
        out += [
            ("params = spec rule at these rows", m["params"] == c.tree_params(spec["model"], spec["rows"]), f"leaf minimum {m['params'].get('min_child_samples', m['params'].get('min_child_weight')):.4g}"),
            ("refit every 10, first forecast row 130", m["refit_every"] == c.REFIT_EVERY and m["tree_start"] == c.TREE_START, f"{m['refit_every']} {m['tree_start']}"),
            ("refit anchors", run["anchors"].tolist() == c.tree_anchors(n_fc), str(len(run["anchors"]))),
            ("training rows at every refit", bool((run["n_train"] == spec["rows"]).all()), str(np.unique(run["n_train"]))),
        ]
    else:
        out.append(("estimator", m["params"] == c.EST_OF[spec["model"]], str(m["params"])))
    return out


def gate() -> None:
    """Refit the first GATE_ANCHORS refits of the two reused LightGBM arms with this script's row
    builder and compare with the stored forecasts (bitwise); the row sets equal the data-size study's."""
    check()
    S = c.sources()
    n_fc = S["n_fc"]
    WORK.mkdir(parents=True, exist_ok=True)
    res = []
    for arm, ds_src in (("lgbm_bars1_r2000", "bar1600"), ("lgbm_bars2_r4000", "pool")):
        spec = ARMS[arm]
        src, _ = source(spec)
        same = bool(np.array_equal(src["X"], S[ds_src]["X"]) and np.array_equal(src["y"], S[ds_src]["y"]) and np.array_equal(src["fc"], S[ds_src]["fc"]))
        anchors = c.tree_anchors(n_fc)[:GATE_ANCHORS]
        t0 = time.time()
        out = c.run_tree("lgbm", src, spec["rows"], n_fc, anchors)
        ref = np.load(DS_WORK / f"{spec['reuse']}.npz")["pred"]
        m = np.isfinite(out["pred"])
        d = float(np.max(np.abs(out["pred"][m] - ref[m])))
        res.append(dict(arm=arm, reused=spec["reuse"], rows_equal=same, forecasts=int(m.sum()), max_abs_diff=d, bitwise=bool(np.array_equal(out["pred"][m], ref[m])), sec=time.time() - t0))
        print(res[-1], flush=True)
    (WORK / "gate_slice.json").write_text(json.dumps(res, indent=1))


# ============================================================================ run
def run(arms: list[str]) -> None:
    check()
    S = c.sources()
    n_fc = S["n_fc"]
    anchors = c.tree_anchors(n_fc)
    WORK.mkdir(parents=True, exist_ok=True)
    for arm in arms:
        spec = ARMS[arm]
        if spec["reuse"]:
            print(f"{arm}: reused from the data-size study ({spec['reuse']})", flush=True)
            continue
        f = WORK / f"{arm}.npz"
        if f.is_file():
            print(f"{arm}: done already", flush=True)
            continue
        if not (c.DESIGN / f"design_{spec['design']}_{c.BUCKET}.npz").is_file():
            print(f"{arm}: design {spec['design']} not captured yet, skipped", flush=True)
            continue
        claim = WORK / f"{arm}.claim"  # two processes never fit the same arm
        try:
            fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            print(f"{arm}: claimed by another process ({claim})", flush=True)
            continue
        os.write(fd, f"pid {os.getpid()} {time.ctime()}\n".encode())
        os.close(fd)
        src, facts = source(spec)
        model = spec["model"]
        print(f"{arm}: model={model} bars={spec['bars']} rows={spec['rows']} design={spec['design']} {facts}", flush=True)
        t0, c0 = time.time(), time.process_time()
        if model in TREES:
            out = c.run_tree(model, src, spec["rows"], n_fc, anchors)
        else:
            out = c.run_linear(model, src, spec["rows"], n_fc)
        wall, cpu = time.time() - t0, time.process_time() - c0
        out["n_sessions"] = sessions_in_windows(src, spec["rows"], anchors)
        meta = dict(
            arm=arm,
            **{k: v for k, v in spec.items()},
            design_facts=facts,
            bucket=c.BUCKET,
            n_columns=int(src["X"].shape[1]),
            refit_every=c.REFIT_EVERY if model in TREES else 1,
            tree_start=c.TREE_START if model in TREES else 0,
            params=c.tree_params(model, spec["rows"]) if model in TREES else c.EST_OF[model],
            wall_sec=wall,
            cpu_sec=cpu,
            versions=c.versions(),
        )
        np.savez_compressed(
            f,
            **{k: v for k, v in out.items() if isinstance(v, np.ndarray)},
            **{k: np.array(v) for k, v in out.items() if not isinstance(v, np.ndarray)},
            meta=json.dumps(meta),
        )
        claim.unlink()
        print(f"{arm}: wrote {f}  wall {wall:.0f}s  cpu {cpu:.0f}s", flush=True)


# ============================================================================ analyze
def load_arm(arm: str) -> dict | None:
    spec = ARMS[arm]
    f = (DS_WORK / f"{spec['reuse']}.npz") if spec["reuse"] else (WORK / f"{arm}.npz")
    if not f.is_file():
        return None
    z = np.load(f, allow_pickle=False)
    out = {k: z[k] for k in z.files}
    out["meta"] = json.loads(str(out["meta"]))
    return out


def asl():
    import atm_straddle_lib as m

    return m


def compare(P: dict, a: int, b: int) -> dict:
    dm = dm_test(P["ql"][:, a], P["ql"][:, b])
    t, lag = asl().newey_west_t(P["pnl"][:, a] - P["pnl"][:, b])
    return dict(dm=dm["dm"], dm_p=dm["p"], t_hac=t, hac_lag=lag)


def describe(arm: str) -> str:
    s = ARMS[arm]
    if s["rows_from"] == "h16":
        return "16:00 rows of the lastbars13 design"
    txt = "one-bar design (shipped)" if s["bars"] == 1 else f"{s['bars']} bars ending 16:00 ({s['design']})"
    if s["cols"] == BARMIN:
        txt += " + column bar_end_minute (design change)"
    return txt


def references(arm: str) -> list[str]:
    s = ARMS[arm]
    refs = [CTRL, TWO]
    if s["model"] != "lgbm":
        refs += [f"{s['model']}_bars1_r2000", f"{s['model']}_bars2_r4000"]
    if s["group"] == "rows4000":
        refs.append("lgbm_bars13_r26000" if s["bars"] == 13 else ("lgbm_bars4_r8000" if s["bars"] == 4 else CTRL))
    if s["cols"] == BARMIN:
        refs.append("lgbm_bars13_r26000")
    if s["model"] in LINEAR:
        refs.append(arm.replace(f"{s['model']}_", "lgbm_", 1))
    return [r for r in dict.fromkeys(refs) if r != arm]


def analyze() -> None:  # noqa: C901 - one linear report
    OUT.mkdir(parents=True, exist_ok=True)
    S = c.sources()
    n_fc = S["n_fc"]
    dz_all = S["dz"]
    dk = dvs.deck_frame()
    sub = np.arange(c.TREE_START, n_fc)
    dz = {k: v[sub] for k, v in dz_all.items()}
    runs = {a: r for a in ARMS if (r := load_arm(a)) is not None}
    arms = list(runs)
    assert CTRL in runs and TWO in runs
    P = dvs.deck_panel([dvs.research_frame(dz, runs[a]["pred"][sub]) for a in arms], dk)
    pt = dvs.point(P)
    col = {a: i for i, a in enumerate(arms)}
    n_days = P["ql"].shape[0]

    # ---------------------------------------------------------------- gates
    gates = []
    ds = pd.read_csv(DS_REPORT / "arms.csv").set_index("arm")
    for a in arms:
        s = ARMS[a]
        if not s["reuse"]:
            continue
        for what, j, k in (("QLIKE", "ql", "qlike"), ("Sharpe mid", "sh", "sharpe_mid")):
            v, ref = float(pt[j][col[a]]), float(ds.loc[s["reuse"], k])
            gates.append(dict(gate=f"{a} = data-size {s['reuse']}: {what} re-scored here = its arms.csv", value=v, reference=ref, abs_diff=abs(v - ref), passed=bool(abs(v - ref) < REPRO_TOL)))
        for name, ok, val in config_matches(a, runs[a]):
            gates.append(dict(gate=f"{a} = data-size {s['reuse']}: configuration, {name} ({val})", value=float(ok), reference=1.0, abs_diff=float(not ok), passed=bool(ok)))
    gf = WORK / "gate_slice.json"
    if gf.is_file():
        for g in json.loads(gf.read_text()):
            gates.append(dict(gate=f"{g['arm']}: rows built here = data-size rows (X, y, forecast rows)", value=float(g["rows_equal"]), reference=1.0, abs_diff=float(not g["rows_equal"]), passed=bool(g["rows_equal"])))
            gates.append(dict(gate=f"{g['arm']}: first {GATE_ANCHORS} refits redone here vs stored {g['reused']} ({g['forecasts']} forecasts), max |difference|", value=g["max_abs_diff"], reference=0.0, abs_diff=g["max_abs_diff"], passed=bool(g["bitwise"])))
    for a in arms:
        s = ARMS[a]
        if s["reuse"]:
            continue
        f = runs[a]["meta"]["design_facts"]
        if s["rows_from"] == "all" and s["bars"] > 1:
            gates.append(dict(gate=f"{a}: calendar column `hour` takes values on the 16:00 rows that no other row has ({s['design']})", value=float(f["hour_isolates_1600"]), reference=1.0, abs_diff=float(not f["hour_isolates_1600"]), passed=bool(f["hour_isolates_1600"])))
    G = pd.DataFrame(gates)
    G.to_csv(OUT / "gates.csv", index=False)

    # ---------------------------------------------------------------- arms
    rows = []
    for a in arms:
        s, r = ARMS[a], runs[a]
        m = r["meta"]
        ns = r["n_sessions"] if "n_sessions" in r else np.array([s["rows"] // s["bars"]])
        cm = {ref: compare(P, col[a], col[ref]) for ref in (CTRL, TWO) if ref != a}
        i = col[a]
        rows.append(
            dict(
                arm=a,
                model=LABEL[s["model"]],
                group=s["group"],
                bars=s["bars"],
                rows=s["rows"],
                sessions_min=int(ns.min()),
                sessions_max=int(ns.max()),
                train_rows=describe(a),
                reused_from=s["reuse"],
                leaf_min=(f"{m['params']['min_child_samples']}" if s["model"] == "lgbm" else (f"{m['params']['min_child_weight']:.3g}" if s["model"] == "xgb" else "")),
                kept_columns=(f"{int(r['kept_n'].min())}..{int(r['kept_n'].max())}" if "kept_n" in r else ""),
                qlike=pt["ql"][i],
                sharpe_mid=pt["sh"][i],
                sharpe_crossed=pt["shx"][i],
                pct_buy=100.0 * P["buy"][i],
                d_qlike_vs_1bar=pt["ql"][i] - pt["ql"][col[CTRL]],
                dm_vs_1bar=cm[CTRL]["dm"] if CTRL in cm else np.nan,
                dm_p_vs_1bar=cm[CTRL]["dm_p"] if CTRL in cm else np.nan,
                t_hac_vs_1bar=cm[CTRL]["t_hac"] if CTRL in cm else np.nan,
                d_qlike_vs_2bar=pt["ql"][i] - pt["ql"][col[TWO]],
                dm_vs_2bar=cm[TWO]["dm"] if TWO in cm else np.nan,
                dm_p_vs_2bar=cm[TWO]["dm_p"] if TWO in cm else np.nan,
                t_hac_vs_2bar=cm[TWO]["t_hac"] if TWO in cm else np.nan,
                fit_sec_mean=float(np.mean(r["fit_sec"])) if "fit_sec" in r else np.nan,
                fits=int(len(r["fit_sec"])) if "fit_sec" in r else int(r["n_solves"]),
                cpu_min=m["cpu_sec"] / 60.0,
                n_days=n_days,
            )
        )
    A = pd.DataFrame(rows)
    A["_o"] = A["group"].map({"ladder": 0, "rows4000": 1, "design change": 2})
    A["_m"] = A["model"].map({"LightGBM": 0, "XGBoost": 1, "ridge": 2, "lasso": 3})
    A = A.sort_values(["_m", "_o", "bars", "rows"]).drop(columns=["_o", "_m"]).reset_index(drop=True)
    A.to_csv(OUT / "arms.csv", index=False)

    vr = []
    for a in A["arm"]:
        for ref in references(a):
            if ref not in runs:
                continue
            cm = compare(P, col[a], col[ref])
            vr.append(
                dict(
                    arm=a,
                    reference=ref,
                    d_qlike=pt["ql"][col[a]] - pt["ql"][col[ref]],
                    pct_qlike=100.0 * (pt["ql"][col[a]] / pt["ql"][col[ref]] - 1.0),
                    dm=cm["dm"],
                    dm_p=cm["dm_p"],
                    d_sharpe_mid=pt["sh"][col[a]] - pt["sh"][col[ref]],
                    t_hac=cm["t_hac"],
                )
            )
    V = pd.DataFrame(vr)
    V.to_csv(OUT / "vs_reference.csv", index=False)

    master = pd.read_csv(MASTER).drop_duplicates("key").set_index("key")
    Cd = pd.DataFrame([dict(key=CITED_KEY, label=master.loc[CITED_KEY, "label"], qlike=float(master.loc[CITED_KEY, "qlike_recal"]), sharpe_mid=float(master.loc[CITED_KEY, "Sharpe_mid"]), sharpe_crossed=float(master.loc[CITED_KEY, "Sharpe_crossed"]), n_days=int(master.loc[CITED_KEY, "n_days"]))])
    Cd.to_csv(OUT / "cited.csv", index=False)
    plot(A, Cd)
    write_summary(G, A, V, Cd)
    print(A[["arm", "bars", "rows", "qlike", "sharpe_mid", "dm_vs_1bar", "t_hac_vs_1bar", "dm_vs_2bar", "fit_sec_mean", "cpu_min"]].to_string(index=False))


def plot(A: pd.DataFrame, Cd: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colr = {"LightGBM": dvs.COLOR["lgbm"], "XGBoost": dvs.COLOR["xgb"], "ridge": dvs.COLOR["ridge"], "lasso": dvs.COLOR["reclasso"]}
    dash = {"LightGBM": dvs.DASH["lgbm"], "XGBoost": dvs.DASH["xgb"], "ridge": dvs.DASH["ridge"], "lasso": dvs.DASH["reclasso"]}
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), constrained_layout=True)
    for ax, ycol, ylab in ((axes[0], "qlike", "QLIKE, 16:00 forecast (866 trade days)"), (axes[1], "sharpe_mid", "sign(s) Sharpe, mid")):
        for m in ("LightGBM", "XGBoost", "ridge", "lasso"):
            h = A[(A["model"] == m) & (A["group"] == "ladder")].sort_values("bars")
            if len(h):
                ax.plot(h["bars"], h[ycol], color=colr[m], linestyle=dash[m], linewidth=2, marker="o", markersize=7, label=f"{m}, 2000 sessions")
        h = A[(A["model"] == "LightGBM") & (A["rows"] == ROWS_FIXED)].sort_values("bars")
        if len(h) > 1:
            ax.plot(h["bars"], h[ycol], color=colr["LightGBM"], linestyle=(0, (1, 2)), linewidth=1.5, marker="s", markersize=8, markerfacecolor="white", markeredgewidth=2, label="LightGBM, 4000 rows (4000 / bars sessions)")
        b = A[A["group"] == "design change"]
        if len(b):
            ax.plot(b["bars"], b[ycol], linestyle="none", marker="*", markersize=15, color=colr["LightGBM"], markeredgecolor="white", label="LightGBM + bar_end_minute column (design change)")
        ax.axhline(Cd["qlike" if ycol == "qlike" else "sharpe_mid"].iloc[0], color="#7a7a7a", linewidth=1, linestyle=(0, (4, 3)), label="pooled 48-bar LightGBM (master table, cited)")
        ax.set_xticks(list(LADDER))
        ax.set_xlim(0.5, 13.6)
        ax.set_xlabel("bars ending 16:00 in the training rows")
        ax.set_ylabel(ylab)
        ax.grid(True, color="#d9d9d9", linewidth=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle("16:00 forecast, all features: training on the N bars ending 16:00, scoring the 16:00 forecast only", fontsize=11)
    fig.savefig(OUT / "bars_ladder.png", dpi=150)
    plt.close(fig)


def _t(v: float, nd: int = 2) -> str:
    return "" if pd.isna(v) else f"{v:.{nd}f}"


def write_summary(G: pd.DataFrame, A: pd.DataFrame, V: pd.DataFrame, Cd: pd.DataFrame) -> None:  # noqa: C901
    Ai = A.set_index("arm")
    ctrl, two = Ai.loc[CTRL], Ai.loc[TWO]
    cited = Cd.iloc[0]
    L = ["# Training the 16:00 trees on more bars than the last two (close study, 2026-10-04)", ""]
    L.append(
        "Question (the user's request): *run ablations on training on more bars than the last 2.* The data-size study trained LightGBM on both "
        "last-hour bars (2000 sessions = 4000 rows) and scored the 16:00 forecast only. This study trains on the N half-hour bars ending 16:00, "
        "N = 1, 2, 3, 4, 5, 7, 13, and again scores only the 16:00 forecast. Written by `experiments/close_trees_morebars.py analyze` from the CSVs "
        "in this folder; every number below is in them."
    )
    L.append("")
    L.append("## Setup")
    L.append(
        "- Design: `experiments/capture_design_close.py`, all features. N = 1 is the shipped one-bar design (`bar1600`), N = 2 is `last30`, "
        "N >= 3 is `lastbars<N>`. Each multi-bar design's rolling robust scaling uses its own window (2000 sessions x the median bars a session), "
        "so the 16:00 rows of two designs differ in the scaling of some columns; the target on the 16:00 rows is the same (asserted)."
    )
    L.append(
        f"- Trees: the tree spec's shipped configs, refit every {c.REFIT_EVERY} sessions from forecast row {c.TREE_START} (2019-01), single-threaded, window mask on, "
        "one model forecasting the 16:00 row; the leaf minimum scaled to the training rows by the spec's rule (LightGBM `min_child_samples` = round(98 x rows / 24000), "
        "98 at the pooled bank's 24000 rows). The model in force for the 16:00 row of day d was fitted on the design rows strictly before that row: the earlier bars of day d "
        "(targets realized by 15:30, when the forecast is issued) may enter it, nothing later does."
    )
    L.append(
        "- Rungs N = 1 and N = 2 are the data-size study's `lgbm_bar1600_w2000` and `lgbm_pool_w4000` (and its XGBoost / ridge / lasso arms of the same rows), reused after the gates below."
    )
    L.append(
        "- Scorer: the master table's research convention (16:00 recalibration over the previous 250 sessions, the 15:30 sign(s) straddle trade), "
        f"{int(ctrl['n_days'])} trade days 2020-01-03 .. 2024-04-30, on the one-bar design's forecast rows. Point estimates (the block bootstrap is off); "
        "DM = Diebold-Mariano on daily QLIKE (negative = the arm has lower loss), HAC t = Newey-West t on the paired daily mid-fill P&L difference (positive = the arm earns more)."
    )
    L.append("")

    lad = A[(A["model"] == "LightGBM") & (A["group"] == "ladder")].sort_values("bars")
    L.append("## Main ladder: LightGBM, 2000 sessions, N bars")
    L.append("")
    L.append("| bars | rows | sessions | leaf min | QLIKE | Sharpe mid | Sharpe crossed | DM vs 1 bar | HAC t vs 1 bar | DM vs 2 bars | HAC t vs 2 bars | fit s (mean) | CPU min |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in lad.iterrows():
        sess = f"{r['sessions_min']}" if r["sessions_min"] == r["sessions_max"] else f"{r['sessions_min']}..{r['sessions_max']}"
        L.append(
            f"| {r['bars']} | {r['rows']} | {sess} | {r['leaf_min']} | {r['qlike']:.4f} | {r['sharpe_mid']:.2f} | {r['sharpe_crossed']:.2f} | {_t(r['dm_vs_1bar'])} | "
            f"{_t(r['t_hac_vs_1bar'])} | {_t(r['dm_vs_2bar'])} | {_t(r['t_hac_vs_2bar'])} | {_t(r['fit_sec_mean'], 1)} | {r['cpu_min']:.1f} |"
        )
    L.append(
        f"| pooled 48-bar LightGBM (master table `{cited['key']}`, cited, not rerun) | 24000 bars | about 500 |  | {cited['qlike']:.4f} | {cited['sharpe_mid']:.2f} | {cited['sharpe_crossed']:.2f} |  |  |  |  |  |  |"
    )
    L.append("")
    L.append("![bars ladder](bars_ladder.png)")
    L.append("")
    L.append("## Reading (numbers from the tables in this file)")
    best = lad.loc[lad["qlike"].idxmin()]
    L.append(
        "- LightGBM, 2000 sessions, QLIKE for N = " + " / ".join(str(b) for b in lad["bars"]) + ": " + " / ".join(f"{q:.4f}" for q in lad["qlike"])
        + "; Sharpe mid " + " / ".join(f"{s:.2f}" for s in lad["sharpe_mid"]) + "."
    )
    L.append(
        f"- Lowest QLIKE on the ladder: N = {best['bars']} ({best['rows']} rows), {best['qlike']:.4f}; against 1 bar DM {_t(best['dm_vs_1bar'])} (p {_t(best['dm_p_vs_1bar'], 3)}), "
        f"against 2 bars DM {_t(best['dm_vs_2bar'])} (p {_t(best['dm_p_vs_2bar'], 3)})."
    )
    big = lad[lad["bars"] > 2]
    if len(big):
        L.append(
            f"- N >= 3 against 2 bars: DM from {big['dm_vs_2bar'].min():.2f} to {big['dm_vs_2bar'].max():.2f}, HAC t from {big['t_hac_vs_2bar'].min():.2f} to {big['t_hac_vs_2bar'].max():.2f}; "
            f"against 1 bar: DM from {big['dm_vs_1bar'].min():.2f} to {big['dm_vs_1bar'].max():.2f}, HAC t from {big['t_hac_vs_1bar'].min():.2f} to {big['t_hac_vs_1bar'].max():.2f}."
        )
    r4 = A[(A["model"] == "LightGBM") & (A["rows"] == ROWS_FIXED)].sort_values("bars")
    if len(r4) > 1:
        L.append(
            "- Rows held at 4000 (LightGBM), N = " + " / ".join(str(b) for b in r4["bars"]) + " (sessions " + " / ".join(f"{s}" for s in r4["sessions_min"]) + "): QLIKE "
            + " / ".join(f"{q:.4f}" for q in r4["qlike"]) + "; Sharpe mid " + " / ".join(f"{s:.2f}" for s in r4["sharpe_mid"]) + "."
        )
    if "lgbm_bars13_r26000_barmin" in Ai.index and "lgbm_bars13_r26000" in Ai.index:
        v = V[(V["arm"] == "lgbm_bars13_r26000_barmin") & (V["reference"] == "lgbm_bars13_r26000")].iloc[0]
        L.append(
            f"- N = 13 with the added column `bar_end_minute` (design change) vs without: QLIKE {Ai.loc['lgbm_bars13_r26000', 'qlike']:.4f} -> {Ai.loc['lgbm_bars13_r26000_barmin', 'qlike']:.4f} "
            f"(DM {v['dm']:.2f}), Sharpe mid {Ai.loc['lgbm_bars13_r26000', 'sharpe_mid']:.2f} -> {Ai.loc['lgbm_bars13_r26000_barmin', 'sharpe_mid']:.2f} (HAC t {v['t_hac']:.2f})."
        )
    for m in ("XGBoost", "ridge", "lasso"):
        h = A[(A["model"] == m) & (A["group"] == "ladder")].sort_values("bars")
        if len(h) > 2:
            L.append(
                f"- {m}, 2000 sessions, QLIKE for N = " + " / ".join(str(b) for b in h["bars"]) + ": " + " / ".join(f"{q:.4f}" for q in h["qlike"])
                + "; Sharpe mid " + " / ".join(f"{s:.2f}" for s in h["sharpe_mid"]) + "."
            )
    L.append(
        f"- Far end, cited: the paper's pooled 48-bar LightGBM (24000 bars, about 500 sessions, a different design and its own recalibration of the 16:00 bar within the master table) "
        f"QLIKE {cited['qlike']:.4f}, Sharpe mid {cited['sharpe_mid']:.2f}; 1-bar control {ctrl['qlike']:.4f} / {ctrl['sharpe_mid']:.2f}, 2 bars {two['qlike']:.4f} / {two['sharpe_mid']:.2f}."
    )
    L.append("")
    oth = A[~((A["model"] == "LightGBM") & (A["group"] == "ladder"))]
    if len(oth):
        L.append("## Other arms")
        L.append("")
        L.append("| arm | model | bars | rows | sessions | training rows | QLIKE | Sharpe mid | Sharpe crossed | DM vs 1 bar | HAC t vs 1 bar | DM vs 2 bars | HAC t vs 2 bars | fit s (mean) | CPU min |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for _, r in oth.iterrows():
            sess = f"{r['sessions_min']}" if r["sessions_min"] == r["sessions_max"] else f"{r['sessions_min']}..{r['sessions_max']}"
            L.append(
                f"| `{r['arm']}` | {r['model']} | {r['bars']} | {r['rows']} | {sess} | {r['train_rows']} | {r['qlike']:.4f} | {r['sharpe_mid']:.2f} | {r['sharpe_crossed']:.2f} | "
                f"{_t(r['dm_vs_1bar'])} | {_t(r['t_hac_vs_1bar'])} | {_t(r['dm_vs_2bar'])} | {_t(r['t_hac_vs_2bar'])} | {_t(r['fit_sec_mean'], 1)} | {r['cpu_min']:.1f} |"
            )
        L.append("")
    L.append("## Pairwise differences (`vs_reference.csv`)")
    L.append("")
    L.append("| arm | reference | dQLIKE | dQLIKE % | DM | p | dSharpe mid | HAC t |")
    L.append("|---|---|---|---|---|---|---|---|")
    for _, r in V.iterrows():
        L.append(f"| `{r['arm']}` | `{r['reference']}` | {r['d_qlike']:+.4f} | {r['pct_qlike']:+.1f} | {r['dm']:.2f} | {r['dm_p']:.3f} | {r['d_sharpe_mid']:+.2f} | {_t(r['t_hac'])} |")
    L.append("")
    L.append("## Gates")
    for _, g in G.iterrows():
        L.append(f"- {g['gate']}: {g['value']:.6g} vs {g['reference']:.6g} (|diff| {g['abs_diff']:.2e}) {'PASS' if g['passed'] else 'FAIL'}")
    L.append("")
    L.append("## Caveats")
    L.append(
        "- The leaf minimum changes with the rows (the spec's rule), so each rung changes the rows, the bars and the leaf minimum together; the rows-held-at-4000 arms hold the rows and the leaf minimum (16) fixed."
    )
    L.append(
        "- Each design has its own rolling scaling, so two rungs differ also in the scaling of some columns; in the data-size study the same LightGBM on two scalings of the same 16:00 rows gave QLIKE 0.1007 vs 0.1006 and Sharpe mid 1.99 vs 1.56."
    )
    L.append(
        "- Linear arms solve at every row (the earlier bars included, not scored) and re-choose the penalty every 250 solves, i.e. every 250 / N sessions; tree arms refit every 10 sessions."
    )
    L.append("- Refit every 10 sessions, one seed, point estimates only; Sharpe ratios of forecasts with nearly equal QLIKE move by several tenths (data-size and tuning studies).")
    L.append("")
    L.append("## Not run")
    missing = [a for a in ARMS if a not in set(A["arm"])]
    L.append("- Arms defined in the script and not run: " + (", ".join(f"`{a}`" for a in missing) if missing else "none") + ".")
    L.append("")
    L.append("## Files")
    L.append("- `experiments/close_trees_morebars.py` (stages gate / run / analyze; fitting machinery from `experiments/close_trees_datasize.py`)")
    L.append("- `arms.csv`, `vs_reference.csv`, `gates.csv`, `cited.csv`, `bars_ladder.png`; forecasts in `_work/<arm>.npz` (not committed; rungs 1 and 2 read from `../trees_datasize/_work/`)")
    own = A[A["reused_from"].fillna("") == ""]
    L.append(f"- CPU: {own['cpu_min'].sum():.0f} min over the {len(own)} arms fitted here (every fit single-threaded, at most four processes at once); the reused arms took {A.loc[A['reused_from'].fillna('') != '', 'cpu_min'].sum():.0f} min in the data-size study.")
    (OUT / "SUMMARY.md").write_text("\n".join(L) + "\n", encoding="utf-8")


# ============================================================================ main
if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("gate", "run", "analyze"):
        raise SystemExit(__doc__)
    if sys.argv[1] == "gate":
        gate()
    elif sys.argv[1] == "run":
        run(sys.argv[2:] or ORDER)
    else:
        analyze()
