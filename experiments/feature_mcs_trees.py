"""Feature Model Confidence Set, the TREE candidates (2026-10-08): the per-bar 16:00 LightGBM / XGBoost / random forest on
the 96 input designs of experiments/feature_mcs_runner.py, fixed top-entry settings, refit every session (T1), window-masked
like the stored runs, forecasts for the 866 MCS trade days 2020-01-03 .. 2024-04-30 only.

  est   model  shipped spec (unchanged; specs/causal_tune_trees_mcs.py patches the design in)   stored T1 tables
  lgbm  lgbm   specs/causal_tune_trees.py         lgbm_opt_00     results/spxw_pnl/yhat_subtree_daily_<set>_lgbm.parquet
  xgb   xgb    specs/causal_tune_trees_xgbtop.py  xgb_opt_00      results/trees_xgbtop_1600/yhat_subtree_daily_<set>_xgbtop.parquet
  rf    rf     specs/causal_tune_trees_rf400.py   trial 255       results/trees_rf400_1600/yhat_subtree_daily_<set>_rf400.parquet

Modes:
  prepare       (local) the designs and rows the cluster reads, from the linear runner's OWN definitions (imported, not
                re-derived): feature_mcs_runner.load_inputs()["models"] -> cluster/feature_mcs_trees_designs.json (+ the 628
                names, the forecast dates); its ref_armf.npz day_row (the 866 trade days' forecast rows) ->
                cluster/feature_mcs_trees_rows.json; the canary task file cluster/feature_mcs_trees_canary.txt
  canary_check  (CARC) the canary's anchor rows vs the stored T1 runs (as the gate); the measured seconds per refit by design
                -> results/feature_mcs_trees_work/canary_costs.csv, canary_gate.csv, CANARY_OK
  plan          (local) the anchor and fleet task files per estimator from the measured costs:
                cluster/feature_mcs_trees_{anchors,fleet}_<est>.txt (packs of <= MCS_PACK_MAX = 32 single-threaded processes, one CPU each)
  gate --est E  (CARC) E's two anchors (baseline, all_features) on all 866 days: bit for bit vs the stored T1 runs (their
                merged results CSVs, parsed exactly), and vs the stored T1 tables within the tables' own CSV parse (<= 2 ulp)
                -> results/feature_mcs_trees_work/gate_<E>.csv, GATE_OK_<E>
  collect       (CARC) every finished (estimator, design) -> results/feature_mcs_2026-10-08/trees/<est>/<design>.npz, the
                layout experiments/feature_mcs_report.py reads: pred = the adjusted-scale forecast on the 866 trade days,
                date = 'YYYY-MM-DD 16:00:00'; <design>.json = cpu (the CPU model), host, cpu_total (CPU seconds), wall;
                census.csv (every chunk's host, CPU, times), status.csv, gate_<est>.csv; TREES_DONE LAST, when all 288 are
                written, the three gates passed and every chunk ran on its estimator's CPU set (CPU_OK: LightGBM EPYC 7513 only;
                XGBoost and the forest EPYC 7513 or 7542, byte-identical classes).
Work (chunks, canary, flags) lives in results/feature_mcs_trees_work/, outside the trees folder the report pulls whole.

Forecast scale: pred_adj (the adjusted target scale, the fit space); the plain forecast is pred_adj^2 x B.  The value delivered is
the spec's npz pred_adj, the stage the stored T1 runs' merged CSVs hold (see read_chunks).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = (
    REPO / "results" / "feature_mcs_2026-10-08" / "trees"
)  # read (and pulled whole) by the MCS report
WORK = REPO / "results" / "feature_mcs_trees_work"
CHUNKS = WORK / "chunks"
# chunk roots, one per chunk geometry: chunks = the first plan (MCS_TARGET_SEC 1800); chunks2 = the backfill-sized
# re-plan (2026-10-09 01:30: 4-process packs, ~700 s chunks).  A row may exist in both (the bytes must agree).
CHUNK_ROOTS = (WORK / "chunks", WORK / "chunks2")
# the CPU models each estimator may run on: LightGBM is not CPU-invariant (the stored class only); XGBoost and the
# forest are byte-identical on EPYC 7513 and 7542 (results/trees_rfdev_preview163/xcpu_carc/SUMMARY.md, the xgbtop
# cross-CPU cell); the anchors and the gates run on the reference class, EPYC 7513
_EPYC = {"AMD EPYC 7513 32-Core Processor", "AMD EPYC 7542 32-Core Processor"}
# 2026-10-13 (the bucket designs, CARC /scratch1 storage failure): Hoffman2's AVX-only models, byte-identical to EPYC 7513
# for the forest (results/trees_rfdev_preview163/xcpu_carc/SUMMARY.md) and XGBoost (coordinator, the xgbtop cross-CPU work)
_H2_AVX = {
    "Intel(R) Xeon(R) CPU E5-2670 0 @ 2.60GHz",
    "Intel(R) Xeon(R) CPU E5-2650 v2 @ 2.60GHz",
}
CPU_OK = {
    "lgbm": {"AMD EPYC 7513 32-Core Processor"},
    "xgb": _EPYC | _H2_AVX,
    "rf": _EPYC | _H2_AVX,
}
CANARY = WORK / "canary"
DESIGNS = REPO / "cluster" / "feature_mcs_trees_designs.json"
ROWS = REPO / "cluster" / "feature_mcs_trees_rows.json"
ESTS = {
    "lgbm": dict(
        model="lgbm",
        base="specs/causal_tune_trees.py",
        token="lgbm",
        stored="results/spxw_pnl",
        label="LightGBM lgbm_opt_00",
    ),
    "xgb": dict(
        model="xgb",
        base="specs/causal_tune_trees_xgbtop.py",
        token="xgbtop",
        stored="results/trees_xgbtop_1600",
        label="XGBoost xgb_opt_00 (xgbtop)",
    ),
    "rf": dict(
        model="rf",
        base="specs/causal_tune_trees_rf400.py",
        token="rf400",
        stored="results/trees_rf400_1600",
        label="random forest trial 255 (rf400)",
    ),
}
ANCHORS = ("baseline", "all_features")
W = 2000  # TRAIN_WIN = the halo: absolute row = W + OOS row
N_ALL = 3469  # rows of the series (W + 1469 forecast rows)
PACK_MAX = int(
    os.environ.get("MCS_PACK_MAX", "32")
)  # processes (= CPUs) per job: few, large jobs (the per-user JOB cap binds, not CPUs)
TARGET_SEC = float(
    os.environ.get("MCS_TARGET_SEC", "1800")
)  # one process's planned compute
STAGGER = 15


def say(msg: str) -> None:
    print(msg, flush=True)


def stored_path(est: str, design: str) -> Path:
    e = ESTS[est]
    return REPO / e["stored"] / f"yhat_subtree_daily_{design}_{e['token']}.parquet"


def chunk_npz(root: Path, est: str, design: str, chunk: int) -> Path:
    m = ESTS[est]["model"]
    return (
        root
        / est
        / design
        / f"c{chunk}"
        / "causal_tune_trees"
        / m
        / "all_features"
        / "trees_bar1600.npz"
    )


def load_designs() -> tuple[dict, list[int], pd.DatetimeIndex]:
    d = json.loads(DESIGNS.read_text(encoding="utf-8"))
    r = json.loads(ROWS.read_text(encoding="utf-8"))
    return (
        d,
        [int(x) for x in r["oos_rows"]],
        pd.DatetimeIndex(pd.to_datetime(d["dates_oos"])),
    )


# ---------------------------------------------------------------- prepare (local)
def prepare(designs_only: bool = False) -> None:
    sys.path.insert(0, str(REPO / "experiments"))
    import feature_mcs_runner as FR  # the linear runner: its design definitions, imported

    inp = FR.load_inputs()
    models = inp["models"]
    names = list(inp["names"])
    w = int(inp["W"])
    assert w == W, w
    dates = list(inp["dates"])
    assert len(dates) == N_ALL, len(dates)
    ref = np.load(FR.INP / "ref_armf.npz")
    day_row = [int(x) for x in ref["day_row"]]
    assert (
        len(day_row) == 866 and day_row == sorted(day_row) and len(set(day_row)) == 866
    )
    order = list(models)
    assert order[:2] == list(ANCHORS) and len(order) >= 96, order[:3]
    if DESIGNS.is_file():  # designs already run must keep their columns (2026-10-12: the 16 bucket designs added)
        old = json.loads(DESIGNS.read_text(encoding="utf-8"))
        for m in old["order"]:
            assert m in models and list(models[m]["cols"]) == list(
                old["models"][m]["cols"]
            ), m
    for m in order:
        cols = models[m]["cols"]
        idx = [names.index(c) for c in cols]
        assert idx == sorted(idx), m
    DESIGNS.write_text(
        json.dumps(
            dict(
                source="experiments/feature_mcs_runner.py load_inputs()['models'] (results/feature_mcs_2026-10-08/_work/inputs/models.csv)",
                names=names,
                backbone=list(inp["backbone"]),
                W=w,
                dates_oos=dates[w:],
                order=order,
                models={
                    m: dict(
                        kind=models[m]["kind"],
                        series=models[m]["series"],
                        cols=list(models[m]["cols"]),
                    )
                    for m in order
                },
            )
        ),
        encoding="utf-8",
    )
    if designs_only:
        say(
            f"prepare --designs-only: {len(order)} designs -> {DESIGNS.relative_to(REPO)}"
        )
        return
    ROWS.write_text(
        json.dumps(
            dict(
                source="results/feature_mcs_2026-10-08/_work/inputs/ref_armf.npz day_row (the 866 trade days' forecast rows)",
                oos_rows=day_row,
                days=[str(pd.Timestamp(dates[w + r]).date()) for r in day_row],
            )
        ),
        encoding="utf-8",
    )
    say(
        f"prepare: {len(order)} designs, {len(names)} columns, {len(day_row)} rows ({dates[w + day_row[0]]} .. {dates[w + day_row[-1]]})"
    )
    # canary: every estimator, the two anchors, one add and one leave-one-out design, a slice of rows each
    kinds = pd.DataFrame(
        [dict(model=m, kind=models[m]["kind"], n=len(models[m]["cols"])) for m in order]
    )
    add_big = (
        kinds[kinds.kind == "add"].sort_values(["n", "model"]).iloc[-1]["model"]
    )  # the largest add-one design
    loo_big = (
        kinds[kinds.kind == "loo"].sort_values(["n", "model"]).iloc[-1]["model"]
    )  # the largest leave-one-out design
    rows_of = {
        "lgbm": (866, 120, 866, 120),
        "xgb": (866, 120, 866, 120),
        "rf": (200, 25, 200, 25),
    }
    lines = []
    for est, (nb, na, nadd, nloo) in rows_of.items():
        for design, n in (
            ("baseline", nb),
            ("all_features", na),
            (add_big, nadd),
            (loo_big, nloo),
        ):
            sel = day_row[:n]
            end = W + sel[-1] + 1
            lines.append(
                f"0 {est} {design} 0 {W + sel[0]} {-1 if end == N_ALL else end} {W} 0"
            )
    (REPO / "cluster" / "feature_mcs_trees_canary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )
    say(
        f"canary: {len(lines)} processes (add {add_big}, loo {loo_big}) -> cluster/feature_mcs_trees_canary.txt"
    )


# ---------------------------------------------------------------- shared readers
def read_chunks(
    root: Path, est: str, design: str, oos_rows: list[int], dates_oos: pd.DatetimeIndex
) -> tuple[pd.Series, list[dict]]:
    """pred_adj by OOS row over every finished chunk of (est, design); each chunk's rows, meta and node."""
    out: dict[int, float] = {}
    info = []
    want = set(oos_rows)
    for d in sorted((root / est / design).glob("c*")):
        if not (d / "DONE").is_file() or not d.name[1:].isdigit():
            continue
        z = np.load(chunk_npz(root, est, design, int(d.name[1:])), allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        assert (
            meta.get("mcs_design") == design and meta.get("model") == ESTS[est]["model"]
        ), (d, meta.get("mcs_design"))
        assert (
            meta.get("window_mask") == 1
            and meta.get("refit_every") == 1
            and meta.get("bucket") == "all_features"
        ), d
        dt = pd.DatetimeIndex(pd.to_datetime(z["date"]))
        pos = dates_oos.get_indexer(dt)
        assert (pos >= 0).all(), d
        # The delivered forecast is the spec's npz pred_adj: the chunk's results CSV (the model's float64 output in
        # full repr) read back by pandas' default parser on CARC -- the same stage as the stored T1 runs' MERGED
        # CSVs (their reducer reads the chunk CSVs the same way): canary 12861536, our npz == the stored merged CSVs
        # (parsed exactly) on every anchor row, and our exact CSV parse == the stored raw chunk CSVs.  The default
        # parse can sit 1-2 ulp from the exact one (checked here), and the stored T1 TABLES add a second default
        # parse (pandas 2.3, locally) of the merged CSVs, so they sit <= 2 ulp from the runs on a few rows.
        res = pd.read_csv(
            chunk_npz(root, est, design, int(d.name[1:])).with_name(
                "results_bar1600.csv"
            ),
            parse_dates=["date"],
            float_precision="round_trip",
        )
        assert len(res) == len(dt) and (pd.DatetimeIndex(res["date"]) == dt).all(), d
        pa = res["pred_adj"].to_numpy(np.float64)
        pz = np.asarray(z["pred_adj"], dtype=np.float64)
        fin = np.isfinite(pa)
        assert (fin == np.isfinite(pz)).all(), d
        ulp = (
            float(np.max(np.abs(pa[fin] - pz[fin]) / np.spacing(np.abs(pa[fin]))))
            if fin.any()
            else 0.0
        )
        assert ulp <= 2.0, (
            d,
            ulp,
        )  # the npz is the same CSV, parsed by the default parser
        pa = pz
        n = 0
        for p, v in zip(pos, pa):
            if p in want:
                assert np.isfinite(v), (d, p)
                assert (
                    p not in out
                    or np.float64(out[p]).tobytes() == np.float64(v).tobytes()
                ), (d, p)
                out[int(p)] = float(v)
                n += 1
            else:
                assert not np.isfinite(v), (d, p)
        node = (
            (d / "NODE").read_text(encoding="utf-8").replace("\n", " ").strip()
            if (d / "NODE").is_file()
            else ""
        )
        cpu = (
            (d / "CPU").read_text(encoding="utf-8").strip()
            if (d / "CPU").is_file()
            else ""
        )
        tm = (
            (d / "TIME").read_text(encoding="utf-8").split()
            if (d / "TIME").is_file()
            else []
        )
        real, cpu_s = (
            (float(tm[-3]), float(tm[-2]) + float(tm[-1]))
            if len(tm) >= 3
            else (np.nan, np.nan)
        )
        info.append(
            dict(
                est=est,
                design=design,
                chunk=d.name,
                rows=n,
                node=node,
                host=node.split(" ")[1] if node.startswith("host ") else "",
                cpu=cpu,
                real_sec=real,
                cpu_sec=cpu_s,
                fit_sec=float(np.sum(z["fit_sec"])),
                refits=len(z["fit_sec"]),
                walk_sec=meta.get("walk_sec"),
                run_sec=meta.get("run_sec"),
                n_features=meta.get("n_features"),
                csv_vs_npz_max_ulp=ulp,
                kept_mean=float(np.mean(z["kept_n"])) if len(z["kept_n"]) else np.nan,
                params=json.dumps(meta.get("params"), sort_keys=True),
            )
        )
    return pd.Series(out, dtype=float).sort_index(), info


def read_all(
    est: str, design: str, oos_rows: list[int], dates_oos: pd.DatetimeIndex
) -> tuple[pd.Series, list[dict]]:
    """read_chunks over every chunk root; a row computed in two roots must carry the same bytes."""
    parts, info = [], []
    for root in CHUNK_ROOTS:
        s_, i_ = read_chunks(root, est, design, oos_rows, dates_oos)
        parts.append(s_)
        info += [dict(i, root=root.name) for i in i_]
    allp = pd.concat(parts)
    dup = allp[allp.index.duplicated(keep=False)]
    for r, g in dup.groupby(level=0):
        assert len({np.float64(v).tobytes() for v in g}) == 1, (est, design, r)
    return allp[~allp.index.duplicated()].sort_index(), info


def stored_rows(est: str, design: str, dates_oos: pd.DatetimeIndex) -> pd.Series:
    """The stored T1 TABLE's yhat (pandas' default CSV parse of the stored run) by OOS row."""
    t = pd.read_parquet(stored_path(est, design), columns=["t", "yhat"])
    et = (
        pd.DatetimeIndex(t["t"])
        .tz_convert("America/New_York")
        .tz_localize(None)
        .as_unit("ns")
    )
    pos = dates_oos.get_indexer(et)
    assert (pos >= 0).all() and len(pos) == len(dates_oos), (est, design)
    return pd.Series(t["yhat"].to_numpy(np.float64), index=pos).sort_index()


STORED_RUN = {
    "lgbm": "results/linear_subsection_trees_mask/t1",
    "xgb": "results/linear_subsection_trees_mask_xgbtop/t1",
    "rf": "results/linear_subsection_trees_mask_rf400/t1",
}


def stored_run_path(est: str, design: str) -> Path:
    m = ESTS[est]["model"]
    return (
        REPO
        / STORED_RUN[est]
        / design
        / "bar1600"
        / m
        / "tw2000"
        / "causal_tune_trees"
        / m
        / design
        / "results_bar1600.csv"
    )


def stored_run_rows(est: str, design: str, dates_oos: pd.DatetimeIndex) -> pd.Series:
    """The stored T1 RUN's forecast (its merged results CSV, parsed exactly: the values its reducer holds) by OOS row."""
    r = pd.read_csv(
        stored_run_path(est, design), parse_dates=["date"], float_precision="round_trip"
    )
    pos = dates_oos.get_indexer(pd.DatetimeIndex(r["date"]))
    assert (pos >= 0).all() and len(pos) == len(dates_oos), (est, design)
    return pd.Series(r["pred_adj"].to_numpy(np.float64), index=pos).sort_index()


def compare(ours: pd.Series, ref: pd.Series) -> dict:
    r = ref.reindex(ours.index)
    a, b = ours.to_numpy(np.float64), r.to_numpy(np.float64)
    same = a.tobytes() == b.tobytes()
    d = np.abs(a - b)
    return dict(
        rows=len(a),
        bit_identical=bool(same),
        rows_differ=int((a != b).sum()),
        max_abs_diff=float(np.nanmax(d)) if len(d) else np.nan,
        max_rel_diff=float(np.nanmax(d / np.abs(b))) if len(d) else np.nan,
    )


def anchor_check(
    est: str, design: str, ours: pd.Series, dates_oos: pd.DatetimeIndex
) -> dict:
    """The gate of one anchor: bit for bit against the stored T1 RUN (its merged results CSV parsed exactly); against the
    stored T1 TABLE (a further default CSV parse of that run, pandas 2.3) within that parse's rounding, at most 2 ulp."""
    run = compare(ours, stored_run_rows(est, design, dates_oos))
    tab_s = stored_rows(est, design, dates_oos)
    tab = compare(ours, tab_s)
    t = tab_s.reindex(ours.index).to_numpy(np.float64)
    a = ours.to_numpy(np.float64)
    ulp = float(np.max(np.abs(a - t) / np.spacing(np.abs(t)))) if len(a) else np.nan
    return dict(
        ok=bool(run["bit_identical"] and ulp <= 2.0),
        rows=run["rows"],
        run_bit_identical=run["bit_identical"],
        run_rows_differ=run["rows_differ"],
        run_max_abs_diff=run["max_abs_diff"],
        table_bit_identical=tab["bit_identical"],
        table_rows_differ=tab["rows_differ"],
        table_max_ulp=ulp,
    )


# ---------------------------------------------------------------- canary_check (CARC)
def canary_check() -> int:
    designs, oos_rows, dates_oos = load_designs()
    lines = [
        ln.split()
        for ln in (REPO / "cluster" / "feature_mcs_trees_canary.txt")
        .read_text()
        .splitlines()
        if ln.strip()
    ]
    rows, costs, ok = [], [], True
    for _, est, design, *_ in lines:
        ours, info = read_chunks(CANARY, est, design, oos_rows, dates_oos)
        if not info:
            rows.append(dict(est=est, design=design, ok=False, why="no finished chunk"))
            ok = False
            continue
        for i in info:
            p = len(designs["models"][design]["cols"])
            costs.append(
                dict(
                    i,
                    p=p,
                    sec_per_refit=i["fit_sec"] / max(1, i["refits"]),
                    overhead_sec=(i["run_sec"] or 0) - (i["walk_sec"] or 0),
                )
            )
        if design in ANCHORS:
            c = anchor_check(est, design, ours, dates_oos)
            rows.append(dict(est=est, design=design, **c))
            ok &= c["ok"]
    WORK.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(WORK / "canary_gate.csv", index=False)
    pd.DataFrame(costs).to_csv(WORK / "canary_costs.csv", index=False)
    say(pd.DataFrame(rows).to_string(index=False))
    say(
        pd.DataFrame(costs)[
            ["est", "design", "p", "rows", "sec_per_refit", "overhead_sec", "node"]
        ].to_string(index=False)
    )
    if ok:
        (WORK / "CANARY_OK").write_text(
            "canary anchor rows bit-identical to the stored T1 runs (the tables within their CSV parse, <= 2 ulp)\n",
            encoding="utf-8",
        )
    return 0 if ok else 1


# ---------------------------------------------------------------- plan (local)
def plan(costs_csv: Path, suffix: str = "", only: tuple[str, ...] = ()) -> None:
    designs, oos_rows, _ = load_designs()
    c = pd.read_csv(costs_csv)
    proj = []
    for est in ESTS:
        ce = c[c.est == est]
        A = np.column_stack([np.ones(len(ce)), ce["p"].to_numpy(float)])
        coef, *_ = np.linalg.lstsq(A, ce["sec_per_refit"].to_numpy(float), rcond=None)
        a, b = float(coef[0]), float(coef[1])
        ov = float(ce["overhead_sec"].max())
        fitted = a + b * ce["p"].to_numpy(float)
        err = float(np.max(np.abs(fitted / ce["sec_per_refit"].to_numpy(float) - 1)))
        say(
            f"{est}: sec/refit = {a:.3f} + {b:.5f} p (max rel error on the canary designs {err:.0%}); overhead {ov:.0f} s/process"
        )
        stages = (
            [("fleet", [m for m in designs["order"] if m.startswith(only)])]
            if only
            else [
                ("anchors", list(ANCHORS)),
                ("fleet", [m for m in designs["order"] if m not in ANCHORS]),
            ]
        )
        for stage, names in stages:
            items = []
            for m in names:
                p = len(designs["models"][m]["cols"])
                spr = max(0.05, a + b * p)
                per = max(1, min(866, math.floor(TARGET_SEC / spr)))
                nch = math.ceil(866 / per)
                per = math.ceil(866 / nch)  # even chunks
                for k in range(nch):
                    sel = oos_rows[k * per : (k + 1) * per]
                    if not sel:
                        continue
                    end = W + sel[-1] + 1
                    items.append(
                        (
                            len(sel) * spr + ov,
                            m,
                            k,
                            W + sel[0],
                            -1 if end == N_ALL else end,
                            len(sel),
                        )
                    )
            items.sort(key=lambda x: -x[0])
            packs = [items[i : i + PACK_MAX] for i in range(0, len(items), PACK_MAX)]
            out = []
            for pk, pack in enumerate(packs, start=1):
                for sec, m, k, s, e, n in pack:
                    out.append(f"{pk} {est} {m} {k} {s} {e} {W} {int(round(sec))}")
            f = REPO / "cluster" / f"feature_mcs_trees_{stage}_{est}{suffix}.txt"
            f.write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
            core = sum(x[0] for x in items) / 3600
            wall = [max(x[0] for x in pk) + STAGGER * (len(pk) - 1) for pk in packs]
            width = max(len(pk) for pk in packs)  # CPUs per task of the array
            alloc = sum(width * w_ for w_ in wall) / 3600
            proj.append(
                dict(
                    est=est,
                    stage=stage,
                    designs=len(names),
                    processes=len(items),
                    packs=len(packs),
                    core_h=round(core, 1),
                    alloc_cpu_h=round(alloc, 1),
                    max_pack_min=round(max(wall) / 60, 1),
                    time_limit_min=int(math.ceil((1.5 * max(wall) + 300) / 300) * 5),
                )
            )
    pj = pd.DataFrame(proj)
    pj.to_csv(
        REPO / "cluster" / f"feature_mcs_trees_projection{suffix}.csv",
        index=False,
        lineterminator="\n",
    )
    say(pj.to_string(index=False))
    say(
        f"total: {pj.core_h.sum():.0f} core-h, {pj.alloc_cpu_h.sum():.0f} allocated CPU-h (packs of <= {PACK_MAX} processes, one CPU each)"
    )


# ---------------------------------------------------------------- gate (CARC)
def gate(est: str) -> int:
    _, oos_rows, dates_oos = load_designs()
    rows, ok = [], True
    for design in ANCHORS:
        ours, info = read_all(est, design, oos_rows, dates_oos)
        if len(ours) != len(oos_rows) or list(ours.index) != oos_rows:
            rows.append(
                dict(
                    est=est,
                    design=design,
                    ok=False,
                    why=f"{len(ours)} of {len(oos_rows)} rows",
                )
            )
            ok = False
            continue
        c = anchor_check(est, design, ours, dates_oos)
        cpus = sorted({i["cpu"] for i in info})
        rows.append(dict(est=est, design=design, cpu=" | ".join(cpus), **c))
        ok &= c["ok"]
    WORK.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(WORK / f"gate_{est}.csv", index=False)
    say(pd.DataFrame(rows).to_string(index=False))
    flag = WORK / f"GATE_OK_{est}"
    if ok:
        flag.write_text(
            f"{ESTS[est]['label']}: baseline and all_features bit-identical to the stored T1 runs on the 866 days "
            "(the stored tables within their CSV parse, <= 2 ulp)\n",
            encoding="utf-8",
        )
    return 0 if ok else 1


# ---------------------------------------------------------------- collect (CARC)
def collect(
    only: tuple[str, ...] = (),
    expect: int = 288,
    flag: str = "TREES_DONE",
    tag: str = "",
    ests: tuple[str, ...] = tuple(ESTS),
) -> int:
    """only = design-name prefixes (default: every design of the first 96 -- the TREES_DONE set); ests = the
    estimators collected (2026-10-13: XGBoost and the forest first, while LightGBM waits for CARC)."""
    designs, oos_rows, dates_oos = load_designs()
    census: list[dict] = []
    status = []
    n_ok = 0
    date_str = np.array(dates_oos[oos_rows].strftime("%Y-%m-%d %H:%M:%S"), dtype="U19")
    for est in ests:
        (OUT / est).mkdir(parents=True, exist_ok=True)
        names = (
            [m for m in designs["order"] if m.startswith(only)]
            if only
            else designs["order"][:96]
        )
        for m in names:
            ours, info = read_all(est, m, oos_rows, dates_oos)
            census += info
            complete = len(ours) == len(oos_rows) and list(ours.index) == oos_rows
            cpus = sorted({i["cpu"] for i in info})
            status.append(
                dict(
                    est=est,
                    model=m,
                    rows=len(ours),
                    complete=complete,
                    chunks=len(info),
                    cpu=" | ".join(cpus),
                )
            )
            if not complete:
                continue
            tmp = OUT / est / f"{m}.tmp.npz"
            np.savez(tmp, pred=ours.to_numpy(np.float64), date=date_str)
            os.replace(tmp, OUT / est / f"{m}.npz")
            js = dict(
                est=est,
                model=m,
                config=ESTS[est]["label"],
                cpu=" | ".join(cpus),
                host=",".join(sorted({i["host"] for i in info})),
                cpu_total=float(np.nansum([i["cpu_sec"] for i in info])),
                wall=float(np.nansum([i["real_sec"] for i in info])),
                chunks=len(info),
                rows=len(ours),
            )
            jt = OUT / est / f"{m}.json.tmp"
            jt.write_text(json.dumps(js), encoding="utf-8")
            os.replace(jt, OUT / est / f"{m}.json")
            n_ok += 1
    pd.DataFrame(census).to_csv(OUT / f"census{tag}.csv", index=False)
    st = pd.DataFrame(status)
    st.to_csv(OUT / f"status{tag}.csv", index=False)
    gates = {e: (WORK / f"GATE_OK_{e}").is_file() for e in ests}
    for e in ESTS:
        if (WORK / f"gate_{e}.csv").is_file():
            (OUT / f"gate_{e}.csv").write_bytes((WORK / f"gate_{e}.csv").read_bytes())
    classes = {e: sorted({i["cpu"] for i in census if i["est"] == e}) for e in ests}
    say(
        f"collect: {n_ok} of {len(st)} (estimator, design) written; gates {gates}; CPU models {classes}"
    )
    if (
        n_ok == expect
        and all(gates.values())
        and all(set(v) <= CPU_OK[e] for e, v in classes.items())
    ):
        cpu_txt = "; ".join(f"{e}: {' | '.join(classes[e])}" for e in ests)
        text = (
            f"{expect} tree candidates ({len(ests)} estimators {', '.join(ests)} x {expect // len(ests)} designs: "
            f"{', '.join(only) or 'the first 96'}), every one complete on the 866 trade days; each estimator's anchors "
            "bit-identical to its stored T1 runs (gate_<est>.csv); every chunk on its estimator's allowed CPU set "
            "(LightGBM: EPYC 7513 only; XGBoost and the forest: EPYC 7513 / 7542 or Hoffman2 AVX E5-2670 / E5-2650 v2, "
            f"byte-identical classes); CPU models used -- {cpu_txt} (census{tag}.csv)\n"
        )
        (OUT / flag).write_text(text, encoding="utf-8")
        return 0
    return 1


def export_buckets(est: str, dest: Path) -> int:
    """2026-10-13 (CARC /scratch1 storage failure: no reads from the login nodes): run INSIDE a batch job, pack every
    bucket-design chunk folder of one estimator (chunks2/<est>/{addb_*,loob_*}) into one tarball on /home1 plus its md5,
    so the collect step can run locally from a login-node read of /home1 only."""
    import hashlib
    import tarfile

    dirs = sorted(
        d
        for d in (WORK / "chunks2" / est).glob("*")
        if d.name.startswith(("addb_", "loob_"))
    )
    n_done = sum(1 for d in dirs for c in d.glob("c*") if (c / "DONE").is_file())
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    with tarfile.open(tmp, "w:gz") as tf:
        for d in dirs:
            tf.add(d, arcname=str(d.relative_to(REPO)))
    os.replace(tmp, dest)
    h = hashlib.md5(dest.read_bytes()).hexdigest()
    dest.with_name(dest.name + ".md5").write_text(
        f"{h}  {dest.name}\n", encoding="utf-8"
    )
    say(
        f"export {est}: {len(dirs)} designs, {n_done} finished chunks -> {dest} (md5 {h})"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "mode",
        choices=(
            "prepare",
            "canary_check",
            "plan",
            "gate",
            "collect",
            "collect_buckets",
            "collect_buckets_xgb_rf",
            "export_buckets",
        ),
    )
    ap.add_argument("--est", choices=tuple(ESTS))
    ap.add_argument("--costs", default=str(WORK / "canary_costs.csv"))
    ap.add_argument(
        "--suffix", default="", help="plan: task / projection file suffix (e.g. _v2)"
    )
    ap.add_argument(
        "--only",
        default="",
        help="plan: comma list of design-name prefixes (e.g. addb_,loob_)",
    )
    ap.add_argument(
        "--designs-only",
        action="store_true",
        help="prepare: rewrite the designs JSON only",
    )
    a = ap.parse_args()
    if a.mode == "prepare":
        prepare(a.designs_only)
        return 0
    if a.mode == "canary_check":
        return canary_check()
    if a.mode == "plan":
        plan(Path(a.costs), a.suffix, tuple(x for x in a.only.split(",") if x))
        return 0
    if a.mode == "gate":
        return gate(a.est)
    if (
        a.mode == "export_buckets"
    ):  # inside a batch job: one estimator's bucket chunks -> a tarball on /home1
        return export_buckets(
            a.est or "lgbm",
            Path(
                os.environ.get(
                    "EXPORT_TO",
                    "/home1/jc_905/feature_mcs_trees_export/buckets_lgbm.tar.gz",
                )
            ),
        )
    if a.mode == "collect_buckets":  # 2026-10-12: the 16 bucket designs -> BUCKETS_DONE
        return collect(("addb_", "loob_"), 48, "BUCKETS_DONE", "_buckets")
    if (
        a.mode == "collect_buckets_xgb_rf"
    ):  # 2026-10-13: XGBoost + forest (Hoffman2) first -> BUCKETS_XGB_RF_DONE
        return collect(
            ("addb_", "loob_"),
            32,
            "BUCKETS_XGB_RF_DONE",
            "_buckets_xgb_rf",
            ("xgb", "rf"),
        )
    return collect()


if __name__ == "__main__":
    sys.exit(main())
