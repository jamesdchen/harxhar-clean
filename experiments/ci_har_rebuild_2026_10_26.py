"""Rebuild of the closing-interval HAR benchmark (CI-HAR, master key sub_ols_baseline) under the current data chain.

RULE  results/ci_har_rebuild_2026-10-26/RULE.txt (pre-registered 2026-10-25 22:49 PDT, before any computation).
OLD   results/linear_subsection_dedup/baseline/bar1600/reclasso/tw2000/causal_tune_linear/incumbent_ols/
      results_bar1600.csv (2026-09-29, Hoffman2; specs/causal_tune_linear.py, OLS-incumbent block).

The model is the spec's incumbent call, verbatim (run_executor with fit_predict_ridge at alpha 0, rank-1 rolling OLS
refitted every session, the empty-exog `baseline` bucket = HAR + calendar, 22 columns, bar1600, global lags, 2,000
sessions).  Only the src data-fix flags change between variants; each variant runs in its own process (module
attributes are set before run_executor is called; every flag is read at call time).

VARIANTS  old      every flag False: the pre-2026-10-15 chain (the gate: must equal OLD bit for bit)
          new      every flag at its current default (True): the rebuilt CI-HAR
          b        old + B_REGULAR_BARS_ONLY
          b_ec     old + B_REGULAR_BARS_ONLY + DROP_EARLY_CLOSE_SESSIONS
          three    old + the three flags that reach this design (B, early-close drop, burn-in in bar space)
          noop     new without those three (must equal old: the other flags do not reach a target-only design)
          old_sandybridge  old, with OpenBLAS forced to its Sandybridge kernel (OPENBLAS_CORETYPE): the size of a
                   BLAS-kernel float-path change on this machine, against which the gate's difference is judged

GATE   bit for bit on the stamps, the horizon, the target (true_adj) and the raw target (true_raw); the forecasts
       (pred_adj, pred_raw) may differ only by a BLAS float path: at most PRED_REL_TOL relative and no more than this
       machine's own change under another OpenBLAS kernel (old_sandybridge vs old).  The old file was computed on
       Hoffman2 (Intel Xeon, cluster/linear_dedup_canary.sh); the rank-1 OLS solves go through BLAS / LAPACK, whose
       kernels differ by CPU.

Run:  python experiments/ci_har_rebuild_2026_10_26.py            (all variants, then the gates; ~1 GB RAM, 1 core)
      python experiments/ci_har_rebuild_2026_10_26.py --variant new   (one variant)
      CI_HAR_OUT=<dir> python experiments/ci_har_rebuild_2026_10_26.py --variant old   (cluster job: outputs to <dir>)
      python experiments/ci_har_rebuild_2026_10_26.py --compare-hoffman2 <dir>   (RULE.txt ADDENDUM: <dir>/runs/old
                  against the old file bit for bit, <dir>/runs/new against the laptop file; <dir>/node.txt is the job's
                  node record)
WRITES results/ci_har_rebuild_2026-10-26/runs/<variant>/results_bar1600.csv (+ feature health, reduce, log),
       results/ci_har_rebuild_2026-10-26/results_bar1600.csv (= runs/new, copied once every gate passes),
       results/ci_har_rebuild_2026-10-26/gates.csv; --compare-hoffman2: <dir>/gates_hoffman2.csv.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"  # as the cluster pack (cluster/linear_dedup_pack.sh)
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("UNIF_SCALE_PROCS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
# CI_HAR_OUT: another output root (the Hoffman2 gate job, RULE.txt ADDENDUM: <deployment>/out); default the repo's
OUT = Path(os.environ.get("CI_HAR_OUT", str(REPO / "results" / "ci_har_rebuild_2026-10-26")))
LAPTOP_OUT = REPO / "results" / "ci_har_rebuild_2026-10-26"
OLD = (
    REPO
    / "results"
    / "linear_subsection_dedup"
    / "baseline"
    / "bar1600"
    / "reclasso"
    / "tw2000"
    / "causal_tune_linear"
    / "incumbent_ols"
    / "results_bar1600.csv"
)
COLS = ["date", "horizon", "true_adj", "pred_adj", "true_raw", "pred_raw"]
PRED_REL_TOL = 1e-13  # a float-path bound on the forecasts (the OLS solves); the target must be bit for bit

# every data-fix flag of src (module, attribute): the 14 of live/close_signal/forecast.py PINNED_AUDIT_FLAGS, the two
# 2026-10-15 flags outside that list (domain rule, name sign) and the two of 2026-10-22
FLAGS: tuple[tuple[str, str], ...] = (
    ("loading", "NULL_INVALID_PRINTS"),
    ("loading", "FOMC_CALENDAR_FIX"),
    ("loading", "SPLICE_VIX_FREE_FEED"),
    ("target", "SIGN_BY_NAME"),
    ("target", "FOMC_FLAGS_IDENTITY"),
    ("target", "CAUSAL_SD_FLOOR"),
    ("target", "CAUSAL_ASINH_ROUTING"),
    ("executor", "APPLY_FFILL_LIMIT"),
    ("executor", "ACTIVE_FROM_OBSERVED"),
    ("executor", "CAUSAL_MODE_INFLATION"),
    ("executor", "CAUSAL_NEUTRAL_FILL"),
    ("executor", "CAUSAL_GUARD_MODE"),
    ("executor", "B_REGULAR_BARS_ONLY"),
    ("executor", "DROP_EARLY_CLOSE_SESSIONS"),
    ("executor", "ERODE_IN_BARS"),
    ("executor", "BURNIN_IN_BAR_SPACE"),
    ("executor", "LATE_START_WARMUP"),
    ("executor", "UNOBSERVED_NEUTRAL"),
)
# model-column switches of the pooled final design: not data rules, kept at their (False) defaults (RULE.txt 1)
MODEL_FLAGS: tuple[str, ...] = (
    "DROP_ACTIVITY_MA",
    "CALENDAR_THROUGH_OVERNIGHT",
    "DROP_OVERNIGHT_INDICATOR",
    "LEAN_GLOBAL_LAGS",
)
BITE = {"B_REGULAR_BARS_ONLY", "DROP_EARLY_CLOSE_SESSIONS", "BURNIN_IN_BAR_SPACE"}
ALL = {a for _, a in FLAGS}
VARIANTS: dict[str, set[str]] = {
    "old": set(),
    "new": set(ALL),
    "b": {"B_REGULAR_BARS_ONLY"},
    "b_ec": {"B_REGULAR_BARS_ONLY", "DROP_EARLY_CLOSE_SESSIONS"},
    "three": set(BITE),
    "noop": ALL - BITE,
    "old_sandybridge": set(),
}
ENV = {"old_sandybridge": {"OPENBLAS_CORETYPE": "Sandybridge"}}


def _modules() -> dict:
    import src.backtest.executor as EX
    import src.data.loading as LD
    import src.features.transforms.target as TG

    return {"loading": LD, "target": TG, "executor": EX}


def run_variant(name: str) -> None:
    """Build one variant in this process: set the flags, then the spec's incumbent call."""
    os.chdir(REPO)  # data_path "data" is relative, as in the spec
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    mods = _modules()
    on = VARIANTS[name]
    settings = {}
    for m, a in FLAGS:
        assert getattr(mods[m], a) is True, f"{m}.{a} is not at its True default"
        setattr(mods[m], a, a in on)
        settings[f"{m}.{a}"] = a in on
    for a in MODEL_FLAGS:
        assert getattr(mods["executor"], a) is False, f"executor.{a} is not at its False default"

    from src.backtest.executor import run_executor
    from src.data.loading import get_bucket
    from src.models.ridge import fit_predict_ridge

    exog = get_bucket("baseline")
    assert exog == [], exog
    d = OUT / "runs" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "flags.json").write_text(json.dumps(settings, indent=1), encoding="utf-8")
    # specs/causal_tune_linear.py, hpc-audit-section baseline: the incumbent call, verbatim (TRAIN_WIN=2000 from the
    # task line, SEGMENT=bar1600, LAG_SCOPE=global, HAR_LAGS=None, START=0, END=-1, HALO=0)
    run_executor(
        method_name="ols_incumbent",
        fit_predict=fit_predict_ridge,
        hyperparams=dict(alpha=0.0, _refit_frequency=1, _incremental=True),
        data_path="data",
        output_file=str(d / "results.csv"),
        horizon=1,
        train_window=2000,
        start=0,
        end=-1,
        halo=0,
        exog_cols=exog,
        segment="bar1600",
        lag_scope="global",
        har_lags=None,
        add_calendar=True,
        target_use_diurnal=True,
        target_winsor_window=240,
        dropna_with_exog=False,
        overnight_fill=True,
        impute_indicate=True,
        diurnal_mode="divide",
        prescale=True,
        seed=42,
    )
    fh = pd.read_csv(d / "results_bar1600_feature_health.csv")
    feats = list(fh["feature"])
    fomc = [c for c in feats if "fomc" in c.lower()]
    if fomc:
        raise SystemExit(f"STOP: FOMC columns in the CI-HAR design: {fomc}")
    print(f"[{name}] design columns ({len(feats)}): {feats}")


def _read(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    assert list(df.columns) == COLS, list(df.columns)
    return df


def gates() -> pd.DataFrame:
    """The gate (old == OLD bit for bit), the no-op check, and the attribution of the change."""
    old_file = _read(OLD)
    runs = {v: _read(OUT / "runs" / v / "results_bar1600.csv") for v in VARIANTS}
    rows = []

    def cmp(label: str, a: pd.DataFrame, b: pd.DataFrame) -> dict:
        same_dates = len(a) == len(b) and bool((a["date"].values == b["date"].values).all())
        r: dict = {"check": label, "rows_a": len(a), "rows_b": len(b), "same_dates": same_dates}
        if same_dates:
            for c in COLS[1:]:
                x, y = a[c].to_numpy(np.float64), b[c].to_numpy(np.float64)
                r[f"bitequal_{c}"] = bool(np.array_equal(x, y))
                r[f"maxabs_{c}"] = float(np.max(np.abs(x - y)))
                r[f"maxrel_{c}"] = float(np.max(np.abs(x / y - 1.0))) if c != "horizon" else 0.0
        else:  # on the common stamps
            m = a.merge(b, on="date", suffixes=("_a", "_b"))
            r["common_dates"] = len(m)
            for c in COLS[2:]:
                x, y = m[f"{c}_a"].to_numpy(np.float64), m[f"{c}_b"].to_numpy(np.float64)
                r[f"bitequal_{c}"] = bool(np.array_equal(x, y))
                r[f"maxabs_{c}"] = float(np.max(np.abs(x - y)))
                r[f"maxrel_{c}"] = float(np.max(np.abs(x / y - 1.0)))
        r["pass"] = same_dates and all(r.get(f"bitequal_{c}", False) for c in COLS[1:])
        return r

    rows.append(cmp("GATE old-rules rebuild == old file (bit for bit)", runs["old"], old_file))
    rows.append(cmp("NO-OP flags-on-except-three == old-rules rebuild", runs["noop"], runs["old"]))
    rows.append(cmp("three-on == all-on (new)", runs["three"], runs["new"]))
    rows.append(cmp("KERNEL old rules, Sandybridge vs default OpenBLAS kernel", runs["old_sandybridge"], runs["old"]))
    rows.append(cmp("KERNEL old rules, Sandybridge kernel vs old file", runs["old_sandybridge"], old_file))
    rows.append(cmp("new vs old file (information)", runs["new"], old_file))
    rows.append(cmp("b vs old file (information)", runs["b"], old_file))
    rows.append(cmp("b_ec vs old file (information)", runs["b_ec"], old_file))
    g = pd.DataFrame(rows)
    g.to_csv(OUT / "gates.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(g[["check", "rows_a", "rows_b", "same_dates", "pass"]].to_string(index=False))
    return g


def compare_hoffman2(d: Path) -> pd.DataFrame:
    """RULE.txt ADDENDUM: the Hoffman2 old-rules run against the old file (bit for bit, every column), the Hoffman2
    new-rules run against the laptop file (max differences), and, for the new file, whether the difference reaches
    the printed precision: QLIKE (4 dp) and MSE on T_oos and on the 866 master days, and the trade sign on the 866
    days (q = +1 if plain > iv_var: every Sharpe ratio and its test are functions of q)."""
    old_file, laptop = _read(OLD), _read(LAPTOP_OUT / "results_bar1600.csv")
    h_old, h_new = _read(d / "runs" / "old" / "results_bar1600.csv"), _read(d / "runs" / "new" / "results_bar1600.csv")
    rows = []
    for label, a, b in (
        ("GATE hoffman2 old-rules == old file (bit for bit)", h_old, old_file),
        ("CPU hoffman2 new-rules vs laptop new file", h_new, laptop),
        (
            "information: hoffman2 old-rules vs laptop old-rules",
            h_old,
            _read(LAPTOP_OUT / "runs" / "old" / "results_bar1600.csv"),
        ),
    ):
        same = len(a) == len(b) and bool((a["date"].values == b["date"].values).all())
        r: dict = {"check": label, "rows": len(a), "same_dates": same}
        for c in COLS[1:]:
            x, y = a[c].to_numpy(np.float64), b[c].to_numpy(np.float64)
            r[f"bitequal_{c}"] = same and bool(np.array_equal(x, y))
            r[f"rows_differ_{c}"] = int((x != y).sum()) if same else -1
            r[f"maxabs_{c}"] = float(np.max(np.abs(x - y))) if same else float("nan")
            r[f"maxrel_{c}"] = float(np.max(np.abs(x / y - 1.0))) if same and c != "horizon" else 0.0
        r["pass"] = same and all(r[f"bitequal_{c}"] for c in COLS[1:])
        rows.append(r)
    # the new file at the printed precision (plain = pred_adj^2 B, B = true_raw / true_adj^2; truth and spans as
    # experiments/rolling_eval_2026_10_15.py)
    sys.path.insert(0, str(REPO / "experiments"))
    sys.path.insert(0, str(REPO / "notebooks"))
    import rolling_eval_2026_10_15 as rolling

    def plain(f: pd.DataFrame) -> pd.Series:
        g = f.assign(day=pd.to_datetime(f["date"]).dt.normalize()).set_index("day")
        g = g[(g["true_adj"] > 0) & (g["true_raw"] > 0)]
        return g["pred_adj"] ** 2 * (g["true_raw"] / g["true_adj"] ** 2)

    p_h, p_l = plain(h_new), plain(laptop)
    hist = rolling.history()
    mtd = pd.read_parquet(
        REPO / "results" / "close_master_table" / "master_table_daily.parquet", columns=["key", "day", "iv_var"]
    )
    mtd = mtd[mtd["key"] == rolling.OLS].assign(day=lambda x: pd.to_datetime(x["day"]).dt.normalize()).set_index("day")
    prec: dict = {"check": "new file at the printed precision (hoffman2 vs laptop)"}
    for nm, days in (("T_oos", rolling.common_span()), ("mt866", pd.DatetimeIndex(mtd.index))):
        rv = hist.reindex(days).to_numpy(float)
        for lab, p in (("hoffman2", p_h), ("laptop", p_l)):
            v = p.reindex(days).to_numpy(float)
            prec[f"qlike_{nm}_{lab}"] = float(rolling.qlike_daily(rv, v).mean())
            prec[f"mse_{nm}_{lab}"] = float(((rv - v) ** 2).mean())
        prec[f"qlike_{nm}_same_4dp"] = f"{prec[f'qlike_{nm}_hoffman2']:.4f}" == f"{prec[f'qlike_{nm}_laptop']:.4f}"
    iv = mtd["iv_var"].to_numpy(float)
    q_h = np.where(p_h.reindex(mtd.index).to_numpy(float) > iv, 1, -1)
    q_l = np.where(p_l.reindex(mtd.index).to_numpy(float) > iv, 1, -1)
    prec["trade_signs_differ_866"] = int((q_h != q_l).sum())
    prec["min_abs_log_plain_over_iv_866"] = float(np.min(np.abs(np.log(p_l.reindex(mtd.index).to_numpy(float) / iv))))
    rows.append(prec)
    out = pd.DataFrame(rows)
    node = d / "node.txt"
    if node.is_file():
        print(node.read_text(encoding="utf-8", errors="replace"))
    out.to_csv(d / "gates_hoffman2.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 60):
        print(out.T.to_string())
    print(f"gate {'CLOSED (bit for bit)' if bool(out.loc[0, 'pass']) else 'NOT bit for bit'}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(VARIANTS), default=None)
    ap.add_argument("--gates-only", action="store_true")
    ap.add_argument("--compare-hoffman2", type=Path, default=None)
    args = ap.parse_args()
    if args.variant:
        run_variant(args.variant)
        return
    if args.compare_hoffman2 is not None:
        compare_hoffman2(args.compare_hoffman2)
        return
    if not args.gates_only:
        for v in ("old", "new", "b", "b_ec", "three", "noop", "old_sandybridge"):
            if (OUT / "runs" / v / "results_bar1600.csv").exists():
                print(f"[{v}] exists, skipped")
                continue
            log = OUT / "runs" / v / "run.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with open(log, "w", encoding="utf-8") as fh:
                subprocess.run(
                    [sys.executable, "-u", __file__, "--variant", v],
                    check=True,
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                    cwd=REPO,
                    env={**os.environ, **ENV.get(v, {})},
                )
            print(f"[{v}] done")
    g = gates()
    gr, kr = g.iloc[0], g.iloc[3]
    exact = bool(gr["same_dates"]) and all(bool(gr[f"bitequal_{c}"]) for c in ("horizon", "true_adj", "true_raw"))
    pred_rel = max(float(gr["maxrel_pred_adj"]), float(gr["maxrel_pred_raw"]))
    kernel_rel = max(float(kr["maxrel_pred_adj"]), float(kr["maxrel_pred_raw"]))
    ok_gate = bool(gr["pass"]) or (exact and pred_rel <= PRED_REL_TOL and pred_rel <= kernel_rel)
    print(
        f"gate: stamps, target bit for bit {exact}; forecasts bit for bit {bool(gr['pass'])}, max rel {pred_rel:.2e} "
        f"(this machine under another BLAS kernel: {kernel_rel:.2e}; bound {PRED_REL_TOL:.0e}) -> "
        f"{'PASS' if ok_gate else 'FAIL'}"
    )
    ok_noop = bool(g.loc[1, "pass"]) and bool(g.loc[2, "pass"])
    print(f"no-op check {'PASS' if ok_noop else 'FAIL'}")
    if ok_gate and ok_noop:
        shutil.copy2(OUT / "runs" / "new" / "results_bar1600.csv", OUT / "results_bar1600.csv")
        print(f"wrote {OUT / 'results_bar1600.csv'}")


if __name__ == "__main__":
    main()
