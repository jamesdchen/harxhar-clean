"""Score the per-bar TREE arms with causal periodic hyperparameter tuning.

What is scored.  ``specs/causal_tune_trees_tuned.py`` fits LightGBM / XGBoost / a
random forest on exactly the per-bar design of the untuned campaign
(``specs/causal_tune_trees.py``), but at every tuning point it first picks a
configuration from a grid on a causal validation window.  Two selection rules
run on the same candidates:

  mse     validation MSE picks the config -- results_<bar>.csv, the arm of record
          (same layout and format as the untuned campaign)
  qlike   validation QLIKE picks it -- results_qsel_<bar>.csv, scored here as the
          model "<model>_qsel"

Every forecast is rebuilt with the ONE causal back-transform of
``score_trees_subsection.py`` (clock: s = mean squared adjusted-scale error of the
arm's own previous 250 sessions at that bar, lagged one session), whose functions
are reused here unchanged -- a difference between two arms is the forecast, not
the calibration.  Step 1 runs that scorer itself on the tuned root.

Tables (``--out``, default = ``--root``):
  written by score_trees_subsection.main on the tuned root (MSE rule), unchanged:
    trees_qlike_by_bar.csv, trees_vs_linear.csv, trees_vs_baseline_tree.csv,
    trees_join_gates.csv, trees_trade_1530.csv, trees_shap_*_<bar>.csv,
    trees_importance_<bar>.csv
  the QLIKE rule:
    trees_qsel_qlike_by_bar.csv    as trees_qlike_by_bar, model <model>_qsel ("common" =
                                   stamps every tree arm of either rule shares at the bar)
    trees_qsel_vs_linear.csv       as trees_vs_linear, for the QLIKE-selected path
    trees_qsel_vs_mse.csv          QLIKE-selected against MSE-selected path of the same arm,
                                   identical stamps (negative diff: the QLIKE rule wins)
    trees_qsel_trade_1530.csv      the deck's 15:30 sign(s) trade on each bar1600
                                   QLIKE-selected path (own stamps)
  tuned against untuned:
    trees_tuned_vs_untuned.csv     each tuned arm (both rules) against the untuned arm of
                                   the same model / bucket / bar / window (--untuned-root)
                                   on identical stamps: clock and as-scored back-transform,
                                   common and deck period, day-block 95 % interval, DM t
                                   (negative diff: tuning wins)
    trees_tuned_trade_1530.csv     the 15:30 trade on the SAME days (the tuned arm's
                                   stamps where every forecast exists) for tuned <model>,
                                   tuned <model> qsel, untuned <model>, linear <est>
  the hyperparameters:
    trees_tuned_trace.csv          every tune_trace file: the chosen config over time,
                                   both rules
    trees_tuned_edges.csv          per (model, bucket, rule, axis): tuning points, picks at
                                   the grid's lo / hi edge (count, share), whether that
                                   edge is a natural parameter bound (grid_<bar>.json;
                                   without one, the trace rows that picked it), and
                                   edge_is_grid_choice_pick_share = picks at an edge that
                                   is NOT a natural bound (the grid was too narrow there);
                                   pseudo-axis rounds_cap (lgbm / xgb): the boosting-round
                                   cap bound (a budget choice, never natural)
    trees_tuned_shipped.csv        per (model, bucket, bar, rule): share of tuning points
                                   at which the shipped (untuned) config was chosen, its
                                   mean rank among the candidates by val_mse and val_qlike
                                   (1 = best), and the in-selection gain mean(shipped /
                                   chosen - 1) -- OPTIMISTIC BY CONSTRUCTION: the chosen
                                   config is the minimum over the candidates on the very
                                   validation rows the gain is measured on
    trees_tuned_join_gates.csv     every gate of this script (below)

Gates (a failed gate is printed, the pair or file is left out, every table is still
written, the exit status is non-zero):
  qsel vs mse csv        the qsel CSV carries the MSE CSV's stamps (same order) and
                         true_raw / true_adj to 1e-9 relative (same run, same rows)
  npz pred_adj_qsel      the npz's pred_adj_qsel is the qsel CSV's pred_adj
  true_raw a vs b        every paired join (qsel vs linear, qsel vs mse, tuned vs
                         untuned): the target agrees to 1e-9 relative
  tune files' keys       model / bucket / segment / train_win in the trace and candidate
                         files are the arm's path
  trace stamps           forecast_date is a naive (ET, bar-end) stamp and equals the
                         results CSV's date at tune_row (tune_row = OOS row index, as the
                         npz refit_row)
  trace vs npz           per rule, the trace's tune_row and cand_idx are the npz's
                         tune_row and chosen_mse / chosen_qlike
  trace vs candidates    per rule, the chosen candidate attains the minimum of its rule's
                         validation loss at that tuning point, the trace's val loss is the
                         candidate's, the candidates' chosen_<rule> flag marks the trace's
                         cand_idx, and the trace's shipped_val_* / is_shipped agree with the
                         candidate flagged is_shipped
  edge labels vs grid    against grid_<bar>.json (the full ascending axes and their natural
                         bounds; the candidates are a fixed SUBSET of the grid, so their
                         range is not the grid's): every chosen and candidate value is on
                         its axis, <axis>_edge is 'lo' / 'hi' exactly when the chosen value
                         is the axis's first / last value, and <axis>_edge_natural is the
                         grid file's flag for that edge (unbounded max_depth "None" sorts
                         above every finite depth); no grid file -> not checked (a note)
Step 1's own gates are in trees_join_gates.csv; its failure is carried to the exit status.

Robust to a partially finished campaign (the cluster scorer runs afterany): missing
arms, qsel CSVs, npz, trace / candidate files, untuned twins or linear comparators
are skipped with a printed note.

Usage:  python experiments/score_trees_tuned.py
            [--root results/linear_subsection_trees_tuned]
            [--untuned-root results/linear_subsection_trees] [--linear-root DIR ...]
            [--tw 2000] [--out DIR] [--min-n 100] [--detail-bars bar1600]
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "notebooks", ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import build_subsection_tree_yhat as bst  # noqa: E402
import score_linear_subsection as base  # noqa: E402
import score_trees_subsection as sts  # noqa: E402

DEFAULT_ROOT = "results/linear_subsection_trees_tuned"
DEFAULT_UNTUNED_ROOT = sts.DEFAULT_ROOT  # results/linear_subsection_trees, read-only
QSEL = "_qsel"  # the spec's file infix (results_qsel_<bar>.csv) and our model suffix
RULES = ("mse", "qlike")  # tune_trace 'rule' values; mse selects the arm of record
RULE_METRIC = {"mse": "val_mse", "qlike": "val_qlike"}  # the loss each rule minimises
RULE_FLAG = {"mse": "chosen_mse", "qlike": "chosen_qlike"}  # npz array / candidate flag
BOOSTED = ("lgbm", "xgb")  # early-stopped models: the boosting-round cap can bind
# the tuned axes the spec is written to (a trace with other axes is scored as it is,
# with a printed note -- the axes are read from the trace's <axis>_edge_natural columns)
EXPECTED_AXES = {
    "lgbm": ("num_leaves", "min_child_samples", "feature_fraction", "lambda_l2"),
    "xgb": (
        "max_depth",
        "min_child_weight",
        "subsample",
        "colsample_bytree",
        "reg_lambda",
    ),
    "rf": ("max_features", "min_samples_leaf", "max_depth"),
}
EDGE_SUFFIX, NATURAL_SUFFIX = "_edge", "_edge_natural"
EDGES = ("lo", "hi")
# the spec writes an unbounded RF max_depth as this string; unbounded is deeper than
# every finite depth, so it is ordered as +inf when the edge labels are checked
UNBOUNDED = "None"
ROUNDS_CAP_AXIS = "rounds_cap"  # pseudo-axis: rounds_cap_hit, a budget choice
# the deck's 15:30 trade forecasts the bar ending 16:00 (base.trade_1530 reads that stamp)
TRADE_SEG, TRADE_HHMM = "bar1600", "16:00"
# 1e-9 relative: same design, same target (the float-text round trip errs by ~1e-16)
GATE_REL = sts.GATE_REL
NPZ_ABS = 1e-12  # sts.npz_gate's bound: npz and CSV are written from one array
MIN_DECK_DAYS = sts.MIN_DECK_DAYS  # the untuned scorer's minimum for a 15:30 trade row
REF_EST = sts.REF_EST  # the summary's reference linear estimator (ridge)
# step 1's tables read back for the summary -- only when this run rewrote them
UNTUNED_OUTPUTS = (
    "trees_qlike_by_bar.csv",
    "trees_vs_linear.csv",
    "trees_join_gates.csv",
)


# ---------------------------------------------------------------- helpers
def bar_end(seg: str) -> str:
    return f"{seg[3:5]}:{seg[5:7]}"


def arm_keys(arm: bst.TreeArm, model: str | None = None) -> dict:
    return {
        "bucket": arm.bucket,
        "model": model or arm.model,
        "train_win": arm.tw,
        "segment": arm.seg,
    }


def read_table(path: Path) -> pd.DataFrame:
    """A tune_trace / tune_candidates CSV; "None" stays a string (pandas >= 2 reads it as NaN)."""
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin(("true", "1", "1.0"))


def axis_value(s: pd.Series) -> pd.Series:
    """Numeric value of a hyperparameter column; UNBOUNDED -> +inf, non-numeric -> NaN."""
    t = s.astype(str).str.strip()
    return pd.to_numeric(t, errors="coerce").mask(t == UNBOUNDED, np.inf)


def max_rel_gap(a, b) -> float:
    """Max |a / b - 1| over both-finite rows; inf when the NaN or zero patterns differ."""
    x, y = np.asarray(a, float), np.asarray(b, float)
    if x.shape != y.shape or (np.isnan(x) ^ np.isnan(y)).any():
        return float("inf")
    if ((y == 0) & (x != 0)).any():
        return float("inf")
    m = np.isfinite(x) & np.isfinite(y) & (y != 0)
    return float(np.abs(x[m] / y[m] - 1.0).max()) if m.any() else 0.0


def max_abs_gap(a, b) -> float:
    x, y = np.asarray(a, float), np.asarray(b, float)
    if x.shape != y.shape or (np.isnan(x) ^ np.isnan(y)).any():
        return float("inf")
    m = ~np.isnan(x)
    return float(np.abs(x[m] - y[m]).max()) if m.any() else 0.0


def gate(
    gates: list[dict],
    keys: dict,
    check: str,
    n: int,
    ok: bool,
    max_rel: float = float("nan"),
    detail: str = "",
) -> bool:
    gates.append(
        keys
        | {"check": check, "n": n, "max_rel": max_rel, "ok": bool(ok), "detail": detail}
    )
    if not ok:
        print(f"GATE FAIL {keys} {check}: {detail or f'gap {max_rel:.2e}'}")
    return bool(ok)


def file_stamp(p: Path) -> tuple[int, int] | None:
    try:
        st = p.stat()
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size


# ---------------------------------------------------------------- step 1
def run_untuned_scorer(
    a: argparse.Namespace, root: Path, out: Path
) -> tuple[str | None, dict[str, pd.DataFrame]]:
    """score_trees_subsection.main on the tuned root; its exit is caught and remembered.

    Returns (failure message or None, {table: frame} for the UNTUNED_OUTPUTS this run
    rewrote -- a file left from an earlier run is not read, whatever its content).
    """
    before = {n: file_stamp(out / n) for n in UNTUNED_OUTPUTS}
    argv = [str(Path(sts.__file__)), "--root", str(root), "--tw", str(a.tw)]
    argv += ["--out", str(out), "--min-n", str(a.min_n), "--detail-bars", a.detail_bars]
    for lr in a.linear_root or []:
        argv += ["--linear-root", lr]
    saved, failure = sys.argv, None
    try:
        sys.argv = argv
        sts.main()
    except SystemExit as e:
        if e.code not in (None, 0):
            failure = f"score_trees_subsection: {e.code}"
    except Exception as e:  # noqa: BLE001 -- the tuned tables are still wanted; exit is non-zero
        traceback.print_exc()
        failure = f"score_trees_subsection raised {type(e).__name__}: {e}"
    finally:
        sys.argv = saved
    fresh: dict[str, pd.DataFrame] = {}
    for n in UNTUNED_OUTPUTS:
        now = file_stamp(out / n)
        if now is not None and now != before[n]:
            fresh[n] = pd.read_csv(out / n)
        elif now is not None:
            print(f"note: {out / n} was not rewritten by this run; not read")
    return failure, fresh


# ---------------------------------------------------------------- step 2 gates
def qsel_gates(
    arm: bst.TreeArm, qpath: Path, z: dict | None, gates: list[dict]
) -> bool:
    """The qsel CSV is the MSE CSV's rows; the npz's pred_adj_qsel is the qsel CSV's."""
    k = arm_keys(arm, arm.model + QSEL)
    try:
        m_raw, q_raw = pd.read_csv(arm.csv), pd.read_csv(qpath)
    except Exception as e:  # noqa: BLE001 -- a half-written file is skipped
        print(f"skip {qpath}: {type(e).__name__}: {e}")
        return False
    qd = q_raw["date"].astype(str).to_numpy()
    same = len(m_raw) == len(q_raw) and bool(
        (m_raw["date"].astype(str).to_numpy() == qd).all()
    )
    gap = (
        max(
            max_rel_gap(q_raw["true_raw"], m_raw["true_raw"]),
            max_rel_gap(q_raw["true_adj"], m_raw["true_adj"]),
        )
        if same
        else float("nan")
    )
    ok = gate(
        gates,
        k,
        "qsel csv vs mse csv (dates, true_raw, true_adj)",
        len(q_raw),
        same and gap < GATE_REL,
        gap,
        "" if same else f"stamps differ ({len(q_raw)} vs {len(m_raw)} rows)",
    )
    if z is None:
        return ok
    if "pred_adj_qsel" not in z:
        print(f"note: npz of {arm.key} has no pred_adj_qsel")
        return ok
    dz = np.asarray(z["date"]).astype(str)
    same_z = len(dz) == len(qd) and bool((dz == qd).all())
    g = max_abs_gap(z["pred_adj_qsel"], q_raw["pred_adj"]) if same_z else float("nan")
    gate(
        gates,
        k,
        "npz pred_adj_qsel vs qsel csv (dates, pred_adj)",
        len(q_raw),
        same_z and g < NPZ_ABS,
        g,
        "" if same_z else "npz dates are not the qsel CSV's",
    )
    return ok


# ---------------------------------------------------------------- step 4: tune files
def same_value(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Equal hyperparameter values (the trace and the candidates print the same floats)."""
    return np.isclose(x, y, rtol=GATE_REL, atol=0.0)


def trace_axes(tr: pd.DataFrame) -> list[str]:
    return [
        c[: -len(NATURAL_SUFFIX)]
        for c in tr.columns
        if c.endswith(NATURAL_SUFFIX) and c[: -len(NATURAL_SUFFIX)] in tr.columns
    ]


def check_tune_files(
    arm: bst.TreeArm,
    tr: pd.DataFrame,
    cd: pd.DataFrame | None,
    z: dict | None,
    gates: list[dict],
) -> None:
    keys = arm_keys(arm)
    # a. the files' own keys are the arm's path
    bad = []
    for name, t in (("trace", tr), ("candidates", cd)):
        if t is None:
            continue
        for col, want in (
            ("model", arm.model),
            ("bucket", arm.bucket),
            ("segment", arm.seg),
            ("train_win", arm.tw),
        ):
            if col not in t:
                continue
            same = (
                pd.to_numeric(t[col], errors="coerce") == want  # 2000 or 2000.0
                if isinstance(want, int)
                else t[col].astype(str) == want
            )
            if not same.all():
                bad.append(f"{name}.{col}")
    gate(
        gates,
        keys,
        "tune files' keys vs arm path",
        len(tr),
        not bad,
        detail=", ".join(bad),
    )

    # b. forecast_date: naive ET, the results CSV's date at tune_row
    dates = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(arm.csv)["date"]))
    detail = ""
    try:
        fd = pd.to_datetime(tr["forecast_date"])
    except Exception as e:  # noqa: BLE001 -- mixed offsets or junk
        fd, detail = None, f"forecast_date unparseable ({type(e).__name__})"
    if fd is not None and getattr(fd.dt, "tz", None) is not None:
        detail = f"forecast_date is tz-aware ({fd.dt.tz}); panel stamps are naive ET"
    elif fd is not None:
        tri = pd.to_numeric(tr["tune_row"], errors="coerce").to_numpy()
        inr = np.isfinite(tri) & (tri >= 0) & (tri < len(dates))
        at = np.full(len(tr), np.datetime64("NaT"), "datetime64[ns]")
        at[inr] = dates.to_numpy()[tri[inr].astype(int)]
        miss = int((~fd.isin(dates)).sum())
        off = int((fd.to_numpy() != at).sum())
        if miss or off:
            detail = f"{miss} forecast_date not in the results stamps, {off} != date[tune_row]"
    gate(
        gates,
        keys,
        "trace forecast_date = results date[tune_row] (naive ET)",
        len(tr),
        not detail,
        detail=detail,
    )

    # c. the trace is the npz's tuning record
    if z is not None and "tune_row" in z:
        bad = []
        zr = np.asarray(z["tune_row"], dtype=np.int64)
        for rule in RULES:
            t = tr[tr["rule"] == rule].sort_values("tune_idx")
            if t["tune_idx"].duplicated().any():
                bad.append(f"{rule}: duplicated tune_idx")
            if not np.array_equal(t["tune_row"].to_numpy(np.int64), zr):
                bad.append(f"{rule}: tune_row")
            flag = RULE_FLAG[rule]
            if flag in z and not np.array_equal(
                t["cand_idx"].to_numpy(np.int64), np.asarray(z[flag], dtype=np.int64)
            ):
                bad.append(f"{rule}: cand_idx vs npz {flag}")
        gate(
            gates,
            keys,
            "trace vs npz (tune_row, chosen)",
            len(tr),
            not bad,
            detail="; ".join(bad),
        )

    if cd is None:
        return
    # d. each rule's pick is the minimum of its loss; the files agree with each other
    cidx = cd.set_index(["tune_idx", "cand_idx"])
    if cidx.index.duplicated().any():
        n_dup = int(cidx.index.duplicated().sum())
        gate(
            gates,
            keys,
            "trace vs candidates",
            len(cd),
            False,
            detail=f"{n_dup} duplicated (tune_idx, cand_idx)",
        )
        return
    shipped_c = (
        cd[as_bool(cd["is_shipped"])].drop_duplicates("tune_idx").set_index("tune_idx")
        if "is_shipped" in cd
        else None
    )
    for rule in RULES:
        metric, flag = RULE_METRIC[rule], RULE_FLAG[rule]
        t = tr[tr["rule"] == rule].set_index("tune_idx")
        bad = [] if len(t) else [f"no trace rows for rule {rule}"]
        pick = cidx[metric].reindex(list(zip(t.index, t["cand_idx"]))).to_numpy(float)
        best = cd.groupby("tune_idx")[metric].min().reindex(t.index).to_numpy(float)
        # a NaN pick (no such candidate, or a failed fit) counts as not the minimum
        not_min = ~(pick <= best * (1.0 + GATE_REL))
        if not_min.any():
            bad.append(f"{int(not_min.sum())} picks not the {metric} minimum")
        gv = max_rel_gap(t[metric], pick) if metric in t else float("nan")
        if gv >= GATE_REL:
            bad.append(f"trace {metric} vs candidate's: gap {gv:.1e}")
        if flag in cd:
            fl = cd[as_bool(cd[flag])].groupby("tune_idx")["cand_idx"]
            if (fl.size() != 1).any() or not np.array_equal(
                fl.first().reindex(t.index).to_numpy(float),
                t["cand_idx"].to_numpy(float),
            ):
                bad.append(f"candidates' {flag} flag is not the trace's cand_idx")
        if shipped_c is not None and "is_shipped" in t:
            want = shipped_c["cand_idx"].reindex(t.index).to_numpy(float) == t[
                "cand_idx"
            ].to_numpy(float)
            if (as_bool(t["is_shipped"]).to_numpy() != want).any():
                bad.append("is_shipped vs candidates' shipped cand_idx")
            for m in ("val_mse", "val_qlike"):
                sc = f"shipped_{m}"
                if sc in t and m in shipped_c:
                    have = shipped_c[m].reindex(t.index)
                    ok_rows = have.notna().to_numpy()
                    g = max_rel_gap(
                        t[sc].to_numpy(float)[ok_rows], have.to_numpy(float)[ok_rows]
                    )
                    if g >= GATE_REL:
                        bad.append(f"{sc} vs shipped candidate's: gap {g:.1e}")
        gate(
            gates,
            keys | {"rule": rule},
            "trace vs candidates",
            len(t),
            not bad,
            detail="; ".join(bad),
        )


def check_edges(
    arm: bst.TreeArm, tr: pd.DataFrame, cd: pd.DataFrame | None, gates: list[dict]
) -> dict:
    """Edge labels and natural flags of the trace against grid_<bar>.json.

    Returns the grid file's natural bounds ({axis: {"lo": bool, "hi": bool}}), {} without one."""
    gp = arm.csv.with_name(f"grid_{arm.seg}.json")
    if not gp.is_file():
        print(f"note: no {gp.name} for {arm.key}; edge labels not checked")
        return {}
    grid = json.loads(gp.read_text(encoding="utf-8"))
    axes_g, nat_g = grid.get("axes", {}), grid.get("natural_bounds", {})
    bad, checked = [], 0
    for ax in trace_axes(tr):
        if ax not in axes_g:
            bad.append(f"{ax}: not an axis of {gp.name}")
            continue
        gv = axis_value(pd.Series(axes_g[ax], dtype=object)).to_numpy(float)
        v = axis_value(tr[ax]).to_numpy(float)

        def on_grid(x: np.ndarray, gv: np.ndarray = gv) -> np.ndarray:
            return np.isclose(x[:, None], gv[None, :], rtol=GATE_REL, atol=0.0).any(1)

        n_off = int((~on_grid(v)).sum())
        if cd is not None and ax in cd:
            n_off += int((~on_grid(axis_value(cd[ax]).to_numpy(float))).sum())
        if len(gv) < 2:  # a one-value axis has no edge (the spec's edge_of)
            want = np.full(len(v), "", dtype=object)
        else:
            want = np.where(
                same_value(v, gv[0]), "lo", np.where(same_value(v, gv[-1]), "hi", "")
            )
        nat = nat_g.get(ax, {})
        want_nat = np.array([bool(e and nat.get(e, False)) for e in want], dtype=bool)
        got = tr[ax + EDGE_SUFFIX].fillna("").astype(str).str.strip().to_numpy()
        got_nat = as_bool(tr[ax + NATURAL_SUFFIX]).to_numpy()
        n_edge, n_nat = int((want != got).sum()), int((want_nat != got_nat).sum())
        checked += len(v)
        if n_off or n_edge or n_nat:
            bad.append(
                f"{ax}: {n_off} values off the grid, {n_edge} edge labels and "
                f"{n_nat} natural flags differ (of {len(v)})"
            )
    gate(
        gates,
        arm_keys(arm),
        "edge labels vs grid file",
        checked,
        not bad,
        detail="; ".join(bad),
    )
    return nat_g


def edge_long(arm: bst.TreeArm, tr: pd.DataFrame) -> pd.DataFrame:
    """One row per (tuning point, rule, axis): the chosen value's edge and its naturalness."""
    parts = []
    axes = trace_axes(tr)
    want = EXPECTED_AXES.get(arm.model)
    if want is not None and set(axes) != set(want):
        print(
            f"note: {arm.key} trace axes {sorted(axes)}; the spec's are {sorted(want)}"
        )
    base_cols = {"model": arm.model, "bucket": arm.bucket, "segment": arm.seg}
    for ax in axes:
        parts.append(
            pd.DataFrame(
                base_cols
                | {
                    "rule": tr["rule"].to_numpy(),
                    "axis": ax,
                    "edge": tr[ax + EDGE_SUFFIX]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .to_numpy(),
                    "natural": as_bool(tr[ax + NATURAL_SUFFIX]).to_numpy(),
                }
            )
        )
    if arm.model in BOOSTED and "rounds_cap_hit" in tr:
        hit = as_bool(tr["rounds_cap_hit"]).to_numpy()
        parts.append(
            pd.DataFrame(
                base_cols
                | {
                    "rule": tr["rule"].to_numpy(),
                    "axis": ROUNDS_CAP_AXIS,
                    "edge": np.where(hit, "hi", ""),
                    "natural": False,  # the cap is a budget the spec chose
                }
            )
        )
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def edge_table(long: pd.DataFrame, grid_nat: dict[tuple, dict]) -> pd.DataFrame:
    """Per (model, bucket, rule, axis); an edge's naturalness from the grid file when
    one was read for that model and bucket, else from the trace rows that picked it."""

    def natural_of(s: pd.Series) -> object:
        if s.empty:
            return ""  # never picked and no grid file: unknown
        return bool(s.iloc[0]) if s.nunique() == 1 else "mixed"

    rows = []
    for (model, bucket, rule, ax), g in long.groupby(
        ["model", "bucket", "rule", "axis"], sort=True
    ):
        n = len(g)
        row = {
            "model": model,
            "bucket": bucket,
            "rule": rule,
            "axis": ax,
            "bars": g["segment"].nunique(),
            "n_tuning_points": n,
        }
        nat = grid_nat.get((model, bucket), {}).get(ax)
        for e in EDGES:
            at = g["edge"] == e
            row |= {
                f"n_{e}": int(at.sum()),
                f"share_{e}": float(at.mean()),
                f"{e}_edge_natural": bool(nat[e])
                if nat is not None and e in nat
                else natural_of(g.loc[at, "natural"]),
            }
        grid = g["edge"].isin(EDGES) & ~g["natural"].astype(bool)
        row |= {
            "n_grid_choice_edge": int(grid.sum()),
            "edge_is_grid_choice_pick_share": float(grid.mean()),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def shipped_rows(
    arm: bst.TreeArm, tr: pd.DataFrame, cd: pd.DataFrame | None
) -> list[dict]:
    rows = []
    for rule in RULES:
        t = tr[tr["rule"] == rule]
        if t.empty:
            continue
        row = arm_keys(arm) | {"bar_end": bar_end(arm.seg), "rule": rule}
        row["n_tuning_points"] = len(t)
        row["n_candidates_mean"] = (
            float(pd.to_numeric(t["n_candidates"], errors="coerce").mean())
            if "n_candidates" in t
            else np.nan
        )
        row["share_shipped_chosen"] = (
            float(as_bool(t["is_shipped"]).mean()) if "is_shipped" in t else np.nan
        )
        for m in ("val_mse", "val_qlike"):
            sc = f"shipped_{m}"
            rank = np.nan
            if cd is not None and sc in t and m in cd:
                j = cd[["tune_idx", m]].merge(t[["tune_idx", sc]], on="tune_idx")
                v, s = j[m].to_numpy(float), j[sc].to_numpy(float)
                better = (v < s) & ~np.isclose(v, s, rtol=GATE_REL, atol=0.0)
                # rank 1 = best; a candidate tied with the shipped config does not push it down
                rank = float(
                    (
                        1 + pd.Series(better).groupby(j["tune_idx"].to_numpy()).sum()
                    ).mean()
                )
            row[f"mean_rank_shipped_{m}"] = rank
            gain = (
                float((t[sc].astype(float) / t[m].astype(float) - 1.0).mean())
                if sc in t and m in t
                else np.nan
            )
            row[f"in_selection_gain_{m.removeprefix('val_')}_optimistic"] = gain
        row["tune_sec_mean"] = (
            float(pd.to_numeric(t["tune_sec"], errors="coerce").mean())
            if "tune_sec" in t
            else np.nan
        )
        rows.append(row)
    return rows


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT, help="tuned tree results root")
    ap.add_argument(
        "--untuned-root",
        default=DEFAULT_UNTUNED_ROOT,
        help="untuned tree root (read-only)",
    )
    ap.add_argument(
        "--linear-root", action="append", default=None, help="repeatable or comma list"
    )
    ap.add_argument("--tw", type=int, default=2000)
    ap.add_argument("--out", default=None, help="default: --root")
    ap.add_argument(
        "--min-n",
        type=int,
        default=100,
        help="interval and DM only from this many rows",
    )
    ap.add_argument(
        "--detail-bars", default="bar1600", help="bars for the SHAP / importance tables"
    )
    a = ap.parse_args()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    root = bst.resolve(a.root)
    untuned_root = bst.resolve(a.untuned_root)
    out = bst.resolve(a.out) if a.out else root
    if not root.is_dir():
        print(f"no tuned results root at {root}; nothing to score")
        return
    out.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    print("=" * 30 + " 1. the untuned scorer on the tuned root (MSE rule)")
    fail, fresh = run_untuned_scorer(a, root, out)
    if fail:
        print(f"FAILED: {fail}")
        failures.append(fail)

    print("\n" + "=" * 30 + " 2. the QLIKE-selected paths")
    lin_roots = sts.split_roots(a.linear_root)
    print(
        f"tuned root {root}; untuned root {untuned_root}; linear roots "
        f"{[str(r) for r in lin_roots]}; tw {a.tw}; out {out}"
    )
    if not untuned_root.is_dir():
        print(f"note: untuned root {untuned_root} does not exist")
    arms = bst.discover_tree_arms(root, a.tw)
    gates: list[dict] = []
    fc: dict[tuple, pd.DataFrame] = {}
    fcq: dict[tuple, pd.DataFrame] = {}
    qarms: dict[tuple, bst.TreeArm] = {}
    npz: dict[tuple, dict | None] = {}
    for arm in arms:
        r = sts.load_forecasts(arm.csv)
        if r is None:
            continue
        fc[arm.key] = r
        z = sts.load_npz(arm)
        if z is not None:
            z.pop("shap", None)  # not read here; step 1 read it on the detail bars
        npz[arm.key] = z
        qp = arm.csv.with_name(f"results_qsel_{arm.seg}.csv")
        if not qp.is_file():
            print(f"note: no qsel CSV for {arm.key}")
            continue
        if not qsel_gates(arm, qp, z, gates):
            continue
        rq = sts.load_forecasts(qp)
        if rq is None:
            continue
        qarms[arm.key] = bst.TreeArm(arm.bucket, arm.seg, arm.model + QSEL, arm.tw, qp)
        fcq[arm.key] = rq
    arms = [m for m in arms if m.key in fc]
    known = {m.csv.parent for m in arms}
    for p in sorted(root.rglob(f"tw{a.tw}/{bst.SPEC_DIR}/*/*/results_qsel_bar*.csv")):
        if p.parent not in known:
            print(f"note: {p} has no scorable MSE-path twin; not scored")
    print(f"{len(arms)} tuned arms, {len(qarms)} with a QLIKE-selected path")

    common: dict[str, pd.DatetimeIndex] = {}
    for d in (fc, fcq):
        for k, r in d.items():
            idx = r.dropna(subset=["pred_clock"]).index
            seg = k[1]
            common[seg] = idx if seg not in common else common[seg].intersection(idx)
    q_rows = [
        row
        for k, qa in qarms.items()
        for row in sts.qlike_rows(qa, fcq[k], common[qa.seg], npz.get(k))
    ]
    tab_qq = pd.DataFrame(q_rows)
    sts.write(tab_qq, out, "trees_qsel_qlike_by_bar.csv")

    deck_days = None
    if base.DECK.is_file():
        deck_days = pd.DatetimeIndex(
            pd.to_datetime(pd.read_parquet(base.DECK).index)
        ).normalize()
    else:
        print(f"no deck parquet at {base.DECK}; the 15:30 trade tables are skipped")

    def n_deck(idx: pd.DatetimeIndex) -> int:
        return (
            int(idx.normalize().isin(deck_days).sum()) if deck_days is not None else 0
        )

    lin_cache: dict[Path, pd.DataFrame | None] = {}

    def linear(bucket: str, seg: str, tw: int) -> dict[str, tuple[pd.DataFrame, str]]:
        found = {}
        for est, (path, src) in sorted(
            sts.linear_paths(lin_roots, bucket, seg, tw).items()
        ):
            if path not in lin_cache:
                lin_cache[path] = sts.load_forecasts(path)
            if lin_cache[path] is not None:
                found[est] = (lin_cache[path], src)
        return found

    vs_lin, vs_mse, q_trades = [], [], []
    for k, qa in qarms.items():
        mine = fcq[k]
        lins = linear(qa.bucket, qa.seg, qa.tw)
        if not lins:
            print(f"note: no linear comparator for {qa.bucket}/{qa.seg}/tw{qa.tw}")
        for est, (lin, src) in lins.items():
            keys = {
                "bucket": qa.bucket,
                "model": qa.model,
                "estimator": est,
                "train_win": qa.tw,
                "segment": qa.seg,
                "bar_end": bar_end(qa.seg),
                "linear_source": src,
            }
            vs_lin += sts.paired(mine, lin, keys, ("tree", "linear"), a.min_n, gates)
        keys = arm_keys(qa, k[2]) | {
            "bar_end": bar_end(qa.seg),
            "comparison": "qsel vs mse",
        }
        vs_mse += sts.paired(mine, fc[k], keys, ("qsel", "mse"), a.min_n, gates)
        if qa.seg == TRADE_SEG and deck_days is not None:
            f = mine["pred_clock"]
            f = f[f.index.strftime("%H:%M") == TRADE_HHMM].dropna()
            if n_deck(f.index) < MIN_DECK_DAYS:
                print(
                    f"skip trade {qa.key}: {n_deck(f.index)} deck days (< {MIN_DECK_DAYS})"
                )
            else:
                q_trades.append(
                    {
                        "bucket": qa.bucket,
                        "model": qa.model,
                        "train_win": qa.tw,
                        "forecast": f"tree {qa.model}",
                        "stamps": "own",
                    }
                    | base.trade_1530(f)
                )
    tab_ql = pd.DataFrame(vs_lin)
    sts.write(tab_ql, out, "trees_qsel_vs_linear.csv")
    tab_qm = pd.DataFrame(vs_mse)
    sts.write(tab_qm, out, "trees_qsel_vs_mse.csv")
    sts.write(pd.DataFrame(q_trades), out, "trees_qsel_trade_1530.csv")

    print("\n" + "=" * 30 + " 3. tuned against untuned")
    unt_cache: dict[Path, pd.DataFrame | None] = {}
    twins: dict[tuple, pd.DataFrame] = {}
    vs_unt: list[dict] = []
    for arm in arms:
        p = bst.tree_arm_path(untuned_root, arm.bucket, arm.seg, arm.model, arm.tw)
        if p is None:
            print(
                f"note: no untuned twin for {arm.model}/{arm.bucket}/{arm.seg}/tw{arm.tw}"
            )
            continue
        if p not in unt_cache:
            unt_cache[p] = sts.load_forecasts(p)
        u = unt_cache[p]
        if u is None:
            continue
        twins[arm.key] = u
        for rule, mine in (("mse", fc[arm.key]), ("qlike", fcq.get(arm.key))):
            if mine is None:
                continue
            keys = arm_keys(arm) | {
                "rule": rule,
                "bar_end": bar_end(arm.seg),
                "comparison": "tuned vs untuned",
                "untuned_source": p.relative_to(untuned_root).as_posix(),
            }
            vs_unt += sts.paired(mine, u, keys, ("tuned", "untuned"), a.min_n, gates)
    tab_vu = pd.DataFrame(vs_unt)
    sts.write(tab_vu, out, "trees_tuned_vs_untuned.csv")

    t_trades: list[dict] = []
    for arm in [m for m in arms if m.seg == TRADE_SEG]:
        if deck_days is None:
            break
        k = arm.key
        series = {f"tuned {arm.model}": fc[k]["pred_clock"]}
        if k in fcq:
            series[f"tuned {arm.model} qsel"] = fcq[k]["pred_clock"]
        if k in twins:
            series[f"untuned {arm.model}"] = twins[k]["pred_clock"]
        for est, (lin, _src) in linear(arm.bucket, arm.seg, arm.tw).items():
            series[f"linear {est}"] = lin["pred_clock"]
        stamps = fc[k].dropna(subset=["pred_clock"]).index
        stamps = stamps[stamps.strftime("%H:%M") == TRADE_HHMM]
        aligned = pd.DataFrame({lab: s.reindex(stamps) for lab, s in series.items()})
        same = aligned.dropna()
        if len(same) < len(aligned):
            lacking = {
                c: int(aligned[c].isna().sum())
                for c in aligned
                if aligned[c].isna().any()
            }
            print(
                f"note: trade {arm.key}: {len(aligned) - len(same)} of the tuned arm's "
                f"{len(aligned)} stamps dropped for every forecast (missing: {lacking})"
            )
        if n_deck(same.index) < MIN_DECK_DAYS:
            print(
                f"skip trade {arm.key}: {n_deck(same.index)} same deck days (< {MIN_DECK_DAYS})"
            )
            continue
        for lab in aligned.columns:
            t_trades.append(
                {
                    "bucket": arm.bucket,
                    "model": arm.model,
                    "train_win": arm.tw,
                    "forecast": lab,
                    "stamps": f"tuned {arm.model}'s, every forecast present",
                    "tuned_stamps": len(aligned),
                    "same_stamps": len(same),
                }
                | base.trade_1530(same[lab])
            )
    tab_tt = pd.DataFrame(t_trades)
    sts.write(tab_tt, out, "trees_tuned_trade_1530.csv")

    print("\n" + "=" * 30 + " 4. the hyperparameters")
    traces, longs, shipped = [], [], []
    grid_nat: dict[
        tuple, dict
    ] = {}  # (model, bucket) -> the grid file's natural bounds
    for arm in arms:
        tp = arm.csv.with_name(f"tune_trace_{arm.seg}.csv")
        cp = arm.csv.with_name(f"tune_candidates_{arm.seg}.csv")
        if not tp.is_file():
            print(f"note: no tune trace for {arm.key}")
            continue
        try:
            tr = read_table(tp)
            cd = read_table(cp) if cp.is_file() else None
        except Exception as e:  # noqa: BLE001 -- a half-written file is skipped
            print(f"skip tune files of {arm.key}: {type(e).__name__}: {e}")
            continue
        if cd is None:
            print(
                f"note: no tune candidates for {arm.key}; ranks and candidate gates skipped"
            )
        try:
            check_tune_files(arm, tr, cd, npz.get(arm.key), gates)
            nb = check_edges(arm, tr, cd, gates)
            if nb:
                grid_nat.setdefault((arm.model, arm.bucket), nb)
            longs.append(edge_long(arm, tr))
            shipped += shipped_rows(arm, tr, cd)
        except Exception as e:  # noqa: BLE001 -- a file off the schema fails its gate, not the run
            traceback.print_exc()
            gate(
                gates,
                arm_keys(arm),
                "tune files readable (schema)",
                len(tr),
                False,
                detail=f"{type(e).__name__}: {e}",
            )
        traces.append(tr)
    tab_tr = pd.concat(traces, ignore_index=True) if traces else pd.DataFrame()
    sts.write(tab_tr, out, "trees_tuned_trace.csv")
    long = (
        pd.concat([x for x in longs if not x.empty], ignore_index=True)
        if any(not x.empty for x in longs)
        else pd.DataFrame()
    )
    tab_e = edge_table(long, grid_nat) if not long.empty else pd.DataFrame()
    sts.write(tab_e, out, "trees_tuned_edges.csv")
    tab_s = pd.DataFrame(shipped)
    sts.write(tab_s, out, "trees_tuned_shipped.csv")
    tab_g = pd.DataFrame(gates)
    sts.write(tab_g, out, "trees_tuned_join_gates.csv")

    # ------------------------------------------------------------ 5. summary
    print("\n" + "=" * 30 + " 5. summary")
    summ = []
    tq = fresh.get("trees_qlike_by_bar.csv")
    for tab, col in ((tq, "Q_tuned_mse_own"), (tab_qq, "Q_tuned_qsel_own")):
        if tab is not None and not tab.empty:
            own = tab[tab["sample"] == "own"].assign(
                model=lambda d: d["model"].str.removesuffix(QSEL)
            )
            summ.append(
                own.groupby(["model", "bucket"])["QLIKE_clock"].mean().rename(col)
            )
    if not tab_vu.empty:
        vu = tab_vu[
            (tab_vu["back_transform"] == "clock")
            & (tab_vu["sample"] == "common")
            & ~tab_vu["blown"]
        ]
        for rule, tag in (("mse", "mse"), ("qlike", "qsel")):
            g = vu[vu["rule"] == rule].groupby(["model", "bucket"])
            if rule == "mse":
                summ.append(g["QLIKE_untuned"].mean().rename("Q_untuned_same_stamps"))
            summ.append(g["pct"].mean().rename(f"pct_{tag}_vs_untuned"))
            summ.append(g["improves"].sum().rename(f"{tag}_better"))
            summ.append(g["worse"].sum().rename(f"{tag}_worse"))
    for tab, tag in ((fresh.get("trees_vs_linear.csv"), "mse"), (tab_ql, "qsel")):
        if tab is not None and not tab.empty:
            r = tab[
                (tab["estimator"] == REF_EST)
                & (tab["back_transform"] == "clock")
                & (tab["sample"] == "common")
                & ~tab["blown"].astype(bool)
            ].assign(model=lambda d: d["model"].str.removesuffix(QSEL))
            summ.append(
                r.groupby(["model", "bucket"])["pct"]
                .mean()
                .rename(f"pct_{tag}_vs_{REF_EST}")
            )
    if summ:
        print(
            "per model x bucket, mean over bars of QLIKE clock: tuned (own sample, mse and qsel rule); "
            "untuned on the tuned arm's stamps (trees_tuned_vs_untuned, common); pct = mean paired "
            f"difference in % (negative: tuned wins), better / worse = bars whose day-block interval "
            f"excludes zero; vs {REF_EST} on common stamps"
        )
        print(pd.concat(summ, axis=1).round(4).to_string())
    if not tab_tt.empty:
        print(
            "\n15:30 sign(s) trade on the same days (deck ridge: 1.338 mid / 0.870 crossed):"
        )
        cols = [
            "bucket",
            "forecast",
            "same_stamps",
            "deck_days",
            "pct_buy",
            "Sharpe_mid",
            "Sharpe_crossed",
        ]
        print(tab_tt[cols].round(3).to_string(index=False))
    if not tab_e.empty:
        ge = tab_e[tab_e["n_grid_choice_edge"] > 0]
        print(
            f"\ngrid-choice edges picked ({len(ge)} of {len(tab_e)} model x bucket x rule x axis cells):"
        )
        if not ge.empty:
            cols = [
                "model",
                "bucket",
                "rule",
                "axis",
                "n_tuning_points",
                "n_lo",
                "lo_edge_natural",
            ]
            cols += ["n_hi", "hi_edge_natural", "edge_is_grid_choice_pick_share"]
            print(ge[cols].round(3).to_string(index=False))
    if not tab_s.empty:
        print("\nshipped config (in-selection gain: optimistic by construction):")
        s = tab_s.groupby(["model", "bucket", "rule"]).agg(
            bars=("segment", "size"),
            share_shipped_chosen=("share_shipped_chosen", "mean"),
            rank_mse=("mean_rank_shipped_val_mse", "mean"),
            rank_qlike=("mean_rank_shipped_val_qlike", "mean"),
            gain_mse=("in_selection_gain_mse_optimistic", "mean"),
            gain_qlike=("in_selection_gain_qlike_optimistic", "mean"),
        )
        print(s.round(4).to_string())
    tg = fresh.get("trees_join_gates.csv")
    if tg is not None and not tg.empty:
        print(
            f"\nstep 1 gates (trees_join_gates.csv): {len(tg)} checked, {int((~tg['ok'].astype(bool)).sum())} failed"
        )
    if not tab_g.empty:
        bad = tab_g[~tab_g["ok"]]
        print(
            f"tuned gates (trees_tuned_join_gates.csv): {len(tab_g)} checked, {len(bad)} failed"
        )
        if not bad.empty:
            print(bad.groupby("check").size().to_string())
            failures.append(
                f"{len(bad)} gate(s) failed -- see trees_tuned_join_gates.csv"
            )
    if failures:
        sys.exit("; ".join(failures))


if __name__ == "__main__":
    main()
