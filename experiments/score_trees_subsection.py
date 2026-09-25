"""Score the per-bar TREE arms against the per-bar linear arms, bar by bar.

What is scored.  ``specs/causal_tune_trees.py`` fits LightGBM / XGBoost / a
random forest on exactly the per-bar linear design (one arm = bucket x one bar
x model x training window).  Every arm's variance forecast is rebuilt from its
adjusted-scale columns with the ONE causal back-transform of
``score_linear_subsection_causal.py`` (``causal_forecasts``: s = mean squared
adjusted-scale error of the arm's own previous 250 sessions at that bar label,
lagged one session) -- the linear comparators get exactly the same treatment,
so a difference is the forecast, not the calibration.  The spec's own
``pred_raw`` (a look-ahead Duan smear) is carried beside it as "as scored".

Tables (``--out``, default = ``--root``):
  trees_qlike_by_bar.csv          per (bucket, model, bar, sample) n, QLIKE with the
                                  causal clock back-transform and as scored,
                                  fit-space MSE, refits and mean fit seconds (npz).
                                  sample: own (the arm's rows past the smear warm-up),
                                  common (stamps every tree arm at that bar shares),
                                  deck period (common within the deck window)
  trees_vs_linear.csv             each tree arm against every linear arm of the same
                                  bucket, window and bar on identical stamps: QLIKE,
                                  paired difference, day-block 95 % interval (tree
                                  minus linear; improves = interval below zero),
                                  Diebold-Mariano t -- THE comparison
  trees_vs_baseline_tree.csv      each model on an exogenous bucket against the same
                                  model on ``baseline`` (HAR + calendar): what the
                                  exogenous columns add to a tree
  trees_join_gates.csv            every join's gate: the target true_raw must agree to
                                  1e-9 relative (same design, same target), and the
                                  npz must carry the CSV's stamps and pred_adj
  trees_trade_1530.csv            the deck's 15:30 sign(s) trade (base.trade_1530) on
                                  each bar1600 tree arm's causal forecast, and on the
                                  linear comparators' on the tree arm's stamps
  trees_shap_<bar>.csv            per feature: mean |SHAP|, mean SHAP, share, rank
  trees_shap_family_<bar>.csv     the same by feature family (rules printed on run)
  trees_importance_<bar>.csv      native importance averaged over the refits
                                  (LightGBM gain, XGBoost total_gain, RF impurity)
for <bar> in ``--detail-bars`` (default bar1600, the bar SHAP is computed on).

Robust to a partially finished campaign (the cluster scorer runs afterany):
missing arms, npz files or linear comparators are skipped with a message.  A
failed gate is printed, the pair is left out, every table is still written, and
the exit status is non-zero.

Usage:  python experiments/score_trees_subsection.py
            [--root results/linear_subsection_trees] [--linear-root DIR ...]
            [--tw 2000] [--out DIR] [--min-n 100] [--detail-bars bar1600]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "notebooks", ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import build_subsection_tree_yhat as bst  # noqa: E402
import score_linear_subsection as base  # noqa: E402
import score_linear_subsection_causal as slc  # noqa: E402
from src.evaluation.diebold_mariano import dm_test  # noqa: E402

DEFAULT_ROOT = "results/linear_subsection_trees"
DEFAULT_LINEAR_ROOTS = (
    "results/linear_subsection/arms_hoffman2",  # flattened Hoffman2 pull
    "results/linear_subsection",  # the cluster layout
)
BASELINE_BUCKET = "baseline"
REF_EST = "ridge"  # the summary's reference linear estimator
GATE_REL = 1e-9
MIN_DECK_DAYS = 20
COLS = ["true_raw", "pred_clock", "pred_raw"]

# ---------------------------------------------------------------- feature families
# Rules, first match wins (printed on every run):
#   har_ladder     ^har_ma_<k>$                  the target's own HAR ladder
#   har_x_session  ^har_ma_<k>_x_<session>$      ladder x open / close interaction
#   calendar       DOW_<d>, hour, is_*, days_to_*
#   availability   <src>_avail_ma_<k>            impute-and-indicate availability flags
#   hurdle_active  <src>_active_ma_<k>           the zero-inflated columns' occurrence flags
#   exog:<group>   [adj_]<src>_ma_<k>            the exogenous source <src>, grouped by the
#                                                bucket it belongs to in src.data.loading
#   exog:other     an exogenous stem in no atomic bucket; other   anything else
_HAR = re.compile(r"^har_ma_\d+$")
_HARX = re.compile(r"^har_ma_\d+_x_\w+$")
_CAL = re.compile(r"^(DOW_\d+|hour|is_\w+|days_to_\w+)$")
_MA = re.compile(r"_ma_\d+$")
_ATOMIC_GROUPS = (
    "moments",
    "liquidity",
    "market_ew",
    "market_vw",
    "sentiment",
    "implied_vol",
    "vol_demand",
    "fomc",
    "options",
    "mfiv_only",
    "ivslice_only",
)
FAMILY_RULES = """feature families (first match wins):
  har_ladder     ^har_ma_<k>$                 the target's HAR ladder
  har_x_session  ^har_ma_<k>_x_<session>$     ladder x open/close interactions
  calendar       ^(DOW_<d>|hour|is_*|days_to_*)$
  availability   <src>_avail_ma_<k>           impute-and-indicate availability flags
  hurdle_active  <src>_active_ma_<k>          zero-inflated columns' occurrence flags
  exog:<group>   [adj_]<src>_ma_<k>           source <src> by its atomic bucket in
                 src.data.loading.SUBGROUPS ({groups}); unmatched -> exog:other
  other          anything else (e.g. unnamed f<j> columns)"""


def _exog_groups() -> dict[str, str]:
    try:
        from src.data.loading import SUBGROUPS
    except Exception as e:  # noqa: BLE001 -- the scorer must run without the loader
        print(
            f"note: src.data.loading not importable ({e}); exogenous sources ungrouped"
        )
        return {}
    out: dict[str, str] = {}
    for g in _ATOMIC_GROUPS:
        for col in SUBGROUPS.get(g, []):
            out.setdefault(col, g.removesuffix("_only"))
    return out


EXOG_GROUP = _exog_groups()


def feature_family(name: str) -> tuple[str, str]:
    """(family, source) of a design column; source is the exogenous stem for exog columns."""
    if _HAR.match(name):
        return "har_ladder", "har_ladder"
    if _HARX.match(name):
        return "har_x_session", "har_x_session"
    if _CAL.match(name):
        return "calendar", "calendar"
    if not _MA.search(name):
        return "other", "other"
    stem = _MA.sub("", name).removeprefix("adj_")
    if stem.endswith("_avail"):
        return "availability", "availability"
    if stem.endswith("_active"):
        return "hurdle_active", "hurdle_active"
    return f"exog:{EXOG_GROUP.get(stem, 'other')}", stem


# ---------------------------------------------------------------- loading
def read_adj(path: Path) -> pd.DataFrame:
    """A results CSV prepared as ``slc.load_adj`` prepares it (any layout)."""
    r = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    if r.index.duplicated().any():
        raise ValueError(f"duplicated stamps in {path}")
    r = r[(r["true_adj"] > 0) & (r["true_raw"] > 0)].copy()
    r["baseline"] = r["true_raw"] / r["true_adj"] ** 2
    r["e2"] = (r["true_adj"] - r["pred_adj"]) ** 2
    r["hhmm"] = r.index.strftime("%H:%M")
    r["day"] = r.index.normalize()
    return r


def load_forecasts(path: Path) -> pd.DataFrame | None:
    try:
        r = read_adj(path)
    except Exception as e:  # noqa: BLE001 -- a truncated or half-written file is skipped
        print(f"skip {path}: {type(e).__name__}: {e}")
        return None
    if r.empty:
        print(f"skip {path}: no scorable rows")
        return None
    return slc.causal_forecasts(r)


def load_npz(arm: bst.TreeArm) -> dict | None:
    if not arm.npz.is_file():
        print(f"note: no npz for {arm.key}")
        return None
    try:
        with np.load(
            arm.npz, allow_pickle=True
        ) as z:  # our own files (older ones store object dates)
            d = {k: z[k] for k in z.files}
        d["meta"] = json.loads(str(d.get("meta", "{}")))
        return d
    except Exception as e:  # noqa: BLE001
        print(f"note: unreadable npz {arm.npz}: {type(e).__name__}: {e}")
        return None


def linear_paths(
    roots: list[Path], bucket: str, seg: str, tw: int
) -> dict[str, tuple[Path, str]]:
    """{estimator: (results CSV, which root)} -- flattened and cluster layouts, first root wins."""
    found: dict[str, tuple[Path, str]] = {}
    for lr in roots:
        for p in sorted((lr / bucket).glob(f"*/tw{tw}/results_{seg}.csv")):
            found.setdefault(p.parts[-3], (p, f"{lr.name} (flattened)"))
        pat = f"*/tw{tw}/causal_tune_linear/*/{bucket}/results_{seg}.csv"
        for p in sorted((lr / bucket / seg).glob(pat)):
            if p.parts[-3] == p.parts[-6]:
                found.setdefault(p.parts[-3], (p, f"{lr.name} (cluster)"))
    if bucket == BASELINE_BUCKET:
        # every cluster-layout arm dir also carries the per-bar OLS incumbent, fitted on the
        # baseline design (HAR + calendar) whatever the arm's bucket -- one copy is taken
        pat = f"*/{seg}/*/tw{tw}/causal_tune_linear/incumbent_ols/results_{seg}.csv"
        for lr in roots:
            hits = sorted(lr.glob(pat))
            if hits:
                found.setdefault("ols", (hits[0], f"{lr.name} (incumbent_ols)"))
                break
    return found


def in_deck(idx: pd.DatetimeIndex) -> np.ndarray:
    return np.asarray((idx >= base.DECK_START) & (idx <= base.DECK_END + " 23:59"))


def rel_gap(a: pd.Series, b: pd.Series) -> float:
    return float((a / b - 1.0).abs().max()) if len(a) else float("nan")


# ---------------------------------------------------------------- paired comparison
def paired(
    a: pd.DataFrame,
    b: pd.DataFrame,
    keys: dict,
    names: tuple[str, str],
    min_n: int,
    gates: list[dict],
) -> list[dict]:
    """QLIKE of a against b on identical stamps, clock and as scored, common and deck period."""
    j = a[COLS].join(b[COLS], how="inner", rsuffix="_b")
    gap = rel_gap(j["true_raw"], j["true_raw_b"])
    ok = bool(len(j) and gap < GATE_REL)
    gates.append(
        keys
        | {"check": "true_raw a vs b", "n_a": len(a), "n_b": len(b), "n": len(j)}
        | {"max_rel": gap, "ok": ok}
    )
    if not len(j):
        print(f"skip {keys}: no shared stamps")
        return []
    if not ok:
        print(
            f"GATE FAIL {keys}: true_raw differs by {gap:.2e} relative on {len(j)} stamps"
        )
        return []
    qa, qb = f"QLIKE_{names[0]}", f"QLIKE_{names[1]}"
    rows = []
    for bt, ca, cb in (
        ("clock", "pred_clock", "pred_clock_b"),
        ("as scored", "pred_raw", "pred_raw_b"),
    ):
        jj = j.dropna(subset=[ca, cb])
        la, lb = slc.qlike(jj["true_raw"], jj[ca]), slc.qlike(jj["true_raw"], jj[cb])
        for smp, m in (
            ("common", np.ones(len(jj), bool)),
            ("deck period", in_deck(jj.index)),
        ):
            n = int(m.sum())
            if n == 0:
                continue
            d = (la - lb)[m]
            lo = hi = dm = float("nan")
            if n >= min_n:
                lo, hi = base.day_block_ci(d)
                dm = float(dm_test(la[m].to_numpy(), lb[m].to_numpy())["dm"])
            ma, mb = float(la[m].mean()), float(lb[m].mean())
            rows.append(
                keys
                | {
                    "back_transform": bt,
                    "sample": smp,
                    "n": n,
                    "first": str(jj.index[m].min().date()),
                    "last": str(jj.index[m].max().date()),
                    qa: ma,
                    qb: mb,
                    "diff": float(d.mean()),
                    "pct": float(100 * d.mean() / mb),
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "dm_stat": dm,
                    "improves": bool(hi < 0.0),
                    "worse": bool(lo > 0.0),
                    "blown": bool(ma > slc.BLOWN or mb > slc.BLOWN),
                }
            )
    return rows


# ---------------------------------------------------------------- tables
def qlike_rows(
    arm: bst.TreeArm, r: pd.DataFrame, common: pd.DatetimeIndex, z: dict | None
) -> list[dict]:
    meta = (z or {}).get("meta", {})
    info = {
        "rows_total": len(r),
        "refits": len(z["refit_row"]) if z is not None and "refit_row" in z else np.nan,
        "fit_sec_mean": float(np.mean(z["fit_sec"]))
        if z is not None and np.size(z.get("fit_sec", []))
        else np.nan,
        "shap_sec_mean": float(np.mean(z["shap_sec"]))
        if z is not None and np.size(z.get("shap_sec", []))
        else np.nan,
        "refit_every": meta.get("refit_every", np.nan),
        "n_features": meta.get("n_features", np.nan),
        "has_shap": bool(meta.get("shap", False)),
    }
    own = r.dropna(subset=["pred_clock"])
    cm = own.loc[own.index.intersection(common)]
    rows = []
    for smp, s in (
        ("own", own),
        ("common", cm),
        ("deck period", cm[in_deck(cm.index)]),
    ):
        row = {
            "bucket": arm.bucket,
            "model": arm.model,
            "train_win": arm.tw,
            "segment": arm.seg,
            "bar_end": f"{arm.seg[3:5]}:{arm.seg[5:7]}",
            "sample": smp,
            "n": len(s),
            "first": str(s.index.min().date()) if len(s) else "",
            "last": str(s.index.max().date()) if len(s) else "",
            "QLIKE_clock": float(slc.qlike(s["true_raw"], s["pred_clock"]).mean())
            if len(s)
            else np.nan,
            "QLIKE_as_scored": float(slc.qlike(s["true_raw"], s["pred_raw"]).mean())
            if len(s)
            else np.nan,
            "mse_adj": float(s["e2"].mean()) if len(s) else np.nan,
        }
        rows.append(row | info)
    return rows


def npz_gate(arm: bst.TreeArm, z: dict, gates: list[dict]) -> None:
    """The npz carries the CSV's stamps and pred_adj (same run, same rows)."""
    keys = {
        "bucket": arm.bucket,
        "model": arm.model,
        "train_win": arm.tw,
        "segment": arm.seg,
    }
    raw = pd.read_csv(arm.csv)
    dz = np.asarray(z["date"]).astype(str)
    same_dates = len(dz) == len(raw) and bool(
        (dz == raw["date"].astype(str).to_numpy()).all()
    )
    gap = (
        float(
            np.max(
                np.abs(
                    np.asarray(z["pred_adj"], float) - raw["pred_adj"].to_numpy(float)
                )
            )
        )
        if same_dates
        else np.nan
    )
    ok = same_dates and gap < 1e-12
    gates.append(
        keys
        | {
            "check": "npz vs csv (dates, pred_adj)",
            "n_a": len(dz),
            "n_b": len(raw),
            "n": len(raw),
            "max_rel": gap,
            "ok": ok,
        }
    )
    if not ok:
        print(
            f"GATE FAIL {keys}: npz and CSV disagree (same dates {same_dates}, pred_adj gap {gap})"
        )


def shap_tables(
    arm: bst.TreeArm, z: dict
) -> tuple[list[dict], list[dict], list[dict], dict]:
    keys = {
        "bucket": arm.bucket,
        "model": arm.model,
        "train_win": arm.tw,
        "segment": arm.seg,
    }
    names = [str(s) for s in np.asarray(z["feature_names"])]
    p = len(names)
    fam = pd.DataFrame([feature_family(n) for n in names], columns=["family", "source"])
    fam.insert(0, "feature", names)
    feat_rows: list[dict] = []
    fam_rows: list[dict] = []
    imp_rows: list[dict] = []
    check: dict = {}
    imp_share: np.ndarray | None = None  # native importance share, in feature order

    imp = np.asarray(z.get("importance", np.zeros((0, 0))), dtype=np.float64)
    if imp.ndim == 2 and imp.shape[0] and imp.shape[1] == p:
        mi = imp.mean(axis=0)
        tot = mi.sum()
        t = fam.assign(
            mean_importance=mi,
            share=mi / tot if tot > 0 else np.nan,
            used_in_refits=(imp > 0).mean(axis=0),
            refits=imp.shape[0],
        )
        t["rank"] = t["mean_importance"].rank(ascending=False, method="min").astype(int)
        imp_rows = [keys | r for r in t.sort_values("rank").to_dict("records")]
        imp_share = t["share"].to_numpy(float)
    elif imp.size:
        print(
            f"note: importance of {arm.key} has shape {imp.shape}, {p} feature names -- skipped"
        )

    sh = np.asarray(z.get("shap", np.zeros((0, 0))))
    if sh.ndim != 2 or sh.size == 0:
        return feat_rows, fam_rows, imp_rows, check
    if sh.shape[1] != p + 1:
        print(
            f"note: SHAP of {arm.key} has {sh.shape[1]} columns for {p} features -- skipped"
        )
        return feat_rows, fam_rows, imp_rows, check
    pred = np.asarray(z["pred_adj"], dtype=np.float64)
    ok = np.isfinite(sh).all(axis=1) & np.isfinite(pred)
    s64 = sh[ok].astype(np.float64)
    phi, ev = s64[:, :p], s64[:, p]
    gap = np.abs(s64.sum(axis=1) - pred[ok])
    check = keys | {
        "rows": int(ok.sum()),
        "rows_nan": int((~ok).sum()),
        "max_abs_additivity_gap": float(gap.max()) if gap.size else np.nan,
        "max_rel_additivity_gap": float((gap / np.abs(pred[ok])).max())
        if gap.size
        else np.nan,
        "recorded_gap_float64": float(np.asarray(z.get("shap_additivity_gap", np.nan))),
        "expected_value_mean": float(ev.mean()) if ev.size else np.nan,
        "pred_adj_mean": float(pred[ok].mean()) if ok.any() else np.nan,
    }
    mabs = np.abs(phi).mean(axis=0)
    tot = mabs.sum()
    t = fam.assign(
        mean_abs_shap=mabs,
        mean_shap=phi.mean(axis=0),
        share=mabs / tot if tot > 0 else np.nan,
    )
    t["rank"] = t["mean_abs_shap"].rank(ascending=False, method="min").astype(int)
    feat_rows = [
        keys | {"rows": int(ok.sum())} | r
        for r in t.sort_values("rank").to_dict("records")
    ]
    for level, col in (("family", "family"), ("source", "source")):
        for g, idx in t.groupby(col).groups.items():
            cols = np.asarray(idx, dtype=int)  # t keeps the feature order: positions
            summed = phi[:, cols].sum(axis=1)
            fam_rows.append(
                keys
                | {
                    "level": level,
                    "family": g,
                    "n_features": len(cols),
                    "sum_mean_abs_shap": float(mabs[cols].sum()),
                    "share": float(mabs[cols].sum() / tot) if tot > 0 else np.nan,
                    "mean_abs_of_family_sum": float(np.abs(summed).mean()),
                    "mean_family_sum": float(summed.mean()),
                    "importance_share": float(imp_share[cols].sum())
                    if imp_share is not None
                    else np.nan,
                }
            )
    return feat_rows, fam_rows, imp_rows, check


# ---------------------------------------------------------------- main
def split_roots(vals: list[str] | None) -> list[Path]:
    raw = [
        v
        for s in (vals or DEFAULT_LINEAR_ROOTS)
        for v in str(s).split(",")
        if v.strip()
    ]
    roots = [bst.resolve(v.strip()) for v in raw]
    have = [r for r in roots if r.is_dir()]
    for r in roots:
        if not r.is_dir():
            print(f"note: linear root {r} does not exist")
    return have


def write(tab: pd.DataFrame, out: Path, name: str) -> None:
    if tab.empty:
        print(f"no rows for {name}; not written")
        return
    tab.to_csv(out / name, index=False)
    print(f"wrote {out / name} ({len(tab)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT, help="tree results root")
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
    out = bst.resolve(a.out) if a.out else root
    lin_roots = split_roots(a.linear_root)
    detail = {s.strip() for s in a.detail_bars.split(",") if s.strip()}
    print(FAMILY_RULES.format(groups=", ".join(sorted(set(EXOG_GROUP.values())))))
    print(
        f"tree root {root}; linear roots {[str(r) for r in lin_roots]}; tw {a.tw}; out {out}"
    )
    if not root.is_dir():
        print("no tree results root; nothing to score")
        return
    out.mkdir(parents=True, exist_ok=True)

    arms = bst.discover_tree_arms(root, a.tw)
    print(
        f"{len(arms)} tree arms: "
        + ", ".join(f"{m.model}/{m.bucket}/{m.seg}" for m in arms)
    )
    fc: dict[tuple, pd.DataFrame] = {}
    npz: dict[tuple, dict | None] = {}
    for arm in arms:
        r = load_forecasts(arm.csv)
        if r is None:
            continue
        fc[arm.key] = r
        z = load_npz(arm)
        if z is not None and arm.seg not in detail:
            z.pop("shap", None)  # SHAP is read on the detail bars only
        npz[arm.key] = z
    arms = [m for m in arms if m.key in fc]

    gates: list[dict] = []
    for arm in arms:
        z = npz.get(arm.key)
        if z is not None and "date" in z and "pred_adj" in z:
            npz_gate(arm, z, gates)

    # a. QLIKE by bar
    common: dict[str, pd.DatetimeIndex] = {}
    for arm in arms:
        idx = fc[arm.key].dropna(subset=["pred_clock"]).index
        common[arm.seg] = (
            idx if arm.seg not in common else common[arm.seg].intersection(idx)
        )
    q = [
        row
        for arm in arms
        for row in qlike_rows(arm, fc[arm.key], common[arm.seg], npz.get(arm.key))
    ]
    tab_q = pd.DataFrame(q)
    write(tab_q, out, "trees_qlike_by_bar.csv")

    # b. against the linear arms, and against the same model on the baseline bucket
    lin_cache: dict[Path, pd.DataFrame | None] = {}
    vs_lin: list[dict] = []
    trades: list[dict] = []
    deck_days = None
    if base.DECK.is_file():
        deck_days = pd.DatetimeIndex(
            pd.to_datetime(pd.read_parquet(base.DECK).index)
        ).normalize()
    else:
        print(f"no deck parquet at {base.DECK}; the 15:30 trade table is skipped")

    def trade(f: pd.Series, label: dict) -> None:
        if deck_days is None:
            return
        f = f[f.index.strftime("%H:%M") == "16:00"].dropna()
        n = int(f.index.normalize().isin(deck_days).sum())
        if n < MIN_DECK_DAYS:
            print(f"skip trade {label}: {n} deck days (< {MIN_DECK_DAYS})")
            return
        trades.append(label | base.trade_1530(f))

    for arm in arms:
        mine = fc[arm.key]
        lins = linear_paths(lin_roots, arm.bucket, arm.seg, arm.tw)
        if not lins:
            print(f"note: no linear comparator for {arm.bucket}/{arm.seg}/tw{arm.tw}")
        if arm.seg == "bar1600":
            trade(
                mine["pred_clock"],
                {
                    "bucket": arm.bucket,
                    "model": arm.model,
                    "train_win": arm.tw,
                    "forecast": f"tree {arm.model}",
                    "stamps": "own",
                },
            )
        for est, (path, src) in sorted(lins.items()):
            if path not in lin_cache:
                lin_cache[path] = load_forecasts(path)
            lin = lin_cache[path]
            if lin is None:
                continue
            keys = {
                "bucket": arm.bucket,
                "model": arm.model,
                "estimator": est,
                "train_win": arm.tw,
                "segment": arm.seg,
                "bar_end": f"{arm.seg[3:5]}:{arm.seg[5:7]}",
                "linear_source": src,
            }
            vs_lin += paired(mine, lin, keys, ("tree", "linear"), a.min_n, gates)
            if arm.seg == "bar1600":
                on = lin["pred_clock"].reindex(mine.dropna(subset=["pred_clock"]).index)
                trade(
                    on,
                    {
                        "bucket": arm.bucket,
                        "model": arm.model,
                        "train_win": arm.tw,
                        "forecast": f"linear {est}",
                        "stamps": f"tree {arm.model}'s",
                    },
                )
    write(pd.DataFrame(vs_lin), out, "trees_vs_linear.csv")

    vs_base: list[dict] = []
    for arm in arms:
        if arm.bucket == BASELINE_BUCKET:
            continue
        ref = fc.get((BASELINE_BUCKET, arm.seg, arm.model, arm.tw))
        if ref is None:
            print(
                f"note: no {BASELINE_BUCKET} twin for {arm.model}/{arm.bucket}/{arm.seg}"
            )
            continue
        keys = {
            "bucket": arm.bucket,
            "model": arm.model,
            "train_win": arm.tw,
            "segment": arm.seg,
            "bar_end": f"{arm.seg[3:5]}:{arm.seg[5:7]}",
        }
        vs_base += paired(fc[arm.key], ref, keys, ("exog", "baseline"), a.min_n, gates)
    write(pd.DataFrame(vs_base), out, "trees_vs_baseline_tree.csv")
    tab_g = pd.DataFrame(gates)
    write(tab_g, out, "trees_join_gates.csv")

    # c. the 15:30 trade
    write(pd.DataFrame(trades), out, "trees_trade_1530.csv")

    # d. SHAP and native importance on the detail bars
    checks: list[dict] = []
    for seg in sorted(detail):
        f_rows, g_rows, i_rows = [], [], []
        for arm in [m for m in arms if m.seg == seg]:
            z = npz.get(arm.key)
            if z is None or "feature_names" not in z:
                continue
            fr, gr, ir, ck = shap_tables(arm, z)
            f_rows += fr
            g_rows += gr
            i_rows += ir
            if ck:
                checks.append(ck)
        write(pd.DataFrame(f_rows), out, f"trees_shap_{seg}.csv")
        write(pd.DataFrame(g_rows), out, f"trees_shap_family_{seg}.csv")
        write(pd.DataFrame(i_rows), out, f"trees_importance_{seg}.csv")
        if g_rows:
            g = pd.DataFrame(g_rows)
            g = g[g["level"] == "family"].sort_values(
                ["bucket", "model", "share"], ascending=[True, True, False]
            )
            print(f"\nSHAP share by family, {seg}:")
            print(
                g[
                    [
                        "bucket",
                        "model",
                        "family",
                        "n_features",
                        "share",
                        "importance_share",
                    ]
                ]
                .round(4)
                .to_string(index=False)
            )

    # summary
    print()
    if checks:
        print("SHAP additivity (from the arrays: max |sum(row) - pred_adj|):")
        print(pd.DataFrame(checks).drop(columns=["train_win"]).to_string(index=False))
    if not tab_g.empty:
        bad = tab_g[~tab_g["ok"]]
        print(
            f"\ngates: {len(tab_g)} checked, {len(bad)} failed; max true_raw rel gap over passing joins "
            f"{tab_g.loc[tab_g['ok'] & tab_g['check'].str.startswith('true_raw'), 'max_rel'].max():.1e}"
        )
    if not tab_q.empty:
        own = tab_q[tab_q["sample"] == "own"]
        summ = own.groupby(["model", "bucket"]).agg(
            bars=("segment", "size"),
            rows=("n", "sum"),
            mean_QLIKE_clock=("QLIKE_clock", "mean"),
            mean_QLIKE_as_scored=("QLIKE_as_scored", "mean"),
        )
        vl = pd.DataFrame(vs_lin)
        if not vl.empty:
            r = vl[
                (vl["estimator"] == REF_EST)
                & (vl["back_transform"] == "clock")
                & (vl["sample"] == "common")
                & ~vl["blown"]
            ]
            if not r.empty:
                summ = summ.join(
                    r.groupby(["model", "bucket"]).agg(
                        bars_vs_ridge=("pct", "size"),
                        mean_pct_vs_ridge=("pct", "mean"),
                        better=("improves", "sum"),
                        worse=("worse", "sum"),
                    )
                )
        print(
            f"\nper model x bucket (QLIKE clock, own sample; vs {REF_EST}: common stamps, day-block 95 % interval):"
        )
        print(summ.round(4).to_string())
    if trades:
        print("\n15:30 sign(s) trade (deck ridge: 1.338 mid / 0.870 crossed):")
        print(pd.DataFrame(trades).round(3).to_string(index=False))
    if not tab_g.empty and (~tab_g["ok"]).any():
        sys.exit(
            f"{int((~tab_g['ok']).sum())} gate(s) failed -- see trees_join_gates.csv"
        )


if __name__ == "__main__":
    main()
