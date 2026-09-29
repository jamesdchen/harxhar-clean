"""Score the per-bar LSTM (the 15:30 forecast of the 15:30-16:00 bar) against the per-bar linear and tree arms.

What is scored.  ``specs/causal_tune_lstm.py`` fits a small LSTM on exactly the
per-bar design of the linear arms (``specs/causal_tune_linear.py``) and the tree
arms (``specs/causal_tune_trees.py``, ``..._tuned.py``), bar1600 only, buckets
live_feasible / all_features / baseline (HAR + calendar), 2000-session window.
Every forecast -- the LSTM's and every comparator's -- is rebuilt from its
adjusted-scale columns with the ONE causal back-transform of the research
scorer (``score_linear_subsection_causal.causal_forecasts``, used through
``score_trees_subsection.load_forecasts``: the "clock" forecast (f^2 + s) x B,
s = mean squared adjusted-scale error of the arm's own previous 250 sessions at
16:00, lagged one session).  This is the research scorer's 16:00-bar
recalibration -- NOT the rv_iv notebook's 13-bar one; the two are never mixed
here.  Beside it: "as scored" (the executor's look-ahead Duan pred_raw) and
"plain" (f^2 x B, no second-moment term) -- the latter shows where the stored
forecasts' rare extreme values come from.

Comparators (same bucket, bar1600, tw 2000; skipped with a note when absent):
  linear  ridge / lasso / enet (+ ols on baseline), ``sts.linear_paths`` roots
  trees   lgbm / xgb / rf untuned (``--tree-root``) and causally tuned
          (``--tuned-root``, MSE rule), ``build_subsection_tree_yhat.tree_arm_path``
  combination  avg_lstm_ridge = 0.5 x LSTM + 0.5 x ridge on the adjusted scale (no
          estimated weight), then the same clock back-transform: does the network add
          anything to the ridge?

Tables (``--out``, default = ``--root``):
  lstm_qlike_1600.csv        per (bucket, forecast, sample): n, QLIKE clock / as
                             scored / plain, the largest one-day clock QLIKE;
                             sample "all common" = the stamps where EVERY forecast
                             of the bucket exists (same rows for all), and its
                             deck-period subset
  lstm_vs_comparators.csv    ``sts.paired``: LSTM (both rules) minus each comparator
                             on identical stamps, clock and as scored, common and deck
                             period: diff, %, day-block 95 % interval, DM t
  lstm_fitspace_1600.csv     per (bucket, forecast) on the same stamps, adjusted scale:
                             MSE, correlation with the target, Mincer-Zarnowitz slope,
                             forecast sd, correlation with the ridge's forecast, and the
                             correlation with the target of the forecast shifted one
                             session (an alignment check: it must be far lower)
  lstm_trade_1530.csv        the deck's 15:30 sign(s) straddle trade (866 days; the
                             logic of ``score_linear_subsection.trade_1530``, whose
                             Sharpe it must reproduce -- a gate) per forecast: % buy,
                             Sharpe mid / crossed with day-block bootstrap intervals
  lstm_trade_vs.csv          paired Sharpe differences LSTM minus comparator on the
                             same days (the same bootstrap draws), mid and crossed,
                             95 % interval, share of days with the same position
  lstm_hyperparameter_path.csv   the chosen configuration at every tuning point (both rules)
  lstm_hyperparameter_share.csv  share of tuning points per axis value
  lstm_seed_spread.csv       each of the N_SEEDS single networks scored alone (QLIKE
                             clock on the bucket's common stamps, trade Sharpe) beside
                             the seed-averaged forecast of record
  lstm_join_gates.csv        every gate (target agreement 1e-9 on each join, trade
                             reproduction, same 866 days for every trade row)
  lstm_score_tables.md       the key tables in markdown (numbers as written above)

Usage:  python experiments/score_lstm_subsection.py [--root results/linear_subsection_lstm]
            [--linear-root DIR ...] [--tree-root results/linear_subsection_trees]
            [--tuned-root results/linear_subsection_trees_tuned] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "notebooks", ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import atm_straddle_lib as asl  # noqa: E402
import build_subsection_lstm_yhat as bly  # noqa: E402
import build_subsection_tree_yhat as bst  # noqa: E402
import score_linear_subsection as base  # noqa: E402
import score_linear_subsection_causal as slc  # noqa: E402
import score_trees_subsection as sts  # noqa: E402

SEG = "bar1600"
TW = 2000
TREE_MODELS = ("lgbm", "xgb", "rf")
LINEAR_NAMES = {
    "ridge": "ridge",
    "reclasso": "lasso",
    "reclasticnet": "enet",
    "ols": "ols",
}
LSTM_LABELS = {"": "lstm", "qsel_": "lstm_qsel"}  # results infix -> forecast label
REFERENCE = "ridge"  # the per-bar forecast of record in the comparisons
# The equal-weight average of the LSTM and the per-bar ridge on the adjusted scale
# (0.5 each: no weight is estimated, so nothing is fitted on the scored rows); if it
# beats the ridge, the network carries information the ridge does not.
COMBO = "avg_lstm_ridge"
COMBO_WEIGHT = 0.5
TRADE_GATE_ABS = (
    1e-12  # the trade rebuild must reproduce base.trade_1530's Sharpe to round-off
)


def resolve(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else ROOT / q


# ---------------------------------------------------------------- forecasts
GATES: list[dict] = []


def load_bucket(
    bucket: str, a: argparse.Namespace, lin_roots: list[Path]
) -> dict[str, pd.DataFrame]:
    """{label: load_forecasts frame} of every forecast of the bucket that exists."""
    out: dict[str, pd.DataFrame] = {}
    src: dict[str, str] = {}
    for infix, label in LSTM_LABELS.items():
        p = bly.lstm_arm_path(resolve(a.root), bucket, a.tw, infix)
        if p is not None:
            src[label] = str(p)
    for est, (p, where) in sorted(
        sts.linear_paths(lin_roots, bucket, SEG, a.tw).items()
    ):
        if est in LINEAR_NAMES:
            src[LINEAR_NAMES[est]] = str(p)
    for root, suffix in ((a.tree_root, ""), (a.tuned_root, "_tuned")):
        if not root:
            continue
        for m in TREE_MODELS:
            p = bst.tree_arm_path(resolve(root), bucket, SEG, m, a.tw)
            if p is not None:
                src[f"{m}{suffix}"] = str(p)
    if "lstm" in src and REFERENCE in src:
        a_, b_ = sts.read_adj(Path(src["lstm"])), sts.read_adj(Path(src[REFERENCE]))
        j = a_.join(b_[["pred_adj", "true_raw"]], how="inner", rsuffix="_ref")
        gap = float((j["true_raw"] / j["true_raw_ref"] - 1.0).abs().max())
        GATES.append(
            {
                "bucket": bucket,
                "check": f"{COMBO}: true_raw lstm vs {REFERENCE}",
                "n": len(j),
                "max_rel": gap,
                "ok": bool(gap < sts.GATE_REL),
            }
        )
        if gap < sts.GATE_REL:
            c = j.drop(columns=["pred_adj_ref", "true_raw_ref"]).copy()
            c["pred_adj"] = (
                COMBO_WEIGHT * j["pred_adj"] + (1.0 - COMBO_WEIGHT) * j["pred_adj_ref"]
            )
            c["e2"] = (c["true_adj"] - c["pred_adj"]) ** 2
            c["pred_raw"] = (c["pred_adj"] ** 2 + c["e2"].mean()) * c[
                "baseline"
            ]  # the executor's Duan form
            f = slc.causal_forecasts(c)
            f = f[f["hhmm"] == "16:00"]
            f["pred_plain"] = f["pred_adj"] ** 2 * f["baseline"]
            out[COMBO] = f
            print(
                f"  {bucket:14s} {COMBO:12s} {len(f):5d} rows  ({COMBO_WEIGHT} x lstm + {1 - COMBO_WEIGHT} x {REFERENCE})"
            )
    for label, path in src.items():
        r = sts.load_forecasts(Path(path))
        if r is None:
            continue
        r = r[r["hhmm"] == "16:00"]
        r["pred_plain"] = r["pred_adj"] ** 2 * r["baseline"]
        out[label] = r
        print(f"  {bucket:14s} {label:12s} {len(r):5d} rows  {path}")
    return out


# ---------------------------------------------------------------- trade
DECK = pd.read_parquet(base.DECK).sort_index() if base.DECK.is_file() else None


def trade_daily(pred: pd.Series) -> pd.DataFrame:
    """Daily sign(s) straddle returns -- score_linear_subsection.trade_1530's logic, per day."""
    assert DECK is not None
    days = pd.DatetimeIndex(pd.to_datetime(DECK.index))
    f = pred[pred.index.strftime("%H:%M") == "16:00"].copy()
    f.index = f.index.normalize()
    f = f.reindex(days)
    ok = f.notna().to_numpy()
    q = np.where(f.to_numpy(float) > DECK["iv_var"].to_numpy(float), 1.0, -1.0)[ok]
    dk = DECK[ok]
    ask = (dk["ask_c"] + dk["ask_p"]).to_numpy(float)
    bid = (dk["bid_c"] + dk["bid_p"]).to_numpy(float)
    ex, r = dk["exit"].to_numpy(float), dk["R"].to_numpy(float)
    return pd.DataFrame(
        {
            "q": q,
            "mid": q * r,
            "crossed": q * np.where(q > 0, ex / ask - 1.0, ex / bid - 1.0),
        },
        index=days[ok],
    )


def sharpe(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return x.mean(axis=axis) / x.std(axis=axis, ddof=1) * base.ANN


def boot_idx(n: int) -> np.ndarray:
    """The research scorer's day-block bootstrap draws (seed, block, B from score_linear_subsection)."""
    return asl.circular_block_bootstrap_idx(
        np.random.default_rng([base.BOOT_SEED, n]), n, base.BOOT_BLOCK, base.BOOT_B
    )


def ci(v: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(v, [2.5, 97.5])
    return float(lo), float(hi)


# ---------------------------------------------------------------- main
def md_table(df: pd.DataFrame, digits: int = 4) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cells.append(
                f"{v:.{digits}f}" if isinstance(v, (float, np.floating)) else str(v)
            )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/linear_subsection_lstm")
    ap.add_argument(
        "--linear-root", action="append", default=None, help="repeatable or comma list"
    )
    ap.add_argument("--tree-root", default="results/linear_subsection_trees")
    ap.add_argument("--tuned-root", default="results/linear_subsection_trees_tuned")
    ap.add_argument("--tw", type=int, default=TW)
    ap.add_argument("--out", default=None, help="default: --root")
    ap.add_argument("--min-n", type=int, default=100)
    a = ap.parse_args()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    root = resolve(a.root)
    out = resolve(a.out) if a.out else root
    out.mkdir(parents=True, exist_ok=True)
    lin_roots = sts.split_roots(a.linear_root)
    print(
        f"lstm root {root}; linear roots {[str(r) for r in lin_roots]}; trees {a.tree_root} / {a.tuned_root}"
    )
    gates = GATES
    q_rows: list[dict] = []
    fs_rows: list[dict] = []
    vs_rows: list[dict] = []
    tr_rows: list[dict] = []
    tv_rows: list[dict] = []
    hp_rows: list[pd.DataFrame] = []
    seed_rows: list[dict] = []
    for bucket in bly.BUCKETS:
        fc = load_bucket(bucket, a, lin_roots)
        lstm_labels = [lab for lab in LSTM_LABELS.values() if lab in fc]
        focus = lstm_labels + ([COMBO] if COMBO in fc else [])
        if not lstm_labels:
            print(f"note: no LSTM arm for {bucket}")
            continue
        # a. QLIKE on the rows every forecast of the bucket shares
        common = None
        for r in fc.values():
            idx = r.dropna(subset=["pred_clock"]).index
            common = idx if common is None else common.intersection(idx)
        assert common is not None
        for label, r in fc.items():
            s = r.loc[common]
            for smp, m in (
                ("all common", np.ones(len(s), bool)),
                ("all common, deck period", sts.in_deck(s.index)),
            ):
                ss = s[m]
                lq = slc.qlike(ss["true_raw"], ss["pred_clock"])
                q_rows.append(
                    {
                        "bucket": bucket,
                        "forecast": label,
                        "sample": smp,
                        "n": len(ss),
                        "first": str(ss.index.min().date()),
                        "last": str(ss.index.max().date()),
                        "QLIKE_clock": float(lq.mean()),
                        "QLIKE_as_scored": float(
                            slc.qlike(ss["true_raw"], ss["pred_raw"]).mean()
                        ),
                        "QLIKE_plain": float(
                            slc.qlike(ss["true_raw"], ss["pred_plain"]).mean()
                        ),
                        "max_day_QLIKE_clock": float(lq.max()),
                        "max_day_QLIKE_plain": float(
                            slc.qlike(ss["true_raw"], ss["pred_plain"]).max()
                        ),
                    }
                )
        # a2. the adjusted scale: accuracy, calibration, similarity to the ridge
        for label, r in fc.items():
            s = r.loc[common]
            pa, ta = s["pred_adj"].to_numpy(float), s["true_adj"].to_numpy(float)
            row = {
                "bucket": bucket,
                "forecast": label,
                "n": len(s),
                "mse_adj": float(np.mean((ta - pa) ** 2)),
                "corr_with_target": float(np.corrcoef(pa, ta)[0, 1]),
                "mz_slope": float(np.polyfit(pa, ta, 1)[0]),
                "sd_forecast": float(pa.std(ddof=1)),
                "corr_shifted_one_session": float(np.corrcoef(pa[:-1], ta[1:])[0, 1]),
            }
            if REFERENCE in fc:
                pr = fc[REFERENCE].loc[common, "pred_adj"].to_numpy(float)
                row["corr_with_ridge_forecast"] = float(np.corrcoef(pa, pr)[0, 1])
            fs_rows.append(row)
        # b. paired QLIKE: LSTM minus every comparator (and the two LSTM rules)
        for lab in focus:
            for comp, r in fc.items():
                if comp == lab or (comp in focus and lab != "lstm"):
                    continue
                keys = {
                    "bucket": bucket,
                    "lstm": lab,
                    "comparator": comp,
                    "segment": SEG,
                    "train_win": a.tw,
                }
                vs_rows += sts.paired(
                    fc[lab], r, keys, ("lstm", "comparator"), a.min_n, gates
                )
        # c. the 15:30 trade
        daily = (
            {lab: trade_daily(r["pred_clock"].dropna()) for lab, r in fc.items()}
            if DECK is not None
            else {}
        )
        if daily and min(len(d) for d in daily.values()) < sts.MIN_DECK_DAYS:
            print(
                f"skip trade {bucket}: fewer than {sts.MIN_DECK_DAYS} deck days for some forecast"
            )
            daily = {}
        if daily:
            for lab, d in daily.items():
                ref = base.trade_1530(fc[lab]["pred_clock"].dropna())
                gap = max(
                    abs(ref["Sharpe_mid"] - float(sharpe(d["mid"].to_numpy()))),
                    abs(ref["Sharpe_crossed"] - float(sharpe(d["crossed"].to_numpy()))),
                )
                gates.append(
                    {
                        "bucket": bucket,
                        "check": f"trade rebuild vs trade_1530 ({lab})",
                        "n": len(d),
                        "max_rel": gap,
                        "ok": bool(gap < TRADE_GATE_ABS),
                    }
                )
            days = sorted({len(d) for d in daily.values()})
            same_days = len(days) == 1 and all(
                d.index.equals(daily[lstm_labels[0]].index) for d in daily.values()
            )
            gates.append(
                {
                    "bucket": bucket,
                    "check": "every trade row on the same deck days",
                    "n": days[0],
                    "max_rel": float(len(days) - 1),
                    "ok": bool(same_days),
                }
            )
            n = len(daily[lstm_labels[0]])
            idx = boot_idx(n)
            for lab, d in daily.items():
                row = {
                    "bucket": bucket,
                    "forecast": lab,
                    "deck_days": len(d),
                    "pct_buy": float(100 * (d["q"] > 0).mean()),
                }
                for leg in ("mid", "crossed"):
                    x = d[leg].to_numpy(float)
                    lo, hi = ci(sharpe(x[idx], axis=1))
                    row |= {
                        f"Sharpe_{leg}": float(sharpe(x)),
                        f"Sharpe_{leg}_lo": lo,
                        f"Sharpe_{leg}_hi": hi,
                    }
                tr_rows.append(row)
            for lab in focus:
                for comp, d in daily.items():
                    if comp == lab or not d.index.equals(daily[lab].index):
                        continue
                    row = {
                        "bucket": bucket,
                        "lstm": lab,
                        "comparator": comp,
                        "days": len(d),
                        "same_position_pct": float(
                            100 * (d["q"] == daily[lab]["q"]).mean()
                        ),
                    }
                    for leg in ("mid", "crossed"):
                        xa, xb = daily[lab][leg].to_numpy(float), d[leg].to_numpy(float)
                        diff = sharpe(xa[idx], axis=1) - sharpe(xb[idx], axis=1)
                        lo, hi = ci(diff)
                        row |= {
                            f"dSharpe_{leg}": float(sharpe(xa) - sharpe(xb)),
                            f"dSharpe_{leg}_lo": lo,
                            f"dSharpe_{leg}_hi": hi,
                        }
                    tv_rows.append(row)
        # d. the hyperparameter path
        arm = bly.lstm_arm_path(root, bucket, a.tw)
        if arm is not None:
            tt = arm.with_name(f"tune_trace_{SEG}.csv")
            if tt.is_file():
                t = pd.read_csv(tt)
                hp_rows.append(
                    t[
                        [
                            "bucket",
                            "rule",
                            "tune_idx",
                            "forecast_date",
                            "seq_len",
                            "hidden",
                            "dropout",
                            "lr",
                            "epochs",
                            "epochs_mean",
                            "epochs_cap_hits",
                            "val_mse",
                            "val_qlike",
                        ]
                    ]
                )
            # e. single networks against the seed average
            npz = arm.with_name(f"lstm_{SEG}.npz")
            if npz.is_file() and DECK is not None:
                with np.load(npz, allow_pickle=False) as z:
                    seeds = np.asarray(z["pred_adj_seeds"])
                    zdates = pd.to_datetime(np.asarray(z["date"]).astype(str))
                    meta = json.loads(str(z["meta"]))
                raw = sts.read_adj(arm)
                for j in range(seeds.shape[1] + 1):
                    r = raw.copy()
                    if j < seeds.shape[1]:
                        r["pred_adj"] = (
                            pd.Series(seeds[:, j], index=zdates)
                            .reindex(r.index)
                            .to_numpy()
                        )
                        r["e2"] = (r["true_adj"] - r["pred_adj"]) ** 2
                        who = f"seed {meta['grid']['seeds'][j]}"
                    else:
                        who = "seed average (of record)"
                    f = slc.causal_forecasts(r)
                    s = f.loc[common]
                    d = trade_daily(f["pred_clock"].dropna())
                    enough = len(d) >= sts.MIN_DECK_DAYS
                    seed_rows.append(
                        {
                            "bucket": bucket,
                            "network": who,
                            "n": len(s),
                            "QLIKE_clock": float(
                                slc.qlike(s["true_raw"], s["pred_clock"]).mean()
                            ),
                            "deck_days": len(d),
                            "Sharpe_mid": float(sharpe(d["mid"].to_numpy()))
                            if enough
                            else np.nan,
                            "Sharpe_crossed": float(sharpe(d["crossed"].to_numpy()))
                            if enough
                            else np.nan,
                        }
                    )

    def write(rows, name: str) -> pd.DataFrame:
        tab = (
            pd.concat(rows, ignore_index=True)
            if rows and isinstance(rows[0], pd.DataFrame)
            else pd.DataFrame(rows)
        )
        if tab.empty:
            print(f"no rows for {name}; not written")
        else:
            tab.to_csv(out / name, index=False)
            print(f"wrote {out / name} ({len(tab)} rows)")
        return tab

    tq = write(q_rows, "lstm_qlike_1600.csv")
    tf = write(fs_rows, "lstm_fitspace_1600.csv")
    tv = write(vs_rows, "lstm_vs_comparators.csv")
    tt = write(tr_rows, "lstm_trade_1530.csv")
    tvs = write(tv_rows, "lstm_trade_vs.csv")
    hp = write(hp_rows, "lstm_hyperparameter_path.csv")
    ts = write(seed_rows, "lstm_seed_spread.csv")
    share = pd.DataFrame()
    if not hp.empty:
        share = (
            hp.melt(
                id_vars=["bucket", "rule"],
                value_vars=["seq_len", "hidden", "dropout", "lr"],
                var_name="axis",
                value_name="value",
            )
            .groupby(["bucket", "rule", "axis", "value"])
            .size()
            .rename("tuning_points")
            .reset_index()
        )
        share["share"] = share["tuning_points"] / share.groupby(
            ["bucket", "rule", "axis"]
        )["tuning_points"].transform("sum")
        write([share], "lstm_hyperparameter_share.csv")
    tg = write(gates, "lstm_join_gates.csv")

    md = [
        "# LSTM 16:00 bar -- score tables (written by experiments/score_lstm_subsection.py)",
        "",
    ]
    if not tq.empty:
        md += [
            "## QLIKE on the stamps every forecast of the bucket shares (clock = research scorer)",
            "",
            md_table(tq[tq["sample"] == "all common"].drop(columns=["sample"])),
            "",
        ]
    if not tf.empty:
        md += [
            "## Adjusted scale (the fitted target), same stamps",
            "",
            md_table(tf, 4),
            "",
        ]
    if not tv.empty:
        sel = tv[(tv["back_transform"] == "clock") & (tv["sample"] == "common")]
        md += [
            "## Paired QLIKE, LSTM minus comparator (clock, common stamps; day-block 95 % interval)",
            "",
            md_table(
                sel[
                    [
                        "bucket",
                        "lstm",
                        "comparator",
                        "n",
                        "QLIKE_lstm",
                        "QLIKE_comparator",
                        "pct",
                        "ci_lo",
                        "ci_hi",
                        "dm_stat",
                    ]
                ],
                5,
            ),
            "",
        ]
    if not tt.empty:
        md += [
            "## 15:30 sign(s) straddle trade, 866 deck days (95 % day-block intervals)",
            "",
            md_table(tt, 3),
            "",
        ]
    if not tvs.empty:
        md += [
            "## Sharpe difference, LSTM minus comparator, same days",
            "",
            md_table(tvs, 3),
            "",
        ]
    if not share.empty:
        md += [
            "## Chosen hyperparameters: share of tuning points",
            "",
            md_table(share, 3),
            "",
        ]
    if not ts.empty:
        md += ["## Single networks vs the seed average", "", md_table(ts, 4), ""]
    if not tg.empty:
        md += [f"## Gates: {len(tg)} checked, {int((~tg['ok']).sum())} failed", ""]
    (out / "lstm_score_tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    if not tg.empty and (~tg["ok"]).any():
        print(tg[~tg["ok"]].to_string(index=False))
        sys.exit(f"{int((~tg['ok']).sum())} gate(s) failed -- see lstm_join_gates.csv")


if __name__ == "__main__":
    main()
