"""Re-score the subsection arms with ONE causal back-transform for every arm.

Why.  ``src.evaluation.metrics.apply_duan_smearing`` turns an adjusted-scale
forecast f into a variance forecast (f^2 + smear) x baseline with
smear = mean((y - f)^2) over the arm's WHOLE out-of-sample run.  That term (a)
looks ahead, and (b) is estimated per arm: a one-bar arm gets a term specific to
its own clock, the pooled arm one term averaged over all 48 bars.  The scorer in
``score_linear_subsection.py`` compares the spec's own ``pred_raw`` columns, so
its "own coefficients per bar" gain mixes two things: different coefficients,
and a clock-specific second-moment calibration the pooled arm was never given.

This script rebuilds every arm's variance forecast from its adjusted-scale
columns with a causal term, s = the mean squared adjusted-scale error over the
previous SMEAR_W sessions (minimum SMEAR_MIN), lagged one session:

  clock   s from the arm's own errors AT THAT BAR LABEL
  pooled  s from the arm's errors over all of its bars (pooled arm only)

and reports, bar by bar, on identical bars:

  coefficients   one-bar arm (clock) against pooled arm (clock): the same
                 calibration on both sides, so only the coefficients differ;
  package        one-bar arm (clock) against pooled arm (pooled): what a desk
                 running the pooled model as it stands would gain;
  calibration    pooled arm (clock) against pooled arm (pooled): what the
                 clock-specific term alone is worth;
  as scored      the spec's own look-ahead columns, for reference.

Usage:  python experiments/score_linear_subsection_causal.py --bucket baseline
            [--root results/linear_subsection]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "notebooks", ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import score_linear_subsection as base  # noqa: E402

SMEAR_W, SMEAR_MIN = 250, 63  # the production recalibration's window and warm-up
BLOWN = 1.0  # a mean QLIKE above one marks a blown arm (the lam2 = 0 pathology)


def load_adj(
    root: Path, bucket: str, seg: str, est: str, tw: int
) -> pd.DataFrame | None:
    d = root / bucket / seg / est / f"tw{tw}" / "causal_tune_linear" / est / bucket
    f = d / ("results.csv" if seg == "none" else f"results_{seg}.csv")
    if not f.exists():
        return None
    r = pd.read_csv(f, parse_dates=["date"]).set_index("date").sort_index()
    ok = (r["true_adj"] > 0) & (r["true_raw"] > 0)
    r = r[ok]
    r["baseline"] = r["true_raw"] / r["true_adj"] ** 2
    r["e2"] = (r["true_adj"] - r["pred_adj"]) ** 2
    r["hhmm"] = r.index.strftime("%H:%M")
    r["day"] = r.index.normalize()
    return r


def causal_forecasts(r: pd.DataFrame) -> pd.DataFrame:
    """Add pred_clock and pred_pooled: (f^2 + s) x baseline with causal s."""
    s_clock = r.groupby("hhmm")["e2"].transform(
        lambda v: v.rolling(SMEAR_W, min_periods=SMEAR_MIN).mean().shift(1)
    )
    daily = r.groupby("day")["e2"].mean()
    s_day = daily.rolling(SMEAR_W, min_periods=SMEAR_MIN).mean().shift(1)
    s_pool = r["day"].map(s_day)
    out = r.copy()
    out["pred_clock"] = (r["pred_adj"] ** 2 + s_clock) * r["baseline"]
    out["pred_pooled"] = (r["pred_adj"] ** 2 + s_pool) * r["baseline"]
    return out


def qlike(y: pd.Series, f: pd.Series) -> pd.Series:
    ratio = y / f
    return ratio - np.log(ratio) - 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", default="baseline")
    ap.add_argument("--root", default="results/linear_subsection")
    a = ap.parse_args()
    root = ROOT / a.root
    pd.set_option("display.width", 250)
    arms = sorted(
        (p.parts[-4], p.parts[-3], int(p.parts[-2][2:]))
        for p in (root / a.bucket).glob("*/*/tw*/causal_tune_linear")
    )
    pooled: dict[tuple[str, int], pd.DataFrame] = {}
    rows, trades = [], []
    for seg, est, tw in arms:
        if seg != "none" and not seg.startswith("bar") and seg != "rth":
            continue  # blocks are a bias-variance variant; the question is per bar (and the whole session, "rth")
        raw = load_adj(root, a.bucket, seg, est, tw)
        if raw is None:
            continue
        r = causal_forecasts(raw)
        if seg == "none":
            pooled[(est, tw)] = r
        last = r[r["hhmm"] == "16:00"]
        if len(last):
            for kind in ("pred_clock", "pred_pooled", "pred_raw"):
                if seg != "none" and kind == "pred_pooled":
                    continue
                f = last[kind].dropna()
                trades.append(
                    {
                        "segment": seg,
                        "estimator": est,
                        "train_win": tw,
                        "back_transform": kind.replace("pred_", ""),
                    }
                    | base.trade_1530(f)
                )
    for seg, est, tw in arms:
        if not seg.startswith("bar"):
            continue
        ref = pooled.get((est, tw))
        raw = load_adj(root, a.bucket, seg, est, tw)
        if ref is None or raw is None:
            continue
        mine = causal_forecasts(raw)
        j = mine.join(
            ref[["pred_clock", "pred_pooled", "pred_raw"]], how="inner", rsuffix="_P"
        )
        j = j.dropna(subset=["pred_clock", "pred_clock_P", "pred_pooled_P"])
        assert j["hhmm"].nunique() == 1, seg
        deck = (j.index >= base.DECK_START) & (j.index <= base.DECK_END + " 23:59")
        y = j["true_raw"]
        pairs = {
            "coefficients": (qlike(y, j["pred_clock"]), qlike(y, j["pred_clock_P"])),
            "package": (qlike(y, j["pred_clock"]), qlike(y, j["pred_pooled_P"])),
            "calibration": (qlike(y, j["pred_clock_P"]), qlike(y, j["pred_pooled_P"])),
            "as scored": (qlike(y, j["pred_raw"]), qlike(y, j["pred_raw_P"])),
        }
        for smp, m in (("common", np.ones(len(j), bool)), ("deck period", deck)):
            if m.sum() < 100:
                continue
            for name, (la, lb) in pairs.items():
                d = (la - lb)[m]
                lo, hi = base.day_block_ci(d)
                rows.append(
                    {
                        "segment": seg,
                        "bar_end": j["hhmm"].iloc[0],
                        "estimator": est,
                        "train_win": tw,
                        "sample": smp,
                        "comparison": name,
                        "n": int(m.sum()),
                        "QLIKE_a": float(la[m].mean()),
                        "QLIKE_b": float(lb[m].mean()),
                        "diff": float(d.mean()),
                        "pct": float(100 * d.mean() / lb[m].mean()),
                        "ci_lo": lo,
                        "ci_hi": hi,
                        "improves": bool(hi < 0.0),
                        "worse": bool(lo > 0.0),
                        "blown": bool(la[m].mean() > BLOWN or lb[m].mean() > BLOWN),
                    }
                )
    tab = pd.DataFrame(rows)
    tab.to_csv(root / a.bucket / "subsection_vs_pooled_causal.csv", index=False)
    pd.DataFrame(trades).to_csv(root / a.bucket / "trade_1530_causal.csv", index=False)
    ok = tab[~tab["blown"]]
    summ = (
        ok.groupby(["sample", "train_win", "estimator", "comparison"])
        .agg(
            mean_pct=("pct", "mean"),
            better=("improves", "sum"),
            worse=("worse", "sum"),
            bars=("pct", "size"),
        )
        .round(2)
    )
    print(f"bucket {a.bucket}: {len(tab)} cells, {int(tab['blown'].sum())} blown")
    print(summ.to_string())
    print(pd.DataFrame(trades).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
