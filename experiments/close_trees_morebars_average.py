"""Average the 16:00 forecasts of the bar-count ladder (close study, 2026-10-04).

The user's question: what if we average the pooled predictions?  The bar-count ladder
(experiments/close_trees_morebars.py) fitted one LightGBM for each pool of the last N bars
ending 16:00 (N = 1, 2, 3, 4, 5, 7, 13), each forecasting the 16:00 row.  Here those stored
forecasts are averaged with equal weights on the model's adjusted scale (the forecast of
sqrt(RV / diurnal scale)), and every average goes through the same scorer as a single arm:
the master table's research convention (16:00-bar recalibration (f^2 + s) B with s from
the average's own errors, QLIKE and the 15:30 sign(s) straddle on the 866 trade days),
rows from the trees' first forecast row (2019-01) on.  Point estimates; DM = Diebold-
Mariano on daily QLIKE, HAC t = Newey-West t of the daily mid P&L difference.

Two controls separate the effect of pooling different bars from plain variance reduction:
averaging two seeds of the same pool (random_state 42 and 43), and averaging the best
single pool with itself across seeds.  The ridge with the HAR + calendar backbone
unpenalized (results/close_studies_2026-10-03/exog_penalty, arm ridge_bb0) is the linear
reference, alone and in an equal-weight average with the tree average.

Writes results/close_studies_2026-10-03/trees_morebars/average.csv.
Usage:  python experiments/close_trees_morebars_average.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "experiments", REPO / "notebooks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import close_trees_morebars as mb  # noqa: E402  (arms, loader, compare; imports close_trees_datasize)
import dense_vs_sparse_1530 as dvs  # noqa: E402

LINEAR_REF = (
    REPO
    / "results"
    / "close_studies_2026-10-03"
    / "exog_penalty"
    / "_work"
    / "runs"
    / "ridge_bb0.npz"
)
LADDER = [f"lgbm_bars{n}_r{n * mb.SESSIONS}" for n in mb.LADDER]
XGB = [f"xgb_bars{n}_r{n * mb.SESSIONS}" for n in mb.LADDER]
SEED = mb.SEED_ALT


def main() -> None:
    S = mb.c.sources()
    n_fc = S["n_fc"]
    sub = np.arange(mb.c.TREE_START, n_fc)
    dz = {k: v[sub] for k, v in S["dz"].items()}
    pred = {a: r["pred"] for a in mb.ARMS if (r := mb.load_arm(a)) is not None}
    missing = [a for a in LADDER + XGB if a not in pred]
    assert not missing, missing
    pred["ridge_bb0"] = np.load(LINEAR_REF)["pred"]
    for a, p in pred.items():
        assert p.shape == (n_fc,) and np.isfinite(p[sub]).all(), a

    def avg(names: list[str]) -> np.ndarray:
        return np.mean([pred[a] for a in names], axis=0)

    seed = lambda n: f"lgbm_bars{n}_r{n * mb.SESSIONS}_seed{SEED}"  # noqa: E731
    combos: dict[str, tuple[str, np.ndarray]] = {
        "lgbm, 1 bar (control)": ("single", pred[mb.CTRL]),
        "lgbm, 2 bars": ("single", pred[mb.TWO]),
        "lgbm, 4 bars (best single pool)": ("single", pred[LADDER[3]]),
        "average of lgbm pools N = 1, 2, 3, 4, 5, 7, 13": (
            "average of pools",
            avg(LADDER),
        ),
        "average of lgbm pools N = 1 .. 5": ("average of pools", avg(LADDER[:5])),
        "average of lgbm pools N = 1 .. 4": ("average of pools", avg(LADDER[:4])),
        "average of lgbm pools N = 2 .. 5": ("average of pools", avg(LADDER[1:5])),
        "average of lgbm pools N = 3, 4": ("average of pools", avg(LADDER[2:4])),
        "average of lgbm 4 bars, seeds 42 and 43": (
            "seed average",
            avg([LADDER[3], seed(4)]),
        ),
        "average of lgbm 2 bars, seeds 42 and 43": (
            "seed average",
            avg([LADDER[1], seed(2)]),
        ),
        "average of lgbm pools N = 1, 2, 3, 4, 13, seeds 42 and 43": (
            "average of pools and seeds",
            avg(
                [
                    *[LADDER[i] for i in (0, 1, 2, 3, 6)],
                    *[seed(n) for n in (1, 2, 3, 4, 13)],
                ]
            ),
        ),
        "average of lgbm and xgb pools, all N": ("average of pools", avg(LADDER + XGB)),
        "ridge, backbone unpenalized (linear reference)": ("linear", pred["ridge_bb0"]),
        "equal weights: ridge backbone unpenalized + average of lgbm pools, all N": (
            "linear + trees",
            0.5 * pred["ridge_bb0"] + 0.5 * avg(LADDER),
        ),
        "equal weights: ridge backbone unpenalized + lgbm 4 bars": (
            "linear + trees",
            0.5 * pred["ridge_bb0"] + 0.5 * pred[LADDER[3]],
        ),
    }
    labels = list(combos)
    frames = [dvs.research_frame(dz, combos[k][1][sub]) for k in labels]
    P = dvs.deck_panel(frames, dvs.deck_frame())
    pt = dvs.point(P)
    col = {k: i for i, k in enumerate(labels)}
    refs = {
        "control": "lgbm, 1 bar (control)",
        "4 bars": "lgbm, 4 bars (best single pool)",
        "ridge": "ridge, backbone unpenalized (linear reference)",
    }
    rows = []
    for k in labels:
        r = dict(
            forecast=k,
            kind=combos[k][0],
            qlike=pt["ql"][col[k]],
            sharpe_mid=pt["sh"][col[k]],
            sharpe_crossed=pt["shx"][col[k]],
            pct_buy=100 * P["buy"][col[k]],
            n_days=P["ql"].shape[0],
        )
        for tag, ref in refs.items():
            if ref == k:
                continue
            cmp = mb.compare(P, col[k], col[ref])
            r[f"dm_vs_{tag}"], r[f"dm_p_vs_{tag}"], r[f"t_hac_vs_{tag}"] = (
                cmp["dm"],
                cmp["dm_p"],
                cmp["t_hac"],
            )
        rows.append(r)
    out = pd.DataFrame(rows)
    f = mb.OUT / "average.csv"
    out.to_csv(f, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 80)
    show = [
        "forecast",
        "qlike",
        "sharpe_mid",
        "dm_vs_control",
        "dm_p_vs_control",
        "dm_vs_4 bars",
        "dm_p_vs_4 bars",
        "dm_vs_ridge",
        "dm_p_vs_ridge",
        "t_hac_vs_ridge",
    ]
    print(out[show].round(4).to_string(index=False))
    print(f"wrote {f}")


if __name__ == "__main__":
    main()
