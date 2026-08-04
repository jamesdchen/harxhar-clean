"""Signed-return channel test: does the SAME info set that carries the VRP edge (HAR core +
sparse-flow survivor bundles + hour dummies, every-bar ridge tw=48000 alpha=1) predict the SIGNED
bar return? This decides whether ANY delta-one (futures-prop-legal) expression of the edge exists:
E[PnL] of a predictable-position futures strategy loads only on E[r|F], never on E[RV|F].

Pre-registered, two cells only (core = HAR-only control, all8 = full survivor superset), no tuning,
no selection. Evaluation excludes bars whose sumret was fabricated by the ffill(0) (uses the
pre-fill availability indicator). Session (9-15) metrics include sign/linear strategy Sharpes with
1-tick cost. PANEL EXTENSION: the same predictions are also scored per bar-hour WINDOW (open 9,
mid 10-14, close 15, after-hours 16-19, overnight 20-8, all 24h) so the null's coverage is the whole
clock, not just the session -- after-hours/overnight are the windows an overnight-permitting futures
account could trade. 6 windows x 2 cells of correlations: read t-stats against a Bonferroni-ish
threshold ~2.9, not 2.0. Writes results/signed_channel/{core,all8}.json + daily .npz."""

from __future__ import annotations

import _bootstrap  # noqa: F401  (repo root + siblings on sys.path)

import json
import os

import numpy as np
import pandas as pd

from src.models.ridge import fit_predict_ridge
from vrp_pnl_worker import SURV
from vrp_screen_worker import build

OUT = "results/signed_channel"
TICK_COST = (
    0.25 / 6000.0
)  # one ES tick crossed per position change, as a return fraction


def daily_sharpe(pnl_day: np.ndarray) -> dict[str, float]:
    return {
        "sh_full": float(pnl_day.mean() / pnl_day.std()),
        "sh_last1000": float(pnl_day[-1000:].mean() / pnl_day[-1000:].std()),
        "t_hac": float(pnl_day.mean() / (pnl_day.std() / np.sqrt(len(pnl_day)))),
    }


def corr_block(f: np.ndarray, r: np.ndarray) -> dict[str, float]:
    """OOS corr + HAC(5) t for E[f*r] > 0 (the tradable moment)."""
    corr = float(np.corrcoef(f, r)[0, 1])
    z = f * r - (f * r).mean()
    gam = [float((z[k:] * z[: len(z) - k]).mean()) for k in range(6)]
    lrv = gam[0] + 2 * sum((1 - k / 6) * gam[k] for k in range(1, 6))
    return {
        "n": len(f),
        "oos_corr": corr,
        "t_cov_hac5": float((f * r).mean() / np.sqrt(lrv / len(z))),
    }


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    df, raw, bundles, har_cols, HD, rv, iv_bar, hour, date = build()
    y = df["sumret"].to_numpy(float)
    fabricated = (
        df["sumret_ind"].to_numpy(float) > 0
    )  # pre-fill missing -> target is a filled 0
    windows = {
        "session9_15": (hour >= 9) & (hour <= 15),
        "open9": hour == 9,
        "mid10_14": (hour >= 10) & (hour <= 14),
        "close15": hour == 15,
        "afterhours16_19": (hour >= 16) & (hour <= 19),
        "overnight20_8": (hour >= 20) | (hour <= 8),
        "all24": np.ones_like(hour, dtype=bool),
    }

    for name, cols in (
        ("core", har_cols),
        ("all8", har_cols + [c for b in SURV for c in bundles[b]]),
    ):
        X = np.column_stack([df[cols].fillna(0).to_numpy(float), HD])
        p = fit_predict_ridge(
            X,
            y,
            train_win_periods=48_000,
            hyperparams=dict(alpha=1.0, _refit_frequency=1),
        )
        pred = np.full(len(df), np.nan)
        pred[len(df) - len(p) :] = p
        base = np.isfinite(pred) & np.isfinite(y) & ~fabricated

        res: dict = {"cell": name, "windows": {}}
        for wname, wmask in windows.items():
            mm = base & wmask
            if mm.sum() >= 1000:
                res["windows"][wname] = corr_block(pred[mm], y[mm])

        mm = base & windows["session9_15"]
        r, f = y[mm], pred[mm]
        res.update(n_bars=int(mm.sum()), **corr_block(f, r))
        for tag, pos in (("sign", np.sign(f)), ("linear", f / np.std(f))):
            pnl = pos * r
            cost = np.abs(np.diff(pos, prepend=0.0)) * TICK_COST
            d = (
                pd.DataFrame({"d": date[mm], "g": pnl, "n": pnl - cost})
                .groupby("d")
                .sum()
            )
            res[tag] = {
                "gross": daily_sharpe(d["g"].to_numpy()),
                "net_1tick": daily_sharpe(d["n"].to_numpy()),
                "turnover_per_bar": float(np.abs(np.diff(pos, prepend=0.0)).mean()),
                "breakeven_cost_ret": float(
                    pnl.mean() / max(np.abs(np.diff(pos, prepend=0.0)).mean(), 1e-12)
                ),
            }
            if tag == "sign":
                np.savez_compressed(
                    f"{OUT}/{name}_daily.npz",
                    days=np.array([str(x) for x in d.index]),
                    gross=d["g"].to_numpy(),
                    net=d["n"].to_numpy(),
                )
        with open(f"{OUT}/{name}.json", "w") as fh:
            json.dump(res, fh, indent=1)
        wl = "  ".join(
            f"{k}:{v['oos_corr']:+.4f}(t{v['t_cov_hac5']:+.1f})"
            for k, v in res["windows"].items()
        )
        print(f"[{name}] {wl}", flush=True)


if __name__ == "__main__":
    main()
