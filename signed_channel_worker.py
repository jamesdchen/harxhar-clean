"""Signed-return channel test: does the SAME info set that carries the VRP edge (HAR core +
sparse-flow survivor bundles + hour dummies, every-bar ridge tw=48000 alpha=1) predict the SIGNED
bar return? This decides whether ANY delta-one (futures-prop-legal) expression of the edge exists:
E[PnL] of a predictable-position futures strategy loads only on E[r|F], never on E[RV|F].

Pre-registered, two cells only (core = HAR-only control, all8 = full survivor superset), no tuning,
no selection. Evaluation excludes bars whose sumret was fabricated by the ffill(0) (uses the
pre-fill availability indicator). Writes results/signed_channel/{core,all8}.json + daily .npz."""

from __future__ import annotations

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


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    df, raw, bundles, har_cols, HD, rv, iv_bar, hour, date = build()
    y = df["sumret"].to_numpy(float)
    fabricated = (
        df["sumret_ind"].to_numpy(float) > 0
    )  # pre-fill missing -> target is a filled 0
    sess = (hour >= 9) & (hour <= 15)

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

        mm = np.isfinite(pred) & np.isfinite(y) & sess & ~fabricated
        r, f = y[mm], pred[mm]
        n = int(mm.sum())
        corr = float(np.corrcoef(f, r)[0, 1])
        # HAC(5) t-stat for E[f*r] > 0 (per-bar covariance = the tradable moment)
        z = f * r - (f * r).mean()
        gam = [float((z[k:] * z[: len(z) - k]).mean()) for k in range(6)]
        lrv = gam[0] + 2 * sum((1 - k / 6) * gam[k] for k in range(1, 6))
        t_cov = float((f * r).mean() / np.sqrt(lrv / len(z)))

        res: dict = {"cell": name, "n_bars": n, "oos_corr": corr, "t_cov_hac5": t_cov}
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
        print(
            f"[{name}] n={n} corr={corr:+.5f} t_cov={t_cov:+.2f} "
            f"sign gross sh={res['sign']['gross']['sh_full']:+.4f} "
            f"net(1tick) sh={res['sign']['net_1tick']['sh_full']:+.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
