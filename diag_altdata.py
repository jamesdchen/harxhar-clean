"""Alt-data representation diagnostics: what does exog information add beyond
own-history — and in which encoding?

Motivation (see writeup/spectral_graph_diagnostics_2026-07-30.md): own-history
is linearly exhausted — ridge on 5 multiscale RV levels dominates every
neighborhood method and its residual is unpredictable from any own-history
representation. The remaining structure is (a) whatever exogenous series know
that the RV path doesn't, and (b) the shared top-decile spike underprediction.
This script measures both, per alt-data block and per *encoding*, with the
same fixed-window / chronological-split protocol as the earlier diagnostics.

Key design point: the production pipeline encodes each exog independently as
``adj_<col>_ma_<w>`` rolling means. It can never form CROSS-SERIES coordinates
(implied-minus-realized, term-structure slope, jump fraction, buy-sell
imbalance, EW/VW breadth). Those relational encodings are exactly the
candidates tested here, against the pipeline-style level encodings of the
same raw series.

For each block B the incremental test is nested-ridge:
    MSE( ridge(Z_base) )  vs  MSE( ridge(Z_base ++ B) )
overall and in the top decile of the true target (the spike bucket), with
per-column standardization fit on the train split only. All features are
strictly causal: rolling stats are shifted one bar before sampling.

Default window ends 2020-07-01: every alt series is live and the test tail
(last 20%) contains the COVID vol regime — the hardest spike test available.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from diag_spectral_graph import TRAIN_WIN, W, delay_views, prepare_series, represent, view_index

LEVEL_WINDOWS = (1, 48, 240)     # bars: last bar, 1 day, 1 week
SLOW_WINDOWS = (8, 48, 240)      # for noisier flow/sentiment series
RV_22D = 22 * 48                 # realized leg of the variance risk premium


def _roll(s: pd.Series, w: int) -> pd.Series:
    """Causal rolling mean: window of the w bars strictly before each row."""
    return s.rolling(w, min_periods=max(1, w // 4)).mean().shift(1)


def _z(s: pd.Series, w: int = 240) -> pd.Series:
    """Causal rolling z-score (surprise vs own recent history)."""
    m = s.rolling(w, min_periods=w // 4).mean().shift(1)
    sd = s.rolling(w, min_periods=w // 4).std().shift(1)
    return (s.shift(1) - m) / sd.replace(0, np.nan)


def _safe_log(s: pd.Series) -> pd.Series:
    return np.log(s.clip(lower=1e-12))


def build_blocks(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Alt-data feature blocks, each a DataFrame aligned to df rows."""
    needed = [
        "vix", "vix3m", "vvix", "RV", "sumbipow", "sumret2",
        "buyturnover_ewstock", "sellturnover_ewstock", "turnover_ewstock",
        "stocktwits_sentiment", "stocktwits_attention",
        "voldemand_spx_open_and_close", "sumret2_ewstock", "sumret2_vwstock",
    ]
    r = {c: df[c].astype(float).ffill() for c in needed}
    out: dict[str, dict[str, pd.Series]] = {}

    # -- VIX level (pipeline-style encoding of the same series) -----------
    lvix = _safe_log(r["vix"])
    out["vix_level"] = {f"lvix_ma{w}": _roll(lvix, w) for w in LEVEL_WINDOWS}

    # -- VIX structure (relational: slope, vol-of-vol, VRP) ---------------
    ts_slope = _safe_log(r["vix3m"]) - lvix
    vvix_r = _safe_log(r["vvix"]) - lvix
    vrp = 2.0 * lvix - _safe_log(_roll(df["RV"].astype(float), RV_22D))
    out["vix_struct"] = {
        **{f"ts_slope_ma{w}": _roll(ts_slope, w) for w in LEVEL_WINDOWS},
        **{f"vvix_r_ma{w}": _roll(vvix_r, w) for w in LEVEL_WINDOWS},
        **{f"vrp_ma{w}": _roll(vrp, w) for w in LEVEL_WINDOWS},
    }

    # -- Jump fraction of SPY variance (bipower vs total) -----------------
    jr = (1.0 - r["sumbipow"] / r["sumret2"].replace(0, np.nan)).clip(-1, 1)
    out["jump"] = {f"jumpfrac_ma{w}": _roll(jr, w) for w in SLOW_WINDOWS}

    # -- Cross-sectional flow (EW stocks): signed turnover imbalance ------
    imb = (r["buyturnover_ewstock"] - r["sellturnover_ewstock"]) / r[
        "turnover_ewstock"
    ].replace(0, np.nan)
    out["flow"] = {
        **{f"imb_ma{w}": _roll(imb, w) for w in SLOW_WINDOWS},
        "turnover_z": _z(_safe_log(r["turnover_ewstock"])),
    }

    # -- Sentiment (stocktwits) -------------------------------------------
    out["sent"] = {
        **{f"sent_ma{w}": _roll(r["stocktwits_sentiment"], w) for w in SLOW_WINDOWS},
        "attention_z": _z(_safe_log(r["stocktwits_attention"].clip(lower=1e-12) + 1.0)),
    }

    # -- Vol demand (hurdle encoding: occurrence + magnitude) -------------
    vd = r["voldemand_spx_open_and_close"]
    active = (vd.fillna(0.0) != 0.0).astype(float)
    out["voldemand"] = {
        **{f"vd_active_ma{w}": _roll(active, w) for w in SLOW_WINDOWS},
        "vd_mag_ma48": _roll(np.sign(vd.fillna(0.0)) * np.sqrt(vd.abs()), 48),
    }

    # -- Breadth: small-cap vs large-cap vol spread -----------------------
    breadth = _safe_log(r["sumret2_ewstock"]) - _safe_log(r["sumret2_vwstock"])
    out["breadth"] = {f"breadth_ma{w}": _roll(breadth, w) for w in SLOW_WINDOWS}

    return {name: pd.DataFrame(cols) for name, cols in out.items()}


def fit_eval(Z_tr, Z_te, t_tr, t_te, dec_te, alpha: float = 1.0) -> dict:
    model = Ridge(alpha=alpha).fit(Z_tr, t_tr)
    p = model.predict(Z_te)
    top = dec_te == 9
    return {
        "mse": float(np.mean((t_te - p) ** 2)),
        "mse_top": float(np.mean((t_te[top] - p[top]) ** 2)),
        "bias_top": float(np.mean(p[top] - t_te[top])),
    }


def standardize_split(M: np.ndarray, split: int) -> np.ndarray:
    mu = M[:split].mean(axis=0)
    sd = M[:split].std(axis=0)
    sd[sd < 1e-12] = 1.0
    return (M - mu) / sd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default="data")
    ap.add_argument("--out", default="results/spectral_graph_diagnostics")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--end-date", default="2020-07-01",
                    help="window end (train+test); COVID lands in the test tail")
    args = ap.parse_args()

    print("Preparing series (pipeline-identical prep)...")
    X, y, dates, df = prepare_series(args.data_path, return_df=True)
    ts = pd.to_datetime(dates)
    t_end = int(np.searchsorted(ts.values, np.datetime64(args.end_date)))
    t0 = t_end - TRAIN_WIN
    if t0 < 0:
        raise SystemExit(f"window before data start (t_end={t_end})")
    y_win = y[t0:t_end]
    df_win = df.iloc[t0:t_end].reset_index(drop=True)
    print(f"window {ts.iloc[t0].date()} .. {ts.iloc[t_end - 1].date()}")

    idx = view_index(len(y_win), args.stride)
    targets = y_win[idx].astype(np.float64)
    Z_ms = represent(delay_views(y_win, idx, W), "multiscale").astype(np.float64)

    blocks = build_blocks(df_win)
    block_mats = {}
    for name, B in blocks.items():
        M = B.to_numpy(dtype=np.float64)[idx]
        nanfrac = float(np.isnan(M).mean())
        M = np.nan_to_num(M, nan=0.0)
        block_mats[name] = M
        print(f"block {name:11s} dim={M.shape[1]:2d} nan%={nanfrac:.3f}")

    split = int(0.8 * len(idx))
    t_tr, t_te = targets[:split], targets[split:]
    dec_te = np.clip((np.argsort(np.argsort(t_te)) * 10) // len(t_te), 0, 9)
    print(f"test tail {ts.iloc[t0 + idx[split]].date()} .. {ts.iloc[t_end - 1].date()}"
          f"  (const mse {np.mean((t_te - t_tr.mean())**2):.5f})")

    Zb = standardize_split(Z_ms, split)
    rows = []
    base = fit_eval(Zb[:split], Zb[split:], t_tr, t_te, dec_te)
    rows.append({"block": "base(multiscale)", "dim": Zb.shape[1], **base})
    print(f"\n{'block':26s} {'mse':>8s} {'d_mse%':>7s} {'mse_top':>8s} {'d_top%':>7s} {'bias_top':>8s}")
    print(f"{'base(multiscale)':26s} {base['mse']:8.5f} {'':7s} {base['mse_top']:8.5f} {'':7s} {base['bias_top']:8.3f}")

    def try_block(name: str, M: np.ndarray):
        Z = np.concatenate([Zb, standardize_split(M, split)], axis=1)
        res = fit_eval(Z[:split], Z[split:], t_tr, t_te, dec_te)
        d = 100 * (res["mse"] / base["mse"] - 1)
        dt = 100 * (res["mse_top"] / base["mse_top"] - 1)
        rows.append({"block": name, "dim": Z.shape[1], **res})
        print(f"{name:26s} {res['mse']:8.5f} {d:+6.2f}% {res['mse_top']:8.5f} {dt:+6.2f}% {res['bias_top']:8.3f}")

    for name, M in block_mats.items():
        try_block(f"+{name}", M)
    try_block("+ALL", np.concatenate(list(block_mats.values()), axis=1))
    struct = np.concatenate(
        [block_mats["vix_struct"], block_mats["jump"], block_mats["breadth"]], axis=1
    )
    try_block("+struct(vix+jump+breadth)", struct)

    os.makedirs(args.out, exist_ok=True)
    out_csv = os.path.join(args.out, f"altdata_{args.end_date}.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nArtifacts -> {out_csv}")


if __name__ == "__main__":
    main()
