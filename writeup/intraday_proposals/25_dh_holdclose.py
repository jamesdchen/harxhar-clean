"""Proposal 25 — delta-hedged t→T: one straddle held to close, Δ rebalanced
every 30 minutes on vendor S.

Unhedged t→T is a terminal-move bet (holdclose notebook). This isolates
(1/2)(v_I-v_M)S^2 Γ by flattening residual delta along the path.

Frozen K from entry. Δ from Black-76 (r=0) at each later stamp using that
stamp's vendor hourly IV as remaining vol and shrinking h. 0.5 bp on |n|S
at each rebalance and at flatten. Sign(s) from remaining = matched (same sign).

Run:  python writeup/intraday_proposals/25_dh_holdclose.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import erf

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "notebooks"))
import atm_straddle_lib as asl  # noqa: E402

REPO = asl.find_repo(Path(__file__).resolve().parent)
HOLD = REPO / "results" / "atm_straddle_intraday_holdclose"
CACHE = HOLD / "cache"
OUT = HOLD / "proposals" / "25"
CLOSE = "15:30"
ANN = float(np.sqrt(asl.PERIODS_PER_YEAR))
UNDERLYING_COST_BP = 0.5
PROFILE_MIN_DAYS = 63
HOURS_YEAR = 252.0 * 6.5


def _sh(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) < 2 or d.std(ddof=1) == 0:
        return float("nan")
    return float(d.mean() / d.std(ddof=1) * ANN)


def _cdf(z):
    return 0.5 * (1.0 + erf(z / np.sqrt(2.0)))


def package_delta(s, F, Kc, Kp):
    s = np.asarray(s, float)
    F = np.asarray(F, float)
    shape = np.broadcast_shapes(s.shape, F.shape)
    Kc = np.broadcast_to(np.asarray(Kc, float), shape)
    Kp = np.broadcast_to(np.asarray(Kp, float), shape)
    F = np.broadcast_to(F, shape)
    s = np.broadcast_to(s, shape)
    out = np.full(shape, np.nan)
    pos = (s > 0) & np.isfinite(s) & (F > 0) & (Kc > 0) & (Kp > 0)
    if pos.any():
        ss, FF, kc, kp = s[pos], F[pos], Kc[pos], Kp[pos]
        d1c = (np.log(FF / kc) + 0.5 * ss * ss) / ss
        d1p = (np.log(FF / kp) + 0.5 * ss * ss) / ss
        out[pos] = _cdf(d1c) + _cdf(d1p) - 1.0
    return out


def w_mean_of_shares(wide, clocks):
    pi = pd.DataFrame(index=wide.index, columns=clocks, dtype=float)
    for i, c in enumerate(clocks):
        rem = wide[clocks[i:]].sum(axis=1)
        pi[c] = wide[c] / rem.replace(0.0, np.nan)
    return pi.expanding(min_periods=PROFILE_MIN_DAYS).mean().shift(1)


def sign_pos(s):
    p = np.where(np.isfinite(s), np.sign(s), 0.0)
    return np.where((p == 0) & np.isfinite(s), -1.0, p)


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    cands = sorted(CACHE.glob("trade_*.parquet"))
    pkg = pd.read_parquet(cands[-1])
    print(f"[{time.time() - t0:5.1f}s] cache {len(pkg):,}")
    panel = asl.load_yhat_panel(asl.yhat_paths(REPO)["blk2"])
    pkg = pkg.copy()
    pkg["t"] = pd.to_datetime(pkg["timestamp"], utc=True)
    clocks = sorted(pkg["hhmm"].unique())
    n_rem = {c: len(clocks) - i for i, c in enumerate(clocks)}
    pkg["h_rem"] = pkg["hhmm"].map(n_rem).astype(float) * 0.5
    pkg["V_M"] = pkg["iv_hourly"].astype(float) ** 2 * pkg["h_rem"]
    pm = panel.set_index("t")[["rv_hat", "rv_raw", "in_fit"]].reset_index()
    pm["t"] = pd.to_datetime(pm["t"], utc=True) - pd.Timedelta(minutes=30)
    work = pkg.merge(pm, on="t", how="left").dropna(subset=["R", "rv_hat"]).copy()
    pf = panel.loc[panel["in_fit"].to_numpy(dtype=bool)].copy()
    clock = pd.to_datetime(pf["t"], utc=True).dt.tz_convert(
        "America/New_York"
    ) - pd.Timedelta(minutes=30)
    pf["pdate"] = clock.dt.normalize().dt.tz_localize(None)
    pf["phhmm"] = clock.dt.strftime("%H:%M")
    wide = pf.pivot_table(
        index="pdate", columns="phhmm", values="rv_raw", aggfunc="mean"
    ).sort_index()
    wide = wide.reindex(columns=clocks)
    w = w_mean_of_shares(wide, clocks)
    mi = pd.MultiIndex.from_arrays([work["date"], work["hhmm"]])
    work["w_slice"] = w.stack().reindex(mi).to_numpy()
    work["s"] = work["rv_hat"] - work["w_slice"] * work["V_M"]
    work = work.sort_values(["date", "hhmm"]).reset_index(drop=True)
    print(
        f"[{time.time() - t0:5.1f}s] work {len(work):,} days {work['date'].nunique()}"
    )

    S_grid = work.pivot_table(index="date", columns="hhmm", values="S", aggfunc="first")
    S_grid = S_grid.reindex(columns=clocks)
    IV_grid = work.pivot_table(
        index="date", columns="hhmm", values="iv_hourly", aggfunc="first"
    )
    IV_grid = IV_grid.reindex(columns=clocks)
    h_row = np.array([n_rem[c] * 0.5 for c in clocks])
    dates = work["date"].to_numpy()
    t_idx = pd.Index(clocks).get_indexer(work["hhmm"].to_numpy())
    Sg = S_grid.loc[dates].to_numpy(float)
    IVg = IV_grid.loc[dates].to_numpy(float)
    ST = work["S_close"].to_numpy(float)
    Kc = work["K_c"].to_numpy(float)
    Kp = work["K_p"].to_numpy(float)
    n, m = Sg.shape
    col = np.arange(m)[None, :]
    active = col >= t_idx[:, None]
    nxt = np.full_like(Sg, np.nan)
    nxt[:, :-1] = Sg[:, 1:]
    last = np.where(active, col, -1).max(axis=1)
    for j in range(m):
        nxt[last == j, j] = ST[last == j]
    tot = np.where(IVg > 0, IVg * np.sqrt(h_row[None, :]), np.nan)
    tot = np.where(active, tot, np.nan)
    dlt = package_delta(tot, Sg, Kc[:, None], Kp[:, None])
    dlt = np.where(active & np.isfinite(Sg) & np.isfinite(dlt), dlt, 0.0)
    dS = np.where(active & np.isfinite(Sg) & np.isfinite(nxt), nxt - Sg, 0.0)
    q = sign_pos(work["s"].to_numpy(float))[:, None]
    n_pos = -q * dlt
    hedge = (n_pos * dS).sum(axis=1)
    prev = np.zeros_like(dlt)
    prev[:, 1:] = dlt[:, :-1]
    turn = np.abs(n_pos - (-q * prev)) * np.where(np.isnan(Sg), 0.0, Sg)
    cost = (turn * UNDERLYING_COST_BP * 1e-4).sum(axis=1)
    opt = q[:, 0] * (work["exit"].to_numpy(float) - work["entry"].to_numpy(float))
    entry = work["entry"].to_numpy(float)
    r_uh = work["R"].to_numpy(float) * q[:, 0]
    r_dh = (opt + hedge) / entry
    r_dh_c = (opt + hedge - cost) / entry
    r_as_uh = work["R"].to_numpy(float) * -1.0
    n_as = -(-1.0) * dlt
    hedge_as = (n_as * dS).sum(axis=1)
    r_as_dh = ((-1.0) * (work["exit"].to_numpy(float) - entry) + hedge_as) / entry
    r_long_dh = (
        work["exit"].to_numpy(float) - entry + ((-dlt) * dS).sum(axis=1)
    ) / entry
    is_close = work["hhmm"].to_numpy() == CLOSE
    q_hyb = np.where(is_close, sign_pos(work["s"].to_numpy(float)), -1.0)
    r_hyb_dh = q_hyb * r_long_dh
    r_hyb_uh = q_hyb * work["R"].to_numpy(float)
    spot_ret = (work["S_close"].to_numpy(float) - work["S"].to_numpy(float)) / work[
        "S"
    ].to_numpy(float)

    def daily(x):
        return pd.Series(x, index=work.index).groupby(work["date"]).sum()

    rows = []
    for name, ser in (
        ("sign(s) unhedged t→T", r_uh),
        ("sign(s) DH t→T 0bp", r_dh),
        ("sign(s) DH t→T 0.5bp", r_dh_c),
        ("always short unhedged t→T", r_as_uh),
        ("always short DH t→T 0bp", r_as_dh),
        ("hybrid AS then sign(s) unhedged", r_hyb_uh),
        ("hybrid AS then sign(s) DH 0bp", r_hyb_dh),
    ):
        d = daily(ser)
        rows.append(
            {
                "rule": name,
                "Sharpe": _sh(d),
                "mean/day": float(d.mean()),
                "n_days": int(d.notna().sum()),
            }
        )
        print(f"{name:32s}  Sharpe {_sh(d):+.3f}  mean/day {float(d.mean()):+.4f}")
    pd.DataFrame(rows).to_csv(OUT / "25_dh_holdclose.csv", index=False)

    print("15:30 only (paper trade):")
    for name, ser in (("unhedged", r_uh), ("DH 0bp", r_dh)):
        d = daily(np.where(is_close, ser, 0.0))
        print(f"  {name:12s} {_sh(d):+.3f}")
    print("corr(always-short DH, spot return entry→close) — ~0 if hedge works:")
    for c in clocks:
        m = work["hhmm"].to_numpy() == c
        a, b = r_as_dh[m], spot_ret[m]
        ok = np.isfinite(a) & np.isfinite(b)
        print(
            f"  {c}  {float(np.corrcoef(a[ok], b[ok])[0, 1]):+.3f}  mean DH {float(np.mean(a[ok])):+.3f}"
        )
    print(f"wrote {OUT}  elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
