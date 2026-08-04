"""Door-3 quantification: EV of a ZERO-MARKET-EDGE trader at a futures eval-prop (Topstep 50K Combine
mechanics, as verified in eval_sim.adp.simulate_topstep 2026-07), where the only informational asset is
a volatility forecast. Prices the eval contract's convexity: P(pass), time-to-resolution, funded-stage
extraction, and the marginal value of forecast quality / sigma-timing.

Design (pre-registered):
  * Env = real SPX session bars (9:00-15:00, 30-min, 1993-2024), sumret DEMEANED globally -> zero-drift
    by construction (the measured signed channel is null, commit 5881635). Direction fixed long
    (short variant for the champion; demeaned skew asymmetry is the only difference).
  * Day game at bar resolution: DLL day-stop $1,000 with two execution models bracketing reality:
    'barclose' = stop fills at the breaching bar's close (full overshoot, conservative);
    'exact'    = stop fills exactly at -DLL (perfect management, optimistic). Day-target tau fills at
    tau (resting limit). Same-bar tie -> stop.
  * Multi-day mechanics = simulate_topstep's verified rules, applied at bar-informed day resolution:
    floor = min(M_eod - D, initial) [EOD ratchet + breakeven lock], intraday floor touch = FAIL,
    consistency: effective target = max(target, best_day/0.5), min 2 trading days.
    Topstep 50K params: initial 50k, target +3k, trailing D 2k, DLL 1k, consistency 0.5.
  * Policies: fixed-notional grid; vol-target grid L_d = v / sigma_hat_d (MES granularity $30k, cap
    5 ES = $1.5M = the 50K account's position limit); day-target grid; sigma-timing (trade only when
    sigma_hat above/below trailing-1000d quantile); sigma-quality {HAR, naive lag-RV, perfect-foresight}.
  * Funded stage: same env+policy, no target, floor locks at initial; withdraw down to a $2k cushion
    whenever cushion >= $2.9k. Analytic bound for ANY zero-edge policy: E[total withdrawn] <= initial
    cushion = $2,000 (optional stopping); the sim shows the friction drag below it.
  * Attempts start every 5th day (overlapping-start dependence acknowledged); eras reported.

Writes results/door3/summary.json.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

OUT = "results/door3"
INITIAL = 50_000.0
TARGET = 3_000.0
DD = 2_000.0
DLL = 1_000.0
CONSIST = 0.5
MIN_DAYS = 2
MAX_CAL = 200
ES_NOTIONAL = 300_000.0  # ~1 ES at SPX 6000; sizing granularity 1 MES = /10
MES = ES_NOTIONAL / 10.0
L_CAP = 5 * ES_NOTIONAL  # 50K account position limit (5 minis)
START_EVERY = 5
BURN = 1_000  # days reserved for sigma-hat burn-in before first attempt


def load_days():
    df = pd.read_parquet(
        "data/core_stats.parquet", columns=["endbartime", "sumret", "sumret2"]
    )
    t = pd.to_datetime(df["endbartime"])
    m = (t.dt.hour >= 9) & (t.dt.hour <= 15)
    d = pd.DataFrame(
        {
            "date": t[m].dt.date.to_numpy(),
            "r": df.loc[m.to_numpy(), "sumret"].fillna(0.0).to_numpy(),
            "rv": df.loc[m.to_numpy(), "sumret2"].fillna(0.0).to_numpy(),
        }  # sumret2 = bar realized variance
    )
    d["r"] = d["r"] - d["r"].mean()  # zero-drift env by construction
    cums, rvs, dates = [], [], []
    for dt, g in d.groupby("date", sort=True):
        if len(g) < 8:
            continue
        cums.append(np.cumsum(g["r"].to_numpy()))
        rvs.append(float(g["rv"].sum()))
        dates.append(dt)
    return cums, np.array(rvs), np.array([str(x) for x in dates])


def har_sigma(rv: np.ndarray) -> np.ndarray:
    """Causal daily HAR on log day-RV, expanding OLS refit every 63 days, 500-day burn-in.
    Returns sigma_hat (day-return sd forecast) with smearing correction; NaN before burn-in."""
    lrv = np.log(np.maximum(rv, 1e-12))
    n = len(lrv)
    x1 = np.concatenate([[np.nan], lrv[:-1]])
    x5 = pd.Series(lrv).rolling(5).mean().shift(1).to_numpy()
    x22 = pd.Series(lrv).rolling(22).mean().shift(1).to_numpy()
    X = np.column_stack([np.ones(n), x1, x5, x22])
    ok = np.isfinite(X).all(1)
    pred = np.full(n, np.nan)
    coef, s2 = None, 0.0
    for i in range(500, n):
        if (i - 500) % 63 == 0:
            tr = ok.copy()
            tr[i:] = False
            A, yy = X[tr], lrv[tr]
            coef, *_ = np.linalg.lstsq(A, yy, rcond=None)
            s2 = float(np.mean((yy - A @ coef) ** 2))
        if ok[i] and coef is not None:
            pred[i] = X[i] @ coef + 0.5 * s2
    return np.sqrt(np.exp(pred))


def day_game(c_dollar: np.ndarray, tau: float | None):
    """One trading day at notional already applied (c_dollar = L * cumsum(r)).
    Returns (pnl_barclose, pnl_exact, mincum_barclose, mincum_exact)."""
    i_dn = np.argmax(c_dollar <= -DLL) if (c_dollar <= -DLL).any() else len(c_dollar)
    i_up = len(c_dollar)
    if tau is not None and (c_dollar >= tau).any():
        i_up = int(np.argmax(c_dollar >= tau))
    if i_dn <= i_up and i_dn < len(c_dollar):  # stop first (tie -> stop)
        pnl_bc = float(c_dollar[i_dn])
        pnl_ex = -DLL
        mc_bc = float(c_dollar[: i_dn + 1].min())
    elif i_up < len(c_dollar):  # limit at tau fills at tau
        pnl_bc = pnl_ex = float(tau)  # type: ignore[arg-type]
        mc_bc = float(c_dollar[: i_up + 1].min())
    else:
        pnl_bc = pnl_ex = float(c_dollar[-1])
        mc_bc = float(c_dollar.min())
    return pnl_bc, pnl_ex, mc_bc, max(mc_bc, -DLL)


def run_config(
    cums,
    sigma,
    name,
    *,
    v_target=None,
    L_fixed=None,
    tau=1500.0,
    timing=None,
    direction=1.0,
    starts=None,
):
    """Simulate eval attempts. timing = (mode, q): trade only if sigma-hat above ('top') / below
    ('bot') its trailing-1000d q-quantile. Returns dict of metrics for both executions."""
    n = len(cums)
    sig = pd.Series(sigma)
    thr = sig.rolling(1000).quantile(timing[1]).shift(1).to_numpy() if timing else None
    # precompute per-day (L, day-game outputs) once
    L_day = np.zeros(n)
    for i in range(n):
        if not np.isfinite(sigma[i]) and v_target is not None:
            continue
        L = (
            L_fixed
            if L_fixed is not None
            else np.clip(
                np.round(v_target / max(sigma[i], 1e-6) / MES) * MES, MES, L_CAP
            )
        )
        if timing and not (
            np.isfinite(thr[i])
            and (
                (timing[0] == "top" and sigma[i] >= thr[i])
                or (timing[0] == "bot" and sigma[i] <= thr[i])
            )
        ):
            L = 0.0
        L_day[i] = L
    games = [
        day_game(direction * L_day[i] * cums[i], tau) if L_day[i] > 0 else None
        for i in range(n)
    ]
    cap_bind = (
        float(np.mean([L_day[i] >= L_CAP for i in range(n) if L_day[i] > 0]))
        if L_day.any()
        else 0.0
    )

    res = {}
    for ex, (ip, im) in (("barclose", (0, 2)), ("exact", (1, 3))):
        outs, durs, tdays, passdays = [], [], [], []
        for s in starts:
            W, M_eod, best, td = INITIAL, INITIAL, 0.0, 0
            floor = INITIAL - DD
            out, cal = 0, 0
            for i in range(s, min(s + MAX_CAL, n)):
                cal += 1
                g = games[i]
                if g is None:
                    continue
                td += 1
                if W + g[im] <= floor:  # intraday floor touch
                    out = 2
                    break
                pnl = g[ip]
                W += pnl
                best = max(best, pnl)
                if W <= floor:
                    out = 2
                    break
                eff = max(INITIAL + TARGET, INITIAL + best / CONSIST)
                if W >= eff and td >= MIN_DAYS:
                    out = 1
                    break
                M_eod = max(M_eod, W)
                floor = min(M_eod - DD, INITIAL)
            outs.append(out)
            durs.append(cal)
            tdays.append(td)
            if out == 1:
                passdays.append(cal)
        o = np.array(outs)
        res[ex] = {
            "p_pass": float((o == 1).mean()),
            "p_fail": float((o == 2).mean()),
            "p_timeout": float((o == 0).mean()),
            "cal_days_mean": float(np.mean(durs)),
            "cal_days_pass": float(np.mean(passdays)) if passdays else None,
            "trading_days_mean": float(np.mean(tdays)),
            "p_pass_last1000": float((o[-len(o) // 6 :] == 1).mean()),
        }
    return {"name": name, "cap_bind_frac": cap_bind, **res}


def funded_sim(cums, sigma, *, v_target, tau, starts):
    """Funded-stage extraction for the zero-edge trader: no target, floor = min(M_eod - DD, INITIAL)
    (locks at initial), withdraw to a $2k cushion whenever cushion >= $2.9k, run to bust or 500 days."""
    n = len(cums)
    L_day = np.zeros(n)
    for i in range(n):
        if np.isfinite(sigma[i]):
            L_day[i] = np.clip(
                np.round(v_target / max(sigma[i], 1e-6) / MES) * MES, MES, L_CAP
            )
    games = [
        day_game(L_day[i] * cums[i], tau) if L_day[i] > 0 else None for i in range(n)
    ]
    res = {}
    for ex, (ip, im) in (("barclose", (0, 2)), ("exact", (1, 3))):
        extracted, lifetimes = [], []
        for s in starts:
            W, M_eod, X = INITIAL, INITIAL, 0.0
            floor = INITIAL - DD
            days = 0
            for i in range(s, min(s + 500, n)):
                days += 1
                g = games[i]
                if g is None:
                    continue
                if W + g[im] <= floor or W + g[ip] <= floor:
                    break
                W += g[ip]
                M_eod = max(M_eod, W)
                floor = min(M_eod - DD, INITIAL)
                cush = W - floor
                if cush >= 2_900.0:
                    X += cush - 2_000.0
                    W -= cush - 2_000.0
            extracted.append(X)
            lifetimes.append(days)
        e = np.array(extracted)
        res[ex] = {
            "extract_mean": float(e.mean()),
            "extract_median": float(np.median(e)),
            "p_zero": float((e == 0).mean()),
            "days_mean": float(np.mean(lifetimes)),
        }
    return res


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    cums, rv, dates = load_days()
    n = len(cums)
    sig_har = har_sigma(rv)
    sig_naive = np.sqrt(np.concatenate([[np.nan], rv[:-1]]))
    sig_perfect = np.sqrt(rv)
    starts = list(range(BURN, n - MAX_CAL, START_EVERY))
    print(f"days={n} attempts={len(starts)} span={dates[0]}..{dates[-1]}", flush=True)

    rows = []
    for v in (1_000, 2_000, 4_000, 8_000, 16_000):
        rows.append(
            run_config(
                cums, sig_har, f"vt{v // 1000}k_tau1.5k", v_target=v, starts=starts
            )
        )
    for tau in (1_000.0, 3_000.0, None):
        rows.append(
            run_config(
                cums, sig_har, f"vt4k_tau{tau}", v_target=4_000, tau=tau, starts=starts
            )
        )
    for L in (1, 2, 4):
        rows.append(
            run_config(
                cums,
                sig_har,
                f"fixed{L}ES_tau1.5k",
                L_fixed=L * ES_NOTIONAL,
                starts=starts,
            )
        )
    rows.append(
        run_config(cums, sig_naive, "vt4k_naive_sigma", v_target=4_000, starts=starts)
    )
    rows.append(
        run_config(
            cums, sig_perfect, "vt4k_perfect_sigma", v_target=4_000, starts=starts
        )
    )
    for mode, q, tag in (
        ("top", 0.5, "top50"),
        ("top", 0.8, "top20"),
        ("bot", 0.2, "bot20"),
    ):
        rows.append(
            run_config(
                cums,
                sig_har,
                f"vt4k_{tag}",
                v_target=4_000,
                timing=(mode, q),
                starts=starts,
            )
        )
    rows.append(
        run_config(
            cums, sig_har, "vt4k_short", v_target=4_000, direction=-1.0, starts=starts
        )
    )

    funded = funded_sim(cums, sig_har, v_target=4_000, tau=1_500.0, starts=starts)

    with open(f"{OUT}/summary.json", "w") as fh:
        json.dump(
            {"configs": rows, "funded": funded, "n_days": n, "n_attempts": len(starts)},
            fh,
            indent=1,
        )
    hdr = f"{'config':<22}{'P(pass) bc/ex':<18}{'cal_days':<10}{'tdays':<8}{'cap%':<6}"
    print(hdr, flush=True)
    for r in rows:
        print(
            f"{r['name']:<22}{r['barclose']['p_pass']:.3f} / {r['exact']['p_pass']:.3f}     "
            f"{r['barclose']['cal_days_mean']:<10.1f}{r['barclose']['trading_days_mean']:<8.1f}"
            f"{r['cap_bind_frac']:<6.2f}",
            flush=True,
        )
    print("funded:", json.dumps(funded), flush=True)


if __name__ == "__main__":
    main()
