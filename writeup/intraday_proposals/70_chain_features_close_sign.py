"""70 - does the 0DTE chain itself know the sign of RV - IV at the close?

The 15:30 trade compares a variance forecast with the 15:30 straddle's implied
slice and takes sign(s).  The chain's implied is the benchmark, not a regressor,
and as a regressor it could not move the sign anyway (a positive weight on the
implied leaves sign(forecast - implied) unchanged).  What the chain CAN add is
information about whether today's price is high or low: how the slice moved into
15:30, the 0DTE skew, what the session has realized against it so far, how rich
the premium is against its own recent past, and the calendar.  None of it is in
the panel-based forecasts, all of it is known at 15:30, and the chain runs to
2025-12-31 -- a year and a half of sessions no forecast in this repository has
seen, so a chain-only rule can be tested truly out of sample.

Features at 15:30 (chain only; 1,279 sessions 2020-01-03 .. 2025-12-31):
  x1  slice change     log( implied per-bar variance at 15:30 / at 15:00 )
                       (15:00's total variance covers two bars: halved)
  x2  skew             (put leg total vol - call leg total vol) / package total
                       vol, each leg inverted from its 15:30 midpoint
  x3  realized so far  log( mean squared 30-min log return 10:00->15:30 /
                       implied per-bar variance at 15:30 )
  x4  premium richness log(package mid / spot) minus its trailing 63-session
                       median (strictly prior sessions)
  x5  month-end        the last session of the month (the override day)
  x6  third Friday     the monthly expiration session
plus, on the 866 deck days only,
  s   the live-feasible per-bar ridge's signal, rv_hat - implied slice
Target: 1[R > 0], the 15:30 nearest-OTM straddle's midpoint return to settlement.

  A  each feature alone, all 1,279 sessions and by sample: AUC for R > 0, mean
     R in its top and bottom quintile, correlation with s on the deck days.
  B  deck days, causal: logistic regression of 1[R > 0] on [s, x1..x6],
     expanding window, 252-session warm-up, refit every 21 sessions, features
     standardized on the fit window; buy if p > 1/2 else short; crossed fills.
     Paired against R0 = sign(s) (the live-feasible per-bar ridge) with the
     circular block bootstrap (21 sessions, 2,000 draws).  Beside it, the same
     logistic on s alone -- what recalibrating s is worth without the chain.
  C  chain only, [x1..x6]: (i) FROZEN -- fitted once on the 866 deck days and
     applied to the 413 holdout sessions 2024-05-01 .. 2025-12-31 that no
     forecast has seen; (ii) expanding causal over all 1,279 sessions.  Sharpe
     at mid and crossed, hit, mean per active day; on the holdout, paired
     against always-short and against a random-sign placebo with the same
     number of buy days (2,000 draws, seed 0).

Written before running:
  (1) the chain ADDS to the forecast if B's crossed Sharpe minus R0's has a
      paired-bootstrap lower bound above zero;
  (2) a chain-only sign rule is SUPPORTED out of sample if C(i)'s holdout
      crossed Sharpe exceeds always-short's with the paired interval above zero
      AND beats 95% of the random-sign placebo.
Everything else is reported, not claimed.

GATES  the chain-built 15:30 straddle return equals the deck's R on all 866
       days (1e-6); the live-feasible per-bar ridge's sign(s) reproduces 1.683
       mid / 1.217 crossed; the two 15:30 legs pulled here sum to proposal 43's
       package midpoint (1e-6) -- the skew is re-inverted from those midpoints,
       and its agreement with the vendor's censored per-leg quotes is reported.

Run:  python writeup/intraday_proposals/70_chain_features_close_sign.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "notebooks", ROOT / "writeup"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import atm_straddle_lib as asl  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = ROOT / "results" / "atm_straddle_intraday_holdclose" / "proposals" / "70"
DECK = ROOT / "results" / "atm_straddle_0dte_1530"
CHAIN = ROOT / "data" / "spxw_chain.parquet"
LIVE_TAG = "sub_live_ridge"
DECK_END, OOS_START = "2024-04-30", "2024-05-01"
ANN = float(np.sqrt(asl.PERIODS_PER_YEAR))
WARMUP, REFIT = 252, 21
TRAIL = 63
B, BLOCK, SEED = 2000, 21, 0
FEATS = [
    "x1_slice_change",
    "x2_skew",
    "x3_realized_so_far",
    "x4_premium_rich",
    "x5_month_end",
    "x6_third_friday",
]
GATE_R_TOL = 1e-6
GATE_R0 = (1.683, 1.217)
GATE_R0_TOL = 5e-3
GATE_MID_TOL = (
    1e-6  # the two legs' midpoints must sum to proposal 43's package midpoint
)


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sharpe(x: np.ndarray) -> float:
    return float(np.mean(x) / np.std(x, ddof=1) * ANN)


def auc(score: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(score)
    s, yy = score[m], y[m].astype(bool)
    n1, n0 = yy.sum(), (~yy).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[yy].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def invert_leg_total_vol(price: float, F: float, K: float, is_call: bool) -> float:
    """Black-76 single leg (rate 0): the total vol sigma*sqrt(T) that prices ``price``."""
    intrinsic = max(F - K, 0.0) if is_call else max(K - F, 0.0)
    if not (np.isfinite(price) and price > intrinsic + 1e-9 and F > 0 and K > 0):
        return float("nan")
    lo, hi = 1e-6, 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        d1 = (np.log(F / K) + 0.5 * mid**2) / mid
        d2 = d1 - mid
        v = (
            F * norm.cdf(d1) - K * norm.cdf(d2)
            if is_call
            else K * norm.cdf(-d2) - F * norm.cdf(-d1)
        )
        if v > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def leg_quotes_1530(
    stamp: pd.DataFrame, idx: pd.DatetimeIndex, kc: np.ndarray, kp: np.ndarray
) -> pd.DataFrame:
    ts = sorted(pd.DatetimeIndex(stamp.loc[idx, "15:30"]).dropna().unique())
    q = pd.read_parquet(
        CHAIN,
        columns=["expiration", "timestamp", "strike", "cp", "bid", "ask"],
        filters=[("timestamp", "in", ts)],
    )
    want = pd.DataFrame(
        {
            "date": idx,
            "timestamp": [pd.Timestamp(stamp.loc[d, "15:30"]) for d in idx],
            "K_c": kc,
            "K_p": kp,
        }
    )
    want["timestamp"] = want["timestamp"].astype(q["timestamp"].dtype)
    want["expiration"] = want["date"].astype(q["expiration"].dtype)
    q = q.merge(want, on=["expiration", "timestamp"], how="inner")
    q = q[
        ((q["cp"] == "C") & (q["strike"] == q["K_c"]))
        | ((q["cp"] == "P") & (q["strike"] == q["K_p"]))
    ]
    dead = (q["bid"] == 0) & (q["ask"] == 0)
    q = q[~dead].assign(
        mid=lambda f: 0.5 * (f["bid"].astype(float) + f["ask"].astype(float))
    )
    piv = q.pivot_table(
        index="date", columns="cp", values="mid", aggfunc="first"
    ).reindex(idx)
    return piv.rename(columns={"C": "mid_c", "P": "mid_p"})


def fit_predict(X: np.ndarray, y: np.ndarray, fit: np.ndarray) -> np.ndarray:
    mu, sd = X[fit].mean(axis=0), X[fit].std(axis=0, ddof=1)
    sd = np.where(sd > 0, sd, 1.0)
    clf = LogisticRegression(C=1.0, max_iter=1000)
    clf.fit((X[fit] - mu) / sd, y[fit])
    z = (X - mu) / sd
    fin = np.isfinite(z).all(axis=1)
    out = np.full(len(X), np.nan)
    out[fin] = clf.predict_proba(z[fin])[:, 1]
    return out


def fit_predict_ev(X: np.ndarray, r: np.ndarray, fit: np.ndarray) -> np.ndarray:
    """OLS of the straddle return on the standardized features: E[R | x]."""
    mu, sd = X[fit].mean(axis=0), X[fit].std(axis=0, ddof=1)
    sd = np.where(sd > 0, sd, 1.0)
    z = (X - mu) / sd
    A = np.column_stack([np.ones(int(fit.sum())), z[fit]])
    beta, *_ = np.linalg.lstsq(A, r[fit], rcond=None)
    fin = np.isfinite(z).all(axis=1)
    out = np.full(len(X), np.nan)
    out[fin] = beta[0] + z[fin] @ beta[1:]
    return out


def causal_probs(
    X: np.ndarray, y: np.ndarray, ok: np.ndarray, ev: bool = False
) -> np.ndarray:
    """Expanding fit: refit every REFIT sessions on all prior usable rows (>= WARMUP).

    ``ev=False``: logistic on 1[R > 0] (returns P(R > 0)); ``ev=True``: OLS of R
    itself (returns E[R]), the decision rule for an asymmetric payoff.
    """
    n = len(y)
    p = np.full(n, np.nan)
    usable = np.flatnonzero(ok)
    last_fit = -1
    probs = None
    for k, t in enumerate(usable):
        prior = usable[:k]
        if len(prior) < WARMUP:
            continue
        if last_fit < 0 or k - last_fit >= REFIT:
            fit = np.zeros(n, bool)
            fit[prior] = True
            probs = fit_predict_ev(X, y, fit) if ev else fit_predict(X, y, fit)
            last_fit = k
        assert probs is not None
        p[t] = probs[t]
    return p


def book(
    pos: np.ndarray, r_mid: np.ndarray, r_ask: np.ndarray, r_bid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Position +1 / -1 / 0 -> returns at mid and crossed, per unit premium."""
    mid = pos * r_mid
    crossed = np.where(pos > 0, r_ask, np.where(pos < 0, -r_bid, 0.0))
    return mid, crossed


def summary(mid: np.ndarray, crossed: np.ndarray, pos: np.ndarray) -> dict[str, float]:
    act = pos != 0
    return {
        "days": int(len(pos)),
        "pct_buy": float((pos > 0).mean()),
        "Sharpe_mid": sharpe(mid),
        "Sharpe_crossed": sharpe(crossed),
        "mean_crossed": float(crossed.mean()),
        "hit_crossed": float((crossed[act] > 0).mean()) if act.any() else float("nan"),
        "worst_crossed": float(crossed.min()),
    }


def main() -> None:  # noqa: PLR0915
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    p43 = _load(HERE / "43_causal_entry_over_time.py", "p43_causal_entry")
    p56 = _load(HERE / "56_close_calendar_family.py", "p56_close_calendar")
    p58 = _load(HERE / "58_third_friday_short.py", "p58_third_friday")
    stamp, _ = p43.session_stamps()
    sessions = pd.DatetimeIndex(stamp.index).difference(p43.half_sessions(stamp))
    ch = p43.build_chain(stamp, sessions)
    idx = pd.DatetimeIndex(ch["dates"])
    k30, k00 = p43.CLOCKS.index("15:30"), p43.CLOCKS.index("15:00")
    S, sc = ch["S"], ch["S_close"]
    mid30, bid30 = ch["entry"][:, k30], ch["bid"][:, k30]
    ask30 = 2.0 * mid30 - bid30
    kc, kp = ch["K_c"][:, k30], ch["K_p"][:, k30]
    pay = np.maximum(sc - kc, 0.0) + np.maximum(kp - sc, 0.0)
    r_mid, r_ask, r_bid = pay / mid30 - 1.0, pay / ask30 - 1.0, pay / bid30 - 1.0
    tot30, tot00 = ch["tot"][:, k30], ch["tot"][:, k00]

    # ---------------------------------------------------------------- gates --
    deck = pd.read_parquet(DECK / f"daily_{LIVE_TAG}.parquet").sort_index()
    di = pd.DatetimeIndex(deck.index)
    pos_d = idx.get_indexer(di)
    assert (pos_d >= 0).all()
    dev = float(np.nanmax(np.abs(r_mid[pos_d] - deck["R"].to_numpy(float))))
    assert dev < GATE_R_TOL, dev
    q_live = np.where(deck["signal"].to_numpy(float) > 0, 1.0, -1.0)
    m0, c0 = book(q_live, r_mid[pos_d], r_ask[pos_d], r_bid[pos_d])
    got = (sharpe(m0), sharpe(c0))
    assert (
        abs(got[0] - GATE_R0[0]) < GATE_R0_TOL
        and abs(got[1] - GATE_R0[1]) < GATE_R0_TOL
    ), got
    print(
        f"GATE  chain-built R = deck R on {len(di)} days ({dev:.1e}); live-feasible per-bar ridge sign(s) {got[0]:.3f} / {got[1]:.3f}"
    )

    # ------------------------------------------------------------- features --
    legs = leg_quotes_1530(stamp, idx, kc, kp)
    tot_c = np.array(
        [
            invert_leg_total_vol(p, f, k, True)
            for p, f, k in zip(legs["mid_c"].to_numpy(float), S[:, k30], kc)
        ]
    )
    tot_p = np.array(
        [
            invert_leg_total_vol(p, f, k, False)
            for p, f, k in zip(legs["mid_p"].to_numpy(float), S[:, k30], kp)
        ]
    )
    skew = (tot_p - tot_c) / tot30
    # alignment gate: the two legs pulled here sum to proposal 43's package
    # midpoint on every day both are quoted (same stamp, strikes and spot)
    both = np.isfinite(legs["mid_c"].to_numpy(float)) & np.isfinite(
        legs["mid_p"].to_numpy(float)
    )
    dev_mid = float(
        np.max(
            np.abs((legs["mid_c"] + legs["mid_p"]).to_numpy(float)[both] - mid30[both])
        )
    )
    assert dev_mid < GATE_MID_TOL, dev_mid
    # the deck's iv_c / iv_p are the vendor's QUOTED per-leg implieds, which are
    # bisection-censored (the chain-defects note); the skew here is re-inverted
    # from the midpoints, so their agreement is reported, not gated
    deck_skew = (deck["iv_p"] - deck["iv_c"]).to_numpy(float)
    ok_sk = (
        np.isfinite(skew[pos_d])
        & np.isfinite(deck_skew)
        & ~deck["iv_capped"].to_numpy(bool)
    )
    corr_sk = float(np.corrcoef(skew[pos_d][ok_sk], deck_skew[ok_sk])[0, 1])
    print(
        f"GATE  the two 15:30 legs sum to the package midpoint on {int(both.sum())} of {len(idx)} sessions (max gap {dev_mid:.1e}); "
        f"re-inverted skew vs the vendor's quoted per-leg skew: correlation {corr_sk:.3f} on {int(ok_sk.sum())} uncensored deck days"
    )

    rets = np.diff(np.log(S), axis=1)  # 11 thirty-minute returns 10:00 -> 15:30
    rv_sofar = np.nanmean(rets**2, axis=1)
    cal = p56.session_calendar(pd.DatetimeIndex(stamp.index))
    types = p56.day_types(cal)
    me = (
        pd.Series(types["month-end"].to_numpy(), index=cal)
        .reindex(idx)
        .fillna(False)
        .to_numpy(bool)
    )
    tf = (
        pd.Series(p58.nth_weekday_sessions(cal, 4, 3), index=cal)
        .reindex(idx)
        .fillna(False)
        .to_numpy(bool)
    )
    prem = pd.Series(np.log(mid30 / S[:, k30]), index=idx)
    rich = (prem - prem.rolling(TRAIL, min_periods=TRAIL).median().shift(1)).to_numpy()
    X = pd.DataFrame(
        {
            "x1_slice_change": np.log(tot30**2 / (tot00**2 / 2.0)),
            "x2_skew": skew,
            "x3_realized_so_far": np.log(rv_sofar / tot30**2),
            "x4_premium_rich": rich,
            "x5_month_end": me.astype(float),
            "x6_third_friday": tf.astype(float),
        },
        index=idx,
    )
    y = (r_mid > 0).astype(int)
    ok = np.isfinite(X.to_numpy()).all(axis=1) & np.isfinite(r_mid) & (bid30 > 0)
    holdout = np.asarray(idx >= OOS_START)
    s_live = np.full(len(idx), np.nan)
    s_live[pos_d] = deck["signal"].to_numpy(float)
    print(
        f"features on {int(ok.sum())} of {len(idx)} sessions ({int((ok & ~holdout).sum())} deck, {int((ok & holdout).sum())} holdout); "
        f"P(R > 0) deck {float(y[pos_d].mean()):.3f}, holdout {float(y[holdout & ok].mean()):.3f}"
    )

    # ------------------------------------------------------------- A --------
    rows = []
    for f in FEATS:
        v = X[f].to_numpy(float)
        for lab, m in (("all", ok), ("deck", ok & ~holdout), ("holdout", ok & holdout)):
            vv, rr, yy = v[m], r_mid[m], y[m]
            if f.startswith(("x5", "x6")):
                top, bot = vv > 0.5, vv <= 0.5
            else:
                qs = np.nanquantile(vv, [0.2, 0.8])
                top, bot = vv >= qs[1], vv <= qs[0]
            rows.append(
                {
                    "feature": f,
                    "sample": lab,
                    "n": int(m.sum()),
                    "AUC_for_R>0": auc(vv, yy),
                    "mean_R_top": float(rr[top].mean()) if top.any() else np.nan,
                    "mean_R_bottom": float(rr[bot].mean()) if bot.any() else np.nan,
                    "corr_with_s (deck)": float(
                        np.corrcoef(
                            vv[np.isfinite(s_live[m])],
                            s_live[m][np.isfinite(s_live[m])],
                        )[0, 1]
                    )
                    if lab == "deck"
                    else np.nan,
                }
            )
    a = pd.DataFrame(rows)
    a.to_csv(OUT / "a_features.csv", index=False)
    print(
        "\nA  each chain feature alone: AUC for R > 0, mean R in its top / bottom quintile (flags: on / off)"
    )
    print(a.round(3).to_string(index=False))

    # ------------------------------------------------------------- B --------
    Xd = np.column_stack([s_live, X.to_numpy(float)])
    okd = ok & np.isfinite(s_live)
    p_full = causal_probs(Xd, y, okd)
    p_s = causal_probs(Xd[:, :1], y, okd)
    # EV variants, ADDED AFTER the logistic rows were seen: a probability
    # threshold at 1/2 buys on 6-10% of days against sign(s)'s 42%, because
    # P(R > 0) is 0.38 and the trade's edge is the size of the wins, not their
    # count.  OLS of R on the same features, buy if E[R] > 0.  Reported, not a
    # criterion.
    e_full = causal_probs(Xd, r_mid, okd, ev=True)
    e_s = causal_probs(Xd[:, :1], r_mid, okd, ev=True)
    scored = (
        np.isfinite(p_full)
        & np.isfinite(p_s)
        & np.isfinite(e_full)
        & np.isfinite(e_s)
        & okd
    )
    rows_b = []
    idx_b = asl.circular_block_bootstrap_idx(
        np.random.default_rng([SEED, int(scored.sum())]), int(scored.sum()), BLOCK, B
    )
    q0 = np.where(s_live[scored] > 0, 1.0, -1.0)
    m_r0, c_r0 = book(q0, r_mid[scored], r_ask[scored], r_bid[scored])
    base_b = c_r0[idx_b]
    sr_base = base_b.mean(axis=1) / base_b.std(axis=1, ddof=1) * ANN
    for name, p, thr in (
        ("R0 sign(s), live-feasible per-bar ridge", None, 0.0),
        ("R1 logistic on s + chain features, buy if P(R>0) > 1/2", p_full, 0.5),
        ("logistic on s alone, buy if P(R>0) > 1/2", p_s, 0.5),
        (
            "EV: OLS of R on s + chain features, buy if E[R] > 0 (added after)",
            e_full,
            0.0,
        ),
        ("EV: OLS of R on s alone, buy if E[R] > 0 (added after)", e_s, 0.0),
    ):
        q = q0 if p is None else np.where(p[scored] > thr, 1.0, -1.0)
        mm, cc = book(q, r_mid[scored], r_ask[scored], r_bid[scored])
        dd = cc[idx_b]
        dsr = dd.mean(axis=1) / dd.std(axis=1, ddof=1) * ANN - sr_base
        rows_b.append(
            {
                "rule": name,
                **summary(mm, cc, q),
                "same_position_as_R0": float((q == q0).mean()),
                "dSharpe_crossed_vs_R0": sharpe(cc) - sharpe(c_r0),
                "ci_lo": float(np.percentile(dsr, 2.5)),
                "ci_hi": float(np.percentile(dsr, 97.5)),
            }
        )
    b = pd.DataFrame(rows_b)
    b.to_csv(OUT / "b_deck_causal.csv", index=False)
    print(
        f"\nB  deck days, causal ({int(scored.sum())} scored sessions from {idx[scored][0].date()}), per unit premium"
    )
    print(b.round(3).to_string(index=False))

    # ------------------------------------------------------------- C --------
    Xc = X.to_numpy(float)
    fit_deck = ok & ~holdout
    p_frozen = fit_predict(Xc, y, fit_deck)
    p_exp = causal_probs(Xc, y, ok)
    e_frozen = fit_predict_ev(Xc, r_mid, fit_deck)
    e_exp = causal_probs(Xc, r_mid, ok, ev=True)
    rows_c = []
    rng = np.random.default_rng(SEED)
    for name, p, m, thr in (
        ("C(i) frozen logistic on the deck -> HOLDOUT", p_frozen, ok & holdout, 0.5),
        (
            "C(i) frozen logistic -> deck (in-sample, reference only)",
            p_frozen,
            fit_deck,
            0.5,
        ),
        (
            "C(ii) expanding logistic -> deck days",
            p_exp,
            np.isfinite(p_exp) & ok & ~holdout,
            0.5,
        ),
        (
            "C(ii) expanding logistic -> HOLDOUT",
            p_exp,
            np.isfinite(p_exp) & ok & holdout,
            0.5,
        ),
        (
            "EV frozen OLS of R on the deck -> HOLDOUT (added after)",
            e_frozen,
            ok & holdout,
            0.0,
        ),
        (
            "EV expanding OLS of R -> deck days (added after)",
            e_exp,
            np.isfinite(e_exp) & ok & ~holdout,
            0.0,
        ),
        (
            "EV expanding OLS of R -> HOLDOUT (added after)",
            e_exp,
            np.isfinite(e_exp) & ok & holdout,
            0.0,
        ),
        ("always short -> HOLDOUT", None, ok & holdout, 0.0),
        ("always short -> deck", None, fit_deck, 0.0),
    ):
        q = (
            np.full(int(m.sum()), -1.0)
            if p is None
            else np.where(p[m] > thr, 1.0, -1.0)
        )
        mm, cc = book(q, r_mid[m], r_ask[m], r_bid[m])
        row = {"rule": name, **summary(mm, cc, q)}
        if "HOLDOUT" in name and p is not None:
            qs = np.full(int(m.sum()), -1.0)
            _, cs = book(qs, r_mid[m], r_ask[m], r_bid[m])
            ib = asl.circular_block_bootstrap_idx(
                np.random.default_rng([SEED, int(m.sum()), 7]), int(m.sum()), BLOCK, B
            )
            d1, d0 = cc[ib], cs[ib]
            dsr = (
                d1.mean(axis=1) / d1.std(axis=1, ddof=1) * ANN
                - d0.mean(axis=1) / d0.std(axis=1, ddof=1) * ANN
            )
            row |= {
                "dSharpe_vs_always_short": sharpe(cc) - sharpe(cs),
                "ci_lo": float(np.percentile(dsr, 2.5)),
                "ci_hi": float(np.percentile(dsr, 97.5)),
            }
            n_buy = int((q > 0).sum())
            draws = []
            for _ in range(B):
                qq = np.full(len(q), -1.0)
                qq[rng.choice(len(q), n_buy, replace=False)] = 1.0
                _, cq = book(qq, r_mid[m], r_ask[m], r_bid[m])
                draws.append(sharpe(cq))
            row["placebo_p"] = float(
                (1 + (np.array(draws) >= sharpe(cc)).sum()) / (1 + B)
            )
        rows_c.append(row)
    c = pd.DataFrame(rows_c)
    c.to_csv(OUT / "c_chain_only.csv", index=False)
    print(
        "\nC  chain-only sign rules (no forecast), per unit premium; the HOLDOUT is 2024-05-01 .. 2025-12-31, unseen by every forecast"
    )
    print(c.round(3).to_string(index=False))
    coef = pd.Series(
        LogisticRegression(C=1.0, max_iter=1000)
        .fit(
            (Xc[fit_deck] - Xc[fit_deck].mean(0)) / Xc[fit_deck].std(0, ddof=1),
            y[fit_deck],
        )
        .coef_[0],
        index=FEATS,
    )
    print(
        "   frozen logistic, standardized coefficients for P(R > 0):",
        coef.round(3).to_dict(),
    )
    pd.DataFrame(
        {
            "p_frozen": p_frozen,
            "p_expanding": p_exp,
            "ev_frozen": e_frozen,
            "ev_expanding": e_exp,
            "R_mid": r_mid,
            "holdout": holdout,
            "ok": ok,
        },
        index=idx,
    ).join(X).to_csv(OUT / "c_daily.csv")

    # ---------------------------------------------------------- verdict -----
    r1 = b[b["rule"].str.startswith("R1")].iloc[0]
    ci = c[c["rule"] == "C(i) frozen logistic on the deck -> HOLDOUT"].iloc[0]
    c1 = bool(r1["ci_lo"] > 0)
    c2 = bool(ci["ci_lo"] > 0 and ci["placebo_p"] < 0.05)
    print(
        f"\nVERDICT  (1) chain features ADD to the forecast: {c1} (dSharpe crossed {r1['dSharpe_crossed_vs_R0']:+.2f}, "
        f"interval [{r1['ci_lo']:+.2f}, {r1['ci_hi']:+.2f}]); (2) chain-only rule SUPPORTED out of sample: {c2} "
        f"(holdout crossed Sharpe {ci['Sharpe_crossed']:.2f} vs always-short, interval [{ci['ci_lo']:+.2f}, {ci['ci_hi']:+.2f}], placebo p {ci['placebo_p']:.3f})"
    )


if __name__ == "__main__":
    main()
