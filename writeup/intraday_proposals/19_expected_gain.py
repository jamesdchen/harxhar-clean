"""Proposal 19 — expected-gain identity G = C_BS(V_H) - C_BS(V_M).

Read-only diagnostic on the cached intraday frame. Nothing is wired into the
notebook. The professor note (deterministic variance disagreement, mu = r-q,
market curve held fixed) maps onto objects the notebook already computes:

    V_M = IV_hr^2 * h_t                         remaining implied variance
    V_H = rv_hat + (1 - w_t) V_M                hybrid: our bar + market after
    s^m = V_H - V_M = rv_hat - w_t V_M          the notebook's matched signal
    G   = C_BS(S, Kc, Kp; sqrt(V_H))
        - C_BS(S, Kc, Kp; sqrt(V_M))            expected package-price change,
                                                index points, Black-76 r = 0

sign(G) = sign(s) for a vanilla package (C increasing in V). The live
comparison is therefore not the sign rule; it is (i) whether G/|s| rises
through the day like ATM gamma 1/(sigma sqrt(T)), (ii) whether R ~ a + b G
is tighter than R ~ a + b s, (iii) unit-median sized by |G| vs |s|.

Vega is left out of the signal. The last paragraph of the note (a moving
implied curve) is diagnosis mechanism 2; a curve-revision table is printed
and not traded.

Pre-registered adopt bar for UM-|G| vs UM-|s|: causality 0/N with teeth,
paired dSharpe bootstrap 95% CI excludes 0 in the claimed direction
(percentile and basic), sizing placebo >= 95th. Sign(s) is a free match.

Run:  python writeup/intraday_proposals/19_expected_gain.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from scipy.special import erf  # noqa: E402


def _bootstrap_repo(start: Path) -> Path:
    for q in [start.resolve(), *start.resolve().parents]:
        if (q / "notebooks" / "atm_straddle_lib.py").exists():
            return q
    raise FileNotFoundError("repo root not found from " + str(start))


sys.path.insert(0, str(_bootstrap_repo(Path(__file__)) / "notebooks"))

import atm_straddle_lib as asl  # noqa: E402

REPO = asl.find_repo(Path(__file__).resolve().parent)
INTRA = REPO / "results" / "atm_straddle_intraday"
CACHE = INTRA / "cache"
OUT = INTRA / "proposals" / "19"
DECK = REPO / "results" / "atm_straddle_0dte_1530" / "daily_blk2.parquet"

PROFILE_MIN_DAYS = 63
CLOSE = "15:30"
SEED = 0
BOOT_B = 2000
BOOT_BLOCK = 21
PLACEBO_DRAWS = 2000
N_CUTS = 10
UM_CAP = 3.0
GATE_TOL = 1e-6
ANN = float(np.sqrt(asl.PERIODS_PER_YEAR))
INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


def tick(t0: float, msg: str) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def rule(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78, flush=True)


def _sh(d) -> float:
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) < 2 or d.std(ddof=1) == 0:
        return float("nan")
    return float(d.mean() / d.std(ddof=1) * ANN)


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + erf(z / np.sqrt(2.0)))


def _norm_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z * z) * INV_SQRT_2PI


def bsm_package_price_vec(
    s: np.ndarray, F: np.ndarray, Kc: np.ndarray, Kp: np.ndarray
) -> np.ndarray:
    """Black-76 package (call Kc + put Kp) on F, r = 0, total vol s. Vectorized."""
    s = np.asarray(s, dtype=np.float64)
    F = np.asarray(F, dtype=np.float64)
    Kc = np.asarray(Kc, dtype=np.float64)
    Kp = np.asarray(Kp, dtype=np.float64)
    out = np.full(np.broadcast_shapes(s.shape, F.shape, Kc.shape, Kp.shape), np.nan)
    ok = (
        np.isfinite(s)
        & np.isfinite(F)
        & (F > 0)
        & np.isfinite(Kc)
        & (Kc > 0)
        & np.isfinite(Kp)
        & (Kp > 0)
    )
    intrinsic = np.maximum(F - Kc, 0.0) + np.maximum(Kp - F, 0.0)
    zero = ok & (s <= 0)
    out[zero] = intrinsic[zero]
    pos = ok & (s > 0)
    if pos.any():
        ss, FF, kc, kp = s[pos], F[pos], Kc[pos], Kp[pos]
        d1c = (np.log(FF / kc) + 0.5 * ss * ss) / ss
        d1p = (np.log(FF / kp) + 0.5 * ss * ss) / ss
        call = FF * _norm_cdf(d1c) - kc * _norm_cdf(d1c - ss)
        put = kp * _norm_cdf(-(d1p - ss)) - FF * _norm_cdf(-d1p)
        out[pos] = call + put
    return out


def package_delta_vec(
    s: np.ndarray, F: np.ndarray, Kc: np.ndarray, Kp: np.ndarray
) -> np.ndarray:
    """Black-76 residual delta of the package, r = 0: N(d1c) + N(d1p) - 1."""
    s = np.asarray(s, dtype=np.float64)
    F = np.asarray(F, dtype=np.float64)
    Kc = np.asarray(Kc, dtype=np.float64)
    Kp = np.asarray(Kp, dtype=np.float64)
    out = np.full(s.shape, np.nan)
    pos = (
        np.isfinite(s)
        & (s > 0)
        & np.isfinite(F)
        & (F > 0)
        & np.isfinite(Kc)
        & (Kc > 0)
        & np.isfinite(Kp)
        & (Kp > 0)
    )
    if pos.any():
        ss, FF, kc, kp = s[pos], F[pos], Kc[pos], Kp[pos]
        d1c = (np.log(FF / kc) + 0.5 * ss * ss) / ss
        d1p = (np.log(FF / kp) + 0.5 * ss * ss) / ss
        out[pos] = _norm_cdf(d1c) + _norm_cdf(d1p) - 1.0
    return out


def atm_dc_dv(
    s: np.ndarray, F: np.ndarray, Kc: np.ndarray, Kp: np.ndarray
) -> np.ndarray:
    """ATM-ish dollar-per-variance of the package: S n(d1)/s summed over legs.

    For a single ATM option dC/dV = S n(d1) / (2 s). The package is two
    options; this prints the sum so it can sit next to G/s.
    """
    s = np.asarray(s, dtype=np.float64)
    F = np.asarray(F, dtype=np.float64)
    Kc = np.asarray(Kc, dtype=np.float64)
    Kp = np.asarray(Kp, dtype=np.float64)
    out = np.full(s.shape, np.nan)
    pos = (
        np.isfinite(s)
        & (s > 0)
        & np.isfinite(F)
        & (F > 0)
        & np.isfinite(Kc)
        & (Kc > 0)
        & np.isfinite(Kp)
        & (Kp > 0)
    )
    if pos.any():
        ss, FF, kc, kp = s[pos], F[pos], Kc[pos], Kp[pos]
        d1c = (np.log(FF / kc) + 0.5 * ss * ss) / ss
        d1p = (np.log(FF / kp) + 0.5 * ss * ss) / ss
        out[pos] = FF * (_norm_pdf(d1c) + _norm_pdf(d1p)) / (2.0 * ss)
    return out


def diurnal_share(panel: pd.DataFrame, clocks: list[str]) -> pd.DataFrame:
    """w_t from the forecast panel's in-fit history, notebook §5b."""
    pf = panel.loc[panel["in_fit"].to_numpy(dtype=bool)].copy()
    clock = pd.to_datetime(pf["t"], utc=True).dt.tz_convert("America/New_York")
    clock = clock - pd.Timedelta(minutes=30)
    pf["pdate"] = clock.dt.normalize().dt.tz_localize(None)
    pf["phhmm"] = clock.dt.strftime("%H:%M")
    prof = pf.pivot_table(
        index="pdate", columns="phhmm", values="rv_raw", aggfunc="mean"
    ).sort_index()
    missing = [c for c in clocks if c not in prof.columns]
    if missing:
        raise AssertionError(f"profile missing clocks {missing}")
    prof_exp = prof.expanding(min_periods=PROFILE_MIN_DAYS).mean().shift(1)
    rem = prof_exp[clocks[::-1]].cumsum(axis=1)[clocks]
    w = prof_exp[clocks] / rem
    if not bool(np.isclose(w[CLOSE].dropna().to_numpy(), 1.0).all()):
        raise AssertionError("w must be 1 at 15:30")
    return w


def attach_share(work: pd.DataFrame, w_slice: pd.DataFrame) -> pd.Series:
    mi = pd.MultiIndex.from_arrays([work["date"], work["hhmm"]])
    return w_slice.stack().reindex(mi).to_numpy()


def build_work(t0: float) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    cands = sorted(CACHE.glob("trade_*.parquet"))
    if len(cands) != 1:
        raise FileNotFoundError(
            f"expected exactly one trade cache in {CACHE}, found {[c.name for c in cands]}"
        )
    pkg = pd.read_parquet(cands[0])
    tick(t0, f"trade cache {cands[0].name}: {len(pkg):,} packages")

    panel = asl.load_yhat_panel(asl.yhat_paths(REPO)["blk2"])
    tick(t0, f"forecast panel: {len(panel):,} stamps")

    pkg = pkg.copy()
    pkg["t"] = pd.to_datetime(pkg["timestamp"], utc=True)
    pm = panel.set_index("t")[
        ["rv_hat", "rv_raw", "in_fit", "early_close"]
    ].reset_index()
    pm["t"] = pd.to_datetime(pm["t"], utc=True) - pd.Timedelta(minutes=30)
    work = pkg.merge(pm, on="t", how="left")
    n_pre = len(work)
    work = work.dropna(subset=["R", "rv_hat"]).copy()
    if not bool(work["in_fit"].all()):
        raise AssertionError("a joined trade bar sits outside the smear's fit mask")

    clocks = sorted(work["hhmm"].unique())
    w_slice = diurnal_share(panel, clocks)
    work["w_slice"] = attach_share(work, w_slice)
    if not bool(np.isfinite(work["w_slice"]).all()):
        raise AssertionError("a scored bar has no diurnal slice")

    n_rem = {c: len(clocks) - i for i, c in enumerate(clocks)}
    work["h_rem"] = work["hhmm"].map(n_rem).astype(float) * 0.5
    work["iv_var_raw"] = work["iv_hourly"].astype(float) ** 2
    work["V_M"] = work["iv_var_raw"] * work["h_rem"]
    work["V_H"] = work["rv_hat"] + (1.0 - work["w_slice"]) * work["V_M"]
    work["s_matched"] = work["rv_hat"] - work["w_slice"] * work["V_M"]
    work = work.sort_values("t").reset_index(drop=True)

    chk = work[(work["hhmm"] == CLOSE) & np.isfinite(work["s_matched"])]
    collapse = float(
        (chk["w_slice"] * work.loc[chk.index, "V_M"] / (chk["iv_var_raw"] * 0.5) - 1.0)
        .abs()
        .max()
    )
    tick(
        t0,
        f"work frame: {len(work):,} bars on {work['date'].nunique()} days; "
        f"dropped {n_pre - len(work):,} at forecast join; "
        f"15:30 |w V_M / (IV^2/2) - 1| max {collapse:.2e}",
    )
    return work, panel, clocks


def attach_g(work: pd.DataFrame) -> pd.DataFrame:
    s_h = np.sqrt(np.clip(work["V_H"].to_numpy(float), 0.0, np.inf))
    s_m = np.sqrt(np.clip(work["V_M"].to_numpy(float), 0.0, np.inf))
    F = work["S"].to_numpy(float)
    Kc = work["K_c"].to_numpy(float)
    Kp = work["K_p"].to_numpy(float)
    finite = (
        np.isfinite(work["V_H"].to_numpy(float))
        & np.isfinite(work["V_M"].to_numpy(float))
        & (work["V_M"].to_numpy(float) > 0)
        & (work["V_H"].to_numpy(float) >= 0)
    )
    c_h = bsm_package_price_vec(s_h, F, Kc, Kp)
    c_m = bsm_package_price_vec(s_m, F, Kc, Kp)
    g = np.where(finite, c_h - c_m, np.nan)
    work = work.copy()
    work["C_M"] = c_m
    work["C_H"] = c_h
    work["G"] = g
    work["G_over_entry"] = g / work["entry"].to_numpy(float)
    work["s_tot"] = np.where(finite, s_m, np.nan)
    work["delta_pkg"] = package_delta_vec(s_m, F, Kc, Kp)
    work["dC_dV"] = atm_dc_dv(s_m, F, Kc, Kp)
    return work


def sign_pos(s: np.ndarray) -> np.ndarray:
    p = np.where(np.isfinite(s), np.sign(s), 0.0)
    return np.where((p == 0) & np.isfinite(s), -1.0, p)


def um_leverage(s: np.ndarray, min_bars: int, cap: float = UM_CAP) -> np.ndarray:
    """q = sign(s) * clip(|s|/lagged expanding median, cap); flat if no scale."""
    s = np.asarray(s, float)
    abs_s = np.abs(s)
    med = pd.Series(abs_s).expanding(min_periods=min_bars).median().shift(1).to_numpy()
    ell = abs_s / med
    ell = np.where(np.isfinite(ell), np.minimum(ell, cap), np.nan)
    q = sign_pos(s) * np.where(np.isfinite(ell), ell, 0.0)
    return np.where(np.isfinite(s) & np.isfinite(ell), q, 0.0)


class Fills:
    def __init__(self, w: pd.DataFrame):
        self.w = w
        self.entry = w["entry"].to_numpy(float)
        self.exit = w["exit"].to_numpy(float)
        self.ret = w["R"].to_numpy(float)
        self.bid_e = w["bid_entry"].to_numpy(float)
        self.ask_e = w["ask_entry"].to_numpy(float)
        self.bid_x = (w["bid_c_nxt"] + w["bid_p_nxt"]).to_numpy(float)
        self.ask_x = (w["ask_c_nxt"] + w["ask_p_nxt"]).to_numpy(float)
        self.is_last = (w["hhmm"] == CLOSE).to_numpy(bool)
        self.same_k = (
            (w["K_c"].shift(-1) == w["K_c"])
            & (w["K_p"].shift(-1) == w["K_p"])
            & (w["date"].shift(-1) == w["date"])
            & ~self.is_last
        ).to_numpy(bool)

    def mid(self, q: np.ndarray):
        r = np.asarray(q, float) * self.ret
        return r, r * self.entry

    def crossed(self, q: np.ndarray):
        q = np.asarray(q, float)
        long, short = q > 0, q < 0
        nxt_q = np.append(q[1:], 0.0)
        hold = self.same_k & (np.sign(nxt_q) == np.sign(q)) & (q != 0)
        held_in = np.concatenate([[False], hold[:-1]])
        entry_px = np.where(
            held_in,
            self.entry,
            np.where(long, self.ask_e, np.where(short, self.bid_e, self.entry)),
        )
        exit_px = np.where(
            self.is_last,
            self.exit,
            np.where(
                hold,
                self.exit,
                np.where(long, self.bid_x, np.where(short, self.ask_x, self.exit)),
            ),
        )
        untradeable = ~held_in & (
            (long & ~(self.ask_e > 0)) | (short & ~(self.bid_e > 0))
        )
        pts = np.where(untradeable, np.nan, q * (exit_px - entry_px))
        return pts / self.entry, pts

    def daily(self, x: np.ndarray) -> pd.Series:
        s = pd.Series(np.asarray(x, float), index=self.w.index)
        return s.groupby(self.w["date"]).sum()


def boot_dsharpe(a, b, rng) -> dict[str, float]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    idx = asl.circular_block_bootstrap_idx(rng, len(a), BOOT_BLOCK, BOOT_B)

    def shr(v):
        return v.mean(axis=1) / v.std(axis=1, ddof=1) * ANN

    d = shr(a[idx]) - shr(b[idx])
    lo, hi = (float(v) for v in np.percentile(d, [2.5, 97.5]))
    hat = _sh(a) - _sh(b)
    return {
        "dSharpe": hat,
        "pct_lo": lo,
        "pct_hi": hi,
        "basic_lo": 2 * hat - hi,
        "basic_hi": 2 * hat - lo,
    }


def ols_hac(y: np.ndarray, x: np.ndarray, name: str) -> dict[str, float | str]:
    m = np.isfinite(y) & np.isfinite(x)
    yv, xv = y[m], x[m]
    if len(yv) < 10:
        return {"name": name, "n": int(len(yv)), "R2": np.nan, "b": np.nan, "t": np.nan}
    X = sm.add_constant(xv)
    lag = asl.newey_west_lag(len(yv))
    fit = sm.OLS(yv, X).fit(
        cov_type="HAC", cov_kwds={"maxlags": lag, "use_correction": False}
    )
    return {
        "name": name,
        "n": int(len(yv)),
        "R2": float(fit.rsquared),
        "b": float(fit.params[1]),
        "t": float(fit.tvalues[1]),
        "lag": int(lag),
    }


def run_gate(f: Fills, work: pd.DataFrame) -> pd.DataFrame:
    s = work["s_matched"].to_numpy(float)
    pos = sign_pos(s)
    is_close = (work["hhmm"] == CLOSE).to_numpy(bool)
    qs = {
        "always short": -np.ones(len(work)),
        "always short, flat at 15:30": np.where(is_close, 0.0, -1.0),
        "sign(s)": pos,
        "always short, sign(s) close": np.where(is_close, pos, -1.0),
    }
    tgt_mid = pd.read_csv(INTRA / "rule_table_intraday_blk2.csv", index_col=0)
    tgt_cr = pd.read_csv(INTRA / "rule_table_intraday_crossed_blk2.csv", index_col=0)
    rows = []
    for name, q in qs.items():
        d_mid = f.daily(f.mid(q)[0])
        d_cr = f.daily(f.crossed(q)[0])
        rows.append(
            {
                "rule": name,
                "Sharpe mid": _sh(d_mid),
                "target Sharpe mid": float(tgt_mid.loc[name, "Sharpe_ann"]),
                "mean/day mid": float(d_mid.mean()),
                "target mean/day mid": float(tgt_mid.loc[name, "mean_daily"]),
                "Sharpe crossed": _sh(d_cr),
                "target Sharpe crossed": float(
                    tgt_cr.loc[name, "Sharpe crossed-spread"]
                ),
            }
        )
    g = pd.DataFrame(rows).set_index("rule")
    g["max |gap|"] = np.maximum.reduce(
        [
            (g["Sharpe mid"] - g["target Sharpe mid"]).abs(),
            (g["mean/day mid"] - g["target mean/day mid"]).abs(),
            (g["Sharpe crossed"] - g["target Sharpe crossed"]).abs(),
        ]
    )
    worst = float(g["max |gap|"].max())
    print(g.to_string(float_format=lambda x: f"{x:+.7f}"))
    print(f"worst reproduction gap against the notebook's tables: {worst:.2e}")
    if worst > GATE_TOL:
        raise AssertionError(f"gate failed: worst gap {worst:.3e} > {GATE_TOL:.0e}")
    print("GATE PASSED")
    return g


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    rng = np.random.default_rng(SEED)

    work, panel, clocks = build_work(t0)
    work = attach_g(work)
    f = Fills(work)
    s = work["s_matched"].to_numpy(float)
    g = work["G"].to_numpy(float)
    hhmm = work["hhmm"].to_numpy()

    # scalar vs vector price, a handful of finite rows
    sample = np.where(np.isfinite(g))[0][:8]
    for i in sample:
        scalar = asl._bsm_package_price(
            float(np.sqrt(work.loc[i, "V_M"])),
            float(work.loc[i, "S"]),
            float(work.loc[i, "K_c"]),
            float(work.loc[i, "K_p"]),
        )
        vec = float(work.loc[i, "C_M"])
        if abs(scalar - vec) > 1e-9:
            raise AssertionError(f"vector C_M != scalar at row {i}: {vec} vs {scalar}")
    tick(t0, f"vector C_M matches scalar _bsm_package_price on {len(sample)} rows")

    rule("0. GATE — notebook rule table under s^m = V_H - V_M")
    gate = run_gate(f, work)
    gate.to_csv(OUT / "19_gate.csv")

    rule("1. Identity checks")
    gap = work["V_H"] - work["V_M"] - work["s_matched"]
    print(f"max |V_H - V_M - s^m| = {float(gap.abs().max()):.3e} (must be ~0)")
    if float(gap.abs().max()) > 1e-15:
        raise AssertionError("s^m is not V_H - V_M")
    both = np.isfinite(s) & np.isfinite(g)
    sign_agree = float(np.mean(np.sign(s[both]) == np.sign(g[both])))
    n_disagree = int(np.sum(np.sign(s[both]) != np.sign(g[both])))
    print(
        f"sign(G) == sign(s) on {sign_agree:.6f} of {int(both.sum())} finite bars "
        f"({n_disagree} disagreements)"
    )
    if n_disagree != 0:
        raise AssertionError("vanilla package: sign(G) must equal sign(s)")
    ratio = work["C_M"] / work["entry"].to_numpy(float)
    print(
        "C_BS(V_M) / entry by clock (1.0000 = the quote is the remaining-window price):"
    )
    rat = (
        work.loc[np.isfinite(ratio)]
        .groupby("hhmm")["C_M"]
        .apply(lambda x: float(np.median(x / work.loc[x.index, "entry"])))
    )
    n_rat = work.loc[np.isfinite(ratio)].groupby("hhmm").size()
    p5 = (
        work.loc[np.isfinite(ratio)]
        .groupby("hhmm", group_keys=False)
        .apply(
            lambda d: float(np.quantile(d["C_M"] / d["entry"], 0.05)),
            include_groups=False,
        )
    )
    p95 = (
        work.loc[np.isfinite(ratio)]
        .groupby("hhmm", group_keys=False)
        .apply(
            lambda d: float(np.quantile(d["C_M"] / d["entry"], 0.95)),
            include_groups=False,
        )
    )
    price_tab = pd.DataFrame({"n": n_rat, "median": rat, "p5": p5, "p95": p95})
    print(price_tab.to_string(float_format=lambda x: f"{x:.4f}"))
    price_tab.to_csv(OUT / "19_price_ratio.csv")

    rule("2. Residual delta of the nearest-OTM re-pick")
    dlt = work["delta_pkg"].to_numpy(float)
    d_rows = []
    for c in clocks:
        m = (hhmm == c) & np.isfinite(dlt)
        v = dlt[m]
        d_rows.append(
            {
                "clock": c,
                "n": int(m.sum()),
                "mean": float(v.mean()),
                "median": float(np.median(v)),
                "mean |delta|": float(np.mean(np.abs(v))),
                "p90 |delta|": float(np.quantile(np.abs(v), 0.90)),
                "pct |delta| > 0.05": 100.0 * float(np.mean(np.abs(v) > 0.05)),
                "pct |delta| > 0.10": 100.0 * float(np.mean(np.abs(v) > 0.10)),
            }
        )
    m = np.isfinite(dlt)
    v = dlt[m]
    d_rows.append(
        {
            "clock": "pooled",
            "n": int(m.sum()),
            "mean": float(v.mean()),
            "median": float(np.median(v)),
            "mean |delta|": float(np.mean(np.abs(v))),
            "p90 |delta|": float(np.quantile(np.abs(v), 0.90)),
            "pct |delta| > 0.05": 100.0 * float(np.mean(np.abs(v) > 0.05)),
            "pct |delta| > 0.10": 100.0 * float(np.mean(np.abs(v) > 0.10)),
        }
    )
    dtab = pd.DataFrame(d_rows).set_index("clock")
    print(dtab.to_string(float_format=lambda x: f"{x:.4f}"))
    print(
        "section 3 of the note: an unhedged option needs mu unless delta is ~0. "
        "This book is nearest-OTM, not a synthetic ATM; residual delta is the "
        "directional leak."
    )
    dtab.to_csv(OUT / "19_residual_delta.csv")

    rule("3. Does G/|s| rise through the day like ATM gamma (1 / sigma sqrt(T))?")
    g_rows = []
    for c in clocks:
        m = (hhmm == c) & np.isfinite(s) & np.isfinite(g) & (np.abs(s) > 0)
        gs = g[m] / s[m]
        dc = work.loc[m, "dC_dV"].to_numpy(float)
        st = work.loc[m, "s_tot"].to_numpy(float)
        g_rows.append(
            {
                "clock": c,
                "n": int(m.sum()),
                "median G/s": float(np.median(gs)),
                "mean G/s": float(np.mean(gs)),
                "median dC/dV": float(np.nanmedian(dc)),
                "median 1/s": float(np.median(1.0 / st)),
                "median G": float(np.median(g[m])),
                "median |s|": float(np.median(np.abs(s[m]))),
            }
        )
    gtab = pd.DataFrame(g_rows).set_index("clock")
    print(gtab.to_string(float_format=lambda x: f"{x:.4g}"))
    order = np.arange(len(clocks))
    med_gs = gtab["median G/s"].to_numpy(float)
    med_invs = gtab["median 1/s"].to_numpy(float)
    rho_clock = float(pd.Series(order).corr(pd.Series(med_gs), method="spearman"))
    rho_gamma = float(pd.Series(med_gs).corr(pd.Series(med_invs), method="spearman"))
    print(f"Spearman(clock order, median G/s) = {rho_clock:+.3f}")
    print(f"Spearman(median G/s, median 1/s)  = {rho_gamma:+.3f}")
    print(
        "ATM gamma ~ 1/(S sigma sqrt(T_rem)); 1/s = 1/sqrt(V_M) is that shape. "
        "A rising G/s is the 15:30-dominance story in dollars."
    )
    gtab.to_csv(OUT / "19_gamma_by_clock.csv")

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.plot(clocks, gtab["median G/s"].to_numpy(), marker="o", label="median G/s")
    ax.plot(
        clocks,
        gtab["median dC/dV"].to_numpy(),
        marker="s",
        label="median dC/dV (BS)",
    )
    ax.set_xlabel("entry clock (ET)")
    ax.set_ylabel("index points per unit variance")
    ax.set_title("Dollar map of the variance gap, by clock")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "19_gamma_by_clock.png", dpi=150)
    plt.close(fig)

    rule("4. Is R ~ a + b G tighter than R ~ a + b s? (long-package return, no q)")
    R = work["R"].to_numpy(float)
    g_ret = work["G_over_entry"].to_numpy(float)
    specs = []
    frames = {
        "all bars": np.ones(len(work), bool),
        "15:30 only": hhmm == CLOSE,
        "10:00-15:00": hhmm != CLOSE,
    }
    for fr_name, msk in frames.items():
        specs.append(ols_hac(R[msk], s[msk], f"{fr_name}: R ~ s"))
        specs.append(ols_hac(R[msk], g[msk], f"{fr_name}: R ~ G (points)"))
        specs.append(ols_hac(R[msk], g_ret[msk], f"{fr_name}: R ~ G/entry"))
    ols_tab = pd.DataFrame(specs).set_index("name")
    print(ols_tab.to_string(float_format=lambda x: f"{x:.4f}"))
    ols_tab.to_csv(OUT / "19_ols.csv")

    rule("5. Unit-median sized by |G| vs |s| (do not adopt unless gates clear)")
    bars_per_day = float(work.groupby("date").size().median())
    min_bars = int(PROFILE_MIN_DAYS * bars_per_day)
    print(
        f"UM warmup: {min_bars} bars = {PROFILE_MIN_DAYS} days x {bars_per_day:.0f} bars/day"
    )
    q_sign = sign_pos(s)
    q_um_s = um_leverage(s, min_bars)
    q_um_g = um_leverage(g, min_bars)
    # same signs, different scale: UM-|G| vs UM-|s|
    sign_um = float(np.mean(np.sign(q_um_s) == np.sign(q_um_g)))
    print(f"sign agreement UM-|G| vs UM-|s|: {sign_um:.6f} (must be 1)")
    if abs(sign_um - 1.0) > 1e-12:
        raise AssertionError("UM-|G| and UM-|s| must share the sign of s")

    daily = {
        "always short": f.daily(f.mid(-np.ones(len(work)))[0]),
        "sign(s)": f.daily(f.mid(q_sign)[0]),
        "UM |s|": f.daily(f.mid(q_um_s)[0]),
        "UM |G|": f.daily(f.mid(q_um_g)[0]),
        "UM |s| crossed": f.daily(f.crossed(q_um_s)[0]),
        "UM |G| crossed": f.daily(f.crossed(q_um_g)[0]),
    }
    um_rows = []
    for name, d in daily.items():
        um_rows.append(
            {
                "rule": name,
                "Sharpe": _sh(d),
                "mean/day": float(d.mean()),
                "t": float(np.sqrt(len(d)) * d.mean() / d.std(ddof=1)),
                "t_HAC": float(asl.newey_west_t(d)[0]),
                "n_days": int(len(d)),
            }
        )
    um_tab = pd.DataFrame(um_rows).set_index("rule")
    print(um_tab.to_string(float_format=lambda x: f"{x:+.4f}"))
    um_tab.to_csv(OUT / "19_um_sharpe.csv")

    a = daily["UM |G|"].to_numpy(float)
    b = daily["UM |s|"].to_numpy(float)
    paired = boot_dsharpe(a, b, rng)
    hac_t, hac_lag = asl.newey_west_t(pd.Series(a - b))
    paired["NW t of daily difference"] = float(hac_t)
    paired["NW lag"] = int(hac_lag)
    excl = (paired["pct_lo"] > 0 or paired["pct_hi"] < 0) and (
        paired["basic_lo"] > 0 or paired["basic_hi"] < 0
    )
    paired["CI excludes 0 (pct and basic)"] = bool(excl)
    print("\npaired dSharpe UM-|G| minus UM-|s| (mid, circular block bootstrap):")
    print(pd.Series(paired).to_string())
    pd.Series(paired).to_csv(OUT / "19_paired.csv")

    # sizing placebo: within-day permutation of |G|, keep sign(s)
    dates = work["date"].to_numpy()
    abs_g = np.abs(g)
    abs_g = np.where(np.isfinite(abs_g), abs_g, np.nan)
    draws = np.empty(PLACEBO_DRAWS)
    for k in range(PLACEBO_DRAWS):
        g_perm = abs_g.copy()
        for d, idx in work.groupby("date").indices.items():
            idx = np.asarray(idx)
            g_perm[idx] = rng.permutation(g_perm[idx])
        # rebuild UM on the permuted |G| with original signs
        signed = np.where(np.isfinite(s), np.sign(s) * g_perm, np.nan)
        signed = np.where((signed == 0) & np.isfinite(s), -g_perm, signed)
        q_p = um_leverage(signed, min_bars)
        draws[k] = _sh(f.daily(f.mid(q_p)[0]))
    real_sh = float(um_tab.loc["UM |G|", "Sharpe"])
    pctile = 100.0 * float(np.mean(draws < real_sh))
    plc = pd.Series(
        {
            "UM |G| Sharpe mid": real_sh,
            "placebo mean": float(np.nanmean(draws)),
            "placebo p95": float(np.nanpercentile(draws, 95)),
            "percentile of the real UM-|G|": pctile,
            "draws": PLACEBO_DRAWS,
            "clears 95th": bool(pctile >= 95.0),
        }
    )
    print("\nsizing placebo: within-day permutation of |G|, signs of s held fixed")
    print(plc.to_string())
    plc.to_csv(OUT / "19_placebo.csv")

    rule("6. Causality")
    days = pd.DatetimeIndex(sorted(work["date"].unique()))
    cut_i = rng.choice(np.arange(200, len(days)), size=N_CUTS, replace=False)
    cut_cl = rng.integers(0, len(clocks) - 1, size=N_CUTS)
    caus_rows = []
    base_w = work["w_slice"].to_numpy(float)
    base_g = work["G"].to_numpy(float)
    for k in range(N_CUTS):
        d = days[cut_i[k]]
        c = clocks[cut_cl[k]]
        row: dict[str, object] = {"date": str(d.date()), "perturbed clock": c}

        # A. panel rv_raw * 10 on this day: same-day w must not move
        pan2 = panel.copy()
        et = pd.to_datetime(pan2["t"], utc=True).dt.tz_convert("America/New_York")
        pdate = (et - pd.Timedelta(minutes=30)).dt.normalize().dt.tz_localize(None)
        hit_pan = pan2["in_fit"].to_numpy(bool) & (pdate == d)
        pan2.loc[hit_pan, "rv_raw"] = pan2.loc[hit_pan, "rv_raw"] * 10.0
        w2 = attach_share(work, diurnal_share(pan2, clocks))
        on_d = (work["date"] == d).to_numpy()
        after_d = (work["date"] > d).to_numpy()
        row["A w moved on cut day"] = int(
            np.nansum(np.abs(w2[on_d] - base_w[on_d]) > 1e-12)
        )
        row["A w moved after cut day"] = int(
            np.nansum(np.abs(w2[after_d] - base_w[after_d]) > 1e-12)
        )

        # B. rv_hat * 10 on the cut bar: G on other same-day clocks unchanged
        wB = work.copy()
        hit = (wB["date"] == d) & (wB["hhmm"] == c)
        wB.loc[hit, "rv_hat"] = wB.loc[hit, "rv_hat"] * 10.0
        wB["V_H"] = wB["rv_hat"] + (1.0 - wB["w_slice"]) * wB["V_M"]
        wB = attach_g(wB)
        same_day_other = on_d & ~(wB["hhmm"] == c).to_numpy()
        row["B G moved on other clocks that day"] = int(
            np.nansum(
                np.abs(wB.loc[same_day_other, "G"].to_numpy() - base_g[same_day_other])
                > 1e-10
            )
        )
        row["B G moved on the cut bar"] = int(
            np.nansum(
                np.abs(wB.loc[hit, "G"].to_numpy() - base_g[hit.to_numpy()]) > 1e-10
            )
        )

        # C. later-clock IV * 10: G at the cut bar unchanged
        wC = work.copy()
        later = (wC["date"] == d) & (wC["hhmm"] > c)
        wC.loc[later, "iv_hourly"] = wC.loc[later, "iv_hourly"] * 10.0
        wC["iv_var_raw"] = wC["iv_hourly"].astype(float) ** 2
        wC["V_M"] = wC["iv_var_raw"] * wC["h_rem"]
        wC["V_H"] = wC["rv_hat"] + (1.0 - wC["w_slice"]) * wC["V_M"]
        wC = attach_g(wC)
        row["C G moved on the cut bar"] = int(
            np.nansum(
                np.abs(wC.loc[hit, "G"].to_numpy() - base_g[hit.to_numpy()]) > 1e-10
            )
        )

        # D. UM scale is lagged: the median at t does not use G_t.
        # q_t itself uses G_t (that is the rule), so it is allowed to move.
        gB = wB["G"].to_numpy(float)
        med0 = (
            pd.Series(np.abs(base_g))
            .expanding(min_periods=min_bars)
            .median()
            .shift(1)
            .to_numpy()
        )
        med1 = (
            pd.Series(np.abs(gB))
            .expanding(min_periods=min_bars)
            .median()
            .shift(1)
            .to_numpy()
        )
        cut_bar = int(np.flatnonzero(hit.to_numpy())[0]) if hit.any() else 0
        row["D lagged median moved on the cut bar"] = int(
            np.isfinite(med0[cut_bar])
            and np.isfinite(med1[cut_bar])
            and abs(med0[cut_bar] - med1[cut_bar]) > 1e-12
        )
        later_all = np.arange(len(work)) > cut_bar
        row["D lagged median moved after the cut bar"] = int(
            np.nansum(np.abs(med1[later_all] - med0[later_all]) > 1e-12)
        )
        caus_rows.append(row)

    caus = pd.DataFrame(caus_rows)
    print(caus.to_string(index=False))
    caus.to_csv(OUT / "19_causality.csv", index=False)
    assert int(caus["A w moved on cut day"].sum()) == 0, "w moved on the perturbed day"
    assert int(caus["A w moved after cut day"].sum()) > 0, (
        "w-causality test is toothless"
    )
    assert int(caus["B G moved on other clocks that day"].sum()) == 0, (
        "G leaked across clocks"
    )
    assert int(caus["B G moved on the cut bar"].sum()) == N_CUTS, "G dead to rv_hat"
    assert int(caus["C G moved on the cut bar"].sum()) == 0, "G used future IV"
    assert int(caus["D lagged median moved on the cut bar"].sum()) == 0, (
        "UM scale peeked at t"
    )
    n_w_teeth = int((caus["A w moved after cut day"] > 0).sum())
    n_um_teeth = int((caus["D lagged median moved after the cut bar"] > 0).sum())
    print(
        "assert passed: w, G and the UM scale are F_t-measurable "
        f"({N_CUTS} cuts). Later-day w teeth: {n_w_teeth} of {N_CUTS} cuts; "
        "later-bar UM median is an expanding median of ~10k points so one "
        f"outlier often does not move it ({n_um_teeth} of {N_CUTS} cuts)."
    )

    rule("7. Curve revision (NOT in the signal)")
    # remaining implied at u vs the fixed-curve leftover (1-w) V_M
    iv_rem = work["V_M"].to_numpy(float)
    leftover = (1.0 - work["w_slice"].to_numpy(float)) * iv_rem
    nxt = pd.Series(iv_rem).groupby(work["date"]).shift(-1).to_numpy()
    rev = nxt - leftover
    last = hhmm == CLOSE
    rev_rows = []
    for c in clocks[:-1]:
        m = (hhmm == c) & np.isfinite(rev) & np.isfinite(iv_rem) & (iv_rem > 0) & ~last
        r = rev[m] / iv_rem[m]
        rev_rows.append(
            {
                "clock": c,
                "n": int(m.sum()),
                "mean (IV_rem next - (1-w) V_M) / V_M": float(np.mean(r)),
                "median": float(np.median(r)),
                "mean |revision| / V_M": float(np.mean(np.abs(r))),
            }
        )
    rtab = pd.DataFrame(rev_rows).set_index("clock")
    print(rtab.to_string(float_format=lambda x: f"{x:+.4f}"))
    print(
        "If the implied curve were the note's fixed v_M, leftover (1-w) V_M "
        "would equal the next stamp's remaining implied. A non-zero revision is "
        "the extra vega term. It is printed, not traded."
    )
    rtab.to_csv(OUT / "19_curve_revision.csv")

    rule("8. Verdict")
    clears_ci = bool(paired["CI excludes 0 (pct and basic)"])
    clears_plc = bool(plc["clears 95th"])
    dsh = float(paired["dSharpe"])
    adopt = bool(clears_ci and clears_plc and dsh > 0)
    print(f"dSharpe UM-|G| - UM-|s| = {dsh:+.4f}")
    print(f"bootstrap CI excludes 0 (pct and basic): {clears_ci}")
    print(f"sizing placebo >= 95th: {clears_plc} (percentile {pctile:.1f})")
    print(f"ADOPT UM-|G| in the notebook: {adopt}")
    if not adopt:
        print("do not adopt. sign(s) is unchanged by construction.")
    verd = pd.Series(
        {
            "dSharpe": dsh,
            "CI excludes 0": clears_ci,
            "placebo >= 95th": clears_plc,
            "placebo percentile": pctile,
            "adopt": adopt,
        }
    )
    verd.to_csv(OUT / "19_verdict.csv")

    print(f"\nwrote {OUT}")
    print(f"elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
