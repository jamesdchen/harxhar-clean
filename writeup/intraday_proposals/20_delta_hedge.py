"""Proposal 20 — delta-hedged P&L of the intraday re-pick, on the cache.

Read-only. Same bars, same q as the notebook. The option hold is already one
30-minute bar; the hedge is opened at entry with the Black-76 package delta
and flattened at the same exit. Nothing is wired into the notebook.

    Delta_t = N(d1_c) + N(d1_p) - 1     Black-76, r = 0, F = S_t,
                                        total vol = IV_hr * sqrt(h_t) = sqrt(V_M)
    n_t     = -q_t * Delta_t            index units (short the residual delta
                                        if long the package)
    Pi_t    = q_t (C_{t+1} - C_t) + n_t (S_{t+1} - S_t)
    R_t^dh  = Pi_t / C_t                same per-premium convention

S_{t+1} is the next trade bar's vendor underlying on 10:00-15:00 and the
official GSPC close at 15:30 (the cash-settle). Delta is frozen at entry.

Underlying cost, when charged: 0.5 bp of S|n| at entry and at flatten, the
DH-legs convention (experiments/spxw_delta_hedged_legs.py).

Gate: unhedged rule tables must reproduce the notebook before any DH number
is reported. Adopt bar for *replacing* the notebook's scored P&L with DH:
causality of Delta, and the DH vs unhedged paired dSharpe is a measurement
not a hunt — report it; do not switch the headline unless the variance
object is the one being claimed and the usual placebo/CI gates are stated
as such. This file does not adopt.

Run:  python writeup/intraday_proposals/20_delta_hedge.py
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
OUT = INTRA / "proposals" / "20"

PROFILE_MIN_DAYS = 63
CLOSE = "15:30"
SEED = 0
BOOT_B = 2000
BOOT_BLOCK = 21
PLACEBO_DRAWS = 2000
N_CUTS = 10
GATE_TOL = 1e-6
UNDERLYING_COST_BP = 0.5
ANN = float(np.sqrt(asl.PERIODS_PER_YEAR))


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


def diurnal_share(panel: pd.DataFrame, clocks: list[str]) -> pd.DataFrame:
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


def attach_share(work: pd.DataFrame, w_slice: pd.DataFrame) -> np.ndarray:
    mi = pd.MultiIndex.from_arrays([work["date"], work["hhmm"]])
    return w_slice.stack().reindex(mi).to_numpy()


def build_work(t0: float) -> tuple[pd.DataFrame, list[str]]:
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
    pm = panel.set_index("t")[["rv_hat", "rv_raw", "in_fit"]].reset_index()
    pm["t"] = pd.to_datetime(pm["t"], utc=True) - pd.Timedelta(minutes=30)
    work = pkg.merge(pm, on="t", how="left").dropna(subset=["R", "rv_hat"]).copy()
    if not bool(work["in_fit"].all()):
        raise AssertionError("a joined trade bar sits outside the smear's fit mask")

    clocks = sorted(work["hhmm"].unique())
    work["w_slice"] = attach_share(work, diurnal_share(panel, clocks))
    if not bool(np.isfinite(work["w_slice"]).all()):
        raise AssertionError("a scored bar has no diurnal slice")

    n_rem = {c: len(clocks) - i for i, c in enumerate(clocks)}
    work["h_rem"] = work["hhmm"].map(n_rem).astype(float) * 0.5
    work["iv_var_raw"] = work["iv_hourly"].astype(float) ** 2
    work["V_M"] = work["iv_var_raw"] * work["h_rem"]
    work["s_matched"] = work["rv_hat"] - work["w_slice"] * work["V_M"]
    work = work.sort_values(["date", "t"]).reset_index(drop=True)

    is_close = (work["hhmm"] == CLOSE).to_numpy(bool)
    s_next = work.groupby("date")["S"].shift(-1).to_numpy(float)
    work["S_exit"] = np.where(is_close, work["S_close"].to_numpy(float), s_next)
    nxt_ts = pd.to_datetime(work["nxt_ts"], utc=True)
    nxt_row_ts = work.groupby("date")["t"].shift(-1)
    match = (~is_close) & nxt_row_ts.notna()
    gap = (nxt_ts[match] - nxt_row_ts[match]).abs().dt.total_seconds().to_numpy()
    tick(
        t0,
        f"work frame: {len(work):,} bars on {work['date'].nunique()} days; "
        f"S_exit finite {int(np.isfinite(work['S_exit']).sum())}; "
        f"non-close nxt_ts vs next row max |dt| {float(np.nanmax(gap) if len(gap) else 0):.0f}s",
    )
    if len(gap) and float(np.nanmax(gap)) > 0:
        raise AssertionError("next-row timestamp is not the option exit stamp")
    if int((~is_close & ~np.isfinite(work["S_exit"])).sum()) != 0:
        raise AssertionError("a daytime bar has no S_exit")
    if int((is_close & ~np.isfinite(work["S_exit"])).sum()) != 0:
        raise AssertionError("a 15:30 bar has no S_close")
    return work, clocks


def attach_delta(work: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    s_m = np.sqrt(np.clip(work["V_M"].to_numpy(float), 0.0, np.inf))
    dlt = package_delta_vec(
        s_m,
        work["S"].to_numpy(float),
        work["K_c"].to_numpy(float),
        work["K_p"].to_numpy(float),
    )
    work = work.copy()
    work["delta_pkg"] = dlt
    work["dS"] = work["S_exit"].to_numpy(float) - work["S"].to_numpy(float)
    if verbose:
        n_miss = int((np.isfinite(work["s_matched"]) & ~np.isfinite(dlt)).sum())
        n_flat = int((~np.isfinite(work["s_matched"])).sum())
        print(
            f"package delta finite on {int(np.isfinite(dlt).sum())} / {len(work)} bars; "
            f"censored-IV (no s, no delta) {n_flat}; "
            f"s finite but delta missing {n_miss}"
        )
    return work


def sign_pos(s: np.ndarray) -> np.ndarray:
    p = np.where(np.isfinite(s), np.sign(s), 0.0)
    return np.where((p == 0) & np.isfinite(s), -1.0, p)


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
        self.delta = np.nan_to_num(w["delta_pkg"].to_numpy(float), nan=0.0)
        self.dS = w["dS"].to_numpy(float)
        self.S = w["S"].to_numpy(float)
        self.S_exit = w["S_exit"].to_numpy(float)

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

    def dh_mid(self, q: np.ndarray, cost_bp: float = 0.0):
        q = np.asarray(q, float)
        opt = q * (self.exit - self.entry)
        n = -q * self.delta
        hdg = n * self.dS
        cost = (cost_bp * 1e-4) * np.abs(n) * (self.S + self.S_exit)
        pts = opt + hdg - cost
        with np.errstate(invalid="ignore", divide="ignore"):
            ret = pts / self.entry
        return ret, pts, hdg, opt, cost

    def dh_crossed(self, q: np.ndarray, cost_bp: float = 0.0):
        _ret, opt_pts = self.crossed(q)
        q = np.asarray(q, float)
        n = -q * self.delta
        hdg = n * self.dS
        cost = (cost_bp * 1e-4) * np.abs(n) * (self.S + self.S_exit)
        pts = opt_pts + hdg - cost
        with np.errstate(invalid="ignore", divide="ignore"):
            ret = pts / self.entry
        return ret, pts

    def daily(self, x: np.ndarray) -> pd.Series:
        s = pd.Series(np.asarray(x, float), index=self.w.index)
        return s.groupby(self.w["date"]).sum()

    def daily_codes(self) -> tuple[np.ndarray, pd.Index]:
        return pd.factorize(self.w["date"], sort=True)


def daily_matrix(bar: np.ndarray, codes: np.ndarray, n_days: int) -> np.ndarray:
    """bar is (B, n_bars) or (n_bars,). Returns (B, n_days) or (n_days,)."""
    bar = np.nan_to_num(np.asarray(bar, float), nan=0.0)
    codes = np.asarray(codes, dtype=np.intp)
    if bar.ndim == 1:
        out = np.zeros(n_days, dtype=float)
        np.add.at(out, codes, bar)
        return out
    b, n = bar.shape
    out2 = np.zeros((b, n_days), dtype=float)
    rows = np.repeat(np.arange(b, dtype=np.intp), n)
    cols = np.tile(codes, b)
    np.add.at(out2, (rows, cols), bar.ravel())
    return out2


def sharpe_rows(daily: np.ndarray) -> np.ndarray:
    mu = daily.mean(axis=1)
    sd = daily.std(axis=1, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(sd > 0, mu / sd * ANN, np.nan)


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
        "CI excludes 0": bool(
            (lo > 0 or hi < 0) and (2 * hat - hi > 0 or 2 * hat - lo < 0)
        ),
    }


def rules_of(work: pd.DataFrame) -> dict[str, np.ndarray]:
    pos = sign_pos(work["s_matched"].to_numpy(float))
    is_close = (work["hhmm"] == CLOSE).to_numpy(bool)
    return {
        "always short": -np.ones(len(work)),
        "always short, flat at 15:30": np.where(is_close, 0.0, -1.0),
        "sign(s)": pos,
        "always short, sign(s) close": np.where(is_close, pos, -1.0),
    }


def run_gate(f: Fills, qs: dict[str, np.ndarray]) -> pd.DataFrame:
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


def score_block(f: Fills, qs: dict[str, np.ndarray], cost_bp: float) -> pd.DataFrame:
    rows = []
    for name, q in qs.items():
        for kind, fn in (
            ("unhedged mid", lambda qq: f.mid(qq)[0]),
            ("unhedged crossed", lambda qq: f.crossed(qq)[0]),
            ("DH mid 0bp", lambda qq: f.dh_mid(qq, 0.0)[0]),
            ("DH crossed 0bp", lambda qq: f.dh_crossed(qq, 0.0)[0]),
            ("DH mid 0.5bp", lambda qq: f.dh_mid(qq, cost_bp)[0]),
            ("DH crossed 0.5bp", lambda qq: f.dh_crossed(qq, cost_bp)[0]),
        ):
            d = f.daily(fn(q))
            rows.append(
                {
                    "rule": name,
                    "book": kind,
                    "Sharpe": _sh(d),
                    "mean/day": float(d.mean()),
                    "t": float(np.sqrt(len(d.dropna())) * d.mean() / d.std(ddof=1))
                    if d.std(ddof=1)
                    else float("nan"),
                    "t_HAC": float(asl.newey_west_t(d)[0]),
                    "n_days": int(d.notna().sum()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    rng = np.random.default_rng(SEED)

    work, clocks = build_work(t0)
    work = attach_delta(work)
    f = Fills(work)
    qs = rules_of(work)
    hhmm = work["hhmm"].to_numpy()
    dlt = work["delta_pkg"].to_numpy(float)

    rule("0. GATE — unhedged notebook rule tables")
    gate = run_gate(f, qs)
    gate.to_csv(OUT / "20_gate.csv")

    rule("1. Residual delta (entry, frozen)")
    d_rows = []
    for c in [*clocks, "pooled"]:
        m = np.isfinite(dlt) if c == "pooled" else ((hhmm == c) & np.isfinite(dlt))
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
    dtab = pd.DataFrame(d_rows).set_index("clock")
    print(dtab.to_string(float_format=lambda x: f"{x:.4f}"))
    dtab.to_csv(OUT / "20_residual_delta.csv")

    rule("2. Unhedged vs delta-hedged, same q")
    scores = score_block(f, qs, UNDERLYING_COST_BP)
    wide = scores.pivot(index="rule", columns="book", values="Sharpe")
    print("Sharpe_ann of the daily sum")
    print(wide.to_string(float_format=lambda x: f"{x:+.4f}"))
    print("\nmean/day")
    print(
        scores.pivot(index="rule", columns="book", values="mean/day").to_string(
            float_format=lambda x: f"{x:+.4f}"
        )
    )
    scores.to_csv(OUT / "20_sharpe.csv", index=False)
    wide.to_csv(OUT / "20_sharpe_wide.csv")

    rule("3. Paired dSharpe DH minus unhedged (mid, 0 bp)")
    pair_rows = []
    for name, q in qs.items():
        a = f.daily(f.dh_mid(q, 0.0)[0]).to_numpy(float)
        b = f.daily(f.mid(q)[0]).to_numpy(float)
        p = boot_dsharpe(a, b, rng)
        p["NW t of daily difference"] = float(asl.newey_west_t(pd.Series(a - b))[0])
        pair_row: dict[str, object] = dict(p)
        pair_row["rule"] = name
        pair_rows.append(pair_row)
        print(
            f"{name:36s}  dSharpe {p['dSharpe']:+.4f}  "
            f"pct [{p['pct_lo']:+.3f}, {p['pct_hi']:+.3f}]  "
            f"excl0={p['CI excludes 0']}  NW t {p['NW t of daily difference']:+.2f}"
        )
    pair = pd.DataFrame(pair_rows).set_index("rule")
    pair.to_csv(OUT / "20_paired_dh_minus_unhedged.csv")

    rule("4. Paired dSharpe DH sign(s) minus DH always short (mid, 0 bp)")
    q_as = qs["always short"]
    q_sg = qs["sign(s)"]
    p_ss = boot_dsharpe(
        f.daily(f.dh_mid(q_sg, 0.0)[0]).to_numpy(float),
        f.daily(f.dh_mid(q_as, 0.0)[0]).to_numpy(float),
        rng,
    )
    p_ss["NW t"] = float(
        asl.newey_west_t(
            f.daily(f.dh_mid(q_sg, 0.0)[0]) - f.daily(f.dh_mid(q_as, 0.0)[0])
        )[0]
    )
    print(pd.Series(p_ss).to_string())
    pd.Series(p_ss).to_csv(OUT / "20_paired_sign_minus_as_dh.csv")

    rule("5. Per-clock Sharpe (one bar per day, sqrt(252))")
    clk_rows = []
    for c in clocks:
        m = hhmm == c
        for rname, q in (("always short", q_as), ("sign(s)", q_sg)):
            uh = f.mid(q)[0][m]
            dh = f.dh_mid(q, 0.0)[0][m]
            clk_rows.append(
                {
                    "clock": c,
                    "rule": rname,
                    "n": int(m.sum()),
                    "Sharpe unhedged": _sh(uh),
                    "Sharpe DH": _sh(dh),
                    "mean unhedged": float(np.nanmean(uh)),
                    "mean DH": float(np.nanmean(dh)),
                    "mean |delta|": float(np.nanmean(np.abs(dlt[m]))),
                }
            )
    ctab = pd.DataFrame(clk_rows)
    print(
        ctab.pivot(index="clock", columns="rule", values="Sharpe DH").to_string(
            float_format=lambda x: f"{x:+.3f}"
        )
    )
    print("\nDH minus unhedged Sharpe by clock, always short / sign(s):")
    dsh_c = ctab.copy()
    dsh_c["dSharpe"] = dsh_c["Sharpe DH"] - dsh_c["Sharpe unhedged"]
    print(
        dsh_c.pivot(index="clock", columns="rule", values="dSharpe").to_string(
            float_format=lambda x: f"{x:+.3f}"
        )
    )
    ctab.to_csv(OUT / "20_by_clock.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for ax, rname in zip(axes, ("always short", "sign(s)")):
        sub = ctab[ctab["rule"] == rname]
        ax.plot(sub["clock"], sub["Sharpe unhedged"], marker="o", label="unhedged")
        ax.plot(sub["clock"], sub["Sharpe DH"], marker="s", label="delta-hedged")
        ax.axhline(0.0, color="0.5", lw=0.8)
        ax.set_title(rname)
        ax.set_xlabel("entry clock (ET)")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    axes[0].set_ylabel("Sharpe_ann (per-clock daily)")
    fig.suptitle("Unhedged vs one-bar delta-hedge, same q", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "20_by_clock.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    rule("6. Decomposition: option vs hedge, always short, mid, 0 bp")
    q = q_as
    _ret, _pts, hdg, opt, _cost = f.dh_mid(q, 0.0)
    dec_rows = []
    for c in [*clocks, "pooled"]:
        m = np.ones(len(work), bool) if c == "pooled" else hhmm == c
        o, h = opt[m], hdg[m]
        ok = np.isfinite(o) & np.isfinite(h)
        o, h = o[ok], h[ok]
        tot = o + h
        dec_rows.append(
            {
                "clock": c,
                "n": int(ok.sum()),
                "mean opt": float(o.mean()),
                "mean hedge": float(h.mean()),
                "mean DH": float(tot.mean()),
                "sd opt": float(o.std(ddof=1)),
                "sd hedge": float(h.std(ddof=1)),
                "sd DH": float(tot.std(ddof=1)),
                "corr(opt, hedge)": float(np.corrcoef(o, h)[0, 1])
                if len(o) > 2
                else float("nan"),
                "var(hedge)/var(DH)": float(h.var(ddof=1) / tot.var(ddof=1))
                if tot.var(ddof=1)
                else float("nan"),
            }
        )
    dec = pd.DataFrame(dec_rows).set_index("clock")
    print(dec.to_string(float_format=lambda x: f"{x:+.4f}"))
    dec.to_csv(OUT / "20_decomp.csv")

    rule("7. Placebo: 2000 rate-matched random signs on DH mid 0 bp")
    pos = qs["sign(s)"]
    p_long = float((pos > 0).mean())
    real = _sh(f.daily(f.dh_mid(pos, 0.0)[0]))
    # Long-package DH return is linear in q: R(q) = q * R(q=+1).
    r_long = f.dh_mid(np.ones(len(work)), 0.0)[0]
    codes, _uniq = f.daily_codes()
    n_days = int(_uniq.size)
    q_p = np.where(rng.random((PLACEBO_DRAWS, len(work))) < p_long, 1.0, -1.0)
    draws = sharpe_rows(daily_matrix(q_p * r_long[None, :], codes, n_days))
    plc = pd.Series(
        {
            "sign(s) DH Sharpe": real,
            "long share": p_long,
            "placebo mean": float(np.nanmean(draws)),
            "placebo p95": float(np.nanpercentile(draws, 95)),
            "percentile of the real rule": 100.0 * float(np.mean(draws < real)),
            "draws": PLACEBO_DRAWS,
        }
    )
    print(plc.to_string())
    plc.to_csv(OUT / "20_placebo.csv")

    rule("8. Causality of Delta (frozen at entry)")
    days = pd.DatetimeIndex(sorted(work["date"].unique()))
    cut_i = rng.choice(np.arange(200, len(days)), size=N_CUTS, replace=False)
    cut_cl = rng.integers(0, len(clocks) - 1, size=N_CUTS)
    base = work["delta_pkg"].to_numpy(float)
    caus_rows = []
    for k in range(N_CUTS):
        d = days[cut_i[k]]
        c = clocks[cut_cl[k]]
        hit = (work["date"] == d) & (work["hhmm"] == c)
        row: dict[str, object] = {"date": str(d.date()), "perturbed clock": c}

        # A. later-bar S * 1.01: Delta at the cut bar must not move
        wA = work.copy()
        later = (wA["date"] == d) & (wA["hhmm"] > c)
        wA.loc[later, "S"] = wA.loc[later, "S"] * 1.01
        wA = attach_delta(wA, verbose=False)
        row["A Delta moved on cut bar"] = int(
            np.nansum(
                np.abs(wA.loc[hit, "delta_pkg"].to_numpy() - base[hit.to_numpy()])
                > 1e-12
            )
        )

        # B. this bar's IV * 10: Delta at the cut bar must move
        wB = work.copy()
        wB.loc[hit, "iv_hourly"] = wB.loc[hit, "iv_hourly"] * 10.0
        wB["iv_var_raw"] = wB["iv_hourly"].astype(float) ** 2
        wB["V_M"] = wB["iv_var_raw"] * wB["h_rem"]
        wB = attach_delta(wB, verbose=False)
        row["B Delta moved on cut bar"] = int(
            np.nansum(
                np.abs(wB.loc[hit, "delta_pkg"].to_numpy() - base[hit.to_numpy()])
                > 1e-10
            )
        )
        other = (work["date"] == d) & (work["hhmm"] != c)
        row["B Delta moved on other clocks that day"] = int(
            np.nansum(
                np.abs(wB.loc[other, "delta_pkg"].to_numpy() - base[other.to_numpy()])
                > 1e-10
            )
        )

        # C. S_exit * 1.01: Delta at entry must not move (exit is an outcome)
        wC = work.copy()
        wC.loc[hit, "S_exit"] = wC.loc[hit, "S_exit"] * 1.01
        wC = attach_delta(wC, verbose=False)
        row["C Delta moved on cut bar"] = int(
            np.nansum(
                np.abs(wC.loc[hit, "delta_pkg"].to_numpy() - base[hit.to_numpy()])
                > 1e-12
            )
        )
        caus_rows.append(row)

    caus = pd.DataFrame(caus_rows)
    print(caus.to_string(index=False))
    caus.to_csv(OUT / "20_causality.csv", index=False)
    assert int(caus["A Delta moved on cut bar"].sum()) == 0, "Delta used later S"
    assert int(caus["B Delta moved on cut bar"].sum()) == N_CUTS, "Delta dead to IV"
    assert int(caus["B Delta moved on other clocks that day"].sum()) == 0, (
        "Delta leaked"
    )
    assert int(caus["C Delta moved on cut bar"].sum()) == 0, "Delta used S_exit"
    print(
        f"assert passed: Delta is F_t-measurable and frozen at entry ({N_CUTS} cuts)."
    )

    rule("9. Verdict")
    as_uh = float(wide.loc["always short", "unhedged mid"])
    as_dh = float(wide.loc["always short", "DH mid 0bp"])
    sg_uh = float(wide.loc["sign(s)", "unhedged mid"])
    sg_dh = float(wide.loc["sign(s)", "DH mid 0bp"])
    hy_uh = float(wide.loc["always short, sign(s) close", "unhedged mid"])
    hy_dh = float(wide.loc["always short, sign(s) close", "DH mid 0bp"])
    print(f"always short   unhedged {as_uh:+.3f}  DH {as_dh:+.3f}")
    print(f"sign(s)        unhedged {sg_uh:+.3f}  DH {sg_dh:+.3f}")
    print(f"hybrid         unhedged {hy_uh:+.3f}  DH {hy_dh:+.3f}")
    print(
        "DH is the variance object the signal is built for. "
        "This file does not replace the notebook's unhedged table."
    )
    verd = pd.Series(
        {
            "always short unhedged": as_uh,
            "always short DH 0bp": as_dh,
            "sign(s) unhedged": sg_uh,
            "sign(s) DH 0bp": sg_dh,
            "hybrid unhedged": hy_uh,
            "hybrid DH 0bp": hy_dh,
            "sign(s) DH minus always short DH dSharpe": float(p_ss["dSharpe"]),
            "sign(s) DH placebo percentile": float(plc["percentile of the real rule"]),
            "adopt DH as the notebook P&L": False,
        }
    )
    verd.to_csv(OUT / "20_verdict.csv")
    print(f"\nwrote {OUT}")
    print(f"elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
