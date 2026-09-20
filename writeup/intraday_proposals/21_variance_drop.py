"""Proposal 21 — per-period implied as a variance drop, non-peeking.

The 30-minute implied from one expiry is the drop in remaining variance:

    ΔV_t = V_M(t) - V_M(t+30m)
         = IV_hr(t)^2 h_t - IV_hr(t+30m)^2 h_{t+30m}

At 15:30 there is no next remaining quote; V_M(16:00)=0 so ΔV = V_M = IV^2/2.

Today's ΔV needs the next stamp (peeking). This file uses only PRIOR days:

    μΔ_{d,c}  = expanding mean of ΔV at clock c over dates < d  (min 63)
    μV_{d,c}  = expanding mean of V_M at clock c over dates < d
    π_{d,c}   = expanding mean of (ΔV/V_M)+ at clock c over dates < d

    abs     slice = μΔ_{d,c}                         (not scaled to today)
    scaled  slice = (μΔ_{d,c} / μV_{d,c}) * V_{d,c}  (ratio of means)
    ratio   slice = π_{d,c} * V_{d,c}                (mean of ratios)

V0 is the notebook's realized-U share w * V_M (panel-seeded). Gate that
sign(s) against the notebook table before anything else.

Peeking ΔV is printed as calibration only (ex post, not a rule).

Run:  python writeup/intraday_proposals/21_variance_drop.py
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
OUT = INTRA / "proposals" / "21"

PROFILE_MIN_DAYS = 63
CLOSE = "15:30"
SEED = 0
BOOT_B = 2000
BOOT_BLOCK = 21
PLACEBO_DRAWS = 2000
N_CUTS = 10
GATE_TOL = 1e-6
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


def build_work(t0: float) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    cands = sorted(CACHE.glob("trade_*.parquet"))
    if len(cands) != 1:
        raise FileNotFoundError(f"expected one trade cache in {CACHE}")
    pkg = pd.read_parquet(cands[0])
    tick(t0, f"trade cache {cands[0].name}: {len(pkg):,} packages")

    panel = asl.load_yhat_panel(asl.yhat_paths(REPO)["blk2"])
    tick(t0, f"forecast panel: {len(panel):,} stamps")

    pkg = pkg.copy()
    pkg["t"] = pd.to_datetime(pkg["timestamp"], utc=True)
    clocks_pkg = sorted(pkg["hhmm"].unique())
    n_rem = {c: len(clocks_pkg) - i for i, c in enumerate(clocks_pkg)}
    pkg["h_rem"] = pkg["hhmm"].map(n_rem).astype(float) * 0.5
    pkg["V_M"] = pkg["iv_hourly"].astype(float) ** 2 * pkg["h_rem"]

    pm = panel.set_index("t")[["rv_hat", "rv_raw", "in_fit"]].reset_index()
    pm["t"] = pd.to_datetime(pm["t"], utc=True) - pd.Timedelta(minutes=30)
    work = pkg.merge(pm, on="t", how="left").dropna(subset=["R", "rv_hat"]).copy()
    if not bool(work["in_fit"].all()):
        raise AssertionError("a joined trade bar sits outside the smear's fit mask")

    clocks = sorted(work["hhmm"].unique())
    work["w_slice"] = attach_share(work, diurnal_share(panel, clocks))
    work["slice_V0"] = work["w_slice"] * work["V_M"]
    work["s_V0"] = work["rv_hat"] - work["slice_V0"]
    work = work.sort_values(["date", "t"]).reset_index(drop=True)
    tick(t0, f"work: {len(work):,} bars on {work['date'].nunique()} days")
    return work, pkg, clocks


def variance_drop_panels(
    pkg: pd.DataFrame, clocks: list[str]
) -> dict[str, pd.DataFrame]:
    """Date × clock panels of V_M and ΔV. 15:30 next remaining is 0."""
    V = pkg.pivot_table(index="date", columns="hhmm", values="V_M", aggfunc="mean")
    V = V.reindex(columns=clocks).sort_index()
    dV = pd.DataFrame(index=V.index, columns=clocks, dtype=float)
    for i, c in enumerate(clocks):
        if c == clocks[-1]:
            dV[c] = V[c]
        else:
            dV[c] = V[c] - V[clocks[i + 1]]
    return {"V": V, "dV": dV}


def lagged_drop_slices(
    work: pd.DataFrame, panels: dict[str, pd.DataFrame], clocks: list[str]
) -> pd.DataFrame:
    V, dV = panels["V"], panels["dV"]
    dV_pos = dV.clip(lower=0.0)
    ratio = dV_pos / V.replace(0.0, np.nan)
    mu_dV = dV_pos.expanding(min_periods=PROFILE_MIN_DAYS).mean().shift(1)
    mu_V = V.expanding(min_periods=PROFILE_MIN_DAYS).mean().shift(1)
    mu_r = ratio.expanding(min_periods=PROFILE_MIN_DAYS).mean().shift(1)
    pi_means = mu_dV / mu_V.replace(0.0, np.nan)

    mi = pd.MultiIndex.from_arrays([work["date"], work["hhmm"]])
    work = work.copy()
    work["dV_peek"] = dV.stack().reindex(mi).to_numpy()
    work["slice_abs"] = mu_dV.stack().reindex(mi).to_numpy()
    work["slice_scaled"] = pi_means.stack().reindex(mi).to_numpy() * work[
        "V_M"
    ].to_numpy(float)
    work["slice_ratio"] = mu_r.stack().reindex(mi).to_numpy() * work["V_M"].to_numpy(
        float
    )
    # 15:30: scaled/ratio must collapse to V_M (share of remaining = 1)
    at_close = work["hhmm"] == CLOSE
    for col in ("slice_scaled", "slice_ratio"):
        close_ratio = work.loc[at_close, col] / work.loc[at_close, "V_M"]
        close_ratio = close_ratio.replace([np.inf, -np.inf], np.nan).dropna()
        if len(close_ratio) and float((close_ratio - 1.0).abs().max()) > 1e-9:
            raise AssertionError(f"{col} is not V_M at 15:30")
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
        "CI excludes 0": bool(
            (lo > 0 or hi < 0) and (2 * hat - hi > 0 or 2 * hat - lo < 0)
        ),
    }


def qlike(rv: np.ndarray, f: np.ndarray) -> float:
    m = np.isfinite(rv) & np.isfinite(f) & (rv > 0) & (f > 0)
    if int(m.sum()) < 10:
        return float("nan")
    return float(np.mean(np.log(f[m]) + rv[m] / f[m]))


def run_gate(f: Fills, s_v0: np.ndarray) -> None:
    tgt = pd.read_csv(INTRA / "rule_table_intraday_blk2.csv", index_col=0)
    tgt_cr = pd.read_csv(INTRA / "rule_table_intraday_crossed_blk2.csv", index_col=0)
    is_close = (f.w["hhmm"] == CLOSE).to_numpy(bool)
    pos = sign_pos(s_v0)
    qs = {
        "always short": -np.ones(len(f.w)),
        "always short, flat at 15:30": np.where(is_close, 0.0, -1.0),
        "sign(s)": pos,
        "always short, sign(s) close": np.where(is_close, pos, -1.0),
    }
    rows = []
    for name, q in qs.items():
        d_mid = f.daily(f.mid(q)[0])
        d_cr = f.daily(f.crossed(q)[0])
        rows.append(
            {
                "rule": name,
                "Sharpe mid": _sh(d_mid),
                "target mid": float(tgt.loc[name, "Sharpe_ann"]),
                "Sharpe crossed": _sh(d_cr),
                "target crossed": float(tgt_cr.loc[name, "Sharpe crossed-spread"]),
            }
        )
    g = pd.DataFrame(rows).set_index("rule")
    g["max |gap|"] = np.maximum(
        (g["Sharpe mid"] - g["target mid"]).abs(),
        (g["Sharpe crossed"] - g["target crossed"]).abs(),
    )
    worst = float(g["max |gap|"].max())
    print(g.to_string(float_format=lambda x: f"{x:+.7f}"))
    print(f"worst gap {worst:.2e}")
    if worst > GATE_TOL:
        raise AssertionError(f"gate failed: {worst:.3e}")
    print("GATE PASSED")
    g.to_csv(OUT / "21_gate.csv")


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    rng = np.random.default_rng(SEED)

    work, pkg, clocks = build_work(t0)
    panels = variance_drop_panels(pkg, clocks)
    work = lagged_drop_slices(work, panels, clocks)
    f = Fills(work)
    hhmm = work["hhmm"].to_numpy()
    rv = work["rv_raw"].to_numpy(float)
    rv_hat = work["rv_hat"].to_numpy(float)

    n_neg = int((work["dV_peek"].to_numpy(float) < 0).sum())
    n_fin = int(np.isfinite(work["dV_peek"]).sum())
    print(
        f"ex post ΔV < 0 on {n_neg} / {n_fin} scored bars "
        f"({100.0 * n_neg / max(n_fin, 1):.1f}%) — remaining implied rose"
    )

    rule("0. GATE — notebook V0 (realized-U share)")
    run_gate(f, work["s_V0"].to_numpy(float))

    slices = {
        "V0 w*V_M": work["slice_V0"].to_numpy(float),
        "abs lagged mean ΔV": work["slice_abs"].to_numpy(float),
        "scaled (μΔ/μV)*V_t": work["slice_scaled"].to_numpy(float),
        "ratio mean(ΔV/V)*V_t": work["slice_ratio"].to_numpy(float),
    }

    rule("1. Calibration: mean(RV) / mean(slice) by clock (1 = fair)")
    cal_rows = []
    for c in clocks:
        m = hhmm == c
        row = {"clock": c, "n": int(m.sum())}
        for name, sl in slices.items():
            a, b = rv[m], sl[m]
            ok = np.isfinite(a) & np.isfinite(b) & (b > 0)
            row[name] = float(a[ok].mean() / b[ok].mean()) if ok.any() else float("nan")
        # peeking, labeled, not a rule
        peek = work["dV_peek"].to_numpy(float)
        a, b = rv[m], np.clip(peek[m], 0.0, None)
        ok = np.isfinite(a) & np.isfinite(b) & (b > 0)
        row["PEEK ΔV (not a rule)"] = (
            float(a[ok].mean() / b[ok].mean()) if ok.any() else float("nan")
        )
        cal_rows.append(row)
    cal = pd.DataFrame(cal_rows).set_index("clock")
    print(cal.to_string(float_format=lambda x: f"{x:.3f}"))
    cal.to_csv(OUT / "21_calibration.csv")

    rule("2. QLIKE of slice as a forecast of RV_bar (lower better)")
    q_rows = []
    for name, sl in {
        **slices,
        "PEEK ΔV (not a rule)": np.clip(work["dV_peek"].to_numpy(float), 0, None),
    }.items():
        q_rows.append(
            {
                "slice": name,
                "QLIKE pooled": qlike(rv, sl),
                "corr(RV, slice)": float(
                    pd.Series(rv).corr(pd.Series(np.clip(sl, 0, None)))
                ),
            }
        )
    qtab = pd.DataFrame(q_rows).set_index("slice")
    print(qtab.to_string(float_format=lambda x: f"{x:.4f}"))
    qtab.to_csv(OUT / "21_qlike.csv")

    rule("3. sign(rv_hat - slice), same q conventions as the notebook")
    is_close = hhmm == CLOSE
    score_rows = []
    daily_sign = {}
    for name, sl in slices.items():
        s = rv_hat - sl
        pos = sign_pos(s)
        qs = {
            "sign(s)": pos,
            "hybrid": np.where(is_close, pos, -1.0),
        }
        for rname, q in qs.items():
            d_mid = f.daily(f.mid(q)[0])
            d_cr = f.daily(f.crossed(q)[0])
            score_rows.append(
                {
                    "slice": name,
                    "rule": rname,
                    "Sharpe mid": _sh(d_mid),
                    "Sharpe crossed": _sh(d_cr),
                    "mean/day mid": float(d_mid.mean()),
                    "pct long": 100.0 * float((pos > 0).mean()),
                    "n finite s": int(np.isfinite(s).sum()),
                }
            )
            if rname == "sign(s)":
                daily_sign[name] = d_mid
    sc = pd.DataFrame(score_rows)
    print(
        sc.pivot(index="slice", columns="rule", values="Sharpe mid").to_string(
            float_format=lambda x: f"{x:+.4f}"
        )
    )
    print("\ncrossed:")
    print(
        sc.pivot(index="slice", columns="rule", values="Sharpe crossed").to_string(
            float_format=lambda x: f"{x:+.4f}"
        )
    )
    sc.to_csv(OUT / "21_sharpe.csv", index=False)

    rule("4. Paired dSharpe sign(s) minus V0 sign(s), mid")
    pair_rows = []
    b = daily_sign["V0 w*V_M"].to_numpy(float)
    for name, d in daily_sign.items():
        if name == "V0 w*V_M":
            continue
        p = boot_dsharpe(d.to_numpy(float), b, rng)
        pair_row: dict[str, object] = dict(p)
        pair_row["slice"] = name
        pair_rows.append(pair_row)
        print(
            f"{name:28s}  dSharpe {p['dSharpe']:+.4f}  "
            f"pct [{p['pct_lo']:+.3f}, {p['pct_hi']:+.3f}]  "
            f"excl0={p['CI excludes 0']}"
        )
    pd.DataFrame(pair_rows).set_index("slice").to_csv(OUT / "21_paired.csv")

    rule("5. Placebo: 2000 rate-matched random signs on scaled-ΔV sign(s), mid")
    s_sc = rv_hat - work["slice_scaled"].to_numpy(float)
    pos = sign_pos(s_sc)
    p_long = float((pos > 0).mean())
    real = _sh(f.daily(f.mid(pos)[0]))
    codes, uniq = pd.factorize(work["date"], sort=True)
    r_long = f.mid(np.ones(len(work)))[0]
    q_p = np.where(rng.random((PLACEBO_DRAWS, len(work))) < p_long, 1.0, -1.0)
    bar = np.nan_to_num(q_p * r_long[None, :], nan=0.0)
    n_days = int(uniq.size)
    daily = np.zeros((PLACEBO_DRAWS, n_days))
    rows = np.repeat(np.arange(PLACEBO_DRAWS), len(work))
    cols = np.tile(codes, PLACEBO_DRAWS)
    np.add.at(daily, (rows, cols), bar.ravel())
    mu, sd = daily.mean(axis=1), daily.std(axis=1, ddof=1)
    draws = np.where(sd > 0, mu / sd * ANN, np.nan)
    plc = pd.Series(
        {
            "scaled sign(s) Sharpe": real,
            "long share": p_long,
            "placebo mean": float(np.nanmean(draws)),
            "placebo p95": float(np.nanpercentile(draws, 95)),
            "percentile": 100.0 * float(np.mean(draws < real)),
        }
    )
    print(plc.to_string())
    plc.to_csv(OUT / "21_placebo.csv")

    rule("6. Causality: lagged μΔ does not use this day's next stamp")
    dates = pd.DatetimeIndex(sorted(work["date"].unique()))
    cut_i = rng.choice(np.arange(200, len(dates) - 5), size=N_CUTS, replace=False)
    V = panels["V"].copy()
    base_abs = work["slice_abs"].to_numpy(float)
    caus_rows = []
    for k in range(N_CUTS):
        d = dates[cut_i[k]]
        c = clocks[int(rng.integers(0, len(clocks) - 1))]
        Vp = V.copy()
        if d in Vp.index:
            Vp.loc[d, clocks[clocks.index(c) + 1] if c != CLOSE else c] = (
                Vp.loc[d, clocks[clocks.index(c) + 1] if c != CLOSE else c] * 10.0
            )
        pkg2 = pkg.copy()
        # rebuild dV from perturbed V
        dVp = pd.DataFrame(index=Vp.index, columns=clocks, dtype=float)
        for i, cc in enumerate(clocks):
            dVp[cc] = Vp[cc] if cc == clocks[-1] else Vp[cc] - Vp[clocks[i + 1]]
        mu = dVp.clip(lower=0).expanding(min_periods=PROFILE_MIN_DAYS).mean().shift(1)
        mi = pd.MultiIndex.from_arrays([work["date"], work["hhmm"]])
        sl = mu.stack().reindex(mi).to_numpy()
        on_d = (work["date"] == d) & (work["hhmm"] == c)
        after = work["date"] > d
        caus_rows.append(
            {
                "date": str(d.date()),
                "clock": c,
                "abs moved on cut bar": int(
                    np.nansum(
                        np.abs(sl[on_d.to_numpy()] - base_abs[on_d.to_numpy()]) > 1e-15
                    )
                ),
                "abs moved after cut day": int(
                    np.nansum(
                        np.abs(sl[after.to_numpy()] - base_abs[after.to_numpy()])
                        > 1e-15
                    )
                ),
            }
        )
    caus = pd.DataFrame(caus_rows)
    print(caus.to_string(index=False))
    caus.to_csv(OUT / "21_causality.csv", index=False)
    assert int(caus["abs moved on cut bar"].sum()) == 0, (
        "lagged ΔV used this day's next V"
    )
    assert int((caus["abs moved after cut day"] > 0).sum()) > 0, (
        "causality test toothless"
    )
    print(f"assert passed: lagged μΔ is F_t ({N_CUTS} cuts).")

    # figure: calibration by clock
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    for col, style in (
        ("V0 w*V_M", "o-"),
        ("scaled (μΔ/μV)*V_t", "s-"),
        ("PEEK ΔV (not a rule)", "x--"),
    ):
        ax.plot(cal.index, cal[col], style, label=col)
    ax.axhline(1.0, color="0.5", lw=0.8)
    ax.set_ylabel("mean(RV) / mean(slice)")
    ax.set_xlabel("entry clock (ET)")
    ax.set_title("Per-period implied as a variance drop")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(OUT / "21_calibration.png", dpi=150)
    plt.close(fig)

    rule("7. Verdict")
    sh_v0 = float(
        sc.loc[
            (sc["slice"] == "V0 w*V_M") & (sc["rule"] == "sign(s)"), "Sharpe mid"
        ].iloc[0]
    )
    sh_sc = float(
        sc.loc[
            (sc["slice"] == "scaled (μΔ/μV)*V_t") & (sc["rule"] == "sign(s)"),
            "Sharpe mid",
        ].iloc[0]
    )
    print(f"sign(s) V0 {sh_v0:+.3f}  scaled-ΔV {sh_sc:+.3f}")
    print("do not adopt unless a paired CI excludes 0 and calibration beats V0.")
    print(f"\nwrote {OUT}")
    print(f"elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
