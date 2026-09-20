"""Proposal 22 — w as trailing mean of daily remaining shares, not ratio of means.

Notebook V0 (ratio of means):

    mhat_{d,c} = expanding mean of RV at clock c over dates < d
    w_{d,c}    = mhat_{d,c} / sum_{s >= c} mhat_{d,s}

This file (mean of shares):

    π_{d',c} = RV_{d',c} / sum_{s >= c} RV_{d',s}     on each prior day
    w_{d,c}  = expanding mean of π_{.,c} over dates < d

Same panel (in-fit, back to 2001), same 63-day min, shift(1). At 15:30 both
are 1. High-vol days dominate V0's clock means; mean-of-shares lets each
prior day vote equally on the remaining pie.

Gate V0 against the notebook. Do not adopt unless calibration and paired
dSharpe clear V0.

Run:  python writeup/intraday_proposals/22_share_mean.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "notebooks"))
import atm_straddle_lib as asl  # noqa: E402

REPO = asl.find_repo(Path(__file__).resolve().parent)
INTRA = REPO / "results" / "atm_straddle_intraday"
CACHE = INTRA / "cache"
OUT = INTRA / "proposals" / "22"
PROFILE_MIN_DAYS = 63
CLOSE = "15:30"
SEED = 0
BOOT_B = 2000
BOOT_BLOCK = 21
GATE_TOL = 1e-6
ANN = float(np.sqrt(asl.PERIODS_PER_YEAR))
N_CUTS = 10


def tick(t0, msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def _sh(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) < 2 or d.std(ddof=1) == 0:
        return float("nan")
    return float(d.mean() / d.std(ddof=1) * ANN)


def sign_pos(s):
    p = np.where(np.isfinite(s), np.sign(s), 0.0)
    return np.where((p == 0) & np.isfinite(s), -1.0, p)


def panel_clocks(panel: pd.DataFrame) -> pd.DataFrame:
    pf = panel.loc[panel["in_fit"].to_numpy(dtype=bool)].copy()
    clock = pd.to_datetime(pf["t"], utc=True).dt.tz_convert("America/New_York")
    clock = clock - pd.Timedelta(minutes=30)
    pf["pdate"] = clock.dt.normalize().dt.tz_localize(None)
    pf["phhmm"] = clock.dt.strftime("%H:%M")
    return pf


def w_ratio_of_means(wide: pd.DataFrame, clocks: list[str]) -> pd.DataFrame:
    mhat = wide[clocks].expanding(min_periods=PROFILE_MIN_DAYS).mean().shift(1)
    rem = mhat[clocks[::-1]].cumsum(axis=1)[clocks]
    w = mhat / rem
    if not bool(np.isclose(w[CLOSE].dropna().to_numpy(), 1.0).all()):
        raise AssertionError("V0 w not 1 at 15:30")
    return w


def w_mean_of_shares(wide: pd.DataFrame, clocks: list[str]) -> pd.DataFrame:
    """Each prior day: bar / remaining-to-close that day; then expanding mean."""
    pi = pd.DataFrame(index=wide.index, columns=clocks, dtype=float)
    for i, c in enumerate(clocks):
        rem = wide[clocks[i:]].sum(axis=1)
        pi[c] = wide[c] / rem.replace(0.0, np.nan)
    w = pi.expanding(min_periods=PROFILE_MIN_DAYS).mean().shift(1)
    close = w[CLOSE].dropna()
    if len(close) and float((close - 1.0).abs().max()) > 1e-9:
        raise AssertionError("mean-of-shares w not 1 at 15:30")
    return w


def attach(work, w):
    mi = pd.MultiIndex.from_arrays([work["date"], work["hhmm"]])
    return w.stack().reindex(mi).to_numpy()


class Fills2:
    def __init__(self, w):
        self.w = w
        self.ret = w["R"].to_numpy(float)
        self.entry = w["entry"].to_numpy(float)
        self.exit = w["exit"].to_numpy(float)
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

    def mid(self, q):
        return np.asarray(q, float) * self.ret

    def crossed(self, q):
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
        return pts / self.entry

    def daily(self, x):
        return (
            pd.Series(np.asarray(x, float), index=self.w.index)
            .groupby(self.w["date"])
            .sum()
        )


def boot_dsharpe(a, b, rng):
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


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    pd.set_option("display.width", 200)

    cands = sorted(CACHE.glob("trade_*.parquet"))
    pkg = pd.read_parquet(cands[0])
    tick(t0, f"cache {len(pkg):,}")
    panel = asl.load_yhat_panel(asl.yhat_paths(REPO)["blk2"])
    tick(t0, f"panel {len(panel):,}")

    pkg = pkg.copy()
    pkg["t"] = pd.to_datetime(pkg["timestamp"], utc=True)
    clocks_pkg = sorted(pkg["hhmm"].unique())
    n_rem = {c: len(clocks_pkg) - i for i, c in enumerate(clocks_pkg)}
    pkg["V_M"] = (
        pkg["iv_hourly"].astype(float) ** 2 * pkg["hhmm"].map(n_rem).astype(float) * 0.5
    )

    pm = panel.set_index("t")[["rv_hat", "rv_raw", "in_fit"]].reset_index()
    pm["t"] = pd.to_datetime(pm["t"], utc=True) - pd.Timedelta(minutes=30)
    work = pkg.merge(pm, on="t", how="left").dropna(subset=["R", "rv_hat"]).copy()
    clocks = sorted(work["hhmm"].unique())

    pf = panel_clocks(panel)
    wide = pf.pivot_table(
        index="pdate", columns="phhmm", values="rv_raw", aggfunc="mean"
    ).sort_index()
    wide = wide.reindex(columns=clocks)
    w0 = w_ratio_of_means(wide, clocks)
    w1 = w_mean_of_shares(wide, clocks)
    work["w_V0"] = attach(work, w0)
    work["w_share"] = attach(work, w1)
    work["slice_V0"] = work["w_V0"] * work["V_M"]
    work["slice_share"] = work["w_share"] * work["V_M"]
    work = work.sort_values(["date", "t"]).reset_index(drop=True)
    tick(t0, f"work {len(work):,}  days {work['date'].nunique()}")

    # how different are the two w's?
    both = np.isfinite(work["w_V0"]) & np.isfinite(work["w_share"])
    print(
        f"corr(w_V0, w_share) {float(pd.Series(work['w_V0'][both]).corr(work['w_share'][both])):.4f}  "
        f"median |w_share/w_V0 - 1| {float(np.median(np.abs(work.loc[both, 'w_share'] / work.loc[both, 'w_V0'] - 1))):.4f}"
    )
    byc = work.loc[both].groupby("hhmm")[["w_V0", "w_share"]].median()
    print("median w by clock:")
    print(byc.to_string(float_format=lambda x: f"{x:.4f}"))
    byc.to_csv(OUT / "22_w_by_clock.csv")

    f = Fills2(work)
    tgt = pd.read_csv(INTRA / "rule_table_intraday_blk2.csv", index_col=0)
    tgt_cr = pd.read_csv(INTRA / "rule_table_intraday_crossed_blk2.csv", index_col=0)
    is_close = (work["hhmm"] == CLOSE).to_numpy(bool)
    pos0 = sign_pos((work["rv_hat"] - work["slice_V0"]).to_numpy(float))
    qs = {
        "always short": -np.ones(len(work)),
        "always short, flat at 15:30": np.where(is_close, 0.0, -1.0),
        "sign(s)": pos0,
        "always short, sign(s) close": np.where(is_close, pos0, -1.0),
    }
    print("\nGATE V0")
    gaps = []
    for name, q in qs.items():
        sm, sc = _sh(f.daily(f.mid(q))), _sh(f.daily(f.crossed(q)))
        tm, tc = (
            float(tgt.loc[name, "Sharpe_ann"]),
            float(tgt_cr.loc[name, "Sharpe crossed-spread"]),
        )
        gap = max(abs(sm - tm), abs(sc - tc))
        gaps.append(gap)
        print(f"  {name:36s} mid {sm:+.7f} ({tm:+.7f})  cr {sc:+.7f} ({tc:+.7f})")
    worst = max(gaps)
    print(f"worst gap {worst:.2e}")
    if worst > GATE_TOL:
        raise AssertionError("gate failed")
    print("GATE PASSED")

    rv = work["rv_raw"].to_numpy(float)
    print("\ncalibration mean(RV)/mean(slice)")
    cal_rows = []
    for c in clocks:
        m = work["hhmm"].to_numpy() == c
        row = {"clock": c}
        for lab, sl in (
            ("V0", work["slice_V0"]),
            ("mean of shares", work["slice_share"]),
        ):
            a, b = rv[m], sl.to_numpy(float)[m]
            ok = np.isfinite(a) & np.isfinite(b) & (b > 0)
            row[lab] = float(a[ok].mean() / b[ok].mean())
        cal_rows.append(row)
    cal = pd.DataFrame(cal_rows).set_index("clock")
    print(cal.to_string(float_format=lambda x: f"{x:.3f}"))
    cal.to_csv(OUT / "22_calibration.csv")

    print("\nsign(s) / hybrid")
    pos1 = sign_pos((work["rv_hat"] - work["slice_share"]).to_numpy(float))
    rows = []
    d0 = f.daily(f.mid(pos0))
    d1 = f.daily(f.mid(pos1))
    for lab, pos in (("V0", pos0), ("mean of shares", pos1)):
        for rname, q in (("sign(s)", pos), ("hybrid", np.where(is_close, pos, -1.0))):
            rows.append(
                {
                    "w": lab,
                    "rule": rname,
                    "Sharpe mid": _sh(f.daily(f.mid(q))),
                    "Sharpe crossed": _sh(f.daily(f.crossed(q))),
                    "pct long": 100.0 * float((pos > 0).mean()),
                }
            )
    sc = pd.DataFrame(rows)
    print(sc.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    sc.to_csv(OUT / "22_sharpe.csv", index=False)

    p = boot_dsharpe(d1.to_numpy(), d0.to_numpy(), rng)
    print("\npaired dSharpe mean-of-shares minus V0, sign(s) mid:")
    print(pd.Series(p).to_string())
    pd.Series(p).to_csv(OUT / "22_paired.csv")

    # causality: perturb RV on day d clock c in the panel-wide; w_share on day d unchanged
    days = pd.DatetimeIndex(sorted(work["date"].unique()))
    cut = rng.choice(np.arange(200, len(days) - 2), size=N_CUTS, replace=False)
    base = work["w_share"].to_numpy(float)
    caus = []
    wide2_src = wide.copy()
    for k in range(N_CUTS):
        d = days[cut[k]]
        c = clocks[int(rng.integers(0, len(clocks) - 1))]
        W = wide2_src.copy()
        if d in W.index:
            W.loc[d, c] = W.loc[d, c] * 10.0
        w2 = w_mean_of_shares(W, clocks)
        sl = attach(work, w2)
        on = (work["date"] == d) & (work["hhmm"] == c)
        after = work["date"] > d
        caus.append(
            {
                "date": str(d.date()),
                "clock": c,
                "moved on cut": int(
                    np.nansum(np.abs(sl[on.to_numpy()] - base[on.to_numpy()]) > 1e-15)
                ),
                "moved after": int(
                    np.nansum(
                        np.abs(sl[after.to_numpy()] - base[after.to_numpy()]) > 1e-15
                    )
                ),
            }
        )
    caus = pd.DataFrame(caus)
    print("\nCAUSALITY")
    print(caus.to_string(index=False))
    assert int(caus["moved on cut"].sum()) == 0
    assert int((caus["moved after"] > 0).sum()) > 0
    print("assert passed: mean-of-shares w is F_t")
    caus.to_csv(OUT / "22_causality.csv", index=False)

    print(f"\nwrote {OUT}  elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
