"""Proposal 23 — short only high-Sharpe daytime clocks, sign(s) at 15:30;
event days as q=0 (days stay in the series).

Two points:

1. Daytime: q = -1 only on clocks whose ALWAYS-SHORT edge is large; 15:30 is
   sign(s). Picking those clocks from the full-sample table is in-sample.
   The causal version uses each clock's expanding Sharpe of always-short R'
   on prior days (min 63, shift 1) and shorts iff that trailing Sharpe > H.
   Proposal 01(a) already rejected trailing-edge clock filters without the
   close sign(s) overlay. This file is 01(a) daytime + 01(c') at the close.

2. Event-day flat: q_15:30 = 0 on FOMC-statement and month-end days, but
   the daily series still has 866 rows (return 0). Dropping those days
   before Sharpe is cherry-picking; both numbers are printed.

Gate the notebook hybrid / always-short / sign(s) first.

Run:  python writeup/intraday_proposals/23_clock_short_event_weight.py
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
OUT = INTRA / "proposals" / "23"
PROFILE_MIN_DAYS = 63
CLOSE = "15:30"
SEED = 0
BOOT_B = 2000
BOOT_BLOCK = 21
GATE_TOL = 1e-3
ANN = float(np.sqrt(asl.PERIODS_PER_YEAR))
H_TRAIL = (0.0, 1.0)


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


def w_mean_of_shares(wide, clocks):
    pi = pd.DataFrame(index=wide.index, columns=clocks, dtype=float)
    for i, c in enumerate(clocks):
        rem = wide[clocks[i:]].sum(axis=1)
        pi[c] = wide[c] / rem.replace(0.0, np.nan)
    w = pi.expanding(min_periods=PROFILE_MIN_DAYS).mean().shift(1)
    return w


class Fills:
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


def trailing_sharpe_ok(
    r_as: pd.Series, dates: pd.Series, clocks: np.ndarray, H: float
) -> np.ndarray:
    """Per clock, expanding Sharpe of always-short R' on prior days; True if > H."""
    out = np.zeros(len(r_as), dtype=bool)
    for c in pd.unique(clocks):
        m = clocks == c
        s = pd.Series(
            r_as.to_numpy()[m], index=pd.DatetimeIndex(dates.to_numpy()[m])
        ).sort_index()
        mu = s.expanding(min_periods=PROFILE_MIN_DAYS).mean().shift(1)
        sd = s.expanding(min_periods=PROFILE_MIN_DAYS).std(ddof=1).shift(1)
        sh = (mu / sd) * ANN
        ok = (sh > H).reindex(dates.to_numpy()[m]).fillna(False).to_numpy()
        out[np.flatnonzero(m)] = ok
    return out


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    pd.set_option("display.width", 220)

    pkg = pd.read_parquet(sorted(CACHE.glob("trade_*.parquet"))[-1])
    tick(t0, f"cache {len(pkg):,}")
    panel = asl.load_yhat_panel(asl.yhat_paths(REPO)["blk2"])
    tick(t0, "panel")

    pkg = pkg.copy()
    pkg["t"] = pd.to_datetime(pkg["timestamp"], utc=True)
    clocks_all = sorted(pkg["hhmm"].unique())
    n_rem = {c: len(clocks_all) - i for i, c in enumerate(clocks_all)}
    pkg["V_M"] = (
        pkg["iv_hourly"].astype(float) ** 2 * pkg["hhmm"].map(n_rem).astype(float) * 0.5
    )

    pm = panel.set_index("t")[["rv_hat", "rv_raw", "in_fit"]].reset_index()
    pm["t"] = pd.to_datetime(pm["t"], utc=True) - pd.Timedelta(minutes=30)
    work = pkg.merge(pm, on="t", how="left").dropna(subset=["R", "rv_hat"]).copy()
    clocks = sorted(work["hhmm"].unique())
    day = work["hhmm"].to_numpy() != CLOSE
    is_close = work["hhmm"].to_numpy() == CLOSE

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
    work = work.sort_values(["date", "t"]).reset_index(drop=True)
    tick(t0, f"work {len(work):,} days {work['date'].nunique()}")

    f = Fills(work)
    pos = sign_pos(work["s"].to_numpy(float))
    q_as = -np.ones(len(work))
    q_sg = pos
    q_hyb = np.where(is_close, pos, -1.0)

    tgt = pd.read_csv(INTRA / "rule_table_intraday_blk2.csv", index_col=0)
    print("GATE")
    for name, q in (
        ("always short", q_as),
        ("sign(s)", q_sg),
        ("always short, sign(s) close", q_hyb),
    ):
        got = _sh(f.daily(f.mid(q)))
        want = float(tgt.loc[name, "Sharpe_ann"])
        print(f"  {name:36s} {got:+.4f}  target {want:+.4f}  gap {abs(got - want):.2e}")
        if abs(got - want) > GATE_TOL:
            raise AssertionError(name)
    print("GATE PASSED")

    # per-clock always-short Sharpe (one bar / day)
    print("\nper-clock always-short Sharpe_ann (one bar/day)")
    clk_rows = []
    for c in clocks:
        m = work["hhmm"].to_numpy() == c
        sh = _sh(work.loc[m, "R"].to_numpy() * -1.0)
        clk_rows.append({"clock": c, "n": int(m.sum()), "AS Sharpe": sh})
        print(f"  {c}  {sh:+.3f}")
    clk = pd.DataFrame(clk_rows).set_index("clock")
    clk.to_csv(OUT / "23_as_by_clock.csv")
    insample_gt1 = [
        c for c in clocks if c != CLOSE and float(clk.loc[c, "AS Sharpe"]) > 1.0
    ]
    insample_gt15 = [
        c for c in clocks if c != CLOSE and float(clk.loc[c, "AS Sharpe"]) > 1.5
    ]
    print("IN-SAMPLE clocks AS Sharpe > 1.0:", insample_gt1)
    print("IN-SAMPLE clocks AS Sharpe > 1.5:", insample_gt15)

    r_as = pd.Series(-work["R"].to_numpy(float), index=work.index)
    flags = asl.fomc_and_monthend(
        pd.DatetimeIndex(pd.to_datetime(work["date"].unique())), REPO
    )
    ev = flags["is_me"].astype(bool) | flags["is_fomc"].astype(bool)
    ev_bar = work["date"].map(ev.to_dict()).fillna(False).astype(bool).to_numpy()
    print(
        f"\nevent days (FOMC or month-end): {int(ev.sum())} of {work['date'].nunique()}"
    )

    def q_subset(clocks_on, event_zero=True):
        q = np.zeros(len(work))
        on = work["hhmm"].isin(clocks_on).to_numpy()
        q[on] = -1.0
        q[is_close] = pos[is_close]
        if event_zero:
            q[is_close & ev_bar] = 0.0
        return q

    rows = []
    dailies = {}

    def add(name, q, note):
        d_mid = f.daily(f.mid(q))
        d_cr = f.daily(f.crossed(q))
        dailies[name] = d_mid
        nzero = int((d_mid == 0).sum())
        rows.append(
            {
                "rule": name,
                "note": note,
                "n_days": int(d_mid.notna().sum()),
                "n_daily_zero": nzero,
                "Sharpe mid": _sh(d_mid),
                "mean/day": float(d_mid.mean()),
                "Sharpe crossed": _sh(d_cr),
                "bars short/day": float(
                    (q < 0).reshape(-1).sum() / work["date"].nunique()
                ),
            }
        )

    add("hybrid (all daytime AS + sign close)", q_hyb, "notebook")
    add(
        "hybrid, 15:30 q=0 on events (days kept)",
        np.where(is_close & ev_bar, 0.0, q_hyb),
        "weight 0",
    )

    # illegal drop: Sharpe on non-event days only
    d_hyb = f.daily(f.mid(q_hyb))
    non_ev_days = ~pd.Series(d_hyb.index.map(ev.to_dict()), index=d_hyb.index).fillna(
        False
    ).astype(bool)
    sh_drop = _sh(d_hyb.loc[non_ev_days.to_numpy()])
    print(
        f"\nCHERRY-PICK vs weight-0: hybrid Sharpe on non-event days only "
        f"(n={int(non_ev_days.to_numpy().sum())}) = {sh_drop:+.3f}; "
        f"weight-0 on all {len(d_hyb)} days is printed in the table"
    )

    for label, subset in (("IS AS>1", insample_gt1), ("IS AS>1.5", insample_gt15)):
        add(
            f"{label} daytime AS + sign close",
            q_subset(subset, event_zero=False),
            "IN-SAMPLE clocks",
        )
        add(
            f"{label} + 15:30 q=0 events",
            q_subset(subset, event_zero=True),
            "IN-SAMPLE + weight 0",
        )

    for H in H_TRAIL:
        ok = trailing_sharpe_ok(r_as, work["date"], work["hhmm"].to_numpy(), H)
        q = np.zeros(len(work))
        q[day & ok] = -1.0
        q[is_close] = pos[is_close]
        add(f"causal trail AS Sharpe>{H:g} + sign close", q, "F_t")
        qe = q.copy()
        qe[is_close & ev_bar] = 0.0
        add(
            f"causal trail AS Sharpe>{H:g} + sign close, events q=0",
            qe,
            "F_t + weight 0",
        )

    tab = pd.DataFrame(rows)
    print("\n" + tab.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    tab.to_csv(OUT / "23_rules.csv", index=False)

    print("\npaired dSharpe vs notebook hybrid (mid)")
    b = dailies["hybrid (all daytime AS + sign close)"].to_numpy(float)
    pair_rows = []
    for name, d in dailies.items():
        if name.startswith("hybrid (all"):
            continue
        p = boot_dsharpe(d.to_numpy(float), b, rng)
        p["rule"] = name
        pair_rows.append(p)
        print(
            f"  {name:55s} {p['dSharpe']:+.3f}  "
            f"[{p['pct_lo']:+.2f}, {p['pct_hi']:+.2f}]  excl0={p['CI excludes 0']}"
        )
    pd.DataFrame(pair_rows).to_csv(OUT / "23_paired.csv", index=False)

    # close-leg only: weight-0 vs drop
    close = work[is_close].copy()
    r0 = pd.Series(
        pos[is_close] * close["R"].to_numpy(), index=pd.DatetimeIndex(close["date"])
    )
    ev_close = close["date"].map(ev.to_dict()).fillna(False).astype(bool).to_numpy()
    r_w0 = pd.Series(np.where(ev_close, 0.0, r0.to_numpy()), index=r0.index)
    r_drop = r0.loc[~ev_close]
    print("\n15:30 sign(s) only:")
    print(f"  unfiltered (n={len(r0)}) Sharpe {_sh(r0):+.3f}")
    print(
        f"  q=0 on events, days kept (n={len(r_w0)}, zeros={int((r_w0 == 0).sum())}) Sharpe {_sh(r_w0):+.3f}"
    )
    print(
        f"  DROP event days (n={len(r_drop)}) Sharpe {_sh(r_drop):+.3f}  <- cherry-pick"
    )
    pd.Series(
        {
            "close unfiltered": _sh(r0),
            "close weight-0": _sh(r_w0),
            "close drop events": _sh(r_drop),
            "n unfiltered": len(r0),
            "n drop": len(r_drop),
        }
    ).to_csv(OUT / "23_event_weight.csv")

    print(f"\nwrote {OUT}  elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
