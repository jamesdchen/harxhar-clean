"""58 - selling the 15:30 straddle on the third Friday of the month: the test.

Disclosure first, because it fixes how much this can prove.  Study 56 judged
eleven scheduled day types on the LONG side by a rule written in advance.  The
third Friday (standard monthly expiration) failed that rule in the opposite
direction in BOTH samples it printed: long at the ask -0.31 premium (t -3.2) on
the deck days and -0.57 (t -4.2) on the 2024-05..2025-12 holdout; short at the
bid +0.25 and +0.55.  So the holdout has been seen.  No options data in this
repository is untouched by that look, and this study cannot be a clean
out-of-sample test.  What it can do is ask the questions that would break the
lead if it were an accident of where we looked:

  A  multiplicity: treat the deck period as the sample that selected the day
     type and the holdout as its confirmation, and charge the holdout for the
     whole family it was picked from (11 day types): one-sided t-distribution
     p-value times 11.
  B  is it an EXPIRATION effect or a Friday effect?  Short at the bid on third
     Fridays against the other Fridays of the month (1st, 2nd, 4th, 5th), and
     against the third Thursday and third Wednesday (the same week, not an
     expiration).  This is the sharp placebo: a lead that is really "Fridays"
     or "mid-month" dies here.
  C  a random-day placebo: the same number of randomly chosen sessions, B =
     2000, seed 0.
  D  mechanism, underlying only, 1998-2019 (before any option in the study) and
     2020-2024: on third Fridays against other Fridays, the last half hour's
     realized variance and squared NET move, each relative to that bar's own
     trailing-63-session median, and the ratio of the two.  A pinned close is a
     small net move for the variance traded.
  E  risk: by year, hit rate, worst days, the mean without its three best and
     three worst days, index points per contract at the quoted bid.
  F  the books: 15:30 sign(s) with third Fridays forced short (alone, and with
     month-ends forced long), paired block bootstrap and a random-days placebo;
     the short afternoon hold books of proposal 43 on third Fridays.

THE RULE UNDER TEST: on the third Friday of each month (the preceding session
if that Friday is a holiday) sell the 15:30 nearest-OTM SPX 0DTE straddle at
the quoted bid and hold it to cash settlement.

Written before running, the lead is called SUPPORTED only if all four hold:
  (1) holdout short-at-the-bid mean > 0 with 11 x one-sided p < 0.05;
  (2) third Fridays beat the other Fridays in the deck period AND in the
      holdout, and the all-sessions bootstrap interval of the difference
      excludes zero;
  (3) random-day placebo p < 0.05 on all sessions;
  (4) the yearly mean is positive in at least two thirds of the years.
Otherwise it stays a lead.  Either way the next truly new evidence is 2026.

GATES  ridge sign(s) 1.338322 / 0.869588; the chain-built 15:30 straddle return
       equals the deck's R; study 56's third-Friday rows reproduce (52 deck days
       long at the ask -0.306, 20 holdout days -0.567).

Run:  python writeup/intraday_proposals/58_third_friday_short.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "notebooks", ROOT / "writeup"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import atm_straddle_lib as asl  # noqa: E402

HERE = Path(__file__).resolve().parent
DECK = ROOT / "results" / "atm_straddle_0dte_1530"
OUT = DECK / "proposals" / "58"
P43_DAILY = (
    ROOT
    / "results"
    / "atm_straddle_intraday_holdclose"
    / "proposals"
    / "43"
    / "a_daily_by_clock.csv"
)
ANN = float(np.sqrt(asl.PERIODS_PER_YEAR))
DECK_END, OOS_START, PRE_OPTIONS_END = "2024-04-30", "2024-05-01", "2019-12-31"
FAMILY_SIZE = 11  # the day types study 56 judged
TRAIL = 63
B, BLOCK, SEED = 2000, 21, 0
GATE_SHARPE = (1.338322, 0.869588)
GATE_TOL = 1e-6
GATE_R_TOL = 1e-6  # the chain stores quotes as float32
GATE_56 = {"deck": (52, -0.306), "holdout": (20, -0.567)}
GATE_56_TOL = 5e-4


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sharpe(x: np.ndarray) -> float:
    return float(np.mean(x) / np.std(x, ddof=1) * ANN)


def tstat(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    return float(x.mean() / x.std(ddof=1) * np.sqrt(len(x))) if len(x) > 2 else np.nan


def welch(a: np.ndarray, b: np.ndarray) -> float:
    return float(
        (a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    )


def boot_diff_ci(
    a: np.ndarray, b: np.ndarray, rng: np.random.Generator
) -> tuple[float, float]:
    """Interval of mean(a) - mean(b) for two independent groups of days."""
    ia = rng.integers(0, len(a), size=(B, len(a)))
    ib = rng.integers(0, len(b), size=(B, len(b)))
    d = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(lo), float(hi)


def nth_weekday_sessions(cal: pd.DatetimeIndex, weekday: int, n: int) -> np.ndarray:
    """Flag the n-th ``weekday`` of each month, moved to the preceding session of
    the same month when that date is not a session (study 56's convention)."""
    months = pd.period_range(cal[0], cal[-1], freq="M").to_timestamp()
    off = (weekday - months.dayofweek) % 7 + 7 * (n - 1)
    days = pd.DatetimeIndex(months + pd.to_timedelta(off, unit="D"))
    days = days[days.month == months.month]  # a 5th occurrence may not exist
    pos = cal.searchsorted(days, side="right") - 1
    ok = pos >= 0
    got = cal[pos[ok]]
    same = (got.year == days[ok].year) & (got.month == days[ok].month)
    out = np.zeros(len(cal), bool)
    out[pos[ok][same]] = True
    return out


def chain_tape(p43: ModuleType) -> pd.DataFrame:
    stamp, _ = p43.session_stamps()
    sessions = pd.DatetimeIndex(stamp.index).difference(p43.half_sessions(stamp))
    ch = p43.build_chain(stamp, sessions)
    k = p43.CLOCKS.index("15:30")
    mid, bid, sc = ch["entry"][:, k], ch["bid"][:, k], ch["S_close"]
    ask = 2.0 * mid - bid
    settle = np.maximum(sc - ch["K_c"][:, k], 0.0) + np.maximum(
        ch["K_p"][:, k] - sc, 0.0
    )
    ok = np.isfinite(mid) & (mid > 0) & (bid > 0) & np.isfinite(settle)
    df = pd.DataFrame(
        {
            "R_mid": settle / mid - 1.0,
            "R_ask": settle / ask - 1.0,
            "S_bid": -(settle / bid - 1.0),
            "S_bid_pts": bid - settle,
            "premium_pts": mid,
        },
        index=ch["dates"],
    )[ok]
    deck = pd.read_parquet(DECK / "daily_blk2.parquet")["R"]
    j = df.join(deck.rename("R_deck"), how="inner")
    dev = float((j["R_mid"] - j["R_deck"]).abs().max())
    assert len(j) == len(deck) and dev < GATE_R_TOL, (len(j), dev)
    print(
        f"GATE  chain-built 15:30 straddle return vs the deck's on {len(j)} days: {dev:.1e}"
    )
    return df


def main() -> None:  # noqa: PLR0915
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    rng = np.random.default_rng(SEED)
    p43 = _load(HERE / "43_causal_entry_over_time.py", "p43_causal_entry")
    p56 = _load(HERE / "56_close_calendar_family.py", "p56_close_calendar")
    df = chain_tape(p43)
    stamp, _ = p43.session_stamps()
    cal = p56.session_calendar(pd.DatetimeIndex(stamp.index))
    flags = pd.DataFrame(
        {
            "third Friday": nth_weekday_sessions(cal, 4, 3),
            "third Thursday": nth_weekday_sessions(cal, 3, 3),
            "third Wednesday": nth_weekday_sessions(cal, 2, 3),
            **{f"Friday #{n}": nth_weekday_sessions(cal, 4, n) for n in (1, 2, 4, 5)},
        },
        index=cal,
    )
    ref = p56.day_types(cal)["third Friday"].to_numpy()
    assert (flags["third Friday"].to_numpy() == ref).all(), (
        "third-Friday rule differs from 56"
    )
    # a moved expiration (holiday Friday) lands on a Thursday: keep it out of the placebos
    tf_all = flags["third Friday"].to_numpy()
    for c in flags.columns[1:]:
        flags[c] = flags[c].to_numpy() & ~tf_all
    flags["other Fridays"] = flags[[f"Friday #{n}" for n in (1, 2, 4, 5)]].any(axis=1)
    f = flags.reindex(df.index)
    periods = {
        "deck": np.asarray(df.index <= DECK_END),
        "holdout": np.asarray(df.index >= OOS_START),
        "all": np.ones(len(df), bool),
    }
    tf = f["third Friday"].to_numpy()
    for name, (n_ref, mean_ref) in GATE_56.items():
        m = periods[name] & tf
        got = (int(m.sum()), float(df.loc[m, "R_ask"].mean()))
        assert got[0] == n_ref and abs(got[1] - mean_ref) < GATE_56_TOL, (name, got)
    print("GATE  study 56's third-Friday rows reproduce (52 deck, 20 holdout)")

    # ------------------------------------------------------------ A, B, C --
    rows = []
    for pname, pm in periods.items():
        for col in flags.columns:
            x = df.loc[pm & f[col].to_numpy(), "S_bid"].to_numpy()
            pts = df.loc[pm & f[col].to_numpy(), "S_bid_pts"].to_numpy()
            rows.append(
                {
                    "period": pname,
                    "day type": col,
                    "n": len(x),
                    "short_bid_mean": x.mean(),
                    "t": tstat(x),
                    "hit": float((x > 0).mean()),
                    "pts_per_contract": pts.mean(),
                    "worst": x.min(),
                }
            )
        x = df.loc[pm & ~tf, "S_bid"].to_numpy()
        rows.append(
            {
                "period": pname,
                "day type": "every other session",
                "n": len(x),
                "short_bid_mean": x.mean(),
                "t": tstat(x),
                "hit": float((x > 0).mean()),
                "pts_per_contract": df.loc[pm & ~tf, "S_bid_pts"].mean(),
                "worst": x.min(),
            }
        )
    ab = pd.DataFrame(rows)
    ab.to_csv(OUT / "ab_short_at_bid_by_day_type.csv", index=False)
    print("\nA/B  short the 15:30 straddle at the bid, premium units")
    print(ab.round(3).to_string(index=False))

    hold = df.loc[periods["holdout"] & tf, "S_bid"].to_numpy()
    p_one = float(stats.t.sf(tstat(hold), df=len(hold) - 1))
    c1 = bool(hold.mean() > 0 and FAMILY_SIZE * p_one < 0.05)
    print(
        f"\n(1) holdout n {len(hold)}, mean {hold.mean():+.3f}, t {tstat(hold):+.2f}, one-sided p "
        f"{p_one:.5f}, x{FAMILY_SIZE} = {FAMILY_SIZE * p_one:.4f} -> {'PASS' if c1 else 'FAIL'}"
    )

    diffs = {}
    for pname, pm in periods.items():
        a = df.loc[pm & tf, "S_bid"].to_numpy()
        b = df.loc[pm & f["other Fridays"].to_numpy(), "S_bid"].to_numpy()
        lo, hi = boot_diff_ci(a, b, rng)
        diffs[pname] = (a.mean() - b.mean(), lo, hi, welch(a, b))
        print(
            f"(2) third Friday minus other Fridays [{pname}]: {a.mean() - b.mean():+.3f} "
            f"[{lo:+.3f}, {hi:+.3f}], Welch t {welch(a, b):+.2f}"
        )
    c2 = bool(diffs["deck"][0] > 0 and diffs["holdout"][0] > 0 and diffs["all"][1] > 0)
    print(f"    -> {'PASS' if c2 else 'FAIL'}")

    s_all = df["S_bid"].to_numpy()
    k = int(tf.sum())
    draws = np.array(
        [s_all[rng.choice(len(s_all), k, replace=False)].mean() for _ in range(B)]
    )
    p_rand = float((1 + (draws >= s_all[tf].mean()).sum()) / (1 + B))
    c3 = p_rand < 0.05
    print(
        f"(3) random-day placebo, {k} sessions: observed {s_all[tf].mean():+.3f}, placebo mean "
        f"{draws.mean():+.3f}, p {p_rand:.4f} -> {'PASS' if c3 else 'FAIL'}"
    )

    # ------------------------------------------------------------------ E --
    yr = df.index.year
    by_year = pd.DataFrame(
        [
            {
                "year": int(y),
                "n": int((tf & (yr == y)).sum()),
                "short_bid_mean": float(df.loc[tf & (yr == y), "S_bid"].mean()),
                "hit": float((df.loc[tf & (yr == y), "S_bid"] > 0).mean()),
                "pts_per_contract": float(df.loc[tf & (yr == y), "S_bid_pts"].mean()),
                "other_fridays_mean": float(
                    df.loc[f["other Fridays"].to_numpy() & (yr == y), "S_bid"].mean()
                ),
            }
            for y in sorted(set(yr))
        ]
    )
    by_year.to_csv(OUT / "e_by_year.csv", index=False)
    c4 = bool((by_year["short_bid_mean"] > 0).mean() >= 2 / 3)
    x = np.sort(s_all[tf])
    print("\nE  third Fridays by year")
    print(by_year.round(3).to_string(index=False))
    print(
        f"(4) years with a positive mean: {int((by_year['short_bid_mean'] > 0).sum())} of "
        f"{len(by_year)} -> {'PASS' if c4 else 'FAIL'}"
    )
    print(
        f"    all {k} third Fridays: mean {x.mean():+.3f}, without the 3 best and 3 worst "
        f"{x[3:-3].mean():+.3f}, worst three {x[:3].round(2).tolist()}, hit {float((x > 0).mean()):.2f}; "
        f"index points per contract at the bid {df.loc[tf, 'S_bid_pts'].mean():+.2f} "
        f"(mean premium {df.loc[tf, 'premium_pts'].mean():.1f})"
    )
    print(
        f"\nVERDICT: {'SUPPORTED' if all((c1, c2, c3, c4)) else 'STAYS A LEAD'} "
        f"(criteria 1-4: {c1}, {c2}, {c3}, {c4})"
    )

    # ------------------------------------------------------------------ D --
    c = pd.read_parquet(
        ROOT / "data" / "core_stats.parquet",
        columns=["endbartime", "sumret", "sumret2"],
    )
    c["t"] = pd.to_datetime(c["endbartime"])
    c = c[(c["t"] >= "1998-01-05") & (c["t"].dt.strftime("%H:%M") == "16:00")]
    u = pd.DataFrame(
        {"rv": c["sumret2"].to_numpy(), "net2": c["sumret"].to_numpy() ** 2},
        index=pd.DatetimeIndex(c["t"].dt.normalize()),
    )
    u = u[(u["rv"] > 0) & (u["net2"] > 0)]
    med = u["rv"].rolling(TRAIL).median().shift(1)
    u = u.assign(l_rv=np.log(u["rv"] / med), l_net=np.log(u["net2"] / med)).dropna()
    fu = flags.reindex(u.index).fillna(False)
    rows = []
    for name, m in (
        ("1998-2019", u.index <= PRE_OPTIONS_END),
        ("2020-2024", u.index > PRE_OPTIONS_END),
    ):
        a, b = m & fu["third Friday"].to_numpy(), m & fu["other Fridays"].to_numpy()
        for col, label in (
            ("l_rv", "realized variance"),
            ("l_net", "squared net move"),
        ):
            rows.append(
                {
                    "sample": name,
                    "quantity": f"last half hour's {label} / own 63-session median",
                    "third_Fridays": float(np.exp(u.loc[a, col].mean())),
                    "n": int(a.sum()),
                    "other_Fridays": float(np.exp(u.loc[b, col].mean())),
                    "Welch_t": welch(
                        u.loc[a, col].to_numpy(), u.loc[b, col].to_numpy()
                    ),
                }
            )
    d_tab = pd.DataFrame(rows)
    d_tab.to_csv(OUT / "d_underlying.csv", index=False)
    print("\nD  the underlying's last half hour, third Fridays against other Fridays")
    print(d_tab.round(3).to_string(index=False))

    # ------------------------------------------------------------------ F --
    deck = pd.read_parquet(DECK / "daily_blk2.parquet").sort_index()
    di = pd.DatetimeIndex(deck.index)
    r = deck["R"].to_numpy(float)
    ask = (deck["ask_c"] + deck["ask_p"]).to_numpy(float)
    bid = (deck["bid_c"] + deck["bid_p"]).to_numpy(float)
    ex = deck["exit"].to_numpy(float)
    q = np.where(deck["signal"].to_numpy(float) > 0, 1.0, -1.0)

    def crossed(qq: np.ndarray) -> np.ndarray:
        return np.where(qq > 0, qq * (ex / ask - 1.0), qq * (ex / bid - 1.0))

    got_s = (sharpe(q * r), sharpe(crossed(q)))
    assert (
        abs(got_s[0] - GATE_SHARPE[0]) < GATE_TOL
        and abs(got_s[1] - GATE_SHARPE[1]) < GATE_TOL
    ), got_s
    tfd = flags["third Friday"].reindex(di).to_numpy(bool)
    me = p56.day_types(cal)["month-end"].reindex(di).to_numpy(bool)
    n = len(r)
    idx = asl.circular_block_bootstrap_idx(
        np.random.default_rng([SEED, n]), n, BLOCK, B
    )
    base = crossed(q)
    rows = []
    for name, qq in (
        ("sign(s)", q),
        ("third Fridays forced short", np.where(tfd, -1.0, q)),
        ("month-ends forced long", np.where(me, 1.0, q)),
        ("both", np.where(tfd, -1.0, np.where(me, 1.0, q))),
    ):
        x = crossed(qq)
        d_s = np.array([sharpe(x[i]) - sharpe(base[i]) for i in idx])
        changed = int((qq != q).sum())
        p = np.nan
        if name == "third Fridays forced short":
            gains = []
            for _ in range(B):
                mm = np.zeros(n, bool)
                mm[rng.choice(n, int(tfd.sum()), replace=False)] = True
                gains.append(sharpe(crossed(np.where(mm, -1.0, q))) - sharpe(base))
            p = float(
                (1 + (np.array(gains) >= sharpe(x) - sharpe(base)).sum()) / (1 + B)
            )
        rows.append(
            {
                "book": name,
                "positions_changed": changed,
                "Sharpe_mid": sharpe(qq * r),
                "Sharpe_crossed": sharpe(x),
                "gain_crossed": sharpe(x) - sharpe(base),
                "ci_lo": float(np.percentile(d_s, 2.5)),
                "ci_hi": float(np.percentile(d_s, 97.5)),
                "placebo_p": p,
            }
        )
    print(
        f"\nF  15:30 sign(s) on the deck days ({int(tfd.sum())} third Fridays, the ridge already "
        f"short on {float((q[tfd] < 0).mean()):.0%} of them)"
    )
    f_tab = pd.DataFrame(rows)
    f_tab.to_csv(OUT / "f_sign_s_overrides.csv", index=False)
    print(f_tab.round(3).to_string(index=False))

    a43 = pd.read_csv(P43_DAILY, index_col=0, parse_dates=True)
    fa = flags.reindex(a43.index).fillna(False)
    rows = []
    for clock in ("11:00", "13:30", "14:30"):
        pts = a43[f"{clock}|hold"] * a43[f"{clock}|entry"]
        for col in ("third Friday", "other Fridays"):
            x = pts[fa[col].to_numpy()].dropna().to_numpy()
            rows.append(
                {
                    "book": f"short straddle sold {clock}, held",
                    "day type": col,
                    "n": len(x),
                    "pts_per_contract": x.mean(),
                    "t": tstat(x),
                }
            )
        x = pts[~fa["third Friday"].to_numpy()].dropna().to_numpy()
        rows.append(
            {
                "book": f"short straddle sold {clock}, held",
                "day type": "every other session",
                "n": len(x),
                "pts_per_contract": x.mean(),
                "t": tstat(x),
            }
        )
    h_tab = pd.DataFrame(rows)
    h_tab.to_csv(OUT / "f_hold_books.csv", index=False)
    print("\n   the short afternoon hold books (index points per contract)")
    print(h_tab.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
