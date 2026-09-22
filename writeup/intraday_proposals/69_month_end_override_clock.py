"""69 - the month-end override's entry clock: buy the straddle at 14:00 instead of 15:30?

The live book's month-end override (proposals 54/55; live.ibkr.run_day) sells
nothing on the last session of the month and instead BUYS the 15:30 nearest-OTM
straddle at the ask, in the short program's full-size count, held to
settlement (+2.93 points per contract on 70 month-ends, t 2.1).  A scan of
seven entry clocks on those same 70 month-ends (2026-09-21, not a study) showed
the long earning more points when bought earlier -- +6.2 at 14:00 (t 2.9), +5.3
at 13:30 -- at about the same return per dollar of premium (0.42 vs 0.43).

DISCLOSURE.  14:00 was picked by looking at the seven clocks on the very data
this study scores; there are no unseen month-ends.  The test therefore charges
the choice to its family and asks what would break it if it were an accident:

  the family   the clocks the runner could buy at instead of 15:30, given that
               it sizes the long off the 13:30 straddle: 13:30, 14:00, 14:30,
               15:00 (FAMILY = 4); every criterion on the candidate is charged
               x4 (Bonferroni)
  the object   per contract, the long straddle bought at the ask at clock c and
               held to the 16:00 settlement, MINUS the same long bought at
               15:30: the paired difference d_c on each month-end (the book buys
               a fixed count, so points per contract is what the book earns;
               premium units are reported beside them)

  A  the family on the 70 month-ends: mean d_c, paired t, one-sided p x4, hit;
     deck (52) and holdout (18) separately; premium units beside.
  B  the placebo: d_c on every non-month-end session (buying earlier COSTS on an
     ordinary day), and the share of 2,000 random 70-session draws (seed 0)
     whose mean d_c is at least the month-ends' -- the p-value of "any 70 days".
  C  where the extra points come from: the c-entry long marked at 15:30 (the
     held legs' 15:30 midpoint, proposal 52's quotes) against 15:30 -> settle;
     the underlying's |c -> close| move against the c-implied straddle on
     month-ends and on other days; MSCI-review month-ends against the rest.
  D  the book: study 65's live book (13:30 short net of 0.5 bp, brake, 10% of
     capital per stress unit, $1M) with the override bought at c instead of
     15:30, every non-month-end day identical: return, Sharpe, max drawdown,
     worst day, on all 1,279 sessions and on the holdout.

Written before running.  The 14:00 override is ADOPTED for the runner if
  (1) mean d_14:00 > 0 on the 70 month-ends with one-sided paired-t p x4 < 0.05;
  (2) mean d_14:00 > 0 in the deck AND in the holdout;
  (3) the random-70-session placebo p < 0.05;
  (4) guard: the book with the 14:00 override has a Sharpe no lower than with
      15:30 and a worst day no worse than 15:30's by more than one point of
      capital (the long at 14:00 carries about twice the premium).
Otherwise the override stays at 15:30 and 14:00 is recorded as a lead.

GATES  study 54's month-end row (+2.931 points at the ask on 70 month-ends,
       52 deck / 18 holdout); study 65's live-book return (9.4863%/yr at 10%).

Run:  python writeup/intraday_proposals/69_month_end_override_clock.py
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
OUT = ROOT / "results" / "atm_straddle_intraday_holdclose" / "proposals" / "69"
CANDIDATE = "14:00"
FAMILY = ("13:30", "14:00", "14:30", "15:00")
REFERENCE = "15:30"
DECK_END, OOS_START = "2024-04-30", "2024-05-01"
B, SEED = 2000, 0
GUARD_WORST_PTS_OF_CAPITAL = 0.01
GATE_ME = {"n": 70, "deck": 52, "holdout": 18, "pts": 2.931}
GATE_ANN_LIVE = 0.094863
GATE_TOL = 5e-4


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def paired(d: np.ndarray) -> dict[str, float]:
    d = d[np.isfinite(d)]
    t = float(d.mean() / d.std(ddof=1) * np.sqrt(len(d)))
    return {
        "n": len(d),
        "mean": float(d.mean()),
        "t": t,
        "p_one_sided": float(stats.t.sf(t, df=len(d) - 1)),
        "hit": float((d > 0).mean()),
    }


def main() -> None:  # noqa: PLR0915
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    p43 = _load(HERE / "43_causal_entry_over_time.py", "p43_causal_entry")
    p50 = _load(HERE / "50_cat_replay_repriced.py", "p50_cat_replay")
    p52 = _load(HERE / "52_flatten_1500_vrp.py", "p52_flatten")
    p56 = _load(HERE / "56_close_calendar_family.py", "p56_close_calendar")
    p65 = _load(HERE / "65_capital_from_history.py", "p65_capital")
    stamp, _ = p43.session_stamps()
    sessions = pd.DatetimeIndex(stamp.index).difference(p43.half_sessions(stamp))
    ch = p43.build_chain(stamp, sessions)
    idx = pd.DatetimeIndex(ch["dates"])
    sc = ch["S_close"]
    cal = p56.session_calendar(pd.DatetimeIndex(stamp.index))
    types = p56.day_types(cal)
    me = (
        pd.Series(types["month-end"].to_numpy(), index=cal)
        .reindex(idx)
        .fillna(False)
        .to_numpy(bool)
    )
    msci = (
        pd.Series(types["MSCI review close"].to_numpy(), index=cal)
        .reindex(idx)
        .fillna(False)
        .to_numpy(bool)
    )
    deck = np.asarray(idx <= DECK_END)

    clocks = (*FAMILY, REFERENCE)
    long_pts, long_units, ask_at, tot_at = {}, {}, {}, {}
    for c in clocks:
        j = p43.CLOCKS.index(c)
        mid, bid = ch["entry"][:, j], ch["bid"][:, j]
        ask = 2.0 * mid - bid
        pay = np.maximum(sc - ch["K_c"][:, j], 0.0) + np.maximum(
            ch["K_p"][:, j] - sc, 0.0
        )
        ok = np.isfinite(ask) & (bid > 0) & np.isfinite(pay)
        long_pts[c] = np.where(ok, pay - ask, np.nan)
        long_units[c] = np.where(ok, (pay - ask) / ask, np.nan)
        ask_at[c] = np.where(ok, ask, np.nan)
        tot_at[c] = ch["tot"][:, j]
    ref = long_pts[REFERENCE]
    got = (
        int(me.sum()),
        int((me & deck).sum()),
        int((me & ~deck).sum()),
        float(np.nanmean(ref[me])),
    )
    assert (
        got[:3] == (GATE_ME["n"], GATE_ME["deck"], GATE_ME["holdout"])
        and abs(got[3] - GATE_ME["pts"]) < GATE_TOL
    ), got
    print(
        f"GATE  study 54's month-end row: {got[0]} month-ends ({got[1]} deck / {got[2]} holdout), long at the ask "
        f"at 15:30 {got[3]:+.3f} points per contract"
    )

    # ------------------------------------------------------------- A --------
    rows = []
    for c in FAMILY:
        d = long_pts[c] - ref
        du = long_units[c] - long_units[REFERENCE]
        for sample, m in (("all 70", me), ("deck", me & deck), ("holdout", me & ~deck)):
            r = paired(d[m])
            rows.append(
                {
                    "clock": c,
                    "sample": sample,
                    **r,
                    "p_x4": min(1.0, r["p_one_sided"] * len(FAMILY)),
                    "mean_units_diff": float(np.nanmean(du[m])),
                    "long_pts_at_c": float(np.nanmean(long_pts[c][m])),
                    "long_pts_at_1530": float(np.nanmean(ref[m])),
                    "premium_at_c": float(np.nanmedian(ask_at[c][m])),
                    "premium_at_1530": float(np.nanmedian(ask_at[REFERENCE][m])),
                }
            )
    a = pd.DataFrame(rows)
    a.to_csv(OUT / "a_family.csv", index=False)
    print(
        "\nA  long straddle bought at c minus bought at 15:30, month-ends, points per contract (paired)"
    )
    print(a.round(3).to_string(index=False))

    # ------------------------------------------------------------- B --------
    rng = np.random.default_rng(SEED)
    rows_b = []
    other = ~me & np.isfinite(long_pts[CANDIDATE]) & np.isfinite(ref)
    pool = np.flatnonzero(np.isfinite(long_pts[CANDIDATE]) & np.isfinite(ref))
    n_me = int(me.sum())
    for c in FAMILY:
        d = long_pts[c] - ref
        obs = float(np.nanmean(d[me]))
        draws = np.array(
            [np.nanmean(d[rng.choice(pool, n_me, replace=False)]) for _ in range(B)]
        )
        rows_b.append(
            {
                "clock": c,
                "other_days_mean_diff": float(np.nanmean(d[other])),
                "other_days_t": paired(d[other])["t"],
                "month_end_mean_diff": obs,
                "placebo_mean": float(draws.mean()),
                "placebo_p": float((1 + (draws >= obs).sum()) / (1 + B)),
            }
        )
    b = pd.DataFrame(rows_b)
    b.to_csv(OUT / "b_placebo.csv", index=False)
    print(
        f"\nB  the same difference on ordinary days, and the share of {B:,} random {n_me}-session draws at or above the month-ends'"
    )
    print(b.round(4).to_string(index=False))

    # ------------------------------------------------------------- C --------
    # proposal 52 quotes the held legs of ITS entry clocks at 15:30; point it at
    # this study's family (the function reads the module constant at call time)
    p52.ENTRY_CLOCKS = tuple(FAMILY)
    quotes = p52.held_leg_quotes(
        stamp, ch
    )  # 15:30 quotes of legs entered at each clock
    rows_c = []
    for c in FAMILY:
        j = p43.CLOCKS.index(c)
        s_c = ch["S"][:, j]
        mark = quotes[REFERENCE][f"{c}_mid"]  # the c-entry straddle's midpoint at 15:30
        to_1530 = mark - ask_at[c]
        after = long_pts[c] - to_1530
        move = np.abs(np.log(sc / s_c))
        for lab, m in (
            ("month-end", me),
            ("MSCI-review month-end", me & msci),
            ("other month-end", me & ~msci),
            ("other days", ~me),
        ):
            rows_c.append(
                {
                    "clock": c,
                    "days": lab,
                    "n": int(m.sum()),
                    "long_pts": float(np.nanmean(long_pts[c][m])),
                    "c_to_1530_pts": float(np.nanmean(to_1530[m])),
                    "1530_to_settle_pts": float(np.nanmean(after[m])),
                    "abs_move_c_to_close_bp": float(np.nanmean(move[m]) * 1e4),
                    "implied_c_to_close_bp": float(np.nanmean(tot_at[c][m]) * 1e4),
                    "realized_over_implied": float(
                        np.sqrt(np.nanmean(move[m] ** 2)) / np.nanmean(tot_at[c][m])
                    ),
                }
            )
    cc = pd.DataFrame(rows_c)
    cc.to_csv(OUT / "c_anatomy.csv", index=False)
    print(
        "\nC  where the points come from: the c-entry long marked at 15:30, then 15:30 -> settle; the move against the implied"
    )
    print(cc.round(3).to_string(index=False))

    # ------------------------------------------------------------- D --------
    rep = p65.load_replay(p50)
    real = p65.load_realized(p43)
    real["m"] = p65.realized_brake(real, rep)
    pos = idx.get_indexer(real.index)
    assert (pos >= 0).all()
    stress = real["stress_dollars"].to_numpy()
    me_r = me[pos]
    plain = real["m"].to_numpy() * real["net_pts"].to_numpy()
    rows_d = []
    hold = np.asarray(real.index >= OOS_START)
    books = {}
    for c in clocks:
        pts = np.where(me_r, long_pts[c][pos], plain)
        r = p65.FRACTION * pts * p65.SPX_INDEX_MULTIPLIER / stress
        books[c] = r
        if c == REFERENCE:
            ann = float(r.mean() * asl.PERIODS_PER_YEAR)
            assert abs(ann - GATE_ANN_LIVE) < 1e-5, ann
            print(
                f"\nGATE  the live book with the 15:30 override: {ann:.4%} a year (study 65)"
            )
        for sample, m in (("all", np.ones(len(r), bool)), ("HOLDOUT", hold)):
            rows_d.append(
                {
                    "override_clock": c,
                    "sample": sample,
                    **p65.evaluate(r[m], real.index[m]),
                    "month_end_days": int(me_r[m].sum()),
                    "month_end_pts_per_contract": float(
                        np.nanmean(long_pts[c][pos][me_r & m])
                    ),
                }
            )
    d = pd.DataFrame(rows_d)
    d.to_csv(OUT / "d_book_by_override_clock.csv", index=False)
    print(
        "D  the live book with the override bought at each clock (fractions of capital)"
    )
    print(d.round(4).to_string(index=False))
    # paired block-bootstrap of the Sharpe difference, candidate vs 15:30 (reported, not a criterion)
    x, y = books[CANDIDATE], books[REFERENCE]
    bi = asl.circular_block_bootstrap_idx(
        np.random.default_rng([SEED, 69]), len(x), 21, B
    )
    ds = (
        x[bi].mean(1) / x[bi].std(1, ddof=1) - y[bi].mean(1) / y[bi].std(1, ddof=1)
    ) * np.sqrt(asl.PERIODS_PER_YEAR)
    print(
        f"   Sharpe difference {CANDIDATE} minus 15:30 override: {float(np.percentile(ds, 2.5)):+.3f} .. {float(np.percentile(ds, 97.5)):+.3f} (95%)"
    )

    # ------------------------------------------------------------ verdict ---
    ra = a[(a["clock"] == CANDIDATE)].set_index("sample")
    c1 = bool(ra.loc["all 70", "mean"] > 0 and ra.loc["all 70", "p_x4"] < 0.05)
    c2 = bool(ra.loc["deck", "mean"] > 0 and ra.loc["holdout", "mean"] > 0)
    c3 = bool(b.set_index("clock").loc[CANDIDATE, "placebo_p"] < 0.05)
    da = d.set_index(["override_clock", "sample"])
    c4 = bool(
        da.loc[(CANDIDATE, "all"), "Sharpe"] >= da.loc[(REFERENCE, "all"), "Sharpe"]
        and da.loc[(CANDIDATE, "all"), "worst_day"]
        >= da.loc[(REFERENCE, "all"), "worst_day"] - GUARD_WORST_PTS_OF_CAPITAL
    )
    print(
        f"\nVERDICT  (1) mean d > 0 with p x4 < 0.05: {c1} (mean {ra.loc['all 70', 'mean']:+.2f}, p x4 {ra.loc['all 70', 'p_x4']:.4f}); "
        f"(2) deck and holdout both > 0: {c2} ({ra.loc['deck', 'mean']:+.2f} / {ra.loc['holdout', 'mean']:+.2f}); "
        f"(3) placebo p < 0.05: {c3} ({b.set_index('clock').loc[CANDIDATE, 'placebo_p']:.4f}); "
        f"(4) book guard: {c4} -> the 14:00 override is {'ADOPTED' if c1 and c2 and c3 and c4 else 'NOT ADOPTED (a lead)'}"
    )


if __name__ == "__main__":
    main()
