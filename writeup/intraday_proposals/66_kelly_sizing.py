"""66 - Kelly sizing for the insurance book, and what it says about third Fridays.

The live book (13:30 short straddle, delta-hedged every 30 minutes, held to
settlement, month-end override, deleveraging brake) commits a FRACTION f of
capital to its stress scenario: contracts = f x capital / (loss of one contract
in a 5% jump), live.ibkr.sizing.  Live f = 10% (Config.stress_fraction).  Write
r_t for one day's P&L of one full-size contract divided by that day's stress
loss, so that a 5% jump is r = -1; the day's return on capital is f x r_t and
the Kelly fraction is the f that maximises the expected log growth

    g(f) = E[ log(1 + f r) ].

On the month-end override day r is the long 15:30 straddle bought at the ask in
the short's full-size count; on braked days r carries the brake's 0.5 or 0.

The difficulty is the tail: the realized tape (2020-2025, net of 0.5 bp) has no
2008 in it.  Study 50's replay has, but its tails are an upper bound (on the
overlap its sd is about twice the real one) and its mean is not the book's.  So
g is evaluated on:

  D1  realized days only
  D2  realized days + the replay's tail: every replayed session BEFORE the tape
      begins (1998-2019, so no day is counted twice) whose loss, at the size the
      brake allowed, is worse than the worst realized day, entered at its
      historical frequency (weight = realized sessions / pre-2020 replayed
      sessions).  Replay losses are put in stress units at the tape's median
      stress fraction of the index.
  D2- as D2 with the realized mean moved to the lower end (2.5%) of its block-
      bootstrap interval: Kelly is proportional to the mean, and the mean is
      the least certain input
  D3  realized days + the stress day itself (r = -1) once in 10, 25, 50 years
      (sensitivity: how rare must a 5% jump be for the live fraction to sit
      where it does against Kelly)

  A  f*, half-Kelly, the Gaussian mu / sigma^2, and the live 10% as a share of
     f*, on each distribution.
  B  five-year paths at the live fraction, quarter, half and full Kelly (D2):
     realized days in 21-session circular blocks with the D2 tail days injected
     at their frequency, 10,000 paths, seed 0: median and 5th-percentile annual
     growth, probability of a 20% and a 50% drawdown, of ending below the start.
  C  third Fridays.  Log growth adds across days and the day type is known in
     advance, so the Kelly fraction can be chosen per day type: f*_TF from third
     Fridays, f*_other from the other sessions, both with the D2 tail at the
     same daily rate.  The ratio f*_TF / f*_other is the size multiplier that
     keeps third Fridays at the same share of Kelly as the rest of the book.
     Interval: realized days resampled within each group, B = 2000, seed 0.
     Written before running: the live third-Friday multiplier is the LOWER end
     (2.5%) of that interval, rounded down to a tenth, and 1 (no change) if that
     is below 1.  What it costs: a 5% jump on a third Friday then takes the
     multiplier x 10% of capital; reported with B's paths re-run with it.

  D  (added after the operator chose 1.5 for the live book) whole contracts on
     $1M in the runner's order of operations, third-Friday multipliers 1.0,
     1.1, 1.5, 1.8: contracts sold, the realized book, the stress-jump cost on
     a third Friday.  Descriptive.

GATES  study 65's realized series (its own gates: gross +0.919, deck net
       +0.4679) and its live-book return at 10% (9.4863% a year, 2020-2025);
       study 58's 72 third-Friday sessions and its 13:30 hold book on them
       (+2.999 points per contract gross).

Run:  python writeup/intraday_proposals/66_kelly_sizing.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "notebooks", ROOT / "writeup"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import atm_straddle_lib as asl  # noqa: E402
from live.ibkr.config import Config  # noqa: E402
from live.ibkr.sizing import scaled_contracts  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = ROOT / "results" / "atm_straddle_intraday_holdclose" / "proposals" / "66"
TAPE_START = "2020-01-01"
DECK_END, OOS_START = "2024-04-30", "2024-05-01"
STRESS_EVERY_YEARS = (10, 25, 50)
YEARS = 5
N_PATHS = 10_000
BLOCK = 21
B, SEED = 2000, 0
MULT_STEP = 0.1  # the live multiplier is rounded down to this step
WHOLE_CAPITAL = 1_000_000.0  # part D: the account the runner's examples use
WHOLE_MULTIPLIERS = (
    1.0,
    1.1,
    1.5,
    1.8,
)  # none, the rule, the operator's choice, the point ratio
EDGE = 1e-12  # keeps the root search strictly inside 1 + f r > 0
GATE_ANN_LIVE = 0.094863
GATE_TF = {"n": 72, "gross_pts": 2.999404}
GATE_TOL = 1e-5


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def growth(r: np.ndarray, w: np.ndarray, f: float) -> float:
    """Weighted mean log growth per day at fraction f."""
    x = 1.0 + f * r
    if (x <= 0).any():
        return float("-inf")
    return float((np.log(x) * w).sum() / w.sum())


def kelly(r: np.ndarray, w: np.ndarray) -> float:
    """argmax_f E_w[log(1 + f r)]: the root of g'(f) = E_w[r / (1 + f r)] (g is concave)."""
    if not (r < 0).any():
        return float("inf")

    def dg(f: float) -> float:
        return float((w * r / (1.0 + f * r)).sum() / w.sum())

    if dg(0.0) <= 0:
        return 0.0
    hi = (1.0 - EDGE) / float(-r.min())
    return hi if dg(hi) > 0 else float(brentq(dg, 0.0, hi, xtol=1e-10))


def with_tail(
    r: np.ndarray, tail: np.ndarray, rate: float
) -> tuple[np.ndarray, np.ndarray]:
    """Realized days at weight 1 and the tail days at the same daily frequency as history."""
    w_tail = len(r) * rate / len(tail) if len(tail) else 0.0
    return np.concatenate([r, tail]), np.concatenate(
        [np.ones(len(r)), np.full(len(tail), w_tail)]
    )


def paths(
    r: np.ndarray, tail: np.ndarray, rate: float, f: float, seed: int
) -> dict[str, float]:
    """N_PATHS five-year paths: realized days in circular blocks, tail days injected."""
    rng = np.random.default_rng([SEED, seed])
    n_days = YEARS * int(asl.PERIODS_PER_YEAR)
    n_blocks = -(-n_days // BLOCK)
    starts = rng.integers(0, len(r), size=(N_PATHS, n_blocks))
    idx = ((starts[:, :, None] + np.arange(BLOCK)[None, None, :]) % len(r)).reshape(
        N_PATHS, -1
    )[:, :n_days]
    x = r[idx]
    hit = rng.random((N_PATHS, n_days)) < rate
    x = np.where(hit, tail[rng.integers(0, len(tail), size=(N_PATHS, n_days))], x)
    step = 1.0 + f * x
    ruined = (step <= 0).any(axis=1)
    cum = np.cumsum(np.log(np.maximum(step, EDGE)), axis=1)
    peak = np.maximum.accumulate(np.maximum(cum, 0.0), axis=1)
    worst = (np.exp(cum - peak) - 1.0).min(axis=1)
    final = np.where(ruined, 0.0, np.exp(cum[:, -1]))
    return {
        "median_annual_growth": float(np.median(final) ** (1.0 / YEARS) - 1.0),
        "p05_annual_growth": float(np.quantile(final, 0.05) ** (1.0 / YEARS) - 1.0),
        "prob_drawdown_20pct": float((worst <= -0.20).mean()),
        "prob_drawdown_50pct": float((worst <= -0.50).mean()),
        "prob_below_start": float((final < 1.0).mean()),
        "prob_ruin": float(ruined.mean()),
    }


def main() -> None:  # noqa: PLR0915
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    p43 = _load(HERE / "43_causal_entry_over_time.py", "p43_causal_entry")
    p50 = _load(HERE / "50_cat_replay_repriced.py", "p50_cat_replay")
    p56 = _load(HERE / "56_close_calendar_family.py", "p56_close_calendar")
    p58 = _load(HERE / "58_third_friday_short.py", "p58_third_friday")
    p65 = _load(HERE / "65_capital_from_history.py", "p65_capital")
    rep = p65.load_replay(p50)
    real = p65.load_realized(p43)
    real["m"] = p65.realized_brake(real, rep)
    c54 = pd.read_csv(p65.P54_DAILY, index_col=0, parse_dates=True).reindex(real.index)
    me = c54["month_end"].fillna(False).astype(bool).to_numpy()
    stress = real["stress_dollars"].to_numpy()
    pts = np.where(
        me,
        c54["pts_ask"].to_numpy(float),
        real["m"].to_numpy() * real["net_pts"].to_numpy(),
    )
    r = pts * p65.SPX_INDEX_MULTIPLIER / stress
    assert np.isfinite(r).all()
    days = real.index
    ann_live = float((p65.FRACTION * r).mean() * asl.PERIODS_PER_YEAR)
    assert abs(ann_live - GATE_ANN_LIVE) < GATE_TOL, ann_live
    print(
        f"GATE  the live book at f = {p65.FRACTION:.0%}: {ann_live:.4%} a year (study 65)"
    )

    stamp, _ = p43.session_stamps()
    cal = p56.session_calendar(pd.DatetimeIndex(stamp.index))
    tf = (
        pd.Series(p58.nth_weekday_sessions(cal, 4, 3), index=cal)
        .reindex(days)
        .fillna(False)
        .to_numpy(bool)
    )
    got_tf = (int(tf.sum()), float(real.loc[tf, "gross_pts"].mean()))
    assert (
        got_tf[0] == GATE_TF["n"] and abs(got_tf[1] - GATE_TF["gross_pts"]) < GATE_TOL
    ), got_tf
    assert not (tf & me).any()
    print(
        f"GATE  {got_tf[0]} third Fridays on the tape, 13:30 book gross {got_tf[1]:+.4f} points (study 58)"
    )

    # ------------------------------------------------------ the tail -------
    stress_frac = float(
        (real["stress_dollars"] / (real["S"] * p65.SPX_INDEX_MULTIPLIER)).median()
    )
    pre = (rep["session"] < TAPE_START).to_numpy()
    r_rep = -(rep["sized_loss_frac"].to_numpy() / stress_frac)[pre]
    order = np.argsort(r_rep)
    tail = r_rep[order][r_rep[order] < r.min()]
    rate = len(tail) / len(r_rep)
    tail_days = (
        rep.loc[pre, "session"].iloc[order[: len(tail)]].dt.date.astype(str).tolist()
    )
    print(
        f"worst realized day {r.min():+.3f} stress units ({days[int(np.argmin(r))].date()}); pre-{TAPE_START[:4]} "
        f"replay days worse: {len(tail)} of {len(r_rep)} ({rate:.4%} a day), worst {tail.min():+.3f}; "
        f"dates {tail_days}"
    )

    # --------------------------------------------------- A Kelly -----------
    rng = np.random.default_rng(SEED)
    idx = asl.circular_block_bootstrap_idx(rng, len(r), BLOCK, B)
    mu_lo = float(np.percentile(r[idx].mean(axis=1), 2.5))
    dists: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "D1 realized only": (r, np.ones(len(r))),
        "D2 realized + pre-2020 replay tail": with_tail(r, tail, rate),
        "D2- as D2, mean at its 2.5% bound": with_tail(
            r - r.mean() + mu_lo, tail, rate
        ),
    }
    for yrs in STRESS_EVERY_YEARS:
        p = 1.0 / (yrs * asl.PERIODS_PER_YEAR)
        dists[f"D3 realized + a 5% jump once in {yrs} years"] = with_tail(
            r, np.array([-1.0]), p
        )
    rows = []
    for name, (x, w) in dists.items():
        f_star = kelly(x, w)
        mu = float((x * w).sum() / w.sum())
        var = float(((x - mu) ** 2 * w).sum() / w.sum())
        rows.append(
            {
                "distribution": name,
                "mean_r": mu,
                "Kelly_f": f_star,
                "half_Kelly_f": f_star / 2,
                "gaussian_mu_over_var": mu / var,
                "live_share_of_Kelly": p65.FRACTION / f_star,
                "ann_log_growth_at_Kelly": growth(x, w, f_star) * asl.PERIODS_PER_YEAR,
                "ann_log_growth_at_half": growth(x, w, f_star / 2)
                * asl.PERIODS_PER_YEAR,
                "ann_log_growth_at_live": growth(x, w, p65.FRACTION)
                * asl.PERIODS_PER_YEAR,
            }
        )
    k = pd.DataFrame(rows)
    k.to_csv(OUT / "a_kelly.csv", index=False)
    print(
        f"\nA  Kelly fraction of capital committed to the 5% stress day (live {p65.FRACTION:.0%}); r in stress units"
    )
    print(k.round(4).to_string(index=False))
    per = []
    for label, mask in (
        ("deck period", days <= DECK_END),
        ("HOLDOUT", days >= OOS_START),
    ):
        x, w = with_tail(r[mask], tail, rate)
        per.append(
            {
                "sample": label,
                "days": int(mask.sum()),
                "mean_r": float(r[mask].mean()),
                "Kelly_f_D2": kelly(x, w),
            }
        )
    pd.DataFrame(per).to_csv(OUT / "a_kelly_by_period.csv", index=False)
    print(pd.DataFrame(per).round(4).to_string(index=False))

    # ---------------------------------------------------- B paths ----------
    f2 = float(k.loc[k["distribution"].str.startswith("D2 "), "Kelly_f"].iloc[0])
    sizings = [
        ("live 10%", p65.FRACTION),
        ("quarter Kelly (D2)", f2 / 4),
        ("half Kelly (D2)", f2 / 2),
        ("Kelly (D2)", f2),
    ]
    rows_b = [
        {"sizing": s, "f": f} | paths(r, tail, rate, f, i)
        for i, (s, f) in enumerate(sizings)
    ]

    # ------------------------------------------------ C third Fridays ------
    a, b = r[tf], r[~tf]
    f_tf = kelly(*with_tail(a, tail, rate))
    f_ot = kelly(*with_tail(b, tail, rate))
    f_tf1, f_ot1 = kelly(a, np.ones(len(a))), kelly(b, np.ones(len(b)))
    rng2 = np.random.default_rng([SEED, 66])
    draws = np.array(
        [
            (
                kelly(*with_tail(a[rng2.integers(0, len(a), len(a))], tail, rate)),
                kelly(*with_tail(b[rng2.integers(0, len(b), len(b))], tail, rate)),
            )
            for _ in range(B)
        ]
    )
    # a draw in which the other sessions earn nothing (their Kelly is 0) while
    # third Fridays earn something is an infinite ratio: it sits in the upper
    # tail and cannot move the lower end the rule reads; 0 / 0 is dropped
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(
            draws[:, 1] > 0,
            draws[:, 0] / draws[:, 1],
            np.where(draws[:, 0] > 0, np.inf, np.nan),
        )
    lo = float(np.nanpercentile(ratios, 2.5))
    hi = float(np.nanpercentile(ratios, 97.5, method="higher"))  # may be unbounded
    print(
        f"   bootstrap: other-session Kelly is 0 in {(draws[:, 1] == 0).mean():.1%} of draws, "
        f"third-Friday Kelly 0 in {(draws[:, 0] == 0).mean():.1%}; third-Friday Kelly interval "
        f"[{np.percentile(draws[:, 0], 2.5):.3f}, {np.percentile(draws[:, 0], 97.5):.3f}], other "
        f"[{np.percentile(draws[:, 1], 2.5):.3f}, {np.percentile(draws[:, 1], 97.5):.3f}]"
    )
    mult = max(1.0, float(np.floor(lo / MULT_STEP) * MULT_STEP))
    ct = pd.DataFrame(
        [
            {
                "group": g,
                "days": len(x),
                "mean_r": float(x.mean()),
                "sd_r": float(x.std(ddof=1)),
                "worst_r": float(x.min()),
                "hit": float((x > 0).mean()),
                "Kelly_f_D1": k1,
                "Kelly_f_D2": k2,
            }
            for g, x, k1, k2 in (
                ("third Fridays", a, f_tf1, f_tf),
                ("other sessions", b, f_ot1, f_ot),
            )
        ]
    )
    ct.to_csv(OUT / "c_third_friday.csv", index=False)
    # descriptive, not the rule: the point ratio, rounded down the same way
    point = float(np.floor(f_tf / f_ot / MULT_STEP) * MULT_STEP)
    r_mult = np.where(tf, mult * r, r)
    r_point = np.where(tf, point * r, r)
    for i, (label, rr) in enumerate(
        (
            (f"live 10%, third Fridays x{mult:.1f} (the rule)", r_mult),
            (f"live 10%, third Fridays x{point:.1f} (point ratio)", r_point),
        )
    ):
        rows_b.append(
            {"sizing": label, "f": p65.FRACTION}
            | paths(rr, tail, rate, p65.FRACTION, len(sizings) + i)
        )
    bt = pd.DataFrame(rows_b)
    bt.to_csv(OUT / "b_paths.csv", index=False)
    print(
        f"\nB  {N_PATHS:,} five-year paths: realized days in {BLOCK}-session blocks, pre-2020 replay tail days "
        f"injected at {rate:.4%} a day"
    )
    print(bt.round(4).to_string(index=False))

    print(
        "\nC  third Fridays against other sessions, r in stress units per full-size contract"
    )
    print(ct.round(4).to_string(index=False))
    print(
        f"   Kelly ratio third Friday / other, D2: {f_tf / f_ot:.2f}, interval [{lo:.2f}, {hi:.2f}]; "
        f"D1: {f_tf1 / f_ot1:.2f}"
    )
    print(
        f"   rule written before running: multiplier = max(1, lower end rounded down to {MULT_STEP}) = "
        f"{mult:.1f}; a 5% jump on a third Friday then costs {mult * p65.FRACTION:.0%} of capital"
    )
    hold = days >= OOS_START
    lv = pd.DataFrame(
        {
            f"{label}, {per}": p65.evaluate(p65.FRACTION * rr[mask], days[mask])
            for per, mask in (("all", np.ones(len(days), bool)), ("HOLDOUT", hold))
            for label, rr in (
                ("live book", r),
                (f"third Fridays x{mult:.1f}", r_mult),
                (f"third Fridays x{point:.1f}", r_point),
            )
        }
    ).T
    lv.to_csv(OUT / "c_live_with_multiplier.csv")
    print(
        "\n   the realized live book, with and without the multiplier (fractions of capital)"
    )
    print(lv.to_string())
    pd.DataFrame(
        [
            {
                "multiplier": mult,
                "point_multiplier_descriptive": point,
                "ratio_D2": f_tf / f_ot,
                "ratio_lo": lo,
                "ratio_hi": hi,
                "ratio_D1": f_tf1 / f_ot1,
            }
        ]
    ).to_csv(OUT / "c_multiplier.csv", index=False)

    # ------------------------------------- D whole contracts at the live size --
    # Added 2026-09-21 after the operator set the live multiplier to 1.5 (between
    # the rule's 1.1 and the point ratio): A-C use fractional contracts; the
    # runner sells whole ones.  The live order of operations on WHOLE_CAPITAL:
    # the stress table floored and capped, the brake floored, the third-Friday
    # multiplier floored (decimal) and capped; the month-end long in the table's
    # count.  Descriptive: the day type was found on these sessions.
    n_table = np.minimum(
        np.floor(p65.FRACTION * WHOLE_CAPITAL / stress), Config.max_straddles
    )
    n_brake = np.floor(n_table * real["m"].to_numpy())
    rows_d = []
    for t in WHOLE_MULTIPLIERS:
        n_tf = np.array(
            [min(scaled_contracts(int(v), t), Config.max_straddles) for v in n_brake]
        )
        n_short = np.where(tf, n_tf, n_brake)
        dollars = p65.SPX_INDEX_MULTIPLIER * np.where(
            me,
            n_table * c54["pts_ask"].to_numpy(float),
            n_short * real["net_pts"].to_numpy(),
        )
        x = dollars / WHOLE_CAPITAL
        at_risk = n_short * stress / WHOLE_CAPITAL
        for pname, mask in (("all", np.ones(len(days), bool)), ("HOLDOUT", hold)):
            rows_d.append(
                {
                    "third_friday_multiplier": t,
                    "sample": pname,
                    "third_friday_contracts_mean": float(n_short[tf & mask].mean()),
                    "other_day_contracts_mean": float(n_short[~tf & ~me & mask].mean()),
                    "third_fridays_with_more_contracts": int(
                        (n_tf[tf & mask] > n_brake[tf & mask]).sum()
                    ),
                    "stress_jump_cost_max_third_friday": float(
                        at_risk[tf & mask].max()
                    ),
                    **p65.evaluate(x[mask], days[mask]),
                }
            )
    dt = pd.DataFrame(rows_d)
    dt.to_csv(OUT / "d_whole_contracts.csv", index=False)
    print(
        f"\nD  WHOLE contracts on ${WHOLE_CAPITAL:,.0f} in the runner's order of operations (cap "
        f"{Config.max_straddles}); fractions of capital"
    )
    print(dt.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
