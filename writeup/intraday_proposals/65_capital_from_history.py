"""65 - how much capital does a contract of the insurance book need?  History against the 5% jump.

The live runner sizes the 13:30 insurance book by a stress table: each short
straddle is charged the loss of a 5% index jump into the settlement, option
leg and futures leg both taken on their adverse side
(live.ibkr.sizing.stress_loss_per_contract), and contracts = capital x fraction
/ that loss (fraction 10%, live.ibkr.config.Config.stress_fraction).  Studies
49 and 50 replayed the book over every session since 1998 and found that loss
5-10 times beyond the historical 1-in-1000 day.  The deleveraging brake (half
size above the expanding 90th percentile of trailing-21 realized variance, flat
above the 97.5th) now keeps the book small in exactly the regimes where the
worst days happened.  This study asks what capital a contract needs when it is
sized to HISTORY at the size the brake allows, and what that does to return on
capital and to drawdowns.

Two sources, both 13:30 entry, hold to settlement, delta-hedged every 30
minutes, per contract:
  replay    study 50's repriced replay, 6,593 sessions 1998-2024 (median-premium
            book).  Its TAILS are an upper bound (on the overlap with the real
            tape its sd is about twice the real one) and its MEAN is not the
            book's (it is negative); it is used for tails only.
  realized  the chain tape, 1,279 sessions 2020-2025 (proposal 43), net of a
            0.5 bp futures cost (study 64); the 413 sessions from 2024-05-01 are
            the holdout.
Losses are carried as a fraction of the index so that a day in 1998 and a day in
2025 are on one scale; dollars are that fraction x the index x 100.

The brake: study 50's definition on the panel's realized variance for every
session the replay covers; the live ledger's own multiplier
(live.ibkr.premium_ledger) for the sessions after it.

Capital per contract, three rules (each day, dollars at that day's index):
  R_stress  the live stress table
  R_worst   the worst loss per contract, at the size the brake allowed, over all
            replayed sessions STRICTLY BEFORE the day (m_s x loss_s, so a
            half-size day counts at half)
  R_q       the same object's 1-in-1000 quantile
Contracts = capital x fraction x m_t / capital-per-contract (fractional
contracts; whole contracts at $1M are reported beside them).

  A  the tails by brake state, replay and realized.
  B  the three rules on the realized sessions: contracts, return on capital,
     volatility, Sharpe, maximum drawdown, worst day, worst 20-session run, and
     what the 5% stress jump would cost at that size.
  C  the same rules walked through the replay (tails only).

Written before running: R_worst is SUPPORTED as the sizing rule if on the
realized sessions (i) no day loses more than the fraction (10%) of capital and
(ii) the worst 20-session run is no worse than the replay's worst 20-session run
under the same rule.  R_q is reported as the aggressive bound.

GATES  study 50's replay rows (no rule mean -0.5418, 1-in-200 -61.75, worst
       -359.79 points; brake 6,134 / 318 / 141 sessions full / half / zero, mean
       -0.1884); the realized 13:30 book's gross mean over all sessions +0.919 and
       its deck-period net-of-0.5 bp mean +0.4679 (studies 60, 64).

Run:  python writeup/intraday_proposals/65_capital_from_history.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "notebooks", ROOT / "writeup"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import atm_straddle_lib as asl  # noqa: E402
from live.ibkr.config import Config  # noqa: E402
from live.ibkr.premium_ledger import PremiumLedger  # noqa: E402
from live.ibkr.sizing import SPX_INDEX_MULTIPLIER, stress_loss_per_contract  # noqa: E402

HERE = Path(__file__).resolve().parent
HOLD = ROOT / "results" / "atm_straddle_intraday_holdclose"
OUT = HOLD / "proposals" / "65"
REPLAY = HOLD / "proposals" / "50" / "a_distribution.csv.gz"
SEED_LEDGER = ROOT / "results" / "live_seed" / "premium_ledger.parquet"
P54_DAILY = (
    ROOT / "results" / "atm_straddle_0dte_1530" / "proposals" / "54" / "c_daily.csv"
)
ANN = float(np.sqrt(asl.PERIODS_PER_YEAR))
ENTRY = "13:30"
FRACTION = Config.stress_fraction  # the live runner's default fraction of capital
CAPITAL = 1_000_000.0  # dollars; every result is also stated as a fraction of it
TAIL_Q = 0.999  # the 1-in-1000 session
MIN_HIST_Q = int(
    round(1.0 / (1.0 - TAIL_Q))
)  # one expected exceedance before the quantile exists
MIN_HIST_WORST = (
    252  # the brake's own warm-up (study 50's DELEVER_MIN_OBS, asserted in main)
)
RUN = 20  # sessions in the "worst run" statistic
DECK_END, OOS_START = "2024-04-30", "2024-05-01"
MEDIAN_ROLE = "median premium"
PRIMARY = "repriced premium + repriced implied"
HOLD_TERMINAL = "hold to cash settlement"
GATE_50 = {
    "mean": -0.5417503,
    "p05": -61.7499666,
    "min": -359.7918183,
    "full": 6134,
    "half": 318,
    "zero": 141,
    "rule_mean": -0.1884265,
}
GATE_REAL = {"gross_all": 0.919, "net_deck": 0.467875}
GATE_TOL = 5e-4


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def rolling_worst(x: np.ndarray, window: int) -> float:
    c = np.concatenate([[0.0], np.cumsum(np.asarray(x, float))])
    return float((c[window:] - c[:-window]).min()) if len(x) >= window else float("nan")


def maxdd(x: np.ndarray) -> float:
    path = np.cumsum(x)
    return float((path - np.maximum(np.maximum.accumulate(path), 0.0)).min())


def brake(rv: np.ndarray, p50: ModuleType) -> np.ndarray:
    """Study 50's multiplier, reproduced from its own helper and constants."""
    q_half = p50.expanding_pct(rv, p50.DELEVER_HALF_PCT, p50.DELEVER_MIN_OBS)
    q_zero = p50.expanding_pct(rv, p50.DELEVER_ZERO_PCT, p50.DELEVER_MIN_OBS)
    half = np.where(np.isfinite(q_half) & (rv > q_half), 0.5, 1.0)
    return np.where(np.isfinite(q_zero) & (rv > q_zero), 0.0, half)


def load_replay(p50: ModuleType) -> pd.DataFrame:
    cols = [
        "day_role",
        "entry_clock",
        "terminal",
        "convention",
        "session",
        "rv21_trail",
        "premium_pts",
        "premium_over_spot",
        "pnl_pts",
    ]
    d = pd.read_csv(REPLAY, usecols=cols)
    d = d[
        (d["day_role"] == MEDIAN_ROLE)
        & (d["entry_clock"] == ENTRY)
        & (d["terminal"] == HOLD_TERMINAL)
        & (d["convention"] == PRIMARY)
    ]
    d = d.sort_values("session").reset_index(drop=True)
    d["session"] = pd.to_datetime(d["session"])
    v = d["pnl_pts"].to_numpy(float)
    m = brake(d["rv21_trail"].to_numpy(float), p50)
    got = {
        "mean": v.mean(),
        "p05": np.percentile(v, 0.5),
        "min": v.min(),
        "full": int((m == 1).sum()),
        "half": int((m == 0.5).sum()),
        "zero": int((m == 0).sum()),
        "rule_mean": float((m * v).mean()),
    }
    for key, ref in GATE_50.items():
        assert abs(float(got[key]) - ref) < GATE_TOL, (key, got[key], ref)
    print(
        f"GATE  study 50's replay reproduced: {len(d)} sessions, brake full/half/zero "
        f"{got['full']}/{got['half']}/{got['zero']}"
    )
    spot = d["premium_pts"] / d["premium_over_spot"]
    d["m"] = m
    d["loss_frac"] = -d["pnl_pts"] / spot  # a loss is positive
    d["sized_loss_frac"] = d["m"] * d["loss_frac"]
    return d


def load_realized(p43: ModuleType) -> pd.DataFrame:
    stamp, _ = p43.session_stamps()
    sessions = pd.DatetimeIndex(stamp.index).difference(p43.half_sessions(stamp))
    ch = p43.build_chain(stamp, sessions)
    arrays = {
        k: ch[k] for k in ("S", "K_c", "K_p", "entry", "bid", "tot", "ask_c", "ask_p")
    }
    arrays["S_close"] = ch["S_close"]
    j = p43.CLOCKS.index(ENTRY)
    bk = p43._clock_book((j, arrays))
    entry = bk["entry"]
    s, kc, kp, v = ch["S"][:, j], ch["K_c"][:, j], ch["K_p"][:, j], ch["tot"][:, j]
    delta = p43.pkg_delta_vec(v, s, kc, kp)
    stress = np.array(
        [
            float(
                stress_loss_per_contract(s[i], kc[i], kp[i], entry[i], delta[i])[
                    "total"
                ]
            )
            if np.isfinite(entry[i])
            else np.nan
            for i in range(len(s))
        ]
    )
    df = pd.DataFrame(
        {
            "S": s,
            "gross_pts": bk["hold"] * entry,
            "net_pts": bk["hold"] * entry - bk["cost_hold_pts"],
            "stress_dollars": stress,
        },
        index=ch["dates"],
    ).dropna()
    got_g = float(df["gross_pts"].mean())
    got_n = float(df.loc[df.index <= DECK_END, "net_pts"].mean())
    assert abs(got_g - GATE_REAL["gross_all"]) < GATE_TOL, got_g
    assert abs(got_n - GATE_REAL["net_deck"]) < GATE_TOL, got_n
    print(
        f"GATE  realized 13:30 book: gross mean {got_g:+.4f}, deck net-of-0.5 bp {got_n:+.4f}; "
        f"stress per contract median ${df['stress_dollars'].median():,.0f}"
    )
    return df


def realized_brake(real: pd.DataFrame, rep: pd.DataFrame) -> np.ndarray:
    """Study 50's multiplier where the replay covers the day, the live ledger's after it."""
    by_day = pd.Series(rep["m"].to_numpy(), index=pd.DatetimeIndex(rep["session"]))
    m = by_day.reindex(real.index).to_numpy(float)
    ledger = PremiumLedger.load(SEED_LEDGER)
    after = ~np.isfinite(m)
    for i in np.flatnonzero(after):
        m[i] = ledger.delever_multiplier(real.index[i].date())[0]
    print(
        f"brake on the realized sessions: {int((~after).sum())} from the panel (study 50), "
        f"{int(after.sum())} from the live ledger; full/half/zero "
        f"{int((m == 1).sum())}/{int((m == 0.5).sum())}/{int((m == 0).sum())}"
    )
    return m


def causal_hist(
    rep: pd.DataFrame, days: pd.DatetimeIndex
) -> tuple[np.ndarray, np.ndarray]:
    """Worst and 1-in-1000 sized loss (fraction of the index) over replay sessions before each day.

    Neither exists until enough history does: the worst loss needs the brake's
    own warm-up (MIN_HIST_WORST sessions), the 1-in-1000 quantile needs at least
    MIN_HIST_Q sessions (one expected exceedance); before that the rule is NaN
    and the day is not scored.
    """
    sess = rep["session"].to_numpy()
    x = rep["sized_loss_frac"].to_numpy(float)
    worst = np.maximum.accumulate(x)
    n_before = np.searchsorted(sess, days.to_numpy(), side="left")
    w = np.where(n_before >= MIN_HIST_WORST, worst[np.maximum(n_before - 1, 0)], np.nan)
    q = np.full(len(days), np.nan)
    for i, n in enumerate(n_before):
        if n >= MIN_HIST_Q:
            q[i] = float(np.quantile(x[:n], TAIL_Q))
    return w, q


def evaluate(pnl_frac: np.ndarray, days: pd.DatetimeIndex) -> dict[str, float | str]:
    """Daily P&L as a fraction of capital -> the summary row."""
    return {
        "days": len(pnl_frac),
        "ann_return": float(pnl_frac.mean() * asl.PERIODS_PER_YEAR),
        "ann_vol": float(pnl_frac.std(ddof=1) * ANN),
        "Sharpe": float(pnl_frac.mean() / pnl_frac.std(ddof=1) * ANN),
        "max_drawdown": maxdd(pnl_frac),
        "worst_day": float(pnl_frac.min()),
        "worst_20": rolling_worst(pnl_frac, RUN),
        "worst_day_date": str(days[int(np.argmin(pnl_frac))].date()),
    }


def main() -> None:  # noqa: PLR0915
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    p43 = _load(HERE / "43_causal_entry_over_time.py", "p43_causal_entry")
    p50 = _load(HERE / "50_cat_replay_repriced.py", "p50_cat_replay")
    rep = load_replay(p50)
    real = load_realized(p43)
    real["m"] = realized_brake(real, rep)
    assert MIN_HIST_WORST == p50.DELEVER_MIN_OBS
    c54 = pd.read_csv(P54_DAILY, index_col=0, parse_dates=True).reindex(real.index)
    real["month_end"] = c54["month_end"].fillna(False).astype(bool)
    real["me_long_pts"] = c54["pts_ask"].to_numpy(float)

    # ------------------------------------------------------------- A tails --
    rows = []
    for name, loss, m, days in (
        (
            "replay 1998-2024",
            rep["loss_frac"].to_numpy(),
            rep["m"].to_numpy(),
            rep["session"],
        ),
        (
            "realized 2020-2025 (net)",
            -real["net_pts"].to_numpy() / real["S"].to_numpy(),
            real["m"].to_numpy(),
            pd.Series(real.index),
        ),
    ):
        for state, mask in (
            ("all days", np.ones(len(m), bool)),
            ("brake full", m == 1),
            ("brake half", m == 0.5),
            ("brake zero", m == 0),
        ):
            x = loss[mask]
            if len(x) == 0:
                continue
            rows.append(
                {
                    "source": name,
                    "state": state,
                    "sessions": len(x),
                    "p99_loss_pct_index": float(np.quantile(x, 0.99) * 100),
                    "p99.9_loss_pct_index": float(np.quantile(x, TAIL_Q) * 100),
                    "worst_loss_pct_index": float(x.max() * 100),
                    "worst_date": str(
                        pd.DatetimeIndex(days)[mask][int(np.argmax(x))].date()
                    ),
                }
            )
    a = pd.DataFrame(rows)
    a.to_csv(OUT / "a_tails_by_brake.csv", index=False)
    print(
        "\nA  loss per contract as % of the index level, by brake state (a loss is positive)"
    )
    print(a.round(4).to_string(index=False))
    stress_frac = real["stress_dollars"] / (real["S"] * SPX_INDEX_MULTIPLIER)
    print(
        f"   the live 5% stress per contract is {stress_frac.median() * 100:.2f}% of the index "
        f"at the median (range {stress_frac.min() * 100:.2f}-{stress_frac.max() * 100:.2f}%)"
    )

    # ------------------------------------------------ B realized, by rule --
    w_frac, q_frac = causal_hist(rep, real.index)
    dollars_per_frac = real["S"].to_numpy() * SPX_INDEX_MULTIPLIER
    cap_per = {
        "R_stress (live)": real["stress_dollars"].to_numpy(),
        "R_worst (history, at the brake's size)": w_frac * dollars_per_frac,
        "R_q (history, 1-in-1000)": q_frac * dollars_per_frac,
    }
    m = real["m"].to_numpy()
    me_np = real["month_end"].to_numpy()
    rows_b = []
    for rule, per_contract in cap_per.items():
        n_full = (
            FRACTION * CAPITAL / per_contract
        )  # full-size contracts the rule allows
        n = n_full * m
        plain = n * real["net_pts"].to_numpy()
        # the live override: no short on a month-end, the short's full-size count
        # of 15:30 straddles bought at the ask instead (no brake on the long)
        live = np.where(me_np, n_full * real["me_long_pts"].to_numpy(), plain)
        for book, contract_pts in (
            ("plain book", plain),
            ("live book (month-end override)", live),
        ):
            pnl = contract_pts * SPX_INDEX_MULTIPLIER / CAPITAL
            stress_cost = n * real["stress_dollars"].to_numpy() / CAPITAL
            for per, mask in (
                ("deck period", real.index <= DECK_END),
                ("HOLDOUT", real.index >= OOS_START),
                ("all", np.ones(len(real), bool)),
            ):
                row = {
                    "rule": rule,
                    "book": book,
                    "sample": per,
                    "contracts_per_1M_median": float(np.median(n_full[mask])),
                    "multiple_of_stress": float(
                        np.median(
                            (
                                n_full
                                / (
                                    FRACTION
                                    * CAPITAL
                                    / real["stress_dollars"].to_numpy()
                                )
                            )[mask]
                        )
                    ),
                    "whole_contracts_per_1M_median": float(
                        np.median(np.floor(n_full[mask]))
                    ),
                    "stress_jump_cost_median": float(
                        np.median(stress_cost[mask & (m > 0)])
                    ),
                    "stress_jump_cost_max": float(np.max(stress_cost[mask])),
                }
                row.update(evaluate(pnl[mask], real.index[mask]))
                rows_b.append(row)
    b = pd.DataFrame(rows_b)
    b.to_csv(OUT / "b_realized_by_rule.csv", index=False)
    print(
        f"\nB  realized 13:30 book net of 0.5 bp, brake on, fraction {FRACTION:.0%} of ${CAPITAL:,.0f}; "
        "returns and losses as fractions of capital"
    )
    print(b.round(4).to_string(index=False))
    print(
        f"   month-end sessions in the sample: {int(me_np.sum())} (the plain book is shown; the live "
        "override replaces them with a long straddle)"
    )

    # ------------------------------------------------- C replay, tails only --
    x = rep["loss_frac"].to_numpy()
    mm = rep["m"].to_numpy()
    sess = pd.DatetimeIndex(rep["session"])
    w_rep, q_rep = causal_hist(rep, sess)
    rows_c = []
    stress_median = float(stress_frac.median())
    for rule, lfrac in (
        (
            "R_stress (live, at its median fraction of the index)",
            np.full(len(sess), stress_median),
        ),
        ("R_worst (history, at the brake's size)", w_rep),
        ("R_q (history, 1-in-1000)", q_rep),
    ):
        ok = np.isfinite(lfrac) & (lfrac > 0)
        day_loss = np.where(ok, FRACTION * mm * x / np.where(ok, lfrac, 1.0), np.nan)
        pnl = -day_loss[ok]
        rows_c.append(
            {
                "rule": rule,
                "sessions": int(ok.sum()),
                "first": str(sess[ok][0].date()),
                "worst_day": float(pnl.min()),
                "worst_day_date": str(sess[ok][int(np.argmin(pnl))].date()),
                "worst_20": rolling_worst(pnl, RUN),
                "days_losing_over_fraction": int((pnl < -FRACTION).sum()),
            }
        )
    c = pd.DataFrame(rows_c)
    c.to_csv(OUT / "c_replay_tails_by_rule.csv", index=False)
    print(
        "\nC  the rules walked through the replay (tails only: the replay's mean is not the book's)"
    )
    print(c.round(4).to_string(index=False))

    # --------------------------------------------------------- verdict -----
    rw = b[
        (b["rule"] == "R_worst (history, at the brake's size)") & (b["sample"] == "all")
    ].iloc[0]
    cw = c[c["rule"] == "R_worst (history, at the brake's size)"].iloc[0]
    ok_i = rw["worst_day"] >= -FRACTION
    ok_ii = rw["worst_20"] >= cw["worst_20"]
    print(
        f"\nVERDICT  R_worst on the realized sessions: worst day {rw['worst_day']:.2%} (limit {-FRACTION:.0%}) -> {ok_i}; "
        f"worst 20-session run {rw['worst_20']:.2%} vs the replay's {cw['worst_20']:.2%} -> {ok_ii}; "
        f"{'SUPPORTED' if ok_i and ok_ii else 'NOT SUPPORTED'}"
    )


if __name__ == "__main__":
    main()
