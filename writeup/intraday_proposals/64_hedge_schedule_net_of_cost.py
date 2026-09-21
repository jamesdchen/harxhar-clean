"""64 - the insurance book's delta-hedge schedule, chosen NET of hedging cost.

Why.  Every insurance-book Sharpe quoted so far is gross of what the futures
hedge costs to run.  Charged 0.5 bp of the index on every unit traded, the 13:30
hold book falls from +0.80 to +0.47 index points per contract per day and from
Sharpe 1.68 to 0.98 (study 45's own table).  Studies 39 and 40 closed the
cadence question on GROSS Sharpe ("every skip lowers gross Sharpe"), which is
the wrong criterion once each rebalance costs money: a schedule that hedges less
gives up some variance reduction and saves some cost, and only the net result
can choose between them.

The tape.  Proposal 43's chain build, unchanged: every chain session
2020-01-03 .. 2025-12-31 (half sessions dropped), the nearest-OTM straddle sold
at the entry clock at the quoted bid, its Black-76 package delta computed at
each 30-minute stamp from that stamp's own midpoint-inverted total volatility,
the futures position carried from each stamp to the next.  Two treatments:
  hold     both legs to the 4pm cash settlement; the futures position held at
           the last stamp is unwound against the settlement print
  flatten  both legs bought back at the 15:30 quoted ask and the futures
           unwound at 15:30 (reference only)
Entry clocks 11:00 and 13:30 are the headline; every entry clock 10:00 .. 15:00
is tabulated.  Per contract, index points.  The deck period (to 2024-04-30, 866
sessions) and the holdout (2024-05-01 .. 2025-12-31, 413 sessions) are reported
separately.

Cost model.  At every change of the futures position, including the unwind,
c x |change| x S, with S the index at that stamp (the settlement print for the
unwind), for c in {0, 0.25, 0.5, 1.0} bp of the index.  0.25 bp is about half
an ES tick at an SPX level near 5000 (a passive or mid fill), 0.5 bp about a
full tick (crossing) or MES all-in, 1.0 bp a pessimistic bound.  0.5 bp is
proposal 43's HEDGE_COST_BP.

Schedules, written before running; none has a fitted parameter:
  H30    every 30 minutes (the book)
  H60    at entry, then on the hour stamps only (the 15:00 position is carried
         through 15:30 to the settlement)
  Hlast  at entry, then only at 15:00 and 15:30
  Hnone  at entry only; that position is carried to the settlement
  H0     no hedge at all
  Hband  at entry, then only when the package delta has moved further from the
         futures position than one 30-minute expected move of the index would
         move it: band = Gamma x S x sigma_30, with sigma_30 the stamp's own
         remaining total volatility spread evenly over the remaining 30-minute
         bars.  In Black-76 units the band is (phi(d1_call) + phi(d1_put)) /
         sqrt(bars remaining): defined by the option at that stamp, not tuned.
         (A stamp without an inverted volatility has no band and is not
         rebalanced; the book at such a stamp holds a zero delta.)
  Hhalf  half the package delta, every 30 minutes

Decision rule, written before running.  A schedule is ADOPTED at a cost level
for an entry clock if its Sharpe beats H30's at the same cost with a paired
circular-block bootstrap interval (block 21, B = 2000, seed 0) above zero in the
deck period AND its Sharpe difference on the holdout has the same sign.  Six
alternative schedules x four costs x two headline clocks = 48 cells for the hold
treatment; the bootstrap leg alone passes about 1.2 by chance, the holdout sign
requirement roughly halves that.

Also reported: futures turnover and rebalances per day, and the stress loss per
contract at 15:30 with each schedule's futures position carried into the close:
(i) the live package's convention (live.ibkr.sizing.stress_loss_per_contract: a
settlement jump of STRESS_JUMP either way, the worse option side plus the
futures leg's loss charged on its adverse side), and (ii) the same jump with the
futures position's P&L signed, worse of the two sides.

GATES  proposal 43's gate_zero and gate_one; H30 equals proposal 43's hold and
       flatten series and its cost_hold_pts / cost_flatten_pts at 0.5 bp,
       exactly; study 45's 13:30 deck-period hedge cost (mean 0.330470 points,
       net mean 0.467875, net Sharpe 0.978806) reproduces.

Run:  python writeup/intraday_proposals/64_hedge_schedule_net_of_cost.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "notebooks", ROOT / "writeup"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import atm_straddle_lib as asl  # noqa: E402
from live.ibkr.sizing import (  # noqa: E402
    SPX_INDEX_MULTIPLIER,
    STRESS_JUMP,
    stress_loss_per_contract,
)

HERE = Path(__file__).resolve().parent
OUT = ROOT / "results" / "atm_straddle_intraday_holdclose" / "proposals" / "64"
ANN = float(np.sqrt(asl.PERIODS_PER_YEAR))
DECK_END, OOS_START = "2024-04-30", "2024-05-01"
HEADLINE = ("11:00", "13:30")
SCHEDULES = ("H30", "H60", "Hlast", "Hnone", "H0", "Hband", "Hhalf")
COSTS_BP = (0.0, 0.25, 0.5, 1.0)
BP = 1e-4
B, BLOCK, SEED = 2000, 21, 0
LAST_CLOCKS = ("15:00", "15:30")
GATE_45 = {
    "mean_cost": 0.3304704615619846,
    "mean_net": 0.46787468126428183,
    "Sharpe_net": 0.9788060579917849,
}
GATE_TOL = 1e-9


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sharpe(x: np.ndarray) -> float:
    v = np.asarray(x, float)
    v = v[np.isfinite(v)]
    sd = float(v.std(ddof=1)) if v.size > 1 else float("nan")
    return float(v.mean() / sd * ANN) if sd > 0 else float("nan")


def sharpe_rows(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1) / x.std(axis=1, ddof=1) * ANN


def maxdd(x: np.ndarray) -> float:
    v = np.asarray(x, float)
    v = v[np.isfinite(v)]
    path = np.cumsum(v)
    peak = np.maximum.accumulate(np.concatenate(([0.0], path)))[1:]
    return float((path - peak).min())


def pkg_band(
    tot: np.ndarray,
    s: np.ndarray,
    kc: np.ndarray,
    kp: np.ndarray,
    bars_left: np.ndarray,
) -> np.ndarray:
    """(phi(d1c) + phi(d1p)) / sqrt(bars left): the delta move of one 30-minute expected move."""
    ok = np.isfinite(tot) & (tot > 0) & np.isfinite(s) & (s > 0)
    v = np.where(ok, tot, 1.0)
    f = np.where(ok, s, 1.0)
    d1c = (np.log(f / kc) + 0.5 * v * v) / v
    d1p = (np.log(f / kp) + 0.5 * v * v) / v
    pdf = (np.exp(-0.5 * d1c * d1c) + np.exp(-0.5 * d1p * d1p)) / np.sqrt(2.0 * np.pi)
    return np.where(ok, pdf / np.sqrt(bars_left), np.nan)


def positions(
    target: np.ndarray, band: np.ndarray, j: int, schedule: str, clocks: tuple[str, ...]
) -> np.ndarray:
    """Futures position held from each stamp to the next under a schedule."""
    n_d, n_k = target.shape
    pos = np.zeros_like(target)
    prev = np.zeros(n_d)
    for k in range(j, n_k):
        tk = target[:, k]
        if schedule == "H30":
            new = tk
        elif schedule == "Hhalf":
            new = 0.5 * tk
        elif schedule == "H0":
            new = np.zeros(n_d)
        elif k == j:
            new = tk  # every other schedule hedges at entry
        elif schedule == "Hnone":
            new = prev
        elif schedule == "H60":
            new = tk if clocks[k].endswith(":00") else prev
        elif schedule == "Hlast":
            new = tk if clocks[k] in LAST_CLOCKS else prev
        elif schedule == "Hband":
            with np.errstate(invalid="ignore"):
                move = np.abs(tk - prev) > band[:, k]
            new = np.where(move, tk, prev)
        else:
            raise ValueError(schedule)
        pos[:, k] = new
        prev = new
    return pos


def book(
    ch: dict[str, Any], p43: ModuleType, j: int, schedule: str
) -> dict[str, np.ndarray]:
    """Proposal 43's _clock_book with the hedge schedule a parameter."""
    s, s_close = ch["S"], ch["S_close"]
    n_d, n_k = s.shape
    kc, kp = ch["K_c"][:, j], ch["K_p"][:, j]
    entry, bid = ch["entry"][:, j], ch["bid"][:, j]
    tot = ch["tot"].copy()
    tot[:, :j] = np.nan
    dlt = p43.pkg_delta_vec(tot, s, kc[:, None], kp[:, None])
    dlt[:, :j] = 0.0
    bars_left = (n_k - np.arange(n_k))[None, :].astype(float)
    band = pkg_band(tot, s, kc[:, None], kp[:, None], bars_left)
    pos = positions(dlt, band, j, schedule, p43.CLOCKS)
    nxt = np.full_like(s, np.nan)
    nxt[:, :-1] = s[:, 1:]
    nxt[:, -1] = s_close
    d_s = np.where(np.isfinite(s) & np.isfinite(nxt), nxt - s, 0.0)
    d_s[:, :j] = 0.0
    d_s_flat = d_s.copy()
    d_s_flat[:, -1] = 0.0
    pos_flat = pos.copy()
    pos_flat[:, -1] = 0.0
    hedge_hold = (pos * d_s).sum(axis=1)
    hedge_flat = (pos_flat * d_s_flat).sum(axis=1)
    turn_h = np.zeros(n_d)
    turn_f = np.zeros(n_d)
    prev_h = np.zeros(n_d)
    prev_f = np.zeros(n_d)
    rebal = np.zeros(n_d)
    for k in range(j, n_k):
        turn_h += np.abs(pos[:, k] - prev_h) * s[:, k]
        turn_f += np.abs(pos_flat[:, k] - prev_f) * s[:, k]
        rebal += (pos[:, k] != prev_h).astype(float)
        prev_h, prev_f = pos[:, k], pos_flat[:, k]
    turn_h += np.abs(prev_h) * s_close
    turn_f += np.abs(prev_f) * s[:, -1]
    settle = np.maximum(s_close - kc, 0.0) + np.maximum(kp - s_close, 0.0)
    buyback = ch["ask_c"][:, j] + ch["ask_p"][:, j]
    ok = np.isfinite(entry) & (entry > 0.0) & np.isfinite(bid) & (bid > 0.0)
    nan = np.full(n_d, np.nan)
    return {
        "hold_pts": np.where(ok, -(settle - bid) + hedge_hold, nan),
        "flat_pts": np.where(ok, -(buyback - bid) + hedge_flat, nan),
        "turn_hold": np.where(ok, turn_h, nan),
        "turn_flat": np.where(ok, turn_f, nan),
        "rebalances": np.where(ok, rebal, nan),
        "entry": np.where(ok, entry, nan),
        "pos_close": np.where(ok, pos[:, -1], nan),
        "S1530": s[:, -1],
        "kc": kc,
        "kp": kp,
    }


def gates(ch: dict[str, Any], p43: ModuleType) -> None:
    arrays = {
        k: ch[k] for k in ("S", "K_c", "K_p", "entry", "bid", "tot", "ask_c", "ask_p")
    }
    arrays["S_close"] = ch["S_close"]
    books = {
        clock: {
            k: pd.Series(v, index=ch["dates"])
            for k, v in p43._clock_book((j, arrays)).items()
        }
        for j, clock in enumerate(p43.ENTRIES)
    }
    p43.gate_zero(books, ch)
    p43.gate_one(books)
    worst = 0.0
    for j, clock in enumerate(p43.ENTRIES):
        ref = books[clock]
        mine = book(ch, p43, j, "H30")
        for key_mine, key_ref, scale in (
            ("hold_pts", "hold", True),
            ("flat_pts", "flatten", True),
        ):
            got = mine[key_mine] / mine["entry"] if scale else mine[key_mine]
            r = ref[key_ref].to_numpy(float)
            both = np.isnan(got) & np.isnan(r)
            worst = max(worst, float(np.nanmax(np.where(both, 0.0, np.abs(got - r)))))
        for key_mine, key_ref in (
            ("turn_hold", "cost_hold_pts"),
            ("turn_flat", "cost_flatten_pts"),
        ):
            got = mine[key_mine] * p43.HEDGE_COST_BP
            r = ref[key_ref].to_numpy(float)
            both = np.isnan(got) & np.isnan(r)
            worst = max(worst, float(np.nanmax(np.where(both, 0.0, np.abs(got - r)))))
    assert worst < GATE_TOL, worst
    print(
        f"GATE  H30 equals proposal 43's hold, flatten and 0.5 bp costs at every entry clock: {worst:.1e}"
    )
    j = p43.ENTRIES.index("13:30")
    b = book(ch, p43, j, "H30")
    m = np.asarray(ch["dates"] <= pd.Timestamp(DECK_END))
    cost = b["turn_hold"][m] * p43.HEDGE_COST_BP
    net = b["hold_pts"][m] - cost
    got = {
        "mean_cost": float(np.nanmean(cost)),
        "mean_net": float(np.nanmean(net)),
        "Sharpe_net": sharpe(net),
    }
    for key, val in GATE_45.items():
        assert abs(got[key] - val) < GATE_TOL, (key, got[key], val)
    print(
        f"GATE  study 45's 13:30 deck-period hedge cost reproduces: cost {got['mean_cost']:.6f}, "
        f"net {got['mean_net']:.6f}, net Sharpe {got['Sharpe_net']:.6f}"
    )


def stress(b: dict[str, np.ndarray], m: np.ndarray) -> dict[str, float]:
    live, signed = [], []
    for i in np.flatnonzero(m & np.isfinite(b["pos_close"]) & np.isfinite(b["S1530"])):
        s0, pos = float(b["S1530"][i]), float(b["pos_close"][i])
        kc, kp, prem = float(b["kc"][i]), float(b["kp"][i]), float(b["entry"][i])
        r = stress_loss_per_contract(s0, kc, kp, prem, pos, jump=STRESS_JUMP)
        live.append(float(r["total"]))
        worst = -np.inf
        for sign in (1.0, -1.0):
            s1 = s0 * (1.0 + sign * STRESS_JUMP)
            opt = max(s1 - kc, 0.0) + max(kp - s1, 0.0) - prem
            worst = max(worst, opt - pos * (s1 - s0))
        signed.append(worst * SPX_INDEX_MULTIPLIER)
    lv, sv = np.asarray(live), np.asarray(signed)
    return {
        "stress_live_median": float(np.median(lv)),
        "stress_live_p95": float(np.quantile(lv, 0.95)),
        "stress_signed_median": float(np.median(sv)),
        "stress_signed_p95": float(np.quantile(sv, 0.95)),
    }


def main() -> None:  # noqa: PLR0915
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 400)
    p43 = _load(HERE / "43_causal_entry_over_time.py", "p43_causal_entry")
    stamp, _ = p43.session_stamps()
    sessions = pd.DatetimeIndex(stamp.index).difference(p43.half_sessions(stamp))
    ch = p43.build_chain(stamp, sessions)
    gates(ch, p43)
    idx = ch["dates"]
    periods = {
        "deck": np.asarray(idx <= pd.Timestamp(DECK_END)),
        "holdout": np.asarray(idx >= pd.Timestamp(OOS_START)),
        "all": np.ones(len(idx), bool),
    }

    rows, stress_rows = [], []
    for j, clock in enumerate(p43.ENTRIES):
        books = {sch: book(ch, p43, j, sch) for sch in SCHEDULES}
        for treat, pts_key, turn_key in (
            ("hold", "hold_pts", "turn_hold"),
            ("flatten", "flat_pts", "turn_flat"),
        ):
            for c in COSTS_BP:
                net = {
                    sch: books[sch][pts_key] - c * BP * books[sch][turn_key]
                    for sch in SCHEDULES
                }
                for pname, pm in periods.items():
                    ok = pm.copy()
                    for sch in SCHEDULES:
                        ok &= np.isfinite(net[sch])
                    n = int(ok.sum())
                    bidx = asl.circular_block_bootstrap_idx(
                        np.random.default_rng([SEED, n]), n, BLOCK, B
                    )
                    base = net["H30"][ok]
                    s_base = sharpe_rows(base[bidx])
                    for sch in SCHEDULES:
                        x = net[sch][ok]
                        d = sharpe_rows(x[bidx]) - s_base
                        rows.append(
                            {
                                "entry": clock,
                                "treatment": treat,
                                "schedule": sch,
                                "cost_bp": c,
                                "period": pname,
                                "days": n,
                                "mean_pts": float(x.mean()),
                                "sd": float(x.std(ddof=1)),
                                "Sharpe": sharpe(x),
                                "vs_H30": sharpe(x) - sharpe(base),
                                "ci_lo": float(np.percentile(d, 2.5))
                                if sch != "H30"
                                else 0.0,
                                "ci_hi": float(np.percentile(d, 97.5))
                                if sch != "H30"
                                else 0.0,
                                "worst": float(x.min()),
                                "max_drawdown": maxdd(x),
                                "turnover_pts": float(books[sch][turn_key][ok].mean()),
                                "cost_pts": float(
                                    (c * BP * books[sch][turn_key][ok]).mean()
                                ),
                                "rebalances": float(
                                    books[sch]["rebalances"][ok].mean()
                                ),
                            }
                        )
        if clock in HEADLINE:
            for sch in SCHEDULES:
                for pname in ("deck", "holdout"):
                    stress_rows.append(
                        {"entry": clock, "schedule": sch, "period": pname}
                        | stress(books[sch], periods[pname])
                    )
        print(f"entry {clock}: {len(SCHEDULES)} schedules x {len(COSTS_BP)} costs done")

    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "a_schedules_by_cost.csv", index=False)
    cols = [
        "schedule",
        "cost_bp",
        "days",
        "mean_pts",
        "sd",
        "Sharpe",
        "vs_H30",
        "ci_lo",
        "ci_hi",
        "worst",
        "max_drawdown",
        "turnover_pts",
        "cost_pts",
        "rebalances",
    ]
    for clock in HEADLINE:
        for pname in ("deck", "holdout"):
            t = tab[
                (tab["entry"] == clock)
                & (tab["treatment"] == "hold")
                & (tab["period"] == pname)
            ]
            print(
                f"\n=== {clock} book, hold, {pname}: index points per contract per day, net of cost"
            )
            print(t[cols].round(3).to_string(index=False))

    adopt = []
    for clock in HEADLINE:
        for c in COSTS_BP:
            for sch in SCHEDULES[1:]:

                def row(p: str) -> pd.Series:
                    return tab[
                        (tab["entry"] == clock)
                        & (tab["treatment"] == "hold")
                        & (tab["schedule"] == sch)
                        & (tab["cost_bp"] == c)
                        & (tab["period"] == p)
                    ].iloc[0]

                dk, ho = row("deck"), row("holdout")
                adopt.append(
                    {
                        "entry": clock,
                        "cost_bp": c,
                        "schedule": sch,
                        "deck_vs_H30": dk["vs_H30"],
                        "deck_ci_lo": dk["ci_lo"],
                        "deck_ci_hi": dk["ci_hi"],
                        "holdout_vs_H30": ho["vs_H30"],
                        "holdout_ci_lo": ho["ci_lo"],
                        "holdout_ci_hi": ho["ci_hi"],
                        "adopted": bool(dk["ci_lo"] > 0 and ho["vs_H30"] > 0),
                    }
                )
    ad = pd.DataFrame(adopt)
    ad.to_csv(OUT / "b_adoption.csv", index=False)
    print(
        f"\n=== adoption ({len(ad)} cells; about 1.2 pass the bootstrap leg by chance)"
    )
    print(ad.round(3).to_string(index=False))
    print(f"ADOPTED cells: {int(ad['adopted'].sum())}")

    rank = []
    for clock in HEADLINE:
        for c in COSTS_BP:
            t = tab[
                (tab["entry"] == clock)
                & (tab["treatment"] == "hold")
                & (tab["period"] == "deck")
                & (tab["cost_bp"] == c)
            ].sort_values("Sharpe", ascending=False)
            rank.append(
                {
                    "entry": clock,
                    "cost_bp": c,
                    "ranking_deck": " > ".join(
                        f"{r.schedule} {r.Sharpe:.2f}" for r in t.itertuples()
                    ),
                }
            )
    rk = pd.DataFrame(rank)
    rk.to_csv(OUT / "c_ranking.csv", index=False)
    print("\n=== deck-period ranking by net Sharpe, hold treatment")
    print(rk.to_string(index=False))

    allc = tab[(tab["treatment"] == "hold") & (tab["cost_bp"] == 0.5)].pivot_table(
        index=["period", "entry"], columns="schedule", values="Sharpe"
    )[list(SCHEDULES)]
    allc.to_csv(OUT / "d_all_clocks_05bp.csv")
    print("\n=== every entry clock, hold, net Sharpe at 0.5 bp")
    print(allc.round(2).to_string())

    st = pd.DataFrame(stress_rows)
    st.to_csv(OUT / "e_stress.csv", index=False)
    print(
        f"\n=== stress per contract at 15:30, dollars (jump {STRESS_JUMP:.0%}), futures position each schedule carries into the close"
    )
    print(st.round(0).to_string(index=False))

    flat = tab[
        (tab["treatment"] == "flatten")
        & (tab["period"] == "deck")
        & (tab["entry"].isin(HEADLINE))
        & (tab["cost_bp"].isin([0.0, 0.5]))
    ]
    print("\n=== reference: flatten treatment, deck period")
    print(
        flat[
            [
                "entry",
                "schedule",
                "cost_bp",
                "mean_pts",
                "Sharpe",
                "vs_H30",
                "ci_lo",
                "ci_hi",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
