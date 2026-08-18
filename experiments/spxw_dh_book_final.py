"""The 0DTE delta-hedged leg book with its two survivors combined.

Regime/exit study (spxw_dh_regime_exit.py) found exactly two rules that
lift the book after costs: (1) do not trade FOMC-announcement sessions
(the remaining-variance forecast cannot see the announcement; the book
goes 97% short at 5x normal |sig| and loses); (2) exit when the
re-measured signal crosses back through zero rather than holding to
settlement (the only early-exit rule that survives the option spread).

This composes them: 2x2 {all days, ex-FOMC} x {hold, signal-cross},
theta grid, per leg, per era, per entry hour, mids and crossed, blk2
and a0 (swap test). Reuses the regime script's machinery verbatim
(imports; no re-implementation) so the numbers tie to its assertions.

FOMC flags end 2024-01-09 (data/releases.parquet); days after that are
UNKNOWN, not "no FOMC" -- the ex-FOMC book keeps them (documented
choice: the exclusion can only be applied where the flag exists) and
the coverage row reports the count.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from spxw_dh_regime_exit import (  # noqa: E402
    DAILY_0DTE,
    OUT,
    _positions,
    build_paths,
    build_state,
    daily_stats,
    exit_pnl,
    rule_exit_index,
)

THETAS = (0.05, 0.10, 0.20)


def _book(
    pos: np.ndarray,
    pe: dict[str, np.ndarray],
    keep: np.ndarray,
    days: np.ndarray,
) -> dict[str, Any]:
    traded = (pos != 0) & keep
    pmid = pos * pe["long_mid"]
    px = np.where(pos > 0, pe["long_x"], np.where(pos < 0, pe["short_x"], np.nan))
    a = daily_stats(pmid, traded, days)
    c = daily_stats(px, traded, days)
    return {
        "n_traded": a["n"],
        "n_days": a["n_days"],
        "frac_traded": float(traded[keep].mean()) if keep.any() else float("nan"),
        "frac_long": float((pos[traded] > 0).mean()) if traded.any() else float("nan"),
        "sh_mid": a["sh"],
        "hit_mid": a["hit"],
        "sh_crossed": c["sh"],
        "hit_crossed": c["hit"],
        "mean_crossed": c["mean"],
    }


def main() -> None:
    led = pd.read_parquet(os.path.join(OUT, "dh_legs_ledger.parquet"))
    led["expiration"] = pd.to_datetime(led["expiration"])
    led = led.sort_values(["expiration", "t", "strike", "cp"]).reset_index(drop=True)
    st = build_state(led)
    led = led.merge(
        st[["expiration", "hhmm", "fomc"]], on=["expiration", "hhmm"], how="left"
    )
    n = len(led)
    days = led["expiration"].to_numpy()
    fomc = led["fomc"].to_numpy(float)
    is_fomc = np.nan_to_num(fomc, nan=0.0) > 0  # unknown -> not excluded
    n_fomc_days = int(led.loc[is_fomc, "expiration"].nunique())
    n_unknown_days = int(led.loc[np.isnan(fomc), "expiration"].nunique())
    print(
        f"ledger rows {n} days {led['expiration'].nunique()}  "
        f"FOMC days {n_fomc_days}  flag-unknown days {n_unknown_days}",
        flush=True,
    )

    p = build_paths(led)
    m = p["m"]
    hold_e = np.full(n, -1)
    pe_hold = exit_pnl(p, hold_e)

    eras = {
        "all": np.ones(n, bool),
        "daily_0dte": led["expiration"].to_numpy() >= np.datetime64(DAILY_0DTE),
    }
    legs = {
        "both": np.ones(n, bool),
        "call": (led["cp"] == "C").to_numpy(),
        "put": (led["cp"] == "P").to_numpy(),
    }
    hh = led["hhmm"].to_numpy()
    hours = sorted(pd.unique(hh))

    rows: list[dict[str, Any]] = []
    byhour: list[dict[str, Any]] = []
    for model in ("b2", "a0"):
        sig = led[f"sig_{model}"].to_numpy(float)
        for th in THETAS:
            pos = _positions(sig, th)
            e_sc = rule_exit_index(p, pos, "sigcross")
            pe_sc = exit_pnl(p, e_sc)
            for excl in ("all_days", "ex_fomc"):
                keep_base = ~is_fomc if excl == "ex_fomc" else np.ones(n, bool)
                for rule, pe in (("hold", pe_hold), ("sigcross", pe_sc)):
                    for era, em in eras.items():
                        for leg, lm in legs.items():
                            r = _book(pos, pe, keep_base & em & lm, days)
                            r.update(
                                model=model,
                                theta=th,
                                days=excl,
                                exit=rule,
                                era=era,
                                leg=leg,
                            )
                            rows.append(r)
                    if th == 0.10 and model == "b2":
                        for h in hours:
                            r = _book(pos, pe, keep_base & (hh == h), days)
                            r.update(days=excl, exit=rule, entry=h)
                            byhour.append(r)
            # paired swap test at the final book (ex_fomc + sigcross)
            if model == "a0":
                continue
        # (swap test rows below, after both models done)

    out = pd.DataFrame(rows)
    # swap test: b2 vs a0 paired daily difference, ex_fomc + sigcross, per theta
    swap: list[dict[str, Any]] = []
    for th in THETAS:
        pb = _positions(led["sig_b2"].to_numpy(float), th)
        pa = _positions(led["sig_a0"].to_numpy(float), th)
        eb = exit_pnl(p, rule_exit_index(p, pb, "sigcross"))
        ea = exit_pnl(p, rule_exit_index(p, pa, "sigcross"))
        keep = ~is_fomc
        db = (
            pd.Series(np.where(keep, pb * eb["long_mid"], np.nan), index=days)
            .groupby(level=0)
            .mean()
        )
        da = (
            pd.Series(np.where(keep, pa * ea["long_mid"], np.nan), index=days)
            .groupby(level=0)
            .mean()
        )
        d = (db - da).dropna().to_numpy(float)
        t = (
            float(d.mean() / (d.std(ddof=1) / np.sqrt(d.size)))
            if d.size > 2
            else float("nan")
        )
        swap.append(
            {
                "theta": th,
                "book": "ex_fomc+sigcross",
                "n_days": int(d.size),
                "t_b2_minus_a0": t,
                "mean_diff": float(d.mean()),
            }
        )
    swap_df = pd.DataFrame(swap)

    cov = pd.DataFrame(
        [
            {
                "ledger_days": int(led["expiration"].nunique()),
                "fomc_days_excluded": n_fomc_days,
                "fomc_flag_unknown_days_kept": n_unknown_days,
                "flag_end": "2024-01-09",
            }
        ]
    )
    out.to_csv(os.path.join(OUT, "dh_book_final.csv"), index=False)
    pd.DataFrame(byhour).to_csv(
        os.path.join(OUT, "dh_book_final_by_hour.csv"), index=False
    )
    swap_df.to_csv(os.path.join(OUT, "dh_book_final_swap.csv"), index=False)
    cov.to_csv(os.path.join(OUT, "dh_book_final_coverage.csv"), index=False)

    pd.set_option("display.width", 240)
    pd.set_option("display.max_rows", 500)
    hl = out[(out.leg == "both") & (out.era == "all") & (out.model == "b2")]
    print("\n=== 2x2 headline: both legs, era all, blk2 ===", flush=True)
    print(
        hl[
            [
                "theta",
                "days",
                "exit",
                "n_days",
                "frac_traded",
                "frac_long",
                "sh_mid",
                "hit_mid",
                "sh_crossed",
                "hit_crossed",
            ]
        ]
        .round(3)
        .to_string(index=False),
        flush=True,
    )
    fin = out[(out.days == "ex_fomc") & (out.exit == "sigcross")]
    print("\n=== FINAL BOOK (ex-FOMC + signal-cross), all cuts ===", flush=True)
    print(
        fin[
            [
                "model",
                "theta",
                "era",
                "leg",
                "n_days",
                "frac_traded",
                "frac_long",
                "sh_mid",
                "hit_mid",
                "sh_crossed",
                "hit_crossed",
            ]
        ]
        .round(3)
        .to_string(index=False),
        flush=True,
    )
    print("\n=== swap test on the final book ===", flush=True)
    print(swap_df.round(3).to_string(index=False), flush=True)
    print("\n=== by entry hour, theta=0.10, blk2 ===", flush=True)
    print(
        pd.DataFrame(byhour)[
            [
                "days",
                "exit",
                "entry",
                "n_days",
                "frac_long",
                "sh_mid",
                "sh_crossed",
                "hit_crossed",
            ]
        ]
        .round(3)
        .to_string(index=False),
        flush=True,
    )
    print("\n" + cov.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
