"""Two robustness bounds of the closing trade, on the plain forecast, over the model set (2026-10-06).

The text of Section 4 (writeup/sections/methods_close_option.tex) states two bounds that were
measured under back-transforms since removed.  This script re-measures both on the PLAIN
forecast f^2 B (2026-10-06: recalibration removed, user decision), point estimates only, over
the MODEL SET: the master table's table A (results/close_master_table/master_table.csv), check
rows (the direct rest-of-day forecast at 15:30) and always short excluded, and every
live-feasible row excluded (2026-10-06: live-feasible removed from the paper, user decision;
the input sets live_feasible, live_vix_only and live_vix_rvol -- keys containing "live_").

(1) B contamination (methods_close_option.tex, "One related contamination ...").  The 16:00
    slot's diurnal factor B is the trailing DIURNAL_WINDOW-session in-slot mean of the bar's
    realized variance, lagged one session (src/features/transforms/target.py diurnal_adjust).
    On an NYSE early close (atm_straddle_lib.EARLY_CLOSE_DATES) the futures panel still prints a
    16:00 bar, made of post-close bars, and that low value enters B for the following sessions.
    Repair (the one the text names: B rebuilt without the early closes' post-close bars): the
    same trailing mean over the most recent DIURNAL_WINDOW REGULAR sessions; on a day whose
    window holds an early close, B_repaired = B x (repaired mean / contaminated mean), every
    other day unchanged.  The forecast is plain = f^2 B, so the repaired forecast is
    plain x B_repaired / B; the trade is re-run (q = sign(plain - iv_var), mid and crossed
    fills).  The old audit (commit 4e02381, 2026-09-05, "a separate audit", read-only auditors,
    no committed script) bounded the effect under the session-bar map at "under a tenth of a
    Sharpe ratio and a handful of flipped positions".
(2) Censored implied volatility (methods_close_option.tex, "Dropping the censored days ...").
    The deck (results/atm_straddle_0dte_1530/daily_blk2.parquet) flags the days whose picked
    leg sat on a solver node of the vendor's IV (iv_capped; their IV is re-inverted from the
    midpoint).  The other treatment drops those days (the notebook's iv_var_dropcens,
    notebooks/_write_0dte_nb.py, section 9): the Sharpe ratio on the remaining days, every
    other day's position and return unchanged.  The old bound (writeup/fill_close_option_
    numbers.py IVDROP_MAX, the rv_iv notebook's scorer, mid fill, the paper's eight columns):
    0.06.

Sources: results/close_master_table/{master_table.csv, master_table_daily.parquet} (the plain
forecast and the trade of every row), experiments/master_table_close.py (load_target: the
16:00 target with B, the unclipped RV and the early-close flag; trade_days; sharpe), the deck.
Gates (asserted): the all-days Sharpe ratios reproduce master_table.csv to GATE_TOL; the
reconstructed contaminated B equals the target's B on every affected trade day; a variant
leaves every unaffected day's position and returns unchanged; the affected trade days number
N_B_DAYS (the text's 94).

Writes results/close_bounds_plain/{b_contamination.csv, censored_iv.csv, SUMMARY.md}.
Run:  python experiments/close_bounds_plain.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "experiments", ROOT / "notebooks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import master_table_close as mtc  # noqa: E402
import score_linear_subsection as base  # noqa: E402
from src.features.transforms.target import (  # noqa: E402
    DIURNAL_MIN_PERIODS,
    DIURNAL_WINDOW,
)

MT = ROOT / "results" / "close_master_table"
OUT = ROOT / "results" / "close_bounds_plain"
CHECK_FAMILY = "direct rest-of-day at 15:30 (check)"
ALWAYS_SHORT = "always_short"
LIVE = "live_"  # live_feasible, live_vix_only, live_vix_rvol (2026-10-06: not in the paper)
GATE_TOL = 1e-12  # the master table's numbers, recomputed with its own functions
B_REL_TOL = (
    1e-9  # the reconstructed B against the target's B (a rolling mean, re-summed)
)
N_B_DAYS = (
    94  # the text: "Ninety-four of the 866 scored days fall inside such a window"
)
OLD_B = "under a tenth of a Sharpe ratio and a handful of flipped positions (session-bar map)"
OLD_CENS = (
    0.06  # writeup/fill_close_option_numbers.py IVDROP_MAX (notebook scorer, mid)
)


def model_set(m: pd.DataFrame) -> pd.DataFrame:
    """Table A minus check rows, always short and every live-feasible row."""
    return m[
        (m["table"] == "A")
        & (m["family"] != CHECK_FAMILY)
        & (m["key"] != ALWAYS_SHORT)
        & ~m["key"].str.contains(LIVE)
    ]


def b_factor() -> tuple[pd.Series, pd.Series]:
    """B_repaired / B on every 16:00 session (by day), and the affected-day flag."""
    t = mtc.load_target()
    rv, early = t["rv_unclipped"], t["early_close"].astype(bool)
    contaminated = (
        rv.rolling(DIURNAL_WINDOW, min_periods=DIURNAL_MIN_PERIODS).mean().shift(1)
    )
    reg = rv[~early]
    repaired = (
        reg.rolling(DIURNAL_WINDOW, min_periods=DIURNAL_MIN_PERIODS).mean().shift(1)
    ).reindex(t.index)
    # a day is affected when an early close is among its window's previous DIURNAL_WINDOW rows
    affected = (
        early.astype(int)
        .rolling(DIURNAL_WINDOW, min_periods=1)
        .sum()
        .shift(1)
        .fillna(0)
        > 0
    ) & ~early
    rel = (contaminated[affected] / t.loc[affected, "B"] - 1.0).abs()
    factor = pd.Series(1.0, index=t.index)
    factor[affected] = repaired[affected] / contaminated[affected]
    day = t["day"]
    return (
        pd.Series(factor.to_numpy(), index=day.to_numpy()),
        pd.Series(affected.to_numpy(), index=day.to_numpy()),
    ), rel.set_axis(day[affected].to_numpy())


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(MT / "master_table.csv")
    ms = model_set(m).set_index("key")
    daily = pd.read_parquet(MT / "master_table_daily.parquet")
    deck = pd.read_parquet(base.DECK).sort_index()
    deck.index = pd.DatetimeIndex(pd.to_datetime(deck.index)).as_unit("ns")
    days = deck.index
    (factor, affected), b_rel = b_factor()
    factor = factor.reindex(days)
    aff = affected.reindex(days).fillna(False).astype(bool)
    assert factor.notna().all(), "every trade day has a 16:00 target row"
    assert int(aff.sum()) == N_B_DAYS, ("affected trade days", int(aff.sum()))
    b_rel_trade = b_rel.reindex(days[aff.to_numpy()])
    assert float(b_rel_trade.max()) <= B_REL_TOL, (
        "the contaminated B is the target's B on the affected days",
        float(b_rel_trade.max()),
    )
    cens = deck["iv_capped"].astype(bool)
    keep = ~cens.to_numpy()
    rows_b, rows_c = [], []
    for k, r in ms.iterrows():
        dk = daily[daily["key"] == k].set_index("day").sort_index()
        dk.index = pd.DatetimeIndex(dk.index).as_unit("ns")
        assert len(dk) == len(days) and (dk.index == days).all(), k
        t0 = mtc.trade_days(dk["plain"], deck)
        g = max(
            abs(mtc.sharpe(t0["mid"].to_numpy()) - r["Sharpe_mid"]),
            abs(mtc.sharpe(t0["crossed"].to_numpy()) - r["Sharpe_crossed"]),
        )
        assert g <= GATE_TOL, (k, "reproduces master_table.csv", g)
        # (1) B repaired on the affected days
        t1 = mtc.trade_days(dk["plain"] * factor, deck)
        un = ~aff.to_numpy()
        assert (t1.loc[un] == t0.loc[un]).all().all(), (k, "unaffected days unchanged")
        rows_b.append(
            {
                "key": k,
                "label": r["label"],
                "family": r["family"],
                "n_days": len(t0),
                "n_days_affected": int(aff.sum()),
                "Sharpe_mid": mtc.sharpe(t0["mid"].to_numpy()),
                "Sharpe_mid_variant": mtc.sharpe(t1["mid"].to_numpy()),
                "Sharpe_crossed": mtc.sharpe(t0["crossed"].to_numpy()),
                "Sharpe_crossed_variant": mtc.sharpe(t1["crossed"].to_numpy()),
                "positions_flipped": int((t1["q"] != t0["q"]).sum()),
                "max_rel_forecast_change": float((factor - 1.0).abs().max()),
            }
        )
        # (2) censored-IV days dropped (positions and returns of the other days unchanged)
        rows_c.append(
            {
                "key": k,
                "label": r["label"],
                "family": r["family"],
                "n_days": int(keep.sum()),
                "n_days_affected": int(cens.sum()),
                "Sharpe_mid": mtc.sharpe(t0["mid"].to_numpy()),
                "Sharpe_mid_variant": mtc.sharpe(t0["mid"].to_numpy()[keep]),
                "Sharpe_crossed": mtc.sharpe(t0["crossed"].to_numpy()),
                "Sharpe_crossed_variant": mtc.sharpe(t0["crossed"].to_numpy()[keep]),
                "positions_flipped": 0,
            }
        )
    out = {}
    for name, rows in (("b_contamination", rows_b), ("censored_iv", rows_c)):
        x = pd.DataFrame(rows)
        for fill in ("mid", "crossed"):
            x[f"dSharpe_{fill}"] = x[f"Sharpe_{fill}_variant"] - x[f"Sharpe_{fill}"]
        cols = [
            "key",
            "label",
            "family",
            "n_days",
            "n_days_affected",
            "Sharpe_mid",
            "Sharpe_mid_variant",
            "dSharpe_mid",
            "Sharpe_crossed",
            "Sharpe_crossed_variant",
            "dSharpe_crossed",
            "positions_flipped",
        ] + [c for c in x.columns if c == "max_rel_forecast_change"]
        x = x[cols]
        x.to_csv(OUT / f"{name}.csv", index=False, lineterminator="\n")
        out[name] = x
    write_summary(out, len(ms), cens)
    return 0


def _worst(x: pd.DataFrame, col: str) -> str:
    i = x[col].abs().idxmax()
    return f"{x.loc[i, col]:+.3f} ({x.loc[i, 'label']})"


def write_summary(out: dict[str, pd.DataFrame], n: int, cens: pd.Series) -> None:
    b, c = out["b_contamination"], out["censored_iv"]
    fl = b["positions_flipped"]
    imax = fl.idxmax()
    L = [
        "# Two bounds of the closing trade on the plain forecast (2026-10-06)",
        "",
        "Script: `experiments/close_bounds_plain.py`; every number is read from the CSVs in this "
        "directory. Forecast: the plain back-transform f^2 B (no recalibration); point estimates "
        f"only. Model set: {n} forecasts of the master table's table A (check rows, always short "
        "and every live-feasible row -- live_feasible, live_vix_only, live_vix_rvol -- excluded), "
        "on the 866 trade days.",
        "",
        "## 1. B contamination (`b_contamination.csv`)",
        "",
        f"- Days affected: {int(b['n_days_affected'].iloc[0])} of 866 (an NYSE early close among the "
        f"previous {DIURNAL_WINDOW} 16:00 sessions); B repaired there to the trailing mean over the "
        f"last {DIURNAL_WINDOW} regular sessions; largest relative change of the forecast "
        f"{float(b['max_rel_forecast_change'].max()):.1%}.",
        f"- Largest |dSharpe|, mid: {_worst(b, 'dSharpe_mid')}; crossed: {_worst(b, 'dSharpe_crossed')}; "
        f"median |dSharpe| mid {b['dSharpe_mid'].abs().median():.3f}.",
        f"- Positions flipped: at most {int(fl.max())} of {int(b['n_days_affected'].iloc[0])} affected "
        f"days ({b.loc[imax, 'label']}); median {fl.median():.0f}; {int((fl == 0).sum())} of {n} "
        "forecasts flip none.",
        f"- Old value (commit 4e02381, no committed script): {OLD_B}.",
        "",
        "## 2. Censored implied volatility (`censored_iv.csv`)",
        "",
        f"- Censored days (deck `iv_capped`): {int(cens.sum())} "
        f"({', '.join(str(d.date()) for d in cens.index[cens.to_numpy()])}); dropped instead of re-inverted.",
        f"- Largest |dSharpe|, mid: {_worst(c, 'dSharpe_mid')}; crossed: {_worst(c, 'dSharpe_crossed')}; "
        f"median |dSharpe| mid {c['dSharpe_mid'].abs().median():.3f}.",
        f"- Old value (writeup/fill_close_option_numbers.py, notebook scorer, mid, the paper's eight): {OLD_CENS}.",
        "",
    ]
    (OUT / "SUMMARY.md").write_text("\n".join(L), encoding="utf-8", newline="\n")
    print("\n".join(L))


if __name__ == "__main__":
    raise SystemExit(main())
