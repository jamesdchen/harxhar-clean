"""68 - the insurance brake on the S&P leg: does it survive the timing the runner can deliver?

Study 67 found that putting the S&P leg of a joint account under the insurance
book's own brake (full weight while the trailing 21-session realized variance
is at or below its expanding lagged 90th percentile, half above it, none above
the 97.5th) roughly halves the worst 20-session run.  It assumed the weight for
session t - known from sessions through t-1 - is on from the CLOSE of t-1.

The live runner cannot do that.  Its ledger takes session t-1 when the operator
reconciles the settlement the next morning, so the weight for session t is
known on the morning of t and the S&P leg is resized during t.  The overnight
gap into t is then carried at the old weight.  The index's open is not reliable
in the daily file, so the realistic timing is BRACKETED:

  L0  weight w_t on from the close of t-1            (study 67, not deliverable)
  L1  weight w_t on from the close of t              (a full session late)
The runner's timing (resized during t) lies between the two.

Rules: BRK the brake; for comparison VM (volatility-managed) and MA200, all as
defined in study 67, each at L0 and L1.  S&P leg = the index's close-to-close
return, price only, cash at zero, 1998-2024.

  A  the S&P leg alone, whole sample and halves: growth, volatility, Sharpe,
     maximum drawdown, worst 20-session run, mean weight, number of weight
     changes a year; the dated drawdowns.
  B  the joint book on the tape 2020-01..2024-04: S&P leg + the realized live
     insurance book at 10% of capital (net of 0.5 bp, brake, month-end
     override).

  C  (written after A and B were seen, before C ran) THE RUNNER'S OWN NUMBERS.
     A and B use the panel's one-minute realized variance, which ends in 2024
     and which the runner never reads.  The runner has the premium ledger's
     session variance (2020-2025).  Both rules computed by the live functions
     (live.ibkr.sp_leg, live.ibkr.premium_ledger) for every index session from
     the first day both are past their warm-up, at L0 and L1, through
     2025-12-31 - so the 2024-05..2025-12 HOLDOUT, with April 2025 in it, is
     scored: the S&P leg alone and the joint book.
     Rule for C: the runner's default is the rule that passed A at L1; it keeps
     that default only if on the HOLDOUT, at L1, it shows a smaller maximum
     drawdown AND a smaller worst 20-session run than buy-and-hold.  If not,
     the default is no S&P-leg report.

Written before running: the brake is ADOPTED for the S&P leg (as a daily target
the runner reports; the operator trades the S&P leg) if AT L1 it still meets
study 67's criterion - smaller maximum drawdown and smaller worst 20-session
run than buy-and-hold in both halves of 1998-2024, and a paired block-bootstrap
interval of the Sharpe difference that is not entirely below zero.

GATE  L0 reproduces study 67's rows for the three rules (whole-sample growth,
      maximum drawdown, worst 20-session run).

Run:  python writeup/intraday_proposals/68_sp_leg_brake_timing.py
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
from live.ibkr.premium_ledger import PremiumLedger  # noqa: E402
from live.ibkr.sp_leg import sp_leg_target  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = ROOT / "results" / "atm_straddle_intraday_holdclose" / "proposals" / "68"
P67_RULES = (
    ROOT
    / "results"
    / "atm_straddle_intraday_holdclose"
    / "proposals"
    / "67"
    / "d_sp_leg_rules.csv"
)
ANN = float(np.sqrt(asl.PERIODS_PER_YEAR))
B, BLOCK, SEED = 2000, 21, 0
GATE_TOL = 1e-9
OOS_START = "2024-05-01"
NAMES_67 = {
    "BRK": "BRK brake on the S&P",
    "VM": "VM volatility-managed",
    "MA200": "MA200 trend",
}


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def lagged(w: np.ndarray, k: int) -> np.ndarray:
    """The weight k sessions late; full weight until the first late value exists."""
    if k == 0:
        return w
    return np.concatenate([np.ones(k), w[:-k]])


def main() -> None:  # noqa: PLR0915
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    p43 = _load(HERE / "43_causal_entry_over_time.py", "p43_causal_entry")
    p49 = _load(HERE / "49_cat_replay.py", "p49_cat_replay")
    p50 = _load(HERE / "50_cat_replay_repriced.py", "p50_cat_replay")
    p65 = _load(HERE / "65_capital_from_history.py", "p65_capital")
    p67 = _load(HERE / "67_crash_forecast.py", "p67_crash")
    d = p67.panel_daily(p49)
    n = len(d)
    r = d["r"].to_numpy()

    sig_vol = np.sqrt(np.exp(d["HAR"].to_numpy()))
    med = (
        pd.Series(sig_vol)
        .expanding(min_periods=p67.MIN_OBS)
        .median()
        .shift(1)
        .to_numpy()
    )
    am = d["above_ma"].to_numpy(float)
    brk = p65.brake(d["RV21"].to_numpy(float), p50)
    base = {
        "BRK": np.where(np.isfinite(brk), brk, 1.0),
        "VM": np.where(
            np.isfinite(med) & np.isfinite(sig_vol), np.minimum(1.0, med / sig_vol), 1.0
        ),
        "MA200": np.where(np.isfinite(am), am, 1.0),
    }
    ref = pd.read_csv(P67_RULES)
    for key, w in base.items():
        got = p67.summary(w * r)
        row = ref[(ref["rule"] == NAMES_67[key]) & (ref["part"] == "whole")].iloc[0]
        for col in ("ann_growth", "max_drawdown", "worst_20"):
            assert abs(got[col] - row[col]) < GATE_TOL, (key, col, got[col], row[col])
    print(
        "GATE  L0 reproduces study 67's S&P-leg rows (growth, maximum drawdown, worst 20-session run)"
    )

    rules: dict[str, np.ndarray] = {"buy and hold": np.ones(n)}
    for key, w in base.items():
        rules[f"{key} L0 (from the prior close)"] = w
        rules[f"{key} L1 (a session late)"] = lagged(w, 1)

    mid = n // 2
    parts = {
        "whole": slice(0, n),
        f"to {d.index[mid - 1].date()}": slice(0, mid),
        f"from {d.index[mid].date()}": slice(mid, n),
    }
    idx = asl.circular_block_bootstrap_idx(
        np.random.default_rng([SEED, 68]), n, BLOCK, B
    )
    rb = r[idx]
    s_bh = rb.mean(axis=1) / rb.std(axis=1, ddof=1) * ANN
    years = n / asl.PERIODS_PER_YEAR
    rows = []
    for name, w in rules.items():
        xb = w[idx] * rb
        dsh = xb.mean(axis=1) / xb.std(axis=1, ddof=1) * ANN - s_bh
        for pname, sl in parts.items():
            row = {
                "rule": name,
                "part": pname,
                "mean_weight": float(w[sl].mean()),
                **p67.summary((w * r)[sl]),
            }
            if pname == "whole":
                row |= {
                    "weight_changes_per_year": float((np.diff(w) != 0).sum() / years),
                    "dSharpe_lo": float(np.percentile(dsh, 2.5)),
                    "dSharpe_hi": float(np.percentile(dsh, 97.5)),
                }
            rows.append(row)
    a = pd.DataFrame(rows)
    a.to_csv(OUT / "a_sp_leg_by_timing.csv", index=False)
    print(
        "\nA  the S&P leg by rule and timing (index close-to-close, price only, cash at zero), fractions"
    )
    print(a.round(3).to_string(index=False))

    named = pd.DataFrame(
        {
            name: {
                lab: float(np.prod(1 + (w * r)[(d.index > a0) & (d.index <= b0)]) - 1)
                for lab, a0, b0 in p67.DRAWDOWNS
            }
            for name, w in rules.items()
        }
    )
    named.to_csv(OUT / "a_named_drawdowns.csv")
    print("\n   return inside the dated S&P drawdowns (peak close -> trough close)")
    print(named.round(3).T.to_string())

    verdict = []
    for name in list(rules)[1:]:
        ok = []
        for pname in list(parts)[1:]:
            bh = a[(a["rule"] == "buy and hold") & (a["part"] == pname)].iloc[0]
            ru = a[(a["rule"] == name) & (a["part"] == pname)].iloc[0]
            ok.append(
                bool(
                    ru["max_drawdown"] > bh["max_drawdown"]
                    and ru["worst_20"] > bh["worst_20"]
                )
            )
        hi = float(
            a[(a["rule"] == name) & (a["part"] == "whole")]["dSharpe_hi"].iloc[0]
        )
        verdict.append(
            {
                "rule": name,
                "smaller_drawdowns_both_halves": all(ok),
                "Sharpe_not_worse": hi > 0,
                "PASSES": bool(all(ok) and hi > 0),
            }
        )
    v = pd.DataFrame(verdict)
    v.to_csv(OUT / "a_verdict.csv", index=False)
    print()
    print(v.to_string(index=False))

    # ------------------------------------------------ B the joint book ------
    rep = p65.load_replay(p50)
    real = p65.load_realized(p43)
    real["m"] = p65.realized_brake(real, rep)
    c54 = pd.read_csv(p65.P54_DAILY, index_col=0, parse_dates=True).reindex(real.index)
    me = c54["month_end"].fillna(False).astype(bool).to_numpy()
    pts = np.where(
        me,
        c54["pts_ask"].to_numpy(float),
        real["m"].to_numpy() * real["net_pts"].to_numpy(),
    )
    live = pd.Series(
        p65.FRACTION
        * pts
        * p65.SPX_INDEX_MULTIPLIER
        / real["stress_dollars"].to_numpy(),
        index=real.index,
    )
    tm = np.asarray(d.index >= p67.TAPE_START)
    live_t = live.reindex(d.index[tm]).fillna(0.0).to_numpy()
    rows_b = [{"book": "insurance alone (realized)", **p67.summary(live_t)}]
    for name, w in rules.items():
        rows_b.append(
            {"book": f"{name} + insurance", **p67.summary((w * r)[tm] + live_t)}
        )
    bt = pd.DataFrame(rows_b)
    bt.to_csv(OUT / "b_joint_book_tape.csv", index=False)
    print(
        f"\nB  the joint book on the tape {p67.TAPE_START[:7]}..{d.index[-1].date()}: S&P leg + the realized live "
        "insurance book at 10%"
    )
    print(bt.round(3).to_string(index=False))

    # ------------------------------------- C the runner's own numbers -------
    ledger = PremiumLedger.load(p65.SEED_LEDGER)
    gs = pd.read_parquet(p67.GSPC)["close"]
    gs = gs[(gs.index >= real.index[0]) & (gs.index <= real.index[-1])]
    rr = gs.pct_change().to_numpy()
    targets = {
        rule: [sp_leg_target(ledger, day.date(), rule) for day in gs.index]
        for rule in ("vol_managed", "brake")
    }
    active = np.ones(len(gs), bool)
    wts = {}
    for rule, recs in targets.items():
        wts[rule] = np.array([float(x["weight"]) for x in recs])
        active &= np.array([x["state"] != "warmup" for x in recs])
    first = int(np.flatnonzero(active)[0]) + 1  # L1 needs the day before it too
    days_c = gs.index[first:]
    live_c = live.reindex(days_c).fillna(0.0).to_numpy()
    books: dict[str, np.ndarray] = {"buy and hold": np.ones(len(gs))}
    for rule, w in wts.items():
        books[f"{rule} L0"] = w
        books[f"{rule} L1"] = lagged(w, 1)
    rows_c = []
    for pname, mask in (
        (f"{days_c[0].date()}..2024-04", np.asarray(days_c < OOS_START)),
        ("HOLDOUT 2024-05..2025-12", np.asarray(days_c >= OOS_START)),
        ("all", np.ones(len(days_c), bool)),
    ):
        for name, w in books.items():
            x = (w * rr)[first:][mask]
            rows_c.append(
                {
                    "sample": pname,
                    "book": name,
                    "leg": "S&P alone",
                    "mean_weight": float(w[first:][mask].mean()),
                    **p67.summary(x),
                }
            )
            rows_c.append(
                {
                    "sample": pname,
                    "book": name,
                    "leg": "S&P + insurance",
                    "mean_weight": float(w[first:][mask].mean()),
                    **p67.summary(x + live_c[mask]),
                }
            )
    ct = pd.DataFrame(rows_c)
    ct.to_csv(OUT / "c_runner_rules_ledger.csv", index=False)
    print(
        f"\nC  the runner's own rules off the premium ledger, {days_c[0].date()}..{days_c[-1].date()} "
        f"({len(days_c)} index sessions); vol_managed below full weight on "
        f"{float((wts['vol_managed'][first:] < 1).mean()):.0%} of days, the brake on "
        f"{float((wts['brake'][first:] < 1).mean()):.0%}"
    )
    print(ct.round(3).to_string(index=False))
    winner = (
        "vol_managed"
        if bool(v[v["rule"] == "VM L1 (a session late)"].iloc[0]["PASSES"])
        else (
            "brake"
            if bool(v[v["rule"] == "BRK L1 (a session late)"].iloc[0]["PASSES"])
            else ""
        )
    )
    keep = False
    if winner:
        h = ct[
            (ct["sample"] == "HOLDOUT 2024-05..2025-12") & (ct["leg"] == "S&P alone")
        ].set_index("book")
        keep = bool(
            h.loc[f"{winner} L1", "max_drawdown"]
            > h.loc["buy and hold", "max_drawdown"]
            and h.loc[f"{winner} L1", "worst_20"] > h.loc["buy and hold", "worst_20"]
        )
    print(
        f"\nVERDICT C  rule that passed A at L1: {winner or 'none'}; on the holdout at L1 it has the smaller "
        f"maximum drawdown and worst 20-session run: {keep} -> the runner's default is "
        f"{winner if keep else 'no S&P-leg report'}"
    )

    l1 = v[v["rule"] == "BRK L1 (a session late)"].iloc[0]
    print(
        f"\nVERDICT  the brake on the S&P leg, a full session late: "
        f"{'ADOPTED as the runner-reported target' if l1['PASSES'] else 'NOT ADOPTED'}"
    )


if __name__ == "__main__":
    main()
