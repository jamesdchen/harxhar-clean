"""Fair comparison of a CARC bucket campaign against the incumbent per-bar arms.

Families (``--family``), one bucket per representation of the implied-vol
information, everything else the live_feasible base:

  mfiv     the model-free 0DTE strip as regressor (mfiv_only, live_feasible_mfiv,
           live_feasible_plus_mfiv; tw 500), plus part B: the HAR ladder of the
           per-bar regression (base-2, base-3, same-clock) against production
           base-5.  Comparison rows 2022-01-03 .. 2024-04-30 (the first 500-day
           window that holds only real chain rows ends late 2021).
  ivslice  the ATM implied slice, the deck's own object, as regressor
           (ivslice_only, live_feasible_ivslice, live_feasible_plus_ivslice; tw
           500; rows 2022-01-03 .. 2024-04-30).
  ivrep    representations of the VIX family (ivrep_target_scale, ivrep_term,
           ivrep_innovations, ivrep_vrp, ivrep_all at tw 2000 on all common
           rows AND on the deck period 2020-01-03 .. 2024-04-30 side by side;
           ivrep_slice_slope at tw 500 on 2022-01-03 .. 2024-04-30).
  vixonly  does the VIX bucket need VVIX and VIX3M (vix_only, vix_rvol,
           live_vix_only, live_vix_rvol at tw 2000 on all common rows and on the
           deck period; vix_rvol = the realized vol-of-VIX from the VIX's own
           path in place of the two indices).

Every forecast is put through the same causal back-transform as the campaign
scorer (``causal_forecasts`` of score_linear_subsection_causal.py: per-clock
smear, 250-day window, 63-day warm-up) and the trade is the deck's 15:30
sign(s) (``trade_1530`` of score_linear_subsection.py) on the deck days from
the comparison window's start.  QLIKE is paired on the common rows of the
variant and the incumbent arm of the same estimator and window; the interval
on the daily QLIKE difference is the scorer's circular block bootstrap.

Inputs: results/linear_subsection/arms_carc/live_feasible (incumbent, flat
layout est/twN/results_<bar>.csv) and results/linear_subsection/<family>_carc
(pulled from CARC, the spec's nested layout).  Outputs beside the pulled arms:
compare_<family>.csv (and compare_harlag.csv for --family mfiv).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "experiments", ROOT / "notebooks"):
    sys.path.insert(0, str(p))

from score_linear_subsection import day_block_ci, trade_1530  # noqa: E402
from score_linear_subsection_causal import causal_forecasts, qlike  # noqa: E402

INCUMBENT = ROOT / "results" / "linear_subsection" / "arms_carc" / "live_feasible"
PULLED_BASE = ROOT / "results" / "linear_subsection"
BARS = [
    f"bar{h:02d}{m:02d}" for h in range(10, 17) for m in (0, 30) if (h, m) <= (16, 0)
]
EST = {"ridge": "ridge", "reclasticnet": "enet", "reclasso": "lasso"}
DECK_START, DECK_END = "2020-01-03", "2024-04-30"
CHAIN_START = "2022-01-03"  # first day with a full 500-day window of real chain rows
HAR_VARIANTS = ("base2", "base3", "intra")
HAR_BARS = ("bar1430", "bar1600")

# Per family: pulled subdirectory, buckets -> train window, comparison windows per
# bucket as (label, lo, hi) with the first one primary, and whether part B runs.
Window = tuple[str, str | None, str | None]
FAMILIES: dict[str, dict] = {
    "mfiv": {
        "pulled": "mfiv_carc",
        "tw": {
            "mfiv_only": 500,
            "live_feasible_mfiv": 500,
            "live_feasible_plus_mfiv": 500,
        },
        "windows": {"*": [("chain", CHAIN_START, DECK_END)]},
        "har": True,
    },
    "ivslice": {
        "pulled": "ivslice_carc",
        "tw": {
            "ivslice_only": 500,
            "live_feasible_ivslice": 500,
            "live_feasible_plus_ivslice": 500,
        },
        "windows": {"*": [("chain", CHAIN_START, DECK_END)]},
        "har": False,
    },
    "ivrep": {
        "pulled": "ivrep_carc",
        "tw": {
            "ivrep_target_scale": 2000,
            "ivrep_term": 2000,
            "ivrep_innovations": 2000,
            "ivrep_vrp": 2000,
            "ivrep_all": 2000,
            "ivrep_slice_slope": 500,
        },
        "windows": {
            "*": [("all", None, None), ("deck", DECK_START, DECK_END)],
            "ivrep_slice_slope": [("chain", CHAIN_START, DECK_END)],
        },
        "har": False,
    },
    "vixonly": {
        "pulled": "vixonly_carc",
        "tw": {
            "vix_only": 2000,
            "vix_rvol": 2000,
            "live_vix_only": 2000,
            "live_vix_rvol": 2000,
        },
        "windows": {"*": [("all", None, None), ("deck", DECK_START, DECK_END)]},
        "har": False,
    },
    # The bucket the free feed can build (live_feasible minus the ES liquidity
    # columns and numobs): what the close-signal service forecasts with.
    "free": {
        "pulled": "free_carc",
        "tw": {
            "free_feasible": 2000,
            "free_feasible_vol": 2000,
            "free_vix_only": 2000,
        },
        "windows": {"*": [("all", None, None), ("deck", DECK_START, DECK_END)]},
        "har": False,
    },
}


def _prep(f: Path) -> pd.DataFrame | None:
    """The causal scorer's load_adj on a path, then the causal back-transform."""
    if not f.exists():
        return None
    r = pd.read_csv(f, parse_dates=["date"]).set_index("date").sort_index()
    r = r[(r["true_adj"] > 0) & (r["true_raw"] > 0)]
    r["baseline"] = r["true_raw"] / r["true_adj"] ** 2
    r["e2"] = (r["true_adj"] - r["pred_adj"]) ** 2
    r["hhmm"] = r.index.strftime("%H:%M")
    r["day"] = r.index.normalize()
    return causal_forecasts(r)


def incumbent(est: str, tw: int, seg: str) -> pd.DataFrame | None:
    return _prep(INCUMBENT / est / f"tw{tw}" / f"results_{seg}.csv")


def pulled(
    pulled_dir: str, root: str, bucket: str, seg: str, est: str, tw: int
) -> pd.DataFrame | None:
    d = (
        PULLED_BASE
        / pulled_dir
        / root
        / bucket
        / seg
        / est
        / f"tw{tw}"
        / "causal_tune_linear"
        / est
        / bucket
    )
    return _prep(d / f"results_{seg}.csv")


def paired_qlike(
    mine: pd.DataFrame, ref: pd.DataFrame, lo: str | None, hi: str | None
) -> dict:
    j = (
        mine[["true_raw", "pred_clock"]]
        .join(ref[["pred_clock"]], how="inner", rsuffix="_ref")
        .dropna()
    )
    if lo:
        j = j[j.index >= lo]
    if hi:
        j = j[j.index <= f"{hi} 23:59"]
    if j.empty:
        return {
            "rows": 0,
            "first": "",
            "pct_vs_ref": np.nan,
            "d_ci_lo": np.nan,
            "d_ci_hi": np.nan,
        }
    q_m = qlike(j["true_raw"], j["pred_clock"])
    q_r = qlike(j["true_raw"], j["pred_clock_ref"])
    d = q_m - q_r
    ci_lo, ci_hi = day_block_ci(d)
    return {
        "rows": int(len(j)),
        "first": str(j.index.min().date()),
        "qlike": float(q_m.mean()),
        "qlike_ref": float(q_r.mean()),
        "pct_vs_ref": float(100 * (q_m.mean() / q_r.mean() - 1)),
        "d_ci_lo": float(ci_lo),
        "d_ci_hi": float(ci_hi),
    }


def trade(mine: pd.DataFrame, lo: str | None) -> dict:
    f = mine["pred_clock"]
    if lo:
        f = f[f.index >= lo]
    return trade_1530(f)


def _ci(pq: dict) -> str:
    return f"[{pq['d_ci_lo']:+.4f}, {pq['d_ci_hi']:+.4f}]"


def part_a(fam: dict) -> pd.DataFrame:
    """Every bucket of the family against the incumbent arm of the same estimator and window."""
    rows = []
    windows_of = fam["windows"]
    for est, label in EST.items():
        root = (
            "linear_subsection_lassofix" if est == "reclasso" else "linear_subsection"
        )
        refs: dict[int, dict[str, pd.DataFrame | None]] = {}
        for b, tw in fam["tw"].items():
            wins: list[Window] = windows_of.get(b, windows_of["*"])
            primary = wins[0]
            if tw not in refs:
                refs[tw] = {s: incumbent(est, tw, s) for s in BARS + ["rth"]}
                ref16 = refs[tw]["bar1600"]
                assert ref16 is not None, f"incumbent {est} tw{tw} bar1600 missing"
                one_tw = len(set(fam["tw"].values())) == 1
                rows.append(
                    {
                        "bucket": "live_feasible (incumbent)"
                        if one_tw
                        else f"live_feasible (incumbent, tw {tw})",
                        "estimator": label,
                        "arm": "bar1600",
                        "qlike_pct_vs_incumbent_16": 0.0,
                        "d_ci": "",
                        "qlike_pct_mean_13_bars": 0.0,
                        "bars_better": "",
                        **{
                            f"trade_{k}": v for k, v in trade(ref16, primary[1]).items()
                        },
                    }
                )
            ref = refs[tw]
            ref16 = ref["bar1600"]
            assert ref16 is not None
            per_bar: dict[str, list[float]] = {w[0]: [] for w in wins}
            for seg in BARS:
                m = pulled(fam["pulled"], root, b, seg, est, tw)
                if m is None or ref[seg] is None:
                    continue
                for w in wins:
                    per_bar[w[0]].append(
                        paired_qlike(m, ref[seg], w[1], w[2])["pct_vs_ref"]
                    )
            m16 = pulled(fam["pulled"], root, b, "bar1600", est, tw)
            assert m16 is not None, f"{b} {est} bar1600 missing"
            pq = paired_qlike(m16, ref16, primary[1], primary[2])
            row = {
                "bucket": b,
                "estimator": label,
                "arm": "bar1600",
                "qlike_pct_vs_incumbent_16": pq["pct_vs_ref"],
                "d_ci": _ci(pq),
                "qlike_pct_mean_13_bars": float(np.mean(per_bar[primary[0]])),
                "bars_better": f"{int(np.sum(np.array(per_bar[primary[0]]) < 0))}/{len(per_bar[primary[0]])}",
                **{f"trade_{k}": v for k, v in trade(m16, primary[1]).items()},
            }
            for w in wins[1:]:
                pqw = paired_qlike(m16, ref16, w[1], w[2])
                row[f"qlike_pct_vs_incumbent_16_{w[0]}"] = pqw["pct_vs_ref"]
                row[f"d_ci_{w[0]}"] = _ci(pqw)
                row[f"qlike_pct_mean_13_bars_{w[0]}"] = float(np.mean(per_bar[w[0]]))
            rows.append(row)
            # the whole-session arm, scored at the close only (families that ran it)
            mr = pulled(fam["pulled"], root, b, "rth", est, tw)
            if mr is not None and ref["rth"] is not None:
                m16r = mr[mr["hhmm"] == "16:00"]
                r16r = ref["rth"][ref["rth"]["hhmm"] == "16:00"]
                pqr = paired_qlike(m16r, r16r, primary[1], primary[2])
                rows.append(
                    {
                        "bucket": b,
                        "estimator": label,
                        "arm": "rth @16:00",
                        "qlike_pct_vs_incumbent_16": pqr["pct_vs_ref"],
                        "d_ci": _ci(pqr),
                        "qlike_pct_mean_13_bars": np.nan,
                        "bars_better": "",
                        **{f"trade_{k}": v for k, v in trade(m16r, primary[1]).items()},
                    }
                )
    return pd.DataFrame(rows)


def part_b(pulled_dir: str) -> pd.DataFrame:
    rows = []
    for est, label in EST.items():
        for tw in (500, 2000):
            for seg in HAR_BARS:
                ref = incumbent(est, tw, seg)
                if ref is None:
                    continue
                for v in HAR_VARIANTS:
                    m = pulled(
                        pulled_dir,
                        "linear_subsection_har/" + v,
                        "live_feasible",
                        seg,
                        est,
                        tw,
                    )
                    if m is None:
                        continue
                    pq_all = paired_qlike(m, ref, None, None)
                    pq_deck = paired_qlike(m, ref, DECK_START, DECK_END)
                    row = {
                        "ladder": v,
                        "estimator": label,
                        "train_win": tw,
                        "bar": seg,
                        "rows": pq_all["rows"],
                        "first": pq_all["first"],
                        "qlike_pct_vs_base5": pq_all["pct_vs_ref"],
                        "d_ci": _ci(pq_all),
                        "qlike_pct_vs_base5_deck": pq_deck["pct_vs_ref"],
                    }
                    if seg == "bar1600":
                        t = trade(m, None)
                        t_ref = trade(ref, None)
                        row.update(
                            {
                                "Sharpe_mid": t["Sharpe_mid"],
                                "Sharpe_crossed": t["Sharpe_crossed"],
                                "pct_buy": t["pct_buy"],
                                "base5_Sharpe_mid": t_ref["Sharpe_mid"],
                                "base5_Sharpe_crossed": t_ref["Sharpe_crossed"],
                                "base5_pct_buy": t_ref["pct_buy"],
                            }
                        )
                    rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=sorted(FAMILIES), default="mfiv")
    args = ap.parse_args()
    fam = FAMILIES[args.family]
    out = PULLED_BASE / fam["pulled"]
    pd.set_option("display.width", 250)
    a = part_a(fam)
    a.to_csv(out / f"compare_{args.family}.csv", index=False)
    print(
        f"A  {args.family} buckets vs the incumbent live_feasible (same estimator and window), QLIKE paired on "
        f"identical rows of the primary comparison window; trade = 15:30 sign(s) on the deck days from that window's start"
    )
    cols = [
        "bucket",
        "estimator",
        "arm",
        "qlike_pct_vs_incumbent_16",
        "d_ci",
        "qlike_pct_mean_13_bars",
        "bars_better",
        "trade_deck_days",
        "trade_pct_buy",
        "trade_Sharpe_mid",
        "trade_Sharpe_crossed",
        "trade_ridge_Sharpe_mid_same_days",
    ]
    extra = [c for c in a.columns if c not in cols]
    print(a[cols + extra].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if fam["har"]:
        b = part_b(fam["pulled"])
        b.to_csv(out / "compare_harlag.csv", index=False)
        print(
            "\nB  HAR ladder vs the production base-5, live_feasible, same bar / estimator / window, common rows "
            "(deck column: 2020-01 .. 2024-04); trade rows for the 16:00 bar on all deck days"
        )
        print(b.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    main()
