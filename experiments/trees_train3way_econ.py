"""The sign strategy at the close for the trees' three-way training-sample comparison
(results/trees_train3way_2014start_w3000/RULE.txt section 7), run locally: experiments/linear_train3way_econ.py with
the tree classes.  The quote book, the master table and the paper's modules are read (only) from a checkout that holds
them (writeup/ is imported, never written).

Construction: q = +1 if plain16 > iv_var else -1 on the 866 option days (the OLS row's days, iv_var and rv_unclipped;
the forecast row's realized variance must equal rv_unclipped), mid return q x R, bid/ask return gui._crossed_return;
the paper's D6 Sharpe ratio (E.sharpe) on the 614 post-warmup days.  Tests: hac_inference.d6_diff_test (paired circular
block bootstrap, block LW_BLOCK, LW_BOOT resamples, D6_SEED) for a minus b.  Check: the RTH13 rows reproduce the
templates' scores/econ.csv (W = 3,000; make_rehaul_economics.add_pooled_trees' construction): D6 mid and bid/ask of the
HAR and full sets to 1e-12 and the full-vs-HAR bootstrap p to 1e-12.

usage: python experiments/trees_train3way_econ.py <checkout with the inputs> <f16.parquet> <out dir>
  -> <out dir>/econ_d6.csv (variant x model), econ_diff.csv (pairs), econ_tables.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

MAIN = Path(sys.argv[1]).resolve()
for _p in (MAIN, MAIN / "notebooks", MAIN / "writeup", MAIN / "experiments"):
    sys.path.insert(0, str(_p))

import hac_inference as hi  # noqa: E402
import make_rehaul_economics as E  # noqa: E402
from master_table_gui import rows as gui  # noqa: E402

assert Path(E.__file__).resolve().is_relative_to(MAIN), E.__file__
F16, OUT = Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve()
OUT.mkdir(parents=True, exist_ok=True)
HERE = Path(__file__).resolve().parents[1]
EST = ("lgbm", "xgb", "rf_skx")
MODELS = [(e, s) for e in EST for s in ("baseline", "all_features")]
TPL = {
    "lgbm": HERE / "results" / "pooled_trees_final_2014start_w3000" / "scores" / "econ.csv",
    "xgb": HERE / "results" / "pooled_trees_final_2014start_w3000" / "scores" / "econ.csv",
    "rf_skx": HERE / "results" / "pooled_rf_skx_2014start_w3000" / "scores" / "econ.csv",
}
PAIRS = [
    ("CLOSE", "RTH13"),
    ("ALL48", "RTH13"),
    ("CLOSE", "ALL48"),
    ("ALL48_N3k", "RTH13_N3k"),
    ("CLOSE", "RTH13_N3k"),
    ("CLOSE", "ALL48_N3k"),
    ("ALL48_N39k", "RTH13"),
    ("ALL48_N3k", "ALL48"),
    ("RTH13_N3k", "RTH13"),
    ("ALL48_N39k", "ALL48"),
]

mt, model_set, series, fam = E.load()
b = series[E.OLS]
days = b.index
pw = E.pw_days()
assert len(days) == 866 and len(pw) == 614
daily = pd.read_parquet(E.RES / "master_table_daily.parquet", columns=["key", "day", "iv_var", "rv_unclipped"])
daily["day"] = pd.to_datetime(daily["day"]).dt.normalize()
ref = daily[daily["key"] == E.OLS].set_index("day").sort_index()
ref.index = ref.index.astype(days.dtype)
ref = ref.reindex(days)
iv, rv = ref["iv_var"].to_numpy(float), ref["rv_unclipped"].to_numpy(float)
assert np.isfinite(iv).all() and np.isfinite(rv).all()
R, ex, bid, ask = (b[c].to_numpy(float) for c in ("R", "ex", "bid", "ask"))

f16 = pd.read_parquet(F16)
f16["day"] = pd.to_datetime(f16["day"]).astype(days.dtype)
RET: dict = {}
rows = []
for (v, m, s), g in f16.groupby(["variant", "model", "predictor_set"]):
    g = g.set_index("day").reindex(days)
    p = g["plain"].to_numpy(float)
    assert np.isfinite(p).all() and (p > 0).all(), (v, m, s)
    assert np.allclose(g["rv_actual"].to_numpy(float), rv, rtol=1e-12, atol=0.0), (v, m, s)
    q = np.where(p > iv, 1.0, -1.0)
    mid, x = q * R, gui._crossed_return(q, ex, bid, ask)
    RET[(v, m, s)] = dict(mid=mid, ba=x, q=q)
    rows.append(
        dict(
            variant=v,
            model=m,
            set=s,
            d6_mid=E.sharpe(mid),
            d6_ba=E.sharpe(x),
            long_share_614=float((E.post_warmup(q) > 0).mean()),
            mean_log_plain_rv_614=float(np.log(p[E.WARMUP :] / rv[E.WARMUP :]).mean()),
        )
    )
D6 = pd.DataFrame(rows)

diffs = []
for m, s in MODELS:
    for a, bb in PAIRS:
        if (a, m, s) not in RET or (bb, m, s) not in RET:
            continue
        row: dict = dict(model=m, set=s, a=a, b=bb)
        for lab in ("mid", "ba"):
            ra, rb_ = E.post_warmup(RET[(a, m, s)][lab]), E.post_warmup(RET[(bb, m, s)][lab])
            t = hi.d6_diff_test(ra, rb_, pw)
            assert abs(t["delta"] - (E.sharpe(RET[(a, m, s)][lab]) - E.sharpe(RET[(bb, m, s)][lab]))) < 1e-10
            row |= {f"d6_{lab}_a": t["sr_a"], f"d6_{lab}_b": t["sr_b"], f"dd6_{lab}": t["delta"], f"p_{lab}": t["p"]}
        row["same_positions_614"] = int(
            (E.post_warmup(RET[(a, m, s)]["q"]) == E.post_warmup(RET[(bb, m, s)]["q"])).sum()
        )
        diffs.append(row)
# full vs HAR within each arm (the templates' test), mid and bid/ask
gaps = []
for v in sorted(f16["variant"].unique()):
    for m in EST:
        if (v, m, "baseline") not in RET or (v, m, "all_features") not in RET:
            continue
        row = dict(variant=v, model=m)
        for lab in ("mid", "ba"):
            t = hi.d6_diff_test(
                E.post_warmup(RET[(v, m, "all_features")][lab]), E.post_warmup(RET[(v, m, "baseline")][lab]), pw
            )
            row |= {
                f"d6_{lab}_full": t["sr_a"],
                f"d6_{lab}_har": t["sr_b"],
                f"dd6_{lab}": t["delta"],
                f"p_{lab}": t["p"],
            }
        gaps.append(row)
DF, GP = pd.DataFrame(diffs), pd.DataFrame(gaps)
D6.to_csv(OUT / "econ_d6.csv", index=False)
DF.to_csv(OUT / "econ_diff.csv", index=False)
GP.to_csv(OUT / "econ_gap.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
txt = (
    f"D6 Sharpe ratio, 614 post-warmup option days\n{D6.to_string(index=False)}\n\nDifferences\n"
    f"{DF.to_string(index=False)}\n\nFull vs HAR\n{GP.to_string(index=False)}\n"
)
(OUT / "econ_tables.txt").write_text(txt)
print(txt)

# check: RTH13 against the templates' scores/econ.csv (W = 3,000)
for m in EST:
    r = pd.read_csv(TPL[m])
    r = r[(r.W == 3000) & (r.model == m)].iloc[0]
    for lab in ("mid", "ba"):
        for s, col in (("baseline", f"d6_{lab}_har"), ("all_features", f"d6_{lab}_full")):
            got = D6[(D6.variant == "RTH13") & (D6.model == m) & (D6.set == s)][f"d6_{lab}"].iloc[0]
            assert abs(got - r[col]) <= 1e-12, (m, s, lab, got, r[col])
        got = GP[(GP.variant == "RTH13") & (GP.model == m)][f"p_{lab}"].iloc[0]
        assert abs(got - r[f"d6_boot_p_{lab}"]) <= 1e-12, (m, lab, got, r[f"d6_boot_p_{lab}"])
print("check: RTH13 reproduces the templates' econ.csv (W = 3,000): D6 mid / bid-ask and the full-vs-HAR bootstrap p")
