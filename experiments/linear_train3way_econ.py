"""The sign strategy at the close for the three-way training-sample comparison (results/linear_train3way_2014start_w3000/
RULE.txt section 8), run locally: the quote book, the master table and the paper's modules are read (only) from a
checkout that holds them (the main worktree; writeup/ is imported, never written).

Construction: writeup/make_rehaul_economics.add_pooled_linear, applied to each variant's 16:00 forecast (scores/
f16.parquet from experiments/linear_train3way_score.py): q = +1 if plain > iv_var else -1 on the 866 option days (the
OLS row's days, iv_var and rv_unclipped; the forecast row's realized variance must equal rv_unclipped), mid return
q x R, bid/ask return gui._crossed_return; the paper's D6 Sharpe ratio (E.sharpe) on the 614 post-warmup days.  Tests:
hac_inference.d6_diff_test (paired circular block bootstrap, block LW_BLOCK, LW_BOOT resamples, D6_SEED) for a minus b.
Check: the RTH13 rows reproduce the window sweep's part2/econ.csv (W = 3,000) to 1e-12, its bootstrap p to 1e-12 (the csv round trip).

usage: python experiments/linear_train3way_econ.py <checkout with the inputs> <f16.parquet> <out dir>
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
MODELS = [("ols", "baseline")] + [(e, s) for s in ("baseline", "all_features") for e in ("ridge", "lasso", "enet")]
PAIRS = [
    ("CLOSE", "RTH13"),
    ("ALL48", "RTH13"),
    ("CLOSE", "ALL48"),
    ("CLOSE_NAT", "RTH13"),
    ("ALL48@16", "RTH13"),
    ("RTH13@16", "RTH13"),
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
for v in f16["variant"].unique():
    for m, s in MODELS:
        g = f16[(f16.variant == v) & (f16.model == m) & (f16.predictor_set == s)].set_index("day").reindex(days)
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

# check: RTH13 against the window sweep's part 2 (W = 3,000): D6 and the full-vs-HAR bootstrap p
p2 = pd.read_csv(HERE / "results" / "window_sweep_2014start_2026-10-22" / "part2" / "econ.csv")
p2 = p2[p2.W == 3000]
for _, r in p2.iterrows():
    for lab in ("mid", "ba"):
        for s, col in (("baseline", f"d6_{lab}_har"), ("all_features", f"d6_{lab}_full")):
            got = D6[(D6.variant == "RTH13") & (D6.model == r.model) & (D6.set == s)][f"d6_{lab}"].iloc[0]
            assert abs(got - r[col]) <= 1e-12, (r.model, s, lab, got, r[col])
        t = hi.d6_diff_test(
            E.post_warmup(RET[("RTH13", r.model, "all_features")][lab]),
            E.post_warmup(RET[("RTH13", r.model, "baseline")][lab]),
            pw,
        )
        assert abs(t["p"] - r[f"d6_boot_p_{lab}"]) <= 1e-12, (r.model, lab, t["p"], r[f"d6_boot_p_{lab}"])
print("check: RTH13 reproduces part2/econ.csv (w3000): D6 mid / bid-ask and the full-vs-HAR bootstrap p", flush=True)

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
DF = pd.DataFrame(diffs)
D6.to_csv(OUT / "econ_d6.csv", index=False)
DF.to_csv(OUT / "econ_diff.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
txt = f"D6 Sharpe ratio, 614 post-warmup option days\n{D6.to_string(index=False)}\n\nDifferences\n{DF.to_string(index=False)}\n"
(OUT / "econ_tables.txt").write_text(txt)
print(txt)
