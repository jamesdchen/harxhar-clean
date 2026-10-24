"""The sign strategy at the close for the close-only window sweep (results/closeonly_wsweep_2014start/RULE.txt sections
5 and 6), run locally on the pulled 16:00 forecasts, and the part-2 tables (results/window_sweep_2014start_2026-10-22/
part2/tables.txt layout) with the 16:00 scores of experiments/closewin_score.py.

Construction (experiments/linear_train3way_econ.py): q = +1 if plain16 > iv_var else -1 on the 866 option days (the OLS
row's days, iv_var and rv_unclipped; the forecast's realized variance must equal rv_unclipped), mid return q x R,
bid/ask return gui._crossed_return; D6 Sharpe ratio (E.sharpe) on the 614 post-warmup days, per W and model.  Full
minus HAR per W and estimator as the window sweep's score14.py (b): ΔD6 with hac_inference.d6_diff_test (mid and
bid/ask), D1 (sharpe_days_d1) with the Ledoit-Wolf p, the pooled 614-day ratio, LLSW, the long shares.  Check: W = 3,000
reproduces the three-way CLOSE rows of econ_d6.csv (D6 mid and bid/ask) to 1e-12.

usage: python experiments/closewin_econ.py <checkout with the inputs> linear|trees <scores dir (closewin_score.py
         outputs, f16.parquet)> <ref econ_d6.csv (three-way)> <out dir>
  -> <out>/econ_d6.csv econ_gap.csv check_econ.json tables.txt
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
MAIN = Path(sys.argv[1]).resolve()
for _p in (MAIN, MAIN / "notebooks", MAIN / "writeup", MAIN / "experiments"):
    sys.path.insert(0, str(_p))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import hac_inference as hi  # noqa: E402
import make_rehaul_economics as E  # noqa: E402
from closewin import W_FIRST, W_REF, WS  # noqa: E402
from master_table_gui import rows as gui  # noqa: E402

assert Path(E.__file__).resolve().is_relative_to(MAIN), E.__file__
FAM = sys.argv[2]
SC, REF_D6, OUT = Path(sys.argv[3]).resolve(), Path(sys.argv[4]).resolve(), Path(sys.argv[5]).resolve()
OUT.mkdir(parents=True, exist_ok=True)
if FAM == "linear":
    EST: tuple[str, ...] = ("ridge", "lasso", "enet")
    MODELS = [("ols", "baseline")] + [(e, s) for s in ("baseline", "all_features") for e in EST]
    LAB = {"ols": "OLS (HAR set)", "ridge": "Ridge", "lasso": "Lasso", "enet": "Elastic net"}
else:
    EST = ("lgbm", "xgb", "rf_skx")
    MODELS = [(e, s) for e in EST for s in ("baseline", "all_features")]
    LAB = {"lgbm": "LightGBM", "xgb": "XGBoost", "rf_skx": "Random forest (oneDAL)"}
SL = {"baseline": "HAR", "all_features": "full"}


def ann252(x) -> float:
    x = np.asarray(x, float)
    return float(x.mean() / x.std(ddof=1) * np.sqrt(252.0))


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

f16 = pd.read_parquet(SC / "f16.parquet")
f16["day"] = pd.to_datetime(f16["day"]).astype(days.dtype)
WSX = tuple(W for W in WS if W in set(f16["W"]))  # 2026-10-24 addendum A: the windows scored
assert W_REF in WSX, WSX
RET: dict = {}
PL: dict = {}
rows = []
for W in WSX:
    for m, s in MODELS:
        g = f16[(f16.W == W) & (f16.model == m) & (f16.predictor_set == s)].set_index("day").reindex(days)
        p = g["plain"].to_numpy(float)
        assert np.isfinite(p).all() and (p > 0).all(), (W, m, s)
        assert np.allclose(g["rv_actual"].to_numpy(float), rv, rtol=1e-12, atol=0.0), (W, m, s)
        q = np.where(p > iv, 1.0, -1.0)
        mid, x = q * R, gui._crossed_return(q, ex, bid, ask)
        RET[(W, m, s)] = dict(mid=mid, ba=x, q=q)
        PL[(W, m, s)] = p
        rows.append(
            dict(
                W=W,
                model=m,
                set=s,
                d6_mid=E.sharpe(mid),
                d6_ba=E.sharpe(x),
                long_share_614=float((E.post_warmup(q) > 0).mean()),
                mean_log_plain_rv_614=float(np.log(p[E.WARMUP :] / rv[E.WARMUP :]).mean()),
            )
        )
D6 = pd.DataFrame(rows)

# check: W = 3,000 = the three-way CLOSE arm's D6
rd = pd.read_csv(REF_D6)
rd = rd[rd.variant == "CLOSE"]
assert len(rd) == len(MODELS), len(rd)
chk: dict = {}
for _, r in rd.iterrows():
    a = D6[(D6.W == W_REF) & (D6.model == r.model) & (D6.set == r.set)].iloc[0]
    d = [float(abs(a.d6_mid - r.d6_mid)), float(abs(a.d6_ba - r.d6_ba))]
    chk[f"{r.model}/{r.set}"] = dict(mine=[a.d6_mid, a.d6_ba], ref=[r.d6_mid, r.d6_ba], pass_=max(d) <= 1e-12)
chk["pass"] = all(v["pass_"] for v in chk.values())
(OUT / "check_econ.json").write_text(json.dumps(chk, indent=1))
print(f"CHECK econ (W = 3,000 = the three-way CLOSE D6): {'PASS' if chk['pass'] else 'FAIL'}", flush=True)

econ = []
for W in WSX:
    for e in EST:
        out: dict = dict(W=W, model=e)
        h, fu = RET[(W, e, "baseline")], RET[(W, e, "all_features")]
        for lab in ("mid", "ba"):
            hp, fp_ = E.post_warmup(h[lab]), E.post_warmup(fu[lab])
            sh, sf = E.sharpe(h[lab]), E.sharpe(fu[lab])
            bt = hi.d6_diff_test(fp_, hp, pw)
            assert abs(bt["delta"] - (sf - sh)) < 1e-10
            ll = hi.llsw_test(fp_ - hp)
            out |= {
                f"d6_{lab}_har": sh,
                f"d6_{lab}_full": sf,
                f"dd6_{lab}": sf - sh,
                f"d6_boot_p_{lab}": bt["p"],
                f"d1_{lab}_har": E.sharpe_days_d1(hp),
                f"d1_{lab}_full": E.sharpe_days_d1(fp_),
                f"lw_p_{lab}": hi.lw_sharpe_test(fp_, hp, block=hi.LW_BLOCK)["p"],
                f"llsw_t_{lab}": ll["t"],
                f"llsw_p_{lab}": ll["p"],
                f"sr614_{lab}_har": ann252(hp),
                f"sr614_{lab}_full": ann252(fp_),
            }
            out[f"dd1_{lab}"] = out[f"d1_{lab}_full"] - out[f"d1_{lab}_har"]
            out[f"dsr614_{lab}"] = out[f"sr614_{lab}_full"] - out[f"sr614_{lab}_har"]
        for lab, s in (("har", "baseline"), ("full", "all_features")):
            out[f"long_share_{lab}"] = float((E.post_warmup(RET[(W, e, s)]["q"]) > 0).mean())
            out[f"mean_log_plain_rv_{lab}"] = float(np.log(PL[(W, e, s)][E.WARMUP :] / rv[E.WARMUP :]).mean())
        econ.append(out)
    print(f"W={W}: economics done", flush=True)
EC = pd.DataFrame(econ)
D6.to_csv(OUT / "econ_d6.csv", index=False)
EC.to_csv(OUT / "econ_gap.csv", index=False)

# ---------------------------------------------------------------------------- tables (part2/tables.txt layout)
Q, DM, G = (pd.read_csv(SC / f"{n}.csv") for n in ("qlike16", "dm", "gap"))
nd = int(Q.n_days.iloc[0])
lines = [
    f"Close-only training (the 16:00 rows of each W design, window W closes), {FAM}; 16:00, 2014-01-02 .. 2024-04-30"
]


def wl(W: int, e: str) -> str:
    """The W label (addendum B: the linear penalized models' W = 1,000 point is a 1,001-close window)."""
    return "1,000 (penalized: 1,001)" if FAM == "linear" and W == 1000 and e != "ols" else f"{W:,}"


def cell(m: str, s: str, W: int, kind: str) -> str:
    x = DM[(DM.model == m) & (DM.set == s) & (DM.W == W) & (DM.kind == kind)]
    return "—" if x.empty else f"{x.pct.iloc[0]:+.2f} ({x.dm_qlike_p.iloc[0]:.3f})"


for e in (["ols"] if FAM == "linear" else []) + list(EST):
    sets = ("baseline",) if e == "ols" else ("baseline", "all_features")
    lines += ["", f"{LAB[e]}: QLIKE 2014-01-02 .. 2024-04-30 ({nd} days, 16:00)"]
    head = "| set | W | 16:00 QLIKE | Δ% step (DM p) | Δ% vs 1,000 (DM p) | Δ% vs 3,000 (DM p) | MSE x1e10 | R2_OOS % |"
    if FAM == "linear" and e != "ols":
        head += " penalty |"
    lines += [head, "|" + "---|" * (head.count("|") - 1)]
    for s in sets:
        for W in WSX:
            x = Q[(Q.model == e) & (Q.set == s) & (Q.W == W)].iloc[0]
            ln = (
                f"| {SL[s]} | {wl(W, e)} | {x.qlike:.5f} | {cell(e, s, W, 'step')} | {cell(e, s, W, 'vs1000')} | "
                f"{cell(e, s, W, 'vs3000')} | {x.mse_e10:.4f} | {x.r2_oos_pct:.2f} |"
            )
            if FAM == "linear" and e != "ols":
                ln += f" {x.alpha:g} |"
            lines.append(ln)
    if e == "ols":
        continue
    lines += ["| full vs HAR | W | 16:00 gap % (DM p) | DM stat |"]
    for W in WSX:
        x = G[(G.model == e) & (G.W == W)].iloc[0]
        lines.append(f"| gap | {wl(W, e)} | {x.pct:+.2f} ({x.dm_qlike_p:.3f}) | {x.dm_qlike:+.2f} |")
    lines += [
        "",
        f"{LAB[e]}: sign strategy, 614 post-warmup option days (2021-08 .. 2024-04)",
        "| W | D6 HAR | D6 full | ΔD6 (boot p) | ΔD6 b/a (boot p) | D6 b/a HAR / full | ΔD1 (LW p) | ΔSR614 | LLSW t (p) | "
        "long HAR / full |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for W in WSX:
        x = EC[(EC.model == e) & (EC.W == W)].iloc[0]
        lines.append(
            f"| {wl(W, e)} | {x.d6_mid_har:.2f} | {x.d6_mid_full:.2f} | {x.dd6_mid:+.2f} ({x.d6_boot_p_mid:.3f}) | "
            f"{x.dd6_ba:+.2f} ({x.d6_boot_p_ba:.3f}) | {x.d6_ba_har:.2f} / {x.d6_ba_full:.2f} | "
            f"{x.dd1_mid:+.2f} ({x.lw_p_mid:.3f}) | {x.dsr614_mid:+.2f} | {x.llsw_t_mid:.2f} ({x.llsw_p_mid:.3f}) | "
            f"{x.long_share_har:.3f} / {x.long_share_full:.3f} |"
        )
if FAM == "linear":
    lines += ["", "OLS (HAR set): sign strategy D6 (mid / bid-ask) by W"]
    lines += ["| W | D6 mid | D6 b/a | long |", "|---|---|---|---|"]
    for W in WSX:
        x = D6[(D6.model == "ols") & (D6.W == W)].iloc[0]
        lines.append(f"| {W:,} | {x.d6_mid:.2f} | {x.d6_ba:.2f} | {x.long_share_614:.3f} |")
lines += ["", f"W = {W_REF:,} check against the three-way CLOSE D6: {'PASS' if chk['pass'] else 'FAIL'}"]
lines += [f"(Δ% vs {W_FIRST:,} / vs {W_REF:,}: QLIKE of W against that W; DM = dm_test HLN, lag 8, on daily QLIKE)"]
txt = "\n".join(lines) + "\n"
(OUT / "tables.txt").write_text(txt, encoding="utf-8")
print(txt)
if not chk["pass"]:
    sys.exit(3)
