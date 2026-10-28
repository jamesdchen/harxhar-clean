"""Addendum to the LINEAR factor test beyond HAR + realized moments + FOMC: split FULL_LIN vs BASE_LIN into a
penalization effect and a series effect.  Requested by the author after seeing the linear factor test's results
(eddf391); not pre-registered; exploratory.  Existing forecasts only, nothing refitted.  Local (laptop) run.

QUESTION  FULL_LIN (Table 1's penalized full set) vs BASE_LIN (OLS on the add_rm_fomc base) is R's denominator in
          experiments/beyond_factor_linear.py.  How much of it is the estimator (penalized vs OLS on the same columns) and
          how much is the six remaining groups (ew, vw, liq, iv, sent, vd) within the same estimator?

SPLIT     for m in ridge, lasso, enet, with ols_base = BASE_LIN:
  total     m full  vs ols_base   (Table 1's patched all_features m vs the ablation's OLS add_rm_fomc)
  penal.    m base  vs ols_base   (the ablation's m add_rm_fomc: same 132 columns, penalized vs OLS)
  series    m full  vs m base     (the six remaining groups added, same estimator)
          The other order (series under OLS, then penalization on the full set) needs OLS on the full set, which Table 1
          does not have (its OLS row is the HAR set only); not computed.

INPUTS    results/ablations_2014start_w3000/forecasts/linear_wide_rmfomc.parquet (columns "<m>|add_rm_fomc", pred_adj)
          and beyond_factor_linear.templates() (Table 1's patched forecasts).  Check: ols_base equals the factor test's
          BASE_LIN (beyond_factor_linear.base_lin()).

SCORING   as beyond_factor_linear cmd_score: QLIKE on rv_actual, plain = pred_adj^2 * B; the scored set (33,527 bars)
          and EVENT / ORDINARY days of experiments/ablate_score; DM = src.evaluation.diebold_mariano.dm_test on daily
          means (HLN, Newey-West lag 8).  Changes in percent of ols_base's mean loss on the span, so penal. + series =
          total; "% own" is the change against the step's own reference.

OUTPUTS   results/beyond_trees_2026-10-27/linear/pen_split.csv and ADDENDUM_pen_split.md.

    python experiments/beyond_factor_linear_pensplit.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (HERE, ROOT, ROOT / "notebooks"):
    sys.path.insert(0, str(p))

import ablate_score as AS  # noqa: E402
import beyond_factor_linear as BFL  # noqa: E402

from src.evaluation.diebold_mariano import dm_test  # noqa: E402

PEN = ("ridge", "lasso", "enet")
STEPS = (
    ("total", "m full vs OLS base", "{m}_full", "ols_base"),
    ("penalization", "m vs OLS, base columns", "{m}_base", "ols_base"),
    ("series", "m full vs m base", "{m}_full", "{m}_base"),
)


def forecasts() -> tuple[pd.DataFrame, float]:
    """Plain forecasts (pred_adj^2 * B) of OLS / ridge / lasso / enet on the base and Table 1's templates, on the bar
    index of the ablation's forecasts, and the max relative gap of ols_base to BASE_LIN."""
    T = BFL.templates()
    bw = pd.read_parquet(BFL.BASE_FC).set_index("date")
    base_t = T[(T["model"] == "ols") & (T["predictor_set"] == "baseline")].set_index("date").reindex(bw.index)
    fr = pd.DataFrame({"B": base_t["B"], "rv_actual": base_t["rv_actual"]}, index=bw.index)
    assert np.isfinite(fr["B"].to_numpy()).all()
    for m in ("ols", *PEN):
        fr[f"{m}_base"] = bw[f"{m}|add_rm_fomc"].to_numpy() ** 2 * fr["B"].to_numpy()
    tag = {"baseline": "har", "all_features": "full"}
    for (m, s), g in T.groupby(["model", "predictor_set"]):
        gi = g.set_index("date").reindex(fr.index)
        assert np.array_equal(gi["B"].to_numpy(), fr["B"].to_numpy()), (m, s)
        fr[f"{m}_{tag[s]}"] = gi["plain"].to_numpy()
    assert "ols_full" not in fr, "Table 1 has OLS on the full set: add the other order"
    bl = BFL.base_lin().reindex(fr.index).to_numpy()
    gap = float(np.max(np.abs(bl**2 * fr["B"].to_numpy() / fr["ols_base"].to_numpy() - 1)))
    return fr, gap


def main() -> int:
    fr, gap = forecasts()
    assert gap < 1e-10, gap
    rv = AS.core_rv()
    days = AS.evaluation_days(rv)
    ev = AS.event_days(days)
    cols = ["ols_base", *(f"{m}_{t}" for m in PEN for t in ("base", "full"))]
    idx = AS.scored({c: fr[c] for c in cols}, rv, days)
    assert len(idx) == BFL.N_SCORED, len(idx)
    dday = pd.DatetimeIndex(idx.normalize())
    spans = {"ordinary": idx[~dday.isin(ev)], "all": idx, "event": idx[dday.isin(ev)]}

    rows = []
    for sp, ix in spans.items():
        y = fr["rv_actual"].reindex(ix).to_numpy(float)
        L = {}
        for c in cols:
            r = y / fr[c].reindex(ix).to_numpy(float)
            L[c] = r - np.log(r) - 1.0
        q0 = L["ols_base"].mean()
        for m in PEN:
            for step, what, a, b in STEPS:
                la, lb = L[a.format(m=m)], L[b.format(m=m)]
                d = dm_test(AS.daily(la, ix), AS.daily(lb, ix), hac_lag=BFL.HAC_LAG)
                rows.append(
                    dict(
                        span=sp,
                        days=int(pd.DatetimeIndex(ix.normalize()).nunique()),
                        bars=len(ix),
                        model=m,
                        step=step,
                        compares=what,
                        pct_of_ols_base=100 * (la.mean() - lb.mean()) / q0,
                        pct_own=100 * (la.mean() / lb.mean() - 1),
                        dm=d["dm"],
                        p=d["p"],
                    )
                )
    out = pd.DataFrame(rows)
    out.to_csv(BFL.OUT / "pen_split.csv", index=False)

    md = [
        "# Addendum: penalization vs series in the linear factor test's denominator",
        "",
        "*Requested by the author after seeing the linear factor test's results (eddf391); not pre-registered; "
        "exploratory.*  Existing forecasts only, nothing refitted.  Code: `experiments/beyond_factor_linear_pensplit.py`"
        " (generated by it).",
        "",
        "FULL_LIN vs BASE_LIN (R's denominator) split, for each penalized class m, into **penalization** (m vs OLS on "
        "the same 132 base columns: the ablation's add_rm_fomc arm) and **series** (Table 1's m on the full set vs m on "
        "the base: the six groups ew, vw, liq, iv, sent, vd added within m).  Changes are in percent of the OLS base's "
        "mean QLIKE on the span, so penalization + series = total; % own is against the step's own reference.  DM (HLN, "
        "daily means, lag 8): negative = the first forecast's loss is lower.",
        "",
        f"- ols_base equals the factor test's BASE_LIN: max relative gap {gap:.1e}.",
        "- The other order (series under OLS, then penalization on the full set) is not computed: Table 1 has no OLS "
        "forecasts on the full set.",
    ]
    for sp in spans:
        o = out[out["span"] == sp]
        md += ["", f"## {sp} ({o['days'].iat[0]} days, {o['bars'].iat[0]} bars)", ""]
        md += ["| m | step | compares | % of OLS base | % own | DM | p |", "|---|---|---|---|---|---|---|"]
        for r in o.itertuples():
            md.append(
                f"| {r.model} | {r.step} | {r.compares} | {r.pct_of_ols_base:+.2f} | {r.pct_own:+.2f} | "
                f"{r.dm:+.2f} | {r.p:.3f} |"
            )
    (BFL.OUT / "ADDENDUM_pen_split.md").write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")
    print(out.drop(columns=["days", "bars", "compares"]).to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
