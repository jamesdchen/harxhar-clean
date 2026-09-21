"""Render the decision-aligned scoring of the 15:30 forecasts for the paper.

Reads the CSVs written by writeup/intraday_proposals/59_trade_aligned_loss.py
(results/atm_straddle_0dte_1530/proposals/59/) and writes, never retyping a
number:

  writeup/figures/murphy_1530.pdf           the Murphy diagram (vector)
  writeup/generated/table_murphy_1530.tex   QLIKE beside the score at the threshold
  writeup/generated/murphy_1530_numbers.tex every number the prose cites

Every qualitative claim the prose makes about the diagram (which family leads
in which range of thresholds, where the ridge ranks at the trade's threshold,
which forecast has the highest crossed Sharpe ratio) is asserted here, so a
change in the inputs that would make a sentence false stops the build instead.
"""

from __future__ import annotations

import os
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "results", "atm_straddle_0dte_1530", "proposals", "59")
GEN = os.path.join(ROOT, "writeup", "generated")
FIG = os.path.join(ROOT, "writeup", "figures")

BASELINE = "baseline (HAR + calendar OLS)"
RIDGE = "block-diagonal ridge"
RIDGE_NO_FOMC = "block-diagonal ridge, without the FOMC columns"
LGBM, XGB = "LightGBM", "XGBoost"
LASSO_T, LASSO_F = "lasso (causally tuned)", "lasso (fixed 1e-4)"
ENET = "elastic net (causally tuned)"
ORDER = [BASELINE, RIDGE, RIDGE_NO_FOMC, LGBM, XGB, LASSO_T, LASSO_F, ENET]
TREES = {LGBM, XGB}
PENALISED_LINEAR = {RIDGE, RIDGE_NO_FOMC, LASSO_T, LASSO_F, ENET}

# the paper's row labels (writeup/make_rule_by_strategy_tex.py): a dagger marks
# a column still fitted on the earlier design panel
ROW_TEX = {
    BASELINE: r"baseline (HAR + calendar OLS)",
    RIDGE: r"block-diagonal ridge",
    RIDGE_NO_FOMC: r"block-diagonal ridge, without the FOMC columns",
    LGBM: r"LightGBM$^{\dagger}$",
    XGB: r"XGBoost$^{\dagger}$",
    LASSO_T: r"lasso (causally tuned)$^{\dagger}$",
    LASSO_F: r"lasso ($\alpha=10^{-4}$)",
    ENET: r"elastic net (causally tuned)$^{\dagger}$",
}
LEGEND = {
    BASELINE: "baseline (HAR + calendar OLS)",
    RIDGE: "block-diagonal ridge",
    RIDGE_NO_FOMC: "block-diagonal ridge, without the FOMC columns",
    LGBM: "LightGBM",
    XGB: "XGBoost",
    LASSO_T: "lasso (causally tuned)",
    LASSO_F: r"lasso ($\alpha=10^{-4}$)",
    ENET: "elastic net (causally tuned)",
}
# greyscale-safe: black for the ridge pair and the baseline, two greys for the
# rest, every line a distinct dash pattern or marker
STYLE: dict[str, dict[str, Any]] = {
    BASELINE: {"color": "0.0", "ls": ":", "marker": "s", "lw": 1.3},
    RIDGE: {"color": "0.0", "ls": "-", "marker": None, "lw": 2.0},
    RIDGE_NO_FOMC: {"color": "0.0", "ls": "--", "marker": None, "lw": 1.3},
    LGBM: {"color": "0.45", "ls": "-", "marker": "o", "lw": 1.3},
    XGB: {"color": "0.45", "ls": "--", "marker": "^", "lw": 1.3},
    LASSO_T: {"color": "0.62", "ls": "-.", "marker": None, "lw": 1.3},
    LASSO_F: {"color": "0.62", "ls": "-", "marker": "D", "lw": 1.3},
    ENET: {"color": "0.62", "ls": ":", "marker": "v", "lw": 1.3},
}
MARK_EVERY = 4  # one marker per half-octave on the 2^(k/8) grid
ORDINAL = {
    1: "first",
    2: "second",
    3: "third",
    4: "fourth",
    5: "fifth",
    6: "sixth",
    7: "seventh",
    8: "eighth",
}
WORD = {
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    15: "fifteen",
}
SLICE_TOL = 1e-12


def load() -> dict[str, pd.DataFrame]:
    scores = pd.read_csv(os.path.join(SRC, "a_scores.csv"))
    fc = scores[scores["kind"] == "forecast"].set_index("forecast")
    assert list(fc.index) == ORDER, list(fc.index)
    murphy = pd.read_csv(os.path.join(SRC, "a_murphy.csv"))
    grid = murphy.pivot(index="c", columns="forecast", values="mean_ES")[ORDER]
    assert np.isclose(grid.index.to_numpy(), 1.0).any(), "c = 1 is not on the grid"
    at_one = grid.loc[grid.index[np.isclose(grid.index, 1.0)][0]]
    gap = float((at_one - fc["ES_at_slice"]).abs().max())
    assert gap < SLICE_TOL, gap
    oracle = pd.read_csv(os.path.join(SRC, "a_variance_vs_payoff.csv")).set_index(
        "oracle"
    )
    rho = pd.read_csv(os.path.join(SRC, "a_rank_correlation.csv"))
    return {"fc": fc, "grid": grid, "oracle": oracle, "rho": rho}


def regions(grid: pd.DataFrame) -> dict[str, float | int]:
    """Where each family leads; asserts the sentences the prose builds on them."""
    c = grid.index.to_numpy()
    names = np.asarray(grid.columns)
    vals = grid.to_numpy()
    # the forecasts attaining the minimum, ties exact: far from the slice several
    # forecasts sit on the same side of theta on the same days and tie exactly
    at_min = vals == vals.min(axis=1, keepdims=True)

    def family(i: int) -> str:
        best = set(names[at_min[i]])
        if best <= PENALISED_LINEAR:
            return "penalised linear"
        if best <= TREES:
            return "trees"
        if best == {BASELINE}:
            return "baseline"
        return "tie across families"

    fam = np.array([family(i) for i in range(len(c))])
    lead = np.where(at_min.sum(axis=1) == 1, names[np.argmin(vals, axis=1)], "")
    # the penalised linear forecasts lead at every threshold up to lin_upper
    lin = fam == "penalised linear"
    k = int(np.argmin(lin)) if not lin.all() else len(c)
    assert k > 0 and c[k - 1] >= 0.5, "no penalised-linear lead below half the slice"
    lin_upper = float(c[k - 1])
    # the trees lead on the contiguous run of thresholds that contains c = 1
    tree = fam == "trees"
    one = int(np.flatnonzero(np.isclose(c, 1.0))[0])
    assert tree[one], "the leader at the trade's threshold is not a tree"
    assert k == int(np.flatnonzero(tree)[0]), "a gap between the linear and tree runs"
    lo = one
    while lo > 0 and tree[lo - 1]:
        lo -= 1
    hi = one
    while hi < len(c) - 1 and tree[hi + 1]:
        hi += 1
    # from the top of the grid down: the run of thresholds where families tie
    tie = fam == "tie across families"
    t0 = len(c)
    while t0 > 0 and tie[t0 - 1]:
        t0 -= 1
    assert t0 < len(c), "no tie at the top of the grid"
    # between the tree run and the ties: the baseline leads most thresholds
    span = np.arange(hi + 1, t0)
    base = fam[span] == "baseline"
    assert span.size and 2 * int(base.sum()) > span.size, "the baseline does not lead"
    # the lines cross within a factor of two of the slice
    within = (c >= 0.5) & (c <= 2.0) & (lead != "")
    assert len(set(lead[within])) >= 3, "fewer than three leaders within a factor of 2"
    for i in range(len(c)):
        print(f"  c {c[i]:.3f}  {fam[i]:22s} {lead[i]}")
    return {
        "lin_upper": lin_upper,
        "tree_lower": float(c[lo]),
        "tree_upper": float(c[hi]),
        "base_lower": float(c[span[0]]),
        "base_upper": float(c[span[-1]]),
        "base_lead_n": int(base.sum()),
        "base_region_n": int(span.size),
        "tie_lower": float(c[t0]),
        "leaders_within_two": len(set(lead[within])),
    }


def figure(grid: pd.DataFrame) -> str:
    c = grid.index.to_numpy()
    centred = grid.sub(grid.mean(axis=1), axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.5), sharex=True)
    for name in ORDER:
        st = STYLE[name]
        kw = {
            "color": st["color"],
            "ls": st["ls"],
            "lw": st["lw"],
            "marker": st["marker"],
            "markevery": MARK_EVERY,
            "ms": 3.5,
            "label": LEGEND[name],
        }
        axes[0].plot(c, grid[name].to_numpy(), **kw)
        axes[1].plot(c, centred[name].to_numpy(), **kw)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks([0.25, 0.5, 1.0, 2.0, 4.0])
        ax.set_xticklabels(["1/4", "1/2", "1", "2", "4"])
        ax.set_xlim(c.min(), c.max())
        ax.axvline(1.0, color="0.3", lw=0.8, ls=(0, (1, 2)))
        ax.tick_params(labelsize=8)
        ax.yaxis.label.set_size(8)
    axes[0].set_ylabel("mean elementary score")
    axes[1].set_ylabel("score minus the mean of the eight")
    axes[1].axhline(0.0, color="0.6", lw=0.6)
    axes[0].set_title("(a) level", fontsize=8)
    axes[1].set_title("(b) relative to the eight-forecast mean", fontsize=8)
    axes[0].annotate(
        "the trade's\nthreshold",
        xy=(1.0, 0.0),
        xytext=(1.12, 0.004),
        fontsize=7,
        va="bottom",
        ha="left",
    )
    fig.supxlabel(
        r"threshold $\theta$, in units of the implied variance of the last half hour",
        fontsize=8,
        y=0.3,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=(0, 0.27, 1, 1))
    os.makedirs(FIG, exist_ok=True)
    dst = os.path.join(FIG, "murphy_1530.pdf")
    fig.savefig(dst)
    plt.close(fig)
    print("wrote", dst)
    return dst


def table(fc: pd.DataFrame, oracle: pd.DataFrame) -> list[str]:
    var = oracle.loc["sign(realized variance - slice)"]
    lines = [
        "% AUTO-GENERATED by writeup/make_murphy_1530_tex.py — do not edit.",
        "% Source: results/atm_straddle_0dte_1530/proposals/59/a_scores.csv,",
        "%         a_variance_vs_payoff.csv",
        r"\begingroup\small\setlength{\tabcolsep}{4.5pt}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"& QLIKE & score at $\theta=\mathrm{IV}^2_{30}$ & same side as $RV$ "
        r"& same side as $R$ & Sharpe$_{\mathrm{ann}}$, crossed \\",
        r"\midrule",
    ]
    for name in ORDER:
        r = fc.loc[name]
        lines.append(
            f"{ROW_TEX[name]} & {r['QLIKE']:.3f} & {r['ES_at_slice']:.3f} & "
            f"{100 * r['hit_vs_variance']:.1f} & {100 * r['hit_vs_payoff']:.1f} & "
            f"{r['Sharpe_crossed']:.2f} \\\\"
        )
    lines += [
        r"\midrule",
        f"realized variance known in advance & 0 & 0 & 100.0 & "
        f"{100 * var['agrees_with_sign_R']:.1f} & {var['Sharpe_crossed']:.2f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\endgroup",
        "",
    ]
    return lines


def signed(x: float) -> str:
    return rf"\ensuremath{{{x:.2f}}}" if x >= 0 else rf"\ensuremath{{-{abs(x):.2f}}}"


def macros(data: dict[str, pd.DataFrame], reg: dict[str, float | int]) -> list[str]:
    fc, grid, oracle, rho = data["fc"], data["grid"], data["oracle"], data["rho"]
    var = oracle.loc["sign(realized variance - slice)"]
    pay = oracle.loc["sign(R)"]
    slice_rank = fc["ES_at_slice"].rank(method="min").astype(int)
    qlike_rank = fc["QLIKE"].rank(method="min").astype(int)
    sharpe_rank = fc["Sharpe_crossed"].rank(ascending=False, method="min").astype(int)
    best = fc["ES_at_slice"].idxmin()
    best_sharpe = fc["Sharpe_crossed"].idxmax()
    # the prose names the fixed lasso as the column with the highest crossed Sharpe
    assert best_sharpe == LASSO_F, best_sharpe
    assert best in TREES, best

    def r(set_prefix: str, loss: str) -> pd.Series:
        m = rho["set"].str.startswith(set_prefix) & (rho["loss"] == loss)
        assert int(m.sum()) == 1, (set_prefix, loss)
        return rho[m].iloc[0]

    eight_q, eight_e = r("(i)", "QLIKE"), r("(i)", "ES_at_slice")
    union_q, union_e = r("(iii)", "QLIKE"), r("(iii)", "ES_at_slice")
    n_union = int(union_q["k"])
    n_ladder = n_union - int(eight_q["k"])
    for row in (eight_q, eight_e):
        assert row["ci_lo"] < 0.0 < row["ci_hi"], (
            "an eight-forecast interval excludes 0"
        )
    out = {
        "murNForecasts": WORD[len(ORDER)],
        "murOracleAgree": f"{100 * var['agrees_with_sign_R']:.1f}",
        "murOracleMiss": f"{100 * (1 - var['agrees_with_sign_R']):.1f}",
        "murOracleShMid": f"{var['Sharpe_mid']:.2f}",
        "murOracleShX": f"{var['Sharpe_crossed']:.2f}",
        "murPayoffOracleShX": f"{pay['Sharpe_crossed']:.1f}",
        "murAgreePayMin": f"{100 * fc['hit_vs_payoff'].min():.1f}",
        "murAgreePayMax": f"{100 * fc['hit_vs_payoff'].max():.1f}",
        "murHitVarMin": f"{100 * fc['hit_vs_variance'].min():.1f}",
        "murHitVarMax": f"{100 * fc['hit_vs_variance'].max():.1f}",
        "murBestSlice": LEGEND[best],
        "murBestSliceScore": f"{fc.loc[best, 'ES_at_slice']:.3f}",
        "murRidgeSliceScore": f"{fc.loc[RIDGE, 'ES_at_slice']:.3f}",
        "murRidgeRank": ORDINAL[int(slice_rank[RIDGE])],
        "murRidgeNoFomcRank": ORDINAL[int(slice_rank[RIDGE_NO_FOMC])],
        "murRidgeNoFomcScore": f"{fc.loc[RIDGE_NO_FOMC, 'ES_at_slice']:.3f}",
        "murRidgeQLIKERank": ORDINAL[int(qlike_rank[RIDGE])],
        "murLassoFixedQLIKERank": ORDINAL[int(qlike_rank[LASSO_F])],
        "murLassoFixedSliceRank": ORDINAL[int(slice_rank[LASSO_F])],
        "murLassoFixedShX": f"{fc.loc[LASSO_F, 'Sharpe_crossed']:.2f}",
        "murLassoFixedShRank": ORDINAL[int(sharpe_rank[LASSO_F])],
        "murRhoQLIKE": signed(float(eight_q["spearman_with_crossed_Sharpe"])),
        "murRhoQLIKELo": signed(float(eight_q["ci_lo"])),
        "murRhoQLIKEHi": signed(float(eight_q["ci_hi"])),
        "murRhoES": signed(float(eight_e["spearman_with_crossed_Sharpe"])),
        "murRhoESLo": signed(float(eight_e["ci_lo"])),
        "murRhoESHi": signed(float(eight_e["ci_hi"])),
        "murNLadder": WORD.get(n_ladder, str(n_ladder)),
        "murNUnion": str(n_union),
        "murRhoUnionQLIKE": signed(float(union_q["spearman_with_crossed_Sharpe"])),
        "murRhoUnionES": signed(float(union_e["spearman_with_crossed_Sharpe"])),
        "murLinUpper": f"{reg['lin_upper']:.2f}",
        "murTreeLower": f"{reg['tree_lower']:.2f}",
        "murTreeUpper": f"{reg['tree_upper']:.2f}",
        "murBaseLower": f"{reg['base_lower']:.2f}",
        "murBaseUpper": f"{reg['base_upper']:.2f}",
        "murBaseLeadN": WORD.get(int(reg["base_lead_n"]), str(reg["base_lead_n"])),
        "murBaseRegionN": WORD.get(
            int(reg["base_region_n"]), str(reg["base_region_n"])
        ),
        "murGridMin": f"{grid.index.min():.2f}",
        "murGridMax": f"{grid.index.max():.0f}",
        "murTieLower": f"{reg['tie_lower']:.2f}",
    }
    lines = [
        "% GENERATED FILE -- do not edit by hand.",
        "% Written by writeup/make_murphy_1530_tex.py from",
        "% results/atm_straddle_0dte_1530/proposals/59/*.csv.",
    ]
    lines += [rf"\newcommand{{\{k}}}{{{v}}}" for k, v in sorted(out.items())]
    lines.append("")
    return lines


def write(name: str, lines: list[str]) -> None:
    dst = os.path.join(GEN, name)
    os.makedirs(GEN, exist_ok=True)
    with open(dst, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines))
    print("wrote", dst)


def main() -> None:
    data = load()
    reg = regions(data["grid"])
    figure(data["grid"])
    write("table_murphy_1530.tex", table(data["fc"], data["oracle"]))
    write("murphy_1530_numbers.tex", macros(data, reg))


if __name__ == "__main__":
    main()
