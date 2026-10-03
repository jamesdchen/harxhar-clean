"""Build Appendix D.19 -- the 16:00 models re-run (design, mask, cadence, tuning).

The 16:00 campaign (writeup/CAMPAIGN_16H_2026-09-29.md) re-ran every per-bar 16:00 model
one change at a time: the de-duplicated design, the per-window mask for trees and the LSTM,
the refit cadence, the tuning (random search, Optuna), an intraday-sequence LSTM, the
CPU-class evidence, the master table's before / after, and the importance / density
re-runs.  This script turns the campaign's executed outputs into

  writeup/generated/appendix_campaign_16h_<table>.tex   one booktabs tabular per table
  writeup/generated/appendix_campaign_16h_numbers.tex   \\newcommand macros (\\camp...)
                                                        for every number quoted in prose

and nothing else.  Every number is read from a committed CSV / JSON of the campaign or from
a summary its own script generated; for the compute table only, from an agent's report (a
commit message or the checklist's agent log).  Each generated file names its sources in a
comment.  Gates assert that two outputs of the same quantity agree before either is used.

Conventions (the campaign's): the research scorer (the 16:00 bar recalibrated on its own),
the same 866 trade days, 95 % paired day-block bootstrap intervals; a difference is a - b.
QLIKE differences are shown in % of b's QLIKE with the interval in the same units; Sharpe
differences are sign(s) Sharpe at the midpoint fill.

Run:  python writeup/make_appendix_campaign_16h_tex.py
"""

from __future__ import annotations

import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "writeup" / "generated"
RES = ROOT / "results"
PREFIX = "appendix_campaign_16h"

A = RES / "linear_subsection_dedup"  # agent A: linear arms + LSTM on the de-dup design
B = RES / "trees_cadence_1600"  # agent B: trees unmasked (T10/T1/RS10/RS1)
H = RES / "trees_mask_1600"  # agent H: the masked ladder
HCLS = (
    RES / "linear_subsection_trees_mask" / "class"
)  # agent H: CPU-class census / gate
C = (
    RES / "linear_subsection_trees_optuna_mask"
)  # agent C: Optuna (single class, canonical)
CR = C / "report"
D = (
    RES / "linear_subsection_trees_optuna" / "crosscluster"
)  # agent D: cross-cluster check
E = (
    RES / "linear_subsection_restofday_dedup"
)  # agent E: rest-of-day arms, de-dup design
F = RES / "feature_importance_1530_dedup"  # agent F: C1 importance re-run
G = RES / "dense_vs_sparse_dedup"  # agent G: C2 dense-vs-sparse re-run
LI = RES / "linear_subsection_lstm_intraday"  # agent I: intraday-sequence LSTM
J = RES / "close_master_table"  # agent J: master table before / after
PROGRESS = ROOT / "writeup" / "PROGRESS_2026-09-29.md"
F_COMMIT = "4fa0697"  # agent F's commit; its message carries F's cluster accounting

BUCKETS = ["baseline", "all_features"]  # live_feasible commented out
TREES = ["lgbm", "xgb", "rf"]
MODEL = {"lgbm": "LightGBM", "xgb": "XGBoost", "rf": "random forest", "lstm": "LSTM"}
# The measures F's summary tables rank (top series per measure); perm P1 MSE is not among them.
FI_MEASURES = ["mdi", "split", "perm_p1_qlike", "perm_p2_qlike", "perm_p2_mse", "shap"]

NUMS: dict[str, str] = {}
WRITTEN: list[str] = []


# --------------------------------------------------------------------------- formatting
def fnum(x: float, d: int = 2, plus: bool = False) -> str:
    """A number for LaTeX text or math: true minus sign, optional explicit plus."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "---"
    s = f"{abs(float(x)):.{d}f}"
    if float(x) < 0 and float(s) != 0.0:
        return r"\ensuremath{-}" + s
    if plus and float(s) != 0.0:
        return r"\ensuremath{+}" + s
    return s


def fci(lo: float, hi: float, d: int = 2) -> str:
    return f"[{fnum(lo, d, True)}, {fnum(hi, d, True)}]"


def fint(x: float) -> str:
    return f"{int(round(float(x))):,}".replace(",", "{,}")


def fsci(x: float, d: int = 1) -> str:
    """Scientific notation, d decimals in the mantissa; 0 stays 0."""
    x = float(x)
    if x == 0.0:
        return "0"
    m, e = f"{x:.{d}e}".split("e")
    return r"\ensuremath{" + m + r"\times10^{" + str(int(e)) + "}}"


def pct(x: float, d: int = 1, plus: bool = False) -> str:
    return fnum(100.0 * float(x), d, plus)


def macro(name: str, value: str) -> None:
    assert re.fullmatch(r"camp[A-Z][A-Za-z]*", name), name
    assert name not in NUMS, f"macro defined twice: {name}"
    NUMS[name] = value


def tt(s: str) -> str:
    return r"\texttt{" + s.replace("_", r"\_") + "}"


def tex_escape(s: str) -> str:
    s = s.replace("{", r"\{").replace("}", r"\}")
    s = (
        s.replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )
    return s.replace("->", r"$\to$")


def stack(cell: str) -> str:
    """A cell whose ``~~`` marks line breaks, as a shortstack."""
    if "~~" not in cell:
        return cell
    parts = [("{}" + x) if x.startswith("[") else x for x in cell.split("~~")]
    return r"\shortstack{" + r"\\".join(parts) + "}"


def row(cells: list[str]) -> str:
    return " & ".join(cells) + r" \\"


def panel(title: str, ncol: int) -> str:
    return r"\multicolumn{" + str(ncol) + r"}{l}{\emph{" + title + r"}} \\"


def write_tab(
    name: str, sources: list[str], colspec: str, header: list[str], body: list[str]
) -> None:
    """One booktabs tabular; ``body`` lines are complete rows or rules."""
    lines = [
        "% AUTO-GENERATED by writeup/make_appendix_campaign_16h_tex.py -- do not edit.",
        *[f"% Source: {s}" for s in sources],
        r"\begin{tabular}{" + colspec + "}",
        r"\toprule",
        " & ".join(stack(h) for h in header) + r" \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
    ]
    (GEN / f"{PREFIX}_{name}.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )
    WRITTEN.append(name)


def qcell(
    q_pct: float, lo_pct: float, hi_pct: float, s: float, slo: float, shi: float
) -> str:
    """Two-line cell: QLIKE % [interval in %] over Delta Sharpe mid [interval]."""
    top = f"{fnum(q_pct, 1, True)} {fci(lo_pct, hi_pct, 1)}"
    bot = f"{fnum(s, 2, True)} {fci(slo, shi, 2)}"
    return r"\shortstack{" + top + r"\\" + bot + "}"


def counts(df: pd.DataFrame, qlo: str, qhi: str, slo: str, shi: str) -> dict[str, int]:
    """Intervals wholly below / above zero, for QLIKE and Sharpe mid."""
    return {
        "qb": int((df[qhi] < 0).sum()),
        "qa": int((df[qlo] > 0).sum()),
        "sb": int((df[shi] < 0).sum()),
        "sa": int((df[slo] > 0).sum()),
        "n": len(df),
    }


def count_macros(stem: str, c: dict[str, int]) -> None:
    macro(f"camp{stem}N", fint(c["n"]))
    macro(f"camp{stem}QBelow", fint(c["qb"]))
    macro(f"camp{stem}QAbove", fint(c["qa"]))
    macro(f"camp{stem}SBelow", fint(c["sb"]))
    macro(f"camp{stem}SAbove", fint(c["sa"]))


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


# =========================================================================== 1 design
def design() -> None:
    ch = pd.read_csv(A / "change_by_arm.csv")
    arms = pd.read_csv(A / "arm_list.csv")
    gates = pd.read_csv(A / "gates.csv")
    lin = ch[ch.estimator.isin(["ridge", "ols", "reclasso", "reclasticnet"])].copy()
    assert len(lin) == len(arms), (len(lin), len(arms))
    tol = gates[gates.gate.str.startswith("(c) lasso / OLS")].bound.unique()
    assert len(tol) == 1
    tol = float(
        tol[0]
    )  # A's gate: a forecast moved when its max rel change exceeds this

    def rng(s: str) -> tuple[int, int]:
        lo, hi = s.split("-")
        return int(lo), int(hi)

    # the production per-bar rows (HAR ladder of six rungs)
    body = []
    nse = set()
    for b in BUCKETS:
        r = lin[lin.forecast == f"sub_ridge_{b}"].iloc[0]
        old, new = rng(r.masked_per_tune_old), rng(r.masked_per_tune_new)
        kept_lo, kept_hi = int(r.p_new) - new[1], int(r.p_new) - new[0]
        assert (int(r.p_old) - old[1], int(r.p_old) - old[0]) == (kept_lo, kept_hi)
        nse.add(int(r.p_old - r.p_new))
        kept = fint(kept_lo) if kept_lo == kept_hi else f"{kept_lo}--{kept_hi}"
        mo = f"{old[0]}" if old[0] == old[1] else f"{old[0]}--{old[1]}"
        mn = f"{new[0]}" if new[0] == new[1] else f"{new[0]}--{new[1]}"
        body.append(
            row(
                [
                    tt(b),
                    fint(r.p_old),
                    fint(r.p_new),
                    fint(r.p_old - r.p_new),
                    mo,
                    mn,
                    kept,
                ]
            )
        )
        short = {"baseline": "Base", "live_feasible": "Live", "all_features": "All"}[b]
        macro(f"campPOld{short}", fint(r.p_old))
        macro(f"campPNew{short}", fint(r.p_new))
        macro(f"campKeptLin{short}", kept.replace("--", "--"))
    assert len(nse) == 1
    n_se = nse.pop()
    macro("campNSessionEdge", fint(n_se))
    macro("campNSessionEdgeHalf", fint(n_se // 2))
    write_tab(
        "design",
        [rel(A / "change_by_arm.csv") + " (rows sub_ridge_<input set>)"],
        "lrrrrrr",
        [
            "input set",
            "columns~~before",
            "columns~~after",
            "removed",
            "masked per~~tune, before",
            "masked per~~tune, after",
            "kept per~~tune",
        ],
        body,
    )

    # the identifiability mask already removed them: masked-per-tune falls by p_old - p_new
    m = lin[lin.masked_per_tune_old.notna()].copy()
    ok = 0
    for _, r in m.iterrows():
        o, n = rng(r.masked_per_tune_old), rng(r.masked_per_tune_new)
        drop = int(r.p_old - r.p_new)
        ok += int(o[0] - n[0] == drop and o[1] - n[1] == drop)
    macro("campNLinear", fint(len(lin)))
    macro("campNLinearMasked", fint(len(m)))
    macro("campMaskDropAgree", fint(ok))

    # per-estimator change of the 16:00 forecast
    body = []
    for est, label, stem in [
        ("ols", "OLS", "OLS"),
        ("ridge", "ridge", "Ridge"),
        ("reclasso", "lasso", "Lasso"),
        ("reclasticnet", "elastic net", "Enet"),
    ]:
        s = lin[lin.estimator == est]
        mx = s.pred_adj_1600_max_rel_change.max()
        med = s.pred_adj_1600_median_rel_change.median()
        moved = int((s.pred_adj_1600_max_rel_change > tol).sum())
        pos = int(s.positions_changed.sum())
        dsh = float((s.Sharpe_mid_new - s.Sharpe_mid_old).abs().max())
        body.append(
            row(
                [
                    label,
                    fint(len(s)),
                    fsci(mx),
                    fsci(med),
                    fint(moved),
                    fint(pos),
                    fnum(dsh, 2),
                ]
            )
        )
        macro(f"camp{stem}N", fint(len(s)))
        macro(f"camp{stem}MaxRel", fsci(mx, 0))
        macro(f"camp{stem}Moved", fint(moved))
    macro("campMovedTol", fsci(tol, 0))
    macro("campPosChanged", fint(lin.positions_changed.sum()))
    days = lin.trade_days.unique()
    assert len(days) == 1
    macro("campTradeDays", fint(days[0]))
    macro(
        "campMaxDSharpeLin",
        fnum((lin.Sharpe_mid_new - lin.Sharpe_mid_old).abs().max(), 2),
    )
    write_tab(
        "linear_change",
        [rel(A / "change_by_arm.csv"), rel(A / "gates.csv") + " (the 'moved' bound)"],
        "lrrrrrr",
        [
            "estimator",
            "forecasts",
            "max rel.~~change",
            "median rel.~~change",
            f"moved~~(> {fsci(tol, 0)})",
            "positions~~changed",
            r"max $|\Delta$Sharpe$|$~~mid",
        ],
        body,
    )

    # control attribution (same-architecture re-runs)
    sa = pd.read_csv(A / "samearch_attribution.csv")
    macro("campAttrN", fint(len(sa)))
    macro("campAttrRepeat", fint((sa.control_rep_vs_control == 0).sum()))
    dz = sa[sa.samearch_vs_control > tol].samearch_vs_control
    mz = sa[sa.control_rep_vs_old > tol].control_rep_vs_old
    macro("campAttrDesignN", fint(len(dz)))
    macro("campAttrDesignLo", fsci(dz.min()))
    macro("campAttrDesignHi", fsci(dz.max()))
    macro("campAttrMachineN", fint(len(mz)))
    macro("campAttrMachineLo", fsci(mz.min()))
    macro("campAttrMachineHi", fsci(mz.max()))

    # the LSTM on the de-dup design, no mask (A): median relative change at 16:00, MSE rule
    ls = ch[ch.estimator == "lstm (mse rule)"]
    macro("campLSTMDedupLo", pct(ls.pred_adj_1600_median_rel_change.min()))
    macro("campLSTMDedupHi", pct(ls.pred_adj_1600_median_rel_change.max()))

    # the rest-of-day check rows (E)
    ck = pd.read_csv(E / "check_1530.csv")
    bef, aft = ck[ck.phase == "before"], ck[ck.phase == "after"]
    assert (aft.same_position == aft.deck_days).all()
    macro("campRodChecks", fint(len(aft)))
    macro("campRodBeforeMax", fsci(bef.max_rel_pred_clock.max()))
    macro("campRodBeforeSameMin", fint(bef.same_position.min()))
    macro("campRodDeck", fint(bef.deck_days.iloc[0]))
    macro("campRodAfterBit", fint((aft.rows_16h_not_bit_identical == 0).sum()))
    macro("campRodRows", fint(aft.rows_16h_common.iloc[0]))
    eg = pd.read_csv(E / "gates.csv")
    mg = eg[eg.gate.str.startswith("(a) masked cols per tune, old minus new")]
    mx = mg.note.str.extract(r"max: (\d+)")[0].astype(int)
    macro("campRodArms", fint(len(mg)))
    macro("campRodMaskSame", fint(((mg.value == n_se) & (mx == n_se)).sum()))


# =========================================================================== 2 mask
def _pairs() -> pd.DataFrame:
    mp = pd.read_csv(H / "mask_pairs.csv")
    for c in ("a", "b"):
        mp[c] = mp[c].astype(str).str.replace("\n", " ", regex=False)
    mp["q_lo"] = 100.0 * mp.qlike_ci_lo / mp.qlike_b
    mp["q_hi"] = 100.0 * mp.qlike_ci_hi / mp.qlike_b
    return mp


def _cell(r: pd.Series) -> str:
    return qcell(
        r.qlike_pct,
        r.q_lo,
        r.q_hi,
        r.sharpe_mid_diff,
        r.sharpe_mid_ci_lo,
        r.sharpe_mid_ci_hi,
    )


def mask() -> None:
    kc = pd.read_csv(H / "kept_counts.csv")
    kcc = pd.read_csv(CR / "kept_columns.csv")
    for b, short in zip(BUCKETS, ["Base", "Live", "All"]):
        s = kc[kc.bucket == b]
        for c in ("kept_n_median", "kept_n_min", "kept_n_max"):
            assert s[c].nunique() == 1, (
                b,
                c,
            )  # every rung and model keeps the same columns
        med, lo, hi = (
            int(s.kept_n_median.iloc[0]),
            int(s.kept_n_min.iloc[0]),
            int(s.kept_n_max.iloc[0]),
        )
        s2 = kcc[kcc.bucket == b]
        assert (
            (s2.stage2_median == med).all()
            and (s2.stage2_min == lo).all()
            and (s2.stage2_max == hi).all()
        )
        macro(f"campKept{short}", fint(med))
        macro(f"campKept{short}Lo", fint(lo))
        macro(f"campKept{short}Hi", fint(hi))
        macro(f"campP{short}", fint(s.p.iloc[0]))
    macro("campKeptRefits", fint(kc[kc.rung == "T1"].refits.iloc[0]))

    mp = _pairs()
    mu = mp[mp.family == "masked - unmasked"].copy()
    mu["rung"] = mu.a.str.split().str[0]
    cols = ["T10", "T1", "RS10", "RS1", "LSTM10"]
    body = []
    for b in BUCKETS:
        body.append(panel(tt(b), 2 + len(cols)))
        for m in TREES + ["lstm"]:
            cells = []
            for rung in cols:
                s = mu[(mu.model == m) & (mu.bucket == b) & (mu.rung == rung)]
                cells.append(_cell(s.iloc[0]) if len(s) else "")
            if any(cells):
                body.append(row(["", MODEL[m], *cells]))
    write_tab(
        "mask",
        [rel(H / "mask_pairs.csv") + " (family 'masked - unmasked')"],
        "ll" + "c" * len(cols),
        ["", "model", *[f"{c}~~masked $-$ unmasked" for c in cols]],
        body,
    )
    tr = mu[mu.model.isin(TREES)]
    count_macros(
        "MaskTree", counts(tr, "q_lo", "q_hi", "sharpe_mid_ci_lo", "sharpe_mid_ci_hi")
    )
    macro("campMaskMedAbsQ", fnum(tr.qlike_pct.abs().median(), 1))
    lm = mu[mu.model == "lstm"]
    count_macros(
        "MaskLSTM", counts(lm, "q_lo", "q_hi", "sharpe_mid_ci_lo", "sharpe_mid_ci_hi")
    )
    la = lm[lm.bucket == "all_features"].iloc[0]
    macro("campMaskLSTMAllQ", fnum(la.qlike_pct, 1, True))
    macro("campMaskLSTMAllQCI", fci(la.q_lo, la.q_hi, 1))


# =========================================================================== 3 cadence + tuning (trees)
def cadence() -> None:
    mp = _pairs()
    ml = mp[mp.family == "masked ladder"].copy()
    comps = [
        ("T1 masked", "T10 masked", "TOneTen", "T1 $-$ T10~~(refit 1 vs 10)"),
        ("RS1 masked", "RS10 masked", "RSOneTen", "RS1 $-$ RS10~~(refit 1 vs 10)"),
        ("RS1 masked", "T1 masked", "RSVsT", "RS1 $-$ T1~~(search vs shipped)"),
    ]
    body = []
    for b in BUCKETS:
        body.append(panel(tt(b), 2 + len(comps)))
        for m in TREES:
            cells = []
            for a, bb, _, _ in comps:
                s = ml[(ml.model == m) & (ml.bucket == b) & (ml.a == a) & (ml.b == bb)]
                assert len(s) == 1, (m, b, a, bb)
                cells.append(_cell(s.iloc[0]))
            body.append(row(["", MODEL[m], *cells]))
        s = ml[
            (ml.model == "lstm")
            & (ml.bucket == b)
            & (ml.a == "LSTM masked")
            & (ml.b == "LSTM10 masked")
        ]
        assert len(s) == 1
        body.append(row(["", r"LSTM (1 vs 10)", _cell(s.iloc[0]), "", ""]))
    write_tab(
        "cadence",
        [rel(H / "mask_pairs.csv") + " (family 'masked ladder')"],
        "ll" + "c" * len(comps),
        ["", "model", *[c[3] for c in comps]],
        body,
    )
    for a, bb, stem, _ in comps:
        s = ml[(ml.a == a) & (ml.b == bb) & ml.model.isin(TREES)]
        count_macros(
            stem, counts(s, "q_lo", "q_hi", "sharpe_mid_ci_lo", "sharpe_mid_ci_hi")
        )
        macro(f"camp{stem}QLo", fnum(s.qlike_pct.min(), 1, True))
        macro(f"camp{stem}QHi", fnum(s.qlike_pct.max(), 1, True))
    s = ml[(ml.a == "LSTM masked") & (ml.b == "LSTM10 masked")]
    count_macros(
        "LSTMCad", counts(s, "q_lo", "q_hi", "sharpe_mid_ci_lo", "sharpe_mid_ci_hi")
    )
    for b, short in zip(BUCKETS, ["Base", "Live", "All"]):
        r = s[s.bucket == b].iloc[0]
        macro(f"campLSTMCad{short}", fnum(r.qlike_pct, 1, True))
        macro(f"campLSTMCad{short}CI", fci(r.q_lo, r.q_hi, 1))

    # the masked rungs against the per-bar ridge
    mr = mp[mp.family == "masked - ridge"].copy()
    tr = mr[mr.model.isin(TREES)]
    count_macros(
        "RidgeTree", counts(tr, "q_lo", "q_hi", "sharpe_mid_ci_lo", "sharpe_mid_ci_hi")
    )
    best = tr.loc[tr.qlike_pct.idxmin()]
    macro(
        "campRidgeBestTree",
        best.a.split()[0] + " " + MODEL[best.model] + " " + tt(best.bucket),
    )
    macro("campRidgeBestTreeQ", fnum(best.qlike_pct, 1, True))
    macro("campRidgeBestTreeQCI", fci(best.q_lo, best.q_hi, 1))
    lr_ = mr[(mr.model == "lstm") & (mr.a == "LSTM masked")]
    count_macros(
        "RidgeLSTM", counts(lr_, "q_lo", "q_hi", "sharpe_mid_ci_lo", "sharpe_mid_ci_hi")
    )
    for b, short in zip(BUCKETS, ["Base", "Live", "All"]):
        r = lr_[lr_.bucket == b].iloc[0]
        macro(f"campRidgeLSTM{short}", fnum(r.qlike_pct, 1, True))


# =========================================================================== 4 Optuna
def _arm_label(m: str, b: str) -> str:
    return f"{MODEL[m]} {tt(b)}"


def _opt_cell(r: pd.Series) -> str:
    qb = r.qlike_a - r.qlike_diff_a_minus_b
    return qcell(
        r.qlike_pct_a_vs_b,
        100.0 * r.qlike_diff_ci_lo / qb,
        100.0 * r.qlike_diff_ci_hi / qb,
        r.dSharpe_mid,
        r.dSharpe_mid_lo,
        r.dSharpe_mid_hi,
    )


def optuna() -> None:
    bp = pd.read_csv(CR / "budget_points.csv")
    assert bp.share_best_in_41_50_if_exchangeable.nunique() == 1
    trials = bp.trials.unique()
    assert len(trials) == 1
    macro("campOptTrials", fint(trials[0]))
    macro("campOptPoints", fint(bp.points.max()))
    assert bp.points.nunique() == 1
    macro("campOptExch", pct(bp.share_best_in_41_50_if_exchangeable.iloc[0], 0))
    macro("campOptLateLo", pct(bp.share_best_in_41_50.min(), 0))
    macro("campOptLateHi", pct(bp.share_best_in_41_50.max(), 0))
    macro("campOptGainTwentyFiveLo", pct(bp.rel_impr_25_to_50_mean.min(), 1))
    macro("campOptGainTwentyFiveHi", pct(bp.rel_impr_25_to_50_mean.max(), 1))
    macro("campOptGainTwentyFiveMedLo", pct(bp.rel_impr_25_to_50_median.min(), 1))
    macro("campOptGainTwentyFiveMedHi", pct(bp.rel_impr_25_to_50_median.max(), 1))
    macro("campOptGainFortyLo", pct(bp.rel_impr_40_to_50_mean.min(), 1))
    macro("campOptGainFortyHi", pct(bp.rel_impr_40_to_50_mean.max(), 1))
    macro("campOptGainFortyMedMax", pct(bp.rel_impr_40_to_50_median.max(), 1))
    macro("campOptPickDiffLo", pct(bp.share_pick_k25_differs_from_k50.min(), 0))
    macro("campOptPickDiffHi", pct(bp.share_pick_k25_differs_from_k50.max(), 0))
    macro("campOptShippedHi", pct(bp.share_best_is_shipped.max(), 1))
    body = []
    for m in TREES:
        for b in BUCKETS:
            r = bp[(bp.model == m) & (bp.bucket == b)].iloc[0]
            body.append(
                row(
                    [
                        _arm_label(m, b),
                        fint(r.best_trial_median),
                        pct(r.share_best_is_shipped, 1),
                        pct(r.share_best_in_41_50, 1),
                        f"{pct(r.rel_impr_10_to_50_mean, 2)} / {pct(r.rel_impr_10_to_50_median, 2)}",
                        f"{pct(r.rel_impr_25_to_50_mean, 2)} / {pct(r.rel_impr_25_to_50_median, 2)}",
                        f"{pct(r.rel_impr_40_to_50_mean, 2)} / {pct(r.rel_impr_40_to_50_median, 2)}",
                        pct(r.share_pick_k25_differs_from_k50, 0),
                    ]
                )
            )
    write_tab(
        "budget",
        [rel(CR / "budget_points.csv")],
        "lrrrccc r".replace(" ", ""),
        [
            "arm",
            "best trial~~(median)",
            r"best =~~shipped (\%)",
            r"best in~~41--50 (\%)",
            r"val.~MSE gain~~10$\to$50 (\%)",
            r"val.~MSE gain~~25$\to$50 (\%)",
            r"val.~MSE gain~~40$\to$50 (\%)",
            r"pick of 25~~$\neq$ of 50 (\%)",
        ],
        body,
    )

    # out of sample: best of 50 against best of 10 / 25
    bo = pd.read_csv(CR / "budget_oos.csv")
    comps = [
        ("tp1", "tp1_k10", "TUNE\\_PER 1:~~50 vs 10"),
        ("tp1", "tp1_k25", "TUNE\\_PER 1:~~50 vs 25"),
        ("tp25", "tp25_k10", "TUNE\\_PER 25:~~50 vs 10"),
        ("tp25", "tp25_k25", "TUNE\\_PER 25:~~50 vs 25"),
    ]
    body = []
    for m in TREES:
        for b in BUCKETS:
            cells = []
            for a, bb, _ in comps:
                s = bo[
                    (bo.model == m)
                    & (bo.bucket == b)
                    & (bo.a_kind == a)
                    & (bo.b_kind == bb)
                ]
                assert len(s) == 1
                cells.append(_opt_cell(s.iloc[0]))
            body.append(row([_arm_label(m, b), *cells]))
    write_tab(
        "budget_oos",
        [rel(CR / "budget_oos.csv")],
        "l" + "c" * len(comps),
        ["arm", *[c[2] for c in comps]],
        body,
    )
    count_macros(
        "Bud",
        counts(
            bo,
            "qlike_diff_ci_lo",
            "qlike_diff_ci_hi",
            "dSharpe_mid_lo",
            "dSharpe_mid_hi",
        ),
    )

    # TUNE_PER ablation (every path refit every session) against TUNE_PER 250
    tp = pd.read_csv(CR / "tuneper_oos.csv")
    tv = tp[tp.comparison == "tune_per vs 250"]
    tp_comps = [
        ("tp1", "TUNE\\_PER 1~~vs 250"),
        ("tp5", "TUNE\\_PER 5~~vs 250"),
        ("tp25", "TUNE\\_PER 25~~vs 250"),
    ]
    body = []
    for m in TREES:
        for b in BUCKETS:
            cells = []
            for a, _ in tp_comps:
                s = tv[(tv.model == m) & (tv.bucket == b) & (tv.a_kind == a)]
                assert len(s) == 1
                cells.append(_opt_cell(s.iloc[0]))
            ref = tp[(tp.model == m) & (tp.bucket == b) & (tp.b_kind == "tp250")].iloc[
                0
            ]
            q250 = ref.qlike_a - ref.qlike_diff_a_minus_b
            body.append(row([_arm_label(m, b), fnum(q250, 4), *cells]))
    write_tab(
        "tuneper",
        [rel(CR / "tuneper_oos.csv") + " (comparison 'tune_per vs 250')"],
        "lr" + "c" * len(tp_comps),
        ["arm", "QLIKE~~TUNE\\_PER 250", *[c[1] for c in tp_comps]],
        body,
    )
    count_macros(
        "TP",
        counts(
            tv,
            "qlike_diff_ci_lo",
            "qlike_diff_ci_hi",
            "dSharpe_mid_lo",
            "dSharpe_mid_hi",
        ),
    )
    below = tv[tv.qlike_diff_ci_hi < 0]
    macro(
        "campTPBelowWho",
        "; ".join(
            f"{MODEL[m]} {tt(b)} (TUNE\\_PER {', '.join(g.a_kind.str[2:].tolist())})"
            for (m, b), g in below.groupby(["model", "bucket"], sort=False)
        ),
    )
    qr = tp[tp.comparison == "tune_per rule qlike vs mse"]
    count_macros(
        "TPRule",
        counts(
            qr,
            "qlike_diff_ci_lo",
            "qlike_diff_ci_hi",
            "dSharpe_mid_lo",
            "dSharpe_mid_hi",
        ),
    )

    # Optuna against the shipped trees refit daily (T1), random search (RS1) and the ridge
    body = []
    for comp, label, stem in [
        ("vs T1", "T1", "OptT"),
        ("vs RS1", "RS1", "OptRS"),
        ("vs ridge", "per-bar ridge", "OptRidge"),
    ]:
        s = tp[tp.comparison == comp]
        c = counts(
            s,
            "qlike_diff_ci_lo",
            "qlike_diff_ci_hi",
            "dSharpe_mid_lo",
            "dSharpe_mid_hi",
        )
        count_macros(stem, c)
        for k in ["tp1", "tp5", "tp25", "tp250"]:
            ck = counts(
                s[s.a_kind == k],
                "qlike_diff_ci_lo",
                "qlike_diff_ci_hi",
                "dSharpe_mid_lo",
                "dSharpe_mid_hi",
            )
            body.append(
                row(
                    [
                        label,
                        k[2:],
                        fint(ck["n"]),
                        fint(ck["qb"]),
                        fint(ck["qa"]),
                        fint(ck["sb"]),
                        fint(ck["sa"]),
                    ]
                )
            )
        body.append(
            row(
                [
                    r"\emph{all}",
                    "",
                    fint(c["n"]),
                    fint(c["qb"]),
                    fint(c["qa"]),
                    fint(c["sb"]),
                    fint(c["sa"]),
                ]
            )
        )
        if comp != "vs ridge":
            body.append(r"\midrule")
    write_tab(
        "optuna_vs",
        [rel(CR / "tuneper_oos.csv") + " (comparisons 'vs T1', 'vs RS1', 'vs ridge')"],
        "llrrrrr",
        [
            "Optuna $-$",
            "TUNE\\_PER",
            "arms",
            "QLIKE~~below 0",
            "QLIKE~~above 0",
            "Sharpe mid~~below 0",
            "Sharpe mid~~above 0",
        ],
        body,
    )
    rs_below = tp[(tp.comparison == "vs RS1") & (tp.qlike_diff_ci_hi < 0)]
    macro(
        "campOptRSBelowWho",
        "; ".join(
            f"{MODEL[m]} {tt(b)} (TUNE\\_PER {', '.join(g.a_kind.str[2:].tolist())})"
            for (m, b), g in rs_below.groupby(["model", "bucket"], sort=False)
        ),
    )

    # where the picks sit
    pb = pd.read_csv(CR / "picks_bounds.csv")
    body = []
    for m in TREES:
        body.append(panel(MODEL[m], 7))
        for ax in pb[pb.model == m].axis.unique():
            s = pb[(pb.model == m) & (pb.axis == ax)].set_index("bucket").loc[BUCKETS]

            def trio(col: str, s: pd.DataFrame = s) -> str:
                v = s[col]
                return " / ".join("---" if pd.isna(x) else pct(x, 1) for x in v)

            def prior(col: str, s: pd.DataFrame = s) -> str:
                v = sorted({pct(x, 1) for x in s[col] if pd.notna(x)})
                return "---" if not v else (v[0] if len(v) == 1 else f"{v[0]}--{v[-1]}")

            meds = []
            for x, shares in zip(s.pick_median, s.pick_shares):
                if pd.notna(x):
                    meds.append(f"{float(x):.3g}")
                    continue
                # a categorical axis: the median of the picks in the space's own order
                cum = 0.0
                for k, v in json.loads(shares).items():
                    cum += float(v)
                    if cum >= 0.5:
                        meds.append(tex_escape(k))
                        break
            pseudo = "(pseudo-axis)" in ax
            name = tt(ax.replace(" (pseudo-axis)", "")) + (
                r"$^{\dagger}$" if pseudo else ""
            )
            body.append(
                row(
                    [
                        name,
                        tex_escape(str(s.space.iloc[0])).replace("<=", r"$\le$"),
                        trio("share_picks_at_low"),
                        trio("share_picks_at_high"),
                        prior("share_startup_at_low"),
                        prior("share_startup_at_high"),
                        " / ".join(meds),
                    ]
                )
            )
    write_tab(
        "picks",
        [rel(CR / "picks_bounds.csv")],
        "llccccc",
        [
            "axis",
            "search space",
            r"picks at low bound (\%)~~base / live / all",
            r"picks at high bound (\%)~~base / live / all",
            r"prior~~low (\%)",
            r"prior~~high (\%)",
            "median pick~~base / live / all",
        ],
        body,
    )
    lg = pb[(pb.model == "lgbm") & (pb.axis == "min_child_samples")]
    macro("campPickLgbmMcsLo", pct(lg.share_picks_at_low.min(), 0))
    macro("campPickLgbmMcsHi", pct(lg.share_picks_at_low.max(), 0))
    rf = pb[(pb.model == "rf") & (pb.axis == "max_depth")]
    macro("campPickRfDepthLo", pct(rf.share_picks_at_high.min(), 0))
    macro("campPickRfDepthHi", pct(rf.share_picks_at_high.max(), 0))
    macro("campPickRfDepthPrior", pct(rf.share_startup_at_high.iloc[0], 0))
    l2 = pb[pb.axis.isin(["lambda_l2", "reg_lambda"])]
    macro("campPickLambdaHighMax", pct(l2.share_picks_at_high.max(), 1))
    macro("campPickLambdaPriorLo", pct(l2.share_startup_at_high.min(), 1))


# =========================================================================== 5 intraday LSTM
def lstmi() -> None:
    sc = LI / "score"
    lv = pd.read_csv(sc / "lstmi_levels.csv")
    pr = pd.read_csv(sc / "lstmi_pairs.csv")
    pr["q_lo"] = 100.0 * pr.qlike_ci_lo / pr.qlike_b
    pr["q_hi"] = 100.0 * pr.qlike_ci_hi / pr.qlike_b
    body = []
    for b in BUCKETS:
        L = lv[lv.bucket == b].set_index("forecast")
        cells = [tt(b)]
        for f in ["intraday", "perbar_lstm", "ridge"]:
            cells.append(fnum(L.loc[f, "qlike"], 4))
        for f in ["intraday", "perbar_lstm", "ridge"]:
            cells.append(fnum(L.loc[f, "sharpe_mid"], 2))
        for other in ["perbar_lstm", "ridge"]:
            s = pr[(pr.bucket == b) & (pr.a == "intraday") & (pr.b == other)]
            assert len(s) == 1
            r = s.iloc[0]
            cells.append(
                qcell(
                    r.qlike_pct,
                    r.q_lo,
                    r.q_hi,
                    r.sharpe_mid_diff,
                    r.sharpe_mid_ci_lo,
                    r.sharpe_mid_ci_hi,
                )
            )
        body.append(row(cells))
    write_tab(
        "lstmi",
        [rel(sc / "lstmi_levels.csv"), rel(sc / "lstmi_pairs.csv")],
        "lrrrrrrcc",
        [
            "input set",
            "QLIKE~~intraday",
            "QLIKE~~per-bar LSTM",
            "QLIKE~~ridge",
            "Sharpe mid~~intraday",
            "Sharpe mid~~per-bar LSTM",
            "Sharpe mid~~ridge",
            "intraday $-$~~per-bar LSTM",
            "intraday $-$~~ridge",
        ],
        body,
    )
    for other, stem in [("perbar_lstm", "LstmiVsLSTM"), ("ridge", "LstmiVsRidge")]:
        s = pr[(pr.a == "intraday") & (pr.b == other)]
        count_macros(
            stem, counts(s, "q_lo", "q_hi", "sharpe_mid_ci_lo", "sharpe_mid_ci_hi")
        )
    kp = pd.read_csv(sc / "lstmi_kept.csv")
    for b, short in zip(BUCKETS, ["Base", "Live", "All"]):
        r = kp[(kp.bucket == b) & (kp.what == "refit (mse rule)")].iloc[0]
        macro(f"campLstmiStep{short}", fint(r.p))
        macro(f"campLstmiKept{short}", fint(r["median"]))
        macro(f"campLstmiKept{short}Lo", fint(r["min"]))
        macro(f"campLstmiKept{short}Hi", fint(r["max"]))
    hs = pd.read_csv(sc / "lstmi_hyperparameter_share.csv")
    sl = (
        hs[(hs.rule == "mse") & (hs.axis == "seq_len")]
        .groupby("value")
        .tuning_points.sum()
    )
    macro("campLstmiTunings", fint(sl.sum()))
    for n, word in [(13, "Thirteen"), (48, "FortyEight"), (96, "NinetySix")]:
        macro(f"campLstmiN{word}", fint(sl.get(float(n), 0)))
    macro("campLstmiRefits", fint(kp[kp.what == "refit (mse rule)"].n.iloc[0]))


# =========================================================================== 6 reproducibility
def repro() -> None:
    dd = pd.read_csv(D / "arch" / "design_diffs.csv")
    ph = pd.read_csv(D / "arch" / "probe_hosts.csv")
    for b in ["all_features"]:  # live_feasible commented out
        s = ph[ph.bucket == b]
        assert (
            s.X_sha.nunique() == 2
            and s.y_sha.nunique() == 1
            and s.kept_sets_sha.nunique() == 1
        ), b
    diff = dd[dd.cols_differ > 0]
    same = dd[dd.cols_differ == 0]
    assert (dd.y_identical).all()
    macro("campXColsLive", fint(diff[diff.bucket == "live_feasible"].cols_differ.max()))
    macro("campXColsAll", fint(diff[diff.bucket == "all_features"].cols_differ.max()))
    macro("campXMaxAbs", fsci(diff.max_abs_diff.max()))
    macro("campXHosts", fint(ph.host.nunique()))
    macro("campXSamePairs", fint(len(same)))

    q1 = pd.read_csv(D / "mask" / "qualify_h2x512_vs_carc_xeon.csv")
    q1r = pd.read_csv(D / "mask" / "qualify_h2x512r_vs_carc_xeon.csv")
    q2 = pd.read_csv(D / "s2class" / "qualify_h2x512_vs_carc_xeon.csv")
    macro(
        "campSameClassTrials",
        f"{fint(q1.stage1_trials_bit_identical.sum())}/{fint(q1.stage1_trials.sum())}",
    )
    macro(
        "campSameClassTrialsRep",
        f"{fint(q1r.stage1_trials_bit_identical.sum())}/{fint(q1r.stage1_trials.sum())}",
    )
    s2i = (
        q1.stage2_forecasts_bit_identical.sum()
        + q2.stage2_forecasts_bit_identical.sum()
    )
    s2n = q1.stage2_forecasts.sum() + q2.stage2_forecasts.sum()
    macro("campSameClassFcst", f"{fint(s2i)}/{fint(s2n)}")
    x1 = pd.read_csv(D / "mask" / "qualify_carc_xeon_vs_carc_epyc.csv").set_index(
        "model"
    )
    x2 = pd.read_csv(D / "s2class" / "qualify_carc_xeon_vs_carc_epyc.csv").set_index(
        "model"
    )
    for m, short in [("lgbm", "Lgbm"), ("xgb", "Xgb"), ("rf", "Rf")]:
        macro(
            f"campXTrials{short}",
            f"{fint(x1.loc[m, 'stage1_trials_bit_identical'])}/{fint(x1.loc[m, 'stage1_trials'])}",
        )
        macro(
            f"campXFcst{short}",
            f"{fint(x2.loc[m, 'stage2_forecasts_bit_identical'])}/{fint(x2.loc[m, 'stage2_forecasts'])}",
        )
        macro(f"campXFcstMax{short}", fnum(x2.loc[m, "stage2_max_rel_diff"], 3))
        macro(
            f"campXPicks{short}",
            "identical"
            if bool(x1.loc[m, "stage1_picks_identical"])
            else "not identical",
        )

    xg = pd.read_csv(C / "xclass" / "xclass_gate.csv").set_index("model")
    xe = pd.read_csv(C / "xclass_epyc" / "xclass_gate.csv").set_index("model")
    assert xe.records_bit_identical.all()
    macro("campXcPoints", fint(xg.points.iloc[0]))
    macro("campXcPickLgbm", fint(xg.loc["lgbm", "best_mse_k50_differs"]))
    macro("campXcPickXgb", fint(xg.loc["xgb", "best_mse_k50_differs"]))
    macro("campXcPickRf", fint(xg.loc["rf", "best_mse_k50_differs"]))
    macro("campXcValLgbm", fnum(xg.loc["lgbm", "val_mse_max_rel_diff"], 1))
    macro("campXcValXgb", fsci(xg.loc["xgb", "val_mse_max_rel_diff"]))
    macro("campXcFcstXgb", fsci(xg.loc["xgb", "tp1_forecast_max_rel_diff"]))
    macro("campXcSame", fint(xe.points.iloc[0]))

    ca = pd.read_csv(HCLS / "class_arms.csv").set_index("rung")
    mp = _pairs()
    cp = mp[mp.family == "CPU class"]
    for rung, short in [("T10", "Ten"), ("T1", "One")]:
        macro(f"campClassRel{short}", pct(ca.loc[rung, "max_rel"], 1))
        r = cp[cp.a.str.startswith(rung + " ")].iloc[0]
        macro(f"campClassQ{short}", fnum(r.qlike_pct, 1, True))
        macro(f"campClassQ{short}CI", fci(r.q_lo, r.q_hi, 1))
        macro(f"campClassS{short}", fnum(r.sharpe_mid_diff, 2, True))
        macro(f"campClassS{short}CI", fci(r.sharpe_mid_ci_lo, r.sharpe_mid_ci_hi, 2))
    macro("campClassRows", fint(ca.rows.iloc[0]))

    xl = pd.read_csv(LI / "gates" / "cross_class_mixed_vs_pinned.csv")
    tpn = xl[xl.what == "tuning point"]
    rch = xl[xl.what == "refit chunk"]
    macro("campLstmiXTune", f"{fint(tpn.bit_identical.sum())}/{fint(len(tpn))}")
    macro("campLstmiXChunk", f"{fint(rch.bit_identical.sum())}/{fint(len(rch))}")

    sm = pd.read_csv(CR / "single_vs_mixed.csv")
    macro("campSMDiffer", fint((sm.forecast_rows_differ > 0).sum()))
    for m, short in [("lgbm", "Lgbm"), ("xgb", "Xgb"), ("rf", "Rf")]:
        macro(
            f"campSMMax{short}",
            fnum(sm[sm.model == m].forecast_max_rel_diff.max(), 2 if m != "rf" else 4),
        )
    c = counts(
        sm, "qlike_diff_ci_lo", "qlike_diff_ci_hi", "dSharpe_mid_lo", "dSharpe_mid_hi"
    )
    count_macros("SM", c)
    macro("campSMSharpeExcl", fint(c["sb"] + c["sa"]))

    chh = pd.read_csv(C / "chunk_hosts.csv")
    macro("campCChunks", fint(len(chh)))
    macro("campCChunkClasses", fint(chh.cpu_class.nunique()))
    cen = pd.read_csv(HCLS / "class_census.csv")
    fleet = cen[cen.root != "xeon"]
    macro("campHChunks", fint(fleet.chunks.sum()))
    macro("campHChunkClasses", fint(fleet["class"].nunique()))
    macro("campHGateChunks", fint(cen[cen.root == "xeon"].chunks.sum()))

    body = [
        row(
            [
                "design $X$, machines of the two vector classes",
                f"{tt('live_feasible')}: {macro_ref('campXColsLive')} of {macro_ref('campPLive')} columns differ; "
                f"{tt('all_features')}: {macro_ref('campXColsAll')} of {macro_ref('campPAll')}; "
                f"largest absolute difference {macro_ref('campXMaxAbs')}; target and kept-column sets identical",
            ]
        ),
        row(
            [
                "Optuna, same class on two clusters",
                f"stage-1 trials bit-identical {macro_ref('campSameClassTrials')} (replicate "
                f"{macro_ref('campSameClassTrialsRep')}); stage-2 forecasts {macro_ref('campSameClassFcst')}",
            ]
        ),
        row(
            [
                "Optuna, across classes on one cluster (same inputs)",
                f"stage-1 trials bit-identical: LightGBM {macro_ref('campXTrialsLgbm')}, XGBoost "
                f"{macro_ref('campXTrialsXgb')}, random forest {macro_ref('campXTrialsRf')}; stage-2 "
                f"forecasts from the same records: {macro_ref('campXFcstLgbm')} (max rel. "
                f"{macro_ref('campXFcstMaxLgbm')}), {macro_ref('campXFcstXgb')} ({macro_ref('campXFcstMaxXgb')}), "
                f"{macro_ref('campXFcstRf')}",
            ]
        ),
        row(
            [
                f"Optuna, first {macro_ref('campXcPoints')} tuning points re-run on the other class",
                f"best-of-50 pick changes: LightGBM {macro_ref('campXcPickLgbm')}, XGBoost "
                f"{macro_ref('campXcPickXgb')}, random forest {macro_ref('campXcPickRf')}; XGBoost forecast "
                f"max rel. {macro_ref('campXcFcstXgb')}; same-class re-run bit-identical at "
                f"{macro_ref('campXcSame')} of {macro_ref('campXcPoints')} points for all three",
            ]
        ),
        row(
            [
                r"masked LightGBM \texttt{live\_feasible}, whole run on the other class",
                f"T10 / T1: max rel. forecast difference {macro_ref('campClassRelTen')} / "
                f"{macro_ref('campClassRelOne')}\\,\\%; QLIKE {macro_ref('campClassQTen')} "
                f"{macro_ref('campClassQTenCI')} / {macro_ref('campClassQOne')} {macro_ref('campClassQOneCI')}\\,\\%; "
                f"$\\Delta$Sharpe mid {macro_ref('campClassSTen')} {macro_ref('campClassSTenCI')} / "
                f"{macro_ref('campClassSOne')} {macro_ref('campClassSOneCI')}",
            ]
        ),
        row(
            [
                "intraday LSTM, mixed-class attempt vs one class",
                f"tuning points bit-identical {macro_ref('campLstmiXTune')}; refit chunks "
                f"{macro_ref('campLstmiXChunk')} (the others ran on the other class)",
            ]
        ),
        row(
            [
                "Optuna, one class vs class-mixed first pass",
                f"{macro_ref('campSMDiffer')} of {macro_ref('campSMN')} tables differ (max rel. LightGBM "
                f"{macro_ref('campSMMaxLgbm')}, XGBoost {macro_ref('campSMMaxXgb')}, random forest "
                f"{macro_ref('campSMMaxRf')}); QLIKE intervals below / above 0: {macro_ref('campSMQBelow')} / "
                f"{macro_ref('campSMQAbove')}; Sharpe mid intervals excluding 0: {macro_ref('campSMSharpeExcl')}",
            ]
        ),
    ]
    write_tab(
        "repro",
        [
            rel(D / "arch" / "design_diffs.csv"),
            rel(D / "arch" / "probe_hosts.csv"),
            rel(D / "mask" / "qualify_h2x512_vs_carc_xeon.csv")
            + " (+ _h2x512r_, s2class/)",
            rel(D / "mask" / "qualify_carc_xeon_vs_carc_epyc.csv") + " (+ s2class/)",
            rel(C / "xclass" / "xclass_gate.csv") + " (+ xclass_epyc/)",
            rel(HCLS / "class_arms.csv")
            + ", "
            + rel(H / "mask_pairs.csv")
            + " (family 'CPU class')",
            rel(LI / "gates" / "cross_class_mixed_vs_pinned.csv"),
            rel(CR / "single_vs_mixed.csv"),
        ],
        r"p{0.30\textwidth}p{0.64\textwidth}",
        ["check", "result"],
        body,
    )


def macro_ref(name: str) -> str:
    assert name in NUMS, f"macro used before it is defined: {name}"
    return "\\" + name + "{}"


# =========================================================================== 7 master table
def master() -> None:
    ba = pd.read_csv(J / "before_after.csv")
    both = ba[(ba.status == "both") & (ba.family != "rule")]
    order = list(dict.fromkeys(both.family))
    body = []
    for fam in order:
        s = both[both.family == fam]
        q = s.qlike_pct_change
        d = s.dSharpe_mid
        body.append(
            row(
                [
                    tex_escape(fam),
                    fint(len(s)),
                    tex_escape(str(s.what_changed.iloc[0])).split(" (")[0],
                    f"{fnum(q.median(), 2, True)} [{fnum(q.min(), 2, True)}, {fnum(q.max(), 2, True)}]",
                    f"{fnum(d.median(), 2, True)} [{fnum(d.min(), 2, True)}, {fnum(d.max(), 2, True)}]",
                    f"{fint((s.dSharpe_mid_lo > 0).sum())} / {fint((s.dSharpe_mid_hi < 0).sum())}",
                    f"{fint(s.positions_changed.median())} [{fint(s.positions_changed.max())}]",
                ]
            )
        )
    write_tab(
        "master",
        [rel(J / "before_after.csv") + " (status 'both')"],
        r"lrp{0.22\textwidth}cccc",
        [
            "family",
            "keys",
            "what changed",
            r"QLIKE change (\%)~~median [min, max]",
            r"$\Delta$Sharpe mid~~median [min, max]",
            r"$\Delta$Sharpe interval~~above / below 0",
            "positions changed~~median [max]",
        ],
        body,
    )
    macro("campMTKeysBoth", fint(len(both)))
    macro("campMTKeysNew", fint((ba.status == "new").sum()))
    h = ba[ba.key == "sub_ridge_live_feasible"].iloc[0]
    macro("campMTHeadQ", fnum(h.qlike_recal_after, 4))
    macro("campMTHeadQBefore", fnum(h.qlike_recal_before, 4))
    macro("campMTHeadPos", fint(h.positions_changed))
    macro("campMTHeadDays", fint(h.n_days_paired))
    macro("campMTHeadS", fnum(h.Sharpe_mid_after, 2))
    macro("campMTHeadSX", fnum(h.Sharpe_crossed_after, 2))
    macro("campMTHeadMaxRel", fsci(h.max_rel_pred_clock_change))
    sig = both[(both.dSharpe_mid_lo > 0) | (both.dSharpe_mid_hi < 0)]
    macro(
        "campMTSigWho",
        "; ".join(
            f"{tex_escape(r.label)}: {fnum(r.Sharpe_mid_before, 2)} $\\to$ {fnum(r.Sharpe_mid_after, 2)}, "
            f"{fnum(r.dSharpe_mid, 2, True)} {fci(r.dSharpe_mid_lo, r.dSharpe_mid_hi, 2)}"
            for r in sig.itertuples()
        ),
    )
    macro("campMTSigN", fint(len(sig)))

    mt = pd.read_csv(J / "master_table.csv")
    vh = pd.read_csv(J / "master_table_vs_headline.csv")
    head = mt[mt.is_headline].key.iloc[0]
    assert head == "sub_ridge_live_feasible"
    other = mt[
        (mt.table == "A")
        & ~mt.family.str.contains("check")
        & (mt.family != "rule")
        & mt.duplicate_of.isna()
        & (mt.key != head)
    ]
    v = vh[vh.key.isin(other.key)]
    assert len(v) == len(other)
    macro("campMTNOther", fint(len(v)))
    macro("campMTBeatMid", fint((v.dSharpe_mid_vs_headline_lo > 0).sum()))
    macro("campMTBeatCrossed", fint((v.dSharpe_crossed_vs_headline_lo > 0).sum()))
    macro("campMTWorseMid", fint((v.dSharpe_mid_vs_headline_hi < 0).sum()))
    better = v[v.qlike_diff_ci_hi < 0].sort_values("qlike_pct_vs_headline")
    macro("campMTQBetter", fint(len(better)))
    macro("campMTQWorse", fint((v.qlike_diff_ci_lo > 0).sum()))
    fams = better.family.unique()
    macro("campMTQBetterFamily", ", ".join(tex_escape(f) for f in fams))
    macro(
        "campMTQBetterWho",
        "; ".join(
            f"{tex_escape(r.label)}: QLIKE {fnum(r.qlike_recal, 4)} ({fnum(r.qlike_pct_vs_headline, 1, True)}\\,\\%, "
            f"interval on the daily difference {fci(r.qlike_diff_ci_lo, r.qlike_diff_ci_hi, 4)}), "
            f"$\\Delta$Sharpe mid {fnum(r.dSharpe_mid_vs_headline, 2, True)} "
            f"{fci(r.dSharpe_mid_vs_headline_lo, r.dSharpe_mid_vs_headline_hi, 2)}"
            for r in better.itertuples()
        ),
    )


# =========================================================================== 8 importance, density
def featimp() -> None:
    ha = pd.read_csv(F / "before_after_har_ma.csv")
    hc = pd.read_csv(F / "before_after_har_ma_columns.csv")
    body = []
    for b in ["all_features"]:  # live_feasible commented out
        for m in TREES:
            s = ha[(ha.bucket == b) & (ha.model == m)].set_index("measure")
            xc = hc[
                (hc.bucket == b) & (hc.model == m) & (hc.measure == "mdi")
            ].old_x_close_share_of_har_columns
            body.append(
                row(
                    [
                        tt(b),
                        MODEL[m],
                        f"{pct(s.loc['mdi', 'value_old'], 0)} $\\to$ {pct(s.loc['mdi', 'value_new'], 0)}",
                        f"{pct(s.loc['shap', 'value_old'], 0)} $\\to$ {pct(s.loc['shap', 'value_new'], 0)}",
                        f"{fnum(s.loc['perm_p2_qlike', 'pct_of_loss_old'], 0, True)} $\\to$ "
                        f"{fnum(s.loc['perm_p2_qlike', 'pct_of_loss_new'], 0, True)}",
                        pct(xc.iloc[0], 0),
                        f"{int(s.loc['mdi', 'rank_old'])} $\\to$ {int(s.loc['mdi', 'rank_new'])}",
                    ]
                )
            )
    write_tab(
        "featimp",
        [
            rel(F / "before_after_har_ma.csv"),
            rel(F / "before_after_har_ma_columns.csv") + " (measure 'mdi')",
        ],
        "llccccc",
        [
            "input set",
            "model",
            r"MDI share (\%)~~old $\to$ new",
            r"TreeSHAP share (\%)~~old $\to$ new",
            r"perm.~P2 QLIKE~~(\% of tail loss)",
            r"old \texttt{\_x\_close} share~~of the ladder's MDI (\%)",
            "MDI rank~~old $\\to$ new",
        ],
        body,
    )
    # the yardstick: the ridge poses the same problem in both runs, so its change is draw noise
    rg = ha[(ha.model == "ridge") & (ha.measure == "perm_p2_mse")]
    macro(
        "campFIRidgeNoise",
        fnum((rg.pct_of_loss_new - rg.pct_of_loss_old).abs().max(), 1),
    )
    sp = pd.read_csv(F / "before_after_spearman.csv")
    t = sp[(sp.level == "series") & sp.model.isin(TREES) & sp.measure.isin(FI_MEASURES)]
    macro("campFILeadN", fint(len(t)))
    macro("campFILead", fint((t.leader_new == "har_ma").sum()))
    tr = sp[(sp.level == "series") & sp.model.isin(TREES)]
    macro("campFISpearmanTrees", fnum(tr.spearman_old_new.median(), 2))
    macro("campFISpearmanTreesMin", fnum(tr.spearman_old_new.min(), 2))


def dvs() -> None:
    dd = pd.read_csv(G / "before_after_density.csv")
    dd = dd[(dd.level == "column") & (dd["sample"] == "all forecasts")]
    se = pd.read_csv(G / "before_after_session_edge.csv")
    models = ["ridge", "lasso", "elastic net", "LightGBM", "XGBoost", "random forest"]
    body = []
    for b in ["all_features"]:  # live_feasible commented out
        for m in models:
            r = dd[(dd.bucket == b) & (dd.model == m)].iloc[0]
            e = se[(se.bucket == b) & (se.model == m)]
            nomask = (
                "" if pd.isna(r.neff__dedup_nomask) else fnum(r.neff__dedup_nomask, 1)
            )
            body.append(
                row(
                    [
                        tt(b),
                        m,
                        fnum(r.neff__first_pass, 1),
                        nomask,
                        fnum(r.neff__dedup_mask, 1),
                        f"{fint(r.k95__first_pass)} $\\to$ {fint(r.k95__dedup_mask)}",
                        f"{fint(r.n_active__first_pass)} $\\to$ {fint(r.n_active__dedup_mask)}",
                        pct(e.first_pass_share_session_edge.iloc[0], 1)
                        if len(e)
                        else "",
                    ]
                )
            )
    write_tab(
        "dvs",
        [
            rel(G / "before_after_density.csv")
            + " (level 'column', sample 'all forecasts')",
            rel(G / "before_after_session_edge.csv"),
        ],
        "llrrrccc",
        [
            "input set",
            "model",
            r"$N_{\mathrm{eff}}$~~first pass",
            r"$N_{\mathrm{eff}}$~~de-dup, no mask",
            r"$N_{\mathrm{eff}}$~~de-dup + mask",
            r"$k_{95}$~~old $\to$ new",
            r"columns with~~weight",
            r"first pass: TreeSHAP~~on session-edge (\%)",
        ],
        body,
    )
    cl = pd.read_csv(G / "before_after_claims.csv")
    macro("campDvsClaims", fint(len(cl)))
    macro("campDvsClaimsSame", fint(cl.same.astype(str).str.lower().eq("true").sum()))
    sw = pd.read_csv(G / "before_after_sweep.csv")
    for b, short in [("live_feasible", "Live"), ("all_features", "All")]:
        r = sw[
            (sw.bucket == b) & (sw.model == "LightGBM") & (sw.k.astype(str) == "all")
        ].iloc[0]
        macro(f"campDvsLgbmQ{short}", fnum(r.QLIKE_new_minus_first_pass, 4, True))
        macro(
            f"campDvsLgbmQ{short}CI",
            fci(r.QLIKE_new_minus_first_pass_lo, r.QLIKE_new_minus_first_pass_hi, 4),
        )
        macro(f"campDvsLgbmS{short}", fnum(r.Sharpe_new_minus_first_pass, 2, True))
        macro(
            f"campDvsLgbmS{short}CI",
            fci(r.Sharpe_new_minus_first_pass_lo, r.Sharpe_new_minus_first_pass_hi, 2),
        )
    lin = sw[sw.model.isin(["ridge", "lasso"])]
    macro("campDvsLinMaxRel", fsci(lin.max_rel_forecast_change.max()))
    rf = dd[(dd.model == "random forest")]
    macro(
        "campDvsRfLive",
        f"{fnum(rf[rf.bucket == 'live_feasible'].neff__first_pass.iloc[0], 1)} $\\to$ "
        f"{fnum(rf[rf.bucket == 'live_feasible'].neff__dedup_mask.iloc[0], 1)}",
    )


# =========================================================================== 9 compute
@dataclass
class Usage:
    """A stage's cluster accounting, summed over its sacct rows."""

    alloc: float  # allocated CPU-hours
    used: float  # used CPU-hours
    peak: int  # peak concurrent allocated CPUs
    first: pd.Timestamp
    last: pd.Timestamp


# The agents' summaries print CPU-hours to one decimal: half a unit of that rounding.
HALF_DECIMAL = 0.05 + 1e-9


def _sacct(path: Path) -> Usage:
    u = pd.read_csv(path)
    st = pd.to_datetime(u.Start)
    en = pd.to_datetime(u.End)
    ev = sorted(
        [(t, int(c)) for t, c in zip(st, u.AllocCPUS)]
        + [(t, -int(c)) for t, c in zip(en, u.AllocCPUS)],
        key=lambda x: (x[0], x[1]),
    )
    cur = peak = 0
    for _, dc in ev:
        cur += dc
        peak = max(peak, cur)
    return Usage(
        alloc=float(u.CPUTimeRAW.sum()) / 3600.0,
        used=float(u.used_cpu_sec.sum()) / 3600.0,
        peak=peak,
        first=st.min(),
        last=en.max(),
    )


def _check_summary(path: Path, got: Usage) -> None:
    """The agent's own summary states allocated CPU-h and the peak; they must agree."""
    txt = path.read_text(encoding="utf-8")
    a = re.search(r"([\d.]+) allocated CPU-hours", txt)
    p = re.search(r"peak concurrent allocated CPUs (\d+)", txt)
    assert a and p, path
    assert abs(float(a.group(1)) - got.alloc) < HALF_DECIMAL, (
        path,
        a.group(1),
        got.alloc,
    )
    assert int(p.group(1)) == got.peak, (path, p.group(1), got.peak)


def _wall(first: pd.Timestamp, last: pd.Timestamp) -> str:
    mins = int(round((last - first).total_seconds() / 60.0))
    return (
        f"{first:%m-%d %H:%M}--{last:%m-%d %H:%M} ({mins // 60} h {mins % 60:02d} min)"
    )


def compute() -> None:
    rows: list[list[str]] = []
    srcs: list[str] = []

    def add(
        stage: str, cluster: str, alloc: str, used: str, peak: str, wall: str
    ) -> None:
        rows.append([stage, cluster, alloc, used, peak, wall])

    def add_usage(stage: str, u: Usage) -> None:
        add(
            stage,
            "A",
            fnum(u.alloc, 1),
            fnum(u.used, 1),
            fint(u.peak),
            _wall(u.first, u.last),
        )

    # A: no CPU-hours were reported for I1 (neither the linear fleet nor the LSTM); left out.
    b = _sacct(B / "cluster_usage.csv")
    _check_summary(B / "SUMMARY.md", b)
    srcs.append(rel(B / "cluster_usage.csv") + " (checked against its SUMMARY.md)")
    add_usage("trees T10 / T1 / RS10 / RS1, no mask", b)

    def kv(path: Path) -> dict[str, str]:
        u = pd.read_csv(path)
        return dict(zip(u["item"], u["value"].astype(str)))

    def kv_wall(d: dict[str, str]) -> str:
        f, l_ = [
            pd.Timestamp(x.strip()) for x in d["first start / last end"].split("/")
        ]
        return _wall(f, l_)

    cu = kv(C / "cluster_usage_unmasked_run.csv")
    srcs.append(rel(C / "cluster_usage_unmasked_run.csv"))
    add(
        "Optuna, no mask (stopped; not used)",
        "A",
        cu["allocated CPU-hours, all jobs"],
        "---",
        cu["peak concurrent CPUs (this campaign)"],
        kv_wall(cu),
    )
    h = _sacct(H / "cluster_usage.csv")
    _check_summary(H / "SUMMARY.md", h)
    srcs.append(rel(H / "cluster_usage.csv") + " (checked against its SUMMARY.md)")
    add_usage("trees and LSTM with the mask, one CPU class", h)
    cm = kv(C / "cluster_usage.csv")
    cs = kv(C / "cluster_usage_single_class_rerun.csv")
    srcs.append(
        rel(C / "cluster_usage.csv")
        + ", "
        + rel(C / "cluster_usage_single_class_rerun.csv")
    )
    add(
        "Optuna with the mask, all jobs",
        "A",
        cm["allocated CPU-hours, all jobs"],
        "---",
        cm["peak concurrent CPUs (this campaign)"],
        kv_wall(cm),
    )
    add(
        "\\quad of which the one-class re-run",
        "A",
        cs["allocated CPU-hours, all jobs"],
        "---",
        cs["peak concurrent CPUs (this campaign)"],
        kv_wall(cs),
    )
    i = _sacct(LI / "score" / "cluster_usage.csv")
    _check_summary(LI / "score" / "SUMMARY.md", i)
    srcs.append(
        rel(LI / "score" / "cluster_usage.csv") + " (checked against its SUMMARY.md)"
    )
    add_usage("intraday-sequence LSTM", i)

    # D: the checklist's agent log (its only accounting record)
    txt = PROGRESS.read_text(encoding="utf-8")
    m = re.search(
        r"D I5 .*?Hoffman2 ([\d.]+) slot-h \(([\d.]+) CPU-h\), peak (\d+) slots; CARC ([\d.]+) CPU-h allocated, peak (\d+) CPUs",
        txt,
    )
    assert m, "agent D's accounting line not found"
    srcs.append(rel(PROGRESS) + " (agent log, agent D's line)")
    add(
        "cross-cluster check (Optuna)",
        "A",
        m.group(4),
        "---",
        m.group(5),
        "not reported",
    )
    add(
        "cross-cluster check (Optuna)",
        "B",
        f"{m.group(1)} slot-h",
        m.group(2),
        f"{m.group(3)} slots",
        "not reported",
    )

    # E: its generated SUMMARY.md (qacct)
    et = (E / "SUMMARY.md").read_text(encoding="utf-8")
    pat = r"Accounting \({}; qacct, (\d+) job tasks\): ([\d.]+) slot-hours wall-clock, ([\d.]+) CPU-hours; first start (\S+) to last end (\S+) \(([\d.]+) min\)"
    e1 = re.search(pat.format("I6 run"), et)
    e2 = re.search(pat.format("control"), et)
    assert e1 and e2
    srcs.append(rel(E / "SUMMARY.md") + " (its accounting lines, from qacct)")
    add(
        "rest-of-day arms, all entry clocks",
        "B",
        f"{e1.group(2)} slot-h",
        e1.group(3),
        "not reported",
        f"{e1.group(4)[:5]}--{e1.group(5)[:5]} ({e1.group(6)} min)",
    )
    add(
        "\\quad attribution control",
        "B",
        f"{e2.group(2)} slot-h",
        e2.group(3),
        "not reported",
        f"{e2.group(4)[:5]}--{e2.group(5)[:5]} ({e2.group(6)} min)",
    )

    # F: its commit message
    msg = subprocess.run(
        ["git", "log", "-1", "--format=%B", F_COMMIT],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    f = re.search(
        r"([\d.]+) core-h allocated / ([\d.]+) used, peak (\d+) CPUs, (\d\d:\d\d)-(\d\d:\d\d)",
        msg,
    )
    assert f, "agent F's accounting not found in its commit message"
    srcs.append(f"git log -1 {F_COMMIT} (agent F's commit message)")
    add(
        "feature importance re-run",
        "A",
        f.group(1),
        f.group(2),
        f.group(3),
        f"{f.group(4)}--{f.group(5)}",
    )

    # G: numbers.json
    g = json.loads((G / "numbers.json").read_text(encoding="utf-8"))["timing"]
    srcs.append(rel(G / "numbers.json") + " (timing)")
    add(
        "dense-vs-sparse re-run",
        "A",
        fnum(g["cluster_sacct"]["alloc_cpu_hours"], 1),
        "---",
        fint(g["cluster_sacct"]["peak_cpus"]),
        f"{fnum(g['cluster_sacct']['wall_min_total'], 0)} min of job time",
    )
    add(
        "\\quad its local part",
        "local",
        "---",
        fnum(g["local"]["cpu_hours"], 1),
        f"{fint(g['local']['peak_processes'])} processes",
        f"{fnum(g['local']['wall_min'], 0)} min",
    )

    write_tab(
        "compute",
        srcs,
        "llrrrl",
        [
            "stage",
            "cluster",
            "allocated~~CPU-h",
            "used~~CPU-h",
            "peak~~CPUs",
            "wall-clock (first start--last end)",
        ],
        [row(r) for r in rows],
    )
    macro("campComputeOptunaAlloc", cm["allocated CPU-hours, all jobs"])
    macro("campComputeOptunaPeak", cm["peak concurrent CPUs (this campaign)"])


# =========================================================================== main
def main() -> None:
    GEN.mkdir(parents=True, exist_ok=True)
    design()
    mask()
    cadence()
    optuna()
    lstmi()
    repro()
    master()
    featimp()
    dvs()
    compute()
    lines = [
        "% AUTO-GENERATED by writeup/make_appendix_campaign_16h_tex.py -- do not edit.",
        "% Every macro is read from the campaign's CSVs / summaries (sources in the table files and",
        "% in the script); prose in sections/appendix_campaign_16h.tex quotes numbers only through these.",
    ]
    lines += [r"\newcommand{\%s}{%s}" % (k, v) for k, v in NUMS.items()]
    (GEN / f"{PREFIX}_numbers.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote {len(WRITTEN)} tables ({', '.join(WRITTEN)}) and {len(NUMS)} macros")


if __name__ == "__main__":
    main()
