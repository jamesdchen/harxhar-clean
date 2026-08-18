"""Emit the options-section macros and table bodies for the writeup.

Reads the committed harvest CSVs under results/spxw_pnl/ and writes
    writeup/generated/options_numbers.tex     scalar macros (\\optXxx)
    writeup/generated/table_options_book.tex  dead-zone x control table body
    writeup/generated/table_options_final.tex composed-book table body
    writeup/generated/table_options_hour.tex  by-entry-hour table body
    writeup/generated/table_options_ttc.tex   time-till-close encompassing body

Same discipline as score_unification.py: every number in the paper's
options section is minted here from the CSVs; nothing is hand-typed.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "spxw_pnl")
GEN = os.path.join(ROOT, "writeup", "generated")

_ROMAN = {
    0.0: "Zero",
    0.05: "FiveHund",
    0.1: "Ten",
    0.2: "Twenty",
}


_DIGIT_WORDS = {
    "0": "Zero",
    "1": "One",
    "2": "Two",
    "3": "Three",
    "4": "Four",
    "5": "Five",
    "6": "Six",
    "7": "Seven",
    "8": "Eight",
    "9": "Nine",
}


def _cs(name: str) -> str:
    """TeX control sequences are letters only: spell out any digit.

    Model keys like ``A_a0`` would otherwise emit ``\\optTtcAa0DM``, which TeX
    reads as ``\\optTtcAa`` followed by the characters ``0DM`` -- a fatal error
    the moment the file is \\input.
    """
    return "".join(_DIGIT_WORDS.get(c, c) for c in name)


def _f(x: Any, nd: int = 2, sign: bool = False) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "---"
    v = float(x)
    s = f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"
    return s.replace("-", "$-$") if v < 0 else s


def _pct(x: Any) -> str:
    return "---" if not np.isfinite(float(x)) else f"{100 * float(x):.0f}\\%"


def _int(x: Any) -> str:
    return f"{int(x):,}".replace(",", "{,}")


def main() -> None:
    import sys

    ft = "--ft" in sys.argv
    sfx = "_ft" if ft else ""
    os.makedirs(GEN, exist_ok=True)
    macros: dict[str, str] = {}
    macros["optForecastMode"] = (
        "the one-step forecast standing at $t$, mapped to the remaining horizon"
        if ft
        else "the per-bar forecasts summed over the remaining bars"
    )

    # ---------------- strategy book (dead-zone table) ----------------
    s = pd.read_csv(os.path.join(RES, f"dh_legs{sfx}_summary.csv"))
    both = s[(s.era == "all") & (s.leg == "both")]
    ctrl = both[both.model == "always_short"].iloc[0]
    macros["optNContracts"] = _int(both[both.model == "b2"]["n_contracts"].max())
    macros["optNDays"] = _int(both["n_days"].max())
    macros["optAlwaysShortSh"] = _f(ctrl["sh_daily_mid"])
    macros["optAlwaysShortHit"] = _pct(ctrl["hit_mid"])
    rows_book = []
    for th in (0.0, 0.05, 0.10, 0.20):
        r = both[(both.model == "b2") & (np.isclose(both.theta, th))].iloc[0]
        a = both[(both.model == "a0") & (np.isclose(both.theta, th))].iloc[0]
        sw = both[(both.model == "b2_minus_a0_t") & (np.isclose(both.theta, th))].iloc[
            0
        ]
        tag = _ROMAN[th]
        macros[f"optTheta{tag}Traded"] = _pct(r["frac_traded"])
        macros[f"optTheta{tag}Long"] = _pct(r["frac_long"])
        macros[f"optTheta{tag}ShMid"] = _f(r["sh_daily_mid"], 1)
        macros[f"optTheta{tag}ShX"] = _f(r["sh_daily_crossed"], 1)
        macros[f"optTheta{tag}Hit"] = _pct(r["hit_mid"])
        macros[f"optTheta{tag}AzeroShMid"] = _f(a["sh_daily_mid"], 1)
        macros[f"optTheta{tag}AzeroShX"] = _f(a["sh_daily_crossed"], 1)
        macros[f"optTheta{tag}SwapT"] = _f(sw["sh_daily_mid"], 2, sign=True)
        rows_book.append(
            f"$\\theta = {th:.2f}$ & {_pct(r['frac_traded'])} & {_pct(r['frac_long'])} & "
            f"{_f(r['sh_daily_mid'], 1)} & {_f(r['sh_daily_crossed'], 1)} & {_pct(r['hit_mid'])} & "
            f"{_f(a['sh_daily_mid'], 1)} & {_f(a['sh_daily_crossed'], 1)} & "
            f"{_f(sw['sh_daily_mid'], 2, sign=True)} \\\\"
        )
    rows_book.append(
        f"Always short (control) & 100\\% & 0\\% & {_f(ctrl['sh_daily_mid'], 1)} & --- & "
        f"{_pct(ctrl['hit_mid'])} & --- & --- & --- \\\\"
    )
    # legs at theta=0.10, era all and daily-0dte
    for era, etag in (("all", ""), ("daily_0dte", "Daily")):
        for leg, ltag in (("call", "Call"), ("put", "Put"), ("both", "Both")):
            r = s[
                (s.era == era)
                & (s.leg == leg)
                & (s.model == "b2")
                & (np.isclose(s.theta, 0.10))
            ].iloc[0]
            macros[f"opt{etag}{ltag}ShMid"] = _f(r["sh_daily_mid"], 1)
            macros[f"opt{etag}{ltag}ShX"] = _f(r["sh_daily_crossed"], 1)
    macros["optDailyNDays"] = _int(
        s[(s.era == "daily_0dte") & (s.leg == "both")]["n_days"].max()
    )
    with open(
        os.path.join(GEN, "table_options_book.tex"), "w", encoding="utf-8", newline="\n"
    ) as fh:
        fh.write("\n".join(rows_book) + "\n\\bottomrule\n")

    # barebones: three rows (always-short control, OLS--HAR, two-block ridge)
    # at theta = 0.10, both legs; columns = all-days and daily-0DTE eras.
    bare = []
    for name, model in (
        ("Always short (no forecast)", "always_short"),
        ("OLS--HAR incumbent", "a0"),
        ("Two-block ridge", "b2"),
    ):
        cells = []
        for era in ("all", "daily_0dte"):
            sub = s[(s.era == era) & (s.leg == "both") & (s.model == model)]
            if model != "always_short":
                sub = sub[np.isclose(sub.theta, 0.10)]
            r = sub.iloc[0]
            trad = "100\%" if model == "always_short" else _pct(r["frac_traded"])
            long_ = "0\%" if model == "always_short" else _pct(r["frac_long"])
            shx = "---" if model == "always_short" else _f(r["sh_daily_crossed"], 1)
            cells.append(
                f"{trad} & {long_} & {_f(r['sh_daily_mid'], 1)} & {shx} & {_pct(r['hit_mid'])}"
            )
            etag = "" if era == "all" else "Daily"
            mtag = {"always_short": "Ctrl", "a0": "Azero", "b2": "Blk"}[model]
            macros[f"optBare{etag}{mtag}ShMid"] = _f(r["sh_daily_mid"], 1)
            macros[f"optBare{etag}{mtag}ShX"] = shx
            macros[f"optBare{etag}{mtag}Hit"] = _pct(r["hit_mid"])
            macros[f"optBare{etag}{mtag}Traded"] = trad
            macros[f"optBare{etag}{mtag}Long"] = long_
        bare.append(f"{name} & " + " & ".join(cells) + " \\\\")
    with open(
        os.path.join(GEN, "table_options_bare.tex"), "w", encoding="utf-8", newline="\n"
    ) as fh:
        fh.write("\n".join(bare) + "\n\\bottomrule\n")
    for era, etag in (("all", ""), ("daily_0dte", "Daily")):
        for leg, ltag in (("call", "Call"), ("put", "Put")):
            for model, mtag in (("a0", "Azero"), ("b2", "Blk")):
                r = s[
                    (s.era == era)
                    & (s.leg == leg)
                    & (s.model == model)
                    & np.isclose(s.theta, 0.10)
                ].iloc[0]
                macros[f"optBare{etag}{ltag}{mtag}ShMid"] = _f(r["sh_daily_mid"], 1)
                macros[f"optBare{etag}{ltag}{mtag}ShX"] = _f(r["sh_daily_crossed"], 1)

    # ---------------- by entry hour ----------------
    h = pd.read_csv(os.path.join(RES, f"dh_legs{sfx}_by_hour.csv"))
    ha = h[h.era == "all"]
    rows_hour = []
    for _, r in ha.iterrows():
        rows_hour.append(
            f"{r['entry']} & {_int(r['n'])} & {_f(r['vrp_med'], 3)} & {_pct(r['frac_long_b2'])} & "
            f"{_f(r['sh_always_short'], 1)} & {_f(r['sh_b2'], 1)} & {_f(r['sh_a0'], 1)} \\\\"
        )
    with open(
        os.path.join(GEN, "table_options_hour.tex"), "w", encoding="utf-8", newline="\n"
    ) as fh:
        fh.write("\n".join(rows_hour) + "\n\\bottomrule\n")
    morn = ha[
        ha.entry.isin(["10:00", "10:30", "11:00", "11:30", "12:00", "12:30", "13:00"])
    ]
    late = ha[ha.entry.isin(["15:00", "15:30"])]
    macros["optMornShLo"] = _f(morn["sh_b2"].min(), 1)
    macros["optMornShHi"] = _f(morn["sh_b2"].max(), 1)
    macros["optMornCtrlLo"] = _f(morn["sh_always_short"].min(), 1)
    macros["optMornCtrlHi"] = _f(morn["sh_always_short"].max(), 1)
    macros["optLateShHi"] = _f(late["sh_b2"].max(), 1)
    macros["optLateVrp"] = _f(late["vrp_med"].max(), 2)
    macros["optLateLong"] = _pct(late["frac_long_b2"].min())

    # ---------------- cost floor ----------------
    q = pd.read_csv(os.path.join(RES, "quote_costs_breakeven.csv"))
    qa = q[q.era == "all"]

    def _cell(hour: int, bucket: str, col: str = "theta_min_med") -> float:
        return float(qa[(qa.hour_et == hour) & (qa.bucket == bucket)][col].iloc[0])

    macros["optCostElevenLo"] = _f(_cell(11, "[0.30,0.50)"), 3)
    macros["optCostElevenMid"] = _f(_cell(11, "[0.50,0.70)"), 3)
    macros["optCostFifteenLo"] = _f(_cell(15, "[0.30,0.50)"), 3)
    macros["optCostFifteenMid"] = _f(_cell(15, "[0.50,0.70)"), 3)
    macros["optCostDeepLo"] = _f(
        qa[qa.bucket == "[0.70,0.90]"]["theta_min_med"].min(), 2
    )
    macros["optCostDeepHi"] = _f(
        qa[qa.bucket == "[0.70,0.90]"]["theta_min_med"].max(), 2
    )

    # ---------------- final composed book ----------------
    fb = pd.read_csv(os.path.join(RES, "dh_book_final.csv"))
    f10 = fb[
        (fb.model == "b2")
        & (fb.leg == "both")
        & (fb.era == "all")
        & np.isclose(fb.theta, 0.10)
    ]
    rows_final = []
    lab = {
        ("all_days", "hold"): "Hold to settlement, all days",
        ("all_days", "sigcross"): "Signal-cross exit, all days",
        ("ex_fomc", "hold"): "Hold to settlement, ex-FOMC",
        ("ex_fomc", "sigcross"): "Signal-cross exit, ex-FOMC",
    }
    for (dd, ex), name in lab.items():
        r = f10[(f10.days == dd) & (f10.exit == ex)].iloc[0]
        tag = ("Ex" if dd == "ex_fomc" else "All") + (
            "Cross" if ex == "sigcross" else "Hold"
        )
        macros[f"optFinal{tag}ShMid"] = _f(r["sh_mid"], 2)
        macros[f"optFinal{tag}ShX"] = _f(r["sh_crossed"], 2)
        macros[f"optFinal{tag}HitX"] = _pct(r["hit_crossed"])
        macros[f"optFinal{tag}NDays"] = _int(r["n_days"])
        rows_final.append(
            f"{name} & {_int(r['n_days'])} & {_pct(r['frac_traded'])} & {_pct(r['frac_long'])} & "
            f"{_f(r['sh_mid'], 2)} & {_pct(r['hit_mid'])} & {_f(r['sh_crossed'], 2)} & {_pct(r['hit_crossed'])} \\\\"
        )
    with open(
        os.path.join(GEN, "table_options_final.tex"),
        "w",
        encoding="utf-8",
        newline="\n",
    ) as fh:
        fh.write("\n".join(rows_final) + "\n\\bottomrule\n")
    f05 = fb[
        (fb.model == "b2")
        & (fb.leg == "both")
        & (fb.era == "all")
        & np.isclose(fb.theta, 0.05)
        & (fb.days == "ex_fomc")
        & (fb.exit == "sigcross")
    ].iloc[0]
    macros["optFinalFiveShMid"] = _f(f05["sh_mid"], 2)
    macros["optFinalFiveShX"] = _f(f05["sh_crossed"], 2)
    macros["optFinalFiveTraded"] = _pct(f05["frac_traded"])
    fd = fb[
        (fb.model == "b2")
        & (fb.leg == "both")
        & (fb.era == "daily_0dte")
        & np.isclose(fb.theta, 0.10)
        & (fb.days == "ex_fomc")
        & (fb.exit == "sigcross")
    ].iloc[0]
    macros["optFinalDailyShMid"] = _f(fd["sh_mid"], 2)
    macros["optFinalDailyShX"] = _f(fd["sh_crossed"], 2)
    sw = pd.read_csv(os.path.join(RES, "dh_book_final_swap.csv"))
    for _, r in sw.iterrows():
        macros[f"optFinalSwapT{_ROMAN[round(float(r['theta']), 2)]}"] = _f(
            r["t_b2_minus_a0"], 2, sign=True
        )
    cov = pd.read_csv(os.path.join(RES, "dh_book_final_coverage.csv")).iloc[0]
    macros["optFomcDays"] = _int(cov["fomc_days_excluded"])
    macros["optFomcUnknownDays"] = _int(cov["fomc_flag_unknown_days_kept"])

    # ---------------- regime: FOMC and exit rules ----------------
    fo = pd.read_csv(os.path.join(RES, "dh_regime_fomc.csv"))
    fo10 = fo[(fo.model == "b2") & np.isclose(fo.theta, 0.10)]
    fr = fo10[fo10.bucket == "fomc"].iloc[0]
    nf = fo10[fo10.bucket == "no_fomc"].iloc[0]

    # column names vary by generator; use what exists
    def _pick(row: pd.Series, *cands: str) -> float:
        for c in cands:
            if c in row.index and np.isfinite(float(row[c])):
                return float(row[c])
        return float("nan")

    macros["optFomcShMid"] = _f(_pick(fr, "sh_mid", "sh_daily_mid"), 2)
    macros["optFomcShX"] = _f(_pick(fr, "sh_crossed", "sh_daily_crossed"), 2)
    macros["optFomcFracShort"] = _pct(1.0 - float(fr["frac_long"]))
    macros["optNoFomcShMid"] = _f(_pick(nf, "sh_mid", "sh_daily_mid"), 2)
    macros["optNoFomcShX"] = _f(_pick(nf, "sh_crossed", "sh_daily_crossed"), 2)
    ex = pd.read_csv(os.path.join(RES, "dh_exit_rules.csv"))
    exa = ex[ex.entry_hour == "ALL"]
    for rule, tag in (
        ("hold", "Hold"),
        ("sigcross", "Cross"),
        ("k1", "KOne"),
        ("k4", "KFour"),
    ):
        r = exa[exa.rule == rule].iloc[0]
        macros[f"optExit{tag}ShMid"] = _f(r["sh_mid"], 2)
        macros[f"optExit{tag}ShX"] = _f(r["sh_crossed"], 2)
    macros["optExitCrossEarly"] = _pct(
        exa[exa.rule == "sigcross"]["frac_exit_early"].iloc[0]
    )
    rf = pd.read_csv(os.path.join(RES, "dh_exit_realized_fraction.csv"))
    rfa = rf[rf.entry_hour == "ALL"] if "entry_hour" in rf else rf
    for k, tag in ((1, "One"), (4, "Four"), (9, "Nine")):
        v = rfa[rfa.k_bars == k]
        macros[f"optRealizedK{tag}"] = (
            _pct(float(v["realized_fraction"].iloc[0])) if len(v) else "---"
        )

    # ---------------- time-till-close forecast structure ----------------
    t = pd.read_csv(os.path.join(RES, "ttc_structure_pooled.csv"))
    tf = t[(t.panel == "fixed")]
    rows_ttc = []
    names = {
        "A_blk2": "Sum of per-bar forecasts (two-block ridge)",
        "A_a0": "Sum of per-bar forecasts (OLS--HAR)",
        "B_direct_blk2": "Direct regression on the sum",
        "B_direct_iv": "Direct regression on implied variance alone",
        "B_joint": "Direct regression on sum $+$ implied variance",
        "B_joint_realized": "\\quad $+$ realized variance so far today",
        "B_ft_blk2": "$\\mathcal{F}_t$-measurable: one-step forecast at $t$",
        "B_ft_joint": "\\quad $+$ implied variance",
    }
    for era, elab in (("all", "All days"), ("daily_0dte", "Daily 0DTE")):
        for key, name in names.items():
            r = tf[(tf.era == era) & (tf.forecast == key)]
            if r.empty:
                continue
            r = r.iloc[0]
            rows_ttc.append(
                f"{name} & {elab} & {_int(r['n_days'])} & {_f(r['qlike'], 4)} & "
                f"{_f(r['dm_t_vs_A_blk2'], 1, sign=True)} & {_f(r['dm_t_vs_nested'], 1, sign=True)} \\\\"
            )
            tag = (
                key.replace("_", "")
                .replace("blk2", "Blk")
                .replace("iv", "Iv")
                .replace("ft", "Ft")
            )
            etag = "" if era == "all" else "Daily"
            macros[f"optTtc{etag}{tag}QLIKE"] = _f(r["qlike"], 4)
            macros[f"optTtc{etag}{tag}DM"] = _f(r["dm_t_vs_A_blk2"], 1, sign=True)
            macros[f"optTtc{etag}{tag}Nested"] = _f(r["dm_t_vs_nested"], 1, sign=True)
    with open(
        os.path.join(GEN, "table_options_ttc.tex"), "w", encoding="utf-8", newline="\n"
    ) as fh:
        fh.write("\n".join(rows_ttc) + "\n\\bottomrule\n")
    macros["optTtcNDays"] = _int(tf[tf.era == "all"]["n_days"].max())

    # ---------------- 10:00 -> close encompassing (options_expression) ----------------
    e = pd.read_csv(os.path.join(RES, "options_expression_encompassing.csv"))
    for _, r in e.iterrows():
        tag = {
            "mfiv_loglog": "Mfiv",
            "blk2_loglog": "Blk",
            "joint_loglog": "Joint",
            "blk2_raw": "BlkRaw",
        }[r["forecast"]]
        macros[f"optEnc{tag}QLIKE"] = _f(r["qlike"], 4)
        macros[f"optEnc{tag}DM"] = _f(r["dm_vs_mfiv_loglog"], 1, sign=True)
    macros["optEncNDays"] = _int(e["n"].max())

    # ---------------- 10:00 -> close QLIKE (a0 vs blk2 on the swap claim) ----------------
    a = pd.read_csv(os.path.join(RES, "a0_vs_blk2_strategy.csv"))
    ql_a = a[a.book.str.startswith("QLIKE remaining a0")].iloc[0]
    ql_b = a[a.book.str.startswith("QLIKE remaining blk2")].iloc[0]
    macros["optTocloseAzeroQLIKE"] = _f(ql_a["mean"], 4)
    macros["optTocloseBlkQLIKE"] = _f(ql_b["mean"], 4)
    macros["optTocloseNDays"] = _int(ql_a["n"])
    inc = a[a.book.str.startswith("QLIKE increment")].iloc[0]
    macros["optTocloseIncrSh"] = _f(inc["sharpe_ann"], 1)
    hess = pd.read_csv(os.path.join(RES, "mfiv_toclose.csv"))
    hh_ = hess[hess.book.str.startswith("paper hess")].iloc[0]
    macros["optTocloseHessSh"] = _f(hh_["sharpe_ann"], 1)
    macros["optTocloseHessHit"] = _pct(hh_["hit"])
    vrp = a[a.book.str.startswith("always short smile")].iloc[0]
    macros["optSmileShortSh"] = _f(vrp["sharpe_ann"], 1)
    strip = hess[hess.book.str.startswith("unsigned strip_pnl")].iloc[0]
    macros["optStripShortSh"] = _f(-float(strip["sharpe_ann"]), 1)

    # ---------------- bumps ----------------
    b = pd.read_csv(os.path.join(RES, "iv_bumps_summary.csv"))
    ba = b[b.era == "all"]
    xs = ba[ba.score.str.startswith("xs_")]
    macros["optBumpsXsMidLo"] = _f(xs["sh_daily_mid"].min(), 1)
    macros["optBumpsXsMidHi"] = _f(xs["sh_daily_mid"].max(), 1)
    macros["optBumpsXsXLo"] = _f(xs["sh_daily_crossed"].min(), 1)
    macros["optBumpsXsXHi"] = _f(xs["sh_daily_crossed"].max(), 1)
    ctrl_b = ba[ba.score == "ctrl_always_short"].iloc[0]
    macros["optBumpsCtrlShMid"] = _f(ctrl_b["sh_daily_mid"], 1)
    macros["optBumpsCtrlShX"] = _f(ctrl_b["sh_daily_crossed"], 1)

    # ---------------- write macros ----------------
    lines = [
        "% GENERATED FILE -- do not edit by hand.",
        "% Written by experiments/spxw_options_tex.py from results/spxw_pnl/*.csv.",
        "% Options-section macros: the delta-hedged 0DTE leg book, its cost floor,",
        "% regime/exit rules, and the time-till-close forecast structure.",
    ]
    for k in sorted(macros):
        lines.append(f"\\newcommand{{\\{_cs(k)}}}{{{macros[k]}}}")
    with open(
        os.path.join(GEN, "options_numbers.tex"), "w", encoding="utf-8", newline="\n"
    ) as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {len(macros)} macros + 4 table bodies to {GEN}", flush=True)
    for k in (
        "optThetaTenShMid",
        "optThetaTenShX",
        "optAlwaysShortSh",
        "optFinalExCrossShMid",
        "optFinalExCrossShX",
        "optCostElevenLo",
        "optTtcBjointNested"
        if "optTtcBjointNested" in macros
        else "optTtcBjointNested",
    ):
        print(f"  {k} = {macros.get(k, '?')}", flush=True)


if __name__ == "__main__":
    main()
