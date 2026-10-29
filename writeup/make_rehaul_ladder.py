"""The feature-count ladder (top-k exogenous series) of the pooled intraday forecasts at W = 3,000 days (2026-10-26;
FOMC-patched by default since 2026-10-28).

READS    default: results/ladder_2026-10-26/fomcfix/, the FOMC-PATCHED ladder (RULE.txt ADDENDUM 3; decision D13, the
         paper's design), as make_rehaul_channels.py reads it for Table tab:rh_channels_sparse, Panel B.
         --prepatch: results/ladder_2026-10-26/, the PRE-PATCH ladder (RULE.txt ADDENDUM 2), kept reachable.
         From the chosen directory: scores.csv, readings.csv, gates.json, rank_primary.csv (written by
         code/ladder_score.py, and for the patched one fomcfix/code/ff_*.py, from the CARC runs) and posthoc_vs_walk.csv
         (below); RULE.txt of results/ladder_2026-10-26 in both cases.  Every number below is read from those files;
         none is recomputed from forecasts here.
POST HOC posthoc_vs_walk.csv (not in the RULE): the monthly rungs against Table 1's every-bar walks of the full set and
         the HAR set, written by code/ladder_posthoc.py.  The patched run did not write one; `--posthoc` writes
         fomcfix/posthoc_vs_walk.csv with that script UNCHANGED, its results directory and walk templates pointed at
         fomcfix/ and at Table 1's patched walks (results/window_sweep_2014start_fomcfix/w3000,
         results/pooled_trees_final_2014start_fomcfix_w3000, results/pooled_rf_skx_2014start_fomcfix_w3000), with the
         script's own gate (each monthly rung's QLIKE = scores.csv to 1e-12).  (The walk_full / walk_har rows of
         fomcfix/scores.csv are the PRE-PATCH Table 1 walks, the scorer's reference template; not read here.)
GATES    the ladder's own per-model pass flag (gates.json); on the patched ladder a model whose flag is false only through
         GP (patch invariance) on a SECONDARY (look-ahead) rung other than S10 passes the gates of the rungs read here,
         as in make_rehaul_channels.py (2026-10-26: LightGBM S3, max |diff| 1.1e-13 on 154 rows).
DESIGN   (RULE.txt sections 1-6) each model refitted on its top k exogenous series only, k = 0 (the HAR benchmark set),
         1, 3, 5, 10, 20 and 46 (the full set); each series enters with its 12 columns; the 12 HAR-benchmark columns in
         every rung; LightGBM, XGBoost, the random forest on oneDAL, ridge, lasso, elastic net with the frozen
         configurations / penalties; monthly refits at 126 origins (every 21st session), each held for its block;
         PRIMARY ranking: one pre-sample fit at 2013-01-02 10:00 explained on the 2013 rows (trees: mean |TreeSHAP| of
         the series' 12 columns; linear: sum |beta| x s.d.), no look-ahead; SECONDARY (look-ahead, descriptive): the
         full-span pooled shares.  Scored on Table 1's 33,527 bars (2,579 days) and on their 16:00 bar; QLIKE; DM (HLN,
         Newey-West lag 8, daily average loss); retained gain R_k = (L_0 - L_k) / (L_0 - L_46) x 100.
WRITES   (tag "" = patched, the names the paper inputs; "_prepatch" with --prepatch)
         writeup/generated/rehaul_ladder_numbers<tag>.tex   \\rhld... and \\ladder... macros (definitions only)
         writeup/generated/rehaul_ladder_fig<tag>.tex       Figure fig:rh_ladder (A_fi.tex, app:rh_fi_ladder)
         writeup/figures/rehaul_ladder<tag>.pdf (and .png)
NAMES    \\rhld<Model><Span>...: Model Lgbm Xgb Rf Ridge Lasso Enet; Span Bar (every bar) / Close (16:00); rung words
         Full (46), Twenty, Ten, Five, Three, One, Zero (the HAR set).  R<k> retained gain (%), P<k>Full the DM p of the
         rung against the full set, Kninety the smallest k keeping 90 %, Verdict the RULE's reading.  Aggregates over
         the trees (LightGBM, XGBoost, forest) and the linear models (ridge, lasso, elastic net): ...Min / ...Max.

Run:  python writeup/make_rehaul_ladder.py [--prepatch] [--posthoc]     (repository root)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import make_rehaul_fi as fi  # noqa: E402  (series names and figure style; importing it only defines them)

LAD_PRE = ROOT / "results" / "ladder_2026-10-26"  # the pre-patch ladder (RULE.txt ADDENDUM 2); RULE.txt for both
LAD_FIX = LAD_PRE / "fomcfix"  # the FOMC-patched ladder (RULE.txt ADDENDUM 3, D13): the paper's design
LAD, TAG = LAD_FIX, ""  # set by the command line (--prepatch: LAD_PRE, "_prepatch")
# Table 1's every-bar walks on the patched design (make_rehaul_channels.py's "Table 1" inputs): the post hoc's templates
WALK_FIX = {
    "lin": ROOT / "results" / "window_sweep_2014start_fomcfix" / "w3000" / "forecasts.parquet",
    "trees": ROOT / "results" / "pooled_trees_final_2014start_fomcfix_w3000" / "forecasts.parquet",
    "rf": ROOT / "results" / "pooled_rf_skx_2014start_fomcfix_w3000" / "forecasts.parquet",
}
LABEL = {
    "": "PATCHED design: the FOMC fix adopted (FOMC_SCHEDULE_2020_FIX, FOMC_PROXIMITY_NO_WINSOR; RULE.txt ADDENDUM 3).",
    "_prepatch": "PRE-PATCH design: the FOMC calendar features before the FOMC fix (RULE.txt ADDENDUM 2).",
}
CAPTION = {
    "": "FOMC-patched design.",
    "_prepatch": r"Pre-patch design: the FOMC calendar features before the fix of Section~\ref{sec:rh_data}.",
}
GEN = HERE / "generated"
FIG = HERE / "figures"
HEAD = "% AUTO-GENERATED by writeup/make_rehaul_ladder.py -- do not edit; re-run the script."

KS = (0, 1, 3, 5, 10, 20, 46)
KTAG = {0: "Zero", 1: "One", 3: "Three", 5: "Five", 10: "Ten", 20: "Twenty", 46: "Full"}
TREES = ("lgbm", "xgb", "rf_skx")
LINS = ("ridge", "lasso", "enet")
MODELS = TREES + LINS
MTAG = {"lgbm": "Lgbm", "xgb": "Xgb", "rf_skx": "Rf", "ridge": "Ridge", "lasso": "Lasso", "enet": "Enet"}
MNAME = {
    "lgbm": "LightGBM",
    "xgb": "XGBoost",
    "rf_skx": "Random forest",
    "ridge": "Ridge",
    "lasso": "Lasso",
    "enet": "Elastic net",
}
# the Section 8 figures' encoding (make_rehaul_fi_pooled.py COLOR / MARK): Okabe-Ito, a marker per model
COLOR = {
    "lgbm": fi.OKABE[0],
    "xgb": fi.OKABE[1],
    "rf_skx": fi.OKABE[2],
    "ridge": fi.OKABE[6],
    "lasso": fi.OKABE[3],
    "enet": fi.OKABE[4],
}
MARK = {"lgbm": "o", "xgb": "s", "rf_skx": "^", "ridge": "D", "lasso": "v", "enet": "P"}
STAG = {"all": "Bar", "c16": "Close"}
VERDICT = {
    "CONCENTRATED": "concentrated",
    "INTERMEDIATE": "intermediate",
    "DENSE AND WEAK": "dense and weak",
    "NO GAIN TO RETAIN": "no gain to retain",
}
N_BARS, N_DAYS, N_ORIGINS, HOLD, N_SERIES, W_DAYS = 33527, 2579, 126, 21, 46, 3000


def thousands(x: int) -> str:
    return f"{x:,}".replace(",", "{,}")


def num(x: float, nd: int = 1) -> str:
    s = f"{x:.{nd}f}"
    if float(s) == 0:
        s = s.lstrip("-")
    return "\\ensuremath{-}" + s[1:] if s.startswith("-") else s


def pval(p: float) -> str:
    """A p-value as the paper prints it (three decimals; below 0.001 as <0.001)."""
    return "\\ensuremath{<}0.001" if p < 0.001 else f"{p:.3f}"


def gate_ok(g: dict) -> bool:
    """The ladder's own pass flag; or (patched ladder) false only through GP on secondary rungs this file does not read
    (every rung but S10), with the core gate (trees: GT, the full rung = the walk at 126 / 126 origins; linear: GL1)."""
    if bool(g.get("pass", False)):
        return True
    core = g.get("GT_origin_bit_equal") == g.get("GT_origin_n") == 126 if "GT_origin_n" in g else g.get("GL1_pass")
    gp = [str(x).split(":")[0] for x in g.get("GP_differ", [])]
    return bool(core) and bool(gp) and all(x.startswith("S") and x != "S10" for x in gp)


def posthoc_patched() -> None:
    """Write fomcfix/posthoc_vs_walk.csv: code/ladder_posthoc.py unchanged, its RES and TEMPL pointed at the patched
    ladder and Table 1's patched walks (the script asserts each monthly rung's QLIKE = scores.csv to 1e-12)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("ladder_posthoc", LAD_PRE / "code" / "ladder_posthoc.py")
    assert spec is not None and spec.loader is not None
    lp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lp)
    for f in WALK_FIX.values():
        assert f.is_file(), f"patched Table 1 walk missing: {f}"
    setattr(lp, "RES", LAD_FIX)  # read by lp.main() at call time
    setattr(lp, "TEMPL", dict(WALK_FIX))
    assert lp.main() == 0
    print(f"wrote {(LAD_FIX / 'posthoc_vs_walk.csv').relative_to(ROOT)}")


def main() -> None:
    S = pd.read_csv(LAD / "scores.csv")
    R = pd.read_csv(LAD / "readings.csv")
    gates = json.loads((LAD / "gates.json").read_text())
    rank = pd.read_csv(LAD / "rank_primary.csv")
    assert (LAD / "posthoc_vs_walk.csv").is_file(), f"{LAD / 'posthoc_vs_walk.csv'} missing: run with --posthoc"
    walk = pd.read_csv(LAD / "posthoc_vs_walk.csv")
    rule = (LAD_PRE / "RULE.txt").read_text(encoding="utf-8")
    if TAG == "_prepatch":
        assert "PRE-PATCH" in rule, "the ladder's RULE no longer labels the design pre-patch: check the caption"
    else:
        assert "ADDENDUM 3" in rule and "RERUN ON THE PATCHED DESIGN" in rule, "the RULE has no patched rerun"
    out_num, out_fig = GEN / f"rehaul_ladder_numbers{TAG}.tex", GEN / f"rehaul_ladder_fig{TAG}.tex"
    M: list[tuple[str, str, str]] = []

    def add(name: str, value: str, note: str) -> None:
        assert name.isalpha(), name
        M.append(("rhld" + name, value, note))

    # ---------------------------------------------------------------- design and gates
    add("NOrigins", str(N_ORIGINS), "monthly refit origins (every 21st session), each fit held for its block")
    add("HoldSessions", str(HOLD), "sessions a refit is held")
    add("NSeries", str(N_SERIES), "exogenous series (the full set's rung)")
    add("WindowDays", thousands(W_DAYS), "rolling window, trading days (39,000 bars)")
    add("NBars", thousands(N_BARS), "scored bars (Table 1's)")
    add("NDays", thousands(N_DAYS), "evaluation days")
    add("HacLag", "8", "Newey-West lag of the DM tests (2,579 days)")
    add("RankYear", "2013", "the year on which the primary ranking's fit is explained (fold 3's validation year)")
    n_pass = sum(gate_ok(g) for g in gates.values())
    assert set(gates) == set(MODELS), sorted(gates)
    gp_note = "; ".join(f"{m} {g['GP_differ']}" for m, g in gates.items() if not g.get("pass", False))
    add("NGates", str(len(gates)), "gated models (trees: bit-equal full rung; linear: replay = walk)")
    add(
        "NGatesPass",
        str(n_pass),
        "gates passed on the rungs read" + (f" (GP, secondary: {gp_note})" if gp_note else ""),
    )
    assert n_pass == len(gates), "a ladder gate failed: the numbers are flagged in the RULE's sense"

    sc = S[S["ranking"].isin(["P", "both"])].copy()  # the primary ranking (k = 46 and k = 0 are shared: "both")
    rows = {}
    for m in MODELS:
        for sp in ("all", "c16"):
            f = sc[(sc.model == m) & (sc.span == sp)].set_index("k")
            assert sorted(f.index.astype(int)) == sorted(KS), (m, sp, sorted(f.index))
            rows[(m, sp)] = f
            r = R[(R.model == m) & (R.ranking == "primary") & (R.span == sp)].iloc[0]
            tag = MTAG[m] + STAG[sp]
            q0, q46 = float(f.loc[0, "qlike"]), float(f.loc[46, "qlike"])
            for k in KS:
                add(f"{tag}Qlike{KTAG[k]}", f"{float(f.loc[k, 'qlike']):.4f}", f"{m} {sp}: QLIKE of the rung k = {k}")
                add(
                    f"{tag}Pct{KTAG[k]}",
                    num(100.0 * (float(f.loc[k, "qlike"]) / q0 - 1.0)),
                    f"{m} {sp}: % QLIKE change of rung k = {k} against the k = 0 rung",
                )
            for k in (1, 3, 5, 10, 20):
                add(f"{tag}R{KTAG[k]}", num(float(f.loc[k, "retained_pct"]), 0), f"{m} {sp}: retained gain R_k, k={k}")
                add(f"{tag}P{KTAG[k]}Full", pval(float(f.loc[k, "p_vs_full"])), f"{m} {sp}: DM p, rung k={k} vs k=46")
                add(f"{tag}Dm{KTAG[k]}Full", num(float(f.loc[k, "dm_vs_full"]), 2), f"{m} {sp}: DM, rung k={k} vs k=46")
            add(f"{tag}PFullHar", pval(float(f.loc[46, "p_vs_har"])), f"{m} {sp}: DM p, full set vs k = 0")
            add(f"{tag}Kninety", str(int(r["k90"])), f"{m} {sp}: smallest k keeping 90% of the full set's gain")
            add(f"{tag}Verdict", VERDICT[str(r["verdict"])], f"{m} {sp}: the RULE's reading (section 6)")
            kbest = int(f["qlike"].astype(float).idxmin())
            add(f"{tag}Kbest", str(kbest), f"{m} {sp}: the rung with the lowest QLIKE")
            add(f"{tag}GainFull", num(1e3 * (q0 - q46), 2), f"{m} {sp}: QLIKE gain of the full set over k=0, x 10^3")

    # ---------------------------------------------------------------- aggregates (primary ranking, every bar)
    def rk(m: str, k: int, sp: str = "all") -> float:
        return float(rows[(m, sp)].loc[k, "retained_pct"])

    for grp, ms in (("Tree", TREES), ("Lin", LINS)):
        for k in (1, 3, 5, 10, 20):
            v = [rk(m, k) for m in ms]
            add(f"{grp}R{KTAG[k]}Min", num(min(v), 0), f"{grp} models, every bar: lowest R_k at k = {k} (%)")
            add(f"{grp}R{KTAG[k]}Max", num(max(v), 0), f"{grp} models, every bar: highest R_k at k = {k} (%)")
        for k in (5, 10, 20):
            p = [float(rows[(m, "all")].loc[k, "p_vs_full"]) for m in ms]
            add(f"{grp}P{KTAG[k]}FullMin", pval(min(p)), f"{grp}, every bar: lowest DM p of k = {k} vs the full set")
            add(f"{grp}P{KTAG[k]}FullMax", pval(max(p)), f"{grp}, every bar: highest DM p of k = {k} vs the full set")
        kb = [int(rows[(m, "all")]["qlike"].astype(float).idxmin()) for m in ms]
        add(f"{grp}KbestMin", str(min(kb)), f"{grp}, every bar: smallest best rung")
        add(f"{grp}KbestMax", str(max(kb)), f"{grp}, every bar: largest best rung")
        k9 = [int(R[(R.model == m) & (R.ranking == "primary") & (R.span == "all")]["k90"].iloc[0]) for m in ms]
        add(f"{grp}KninetyMin", str(min(k9)), f"{grp}, every bar: smallest k90")
        add(f"{grp}KninetyMax", str(max(k9)), f"{grp}, every bar: largest k90")
        # rungs below the full set (every bar): reduced rungs with a lower QLIKE than k = 46
        nb = [
            int(
                sum(
                    float(rows[(m, "all")].loc[k, "qlike"]) < float(rows[(m, "all")].loc[46, "qlike"]) for k in KS[1:-1]
                )
            )
            for m in ms
        ]
        add(f"{grp}NRungsBelowFullMin", str(min(nb)), f"{grp}, every bar: fewest reduced rungs below the full set")
        add(f"{grp}NRungsBelowFullMax", str(max(nb)), f"{grp}, every bar: most reduced rungs below the full set")
    # the text's claims, checked (A_fi.tex, app:rh_fi_ladder): the trees' top 10 keep most of the gain and their top 5
    # do not; the linear models' 10-series rungs beat the full set (DM p < 0.05) and their 5-series rungs are no worse
    # than it (R_5 > 100, or DM p > 0.10 against the full set). 2026-10-28 (patched ladder): was "the linear models'
    # 5- and 10-series rungs beat the full set" (pre-patch R_5 132-211); patched, the elastic net's R_5 is 99.
    assert min(rk(m, 10) for m in TREES) >= 90 and max(rk(m, 5) for m in TREES) < 50, "trees: claim 'top 10 / top 5'"
    assert min(rk(m, 10) for m in LINS) > 100, "linear: claim '10 series beat the full set'"
    assert max(float(rows[(m, "all")].loc[10, "p_vs_full"]) for m in LINS) < 0.05, "linear: claim 'top 10 > full'"
    assert all(rk(m, 5) > 100 or float(rows[(m, "all")].loc[5, "p_vs_full"]) > 0.10 for m in LINS), (
        "linear: claim '5 series no worse than the full set'"
    )
    # the FOMC proximity series in the trees' primary ranking
    fr_ = rank[(rank.ranking == "primary") & (rank.series == "fomc_until_inv")].set_index("model")["rank"]
    tr = [int(fr_[m]) for m in TREES]
    add("TreeRankFomcUntilMin", str(min(tr)), "trees: best primary rank of fomc_until_inv")
    add("TreeRankFomcUntilMax", str(max(tr)), "trees: worst primary rank of fomc_until_inv")
    assert 5 < min(tr) and max(tr) <= 10, "claim: fomc_until_inv enters between the top-5 and top-10 rungs"
    for m in MODELS:
        top = rank[(rank.model == m) & (rank.ranking == "primary")].sort_values("rank")["series"].iloc[0]
        add(f"{MTAG[m]}TopOne", fi.grp(top), f"{m}: the top series of the primary ranking ({top})")
    # post hoc (labelled; not in the RULE): the monthly top-10 rung against Table 1's every-bar walk of the full set
    for grp, ms in (("Tree", TREES), ("Lin", LINS)):
        p = [float(walk[(walk.model == m) & (walk.k == 10) & (walk.vs == "walk_full")]["p"].iloc[0]) for m in ms]
        d = [float(walk[(walk.model == m) & (walk.k == 10) & (walk.vs == "walk_full")]["dm"].iloc[0]) for m in ms]
        add(f"{grp}PTenVsWalkMin", pval(min(p)), f"post hoc, {grp}: lowest DM p, monthly top 10 vs the walk's full set")
        add(
            f"{grp}PTenVsWalkMax", pval(max(p)), f"post hoc, {grp}: highest DM p, monthly top 10 vs the walk's full set"
        )
        add(f"{grp}DmTenVsWalkMin", num(min(d), 2), f"post hoc, {grp}: lowest DM, monthly top 10 vs walk full")
        add(f"{grp}DmTenVsWalkMax", num(max(d), 2), f"post hoc, {grp}: highest DM, monthly top 10 vs walk full")
    for m in MODELS:  # the cost of the monthly refit: the full rung against the every-bar walk's full set
        w = walk[(walk.model == m) & (walk.k == 46) & (walk.vs == "walk_full")].iloc[0]
        add(
            f"{MTAG[m]}PctMonthlyVsWalk",
            num(100.0 * (float(w["qlike_monthly"]) / float(w["qlike_walk"]) - 1.0)),
            f"post hoc, {m}: % QLIKE of the monthly full rung against the every-bar walk's full set",
        )
    # secondary (look-ahead) ranking: the trees' top-10 retained gain, for the robustness sentence
    ss = S[S["ranking"].isin(["S", "both"])]
    for m in MODELS:
        f = ss[(ss.model == m) & (ss.span == "all")].set_index("k")
        add(f"{MTAG[m]}SecRTen", num(float(f.loc[10, "retained_pct"]), 0), f"{m}: R_10, secondary (look-ahead) ranking")

    # FRONT's names (abstract, introduction, conclusion; no "rh" prefix): aliases of the quantities above
    ten = [rk(m, 10) for m in TREES]
    kb_lin = sorted({int(rows[(m, "all")]["qlike"].astype(float).idxmin()) for m in LINS})

    def rng(lo: str, hi: str) -> str:
        return lo if lo == hi else f"{lo}--{hi}"

    alias = [
        ("ladderTopK", "10", "the rung FRONT quotes (top 10 series)"),
        ("ladderTreeRetainRng", rng(num(min(ten), 0), num(max(ten), 0)), "= rhldTreeRTenMin--rhldTreeRTenMax (%)"),
        ("ladderLinKstarRng", rng(str(kb_lin[0]), str(kb_lin[-1])), "= rhldLinKbestMin--rhldLinKbestMax"),
    ]
    # VI's names (08_feature_importance.tex): the every-bar DM p of the top 10 against the full set, without "Bar"
    for m in TREES:
        alias.append(
            (
                f"rhld{MTAG[m]}PTenFull",
                pval(float(rows[(m, "all")].loc[10, "p_vs_full"])),
                f"= rhld{MTAG[m]}BarPTenFull (VI's name)",
            )
        )
    M.extend(alias)

    names = [n for n, _, _ in M]
    assert len(set(names)) == len(names), "a macro name twice"
    lines = [
        HEAD,
        f"% \\rhld... macros of the feature-count ladder (Figure fig:rh_ladder; {LAD.relative_to(ROOT).as_posix()},",
        "% RULE.txt sections 1-6): each model refitted on its top k exogenous series, monthly refits at 126 origins,",
        "% W = 3,000 days, primary (pre-sample) ranking unless named Sec; scored on Table 1's bars. Names in the",
        f"% generator's docstring. {LABEL[TAG]}",
    ]
    for name, value, note in M:
        lines.append(f"\\providecommand{{\\{name}}}{{}}\\renewcommand{{\\{name}}}{{{value}}}  % {note}")
    GEN.mkdir(parents=True, exist_ok=True)
    out_num.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    figure(rows)
    floats = rf"""{HEAD}
% Figure fig:rh_ladder (Appendix app:rh_fi_ladder): the feature-count ladder, {LAD.relative_to(ROOT).as_posix()}
% (RULE.txt sections 1-6). {LABEL[TAG]}
\begin{{figure}}[tbp]
  \centering
  \includegraphics[width=\linewidth]{{figures/rehaul_ladder{TAG}.pdf}}
  \caption{{How many exogenous series carry the gain. Each model is refitted on its top $k$ exogenous series only
    ($k = 0$ is the HAR benchmark set, $k = \rhldNSeries$ the full set), ranked once by the model's own importance
    measure on a fit explained on \rhldRankYear{{}} data, before the evaluation span. Vertical axis: the QLIKE of the
    rung relative to the same model's $k = 0$ rung, in percent (negative: more accurate), on the \rhldNBars{{}} bars
    of Table~\ref{{tab:rh_oos}} (left) and on their 16:00 bar (right). Monthly refits at \rhldNOrigins{{}} origins, each
    held for \rhldHoldSessions{{}} sessions, window $W = \rhldWindowDays$ trading days, hyperparameters frozen.
    {CAPTION[TAG]}}}
  \label{{fig:rh_ladder}}
\end{{figure}}
"""
    out_fig.write_text(floats, encoding="utf-8", newline="\n")
    print(
        f"{LAD.relative_to(ROOT).as_posix()}: wrote {len(M)} macros to {out_num.relative_to(ROOT)}, "
        f"{out_fig.relative_to(ROOT)}, figures/rehaul_ladder{TAG}.pdf"
    )


def figure(rows: dict) -> None:
    import matplotlib.pyplot as plt

    ink, ink2, gridc = "#0b0b0b", "#52514e", "#e4e3df"
    FIG.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(KS))
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.9), sharex=True)
    for ax, sp, title in zip(axes, ("all", "c16"), ("Every bar", "16:00 bar")):
        ax.axhline(0, color=ink2, lw=0.8)
        for m in MODELS:
            f = rows[(m, sp)]
            q0 = float(f.loc[0, "qlike"])
            y = [100.0 * (float(f.loc[k, "qlike"]) / q0 - 1.0) for k in KS]
            ax.plot(
                x,
                y,
                color=COLOR[m],
                marker=MARK[m],
                ms=4,
                lw=1.4,
                ls="-" if m in TREES else "--",
                label=MNAME[m],
            )
        ax.set_xticks(x, [str(k) for k in KS])
        ax.set_xlabel("exogenous series $k$", fontsize=8, color=ink2)
        ax.set_title(title, fontsize=9, color=ink)
        ax.grid(axis="y", color=gridc, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(ink2)
        ax.tick_params(colors=ink2, labelsize=7)
    axes[0].set_ylabel("QLIKE vs $k = 0$ (%)", fontsize=8)
    h, lb = axes[0].get_legend_handles_labels()
    fig.legend(h, lb, loc="upper center", ncol=6, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIG / f"rehaul_ladder{TAG}.pdf")
    fig.savefig(FIG / f"rehaul_ladder{TAG}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--prepatch", action="store_true", help="the pre-patch ladder; writes the *_prepatch names")
    ap.add_argument("--posthoc", action="store_true", help="first write fomcfix/posthoc_vs_walk.csv (patched only)")
    a = ap.parse_args()
    if a.prepatch:
        assert not a.posthoc, "--posthoc writes the patched ladder's post hoc only (the pre-patch one is on file)"
        LAD, TAG = LAD_PRE, "_prepatch"
    if a.posthoc:
        posthoc_patched()
    main()
