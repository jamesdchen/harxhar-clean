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
         writeup/generated/rehaul_ladder_fig<tag>.tex       Figure fig:rh_ladder_monthly<tag> (2026-10-29,
             Q-2026-10-29-30: no longer input in A_fi.tex, whose fig:rh_ladder is the every-bar figure below; the
             source of the 16:00 figure of app:rh_closing; was fig:rh_ladder)
         writeup/figures/rehaul_ladder<tag>.pdf (and .png)
NAMES    \\rhld<Model><Span>...: Model Lgbm Xgb Rf Ridge Lasso Enet; Span Bar (every bar) / Close (16:00); rung words
         Full (46), Twenty, Ten, Five, Three, One, Zero (the HAR set).  R<k> retained gain (%), P<k>Full the DM p of the
         rung against the full set, Kninety the smallest k keeping 90 %, Verdict the RULE's reading.  Aggregates over
         the trees (LightGBM, XGBoost, forest) and the linear models (ridge, lasso, elastic net): ...Min / ...Max.

EVERY BAR (--everybar; 2026-10-29): the same ladder refitted at EVERY BAR (results/ablations_2014start_w3000/RULE.txt
         ADDENDUM K; every output labelled "every-bar refit, requested by the author after the monthly-vs-every-bar
         comparison; not pre-registered").  READS results/ablations_2014start_w3000/scores_fomcfix/ladall_scores.csv,
         ladall_readings.csv and ladall_checks.json (its "pass" flag must be true), written by the ladder's scorer
         (ADDENDUM K (h)): LightGBM, XGBoost, ridge, lasso, elastic net (no random forest, so no forest macros), the
         primary ranking, spans all (every bar) / ord (ORDINARY days) / ev (EVENT days: the scheduled FOMC statement
         days); k = 0 and k = 46 are Table 1's own every-bar walks (HAR set, patched full set).  Also reads, from the
         monthly fomcfix/ ladder: scores.csv (the figure's dashed monthly rungs, and the R_monthly check),
         posthoc_vs_walk.csv (check: the every-bar k = 0 / 46 QLIKE = Table 1's walks there) and rank_primary.csv (the
         every-bar rungs' ranking, ADDENDUM K (a)) and readings.csv (the monthly verdicts).  Since Q-2026-10-29-30
         this is the paper's ladder; the claims the text makes on it (Q-2026-10-29-31: A_fi.tex app:rh_fi_ladder, the
         introduction and the conclusion) are ASSERTED here and listed in the numbers file's header comment.
         WRITES writeup/generated/rehaul_ladder_numbers_everybar.tex, writeup/generated/rehaul_ladder_fig_everybar.tex
         (Figure fig:rh_ladder, the figure of app:rh_fi_ladder since 2026-10-29; was fig:rh_ladder_everybar),
         writeup/figures/rehaul_ladder_everybar.pdf (and .png): every-bar rungs solid, monthly rungs dashed (all-bars
         panel only: the monthly ladder has no day-type spans), QLIKE vs k = 0 (%).  The addendum K label stays in the
         files' header comments and the figure's metadata, not in the printed caption (Q-2026-10-29-30 (2)).
         NAMES: the monthly name with EB inserted after its prefix, \\rhld<X> -> \\rhldEB<X>, \\ladder<X> ->
         \\ladderEB<X>; Span tags Bar (every bar), Ord (ORDINARY days), Ev (EVENT days); aggregates Tree = LightGBM and
         XGBoost (no forest), Lin = ridge, lasso, elastic net, unsuffixed on every bar and Tree<Ord|Ev>... /
         Lin<Ord|Ev>... by day type.  Added (no monthly name): <Model><Span>Kfifty / Kseventyfive (k50 / k75),
         <Model>BarRMonthly<k> (the monthly R_k beside the every-bar one), <Model>BarPctMonthly<k> (% QLIKE of the
         monthly rung against the every-bar rung), NBars<Span>, NDays<Span>, HacLag<Span> (Ord / Ev).  R_k is n/a when
         L_46 >= L_0 (ADDENDUM K (h)).  Monthly names without an every-bar meaning (origins, gates, post hoc vs the
         walk, secondary ranking, the 16:00 bar) are not written.
         Added 2026-10-29 (Q-2026-10-29-30 / -31): LinPctMonthly<Full|Ten><Min|Max> (extremes of <Model>BarPctMonthly
         over the linear models), AllBarVerdict (the five every-bar verdicts, written only when they agree),
         LinVerdictMonthly (the three MONTHLY linear verdicts, written only when they agree); and, without EB because
         they are not every-bar quantities (written here so that the monthly file is unchanged): \\rhldPctMonthlyVsWalk
         <Min|Max> (extremes of the monthly file's \\rhld<Model>PctMonthlyVsWalk, all six models), \\rhldRuleKConc and
         \\rhldRulePct (the reading rule's bound on k90 for "concentrated" and its retained-share threshold, parsed from
         results/ladder_2026-10-26/RULE.txt section 6).

Run:  python writeup/make_rehaul_ladder.py [--prepatch] [--posthoc]     (repository root)
      python writeup/make_rehaul_ladder.py --everybar
      Either with --in-dir DIR (the ladder files read: the monthly ladder's dir, or the ladall_* dir with --everybar)
      and --out-dir DIR (writes DIR/generated/ and DIR/figures/ instead of writeup/generated/ and writeup/figures/).
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable
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
ROLE = {  # the monthly figure's use since Q-2026-10-29-30 (its header comment)
    "": "the monthly ladder; the source of the 16:00 figure of app:rh_closing",
    "_prepatch": "the monthly ladder on the pre-patch design; kept reachable, input nowhere",
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

# ---------------------------------------------------------------- the every-bar ladder (--everybar; RULE.txt ADDENDUM K)
EB_DIR = ROOT / "results" / "ablations_2014start_w3000" / "scores_fomcfix"  # ladall_scores.csv / _readings / _checks
EB_RULE = ROOT / "results" / "ablations_2014start_w3000" / "RULE.txt"
EB_IN = EB_DIR  # set by the command line (--in-dir)
EB_TAG = "_everybar"
EB_LABEL = "every-bar refit, requested by the author after the monthly-vs-every-bar comparison; not pre-registered"
EB_HEAD = HEAD.replace("make_rehaul_ladder.py", "make_rehaul_ladder.py --everybar")
EB_TREES = ("lgbm", "xgb")  # no random forest in the every-bar ladder (ADDENDUM K (b))
EB_MODELS = EB_TREES + LINS
EB_SPANS = ("all", "ord", "ev")
EB_STAG = {"all": "Bar", "ord": "Ord", "ev": "Ev"}
EB_TITLE = {"all": "All bars", "ord": "Ordinary days", "ev": "FOMC statement days"}
EB_SCOLS = (
    "model,ranking,k,rung,fitted_as,span,n_bars,qlike,mse,dm_vs_full,p_vs_full,dm_vs_har,p_vs_har,hac_lag,R,R_monthly"
).split(",")
EB_RCOLS = (
    "model,ranking,span,qlike_full,qlike_har,gain_full_vs_har,k50,k75,k90,verdict,rungs_below_full,R1,R3,R5,R10,R20,"
    "p20_vs_full,p10_vs_full,p5_vs_full,dm20_vs_full,dm10_vs_full,dm5_vs_full"
).split(",")
BARS_PER_DAY = 13  # Table 1's half-hour bars, 10:00 .. 16:00 (33,527 = 2,579 x 13)
NA = "n/a"


def rel(p: Path) -> str:
    """A path as the outputs print it: relative to the repository root when under it, else absolute."""
    for q in (p, p.resolve()):
        try:
            return q.relative_to(ROOT).as_posix()
        except ValueError:
            pass
    return p.resolve().as_posix()


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
        f"% \\rhld... macros of the feature-count ladder (Figure fig:rh_ladder_monthly{TAG}; {rel(LAD)},",
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
% Figure fig:rh_ladder_monthly{TAG} ({ROLE[TAG]}; not
% input in app:rh_fi_ladder, whose fig:rh_ladder is the every-bar ladder): the feature-count ladder, {rel(LAD)}
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
  \label{{fig:rh_ladder_monthly{TAG}}}
\end{{figure}}
"""
    out_fig.write_text(floats, encoding="utf-8", newline="\n")
    print(
        f"{rel(LAD)}: wrote {len(M)} macros to {rel(out_num)}, {rel(out_fig)}, {rel(FIG / f'rehaul_ladder{TAG}.pdf')}"
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


# ====================================================================== every-bar mode (--everybar; RULE.txt ADDENDUM K)
def na_num(x: float, nd: int = 1) -> str:
    return num(x, nd) if np.isfinite(x) else NA


def na_pval(p: float) -> str:
    return pval(p) if np.isfinite(p) else NA


def na_k(x: float) -> str:
    return str(int(x)) if np.isfinite(x) else NA


def finite(v: list[float]) -> list[float]:
    return [x for x in v if np.isfinite(x)]


def span_rng(lo: str, hi: str) -> str:
    return lo if lo == hi else f"{lo}--{hi}"


def main_everybar() -> None:
    """The ladder refitted at every bar (RULE.txt ADDENDUM K (h)): macros, float and figure under the *_everybar names.
    Every number is read from the scorer's ladall_* files (and, for the monthly comparison, the monthly ladder's)."""
    src = EB_IN
    S = pd.read_csv(src / "ladall_scores.csv")
    R = pd.read_csv(src / "ladall_readings.csv")
    checks = json.loads((src / "ladall_checks.json").read_text(encoding="utf-8"))
    for fname, df, cols in (("ladall_scores.csv", S, EB_SCOLS), ("ladall_readings.csv", R, EB_RCOLS)):
        miss = [c for c in cols if c not in df.columns]
        assert not miss, f"{src / fname}: columns missing against the input contract: {miss}"
    assert checks.get("pass") is True, f"ladall_checks.json: the every-bar ladder's checks failed: {checks.get('fail')}"
    rule = EB_RULE.read_text(encoding="utf-8")
    assert "ADDENDUM K" in rule and "NOT PRE-REGISTERED" in rule, "RULE.txt has no ADDENDUM K (the every-bar ladder)"
    S = S.assign(k=S["k"].astype(float).astype(int), model=S["model"].astype(str), span=S["span"].astype(str))
    R = R.assign(model=R["model"].astype(str), span=R["span"].astype(str))
    assert set(S.model) == set(R.model) == set(EB_MODELS), f"every-bar models {sorted(set(S.model))} / {EB_MODELS}"
    assert set(S.ranking) == set(R.ranking) == {"primary"}, (set(S.ranking), set(R.ranking))
    assert set(S.fitted_as) == {"everybar"}, set(S.fitted_as)
    assert set(S.span) == set(R.span) == set(EB_SPANS), (set(S.span), set(R.span))
    # the monthly ladder (patched): the figure's dashed rungs, R_monthly, the walks' QLIKE, the rungs' ranking
    mon = pd.read_csv(LAD_FIX / "scores.csv")
    mon = mon[mon["ranking"].isin(["P", "both"]) & (mon["span"] == "all")]
    walk = pd.read_csv(LAD_FIX / "posthoc_vs_walk.csv")
    rank = pd.read_csv(LAD_FIX / "rank_primary.csv")
    mread = pd.read_csv(LAD_FIX / "readings.csv")  # the monthly verdicts (LinVerdictMonthly, the forest's check)
    out_num, out_fig = GEN / f"rehaul_ladder_numbers{EB_TAG}.tex", GEN / f"rehaul_ladder_fig{EB_TAG}.tex"
    M: list[tuple[str, str, str]] = []

    def add(name: str, value: str, note: str) -> None:
        assert name.isalpha(), name
        M.append(("rhldEB" + name, value, note))

    # ---------------------------------------------------------------- design (bars and days by span, from the file)
    nbar: dict[str, int] = {}
    lag: dict[str, int] = {}
    for sp in EB_SPANS:
        nb_ = sorted(set(S.loc[S.span == sp, "n_bars"].astype(int)))
        lg_ = sorted(set(S.loc[S.span == sp, "hac_lag"].dropna().astype(int)))
        assert len(nb_) == 1 and len(lg_) == 1, (sp, nb_, lg_)
        nbar[sp], lag[sp] = nb_[0], lg_[0]
    assert nbar["all"] == N_BARS and nbar["ord"] + nbar["ev"] == N_BARS, nbar
    assert all(n % BARS_PER_DAY == 0 for n in nbar.values()), nbar
    nday = {sp: n // BARS_PER_DAY for sp, n in nbar.items()}
    assert nday["all"] == N_DAYS, nday
    add("NSeries", str(N_SERIES), "exogenous series (the full set's rung)")
    add("WindowDays", thousands(W_DAYS), "rolling window, trading days (39,000 bars), refitted at every bar")
    add("RankYear", "2013", "the year on which the primary ranking's fit is explained (the monthly ladder's ranking)")
    for sp in EB_SPANS:
        sx = "" if sp == "all" else EB_STAG[sp]
        add(f"NBars{sx}", thousands(nbar[sp]), f"scored bars, span {sp}")
        add(f"NDays{sx}", thousands(nday[sp]), f"evaluation days, span {sp} ({BARS_PER_DAY} bars a day)")
        add(f"HacLag{sx}", str(lag[sp]), f"Newey-West lag of the DM tests, span {sp}")

    # ---------------------------------------------------------------- per model and span
    rows: dict[tuple[str, str], pd.DataFrame] = {}
    reads: dict[tuple[str, str], pd.Series] = {}
    mrows: dict[tuple[str, str], pd.DataFrame] = {}
    for m in EB_MODELS:
        g = mon[mon.model == m].assign(k=lambda d: d["k"].astype(float).astype(int)).set_index("k")
        assert sorted(g.index) == sorted(KS) and g.index.is_unique, (m, "monthly", list(g.index))
        mrows[(m, "all")] = g
        for sp in EB_SPANS:
            f = S[(S.model == m) & (S.span == sp)].set_index("k")
            assert sorted(f.index) == sorted(KS) and f.index.is_unique, (m, sp, list(f.index))
            for k in KS:
                want = "har" if k == 0 else "full" if k == N_SERIES else f"lrung_{m}_{k}"
                assert str(f.loc[k, "rung"]) == want, (m, sp, k, f.loc[k, "rung"], want)
            rr = R[(R.model == m) & (R.span == sp)]
            assert len(rr) == 1, (m, sp, len(rr))
            r = rr.iloc[0]
            q0, q46 = float(f.loc[0, "qlike"]), float(f.loc[46, "qlike"])
            assert abs(float(r["qlike_har"]) - q0) < 1e-12 and abs(float(r["qlike_full"]) - q46) < 1e-12, (m, sp)
            for k in (1, 3, 5, 10, 20):
                a_, b_ = float(r[f"R{k}"]), float(f.loc[k, "R"])
                assert np.isclose(a_, b_, rtol=1e-9, atol=1e-9, equal_nan=True), (m, sp, k, "readings R vs scores R")
            if sp == "all":  # k = 0 / 46 are Table 1's every-bar walks: the monthly post hoc's walk QLIKE
                for k, vs in ((0, "walk_har"), (46, "walk_full")):
                    qw = sorted(set(walk.loc[(walk.model == m) & (walk.vs == vs), "qlike_walk"].astype(float)))
                    assert len(qw) == 1 and abs(qw[0] - float(f.loc[k, "qlike"])) < 1e-12, (m, k, vs, qw)
                for k in KS:  # R_monthly = the monthly ladder's R_k (primary, every bar), where the scorer gives one
                    rm = float(f.loc[k, "R_monthly"])
                    assert not np.isfinite(rm) or abs(rm - float(g.loc[k, "retained_pct"])) < 1e-9, (m, k, "R_monthly")
            rows[(m, sp)], reads[(m, sp)] = f, r
            tag = MTAG[m] + EB_STAG[sp]
            for k in KS:
                add(f"{tag}Qlike{KTAG[k]}", f"{float(f.loc[k, 'qlike']):.4f}", f"{m} {sp}: QLIKE of the rung k = {k}")
                add(
                    f"{tag}Pct{KTAG[k]}",
                    num(100.0 * (float(f.loc[k, "qlike"]) / q0 - 1.0)),
                    f"{m} {sp}: % QLIKE change of rung k = {k} against the k = 0 rung",
                )
            for k in (1, 3, 5, 10, 20):
                add(f"{tag}R{KTAG[k]}", na_num(float(f.loc[k, "R"]), 0), f"{m} {sp}: retained gain R_k, k={k}")
                add(
                    f"{tag}P{KTAG[k]}Full",
                    na_pval(float(f.loc[k, "p_vs_full"])),
                    f"{m} {sp}: DM p, rung k={k} vs k=46",
                )
                add(
                    f"{tag}Dm{KTAG[k]}Full",
                    na_num(float(f.loc[k, "dm_vs_full"]), 2),
                    f"{m} {sp}: DM, rung k={k} vs k=46",
                )
            add(f"{tag}PFullHar", na_pval(float(f.loc[46, "p_vs_har"])), f"{m} {sp}: DM p, full set vs k = 0")
            add(f"{tag}Kfifty", na_k(float(r["k50"])), f"{m} {sp}: smallest k keeping 50% of the full set's gain")
            add(f"{tag}Kseventyfive", na_k(float(r["k75"])), f"{m} {sp}: smallest k keeping 75% of the full set's gain")
            add(f"{tag}Kninety", na_k(float(r["k90"])), f"{m} {sp}: smallest k keeping 90% of the full set's gain")
            add(f"{tag}Verdict", VERDICT[str(r["verdict"])], f"{m} {sp}: the RULE's reading (ladder RULE section 6)")
            add(f"{tag}Kbest", str(int(f["qlike"].astype(float).idxmin())), f"{m} {sp}: the rung with the lowest QLIKE")
            add(f"{tag}GainFull", num(1e3 * (q0 - q46), 2), f"{m} {sp}: QLIKE gain of the full set over k=0, x 10^3")
            if sp == "all":  # the monthly ladder beside it (every bar; the monthly ladder has no day-type spans)
                for k in (1, 3, 5, 10, 20):
                    add(
                        f"{tag}RMonthly{KTAG[k]}",
                        na_num(float(g.loc[k, "retained_pct"]), 0),
                        f"{m}: the MONTHLY ladder's R_k, k={k} (= rhld{tag}R{KTAG[k]} of the monthly file)",
                    )
                for k in KS:
                    add(
                        f"{tag}PctMonthly{KTAG[k]}",
                        num(100.0 * (float(g.loc[k, "qlike"]) / float(f.loc[k, "qlike"]) - 1.0)),
                        f"{m}: % QLIKE of the monthly rung k = {k} against the every-bar rung k = {k}",
                    )

    # ---------------------------------------------------------------- aggregates (unsuffixed: every bar)
    def val(m: str, sp: str, k: int, col: str) -> float:
        return float(rows[(m, sp)].loc[k, col])

    def kbest(m: str, sp: str) -> int:
        return int(rows[(m, sp)]["qlike"].astype(float).idxmin())

    def lohi(name: str, v: list[float], fmt: Callable[[float], str], note: str) -> None:
        v = finite(v)
        add(f"{name}Min", fmt(min(v)) if v else NA, f"{note}: lowest")
        add(f"{name}Max", fmt(max(v)) if v else NA, f"{note}: highest")

    for grp, ms in (("Tree", EB_TREES), ("Lin", LINS)):
        for sp in EB_SPANS:
            gs = grp + ("" if sp == "all" else EB_STAG[sp])
            for k in (1, 3, 5, 10, 20):
                lohi(f"{gs}R{KTAG[k]}", [val(m, sp, k, "R") for m in ms], lambda x: num(x, 0), f"{grp} {sp}: R_{k} (%)")
            for k in (5, 10, 20):
                lohi(
                    f"{gs}P{KTAG[k]}Full",
                    [val(m, sp, k, "p_vs_full") for m in ms],
                    pval,
                    f"{grp} {sp}: DM p of k = {k} vs the full set",
                )
            lohi(f"{gs}Kbest", [float(kbest(m, sp)) for m in ms], lambda x: str(int(x)), f"{grp} {sp}: best rung")
            lohi(f"{gs}Kninety", [float(reads[(m, sp)]["k90"]) for m in ms], lambda x: str(int(x)), f"{grp} {sp}: k90")
            nbf = [float(sum(val(m, sp, k, "qlike") < val(m, sp, 46, "qlike") for k in KS[1:-1])) for m in ms]
            lohi(f"{gs}NRungsBelowFull", nbf, lambda x: str(int(x)), f"{grp} {sp}: reduced rungs below the full set")

    # ---------------------------------------------------------------- the ranking (ADDENDUM K (a): the monthly one)
    fr_ = rank[(rank.ranking == "primary") & (rank.series == "fomc_until_inv")].set_index("model")["rank"]
    tr = [int(fr_[m]) for m in EB_TREES]
    add("TreeRankFomcUntilMin", str(min(tr)), "LightGBM / XGBoost: best primary rank of fomc_until_inv")
    add("TreeRankFomcUntilMax", str(max(tr)), "LightGBM / XGBoost: worst primary rank of fomc_until_inv")
    for m in EB_MODELS:
        top = rank[(rank.model == m) & (rank.ranking == "primary")].sort_values("rank")["series"].iloc[0]
        add(f"{MTAG[m]}TopOne", fi.grp(top), f"{m}: the top series of the primary ranking ({top})")

    # ---------------------------------------------------------------- 2026-10-29 (Q-2026-10-29-30 / -31): the new text's
    # numbers.  The monthly hold's cost on the linear rungs: extremes of <Model>BarPctMonthly<k>, k = 46 and 10
    def pct_monthly(m: str, k: int) -> float:
        return 100.0 * (float(mrows[(m, "all")].loc[k, "qlike"]) / val(m, "all", k, "qlike") - 1.0)

    for k in (46, 10):
        lohi(
            f"LinPctMonthly{KTAG[k]}",
            [pct_monthly(m, k) for m in LINS],
            num,
            f"Lin: % QLIKE of the MONTHLY rung k = {k} against the every-bar rung (rhldEB<M>BarPctMonthly{KTAG[k]})",
        )
    # the readings: the five every-bar ones (span all), and the monthly ladder's (readings.csv, primary, span all)
    eb_verd = {m: str(reads[(m, "all")]["verdict"]) for m in EB_MODELS}
    if len(set(eb_verd.values())) == 1:
        add(
            "AllBarVerdict",
            VERDICT[eb_verd[EB_MODELS[0]]],
            "the five models' every-bar reading (written if they agree)",
        )
    mr = mread[(mread.ranking == "primary") & (mread.span == "all")].set_index("model")
    assert mr.index.is_unique and set(MODELS) <= set(mr.index), sorted(mr.index)
    mon_verd = {m: str(mr.loc[m, "verdict"]) for m in MODELS}
    mon_k90 = {m: int(mr.loc[m, "k90"]) for m in MODELS}
    if len({mon_verd[m] for m in LINS}) == 1:
        add(
            "LinVerdictMonthly",
            VERDICT[mon_verd[LINS[0]]],
            "MONTHLY ladder: the three linear models' reading (rhld<M>BarVerdict of the monthly file; written if they agree)",
        )
    # the reading rule (ladder RULE section 6), parsed; the readings files follow it
    lrule = (LAD_PRE / "RULE.txt").read_text(encoding="utf-8")
    m_thr = re.search(r"k90 = the smallest k with R_k >= (\d+) %", lrule)
    m_conc = re.search(r"CONCENTRATED\s+k90 <= (\d+)\s", lrule)
    assert m_thr is not None and m_conc is not None, "the ladder RULE's reading (section 6) not found"
    thr, kconc = int(m_thr.group(1)), int(m_conc.group(1))
    for (m, sp), rd_ in reads.items():
        k9_ = float(rd_["k90"])
        assert (str(rd_["verdict"]) == "CONCENTRATED") == bool(np.isfinite(k9_) and k9_ <= kconc), (
            m,
            sp,
            rd_["verdict"],
        )
    # the monthly full sets against Table 1's every-bar walks (the monthly file's rhld<M>PctMonthlyVsWalk, six models)
    w46 = walk[(walk.k == 46) & (walk.vs == "walk_full")].set_index("model")
    assert w46.index.is_unique and set(w46.index) == set(MODELS), sorted(w46.index)
    pmw = {m: 100.0 * (float(w46.loc[m, "qlike_monthly"]) / float(w46.loc[m, "qlike_walk"]) - 1.0) for m in MODELS}
    topk = 10  # the rung FRONT quotes (\ladderEBTopK)
    # names without EB: not every-bar quantities, written here so that the monthly file is unchanged (Q-2026-10-29-30)
    plain = [
        (
            "rhldPctMonthlyVsWalkMin",
            num(min(pmw.values())),
            "MONTHLY ladder, post hoc: lowest rhld<M>PctMonthlyVsWalk over the six models (% QLIKE of the monthly full "
            "rung against Table 1's every-bar full set)",
        ),
        (
            "rhldPctMonthlyVsWalkMax",
            num(max(pmw.values())),
            "MONTHLY ladder, post hoc: highest rhld<M>PctMonthlyVsWalk",
        ),
        ("rhldRuleKConc", str(kconc), "ladder RULE section 6: a model reads concentrated if k90 <= this"),
        ("rhldRulePct", str(thr), "ladder RULE section 6: k90 = the smallest k whose R_k reaches this share (%)"),
    ]

    # ---------------------------------------------------------------- the text's claims (Q-2026-10-29-31), asserted
    def rk(m: str, k: int) -> float:
        return val(m, "all", k, "R")

    def pk(m: str, k: int) -> float:
        return val(m, "all", k, "p_vs_full")

    def dk(m: str, k: int) -> float:
        return val(m, "all", k, "dm_vs_full")

    def show(f: Callable[[str], str], ms: tuple[str, ...]) -> str:
        return ", ".join(f"{m} {f(m)}" for m in ms)

    def r_(k: int, ms: tuple[str, ...] = LINS) -> str:
        return f"R_{k} " + show(lambda m: f"{rk(m, k):.0f}", ms)

    def p_(k: int, ms: tuple[str, ...] = LINS) -> str:
        return f"p_{k} / DM " + show(lambda m: f"{pk(m, k):.3g} / {dk(m, k):+.2f}", ms)

    nbf_lin = {m: int(sum(val(m, "all", k, "qlike") < val(m, "all", 46, "qlike") for k in KS[1:-1])) for m in LINS}
    k90_eb = {m: float(reads[(m, "all")]["k90"]) for m in EB_MODELS}
    by_type = {f"{m} ord": str(reads[(m, "ord")]["verdict"]) for m in EB_TREES} | {
        f"{m} ev": str(reads[(m, "ev")]["verdict"]) for m in LINS
    }
    pm46, pm10 = [pct_monthly(m, 46) for m in LINS], [pct_monthly(m, 10) for m in LINS]
    claims = [
        (
            "linear: the top 10 and the top 20 keep more than 100% of the gain (rhldEBLinRTenMin, rhldEBLinRTwentyMin)",
            min(rk(m, 10) for m in LINS) > 100 and min(rk(m, 20) for m in LINS) > 100,
            f"{r_(10)}; {r_(20)}",
        ),
        (
            "linear: two reduced rungs below the full set at the fewest (rhldEBLinNRungsBelowFullMin = 2)",
            min(nbf_lin.values()) == 2,
            show(lambda m: str(nbf_lin[m]), LINS),
        ),
        (
            "ridge: the top 10 significantly more accurate than the full set (DM p < 0.05, DM < 0)",
            pk("ridge", 10) < 0.05 and dk("ridge", 10) < 0,
            p_(10, ("ridge",)),
        ),
        (
            "lasso, elastic net: the top 10 against the full set not significant (DM p >= 0.05)",
            all(pk(m, 10) >= 0.05 for m in ("lasso", "enet")),
            p_(10, ("lasso", "enet")),
        ),
        (
            "linear: the top 20 against the full set not significant in any class (rhldEBLinPTwentyFullMin >= 0.05)",
            min(pk(m, 20) for m in LINS) >= 0.05,
            p_(20),
        ),
        (
            "lasso, elastic net: the full set significantly more accurate than the top 5 (DM p < 0.05, DM > 0)",
            all(pk(m, 5) < 0.05 and dk(m, 5) > 0 for m in ("lasso", "enet")),
            p_(5, ("lasso", "enet")),
        ),
        (
            "linear: the top 5 keep less than 100% of the gain (rhldEBLinRFiveMax < 100)",
            max(rk(m, 5) for m in LINS) < 100,
            r_(5),
        ),
        (
            "boosted trees: the full set significantly more accurate than the top 10 and the top 20 "
            "(rhldEBTreePTenFullMax, rhldEBTreePTwentyFullMax < 0.05, DM > 0)",
            all(pk(m, k) < 0.05 and dk(m, k) > 0 for m in EB_TREES for k in (10, 20)),
            f"{p_(10, EB_TREES)}; {p_(20, EB_TREES)}",
        ),
        (
            f"k90 = ladderEBTopK ({topk}) in all five models (rhldEBTree/LinKninetyMin = Max)",
            all(k90_eb[m] == topk for m in EB_MODELS),
            show(lambda m: f"{k90_eb[m]:.0f}", EB_MODELS),
        ),
        (
            f"every model's top 5 keep less than rhldRulePct ({thr}%) of its gain",
            all(rk(m, 5) < thr for m in EB_MODELS),
            r_(5, EB_MODELS),
        ),
        (
            "the five every-bar readings agree and are intermediate (rhldEBAllBarVerdict)",
            set(eb_verd.values()) == {"INTERMEDIATE"},
            show(lambda m: eb_verd[m], EB_MODELS),
        ),
        (
            "LightGBM and XGBoost on ordinary days read as the linear models on FOMC statement days "
            "(rhldEBLgbmOrdVerdict)",
            len(set(by_type.values())) == 1,
            ", ".join(f"{k} {v}" for k, v in by_type.items()),
        ),
        (
            "the forest's monthly reading = LightGBM's and XGBoost's, monthly and every bar (rhldRfBarVerdict)",
            all(mon_verd["rf_skx"] == mon_verd[m] == eb_verd[m] for m in EB_TREES),
            f"monthly {show(lambda m: mon_verd[m], TREES)}; every bar {show(lambda m: eb_verd[m], EB_TREES)}",
        ),
        (
            "monthly: the three linear readings agree, with one k90 (rhldEBLinVerdictMonthly, rhldLinKninetyMax)",
            len({mon_verd[m] for m in LINS}) == 1 and len({mon_k90[m] for m in LINS}) == 1,
            show(lambda m: f"{mon_verd[m]} k90 {mon_k90[m]}", LINS),
        ),
        (
            "boosted trees: fomc_until_inv ranked between the top-5 and top-10 rungs (rhldEBTreeRankFomcUntil)",
            5 < min(tr) and max(tr) <= 10,
            f"ranks {tr}",
        ),
        (
            "a month's hold raises the linear full sets' loss (rhldEBLinPctMonthlyFullMin > 0) by more than any "
            "linear 10-series rung's (rhldEBLinPctMonthlyTenMax < FullMin)",
            min(pm46) > 0 and max(pm10) < min(pm46),
            f"full {', '.join(f'{x:.2f}' for x in pm46)}; top 10 {', '.join(f'{x:.2f}' for x in pm10)}",
        ),
        (
            "monthly: each full set less accurate than Table 1's every-bar full set (rhldPctMonthlyVsWalkMin > 0)",
            min(pmw.values()) > 0,
            show(lambda m: f"{pmw[m]:.2f}", MODELS),
        ),
    ]
    bad = [c for c, ok, _ in claims if not ok]
    assert not bad, f"the text's ladder claims (Q-2026-10-29-31) fail on these numbers: {bad}"
    claim_lines = [f"%   [HOLDS] {c} ({d})" for c, _, d in claims]

    # ---------------------------------------------------------------- FRONT's and VI's aliases, with EB after the prefix
    ten = finite([rk(m, 10) for m in EB_TREES])
    kb_lin = sorted({kbest(m, "all") for m in LINS})
    M.append(("ladderEBTopK", str(topk), "the rung FRONT quotes (top 10 series)"))
    M.append(
        (
            "ladderEBTreeRetainRng",
            span_rng(num(min(ten), 0), num(max(ten), 0)) if ten else NA,
            "= rhldEBTreeRTenMin--rhldEBTreeRTenMax (%)",
        )
    )
    M.append(
        ("ladderEBLinKstarRng", span_rng(str(kb_lin[0]), str(kb_lin[-1])), "= rhldEBLinKbestMin--rhldEBLinKbestMax")
    )
    for m in EB_TREES:
        M.append((f"rhldEB{MTAG[m]}PTenFull", na_pval(pk(m, 10)), f"= rhldEB{MTAG[m]}BarPTenFull (VI's name)"))
    M.extend(plain)

    names = [n for n, _, _ in M]
    assert len(set(names)) == len(names), "a macro name twice"
    clabel = " ".join(str(checks.get("label", "")).split())
    lines = [
        EB_HEAD,
        f"% Label: {EB_LABEL}.",
        "% \\rhldEB... / \\ladderEB... macros of the feature-count ladder REFITTED AT EVERY BAR (Figure fig:rh_ladder,",
        f"% the paper's ladder since 2026-10-29; {rel(src)}/ladall_scores.csv, ladall_readings.csv,",
        "% ladall_checks.json; results/ablations_2014start_w3000/RULE.txt ADDENDUM K): each model refitted at every bar on",
        "% its top k exogenous series (its own primary ranking, results/ladder_2026-10-26/fomcfix/rank_primary.csv),",
        "% W = 3,000 days; k = 0 and k = 46 are Table 1's every-bar walks (the HAR set; the patched full set); LightGBM,",
        "% XGBoost, ridge, lasso, elastic net (no random forest); scored on Table 1's bars. FOMC-patched design.",
        "% NAMING RULE: each name is the monthly file's (rehaul_ladder_numbers.tex) with EB inserted after its prefix:",
        "% \\rhld<X> -> \\rhldEB<X>, \\ladder<X> -> \\ladderEB<X>. Span tags: Bar (every bar), Ord (ORDINARY days), Ev",
        "% (EVENT days, the scheduled FOMC statement days). Tree = LightGBM and XGBoost (no forest), Lin = ridge, lasso,",
        "% elastic net; their ...Min / ...Max aggregates are unsuffixed on every bar and Tree<Ord|Ev>... / Lin<Ord|Ev>...",
        "% by day type. Added (no monthly name): <Model><Span>Kfifty / Kseventyfive (k50 / k75), <Model>BarRMonthly<k> (the",
        "% monthly ladder's R_k), <Model>BarPctMonthly<k> (% QLIKE of the monthly rung against the every-bar rung),",
        "% LinPctMonthly<Full|Ten><Min|Max>, AllBarVerdict, LinVerdictMonthly, NBars / NDays / HacLag with Ord / Ev.",
        "% n/a: R_k undefined (L_46 >= L_0) or no finite value. Without EB (not every-bar quantities; kept out of the",
        "% monthly file so that it is unchanged): \\rhldPctMonthlyVsWalkMin/Max (monthly ladder, six models),",
        "% \\rhldRuleKConc and \\rhldRulePct (the reading rule, results/ladder_2026-10-26/RULE.txt section 6).",
        f"% Scorer's checks (ladall_checks.json): pass.{(' Label: ' + clabel) if clabel else ''}",
        "% The text's ladder claims (Q-2026-10-29-31: A_fi.tex app:rh_fi_ladder, 01_introduction.tex, 09_conclusion.tex)",
        "% on these numbers, every bar unless named (asserted: a rerun that breaks one writes nothing):",
        *claim_lines,
    ]
    for name, value, note in M:
        lines.append(f"\\providecommand{{\\{name}}}{{}}\\renewcommand{{\\{name}}}{{{value}}}  % {note}")
    GEN.mkdir(parents=True, exist_ok=True)
    out_num.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    figure_everybar(rows, mrows)
    floats = rf"""{EB_HEAD}
% Figure fig:rh_ladder (Appendix app:rh_fi_ladder, since 2026-10-29; was fig:rh_ladder_everybar): the feature-count
% ladder refitted at every bar, {rel(src)} (results/ablations_2014start_w3000/RULE.txt ADDENDUM K); dashed: the
% monthly ladder, {rel(LAD_FIX)}. Label: {EB_LABEL}.
\begin{{figure}}[tbp]
  \centering
  \includegraphics[width=\linewidth]{{figures/rehaul_ladder{EB_TAG}.pdf}}
  \caption{{How many exogenous series carry the gain when every rung is refitted at every bar. Each model is refitted
    at every bar on its top $k$ exogenous series only, ranked once by the model's own importance measure on a fit
    explained on \rhldEBRankYear{{}} data, before the evaluation span; $k = 0$ (the HAR benchmark set) and
    $k = \rhldEBNSeries$ (the full set) are the walks of Table~\ref{{tab:rh_oos}}. Vertical axis: the QLIKE of the
    rung relative to the same model's $k = 0$ rung, in percent (negative: more accurate), on the \rhldEBNBars{{}} bars
    of Table~\ref{{tab:rh_oos}} (left), on the \rhldEBNDaysOrd{{}} ordinary days (middle) and on the
    \rhldEBNDaysEv{{}} FOMC statement days (right). Solid: refitted at every bar, window $W = \rhldEBWindowDays$
    trading days, hyperparameters frozen. Dashed (left panel): the same rungs refitted at the \rhldNOrigins{{}} monthly
    origins and held for \rhldHoldSessions{{}} sessions, the pre-specified version of
    Appendix~\ref{{app:rh_fi_ladder}}, each against its own $k = 0$ rung. LightGBM, XGBoost and the linear models.
    The random forest is refitted only monthly (Appendix~\ref{{app:rh_fi_ladder}}). FOMC-patched design.}}
  \label{{fig:rh_ladder}}
\end{{figure}}
"""
    out_fig.write_text(floats, encoding="utf-8", newline="\n")
    print(
        f"{rel(src)}: wrote {len(M)} macros to {rel(out_num)}, {rel(out_fig)}, {rel(FIG / f'rehaul_ladder{EB_TAG}.pdf')}"
    )
    print(f"label (header comments and figure metadata): {EB_LABEL}")
    print("the text's ladder claims (Q-2026-10-29-31), asserted:")
    for c, _, d in claims:
        print(f"  [HOLDS] {c} ({d})")


def figure_everybar(rows: dict, mrows: dict) -> None:
    """Every-bar rungs solid, the monthly rungs dashed (every-bar panel only), QLIKE vs each ladder's own k = 0."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ink, ink2, gridc = "#0b0b0b", "#52514e", "#e4e3df"
    FIG.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(KS))

    def pct(f: pd.DataFrame) -> list[float]:
        q0 = float(f.loc[0, "qlike"])
        return [100.0 * (float(f.loc[k, "qlike"]) / q0 - 1.0) for k in KS]

    fig, axes = plt.subplots(1, len(EB_SPANS), figsize=(6.5, 2.9), sharex=True)
    for ax, sp in zip(axes, EB_SPANS):
        ax.axhline(0, color=ink2, lw=0.8)
        for m in EB_MODELS:
            if (m, sp) in mrows:
                ax.plot(x, pct(mrows[(m, sp)]), color=COLOR[m], marker=MARK[m], ms=3.5, mfc="white", lw=1.0, ls="--")
        for m in EB_MODELS:
            ax.plot(x, pct(rows[(m, sp)]), color=COLOR[m], marker=MARK[m], ms=4, lw=1.4, ls="-", label=MNAME[m])
        ax.set_xticks(x, [str(k) for k in KS])
        ax.set_xlabel("exogenous series $k$", fontsize=8, color=ink2)
        ax.set_title(EB_TITLE[sp], fontsize=9, color=ink)
        ax.grid(axis="y", color=gridc, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(ink2)
        ax.tick_params(colors=ink2, labelsize=7)
    axes[0].set_ylabel("QLIKE vs $k = 0$ (%)", fontsize=8)
    h, lb = axes[0].get_legend_handles_labels()
    h += [Line2D([], [], color=ink2, lw=1.4, ls="-"), Line2D([], [], color=ink2, lw=1.0, ls="--")]
    lb += ["every-bar refit", "monthly refit"]
    fig.legend(h, lb, loc="upper center", ncol=7, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIG / f"rehaul_ladder{EB_TAG}.pdf", metadata={"Subject": EB_LABEL})
    fig.savefig(FIG / f"rehaul_ladder{EB_TAG}.png", dpi=150, metadata={"Description": EB_LABEL})
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--prepatch", action="store_true", help="the pre-patch ladder; writes the *_prepatch names")
    ap.add_argument("--posthoc", action="store_true", help="first write fomcfix/posthoc_vs_walk.csv (patched only)")
    ap.add_argument(
        "--everybar",
        action="store_true",
        help="the ladder refitted at every bar (ADDENDUM K): reads ladall_*; writes the *_everybar names",
    )
    ap.add_argument(
        "--in-dir",
        type=Path,
        help="read the ladder files from DIR (default: fomcfix/; --prepatch: its parent; --everybar: scores_fomcfix/)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        help="write DIR/generated/ and DIR/figures/ instead of writeup/generated/ and writeup/figures/ (tests)",
    )
    a = ap.parse_args()
    if a.out_dir is not None:
        GEN, FIG = a.out_dir / "generated", a.out_dir / "figures"
    if a.everybar:
        assert not (a.prepatch or a.posthoc), "--everybar reads the every-bar ladder only (patched; no post hoc)"
        if a.in_dir is not None:
            EB_IN = a.in_dir
        main_everybar()
    else:
        if a.prepatch:
            assert not a.posthoc, "--posthoc writes the patched ladder's post hoc only (the pre-patch one is on file)"
            LAD, TAG = LAD_PRE, "_prepatch"
        if a.in_dir is not None:
            assert not a.posthoc, "--posthoc writes fomcfix/posthoc_vs_walk.csv itself: not with --in-dir"
            LAD = a.in_dir
        if a.posthoc:
            posthoc_patched()
        main()
