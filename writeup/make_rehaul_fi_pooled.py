"""Section 8 (variable importance) of the rehauled paper on the FINAL pooled runs (2026-10-23; FOMC-patched by default
since 2026-10-28).

Reads two finished studies (no model is fitted here; every number is read or recomputed from their files):

  results/tree_shap_w3000_2014start_2026-10-23/   TreeSHAP of the final W = 3,000 LightGBM, XGBoost and random forest
                                                  (RULE_shap.txt, M1-M5), 126 monthly origins, all 33,970 bars
  results/fi_by_window_2014start_2026-10-22/      the pooled ridge / lasso / elastic net coefficients on the window-
                                                  standardized predictors at the same 126 origins, W = 1,000-3,500
                                                  (RULE_fiw.txt, P1 / P2), fiw_gamma.npz

and, read-only, the merged every-bar walks of the three trees (results/pooled_trees_final_2014start_w3000,
results/pooled_rf_skx_2014start_w3000) to re-check that every origin refit the SHAP explains equals the walk's
forecast at the origin bar (2026-10-23, coordinator: the forest's walk is finished; compare all 126 origins).

2026-10-28 (decision D13, the paper's design is the FOMC-patched one): by DEFAULT the trees' attribution is the
patched TreeSHAP study, results/tree_shap_w3000_2014start_fomcfix_2026-10-26/ (RULE_pointer.txt: RULE_shap.txt
unchanged, on fomcfix cell B), gated against the patched every-bar walks (results/pooled_trees_final_2014start_fomcfix_
w3000, results/pooled_rf_skx_2014start_fomcfix_w3000); the linear models' pooled W = 3,000 shares (M2's reference,
Table tab:rh_fip_shares, the by-year panel) are the patched walks' gamma at the 126 origins (the study's
inputs/fiw_gamma_fomcfix.npz), their replay gate the patched ladder's GL1 (results/ladder_2026-10-26/fomcfix/
gates.json), their penalties results/window_sweep_2014start_fomcfix/w3000/frozen_penalties.json.  The window study
(fiw_summary / fiw_trend / fiw_verdicts: Table tab:rh_fip_window and the \\rhfip...W... macros) has no patched rerun
and stays the pre-patch one in both modes.  --prepatch reads the pre-patch study, walks and gamma (the run of
2026-10-23, as before) and writes every output under the *_prepatch names below (tag "" = patched, the names the
paper inputs).

Writes (labels for the other sections; <tag> before the extension):

  writeup/generated/rehaul_fip_numbers.tex     \\rhfip... prose macros
  writeup/generated/rehaul_fip_shares.tex      tab:rh_fip_shares   shares by category and class, D, concentration
  writeup/generated/rehaul_fip_nonstd.tex      tab:rh_fip_nonstd   leading non-standard series: share, rank, shape,
                                                                   interaction fraction
  writeup/generated/rehaul_fip_inter.tex       tab:rh_fip_inter    SHAP interactions (LightGBM, XGBoost)
  writeup/generated/rehaul_fip_window.tex      tab:rh_fip_window   the linear models across training windows
  writeup/generated/rehaul_fip_fig_rank.tex    fig:rh_fip_rank     figures/fi_pooled/fip_rank.pdf
  writeup/generated/rehaul_fip_fig_time.tex    fig:rh_fip_time     figures/fi_pooled/fip_time.pdf
  writeup/generated/rehaul_fip_fig_cond.tex    fig:rh_fip_cond     figures/fi_pooled/fip_cond.pdf
  writeup/generated/rehaul_fip_fig_shapes.tex  fig:rh_fip_shapes   figures/fi_pooled/fip_shapes.pdf

Quantities computed here that the two studies' rules did not fix in advance are DESCRIPTIVE and say so in the text:
the effective number of exogenous series N_eff = 1 / sum_A s_A^2 of the pooled shares (for the linear models it equals
fiw_summary.csv's Neff_pooled, asserted), the linear models' shares by calendar year (from fiw_gamma.npz, by origin
date), and the interaction fraction of the standard exogenous series without the HAR components and calendar.

Run:  python writeup/make_rehaul_fi_pooled.py [--prepatch] [--preview DIR]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WRITEUP = Path(__file__).resolve().parent
sys.path.insert(0, str(WRITEUP))
import make_rehaul_fi as fi  # noqa: E402  (series names, figure style; importing it only defines them)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import PowerNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = WRITEUP.parent
GEN = WRITEUP / "generated"
FIG = WRITEUP / "figures" / "fi_pooled"
SHAP_PRE = ROOT / "results" / "tree_shap_w3000_2014start_2026-10-23"  # pre-patch design (RULE_shap.txt)
SHAP_FIX = ROOT / "results" / "tree_shap_w3000_2014start_fomcfix_2026-10-26"  # FOMC-patched (RULE_pointer.txt; D13)
FIW = ROOT / "results" / "fi_by_window_2014start_2026-10-22"
SWEEP = ROOT / "results" / "window_sweep_2014start_2026-10-22"
SWEEP_FIX = ROOT / "results" / "window_sweep_2014start_fomcfix"  # the patched W = 3,000 walks' frozen penalties
LADDER_GATES_FIX = ROOT / "results" / "ladder_2026-10-26" / "fomcfix" / "gates.json"  # GL1: patched linear replays
WALK_PRE = {
    "lgbm": ROOT / "results" / "pooled_trees_final_2014start_w3000" / "forecasts.parquet",
    "xgb": ROOT / "results" / "pooled_trees_final_2014start_w3000" / "forecasts.parquet",
    "rf_skx": ROOT / "results" / "pooled_rf_skx_2014start_w3000" / "forecasts.parquet",
}
WALK_FIX = {
    "lgbm": ROOT / "results" / "pooled_trees_final_2014start_fomcfix_w3000" / "forecasts.parquet",
    "xgb": ROOT / "results" / "pooled_trees_final_2014start_fomcfix_w3000" / "forecasts.parquet",
    "rf_skx": ROOT / "results" / "pooled_rf_skx_2014start_fomcfix_w3000" / "forecasts.parquet",
}
SHAP, WALK, TAG = SHAP_FIX, WALK_FIX, ""  # set by the command line (--prepatch: SHAP_PRE, WALK_PRE, "_prepatch")
W_MAIN = 3000
PREVIEW: Path | None = None

TREES = ("lgbm", "xgb", "rf_skx")
LINS = ("ridge", "lasso", "enet")
NAME = {
    "lgbm": "LightGBM",
    "xgb": "XGBoost",
    "rf_skx": "Random forest",
    "ridge": "Ridge",
    "lasso": "Lasso",
    "enet": "Elastic net",
}
MAC = {
    "lgbm": "Lgbm",
    "xgb": "Xgb",
    "rf_skx": "Rf",
    "ridge": "Ridge",
    "lasso": "Lasso",
    "enet": "Enet",
}
COLOR = {
    "lgbm": fi.OKABE[0],
    "xgb": fi.OKABE[1],
    "rf_skx": fi.OKABE[2],
    "ridge": fi.OKABE[6],
    "lasso": fi.OKABE[3],
    "enet": fi.OKABE[4],
}
MARK = {"lgbm": "o", "xgb": "s", "rf_skx": "^", "ridge": "D", "lasso": "v", "enet": "P"}
# the paper's category words (rehaul/04_features.tex), the backbone shown as its two members
BUCKETS = [
    "har_ma",
    "calendar",
    "moments",
    "liquidity",
    "market_ew",
    "market_vw",
    "sentiment",
    "implied_vol",
    "vol_demand",
    "fomc",
]
BUCKET_WORD = {
    "har_ma": "HAR components",
    "calendar": "Calendar dummies",
    "moments": "Realized moments",
    "liquidity": "Liquidity",
    "market_ew": "Market moments, EW",
    "market_vw": "Market moments, VW",
    "sentiment": "Sentiment",
    "implied_vol": "Implied volatility",
    "vol_demand": "Volatility demand",
    "fomc": "FOMC announcements",
}
# 2026-10-19 (user): the draft names each series by its raw parquet column name (\rawseriestrue in paper_rehaul.tex);
# the figures follow (False: the symbols of the notation guide, as writeup/make_rehaul_fi.py's figures).
RAW_NAMES = True

HEAD = "% AUTO-GENERATED by writeup/make_rehaul_fi_pooled.py -- do not edit; re-run the script.\n" + fi.SER_PROVIDE


def load_line() -> str:
    """The tables' and floats' guarded input of this mode's macro file."""
    return rf"\ifdefined\rhfipNOrigins\else\input{{generated/rehaul_fip_numbers{TAG}}}\fi"


# ============================================================================ helpers
def rel(p: Path) -> str:
    return fi.rel(p)


check = fi.check


def num(n: float) -> str:
    """Integer with a LaTeX thousands separator."""
    return f"{int(round(n)):,}".replace(",", "{,}")


def p1(x: float, nd: int = 1) -> str:
    """A share (fraction) in percent."""
    return fi.pct(x, nd)


def pp(x: float, nd: int = 1) -> str:
    """A difference of shares in percentage points, signed."""
    s = f"{100 * x:+.{nd}f}"
    return s.replace("-", r"\ensuremath{-}").replace("+", r"\ensuremath{+}")


def ci(lo: float, hi: float, nd: int = 1, signed: bool = False) -> str:
    g = (lambda v: pp(v, nd)) if signed else (lambda v: p1(v, nd))
    return f"[{g(lo)}, {g(hi)}]"


def date_tex(d) -> str:
    return fi.date_tex(d)


def sci(x: float) -> str:
    m, e = f"{x:.1e}".split("e")
    return rf"\ensuremath{{{m}\times 10^{{{int(e)}}}}}"


def fomc_statement_days() -> int:
    """Scheduled FOMC statement days 2014-01-02 .. 2024-04-30 on the loader's calendar: data/releases.parquet's
    'fomc release' bars, less the unscheduled actions (UNSCHEDULED_FOMC_DAYS), plus FOMC_MISSING_STATEMENTS
    (2026-10-26, audit results-9; 81 days)."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.data import loading as ld

    rel = pd.read_parquet(ROOT / "data" / "releases.parquet", columns=["endbartime", "fomc release"])
    days = set(pd.to_datetime(rel.loc[rel["fomc release"] > 0, "endbartime"]).dt.normalize())
    days -= set(ld.UNSCHEDULED_FOMC_DAYS)
    if ld.FOMC_CALENDAR_FIX:
        days |= {pd.Timestamp(x).normalize() for x in ld.FOMC_MISSING_STATEMENTS}
    n = sum(pd.Timestamp("2014-01-02") <= d <= pd.Timestamp("2024-04-30") for d in days)
    assert n == 81, n
    return n


def fomc_days_shap(T: dict) -> int:
    """The FOMC statement days of the SHAP study's M4 subset (its FOMC-day bars, 13 a day)."""
    sub = T["m4_subsets"]
    fr = int(sub[(sub["est"] == "lgbm") & (sub["subset"] == "FOMC day")]["rows"].iloc[0])
    assert fr % 13 == 0, fr
    return fr // 13


def name_tex(v: str) -> str:
    """A predictor group in the text and tables: the raw name / symbol switch, or the group words."""
    return fi.grp(v)


def name_mpl(v: str) -> str:
    if v in fi.GROUP_WORDS:
        return fi.GROUP_WORDS[v]
    return v if RAW_NAMES else f"${fi.series_sym(v)}$"


def year_range(years: list[int]) -> tuple[int, int]:
    ys = sorted(years)
    assert ys == list(range(ys[0], ys[-1] + 1)), f"years not contiguous: {ys}"
    return ys[0], ys[-1]


def group_of(col: str) -> str:
    """Column -> predictor group, as results/tree_shap_w3000_2014start_2026-10-23/code/shap_common.py."""
    if col.startswith("har_ma_"):
        return "har_ma"
    if col.startswith("DOW_") or col == "hour":
        return "calendar"
    if col.startswith("adj_") and "_ma_" in col:
        return col[4:].rsplit("_ma_", 1)[0]
    if "_avail_ma_" in col:
        return col.rsplit("_avail_ma_", 1)[0]
    raise ValueError(col)


def horizon(col: str) -> str:
    """The averaging horizon k of a value-average predictor, adj_<series>_ma_<k>."""
    m = re.search(r"_ma_(\d+)$", col)
    assert m is not None, col
    return m.group(1)


def neff(s: np.ndarray) -> tuple[float, int, int]:
    """1 / HHI of shares summing to one; the fewest series holding half and 80% of the total."""
    s = np.sort(np.asarray(s, float))[::-1]
    assert abs(s.sum() - 1) < 1e-9
    c = np.cumsum(s)
    return (
        float(1 / np.sum(s**2)),
        int(np.searchsorted(c, 0.5) + 1),
        int(np.searchsorted(c, 0.8) + 1),
    )


# ============================================================================ macros
class Macros:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def __setitem__(self, name: str, val) -> None:
        assert re.fullmatch(r"[A-Za-z]+", name), name
        v, c = val if isinstance(val, tuple) else (val, "")
        self.rows.append((name, str(v), c))

    def write(self, path: Path) -> None:
        if TAG == "_prepatch":
            src = (
                "% \\rhfip... macros for writeup/rehaul/08_feature_importance.tex (results/tree_shap_w3000_2014start_"
                "2026-10-23,\n% results/fi_by_window_2014start_2026-10-22; the final pooled runs, W = 3,000 trading days).\n"
            )
        else:
            src = (
                f"% \\rhfip... macros for writeup/rehaul/08_feature_importance.tex (results/{SHAP.name}, the\n"
                "% FOMC-patched design, its linear reference inputs/fiw_gamma_fomcfix.npz; the window study's ...W... "
                "macros:\n% results/fi_by_window_2014start_2026-10-22, pre-patch; the final pooled runs, W = 3,000 "
                "trading days).\n"
            )
        lines = [HEAD, src]
        seen = set()
        for n, v, c in self.rows:
            assert n not in seen, n
            seen.add(n)
            lines.append(
                f"\\providecommand{{\\rhfip{n}}}{{}}\\renewcommand{{\\rhfip{n}}}{{{v}}}"
                + (f"  % {c}" if c else "")
                + "\n"
            )
        path.write_text("".join(lines), encoding="utf-8")
        print(f"wrote {rel(path)} ({len(self.rows)} macros)")


# ============================================================================ load
def load() -> dict:
    T: dict = {}
    for n in (
        "m1_variables",
        "m1_buckets",
        "m1_classes",
        "m1_by_year",
        "m2_linear_shares",
        "m2_nonstandard_D",
        "m2_rank",
        "m3_shapes",
        "m3_bins",
        "m4_subsets",
        "m4_contrasts",
        "m4_origins_vix",
        "m5_summary",
        "m5_variables",
        "m5_buckets",
        "m5_pairs_nonstandard",
        "gate",
    ):
        T[n] = pd.read_csv(SHAP / f"{n}.csv")
    T["shap_info"] = json.loads((SHAP / "run_info.json").read_text())
    for n in (
        "fiw_summary",
        "fiw_trend",
        "fiw_verdicts",
        "fiw_variables",
        "fiw_origins",
    ):
        T[n] = pd.read_csv(FIW / f"{n}.csv")
    T["fiw_info"] = json.loads((FIW / "run_info.json").read_text())
    # the linear models' coefficients at the 126 origins, W = 3,000: the patched study's reference (its M2) or the
    # pre-patch window study's
    patched = TAG == ""
    z = np.load(SHAP / "inputs" / "fiw_gamma_fomcfix.npz" if patched else FIW / "fiw_gamma.npz")
    T["gamma_cols"] = [str(c) for c in z["columns"]]
    T["gamma_origins"] = pd.to_datetime([str(o) for o in z["origins"]])
    T["gamma"] = {m: z[f"walk_w{W_MAIN}_{m}_gamma"] for m in LINS}
    T["folds"] = json.loads((SWEEP / "folds14.json").read_text())
    T["pen"] = json.loads(((SWEEP_FIX if patched else SWEEP) / f"w{W_MAIN}" / "frozen_penalties.json").read_text())
    if patched:
        T["lin_gate_fix"] = json.loads(LADDER_GATES_FIX.read_text())
        T["shap_origins"] = pd.read_csv(SHAP / "inputs" / "fiw_origins.csv")
    walks = {}
    for e, f in WALK.items():
        d = pd.read_parquet(f, columns=["model", "predictor_set", "date", "pred_adj"])
        d = d[(d["model"] == e) & (d["predictor_set"] == "all_features")]
        walks[e] = d.set_index("date")["pred_adj"]
    T["walk"] = walks
    return T


# ============================================================================ derived tables
def derive(T: dict) -> dict:
    """Linear group shares (pooled and by origin year), tree exogenous shares, class maps."""
    v = T["m1_variables"]
    D: dict = {}
    groups = v[v["est"] == "lgbm"]["variable"].tolist()
    D["groups"] = groups
    D["bucket"] = v[v["est"] == "lgbm"].set_index("variable")["bucket"].to_dict()
    D["cls"] = v[v["est"] == "lgbm"].set_index("variable")["cls"].to_dict()
    D["exog"] = [g for g in groups if g not in ("har_ma", "calendar")]
    D["nonstd"] = [g for g in groups if D["cls"][g] == "NON-STANDARD"]
    D["std"] = [g for g in groups if D["cls"][g] == "STANDARD"]
    # linear |gamma| by group at each origin (126 x 48)
    P = np.zeros((len(groups), len(T["gamma_cols"])))
    gi = {g: i for i, g in enumerate(groups)}
    for j, c in enumerate(T["gamma_cols"]):
        P[gi[group_of(c)], j] = 1.0
    ex = np.array([g in D["exog"] for g in groups])
    ns = np.array([D["cls"][g] == "NON-STANDARD" for g in groups])
    D["ex_mask"], D["ns_mask"] = ex, ns
    D["lin_k"] = {m: np.abs(T["gamma"][m]) @ P.T for m in LINS}  # K x 48
    years = T["gamma_origins"].year
    lin_year = {}
    for m in LINS:
        A = pd.DataFrame(D["lin_k"][m], columns=groups)
        A["year"] = years
        Y = A.groupby("year").sum()
        lin_year[m] = Y[D["exog"]].div(Y[D["exog"]].sum(1), axis=0)  # share of the exogenous total, by origin year
    D["lin_year"] = lin_year
    # trees: share of the exogenous total by forecast year
    by = T["m1_by_year"]
    tree_year = {}
    for e in TREES:
        Y = by[by["est"] == e].pivot(index="year", columns="variable", values="share")
        tree_year[e] = Y[D["exog"]].div(Y[D["exog"]].sum(1), axis=0)
    D["tree_year"] = tree_year
    # pooled shares of the exogenous total
    L = T["m2_linear_shares"]
    D["lin_ex"] = {m: L[L["linear"] == m].set_index("variable")["share_exog"].reindex(D["exog"]) for m in LINS}
    D["lin_all"] = {m: L[L["linear"] == m].set_index("variable")["share_all"].reindex(groups) for m in LINS}
    D["tree_ex"] = {e: v[v["est"] == e].set_index("variable")["share_exog"].reindex(D["exog"]) for e in TREES}
    D["tree_all"] = {e: v[v["est"] == e].set_index("variable")["share"].reindex(groups) for e in TREES}
    # rank among the 46 exogenous series (1 = largest share)
    D["rank_ex"] = {
        k: s.rank(ascending=False, method="min").astype(int) for k, s in {**D["tree_ex"], **D["lin_ex"]}.items()
    }
    D["conc"] = {k: neff(s.to_numpy()) for k, s in {**D["tree_ex"], **D["lin_ex"]}.items()}
    return D


# ============================================================================ gates
def gates(T: dict, D: dict) -> dict:
    print("gates")
    G: dict = {}
    g = T["gate"]
    g["origin"] = pd.to_datetime(g["origin"])
    o = T["fiw_origins"]
    origins = pd.to_datetime(o["origin"])
    check(
        len(o) == 126 and int(o["block_rows"].sum()) == 33970,
        "126 origins, blocks partition 33,970 bars",
    )
    check(
        list(T["gamma_origins"]) == list(origins),
        "fiw_gamma.npz origins = fiw_origins.csv",
    )
    for e in TREES:
        r = g[g["est"] == e].sort_values("k")
        check(list(r["origin"]) == list(origins), f"{e}: gate origins = the 126 origins")
        ref = r["pred_origin_hex"].map(float.fromhex).to_numpy()
        w = T["walk"][e]
        check(w.index.is_unique and len(w) == 33970, f"{e}: merged walk holds 33,970 bars")
        wv = r["origin"].map(w).to_numpy()
        eq = int(np.sum(wv == ref))
        G[e + "_eq"] = eq
        check(
            eq == 126,
            f"{e}: origin refit = merged every-bar walk pred_adj, bit-equal at {eq}/126 origins",
        )
        m = r["walk_hex"].notna()
        G[e + "_eq_at_shap"] = int(r.loc[m, "walk_equal"].astype(bool).sum())
        G[e + "_n_at_shap"] = int(m.sum())
        check(bool(r["add_pass"].all()), f"{e}: SHAP additivity passes at 126/126 origins")
        G[e + "_add"] = float(r["add_err_max"].max())
    check(G["xgb_add"] <= 1e-5, f"XGBoost additivity {G['xgb_add']:.1e} <= 1e-5 (float32)")
    check(
        max(G["lgbm_add"], G["rf_skx_add"]) <= 1e-9,
        f"LightGBM / forest additivity {max(G['lgbm_add'], G['rf_skx_add']):.1e} <= 1e-9",
    )
    if TAG == "":
        # patched: the SHAP study's origins are the window study's, and the linear gamma is the patched ladder's
        # full-set replays, gated to the patched linear walks (GL1)
        check(T["shap_origins"].equals(o), "patched SHAP inputs/fiw_origins.csv = the window study's origins")
        gl = T["lin_gate_fix"]
        check(
            all(bool(gl[m]["GL1_pass"]) and gl[m]["GL1_maxabs"] <= 1e-12 for m in LINS),
            "patched linear replay gate (ladder fomcfix GL1): every model <= 1e-12",
        )
        G["lin_gate_max"] = max(float(gl[m]["GL1_maxabs"]) for m in LINS)
        vl = T["pen"]["info"]["folds"][-1]["val_last"]
        f3 = [f for f in T["folds"]["folds"] if f["fold"] == 3][0]
        check(
            pd.Timestamp(vl) == pd.Timestamp(f3["val_last"]),
            f"patched penalties' last tuning fold ends {vl} = folds14.json fold 3",
        )
    else:
        # linear replay gate of the window study
        gi = T["fiw_info"]["gate"]
        check(bool(gi["all_pass_1e12"]), "fi-window replay gate: every W x model <= 1e-12")
        G["lin_gate_max"] = max(
            v["replay_max_abs_all_rows"] for k, v in gi["per_W_model"].items() if k.startswith(f"w{W_MAIN}/")
        )
    # shares
    v = T["m1_variables"]
    for e in TREES:
        s = v[v["est"] == e]
        check(abs(s["share"].sum() - 1) < 1e-9, f"{e}: 48 shares sum to one")
        c = T["m1_classes"].set_index(["est", "cls"])["share"]
        check(
            abs(s[s["cls"] == "NON-STANDARD"]["share"].sum() - c[(e, "NON-STANDARD")]) < 1e-9,
            f"{e}: non-standard class share = sum of its 36 series",
        )
    check(
        len(D["groups"]) == 48 and len(D["exog"]) == 46 and len(D["nonstd"]) == 36,
        "48 groups, 46 exogenous, 36 non-standard",
    )
    # linear shares recomputed from fiw_gamma.npz = the SHAP study's m2_linear_shares = fiw_variables.csv
    fv = T["fiw_variables"]
    for m in LINS:
        tot = D["lin_k"][m].sum(0)
        sh = tot[D["ex_mask"]] / tot[D["ex_mask"]].sum()
        check(
            np.allclose(sh, D["lin_ex"][m].to_numpy(), rtol=1e-9, atol=1e-12),
            f"{m}: pooled |gamma| shares from fiw_gamma.npz = m2_linear_shares.csv",
        )
        if TAG == "":
            continue  # the window study's files below describe the pre-patch walks (no patched window study)
        ref = fv[(fv["W"] == W_MAIN) & (fv["model"] == m) & (fv["source"] == "walk")].set_index("variable")[
            "pooled_share"
        ]
        check(
            np.allclose(sh, ref.reindex(D["exog"]).to_numpy(), rtol=1e-9, atol=1e-12),
            f"{m}: = fiw_variables.csv pooled_share (W = {W_MAIN} walk)",
        )
        fs = T["fiw_summary"]
        r = fs[(fs["W"] == W_MAIN) & (fs["model"] == m) & (fs["source"] == "walk")].iloc[0]
        ne, n50, n80 = D["conc"][m]
        check(
            abs(ne - r["Neff_pooled"]) < 1e-9 and n50 == r["n50_pooled"] and n80 == r["n80_pooled"],
            f"{m}: N_eff {ne:.2f}, n50 {n50}, n80 {n80} = fiw_summary.csv pooled values",
        )
    # D = tree minus linear, of the exogenous total
    Dt = T["m2_nonstandard_D"]
    cl = T["m1_classes"].set_index(["est", "cls"])
    for e in TREES:
        for m in LINS:
            r = Dt[(Dt["est"] == e) & (Dt["linear"] == m) & (Dt["denominator"] == "of exogenous total")].iloc[0]
            lin_ns = D["lin_ex"][m][[x for x in D["exog"] if x in D["nonstd"]]].sum()
            check(
                abs(r["linear_share"] - lin_ns) < 1e-9
                and abs(r["tree"] - cl.loc[(e, "NON-STANDARD"), "share_exog"]) < 1e-9
                and abs(r["D"] - (r["tree"] - r["linear_share"])) < 1e-12,
                f"D {e} - {m}: {100 * r['D']:+.1f} pp = tree minus linear non-standard share",
            )
    # tuning fold before the first origin
    f3 = [f for f in T["folds"]["folds"] if f["fold"] == 3][0]
    G["tune_last"] = pd.Timestamp(f3["val_last"])
    check(
        G["tune_last"] < origins.iloc[0],
        f"last tuning-fold bar {f3['val_last']} precedes the first origin",
    )
    return G


# ============================================================================ macros
def macros(T: dict, D: dict, G: dict) -> Macros:
    M = Macros()
    o = T["fiw_origins"]
    origins = pd.to_datetime(o["origin"])
    info = T["shap_info"]
    M["NOrigins"] = (len(o), "monthly origins (re-estimations explained)")
    M["NRows"] = (
        num(info["forecast_rows"]),
        "bars explained, every bar of the evaluation span",
    )
    M["OriginEvery"] = (
        int(o["block_sessions"].iloc[:-1].mode()[0]),
        "trading days between origins",
    )
    M["LastBlock"] = (
        int(o["block_sessions"].iloc[-1]),
        "trading days of the last origin's block",
    )
    M["FirstOrigin"] = (date_tex(origins.iloc[0]), "first origin (10:00 bar)")
    M["LastOrigin"] = (date_tex(origins.iloc[-1]), "last origin")
    last = T["walk"]["lgbm"].index.max()
    assert all(T["walk"][e].index.max() == last for e in TREES)
    M["LastDay"] = (date_tex(last), "day of the last bar explained")
    M["WindowDays"] = (num(W_MAIN), "rolling window, trading days")
    M["WindowBars"] = (num(W_MAIN * 13), "rolling window, bars")
    M["NCols"] = (len(T["gamma_cols"]), "predictors of the full set")
    M["NGroups"] = (
        len(D["groups"]),
        "predictor groups: 46 exogenous series + HAR components + calendar dummies",
    )
    M["NExog"] = (len(D["exog"]), "exogenous series")
    M["NPerSeries"] = (
        12,
        "predictors per exogenous series: value and availability averages at six horizons",
    )
    assert all(sum(group_of(c) == g for c in T["gamma_cols"]) == 12 for g in D["exog"])
    M["NStd"] = (len(D["std"]), "standard groups")
    M["NNonStd"] = (len(D["nonstd"]), "non-standard series")
    M["NStdExog"] = (
        len([g for g in D["std"] if g in D["exog"]]),
        "standard exogenous series",
    )
    M["TuneLast"] = (
        date_tex(G["tune_last"]),
        "last day of the last tuning fold (fold 3)",
    )
    pen = T["pen"]
    M["RidgePen"] = fi.pow10(pen["ridge"]["all_features"])
    M["LassoPen"] = fi.pow10(pen["lasso"]["all_features"])
    M["EnetPen"] = fi.pow10(pen["enet"]["all_features"])
    M["HarPf"] = (
        pen["designs"]["all_features"]["penalty_factor_on_har_benchmark_block"],
        "penalty factor, HAR block",
    )
    bs = info["bootstrap"]
    assert bs["block"] == T["fiw_info"]["bootstrap"]["block"]
    M["BootBlock"] = (bs["block"], "bootstrap block, origins")
    M["BootB"] = num(bs["B"])
    M["CiLevel"] = (
        95,
        "percentile intervals of both rules (RULE_shap.txt, RULE_fiw.txt)",
    )
    sel = set(T["m3_shapes"]["selection"])
    assert sel == {">= 0.5%"}, sel
    M["ShapeMin"] = (fi.f(0.5, 1), "M3 selection: share of at least this many percent")
    assert "top10_overlap" in T["m2_rank"].columns
    M["TopN"] = (10, "length of the top lists compared (m2_rank.csv top10_overlap)")
    assert "slope_per_1000d" in T["fiw_trend"].columns
    M["SlopeUnit"] = (num(1000), "trading days per unit of the window slope")
    M["NMom"] = (
        fi.word(sum(D["bucket"][g] == "moments" for g in D["std"])),
        "the futures' realized moments (standard)",
    )
    # gates
    for e in TREES:
        M[MAC[e] + "GateEq"] = (
            G[e + "_eq"],
            "origins bit-equal to the merged every-bar walk",
        )
    M["RfGateAtShap"] = (
        G["rf_skx_n_at_shap"],
        "forest origins the walk had written when the SHAP gate ran",
    )
    M["XgbAddMax"] = (sci(G["xgb_add"]), "XGBoost additivity, float32")
    M["XgbAddTol"] = fi.pow10(1e-5)
    M["AddMaxDouble"] = sci(max(G["lgbm_add"], G["rf_skx_add"]))
    M["LinGateMax"] = (
        sci(G["lin_gate_max"]),
        "linear replay vs walk forecasts, W = 3,000",
    )
    m5 = T["m5_summary"].set_index("est")
    M["InterRows"] = (num(m5.loc["lgbm", "rows"]), "M5 rows")
    M["InterPerOrigin"] = (int(m5.loc["lgbm", "rows"]) // len(o), "M5 rows per origin")
    # FOMC and VIX subsets
    sub = T["m4_subsets"]
    fr = sub[(sub["est"] == "lgbm") & (sub["subset"] == "FOMC day")]["rows"].iloc[0]
    M["FomcRows"] = num(fr)
    # 2026-10-26 (integration, audit results-9): the reducer (shap_reduce.py M4, run of 2026-10-23) took the FOMC days
    # from data/releases.parquet alone, without the three statements the loader adds (src/data/loading.py
    # FOMC_MISSING_STATEMENTS, FOMC_CALENDAR_FIX): 78 of the span's 81 statement days; the 39 bars of the other three
    # are counted with the other days until the SHAP rerun on the patched design (the reducer is fixed for it).
    # rhfipFomcDays is now the statement days of the span (81); rhfipFomcDaysShap the days of the SHAP subset.
    # Was: M["FomcDays"] = (int(fr) // 13, "FOMC statement days among the bars (13 bars a day)")
    # 2026-10-28: the patched study's reducer (fomcfix code/shap_reduce.py M4) still takes the FOMC days from
    # releases.parquet alone (78 of 81; the fix above did not reach it), so the three days stay with the other days in
    # both modes; only the comment of FomcDaysMissing differs (no rerun pending).
    assert fr % 13 == 0 and int(fr) // 13 == fomc_days_shap(T)
    M["FomcDaysShap"] = (int(fr) // 13, "FOMC statement days in the SHAP subset (M4; releases.parquet only)")
    M["FomcDays"] = (fomc_statement_days(), "scheduled FOMC statement days in the span (loader calendar)")
    M["FomcDaysMissing"] = (
        fomc_statement_days() - int(fr) // 13,
        "statement days the SHAP subset leaves out ("
        + (
            "counted with the other days until the patched SHAP rerun)"
            if TAG == "_prepatch"
            else "counted with the other days; the patched rerun's reducer keeps releases.parquet's days)"
        ),
    )
    # 2026-10-26 (integration, audit results-8): the NYSE-holiday sessions among the explained bars
    hol = pd.read_csv(ROOT / "results" / "fomcfix_2026-10-26" / "holiday_sessions.csv", parse_dates=["date"])
    hol = hol[(hol.date >= pd.Timestamp("2014-01-02")) & (hol.date <= pd.Timestamp("2024-04-30"))]
    assert int(info["forecast_rows"]) - int(hol["n_rth_bars"].sum()) == 33527, "explained - holiday bars = Table 1's"
    M["NHolSessions"] = (len(hol), "NYSE-holiday futures sessions among the explained bars (not scored in Table 1)")
    M["NHolBars"] = (int(hol["n_rth_bars"].sum()), "their regular-hours bars")
    vx = T["m4_origins_vix"]["vix_prev_close"].to_numpy()
    q = np.quantile(vx, [1 / 3, 2 / 3])
    M["VixLo"], M["VixHi"] = fi.f(q[0], 2), fi.f(q[1], 2)

    # ---------------------------------------------------------------- M1 shares
    bk = T["m1_buckets"].set_index(["est", "bucket"])
    cl = T["m1_classes"].set_index(["est", "cls"])
    for e in TREES:
        k = MAC[e]
        M[k + "Har"] = p1(bk.loc[(e, "har_ma"), "share"])
        M[k + "HarLo"], M[k + "HarHi"] = (
            p1(bk.loc[(e, "har_ma"), "lo"]),
            p1(bk.loc[(e, "har_ma"), "hi"]),
        )
        M[k + "Mom"] = p1(bk.loc[(e, "moments"), "share"])
        r = cl.loc[(e, "NON-STANDARD")]
        M[k + "Ns"], M[k + "NsLo"], M[k + "NsHi"] = (
            p1(r["share"]),
            p1(r["lo"]),
            p1(r["hi"]),
        )
        M[k + "NsEx"], M[k + "NsExLo"], M[k + "NsExHi"] = (
            p1(r["share_exog"]),
            p1(r["lo_exog"]),
            p1(r["hi_exog"]),
        )
    for m in LINS:
        k = MAC[m]
        M[k + "NsEx"] = p1(D["lin_ex"][m][[x for x in D["exog"] if x in D["nonstd"]]].sum())
        M[k + "Ns"] = p1(D["lin_all"][m][D["nonstd"]].sum())
        M[k + "Har"] = p1(D["lin_all"][m]["har_ma"])
    rng = [bk.loc[(e, "har_ma"), "share"] for e in TREES]
    M["HarMin"], M["HarMax"] = p1(min(rng)), p1(max(rng))
    rng = [bk.loc[(e, "moments"), "share"] for e in TREES]
    M["MomMin"], M["MomMax"] = p1(min(rng)), p1(max(rng))
    rng = [cl.loc[(e, "NON-STANDARD"), "share_exog"] for e in TREES]
    M["NsExMin"], M["NsExMax"] = p1(min(rng)), p1(max(rng))
    for b, n in (("sentiment", "Sent"), ("vol_demand", "Vd"), ("fomc", "FomcBucket")):
        rng = [bk.loc[(e, b), "share"] for e in TREES]
        M[n + "Min"], M[n + "Max"] = p1(min(rng)), p1(max(rng))
    ta = D["tree_all"]
    for vname, n in (("vvix", "Vvix"), ("vix3m", "VixThreeM"), ("fomc_day", "FomcDay")):
        rng = [ta[e][vname] for e in TREES]
        M[n + "Min"], M[n + "Max"] = p1(min(rng)), p1(max(rng))
    # leading non-standard series by the mean of the three trees' shares
    mean_ns = sum(ta[e][D["nonstd"]] for e in TREES) / 3
    lead = mean_ns.sort_values(ascending=False).index[:5].tolist()
    D["lead_ns"] = lead
    for i, vname in enumerate(lead):
        n = "Ns" + ["One", "Two", "Three", "Four", "Five"][i]
        rng = [ta[e][vname] for e in TREES]
        M[n] = name_tex(vname)
        M[n + "Min"], M[n + "Max"] = p1(min(rng)), p1(max(rng))
        for e in TREES:
            M[n + MAC[e]] = p1(ta[e][vname])
    # leading exogenous series of the trees (the same two for all three, asserted)
    tops = {e: D["tree_ex"][e].sort_values(ascending=False).index[:2].tolist() for e in TREES}
    assert tops["lgbm"] == tops["xgb"] == tops["rf_skx"], tops
    M["TreeTopOne"], M["TreeTopTwo"] = (
        name_tex(tops["lgbm"][0]),
        name_tex(tops["lgbm"][1]),
    )
    for e in TREES:
        M[MAC[e] + "TopOneEx"] = p1(D["tree_ex"][e][tops["lgbm"][0]])
    rtop = D["lin_ex"]["ridge"].sort_values(ascending=False).index[:1].tolist()[0]
    M["RidgeTopOne"] = name_tex(rtop)
    M["RidgeTopOneEx"] = p1(D["lin_ex"]["ridge"][rtop])
    # usage: origins at which each tree splits on the series
    v = T["m1_variables"]
    for e in TREES:
        s = v[(v["est"] == e) & (v["variable"].isin(D["exog"]))]
        M[MAC[e] + "NUsedAll"] = (
            int((s["K_used"] == len(o)).sum()),
            "exogenous series split on at every origin",
        )
        M[MAC[e] + "UsedMin"] = int(s["K_used"].min())
    # ---------------------------------------------------------------- M2 vs the linear models
    Dt = T["m2_nonstandard_D"]
    for e in TREES:
        for m in LINS:
            r = Dt[(Dt["est"] == e) & (Dt["linear"] == m) & (Dt["denominator"] == "of exogenous total")].iloc[0]
            assert r["verdict"] == ("LESS" if r["hi"] < 0 else r["verdict"])
            k = MAC[e] + "D" + MAC[m]
            M[k], M[k + "Lo"], M[k + "Hi"] = pp(r["D"]), pp(r["lo"]), pp(r["hi"])
    M["NDLess"] = (
        int((Dt["verdict"] == "LESS").sum()),
        "tree-minus-linear D with the 95% interval below 0",
    )
    M["NDAll"] = (len(Dt), "D comparisons: 3 trees x 3 linear models x 3 denominators")
    assert (Dt["hi"] < 0).sum() == (Dt["verdict"] == "LESS").sum()
    R = T["m2_rank"]
    for e in TREES:
        for m in LINS:
            r = R[(R["est"] == e) & (R["linear"] == m) & (R["scope"] == "46 exogenous")].iloc[0]
            M[MAC[e] + "Sp" + MAC[m]] = fi.f(r["spearman"], 2)
            M[MAC[e] + "Top" + MAC[m]] = int(r["top10_overlap"])
    for a, b in (("lgbm", "xgb"), ("lgbm", "rf_skx"), ("xgb", "rf_skx")):
        r = R[(R["est"] == a) & (R["linear"] == b)].iloc[0]
        M["Sp" + MAC[a] + MAC[b]] = fi.f(r["spearman"], 2)
    rk = D["rank_ex"]
    for vname, n in (("fomc_until_inv", "FomcUntil"), ("fomc_day", "FomcDay")):
        M["RidgeRank" + n] = rk["ridge"][vname]
        rr = [rk[e][vname] for e in TREES]
        M["TreeRank" + n + "Min"], M["TreeRank" + n + "Max"] = min(rr), max(rr)
    M["FomcUntil"] = name_tex("fomc_until_inv")
    M["FomcDay"] = name_tex("fomc_day")
    M["AbsVw"] = name_tex("sumabsret_vwstock")
    M["Ofi"] = name_tex("ofi_vwstock")
    # concentration (descriptive, not in either rule)
    for k, (ne, n50, n80) in D["conc"].items():
        M[MAC[k] + "Neff"] = fi.f(ne, 1)
        M[MAC[k] + "NHalf"] = n50
        M[MAC[k] + "NEighty"] = n80
    # ---------------------------------------------------------------- by year
    by = T["m1_by_year"]
    yrs = sorted(by["year"].unique())
    M["YearFirst"], M["YearLast"] = yrs[0], yrs[-1]
    dip_sets = []
    for e in TREES:
        y = by[(by["est"] == e) & (by["cls"] == "NON-STANDARD")].groupby("year")["share"].sum()
        h = by[(by["est"] == e) & (by["variable"] == "har_ma")].set_index("year")["share"]
        pooled = cl.loc[(e, "NON-STANDARD"), "share"]
        dip_sets.append({int(t) for t in y.index if y[t] < pooled})
        k = MAC[e]
        M[k + "NsYearMin"], M[k + "NsYearMinYr"] = p1(y.min()), int(y.idxmin())
        M[k + "NsYearMax"], M[k + "NsYearMaxYr"] = p1(y.max()), int(y.idxmax())
        M[k + "HarYearMax"], M[k + "HarYearMaxYr"] = p1(h.max()), int(h.idxmax())
        M[k + "HarYearMin"], M[k + "HarYearMinYr"] = p1(h.min()), int(h.idxmin())
    yr_min = {
        int(by[(by["est"] == e) & (by["cls"] == "NON-STANDARD")].groupby("year")["share"].sum().idxmin()) for e in TREES
    }
    assert len(yr_min) == 1, yr_min
    M["NsYearMinYr"] = (
        yr_min.pop(),
        "the year of every tree's lowest non-standard share",
    )
    dip = sorted(set.intersection(*dip_sets))
    dip_a, dip_b = year_range(dip)
    M["DipFirst"] = (
        dip_a,
        "first of the years in which every tree's non-standard share is below its pooled share",
    )
    M["DipLast"] = dip_b
    for m in LINS:
        y = D["lin_year"][m][[x for x in D["exog"] if x in D["nonstd"]]].sum(1)
        k = MAC[m]
        M[k + "NsExYearMin"], M[k + "NsExYearMinYr"] = p1(y.min()), int(y.idxmin())
        M[k + "NsExYearMax"], M[k + "NsExYearMaxYr"] = p1(y.max()), int(y.idxmax())
    for e in TREES:
        y = D["tree_year"][e][[x for x in D["exog"] if x in D["nonstd"]]].sum(1)
        k = MAC[e]
        M[k + "NsExYearMin"], M[k + "NsExYearMax"] = p1(y.min()), p1(y.max())
    # ---------------------------------------------------------------- M3 shapes
    sh = T["m3_shapes"]
    for e in TREES:
        s = sh[sh["est"] == e]
        k = MAC[e]
        assert (s["selection"] == ">= 0.5%").all()
        cnt = s["shape"].str.split().str[0].value_counts()
        M[k + "NShape"] = (len(s), "non-standard series at >= 0.5% (M3)")
        for lab, n in (
            ("MONOTONE", "NMono"),
            ("IRREGULAR", "NIrr"),
            ("STEP", "NStep"),
            ("FLAT", "NFlat"),
            ("THRESHOLD", "NThresh"),
        ):
            M[k + n] = int(cnt.get(lab, 0))
        M[k + "NU"] = int(s["shape"].str.contains("U-SHAPED").sum())
        M[k + "NAsym"] = int(s["asymmetry"].notna().sum())
    M["NUThreshAll"] = (
        int(sh["shape"].str.contains("U-SHAPED|INVERTED|THRESHOLD").sum()),
        "U-shaped, inverted-U or threshold profiles in any tree",
    )
    bins = T["m3_bins"]
    for vname, n in (
        ("sumabsret_vwstock", "AbsVw"),
        ("ofi_vwstock", "Ofi"),
        ("fomc_until_inv", "Until"),
    ):
        for e in TREES:
            b = bins[(bins["est"] == e) & (bins["variable"] == vname)].sort_values("bin")
            if len(b) == 0:
                continue
            M[n + MAC[e] + "Lo"] = fi.f(b["mean_shap_var_pct_forecast"].iloc[0], 1)
            M[n + MAC[e] + "Hi"] = fi.f(b["mean_shap_var_pct_forecast"].iloc[-1], 1)
    # 2026-10-28 (patched study): XGBoost's dominant column of fomc_until_inv is the 1-sample average, the other two
    # trees' the 25-sample one. UntilHorizon is the horizon most trees share; a tree that differs gets its own macro.
    # Was: hz = ...unique(); assert len(hz) == 1; M["UntilHorizon"] = (horizon(hz[0]), ...)
    hz = sh[sh["variable"] == "fomc_until_inv"].set_index("est")["dominant_column"].map(horizon).to_dict()
    assert set(hz) == set(TREES), hz
    hs = [hz[e] for e in TREES]
    common = max(hs, key=hs.count)
    assert hs.count(common) >= 2, hz
    M["UntilHorizon"] = (common, "horizon of fomc_until_inv's dominant column")
    for e in TREES:
        if hz[e] != common:
            M["UntilHorizon" + MAC[e]] = (hz[e], f"the same in {NAME[e]} (differs from the other trees')")
    # ---------------------------------------------------------------- M4 conditional
    C = T["m4_contrasts"]
    # the bars: first, last, and the two with the largest FOMC-announcement share (the same two in every tree)
    peaks = []
    for e in TREES:
        s = sub[(sub["est"] == e) & (sub["kind"] == "bar")].set_index("subset")
        peaks.append(tuple(sorted(s["bucket_fomc"].sort_values(ascending=False).index[:2])))
    assert len(set(peaks)) == 1, peaks
    pk = list(peaks[0])
    bar_list = list(sub[(sub["est"] == "lgbm") & (sub["kind"] == "bar")]["subset"])
    first_bar, last_bar = bar_list[0], bar_list[-1]
    assert bar_list == sorted(bar_list) and len(bar_list) == 13
    M["BarFirst"], M["BarLast"] = first_bar, last_bar
    M["BarPeakA"], M["BarPeakB"] = pk[0], pk[1]
    M["NBars"] = len(bar_list)
    assert (C["contrast"] == f"{last_bar} - {first_bar}").any(), "the RULE's last-minus-first bar contrast"
    for e in TREES:
        k = MAC[e]
        s = sub[sub["est"] == e].set_index("subset")
        M[k + "NsFomc"], M[k + "NsOther"] = (
            p1(s.loc["FOMC day", "ns_share"]),
            p1(s.loc["other day", "ns_share"]),
        )
        tf = s.loc["FOMC day", "top5_nonstandard"].split()
        to = s.loc["other day", "top5_nonstandard"].split()
        fu = [float(t.split(":")[1]) for t in tf if t.startswith("fomc_until_inv:")]
        assert fu, e
        M[k + "UntilFomc"] = fi.f(fu[0], 1)
        fo = [float(t.split(":")[1]) for t in to if t.startswith("fomc_until_inv:")]
        # the other-day value is printed only where fomc_until_inv is among that subset's five largest
        if fo:
            M[k + "UntilOther"] = fi.f(fo[0], 2)
        M[k + "FomcBucketFomc"] = p1(s.loc["FOMC day", "bucket_fomc"])
        M[k + "FomcBucketOther"] = p1(s.loc["other day", "bucket_fomc"])
        M[k + "HarFomc"] = p1(s.loc["FOMC day", "bucket_har_ma"])
        M[k + "HarOther"] = p1(s.loc["other day", "bucket_har_ma"])
        bars = s.loc[bar_list]
        M[k + "NsFirstBar"] = p1(bars.loc[first_bar, "ns_share"])
        M[k + "NsLastBar"] = p1(bars.loc[last_bar, "ns_share"])
        aft = bars.loc[pk, "ns_share"]
        M[k + "NsAftMin"], M[k + "NsAftMax"] = p1(aft.min()), p1(aft.max())
        mid = bars.drop(index=[first_bar, *pk])["ns_share"]
        M[k + "NsMidMin"], M[k + "NsMidMax"] = p1(mid.min()), p1(mid.max())
        M[k + "FomcBucketAft"] = p1(bars.loc[pk, "bucket_fomc"].max())
        M[k + "FomcBucketOtherBars"] = (
            p1(bars.drop(index=pk)["bucket_fomc"].median()),
            "median FOMC-announcement share over the other 11 bars",
        )
        for con, n in (
            ("FOMC day - other day", "FomcDiff"),
            (f"{last_bar} - {first_bar}", "LateDiff"),
            ("VIX high - low", "VixDiff"),
        ):
            r = C[(C["est"] == e) & (C["contrast"] == con) & (C["scope"] == "of total")].iloc[0]
            M[k + n], M[k + n + "Lo"], M[k + n + "Hi"] = (
                pp(r["diff"]),
                pp(r["lo"]),
                pp(r["hi"]),
            )
        r = C[(C["est"] == e) & (C["contrast"] == "VIX high - low") & (C["scope"] == "bucket har_ma of total")].iloc[0]
        M[k + "VixHar"], M[k + "VixHarLo"], M[k + "VixHarHi"] = (
            pp(r["diff"]),
            pp(r["lo"]),
            pp(r["hi"]),
        )
        r = C[(C["est"] == e) & (C["contrast"] == "VIX high - low") & (C["scope"] == "of exogenous")].iloc[0]
        M[k + "VixDiffEx"], M[k + "VixDiffExLo"], M[k + "VixDiffExHi"] = (
            pp(r["diff"]),
            pp(r["lo"]),
            pp(r["hi"]),
        )
    # ---------------------------------------------------------------- M5 interactions
    m5v = T["m5_variables"]
    for e in ("lgbm", "xgb"):
        k = MAC[e]
        M[k + "InterVar"] = p1(m5.loc[e, "inter_share_variable"])
        M[k + "InterCol"] = p1(m5.loc[e, "inter_share_column"])
        M[k + "InterNs"] = p1(m5.loc[e, "inter_frac_nonstandard"])
        M[k + "InterStd"] = p1(m5.loc[e, "inter_frac_standard"])
        s = m5v[m5v["est"] == e]
        for c, col in (
            ("NON-STANDARD", "inter_frac_nonstandard"),
            ("STANDARD", "inter_frac_standard"),
        ):
            h = s[s["cls"] == c]
            chk = h["inter_abs"].sum() / (h["main_abs"] + h["inter_abs"]).sum()
            assert abs(chk - m5.loc[e, col]) < 1e-9, (e, c, chk)
        h = s[(s["cls"] == "STANDARD") & (s["variable"].isin(D["exog"]))]
        M[k + "InterStdEx"] = (
            p1(h["inter_abs"].sum() / (h["main_abs"] + h["inter_abs"]).sum()),
            "descriptive: standard exogenous series only (no HAR, no calendar)",
        )
        M[k + "InterHar"] = p1(s.set_index("variable").loc["har_ma", "inter_frac"])
        weak = [
            "stocktwits_attention",
            "stocktwits_sentiment",
            "stocktwits_sentcount",
            "vvix",
            "vix3m",
            "fomc_day",
        ]
        w = s[s["variable"].isin(weak)]["inter_frac"]
        M[k + "InterWeakMin"], M[k + "InterWeakMax"] = p1(w.min(), 0), p1(w.max(), 0)
        pr = T["m5_pairs_nonstandard"]
        pr = pr[pr["est"] == e].sort_values("share", ascending=False)
        M[k + "PairOne"] = p1(pr["share"].iloc[0], 2)
        M[k + "PairTwo"] = p1(pr["share"].iloc[1], 2)
        tb = T["m5_buckets"]
        tb = tb[tb["est"] == e].set_index(["bucket_a", "bucket_b"])["share"]
        M[k + "CellHarHar"] = p1(tb[("har_ma", "har_ma")])
    pr = T["m5_pairs_nonstandard"]
    one = {
        e: tuple(pr[pr["est"] == e].sort_values("share", ascending=False).iloc[0][["var_a", "var_b"]])
        for e in ("lgbm", "xgb")
    }
    two = {
        e: tuple(pr[pr["est"] == e].sort_values("share", ascending=False).iloc[1][["var_a", "var_b"]])
        for e in ("lgbm", "xgb")
    }
    assert one["lgbm"] == one["xgb"] and two["lgbm"] == two["xgb"], (one, two)
    M["PairOneA"], M["PairOneB"] = name_tex(one["lgbm"][0]), name_tex(one["lgbm"][1])
    M["PairTwoA"], M["PairTwoB"] = name_tex(two["lgbm"][0]), name_tex(two["lgbm"][1])
    # how many of the top-10 non-standard pairs involve the HAR components
    for e in ("lgbm", "xgb"):
        p = pr[pr["est"] == e].sort_values("share", ascending=False).head(10)
        M[MAC[e] + "PairsHar"] = int(((p["var_a"] == "har_ma") | (p["var_b"] == "har_ma")).sum())
    # ---------------------------------------------------------------- linear across windows
    tr = T["fiw_trend"]
    ver = T["fiw_verdicts"]
    W = T["fiw_info"]["windows_days"]
    M["WMin"], M["WMax"], M["NWindows"] = num(min(W)), num(max(W)), len(W)
    s1 = tr[tr["source"].str.startswith("S1")]
    M["SOneMeasures"] = (
        len(s1),
        "measure x common-penalty fits of the window study (P1 and P2)",
    )
    M["SOneRises"] = (int((s1["verdict"] == "rises").sum()), "... of which rise with W")
    M["SOneFalls"] = int((s1["verdict"] == "falls").sum())
    M["SOneFlat"] = int((s1["verdict"] == "flat").sum())
    assert M_get(M, "SOneRises") == "0"
    s1v = ver[ver["source"].str.startswith("S1")]
    M["SOneNotSupported"] = (
        int((s1v["verdict"] == "NOT SUPPORTED").sum()),
        "common-penalty verdicts NOT SUPPORTED",
    )
    M["SOneVerdicts"] = len(s1v)
    fs = T["fiw_summary"]

    def stat(model: str, source: str, w: int, col: str) -> float:
        return float(fs[(fs["W"] == w) & (fs["model"] == model) & (fs["source"] == source)][col].iloc[0])

    M["RidgeNeffWMin"] = fi.f(stat("ridge", "walk", min(W), "Neff_mean"), 1)
    M["RidgeNeffWMain"] = fi.f(stat("ridge", "walk", W_MAIN, "Neff_mean"), 1)
    M["RidgeNeffWMax"] = fi.f(stat("ridge", "walk", max(W), "Neff_mean"), 1)
    r = tr[(tr["model"] == "ridge") & (tr["source"] == "walk") & (tr["measure"] == "Neff_mean")].iloc[0]
    M["RidgeNeffSlope"], M["RidgeNeffSlopeLo"], M["RidgeNeffSlopeHi"] = (
        fi.f(r["slope_per_1000d"], 2),
        fi.f(r["slope_lo"], 2),
        fi.f(r["slope_hi"], 2),
    )
    for src, n in (("S1_1000", "RidgeSOneLo"), ("S1_10000", "RidgeSOneHi")):
        M[n + "NeffWMin"] = fi.f(stat("ridge", src, min(W), "Neff_mean"), 1)
        M[n + "NeffWMax"] = fi.f(stat("ridge", src, max(W), "Neff_mean"), 1)
    M["LassoSOneVWMin"] = fi.f(stat("lasso", "S1_0.001", min(W), "V_mean"), 1)
    M["LassoSOneVWMax"] = fi.f(stat("lasso", "S1_0.001", max(W), "V_mean"), 1)
    M["LassoSOneCWMin"] = fi.f(stat("lasso", "S1_0.0001", min(W), "C_mean"), 1)
    M["LassoSOneCWMax"] = fi.f(stat("lasso", "S1_0.0001", max(W), "C_mean"), 1)
    M["LassoWalkVWMin"] = fi.f(stat("lasso", "walk", min(W), "V_mean"), 1)
    M["LassoWalkVWMax"] = fi.f(stat("lasso", "walk", max(W), "V_mean"), 1)
    M["LassoSOneLo"] = fi.pow10(1e-3)
    M["RidgeSOneLo"] = fi.pow10(1e3)
    M["RidgeSOneHi"] = fi.pow10(1e4)
    fp = T["fiw_info"]["frozen_penalties"]
    M["EnetPenWMin"] = fi.pow10(fp[str(min(W))]["enet"])
    M["EnetPenWNext"] = fi.pow10(fp[str(sorted(W)[1])]["enet"])
    M["WNext"] = num(sorted(W)[1])
    for m in LINS:
        for p in ("P1", "P2"):
            r = ver[(ver["model"] == m) & (ver["source"] == "walk") & (ver["prediction"] == p)].iloc[0]
            M[MAC[m] + p.replace("1", "One").replace("2", "Two") + "Walk"] = r["verdict"].lower()
    return M


def M_get(M: Macros, name: str) -> str:
    return [v for n, v, _ in M.rows if n == name][0]


# ============================================================================ tables
def table(
    path: Path,
    caption: str,
    label: str,
    spec: str,
    rows: list[str],
    notes: str,
    size: str = r"\small",
    sep: str = "4pt",
) -> None:
    lines = [
        HEAD,
        load_line(),
        r"\begin{table}[tbp]",
        r"\centering",
        size,
        rf"\setlength{{\tabcolsep}}{{{sep}}}",
        r"\begin{threeparttable}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{spec}}}",
        r"\toprule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]",
        r"\footnotesize",
        rf"\item \textit{{Notes:}} {notes}",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {rel(path)}")


def panel(title: str, ncol: int) -> str:
    return rf"\multicolumn{{{ncol}}}{{@{{}}l}}{{\textit{{{title}}}}} \\"


def cell_ci(v: float, lo: float, hi: float, signed: bool = False) -> str:
    g = pp if signed else p1
    return rf"{g(v)} {{\scriptsize{ci(lo, hi, signed=signed)}}}"


SAMPLE = (
    r"Pooled intraday models of the final design, rolling window of \rhfipWindowDays{} trading days, full predictor "
    r"set (\rhfipNCols{} predictors in \rhfipNGroups{} groups: \rhfipNExog{} exogenous series, the HAR components, the "
    r"calendar dummies), re-estimated at \rhfipNOrigins{} monthly origins (\rhfipFirstOrigin{} to \rhfipLastOrigin)"
    # 2026-10-26 (integration, audit results-8): the explained bars include the NYSE-holiday futures sessions, which
    # Table 1 does not score (33,970 explained vs 33,527 scored bars)
    r"; the \rhfipNRows{} explained bars include the \rhfipNHolBars{} bars of \rhfipNHolSessions{} shortened futures "
    r"sessions on NYSE holidays, which Table~\ref{tab:rh_oos} does not score (it scores \rhorNIntB{} bars)"
)


def tab_shares(T: dict, D: dict) -> None:
    bk = T["m1_buckets"].set_index(["est", "bucket"])
    cl = T["m1_classes"].set_index(["est", "cls"])
    Dt = T["m2_nonstandard_D"]
    R = T["m2_rank"]
    n = 7
    rows = [
        r" & \multicolumn{3}{c}{Tree ensembles: mean $|\phi|$} & \multicolumn{3}{c}{Linear models: $|\gamma|$} \\",
        r"\cmidrule(lr){2-4}\cmidrule(l){5-7}",
        r" & " + " & ".join(NAME[k] for k in (*TREES, *LINS)) + r" \\",
        r"\midrule",
        panel("Panel A: Share of the total attribution, by category (\\%)", n),
    ]
    for b in BUCKETS:
        lin = [p1(sum(D["lin_all"][m][g] for g in D["groups"] if D["bucket"][g] == b)) for m in LINS]
        tre = [cell_ci(bk.loc[(e, b), "share"], bk.loc[(e, b), "lo"], bk.loc[(e, b), "hi"]) for e in TREES]
        rows.append(BUCKET_WORD[b] + " & " + " & ".join(tre + lin) + r" \\")
    rows += [r"\addlinespace", panel("Panel B: Share by class (\\%)", n)]
    for c, lab, col in (
        ("STANDARD", "Standard", "share"),
        ("NON-STANDARD", "Non-standard", "share"),
        ("NON-STANDARD", r"\quad of the exogenous total", "share_exog"),
    ):
        lo, hi = ("lo", "hi") if col == "share" else ("lo_exog", "hi_exog")
        tre = [cell_ci(cl.loc[(e, c), col], cl.loc[(e, c), lo], cl.loc[(e, c), hi]) for e in TREES]
        if col == "share":
            mem = D["std"] if c == "STANDARD" else D["nonstd"]
            lin = [p1(D["lin_all"][m][mem].sum()) for m in LINS]
        else:
            lin = [p1(D["lin_ex"][m][[x for x in D["exog"] if x in D["nonstd"]]].sum()) for m in LINS]
        rows.append(lab + " & " + " & ".join(tre + lin) + r" \\")
    rows += [
        r"\addlinespace",
        panel(
            "Panel C: $D$, tree minus linear model, non-standard share of the exogenous total (pp)",
            n,
        ),
    ]
    for m in LINS:
        cells = []
        for e in TREES:
            r = Dt[(Dt["est"] == e) & (Dt["linear"] == m) & (Dt["denominator"] == "of exogenous total")].iloc[0]
            cells.append(cell_ci(r["D"], r["lo"], r["hi"], signed=True))
        rows.append(rf"vs {NAME[m].lower()} & " + " & ".join(cells) + r" & & & \\")
    rows += [
        r"\addlinespace",
        panel(
            r"Panel D: Concentration and rank agreement over the \rhfipNExog{} exogenous series",
            n,
        ),
    ]
    rows.append(r"$N_{\mathrm{eff}}$ & " + " & ".join(fi.f(D["conc"][k][0], 1) for k in (*TREES, *LINS)) + r" \\")
    rows.append(r"Series holding half & " + " & ".join(str(D["conc"][k][1]) for k in (*TREES, *LINS)) + r" \\")
    rows.append(r"Series holding 80\% & " + " & ".join(str(D["conc"][k][2]) for k in (*TREES, *LINS)) + r" \\")
    for m in LINS:
        sp = []
        for e in TREES:
            r = R[(R["est"] == e) & (R["linear"] == m) & (R["scope"] == "46 exogenous")].iloc[0]
            sp.append(rf"{fi.f(r['spearman'], 2)} ({int(r['top10_overlap'])})")
        rows.append(rf"Spearman, {NAME[m].lower()} & " + " & ".join(sp) + r" & & & \\")
    notes = (
        SAMPLE + r"; together the origins' models forecast all \rhfipNRows{} bars from \rhfipFirstOrigin{} to "
        r"\rhfipLastDay. Tree ensembles: the share $s_A$ of group $A$ in the total mean absolute TreeSHAP value over "
        r"all bars, $\phi_A$ summed over the group's predictors, with \rhfipCiLevel\% moving-block bootstrap intervals over the "
        r"origins (blocks of \rhfipBootBlock{} origins, \rhfipBootB{} resamples) in brackets. Linear models: the share of "
        r"the absolute coefficients on the window-standardized predictors, $|\gamma_{k,j}| = |\hat\beta_{k,j}|\,"
        r"\mathrm{sd}_{k,j}$, summed over the origins (penalties: ridge \rhfipRidgePen, lasso \rhfipLassoPen, elastic net "
        r"\rhfipEnetPen); the HAR components carry the penalty factor \rhfipHarPf, so their linear shares and the "
        r"linear ``of the total'' rows are not comparable with the trees'. The two measures differ, so the table "
        r"compares shares of each model's own measure. Standard: the HAR components, the calendar dummies, the \rhfipNMom{} "
        r"futures realized moments, the VIX, the futures volume and trade count (\rhfipNStd{} groups); non-standard: the "
        r"other \rhfipNNonStd{} series. $D$: paired bootstrap interval. Panel D, computed here from the pooled shares "
        r"(descriptive): $N_{\mathrm{eff}}$, the effective number of series, and the fewest series whose shares sum to "
        r"half and to 80\% of the exogenous total; Spearman rank correlation of the \rhfipNExog{} shares with the "
        r"linear model's, and in parentheses the overlap of the two top-10 lists."
    )
    table(
        GEN / f"rehaul_fip_shares{TAG}.tex",
        "Attribution by category and class: tree ensembles and linear models",
        "tab:rh_fip_shares",
        r"@{}lccc rrr@{}",
        rows,
        notes,
        size=r"\footnotesize",
        sep="3pt",
    )


SHAPE_CODE = {
    "MONOTONE up": r"M$\uparrow$",
    "MONOTONE down": r"M$\downarrow$",
    "IRREGULAR": "I",
    "STEP up": r"S$\uparrow$",
    "STEP down": r"S$\downarrow$",
    "FLAT": "F",
    # 2026-10-28: the patched study classifies one profile THRESHOLD (XGBoost, effspread_ewstock)
    "THRESHOLD up": r"T$\uparrow$",
    "THRESHOLD down": r"T$\downarrow$",
}


def shape_cell(r: pd.Series | None) -> str:
    if r is None:
        return "--"
    s = str(r["shape"])
    code = SHAPE_CODE.get(s)
    if code is None:  # U-SHAPED / INVERTED-U were never assigned (asserted in main); print the word
        code = s.lower()
    a = r["asymmetry"]
    if isinstance(a, str):
        code += r"$^{\mathrm{t}}$" if "top" in a else r"$^{\mathrm{b}}$"
    return code


def tab_nonstd(T: dict, D: dict) -> None:
    sh = T["m3_shapes"]
    n_thresh = int(sh["shape"].str.startswith("THRESHOLD").sum())
    m5v = T["m5_variables"].set_index(["est", "variable"])["inter_frac"]
    ta, rk = D["tree_all"], D["rank_ex"]
    sel = sorted(set(sh["variable"]))
    ridge_top = D["lin_ex"]["ridge"].sort_values(ascending=False).index[:10]
    extra = [v for v in ridge_top if v in D["nonstd"] and v not in sel]
    mean_share = sum(ta[e] for e in TREES) / 3
    sel = sorted(sel, key=lambda v: -mean_share[v])
    extra = sorted(extra, key=lambda v: -mean_share[v])
    D["profile_rows"] = sel + extra
    n = 10

    def row(v: str) -> str:
        sh_c = [p1(ta[e][v], 2) for e in TREES]
        tr = int(np.median([rk[e][v] for e in TREES]))
        rr = int(rk["ridge"][v])
        shp = []
        for e in TREES:
            s = sh[(sh["est"] == e) & (sh["variable"] == v)]
            shp.append(shape_cell(s.iloc[0] if len(s) else None))
        it = [p1(m5v[(e, v)], 0) for e in ("lgbm", "xgb")]
        return " & ".join([name_tex(v), *sh_c, str(tr), str(rr), *shp, *it]) + r" \\"

    rows = [
        r" & \multicolumn{3}{c}{Share (\%)} & \multicolumn{2}{c}{Rank} & \multicolumn{3}{c}{Response shape} & "
        r"\multicolumn{2}{c}{Interaction (\%)} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(lr){7-9}\cmidrule(l){10-11}",
        r"Series & LGBM & XGB & RF & Trees & Ridge & LGBM & XGB & RF & LGBM & XGB \\",
        r"\midrule",
        panel(
            r"Panel A: Non-standard series with a share of at least \rhfipShapeMin\% in some tree ensemble",
            n + 1,
        ),
        *[row(v) for v in sel],
        r"\addlinespace",
        panel(
            r"Panel B: Non-standard series in the ridge's top ten, below \rhfipShapeMin\% in every tree ensemble",
            n + 1,
        ),
        *[row(v) for v in extra],
    ]
    notes = (
        SAMPLE + r". Share: of the total mean absolute SHAP value (LGBM: LightGBM, XGB: XGBoost, RF: random forest). "
        r"Rank: among the \rhfipNExog{} exogenous series, the median over the three tree ensembles and the ridge's "
        r"(by its share of $|\gamma|$, Table~\ref{tab:rh_fip_shares}). Response shape: the mean SHAP value of the "
        r"series in each decile of its dominant predictor (the value average with the largest mean absolute SHAP), "
        r"classified by the rule fixed before computing (Figure~\ref{fig:rh_fip_shapes}): M$\uparrow$/M$\downarrow$ "
        r"monotone up/down (absolute Spearman correlation of the decile means with the decile index at least 0.9), "
        # 2026-10-28: the threshold code and the closing sentence follow the profiles (one threshold in the patched
        # study); with none, the note is the one of 2026-10-23
        + (
            r"T$\uparrow$/T$\downarrow$ threshold (monotone, with one step between adjacent deciles at least half the "
            r"total change), "
            if n_thresh
            else ""
        )
        + r"S "
        r"step (three bins or fewer), F flat (the decile means move less than half the mean absolute SHAP value), I "
        r"irregular; t/b: asymmetric, the top/bottom arm at least twice the other; -- below \rhfipShapeMin\% in that model, not "
        + (
            r"classified. No profile was classified as U-shaped or inverted-U. Interaction: the share of the "
            if n_thresh
            else r"classified. No profile was classified as a threshold, U-shaped or inverted-U. Interaction: the share of the "
        )
        + r"series' SHAP interaction attribution that sits off the diagonal (with other groups), on "
        r"\rhfipInterRows{} bars (\rhfipInterPerOrigin{} drawn per origin); not computed for the random forest."
    )
    table(
        GEN / f"rehaul_fip_nonstd{TAG}.tex",
        "The leading non-standard series in the tree ensembles",
        "tab:rh_fip_nonstd",
        r"@{}l rrr rr ccc rr@{}",
        rows,
        notes,
        size=r"\footnotesize",
        sep="3.5pt",
    )


def tab_inter(T: dict, D: dict) -> None:
    m5 = T["m5_summary"].set_index("est")
    m5v = T["m5_variables"]
    pr = T["m5_pairs_nonstandard"]
    E = ("lgbm", "xgb")
    rows = [
        r" & LightGBM & XGBoost \\",
        r"\midrule",
        panel(r"Panel A: Interaction share of the attribution (\%)", 3),
        r"All groups, group level & " + " & ".join(p1(m5.loc[e, "inter_share_variable"]) for e in E) + r" \\",
        r"All predictors, predictor level & " + " & ".join(p1(m5.loc[e, "inter_share_column"]) for e in E) + r" \\",
        r"Non-standard series & " + " & ".join(p1(m5.loc[e, "inter_frac_nonstandard"]) for e in E) + r" \\",
        r"Standard groups & " + " & ".join(p1(m5.loc[e, "inter_frac_standard"]) for e in E) + r" \\",
    ]
    cells = []
    for e in E:
        h = m5v[(m5v["est"] == e) & (m5v["cls"] == "STANDARD") & (m5v["variable"].isin(D["exog"]))]
        cells.append(p1(h["inter_abs"].sum() / (h["main_abs"] + h["inter_abs"]).sum()))
    rows.append(r"\quad standard exogenous series only$^{a}$ & " + " & ".join(cells) + r" \\")
    hv = [p1(m5v[(m5v["est"] == e) & (m5v["variable"] == "har_ma")]["inter_frac"].iloc[0]) for e in E]
    rows.append(r"\quad HAR components & " + " & ".join(hv) + r" \\")
    rows += [
        r"\addlinespace",
        panel(
            r"Panel B: Largest pairs with a non-standard series, share of the pair total (\%)",
            3,
        ),
    ]
    pl = pr[pr["est"] == "lgbm"].sort_values("share", ascending=False).head(10)
    px = pr[pr["est"] == "xgb"].set_index(["var_a", "var_b"])["share"]
    for _, r in pl.iterrows():
        rows.append(
            rf"{name_tex(r['var_a'])} $\times$ {name_tex(r['var_b'])} & {p1(r['share'], 2)} & "
            rf"{p1(px[(r['var_a'], r['var_b'])], 2)} \\"
        )
    notes = (
        r"LightGBM and XGBoost of Table~\ref{tab:rh_fip_shares}; SHAP interaction values $\phi_{jl}$ on "
        r"\rhfipInterRows{} bars, \rhfipInterPerOrigin{} drawn at random from each origin's block and explained by that "
        r"origin's model; summed over the predictors of each group, $G_{AB} = \sum_{j \in A}\sum_{l \in B} \phi_{jl}$. "
        r"Panel A: the share of $\sum |G_{AB}|$ off the diagonal ($A \ne B$), for all groups and at the predictor "
        r"level, and pooled over the members of each class. $^{a}$Computed here from the per-series file (descriptive): "
        r"the standard class without the HAR components and the calendar dummies. Panel B: the ten largest pairs by "
        r"LightGBM, mean $|2G_{AB}|$ as a share of the total over all pairs $A \le B$ (diagonal included). No intervals: "
        r"the interaction values are descriptive."
    )
    table(
        GEN / f"rehaul_fip_inter{TAG}.tex",
        "SHAP interactions in the boosted trees",
        "tab:rh_fip_inter",
        r"@{}lrr@{}",
        rows,
        notes,
    )


def tab_window(T: dict, D: dict) -> None:
    fs, tr, ver = T["fiw_summary"], T["fiw_trend"], T["fiw_verdicts"]
    W = sorted(T["fiw_info"]["windows_days"])
    fp = T["fiw_info"]["frozen_penalties"]

    def st(m: str, s: str, w: int, c: str) -> float:
        return float(fs[(fs["W"] == w) & (fs["model"] == m) & (fs["source"] == s)][c].iloc[0])

    def verdict(m: str, s: str, p: str) -> str:
        r = ver[(ver["model"] == m) & (ver["source"] == s) & (ver["prediction"] == p)].iloc[0]
        word = {"SUPPORTED": "yes", "PARTLY": "partly", "NOT SUPPORTED": "no"}[r["verdict"]]
        return rf"{word} ({int(r['rises'])}/{int(r['flat'])}/{int(r['falls'])})"

    def slope(m: str, s: str) -> str:
        r = tr[(tr["model"] == m) & (tr["source"] == s) & (tr["measure"] == "Neff_mean")].iloc[0]
        return rf"{fi.f(r['slope_per_1000d'], 2)} {{\scriptsize[{fi.f(r['slope_lo'], 2)}, {fi.f(r['slope_hi'], 2)}]}}"

    def pexp(x: float) -> str:
        e = int(round(np.log10(x)))
        assert abs(x - 10.0**e) < 1e-12 * max(1.0, x)
        return f"10^{{{e}}}"

    def pen_walk(m: str) -> str:
        uniq = list(dict.fromkeys(pexp(fp[str(w)][m]) for w in W))
        return "tuned: $" + ", ".join(uniq) + "$"

    def src_label(m: str, s: str) -> str:
        if s == "walk":
            return "Walk"
        a = s.split("_", 1)[1]
        if a == "rel":
            return r"Common, $10^{4}\,W/1{,}000$"
        return "Common, " + fi.pow10(float(a))

    rows = [
        r" & & \multicolumn{2}{c}{Series selected} & \multicolumn{3}{c}{$N_{\mathrm{eff}}$, mean over origins} & "
        r"\multicolumn{2}{c}{Prediction supported} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-7}\cmidrule(l){8-9}",
        rf"Model & Penalty & $W={num(W[0])}$ & ${num(W[-1])}$ & $W={num(W[0])}$ & ${num(W[-1])}$ & "
        r"Slope$^{a}$ & P1 & P2 \\",
        r"\midrule",
    ]
    order = {
        "ridge": ["walk", "S1_1000", "S1_10000", "S1_rel"],
        "lasso": ["walk", "S1_0.0001", "S1_0.001"],
        "enet": ["walk", "S1_0.001", "S1_0.01"],
    }
    for m in LINS:
        for i, s in enumerate(order[m]):
            lab = NAME[m] if i == 0 else ""
            pen = pen_walk(m) if s == "walk" else src_label(m, s).replace("Common, ", "fixed: ")
            vsel = ["--", "--"] if m == "ridge" else [fi.f(st(m, s, w, "V_mean"), 1) for w in (W[0], W[-1])]
            ne = [fi.f(st(m, s, w, "Neff_mean"), 1) for w in (W[0], W[-1])]
            p1v = "--" if m == "ridge" else verdict(m, s, "P1")
            rows.append(
                " & ".join(
                    [
                        f"{lab}",
                        pen,
                        *vsel,
                        *ne,
                        slope(m, s),
                        p1v,
                        verdict(m, s, "P2"),
                    ]
                )
                + r" \\"
            )
        if m != LINS[-1]:
            rows.append(r"\addlinespace")
    notes = (
        r"The pooled ridge, lasso and elastic net on the full predictor set, rolling windows $W = "
        + ", ".join(num(w) for w in W)
        + r"$ trading days, each walk's coefficients at the same \rhfipNOrigins{} monthly origins; tuned: each $W$'s walk at "
        r"the penalty its pre-sample tuning chose (listed from the shortest to the longest window where it changes); "
        r"fixed: the fit at the origin at one penalty for every $W$ (ridge also at a penalty proportional to $W$). Series selected: mean number of exogenous series with a "
        r"nonzero coefficient (the ridge keeps every series that has data). $N_{\mathrm{eff}}$: effective number of "
        r"exogenous series, $1/\sum_A w_A^2$ with $w_A$ the series' share of $\sum_j |\gamma_j|$ over the exogenous "
        r"predictors at an origin; $^{a}$slope of the per-$W$ value on $W$, per \rhfipSlopeUnit{} days, with its \rhfipCiLevel\% paired moving-block bootstrap "
        r"interval. If the exogenous signal were dense and weak, a longer window would select more series (P1, lasso "
        r"and elastic net: series and predictors selected, mean, median and in half the origins; six measures) and "
        r"spread the weight over more series (P2: $N_{\mathrm{eff}}$ and the fewest series holding half and 80\% of "
        r"the weight, per origin and pooled; nine measures). Supported: every measure rises; partly: some rise; no: "
        r"none rises; in parentheses the numbers of measures that rise, stay flat and fall. Predictions, measures and "
        r"rule were fixed before computing."
    )
    table(
        GEN / f"rehaul_fip_window{TAG}.tex",
        "Does a longer window bring in more exogenous series? The linear models across training windows",
        "tab:rh_fip_window",
        r"@{}ll rr rr r ll@{}",
        rows,
        notes,
        size=r"\footnotesize",
        sep="2.4pt",
    )


# ============================================================================ figures
def _save(fig, stem: str) -> None:
    w, h = fig.get_size_inches()
    assert abs(w - fi.FIG_W) < 1e-9, (stem, w)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{stem}{TAG}.pdf")
    if PREVIEW is not None:
        PREVIEW.mkdir(parents=True, exist_ok=True)
        fig.savefig(PREVIEW / f"{stem}{TAG}.png", dpi=170)
    plt.close(fig)
    print(f"wrote {rel(FIG / (stem + TAG + '.pdf'))} ({w:.2f} x {h:.2f} in)")


def float_fig(stem: str, caption: str, label: str) -> None:
    path = GEN / f"rehaul_fip_fig_{stem.removeprefix('fip_')}{TAG}.tex"
    path.write_text(
        HEAD
        + load_line()
        + "\n"
        + "\n".join(
            [
                r"\begin{figure}[tbp]",
                r"\centering",
                rf"\includegraphics[width=\linewidth]{{figures/fi_pooled/{stem}{TAG}.pdf}}",
                rf"\caption{{{caption}}}",
                rf"\label{{{label}}}",
                r"\end{figure}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {rel(path)}")


def _label_style(ax, names: list[str], ns: set[str]) -> None:
    for t, v in zip(ax.get_yticklabels(), names):
        if RAW_NAMES and v not in fi.GROUP_WORDS:
            t.set_fontfamily("monospace")
            t.set_fontsize(7)
        if v in ns:
            t.set_fontweight("bold")


def fig_rank(T: dict, D: dict) -> None:
    ex = D["exog"]
    mean_tree = sum(D["tree_ex"][e] for e in TREES) / 3
    top = mean_tree.sort_values(ascending=False).index[:20].tolist()
    rtop = D["lin_ex"]["ridge"].sort_values(ascending=False).index[:10].tolist()
    rows = top + [v for v in rtop if v not in top]
    D["rank_rows"] = rows
    ns = {v for v in ex if v in D["nonstd"]}
    fig, ax = plt.subplots(figsize=(fi.FIG_W, 5.2), layout="constrained")
    y = np.arange(len(rows))
    off = {"lgbm": -0.21, "xgb": -0.07, "rf_skx": 0.07, "ridge": 0.21}
    for k in ("lgbm", "xgb", "rf_skx", "ridge"):
        s = (D["tree_ex"][k] if k in TREES else D["lin_ex"][k]).reindex(rows) * 100
        filled = k in TREES
        ax.scatter(
            s.to_numpy(),
            y + off[k],
            marker=MARK[k],
            s=16 if filled else 20,
            facecolors=COLOR[k] if filled else "none",
            edgecolors=COLOR[k],
            linewidths=0.8,
            zorder=3,
            label=NAME[k] + (" (mean $|\\phi|$)" if filled else " ($|\\gamma|$)"),
        )
    for i in range(len(rows)):
        ax.axhline(i, color=fi.GRID, lw=0.4, zorder=0)
    if len(rows) > len(top):
        ax.axhline(len(top) - 0.5, color=fi.INK2, lw=0.6, ls=":")
    ax.set_yticks(y)
    ax.set_yticklabels([name_mpl(v) for v in rows])
    _label_style(ax, rows, ns)
    ax.set_ylim(len(rows) - 0.5, -0.6)
    ax.set_xlim(left=0)
    ax.set_xlabel(r"share of the exogenous total (%), each model's own measure")
    ax.grid(axis="x", color=fi.GRID, lw=0.4)
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color="none"))
    labels.append("non-standard series in bold")
    ax.legend(handles, labels, loc="lower right", frameon=False)
    _save(fig, "fip_rank")
    nr = len(rows) - len(top)
    cap = (
        r"Which exogenous series the models use. Each series' share of the exogenous attribution, pooled over the "
        r"\rhfipNOrigins{} monthly origins: the tree ensembles' share of the mean absolute SHAP value (filled markers) "
        r"and the ridge's share of the absolute standardized coefficients $|\gamma|$ (open diamonds). Rows: the 20 "
        r"series with the largest mean share over the three tree ensembles, in that order"
        + (
            rf", then (below the dotted line) the {fi.word(nr)} series in the ridge's top ten that are not among them"
            if nr
            else ""
        )
        + r". Non-standard series are labeled in bold. The two measures differ; the figure compares rankings and "
        r"shares, not magnitudes (Table~\ref{tab:rh_fip_shares})."
    )
    float_fig("fip_rank", cap, "fig:rh_fip_rank")


def fig_time(T: dict, D: dict) -> None:
    ns_ex = [v for v in D["exog"] if v in D["nonstd"]]
    fig = plt.figure(figsize=(fi.FIG_W, 6.6), layout="constrained")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.9], width_ratios=[1.0, 1.0, 0.035])
    ax = fig.add_subplot(gs[0, :])
    for e in TREES:
        y = D["tree_year"][e][ns_ex].sum(1) * 100
        ax.plot(
            y.index,
            y.to_numpy(),
            color=COLOR[e],
            marker=MARK[e],
            ms=3.5,
            lw=1.2,
            label=NAME[e],
        )
    for m in LINS:
        y = D["lin_year"][m][ns_ex].sum(1) * 100
        ax.plot(
            y.index,
            y.to_numpy(),
            color=COLOR[m],
            marker=MARK[m],
            ms=3.5,
            lw=1.0,
            ls="--",
            label=NAME[m],
        )
    yrs = sorted(D["tree_year"]["lgbm"].index)
    ax.set_xticks(yrs)
    ax.set_xticklabels([str(y) if y != yrs[-1] else f"{y}\n(Jan-Apr)" for y in yrs])
    ax.set_ylabel("non-standard share of\nthe exogenous total (%)")
    ax.grid(axis="y", color=fi.GRID, lw=0.4)
    ax.set_axisbelow(True)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.22 * (hi - lo))
    ax.legend(
        ncol=6,
        loc="upper center",
        frameon=False,
        fontsize=7.5,
        handlelength=2.2,
        columnspacing=1.2,
    )
    ax.set_title("(a) Non-standard share by year", loc="left", fontsize=9)
    # heatmaps: LightGBM and ridge, the union of their top-8 series
    lt = D["tree_ex"]["lgbm"].sort_values(ascending=False).index[:8].tolist()
    rt = D["lin_ex"]["ridge"].sort_values(ascending=False).index[:8].tolist()
    rows = lt + [v for v in rt if v not in lt]
    D["time_rows"] = rows
    mats = {
        "lgbm": D["tree_year"]["lgbm"][rows].T.to_numpy() * 100,
        "ridge": D["lin_year"]["ridge"].reindex(index=yrs)[rows].T.to_numpy() * 100,
    }
    vmax = max(np.nanmax(a) for a in mats.values())
    norm = PowerNorm(0.5, vmin=0, vmax=vmax)
    cm = fi._cmap()
    ns = set(D["nonstd"])
    for j, (k, title) in enumerate((("lgbm", "(b) LightGBM, mean $|\\phi|$"), ("ridge", "(c) Ridge, $|\\gamma|$"))):
        a = fig.add_subplot(gs[1, j])
        A = mats[k]
        im = a.imshow(A, aspect="auto", cmap=cm, norm=norm, interpolation="nearest")
        for (i, jj), val in np.ndenumerate(A):
            if np.isfinite(val):
                a.text(
                    jj,
                    i,
                    f"{val:.1f}" if val < 9.95 else f"{val:.0f}",
                    ha="center",
                    va="center",
                    fontsize=5,
                    color="white" if norm(val) < 0.6 else "black",
                )
        a.set_xticks(range(len(yrs)))
        a.set_xticklabels([f"{y % 100:02d}" for y in yrs], fontsize=7)
        a.set_yticks(range(len(rows)))
        if j == 0:
            a.set_yticklabels([name_mpl(v) for v in rows])
            _label_style(a, rows, ns)
        else:
            a.set_yticklabels([])
        a.tick_params(length=0)
        for sp in a.spines.values():
            sp.set_visible(False)
        a.set_title(title, loc="left", fontsize=9)
        a.set_xlabel("year (20..)")
    cax = fig.add_subplot(gs[1, 2])
    fig.colorbar(im, cax=cax, label="share of the exogenous total (%), square-root scale")
    _save(fig, "fip_time")
    cap = (
        r"Variable importance over time. (a) The \rhfipNNonStd{} non-standard series' share of each model's "
        r"exogenous attribution by calendar year: tree ensembles (solid) by the year of the forecast bar, linear "
        r"models (dashed) by the year of the origin whose coefficients enter. (b), (c) The share of the exogenous "
        r"total by year for the leading series of LightGBM and of the ridge (the union of each model's eight largest "
        r"pooled shares; non-standard series in bold), on one square-root color scale. Shares of each model's own "
        r"measure: mean absolute SHAP value for the trees, $|\gamma|$ for the linear models. The linear shares by year "
        r"are computed here from the coefficients at the origins (descriptive). \rhfipYearLast{} covers January to "
        r"April."
    )
    float_fig("fip_time", cap, "fig:rh_fip_time")


def fig_cond(T: dict, D: dict) -> None:
    sub = T["m4_subsets"]
    fig = plt.figure(figsize=(fi.FIG_W, 3.9), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.75, 1.0], height_ratios=[1.5, 1.0])
    a1 = fig.add_subplot(gs[0, 0])
    a2 = fig.add_subplot(gs[1, 0], sharex=a1)
    a3 = fig.add_subplot(gs[:, 1])
    bars = [b for b in sub[sub["est"] == "lgbm"]["subset"] if re.fullmatch(r"\d\d:\d\d", str(b))]
    x = np.arange(len(bars))
    for e in TREES:
        s = sub[(sub["est"] == e) & (sub["kind"] == "bar")].set_index("subset").reindex(bars)
        a1.fill_between(x, s["ns_lo"] * 100, s["ns_hi"] * 100, color=COLOR[e], alpha=0.15, lw=0)
        a1.plot(
            x,
            s["ns_share"] * 100,
            color=COLOR[e],
            marker=MARK[e],
            ms=3,
            lw=1.1,
            label=NAME[e],
        )
        a2.plot(x, s["bucket_fomc"] * 100, color=COLOR[e], marker=MARK[e], ms=3, lw=1.0)
    a1.set_ylabel("non-standard\nshare (%)")
    a2.set_ylabel("FOMC-announcement\nshare (%)")
    a2.set_xticks(x)
    a2.set_xticklabels(bars, rotation=90, fontsize=7)
    a2.set_xlabel("bar (end, ET)")
    plt.setp(a1.get_xticklabels(), visible=False)
    for a in (a1, a2):
        a.grid(axis="y", color=fi.GRID, lw=0.4)
        a.set_axisbelow(True)
    a1.set_title("(a) By bar of the day", loc="left", fontsize=9)
    a1.legend(loc="upper center", ncol=3, frameon=False, fontsize=7.5)
    cats = [
        ("fomc", "FOMC day"),
        ("fomc", "other day"),
        ("vix", "VIX low"),
        ("vix", "VIX mid"),
        ("vix", "VIX high"),
    ]
    lab = {
        "FOMC day": "FOMC days",
        "other day": "other days",
        "VIX low": "VIX low",
        "VIX mid": "VIX middle",
        "VIX high": "VIX high",
    }
    yy = np.arange(len(cats))
    offs = {"lgbm": -0.2, "xgb": 0.0, "rf_skx": 0.2}
    for e in TREES:
        s = sub[sub["est"] == e].set_index("subset")
        for i, (_, c) in enumerate(cats):
            r = s.loc[c]
            a3.errorbar(
                r["ns_share"] * 100,
                i + offs[e],
                xerr=[
                    [100 * (r["ns_share"] - r["ns_lo"])],
                    [100 * (r["ns_hi"] - r["ns_share"])],
                ],
                fmt=MARK[e],
                color=COLOR[e],
                ms=3.5,
                lw=0.9,
                capsize=0,
            )
    a3.set_yticks(yy)
    a3.set_yticklabels([lab[c] for _, c in cats])
    a3.set_ylim(len(cats) - 0.5, -0.5)
    a3.axhline(1.5, color=fi.INK2, lw=0.5, ls=":")
    a3.set_xlabel("non-standard share (%)")
    a3.grid(axis="x", color=fi.GRID, lw=0.4)
    a3.set_axisbelow(True)
    a3.set_title("(b) By day type and VIX tercile", loc="left", fontsize=9)
    _save(fig, "fip_cond")
    cap = (
        r"When the tree ensembles use the non-standard series. Share of the non-standard series in the total mean "
        r"absolute SHAP value on subsets of the \rhfipNRows{} bars, with \rhfipCiLevel\% block-bootstrap intervals over the "
        r"origins. (a) By bar of the day (top; bands: intervals) and the FOMC-announcement category's share (bottom). "
        # 2026-10-26 (audit results-9): the SHAP subset holds 78 of the 81 statement days (shap_reduce.py, fixed for
        # the rerun). Was: r"(b) On the \rhfipFomcDays{} scheduled FOMC statement days and on all other days, ..."
        # 2026-10-28: the patched rerun's subset also holds 78 (its reducer kept releases.parquet's days); the full
        # wording returns by itself once a subset holds all 81.
        + (
            r"(b) On the \rhfipFomcDays{} scheduled FOMC statement days and on all other days, and by the tercile of "
            r"the "
            if fomc_days_shap(T) == fomc_statement_days()
            else r"(b) On \rhfipFomcDaysShap{} of the \rhfipFomcDays{} scheduled FOMC statement days (the other "
            r"\rhfipFomcDaysMissing{} are counted with the other days) and on all other days, and by the tercile of the "
        )
        + r"VIX close before the origin (cut points \rhfipVixLo{} and \rhfipVixHi; every bar of an origin's block takes "
        r"that origin's tercile)."
    )
    float_fig("fip_cond", cap, "fig:rh_fip_cond")


def fig_shapes(T: dict, D: dict) -> None:
    bins = T["m3_bins"]
    sh = T["m3_shapes"]
    lead = D["lead_ns"][:5]
    extra = [v for v in D["profile_rows"] if v not in lead][:1]
    vs = lead + extra
    fig, axes = plt.subplots(2, 3, figsize=(fi.FIG_W, 4.2), layout="constrained", sharex=True)
    for a, v in zip(axes.ravel(), vs):
        for e in TREES:
            b = bins[(bins["est"] == e) & (bins["variable"] == v)].sort_values("bin")
            if len(b) == 0:
                continue
            a.plot(
                b["bin"],
                b["mean_shap_var_pct_forecast"],
                color=COLOR[e],
                marker=MARK[e],
                ms=2.8,
                lw=1.0,
                label=NAME[e],
            )
        a.axhline(0, color=fi.INK2, lw=0.5)
        dom = sh[sh["variable"] == v]["dominant_column"].iloc[0]
        h = horizon(dom)
        # 2026-10-28: where the trees' dominant horizons differ (patched study: XGBoost's fomc_until_inv), the title
        # gives the most common one and, on a second line, the exception. Was: f"{name_mpl(v)}, k = {h}" alone
        hz = sh[sh["variable"] == v].set_index("est")["dominant_column"].map(horizon)
        if hz.nunique() > 1:
            vc = hz.value_counts()
            h = str(vc.index[0]) if vc.iloc[0] > 1 else h
            h += "\n" + ", ".join(f"{NAME[e]}: k = {x}" for e, x in hz.items() if x != h)  # a second line
        a.set_title(
            f"{name_mpl(v)}, k = {h}",
            fontsize=8,
            loc="left",
            fontdict={"family": "monospace"} if RAW_NAMES else None,
        )
        a.grid(axis="y", color=fi.GRID, lw=0.4)
        a.set_axisbelow(True)
        a.set_xticks(range(1, 11))
    for a in axes[1]:
        a.set_xlabel("decile of the dominant predictor")
    for a in axes[:, 0]:
        a.set_ylabel("mean SHAP\n(% of mean forecast)")
    axes[0, 0].legend(frameon=False, fontsize=7, loc="upper left")
    _save(fig, "fip_shapes")
    cap = (
        r"Response shapes of the leading non-standard series. For each series, the mean SHAP value of the series "
        r"(all its predictors) in each decile of its dominant predictor, the value average at horizon $k$ with the "
        r"largest mean absolute SHAP value, as a percentage of the mean forecast; over all \rhfipNRows{} bars. The "
        r"predictor values are the robust-scaled inputs the trees see. A model's line is drawn where the series' share "
        r"is at least \rhfipShapeMin\% (the classification in Table~\ref{tab:rh_fip_nonstd}); the five series with the largest "
        r"mean share over the three tree ensembles and the next row of that table."
    )
    float_fig("fip_shapes", cap, "fig:rh_fip_shapes")


# ============================================================================ claims
def claims(T: dict, D: dict) -> None:
    """The qualitative words of rehaul/08_feature_importance.tex ("lower", "most", "more than half", ...), checked
    against the files so a regeneration that changes them fails here instead of printing a wrong sentence."""
    print("claims in the text")
    sh, sub, C = T["m3_shapes"], T["m4_subsets"], T["m4_contrasts"]
    m5v, by = T["m5_variables"], T["m1_by_year"]
    cl = T["m1_classes"].set_index(["est", "cls"])

    def con(e: str, c: str, scope: str) -> pd.Series:
        return C[(C["est"] == e) & (C["contrast"] == c) & (C["scope"] == scope)].iloc[0]

    for e in TREES:
        s = sh[sh["est"] == e]
        check(
            s["asymmetry"].notna().sum() > len(s) / 2,
            f"{e}: most M3 profiles are asymmetric ('most responses are one-sided')",
        )
        ns = by[(by["est"] == e) & (by["cls"] == "NON-STANDARD")].groupby("year")["share"].sum()
        har = by[(by["est"] == e) & (by["variable"] == "har_ma")].set_index("year")["share"]
        dip = ns < cl.loc[(e, "NON-STANDARD"), "share"]
        check(
            har[dip].mean() > har[~dip].mean(),
            f"{e}: HAR share higher in the dip years ('the HAR components gain in the same years')",
        )
        v = T["m1_variables"]
        used = v[(v["est"] == e) & v["variable"].isin(D["exog"])]["K_used"]
        check(
            (used == len(T["fiw_origins"])).sum() > len(D["exog"]) / 2,
            f"{e}: most exogenous series split on at every origin",
        )
        check(D["conc"][e][0] < D["conc"]["ridge"][0], f"{e}: N_eff below the ridge's")
        bars = sub[(sub["est"] == e) & (sub["kind"] == "bar")].set_index("subset")["ns_share"]
        fomc = sub[(sub["est"] == e) & (sub["kind"] == "bar")].set_index("subset")["bucket_fomc"]
        top3 = set(bars.sort_values(ascending=False).index[:3])
        peaks = set(fomc.sort_values(ascending=False).index[:2])
        check(
            top3 == {bars.index[0], *peaks},
            f"{e}: the non-standard share is highest at the first bar and the two FOMC-peak bars",
        )
        r = con(e, f"{bars.index[-1]} - {bars.index[0]}", "of total")
        check(
            r["hi"] < 0,
            f"{e}: lower at the last bar than at the first (interval below 0)",
        )
        r = con(e, "FOMC day - other day", "of total")
        check(r["lo"] > 0, f"{e}: higher on FOMC statement days (interval above 0)")
        r = con(e, "VIX high - low", "of total")
        rh = con(e, "VIX high - low", "bucket har_ma of total")
        check(
            r["hi"] < 0 and rh["lo"] > 0,
            f"{e}: high-VIX tercile lower, HAR share higher",
        )
    if TAG == "_prepatch":
        check(
            con("xgb", "VIX high - low", "of exogenous")["hi"] < 0
            and con("rf_skx", "VIX high - low", "of exogenous")["hi"] < 0
            and con("lgbm", "VIX high - low", "of exogenous")["lo"]
            < 0
            < con("lgbm", "VIX high - low", "of exogenous")["hi"],
            "of the exogenous total: lower for XGBoost and the forest, no clear difference for LightGBM",
        )
    else:
        # 2026-10-28 (patched study): LightGBM's interval is below 0 as well (A_fi.tex reworded: "clear for all three,
        # least for LightGBM")
        d_ex = {e: con(e, "VIX high - low", "of exogenous") for e in TREES}
        check(
            all(d_ex[e]["hi"] < 0 for e in TREES)
            and abs(d_ex["lgbm"]["diff"]) < min(abs(d_ex["xgb"]["diff"]), abs(d_ex["rf_skx"]["diff"])),
            "of the exogenous total: lower for all three tree ensembles, least for LightGBM",
        )
    for e in ("lgbm", "xgb"):
        s = sub[sub["est"] == e].set_index("subset")

        def until(subset: str, s: pd.DataFrame = s) -> float:
            tok = [t for t in s.loc[subset, "top5_nonstandard"].split() if t.startswith("fomc_until_inv:")]
            return float(tok[0].split(":")[1]) / 100

        diff = con(e, "FOMC day - other day", "of total")["diff"]
        check(
            until("FOMC day") - until("other day") >= 0.8 * diff,
            f"{e}: fomc_until_inv accounts for 'almost all' of the FOMC-day rise",
        )
        g = m5v[m5v["est"] == e]
        f_ns = T["m5_summary"].set_index("est").loc[e, "inter_frac_nonstandard"]
        f_std = T["m5_summary"].set_index("est").loc[e, "inter_frac_standard"]
        h = g[(g["cls"] == "STANDARD") & g["variable"].isin(D["exog"])]
        f_sx = h["inter_abs"].sum() / (h["main_abs"] + h["inter_abs"]).sum()
        check(
            (f_sx - f_std) / (f_ns - f_std) > 0.5,
            f"{e}: 'more than half' of the interaction gap is the HAR components and calendar dummies",
        )
        check(0.4 < f_ns < 0.6, f"{e}: non-standard interaction fraction is 'about half'")
        pr = T["m5_pairs_nonstandard"]
        p = pr[pr["est"] == e].sort_values("share", ascending=False)
        check(
            "har_ma" in tuple(p.iloc[0][["var_a", "var_b"]]),
            f"{e}: the largest non-standard pair has the HAR components",
        )
    # trees below the ridge on N_eff and the ridge's FOMC ranks near the top
    if TAG == "_prepatch":
        check(
            D["rank_ex"]["ridge"]["fomc_until_inv"] <= 5 and D["rank_ex"]["ridge"]["fomc_day"] <= 5,
            "ridge puts the FOMC calendar 'near the top' (both in its top five)",
        )
    else:
        # 2026-10-28 (patched study): fomc_until_inv leads all three linear models, fomc_day falls to the ridge's 9th;
        # the text quotes fomc_until_inv's rank only
        check(
            all(D["rank_ex"][m]["fomc_until_inv"] <= 5 for m in LINS),
            "the linear models put the FOMC calendar 'near the top' (fomc_until_inv in each one's top five)",
        )
    # dense-and-weak: the elastic net's walk counts rise only at the first window step (its penalty step)
    fs = T["fiw_summary"]
    W = sorted(T["fiw_info"]["windows_days"])
    fp = T["fiw_info"]["frozen_penalties"]
    for m in ("enet", "lasso"):
        vm = [float(fs[(fs["W"] == w) & (fs["model"] == m) & (fs["source"] == "walk")]["V_mean"].iloc[0]) for w in W]
        for i in range(1, len(W)):
            if vm[i] > vm[i - 1]:
                check(
                    fp[str(W[i])][m] < fp[str(W[i - 1])][m],
                    f"{m}: walk V rises from W {W[i - 1]} to {W[i]} only with a lower tuned penalty",
                )
    check(
        T["fiw_verdicts"].set_index(["model", "source", "prediction"]).loc[("ridge", "walk", "P2"), "verdict"]
        == "NOT SUPPORTED",
        "ridge walk P2 not supported ('the ridge's weight does not spread')",
    )


# ============================================================================ main
def main() -> None:
    import argparse

    global PREVIEW, SHAP, WALK, TAG
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepatch", action="store_true", help="the pre-patch TreeSHAP study; writes the *_prepatch names")
    ap.add_argument("--preview", type=Path, default=None, help="also write PNG previews of the figures here")
    a = ap.parse_args()
    PREVIEW = a.preview
    if a.prepatch:
        SHAP, WALK, TAG = SHAP_PRE, WALK_PRE, "_prepatch"
    rule = SHAP / ("RULE_shap.txt" if a.prepatch else "RULE_pointer.txt")
    assert rule.is_file(), rule
    if not a.prepatch:
        assert "FOMC-PATCHED" in rule.read_text(encoding="utf-8"), f"{rule} does not label the design patched"
        gate = (SHAP / "gate.txt").read_text(encoding="utf-8")
        for e in ("LightGBM", "XGBoost", "Random forest"):
            line = next(ln for ln in gate.splitlines() if ln.startswith(e))
            assert "walk rows 126/126; bit-equal 126/126" in line and "additivity pass 126/126" in line, line
    print(f"SHAP study: {rel(SHAP)} (tag '{TAG}')")
    T = load()
    D = derive(T)
    G = gates(T, D)
    M = macros(T, D, G)
    claims(T, D)
    sh = T["m3_shapes"]
    # 2026-10-28: a threshold profile is now possible (the patched study has one; the note and the codes follow it)
    # Was: check(not sh["shape"].str.contains("THRESHOLD|U-SHAPED|INVERTED").any(), ...)
    check(
        not sh["shape"].str.contains("U-SHAPED|INVERTED").any() and sh["shape"].isin(list(SHAPE_CODE)).all(),
        f"no M3 profile is U-shaped or inverted-U, every one has a table code "
        f"({int(sh['shape'].str.startswith('THRESHOLD').sum())} threshold; the table notes say so)",
    )
    M.write(GEN / f"rehaul_fip_numbers{TAG}.tex")
    tab_shares(T, D)
    tab_nonstd(T, D)
    tab_inter(T, D)
    tab_window(T, D)
    fig_rank(T, D)
    fig_time(T, D)
    fig_cond(T, D)
    fig_shapes(T, D)


if __name__ == "__main__":
    main()
