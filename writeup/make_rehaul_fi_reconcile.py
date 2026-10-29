"""Attribution against the ablations, one row per predictor group: Table tab:rh_fi_reconcile (Section 6.3.5) and the
\\rhfrc... macros.

POST HOC.  Requested by the author on 28 October 2026 (17:50 PDT: "I need a better reconciliation of the attribution
with the ablations"); deliberation round 2026-10-29, adjudication C5 (T2): "The reconciliation table in 6.3.5, one row
per group on the D18 partition: attribution share (trees, all / statement / ordinary days; linear) next to the
add-alone and drop-one changes.  No compute."  Not pre-registered: RULE_shap.txt fixes the attribution measures and
results/ablations_2014start_w3000/RULE.txt the arms, neither this comparison.  The D18 partition
(writeup/INTEGRATION_PLAN_2026-10-26.md): base = HAR + the futures' realized moments (RM) + the FOMC calendar; the
other six groups are "beyond the base".

No model is fitted; every number is read or recomputed from these files:

  results/tree_shap_w3000_2014start_fomcfix_2026-10-26/   the FOMC-patched TreeSHAP study (RULE_pointer.txt ->
                                                          RULE_shap.txt) of the W = 3,000 LightGBM, XGBoost, forest
      m1_variables.csv      variable -> bucket (asserted = the ablation groups' series, below)
      m1_buckets.csv        M1: bucket share of the total mean |SHAP| over every explained bar
      m4_subsets.csv        M4 (c): the same on the FOMC statement days and on the other days
      m2_linear_shares.csv  M2's linear reference: share of sum_k |gamma_kj| (coefficient x window s.d., 126 origins)
      run_info.json         explained rows;  tables.txt  the statement-day count of M4 (c)
  results/ablations_2014start_w3000/scores_fomcfix/       experiments/ablate_score.py on the patched design
      metrics.csv, dm.csv, checks.json
  writeup/generated/rehaul_ablation_numbers.tex           read only: \\rhab<C>BaseShareAll, cross-checked

Group mapping (ablation group, experiments/ablate_arms.py GROUPS <- SHAP bucket, the study's shap_common.CATEGORIES;
both are src/data/loading.py SUBGROUPS, asserted series by series on m1_variables.csv):
    rm <- moments     liq <- liquidity    ew <- market_ew    vw <- market_vw
    sent <- sentiment iv <- implied_vol   vd <- vol_demand   fomc <- fomc
The SHAP backbone (har_ma, calendar) is the HAR benchmark set every ablation arm keeps (ablate_arms.HAR12).  The
table's shares are of the EXOGENOUS total (the eight groups, which sum to 100): comparable between the trees and the
linear models, whose HAR components carry a penalty factor (tab:rh_fip_shares note), and rank-equivalent to the shares
of the total.

Statement days: the attribution study's M4 (c) set, 'fomc release' > 0 in data/releases.parquet minus
loading.UNSCHEDULED_FOMC_DAYS (78 days); the ablation's EVENT days (results/channels_2026-10-26/RULE.txt section 0, 81)
add loading.FOMC_MISSING_STATEMENTS (three 2023-24 statements), which the attribution counts as other days.  The
linear attribution is pooled over origins: no statement-day split.  The ablation cells are every bar only.

Writes
  writeup/generated/rehaul_fi_reconcile_table.tex     tab:rh_fi_reconcile
  writeup/generated/rehaul_fi_reconcile_numbers.tex   \\rhfrc... macros (\\providecommand + \\renewcommand; letters only;
                                                      no other generated file uses the prefix):
    T in {Lgbm, Xgb, Rf}; L in {Ridge, Lasso, Enet}; C in L + T; G in {Rm, Liq, Ew, Vw, Sent, Iv, Vd, Fomc};
    S in {All, Stmt, Ord} (every explained bar / FOMC statement days / other days)
    D18 post hoc shares
      \\rhfrc<T>BeyondShare               six beyond-base groups, % of the total mean |SHAP| (every bar)
      \\rhfrcBeyondShareMin, ...Max       its range over the three trees
      \\rhfrc<T>BeyondShareExog<S>        six groups, % of the exogenous total (All: the C5 "of exogenous" figures)
      \\rhfrcBeyondShareExogMin, ...Max   range of <T>BeyondShareExogAll over the trees
      \\rhfrc<L>BeyondShareExog           the same for the linear models (|gamma| shares; pooled, every bar)
      \\rhfrcLinBeyondShareExogMin, ...Max
      \\rhfrc<C>BeyondGain                100 - \\rhab<C>BaseShareAll: % of the full set's QLIKE gain over its own HAR
                                          that the base does not reach (negative: the base beats the full set)
      \\rhfrcTreeBeyondGainMin, ...Max; \\rhfrcLinBeyondGainMin, ...Max
      \\rhfrcTreeHarCalShareMin, ...Max   HAR components + calendar, % of the trees' total (every bar)
    Shares by group
      \\rhfrc<T><G>Share<S>               the group's % of the exogenous total, tree T, span S
      \\rhfrc<L><G>ShareAll               the same, linear model L (pooled)
      \\rhfrcOrdMaxDiff                   largest |ordinary-day - every-bar| share over trees x groups (pp)
    Rank correlation (Spearman over the eight groups; the model's own every-bar exogenous share)
      \\rhfrc<C>RhoDrop                   with the drop-one QLIKE change (no Rf: the forest has one removal)
      \\rhfrc<C>RhoAdd                    with the add-alone QLIKE change
      \\rhfrcTreeRhoDropMin, ...Max       range of <T>RhoDrop over LightGBM and XGBoost
    Counts
      \\rhfrcNGroups (word), \\rhfrcNShapRows, \\rhfrcNStmtDaysShap, \\rhfrcNStmtRowsShap, \\rhfrcNAblBars

Run:  python writeup/make_rehaul_fi_reconcile.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

WRITEUP = Path(__file__).resolve().parent
ROOT = WRITEUP.parent
for _p in (str(ROOT), str(ROOT / "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import ablate_arms as AA  # noqa: E402  (group definitions only)

GEN = WRITEUP / "generated"
SHAP = ROOT / "results" / "tree_shap_w3000_2014start_fomcfix_2026-10-26"
ABL = ROOT / "results" / "ablations_2014start_w3000" / "scores_fomcfix"
ABL_NUM = GEN / "rehaul_ablation_numbers.tex"
OUT_TAB = GEN / "rehaul_fi_reconcile_table.tex"
OUT_NUM = GEN / "rehaul_fi_reconcile_numbers.tex"
HEAD = "% AUTO-GENERATED by writeup/make_rehaul_fi_reconcile.py -- do not edit; re-run the script."
POST_HOC = "post hoc, requested by the author on 28 October; not pre-registered"

# ablation group -> SHAP bucket (both = src/data/loading.py SUBGROUPS names; asserted in check_mapping)
BUCKET = dict(AA.GROUPS)
BASE = ("rm", "fomc")  # D18: with the HAR set, the base HAR + RM + FOMC
BEYOND = tuple(g for g in AA.GROUPS if g not in BASE)  # the six, in the ablation tables' row order
assert set(BEYOND) == set(AA.BEYOND_GROUPS), (BEYOND, AA.BEYOND_GROUPS)
COMBO = "add_rm_fomc"
G_TEX = {"rm": "Rm", "liq": "Liq", "ew": "Ew", "vw": "Vw", "sent": "Sent", "iv": "Iv", "vd": "Vd", "fomc": "Fomc"}
ROW_LABEL = {
    "rm": "Realized moments",
    "liq": "Liquidity",
    "ew": "EW market",
    "vw": "VW market",
    "sent": "Sentiment",
    "iv": "Implied volatility",
    "vd": "Volatility demand",
    "fomc": "FOMC",
}

TREES = ("lgbm", "xgb", "rf")  # ablation class names; EST: the SHAP study's estimator keys
EST = {"lgbm": "lgbm", "xgb": "xgb", "rf": "rf_skx"}
LINS = ("ridge", "lasso", "enet")
LIN_TAB = "ridge"  # the table's linear column: RULE_shap M2's registered linear reference
ABL_COLS = (LIN_TAB, *TREES)  # the ablation tables' order: linear first
CLS_TEX = {"ridge": "Ridge", "lasso": "Lasso", "enet": "Enet", "lgbm": "Lgbm", "xgb": "Xgb", "rf": "Rf"}
HEAD_NAME = {"ridge": "Ridge", "lgbm": "LGBM", "xgb": "XGB", "rf": "RF"}
SPANS = {"All": None, "Stmt": "FOMC day", "Ord": "other day"}  # macro tag -> m4_subsets subset (None: m1_buckets)
WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}
# adjudication C5, refutation 2: the figures this generator reproduces (printed as a check, not asserted)
C5_CHECK = {
    "BeyondShare": {"lgbm": 22.8, "xgb": 21.3, "rf": 20.0},
    "BeyondShareExog": {"lgbm": 48.0, "xgb": 45.2, "rf": 47.4},
    "BeyondGain": {"lgbm": 10.0, "xgb": 11.0, "rf": 4.0},
}


class Change(NamedTuple):
    pct: float  # 100 (L_arm / L_ref - 1)
    dm: float  # Diebold-Mariano, arm vs ref (same sign)
    p: float
    pre: bool  # tested against the pre-patch full set


# ============================================================================ formatting
def fmt(x: float, dp: int) -> str:
    v = round(float(x), dp)
    if v == 0:
        v = 0.0
    return f"{'$-$' if v < 0 else ''}{abs(v):.{dp}f}"


def num(n: int) -> str:
    return f"{int(n):,}".replace(",", "{,}")


def tt(path: str) -> str:
    """A file path in typewriter type, breakable after '/' and '_'."""
    return r"\texttt{" + path.replace("_", r"\_\allowbreak{}").replace("/", r"/\allowbreak{}") + "}"


def cell(x: Change | None) -> str:
    """% change, * at DM p < 0.05, dagger on the pre-patch reference; '--' when the arm was not run."""
    if x is None:
        return "--"
    marks = ("*" if x.p < 0.05 else "") + (r"\dagger" if x.pre else "")
    return fmt(x.pct, 1) + (f"$^{{{marks}}}$" if marks else "")


class Macros:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def put(self, name: str, val: str, comment: str = "") -> None:
        assert re.fullmatch(r"[A-Za-z]+", name), name
        assert name not in {n for n, _v, _c in self.rows}, name
        self.rows.append((name, val, comment))

    def text(self) -> str:
        lines = [
            HEAD,
            "% \\rhfrc... macros of Table tab:rh_fi_reconcile (" + POST_HOC + ").",
            f"% Attribution: {SHAP.relative_to(ROOT).as_posix()} (FOMC-patched TreeSHAP; linear: M2's |gamma| shares).",
            f"% Ablations: {ABL.relative_to(ROOT).as_posix()} (FOMC-patched design).  Shares in %; changes in % of QLIKE.",
        ]
        for n, v, c in self.rows:
            lines.append(
                f"\\providecommand{{\\rhfrc{n}}}{{}}\\renewcommand{{\\rhfrc{n}}}{{{v}}}" + (f"  % {c}" if c else "")
            )
        return "\n".join(lines) + "\n"


# ============================================================================ attribution
def check_mapping(v: pd.DataFrame) -> None:
    """Each ablation group's series (ablate_arms.group_series = loading.SUBGROUPS) are exactly the SHAP bucket's
    variables, for every estimator; the rest of the SHAP variables are the backbone har_ma + calendar."""
    gs = AA.group_series()
    for e in v["est"].unique():
        b = v[v["est"] == e].set_index("variable")["bucket"]
        for g, bucket in BUCKET.items():
            assert sorted(b[b == bucket].index) == sorted(gs[g]), (e, g)
        assert sorted(b[~b.isin(list(BUCKET.values()))].index) == ["calendar", "har_ma"], e


def tree_shares(v: pd.DataFrame, bk: pd.DataFrame, m4: pd.DataFrame) -> dict[tuple[str, str], pd.Series]:
    """(tree, span tag) -> % of the exogenous total by ablation group."""
    out: dict[tuple[str, str], pd.Series] = {}
    for t in TREES:
        e = EST[t]
        for s, sub in SPANS.items():
            if sub is None:
                raw = bk[bk["est"] == e].set_index("bucket")["share"]
            else:
                r = m4[(m4["est"] == e) & (m4["kind"] == "fomc") & (m4["subset"] == sub)]
                assert len(r) == 1, (e, sub)
                raw = pd.Series({b: float(r[f"bucket_{b}"].iloc[0]) for b in BUCKET.values()})
            x = pd.Series({g: float(raw[BUCKET[g]]) for g in AA.GROUPS})
            out[t, s] = 100.0 * x / x.sum()
        # every bar: the bucket sums equal the variables' own exogenous shares (m1_variables share_exog)
        ve = v[(v["est"] == e) & v["bucket"].isin(list(BUCKET.values()))].groupby("bucket")["share_exog"].sum()
        chk = np.array([100.0 * float(ve[BUCKET[g]]) for g in AA.GROUPS])
        assert np.allclose(chk, out[t, "All"].to_numpy(), atol=1e-9), t
    return out


def lin_shares(lin: pd.DataFrame, v: pd.DataFrame) -> dict[str, pd.Series]:
    """linear model -> % of the exogenous |gamma| total by ablation group (pooled over the origins)."""
    bucket_of = v[v["est"] == "lgbm"].set_index("variable")["bucket"]
    out = {}
    for m in LINS:
        d = lin[lin["linear"] == m].copy()
        d["bucket"] = d["variable"].map(bucket_of)
        d = d[d["bucket"].isin(list(BUCKET.values()))]
        s = d.groupby("bucket")["share_exog"].sum()
        out[m] = pd.Series({g: 100.0 * float(s[BUCKET[g]]) for g in AA.GROUPS})
        assert abs(out[m].sum() - 100.0) < 1e-9, m
    return out


def stmt_count(m4: pd.DataFrame) -> tuple[int, int]:
    """(statement days, their rows) of the attribution study's M4 (c), as its tables.txt prints them."""
    txt = (SHAP / "tables.txt").read_text(encoding="utf-8")
    m = re.search(r"FOMC statement days among forecast rows: (\d+) days, (\d+) rows", txt)
    assert m, "tables.txt: statement-day line"
    days, rows = int(m.group(1)), int(m.group(2))
    r = m4[(m4["kind"] == "fomc") & (m4["subset"] == "FOMC day")]["rows"].unique()
    assert list(r) == [rows], (r, rows)
    return days, rows


# ============================================================================ ablations
class Ablation:
    """Every-bar QLIKE changes of experiments/ablate_score.py's patched panel."""

    def __init__(self) -> None:
        M = pd.read_csv(ABL / "metrics.csv")
        D = pd.read_csv(ABL / "dm.csv")
        self.checks = json.loads((ABL / "checks.json").read_text())
        assert self.checks["pass"], self.checks["fail"]
        self.q = M[M["span"] == "all"].set_index(["cls", "arm"])["qlike"]
        D = D[(D["loss"] == "qlike") & (D["span"] == "all")]
        self.dq = D.set_index(["cls", "arm", "vs_cls", "vs_arm"])
        lags = sorted(D["hac_lag"].unique())
        assert len(lags) == 1, lags
        self.hac = int(lags[0])
        # the removals tested against the pre-patch full set (addendum J: only the forest's realized moments remain)
        pre = D[D["vs_arm"] == "full_pre"]
        self.pre_ref = set(zip(pre["cls"], pre["arm"]))

    def change(self, c: str, arm: str, ref: str) -> Change | None:
        """arm against ref (the stored test is arm vs ref); None when the arm was not run."""
        k = (c, arm, c, ref)
        if (c, arm) not in self.q.index or (c, ref) not in self.q.index or k not in self.dq.index:
            return None
        pct = 100.0 * (float(self.q[c, arm]) / float(self.q[c, ref]) - 1.0)
        return Change(pct, float(self.dq.loc[k, "dm"]), float(self.dq.loc[k, "p"]), ref == "full_pre")

    def add(self, c: str, g: str) -> Change | None:
        return self.change(c, f"add_{g}", "har")

    def drop(self, c: str, g: str) -> Change | None:
        a = f"lo_{g}"
        return self.change(c, a, "full_pre" if (c, a) in self.pre_ref else "full")

    def beyond_gain(self, c: str) -> float:
        """100 - BaseShare: % of the full set's gain over its own HAR beyond HAR + RM + FOMC."""
        lr, lb, lf = (float(self.q[c, a]) for a in ("har", COMBO, "full"))
        return 100.0 - 100.0 * (lr - lb) / (lr - lf)


def abl_base_share() -> dict[str, int]:
    """\\rhab<C>BaseShareAll as printed by writeup/make_rehaul_ablations.py (read only)."""
    txt = ABL_NUM.read_text(encoding="utf-8")
    out = {}
    for c in (*LINS, *TREES):
        m = re.search(r"\\newcommand\{\\rhab" + CLS_TEX[c] + r"BaseShareAll\}\{([^}]*)\}", txt)
        assert m, c
        out[c] = int(m.group(1).replace("$-$", "-"))
    return out


# ============================================================================ main
def main() -> int:
    v = pd.read_csv(SHAP / "m1_variables.csv")
    bk = pd.read_csv(SHAP / "m1_buckets.csv")
    m4 = pd.read_csv(SHAP / "m4_subsets.csv")
    lin = pd.read_csv(SHAP / "m2_linear_shares.csv")
    info = json.loads((SHAP / "run_info.json").read_text())
    check_mapping(v)
    ts = tree_shares(v, bk, m4)
    ls = lin_shares(lin, v)
    n_stmt_days, n_stmt_rows = stmt_count(m4)
    A = Ablation()
    # the ablation's EVENT days = the attribution's statement days + loading.FOMC_MISSING_STATEMENTS (2023-24)
    from src.data import loading as LD

    n_missing = len(LD.FOMC_MISSING_STATEMENTS)
    assert all(s[:4] in ("2023", "2024") for s in LD.FOMC_MISSING_STATEMENTS), LD.FOMC_MISSING_STATEMENTS
    assert n_stmt_days + n_missing == A.checks["day_types"]["event_days"], (n_stmt_days, n_missing)

    # ------------------------------------------------------------------ D18 shares and the C5 check
    def bucket_sum(t: str, buckets: list[str]) -> float:
        return 100.0 * float(bk[(bk["est"] == EST[t]) & bk["bucket"].isin(buckets)]["share"].sum())

    beyond_tot = {t: bucket_sum(t, [BUCKET[g] for g in BEYOND]) for t in TREES}
    harcal = {t: bucket_sum(t, ["har_ma", "calendar"]) for t in TREES}
    beyond_ex = {(t, s): float(ts[t, s][list(BEYOND)].sum()) for t in TREES for s in SPANS}
    lin_beyond = {m: float(ls[m][list(BEYOND)].sum()) for m in LINS}
    gain = {c: A.beyond_gain(c) for c in (*LINS, *TREES)}
    base_tab = abl_base_share()
    for c, x in gain.items():
        assert round(100.0 - x) == base_tab[c], (c, 100.0 - x, base_tab[c])
    found = {
        "BeyondShare": {t: round(beyond_tot[t], 1) for t in TREES},
        "BeyondShareExog": {t: round(beyond_ex[t, "All"], 1) for t in TREES},
        "BeyondGain": {t: float(round(gain[t])) for t in TREES},
    }
    print("C5 check (adjudication 2026-10-29, refutation 2):")
    for k, want in C5_CHECK.items():
        print(f"  {k:16s} found {found[k]}  quoted {want}  {'REPRODUCED' if found[k] == want else 'DIFFERS'}")

    Mc = Macros()
    for t in TREES:
        Mc.put(f"{CLS_TEX[t]}BeyondShare", fmt(beyond_tot[t], 1), f"{t}: six beyond-base groups, % of total |SHAP|")
    Mc.put("BeyondShareMin", fmt(min(beyond_tot.values()), 1), "lowest over the trees")
    Mc.put("BeyondShareMax", fmt(max(beyond_tot.values()), 1), "highest over the trees")
    for t in TREES:
        for s in SPANS:
            Mc.put(f"{CLS_TEX[t]}BeyondShareExog{s}", fmt(beyond_ex[t, s], 1), f"{t}: six groups, % of exogenous, {s}")
    ex_all = [beyond_ex[t, "All"] for t in TREES]
    Mc.put("BeyondShareExogMin", fmt(min(ex_all), 1), "lowest over the trees, every bar")
    Mc.put("BeyondShareExogMax", fmt(max(ex_all), 1), "highest over the trees, every bar")
    for m in LINS:
        Mc.put(f"{CLS_TEX[m]}BeyondShareExog", fmt(lin_beyond[m], 1), f"{m}: six groups, % of exogenous |gamma|")
    Mc.put("LinBeyondShareExogMin", fmt(min(lin_beyond.values()), 1), "lowest over ridge, lasso, enet")
    Mc.put("LinBeyondShareExogMax", fmt(max(lin_beyond.values()), 1), "highest over ridge, lasso, enet")
    for c in (*LINS, *TREES):
        Mc.put(f"{CLS_TEX[c]}BeyondGain", fmt(gain[c], 0), f"100 - rhab{CLS_TEX[c]}BaseShareAll (%)")
    for nm, cs in (("Tree", TREES), ("Lin", LINS)):
        Mc.put(f"{nm}BeyondGainMin", fmt(min(gain[c] for c in cs), 0), f"lowest over {', '.join(cs)}")
        Mc.put(f"{nm}BeyondGainMax", fmt(max(gain[c] for c in cs), 0), f"highest over {', '.join(cs)}")
    Mc.put("TreeHarCalShareMin", fmt(min(harcal.values()), 1), "HAR + calendar, % of the trees' total, lowest")
    Mc.put("TreeHarCalShareMax", fmt(max(harcal.values()), 1), "HAR + calendar, % of the trees' total, highest")

    # ------------------------------------------------------------------ shares by group
    for t in TREES:
        for g in AA.GROUPS:
            for s in SPANS:
                Mc.put(f"{CLS_TEX[t]}{G_TEX[g]}Share{s}", fmt(ts[t, s][g], 1), f"{t} {g} % of exogenous, {s}")
    for m in LINS:
        for g in AA.GROUPS:
            Mc.put(f"{CLS_TEX[m]}{G_TEX[g]}ShareAll", fmt(ls[m][g], 1), f"{m} {g} % of exogenous |gamma|")
    diffs = {(t, g): abs(float(ts[t, "Ord"][g] - ts[t, "All"][g])) for t in TREES for g in AA.GROUPS}
    worst = max(diffs, key=lambda k: diffs[k])
    Mc.put("OrdMaxDiff", fmt(diffs[worst], 1), f"largest |ordinary - every bar| share, pp ({worst[0]}, {worst[1]})")

    # ------------------------------------------------------------------ rank correlations
    rho: dict[tuple[str, str], float | None] = {}
    for c in (*LINS, *TREES):
        share = ls[c] if c in LINS else ts[c, "All"]
        for kind in ("Add", "Drop"):
            ch = {g: (A.add if kind == "Add" else A.drop)(c, g) for g in AA.GROUPS}
            vals = [x.pct for x in ch.values() if x is not None]
            if len(vals) < len(AA.GROUPS):
                rho[c, kind] = None
                print(f"  Spearman {c} {kind}: {len(vals)} of {len(AA.GROUPS)} groups run -- not computed")
                continue
            # Spearman = Pearson on the (average-tie) ranks; no scipy (the base env's build is broken)
            y = pd.Series(vals, index=list(AA.GROUPS))
            r = float(share.rank().corr(y.rank()))
            rho[c, kind] = r
            Mc.put(f"{CLS_TEX[c]}Rho{kind}", fmt(r, 2), f"Spearman, {c} every-bar share vs {kind.lower()} change")
    td = [r for r in (rho["lgbm", "Drop"], rho["xgb", "Drop"]) if r is not None]
    assert len(td) == 2
    Mc.put("TreeRhoDropMin", fmt(min(td), 2), "lowest of lgbm, xgb")
    Mc.put("TreeRhoDropMax", fmt(max(td), 2), "highest of lgbm, xgb")

    # ------------------------------------------------------------------ counts
    Mc.put("NGroups", WORDS[len(AA.GROUPS)], "ablation groups = table rows")
    Mc.put("NShapRows", num(info["forecast_rows"]), "explained bars of the attribution study")
    Mc.put("NStmtDaysShap", str(n_stmt_days), "FOMC statement days of the attribution study (M4 c)")
    Mc.put("NStmtRowsShap", num(n_stmt_rows), "their bars")
    Mc.put("NAblBars", num(A.checks["n_bars_all"]), "bars scored by the ablations (every bar)")
    GEN.mkdir(parents=True, exist_ok=True)
    OUT_NUM.write_text(Mc.text(), encoding="utf-8")
    print(f"wrote {OUT_NUM.relative_to(ROOT).as_posix()} ({len(Mc.rows)} macros)")

    # ------------------------------------------------------------------ table
    n = 1 + 4 + 3 + 4 + 4

    def head(txt: str) -> str:
        return rf"\multicolumn{{{n}}}{{@{{}}l}}{{\textit{{{txt}}}}} \\"

    def row(g: str) -> str:
        cells = [fmt(ls[LIN_TAB][g], 1)] + [fmt(ts[t, "All"][g], 1) for t in TREES]
        cells += [fmt(ts[t, "Stmt"][g], 1) for t in TREES]
        cells += [cell(A.add(c, g)) for c in ABL_COLS] + [cell(A.drop(c, g)) for c in ABL_COLS]
        return f"{ROW_LABEL[g]} & " + " & ".join(cells) + r" \\"

    out = [
        HEAD,
        "% tab:rh_fi_reconcile (" + POST_HOC + "; adjudication 2026-10-29 C5).",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{1.5pt}",
        r"\begin{threeparttable}",
        r"\caption{Attribution and ablation by predictor group}",
        r"\label{tab:rh_fi_reconcile}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l" + "r" * (n - 1) + "@{}}",
        r"\toprule",
        r" & \multicolumn{7}{c}{Attribution share (\%)} & \multicolumn{4}{c}{Added to HAR (\%)}"
        r" & \multicolumn{4}{c}{Dropped from full set (\%)} \\",
        r"\cmidrule(lr){2-8}\cmidrule(lr){9-12}\cmidrule(l){13-16}",
        r" & \multicolumn{4}{c}{Every bar} & \multicolumn{3}{c}{Statement days}"
        r" & \multicolumn{4}{c}{\textit{$-$: adds accuracy}} & \multicolumn{4}{c}{\textit{$+$: adds accuracy}} \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-8}",
        " & "
        + " & ".join(
            [HEAD_NAME[c] for c in ABL_COLS] + [HEAD_NAME[t] for t in TREES] + [HEAD_NAME[c] for c in ABL_COLS] * 2
        )
        + r" \\",
        r"\midrule",
        head("In the base HAR + RM + FOMC"),
        *(row(g) for g in BASE),
        head("Beyond the base"),
        *(row(g) for g in BEYOND),
        r"\addlinespace[2pt]",
    ]
    # the six together: shares summed; dropped = the base HAR + RM + FOMC against the full set
    six = [fmt(lin_beyond[LIN_TAB], 1)] + [fmt(beyond_ex[t, "All"], 1) for t in TREES]
    six += [fmt(beyond_ex[t, "Stmt"], 1) for t in TREES] + [""] * 4
    six += [cell(A.change(c, COMBO, "full")) for c in ABL_COLS]
    out.append("The six together & " + " & ".join(six) + r" \\")

    def rho_cell(c: str, k: str) -> str:
        r = rho[c, k]
        return "--" if r is None else fmt(r, 2)

    rr = [""] * 7 + [rho_cell(c, k) for k in ("Add", "Drop") for c in ABL_COLS]
    out.append(r"Spearman $\rho$ & " + " & ".join(rr) + r" \\")
    sign = (
        r"\item \textit{Sign:} ablation cells are $100\,(L_{\text{row}}/L_{\text{ref}} - 1)$, the percentage change in"
        r" mean QLIKE against the reference, as in Table~\ref{tab:rh_ablation}.  Added to HAR (reference: the class's"
        r" own HAR benchmark fit): negative, the loss falls when the group is added, i.e.\ the group adds accuracy."
        r"  Dropped from the full set (reference: the full set): positive, the loss rises when the group is dropped,"
        r" i.e.\ the group adds accuracy.  $^{*}$ Diebold--Mariano test rejects equal accuracy at 5\% (two-sided;"
        rf" daily average QLIKE, Newey--West lag {A.hac})."
    )
    ng = WORDS[len(AA.GROUPS)]
    notes = (
        r"\item \textit{Notes:} Post hoc, requested by the author on 28 October; not pre-registered.  LGBM, XGB, RF:"
        r" LightGBM, XGBoost, random forest.  Base: the HAR benchmark set plus the futures' realized moments (RM) and"
        r" the FOMC calendar.  Attribution: mean $|\mathrm{SHAP}|$ share of the group's series in the tree's exogenous"
        rf" total (the {ng} groups, which sum to 100; the HAR components and calendar dummies, {fmt(min(harcal.values()), 0)}--"
        rf"{fmt(max(harcal.values()), 0)}\% of the trees' total, are left out), over all"
        rf" {num(info['forecast_rows'])} explained bars or the {num(n_stmt_rows)} bars of the {n_stmt_days} scheduled"
        rf" FOMC statement days with a release stamp (the {WORDS.get(n_missing, str(n_missing))} 2023--24 statements"
        r" without one count as ordinary days); the ordinary-day shares differ from the every-bar shares by at most"
        rf" {fmt(diffs[worst], 1)} points.  Ridge: share of the absolute coefficients on the window-standardized"
        r" predictors, summed over the origins (Table~\ref{tab:rh_fip_shares}), a different measure.  Ablations:"
        rf" FOMC-patched design, the {num(A.checks['n_bars_all'])} scored bars of Table~\ref{{tab:rh_oos}}, every arm at"
        r" the full set's frozen hyperparameters; \dag{} on the pre-patch design against the pre-patch full set; the"
        r" forest's other removals were not run.  The six together: shares summed; dropped, HAR + RM + FOMC against"
        rf" the full set.  Spearman $\rho$: rank correlation over the {ng} groups of the model's every-bar share with"
        r" its change (larger shares going with larger gains: negative under Added, positive under Dropped).  Sources: "
        + tt(SHAP.relative_to(ROOT).as_posix())
        + r" (m1\_buckets, m4\_subsets, m2\_linear\_shares) and "
        + tt(ABL.relative_to(ROOT).as_posix())
        + " (metrics, dm)."
    )
    out += [
        r"\bottomrule",
        r"\end{tabular*}",
        r"\begin{tablenotes}[flushleft]",
        r"\footnotesize",
        sign,
        notes,
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
    ]
    OUT_TAB.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {OUT_TAB.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
