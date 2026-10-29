"""The two channels of the pooled gain: FOMC statement days and ordinary days (results/channels_2026-10-26/RULE.txt).

RULE sections 0-3 (section 4, the factor-augmented HAR, belongs to another run and is not read here).  Every number is
computed here from stored forecasts; no model is refitted.

    python writeup/make_rehaul_channels.py          (repository root, the 285J environment)

DESIGN OF RECORD  Table 1: 13 regular-hours bars pooled, W = 3,000 days, FOMC-patched design (D13), 2014-01-02 ..
         2024-04-30; scored set = Table 1's 33,527 bars on 2,579 days (every session with all 13 bars).  plain =
         pred_adj^2 x B; QLIKE = r - ln r - 1, r = RV / plain, RV the bar's realized variance (data/core_stats.parquet
         sumret2 = the forecasts' rv_actual, checked); squared error alongside.  DM: src/evaluation/diebold_mariano.
         dm_test (HLN) on the daily mean of the day's bar losses, HAC lag 8, two-sided, 5 %.
DAY TYPES EVENT = the scheduled FOMC statement days of the span (data/releases.parquet loading.FOMC_RAW_COL, plus
         loading.FOMC_MISSING_STATEMENTS, minus loading.UNSCHEDULED_FOMC_DAYS) = 81 (asserted); ORDINARY = the other
         2,498 scored days.  "Bars 14:30-16:00" = the bars with END stamps 14:30, 15:00, 15:30, 16:00 (the statement's
         14:00 release falls in the bar ending 14:30, the stamp of every statement in the span, asserted).
BENCHMARKS the OLS on the HAR set (D2) for ridge, lasso, elastic net; each tree's own HAR-set fit.
INPUTS   (each found in this worktree first, else as a COMMITTED file of its run's branch -- read with git cat-file,
         never from another worktree's uncommitted files; REHAUL_CHANNELS_NO_BRANCH=1 turns the branch read off)
  full / HAR, patched   results/window_sweep_2014start_fomcfix/w3000, results/pooled_trees_final_2014start_fomcfix_w3000,
                        results/pooled_rf_skx_2014start_fomcfix_w3000 (Table 1)
  full, pre-patch       results/window_sweep_2014start_2026-10-22/w3000, results/pooled_trees_final_2014start_w3000,
                        results/pooled_rf_skx_2014start_w3000 (the ablation templates; reference of pre-patch arms)
  ablation arms         results/ablations_2014start_w3000/forecasts/: linear_wide_fix.parquet (every linear arm on the
                        patched design, addendum D), else linear_wide.parquet (pre-patch); trees_wide.parquet (pre-patch;
                        design-invariant where the arm holds no FOMC column); trees_wide_fix.parquet (the trees' patched
                        add_fomc, addendum D (d)); trees_wide_rflofomc.parquet (the forest's lo_fomc, addendum E;
                        design-invariant).  A missing file prints \\rhPending{} in its cells and macros.
  ladder                results/ladder_2026-10-26/fomcfix/ (patched, primary ranking), else results/ladder_2026-10-26/
                        (pre-patch, labelled)
  An arm holding a FOMC column is used only on the patched design or labelled pre-patch: the trees' lo_<g>, g != fomc,
  are pre-patch and compared with the pre-patch full set (dagger in the tables).
GATES    (asserted, before any day split; written to results/channels_2026-10-26/gates.json first)
  scored set 33,527 bars = 2,579 days x 13; EVENT days 81; Table 1's every-bar QLIKE and MSE of its 13 rows (1e-12
  relative) and its full-vs-HAR and vs-OLS DM of all six classes (1e-9); RV = rv_actual; B equal across inputs; every
  arm finite and positive on the scored set; each arm's every-bar QLIKE = the ablation scorer's (scores/ or
  scores_fomcfix/ metrics.csv, 1e-12); the linear FOMC-free arms of linear_wide_fix = their pre-patch walks bit for bit;
  the trees' HAR fits equal across the patched and pre-patch runs; the ladder's every-bar QLIKE = its scores.csv.
WRITES   results/channels_2026-10-26/*.csv, gates.json, provenance.csv, SUMMARY.md
         writeup/generated/rehaul_channels_numbers.tex           \\rhch... macros
         writeup/generated/rehaul_channels_corrections_table.tex  tab:rh_channels_corr   (item 1)
         writeup/generated/rehaul_channels_ablation_table.tex     tab:rh_channels_abl    (item 3 ii)
         writeup/generated/rehaul_channels_ablation_full_table.tex tab:rh_channels_abl_full (3 ii, every row)
         writeup/generated/rehaul_channels_sparsity_table.tex     tab:rh_channels_sparse (items 3 i, 3 iii)
         writeup/figures/rehaul_channels_cum.pdf, writeup/generated/rehaul_channels_fig.tex   fig:rh_channels (item 2)
ADDITIONAL, NOT PRE-REGISTERED (coordinator, 2026-10-26, after the first scoring; labelled so in the CSV, SUMMARY.md
         and the macros \\rhchX...): the full set against HAR + realized moments (add_rm) and HAR + realized moments
         against HAR, every class, every bar / EVENT / ORDINARY days (itemX_additional_rm_not_preregistered.csv); and
         the readings with REDUNDANT tested before CONCENTRATED (...ReadingRedFirst; the RULE's two definitions overlap).
NAMES    \\rhch<Class><...>: Class Ridge Lasso Enet Lgbm Xgb Rf; day type Bar (every bar) / Event / Ord; group Rm Liq Ew Vw
         Sent Iv Vd Fomc; Corr{A,B,C} the three corrections of item 1; Pct = % QLIKE change, Dm, P; Lin / Tree the model
         types.  Choices where the RULE is silent: SUMMARY.md, section "Choices".
PROSE    (2026-10-28, draft v2 of Sections 6-7, MAP_06_07.md "Numbers still needed"; prose_ranges) ranges and extremes
         from the item 1 / 3 (ii) results: \\rhchCorr{Lin,Tree}Rng, \\rhchEvFomcShare{Lin,Tree}Rng, \\rhchEvLoFomcRng,
         \\rhchEvOtherMaxShare, \\rhchOrdAddFomc{Lin,Tree}Rng, \\rhchOrdAddRmTreeRng, \\rhchOrdRmShareTreeRng,
         \\rhchOrdAddOtherTreeRng, \\rhchOrdSumRatio{,Lin,Tree}Rng, \\rhchOrdLoOtherTreeMaxAbs, \\rhchAltLoMaxAbs;
         magnitudes "a--b" where the sentence gives the direction (\\rhorVsOlsRedTreeRng), signed where it may straddle 0.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(ROOT), str(ROOT / "notebooks"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import final_run as fr  # noqa: E402  (PENDING_TEX / PENDING_DEF only)
from src.data import loading  # noqa: E402
from src.evaluation.diebold_mariano import dm_test  # noqa: E402

RES = ROOT / "results" / "channels_2026-10-26"
GEN = HERE / "generated"
FIG = HERE / "figures" / "rehaul_channels_cum.pdf"
HEAD = "% AUTO-GENERATED by writeup/make_rehaul_channels.py -- do not edit; re-run the script."
PEND = fr.PENDING_TEX
FIRST, LAST = pd.Timestamp("2014-01-02"), pd.Timestamp("2024-04-30")
N_BARS, N_DAYS, N_EVENT, BARS_A_DAY, HAC_LAG, ALPHA, AGREE_PP = 33527, 2579, 81, 13, 8, 0.05, 2.0
PM_FROM = (14, 30)  # bar END stamps 14:30 .. 16:00
BRANCH_ABL = "insurance-selling-2026-09-19-e34-ablations"
BRANCH_LAD = "insurance-selling-2026-09-19-e34-ladder"
NO_BRANCH = os.environ.get("REHAUL_CHANNELS_NO_BRANCH", "") == "1"
# the forest's lo_fomc (ablation addendum E) was STOPPED by the author (2026-10-26 19:06, addendum E (i)): its cells read
# "not run" (--) and the forest leaves the trees' vote where it decides.  Default since 2026-10-28 (the outputs of
# a3779cf were made with the variable set to 1); REHAUL_CHANNELS_RF_LOFOMC_DROPPED=0 restores the pending cells.
RF_LOFOMC_DROPPED = os.environ.get("REHAUL_CHANNELS_RF_LOFOMC_DROPPED", "1") == "1"

LIN = ("ridge", "lasso", "enet")
TREES = ("lgbm", "xgb", "rf")
CLASSES = LIN + TREES
CTAG = {"ridge": "Ridge", "lasso": "Lasso", "enet": "Enet", "lgbm": "Lgbm", "xgb": "Xgb", "rf": "Rf"}
CNAME = {
    "ridge": "Ridge",
    "lasso": "Lasso",
    "enet": "Elastic net",
    "lgbm": "LightGBM",
    "xgb": "XGBoost",
    "rf": "Random forest",
}
CSHORT = {"ridge": "Ridge", "lasso": "Lasso", "enet": "E.\\ net", "lgbm": "LGBM", "xgb": "XGB", "rf": "Forest"}
GROUPS = ("rm", "liq", "ew", "vw", "sent", "iv", "vd", "fomc")
GTAG = {"rm": "Rm", "liq": "Liq", "ew": "Ew", "vw": "Vw", "sent": "Sent", "iv": "Iv", "vd": "Vd", "fomc": "Fomc"}
GNAME = {
    "rm": "Realized moments",
    "liq": "Liquidity",
    "ew": "Equal-weighted market",
    "vw": "Value-weighted market",
    "sent": "Sentiment",
    "iv": "Implied volatility",
    "vd": "Volatility demand",
    "fomc": "FOMC",
}
SPANS = ("all", "event", "ord")
STAG = {"all": "Bar", "event": "Event", "ord": "Ord"}
SNAME = {"all": "every bar", "event": "FOMC statement days", "ord": "ordinary days"}
KS = (1, 3, 5, 10, 20)
KTAG = {1: "One", 3: "Three", 5: "Five", 10: "Ten", 20: "Twenty"}
LAD_KEY = {"rf": "rf_skx"}
T1_ROWS = [("ols", "baseline")] + [(c, s) for c in CLASSES for s in ("all_features", "baseline")]
SET_ARM = {"all_features": "full", "baseline": "har"}
OKABE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]  # make_rehaul_fi.OKABE
COLOR = {"lgbm": OKABE[0], "xgb": OKABE[1], "rf": OKABE[2], "ridge": OKABE[6], "lasso": OKABE[3], "enet": OKABE[4]}
MARK = {"lgbm": "o", "xgb": "s", "rf": "^", "ridge": "D", "lasso": "v", "enet": "P"}
READ_WORD = {
    "CONCENTRATED": "concentrated",
    "REDUNDANT": "redundant",
    "MIXED": "mixed",
    "SPLIT": "split",
    "NO JOINT GAIN": "no joint gain",
    "INCOMPLETE": "n/a",
}
READ_SHORT = {
    "CONCENTRATED": "conc.",
    "REDUNDANT": "redund.",
    "MIXED": "mixed",
    "SPLIT": "split",
    "NO JOINT GAIN": "no gain",
    "INCOMPLETE": "n/a",
}


def say(msg: str) -> None:
    print(msg, flush=True)


# ------------------------------------------------------------------------------------------------ input resolution
@dataclass
class Src:
    rel: str
    where: str  # "worktree" or "branch <name> @ <sha>"
    data: bytes | None = None  # a branch blob

    def read_parquet(self, columns: list[str] | None = None) -> pd.DataFrame:
        if self.data is None:
            return pd.read_parquet(ROOT / self.rel, columns=columns)
        return pd.read_parquet(io.BytesIO(self.data), columns=columns)

    def read_csv(self) -> pd.DataFrame:
        return pd.read_csv(ROOT / self.rel if self.data is None else io.BytesIO(self.data))

    def read_text(self) -> str:
        return (ROOT / self.rel).read_text(encoding="utf-8") if self.data is None else self.data.decode("utf-8")


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True)


def find(rel: str, branch: str | None = None) -> Src | None:
    """The input at `rel`: this worktree's file, else (unless NO_BRANCH) the committed blob on `branch`, else None."""
    if (ROOT / rel).is_file():
        return Src(rel, "worktree")
    if branch is None or NO_BRANCH:
        return None
    if git("rev-parse", "--verify", "--quiet", f"{branch}:{rel}").returncode != 0:
        return None
    sha = git("rev-parse", "--short", branch).stdout.decode().strip()
    out = git("cat-file", "blob", f"{branch}:{rel}")
    assert out.returncode == 0, out.stderr
    return Src(rel, f"branch {branch} @ {sha}", out.stdout)


# ------------------------------------------------------------------------------------------------ data
def core_rv() -> pd.Series:
    c = pd.read_parquet(ROOT / "data" / "core_stats.parquet", columns=["endbartime", "sumret2"])
    s = pd.Series(c["sumret2"].to_numpy(float), index=pd.DatetimeIndex(pd.to_datetime(c["endbartime"])))
    return s[~s.index.duplicated()].sort_index()


def evaluation_days(rv: pd.Series) -> pd.DatetimeIndex:
    """Table 1's 2,579 days (experiments/ablate_score.py): a positive 16:00 print, early closes dropped."""
    import atm_straddle_lib as asl

    early = pd.DatetimeIndex(pd.to_datetime(list(asl.EARLY_CLOSE_DATES)))
    s = rv[(rv.index.hour == 16) & (rv.index.minute == 0)]
    d = pd.DatetimeIndex(s.index.normalize())
    keep = (s.to_numpy() > 0) & np.isfinite(s.to_numpy()) & ~d.isin(early)
    days = d[keep]
    days = days[(days >= FIRST) & (days <= LAST)]
    assert len(days) == N_DAYS and days[0] == FIRST and days[-1] == LAST, (len(days), days[0], days[-1])
    return days


def statement_stamps() -> pd.DatetimeIndex:
    """The scheduled FOMC statements (bar end stamps), loading.add_fomc_features' rule."""
    rel = pd.read_parquet(ROOT / "data" / "releases.parquet", columns=["endbartime", loading.FOMC_RAW_COL])
    t = pd.DatetimeIndex(pd.to_datetime(rel["endbartime"]))
    raw = np.nan_to_num(pd.to_numeric(rel[loading.FOMC_RAW_COL], errors="coerce").to_numpy(float), nan=0.0)
    st = t[raw > 0].append(pd.DatetimeIndex(pd.to_datetime(list(loading.FOMC_MISSING_STATEMENTS))))
    st = st[~pd.DatetimeIndex(st.normalize()).isin(list(loading.UNSCHEDULED_FOMC_DAYS))]
    return st.unique().sort_values()


def long_panel(src: Src, models: dict[str, str], pre: bool, first: pd.Timestamp = FIRST) -> dict:
    """(class, arm) -> (pred_adj Series, B Series, rv_actual Series) from a long forecasts file; models maps the file's
    model name to the class key; arm full / har (pre: full_pre / har_pre)."""
    cols = ["model", "predictor_set", "date", "pred_adj", "B", "rv_actual"]
    d = src.read_parquet(cols)
    d = d[d["model"].isin(list(models)) & (pd.to_datetime(d["date"]) >= first)]
    out = {}
    for (m, s), f in d.groupby(["model", "predictor_set"]):
        ix = pd.DatetimeIndex(pd.to_datetime(f["date"]))
        assert ix.is_unique, (src.rel, m, s)
        arm = SET_ARM[s] + ("_pre" if pre else "")
        out[(models[m], arm)] = tuple(pd.Series(f[c].to_numpy(float), index=ix) for c in ("pred_adj", "B", "rv_actual"))
    return out


# ------------------------------------------------------------------------------------------------ scoring
class Scorer:
    def __init__(self, idx: pd.DatetimeIndex, rv: np.ndarray, event_days: pd.DatetimeIndex):
        self.idx = idx
        self.rv = rv
        day = pd.DatetimeIndex(idx.normalize())
        self.days, self.code = np.unique(day.to_numpy(), return_inverse=True)
        self.is_event_day = pd.DatetimeIndex(self.days).isin(event_days)
        ev = self.is_event_day[self.code]
        self.mask = {"all": np.ones(len(idx), bool), "event": ev, "ord": ~ev}
        mins = idx.hour * 60 + idx.minute
        self.pm = np.asarray(mins >= PM_FROM[0] * 60 + PM_FROM[1])
        self.L: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}

    def add(self, key: tuple[str, str], plain: np.ndarray) -> None:
        assert np.isfinite(plain).all() and (plain > 0).all(), key
        r = self.rv / plain
        self.L[key] = (r - np.log(r) - 1.0, (self.rv - plain) ** 2)

    def daily(self, x: np.ndarray, sp: str) -> np.ndarray:
        m = self.mask[sp]
        s = np.bincount(self.code[m], weights=x[m], minlength=len(self.days))
        n = np.bincount(self.code[m], minlength=len(self.days))
        return s[n > 0] / n[n > 0]

    def mean(self, key: tuple[str, str], sp: str, loss: int = 0) -> float:
        return float(self.L[key][loss][self.mask[sp]].mean())

    def comp(self, a: tuple[str, str], b: tuple[str, str], sp: str) -> dict:
        out: dict = dict(span=sp, n_bars=int(self.mask[sp].sum()), n_days=int(len(self.daily(self.L[a][0], sp))))
        for li, nm in ((0, "qlike"), (1, "mse")):
            xa, xb = self.L[a][li], self.L[b][li]
            r = dm_test(self.daily(xa, sp), self.daily(xb, sp), hac_lag=HAC_LAG)
            qa, qb = float(xa[self.mask[sp]].mean()), float(xb[self.mask[sp]].mean())
            out.update(
                {
                    f"{nm}_a": qa,
                    f"{nm}_b": qb,
                    f"pct_{nm}": 100.0 * (qa / qb - 1.0),
                    f"dm_{nm}": float(r["dm"]),
                    f"p_{nm}": float(r["p"]),
                }
            )
            assert int(r["hac_lag"]) == HAC_LAG
        out["gain"] = out["qlike_b"] - out["qlike_a"]  # loss of b minus loss of a (> 0: a better)
        return out


def har_of(c: str) -> tuple[str, str]:
    return ("ols", "har") if c in LIN else (c, "har")


# ------------------------------------------------------------------------------------------------ formatting
def fmt(x: float, dp: int) -> str:
    v = round(float(x), dp)
    if v == 0:
        v = 0.0
    return f"{'$-$' if v < 0 else ''}{abs(v):.{dp}f}"


def sfmt(x: float, dp: int) -> str:
    """fmt with an explicit $+$ on a positive value (a range that may straddle zero)."""
    s = fmt(x, dp)
    return s if s.startswith("$-$") or float(s) == 0 else f"$+${s}"


def rng_txt(vals: list[float], dp: int, signed: bool = False) -> tuple[str, bool]:
    """A range for the prose; the sentence gives the direction (as \\rhorVsOlsRedTreeRng): magnitudes "a--b" when the
    values share one sign, else (or with signed=True) "a to b" with signs; one value when both ends print alike.
    Returns (text, mixed): mixed = an unsigned range whose values have both signs (printed signed instead)."""
    v = np.asarray(vals, float)
    assert len(v) and np.isfinite(v).all(), vals
    mixed = not ((v > 0).all() or (v < 0).all())
    if signed or mixed:
        lo, hi = sfmt(v.min(), dp), sfmt(v.max(), dp)
        return (lo if lo == hi else f"{lo} to {hi}"), mixed and not signed
    a = np.abs(v)
    lo, hi = fmt(a.min(), dp), fmt(a.max(), dp)
    return (lo if lo == hi else f"{lo}--{hi}"), False


def fmt_p(p: float) -> str:
    return "\\ensuremath{<}0.001" if p < 0.001 else f"{p:.3f}"


def thousands(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def txt(x: float, dp: int = 2) -> str:
    return f"{x:.{dp}f}"


def ptxt(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# ------------------------------------------------------------------------------------------------ readings (item 3)
def sig_lo(r: dict | str, literal: bool = False) -> bool | str:
    """Leave-one-out significant: the full set without g has the significantly HIGHER loss (DM > 0, p < 0.05); with
    literal=True the two-sided test alone (either sign).  'pending' / 'na' pass through."""
    if isinstance(r, str):
        return r
    if literal:
        return bool(r["p_qlike"] < ALPHA)
    return bool(r["p_qlike"] < ALPHA and r["dm_qlike"] > 0)


def class_reading(J: float, A: dict[str, float | str], lo: dict[str, bool | str]) -> tuple[str, dict]:
    """RULE item 3 for one class and day type.  J joint gain (HAR loss - full loss); A[g] add-alone gain (HAR - add_g)
    or 'pending'; lo[g] leave-one-out significant (bool) or 'pending' / 'na' (arm not run).  Order: CONCENTRATED,
    REDUNDANT, MIXED; an undetermined test -> PENDING (an input pending) or INCOMPLETE (an arm not run)."""
    known = {g: float(a) for g, a in A.items() if not isinstance(a, str)}
    det: dict = dict(J=J, sum_add=np.nan, ratio_sum=np.nan, top_group="", top_share=np.nan, n_lo_sig=0, n_lo_unknown=0)
    if known:
        g0 = max(known, key=lambda g: known[g])
        det.update(top_group=g0, top_share=known[g0] / J if J > 0 else np.nan)
    det["n_lo_sig"] = sum(v is True for v in lo.values())
    det["n_lo_unknown"] = sum(isinstance(v, str) for v in lo.values())
    det["n_lo_na"] = sum(v == "na" for v in lo.values())
    if len(known) == len(A):
        det["sum_add"] = sum(known.values())
        det["ratio_sum"] = det["sum_add"] / J if J > 0 else np.nan
    det.update(conc_test=None, red_test=None)  # each definition on its own (True / False / None = undetermined)
    if J <= 0:
        return "NO JOINT GAIN", det
    blockers: list[str] = []
    conc = False
    for g, a in A.items():
        if isinstance(a, str):
            blockers.append(a)
            continue
        if a >= 0.75 * J:
            if lo[g] is True:
                conc = True
            elif isinstance(lo[g], str):
                blockers.append(str(lo[g]))
    det["conc_test"] = True if conc else (None if blockers else False)
    red_blk: list[str] = []
    if not np.isfinite(det["sum_add"]):
        red_blk = ["pending"]
    elif det["sum_add"] > 1.5 * J:
        if det["n_lo_sig"] >= 2:
            det["red_test"] = False
        elif det["n_lo_sig"] + det["n_lo_unknown"] <= 1:
            det["red_test"] = True
        else:
            red_blk = [v for v in lo.values() if isinstance(v, str)]
    else:
        det["red_test"] = False
    if conc:
        return "CONCENTRATED", det
    if blockers:
        return ("INCOMPLETE" if "na" in blockers else "PENDING"), det
    if det["red_test"] is True:
        return "REDUNDANT", det
    if red_blk:
        return ("INCOMPLETE" if "na" in red_blk else "PENDING"), det
    return "MIXED", det


def tri(v: object) -> bool | None:
    """A test outcome read back from a DataFrame: True / False, or None (undetermined; None or NaN)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return bool(v)


def type_reading(reads: dict[str, str]) -> str:
    """Majority of the classes with a determinable reading (INCOMPLETE ones excluded: 'the forest where its arms
    exist'); PENDING if a pending class could still decide it; SPLIT when no majority."""
    elig = {c: r for c, r in reads.items() if r != "INCOMPLETE"}
    n = len(elig)
    votes: dict[str, int] = {}
    for r in elig.values():
        if r != "PENDING":
            votes[r] = votes.get(r, 0) + 1
    for r, v in votes.items():
        if v > n / 2:
            return r
    return "PENDING" if any(r == "PENDING" for r in elig.values()) else "SPLIT"


# ------------------------------------------------------------------------------------------------ main
def main() -> None:
    RES.mkdir(parents=True, exist_ok=True)
    gates: dict = {"fail": []}
    prov: list[dict] = []

    def chk(ok: bool, msg: str) -> None:
        if not ok:
            gates["fail"].append(msg)
            say(f"GATE FAIL: {msg}")

    # ---------------------------------------------------------------- inputs
    def need(rel: str) -> Src:
        s = find(rel)
        assert s is not None, f"input {rel} missing"
        return s

    srcs = {
        "lin_fix": need("results/window_sweep_2014start_fomcfix/w3000/forecasts.parquet"),
        "tree_fix": need("results/pooled_trees_final_2014start_fomcfix_w3000/forecasts.parquet"),
        "rf_fix": need("results/pooled_rf_skx_2014start_fomcfix_w3000/forecasts.parquet"),
        "lin_pre": need("results/window_sweep_2014start_2026-10-22/w3000/forecasts.parquet"),
        "tree_pre": need("results/pooled_trees_final_2014start_w3000/forecasts.parquet"),
        "rf_pre": need("results/pooled_rf_skx_2014start_w3000/forecasts.parquet"),
    }
    abl = "results/ablations_2014start_w3000/"
    a_lin_fix = find(abl + "forecasts/linear_wide_fix.parquet", BRANCH_ABL)
    a_lin_pre = find(abl + "forecasts/linear_wide.parquet", BRANCH_ABL)
    a_tree = find(abl + "forecasts/trees_wide.parquet", BRANCH_ABL)
    a_tree_fix = find(abl + "forecasts/trees_wide_fix.parquet", BRANCH_ABL)
    # 2026-10-28 ablation addendum J: LightGBM's and XGBoost's lo_<g>, g != fomc, on the patched design (when merged)
    a_tree_lofix = find(abl + "forecasts/trees_wide_lofix.parquet", BRANCH_ABL)
    a_rf_lofomc = find(abl + "forecasts/trees_wide_rflofomc.parquet", BRANCH_ABL)
    a_sc_pre = find(abl + "scores/metrics.csv", BRANCH_ABL)
    a_sc_fix = find(abl + "scores_fomcfix/metrics.csv", BRANCH_ABL)
    assert a_lin_pre is not None and a_tree is not None and a_sc_pre is not None
    lad_fix = find("results/ladder_2026-10-26/fomcfix/forecasts_wide.parquet", BRANCH_LAD)
    lad_patched = lad_fix is not None
    lad_dir = "results/ladder_2026-10-26/" + ("fomcfix/" if lad_patched else "")
    lad = lad_fix if lad_patched else find(lad_dir + "forecasts_wide.parquet", BRANCH_LAD)
    lad_sc = find(lad_dir + "scores.csv", BRANCH_LAD)
    lad_gates = find(lad_dir + "gates.json", BRANCH_LAD)
    assert lad is not None and lad_sc is not None and lad_gates is not None

    def note(name: str, s: Src | None, design: str, use: str, missing: str = "PENDING") -> None:
        prov.append(
            dict(
                input=name,
                path=s.rel if s else "",
                source=s.where if s else "",
                design=design if s else "",
                status="ok" if s else missing,
                used_for=use,
            )
        )

    note("Table 1 linear (full, HAR, OLS)", srcs["lin_fix"], "patched", "full / HAR rows, every item")
    note("Table 1 LightGBM, XGBoost", srcs["tree_fix"], "patched", "full / HAR rows, every item")
    note("Table 1 random forest", srcs["rf_fix"], "patched", "full / HAR rows, every item")
    note("pre-patch linear full set", srcs["lin_pre"], "pre-patch", "gate (linear_wide_fix invariance) / fallback")
    note("pre-patch trees full set", srcs["tree_pre"], "pre-patch", "reference of the trees' pre-patch lo_<g>")
    note("pre-patch forest full set", srcs["rf_pre"], "pre-patch", "reference of the forest's pre-patch lo_rm")
    note("linear ablation arms (patched)", a_lin_fix, "patched (FOMC arms) / invariant", "items 1 (a), (b), 3 (ii)")
    note("linear ablation arms (pre-patch)", a_lin_pre, "pre-patch", "gate (invariance); fallback")
    note(
        "tree ablation arms", a_tree, "invariant (add_g, g!=fomc; lo_fomc) / pre-patch (lo_g, g!=fomc)", "1 (b), 3 (ii)"
    )
    note("tree add_fomc (patched)", a_tree_fix, "patched", "items 1 (a), 3 (ii)")
    if a_tree_lofix is not None:
        note("boosted trees' lo_g, g!=fomc (patched, addendum J)", a_tree_lofix, "patched", "1 (b), 3 (ii)")
    note(
        "forest lo_fomc",
        a_rf_lofomc,
        "invariant",
        "items 1 (b), 3 (ii)",
        "NOT RUN (stopped, ablation addendum E (i))" if RF_LOFOMC_DROPPED else "PENDING",
    )
    note("ablation scores, pre-patch", a_sc_pre, "pre-patch", "gate (arm QLIKE)")
    note("ablation scores, patched", a_sc_fix, "patched", "gate (arm QLIKE)")
    note("feature-count ladder", lad, "patched" if lad_patched else "pre-patch", "item 3 (iii)")
    note("ladder scores", lad_sc, "patched" if lad_patched else "pre-patch", "gate (ladder QLIKE)")

    # ---------------------------------------------------------------- templates
    raw: dict = {}
    raw.update(long_panel(srcs["lin_fix"], {"ridge": "ridge", "lasso": "lasso", "enet": "enet", "ols": "ols"}, False))
    raw.update(long_panel(srcs["tree_fix"], {"lgbm": "lgbm", "xgb": "xgb"}, False))
    raw.update(long_panel(srcs["rf_fix"], {"rf_skx": "rf"}, False))
    raw.update(long_panel(srcs["tree_pre"], {"lgbm": "lgbm", "xgb": "xgb"}, True))
    raw.update(long_panel(srcs["rf_pre"], {"rf_skx": "rf"}, True))
    raw.update(long_panel(srcs["lin_pre"], {"ridge": "ridge", "lasso": "lasso", "enet": "enet"}, True))
    B = raw[("lgbm", "full")][1]
    for k, (_, b, _) in raw.items():
        bb = b.reindex(B.index).to_numpy()
        chk(np.array_equal(bb, B.to_numpy()), f"B of {k} differs from the LightGBM template's")
    rv_core = core_rv()
    days = evaluation_days(rv_core)

    # scored set (Table 1's 13 rows)
    t1keys = [(c if c != "rf_skx" else "rf", SET_ARM[s]) for c, s in T1_ROWS]
    idx = None
    for k in t1keys:
        p, b, _ = raw[k]
        pl = p.to_numpy() ** 2 * b.to_numpy()
        good = p.index[np.isfinite(pl) & (pl > 0)]
        idx = good if idx is None else idx.intersection(good)
    assert idx is not None
    idx = idx[pd.DatetimeIndex(idx.normalize()).isin(days)]
    t = rv_core.reindex(idx).to_numpy()
    idx = idx[(t > 0) & np.isfinite(t)].sort_values()
    per_day = pd.Series(1, index=idx.normalize()).groupby(level=0).size()
    gates["n_bars"], gates["n_days"] = len(idx), int(per_day.size)
    chk(len(idx) == N_BARS, f"scored bars {len(idx)} != {N_BARS}")
    chk(per_day.size == N_DAYS and bool((per_day == BARS_A_DAY).all()), "not 2,579 days of 13 bars")
    rv = rv_core.reindex(idx).to_numpy(float)
    for k in t1keys:  # RV = the forecasts' rv_actual
        ra = raw[k][2].reindex(idx).to_numpy()
        chk(bool(np.allclose(ra, rv, rtol=1e-12, atol=0)), f"rv_actual of {k} != core_stats sumret2")

    # day types
    st = statement_stamps()
    st_span = st[pd.DatetimeIndex(st.normalize()).isin(per_day.index)]
    event_days = pd.DatetimeIndex(st_span.normalize().unique())
    gates["n_event_days"] = len(event_days)
    gates["statement_stamps"] = sorted(set(st_span.strftime("%H:%M")))
    chk(len(event_days) == N_EVENT, f"EVENT days {len(event_days)} != {N_EVENT}")
    chk(gates["statement_stamps"] == ["14:30"], f"statement stamps {gates['statement_stamps']} (expected 14:30)")
    S = Scorer(idx, rv, event_days)
    n_ev_bars = int(S.mask["event"].sum())
    gates["n_event_bars"] = n_ev_bars

    def plain_of(p: pd.Series) -> np.ndarray:
        return p.reindex(idx).to_numpy(float) ** 2 * B.reindex(idx).to_numpy(float)

    PROV: dict[tuple[str, str], str] = {}
    for k, (p, _, _) in raw.items():
        S.add(k, plain_of(p))
        PROV[k] = "pre-patch" if k[1].endswith("_pre") else "patched"
    for c in TREES:  # the trees' HAR fits: no FOMC column, equal on both designs
        chk(
            np.array_equal(raw[(c, "har")][0].reindex(idx).to_numpy(), raw[(c, "har_pre")][0].reindex(idx).to_numpy()),
            f"{c}: HAR fit differs across the patched and pre-patch runs",
        )

    # ---------------------------------------------------------------- gate: Table 1
    t1m = pd.read_csv(ROOT / "results" / "oos_final_2014start" / "table1_w3000" / "metrics.csv")
    t1d = pd.read_csv(ROOT / "results" / "oos_final_2014start" / "table1_w3000" / "dm.csv")
    t1rows = []
    for _, r in t1m.iterrows():
        k = (r["model"], SET_ARM[r["set"]])
        qv, mv = S.mean(k, "all", 0), S.mean(k, "all", 1)
        chk(bool(np.isclose(qv, r["qlike"], rtol=1e-12, atol=0)), f"Table 1 QLIKE {k}: {qv} vs {r['qlike']}")
        chk(bool(np.isclose(mv, r["mse"], rtol=1e-12, atol=0)), f"Table 1 MSE {k}: {mv} vs {r['mse']}")
        t1rows.append(dict(model=k[0], set=r["set"], qlike=qv, qlike_table1=r["qlike"], mse=mv, mse_table1=r["mse"]))
    for comp_name, vs in (("all features vs HAR benchmark", "own"), ("vs OLS", "ols")):
        dd = t1d[(t1d["comparison"] == comp_name) & (t1d["loss"] == "qlike") & (t1d["set"] == "all_features")]
        for _, r in dd.iterrows():
            c = r["model"]
            ref = (c, "har") if vs == "own" else ("ols", "har")
            x = S.comp((c, "full"), ref, "all")
            chk(bool(np.isclose(x["dm_qlike"], r["dm"], rtol=1e-9, atol=0)), f"Table 1 DM {c} vs {vs}: {x['dm_qlike']}")
            t1rows.append(dict(model=c, set=f"DM vs {vs}", qlike=x["dm_qlike"], qlike_table1=r["dm"]))
        chk(len(dd) == 6, f"Table 1 dm.csv: {len(dd)} rows of '{comp_name}'")
    pd.DataFrame(t1rows).to_csv(RES / "gate_table1.csv", index=False)

    # ---------------------------------------------------------------- ablation arms
    AVAIL: dict[tuple[str, str], str] = {}  # (class, arm) -> "ok" | "pending" | "na"
    REF: dict[tuple[str, str], tuple[str, str]] = {}  # (class, lo_g) -> its full-set reference

    def put(key: tuple[str, str], pred: np.ndarray, d: pd.DatetimeIndex, prov_: str) -> None:
        s = pd.Series(pred, index=d)
        S.add(key, plain_of(s))
        PROV[key] = prov_
        AVAIL[key] = "ok"

    wl_pre = a_lin_pre.read_parquet()
    dl_pre = pd.DatetimeIndex(wl_pre["date"])
    if a_lin_fix is not None:
        wl = a_lin_fix.read_parquet()
        dl = pd.DatetimeIndex(wl["date"])
        chk(dl.equals(dl_pre), "linear_wide_fix dates != linear_wide")
        n_inv, n_chg = 0, 0
        for c in LIN:
            for g in GROUPS:
                for kind in ("add", "lo"):
                    arm = f"{kind}_{g}"
                    col = f"{c}|{arm}"
                    has_fomc = (kind == "add") == (g == "fomc")
                    same = np.array_equal(wl[col].to_numpy(), wl_pre[col].to_numpy())
                    if has_fomc:
                        n_chg += int(not same)
                        chk(not same, f"{col}: holds FOMC columns but equals its pre-patch walk")
                    else:
                        n_inv += int(same)
                        chk(same, f"{col}: FOMC-free arm differs from its pre-patch walk")
                    put((c, arm), wl[col].to_numpy(float), dl, "patched" if has_fomc else "invariant")
                    if kind == "lo":
                        REF[(c, arm)] = (c, "full")
        gates["linear_fix_invariant_equal"], gates["linear_fix_fomc_changed"] = n_inv, n_chg
        del wl
    else:
        for c in LIN:
            for g in GROUPS:
                for kind in ("add", "lo"):
                    arm = f"{kind}_{g}"
                    has_fomc = (kind == "add") == (g == "fomc")
                    if kind == "add" and g == "fomc":
                        AVAIL[(c, arm)] = "pending"
                        continue
                    put(
                        (c, arm), wl_pre[f"{c}|{arm}"].to_numpy(float), dl_pre, "pre-patch" if has_fomc else "invariant"
                    )
                    if kind == "lo":
                        REF[(c, arm)] = (c, "full_pre" if has_fomc else "full")
    del wl_pre

    wt = a_tree.read_parquet()
    dt_ = pd.DatetimeIndex(wt["date"])
    wt.columns = ["date"] + [col.replace("rf_skx|", "rf|") for col in wt.columns[1:]]
    if a_tree_lofix is not None:  # addendum J: the patched lo_<g> replace the pre-patch ones
        wj = a_tree_lofix.read_parquet()
        dj = pd.DatetimeIndex(wj["date"])
        for col in wj.columns[1:]:
            cl, arm = col.split("|")
            assert cl in ("lgbm", "xgb") and arm.startswith("lo_") and arm != "lo_fomc", col
            put((cl, arm), wj[col].to_numpy(float), dj, "patched")
            REF[(cl, arm)] = (cl, "full")
        done_j = {tuple(col.split("|")) for col in wj.columns[1:]}
        del wj
    else:
        done_j = set()
    for c in TREES:
        for g in GROUPS:
            for kind in ("add", "lo"):
                arm = f"{kind}_{g}"
                col = f"{c}|{arm}"
                has_fomc = (kind == "add") == (g == "fomc")
                if arm == "add_fomc":
                    if col in wt.columns:  # pre-patch, kept for the provisional readings only
                        S.add((c, "add_fomc_pre"), plain_of(pd.Series(wt[col].to_numpy(float), index=dt_)))
                        PROV[(c, "add_fomc_pre")] = "pre-patch"
                    AVAIL[(c, arm)] = "pending"
                    continue
                if (c, arm) in done_j:
                    continue  # addendum J: already put from the patched walks
                if col not in wt.columns:
                    lofomc_wait = (c, arm) == ("rf", "lo_fomc") and not RF_LOFOMC_DROPPED
                    AVAIL[(c, arm)] = "pending" if lofomc_wait else "na"
                    continue
                put((c, arm), wt[col].to_numpy(float), dt_, "pre-patch" if has_fomc else "invariant")
                if kind == "lo":
                    REF[(c, arm)] = (c, "full_pre" if has_fomc else "full")
    del wt
    if a_tree_fix is not None:
        wf = a_tree_fix.read_parquet()
        for col in wf.columns[1:]:
            cl, arm = col.split("|")
            cl = {"rf_skx": "rf"}.get(cl, cl)
            assert arm == "add_fomc", col
            put((cl, arm), wf[col].to_numpy(float), pd.DatetimeIndex(wf["date"]), "patched")
    if a_rf_lofomc is not None:
        wr = a_rf_lofomc.read_parquet()
        col = "rf_skx|lo_fomc" if "rf_skx|lo_fomc" in wr.columns else "rf|lo_fomc"
        put(("rf", "lo_fomc"), wr[col].to_numpy(float), pd.DatetimeIndex(wr["date"]), "invariant")
        REF[("rf", "lo_fomc")] = ("rf", "full")
    gates["arms_ok"] = sorted(f"{c}|{a}" for (c, a), v in AVAIL.items() if v == "ok")
    gates["arms_pending"] = sorted(f"{c}|{a}" for (c, a), v in AVAIL.items() if v == "pending")
    gates["arms_not_run"] = sorted(f"{c}|{a}" for (c, a), v in AVAIL.items() if v == "na")

    # gate: every arm's every-bar QLIKE = the ablation scorer's
    scp = a_sc_pre.read_csv()
    scp = scp[scp["span"] == "all"]
    scf = None
    if a_sc_fix is not None:
        scf = a_sc_fix.read_csv()
        scf = scf[scf["span"] == "all"]
    n_cmp = 0
    for (c, arm), v in AVAIL.items():
        if v != "ok":
            continue
        use_fix = PROV[(c, arm)] == "patched" or (c in LIN and a_lin_fix is not None)
        tab = scf if use_fix else scp
        if tab is None:
            continue
        r = tab[(tab["cls"] == c) & (tab["arm"] == arm)]
        if len(r) != 1:
            continue
        q = S.mean((c, arm), "all")
        chk(bool(np.isclose(q, float(r["qlike"].iloc[0]), rtol=1e-12, atol=0)), f"arm {c}|{arm}: QLIKE {q} vs scorer")
        n_cmp += 1
    gates["arms_checked_against_ablation_scorer"] = n_cmp

    # ---------------------------------------------------------------- ladder
    lw = lad.read_parquet()
    ld = pd.DatetimeIndex(lw["date"])
    lsc = lad_sc.read_csv()
    lg = json.loads(lad_gates.read_text())
    gates["ladder_design"] = "patched" if lad_patched else "pre-patch"
    gates["ladder_source"] = lad.where
    gates["ladder_gate_notes"] = {}
    for c in CLASSES:
        m = LAD_KEY.get(c, c)
        g_ = lg.get(m, {})
        # the gates the PRIMARY rungs depend on (the only rungs read here): GT (trees: the full rung = the patched walk
        # at all 126 origins) or GL1 (linear replays = the walks), and GP (patch invariance) on primary rungs; a GP
        # mismatch on a secondary (look-ahead) rung S<k> is recorded, not failed (2026-10-26: LightGBM S3, 1.1e-13)
        core = (
            g_.get("GT_origin_bit_equal") == g_.get("GT_origin_n") == 126
            if c in TREES
            else bool(g_.get("GL1_pass", False))
        )
        gp_bad = [str(x) for x in g_.get("GP_differ", [])]
        gp_primary = [x for x in gp_bad if not x.startswith("S")]
        chk(bool(core) and not gp_primary, f"ladder gate of {m} (primary rungs): core {core}, GP {gp_primary}")
        if not g_.get("pass", False):
            gates["ladder_gate_notes"][m] = f"ladder's own pass flag false: GP on secondary rungs {gp_bad}"
        for rung in ("full", "har", *(f"P{k}" for k in KS)):
            key = (f"lad_{c}", rung)
            S.add(key, plain_of(pd.Series(lw[f"{m}|{rung}"].to_numpy(float), index=ld)))
            PROV[key] = gates["ladder_design"]
            r = lsc[(lsc["model"] == m) & (lsc["rung"] == rung) & (lsc["span"] == "all")]
            r = r[r["ranking"].isin(["P", "both"])]
            q = S.mean(key, "all")
            chk(len(r) == 1 and bool(np.isclose(q, float(r["qlike"].iloc[0]), rtol=1e-12, atol=0)), f"ladder {key}")
    del lw

    gates["pass"] = not gates["fail"]
    (RES / "gates.json").write_text(json.dumps(gates, indent=1, default=str), encoding="utf-8")
    pd.DataFrame(prov).to_csv(RES / "provenance.csv", index=False)
    assert gates["pass"], gates["fail"]
    say(f"gates PASS: {N_BARS} bars, {N_DAYS} days, {N_EVENT} EVENT days ({n_ev_bars} bars)")

    def ok(key: tuple[str, str]) -> bool:
        return key in S.L

    # ---------------------------------------------------------------- item 1
    rows1 = []
    for c in CLASSES:
        H = har_of(c)
        specs = [
            ("t1", (c, "full"), H, "all"),
            ("a", (c, "full"), (c, "add_fomc"), "all"),
            ("b", (c, "lo_fomc"), H, "all"),
            ("c", (c, "full"), H, "ord"),
        ]
        for corr, ka, kb, sp in specs:
            base = dict(cls=c, corr=corr, arm_a=ka[1], arm_b=kb[1], role="rule")
            if ok(ka) and ok(kb):
                design = "patched" if "pre-patch" not in (PROV[ka], PROV[kb]) else "pre-patch"
                rows1.append({**base, **S.comp(ka, kb, sp), "status": "ok", "design": design})
            else:
                na = "na" in (AVAIL.get(ka), AVAIL.get(kb))
                rows1.append({**base, "span": sp, "status": "NA" if na else "PENDING"})
        if ok((c, "add_fomc_pre")) and not ok((c, "add_fomc")):  # information: the pre-patch (a)
            rows1.append(
                dict(
                    cls=c,
                    corr="a",
                    arm_a="full_pre",
                    arm_b="add_fomc_pre",
                    role="info_prepatch",
                    **S.comp((c, "full_pre"), (c, "add_fomc_pre"), "all"),
                    status="ok",
                    design="pre-patch",
                )
            )
    C1 = pd.DataFrame(rows1)
    agree = {}
    for c in CLASSES:
        f = C1[(C1.cls == c) & (C1.role == "rule") & C1["corr"].isin(["a", "b", "c"])]
        known = f[f.status == "ok"]["pct_qlike"].to_numpy(float)
        spread = float(known.max() - known.min()) if len(known) >= 2 else np.nan
        if len(known) == 3:
            agree[c] = ("agree" if spread <= AGREE_PP else "disagree", spread)
        elif len(known) == 2 and spread > AGREE_PP:
            agree[c] = ("disagree", spread)
        elif (f.status == "NA").any():  # a correction not run: not decidable
            agree[c] = ("n/a", spread)
        else:
            agree[c] = ("PENDING", spread)
    C1["agree"] = C1["cls"].map({c: a for c, (a, _) in agree.items()})
    C1["spread_pp"] = C1["cls"].map({c: s for c, (_, s) in agree.items()})
    C1.to_csv(RES / "item1_corrections.csv", index=False)

    # ---------------------------------------------------------------- item 2
    dd = {}
    for c in CLASSES:
        H = har_of(c)
        dd[c] = S.daily(S.L[(c, "full")][0] - S.L[H][0], "all")
    D2 = pd.DataFrame({"date": pd.DatetimeIndex(S.days), "event": S.is_event_day})
    for c in CLASSES:
        D2[f"d_{c}"] = dd[c]
        D2[f"cum_{c}"] = np.cumsum(dd[c])
        D2[f"cum_ord_{c}"] = np.cumsum(np.where(S.is_event_day, 0.0, dd[c]))
    D2.to_csv(RES / "item2_daily_diff.csv", index=False, float_format="%.10g")
    stamps = idx.strftime("%H:%M")
    bars = sorted(set(stamps))
    assert len(bars) == BARS_A_DAY
    rows2 = []
    for c in CLASSES:
        H = har_of(c)
        diff = S.L[(c, "full")][0] - S.L[H][0]
        for sp in ("event", "ord"):
            for b in bars:
                m = S.mask[sp] & (stamps == b)
                rows2.append(
                    dict(
                        cls=c,
                        daytype=sp,
                        bar_end=b,
                        n=int(m.sum()),
                        mean_diff=float(diff[m].mean()),
                        qlike_full=float(S.L[(c, "full")][0][m].mean()),
                        qlike_har=float(S.L[H][0][m].mean()),
                    )
                )
    B2 = pd.DataFrame(rows2)
    B2.to_csv(RES / "item2_by_bar.csv", index=False)
    rows2s = []
    for c in CLASSES:
        H = har_of(c)
        gain = S.L[H][0] - S.L[(c, "full")][0]  # > 0: the full set better
        tot = float(gain.sum())
        ev = float(gain[S.mask["event"]].sum())
        evpm = float(gain[S.mask["event"] & S.pm].sum())
        rows2s.append(
            dict(
                cls=c,
                total_gain_sum=tot,
                event_gain_sum=ev,
                event_pm_gain_sum=evpm,
                share_event_pct=100.0 * ev / tot,
                share_event_pm_pct=100.0 * evpm / tot,
                share_ord_pct=100.0 * (tot - ev) / tot,
                event_bars_pct=100.0 * S.mask["event"].mean(),
                event_pm_bars_pct=100.0 * (S.mask["event"] & S.pm).mean(),
            )
        )
    SH = pd.DataFrame(rows2s)
    SH.to_csv(RES / "item2_shares.csv", index=False)

    # ---------------------------------------------------------------- item 3 (i)
    rows3 = []
    for b in ("ridge", "enet"):
        for sp in SPANS:
            rows3.append(dict(pair=f"lasso_vs_{b}", **S.comp(("lasso", "full"), (b, "full"), sp)))
    P3 = pd.DataFrame(rows3)
    P3.to_csv(RES / "item3i_lasso_pairs.csv", index=False)

    # ---------------------------------------------------------------- item 3 (ii)
    rows4 = []
    for c in CLASSES:
        H = har_of(c)
        for sp in SPANS:
            rows4.append(
                dict(cls=c, span=sp, group="", kind="joint", ref=f"{H[0]}|{H[1]}", design="patched", status="ok")
                | S.comp((c, "full"), H, sp)
            )
            for g in GROUPS:
                for kind in ("add", "lo"):
                    arm = f"{kind}_{g}"
                    key = (c, arm)
                    base = dict(cls=c, span=sp, group=g, kind=kind)
                    if not ok(key):
                        rows4.append({**base, "status": AVAIL.get(key, "na").upper()})
                        continue
                    ref = H if kind == "add" else REF[key]
                    rows4.append(
                        {
                            **base,
                            "ref": f"{ref[0]}|{ref[1]}",
                            "design": PROV[key],
                            "status": "ok",
                            **S.comp(key, ref, sp),
                        }
                    )
    A4 = pd.DataFrame(rows4)
    A4.to_csv(RES / "item3ii_ablation_daytype.csv", index=False)

    def readings(provisional: bool, literal: bool) -> pd.DataFrame:
        out = []
        for sp in ("event", "ord", "all"):
            per: dict[str, dict[str, str]] = {"lin": {}, "tree": {}}
            for c in CLASSES:
                f = A4[(A4.cls == c) & (A4.span == sp)]
                J = float(f[f.kind == "joint"]["gain"].iloc[0])
                A: dict[str, float | str] = {}
                lo: dict[str, bool | str] = {}
                for g in GROUPS:
                    ra = f[(f.kind == "add") & (f.group == g)].iloc[0]
                    if ra["status"] == "ok":
                        A[g] = float(ra["gain"])
                    elif provisional and g == "fomc" and ok((c, "add_fomc_pre")):
                        A[g] = S.comp((c, "add_fomc_pre"), har_of(c), sp)["gain"]
                    else:
                        A[g] = "pending"
                    rl = f[(f.kind == "lo") & (f.group == g)].iloc[0]
                    lo[g] = sig_lo(rl.to_dict(), literal) if rl["status"] == "ok" else rl["status"].lower()
                rd, det = class_reading(J, A, lo)
                per["lin" if c in LIN else "tree"][c] = rd
                out.append(
                    dict(
                        span=sp,
                        level="class",
                        who=c,
                        reading=rd,
                        **det,
                        lo_sig=" ".join(g for g in GROUPS if lo[g] is True),
                    )
                )
            for t, rr in per.items():
                out.append(dict(span=sp, level="type", who=t, reading=type_reading(rr)))
        return pd.DataFrame(out)

    RD = readings(False, False)
    # the RULE's CONCENTRATED and REDUNDANT definitions can both hold (coordinator, 2026-10-26: report both): the
    # reading with REDUNDANT tested first, by class and by model type, beside the RULE's order (CONCENTRATED first)
    alt = []
    for _, r in RD.iterrows():
        if r["level"] != "class":
            alt.append("")
            continue
        rt, ct = tri(r["red_test"]), tri(r["conc_test"])
        if r["reading"] == "NO JOINT GAIN":
            alt.append(r["reading"])
        elif rt is True:
            alt.append("REDUNDANT")
        elif rt is None:
            alt.append("PENDING" if not np.isfinite(r["ratio_sum"]) or r["n_lo_na"] == 0 else "INCOMPLETE")
        elif ct is True:
            alt.append("CONCENTRATED")
        else:
            alt.append(r["reading"])  # MIXED, or undetermined as in the RULE's order
    RD["reading_redundant_first"] = alt
    for i, r in RD.iterrows():
        if r["level"] == "type":
            cs = LIN if r["who"] == "lin" else TREES
            sub = RD[(RD.level == "class") & (RD.span == r["span"]) & RD.who.isin(cs)]
            RD.loc[i, "reading_redundant_first"] = type_reading(dict(zip(sub.who, sub.reading_redundant_first)))
    RD.to_csv(RES / "item3_readings.csv", index=False)
    RDl = readings(False, True)
    RDl.to_csv(RES / "item3_readings_literal_twosided.csv", index=False)
    RDp = readings(True, False)
    RDp.to_csv(RES / "item3_readings_provisional_prepatch_addfomc.csv", index=False)

    # ---------------------------------------------------------------- item 3 (iii)
    rows5 = []
    for c in CLASSES:
        lk = f"lad_{c}"
        for sp in SPANS:
            l0, l46 = S.mean((lk, "har"), sp), S.mean((lk, "full"), sp)
            for kk in (0, *KS, 46):
                rung = {0: "har", 46: "full"}.get(kk, f"P{kk}")
                lk_ = S.mean((lk, rung), sp)
                rr: dict = dict(cls=c, span=sp, k=kk, rung=rung, qlike=lk_, design=gates["ladder_design"])
                rr["R_pct"] = 100.0 * (l0 - lk_) / (l0 - l46) if (l0 - l46) > 0 and kk not in (0, 46) else np.nan
                rr["gain_full_vs_k0_pct"] = 100.0 * (l46 / l0 - 1.0)
                if kk != 46:
                    x = S.comp((lk, rung), (lk, "full"), sp)
                    rr.update(dm_vs_full=x["dm_qlike"], p_vs_full=x["p_qlike"])
                rows5.append(rr)
    LD = pd.DataFrame(rows5)
    LD.to_csv(RES / "item3iii_ladder_daytype.csv", index=False)

    # ---------------------------------------------------------------- ADDITIONAL, NOT PRE-REGISTERED (coordinator,
    # 2026-10-26 after the first scoring): the full set against HAR + the futures' realized moments (add_rm), and HAR +
    # realized moments against HAR, on every bar, ORDINARY and EVENT days, every class; the same tests.  add_rm holds no
    # FOMC column (the same forecast on both designs); the full set and HAR are Table 1's (patched).
    rows6 = []
    for c in CLASSES:
        H = har_of(c)
        for sp in SPANS:
            for cmp_, ka, kb in (("full_vs_har_rm", (c, "full"), (c, "add_rm")), ("har_rm_vs_har", (c, "add_rm"), H)):
                x = S.comp(ka, kb, sp)
                j = S.comp((c, "full"), H, sp)["gain"]
                rows6.append(
                    dict(
                        label="additional, not pre-registered",
                        cls=c,
                        comparison=cmp_,
                        arm_a=f"{ka[0]}|{ka[1]}",
                        arm_b=f"{kb[0]}|{kb[1]}",
                        design=f"{PROV[ka]} vs {PROV[kb]}",
                        share_of_joint_gain=x["gain"] / j if j > 0 else np.nan,
                        **x,
                    )
                )
    XD = pd.DataFrame(rows6)
    XD.to_csv(RES / "itemX_additional_rm_not_preregistered.csv", index=False)

    write_tex(S, C1, agree, SH, B2, P3, A4, RD, LD, gates, PROV, AVAIL, n_ev_bars, XD)
    figure(D2, B2, bars, gates)
    summary(C1, agree, SH, B2, P3, A4, RD, RDl, RDp, LD, gates, prov, n_ev_bars, XD, AVAIL)
    say(f"wrote {RES}/ and the generated files")


# ------------------------------------------------------------------------------------------------ LaTeX
def write_tex(S, C1, agree, SH, B2, P3, A4, RD, LD, gates, PROV, AVAIL, n_ev_bars, XD) -> None:  # noqa: ANN001
    M: list[tuple[str, str, str]] = []

    def add(name: str, value: str, note: str) -> None:
        assert name.isalpha(), name
        M.append(("rhch" + name, value, note))

    n_ord = N_DAYS - N_EVENT
    add("NBars", thousands(N_BARS), "scored bars (Table 1)")
    add("NDays", thousands(N_DAYS), "scored days")
    add("NEventDays", str(N_EVENT), "EVENT days: scheduled FOMC statement days of the span")
    add("NOrdDays", thousands(n_ord), "ORDINARY days")
    add("NEventBars", thousands(n_ev_bars), "bars on EVENT days")
    add("EventBarPct", fmt(100.0 * n_ev_bars / N_BARS, 1), "EVENT-day bars, % of the scored bars")
    add("HacLag", str(HAC_LAG), "Newey-West lag of every DM test")
    add("AgreePp", fmt(AGREE_PP, 0), "item 1: the corrections agree within this many percentage points")
    add("LadderDesign", "patched" if gates["ladder_design"] == "patched" else "pre-patch", "design of the ladder")

    # item 1
    n_ag = {"agree": 0, "disagree": 0, "PENDING": 0}
    for c in CLASSES:
        T = CTAG[c]
        for corr, nm in (("t1", "TabOne"), ("a", "CorrA"), ("b", "CorrB"), ("c", "CorrC")):
            r = C1[(C1.cls == c) & (C1["corr"] == corr) & (C1.role == "rule")].iloc[0]
            what = {
                "t1": "full vs HAR, every bar (Table 1)",
                "a": "full vs HAR+FOMC, every bar",
                "b": "full without FOMC vs HAR, every bar",
                "c": "full vs HAR, ORDINARY days",
            }[corr]
            if r["status"] == "ok":
                add(f"{T}{nm}Pct", fmt(r["pct_qlike"], 1), f"{c}: {what}: % QLIKE change")
                add(f"{T}{nm}Dm", fmt(r["dm_qlike"], 2), f"{c}: {what}: DM")
                add(f"{T}{nm}P", fmt_p(r["p_qlike"]), f"{c}: {what}: DM p")
                add(f"{T}{nm}MsePct", fmt(r["pct_mse"], 1), f"{c}: {what}: % MSE change")
                add(f"{T}{nm}MseP", fmt_p(r["p_mse"]), f"{c}: {what}: DM p on MSE")
            else:
                na = r["status"] == "NA"
                for s in ("Pct", "Dm", "P", "MsePct", "MseP"):
                    add(f"{T}{nm}{s}", "--" if na else PEND, f"{c}: {what}: {'not run' if na else 'pending'}")
        a, sp = agree[c]
        n_ag[a] = n_ag.get(a, 0) + 1
        a_tex = {"PENDING": PEND, "n/a": "--"}.get(a, a)
        add(f"{T}CorrAgree", a_tex, f"{c}: the three corrections within {AGREE_PP} pp?")
        sp_tex = PEND if a == "PENDING" else (fmt(sp, 1) if np.isfinite(sp) else "--")
        add(f"{T}CorrSpread", sp_tex, f"{c}: max - min of the three (pp)")
    add("NCorrAgree", str(n_ag["agree"]), "classes whose corrections agree")
    add("NCorrDisagree", str(n_ag["disagree"]), "classes whose corrections disagree")
    add("NCorrPending", str(n_ag["PENDING"]), "classes with a correction pending")

    # item 2
    for c in CLASSES:
        r = SH[SH.cls == c].iloc[0]
        add(f"{CTAG[c]}ShareEvent", fmt(r["share_event_pct"], 0), f"{c}: % of the every-bar gain on EVENT days")
        add(f"{CTAG[c]}ShareEventPm", fmt(r["share_event_pm_pct"], 0), f"{c}: % of the gain on EVENT days 14:30-16:00")
        for sp in ("event", "ord"):
            f = B2[(B2.cls == c) & (B2.daytype == sp)].set_index("bar_end")["mean_diff"]
            add(f"{CTAG[c]}{STAG[sp]}PeakBar", str(f.idxmin()), f"{c}: {sp}: bar end with the most negative mean diff")
            add(f"{CTAG[c]}{STAG[sp]}PeakDiff", fmt(1e3 * f.min(), 1), f"{c}: {sp}: that mean diff (full - HAR) x 10^3")
    add("EventPmBarPct", fmt(100.0 * (S.mask["event"] & S.pm).mean(), 1), "EVENT 14:30-16:00 bars, % of all bars")
    for grp, cs in (("Lin", LIN), ("Tree", TREES)):
        for col, nm in (("share_event_pct", "ShareEvent"), ("share_event_pm_pct", "ShareEventPm")):
            v = SH[SH.cls.isin(cs)][col]
            add(f"{grp}{nm}Min", fmt(v.min(), 0), f"{grp}: lowest {col}")
            add(f"{grp}{nm}Max", fmt(v.max(), 0), f"{grp}: highest {col}")

    # item 3 (i)
    for b, nm in (("ridge", "LassoRidge"), ("enet", "LassoEnet")):
        for sp in SPANS:
            r = P3[(P3.pair == f"lasso_vs_{b}") & (P3.span == sp)].iloc[0]
            add(f"{nm}{STAG[sp]}Pct", fmt(r["pct_qlike"], 1), f"lasso vs {b}, full sets, {sp}: % QLIKE change")
            add(f"{nm}{STAG[sp]}Dm", fmt(r["dm_qlike"], 2), f"lasso vs {b}, {sp}: DM")
            add(f"{nm}{STAG[sp]}P", fmt_p(r["p_qlike"]), f"lasso vs {b}, {sp}: DM p")

    # item 3 (ii)
    for c in CLASSES:
        T = CTAG[c]
        for sp in SPANS:
            f = A4[(A4.cls == c) & (A4.span == sp)]
            j = f[f.kind == "joint"].iloc[0]
            add(f"{T}{STAG[sp]}JointPct", fmt(j["pct_qlike"], 1), f"{c} {sp}: full vs HAR, % QLIKE change")
            add(f"{T}{STAG[sp]}JointDm", fmt(j["dm_qlike"], 2), f"{c} {sp}: full vs HAR, DM")
            add(f"{T}{STAG[sp]}JointP", fmt_p(j["p_qlike"]), f"{c} {sp}: full vs HAR, DM p")
            for g in GROUPS:
                for kind, nm in (("add", "Add"), ("lo", "Lo")):
                    r = f[(f.kind == kind) & (f.group == g)].iloc[0]
                    stem = f"{T}{STAG[sp]}{nm}{GTAG[g]}"
                    if r["status"] == "ok":
                        lab = " [pre-patch]" if r["design"] == "pre-patch" else ""
                        add(f"{stem}Pct", fmt(r["pct_qlike"], 1), f"{c} {sp} {kind}_{g} vs {r['ref']}: %{lab}")
                        add(f"{stem}P", fmt_p(r["p_qlike"]), f"{c} {sp} {kind}_{g}: DM p{lab}")
                    elif r["status"] == "PENDING":
                        add(f"{stem}Pct", PEND, f"{c} {sp} {kind}_{g}: pending")
                        add(f"{stem}P", PEND, f"{c} {sp} {kind}_{g}: pending")
            if sp != "all":
                rd = RD[(RD.level == "class") & (RD.who == c) & (RD.span == sp)].iloc[0]
                add(f"{T}{STAG[sp]}Reading", rd_tex(rd["reading"]), f"{c} {sp}: item 3 reading")
                ok_ = rd["reading"] not in ("PENDING",)
                add(
                    f"{T}{STAG[sp]}TopAddShare",
                    fmt(100 * rd["top_share"], 0) if ok_ and np.isfinite(rd["top_share"]) else PEND,
                    f"{c} {sp}: largest add-alone gain, % of the joint gain ({rd['top_group']})",
                )
                add(
                    f"{T}{STAG[sp]}SumAddRatio",
                    fmt(rd["ratio_sum"], 2) if np.isfinite(rd["ratio_sum"]) else PEND,
                    f"{c} {sp}: add-alone gains summed over groups / joint gain",
                )
                add(f"{T}{STAG[sp]}NLoSig", str(int(rd["n_lo_sig"])), f"{c} {sp}: significant leave-one-outs")
                n_run = len(GROUPS) - int(rd["n_lo_unknown"])
                add(f"{T}{STAG[sp]}NLoRun", str(n_run), f"{c} {sp}: leave-one-outs available (of 8)")
                for tst, nm in (("conc_test", "ConcTest"), ("red_test", "RedTest")):
                    v = tri(rd[tst])
                    add(
                        f"{T}{STAG[sp]}{nm}",
                        "yes" if v is True else ("no" if v is False else PEND),
                        f"{c} {sp}: the {tst.split('_')[0].upper()} definition on its own (pending: undetermined)",
                    )
                add(
                    f"{T}{STAG[sp]}ReadingRedFirst",
                    rd_tex(rd["reading_redundant_first"]),
                    f"{c} {sp}: reading with REDUNDANT tested before CONCENTRATED (the RULE tests CONCENTRATED first)",
                )
    for sp in ("event", "ord"):
        for t, nm in (("lin", "Lin"), ("tree", "Tree")):
            r = RD[(RD.level == "type") & (RD.who == t) & (RD.span == sp)].iloc[0]
            add(f"{nm}{STAG[sp]}Reading", rd_tex(r["reading"]), f"{t} {sp}: item 3 reading by model type")
            add(
                f"{nm}{STAG[sp]}ReadingRedFirst",
                rd_tex(r["reading_redundant_first"]),
                f"{t} {sp}: by model type, REDUNDANT tested first",
            )
            cs = LIN if t == "lin" else TREES
            sub = RD[(RD.level == "class") & (RD.span == sp) & RD.who.isin(cs)]
            for tst, lab in (("conc_test", "Conc"), ("red_test", "Red")):
                add(
                    f"{nm}{STAG[sp]}N{lab}Test",
                    str(sum(tri(v) is True for v in sub[tst])),
                    f"{t} {sp}: classes meeting the {lab} definition (of {len(cs)})",
                )

    prose_ranges(add, C1, A4, RD)

    # ADDITIONAL, NOT PRE-REGISTERED (coordinator, 2026-10-26): \rhchX...
    for c in CLASSES:
        for sp in SPANS:
            for cmp_, nm in (("full_vs_har_rm", "FullVsRm"), ("har_rm_vs_har", "RmVsHar")):
                r = XD[(XD.cls == c) & (XD.span == sp) & (XD.comparison == cmp_)].iloc[0]
                what = {"full_vs_har_rm": "full vs HAR+RM", "har_rm_vs_har": "HAR+RM vs HAR"}[cmp_]
                stem = f"X{CTAG[c]}{STAG[sp]}{nm}"
                note = f"ADDITIONAL, NOT PRE-REGISTERED: {c} {sp}: {what}"
                add(f"{stem}Pct", fmt(r["pct_qlike"], 1), f"{note}: % QLIKE change")
                add(f"{stem}Dm", fmt(r["dm_qlike"], 2), f"{note}: DM")
                add(f"{stem}P", fmt_p(r["p_qlike"]), f"{note}: DM p")
                sh = r["share_of_joint_gain"]
                add(f"{stem}Share", fmt(100 * sh, 0) if np.isfinite(sh) else "--", f"{note}: % of the joint gain")
    for grp, cs in (("Lin", LIN), ("Tree", TREES)):
        for sp in SPANS:
            f = XD[XD.cls.isin(cs) & (XD.span == sp) & (XD.comparison == "full_vs_har_rm")]
            add(f"X{grp}{STAG[sp]}FullVsRmPctMin", fmt(f.pct_qlike.min(), 1), f"ADDITIONAL: {grp} {sp}: lowest %")
            add(f"X{grp}{STAG[sp]}FullVsRmPctMax", fmt(f.pct_qlike.max(), 1), f"ADDITIONAL: {grp} {sp}: highest %")
            add(
                f"X{grp}{STAG[sp]}FullVsRmNSig",
                str(int(((f.p_qlike < ALPHA) & (f.dm_qlike < 0)).sum())),
                f"ADDITIONAL: {grp} {sp}: classes whose full set beats HAR+RM at 5%",
            )

    # item 3 (iii)
    for c in CLASSES:
        for sp in SPANS:
            f = LD[(LD.cls == c) & (LD.span == sp)].set_index("k")
            add(
                f"{CTAG[c]}{STAG[sp]}LadFullPct",
                fmt(f.loc[46, "gain_full_vs_k0_pct"], 1),
                f"{c} {sp}: ladder k=46 vs 0 %",
            )
            for k in KS:
                v = f.loc[k, "R_pct"]
                add(
                    f"{CTAG[c]}{STAG[sp]}R{KTAG[k]}",
                    fmt(v, 0) if np.isfinite(v) else "--",
                    f"{c} {sp}: ladder R_{k} (%)",
                )

    names = [n for n, _, _ in M]
    assert len(set(names)) == len(names), "a macro name twice"
    lines = [
        HEAD,
        "% \\rhch... macros: the pooled gain by day type (results/channels_2026-10-26/RULE.txt items 1-3):",
        "% FOMC statement days (EVENT) and ordinary days, Table 1's scored bars, the FOMC-patched design of record;",
        "% DM (HLN) on daily average loss, lag 8.  Names in the generator's docstring.",
        "% \\rhchX... = ADDITIONAL, NOT PRE-REGISTERED (coordinator, 2026-10-26, after the first scoring): the full set",
        "% against HAR + the futures' realized moments (FullVsRm) and HAR + realized moments against HAR (RmVsHar).",
    ]
    if any(PEND in v for _, v, _ in M):
        lines.append(fr.PENDING_DEF)
    lines += [f"\\providecommand{{\\{n}}}{{}}\\renewcommand{{\\{n}}}{{{v}}}  % {c}" for n, v, c in M]
    (GEN / "rehaul_channels_numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    say(f"wrote rehaul_channels_numbers.tex: {len(M)} macros")
    table_corr(C1, agree)
    table_abl(A4, RD, n_ev_bars)
    table_sparse(P3, LD, gates)


CONLIQ = ("ew", "vw", "liq")  # the constituents' moments (EW, VW) and liquidity (which holds the order-flow imbalance)
REST = ("ew", "vw", "liq", "sent", "iv", "vd")  # the groups beyond HAR + RM + FOMC
ALT = ("sent", "vd", "iv")  # the alternative data: StockTwits (sent), volatility demand, implied volatility


def prose_ranges(add: Callable[[str, str, str], None], C1: pd.DataFrame, A4: pd.DataFrame, RD: pd.DataFrame) -> None:
    """Ranges and extremes for the prose of draft v2 of Sections 6-7 (writeup/rehaul/drafts/MAP_06_07.md, "Numbers
    still needed"), from the item 1 and item 3 (ii) results above; no new forecast.  % = % QLIKE change; share = an
    add-alone QLIKE gain as % of the joint gain (full vs HAR) on the same day type.  Magnitudes "a--b" where the
    sentence gives the direction (rng_txt); a mixed-sign range is printed signed and flagged in its comment."""

    def av(c: str, sp: str, kind: str, g: str = "") -> pd.Series | None:
        f = A4[(A4.cls == c) & (A4.span == sp) & (A4.kind == kind) & (A4.group == g)]
        return f.iloc[0] if len(f) and f.iloc[0]["status"] == "ok" else None

    def pct(c: str, sp: str, kind: str, g: str = "") -> float:
        r = av(c, sp, kind, g)
        assert r is not None, (c, sp, kind, g)
        return float(r["pct_qlike"])

    def share(c: str, sp: str, g: str) -> float:
        a, j = av(c, sp, "add", g), av(c, sp, "joint")
        assert a is not None and j is not None and j["gain"] > 0, (c, sp, g)
        return 100.0 * float(a["gain"]) / float(j["gain"])

    def prose(name: str, vals: list[float], dp: int, note: str, signed: bool = False, change: bool = True) -> None:
        """change: the values are % changes (the comment says which way the printed magnitudes go)."""
        txt, mixed = rng_txt(vals, dp, signed)
        v = np.asarray(vals, float)
        way = "" if signed or mixed or not change else (": increases" if (v > 0).all() else ": reductions")
        if mixed:
            say(f"NOTE {name}: values of both signs {np.round(v, 2).tolist()}; printed signed")
        add(name, txt, note + way + (" [MIXED SIGNS: printed signed]" if mixed else ""))

    def names(cs: tuple[str, ...] | list[str]) -> str:
        return ", ".join(cs)

    # item 1: the three corrections, by model type
    corr = C1[(C1.role == "rule") & C1["corr"].isin(["a", "b", "c"])]
    for grp, cs in (("Lin", LIN), ("Tree", TREES)):
        f = corr[corr.cls.isin(cs)]
        ok_ = f[f.status == "ok"]
        miss = ", ".join(f"{r['cls']} ({r['corr']})" for _, r in f[f.status != "ok"].iterrows())
        prose(
            f"Corr{grp}Rng",
            ok_["pct_qlike"].tolist(),
            1,
            f"item 1: % QLIKE change of corrections (a) full vs HAR+FOMC, (b) full w/o FOMC vs HAR, (c) full vs HAR on ORDINARY days, "
            f"range over {names(cs)} x (a)-(c){'; not run: ' + miss if miss else ''}",
        )
    # EVENT days: the FOMC group
    for grp, cs in (("Lin", LIN), ("Tree", TREES)):
        prose(
            f"EvFomcShare{grp}Rng",
            [share(c, "event", "fomc") for c in cs],
            0,
            f"EVENT days: (HAR+FOMC vs HAR) gain as % of (full vs HAR) gain, range over {names(cs)}",
            change=False,
        )
    have = [c for c in CLASSES if av(c, "event", "lo", "fomc") is not None]
    prose(
        "EvLoFomcRng",
        [pct(c, "event", "lo", "fomc") for c in have],
        1,
        f"EVENT days: full without FOMC vs full, %, range over the classes with the arm: {names(have)} (not run: "
        f"{names([c for c in CLASSES if c not in have]) or 'none'}); the arm holds no FOMC column, vs the patched full",
    )
    sh_ev, c_ev, g_ev = max((share(c, "event", g), c, g) for c in CLASSES for g in GROUPS if g != "fomc")
    add(
        "EvOtherMaxShare",
        fmt(sh_ev, 0),
        f"EVENT days: largest add-alone gain of a non-FOMC group, % of the joint gain, over 6 classes x 7 groups "
        f"({c_ev}, {GNAME[g_ev]})",
    )
    # ORDINARY days: FOMC added alone
    for grp, cs, sg in (("Lin", LIN, False), ("Tree", TREES, True)):
        prose(
            f"OrdAddFomc{grp}Rng",
            [pct(c, "ord", "add", "fomc") for c in cs],
            1,
            f"ORDINARY days: HAR+FOMC vs HAR, %, range over {names(cs)}" + (" (signed)" if sg else ""),
            signed=sg,
        )
    # ORDINARY days: the realized moments and the other groups, trees
    prose(
        "OrdAddRmTreeRng",
        [pct(c, "ord", "add", "rm") for c in TREES],
        1,
        f"ORDINARY days: HAR+RM vs HAR, %, range over {names(TREES)}",
    )
    prose(
        "OrdRmShareTreeRng",
        [share(c, "ord", "rm") for c in TREES],
        0,
        f"ORDINARY days: (HAR+RM vs HAR) gain as % of (full vs HAR) gain, range over {names(TREES)}",
        change=False,
    )
    prose(
        "OrdAddOtherTreeRng",
        [pct(c, "ord", "add", g) for c in TREES for g in CONLIQ],
        1,
        f"ORDINARY days: HAR+g vs HAR, %, g = Equal-weighted market, Value-weighted market (the constituents' moments) "
        f"and Liquidity (holds the constituents' order-flow imbalance; there is no separate order-flow group), range "
        f"over {names(TREES)} x the 3 groups",
    )
    rs = RD[(RD.level == "class") & (RD.span == "ord")].set_index("who")["ratio_sum"]
    by_type: tuple[tuple[str, tuple[str, ...]], ...] = (("", CLASSES), ("Lin", LIN), ("Tree", TREES))
    for grp, cls_ in by_type:
        cs_ = [c for c in cls_ if np.isfinite(rs[c])]
        prose(
            f"OrdSumRatio{grp}Rng",
            [float(rs[c]) for c in cs_],
            1,
            f"ORDINARY days: add-alone gains summed over the 8 groups / joint gain, range over {names(cs_)}",
            change=False,
        )
    # leave-one-outs: the groups beyond HAR + RM + FOMC (trees, ORDINARY days) and the alternative data (both day types)
    lo_ord: list[tuple[float, str, str, str, float]] = []  # (|%|, class, group, design, p)
    for c in TREES:
        for g in REST:
            r = av(c, "ord", "lo", g)
            if r is not None:
                lo_ord.append((abs(float(r["pct_qlike"])), c, g, str(r["design"]), float(r["p_qlike"])))
    x_ord = max(lo_ord)
    pre = sorted({f"{x[1]} {x[2]}" for x in lo_ord if x[3] == "pre-patch"})
    add(
        "OrdLoOtherTreeMaxAbs",
        fmt(x_ord[0], 1),
        f"ORDINARY days: largest |full w/o g vs full| % over {names(sorted({x[1] for x in lo_ord}, key=CLASSES.index))} "
        f"(the forest's arms not run), g = {names(REST)} (non-RM, non-FOMC) ({x_ord[1]} {x_ord[2]}, p {x_ord[4]:.3f})"
        + (f"; PRE-PATCH arms vs the pre-patch full set: {len(pre)} of {len(lo_ord)}" if pre else ""),
    )
    lo_alt: list[tuple[float, str, str, str, str]] = []  # (|%|, class, day type, group, design)
    for c in CLASSES:
        for sp in ("event", "ord"):
            for g in ALT:
                r = av(c, sp, "lo", g)
                if r is not None:
                    lo_alt.append((abs(float(r["pct_qlike"])), c, sp, g, str(r["design"])))
    x_alt = max(lo_alt)
    n_pre = sum(x[4] == "pre-patch" for x in lo_alt)
    add(
        "AltLoMaxAbs",
        fmt(x_alt[0], 1),
        f"largest |full w/o g vs full| %, g = Sentiment (StockTwits), Volatility demand, Implied volatility, over "
        f"{names(sorted({x[1] for x in lo_alt}, key=CLASSES.index))} and EVENT / ORDINARY days ({x_alt[1]} {x_alt[2]} "
        f"{GNAME[x_alt[3]]}, {x_alt[4]})"
        + (f"; PRE-PATCH arms (the trees'): {n_pre} of {len(lo_alt)}" if n_pre else ""),
    )


def rd_tex(r: str, short: bool = False) -> str:
    if r == "PENDING":
        return PEND
    return READ_SHORT[r] if short else READ_WORD[r]


def cell(r: pd.Series | None, star: bool = False, dagger: bool = False) -> str:
    if r is None:
        return "--"
    if r["status"] != "ok":
        return PEND if r["status"] == "PENDING" else "--"
    s = fmt(r["pct_qlike"], 1)
    sup = ("*" if star and r["p_qlike"] < ALPHA else "") + ("\\dagger" if dagger and r["design"] == "pre-patch" else "")
    return s + (f"$^{{{sup}}}$" if sup else "")


def with_def(body: str) -> str:
    """The numbers file's \\ifdefined guard; PENDING_DEF when the float prints \\rhPending."""
    lines = [HEAD, "\\ifdefined\\rhchNBars\\else\\input{generated/rehaul_channels_numbers}\\fi"]
    if PEND in body:
        lines.append(fr.PENDING_DEF)
    return "\n".join(lines) + "\n" + body


def table_corr(C1: pd.DataFrame, agree: dict) -> None:
    rows = []
    for c in CLASSES:
        cells = []
        for corr in ("t1", "a", "b", "c"):
            r = C1[(C1.cls == c) & (C1["corr"] == corr) & (C1.role == "rule")].iloc[0]
            if r["status"] == "ok":
                cells.append(f"{fmt(r['pct_qlike'], 1)} ({fmt(r['dm_qlike'], 2)})")
            else:
                cells.append("--" if r["status"] == "NA" else PEND)
        a, sp = agree[c]
        # the spread of the known two when it decides
        cells.append(PEND if a == "PENDING" else (fmt(sp, 1) if np.isfinite(sp) else "--"))
        cells.append({"PENDING": PEND, "n/a": "--", "agree": "yes", "disagree": "no"}[a])
        rows.append(f"{CNAME[c]} & " + " & ".join(cells) + " \\\\")
        if c == "enet":
            rows.append("\\addlinespace[3pt]")
    na_note = " --: not run." if (C1.status == "NA").any() else ""
    # 2026-10-28 (author: prune "pooled"). Was (notes): "Benchmarks as in Table~\ref{{tab:rh_oos}}: the pooled OLS"
    body = rf"""\begin{{table}}[htbp]
\centering
\footnotesize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{threeparttable}}
\caption{{What the other predictors add beyond the FOMC calendar: three corrections of the full set's gain (\%
  change in QLIKE, DM statistic in parentheses)}}
\label{{tab:rh_channels_corr}}
\begin{{tabular*}}{{\linewidth}}{{@{{\extracolsep{{\fill}}}}lrrrrrc@{{}}}}
\toprule
 & Table~\ref{{tab:rh_oos}} & (a) & (b) & (c) & Spread & Agree \\
 & Full vs HAR & Full vs HAR + FOMC & Full without FOMC & Full vs HAR, & (pp) & \\
 & every bar & every bar & vs HAR, every bar & ordinary days & & \\
\cmidrule(lr){{2-5}}
 & \multicolumn{{4}}{{c}}{{\textit{{$-$: the full set (in (b), without FOMC) is more accurate; $+$: less accurate}}}} & & \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular*}}
\begin{{tablenotes}}[flushleft]
\footnotesize
\item \textit{{Notes:}} \textit{{Sign:}} each cell is $100\,(L/L_{{\mathrm{{ref}}}} - 1)$, the percentage change in mean
QLIKE of the column's first forecast (the full set; in (b), the full set without the FOMC group) against its reference,
the forecast after ``vs''; the Diebold--Mariano statistic of the difference, in parentheses, has the same sign. Negative
= the full set is more accurate than its reference; positive = less accurate. (a)--(c) set the FOMC calendar aside, so
there negative = the other predictors add accuracy beyond it; positive = they hurt. HLN correction, daily average loss,
Newey--West lag \rhchHacLag. Benchmarks as in Table~\ref{{tab:rh_oos}}: the OLS
on the HAR set for ridge, lasso and elastic net, each tree's own HAR-set fit. (a) sets the full set against the HAR
benchmark set plus the FOMC group, (b) the full set without the FOMC group against the HAR benchmark, both on every bar
of Table~\ref{{tab:rh_oos}}; (c) sets the full set against the HAR benchmark on the \rhchNOrdDays{{}} days that are not
scheduled FOMC statement days. Spread: the largest minus the smallest of (a)--(c), in percentage points; the corrections
agree when the spread is at most \rhchAgreePp{{}} points. The arm in (b) holds no FOMC column, so it is the same forecast
on the patched and the unpatched design; every forecast here is on the patched design.{na_note}
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table}}
"""
    (GEN / "rehaul_channels_corrections_table.tex").write_text(with_def(body), encoding="utf-8", newline="\n")


def table_abl(A4: pd.DataFrame, RD: pd.DataFrame, n_ev_bars: int) -> None:
    """tab:rh_channels_abl (Section 7) and tab:rh_channels_abl_full (Appendix app:rh_channels, every row).  The full
    table: * on a significant cell, the dagger on each pre-patch cell, the add-alone sum, the significant leave-outs and
    the readings by class.  The main table (author, 2026-10-28): a significant cell in bold; each class's top add-alone
    group (the readings' top_group) shaded in both blocks; the dagger once on a class header whose removals other than
    FOMC are all pre-patch (a class whose removals mix the designs keeps it on its pre-patch cells); the joint row and
    the readings by model type only."""

    def get(c: str, sp: str, kind: str, g: str = "") -> pd.Series | None:
        f = A4[(A4.cls == c) & (A4.span == sp) & (A4.kind == kind) & (A4.group == g)]
        return f.iloc[0] if len(f) else None

    hdr_dag: dict[str, bool] = {}  # main table: the dagger on the class header
    cell_dag: dict[str, bool] = {}  # main table: the dagger on the class's pre-patch cells (designs mixed)
    for c in CLASSES:
        lo = A4[(A4.cls == c) & (A4.kind == "lo") & (A4.status == "ok")]
        assert not (lo[lo.group == "fomc"]["design"] == "pre-patch").any(), f"{c}: lo_fomc pre-patch"  # the note
        des = set(lo[lo.group != "fomc"]["design"])
        hdr_dag[c] = des == {"pre-patch"}
        cell_dag[c] = "pre-patch" in des and not hdr_dag[c]

    def bold(r: pd.Series | None, c: str = "", shade: bool = False) -> str:
        """Main-table cell: bold when significant at ALPHA (the full table's * criterion), shaded on request."""
        s = cell(r)
        if r is not None and r["status"] == "ok":
            if r["p_qlike"] < ALPHA:
                s = f"\\textbf{{\\boldmath {s}}}"
            if c and cell_dag[c] and r["design"] == "pre-patch":
                s += "$^{\\dagger}$"
        return ("\\cellcolor{black!8}" if shade else "") + s

    out_f: list[str] = []  # the full table
    out_m: list[str] = []  # the main table
    for sp, title in (
        ("event", f"A. FOMC statement days (\\rhchNEventDays{{}} days, {thousands(n_ev_bars)} bars)"),
        ("ord", "B. Ordinary days (\\rhchNOrdDays{} days)"),
    ):
        for out in (out_f, out_m):
            out.append(f"\\multicolumn{{13}}{{@{{}}l}}{{\\textit{{{title}}}}} \\\\")
        rds = RD[(RD.level == "class") & (RD.span == sp)].set_index("who")
        top = {c: str(rds.loc[c, "top_group"]) for c in CLASSES}
        for g in GROUPS:
            cells = [cell(get(c, sp, "add", g), star=True) for c in CLASSES]
            cells += [cell(get(c, sp, "lo", g), star=True, dagger=True) for c in CLASSES]
            out_f.append(f"{GNAME[g]} & " + " & ".join(cells) + " \\\\")
            cells = [bold(get(c, sp, "add", g), shade=top[c] == g) for c in CLASSES]
            cells += [bold(get(c, sp, "lo", g), c, shade=top[c] == g) for c in CLASSES]
            out_m.append(f"{GNAME[g]} & " + " & ".join(cells) + " \\\\")
        for out in (out_f, out_m):
            out.append("\\addlinespace[2pt]")
        cells = [cell(get(c, sp, "joint"), star=True) for c in CLASSES] + [""] * 6
        out_f.append("Full set vs HAR (joint) & " + " & ".join(cells) + " \\\\")
        cells = [bold(get(c, sp, "joint")) for c in CLASSES] + [""] * 6
        out_m.append("Full set vs HAR (joint) & " + " & ".join(cells) + " \\\\")
        cells = [fmt(rds.loc[c, "ratio_sum"], 2) if np.isfinite(rds.loc[c, "ratio_sum"]) else PEND for c in CLASSES] + [
            f"{int(rds.loc[c, 'n_lo_sig'])}/{len(GROUPS) - int(rds.loc[c, 'n_lo_unknown'])}" for c in CLASSES
        ]
        out_f.append("Sum add / joint; sig.\\ leave-outs & " + " & ".join(cells) + " \\\\")
        cells = [rd_tex(rds.loc[c, "reading"], short=True) for c in CLASSES] + [""] * 6
        out_f.append("Reading, by class & " + " & ".join(cells) + " \\\\")
        rt = RD[(RD.level == "type") & (RD.span == sp)].set_index("who")["reading"]
        for out in (out_f, out_m):
            out.append(
                f"Reading, by model type & \\multicolumn{{3}}{{c}}{{{rd_tex(rt['lin'])}}} & "
                f"\\multicolumn{{3}}{{c}}{{{rd_tex(rt['tree'])}}} & & & & & & \\\\"
            )
            if sp == "event":
                out.append("\\addlinespace[4pt]")
    hdr = " & ".join(CSHORT[c] for c in CLASSES)
    hdr_lo = " & ".join(CSHORT[c] + ("$^{\\dagger}$" if hdr_dag[c] else "") for c in CLASSES)
    head = r"""\begin{table}[htbp]
\centering
\scriptsize
\setlength{\tabcolsep}{1.5pt}
\begin{threeparttable}
\caption{Predictor groups by day type: each group added alone to the HAR benchmark and left out of the full set, on
  FOMC statement days and on ordinary days (\% change in QLIKE"""
    cols = r"""\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrrrrrrrrr@{}}
\toprule
 & \multicolumn{6}{c}{HAR benchmark plus the group, vs HAR} & \multicolumn{6}{c}{Full set without the group, vs full set} \\
 & \multicolumn{6}{c}{\textit{$-$: the group adds accuracy; $+$: it hurts}}
 & \multicolumn{6}{c}{\textit{$+$: the group adds accuracy; $-$: it hurts}} \\
\cmidrule(lr){2-7}\cmidrule(l){8-13}
"""
    sign = r"""\item \textit{Notes:} \textit{Sign:} every cell is $100\,(L_{\mathrm{row}}/L_{\mathrm{ref}} - 1)$, the
percentage change in mean QLIKE of the row's forecast against its reference on the bars of the day type; the
Diebold--Mariano statistic has the same sign. Left block, the group added to the HAR benchmark set (reference: the HAR
benchmark): negative = the loss falls when the group is added, i.e.\ the group adds accuracy; positive = the group
hurts. Right block, the group removed from the full set (reference: the full set): positive = the loss rises when the
group is dropped, i.e.\ the group adds accuracy; negative = the group hurts. Joint: the full set against the HAR
benchmark on the same days; negative = the full set is more accurate."""
    # 2026-10-28 (author: prune "pooled"). Was: "HAR benchmark: the pooled OLS on the HAR set for"
    test = r"""the Diebold--Mariano test rejects equal accuracy
at 5\% (two-sided; daily average loss, Newey--West lag \rhchHacLag). HAR benchmark: the OLS on the HAR set for
the linear models, each tree's own HAR-set fit."""
    # 2026-10-28: the forest's FOMC removal was stopped (ablation addendum E (i)), so it is not "run"; was: "for leaving
    # out the realized moments and the FOMC group only" in every case
    rf_lo = "the realized moments only" if RF_LOFOMC_DROPPED else "the realized moments and the FOMC group only"
    tail = (
        r"""Arms without a FOMC column are the same forecasts on both designs. The forest is run for the
added groups and for leaving out """
        + rf_lo
        + r""" (--: not run).
\end{tablenotes}"""
    )
    tail += r"""
\end{threeparttable}
\end{table}
"""
    full = (
        head
        + r"""; full detail)}
\label{tab:rh_channels_abl_full}
"""
        + cols
        + f" & {hdr} & {hdr} \\\\\n\\midrule\n{chr(10).join(out_f)}\n"
        + r"""\bottomrule
\end{tabular*}
\begin{tablenotes}[flushleft]
\footnotesize
"""
        + sign
        + " * "
        + test
        + r""" Sum add / joint: the add-alone QLIKE gains (falls in the loss; a group
that hurts counts negative) summed over the eight groups, divided by the joint gain; sig.\ leave-outs: groups whose removal raises the loss significantly, of the
leave-outs run. Readings (fixed before scoring, tested in this order): concentrated (conc.)\ if one group's add-alone
gain is at least 75\% of the joint gain and its leave-out is significant; redundant (redund.)\ if the add-alone gains
sum to more than 1.5 times the joint gain and at most one leave-out is significant; otherwise mixed; by model type the
majority of ridge, lasso and elastic net, and of LightGBM and XGBoost with the forest where its arms exist. $^{\dagger}$ Unpatched design (the FOMC proximity
series before the fix), compared with the unpatched full set: the trees' leave-outs other than FOMC were not rerun on
the patched design. """
        + tail
    )
    dag = []
    if any(hdr_dag.values()):
        dag.append("On a class: its removals other than the FOMC group are")
    if any(cell_dag.values()):
        dag.append("on a cell: that removal is" if dag else "On a cell: that removal is")
    dag_note = (
        "$^{\\dagger}$ "
        + "; ".join(dag)
        + " on the unpatched design (the FOMC proximity series before the fix), compared with the unpatched full set;"
        + " they were not rerun on the patched design. "
        if dag
        else ""
    )
    main = (
        head
        + r""")}
\label{tab:rh_channels_abl}
"""
        + cols
        + f" & {hdr} & {hdr_lo} \\\\\n\\midrule\n{chr(10).join(out_m)}\n"
        + r"""\bottomrule
\end{tabular*}
\begin{tablenotes}[flushleft]
\footnotesize
"""
        + sign
        + " Bold: "
        + test
        + r""" Shaded: in each class's column, the group with the largest add-alone gain on the day type, in
both blocks. Readings by model type (fixed before scoring, tested in this order): concentrated if one group's add-alone
gain is at least 75\% of the joint gain and its removal raises the loss significantly; redundant if the add-alone gains
sum to more than 1.5 times the joint gain and at most one removal raises the loss significantly; otherwise mixed; the
majority of ridge, lasso and elastic net, and of LightGBM and XGBoost with the forest where its arms exist.
Table~\ref{tab:rh_channels_abl_full} (Appendix~\ref{app:rh_channels}) adds the per-class readings, the add-alone gains
summed over the groups against the joint gain and the counts of significant leave-outs. """
        + dag_note
        + tail
    )
    (GEN / "rehaul_channels_ablation_table.tex").write_text(with_def(main), encoding="utf-8", newline="\n")
    (GEN / "rehaul_channels_ablation_full_table.tex").write_text(with_def(full), encoding="utf-8", newline="\n")


def table_sparse(P3: pd.DataFrame, LD: pd.DataFrame, gates: dict) -> None:
    ra = []
    for b, nm in (("ridge", "Lasso vs ridge"), ("enet", "Lasso vs elastic net")):
        cells = []
        for sp in SPANS:
            r = P3[(P3.pair == f"lasso_vs_{b}") & (P3.span == sp)].iloc[0]
            cells.append(f"{fmt(r['pct_qlike'], 1)} ({fmt(r['dm_qlike'], 2)})")
        ra.append(f"{nm} & " + " & ".join(cells) + " \\\\")
    rb = []
    for c in CLASSES:
        cells = []
        for sp in ("event", "ord"):
            f = LD[(LD.cls == c) & (LD.span == sp)].set_index("k")
            cells.append(fmt(f.loc[46, "gain_full_vs_k0_pct"], 1))
            cells += [fmt(f.loc[k, "R_pct"], 0) if np.isfinite(f.loc[k, "R_pct"]) else "--" for k in KS]
        rb.append(f"{CNAME[c]} & " + " & ".join(cells) + " \\\\")
        if c == "enet":
            rb.append("\\addlinespace[2pt]")
    lab = (
        "the FOMC-patched design"
        if gates["ladder_design"] == "patched"
        else "the unpatched design (the FOMC proximity series before the fix)"
    )
    kh = " & ".join(f"$k{{=}}{k}$" for k in KS)
    body = rf"""\begin{{table}}[htbp]
\centering
\footnotesize
\setlength{{\tabcolsep}}{{2pt}}
\caption{{Concentrated or spread, by day type: the lasso against the dense penalties, and the share of the full set's
  gain kept by the top $k$ series}}
\label{{tab:rh_channels_sparse}}
\begin{{tabular*}}{{\linewidth}}{{@{{\extracolsep{{\fill}}}}lrrr@{{}}}}
\toprule
\multicolumn{{4}}{{@{{}}l}}{{\textit{{A. Full sets: \% change in QLIKE (DM statistic); $-$: the lasso is more accurate, $+$: less accurate}}}} \\
 & Every bar & FOMC statement days & Ordinary days \\
\midrule
{chr(10).join(ra)}
\bottomrule
\end{{tabular*}}

\medskip
\begin{{tabular*}}{{\linewidth}}{{@{{\extracolsep{{\fill}}}}lrrrrrrrrrrrr@{{}}}}
\toprule
\multicolumn{{13}}{{@{{}}l}}{{\textit{{B. Feature-count ladder: retained gain $R_k$ (\%) by day type}}}} \\
\multicolumn{{13}}{{@{{}}l}}{{\textit{{Full: $-$ = the full set beats HAR. $R_k$: 100 = all of that gain kept, 0 = none, below 0 = worse than HAR}}}} \\
 & \multicolumn{{6}}{{c}}{{FOMC statement days}} & \multicolumn{{6}}{{c}}{{Ordinary days}} \\
\cmidrule(lr){{2-7}}\cmidrule(l){{8-13}}
 & Full (\%) & {kh} & Full (\%) & {kh} \\
\midrule
{chr(10).join(rb)}
\bottomrule
\end{{tabular*}}

\smallskip
\begin{{minipage}}{{\linewidth}}
\footnotesize
\textit{{Notes:}} \textit{{Sign:}} a \% change is $100\,(L_{{\mathrm{{row}}}}/L_{{\mathrm{{ref}}}} - 1)$, the percentage
change in mean QLIKE of the row's forecast against its reference, and Panel A's Diebold--Mariano statistic has the same
sign. Panel A (reference: the ridge's or the elastic net's full set): negative = the lasso is more accurate. Panel B,
Full (the full set against the HAR benchmark set): negative = the full set is more accurate; $R_k$: 100 = the top $k$
series keep all of the full set's gain over the HAR benchmark set, 0 = none of it, below 0 = worse than the HAR
benchmark set, above 100 = better than the full set; --: no gain to keep. Panel A: the lasso's full-set forecast against
the ridge's and the elastic net's, percentage change in QLIKE with the Diebold--Mariano statistic in parentheses (HLN
correction, daily average loss, Newey--West lag \rhchHacLag). Panel B: each model refitted monthly on its $k$
highest-ranked exogenous series (ranking from a pre-sample fit; the feature-count ladder of
Section~\ref{{sec:rh_fi}}), {lab}. Full: the $k = 46$ rung (the full set) against the $k = 0$ rung (the HAR benchmark
set), \% change in QLIKE on the day type; $R_k = (L_0 - L_k) / (L_0 - L_{{46}})$, the share of that gain the top $k$
series keep.
\end{{minipage}}
\end{{table}}
"""
    (GEN / "rehaul_channels_sparsity_table.tex").write_text(with_def(body), encoding="utf-8", newline="\n")


# ------------------------------------------------------------------------------------------------ figure
def figure(D2: pd.DataFrame, B2: pd.DataFrame, bars: list[str], gates: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.edgecolor": "#52514e",
            "xtick.color": "#52514e",
            "ytick.color": "#52514e",
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
        }
    )
    INK, GRIDC = "#0b0b0b", "#e6e5e0"
    fig = plt.figure(figsize=(7.2, 5.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0])
    ax = fig.add_subplot(gs[0, :])
    dates = pd.DatetimeIndex(D2["date"])
    ev = D2["event"].to_numpy(bool)
    ends: list[tuple[float, str]] = []
    lo_y = 0.0
    for c in ("enet", "lgbm"):
        y = D2[f"cum_{c}"].to_numpy()
        yo = D2[f"cum_ord_{c}"].to_numpy()
        ax.plot(dates, y, color=COLOR[c], lw=1.4, zorder=3)
        ax.plot(dates, yo, color=COLOR[c], lw=1.0, ls=(0, (3, 2)), zorder=2)
        ends += [(float(y[-1]), f"{CNAME[c]}, every day"), (float(yo[-1]), f"{CNAME[c]}, ordinary days")]
        lo_y = min(lo_y, float(y.min()))
    ends.sort()
    span = abs(lo_y)
    pos = [e[0] for e in ends]
    for i in range(1, len(pos)):  # keep the end labels apart
        pos[i] = max(pos[i], pos[i - 1] + 0.05 * span)
    for (_, lab), yv in zip(ends, pos):
        ax.text(dates[-1] + pd.Timedelta(days=45), yv, lab, va="center", fontsize=6.5, color=INK)
    # the statement days: a rug under the curves
    rug = lo_y - 0.06 * span
    ax.plot(dates[ev], np.full(int(ev.sum()), rug), ls="none", marker="|", ms=6, mew=0.8, color="#52514e", zorder=2)
    ax.text(dates[0], rug - 0.045 * span, "FOMC statement days", fontsize=6.5, color="#52514e", va="top")
    ax.axhline(0, color="#52514e", lw=0.6)
    ax.set_ylim(rug - 0.1 * span, 0.04 * span)
    ax.set_xlim(dates[0] - pd.Timedelta(days=30), dates[-1] + pd.Timedelta(days=640))
    ax.set_xticks([pd.Timestamp(f"{y}-01-01") for y in range(2014, 2025, 2)])
    ax.set_xticklabels([str(y) for y in range(2014, 2025, 2)])
    ax.set_ylabel("Cumulative daily QLIKE difference\n(full set minus HAR benchmark)")
    ax.set_title(
        "a. Cumulative difference; solid: every day, dashed: the statement days left out",
        fontsize=7.5,
        loc="left",
        color=INK,
    )
    ax.grid(axis="y", color=GRIDC, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    x = np.arange(len(bars))
    for j, (sp, title) in enumerate(
        (
            ("event", "b. FOMC statement days: mean difference by bar"),
            ("ord", "c. Ordinary days: mean difference by bar"),
        )
    ):
        a = fig.add_subplot(gs[1, j])
        for c in CLASSES:
            f = B2[(B2.cls == c) & (B2.daytype == sp)].set_index("bar_end").reindex(bars)["mean_diff"].to_numpy()
            a.plot(x, f, color=COLOR[c], lw=1.2, marker=MARK[c], ms=2.8, label=CNAME[c], zorder=3)
        a.axhline(0, color="#52514e", lw=0.6)
        a.set_xticks(x[::2])
        a.set_xticklabels(bars[::2], fontsize=6.5)
        a.set_xlabel("Bar (end, ET)")
        a.set_title(title, fontsize=7.5, loc="left", color=INK)
        a.grid(axis="y", color=GRIDC, lw=0.6)
        a.spines[["top", "right"]].set_visible(False)
        if j == 0:
            a.set_ylabel("Mean QLIKE difference\n(full set minus HAR benchmark)")
            a.legend(fontsize=6, frameon=False, ncol=2, loc="lower left")
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG)
    plt.close(fig)
    caption = (
        "Where the full set's gain over the HAR benchmark falls. (a) Cumulative sum through time of the daily QLIKE "
        "difference, full set minus HAR benchmark (the day's mean over its 13 bars; falling: the full set has the lower "
        "loss), for the elastic net and LightGBM; solid: every day; dashed: the same sum with the \\rhchNEventDays{} "
        "scheduled FOMC statement days (ticks under the curves) left out. (b), (c) Mean difference by bar of the "
        "day (end stamps, ET) on the statement days and on the \\rhchNOrdDays{} ordinary days, every class; note the "
        "different scales. Benchmarks as in Table~\\ref{tab:rh_oos}; FOMC-patched design, \\rhchNBars{} bars."
    )
    (GEN / "rehaul_channels_fig.tex").write_text(
        "\n".join(
            [
                HEAD,
                "\\ifdefined\\rhchNBars\\else\\input{generated/rehaul_channels_numbers}\\fi",
                "\\begin{figure}[tbp]",
                "\\centering",
                "\\includegraphics[width=\\linewidth]{figures/rehaul_channels_cum.pdf}",
                f"\\caption{{{caption}}}",
                "\\label{fig:rh_channels}",
                "\\end{figure}",
            ]
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    say(f"wrote {FIG.name}, rehaul_channels_fig.tex")


# ------------------------------------------------------------------------------------------------ SUMMARY.md
def summary(C1, agree, SH, B2, P3, A4, RD, RDl, RDp, LD, gates, prov, n_ev_bars, XD, AVAIL) -> None:  # noqa: ANN001
    L: list[str] = [
        "# Channels of the pooled gain: FOMC statement days and ordinary days (RULE.txt items 1-3)",
        "",
        "AUTO-GENERATED by writeup/make_rehaul_channels.py -- do not edit; re-run the script.  Section 4 of the RULE "
        "(factor-augmented HAR) is another run's.",
        "",
        f"Scored set: Table 1's {N_BARS:,} bars on {N_DAYS:,} days (13 a day); EVENT = {N_EVENT} scheduled FOMC "
        f"statement days ({n_ev_bars:,} bars, {100 * n_ev_bars / N_BARS:.1f} % of bars), ORDINARY = "
        f"{N_DAYS - N_EVENT:,} days.  QLIKE on the bar's RV, plain = pred_adj^2 x B; DM (HLN) on daily mean loss, "
        "lag 8, two-sided.  % = 100 (L_a / L_b - 1), negative: a better.",
        "",
        "## Inputs and provenance",
        "",
        "| input | source | design | status | used for |",
        "|---|---|---|---|---|",
    ]
    for p in prov:
        L.append(
            f"| {p['input']} | {p['source'] or '-'} {('`' + p['path'] + '`') if p['path'] else ''} | "
            f"{p['design'] or '-'} | {p['status']} | {p['used_for']} |"
        )
    L += [
        "",
        f"Arms pending: {', '.join(gates['arms_pending']) or 'none'}.  Arms not run (by the ablation RULE): "
        f"{', '.join(gates['arms_not_run']) or 'none'}.",
        "",
        "## Gates (all PASS, asserted before any day split)",
        "",
        f"- scored set {gates['n_bars']:,} bars, {gates['n_days']:,} days x 13; EVENT days {gates['n_event_days']} "
        f"(statement stamps {', '.join(gates['statement_stamps'])})",
        "- Table 1: every-bar QLIKE and MSE of its 13 rows (1e-12), DM full vs own HAR and vs OLS-HAR, six classes "
        "(1e-9): gate_table1.csv",
        f"- every arm's every-bar QLIKE = the ablation scorer's ({gates['arms_checked_against_ablation_scorer']} arms)",
        f"- linear_wide_fix: {gates.get('linear_fix_invariant_equal', '-')} FOMC-free arms bit-identical to the "
        f"pre-patch walks, {gates.get('linear_fix_fomc_changed', '-')} FOMC arms changed",
        "- the trees' HAR fits equal across the patched and pre-patch runs; B equal across every input; RV = rv_actual",
        f"- ladder ({gates['ladder_design']}, {gates['ladder_source']}): every rung's every-bar QLIKE = its scores.csv",
        "",
        "## Item 1: what the other predictors add beyond FOMC",
        "",
        "(a) full vs HAR+FOMC, every bar; (b) full without FOMC vs HAR, every bar; (c) full vs HAR, ORDINARY days.  "
        "% QLIKE change (DM, p).  Agree = the three within 2 pp.",
        "",
        "| class | Table 1 every bar | (a) | (b) | (c) | spread pp | reading |",
        "|---|---|---|---|---|---|---|",
    ]

    def c1(c: str, corr: str, role: str = "rule") -> str:
        f = C1[(C1.cls == c) & (C1["corr"] == corr) & (C1.role == role)]
        if not len(f) or f.iloc[0]["status"] != "ok":
            return "not run" if len(f) and f.iloc[0]["status"] == "NA" else "PENDING"
        r = f.iloc[0]
        return f"{r['pct_qlike']:+.2f} ({r['dm_qlike']:+.2f}, p {ptxt(r['p_qlike'])})"

    for c in CLASSES:
        a, sp = agree[c]
        L.append(
            f"| {CNAME[c]} | {c1(c, 't1')} | {c1(c, 'a')} | {c1(c, 'b')} | {c1(c, 'c')} | "
            f"{'' if not np.isfinite(sp) else f'{sp:.2f}'} | {a.upper()} |"
        )
    info = C1[C1.role == "info_prepatch"]
    if len(info):
        L += [
            "",
            "Information only (not the RULE's input): (a) on the PRE-PATCH design (pre-patch full vs pre-patch "
            "add_fomc), for the classes whose patched add_fomc is pending:",
            "",
        ]
        for _, r in info.iterrows():
            L.append(f"- {CNAME[r['cls']]}: {r['pct_qlike']:+.2f} % (DM {r['dm_qlike']:+.2f}, p {ptxt(r['p_qlike'])})")
    L += [
        "",
        "MSE (same comparisons, % change, p): "
        + "; ".join(
            f"{CNAME[c]} "
            + ", ".join(
                f"({k}) {r['pct_mse']:+.1f} (p {ptxt(r['p_mse'])})"
                for k in ("a", "b", "c")
                for _, r in C1[(C1.cls == c) & (C1["corr"] == k) & (C1.role == "rule") & (C1.status == "ok")].iterrows()
            )
            for c in CLASSES
        ),
    ]
    L += [
        "",
        "## Item 2: where the gain comes from (descriptive)",
        "",
        "Share of each class's every-bar gain (sum over bars of HAR loss minus full loss) on EVENT days and on the "
        f"EVENT bars ending 14:30-16:00 (EVENT days are {100 * N_EVENT / N_DAYS:.1f} % of days; their 14:30-16:00 "
        f"bars {float(SH['event_pm_bars_pct'].iloc[0]):.2f} % of bars).",
        "",
        "| class | total gain (sum) | EVENT share % | EVENT 14:30-16:00 share % | ORDINARY share % |",
        "|---|---|---|---|---|",
    ]
    for _, r in SH.iterrows():
        L.append(
            f"| {CNAME[r['cls']]} | {r['total_gain_sum']:.2f} | {r['share_event_pct']:.1f} | "
            f"{r['share_event_pm_pct']:.1f} | {r['share_ord_pct']:.1f} |"
        )
    L += [
        "",
        "Mean differential (full - HAR, x 1e3) by bar end, EVENT / ORDINARY:",
        "",
        "| class | " + " | ".join(sorted(set(B2.bar_end))) + " |",
        "|---|" + "---|" * BARS_A_DAY,
    ]
    for c in CLASSES:
        for sp in ("event", "ord"):
            f = B2[(B2.cls == c) & (B2.daytype == sp)].sort_values("bar_end")
            L.append(f"| {CNAME[c]} {sp} | " + " | ".join(f"{1e3 * v:+.1f}" for v in f["mean_diff"]) + " |")
    L += [
        "",
        "Figure: writeup/figures/rehaul_channels_cum.pdf (daily series: item2_daily_diff.csv).",
        "",
        "## Item 3 (i): lasso vs ridge and elastic net, full sets",
        "",
        "| pair | every bar | EVENT | ORDINARY |",
        "|---|---|---|---|",
    ]
    for b in ("ridge", "enet"):
        cells = []
        for sp in SPANS:
            r = P3[(P3.pair == f"lasso_vs_{b}") & (P3.span == sp)].iloc[0]
            cells.append(f"{r['pct_qlike']:+.2f} ({r['dm_qlike']:+.2f}, p {ptxt(r['p_qlike'])})")
        L.append(f"| lasso vs {b} | " + " | ".join(cells) + " |")
    L += [
        "",
        "## Item 3 (ii): groups by day type; readings",
        "",
        "Per class and day type: J = joint gain (HAR loss - full loss); top = the group with the largest add-alone "
        "gain and its share of J; sum = add-alone gains summed / J; sig LO = groups whose leave-one-out raises the "
        "loss significantly (DM > 0, p < 0.05); conc / red = the CONCENTRATED and REDUNDANT definitions tested each "
        "on its own (yes / no / ? undetermined); reading = the RULE's order (CONCENTRATED first); red-first = the "
        "same tests with REDUNDANT first (the two definitions can both hold).",
        "",
        "| day type | class | J (x1e3) | full vs HAR % | top group (share of J) | sum / J | sig LO | conc / red "
        "| reading | red-first |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    def yn(v: object) -> str:
        return "?" if v is None or (isinstance(v, float) and not np.isfinite(v)) else ("yes" if v else "no")

    for sp in ("event", "ord"):
        for c in CLASSES:
            r = RD[(RD.level == "class") & (RD.span == sp) & (RD.who == c)].iloc[0]
            j = A4[(A4.cls == c) & (A4.span == sp) & (A4.kind == "joint")].iloc[0]
            ratio = "pending" if not np.isfinite(r["ratio_sum"]) else f"{r['ratio_sum']:.2f}"
            n_run = len(GROUPS) - int(r["n_lo_unknown"])
            L.append(
                f"| {sp} | {CNAME[c]} | {1e3 * r['J']:.2f} | {j['pct_qlike']:+.2f} (p {ptxt(j['p_qlike'])}) | "
                f"{GNAME.get(r['top_group'], '-')} ({100 * r['top_share']:.0f} %) | {ratio} | "
                f"{r['lo_sig'] or '-'} (of {n_run} run) | {yn(r['conc_test'])} / {yn(r['red_test'])} | "
                f"{r['reading']} | {r['reading_redundant_first']} |"
            )
        rt = RD[(RD.level == "type") & (RD.span == sp)].set_index("who")
        for t, nm in (("lin", "linear"), ("tree", "trees")):
            L.append(
                f"| {sp} | **{nm}** | | | | | | | **{rt.loc[t, 'reading']}** | "
                f"**{rt.loc[t, 'reading_redundant_first']}** |"
            )
    diff = []
    for _, r in RDl[RDl.level == "class"].iterrows():
        r0 = RD[(RD.level == "class") & (RD.span == r["span"]) & (RD.who == r["who"])].iloc[0]
        if r0["reading"] != r["reading"]:
            diff.append(f"{r['who']} {r['span']}: {r0['reading']} -> {r['reading']}")
    L += [
        "",
        "Literal two-sided leave-one-out significance (either sign) instead of 'removal hurts': "
        + ("no reading changes." if not diff else "changes: " + "; ".join(diff)),
    ]
    pv = RDp[(RDp.level == "type") & RDp.span.isin(["event", "ord"])]
    if any(AVAIL.get((c, "add_fomc")) != "ok" for c in TREES):  # only while a tree's patched add_fomc is pending
        L += [
            "",
            "PROVISIONAL (information only; the trees' add_fomc on the PRE-PATCH design while the patched one is "
            "pending; the forest's lo_fomc as available): "
            + "; ".join(f"{r['who']} {r['span']}: {r['reading']}" for _, r in pv.iterrows())
            + ".  By class: "
            + "; ".join(
                f"{r['who']} {r['span']} {r['reading']} (top {r['top_group']} {100 * r['top_share']:.0f} %, sum/J "
                f"{r['ratio_sum']:.2f}, sig LO {r['lo_sig'] or '-'})"
                for _, r in RDp[
                    (RDp.level == "class") & RDp.who.isin(TREES) & RDp.span.isin(["event", "ord"])
                ].iterrows()
            ),
        ]
    L += [
        "",
        "## ADDITIONAL, NOT PRE-REGISTERED: the full set beyond HAR + the futures' realized moments",
        "",
        "Asked by the coordinator after the first scoring (not in RULE.txt; macros \\rhchX...; "
        "itemX_additional_rm_not_preregistered.csv).  Same loss and tests.  'full vs HAR+RM' = the full set against "
        "the add_rm arm (HAR benchmark set plus the realized moments; no FOMC column, the same forecast on both "
        "designs); 'HAR+RM vs HAR' against Table 1's HAR benchmark.  share = the comparison's QLIKE gain as % of the "
        "joint gain (full vs HAR) on the same days.",
        "",
        "| class | day type | full vs HAR+RM % (DM, p) | share of J | HAR+RM vs HAR % (DM, p) | share of J |",
        "|---|---|---|---|---|---|",
    ]
    for c in CLASSES:
        for sp in ("ord", "event", "all"):
            cells = []
            for cmp_ in ("full_vs_har_rm", "har_rm_vs_har"):
                r = XD[(XD.cls == c) & (XD.span == sp) & (XD.comparison == cmp_)].iloc[0]
                cells.append(f"{r['pct_qlike']:+.2f} ({r['dm_qlike']:+.2f}, p {ptxt(r['p_qlike'])})")
                cells.append(f"{100 * r['share_of_joint_gain']:.0f} %")
            L.append(f"| {CNAME[c]} | {sp} | " + " | ".join(cells) + " |")
    L += [
        "",
        "Every (class, day type, group) cell with % and DM, MSE alongside: item3ii_ablation_daytype.csv.",
        "",
        f"## Item 3 (iii): feature-count ladder by day type ({gates['ladder_design']} design, primary ranking)",
        "",
        "| class | day type | full vs k=0 % | " + " | ".join(f"R{k}" for k in KS) + " |",
        "|---|---|---|" + "---|" * 5,
    ]
    for c in CLASSES:
        for sp in ("event", "ord", "all"):
            f = LD[(LD.cls == c) & (LD.span == sp)].set_index("k")
            L.append(
                f"| {CNAME[c]} | {sp} | {f.loc[46, 'gain_full_vs_k0_pct']:+.2f} | "
                + " | ".join("-" if not np.isfinite(f.loc[k, "R_pct"]) else f"{f.loc[k, 'R_pct']:.0f}" for k in KS)
                + " |"
            )
    L += [
        "",
        "## Choices where the RULE is silent",
        "",
        "- Inputs on a branch: an input missing from this worktree is read as a COMMITTED file of its run's branch "
        f"({BRANCH_ABL}, {BRANCH_LAD}) with git cat-file; the source and commit are in the provenance table.  Once the "
        "branches are merged the worktree files are read instead (same bytes).",
        "- Linear ablation arms: every linear arm (incl. lo_<g>, g != fomc, and add_fomc) is on the patched design "
        "(linear_wide_fix, ablation addendum D), so no linear number here is pre-patch.  The trees' lo_<g>, g != fomc, "
        "are pre-patch and set against the pre-patch full set (dagger).",
        "- 'Bars 14:30-16:00' = the bars with END stamps 14:30, 15:00, 15:30, 16:00 (the paper's bar labels); the "
        "statement (14:00) falls in the 14:30 bar.",
        "- Gain shares (item 2) are shares of the sum over bars of (HAR loss - full loss); every day has 13 bars, so "
        "the sum of daily means is proportional to it.  Panel b of the figure is drawn as two axes (b: EVENT, c: "
        "ORDINARY) because the scales differ by an order of magnitude; panel a adds a dashed line with the EVENT days "
        "left out.",
        "- DM on a day type: daily means over that day type's days only (81 or 2,498), HAC lag fixed at 8 for both.",
        "- 'Leave-one-out is significant' = the full set without g has the significantly HIGHER loss (DM > 0, "
        "two-sided p < 0.05); a significant improvement from dropping g does not count (the literal two-sided "
        "variant is checked above).",
        "- Item 3 (iii) uses the ladder's PRIMARY (pre-sample) ranking and its own k = 0 rung; R_k is not defined "
        "(-) when the full rung does not beat the k = 0 rung on that day type.",
        "- Readings: tests in the RULE's order (CONCENTRATED, then REDUNDANT, then MIXED); a class whose reading "
        "depends on a pending arm is PENDING; one that depends on an arm not run (the forest's lo_<g> other than rm "
        "and fomc) is INCOMPLETE and left out of the model-type vote ('the forest where its arms exist'); if the "
        "joint gain is not positive the reading is NO JOINT GAIN.  Model type: the reading held by a strict majority "
        "of the eligible classes (2 of 3, or both of LightGBM and XGBoost when the forest is out); SPLIT otherwise.",
        "- Item 1 agreement is decided before every correction is in when the two known ones already differ by more "
        "than 2 pp (then 'disagree'); a correction that is not run makes it n/a unless the other two already differ "
        "by more than 2 pp.",
        "",
        "Rerun when an input lands: `python writeup/make_rehaul_channels.py` (285J environment, repository root).  "
        "The patched tree add_fomc (trees_wide_fix.parquet) and the forest's lo_fomc (trees_wide_rflofomc.parquet) "
        "are read as soon as they are in this worktree or committed on the ablations branch.  The forest's lo_fomc was "
        "stopped (ablation addendum E (i)); its cells read 'not run' by default (REHAUL_CHANNELS_RF_LOFOMC_DROPPED=0 "
        "restores the pending cells if it is resumed).  "
        "REHAUL_CHANNELS_NO_BRANCH=1 reads this worktree only.",
        "",
    ]
    (RES / "SUMMARY.md").write_text("\n".join(L), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
