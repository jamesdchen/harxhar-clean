"""Compounding at a fixed fraction of wealth, for the HEADLINE forecast, plain forecast.

2026-10-06 (user decision): the paper has NO headline model and no live-feasible forecasts;
D.18 is re-based on the paper's MODEL SET (results/close_master_table/paper_model_set.csv):
main() compounds every model-set forecast (plain f^2 B positions) at the same fixed
fraction and reports, by group and over the set, how many end above one unit of wealth,
the median and range of terminal wealth, the log-growth, the drawdowns and the ruin-bound
check, beside always short.  Outputs: fixedfrac_modelset.csv, fixedfrac_modelset_summary.csv,
peryear_growth_modelset.csv, gates.csv, wealth_modelset.png, SUMMARY.md,
writeup/generated/appendix_close_compounding.tex.  The headline run is kept as
main_headline_parked() (not called); its description follows.

2026-10-06: recalibration removed, user decision.  The positions are the master table's
on the PLAIN forecast f^2 * B (q = sign(plain - iv_var), gated here against the per-day
frame's ``plain`` column), never the research scorer's (f^2 + s) * B.  The committed-
output gates become reported-only comparisons for the rows that read a forecast (the
headline, the reference: the numbers change by design); always short's rows read no
forecast and are still asserted equal to the committed copy.  This script reports no
intervals.  The text below is the pre-2026-10-06 description, kept for the record.

The last of the diagnostics parked from Section 5.4 (writeup/sections/appendix_running_parked.tex,
"Compounded at a fixed three percent of wealth": the block-diagonal ridge under the deck's
session-bar recalibration, notebook section 15), re-run for the headline under the 16:00-bar
recalibration of the master table.  Definition (the notebook's, unchanged):

    f = 0.03 of current wealth deployed as straddle premium every day, the same on every
    day and for every rule, a chosen round number;  W_T = prod_t (1 + f R'_t) with
    R'_t = q_t R_t the rule's daily return;  g = 252 * mean log(1 + f R'_t) the annualized
    log-growth;  maxDD_frac = the largest fall of wealth from its running peak as a fraction
    of that peak;  worst_day_factor = 1 + f min_t R'_t;  ruin_bound_f = 1 / |min_t R'_t|
    (the sample's worst day, a realized draw);  per-year growth annualizes each calendar
    year from the days it traded.  Every wealth factor is asserted positive.

Input: the master table's per-day frame (results/close_master_table/master_table_daily.parquet):
ret_mid = q R at the midpoint and ret_crossed at the crossed spread for every table-A forecast on
the 866 deck days (the headline's Sharpe is gated against master_table.csv).  The midpoint frame
is the parked exercise's; the crossed frame is added beside it.  Always short is q = -1 every day.

Which forecasts count (I4c, 2026-09-30): fixedfrac_all_forecasts.csv carries every table-A
forecast of the per-day frame, flagged in_rank_set; the counts, range and ranks over "the
forecasts of the master table" use the master table's own rank set (table A without the
rest-of-day check rows, without exact duplicates and without always short: 220 after the 16:00
campaign), so the check rows, now bit-for-bit copies of their per-bar twins, are not counted
twice.

Committed-output gate (I4c): the headline's forecast and positions are unchanged by the 16:00
campaign and the reference and always short are untouched, so fixedfrac_headline.csv,
peryear_growth.csv and the headline's and the reference's rows of fixedfrac_all_forecasts.csv
must equal their committed copies (git show PREV_REV:<path>; PREV_REV = HEAD unless the
environment variable CLOSE_PREV_REV names another revision), cell by cell, to
|new - old| <= 1e-9 max(1, |old|).

Outputs (results/close_compounding/): fixedfrac_headline.csv (the headline, the reference scored
this way, always short, both fills), fixedfrac_all_forecasts.csv (every table-A forecast, both
fills), peryear_growth.csv, wealth_headline.png, gates.csv, SUMMARY.md (from the CSVs);
writeup/generated/appendix_close_compounding.tex (table + number macros).
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "notebooks", ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import atm_straddle_lib as asl  # noqa: E402
import master_table_close as mtc  # noqa: E402
import score_linear_subsection as base  # noqa: E402

OUT = ROOT / "results" / "close_compounding"
TEX = ROOT / "writeup" / "generated" / "appendix_close_compounding.tex"
# 2026-10-15 (number macros): macro-only file for the rehauled paper (paper_rehaul.tex loads it in its preamble; TEX
# above mixes tables and macros, so it is not loaded there). Written by main() and by --numbers-only (from the
# stored CSVs; no re-computation beyond a light consistency check).
NUMBERS_TEX = ROOT / "writeup" / "generated" / "close_compounding_numbers.tex"
# the values 07_economics.tex typed on 2026-10-15; write_numbers() stops if a stored value differs
NUMBERS_TYPED = {
    "cmpWMedMid": "4.4",
    "cmpWMedX": "1.8",
    "cmpDDMaxMid": "64",
    "cmpDDMaxX": "78",
}
DAILY = mtc.OUT / "master_table_daily.parquet"
MASTER = mtc.OUT / "master_table.csv"
HEADLINE = mtc.HEADLINE
REFERENCE = mtc.REFERENCE
F_FIXED = 0.03  # share of wealth deployed as straddle premium, every day (the notebook's constant)
PPY = float(asl.PERIODS_PER_YEAR)
GATE_REL = 1e-9
CHECK_FAMILY = "direct rest-of-day at 15:30 (check)"  # the master table's check rows
# the committed outputs the headline rows are gated against (the headline is unchanged by the campaign)
PREV_REV = os.environ.get("CLOSE_PREV_REV", "HEAD")
COMMITTED_TOL = 1e-9  # |new - old| <= COMMITTED_TOL * max(1, |old|), cell by cell
# 2026-10-06: recalibration removed, user decision -- the forecast column the positions read
FCOL = "plain"
PLAIN_HEADLINE_SHARPE_MID = "1.79"  # the GUI's independent recomputation, 866 days


def rank_set(mt: pd.DataFrame) -> set[str]:
    """The master table's rank set: table A, no check rows, no exact duplicates, no always short."""
    dup = mt["duplicate_of"].fillna("").astype(str)
    keep = (
        (mt["table"] == "A")
        & (mt["family"] != CHECK_FAMILY)
        & (dup == "")
        & (mt.index != mtc.ALWAYS_SHORT)
    )
    return set(mt.index[keep])


def committed_csv(path: Path, **kw) -> tuple[pd.DataFrame, str]:
    """The committed copy of an output CSV (git show PREV_REV:<path>) and the revision's short sha."""
    rel = path.relative_to(ROOT).as_posix()
    sha = subprocess.check_output(
        ["git", "rev-parse", "--short", PREV_REV], cwd=ROOT, text=True
    ).strip()
    raw = subprocess.check_output(["git", "show", f"{PREV_REV}:{rel}"], cwd=ROOT)
    return pd.read_csv(io.BytesIO(raw), **kw), sha


def committed_diff(new: pd.DataFrame, old: pd.DataFrame) -> float:
    """Largest |new - old| / max(1, |old|) over the committed frame's cells (NaN = NaN);
    inf when a committed row or column is missing or a text cell differs."""
    if not (old.index.isin(new.index).all() and old.columns.isin(new.columns).all()):
        return float("inf")
    n = new.loc[old.index, old.columns]
    worst = 0.0
    for c in old.columns:
        a, b = n[c], old[c]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            a_, b_ = a.to_numpy(float), b.to_numpy(float)
            same_nan = np.isnan(a_) & np.isnan(b_)
            d = np.where(same_nan, 0.0, np.abs(a_ - b_) / np.maximum(1.0, np.abs(b_)))
            if np.isnan(d).any():
                return float("inf")
            worst = max(worst, float(d.max(initial=0.0)))
        elif not (a.astype(str) == b.astype(str)).all():
            return float("inf")
    return worst


def wealth_stats(f: float, r: np.ndarray) -> dict[str, float]:
    """The notebook's wealth_stats, line for line."""
    factors = 1.0 + f * np.asarray(r, float)
    assert (factors > 0).all(), "a wealth factor hit zero - ruin"
    w = np.cumprod(factors)
    return {
        "f": f,
        "g_ann": PPY * float(np.mean(np.log(factors))),
        "terminal": float(w[-1]),
        "maxDD_frac": float((w / np.maximum.accumulate(w) - 1.0).min()),
        "worst_day_factor": float(factors.min()),
        "ruin_bound_f": 1.0 / abs(float(np.min(r))),
        "n": len(r),
    }


# --------------------------------------------------------------------------- the paper's model set
# 2026-10-06 (user decision): the paper has NO headline model and no live-feasible forecasts;
# this diagnostic is re-based on the paper's MODEL SET.  The exact list is the macros agent's
# results/close_master_table/paper_model_set.csv (column key).  Without that file the script
# stops, unless MODELSET_PROVISIONAL=1 (development only): then the set is read off the
# master table's families by the rule below.
MODEL_SET_CSV = mtc.OUT / "paper_model_set.csv"
SET_BUCKETS = ("baseline", "all_features", "lean_vol_vix")
# checked before the shorter names that they contain
BUCKET_NAMES = (
    "lean_vol_vix",
    "moments_vol_vix",
    "moments_vix",
    "all_features",
    "baseline",
    "live_feasible",
)
GROUPS = ("paper", "pooled linear", "per-bar linear", "per-bar trees")
GROUP_COLOURS = {
    "paper": "#1b1b1b",
    "pooled linear": "#2a78d6",
    "per-bar linear": "#1baf7a",
    "per-bar trees": "#eb6834",
}
# families the user excluded from the model set (asserted absent)
EXCLUDED_FAMILY_WORDS = ("LSTM", "VIX-only", "implied-vol", "chain-period", "check")
EXCLUDED_BUCKETS = ("live_feasible", "moments_vix", "moments_vol_vix")


def bucket_of(key: str) -> str:
    """The input set named in a master-table key ('' for the paper's own forecasts)."""
    return next((b for b in BUCKET_NAMES if b in key), "")


def group_of(family: str) -> str:
    """The model-set group of a master-table family."""
    if family == "paper":
        return "paper"
    # 2026-10-06: the pre-tuned pooled rows (poolpt_*) may carry their own family name
    if family == "pooled twin" or family.startswith("pooled"):
        return "pooled linear"
    if family == "per-bar linear":
        return "per-bar linear"
    if family.startswith("per-bar tree"):
        return "per-bar trees"
    return "other"


def model_set(mt: pd.DataFrame) -> tuple[list[str], str]:
    """The model set's keys, in master-table order, and where they came from."""
    if MODEL_SET_CSV.exists():
        keys = pd.read_csv(MODEL_SET_CSV)["key"].astype(str).tolist()
        src = MODEL_SET_CSV.relative_to(ROOT).as_posix()
    elif os.environ.get("MODELSET_PROVISIONAL") == "1":
        a = mt[mt["table"] == "A"]
        grp = a["family"].map(group_of)
        bk = pd.Series([bucket_of(k) for k in a.index], index=a.index)
        keep = (grp == "paper") | (grp.isin(GROUPS) & bk.isin(SET_BUCKETS))
        keys = a.index[keep].tolist()
        if any(
            k.startswith("subpt_") for k in keys
        ):  # the pre-tuned per-bar linear tables
            keys = [k for k in keys if not k.startswith("sub_")]
        src = "PROVISIONAL rule on the master table's families (MODELSET_PROVISIONAL=1)"
    else:
        raise SystemExit(
            f"{MODEL_SET_CSV} is missing (MODELSET_PROVISIONAL=1 for a dry run)"
        )
    assert len(keys) == len(set(keys)), "duplicated keys in the model set"
    missing = [k for k in keys if k not in mt.index]
    assert not missing, f"model-set keys not in the master table: {missing}"
    fam = mt.loc[keys, "family"].astype(str)
    bad = [
        k
        for k in keys
        if any(w in fam[k] for w in EXCLUDED_FAMILY_WORDS)
        or bucket_of(k) in EXCLUDED_BUCKETS
        or group_of(fam[k]) == "other"
    ]
    assert not bad, f"excluded forecasts in the model set: {bad}"
    order = {k: i for i, k in enumerate(mt.index)}
    return sorted(keys, key=order.__getitem__), src


# 2026-10-06: the headline macros, parked (no headline model; user decision).  main() writes
# each one that it does not redefine as a visible ?? stub so the paper still compiles.
HEADLINE_MACROS_PARKED = (
    "cmpF",
    "cmpHW",
    "cmpHG",
    "cmpHDD",
    "cmpHWorst",
    "cmpHRuin",
    "cmpHWX",
    "cmpHGX",
    "cmpHDDX",
    "cmpRW",
    "cmpRG",
    "cmpRWX",
    "cmpAW",
    "cmpAG",
    "cmpARuin",
    "cmpAWX",
    "cmpNAll",
    "cmpNPosMid",
    "cmpNPosX",
    "cmpWLoMid",
    "cmpWHiMid",
    "cmpWLoX",
    "cmpWHiX",
    "cmpHRankMid",
    "cmpHRankX",
    "cmpNYears",
    "cmpNPosYears",
    "cmpNegYears",
    "cmpNPosYearsX",
    "cmpNegYearsX",
    "cmpYearsMid",
    "cmpYearsX",
)


def main_headline_parked() -> None:
    """2026-10-06: PARKED (no headline model; user decision) -- the headline run, kept for
    the record and never called; main() below runs the paper's model set."""
    OUT.mkdir(parents=True, exist_ok=True)
    gates: list[dict] = []

    def gate(name: str, value: float, bound: float, n: int) -> None:
        gates.append(
            {
                "gate": name,
                "value": float(value),
                "bound": bound,
                "n": n,
                "ok": bool(value <= bound),
            }
        )

    if not DAILY.exists():
        raise SystemExit(
            f"{DAILY} is missing: run experiments/master_table_close.py first"
        )
    D = pd.read_parquet(DAILY)
    D["day"] = pd.to_datetime(D["day"])
    mt = pd.read_csv(MASTER).set_index("key")
    deck = pd.read_parquet(base.DECK).sort_index()
    deck.index = pd.DatetimeIndex(pd.to_datetime(deck.index)).as_unit("ns")
    days = deck.index
    assert len(days) == 866
    keys = [
        k
        for k in D["key"].unique()
        if (D["key"] == k).sum() == len(days) and k in mt.index
    ]
    frames = {k: D[D["key"] == k].set_index("day").reindex(days) for k in keys}
    in_set = rank_set(mt)
    assert in_set <= set(keys), sorted(in_set - set(keys))
    R = deck["R"].to_numpy(float)
    ex = deck["exit"].to_numpy(float)
    bid = (deck["bid_c"] + deck["bid_p"]).to_numpy(float)
    h = frames[HEADLINE]
    gate(
        "headline Sharpe mid = master_table.csv",
        abs(
            mtc.sharpe(h["ret_mid"].to_numpy(float))
            / float(mt.loc[HEADLINE, "Sharpe_mid"])
            - 1
        ),
        GATE_REL,
        len(days),
    )
    gate(
        "headline Sharpe crossed = master_table.csv",
        abs(
            mtc.sharpe(h["ret_crossed"].to_numpy(float))
            / float(mt.loc[HEADLINE, "Sharpe_crossed"])
            - 1
        ),
        GATE_REL,
        len(days),
    )
    gate(
        "headline ret_mid = q R",
        np.abs(h["ret_mid"].to_numpy(float) - h["q"].to_numpy(float) * R).max(),
        1e-12,
        len(days),
    )
    # 2026-10-06: recalibration removed, user decision -- every position read here is
    # sign(plain - iv_var) on the per-day frame's own columns
    for k in (HEADLINE, REFERENCE):
        fk = frames[k]
        gate(
            f"q = sign({FCOL} - iv_var) ({k})",
            np.abs(
                fk["q"].to_numpy(float)
                - np.where(
                    fk[FCOL].to_numpy(float) > deck["iv_var"].to_numpy(float), 1.0, -1.0
                )
            ).max(),
            0,
            len(days),
        )
    gate(
        f"headline Sharpe mid = {PLAIN_HEADLINE_SHARPE_MID} to two decimals (the GUI's recomputation)",
        float(
            f"{mtc.sharpe(h['ret_mid'].to_numpy(float)):.2f}"
            != PLAIN_HEADLINE_SHARPE_MID
        ),
        0,
        len(days),
    )

    # ---- the three series, both fills
    series = {
        "headline sign(s)": (
            h["ret_mid"].to_numpy(float),
            h["ret_crossed"].to_numpy(float),
        ),
        "reference sign(s) (this scorer)": (
            frames[REFERENCE]["ret_mid"].to_numpy(float),
            frames[REFERENCE]["ret_crossed"].to_numpy(float),
        ),
        "always short": (-R, 1.0 - ex / bid),
    }
    rows = []
    for name, (rm, rc) in series.items():
        for fill, r in (("mid", rm), ("crossed", rc)):
            rows.append({"series": name, "fill": fill, **wealth_stats(F_FIXED, r)})
    ff = pd.DataFrame(rows)
    ff.to_csv(OUT / "fixedfrac_headline.csv", index=False)

    # ---- every table-A forecast
    rows = []
    for k in keys:
        for fill in ("mid", "crossed"):
            rows.append(
                {
                    "key": k,
                    "label": mt.loc[k, "label"],
                    "fill": fill,
                    **wealth_stats(F_FIXED, frames[k][f"ret_{fill}"].to_numpy(float)),
                    "in_rank_set": k in in_set,
                }
            )
    fa = pd.DataFrame(rows)
    fa.to_csv(OUT / "fixedfrac_all_forecasts.csv", index=False)

    # ---- per calendar year (annualized from the days traded that year)
    yr = days.year
    py_rows = []
    for name, (rm, rc) in series.items():
        for fill, r in (("mid", rm), ("crossed", rc)):
            g = pd.Series(np.log1p(F_FIXED * r), index=days)
            gy = PPY * g.groupby(yr).mean()
            for y, v in gy.items():
                py_rows.append(
                    {
                        "series": name,
                        "fill": fill,
                        "year": int(y),
                        "days": int((yr == y).sum()),
                        "g_ann": float(v),
                    }
                )
    py = pd.DataFrame(py_rows)
    py.to_csv(OUT / "peryear_growth.csv", index=False)

    # ---- the committed copies
    # 2026-10-06: recalibration removed, user decision -- the committed copies are the
    # recalibrated run, so the headline's and the reference's rows are compared for the
    # record only (`reported`, SUMMARY.md); always short's rows read no forecast and stay
    # asserted equal.  The old loop, which asserted every row:
    # for fname, idx, keep in (
    #     ("fixedfrac_headline.csv", ["series", "fill"], None),
    #     ("peryear_growth.csv", ["series", "fill", "year"], None),
    #     ("fixedfrac_all_forecasts.csv", ["key", "fill"], [HEADLINE, REFERENCE]),
    # ):
    #     ... gate(f"{fname} = the committed copy ...", committed_diff(new, old), COMMITTED_TOL, ...)
    reported: list[dict] = []
    committed: dict[str, pd.DataFrame] = {}
    for fname, idx, col, keep, asserted in (
        (
            "fixedfrac_headline.csv",
            ["series", "fill"],
            "series",
            ["always short"],
            True,
        ),
        (
            "peryear_growth.csv",
            ["series", "fill", "year"],
            "series",
            ["always short"],
            True,
        ),
        (
            "fixedfrac_headline.csv",
            ["series", "fill"],
            "series",
            ["headline sign(s)", "reference sign(s) (this scorer)"],
            False,
        ),
        (
            "peryear_growth.csv",
            ["series", "fill", "year"],
            "series",
            ["headline sign(s)", "reference sign(s) (this scorer)"],
            False,
        ),
        (
            "fixedfrac_all_forecasts.csv",
            ["key", "fill"],
            "key",
            [HEADLINE, REFERENCE],
            False,
        ),
    ):
        old, sha = committed_csv(OUT / fname)
        committed[fname] = old
        new = pd.read_csv(OUT / fname)
        old, new = old[old[col].isin(keep)], new[new[col].isin(keep)]
        old, new = old.set_index(idx), new.set_index(idx)
        if asserted:
            gate(
                f"{fname} = the committed copy ({PREV_REV} = {sha}), always short's rows "
                "(no forecast), max |new - old| / max(1, |old|)",
                committed_diff(new, old),
                COMMITTED_TOL,
                int(old.size),
            )
        else:
            reported.append(
                {
                    "file": fname,
                    "rows": ", ".join(keep),
                    "max_rel_diff": committed_diff(new, old),
                    "n_cells": int(old.size),
                }
            )
    pd.DataFrame(gates).to_csv(OUT / "gates.csv", index=False)
    assert all(g["ok"] for g in gates), [g for g in gates if not g["ok"]]

    # ---- figure: wealth paths, midpoint and crossed
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (fi, fill) in zip(axes, enumerate(("mid", "crossed"))):
        for name, c in (
            ("always short", "0.5"),
            ("reference sign(s) (this scorer)", "C1"),
            ("headline sign(s)", "C0"),
        ):
            r = series[name][fi]
            w = np.cumprod(1.0 + F_FIXED * r)
            ax.plot(
                days,
                w,
                lw=1.1 if name != "headline sign(s)" else 1.6,
                color=c,
                label=name,
            )
            ax.annotate(
                f"{w[-1]:.1f}×",
                (days[-1], w[-1]),
                xytext=(4, 0),
                textcoords="offset points",
                fontsize=8,
                va="center",
                color=c,
            )
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(FixedLocator([0.25, 0.5, 1, 2, 5, 10, 20, 50, 100]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.axhline(1.0, color="k", lw=0.6, ls="--")
        ax.grid(axis="y", which="major", alpha=0.3)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.set_title(
            f"{fill} fill: wealth at a fixed {F_FIXED:.0%} of wealth per day",
            fontsize=9,
        )
    axes[0].set_ylabel("wealth multiple (log scale; 1× = starting stake)")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "wealth_headline.png", dpi=130, bbox_inches="tight")

    # ---- tex: table + macros
    def row(name: str, fill: str) -> pd.Series:
        return ff[(ff["series"] == name) & (ff["fill"] == fill)].iloc[0]

    L = [
        "% AUTO-GENERATED by experiments/close_compounding_headline.py from results/close_compounding/*.csv -- do not edit.",
        "% 2026-10-06: plain 16:00 forecast f^2 B (recalibration removed, user decision).",
        r"\begingroup\footnotesize\setlength{\tabcolsep}{3pt}",
        r"\begin{longtable}{llrrrrr}",
        # 2026-10-06: recalibration removed, user decision (was "under the 16:00-bar recalibration")
        r"\caption{Compounding at a fixed $f=0.03$ of wealth per day on the plain forecast $\hat y^2B$ (no recalibration), 866 days: terminal wealth $W_T=\prod_t(1+f\,q_tR_t)$, "
        r"annualized log-growth $g$, the largest fall of wealth from its running peak, the worst single-day wealth factor $1+f\min_t R'_t$, and the ruin bound "
        r"$1/|\min_t R'_t|$ (the sample's worst day). The headline, the paper's block-diagonal ridge scored the same way, and always short, at both fills.}"
        r"\label{tab:app_compounding_headline}\\",
        r"\toprule",
        r"series & fill & $W_T$ & $g$ & max DD & worst day & ruin bound \\",
        r"\midrule",
        r"\endfirsthead",
        r"\caption[]{(continued)}\\",
        r"\toprule",
        r"series & fill & $W_T$ & $g$ & max DD & worst day & ruin bound \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for _, r_ in ff.iterrows():
        L.append(
            f"{r_['series']} & {r_['fill']} & {r_['terminal']:.2f} & {r_['g_ann']:+.2f} & {100 * abs(r_['maxDD_frac']):.0f}\\,\\% & "
            f"{r_['worst_day_factor']:.3f} & {r_['ruin_bound_f']:.2f} \\\\"
        )
    L += [r"\end{longtable}", r"\endgroup", ""]
    hm, hc = row("headline sign(s)", "mid"), row("headline sign(s)", "crossed")
    rm_, rc_ = (
        row("reference sign(s) (this scorer)", "mid"),
        row("reference sign(s) (this scorer)", "crossed"),
    )
    am, ac = row("always short", "mid"), row("always short", "crossed")
    fam = fa[(fa["fill"] == "mid") & fa["in_rank_set"]]
    fac = fa[(fa["fill"] == "crossed") & fa["in_rank_set"]]
    n_set = len(fam)
    pyh = py[(py["series"] == "headline sign(s)") & (py["fill"] == "mid")].set_index(
        "year"
    )["g_ann"]
    pyhc = py[
        (py["series"] == "headline sign(s)") & (py["fill"] == "crossed")
    ].set_index("year")["g_ann"]
    neg = pyh[pyh <= 0]
    negc = pyhc[pyhc <= 0]
    macros = {
        "cmpF": f"{F_FIXED:.2f}",
        "cmpHW": f"{hm['terminal']:.1f}",
        "cmpHG": f"{hm['g_ann']:+.2f}",
        "cmpHDD": f"{100 * abs(hm['maxDD_frac']):.0f}",
        "cmpHWorst": f"{100 * (1 - hm['worst_day_factor']):.0f}",
        "cmpHRuin": f"{hm['ruin_bound_f']:.2f}",
        "cmpHWX": f"{hc['terminal']:.1f}",
        "cmpHGX": f"{hc['g_ann']:+.2f}",
        "cmpHDDX": f"{100 * abs(hc['maxDD_frac']):.0f}",
        "cmpRW": f"{rm_['terminal']:.1f}",
        "cmpRG": f"{rm_['g_ann']:+.2f}",
        "cmpRWX": f"{rc_['terminal']:.1f}",
        "cmpAW": f"{am['terminal']:.2f}",
        "cmpAG": f"{am['g_ann']:+.2f}",
        "cmpARuin": f"{am['ruin_bound_f']:.2f}",
        "cmpAWX": f"{ac['terminal']:.2f}",
        "cmpNAll": f"{n_set}",
        "cmpNPosMid": f"{int((fam['terminal'] > 1).sum())}",
        "cmpNPosX": f"{int((fac['terminal'] > 1).sum())}",
        "cmpWLoMid": f"{fam['terminal'].min():.1f}",
        "cmpWHiMid": f"{fam['terminal'].max():.1f}",
        "cmpWLoX": f"{fac['terminal'].min():.1f}",
        "cmpWHiX": f"{fac['terminal'].max():.1f}",
        "cmpHRankMid": f"{int((fam['terminal'] > hm['terminal']).sum() + 1)}",
        "cmpHRankX": f"{int((fac['terminal'] > hc['terminal']).sum() + 1)}",
        "cmpNYears": f"{len(pyh)}",
        "cmpNPosYears": f"{int((pyh > 0).sum())}",
        "cmpNegYears": ", ".join(f"{y} is ${v:+.2f}$" for y, v in neg.items())
        if len(neg)
        else "none",
        "cmpNPosYearsX": f"{int((pyhc > 0).sum())}",
        "cmpNegYearsX": ", ".join(f"{y} is ${v:+.2f}$" for y, v in negc.items())
        if len(negc)
        else "none",
        "cmpYearsMid": ", ".join(f"{y}: ${v:+.2f}$" for y, v in pyh.items()),
        "cmpYearsX": ", ".join(f"{y}: ${v:+.2f}$" for y, v in pyhc.items()),
    }
    L += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]
    TEX.parent.mkdir(parents=True, exist_ok=True)
    TEX.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")

    # ---- SUMMARY.md from the CSVs
    # 2026-10-06: recalibration removed (user decision): "plain forecast", and the committed
    # (recalibrated) run set beside this one for the record
    S = [
        "# Compounding at a fixed fraction of wealth: the headline forecast, plain 16:00 forecast (no recalibration)",
        "",
        "Written by `experiments/close_compounding_headline.py` from `fixedfrac_headline.csv`, `fixedfrac_all_forecasts.csv`, `peryear_growth.csv` and `gates.csv` in this folder. "
        "Definition: the notebook's section 15 (f = 0.03 of wealth deployed as straddle premium every day, W_T = Π(1 + f q_t R_t), g = 252·mean log(1 + f R'), the drawdown as a "
        "fraction of the running peak, the worst-day factor 1 + f·min R', the ruin bound 1/|min R'|). Input: the master table's per-day returns of every table-A forecast "
        "(`results/close_master_table/master_table_daily.parquet`; the headline's Sharpe gated against `master_table.csv`), every position q = sign(f²·B − implied variance) "
        "on the plain forecast f²·B (no recalibration, 2026-10-06 user decision; gated against the frame's `plain` column). The parked exercise (Appendix D.15) is the paper's "
        "block-diagonal ridge under the deck's session-bar recalibration and is not set beside these numbers.",
        "",
        "## The headline, the reference scored this way, always short",
        "",
        "| series | fill | W_T | g (ann. log-growth) | max drawdown | worst-day factor | ruin bound f |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r_ in ff.iterrows():
        S.append(
            f"| {r_['series']} | {r_['fill']} | {r_['terminal']:.2f} | {r_['g_ann']:+.3f} | {100 * abs(r_['maxDD_frac']):.0f} % | {r_['worst_day_factor']:.3f} | {r_['ruin_bound_f']:.2f} |"
        )
    S += [
        "",
        f"## Every table-A forecast (`fixedfrac_all_forecasts.csv`: {len(keys)} forecasts; counted over the master table's rank set of {n_set}, "
        "`in_rank_set`: check rows, exact duplicates and always short left out)",
        f"- Midpoint: {int((fam['terminal'] > 1).sum())} of {n_set} sign(s) portfolios end above 1 (terminal wealth {fam['terminal'].min():.2f} to {fam['terminal'].max():.2f}); "
        f"the headline's {hm['terminal']:.2f} ranks {int((fam['terminal'] > hm['terminal']).sum() + 1)}.",
        f"- Crossed: {int((fac['terminal'] > 1).sum())} of {n_set} end above 1 ({fac['terminal'].min():.2f} to {fac['terminal'].max():.2f}); the headline's {hc['terminal']:.2f} ranks "
        f"{int((fac['terminal'] > hc['terminal']).sum() + 1)}.",
        "",
        "## Per calendar year, headline sign(s) (annualized from the days traded that year; `peryear_growth.csv`)",
        "",
        "| year | days | g mid | g crossed |",
        "|---|---|---|---|",
    ]
    dpy = py[(py["series"] == "headline sign(s)") & (py["fill"] == "mid")].set_index(
        "year"
    )["days"]
    for y in pyh.index:
        S.append(f"| {y} | {int(dpy[y])} | {pyh[y]:+.3f} | {pyhc[y]:+.3f} |")
    # 2026-10-06: before / after against the committed (recalibrated) run, reported only
    ffo = committed["fixedfrac_headline.csv"].set_index(["series", "fill"])
    pyo = committed["peryear_growth.csv"].set_index(["series", "fill", "year"])
    fao = committed["fixedfrac_all_forecasts.csv"]
    fao = fao[fao["in_rank_set"].astype(bool)]
    S += [
        "",
        f"## Before / after: this run against the committed run ({PREV_REV} = {sha}, the 16:00-bar recalibration (f² + s)·B; reported, not asserted)",
        "",
        "| series | fill | W_T before → after | g before → after | max drawdown before → after |",
        "|---|---|---|---|---|",
    ]
    for _, r_ in ff.iterrows():
        o_ = ffo.loc[(r_["series"], r_["fill"])]
        S.append(
            f"| {r_['series']} | {r_['fill']} | {o_['terminal']:.2f} → {r_['terminal']:.2f} | {o_['g_ann']:+.3f} → {r_['g_ann']:+.3f} | "
            f"{100 * abs(o_['maxDD_frac']):.0f} % → {100 * abs(r_['maxDD_frac']):.0f} % |"
        )
    S += [
        "",
        "- Headline per-year g, mid, before → after: "
        + "; ".join(
            f"{y} {pyo.loc[('headline sign(s)', 'mid', y), 'g_ann']:+.2f} → {pyh[y]:+.2f}"
            for y in pyh.index
        )
        + ".",
        f"- Rank set: forecasts ending above 1 at the midpoint {int((fao[fao['fill'] == 'mid']['terminal'] > 1).sum())} of "
        f"{int((fao['fill'] == 'mid').sum())} → {int((fam['terminal'] > 1).sum())} of {n_set}; crossed "
        f"{int((fao[fao['fill'] == 'crossed']['terminal'] > 1).sum())} → {int((fac['terminal'] > 1).sum())}.",
        "- Reported (max |new − old| / max(1, |old|)): "
        + "; ".join(
            f"`{r_['file']}` ({r_['rows']}) {r_['max_rel_diff']:.3g}" for r_ in reported
        )
        + ".",
    ]
    S += ["", "## Gates"]
    S += [
        f"- {g['gate']}: {g['value']:.3g} ≤ {g['bound']:.3g} on n = {g['n']} — {'PASS' if g['ok'] else 'FAIL'}"
        for g in gates
    ]
    (OUT / "SUMMARY.md").write_text("\n".join(S) + "\n", encoding="utf-8")
    print(ff.round(4).to_string(index=False))
    print(
        py[py["series"] == "headline sign(s)"]
        .pivot(index="year", columns="fill", values="g_ann")
        .round(3)
        .to_string()
    )
    print(
        f"table-A forecasts ending above 1: mid {int((fam['terminal'] > 1).sum())} / crossed {int((fac['terminal'] > 1).sum())} of {n_set}"
    )
    print("gates:", sum(g["ok"] for g in gates), "of", len(gates), "pass")
    print("wrote", OUT, "and", TEX)


def write_numbers() -> None:
    """2026-10-15 (number macros): NUMBERS_TEX, macro definitions only, from the stored fixedfrac_modelset*.csv
    (python experiments/close_compounding_headline.py --numbers-only).  Same names and formats as main()'s macros.

    Light check first: the stored rows are the current model set (paper_model_set.csv), and every stored terminal
    wealth equals prod(1 + f r_t) of the current master_table_daily.parquet returns.  The positions read only the
    forecast and the implied variance, so the scoring switch to the actual realized variance (data audit #10,
    2026-10-15) cannot move these numbers; the check confirms it."""
    fa = pd.read_csv(OUT / "fixedfrac_modelset.csv")
    summ = pd.read_csv(OUT / "fixedfrac_modelset_summary.csv")
    keys = set(pd.read_csv(MODEL_SET_CSV)["key"].astype(str))
    stored = fa[fa["key"] != "always short"]
    assert set(stored["key"].astype(str)) == keys, (
        "fixedfrac_modelset.csv is not the current model set: re-run main()"
    )
    D = pd.read_parquet(DAILY, columns=["key", "day", "ret_mid", "ret_crossed"])
    D = D[D["key"].isin(keys)].sort_values(["key", "day"])
    term = {
        (k, fill): float(np.prod(1.0 + F_FIXED * x[f"ret_{fill}"].to_numpy(float)))
        for k, x in D.groupby("key")
        for fill in ("mid", "crossed")
    }
    worst = max(
        abs(term[(str(r_["key"]), r_["fill"])] / float(r_["terminal"]) - 1)
        for _, r_ in stored.iterrows()
    )
    assert worst <= GATE_REL, (
        f"stored terminal wealth differs from the current daily returns ({worst:.2g}): re-run main()"
    )
    sm = summ[summ["group"] == "model set"].set_index("fill")
    for fill in ("mid", "crossed"):
        x = stored[stored["fill"] == fill]
        assert abs(sm.loc[fill, "W_median"] - x["terminal"].median()) < 1e-12
        assert (
            abs(sm.loc[fill, "maxDD_max_pct"] - 100 * x["maxDD_frac"].abs().max())
            < 1e-9
        )
    macros = {
        "cmpWMedMid": f"{sm.loc['mid', 'W_median']:.1f}",
        "cmpWMedX": f"{sm.loc['crossed', 'W_median']:.1f}",
        "cmpDDMaxMid": f"{sm.loc['mid', 'maxDD_max_pct']:.0f}",
        "cmpDDMaxX": f"{sm.loc['crossed', 'maxDD_max_pct']:.0f}",
    }
    assert macros == NUMBERS_TYPED, (
        f"{macros} != the values the text typed {NUMBERS_TYPED}: update writeup/rehaul/07_economics.tex and "
        "NUMBERS_TYPED together"
    )
    what = {
        "cmpWMedMid": "median terminal wealth W_oos across the model set, mid-quotes",
        "cmpWMedX": "the same at quoted bid/ask prices",
        "cmpDDMaxMid": "largest maximum drawdown across the model set, %, mid-quotes",
        "cmpDDMaxX": "the same at quoted bid/ask prices, %",
    }
    L = [
        "% AUTO-GENERATED by experiments/close_compounding_headline.py (write_numbers) from",
        "% results/close_compounding/fixedfrac_modelset*.csv -- do not edit. Macro definitions only (no tables);",
        "% loaded once by the preamble of writeup/paper_rehaul.tex (same names and values as the table-mixed",
        "% appendix_close_compounding.tex, which the rehauled paper does not load).",
        f"% Fixed fraction f = {F_FIXED:.2f} of wealth per day, sign strategy, {len(keys)} forecasts of the model set.",
    ]
    L += [rf"\newcommand{{\{k}}}{{{v}}}  % {what[k]}" for k, v in macros.items()]
    NUMBERS_TEX.parent.mkdir(parents=True, exist_ok=True)
    NUMBERS_TEX.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print(
        "wrote",
        NUMBERS_TEX,
        f"(terminal wealth = current daily returns to {worst:.1g})",
    )


def main() -> None:
    """Fixed-fraction compounding of every forecast of the paper's model set.

    2026-10-06 (user decision): no headline; the same fixed fraction f for every model-set
    forecast (plain f^2 B positions, both fills): terminal wealth (how many end above 1,
    the median and the range), annualized log-growth, the largest wealth drawdown and the
    ruin-bound check 1 / |min R'| > f, by group and over the set, beside always short.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    gates: list[dict] = []

    def gate(name: str, value: float, bound: float, n: int) -> None:
        gates.append(
            {
                "gate": name,
                "value": float(value),
                "bound": bound,
                "n": n,
                "ok": bool(value <= bound),
            }
        )

    if not DAILY.exists():
        raise SystemExit(
            f"{DAILY} is missing: run experiments/master_table_close.py first"
        )
    mt = pd.read_csv(MASTER).set_index("key")
    keys, set_src = model_set(mt)
    D = pd.read_parquet(
        DAILY, columns=["key", "day", FCOL, "iv_var", "q", "ret_mid", "ret_crossed"]
    )
    D["day"] = pd.to_datetime(D["day"])
    D = D[D["key"].isin(keys)]
    deck = pd.read_parquet(base.DECK).sort_index()
    deck.index = pd.DatetimeIndex(pd.to_datetime(deck.index)).as_unit("ns")
    days = deck.index
    assert len(days) == 866
    frames = {k: D[D["key"] == k].set_index("day").reindex(days) for k in keys}
    grp = {k: group_of(str(mt.loc[k, "family"])) for k in keys}
    R = deck["R"].to_numpy(float)
    ex = deck["exit"].to_numpy(float)
    bid = (deck["bid_c"] + deck["bid_p"]).to_numpy(float)
    iv = deck["iv_var"].to_numpy(float)
    worst = {"sharpe_mid": 0.0, "sharpe_x": 0.0, "qR": 0.0, "q": 0.0}
    for k in keys:
        fk = frames[k]
        assert fk["ret_mid"].notna().all(), k
        worst["sharpe_mid"] = max(
            worst["sharpe_mid"],
            abs(
                mtc.sharpe(fk["ret_mid"].to_numpy(float))
                / float(mt.loc[k, "Sharpe_mid"])
                - 1
            ),
        )
        worst["sharpe_x"] = max(
            worst["sharpe_x"],
            abs(
                mtc.sharpe(fk["ret_crossed"].to_numpy(float))
                / float(mt.loc[k, "Sharpe_crossed"])
                - 1
            ),
        )
        worst["qR"] = max(
            worst["qR"],
            float(
                np.abs(
                    fk["ret_mid"].to_numpy(float) - fk["q"].to_numpy(float) * R
                ).max()
            ),
        )
        worst["q"] = max(
            worst["q"],
            float(
                np.abs(
                    fk["q"].to_numpy(float)
                    - np.where(fk[FCOL].to_numpy(float) > iv, 1.0, -1.0)
                ).max()
            ),
        )
    gate(
        "Sharpe mid = master_table.csv, every forecast",
        worst["sharpe_mid"],
        GATE_REL,
        len(keys),
    )
    gate(
        "Sharpe crossed = master_table.csv, every forecast",
        worst["sharpe_x"],
        GATE_REL,
        len(keys),
    )
    gate("ret_mid = q R, every forecast", worst["qR"], 1e-12, len(keys))
    gate(f"q = sign({FCOL} - iv_var), every forecast", worst["q"], 0, len(keys))

    # ---- every forecast of the set and always short, both fills (wealth_stats asserts every factor > 0)
    rows = []
    for k in keys:
        for fill in ("mid", "crossed"):
            rows.append(
                {
                    "key": k,
                    "label": mt.loc[k, "label"],
                    "group": grp[k],
                    "inputs": bucket_of(k),
                    "fill": fill,
                    **wealth_stats(F_FIXED, frames[k][f"ret_{fill}"].to_numpy(float)),
                }
            )
    short = {"mid": -R, "crossed": 1.0 - ex / bid}
    for fill, r in short.items():
        rows.append(
            {
                "key": "always short",
                "label": "always short",
                "group": "always short",
                "inputs": "",
                "fill": fill,
                **wealth_stats(F_FIXED, r),
            }
        )
    fa = pd.DataFrame(rows)
    fa.to_csv(OUT / "fixedfrac_modelset.csv", index=False)
    gate(
        f"the ruin bound exceeds f = {F_FIXED} for every forecast and fill (no wealth factor <= 0)",
        float(F_FIXED - fa["ruin_bound_f"].min())
        if fa["ruin_bound_f"].min() <= F_FIXED
        else 0.0,
        0,
        len(fa),
    )

    # ---- summary by group and over the set, both fills
    sm_rows = []
    fs = fa[fa["key"] != "always short"]
    for fill in ("mid", "crossed"):
        x_fill = fs[fs["fill"] == fill]
        for g, x in [
            (g, x_fill[x_fill["group"] == g])
            for g in GROUPS
            if (x_fill["group"] == g).any()
        ] + [("model set", x_fill)]:
            sm_rows.append(
                {
                    "group": g,
                    "fill": fill,
                    "n": len(x),
                    "n_above_1": int((x["terminal"] > 1).sum()),
                    "W_median": float(x["terminal"].median()),
                    "W_min": float(x["terminal"].min()),
                    "W_max": float(x["terminal"].max()),
                    "g_median": float(x["g_ann"].median()),
                    "g_min": float(x["g_ann"].min()),
                    "g_max": float(x["g_ann"].max()),
                    "maxDD_median_pct": float(100 * x["maxDD_frac"].abs().median()),
                    "maxDD_max_pct": float(100 * x["maxDD_frac"].abs().max()),
                    "worst_day_min_factor": float(x["worst_day_factor"].min()),
                    "ruin_bound_min": float(x["ruin_bound_f"].min()),
                    "n_ruin_bound_above_f": int((x["ruin_bound_f"] > F_FIXED).sum()),
                }
            )
        a_ = fa[(fa["key"] == "always short") & (fa["fill"] == fill)].iloc[0]
        sm_rows.append(
            {
                "group": "always short",
                "fill": fill,
                "n": 1,
                "n_above_1": int(a_["terminal"] > 1),
                "W_median": a_["terminal"],
                "W_min": a_["terminal"],
                "W_max": a_["terminal"],
                "g_median": a_["g_ann"],
                "g_min": a_["g_ann"],
                "g_max": a_["g_ann"],
                "maxDD_median_pct": 100 * abs(a_["maxDD_frac"]),
                "maxDD_max_pct": 100 * abs(a_["maxDD_frac"]),
                "worst_day_min_factor": a_["worst_day_factor"],
                "ruin_bound_min": a_["ruin_bound_f"],
                "n_ruin_bound_above_f": int(a_["ruin_bound_f"] > F_FIXED),
            }
        )
    summ = pd.DataFrame(sm_rows)
    summ.to_csv(OUT / "fixedfrac_modelset_summary.csv", index=False)

    # ---- per calendar year (annualized from the days traded that year), every forecast
    yr = days.year
    py_rows = []
    for k in keys + ["always short"]:
        for fill in ("mid", "crossed"):
            r = (
                short[fill]
                if k == "always short"
                else frames[k][f"ret_{fill}"].to_numpy(float)
            )
            gy = PPY * pd.Series(np.log1p(F_FIXED * r), index=days).groupby(yr).mean()
            for y_, v in gy.items():
                py_rows.append(
                    {
                        "key": k,
                        "group": grp.get(k, "always short"),
                        "fill": fill,
                        "year": int(y_),
                        "days": int((yr == y_).sum()),
                        "g_ann": float(v),
                    }
                )
    py = pd.DataFrame(py_rows)
    py.to_csv(OUT / "peryear_growth_modelset.csv", index=False)

    # ---- always short reads no forecast: it still equals its committed rows
    for fname, idx, newframe in (
        (
            "fixedfrac_headline.csv",
            ["series", "fill"],
            fa[fa["key"] == "always short"].assign(series="always short"),
        ),
        (
            "peryear_growth.csv",
            ["series", "fill", "year"],
            py[py["key"] == "always short"].assign(series="always short"),
        ),
    ):
        old, sha = committed_csv(OUT / fname)
        old = old[old["series"] == "always short"].set_index(idx)
        new = newframe.set_index(idx)
        old = old[[c for c in old.columns if c in new.columns]]
        gate(
            f"{fname} = the committed copy ({PREV_REV} = {sha}), always short's rows (no forecast)",
            committed_diff(new, old),
            COMMITTED_TOL,
            int(old.size),
        )
    pd.DataFrame(gates).to_csv(OUT / "gates.csv", index=False)
    assert all(g["ok"] for g in gates), [g for g in gates if not g["ok"]]

    # ---- figure: every forecast's wealth path (thin, by group), always short (dashed)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, fill in zip(axes, ("mid", "crossed")):
        for k in keys:
            w = np.cumprod(1.0 + F_FIXED * frames[k][f"ret_{fill}"].to_numpy(float))
            ax.plot(days, w, lw=0.6, alpha=0.45, color=GROUP_COLOURS[grp[k]])
        ax.plot(
            days, np.cumprod(1.0 + F_FIXED * short[fill]), lw=1.4, color="0.4", ls="--"
        )
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(FixedLocator([0.25, 0.5, 1, 2, 5, 10, 20, 50, 100]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.axhline(1.0, color="k", lw=0.6, ls="--")
        ax.grid(axis="y", which="major", alpha=0.3)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.set_title(
            f"{fill} fill: {len(keys)} forecasts, a fixed {F_FIXED:.0%} of wealth per day",
            fontsize=9,
        )
    axes[0].set_ylabel("wealth multiple (log scale; 1× = starting stake)")
    handles = [
        plt.Line2D([], [], color=GROUP_COLOURS[g], lw=1.2, label=g)
        for g in GROUPS
        if g in grp.values()
    ]
    handles.append(
        plt.Line2D([], [], color="0.4", lw=1.4, ls="--", label="always short")
    )
    axes[0].legend(handles=handles, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "wealth_modelset.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---- tex: table + macros (no headline macros)
    L = [
        "% AUTO-GENERATED by experiments/close_compounding_headline.py (main: the paper's model set) from",
        "% results/close_compounding/fixedfrac_modelset*.csv -- do not edit.",
        f"% 2026-10-06: the paper's model set ({set_src}); no headline; plain forecast f^2 B.",
        r"\begingroup\scriptsize\setlength{\tabcolsep}{3pt}",
        r"\begin{longtable}{llrrrrrrr}",
        rf"\caption{{Compounding at a fixed $f={F_FIXED:.2f}$ of wealth per day across the paper's model set ({len(keys)} "
        r"forecasts, each its plain forecast $\hat y^2B$, no recalibration), 866 days, by group: how many end above one "
        r"unit of wealth, the median terminal wealth $W_T=\prod_t(1+f\,q_tR_t)$ and its range, the median annualized "
        r"log-growth $g$, the median and the largest fall of wealth from its running peak, and the smallest ruin bound "
        r"$1/|\min_t R'_t|$ (the sample's worst day). Always short on the same days.}\label{tab:app_compounding_headline}\\",
        r"\toprule",
        r"group & fill & above 1 & median $W_T$ & range & median $g$ & median DD & largest DD & min ruin bound \\",
        r"\midrule",
        r"\endfirsthead",
        r"\caption[]{(continued)}\\",
        r"\toprule",
        r"group & fill & above 1 & median $W_T$ & range & median $g$ & median DD & largest DD & min ruin bound \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    order = [g for g in GROUPS if g in grp.values()] + ["model set", "always short"]
    for g in order:
        for fill in ("mid", "crossed"):
            r_ = summ[(summ["group"] == g) & (summ["fill"] == fill)].iloc[0]
            lab = r"\textbf{model set}" if g == "model set" else g
            above = (
                ""
                if g == "always short"
                else f"{int(r_['n_above_1'])} of {int(r_['n'])}"
            )
            rng = (
                "" if g == "always short" else f"{r_['W_min']:.2f} to {r_['W_max']:.2f}"
            )
            L.append(
                f"{lab} & {fill} & {above} & {r_['W_median']:.2f} & {rng} & {r_['g_median']:+.2f} & "
                f"{r_['maxDD_median_pct']:.0f}\\,\\% & {r_['maxDD_max_pct']:.0f}\\,\\% & {r_['ruin_bound_min']:.2f} \\\\"
            )
        if g in ("per-bar trees", "model set") or g == order[-3]:
            L.append(r"\addlinespace")
    L += [r"\end{longtable}", r"\endgroup", ""]

    def S_(g: str, fill: str) -> pd.Series:
        return summ[(summ["group"] == g) & (summ["fill"] == fill)].iloc[0]

    sm_mid, sm_x = S_("model set", "mid"), S_("model set", "crossed")
    am, ac = S_("always short", "mid"), S_("always short", "crossed")
    fm_mid = fs[fs["fill"] == "mid"].set_index("key")
    best, worst_k = fm_mid["terminal"].idxmax(), fm_mid["terminal"].idxmin()
    pym = py[(py["key"] != "always short") & (py["fill"] == "mid")]
    pyx = py[(py["key"] != "always short") & (py["fill"] == "crossed")]
    yrs = sorted(pym["year"].unique())

    def tex_label(k: str) -> str:
        return str(mt.loc[k, "label"]).replace("_", r"\_").replace("&", r"\&")

    macros = {
        "cmpF": f"{F_FIXED:.2f}",
        "cmpNSet": f"{len(keys)}",
        "cmpNPosMid": f"{int(sm_mid['n_above_1'])}",
        "cmpNPosX": f"{int(sm_x['n_above_1'])}",
        "cmpWMedMid": f"{sm_mid['W_median']:.1f}",
        "cmpWLoMid": f"{sm_mid['W_min']:.1f}",
        "cmpWHiMid": f"{sm_mid['W_max']:.1f}",
        "cmpWMedX": f"{sm_x['W_median']:.1f}",
        "cmpWLoX": f"{sm_x['W_min']:.1f}",
        "cmpWHiX": f"{sm_x['W_max']:.1f}",
        "cmpGMedMid": f"{sm_mid['g_median']:+.2f}",
        "cmpGLoMid": f"{sm_mid['g_min']:+.2f}",
        "cmpGHiMid": f"{sm_mid['g_max']:+.2f}",
        "cmpGMedX": f"{sm_x['g_median']:+.2f}",
        "cmpDDMedMid": f"{sm_mid['maxDD_median_pct']:.0f}",
        "cmpDDMaxMid": f"{sm_mid['maxDD_max_pct']:.0f}",
        "cmpDDMedX": f"{sm_x['maxDD_median_pct']:.0f}",
        "cmpDDMaxX": f"{sm_x['maxDD_max_pct']:.0f}",
        "cmpWorstDayMax": f"{100 * (1 - min(sm_mid['worst_day_min_factor'], sm_x['worst_day_min_factor'])):.0f}",
        "cmpRuinMin": f"{min(sm_mid['ruin_bound_min'], sm_x['ruin_bound_min']):.2f}",
        "cmpNRuinOK": f"{int(sm_mid['n_ruin_bound_above_f'])}",
        "cmpBestLabel": tex_label(best),
        "cmpWorstLabel": tex_label(worst_k),
        "cmpAW": f"{am['W_median']:.2f}",
        "cmpAG": f"{am['g_median']:+.2f}",
        "cmpARuin": f"{am['ruin_bound_min']:.2f}",
        "cmpAWX": f"{ac['W_median']:.2f}",
        "cmpNYears": f"{len(yrs)}",
        "cmpYearsNPosMid": ", ".join(
            f"{y_}: {int((pym[pym['year'] == y_]['g_ann'] > 0).sum())}" for y_ in yrs
        ),
        "cmpYearsNPosX": ", ".join(
            f"{y_}: {int((pyx[pyx['year'] == y_]['g_ann'] > 0).sum())}" for y_ in yrs
        ),
        "cmpYearsMedMid": ", ".join(
            f"{y_}: ${pym[pym['year'] == y_]['g_ann'].median():+.2f}$" for y_ in yrs
        ),
        "cmpYearsMedX": ", ".join(
            f"{y_}: ${pyx[pyx['year'] == y_]['g_ann'].median():+.2f}$" for y_ in yrs
        ),
    }
    L += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]
    stubs = [m for m in HEADLINE_MACROS_PARKED if m not in macros]
    L += [
        "% 2026-10-06: the headline macros are parked (no headline model; user decision). Each is a",
        "% visible ?? stub until sections/appendix_close_compounding.tex stops using it.",
    ]
    L += [f"\\newcommand{{\\{m}}}{{\\textbf{{??}}}}" for m in stubs]
    TEX.parent.mkdir(parents=True, exist_ok=True)
    TEX.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")

    # ---- SUMMARY.md
    S = [
        "# Compounding at a fixed fraction of wealth across the paper's model set (plain forecast f²·B)",
        "",
        "Written by `experiments/close_compounding_headline.py` (`main`, 2026-10-06: no headline model, user decision) from "
        "`fixedfrac_modelset.csv`, `fixedfrac_modelset_summary.csv`, `peryear_growth_modelset.csv` and `gates.csv`. Definition: "
        "the notebook's section 15 (f = 0.03 of wealth as straddle premium every day, W_T = Π(1 + f q_t R_t), g = 252·mean "
        "log(1 + f R'), the drawdown as a fraction of the running peak, the ruin bound 1/|min R'|). Input: the master table's "
        f"per-day returns. Model set: {len(keys)} forecasts ({set_src}).",
        "",
        "| group | fill | above 1 | median W_T | range | median g | median DD | largest DD | smallest ruin bound |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for g in order:
        for fill in ("mid", "crossed"):
            r_ = S_(g, fill)
            S.append(
                f"| {g} | {fill} | {int(r_['n_above_1'])} of {int(r_['n'])} | {r_['W_median']:.2f} | {r_['W_min']:.2f} to "
                f"{r_['W_max']:.2f} | {r_['g_median']:+.3f} | {r_['maxDD_median_pct']:.0f} % | {r_['maxDD_max_pct']:.0f} % | "
                f"{r_['ruin_bound_min']:.2f} |"
            )
    S += [
        "",
        f"- Largest terminal wealth at the midpoint: {mt.loc[best, 'label']} ({fm_mid.loc[best, 'terminal']:.2f}); smallest: "
        f"{mt.loc[worst_k, 'label']} ({fm_mid.loc[worst_k, 'terminal']:.2f}).",
        "- Per year, forecasts with positive log-growth (mid): "
        + ", ".join(
            f"{y_} {int((pym[pym['year'] == y_]['g_ann'] > 0).sum())}" for y_ in yrs
        )
        + f" of {len(keys)}; median g: "
        + ", ".join(
            f"{y_} {pym[pym['year'] == y_]['g_ann'].median():+.2f}" for y_ in yrs
        )
        + ".",
        "",
        "## Gates",
    ]
    S += [
        f"- {g['gate']}: {g['value']:.3g} ≤ {g['bound']:.3g} on n = {g['n']} — {'PASS' if g['ok'] else 'FAIL'}"
        for g in gates
    ]
    (OUT / "SUMMARY.md").write_text("\n".join(S) + "\n", encoding="utf-8", newline="\n")
    print(summ.round(3).to_string(index=False))
    print("gates:", sum(g["ok"] for g in gates), "of", len(gates), "pass")
    print("wrote", OUT, "and", TEX)
    write_numbers()  # 2026-10-15 (number macros): the macro-only file, from the CSVs just written


if __name__ == "__main__":
    # 2026-10-15 (number macros): --numbers-only writes NUMBERS_TEX from the stored CSVs (no re-run)
    if "--numbers-only" in sys.argv[1:]:
        write_numbers()
    else:
        main()
