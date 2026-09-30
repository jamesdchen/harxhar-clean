"""Compounding at a fixed fraction of wealth, for the HEADLINE forecast under the research scorer.

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


def main() -> None:
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

    # ---- the committed copies: the headline's, the reference's and always short's rows are unchanged
    for fname, idx, keep in (
        ("fixedfrac_headline.csv", ["series", "fill"], None),
        ("peryear_growth.csv", ["series", "fill", "year"], None),
        ("fixedfrac_all_forecasts.csv", ["key", "fill"], [HEADLINE, REFERENCE]),
    ):
        old, sha = committed_csv(OUT / fname)
        new = pd.read_csv(OUT / fname)
        if keep is not None:
            old, new = old[old["key"].isin(keep)], new[new["key"].isin(keep)]
        old, new = old.set_index(idx), new.set_index(idx)
        rows_ = (
            "every row" if keep is None else "the headline's and the reference's rows"
        )
        gate(
            f"{fname} = the committed copy ({PREV_REV} = {sha}), {rows_}, "
            "max |new - old| / max(1, |old|)",
            committed_diff(new, old),
            COMMITTED_TOL,
            int(old.size),
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
        r"\begingroup\footnotesize\setlength{\tabcolsep}{3pt}",
        r"\begin{longtable}{llrrrrr}",
        r"\caption{Compounding at a fixed $f=0.03$ of wealth per day under the 16:00-bar recalibration, 866 days: terminal wealth $W_T=\prod_t(1+f\,q_tR_t)$, "
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
    TEX.write_text("\n".join(L) + "\n", encoding="utf-8")

    # ---- SUMMARY.md from the CSVs
    S = [
        "# Compounding at a fixed fraction of wealth: the headline forecast under the 16:00-bar recalibration",
        "",
        "Written by `experiments/close_compounding_headline.py` from `fixedfrac_headline.csv`, `fixedfrac_all_forecasts.csv`, `peryear_growth.csv` and `gates.csv` in this folder. "
        "Definition: the notebook's section 15 (f = 0.03 of wealth deployed as straddle premium every day, W_T = Π(1 + f q_t R_t), g = 252·mean log(1 + f R'), the drawdown as a "
        "fraction of the running peak, the worst-day factor 1 + f·min R', the ruin bound 1/|min R'|). Input: the master table's per-day returns of every table-A forecast "
        "(`results/close_master_table/master_table_daily.parquet`; the headline's Sharpe gated against `master_table.csv`). The parked exercise (Appendix D.15) is the paper's "
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


if __name__ == "__main__":
    main()
