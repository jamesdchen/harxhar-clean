"""The lean input set as a follow-up to the feature-importance study (2026-10-06).

User decision 2026-10-06: "make the lean set the third input set in the paper, but it has to be a
follow up to the feature importance study."  The lean set is SUBGROUPS["lean_vol_vix"]
(src/data/loading.py): sumret, sumabsret, sumret4, sumautocov, sumvolume, vix.  This script
writes the two pieces of evidence the text needs:

(A) IMPORTANCE -- why these series.  From the importance study re-run on the de-duplicated design
    (results/feature_importance_1530_dedup/; experiments/feature_importance_1530_dedup.py), on
    all_features, SERIES level (a series = every column of one input: its lags and flags): for
    each of the seven ES return-moment series of the design (sumret, sumabsret, sumret3, sumret4,
    sumpret2 = upside squared, sumbipow = bipower, sumautocov), ES volume (sumvolume) and VIX, the
    median rank and the number of top-10 ranks over the 37 model x measure cells (one cell = one
    by_measure/imp_all_features_<model>_<measure>.csv: ridge and lasso x 8 measures, LightGBM,
    XGBoost and random forest x 7; a series' rank within the cell's 49 series, 1 = most important,
    the study's own `rank` column), beside the leader (har_ma) and the best-ranked other series;
    and the correlation clusters (clusters.csv; complete linkage at |corr| >= 0.8) that put
    absolute, quartic, upside-squared and bipower returns together, one per lag, with each
    cluster's minimum absolute pairwise correlation.

    PRE-2020 (the same ranks on the data before the 2020 crash): a series' rank in a cell is the
    order of its refit-averaged measure value, highest first, among the cell's eligible series
    (gate: on the all-period `value` column this reproduces the files' `rank` column in all 37
    cells).  The pre-2020 value is the tail-day-weighted mean of the files' y2018 and y2019
    columns.  Those columns average over the refits whose causal tail STARTS in that year
    (experiments/feature_importance_1530.py _vix_labels / aggregate), so their weights are the
    tail days of those refits: 13 refits x 10 = 130 and 26 x 10 = 260 (gate: the same label
    weights rebuild the all-period `value` from y2018..y2024).  The 39 refits' tails run from
    2018-06-25 to 2020-01-14; their last refit's tail starts 2019-12-31.  (The calendar session
    counts, 129 and 252, give the same medians and top-10 counts.)  Ties (the lasso zeroes
    28-29 series before 2020 in its 8 cells) share the best rank of the tie (competition
    ranking, method="min").  The cluster check on the same period: for each HAR lag k the
    minimum absolute pairwise correlation among adj_sumabsret_ma_k, adj_sumret4_ma_k,
    adj_sumpret2_ma_k and adj_sumbipow_ma_k over the all_features design rows before 2020-01-03
    (results/close_studies_2026-10-03/forecasts/design_bar1600_all_features.parquet).

(B) RESULTS -- what the lean set does.  Every lean forecast of the paper model set
    (make_appendix_campaign_16h_tex.paper_model_set on results/close_master_table/master_table.csv)
    against its twin on all_features and on baseline (the same key, the input-set token swapped):
    QLIKE of the plain forecast (qlike_raw), sign(s) Sharpe mid / crossed, the Diebold-Mariano
    statistic of the daily QLIKE difference (lean minus other; src.evaluation.diebold_mariano)
    and the Newey-West HAC t of the daily mid-return difference (notebooks/atm_straddle_lib
    newey_west_t; p-value master_table_close.hac_p), from master_table_daily.parquet.  Per model
    family group and overall: medians of the levels and of the differences, counts of lean
    better / worse (point estimates) and of those significant at camp.SIG (DM for QLIKE, HAC t
    for Sharpe).  Groups: per-bar linear; pooled linear (the pooled twins); trees with the
    shipped settings (refit every 10 and daily); trees with the random search (refit every 10 and
    daily, both selection rules); trees tuned by Optuna (every TUNE_PER and trial budget).

Point estimates only; DM and HAC t are statistics, not intervals (2026-10-06: intervals removed,
user decision).  Display names carry no "adj_".  If the master table has no (or only some) lean
rows, part B writes what exists and warns; it does not fail.

Writes writeup/generated/featimp_lean_numbers.tex (\\fl... macros),
featimp_lean_importance.tex and featimp_lean_results.tex (booktabs tabulars).

Run:  python writeup/make_featimp_lean_tex.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "writeup", ROOT / "experiments", ROOT / "notebooks", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import make_appendix_campaign_16h_tex as camp  # noqa: E402  (the paper model set)

FI = ROOT / "results" / "feature_importance_1530_dedup"
MASTER = ROOT / "results" / "close_master_table" / "master_table.csv"
DAILY = ROOT / "results" / "close_master_table" / "master_table_daily.parquet"
GEN = ROOT / "writeup" / "generated"
NUMBERS_OUT = GEN / "featimp_lean_numbers.tex"
IMP_OUT = GEN / "featimp_lean_importance.tex"
RES_OUT = GEN / "featimp_lean_results.tex"

BUCKET = (
    "all_features"  # the importance study's design that holds every candidate series
)
N_CELLS = (
    37  # model x measure cells on all_features (ridge, lasso x 8; three trees x 7)
)
TOP = 10  # "top-10 count": cells in which the series ranks 1..TOP
N_OTHER = 5  # best-ranked other series shown beside the candidates
GATE_TOL = 1e-12  # the daily frame reproduces master_table.csv
LEAN_SERIES = ("sumret", "sumabsret", "sumret4", "sumautocov", "sumvolume", "vix")
# the seven ES return moments of the design, ES volume and VIX (the series the lean set is cut from)
CANDIDATES = [
    ("sumret", "Sumret", "signed return"),
    ("sumabsret", "Sumabsret", "absolute return"),
    ("sumret3", "Sumretthree", "cubed return"),
    ("sumret4", "Sumretfour", "quartic return"),
    ("sumpret2", "Sumprettwo", "upside squared return"),
    ("sumbipow", "Sumbipow", "bipower variation"),
    ("sumautocov", "Sumautocov", "return autocovariance"),
    ("sumvolume", "Sumvolume", "ES volume"),
    ("vix", "Vix", "VIX"),
]
LEADER = "har_ma"  # the target's own HAR ladder (the leader of the study)
ABS_FAMILY = {
    "sumabsret",
    "sumret4",
    "sumpret2",
    "sumbipow",
}  # the absolute-return cluster
LAG_WORD = {
    "1": "One",
    "5": "Five",
    "25": "TwentyFive",
    "125": "OneTwentyFive",
    "625": "SixTwentyFive",
    "3125": "ThreeOneTwoFive",
}
ORDINAL = ("One", "Two", "Three", "Four", "Five")
# pre-2020 ranks and cluster check
PRE_YEARS = {2018: "Eighteen", 2019: "Nineteen"}
REFIT_EVERY = 10  # sessions per refit tail (the study's refit cadence)
DESIGN = (
    ROOT
    / "results"
    / "close_studies_2026-10-03"
    / "forecasts"
    / "design_bar1600_all_features.parquet"
)
PRE_CUTOFF = "2020-01-03"  # cluster check: design rows dated before this day
ABS_COLS = ("adj_sumabsret", "adj_sumret4", "adj_sumpret2", "adj_sumbipow")
PRE_TAGS = {"Har"} | {
    tag for _, tag, _ in CANDIDATES
}  # the rows that get pre-2020 macros
# part B: the master table's families, grouped
GROUPS = [
    ("Linear", "per-bar linear"),
    ("Pooled", "pooled linear"),
    ("TreeShip", "trees, shipped settings"),
    ("TreeRs", "trees, random search"),
    ("TreeOpt", "trees, Optuna"),
]
VS = [("AllF", "all_features"), ("Base", "baseline")]

NUMS: dict[str, str] = {}


# --------------------------------------------------------------------------- formatting
def macro(name: str, value: str) -> None:
    assert re.fullmatch(r"fl[A-Z][A-Za-z]*", name), name
    assert name not in NUMS, name
    assert "??" not in value and "nan" not in value.lower() and "[" not in value, (
        name,
        value,
    )
    assert not re.search(r"(?<!\\)_", value), (name, value)
    NUMS[name] = value


def num(x: float, nd: int = 2, plus: bool = False) -> str:
    s = f"{x:+.{nd}f}" if plus else f"{x:.{nd}f}"
    return s.replace("-", r"\ensuremath{-}")


def med(x: pd.Series | np.ndarray) -> float:
    return float(np.median(np.asarray(x, float)))


def disp(name: str) -> str:
    """A series' display name: no "adj_" (presentation rule, 2026-10-06), texttt."""
    name = re.sub(r"\badj_", "", name)
    return r"\texttt{" + name.replace("_", r"\_") + "}"


def write_tab(path: Path, sources: list[str], colspec: str, lines: list[str]) -> None:
    out = [
        "% AUTO-GENERATED by writeup/make_featimp_lean_tex.py -- do not edit.",
        *[f"% Source: {s}" for s in sources],
        r"\begin{tabular}{" + colspec + "}",
        r"\toprule",
        *lines,
        r"\bottomrule",
        r"\end{tabular}",
    ]
    path.write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


# --------------------------------------------------------------------------- (A) importance
def rank_desc(s: pd.DataFrame, col: str) -> pd.Series:
    """A series' rank in its cell: order of the value, highest first; ties share the best rank."""
    return s.groupby(["model", "measure"])[col].rank(ascending=False, method="min")


def year_weights() -> tuple[dict[int, int], dict[int, int]]:
    """Refits and tail days behind each yYYYY column (the refits whose tail starts in that year)."""
    import json

    meta = json.loads((FI / "aggregate_meta.json").read_text(encoding="utf-8"))[BUCKET]
    k, n = int(meta["K"]), int(meta["n"])
    assert k == -(-n // REFIT_EVERY), (k, n)
    tail = np.full(k, REFIT_EVERY)
    tail[-1] = n - REFIT_EVERY * (
        k - 1
    )  # the last refit forecasts the remaining sessions
    refits, days, i = {}, {}, 0
    for y, cnt in sorted((int(y), int(c)) for y, c in meta["years"].items()):
        refits[y], days[y] = cnt, int(tail[i : i + cnt].sum())
        i += cnt
    assert i == k and sum(days.values()) == n
    return refits, days


def pre_tail_span(n_days: int) -> tuple[str, str]:
    """First and last forecast session of the first n_days tail days (the design's dates)."""
    x = pd.read_parquet(DESIGN, columns=["date", "forecast_row"])
    f = pd.to_datetime(x.loc[x["forecast_row"] >= 0, "date"]).reset_index(drop=True)
    assert f.iloc[0] == pd.Timestamp(
        "2018-06-25 16:00"
    )  # the study's first forecast session
    return f.iloc[0].strftime("%Y-%m-%d"), f.iloc[n_days - 1].strftime("%Y-%m-%d")


def importance() -> None:
    files = sorted((FI / "by_measure").glob(f"imp_{BUCKET}_*.csv"))
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    s = d[(d["level"] == "series") & d["eligible"].astype(bool)].copy()
    cells = s.groupby(["model", "measure"])
    assert cells.ngroups == len(files) == N_CELLS, (cells.ngroups, len(files))
    n_series = cells.size().unique()
    assert len(n_series) == 1, n_series
    # one rank per series per cell, 1..n_series (the study's own ranking)
    assert (cells["rank"].nunique() == n_series[0]).all()
    # gate: ordering the all-period mean value reproduces the study's rank in every cell
    assert (rank_desc(s, "value") == s["rank"]).all()
    # gate: the yYYYY columns are tail-day-weighted means with the year_weights() weights
    pre_refits, w = year_weights()
    rebuilt = sum(wy * s[f"y{y}"] for y, wy in w.items()) / sum(w.values())
    scale = cells["value"].transform(lambda v: v.abs().max())
    assert (
        (rebuilt - s["value"]).abs() <= 1e-5 * scale
    ).all()  # the CSVs carry 6 digits
    s["pre"] = sum(w[y] * s[f"y{y}"] for y in PRE_YEARS) / sum(w[y] for y in PRE_YEARS)
    s["pre_rank"] = rank_desc(s, "pre")
    pre_tied = s.groupby(["model", "measure"])["pre"].agg(
        lambda v: v.duplicated().any()
    )
    r = s.pivot_table(index="unit", columns=["model", "measure"], values="rank")
    assert r.shape == (n_series[0], N_CELLS) and r.notna().all().all()
    pr = s.pivot_table(index="unit", columns=["model", "measure"], values="pre_rank")
    assert pr.shape == r.shape and pr.notna().all().all()
    label = s.groupby("unit")["label"].first()
    stats = pd.DataFrame(
        {
            "median": r.median(axis=1),
            "top": (r <= TOP).sum(axis=1),
            "best": r.min(axis=1),
            "worst": r.max(axis=1),
            "pre_median": pr.median(axis=1),
            "pre_top": (pr <= TOP).sum(axis=1),
        }
    ).sort_values(["median", "best"])
    macro("flPreRefits", str(sum(pre_refits[y] for y in PRE_YEARS)))
    macro("flPreTailDays", str(sum(w[y] for y in PRE_YEARS)))
    for y, word in PRE_YEARS.items():
        macro(f"flPreRefits{word}", str(pre_refits[y]))
        macro(f"flPreTailDays{word}", str(w[y]))
    first, last = pre_tail_span(sum(w[y] for y in PRE_YEARS))
    macro("flPreTailFirst", first)
    macro("flPreTailLast", last)
    macro("flPreTiedCells", str(int(pre_tied.sum())))
    stats["order"] = np.arange(
        1, len(stats) + 1
    )  # place by median rank among all series
    macro("flCells", str(N_CELLS))
    macro("flSeriesN", str(int(n_series[0])))
    macro("flModelsN", str(s["model"].nunique()))
    macro("flTop", str(TOP))
    rows = []

    def row(unit: str, what: str, tag: str | None) -> None:
        x = stats.loc[unit]
        if tag is not None:
            macro(
                f"flRankMed{tag}", num(x["median"], 0)
            )  # the median of 37 ranks: an integer
            macro(f"flTopTen{tag}", str(int(x["top"])))
            macro(f"flRankBest{tag}", str(int(x["best"])))
            macro(f"flPlace{tag}", str(int(x["order"])))
        if tag in PRE_TAGS:
            macro(f"flPreRank{tag}", num(x["pre_median"], 0))
            macro(f"flPreTopTen{tag}", str(int(x["pre_top"])))
        lean = r"\checkmark" if unit in LEAN_SERIES else ""
        rows.append(
            " & ".join(
                [
                    disp(label[unit]),
                    what,
                    lean,
                    num(x["median"], 0),
                    str(int(x["top"])),
                    num(x["pre_median"], 0),
                    str(int(x["pre_top"])),
                    str(int(x["best"])),
                    str(int(x["order"])),
                ]
            )
            + r" \\"
        )

    row(LEADER, "realized variance (the target's HAR ladder)", "Har")
    rows.append(r"\midrule")
    for unit, tag, what in CANDIDATES:
        assert unit in stats.index, unit
        row(unit, what, tag)
    rows.append(r"\midrule")
    known = {LEADER} | {u for u, _, _ in CANDIDATES}
    other = [u for u in stats.index if u not in known][:N_OTHER]
    for i, unit in enumerate(other):
        macro(f"flOther{ORDINAL[i]}Label", disp(label[unit]))
        row(unit, "best-ranked other series", f"Other{ORDINAL[i]}")
    for extra in (
        "vix3m",
        "stocktwits_sentiment",
    ):  # the comparison series the text names
        if extra not in other:
            tag = "Vixthreem" if extra == "vix3m" else "Stocktwits"
            row(extra, "comparison", tag)
    lean_med = stats.loc[list(LEAN_SERIES), "median"]
    macro(
        "flLeanInTopTenN",
        str(int((stats.loc[list(LEAN_SERIES), "order"] <= TOP).sum())),
    )
    macro("flLeanRankMedMax", num(lean_med.max(), 0))
    macro("flLeanN", str(len(LEAN_SERIES)))
    top_hdr = r"\shortstack{top-" + str(TOP) + r"\\cells}"
    head = [
        r" & & & \multicolumn{2}{c}{all tails, 2018--2024} & \multicolumn{2}{c}{pre-2020 tails} & & \\",
        r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"series & what & lean & \shortstack{median\\rank} & "
        + top_hdr
        + r" & \shortstack{median\\rank} & "
        + top_hdr
        + r" & \shortstack{best\\rank} & \shortstack{place by\\median} \\",
        r"\midrule",
    ]
    write_tab(
        IMP_OUT,
        [
            rel(FI / "by_measure")
            + f"/imp_{BUCKET}_<model>_<measure>.csv (level 'series', "
            f"{N_CELLS} cells: rank 1 = most important among {int(n_series[0])} series); "
            "pre-2020 = the y2018 / y2019 columns weighted by their tail days, re-ranked"
        ],
        "llcrrrrrr",
        head + rows,
    )

    # the absolute-return cluster, one per lag (clusters.csv, complete linkage at |corr| >= 0.8)
    c = pd.read_csv(FI / "clusters.csv")
    c = c[c["bucket"] == BUCKET]
    ab = c[c["series"].astype(str).map(lambda x: set(x.split(",")) == ABS_FAMILY)]
    assert len(ab) > 0
    corr = []
    for rr in ab.itertuples():
        lags = sorted({m.rsplit("_", 1)[1] for m in str(rr.members).split()}, key=int)
        tag = "".join(LAG_WORD[g] for g in lags)
        macro(f"flCluster{tag}Corr", num(rr.min_abs_corr, 2))
        corr.append(float(rr.min_abs_corr))
    macro("flClusterN", str(len(ab)))
    macro("flClusterCorrMin", num(min(corr), 2))
    macro("flClusterCorrMax", num(max(corr), 2))
    macro(
        "flClusterLagsN",
        str(
            sum(
                len({m.rsplit("_", 1)[1] for m in str(x).split()})
                for x in ab["members"]
            )
        ),
    )

    # pre-2020: min |corr| among the four absolute-return columns of each HAR lag
    cols = [f"{b}_ma_{k}" for k in LAG_WORD for b in ABS_COLS]
    x = pd.read_parquet(DESIGN, columns=["date", *cols])
    x = x[pd.to_datetime(x["date"]) < pd.Timestamp(PRE_CUTOFF)]
    assert len(x) and not x[cols].isna().any().any()
    macro("flPreClusterRows", str(len(x)))
    macro("flPreClusterFirst", pd.Timestamp(x["date"].min()).strftime("%Y-%m-%d"))
    macro("flPreClusterLast", pd.Timestamp(x["date"].max()).strftime("%Y-%m-%d"))
    for k, word in LAG_WORD.items():
        m = x[[f"{b}_ma_{k}" for b in ABS_COLS]].corr().abs().to_numpy()
        macro(
            f"flPreClusterMinCorrLag{word}",
            num(m[np.triu_indices(len(ABS_COLS), 1)].min(), 2),
        )


# --------------------------------------------------------------------------- (B) results
def group_of(family: str) -> str | None:
    if family == "per-bar linear":
        return "Linear"
    if family == "pooled twin":
        return "Pooled"
    if family in ("per-bar tree (untuned)", "per-bar tree (untuned, daily refit)"):
        return "TreeShip"
    if family in ("per-bar tree (tuned)", "per-bar tree (random search, daily)"):
        return "TreeRs"
    if family.startswith("per-bar tree (Optuna"):
        return "TreeOpt"
    return None


def results() -> None:
    import atm_straddle_lib as asl
    import master_table_close as mtc
    from src.evaluation.diebold_mariano import dm_test

    master = pd.read_csv(MASTER).set_index("key")
    ms = camp.paper_model_set(master)
    lean = [k for k in ms.index if camp.input_set(k) == camp.LEAN]
    macro("flLeanForecastsN", str(len(lean)))
    head = [
        r" & & \multicolumn{3}{c}{QLIKE (median)} & \multicolumn{3}{c}{Sharpe mid (median)}"
        r" & \multicolumn{2}{c}{lean $-$ \texttt{all\_features}} & \multicolumn{2}{c}{lean $-$ \texttt{baseline}} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-10}\cmidrule(lr){11-12}",
        r"family & pairs & lean & all & base & lean & all & base & \shortstack{QLIKE lower\\(DM sig.)}"
        r" & \shortstack{Sharpe higher\\(HAC sig.)} & \shortstack{QLIKE lower\\(DM sig.)}"
        r" & \shortstack{Sharpe higher\\(HAC sig.)} \\",
        r"\midrule",
    ]
    src = [
        rel(MASTER),
        rel(DAILY)
        + " (DM of the daily QLIKE difference, HAC t of the daily mid-return difference)",
    ]
    if not lean:
        print(
            "WARNING: the master table has no lean_vol_vix forecast yet -- part B (results) is empty"
        )
        write_tab(
            RES_OUT,
            src,
            "lrrrrrrrrrrr",
            head
            + [
                r"\multicolumn{12}{l}{\emph{no lean forecast in the master table yet}} \\"
            ],
        )
        return
    daily = pd.read_parquet(DAILY)
    frames = {k: g.set_index("day").sort_index() for k, g in daily.groupby("key")}
    rows = []
    missing = []
    for k in lean:
        g = group_of(str(ms.loc[k, "family"]))
        if g is None:
            missing.append(k)
            continue
        rec = {"key": k, "group": g, "family": ms.loc[k, "family"]}
        for tag, b in [("Lean", camp.LEAN)] + [(t, bb) for t, bb in VS]:
            kk = k.replace(camp.LEAN, b)
            assert kk in ms.index, (k, kk, "the twin must be in the model set")
            f = frames[kk]
            # gate: the daily frame reproduces master_table.csv
            assert (
                abs(mtc.sharpe(f["ret_mid"].to_numpy()) - ms.loc[kk, "Sharpe_mid"])
                < GATE_TOL
            ), kk
            assert (
                abs(float(f["qlike_raw"].mean()) - ms.loc[kk, "qlike_raw"]) < GATE_TOL
            ), kk
            rec[f"q{tag}"] = float(ms.loc[kk, "qlike_raw"])
            rec[f"sh{tag}"] = float(ms.loc[kk, "Sharpe_mid"])
            rec[f"shx{tag}"] = float(ms.loc[kk, "Sharpe_crossed"])
            if tag != "Lean":
                a, o = frames[k], f
                assert (a.index == o.index).all()
                dm = dm_test(a["qlike_raw"].to_numpy(), o["qlike_raw"].to_numpy())
                t = asl.newey_west_t(a["ret_mid"].to_numpy() - o["ret_mid"].to_numpy())[
                    0
                ]
                rec[f"dq{tag}"] = 100.0 * (rec["qLean"] / rec[f"q{tag}"] - 1.0)
                rec[f"dm{tag}"] = float(dm["dm"])
                rec[f"dmp{tag}"] = float(dm["p"])
                rec[f"dsh{tag}"] = rec["shLean"] - rec[f"sh{tag}"]
                rec[f"t{tag}"] = float(t)
                rec[f"tp{tag}"] = mtc.hac_p(t)
        rows.append(rec)
    if missing:
        print(
            f"WARNING: lean forecasts outside the five family groups, left out: {missing}"
        )
    p = pd.DataFrame(rows)
    sig = camp.SIG
    body = []
    for g, name in GROUPS + [("Set", r"\emph{all}")]:
        x = p if g == "Set" else p[p["group"] == g]
        if not len(x):
            continue
        macro(f"fl{g}N", str(len(x)))
        for tag in ("Lean", "AllF", "Base"):
            macro(f"fl{g}Q{tag}", num(med(x[f"q{tag}"]), 4))
            macro(f"fl{g}Sh{tag}", num(med(x[f"sh{tag}"])))
            macro(f"fl{g}ShX{tag}", num(med(x[f"shx{tag}"])))
        cells = [name, str(len(x))]
        cells += [num(med(x[f"q{t}"]), 4) for t in ("Lean", "AllF", "Base")]
        cells += [num(med(x[f"sh{t}"])) for t in ("Lean", "AllF", "Base")]
        for tag, _ in VS:
            ql, qs = x[f"dq{tag}"] < 0, x[f"dmp{tag}"] < sig
            sh, ss = x[f"dsh{tag}"] > 0, x[f"tp{tag}"] < sig
            v = f"Vs{tag}"
            macro(f"fl{g}{v}QPctMed", num(med(x[f"dq{tag}"]), 1, True))
            macro(f"fl{g}{v}ShMed", num(med(x[f"dsh{tag}"]), 2, True))
            macro(f"fl{g}{v}QBetter", str(int(ql.sum())))
            macro(f"fl{g}{v}QBetterSig", str(int((ql & qs).sum())))
            macro(f"fl{g}{v}QWorse", str(int((~ql).sum())))
            macro(f"fl{g}{v}QWorseSig", str(int((~ql & qs).sum())))
            macro(f"fl{g}{v}ShAbove", str(int(sh.sum())))
            macro(f"fl{g}{v}ShAboveSig", str(int((sh & ss).sum())))
            macro(f"fl{g}{v}ShBelow", str(int((~sh).sum())))
            macro(f"fl{g}{v}ShBelowSig", str(int((~sh & ss).sum())))
            cells += [
                f"{int(ql.sum())} ({int((ql & qs).sum())})",
                f"{int(sh.sum())} ({int((sh & ss).sum())})",
            ]
        if g == "Set":
            body.append(r"\midrule")
        body.append(" & ".join(cells) + r" \\")
    write_tab(RES_OUT, src, "lrrrrrrrrrrr", head + body)


# --------------------------------------------------------------------------- main
def main() -> None:
    GEN.mkdir(parents=True, exist_ok=True)
    importance()
    results()
    lines = [
        "% AUTO-GENERATED by writeup/make_featimp_lean_tex.py -- do not edit.",
        "% The lean set (lean_vol_vix) as a follow-up to the feature-importance study (2026-10-06):",
        "% importance ranks (results/feature_importance_1530_dedup/, all_features, series level) and",
        "% lean vs all_features vs baseline over the paper model set (point estimates; DM / HAC t).",
    ]
    lines += [rf"\newcommand{{\{k}}}{{{v}}}" for k, v in NUMS.items()]
    NUMBERS_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(
        f"wrote {rel(NUMBERS_OUT)} ({len(NUMS)} macros), {rel(IMP_OUT)}, {rel(RES_OUT)}"
    )


if __name__ == "__main__":
    main()
