"""The bucket story check (feature MCS study, 2026-10-13) for the paper: macros and one table.

QUESTION (the professor's story, user 2026-10-12): "each individual bucket didn't see much improvement,
but all features together saw a discernible difference" over HAR + calendar.  Tested by
experiments/feature_mcs_buckets.py (designs, losses, pairwise and joint tests) and summarised in
results/feature_mcs_2026-10-08/SUMMARY.md, section "The bucket story check".  Nothing is re-estimated
here: every number is read from the study's CSVs (the source of truth), and every qualitative claim
the appendix text makes is asserted below.

SOURCES (results/feature_mcs_2026-10-08/)
  buckets_pairwise.csv   per family (6 estimators, 4 pools) and design: mean QLIKE / winsorized QLIKE /
                         fit-space MSE and their DM against the reference; sign(s) Sharpe mid / crossed of
                         design and reference; Newey-West HAC t of the daily mid-P&L difference
  buckets_stepm.csv      Romano-Wolf StepM per family, loss (qlike, qlike_w, trade), test (A: beat HAR +
                         calendar; B: removing helps; C: removing hurts), block (5, 10, 21), FWER (5 %, 10 %)
  buckets_spa.csv        Hansen's SPA (consistent p) for the same cells
  buckets_verdicts.csv   the story check per family and loss (feature_mcs_report.bucket_verdicts: StepM at
                         FWER 5 % naming a design at EVERY block length)
  buckets_gates.csv      the study's gates (all must be ok)
  bucket_losses.parquet  daily losses of the 16 bucket designs (the day count)
  mcs_sets.csv           the MCS over the study's 576 single-series candidates (context sentence)
The estimators' settings and the bootstrap constants are read from experiments/feature_mcs_report.py and
experiments/feature_mcs_spa.py; the buckets from src/data/loading.py SUBGROUPS.

LSTM: the study's estimators are the pre-tuned ridge / lasso / elastic net and the three trees; no LSTM
is in any estimator set or pool (asserted below).

Writes writeup/generated/buckets_numbers.tex (\\bk... macros) and writeup/generated/table_buckets_story.tex.

Run:  python writeup/make_buckets_tex.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (
    ROOT,
    ROOT / "experiments",
    ROOT / "notebooks",
    ROOT / "specs",
    ROOT / "writeup",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import feature_mcs_report as R  # noqa: E402
import feature_mcs_spa as SP  # noqa: E402

from src.data.loading import SUBGROUPS  # noqa: E402

RES = ROOT / "results" / "feature_mcs_2026-10-08"
GEN = ROOT / "writeup" / "generated"
OUT_NUM = GEN / "buckets_numbers.tex"
OUT_TAB = GEN / "table_buckets_story.tex"
SIG = 1.96  # two-sided 5 % for DM and HAC t, the master table's convention
FW = 0.05  # StepM family-wise error rate of the verdict

BUCKETS = (
    "moments",
    "liquidity",
    "market_ew",
    "market_vw",
    "sentiment",
    "implied_vol",
    "vol_demand",
    "fomc",
)
BTAG = {
    "moments": "Mom",
    "liquidity": "Liq",
    "market_ew": "Mew",
    "market_vw": "Mvw",
    "sentiment": "Sent",
    "implied_vol": "Iv",
    "vol_demand": "Vd",
    "fomc": "Fomc",
}
FAMS = (
    "enet",
    "ridge",
    "lasso",
    "lgbm",
    "xgb",
    "rf",
    "pool_linear",
    "pool_trees",
    "pool_all",
    "pool_5",
)
FAM_NAME = {
    "enet": "elastic net",
    "ridge": "ridge",
    "lasso": "lasso",
    "lgbm": "LightGBM",
    "xgb": "XGBoost",
    "rf": "random forest",
    "pool_linear": "linear pool",
    "pool_trees": "tree pool",
    "pool_all": "pool of all six",
    "pool_5": "pool, no LightGBM",
}
LOSS_NAME = {"qlike": "QLIKE", "qlike_w": "winsorized QLIKE", "trade": "the trade"}
LOSS_TAG = {"qlike": "Q", "qlike_w": "Q$_w$", "trade": "T"}  # the table's StepM column

MACROS: list[tuple[str, str, str]] = []


def macro(name: str, val: str, note: str = "") -> None:
    assert name.isalpha(), name
    assert name not in {m[0] for m in MACROS}, name
    MACROS.append((name, val, note))


def fnum(x: float, d: int = 2, plus: bool = False) -> str:
    s = f"{abs(x):.{d}f}"
    if x < 0 and float(s) != 0.0:
        return r"\ensuremath{-}" + s
    return (r"\ensuremath{+}" + s) if plus else s


def fint(x: int) -> str:
    return f"{int(x):,}".replace(",", "{,}")


def tt(s: str) -> str:
    return r"\texttt{" + s.replace("_", r"\_") + "}"


def words(n: int) -> str:
    return {
        0: "none",
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
    }.get(n, str(n))


def main() -> None:  # noqa: C901
    pw = pd.read_csv(RES / "buckets_pairwise.csv")
    st = pd.read_csv(RES / "buckets_stepm.csv")
    sp = pd.read_csv(RES / "buckets_spa.csv")
    vd = pd.read_csv(RES / "buckets_verdicts.csv")
    gg = pd.read_csv(RES / "buckets_gates.csv")
    bl = pd.read_parquet(RES / "bucket_losses.parquet", columns=["est", "model", "day"])

    # ---- gates and the estimator sets (no LSTM anywhere)
    assert gg["ok"].astype(bool).all(), "a bucket-study gate failed"
    assert set(pw["family"]) == set(FAMS), sorted(set(pw["family"]))
    members = pw.groupby("family")["members"].first()
    for f in FAMS:
        assert "lstm" not in str(members[f]).lower(), (f, members[f])
    est = members[[f for f in FAMS if not f.startswith("pool")]].tolist()
    assert est == [f for f in FAMS if not f.startswith("pool")]
    assert members["pool_all"].split(",") == list(R.SHORTS)
    assert members["pool_5"].split(",") == [*R.LIN, "xgb", "rf"]
    assert members["pool_linear"].split(",") == list(R.LIN)
    assert members["pool_trees"].split(",") == list(R.TREES)

    # ---- design
    sizes = {b: len(SUBGROUPS[b]) for b in BUCKETS}
    flat = sum((list(SUBGROUPS[b]) for b in BUCKETS), [])
    assert len(flat) == len(set(flat)), "the buckets overlap"
    macro("bkNBuckets", str(len(BUCKETS)), "buckets")
    macro("bkNSeries", str(len(flat)), "series of all_features the buckets partition")
    for b in BUCKETS:
        macro(f"bkSize{BTAG[b]}", str(sizes[b]), f"series in bucket {b}")
    assert {"ofi_ewstock", "ofi_vwstock"} <= set(SUBGROUPS["liquidity"])
    macro("bkNEst", str(len(R.SHORTS)), "estimators")
    n_designs = 2 * len(BUCKETS) + 2
    macro(
        "bkNDesigns",
        str(n_designs),
        "designs per estimator: 8 add-bucket, 8 leave-bucket-out, 2 anchors",
    )
    days = bl["day"].nunique()
    assert (bl.groupby(["est", "model"])["day"].nunique() == days).all()
    macro("bkDays", str(days), "trade days")
    macro("bkBoot", fint(R.B_BOOT), "bootstrap replicates (circular block)")
    assert tuple(R.BLOCKS) == (5, 10, 21)
    macro("bkBlockA", str(R.BLOCKS[0]))
    macro("bkBlockB", str(R.BLOCKS[1]))
    macro("bkBlockC", str(R.BLOCKS[2]))
    assert FW in SP.FWER
    macro("bkFwer", f"{100 * FW:.0f}", "StepM family-wise error rate of the verdict, %")
    macro("bkSigPct", "5", "two-sided level of the pairwise DM and HAC t (|.| > 1.96)")
    assert abs(SIG - 1.96) < 1e-12
    macro("bkNPools", str(sum(f.startswith("pool") for f in FAMS)), "pools")
    macro("bkNFam", str(len(FAMS)), "families: six estimators and four pools")

    # ---- the verdicts (StepM, FWER 5 %, test A, at every block length)
    assert len(vd) == len(FAMS) * 3
    macro("bkCells", str(len(vd)), "family x loss cells of the verdict")
    holds = (vd["verdict"] == "holds").sum()
    assert holds == 0, "the story holds somewhere: rewrite the appendix"
    macro("bkHolds", words(int(holds)), "cells where the story holds")
    all_beats = vd["all_features_beats_har"].astype(bool).sum()
    assert all_beats == 0
    macro(
        "bkAllBeats",
        words(int(all_beats)),
        "cells where all features beats HAR + cal. (StepM, every block)",
    )
    # stronger: StepM names all_features in test A at NO block length and at neither FWER
    a_any = st[st["test"] == "A: beat HAR + calendar"]
    named_all = a_any["rejected"].map(lambda s: "all_features" in json.loads(s)).sum()
    assert named_all == 0
    vb = vd[vd["buckets_beating_har"] != "[]"]
    named = sorted({x for s in vb["buckets_beating_har"] for x in json.loads(s)})
    assert named == ["addb_market_vw"], named
    assert set(vb["family"]) <= {"enet", "ridge", "lasso", "pool_linear"}, vb
    macro(
        "bkBucketCells",
        words(len(vb)),
        "cells where one bucket beats HAR + cal. (all market_vw, linear)",
    )
    vb_list = ", ".join(
        f"{FAM_NAME[f]} on {LOSS_NAME[lo]}" for f, lo in zip(vb["family"], vb["loss"])
    )
    macro(
        "bkBucketCellsWho",
        "the "
        + vb_list.replace(", lasso", ", the lasso").replace(", linear", ", the linear"),
        "those cells",
    )
    vc = vd[vd["removal_hurts"] != "[]"]
    named_c = sorted({x for s in vc["removal_hurts"] for x in json.loads(s)})
    assert named_c == ["loob_market_vw"], named_c
    assert set(vc["family"]) <= {"enet", "ridge", "lasso", "pool_linear"}
    macro(
        "bkRemoveCells",
        words(len(vc)),
        "cells where StepM (test C, every block) says removing a bucket hurts",
    )

    def P(fam: str, design: str) -> pd.Series:
        r = pw[(pw["family"] == fam) & (pw["design"] == design)]
        assert len(r) == 1, (fam, design)
        return r.iloc[0]

    # ---- all features against HAR + calendar, every family: never significant
    allf = pw[pw["design"] == "all_features"].set_index("family").loc[list(FAMS)]
    assert (allf["dm_qlike"] > -SIG).all() and (allf["dm_qlike_w"] > -SIG).all()
    assert (allf["hac_t_mid"] < SIG).all()
    i = allf["dm_qlike"].idxmin()
    macro(
        "bkAllDMMin",
        fnum(allf.loc[i, "dm_qlike"]),
        f"most negative DM QLIKE, all features vs HAR + cal. ({i})",
    )
    macro("bkAllDMMinWho", FAM_NAME[i])
    j = allf["hac_t_mid"].idxmax()
    macro(
        "bkAllTMax",
        fnum(allf.loc[j, "hac_t_mid"]),
        f"largest HAC t, all features vs HAR + cal. ({j})",
    )
    macro("bkAllTMaxWho", FAM_NAME[j])
    macro(
        "bkAllShHigher",
        str(int((allf["sharpe_mid"] > allf["sharpe_mid_ref"]).sum())),
        "families where all features trades higher",
    )

    # ---- linear models on QLIKE: market_vw
    r = P("lasso", "addb_market_vw")
    assert r["dm_qlike"] < -SIG
    macro(
        "bkLassoVwDQ",
        fnum(r["dpct_qlike"], 1),
        "lasso + market_vw vs HAR + cal., QLIKE change %",
    )
    macro("bkLassoVwDM", fnum(r["dm_qlike"]))
    p = sp[
        (sp["family"] == "lasso")
        & (sp["loss"] == "qlike")
        & (sp["test"] == "A: beat HAR + calendar")
    ]
    assert (p["best"] == "addb_market_vw").all()
    macro(
        "bkLassoVwSpaMax",
        f"{p['p_consistent'].max():.3f}",
        "lasso, SPA consistent p (test A, QLIKE), largest over blocks",
    )
    r = P("pool_linear", "addb_market_vw")
    assert r["dm_qlike"] < -SIG and r["dm_qlike_w"] < -SIG
    macro(
        "bkPoolLinVwDM",
        fnum(r["dm_qlike"]),
        "linear pool + market_vw vs HAR + cal., DM QLIKE",
    )
    macro("bkPoolLinVwDMw", fnum(r["dm_qlike_w"]), "... DM winsorized QLIKE")
    r = P("pool_linear", "all_features")
    macro(
        "bkPoolLinAllDM",
        fnum(r["dm_qlike"]),
        "linear pool, all features vs HAR + cal., DM QLIKE",
    )
    macro("bkPoolLinAllDMw", fnum(r["dm_qlike_w"]), "... DM winsorized QLIKE")
    r = P("pool_linear", "loob_market_vw")
    assert r["dm_qlike"] > SIG and r["dm_qlike_w"] > SIG
    macro(
        "bkPoolLinVwOutDM",
        fnum(r["dm_qlike"], 2, True),
        "linear pool, all features - market_vw vs all features, DM QLIKE",
    )
    macro("bkPoolLinVwOutDMw", fnum(r["dm_qlike_w"], 2, True), "... winsorized")
    # removing market_vw is the only removal that significantly hurts a linear QLIKE (pairwise, raw QLIKE)
    lin_out = pw[
        pw["family"].isin(R.LIN)
        & (pw["comparison"] == "leave bucket out vs all features")
    ]
    hurt = lin_out[lin_out["dm_qlike"] > SIG]
    assert set(hurt["design"]) <= {"loob_market_vw", "loob_implied_vol"}, hurt[
        ["family", "design"]
    ]
    # liquidity added alone makes the linear models worse
    liq = pw[pw["family"].isin(R.LIN) & (pw["design"] == "addb_liquidity")]
    assert (liq["dpct_qlike"] > 0).all() and (liq["dm_qlike"] >= SIG - 0.005).all()
    macro(
        "bkLiqLinDQMin",
        fnum(liq["dpct_qlike"].min(), 1),
        "linear + liquidity vs HAR + cal., QLIKE change %, smallest",
    )
    macro("bkLiqLinDQMax", fnum(liq["dpct_qlike"].max(), 1), "... largest")
    macro("bkLiqLinDMMin", fnum(liq["dm_qlike"].min(), 2, True), "... DM smallest")
    macro("bkLiqLinDMMax", fnum(liq["dm_qlike"].max(), 2, True), "... DM largest")

    # ---- trees on the trade: single buckets, pairwise
    for fam, tag in (("xgb", "Xgb"), ("rf", "Rf"), ("lgbm", "Lgbm")):
        b0 = P(fam, "all_features")
        macro(
            f"bk{tag}BaseSh",
            fnum(b0["sharpe_mid_ref"]),
            f"{fam}: HAR + cal. Sharpe mid",
        )
        macro(
            f"bk{tag}AllSh", fnum(b0["sharpe_mid"]), f"{fam}: all features Sharpe mid"
        )
        macro(
            f"bk{tag}AllT",
            fnum(b0["hac_t_mid"]),
            f"{fam}: all features vs HAR + cal., HAC t",
        )
        for b in ("moments", "liquidity", "market_ew"):
            rb = P(fam, f"addb_{b}")
            macro(
                f"bk{tag}{BTAG[b]}Sh",
                fnum(rb["sharpe_mid"]),
                f"{fam} + {b}: Sharpe mid",
            )
            macro(
                f"bk{tag}{BTAG[b]}T",
                fnum(rb["hac_t_mid"]),
                f"{fam} + {b} vs HAR + cal.: HAC t",
            )
    assert (
        P("xgb", "addb_liquidity")["hac_t_mid"] > SIG
        and P("xgb", "addb_market_ew")["hac_t_mid"] > SIG
    )
    assert (
        P("rf", "addb_moments")["hac_t_mid"] > SIG
        and P("rf", "addb_market_ew")["hac_t_mid"] > SIG
    )
    for fam in (
        *R.TREES,
        "pool_trees",
    ):  # the trees' best single bucket (largest HAC t) trades above all features
        add = pw[
            (pw["family"] == fam) & (pw["comparison"] == "add bucket vs HAR + calendar")
        ]
        assert (
            add.loc[add["hac_t_mid"].idxmax(), "sharpe_mid"]
            > P(fam, "all_features")["sharpe_mid"]
        ), fam
    for fam in (
        "xgb",
        "rf",
    ):  # each significant single bucket trades above all features
        a = P(fam, "all_features")["sharpe_mid"]
        for b in (
            ("liquidity", "market_ew") if fam == "xgb" else ("moments", "market_ew")
        ):
            assert P(fam, f"addb_{b}")["sharpe_mid"] > a
    # tree estimators' pairwise-significant single-bucket trade gains
    tr = pw[
        pw["family"].isin(R.TREES)
        & (pw["comparison"] == "add bucket vs HAR + calendar")
    ]
    sigtr = tr[tr["hac_t_mid"] > SIG]
    macro(
        "bkTreeTradeSigN",
        words(len(sigtr)),
        "tree x bucket trade gains with HAC t > 1.96",
    )
    assert set(sigtr["family"]) == {"xgb", "rf"} and set(sigtr["design"]) <= {
        "addb_moments",
        "addb_liquidity",
        "addb_market_ew",
    }
    # StepM on XGBoost's trade: market_ew named at some block lengths, not all
    xs = st[
        (st["family"] == "xgb")
        & (st["loss"] == "trade")
        & (st["test"] == "A: beat HAR + calendar")
        & (st["fwer"] == FW)
    ]
    blk_mew = sorted(
        int(b)
        for b, s in zip(xs["block"], xs["rejected"])
        if "addb_market_ew" in json.loads(s)
    )
    assert 0 < len(blk_mew) < len(R.BLOCKS), blk_mew
    macro(
        "bkXgbMewStepmN",
        words(len(blk_mew)),
        "block lengths at which StepM (5 %) names XGBoost + market_ew",
    )
    macro("bkXgbMewStepmBlocks", " and ".join(str(b) for b in blk_mew))
    miss = [b for b in R.BLOCKS if b not in blk_mew]
    macro(
        "bkXgbMewStepmMiss",
        " and ".join(str(b) for b in miss),
        "block lengths at which it is not named",
    )
    # LightGBM: StepM rejects nothing anywhere
    assert (st[st["family"] == "lgbm"]["n_rejected"] == 0).all()
    lg = pw[
        (pw["family"] == "lgbm") & (pw["comparison"] == "add bucket vs HAR + calendar")
    ]
    lgs = lg[lg["dm_qlike"] < -SIG]
    assert list(lgs["design"]) == ["addb_moments"], lgs
    assert (lg["hac_t_mid"] < SIG).all()
    macro(
        "bkLgbmMomDM",
        fnum(lgs.iloc[0]["dm_qlike"]),
        "LightGBM + moments vs HAR + cal., DM QLIKE",
    )

    # ---- the pools
    r = P("pool_all", "all_features")
    macro(
        "bkPoolAllDM",
        fnum(r["dm_qlike"]),
        "six-estimator pool, all features vs HAR + cal., DM QLIKE",
    )
    macro("bkPoolAllSh", fnum(r["sharpe_mid"]))
    macro("bkPoolAllBaseSh", fnum(r["sharpe_mid_ref"]))
    macro("bkPoolAllT", fnum(r["hac_t_mid"]))
    assert abs(r["hac_t_mid"]) < SIG and abs(r["dm_qlike"]) < SIG
    r = P("pool_5", "loob_liquidity")
    assert r["hac_t_mid"] > SIG and r["sharpe_mid"] > r["sharpe_mid_ref"]
    macro(
        "bkPoolFiveLiqOutSh",
        fnum(r["sharpe_mid"]),
        "five-estimator pool, all features - liquidity: Sharpe mid",
    )
    macro("bkPoolFiveAllSh", fnum(r["sharpe_mid_ref"]), "... all features: Sharpe mid")
    macro("bkPoolFiveLiqOutT", fnum(r["hac_t_mid"]), "... HAC t")
    sb = st[
        (st["family"] == "pool_5")
        & (st["loss"] == "trade")
        & (st["test"] == "B: removing helps")
        & (st["fwer"] == FW)
    ]
    assert len(sb) == len(R.BLOCKS) and all(
        "loob_liquidity" in json.loads(s) for s in sb["rejected"]
    )
    p = sp[
        (sp["family"] == "pool_5")
        & (sp["loss"] == "trade")
        & (sp["test"] == "B: removing helps")
    ]
    assert (p["best"] == "loob_liquidity").all()
    macro(
        "bkPoolFiveLiqOutSpaMax",
        f"{p['p_consistent'].max():.3f}",
        "SPA consistent p (test B, trade), largest over blocks",
    )
    # the trees' pool on the trade: single buckets, pairwise
    for b in ("moments", "market_ew"):
        rb = P("pool_trees", f"addb_{b}")
        assert rb["hac_t_mid"] > SIG
        macro(
            f"bkPoolTree{BTAG[b]}T",
            fnum(rb["hac_t_mid"]),
            f"tree pool + {b} vs HAR + cal.: HAC t",
        )
    macro(
        "bkPoolTreeAllT",
        fnum(P("pool_trees", "all_features")["hac_t_mid"]),
        "tree pool, all features: HAC t",
    )

    # ---- the MCS over the study's single-series candidates (context): HAR + calendar never excluded
    ms = pd.read_csv(RES / "mcs_sets.csv")
    g = ms[(ms["scope"] == "grand") & ms["loss"].isin(["qlike", "qlike_w", "trade"])]
    anchors = [f"{e}:baseline" for e in R.SHORTS]
    assert all(set(anchors) <= set(json.loads(s)) for s in g["models"]), (
        "an MCS excluded HAR + calendar"
    )
    macro("bkMcsN", str(int(g["n"].iloc[0])), "candidates of the grand MCS")
    macro(
        "bkMcsSets",
        str(len(g)),
        "grand MCS sets on QLIKE, winsorized QLIKE and trade (blocks x stats x levels)",
    )

    # ---- the table: one row per family
    rows = []
    for f in FAMS:
        a = P(f, "all_features")
        add = pw[
            (pw["family"] == f) & (pw["comparison"] == "add bucket vs HAR + calendar")
        ]
        bq = add.loc[add["dm_qlike"].idxmin()]
        bt = add.loc[add["hac_t_mid"].idxmax()]
        v = vd[vd["family"] == f].set_index("loss")
        named_a = sorted(
            {
                (x, lo)
                for lo in v.index
                for x in json.loads(v.loc[lo, "buckets_beating_har"])
            }
        )
        named_c = sorted(
            {(x, lo) for lo in v.index for x in json.loads(v.loc[lo, "removal_hurts"])}
        )
        sbb = st[
            (st["family"] == f)
            & (st["test"] == "B: removing helps")
            & (st["fwer"] == FW)
        ]
        named_b = sorted(
            {
                (x, lo)
                for lo, gb in sbb.groupby("loss")
                for x in set.intersection(*[set(json.loads(s)) for s in gb["rejected"]])
            }
        )

        def lab(pairs, test):
            by: dict[str, list[str]] = {}
            for x, lo in pairs:
                by.setdefault(x.split("_", 1)[1], []).append(LOSS_TAG[lo])
            return "; ".join(
                f"{test}: {tt(b)} ({', '.join(ls)})" for b, ls in by.items()
            )

        parts = [
            x for x in (lab(named_a, "A"), lab(named_b, "B"), lab(named_c, "C")) if x
        ]
        # one test per line (a left-aligned stack), so the column stays narrow
        stepm = (
            parts[0]
            if len(parts) == 1
            else (
                r"\begin{tabular}[t]{@{}l@{}}" + r"\\".join(parts) + r"\end{tabular}"
                if parts
                else ""
            )
        )
        rows.append(
            [
                FAM_NAME[f],
                fnum(a["dpct_qlike"], 1),
                fnum(a["dm_qlike"]),
                f"{tt(bq['design'].split('_', 1)[1])} {fnum(bq['dm_qlike'])}",
                fnum(a["sharpe_mid_ref"]),
                f"{fnum(a['sharpe_mid'])} ({fnum(a['hac_t_mid'])})",
                f"{tt(bt['design'].split('_', 1)[1])} {fnum(bt['sharpe_mid'])} ({fnum(bt['hac_t_mid'])})",
                stepm if stepm else "---",
            ]
        )
    head = [
        "% AUTO-GENERATED by writeup/make_buckets_tex.py -- do not edit.",
        "% Source: results/feature_mcs_2026-10-08/buckets_pairwise.csv, buckets_stepm.csv, buckets_verdicts.csv",
        r"\begin{tabular}{lrrlrlll}",
        r"\toprule",
        r" & \multicolumn{3}{c}{QLIKE} & \multicolumn{3}{c}{trade: Sharpe (HAC $t$)} & \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"family & \shortstack{all features\\$\Delta$\,\%} & \shortstack{all features\\DM} & "
        r"\shortstack{best bucket\\(DM)} & \shortstack{HAR\\+ cal.} & \shortstack{all\\features} & "
        r"\shortstack{best bucket} & \shortstack[l]{StepM, every\\block length} \\",
        r"\midrule",
    ]
    body = [" & ".join(r) + r" \\" for r in rows]
    for k, r in enumerate(body):
        if k == 6:
            body[k] = r"\midrule" + "\n" + r
    tail = [r"\bottomrule", r"\end{tabular}"]
    GEN.mkdir(parents=True, exist_ok=True)
    OUT_TAB.write_text(
        "\n".join(head + body + tail) + "\n", encoding="utf-8", newline="\n"
    )

    lines = [
        "% AUTO-GENERATED by writeup/make_buckets_tex.py -- do not edit.",
        "% The bucket story check (experiments/feature_mcs_buckets.py), read from",
        "% results/feature_mcs_2026-10-08/buckets_{pairwise,stepm,spa,verdicts,gates}.csv, bucket_losses.parquet",
        "% and mcs_sets.csv.  Point estimates; DM and HAC t significant at |.| > 1.96; StepM at FWER 5 %.",
    ]
    for n, v, note in MACROS:
        lines.append(r"\newcommand{\%s}{%s}" % (n, v) + (f"  % {note}" if note else ""))
    OUT_NUM.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT_NUM} ({len(MACROS)} macros) and {OUT_TAB}")


if __name__ == "__main__":
    main()
