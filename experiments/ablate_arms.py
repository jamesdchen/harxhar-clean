"""Arms of the predictor-group ablations and augmented-HAR benchmarks (results/ablations_2014start_w3000/RULE.txt).

Shared by experiments/ablate_design.py (column-subset design caches, the new-series design), experiments/ablate_linear.py
(the linear walks), experiments/ablate_trees.py (the tree walks) and experiments/ablate_score.py (scores).

An arm is a list of column names of a design cache:
  full      the final full set (564 columns), the template
  har       the HAR benchmark set (12 columns: har_ma_k, DOW_0 .. DOW_4, hour), the template
  lo_<g>    the full set without group g's columns (a series' 6 value and 6 availability averages)
  add_<g>   the HAR benchmark set plus group g's columns
  add_<g1>_<g2>   the HAR benchmark set plus the columns of groups g1 and g2 (COMBO; 2026-10-26 addendum F: add_rm_fomc)
  shar, harq, lhar, sharql   classical HAR extensions (value averages of the new series of bucket abl_new)
Column order: the cache's own order (har ladder, value ladders, availability ladders, calendar).
"""

from __future__ import annotations

import re

HAR_K = (1, 5, 25, 125, 625, 3125)
HAR_COLS = [f"har_ma_{k}" for k in HAR_K]
CAL_COLS = [f"DOW_{d}" for d in range(5)] + ["hour"]
HAR12 = HAR_COLS + CAL_COLS

# the paper's 8 groups = src/data/loading.py SUBGROUPS (asserted against it in group_series)
GROUPS = {
    "rm": "moments",
    "liq": "liquidity",
    "ew": "market_ew",
    "vw": "market_vw",
    "sent": "sentiment",
    "iv": "implied_vol",
    "vd": "vol_demand",
    "fomc": "fomc",
}
GROUP_LABEL = {
    "rm": "Realized moments",
    "liq": "Liquidity",
    "ew": "Equal-weighted market",
    "vw": "Value-weighted market",
    "sent": "Sentiment",
    "iv": "Implied volatility",
    "vd": "Volatility demand",
    "fomc": "FOMC",
}
N_SERIES = {"rm": 7, "liq": 14, "ew": 6, "vw": 6, "sent": 3, "iv": 3, "vd": 4, "fomc": 3}
LATE_GROUPS = ("sent", "iv", "vd")  # series that start after the data (LATE_START_WARMUP)

# the new series of bucket abl_new (RULE.txt section 2): the first three are reference series (= the full design's)
NEW_REF = ["sumret", "sumret4", "sumpret2"]
NEW_SERIES = ["sumnret2", "rvsqrq", "negret"]
ABL_NEW = NEW_REF + NEW_SERIES
CLASSICAL = {
    "shar": ["adj_sumpret2_ma_1", "adj_sumnret2_ma_1"],
    "harq": ["adj_rvsqrq_ma_1"],
    "lhar": [f"adj_negret_ma_{k}" for k in HAR_K],
}
CLASSICAL["sharql"] = CLASSICAL["shar"] + CLASSICAL["harq"] + CLASSICAL["lhar"]
CLASSICAL_LABEL = {
    "shar": "SHAR (RS$^{+}$, RS$^{-}$)",
    "harq": "HARQ (RV$\\cdot$RQ$^{1/2}$)",
    "lhar": "LHAR (leverage)",
    "sharql": "SHAR + HARQ + LHAR",
}

LO = [f"lo_{g}" for g in GROUPS]
ADD = [f"add_{g}" for g in GROUPS]
GROUP_ARMS = LO + ADD
TREE_ARMS = GROUP_ARMS
LINEAR_ARMS = GROUP_ARMS + list(CLASSICAL)
OLS_ARMS = ["add_rm"] + list(CLASSICAL)
# the forest (RULE.txt section 3): the add arms and lo_rm, subject to the projection
RF_ARMS = ADD + ["lo_rm", "lo_fomc"]  # 2026-10-26 addendum E: + lo_fomc
# 2026-10-26 addendum F (the author's request after the channel results): HAR + realized moments + FOMC, every class
# (ridge / lasso / elastic net, OLS, the three trees), on the FOMC-patched design.  Kept out of the lists above, so the
# task files those generate stay as they were.
COMBO = ["add_rm_fomc"]
# 2026-10-26 addendum G (requested by the author after the HAR + RM + FOMC results; not pre-registered): the base
# add_rm_fomc plus each remaining group -- which groups give the trees' gain beyond HAR + RM + FOMC
BEYOND_GROUPS = ("ew", "vw", "liq", "iv", "sent", "vd")
BEYOND = [f"add_rm_fomc_{g}" for g in BEYOND_GROUPS]
COMBO += BEYOND
# 2026-10-27 addendum I (results/beyond_trees_2026-10-27/RULE.txt section 2; requested by the author after the
# beyond-base results, not pre-registered): the within-block ladder.  Rung brung_<est>_<k> = BASE (add_rm_fomc) + the top
# k series of the remaining block (ew, vw, liq, iv, sent, vd) in the model's own patched primary ladder ranking
# (results/ladder_2026-10-26/fomcfix/rank_primary.csv, git blob 53d057da; the 2013 pre-span fit), all 12 columns of
# each series.  The restricted top 20, in order (checked against the csv by check_ladder):
LADDER = {
    "lgbm": [
        "sumabsret_vwstock",
        "sumabsret_ewstock",
        "effspread_vwstock",
        "vix",
        "sumret3_vwstock",
        "stocktwits_sentiment",
        "sumret3_ewstock",
        "spread_vwstock",
        "effspread_ewstock",
        "spread_ewstock",
        "sumvolume",
        "ofi_ewstock",
        "sumret2_ewstock",
        "sumret2_vwstock",
        "sumbipow_ewstock",
        "sellturnover_ewstock",
        "ofi_vwstock",
        "stocktwits_sentcount",
        "sellturnover_vwstock",
        "buyturnover_vwstock",
    ],
    "xgb": [
        "stocktwits_sentiment",
        "sumabsret_vwstock",
        "effspread_vwstock",
        "sumabsret_ewstock",
        "vix",
        "sumret3_vwstock",
        "sumret3_ewstock",
        "stocktwits_sentcount",
        "spread_vwstock",
        "spread_ewstock",
        "effspread_ewstock",
        "sumvolume",
        "sellturnover_ewstock",
        "ofi_ewstock",
        "sumret2_ewstock",
        "sumret2_vwstock",
        "sellturnover_vwstock",
        "ofi_vwstock",
        "sumbipow_ewstock",
        "buyturnover_vwstock",
    ],
}
LADDER_K = (1, 3, 5, 10, 20)
RUNGS = {est: [f"brung_{est}_{k}" for k in LADDER_K] for est in LADDER}


def rung_parts(arm: str) -> tuple[str, int]:
    _b, est, k = arm.split("_")
    assert _b == "brung" and est in LADDER and int(k) in LADDER_K, arm
    return est, int(k)


def check_ladder(csv: str) -> None:
    """LADDER = the csv's primary ranking per model restricted to the remaining block, first 20, in rank order."""
    import pandas as pd

    r = pd.read_csv(csv)
    gs = group_series()
    rem = {s for g in BEYOND_GROUPS for s in gs[g]}
    for est, top in LADDER.items():
        f = r[(r["model"] == est) & (r["ranking"] == "primary")].sort_values("rank")
        assert len(f) == 46 and f["rank"].is_unique, (est, len(f))
        assert list(f[f["series"].isin(rem)]["series"].head(20)) == top, est


def add_groups(arm: str) -> list[str]:
    """The groups of an add arm: add_rm -> [rm]; add_rm_fomc -> [rm, fomc] (group keys hold no underscore)."""
    assert arm.startswith("add_"), arm
    gs = arm[4:].split("_")
    assert gs and all(g in GROUPS for g in gs) and len(set(gs)) == len(gs), arm
    return gs


def group_series() -> dict[str, list[str]]:
    """g -> the raw series of group g (src/data/loading.py SUBGROUPS, the final design's EXCLUDE_FOMC_RELEASE)."""
    from src.data import loading as LD

    out = {g: list(LD.SUBGROUPS[s]) for g, s in GROUPS.items()}
    assert {g: len(v) for g, v in out.items()} == N_SERIES, {g: len(v) for g, v in out.items()}
    allf = [s for v in out.values() for s in v]
    assert sorted(allf) == sorted(LD.ALL_FEATURES) and len(set(allf)) == 46, "groups != ALL_FEATURES"
    return out


def series_cols(s: str) -> list[str]:
    return [f"adj_{s}_ma_{k}" for k in HAR_K] + [f"{s}_avail_ma_{k}" for k in HAR_K]


def group_cols(g: str, gs: dict[str, list[str]]) -> set[str]:
    return {c for s in gs[g] for c in series_cols(s)}


def series_of(col: str) -> str | None:
    """The raw series of a value / availability average (None for HAR and calendar columns)."""
    m = re.match(r"^adj_(.+)_ma_\d+$", col) or re.match(r"^(.+)_avail_ma_\d+$", col)
    if m is None or col.startswith("har_ma_"):
        return None
    return m.group(1)


def arm_names(arm: str, names: list[str], gs: dict[str, list[str]] | None = None) -> list[str]:
    """The arm's columns, in the order of the cache `names` (the full design, or abl_new for a classical arm)."""
    if arm in CLASSICAL:
        want = set(HAR12) | set(CLASSICAL[arm])
        out = [n for n in names if n in want]
        assert len(out) == len(want), (arm, sorted(want - set(out)))
        return out
    gs = gs or group_series()
    if arm.startswith("brung_"):  # addendum I: BASE + the top-k remaining-block series of the model's ranking
        est, k = rung_parts(arm)
        keep = set(HAR12).union(
            group_cols("rm", gs), group_cols("fomc", gs), *(series_cols(s) for s in LADDER[est][:k])
        )
        assert keep <= set(names), arm
        return [n for n in names if n in keep]
    if arm == "full":
        out = list(names)
    elif arm == "har":
        out = [n for n in names if n in set(HAR12)]
    elif arm.startswith("lo_"):
        drop = group_cols(arm[3:], gs)
        assert drop <= set(names), arm
        out = [n for n in names if n not in drop]
    elif arm.startswith("add_"):
        keep = set(HAR12).union(*(group_cols(g, gs) for g in add_groups(arm)))
        assert keep <= set(names), arm
        out = [n for n in names if n in keep]
    else:
        raise KeyError(arm)
    return out


def check_full_names(names: list[str], gs: dict[str, list[str]] | None = None) -> None:
    """The full design's 564 columns = HAR12 + every group's columns, each column in exactly one group."""
    gs = gs or group_series()
    assert len(names) == 564 and len(set(names)) == 564, len(names)
    cover: dict[str, str] = {}
    for g in GROUPS:
        for c in group_cols(g, gs):
            assert c not in cover, (c, g, cover.get(c))
            cover[c] = g
    assert set(names) == set(HAR12) | set(cover), sorted(set(names) ^ (set(HAR12) | set(cover)))[:6]


def n_cols(arm: str) -> int:
    """Column count of an arm (without reading a cache)."""
    if arm == "full":
        return 564
    if arm == "har":
        return 12
    if arm in CLASSICAL:
        return 12 + len(CLASSICAL[arm])
    if arm.startswith("add_"):
        return 12 + 12 * sum(N_SERIES[g] for g in add_groups(arm))
    if arm.startswith("brung_"):
        return 12 + 12 * (N_SERIES["rm"] + N_SERIES["fomc"] + rung_parts(arm)[1])
    g = arm.split("_", 1)[1]
    return 564 - 12 * N_SERIES[g]
