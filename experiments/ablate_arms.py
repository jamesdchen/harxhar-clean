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
  brung_<est>_<k>   addendum I: add_rm_fomc + the top k remaining-block series of the model's ranking (LADDER)
  lrung_<est>_<k>   addendum K: the HAR set + the top k of the 46 series of the model's own ranking (LADALL)
  add_mac, add_rm_fomc_mac, full_mac, add_rm_fomc_mac_<r>   addendum L: the macro release calendars (MAC_ARMS); read
                    from the macro cache (bucket all_features_mac), on which every other arm means what it means on
                    the full cache (the macro columns are dropped first: "full" stays the 564-column set)
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


# 2026-10-28 addendum K (requested by the author on 28 October after the monthly-vs-every-bar comparison; not
# pre-registered): the feature-count ladder of results/ladder_2026-10-26 (RULE.txt sections 1 and 3) refitted at EVERY
# bar on the FOMC-patched design.  Rung lrung_<est>_<k> = the HAR benchmark set + all 12 columns of each of the top k of
# the 46 exogenous series in the model's own patched PRIMARY ranking (results/ladder_2026-10-26/fomcfix/rank_primary.csv,
# git blob 53d057da; the 2013 pre-span fit; ladder RULE section 3: "each model its own", so the three linear classes
# have one ranking each).  Production k in LADDER_K; k = 0 (the HAR set) and k = 46 (the full set) are the gate rungs
# only.  The top 20 per model, in rank order (checked against the csv by check_ladall):
LADALL = {
    "lgbm": [
        "sumabsret",
        "sumret",
        "sumabsret_vwstock",
        "sumabsret_ewstock",
        "sumautocov",
        "effspread_vwstock",
        "fomc_until_inv",
        "sumret3",
        "vix",
        "sumret3_vwstock",
        "stocktwits_sentiment",
        "sumret3_ewstock",
        "spread_vwstock",
        "effspread_ewstock",
        "spread_ewstock",
        "sumvolume",
        "ofi_ewstock",
        "sumret4",
        "sumret2_ewstock",
        "sumret2_vwstock",
    ],
    "xgb": [
        "sumabsret",
        "stocktwits_sentiment",
        "sumabsret_vwstock",
        "sumret",
        "effspread_vwstock",
        "sumabsret_ewstock",
        "sumautocov",
        "fomc_until_inv",
        "vix",
        "sumret3",
        "sumret3_vwstock",
        "sumret3_ewstock",
        "stocktwits_sentcount",
        "spread_vwstock",
        "spread_ewstock",
        "effspread_ewstock",
        "sumvolume",
        "sellturnover_ewstock",
        "ofi_ewstock",
        "sumpret2",
    ],
    "ridge": [
        "fomc_until_inv",
        "fomc_since_inv",
        "sumabsret",
        "fomc_day",
        "sumret",
        "sumabsret_vwstock",
        "sumautocov",
        "turnover_ewstock",
        "sumret4",
        "sellturnover_vwstock",
        "spread_vwstock",
        "buyturnover_vwstock",
        "sumret3",
        "sumret2_vwstock",
        "sumret3_vwstock",
        "sumret2_ewstock",
        "sumbipow",
        "sumvolume",
        "buyturnover_ewstock",
        "sellturnover_ewstock",
    ],
    "lasso": [
        "fomc_until_inv",
        "fomc_since_inv",
        "sumabsret",
        "fomc_day",
        "sumret",
        "turnover_ewstock",
        "sumabsret_vwstock",
        "buyturnover_ewstock",
        "sumret4",
        "sumautocov",
        "sumbipow",
        "sellturnover_vwstock",
        "sumret3",
        "sumret2_ewstock",
        "spread_vwstock",
        "sumret2_vwstock",
        "sumvolume",
        "sumbipow_vwstock",
        "sumret3_vwstock",
        "vvix",
    ],
    "enet": [
        "fomc_until_inv",
        "fomc_since_inv",
        "sumabsret",
        "sumret",
        "fomc_day",
        "sumabsret_vwstock",
        "sumautocov",
        "sumret4",
        "sumret2_ewstock",
        "sumret3_vwstock",
        "sumret3",
        "sumvolume",
        "spread_vwstock",
        "sellturnover_vwstock",
        "buyturnover_vwstock",
        "ofi_ewstock",
        "vix",
        "numobs",
        "sumret2_vwstock",
        "sumret3_ewstock",
    ],
}
LADALL_GATE_K = (0, 46)
LRUNGS = {est: [f"lrung_{est}_{k}" for k in LADDER_K] for est in LADALL}


def lrung_parts(arm: str) -> tuple[str, int]:
    """Addendum K: lrung_<est>_<k> -> (est, k); k in LADDER_K or a gate rung (0, 46)."""
    _l, est, k = arm.split("_")
    assert _l == "lrung" and est in LADALL and int(k) in (*LADDER_K, *LADALL_GATE_K), arm
    return est, int(k)


# 2026-10-29 addendum L (requested by the author on 29 October; not pre-registered): six US macro release calendars
# built as the FOMC calendar (src/data/loading.py ex-flag MACRO_CALENDARS; series stem mac_<r>, three series a release
# as FOMC's; SUBGROUPS["macro"], 18 series; SUBGROUPS["all_features_mac"] = ALL_FEATURES + macro, the macro series kept
# OUT of ALL_FEATURES).  The macro cache (bucket all_features_mac) holds the full set's 564 columns bit for bit plus the
# 216 macro columns, in the order: har ladder, the 46 series' value averages, the macro value averages, the 46 series'
# availability averages, the macro availability averages, calendar.  Groups "mac" (all 18 series, 216 columns) and
# "mac_<r>" (one release, 36 columns).  Kept out of every list above, so the task files those generate stay as they were.
MAC_RELEASES = ("cpi", "emp", "ppi", "gdp", "trade", "retail")
MAC_SUBGROUP = "macro"
MAC_BUCKET = "all_features_mac"
MAC_PER_RELEASE = 3  # the FOMC calendar's three series (day flag, until_inv, since_inv), per release
MAC_GROUPS = ["mac"] + [f"mac_{r}" for r in MAC_RELEASES]
MAC_N_SERIES = {"mac": MAC_PER_RELEASE * len(MAC_RELEASES), **{f"mac_{r}": MAC_PER_RELEASE for r in MAC_RELEASES}}
MAC_LABEL = {
    "mac": "Macro releases",
    "mac_cpi": "CPI",
    "mac_emp": "Employment",
    "mac_ppi": "PPI",
    "mac_gdp": "GDP (advance)",
    "mac_trade": "Trade",
    "mac_retail": "Retail sales",
}
MAC_PER = [f"add_rm_fomc_mac_{r}" for r in MAC_RELEASES]
MAC_ARMS = ["full_mac", "add_rm_fomc_mac", "add_mac", *MAC_PER]  # the largest first
# the OLS where the existing add arms run it: the augmented-HAR benchmarks HAR + RM (+ FOMC) (OLS_ARMS, addendum F),
# so here HAR + RM + FOMC + macro and HAR + RM + FOMC + one release; not add_mac (as add_fomc) nor full_mac (as full)
MAC_OLS_ARMS = ["add_rm_fomc_mac", *MAC_PER]
N_FULL = 564
N_FULL_MAC = N_FULL + 12 * MAC_N_SERIES["mac"]  # 780


def is_mac_series(s: str | None) -> bool:
    """A macro calendar series (stem mac_<r>, r in MAC_RELEASES); no series of the 46 or of abl_new is one."""
    return s is not None and any(s == f"mac_{r}" or s.startswith(f"mac_{r}_") for r in MAC_RELEASES)


def is_mac_col(col: str) -> bool:
    return is_mac_series(series_of(col))


def mac_series() -> dict[str, list[str]]:
    """'mac' -> the 18 macro series (src/data/loading.py SUBGROUPS["macro"], its order); 'mac_<r>' -> release r's 3.
    Asserts the interface: the stems, 3 a release, none in ALL_FEATURES, all_features_mac = ALL_FEATURES + macro."""
    from src.data import loading as LD

    mac = list(LD.SUBGROUPS[MAC_SUBGROUP])
    assert len(mac) == MAC_N_SERIES["mac"] and len(set(mac)) == len(mac), mac
    assert all(is_mac_series(s) for s in mac), [s for s in mac if not is_mac_series(s)]
    assert not set(mac) & set(LD.ALL_FEATURES), sorted(set(mac) & set(LD.ALL_FEATURES))
    assert list(LD.SUBGROUPS[MAC_BUCKET]) == list(LD.ALL_FEATURES) + mac, "all_features_mac != ALL_FEATURES + macro"
    out = {"mac": mac}
    for r in MAC_RELEASES:
        out[f"mac_{r}"] = [s for s in mac if s == f"mac_{r}" or s.startswith(f"mac_{r}_")]
    assert {g: len(v) for g, v in out.items()} == MAC_N_SERIES, {g: len(v) for g, v in out.items()}
    return out


def mac_parts(arm: str) -> tuple[list[str], str] | None:
    """An add arm with macro columns -> (its groups before 'mac', the macro group): add_mac -> ([], 'mac'),
    add_rm_fomc_mac -> (['rm', 'fomc'], 'mac'), add_rm_fomc_mac_cpi -> (['rm', 'fomc'], 'mac_cpi'); None otherwise."""
    if not arm.startswith("add_"):
        return None
    tok = arm[4:].split("_")
    if "mac" not in tok:
        return None
    i = tok.index("mac")
    base, rest = tok[:i], tok[i + 1 :]
    key = "mac" if not rest else "mac_" + "_".join(rest)
    assert key in MAC_GROUPS and all(g in GROUPS for g in base) and len(set(base)) == len(base), arm
    return base, key


def is_mac_arm(arm: str) -> bool:
    return arm == "full_mac" or mac_parts(arm) is not None


def mac_names(names: list[str], ms: dict[str, list[str]] | None = None) -> list[str]:
    """The macro cache's expected names: the non-macro names of `names` (the full set's, in its order) with the macro
    value averages after the last value average and the macro availability averages after the last availability
    average (series in SUBGROUPS["macro"] order, k ascending, as the full set's own blocks)."""
    ms = ms or mac_series()
    base = [n for n in names if not is_mac_col(n)]
    har = [n for n in base if n.startswith("har_ma_")]
    val = [n for n in base if n.startswith("adj_")]
    av = [n for n in base if "_avail_ma_" in n]
    cal = [n for n in base if n not in set(har) | set(val) | set(av)]
    assert base == har + val + av + cal, "full set's order is not har, values, availabilities, calendar"
    mv = [f"adj_{s}_ma_{k}" for s in ms["mac"] for k in HAR_K]
    ma = [f"{s}_avail_ma_{k}" for s in ms["mac"] for k in HAR_K]
    return har + val + mv + av + ma + cal


def check_mac_names(names: list[str], gs: dict | None = None, ms: dict | None = None) -> None:
    """The macro cache's 780 columns: the full set's 564 (check_full_names) in their order, the 216 macro columns
    (each macro series' 6 value and 6 availability averages), in mac_names' order."""
    ms = ms or mac_series()
    base = [n for n in names if not is_mac_col(n)]
    check_full_names(base, gs)
    assert len(names) == N_FULL_MAC and len(set(names)) == N_FULL_MAC, len(names)
    assert {n for n in names if is_mac_col(n)} == {c for s in ms["mac"] for c in series_cols(s)}, "macro columns"
    assert names == mac_names(names, ms), "macro cache column order"


def check_ladall(csv: str) -> None:
    """LADALL = the csv's primary ranking per model over all 46 series, first 20, in rank order."""
    import pandas as pd

    r = pd.read_csv(csv)
    allf = {s for v in group_series().values() for s in v}
    for est, top in LADALL.items():
        f = r[(r["model"] == est) & (r["ranking"] == "primary")].sort_values("rank")
        assert len(f) == 46 and f["rank"].is_unique and set(f["series"]) == allf, (est, len(f))
        assert list(f["series"].head(20)) == top, est


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


def arm_names(
    arm: str, names: list[str], gs: dict[str, list[str]] | None = None, ms: dict[str, list[str]] | None = None
) -> list[str]:
    """The arm's columns, in the order of the cache `names` (the full design, or abl_new for a classical arm, or
    (addendum L) the macro cache: a macro arm takes its macro columns from it, every other arm drops them first)."""
    if is_mac_arm(arm):  # addendum L
        gs = gs or group_series()
        ms = ms or mac_series()
        if arm == "full_mac":
            check_mac_names(names, gs, ms)
            return list(names)
        base, key = mac_parts(arm) or ([], "")
        keep = set(HAR12).union(*(group_cols(g, gs) for g in base), *(series_cols(s) for s in ms[key]))
        assert keep <= set(names), (arm, sorted(keep - set(names))[:4])
        return [n for n in names if n in keep]
    names = [n for n in names if not is_mac_col(n)]  # addendum L: a no-op on every cache without macro columns
    if arm in CLASSICAL:
        want = set(HAR12) | set(CLASSICAL[arm])
        out = [n for n in names if n in want]
        assert len(out) == len(want), (arm, sorted(want - set(out)))
        return out
    gs = gs or group_series()
    if arm.startswith("lrung_"):  # addendum K: the HAR set + the top-k of the 46 series in the model's own ranking
        est, k = lrung_parts(arm)
        ser = [s for v in gs.values() for s in v] if k == 46 else LADALL[est][:k]
        keep = set(HAR12).union(*(series_cols(s) for s in ser))
        assert keep <= set(names), arm
        return [n for n in names if n in keep]
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
    if arm == "full_mac":  # addendum L
        return N_FULL_MAC
    mp = mac_parts(arm)
    if mp is not None:  # addendum L
        return 12 + 12 * (sum(N_SERIES[g] for g in mp[0]) + MAC_N_SERIES[mp[1]])
    if arm.startswith("add_"):
        return 12 + 12 * sum(N_SERIES[g] for g in add_groups(arm))
    if arm.startswith("brung_"):
        return 12 + 12 * (N_SERIES["rm"] + N_SERIES["fomc"] + rung_parts(arm)[1])
    if arm.startswith("lrung_"):  # addendum K
        return 12 + 12 * lrung_parts(arm)[1]
    g = arm.split("_", 1)[1]
    return 564 - 12 * N_SERIES[g]
