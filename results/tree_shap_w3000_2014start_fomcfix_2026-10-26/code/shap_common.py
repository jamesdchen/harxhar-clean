"""Shared definitions of the tree SHAP study (RULE_shap.txt): paths, the frozen runner, origins, neighbourhoods,
variable groups, buckets and the STANDARD / NON-STANDARD classes.  Runs on the laptop (git worktree) and on CARC
(code/carc_job.sbatch sets TREE_SHAP_DESIGN / TREE_SHAP_WORK / TREE_SHAP_RUNNER_DIR)."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE.parent  # results/tree_shap_w3000_2014start_2026-10-23
REPO = RES.parents[1]  # the worktree (laptop)
_TMP = Path(r"C:\Users\james\AppData\Local\Temp")
DESIGN = Path(
    os.environ.get("TREE_SHAP_DESIGN", str(_TMP / "shap_design" / "w3000"))
)  # md5-verified design copy
WORK = Path(
    os.environ.get("TREE_SHAP_WORK", str(_TMP / "shap_work"))
)  # per-origin SHAP arrays (not committed)
RUNNER_DIR = os.environ.get(
    "TREE_SHAP_RUNNER_DIR"
)  # CARC: the runner file + cluster/ configs staged from git
RUNNER_SHA = "3cbb85f"  # -trees-final: model_factory / fit_predict incl. rf_skx
FIW_SHA = "0c44dca"  # -fi-window: fiw_origins.csv, fiw_gamma.npz
# sha256 of the git blobs at RUNNER_SHA (LF), checked wherever the runner is loaded from
BLOB_SHA256 = {
    "experiments/pooled_trees_rerun_13.py": "c2a5b45724f8ef1105484e8d7db41c974766b36cc8daa29b581905d2af2bbad8",
    "cluster/pooltree_lgbm_config.json": "1349f96a9ae69198db9f754c07993c729fe02b6c5a0a45c66d0ce4dae3e80d0d",
    "cluster/xgbtop_xgb_config.json": "961ca7096386aba2aad9fa54a50a7be5c9a711b4399fa80b49c8ec6f6854f3dc",
    "cluster/pooltree_rf400_config.json": "ba5c99b02b9359e070813a7ed8a421cb96205176a7dc91aa55c6b0288f164567",
}
ORIGINS_SHA256 = "886e0ffea166eaf753d116382647f6acffc67e911698d8cdcd8ad8798062fce8"  # FIW_SHA's fiw_origins.csv
W_DAYS, PPD = 3000, 13
W = W_DAYS * PPD
EST = ("lgbm", "xgb", "rf_skx")
EST_NAME = {"lgbm": "LightGBM", "xgb": "XGBoost", "rf_skx": "Random forest"}
THREADS = 8
SEED = 20261023
N_INTER = 8  # M5 rows per neighbourhood
REPEAT_KS = (
    0,
    62,
    125,
)  # G2(a): the forest refitted twice at origins 1, 63, 126 (k is 0-based)

CATEGORIES = {
    "moments": [
        "sumret",
        "sumabsret",
        "sumret3",
        "sumret4",
        "sumpret2",
        "sumbipow",
        "sumautocov",
    ],
    "liquidity": [
        "sumvolume",
        "numobs",
        "turnover_ewstock",
        "buyturnover_ewstock",
        "sellturnover_ewstock",
        "effspread_ewstock",
        "spread_ewstock",
        "turnover_vwstock",
        "buyturnover_vwstock",
        "sellturnover_vwstock",
        "effspread_vwstock",
        "spread_vwstock",
        "ofi_ewstock",
        "ofi_vwstock",
    ],
    "market_ew": [
        "sumret2_ewstock",
        "sumret3_ewstock",
        "sumret4_ewstock",
        "sumabsret_ewstock",
        "sumbipow_ewstock",
        "sumpret2_ewstock",
    ],
    "market_vw": [
        "sumret2_vwstock",
        "sumret3_vwstock",
        "sumret4_vwstock",
        "sumabsret_vwstock",
        "sumbipow_vwstock",
        "sumpret2_vwstock",
    ],
    "sentiment": [
        "stocktwits_attention",
        "stocktwits_sentiment",
        "stocktwits_sentcount",
    ],
    "implied_vol": ["vix", "vvix", "vix3m"],
    "vol_demand": [
        "voldemand_spx_open_and_close",
        "voldemand_spx_open_only",
        "voldemand_all_open_and_close",
        "voldemand_all_open_only",
    ],
    "fomc": ["fomc_day", "fomc_until_inv", "fomc_since_inv"],
}
EXOG = [v for vs in CATEGORIES.values() for v in vs]
GROUPS = ["har_ma", "calendar"] + EXOG  # 48 variables
CAT_OF = {"har_ma": "har_ma", "calendar": "calendar"} | {
    v: c for c, vs in CATEGORIES.items() for v in vs
}
BUCKETS = ["har_ma", "calendar"] + list(CATEGORIES)  # backbone shown as its two members
BUCKET_NAME = {
    "har_ma": "HAR terms",
    "calendar": "Calendar",
    "moments": "Realized moments (E-mini)",
    "liquidity": "Liquidity",
    "market_ew": "Stock-market moments (EW)",
    "market_vw": "Stock-market moments (VW)",
    "sentiment": "StockTwits",
    "implied_vol": "Implied volatility",
    "vol_demand": "Volatility demand",
    "fomc": "FOMC calendar",
}
STANDARD = [
    "har_ma",
    "calendar",
    "sumret",
    "sumabsret",
    "sumret3",
    "sumret4",
    "sumpret2",
    "sumbipow",
    "sumautocov",
    "vix",
    "sumvolume",
    "numobs",
]
EMINI_ACTIVITY = ["sumvolume", "numobs"]  # sensitivity: moved to NON-STANDARD
NONSTANDARD = [g for g in GROUPS if g not in STANDARD]
assert len(GROUPS) == 48 and len(NONSTANDARD) == 36 and len(EXOG) == 46
CLASS_OF = {g: ("STANDARD" if g in STANDARD else "NON-STANDARD") for g in GROUPS}


def say(msg: str) -> None:
    import time

    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def git_show(rev_path: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(REPO), "show", rev_path])


def load_runner() -> types.ModuleType:
    """experiments/pooled_trees_rerun_13.py at RUNNER_SHA, executed from the git object (not edited); its ROOT is this
    worktree, whose cluster/ config files are the same git blobs as at RUNNER_SHA (asserted).  On CARC
    (TREE_SHAP_RUNNER_DIR) the same file and configs, staged from git, are checked by the sha256 of the blobs."""
    if RUNNER_DIR:
        root = Path(RUNNER_DIR)
        for f, h in BLOB_SHA256.items():
            assert hashlib.sha256((root / f).read_bytes()).hexdigest() == h, f
        src = (root / "experiments" / "pooled_trees_rerun_13.py").read_bytes()
    else:
        root = REPO
        for f in BLOB_SHA256:
            if f.startswith("cluster/"):
                a = subprocess.check_output(
                    ["git", "-C", str(REPO), "rev-parse", f"HEAD:{f}"]
                ).strip()
                b = subprocess.check_output(
                    ["git", "-C", str(REPO), "rev-parse", f"{RUNNER_SHA}:{f}"]
                ).strip()
                assert a == b, (f, a, b)
        src = git_show(f"{RUNNER_SHA}:experiments/pooled_trees_rerun_13.py")
    assert (
        hashlib.sha256(src).hexdigest()
        == BLOB_SHA256["experiments/pooled_trees_rerun_13.py"]
    )
    mod = types.ModuleType("pooled_trees_rerun_13_at_" + RUNNER_SHA)
    mod.__file__ = str(root / "experiments" / "pooled_trees_rerun_13.py")
    sys.modules[mod.__name__] = mod
    exec(compile(src, f"pooled_trees_rerun_13.py@{RUNNER_SHA}", "exec"), mod.__dict__)
    mod.RUNNER_SRC_SHA256 = hashlib.sha256(src).hexdigest()  # type: ignore[attr-defined]
    return mod


def design_meta() -> dict:
    return json.loads(
        (DESIGN / "_work" / "design_all_features" / "design.json").read_text()
    )


def names() -> list[str]:
    return list(design_meta()["names"])


def group_of(col: str) -> str:
    if col.startswith("har_ma_"):
        return "har_ma"
    if col.startswith("DOW_") or col == "hour":
        return "calendar"
    if col.startswith("adj_") and "_ma_" in col:
        return col[4:].rsplit("_ma_", 1)[0]
    if "_avail_ma_" in col:
        return col.rsplit("_avail_ma_", 1)[0]
    raise ValueError(col)


def kind_of(col: str) -> str:
    if col.startswith("har_ma_"):
        return "har"
    if col.startswith("DOW_") or col == "hour":
        return "calendar"
    return "avail" if "_avail_ma_" in col else "value"


def column_table() -> pd.DataFrame:
    n = names()
    t = pd.DataFrame(
        {
            "column": n,
            "group": [group_of(c) for c in n],
            "kind": [kind_of(c) for c in n],
        }
    )
    t["bucket"] = t["group"].map(CAT_OF)
    t["class"] = t["group"].map(CLASS_OF)
    assert (
        set(t["group"]) == set(GROUPS)
        and t["group"]
        .value_counts()
        .eq(pd.Series({g: (6 if g in ("har_ma", "calendar") else 12) for g in GROUPS}))
        .all()
    )
    fi = pd.read_csv(
        RES / "inputs" / "fi_predictors.csv"
    )  # = results/rehaul_fi_pooled13_2026-10-22/
    chk = fi.set_index("column").reindex(t["column"])
    assert chk["group"].tolist() == t["group"].tolist(), (
        "groups differ from the linear FI"
    )
    cat = chk["category"].where(chk["category"] != "backbone", t["group"].values)
    assert cat.tolist() == t["bucket"].tolist(), (
        "buckets differ from fi_groups category"
    )
    return t


def group_matrix(t: pd.DataFrame) -> np.ndarray:
    """48 x 564 indicator (rows in GROUPS order)."""
    P = np.zeros((len(GROUPS), len(t)))
    gi = {g: i for i, g in enumerate(GROUPS)}
    for j, g in enumerate(t["group"]):
        P[gi[g], j] = 1.0
    return P


def oos_dates() -> pd.DatetimeIndex:
    return pd.DatetimeIndex(
        pd.read_csv(
            DESIGN / "_work" / "design_all_features" / "dates_oos.csv",
            parse_dates=["date"],
        )["date"]
    )


def origins() -> pd.DataFrame:
    """The 126 origins of fi_by_window_2014start (inputs/fiw_origins.csv = git show FIW_SHA:..., sha256 checked), their
    design rows r_k = W + oos_row_w3000 and neighbourhoods [r_k, r_{k+1})."""
    f = RES / "inputs" / "fiw_origins.csv"
    assert (
        hashlib.sha256(f.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        == ORIGINS_SHA256
    )
    o = pd.read_csv(f)
    d = oos_dates()
    assert len(o) == 126
    o["oos_row"] = o["oos_row_w3000"].astype(int)
    o["r"] = W + o["oos_row"]
    n_rows = W + len(d)
    o["r_end"] = list(o["r"].iloc[1:]) + [n_rows]
    assert (pd.DatetimeIndex(d[o["oos_row"]]) == pd.to_datetime(o["origin"])).all()
    assert (o["r_end"] - o["r"]).sum() == 33970 and o["r"].iloc[0] == 50777
    assert d[o["oos_row"].iloc[0]] == pd.Timestamp("2014-01-02 10:00")
    o["origin"] = pd.to_datetime(o["origin"])
    return o[["k", "origin", "oos_row", "r", "r_end"]]


def inter_rows(o: pd.DataFrame) -> dict[int, np.ndarray]:
    """M5 fixed subsample: N_INTER rows of each neighbourhood, default_rng(SEED), origin by origin in order."""
    rng = np.random.default_rng(SEED)
    out = {}
    for _, row in o.iterrows():
        n = int(row["r_end"] - row["r"])
        out[int(row["k"])] = np.sort(rng.choice(n, size=N_INTER, replace=False))
    return out
