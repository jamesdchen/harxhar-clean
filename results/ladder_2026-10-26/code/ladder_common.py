"""Shared definitions of the feature-count ladder (RULE.txt): paths, origins, series, rungs, rankings.

Runs on CARC (the tree venv and the linear venv) and on the laptop (scoring).  No third-party import beyond numpy /
pandas.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE.parent  # results/ladder_2026-10-26 (laptop) or the staged copy (CARC)
WORK = Path(os.environ.get("LADDER_WORK", str(RES / "work")))
TREE_D = Path(os.environ.get("LADDER_TREE_DESIGN", "/home1/jc_905/trees_final/designs/w3000/_work/design_all_features"))
LIN_D = Path(os.environ.get("LADDER_LIN_DESIGN", "/scratch1/jc_905/fi_window/w3000/design_all_features"))
X_MD5 = "374bea97f35e33a02404dc47c82c3138"
ORIGINS_SHA256 = "886e0ffea166eaf753d116382647f6acffc67e911698d8cdcd8ad8798062fce8"  # fiw_origins.csv (LF)
W_DAYS, PPD = 3000, 13
W = W_DAYS * PPD  # 39,000 rows
N_ROWS = 84747
R_FIRST = 50777  # design row of 2014-01-02 10:00 (the first forecast row)
RANK_ORIGIN = pd.Timestamp("2013-01-02 10:00")
KS = (46, 20, 10, 5, 3, 1, 0)
TREES = ("lgbm", "xgb", "rf_skx")
LINEAR = ("ridge", "lasso", "enet")
MODELS = ("lgbm", "xgb", "ridge", "lasso", "enet", "rf_skx")  # priority order
FROZEN = {"ridge": 1000.0, "lasso": 1e-4, "enet": 1e-3}  # w3000 frozen_penalties.json, all_features
R_BB = 0.01  # penalty factor on the 12 HAR-benchmark columns (every rung)
GATE_ARMS = {"add_iv": ["vix", "vvix", "vix3m"], "add_fomc": ["fomc_day", "fomc_until_inv", "fomc_since_inv"]}


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def series_of(col: str) -> str | None:
    """The exogenous series of a design column; None for the 12 HAR-benchmark columns."""
    if col.startswith("har_ma_") or col.startswith("DOW_") or col == "hour":
        return None
    if col.startswith("adj_") and "_ma_" in col:
        return col[4:].rsplit("_ma_", 1)[0]
    if "_avail_ma_" in col:
        return col.rsplit("_avail_ma_", 1)[0]
    raise ValueError(col)


def names(d: Path) -> list[str]:
    return list(json.loads((d / "design.json").read_text())["names"])


def series_list(nm: list[str]) -> list[str]:
    """The 46 series in design order (order of their first column): the tie-break order."""
    out: list[str] = []
    for c in nm:
        s = series_of(c)
        if s is not None and s not in out:
            out.append(s)
    assert len(out) == 46, len(out)
    for s in out:
        assert sum(series_of(c) == s for c in nm) == 12, s
    assert sum(series_of(c) is None for c in nm) == 12
    return out


def columns_for(nm: list[str], keep: list[str] | set[str]) -> np.ndarray:
    """Design-order indices of the 12 HAR-benchmark columns plus every column of the kept series."""
    keep = set(keep)
    idx = [j for j, c in enumerate(nm) if series_of(c) is None or series_of(c) in keep]
    assert len(idx) == 12 + 12 * len(keep), (len(idx), len(keep))
    return np.array(idx, dtype=np.int64)


def oos_dates(d: Path) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.read_csv(d / "dates_oos.csv", parse_dates=["date"])["date"])


def origins(d: Path) -> pd.DataFrame:
    """The 126 origins (fiw_origins.csv, sha256 checked): design rows r = W + oos_row_w3000, blocks [r, r_end)."""
    f = HERE / "fiw_origins.csv"
    assert hashlib.sha256(f.read_bytes().replace(b"\r\n", b"\n")).hexdigest() == ORIGINS_SHA256
    o = pd.read_csv(f)
    dd = oos_dates(d)
    assert len(o) == 126 and len(dd) == N_ROWS - W
    o["oos_row"] = o["oos_row_w3000"].astype(int)
    o["r"] = W + o["oos_row"]
    o["r_end"] = list(o["r"].iloc[1:]) + [N_ROWS]
    assert (pd.DatetimeIndex(dd[o["oos_row"]]) == pd.to_datetime(o["origin"])).all()
    assert (o["r_end"] - o["r"]).sum() == 33970 and o["r"].iloc[0] == R_FIRST
    return o[["k", "origin", "oos_row", "r", "r_end"]].reset_index(drop=True)


def rank_origin_row(d: Path) -> int:
    """Design row r_0 of 2013-01-02 10:00 (the primary ranking's pre-sample fit)."""
    dd = oos_dates(d)
    i0 = int(np.flatnonzero(dd == RANK_ORIGIN)[0])
    r0 = W + i0
    assert r0 - W >= 0 and r0 < R_FIRST
    yrs = dd[i0 : R_FIRST - W]
    assert (yrs.year == 2013).all() and dd[R_FIRST - W - 1].year == 2013 and dd[i0 - 1].year == 2012
    return r0


def order_from_importance(imp: dict[str, float], tie_order: list[str]) -> list[str]:
    """Descending importance; ties by design order."""
    pos = {s: i for i, s in enumerate(tie_order)}
    return sorted(tie_order, key=lambda s: (-float(imp[s]), pos[s]))


def rung_sets(order_p: list[str], order_s: list[str] | None) -> dict[str, list[str]]:
    """{rung id: kept series}: full, har, P<k>, S<k> (k = 20, 10, 5, 3, 1)."""
    out: dict[str, list[str]] = {"full": list(order_p), "har": []}
    for tag, order in (("P", order_p), ("S", order_s)):
        if order is None:
            continue
        for k in KS:
            if k in (46, 0):
                continue
            out[f"{tag}{k}"] = list(order[:k])
    return out


def unique_sets(sets: dict[str, list[str]]) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Fit each distinct series set once: (fitted id -> series, every id -> the fitted id)."""
    fit: dict[str, list[str]] = {}
    alias: dict[str, str] = {}
    seen: dict[frozenset, str] = {}
    for rid, ss in sets.items():
        key = frozenset(ss)
        if key in seen:
            alias[rid] = seen[key]
        else:
            seen[key] = rid
            fit[rid] = ss
            alias[rid] = rid
    return fit, alias


def secondary_order(model: str, tie_order: list[str]) -> list[str]:
    """SECONDARY (look-ahead) ranking from code/secondary_importance.json (the SHAP study's m1_variables.csv /
    m2_linear_shares.csv, extracted on the laptop)."""
    imp = json.loads((HERE / "secondary_importance.json").read_text())[model]
    return order_from_importance(imp, tie_order)


def primary_order(model: str, tie_order: list[str]) -> list[str]:
    fam = "linear" if model in LINEAR else model
    rk = json.loads((WORK / f"rank_{fam}.json").read_text())
    imp = rk["importance"][model]
    return order_from_importance(imp, tie_order)
