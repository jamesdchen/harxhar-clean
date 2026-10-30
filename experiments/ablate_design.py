"""Design caches of the ablation run (results/ablations_2014start_w3000/RULE.txt, sections 2 and 5 gate 0).

usage: python experiments/ablate_design.py <mode> [options]
  check     --full D --har D --out J     gate 0a: md5 of the full / baseline caches = the templates'; the full cache's
                                         12 HAR-benchmark columns = the baseline X bit for bit; y / b_oos / dates
                                         equal; group columns cover the full set; sumret / sumpret2 availability
                                         averages constant 1 (exit 3 on a failure)
  newseries --out D [--data data]        the abl_new design (sumret, sumret4, sumpret2 + sumnret2, rvsqrq, negret)
                                         through experiments/pool_linear_pretune.py design with the final run's
                                         configuration (RULE.txt section 2)
  cmpnew    --new D --full D --out J     gate 0b: abl_new's HAR, calendar and reference-series columns = the full
                                         cache's bit for bit; y / b_oos / dates equal (exit 3 on a failure)
  subsets   --full D --new D --root R    the linear arms' caches R/<arm>/ (column subsets, F order, the source's
                                         meta with the arm's names; har_sub = the 12 HAR columns of the full cache,
                                         labelled baseline for gate L)
  latestart --full D --out C [--data data] [--rule CSV]
                                         per series: first print, and per value column the zero prefix of the full
                                         cache (LATE_START_WARMUP), its first live row, dates where the cache has
                                         them, live share of the 2014-2024 evaluation rows; checked against the
                                         window sweep's latestart_rule_all.csv (w3000)
  2026-10-29 addendum L (the macro release calendars):
  buildmac  --out D [--data data]        the macro cache (bucket all_features_mac): pool_linear_pretune.py design with
                                         the FOMC-patched design's arguments (CFG + cell B's flags, as fomcfix's
                                         lin_stage.sh design stage ran them through the orchestrator: one thread)
                                         + MACRO_CALENDARS=1; its names checked (ablate_arms.check_mac_names)
  checkmac  --mac D --full D --out J     gate GM0: the patched full cache's X md5 = 725eb201...; the macro cache's y /
                                         b_oos / dates bytes = its; bucket, config (cell B + MACRO_CALENDARS) and names
                                         (the 564 in their order + the 216 macro columns in place); the 564 columns
                                         bit-identical; the macro columns finite (exit 3 on a failure)
  submac    --mac D --root R [--arms A,..]
                                         the linear caches from the macro cache: R/har_sub (12, baseline), R/full (the
                                         564, all_features: gate LF's full set; X md5 must be 725eb201..., har_sub's
                                         05da4b5b...) and R/<arm> for every macro arm (exit 3 on an md5 failure)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(HERE), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ablate_arms as AA  # noqa: E402

W = 39000
TEMPLATE_MD5 = {
    "all_features": {
        "X.npy": "374bea97f35e33a02404dc47c82c3138",
        "y.npy": "996504f16da7982d64656d1ba0d9722a",
        "b_oos.npy": "7670f01a3055265dfac28ebac1eca864",
        "dates_oos.csv": "d8a203fe273cd3e8e34987fe5a73048e",
        "design.json": "224b086bde8a54ef06e62ff23bfdf5e9",
    },
    "baseline": {
        "X.npy": "05da4b5b30ec4579df77f1a3b6dd57ed",
        "y.npy": "996504f16da7982d64656d1ba0d9722a",
        "b_oos.npy": "7670f01a3055265dfac28ebac1eca864",
        "dates_oos.csv": "d8a203fe273cd3e8e34987fe5a73048e",
        "design.json": "91b0800f1ff1aed3cc53af28cfc1fabc",
    },
}
# the final run's runner configuration (pooled_linear_rerun_2026_10_19_13.CFG + --window-days 3000 +
# --ex-flag LATE_START_WARMUP=1, in the orchestrator's order)
CFG = [
    "--sample",
    "rth13",
    "--har-base",
    "5",
    "--ex-flag",
    "DROP_ACTIVITY_MA=1",
    "--ex-flag",
    "CALENDAR_THROUGH_OVERNIGHT=1",
    "--ex-flag",
    "DROP_OVERNIGHT_INDICATOR=1",
    "--ex-flag",
    "LEAN_GLOBAL_LAGS=1",
    "--window-days",
    "3000",
    "--ex-flag",
    "LATE_START_WARMUP=1",
]
FIRST_EVAL, LAST_EVAL = pd.Timestamp("2014-01-02"), pd.Timestamp("2024-04-30")
# the final design's configuration as the design caches record it (design.json "config")
CONFIG = {
    "sample": "rth13",
    "periods_per_day": 13,
    "window": 39000,
    "har_base": "5",
    "drop_fomc": False,
    "ex_flags": {
        "DROP_ACTIVITY_MA": True,
        "CALENDAR_THROUGH_OVERNIGHT": True,
        "DROP_OVERNIGHT_INDICATOR": True,
        "LEAN_GLOBAL_LAGS": True,
        "LATE_START_WARMUP": True,
    },
}
# 2026-10-25 addendum A (b): reference columns through numpy's power function (sumret4's fourth root) may differ from
# CARC's build at the last bits (the container's libm); every other reference column must be bit-identical
POW_COLS_TOL = 1e-12


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def meta(d: Path) -> dict:
    return json.loads((d / "design.json").read_text())


def load_X(d: Path) -> np.ndarray:
    return np.load(d / "X.npy", mmap_mode="r")


# ============================================================================ gate 0a
def cmd_check(a) -> int:
    full, har = Path(a.full), Path(a.har)
    rec: dict = {"full": str(full), "har": str(har), "md5": {}, "fail": []}
    # 2026-10-25 addendum A (a): design.json is recorded, not required (the Hoffman2 copies are the linear three-way
    # comparison's rebuilt caches, whose design.json records that build); its names and config are checked instead.
    # Was: every file of TEMPLATE_MD5 required.
    for s, d in (("all_features", full), ("baseline", har)):
        for f, want in TEMPLATE_MD5[s].items():
            got = md5(d / f)
            rec["md5"][f"{s}/{f}"] = got
            if got != want and f != "design.json":
                rec["fail"].append(f"md5 {s}/{f} {got} != {want}")
    mf, mh = meta(full), meta(har)
    fz = json.loads(
        (ROOT / "results" / "window_sweep_2014start_2026-10-22" / "w3000" / "frozen_penalties.json").read_text()
    )
    for s, m_ in (("all_features", mf), ("baseline", mh)):
        if m_["names"] != fz["designs"][s]["columns"]:
            rec["fail"].append(f"{s}: names differ from w3000 frozen_penalties.json")
        if m_.get("config") != CONFIG:
            rec["fail"].append(f"{s}: config {m_.get('config')} != {CONFIG}")
    names = mf["names"]
    AA.check_full_names(names)
    assert mh["names"] == AA.HAR12, mh["names"]
    Xf, Xh = load_X(full), load_X(har)
    assert mf["f_order"] and mh["f_order"] and Xf.flags.f_contiguous and Xh.flags.f_contiguous
    idx = [names.index(n) for n in AA.HAR12]
    sub = np.asfortranarray(Xf[:, idx])
    rec["har_subset_equal"] = bool(
        np.array_equal(sub, np.asarray(Xh)) and sub.tobytes(order="A") == np.asarray(Xh).tobytes(order="A")
    )
    if not rec["har_subset_equal"]:
        rec["fail"].append("the full cache's 12 HAR columns differ from the baseline X")
    for f in ("y.npy", "b_oos.npy", "dates_oos.csv"):
        if (full / f).read_bytes() != (har / f).read_bytes():
            rec["fail"].append(f"{f} differs between the caches")
    const = {}
    for s in ("sumret", "sumpret2"):
        for k in AA.HAR_K:
            c = Xf[:, names.index(f"{s}_avail_ma_{k}")]
            const[f"{s}_avail_ma_{k}"] = [float(np.min(c)), float(np.max(c))]
            if not (np.min(c) == 1.0 and np.max(c) == 1.0):
                rec["fail"].append(f"{s}_avail_ma_{k} not constant 1")
    rec["emini_avail_min_max"] = const
    rec["shape"] = list(Xf.shape)
    rec["pass"] = not rec["fail"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec, indent=1))
    say(f"gate 0a {'PASS' if rec['pass'] else 'FAIL'}: {rec['fail']}")
    return 0 if rec["pass"] else 3


# ============================================================================ gate F0 (addendum D)
FIX_X_MD5 = "725eb201044783d3cdc62501307dcc3a"  # fomcfix cell B, design_all_features/X.npy (its gate G3)
FOMC_PROX = [f"adj_fomc_{s}_inv_ma_{k}" for s in ("until", "since") for k in AA.HAR_K]


def cmd_checkfix(a) -> int:
    """Gate F0: the FOMC-patched full cache against the final one: X md5 = fomcfix's, y / b_oos / dates bytes equal,
    names equal, every column bit-identical except exactly the 12 FOMC proximity value averages (which differ)."""
    old, new = Path(a.full), Path(a.new)
    rec: dict = {"old": str(old), "new": str(new), "fail": []}
    rec["x_md5"] = md5(new / "X.npy")
    if rec["x_md5"] != FIX_X_MD5:
        rec["fail"].append(f"X md5 {rec['x_md5']} != {FIX_X_MD5}")
    for f in ("y.npy", "b_oos.npy", "dates_oos.csv"):
        if (old / f).read_bytes() != (new / f).read_bytes():
            rec["fail"].append(f"{f} differs")
    mo, mn = meta(old), meta(new)
    if mo["names"] != mn["names"]:
        rec["fail"].append("names differ")
    rec["config_new"] = mn.get("config")
    Xo, Xn = load_X(old), load_X(new)
    diff = []
    for j, n in enumerate(mo["names"]):
        x, y = np.asarray(Xo[:, j]), np.asarray(Xn[:, j])
        if not (np.array_equal(x, y) and x.tobytes() == y.tobytes()):
            diff.append(n)
    rec["differing_columns"] = diff
    if sorted(diff) != sorted(FOMC_PROX):
        rec["fail"].append(f"differing columns {diff} != the 12 FOMC proximity averages")
    rec["pass"] = not rec["fail"]
    Path(a.out).write_text(json.dumps(rec, indent=1))
    say(f"gate F0 {'PASS' if rec['pass'] else 'FAIL'}: {len(diff)} differing columns; {rec['fail']}")
    return 0 if rec["pass"] else 3


# ============================================================================ addendum L: the macro cache
# fomcfix cell B's design flags (results/fomcfix_2026-10-26/RULE.txt; lin_stage.sh CELL, after LATE_START_WARMUP), then
# the macro calendars
CELL_B = ["FOMC_SCHEDULE_2020_FIX=1", "FOMC_PROXIMITY_NO_WINSOR=1", "DROP_HOLIDAY_SESSIONS=0"]
MAC_FLAGS = [*CELL_B, "MACRO_CALENDARS=1"]
RUNNER = ROOT / "experiments" / "pool_linear_pretune.py"


def mac_cfg() -> list[str]:
    """The macro cache's runner arguments after "design --bucket all_features_mac --out D": the orchestrator's CFG (=
    CFG here) followed by the cell's flags, as pooled_linear_rerun_2026_10_19_13.py main extends its CFG for
    lin_stage.sh's design stage, + MACRO_CALENDARS=1."""
    out = list(CFG)
    for f in MAC_FLAGS:
        out += ["--ex-flag", f]
    return out


def mac_config(full_config: dict) -> dict:
    """The macro cache's design.json config: the patched full cache's, with MACRO_CALENDARS on."""
    c = json.loads(json.dumps(full_config))
    c["ex_flags"] = dict(c["ex_flags"], MACRO_CALENDARS=True)
    return c


def cmd_buildmac(a) -> int:
    """The macro cache as the patched cache was built: pooled_linear_rerun_2026_10_19_13.run's subprocess (one thread
    for OpenMP / OpenBLAS / MKL / numba, cwd = the code tree), the runner's design mode on bucket all_features_mac."""
    import subprocess

    out = Path(a.out).resolve()
    args = ["design", "--bucket", AA.MAC_BUCKET, "--out", str(out)]
    if a.data != "data":
        args += ["--data", a.data]
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMBA_NUM_THREADS="1",
        PYTHONUNBUFFERED="1",
    )
    cmd = [sys.executable, "-u", str(RUNNER), *args, *mac_cfg()]
    say(f"buildmac: {' '.join(cmd[2:])}")
    t0 = time.time()
    r = subprocess.run(cmd, env=env, cwd=ROOT)
    if r.returncode:
        say(f"buildmac: runner exit {r.returncode}")
        return r.returncode
    m = meta(out)
    AA.check_mac_names(m["names"])
    assert m["bucket"] == AA.MAC_BUCKET and m["f_order"], (m["bucket"], m["f_order"])
    assert m["config"]["ex_flags"].get("MACRO_CALENDARS") is True, m["config"]
    say(
        f"buildmac done in {time.time() - t0:.0f} s: {m['shape']}, {len(m['names'])} columns; X md5 {md5(out / 'X.npy')}"
    )
    return 0


def _npy_md5(a: np.ndarray) -> str:
    import io

    buf = io.BytesIO()
    np.save(buf, a)
    return hashlib.md5(buf.getvalue()).hexdigest()


def cmd_checkmac(a) -> int:
    """Gate GM0 (addendum L): the macro cache against the FOMC-patched full cache."""
    mac, full = Path(a.mac), Path(a.full)
    rec: dict = {"mac": str(mac), "full": str(full), "fail": []}
    rec["full_x_md5"] = md5(full / "X.npy")
    if rec["full_x_md5"] != FIX_X_MD5:
        rec["fail"].append(f"patched cache X md5 {rec['full_x_md5']} != {FIX_X_MD5}")
    rec["mac_md5"] = {f: md5(mac / f) for f in ("X.npy", "y.npy", "b_oos.npy", "dates_oos.csv", "design.json")}
    for f in ("y.npy", "b_oos.npy", "dates_oos.csv"):
        if (mac / f).read_bytes() != (full / f).read_bytes():
            rec["fail"].append(f"{f} differs")
    mm, mf = meta(mac), meta(full)
    if mm.get("bucket") != AA.MAC_BUCKET:
        rec["fail"].append(f"bucket {mm.get('bucket')} != {AA.MAC_BUCKET}")
    rec["config_mac"], rec["config_full"] = mm.get("config"), mf.get("config")
    if mm.get("config") != mac_config(mf["config"]):
        rec["fail"].append(f"config {mm.get('config')} != the patched cache's + MACRO_CALENDARS")
    names, fn = list(mm["names"]), list(mf["names"])
    gs, ms = AA.group_series(), AA.mac_series()
    rec["macro_series"] = ms
    try:
        AA.check_mac_names(names, gs, ms)
    except AssertionError as e:
        rec["fail"].append(f"names: {e}")
    base = [n for n in names if not AA.is_mac_col(n)]
    if base != fn:
        rec["fail"].append("the non-macro names (in order) differ from the patched cache's")
    if names != AA.mac_names(fn, ms):
        rec["fail"].append("names != the patched names with the macro blocks in place")
    Xm, Xf = load_X(mac), load_X(full)
    rec["shape"], rec["f_order"] = list(Xm.shape), bool(mm.get("f_order"))
    if Xm.shape[0] != Xf.shape[0] or not (Xm.flags.f_contiguous and Xf.flags.f_contiguous):
        rec["fail"].append(f"rows / layout: {Xm.shape} {Xf.shape}")
    diff = []
    for j, n in enumerate(fn):
        if n not in names:
            diff.append(n)
            continue
        x, y = np.asarray(Xm[:, names.index(n)]), np.asarray(Xf[:, j])
        if not (np.array_equal(x, y) and x.tobytes() == y.tobytes()):
            diff.append(n)
    rec["differing_columns"] = diff
    if diff:
        rec["fail"].append(f"{len(diff)} of the 564 columns not bit-identical: {diff[:6]}")
    # recorded: the 564 columns written as a cache would be (F order) -- = the patched X.npy byte for byte when the
    # columns are and np.save writes the same header
    idx = [names.index(n) for n in fn if n in names]
    if len(idx) == len(fn):
        rec["subset_npy_md5"] = _npy_md5(np.asfortranarray(Xm[:, idx]))
        rec["subset_npy_md5_equal"] = rec["subset_npy_md5"] == FIX_X_MD5
    st = {}
    for n in [n for n in names if AA.is_mac_col(n)]:
        c = np.asarray(Xm[:, names.index(n)])
        nz = np.flatnonzero(c != 0.0)
        st[n] = dict(
            finite=bool(np.isfinite(c).all()),
            min=float(np.nanmin(c)),
            max=float(np.nanmax(c)),
            zeros=int(np.sum(c == 0.0)),
            first_nonzero_row=int(nz[0]) if len(nz) else -1,
            sd_after_w=float(np.std(c[W:])) if np.isfinite(c).all() else float("nan"),
        )
        if not st[n]["finite"]:
            rec["fail"].append(f"macro column {n} not finite")
        if "_avail_ma_" in n and not (st[n]["min"] >= 0.0 and st[n]["max"] <= 1.0):
            rec["fail"].append(f"availability column {n} outside [0, 1]")
    rec["macro_columns"] = st
    rec["macro_constant"] = sorted(n for n, v in st.items() if v["sd_after_w"] == 0.0)  # recorded
    rec["pass"] = not rec["fail"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec, indent=1))
    say(
        f"gate GM0 {'PASS' if rec['pass'] else 'FAIL'}: {len(names)} columns, {len(st)} macro, X md5 "
        f"{rec['mac_md5']['X.npy']}; {rec['fail'][:5]}"
    )
    return 0 if rec["pass"] else 3


def cmd_submac(a) -> int:
    """The linear caches of addendum L from the macro cache (column subsets, F order, labelled as every linear arm):
    har_sub (baseline) and full (all_features) for gate LF, and the macro arms."""
    mac, root = Path(a.mac), Path(a.root)
    m = meta(mac)
    assert m["bucket"] == AA.MAC_BUCKET, m["bucket"]
    names = list(m["names"])
    gs, ms = AA.group_series(), AA.mac_series()
    AA.check_mac_names(names, gs, ms)
    only = [x for x in a.arms.split(",") if x]
    recs, fail = [], []
    if not only or "har_sub" in only:
        r = write_subset(mac, AA.arm_names("har", names, gs), root / "har_sub", "har_sub", "baseline")
        if r["x_md5"] != TEMPLATE_MD5["baseline"]["X.npy"]:
            fail.append(f"har_sub X md5 {r['x_md5']} != the baseline cache's")
        recs.append(r)
    if not only or "full" in only:
        cols = AA.arm_names("full", names, gs)
        assert len(cols) == AA.N_FULL, len(cols)
        r = write_subset(mac, cols, root / "full", "full", "all_features")
        if r["x_md5"] != FIX_X_MD5:
            fail.append(f"full X md5 {r['x_md5']} != the patched cache's {FIX_X_MD5}")
        recs.append(r)
    for arm in [x for x in only if x not in ("har_sub", "full")] or AA.MAC_ARMS:
        assert AA.is_mac_arm(arm), arm
        cols = AA.arm_names(arm, names, gs, ms)
        assert len(cols) == AA.n_cols(arm), (arm, len(cols), AA.n_cols(arm))
        recs.append(write_subset(mac, cols, root / arm, arm, "all_features"))
        say(f"subset {arm}: {len(cols)} columns")
    pd.DataFrame(recs).to_csv(
        root / ("subsets_mac.csv" if not only else f"subsets_mac_{'_'.join(only)}.csv"), index=False
    )
    say(f"submac: {len(recs)} caches -> {root}; {'PASS' if not fail else 'FAIL ' + str(fail)}")
    return 0 if not fail else 3


# ============================================================================ the new series
def register_new_series() -> None:
    """Add sumnret2, rvsqrq, negret to the loaded frame (as add_derived_features adds the OFI ratios: after the domain
    rule, before the dead-session drop) and to the name-based sign registry; register bucket abl_new."""
    from src.data import loading as LD

    orig = LD.add_derived_features
    NEW = list(AA.NEW_SERIES)

    def add_derived_plus(df: pd.DataFrame) -> list[str]:
        added = orig(df)
        rv = df["RV"].astype("float64")
        # RS- = RV - RS+ (>= 0 on every bar of data/core_stats.parquet; the clip guards rounding only)
        df["sumnret2"] = (rv - df["sumpret2"].astype("float64")).clip(lower=0.0)
        # RV x sqrt(RQ), RQ = (N / 3) sum r^4 (BPQ's HARQ interaction)
        rq = df["numobs"].astype("float64") / 3.0 * df["sumret4"].astype("float64")
        df["rvsqrq"] = rv * np.sqrt(rq)
        # the half hour's negative return (Corsi and Reno's leverage term)
        df["negret"] = df["sumret"].astype("float64").clip(upper=0.0)
        return added + NEW

    LD.add_derived_features = add_derived_plus
    LD.NON_NEGATIVE_SERIES = frozenset(LD.NON_NEGATIVE_SERIES | {"sumnret2", "rvsqrq"})
    LD.SIGNED_SERIES = frozenset(LD.SIGNED_SERIES | {"negret"})
    LD.SUBGROUPS["abl_new"] = list(AA.ABL_NEW)


def cmd_newseries(a) -> int:
    register_new_series()
    import pool_linear_pretune as PLP

    out = Path(a.out).resolve()
    sys.argv = ["pool_linear_pretune.py", "design", "--bucket", "abl_new", "--out", str(out), "--data", a.data, *CFG]
    say(f"newseries: {' '.join(sys.argv[1:])}")
    PLP.main()
    m = meta(out)
    assert m["raw_cols"] == AA.ABL_NEW, m["raw_cols"]
    say(f"newseries done: {m['shape']}, {len(m['names'])} columns")
    return 0


# ============================================================================ gate 0b
def cmd_cmpnew(a) -> int:
    new, full = Path(a.new), Path(a.full)
    mn, mf = meta(new), meta(full)
    Xn, Xf = load_X(new), load_X(full)
    rec: dict = {"new": str(new), "full": str(full), "shape_new": list(Xn.shape), "fail": [], "columns": {}}
    if mn.get("config") != mf.get("config"):
        rec["fail"].append(f"config {mn.get('config')} != {mf.get('config')}")
    common = [n for n in mn["names"] if n in mf["names"]]
    want = set(AA.HAR12) | {c for s in AA.NEW_REF for c in AA.series_cols(s)}
    assert want <= set(common), sorted(want - set(common))
    for n in common:
        x, y = np.asarray(Xn[:, mn["names"].index(n)]), np.asarray(Xf[:, mf["names"].index(n)])
        eq = bool(np.array_equal(x, y) and x.tobytes() == y.tobytes())
        rec["columns"][n] = eq if eq else float(np.nanmax(np.abs(x - y)))
        # 2026-10-25 addendum A (b): sumret4's value averages within POW_COLS_TOL (recorded); was: every column exact
        pow_ok = n.startswith("adj_sumret4_ma_") and rec["columns"][n] <= POW_COLS_TOL
        if not eq and not pow_ok:
            rec["fail"].append(f"column {n} differs (max abs {rec['columns'][n]:.3e})")
    for f in ("y.npy", "b_oos.npy", "dates_oos.csv"):
        if (new / f).read_bytes() != (full / f).read_bytes():
            rec["fail"].append(f"{f} differs")
    newc = [n for n in mn["names"] if n not in common]
    rec["new_columns"] = newc
    rec["new_columns_stats"] = {
        n: dict(
            mean=float(np.mean(Xn[W:, mn["names"].index(n)])),
            sd=float(np.std(Xn[W:, mn["names"].index(n)])),
            zeros=int(np.sum(Xn[:, mn["names"].index(n)] == 0.0)),
            finite=bool(np.isfinite(Xn[:, mn["names"].index(n)]).all()),
        )
        for n in newc
    }
    for n, st in rec["new_columns_stats"].items():
        if not st["finite"]:
            rec["fail"].append(f"new column {n} not finite")
    need = {c for v in AA.CLASSICAL.values() for c in v}
    if not need <= set(mn["names"]):
        rec["fail"].append(f"classical columns missing: {sorted(need - set(mn['names']))}")
    rec["pass"] = not rec["fail"]
    Path(a.out).write_text(json.dumps(rec, indent=1))
    say(
        f"gate 0b {'PASS' if rec['pass'] else 'FAIL'}: {len(common)} common columns, {len(newc)} new; {rec['fail'][:5]}"
    )
    return 0 if rec["pass"] else 3


# ============================================================================ the linear arms' caches
def write_subset(src: Path, names: list[str], out: Path, arm: str, bucket: str, extra: Path | None = None) -> dict:
    """The arm's cache: the columns `names` of `src` (F order), and (2026-10-25 addendum A (b)) columns `src` does not
    have from `extra` (the classical arms: HAR, calendar and RS+ from the full cache, the new series from abl_new)."""
    m = meta(src)
    X = load_X(src)
    sub = np.empty((X.shape[0], len(names)), dtype=np.float64, order="F")
    me = meta(extra) if extra is not None else None
    Xe = load_X(extra) if extra is not None else None
    from_extra = []
    for j, n in enumerate(names):
        if n in m["names"]:
            sub[:, j] = X[:, m["names"].index(n)]
        else:
            assert me is not None and Xe is not None and n in me["names"], n
            sub[:, j] = Xe[:, me["names"].index(n)]
            from_extra.append(n)
    if extra is not None:
        for f in ("y.npy", "b_oos.npy", "dates_oos.csv"):
            assert (src / f).read_bytes() == (extra / f).read_bytes(), f
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "X.npy", sub)
    for f in ("y.npy", "b_oos.npy", "dates_oos.csv"):
        shutil.copyfile(src / f, out / f)
    raw = []
    for n in names:
        s = AA.series_of(n)
        if s is not None and s not in raw:
            raw.append(s)
    mm = dict(
        m,
        bucket=bucket,
        raw_cols=raw,
        names=names,
        shape=list(sub.shape),
        f_order=bool(sub.flags.f_contiguous),
        ablation=dict(
            arm=arm,
            source=str(src),
            source_x_md5=md5(src / "X.npy"),
            n_cols=len(names),
            from_extra=from_extra,
            extra=str(extra) if extra is not None else None,
        ),
    )
    (out / "design.json").write_text(json.dumps(mm))
    return dict(arm=arm, bucket=bucket, n_cols=len(names), x_md5=md5(out / "X.npy"), source=str(src))


def cmd_subsets(a) -> int:
    full, new, root = Path(a.full), Path(a.new), Path(a.root)
    gs = AA.group_series()
    fn = meta(full)["names"]
    nn = meta(new)["names"]
    recs = []
    # 2026-10-26 addendum F: --arms builds only the listed arms (e.g. add_rm_fomc), the other caches left as they are
    only = [x for x in a.arms.split(",") if x]
    if not only or "har_sub" in only:
        recs.append(write_subset(full, AA.arm_names("har", fn, gs), root / "har_sub", "har_sub", "baseline"))
    for arm in [x for x in only if x != "har_sub"] or AA.LINEAR_ARMS:
        # 2026-10-25 addendum A (b): a classical arm takes every column the full cache has from it (HAR, calendar,
        # RS+) and only the new series from abl_new.  Was: every column of a classical arm from abl_new.
        cols = AA.arm_names(arm, nn if arm in AA.CLASSICAL else fn, gs)
        assert len(cols) == AA.n_cols(arm), (arm, len(cols), AA.n_cols(arm))
        extra = new if arm in AA.CLASSICAL else None
        recs.append(write_subset(full, cols, root / arm, arm, "all_features", extra=extra))
        say(f"subset {arm}: {len(cols)} columns")
    pd.DataFrame(recs).to_csv(root / ("subsets.csv" if not only else f"subsets_{'_'.join(only)}.csv"), index=False)
    return 0


# ============================================================================ late-start windows
def cmd_latestart(a) -> int:
    from src.data.loading import load_raw_data

    full = Path(a.full)
    m = meta(full)
    names = m["names"]
    X = load_X(full)
    n = X.shape[0]
    d_oos = pd.DatetimeIndex(pd.read_csv(full / "dates_oos.csv", parse_dates=["date"])["date"])
    assert len(d_oos) == n - W
    d_all = None
    if (full / "dates_all.csv").is_file():
        d_all = pd.DatetimeIndex(pd.read_csv(full / "dates_all.csv", parse_dates=["date"])["date"])
        assert len(d_all) == n and d_all[W:].equals(d_oos), "dates_all"
    ev = np.flatnonzero((d_oos.normalize() >= FIRST_EVAL) & (d_oos.normalize() <= LAST_EVAL)) + W
    df = load_raw_data(a.data, allow_missing=True)
    gs = AA.group_series()
    grp = {s: g for g, v in gs.items() for s in v}
    rule = pd.read_csv(a.rule) if a.rule else None
    if rule is not None:
        rule = rule[rule["window"] == "w3000"].set_index("column")
    rows = []
    sers: set[str] = {x for x in (AA.series_of(c) for c in names) if x is not None}
    for s in sorted(sers, key=lambda u: (list(gs).index(grp[u]), u)):
        obs = df[s].notna().to_numpy() if s in df.columns else np.zeros(len(df), bool)
        first = df["t"].iloc[int(np.argmax(obs))] if obs.any() else pd.NaT
        last = df["t"].iloc[len(obs) - 1 - int(np.argmax(obs[::-1]))] if obs.any() else pd.NaT
        for k in AA.HAR_K:
            c = f"adj_{s}_ma_{k}"
            j = names.index(c)
            col = np.asarray(X[:, j])
            nz = np.flatnonzero(col != 0.0)
            z = int(nz[0]) if len(nz) else n
            live_ev = int(np.count_nonzero(col[ev] != 0.0))
            rec = dict(
                group=grp[s],
                series=s,
                k=k,
                column=c,
                first_print=str(first),
                last_print=str(last),
                zero_prefix_rows=z,
                first_live_row=z - W if z > 0 else 0,
                first_live_date=(
                    str(d_all[z - W]) if d_all is not None and z >= W else ("" if z == 0 else "< design row 39,000")
                ),
                live_from=str(d_oos[z - W])
                if W <= z < n
                else ("whole design" if z == 0 else ("< row 39,000" if z < W else "never")),
                eval_rows=len(ev),
                eval_rows_nonzero=live_ev,
                eval_share_nonzero=live_ev / len(ev),
            )
            if rule is not None and c in rule.index:
                rz = int(rule.loc[c, "zero_prefix_rows"])
                rec["rule_zero_prefix_rows"] = rz
                # 2026-10-25 addendum B (a): UNOBSERVED_NEUTRAL may zero rows right after the late-start prefix, so the
                # first non-zero row is at or after the rule's prefix.  Was: rule_match = (prefix == z)
                rec["rule_exact"] = rz == z
                rec["rule_match"] = z >= rz
            rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    bad = out[out.get("rule_match", pd.Series(True, index=out.index)) == False]  # noqa: E712
    say(
        f"latestart: {len(out)} value columns, {int((out['zero_prefix_rows'] > 0).sum())} with a zero prefix; rule mismatches {len(bad)}"
    )
    return 0 if len(bad) == 0 else 3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "mode",
        choices=("check", "newseries", "cmpnew", "subsets", "latestart", "checkfix", "buildmac", "checkmac", "submac"),
    )
    ap.add_argument("--mac", default="", help="checkmac / submac: the macro cache (addendum L)")
    ap.add_argument("--full", default="")
    ap.add_argument("--har", default="")
    ap.add_argument("--new", default="")
    ap.add_argument("--root", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--data", default="data")
    ap.add_argument("--rule", default="")
    ap.add_argument("--arms", default="", help="subsets: only these arms, comma-separated (addendum F)")
    a = ap.parse_args()
    os.chdir(ROOT)
    return {
        "check": cmd_check,
        "newseries": cmd_newseries,
        "cmpnew": cmd_cmpnew,
        "subsets": cmd_subsets,
        "latestart": cmd_latestart,
        "checkfix": cmd_checkfix,
        "buildmac": cmd_buildmac,
        "checkmac": cmd_checkmac,
        "submac": cmd_submac,
    }[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
