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
    for s, d in (("all_features", full), ("baseline", har)):
        for f, want in TEMPLATE_MD5[s].items():
            got = md5(d / f)
            rec["md5"][f"{s}/{f}"] = got
            if got != want:
                rec["fail"].append(f"md5 {s}/{f} {got} != {want}")
    mf, mh = meta(full), meta(har)
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
        if not eq:
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
def write_subset(src: Path, names: list[str], out: Path, arm: str, bucket: str) -> dict:
    m = meta(src)
    X = load_X(src)
    idx = [m["names"].index(n) for n in names]
    sub = np.asfortranarray(X[:, idx], dtype=np.float64)
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
        ablation=dict(arm=arm, source=str(src), source_x_md5=md5(src / "X.npy"), n_cols=len(names)),
    )
    (out / "design.json").write_text(json.dumps(mm))
    return dict(arm=arm, bucket=bucket, n_cols=len(names), x_md5=md5(out / "X.npy"), source=str(src))


def cmd_subsets(a) -> int:
    full, new, root = Path(a.full), Path(a.new), Path(a.root)
    gs = AA.group_series()
    fn = meta(full)["names"]
    nn = meta(new)["names"]
    recs = []
    recs.append(write_subset(full, AA.arm_names("har", fn, gs), root / "har_sub", "har_sub", "baseline"))
    for arm in AA.LINEAR_ARMS:
        src, names = (new, nn) if arm in AA.CLASSICAL else (full, fn)
        cols = AA.arm_names(arm, names, gs)
        assert len(cols) == AA.n_cols(arm), (arm, len(cols), AA.n_cols(arm))
        recs.append(write_subset(src, cols, root / arm, arm, "all_features"))
        say(f"subset {arm}: {len(cols)} columns")
    pd.DataFrame(recs).to_csv(root / "subsets.csv", index=False)
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
                rec["rule_zero_prefix_rows"] = int(rule.loc[c, "zero_prefix_rows"])
                rec["rule_match"] = rec["rule_zero_prefix_rows"] == z
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
    ap.add_argument("mode", choices=("check", "newseries", "cmpnew", "subsets", "latestart"))
    ap.add_argument("--full", default="")
    ap.add_argument("--har", default="")
    ap.add_argument("--new", default="")
    ap.add_argument("--root", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--data", default="data")
    ap.add_argument("--rule", default="")
    a = ap.parse_args()
    os.chdir(ROOT)
    return {
        "check": cmd_check,
        "newseries": cmd_newseries,
        "cmpnew": cmd_cmpnew,
        "subsets": cmd_subsets,
        "latestart": cmd_latestart,
    }[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
