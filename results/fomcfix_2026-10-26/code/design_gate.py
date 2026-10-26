"""Design gates of results/fomcfix_2026-10-26/RULE.txt (G2, G3, G3b, GC, GD): two design roots compared, read-only.

A design root holds _work/design_{baseline,all_features}/ (X.npy, y.npy, b_oos.npy, dates_oos.csv, design.json) as
experiments/pooled_linear_rerun_2026_10_19_13.py design writes them.

  --expect identical  every file of both sets md5-equal (X, y, b_oos, dates_oos; the column names equal)        G2
  --expect fomc       the baseline set md5-equal; the full set's y, b_oos, dates_oos md5-equal and X equal bit for
                      bit in every column except the 12 value averages adj_fomc_{until,since}_inv_ma_k            G3, GD
                      (--first-stamp T: and no row before the design row of bar T differs, G3b)
  --expect holiday    the new design = the reference without the rows of the holiday sessions
                      (src/data/loading.py NYSE_HOLIDAY_FUTURES_SESSIONS): its OOS dates = the reference's OOS dates
                      from its own first OOS date on, less the holiday dates; every 2014-01-02 .. 2024-04-30 row of the
                      reference outside the holidays present; the row counts reported                               GC
Every differing column is reported with its first and last differing row (date where the row is an OOS row), the
number of differing rows and the largest absolute difference.

usage: python design_gate.py <reference root> <new root> <out json> --expect identical|fomc|holiday [--first-stamp T]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CODE))
FILES = ("X.npy", "y.npy", "b_oos.npy", "dates_oos.csv")
SETS = ("baseline", "all_features")
FOMC_VALUE = {f"adj_fomc_{s}_inv_ma_{k}" for s in ("until", "since") for k in (1, 5, 25, 125, 625, 3125)}


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def dz(root: Path, s: str) -> Path:
    return root / "_work" / f"design_{s}"


def meta(root: Path, s: str) -> dict:
    return json.loads((dz(root, s) / "design.json").read_text())


def dates(root: Path, s: str) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.read_csv(dz(root, s) / "dates_oos.csv", parse_dates=["date"])["date"])


def col_diff(ref: Path, new: Path, s: str) -> list[dict]:
    """Column-by-column comparison of two X.npy of the same shape and names."""
    Xa = np.load(dz(ref, s) / "X.npy", mmap_mode="r")
    Xb = np.load(dz(new, s) / "X.npy", mmap_mode="r")
    names = meta(ref, s)["names"]
    assert Xa.shape == Xb.shape and names == meta(new, s)["names"], (Xa.shape, Xb.shape)
    W = int(meta(ref, s)["config"]["window"])
    d = dates(ref, s)
    out = []
    for j, n in enumerate(names):
        a, b = np.asarray(Xa[:, j]), np.asarray(Xb[:, j])
        neq = ~((a == b) | (np.isnan(a) & np.isnan(b)))
        if not neq.any():
            continue
        r = np.flatnonzero(neq)
        lab = [str(d[i - W]) if i >= W else f"pre-OOS row {i}" for i in (r[0], r[-1])]
        out.append(
            dict(
                column=n,
                n_rows=int(neq.sum()),
                first_row=int(r[0]),
                first=lab[0],
                last_row=int(r[-1]),
                last=lab[1],
                max_abs=float(np.nanmax(np.abs(a[neq] - b[neq]))),
            )
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("new")
    ap.add_argument("out")
    ap.add_argument("--expect", choices=("identical", "fomc", "holiday"), required=True)
    ap.add_argument("--first-stamp", default="")
    a = ap.parse_args()
    ref, new = Path(a.ref).resolve(), Path(a.new).resolve()
    rec: dict = dict(ref=str(ref), new=str(new), expect=a.expect, fail=[])
    for s in SETS:
        rec[s] = {
            "md5_ref": {f: md5(dz(ref, s) / f) for f in FILES},
            "md5_new": {f: md5(dz(new, s) / f) for f in FILES},
            "shape_ref": meta(ref, s)["shape"],
            "shape_new": meta(new, s)["shape"],
            "config_new": meta(new, s).get("config"),
        }
        same = {f: rec[s]["md5_ref"][f] == rec[s]["md5_new"][f] for f in FILES}
        rec[s]["md5_equal"] = same
        names_eq = meta(ref, s)["names"] == meta(new, s)["names"]
        if not names_eq:
            rec["fail"].append(f"{s}: column names differ")
        if a.expect == "identical" or (a.expect == "fomc" and s == "baseline"):
            if not all(same.values()):
                rec["fail"].append(f"{s}: md5 differ {[f for f, v in same.items() if not v]}")
        elif a.expect == "fomc":
            for f in FILES[1:]:
                if not same[f]:
                    rec["fail"].append(f"{s}: {f} differs")
            cd = col_diff(ref, new, s)
            rec[s]["differing_columns"] = cd
            bad = [c["column"] for c in cd if c["column"] not in FOMC_VALUE]
            if bad:
                rec["fail"].append(f"{s}: columns outside the FOMC value averages differ: {bad[:10]}")
            if a.first_stamp and cd:
                d = dates(ref, s)
                W = int(meta(ref, s)["config"]["window"])
                r0 = W + int(np.flatnonzero(d >= pd.Timestamp(a.first_stamp))[0])
                early = [c["column"] for c in cd if c["first_row"] < r0]
                rec[s]["first_stamp_row"] = r0
                if early:
                    rec["fail"].append(f"{s}: rows before {a.first_stamp} (row {r0}) differ in {early}")
        else:  # holiday
            from src.data.loading import NYSE_HOLIDAY_FUTURES_SESSIONS as HOL

            hol = pd.DatetimeIndex(pd.to_datetime(list(HOL)))
            dr, dn = dates(ref, s), dates(new, s)
            want = dr[(dr >= dn[0]) & ~dr.normalize().isin(hol)]
            rec[s]["oos_first_ref"], rec[s]["oos_first_new"] = str(dr[0]), str(dn[0])
            rec[s]["rows_ref"], rec[s]["rows_new"] = len(dr), len(dn)
            rec[s]["holiday_rows_ref_oos"] = int(dr.normalize().isin(hol).sum())
            if not dn.equals(want):
                rec["fail"].append(f"{s}: OOS dates are not the reference's less the holiday rows")
            if dn.normalize().isin(hol).any():
                rec["fail"].append(f"{s}: holiday rows left")
            span = dr[(dr >= "2014-01-02") & (dr < "2024-05-01") & ~dr.normalize().isin(hol)]
            if not span.isin(dn).all():
                rec["fail"].append(f"{s}: 2014-2024 non-holiday rows missing")
            rec[s]["holiday_rows_2014_2024_ref"] = int(
                (dr.normalize().isin(hol) & (dr >= "2014-01-02") & (dr < "2024-05-01")).sum()
            )
    rec["pass"] = not rec["fail"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: v for k, v in rec.items() if k in ("expect", "fail", "pass")}, indent=1))
    for s in SETS:
        print(s, "md5_equal", rec[s]["md5_equal"])
        for c in rec[s].get("differing_columns", []):
            print("  ", c)
    return 0 if rec["pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
