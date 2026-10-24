"""Close-only window sweep of the pooled linear models and trees (results/closeonly_wsweep_2014start/RULE.txt, written
before any job): the shared constants and the design stage.

The close sample at window W is the 16:00 rows of the pooled window sweep's 13-bar design at W
(results/window_sweep_2014start_2026-10-22/w<W>, CARC /scratch1/jc_905/wsweep/...; its scaler and late-start warm-up at
13 W rows).  The sweep's design caches do not store dates_all.csv (the date of every design row); stage copies them
off /scratch1 (md5-checked) and adds the linear three-way comparison's rebuilt RTH13 dates_all.csv after checking that
every W design has the same rows (RULE.txt section 1).

usage: python experiments/closewin.py stage --sweep <window sweep results dir> --rth13 <three-way RTH13 _work dir>
          --design-root <dest> --out <dir for design_check.json>
  -> <dest>/RTH13_w<W>/_work/design_<set>/ {X.npy, y.npy, b_oos.npy, dates_oos.csv, design.json, dates_all.csv,
     MD5SUMS}; <out>/design_check.json (exit 3 on a failed check)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WS = (1000, 1500, 2000, 2500, 3000, 3500)
W_REF, W_FIRST = 3000, 1000
SETS = ("baseline", "all_features")
COPY = ("X.npy", "y.npy", "b_oos.npy", "dates_oos.csv", "design.json")
FLAGS_RTH13 = {  # the template's executor flags (experiments/trees_train3way.py FLAGS["RTH13"])
    "DROP_ACTIVITY_MA": True,
    "CALENDAR_THROUGH_OVERNIGHT": True,
    "DROP_OVERNIGHT_INDICATOR": True,
    "LEAN_GLOBAL_LAGS": True,
    "LATE_START_WARMUP": True,
}
# the three-way templates' design = the sweep's w3000 design (experiments/trees_train3way.py TEMPLATE_MD5)
TEMPLATE_MD5 = {
    "baseline": {
        "X.npy": "05da4b5b30ec4579df77f1a3b6dd57ed",
        "y.npy": "996504f16da7982d64656d1ba0d9722a",
        "b_oos.npy": "7670f01a3055265dfac28ebac1eca864",
        "dates_oos.csv": "d8a203fe273cd3e8e34987fe5a73048e",
    },
    "all_features": {
        "X.npy": "374bea97f35e33a02404dc47c82c3138",
        "y.npy": "996504f16da7982d64656d1ba0d9722a",
        "b_oos.npy": "7670f01a3055265dfac28ebac1eca864",
        "dates_oos.csv": "d8a203fe273cd3e8e34987fe5a73048e",
    },
}


def design_name(W: int) -> str:
    return f"RTH13_w{W}"


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def cmd_stage(a) -> int:
    sweep, rth13, dest = Path(a.sweep).resolve(), Path(a.rth13).resolve(), Path(a.design_root).resolve()
    g: dict = {"sweep": str(sweep), "rth13": str(rth13), "design_root": str(dest)}
    ok = True
    for s in SETS:
        ref_dates = rth13 / f"design_{s}" / "dates_all.csv"
        da = pd.read_csv(ref_dates, dtype=str)["date"]
        y_ref = md5(rth13 / f"design_{s}" / "y.npy")
        names0: list[str] | None = None
        b0: tuple[int, np.ndarray] | None = None
        for W in WS:
            src = sweep / f"w{W}" / "_work" / f"design_{s}"
            dst = dest / design_name(W) / "_work" / f"design_{s}"
            dst.mkdir(parents=True, exist_ok=True)
            r: dict = {}
            m_src = {fn: md5(src / fn) for fn in COPY}
            for fn in COPY:
                tmp = dst / (fn + ".part")
                shutil.copyfile(src / fn, tmp)
                tmp.replace(dst / fn)
            shutil.copyfile(ref_dates, dst / "dates_all.csv")
            m_dst = {fn: md5(dst / fn) for fn in COPY + ("dates_all.csv",)}
            m_src2 = {fn: md5(src / fn) for fn in COPY}  # the source unchanged after the copy
            r["copy_md5_equal"] = all(m_dst[fn] == m_src[fn] == m_src2[fn] for fn in COPY)
            r["md5"] = m_dst
            (dst / "MD5SUMS").write_text("".join(f"{h}  {fn}\n" for fn, h in m_dst.items()))
            meta = json.loads((dst / "design.json").read_text())
            cf = meta["config"]
            n = int(meta["shape"][0])
            r["config_ok"] = bool(
                cf["sample"] == "rth13"
                and int(cf["periods_per_day"]) == 13
                and int(cf["window"]) == 13 * W
                and cf["har_base"] == "5"
                and cf["ex_flags"] == FLAGS_RTH13
                and list(meta["shape"]) == [n, len(meta["names"])]
            )
            # (a) the same rows in every W: the W-independent target y, the row count, the OOS dates
            oos = pd.read_csv(dst / "dates_oos.csv", dtype=str)["date"]
            r["rows"] = n
            r["y_md5_equal_rth13"] = m_dst["y.npy"] == y_ref
            r["dates_oos_equal_tail"] = bool(
                len(da) == n and len(oos) == n - 13 * W and da.iloc[13 * W :].reset_index(drop=True).equals(oos)
            )
            # (b) the column names equal across W
            names0 = names0 or meta["names"]
            r["names_equal_w1000"] = meta["names"] == names0
            # (c) W = 3,000: the three-way templates' design
            if W == W_REF:
                r["template_md5_equal"] = all(m_dst[fn] == h for fn, h in TEMPLATE_MD5[s].items())
            # (d) B at the common OOS rows equal to W = 1,000's (B is built on the full grid)
            b = np.load(dst / "b_oos.npy")
            if b0 is None:
                b0 = (13 * W, b)
                r["b_equal_w1000_common"] = True
            else:
                off = 13 * W - b0[0]
                r["b_equal_w1000_common"] = bool(
                    len(b) == len(b0[1]) - off and np.array_equal(b, b0[1][off:], equal_nan=True)
                )
            # the 16:00 rows (the close sample) and the first one with a stored B
            k16 = np.flatnonzero(da.str.slice(11, 16).to_numpy() == "16:00")
            r["rows_1600"] = int(len(k16))
            r["first_1600_oos"] = da.iloc[int(k16[np.searchsorted(k16, 13 * W)])]
            good = all(v for k, v in r.items() if isinstance(v, bool))
            r["pass"] = bool(good)
            ok &= good
            g[f"{design_name(W)}/{s}"] = r
            print(f"{design_name(W)} {s}: {json.dumps(r)}", flush=True)
    g["pass"] = bool(ok)
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "design_check.json").write_text(json.dumps(g, indent=1))
    print(f"DESIGN CHECK {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("stage",))
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--rth13", required=True)
    ap.add_argument("--design-root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    return {"stage": cmd_stage}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
