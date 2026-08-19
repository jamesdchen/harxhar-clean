"""Emit macros + table bodies for the dense direct multi-horizon result.

Reads results/direct_mh/dense_{byhorizon,path}.csv (from
direct_multihorizon_reduce.py) and writes
    writeup/generated/dense_numbers.tex        scalar macros (\\dmhXxx)
    writeup/generated/table_dense_byk.tex      per-horizon body (era all)
    writeup/generated/table_dense_path.tex     cumulative-sum body
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "direct_mh")
GEN = os.path.join(ROOT, "writeup", "generated")
_W = {"1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine", "10": "Ten", "11": "Eleven"}


def _f(x: Any, nd: int = 2, sign: bool = False) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "---"
    v = float(x)
    s = f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"
    return s.replace("-", "$-$") if v < 0 else s


def _int(x: Any) -> str:
    return f"{int(x):,}".replace(",", "{,}")


def main() -> None:
    os.makedirs(GEN, exist_ok=True)
    m: dict[str, str] = {}
    b = pd.read_csv(os.path.join(RES, "dense_byhorizon.csv"))
    p = pd.read_csv(os.path.join(RES, "dense_path.csv"))
    rows = []
    for era, etag in (("all", ""), ("rth", "Rth")):
        s = b[b.era == era].sort_values("k")
        for _, r in s.iterrows():
            k = int(r["k"])
            m[f"dmh{etag}K{_W[str(k)]}Azero"] = _f(r["q_a0"], 4)
            m[f"dmh{etag}K{_W[str(k)]}Blk"] = _f(r["q_blk2"], 4)
            m[f"dmh{etag}K{_W[str(k)]}DM"] = _f(r["dm"], 1, sign=True)
            if era == "all":
                rows.append(
                    f"$t+{k}$ & {_int(r['n'])} & {_f(r['q_a0'], 4)} & {_f(r['q_blk2'], 4)} & "
                    f"{_f(r['dq'], 4, sign=True)} & {_f(r['dm'], 1, sign=True)} \\\\"
                )
        m[f"dmh{etag}N"] = _int(s["n"].max())
        m[f"dmh{etag}DMmin"] = _f(s["dm"].min(), 1, sign=True)
        m[f"dmh{etag}DMmax"] = _f(s["dm"].max(), 1, sign=True)
    rows.append("\\bottomrule")
    with open(os.path.join(GEN, "table_dense_byk.tex"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(rows) + "\n")
    prow = []
    for _, r in p[p.era == "all"].sort_values("bars_summed").iterrows():
        kk = int(r["bars_summed"])
        prow.append(
            f"{kk} & {_int(r['n'])} & {_f(r['q_a0'], 4)} & {_f(r['q_blk2'], 4)} & "
            f"{_f(r['dq'], 4, sign=True)} & {_f(r['dm'], 1, sign=True)} \\\\"
        )
        m[f"dmhPath{_W[str(kk)]}DM"] = _f(r["dm"], 1, sign=True)
    prow.append("\\bottomrule")
    with open(os.path.join(GEN, "table_dense_path.tex"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(prow) + "\n")
    m["dmhRefit"] = "48"
    lines = ["% GENERATED FILE -- do not edit by hand.", "% Written by experiments/direct_mh_tex.py from results/direct_mh/dense_*.csv."]
    for k in sorted(m):
        lines.append(f"\\newcommand{{\\{k}}}{{{m[k]}}}")
    with open(os.path.join(GEN, "dense_numbers.tex"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {len(m)} macros; byk rows {len(rows)-1}; path rows {len(prow)-1}", flush=True)
    for k in ("dmhKOneDM", "dmhKElevenDM", "dmhRthKOneDM", "dmhRthKElevenDM", "dmhPathElevenDM"):
        print(f"  {k} = {m[k]}", flush=True)


if __name__ == "__main__":
    main()
