"""Emit macros + table body for the direct multi-horizon probe (results/multih).

Reads results/multih/h{h}.npz (per-bar contract losses per arm at each
horizon h) and writes
    writeup/generated/multih_numbers.tex           scalar macros (\\mhpXxx)
    writeup/generated/table_multih_profile.tex     horizon-profile table body

Every number in the horizon-profile table is minted here.
"""

from __future__ import annotations

import glob
import os
import sys
from typing import Any

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import score_unification as su  # noqa: E402

RES = os.path.join(ROOT, "results", "multih")
GEN = os.path.join(ROOT, "writeup", "generated")
ARMS = ("a0_ols_har", "blk2_user", "blk3_user")
_WORD = {
    "1": "One",
    "4": "Four",
    "16": "Sixteen",
    "48": "FortyEight",
    "240": "TwoForty",
}
_LABEL = {
    1: "1 bar (30 min)",
    4: "4 bars (2 h)",
    16: "16 bars (8 h)",
    48: "48 bars (1 day)",
    240: "240 bars (5 days)",
}


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
    macros: dict[str, str] = {}
    files = sorted(
        glob.glob(os.path.join(RES, "h*.npz")),
        key=lambda p: int(os.path.basename(p)[1:].split(".")[0]),
    )
    rows = []
    d1: dict[str, float] = {}
    for f in files:
        z = np.load(f)
        h = int(z["horizon"])
        L = {a: z[f"loss_{a}"] for a in ARMS}
        base = L["a0_ols_har"]
        ok_all = np.isfinite(base)
        for a in ARMS:
            ok_all &= np.isfinite(L[a])
        n = int(ok_all.sum())
        q0 = float(np.mean(base[ok_all]))
        tag = _WORD[str(h)]
        macros[f"mhpH{tag}N"] = _int(n)
        macros[f"mhpH{tag}AzeroQLIKE"] = _f(q0, 4)
        cells = [_LABEL[h], _int(n), _f(q0, 4)]
        for a, atag in (("blk2_user", "Blk"), ("blk3_user", "BlkThree")):
            d = L[a][ok_all] - base[ok_all]
            dq = float(np.mean(d))
            dm = float(su.dm_test(L[a][ok_all], base[ok_all], h=1)["dm"])
            q = float(np.mean(L[a][ok_all]))
            macros[f"mhpH{tag}{atag}QLIKE"] = _f(q, 4)
            macros[f"mhpH{tag}{atag}DQ"] = _f(dq, 4, sign=True)
            macros[f"mhpH{tag}{atag}DM"] = _f(dm, 1, sign=True)
            macros[f"mhpH{tag}{atag}Rel"] = _f(100.0 * dq / q0, 1, sign=True)  # % of a0
            if a == "blk2_user":  # product block is parked: table shows a0 vs blk2 only
                cells += [_f(q, 4), _f(dq, 4, sign=True), _f(dm, 1, sign=True)]
            if h == 1:
                d1[a] = dq
        rows.append(" & ".join(cells) + " \\\\")
    rows.append("\\bottomrule")
    with open(
        os.path.join(GEN, "table_multih_profile.tex"),
        "w",
        encoding="utf-8",
        newline="\n",
    ) as fh:
        fh.write("\n".join(rows) + "\n")
    macros["mhpNBars"] = _int(57200)
    macros["mhpNHorizons"] = _int(len(files))
    lines = [
        "% GENERATED FILE -- do not edit by hand.",
        "% Written by experiments/multih_probe_tex.py from results/multih/h*.npz.",
    ]
    for k in sorted(macros):
        lines.append(f"\\newcommand{{\\{k}}}{{{macros[k]}}}")
    with open(
        os.path.join(GEN, "multih_numbers.tex"), "w", encoding="utf-8", newline="\n"
    ) as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {len(macros)} macros; table rows: {len(rows) - 1}", flush=True)
    for k in (
        "mhpHOneBlkDM",
        "mhpHFortyEightBlkDM",
        "mhpHTwoFortyBlkDM",
        "mhpHTwoFortyBlkRel",
        "mhpHFourBlkDM",
    ):
        print(f"  {k} = {macros[k]}", flush=True)


if __name__ == "__main__":
    main()
