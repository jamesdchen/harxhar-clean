"""Emit the multi-horizon section's macros and table bodies for the writeup.

Reads results/multihorizon/ttc_{pooled,byhour,coefs}.csv and writes
    writeup/generated/multihorizon_numbers.tex     scalar macros (\\mhXxx)
    writeup/generated/table_multihorizon_pooled.tex  pooled table body
    writeup/generated/table_multihorizon_byhour.tex  by-entry-hour body (M2)

Every number in the section is minted here from the CSVs.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "multihorizon")
GEN = os.path.join(ROOT, "writeup", "generated")

_WORD = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five"}


def _f(x: Any, nd: int = 2, sign: bool = False) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "---"
    v = float(x)
    s = f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"
    return s.replace("-", "$-$") if v < 0 else s


def _int(x: Any) -> str:
    return f"{int(x):,}".replace(",", "{,}")


def _cs(name: str) -> str:
    """Letters only for TeX control sequences."""
    out = []
    for c in name:
        if c.isdigit():
            out.append(_WORD[c])
        elif c == "_":
            continue
        else:
            out.append(c)
    return "".join(out)


def _hour_tag(h: str) -> str:
    return "H" + _cs(h.replace(":", ""))


def main() -> None:
    os.makedirs(GEN, exist_ok=True)
    macros: dict[str, str] = {}
    pl = pd.read_csv(os.path.join(RES, "ttc_pooled.csv"))
    bh = pd.read_csv(os.path.join(RES, "ttc_byhour.csv"))

    # ---- pooled table: rows = constructions, cols = a0 QLIKE, blk2 QLIKE, DM blk2 vs a0, DM vs naive (blk2)
    names = {
        "M1": "One-step forecast, mapped to the horizon (M1)",
        "M2": "\\quad $+$ realized so far this session (M2)",
        "M3": "\\quad $+$ previous session, same horizon (M3)",
        "M2pooled": "M2, one regression with hour dummies",
    }
    rows = []
    for era, elab in (("all", "All sessions"), ("post2016", "2016--2024")):
        p = pl[pl.era == era]
        n0 = p[p.forecast == "M0"].iloc[0]
        macros[f"mh{'' if era == 'all' else 'Post'}NDays"] = _int(n0["n_days"])
        macros[f"mh{'' if era == 'all' else 'Post'}NaiveQLIKE"] = _f(n0["qlike"], 4)
        rows.append(
            f"Horizon benchmark (no model) & {elab} & {_f(n0['qlike'], 4)} & --- & --- & --- \\\\"
        )
        for key, name in names.items():
            a = p[p.forecast == f"a0_{key}"].iloc[0]
            b = p[p.forecast == f"blk2_{key}"].iloc[0]
            rows.append(
                f"{name} & {elab} & {_f(a['qlike'], 4)} & {_f(b['qlike'], 4)} & "
                f"{_f(b['dm_blk2_vs_a0'], 1, sign=True)} & {_f(b['dm_vs_naive'], 1, sign=True)} \\\\"
            )
            etag = "" if era == "all" else "Post"
            ktag = _cs(key)
            macros[f"mh{etag}Azero{ktag}QLIKE"] = _f(a["qlike"], 4)
            macros[f"mh{etag}Blk{ktag}QLIKE"] = _f(b["qlike"], 4)
            macros[f"mh{etag}{ktag}DMBlkVsAzero"] = _f(b["dm_blk2_vs_a0"], 1, sign=True)
            macros[f"mh{etag}Blk{ktag}DMVsNaive"] = _f(b["dm_vs_naive"], 1, sign=True)
            macros[f"mh{etag}Azero{ktag}DMVsNaive"] = _f(a["dm_vs_naive"], 1, sign=True)
            if key == "M2pooled":
                macros[f"mh{etag}PooledVsPerHourBlk"] = _f(
                    b["dm_pooled_vs_perhour"], 1, sign=True
                )
                macros[f"mh{etag}PooledVsPerHourAzero"] = _f(
                    a["dm_pooled_vs_perhour"], 1, sign=True
                )
        rows.append("\\midrule")
    if rows and rows[-1] == "\\midrule":
        rows.pop()
    rows.append("\\bottomrule")
    with open(
        os.path.join(GEN, "table_multihorizon_pooled.tex"),
        "w",
        encoding="utf-8",
        newline="\n",
    ) as fh:
        fh.write("\n".join(rows) + "\n")

    # ---- by hour (era all): M2 for a0 and blk2, benchmark, DM blk2 vs a0, DM blk2 vs naive
    rows_h = []
    ba = bh[bh.era == "all"]
    hours = list(dict.fromkeys(ba["hour"]))
    for h in hours:
        n0 = ba[(ba.hour == h) & (ba.forecast == "M0")].iloc[0]
        a = ba[(ba.hour == h) & (ba.forecast == "a0_M2")].iloc[0]
        b = ba[(ba.hour == h) & (ba.forecast == "blk2_M2")].iloc[0]
        rows_h.append(
            f"{h} & {_f(n0['qlike'], 4)} & {_f(a['qlike'], 4)} & {_f(b['qlike'], 4)} & "
            f"{_f(b['dm_blk2_vs_a0'], 1, sign=True)} & {_f(b['dm_vs_naive'], 1, sign=True)} \\\\"
        )
        tag = _hour_tag(h)
        macros[f"mh{tag}BlkMTwoQLIKE"] = _f(b["qlike"], 4)
        macros[f"mh{tag}AzeroMTwoQLIKE"] = _f(a["qlike"], 4)
        macros[f"mh{tag}DMBlkVsAzero"] = _f(b["dm_blk2_vs_a0"], 1, sign=True)
    rows_h.append("\\bottomrule")
    with open(
        os.path.join(GEN, "table_multihorizon_byhour.tex"),
        "w",
        encoding="utf-8",
        newline="\n",
    ) as fh:
        fh.write("\n".join(rows_h) + "\n")

    # summary macros for prose: range of by-hour blk2-vs-a0 DM, count of hours favouring blk2
    dms = ba[ba.forecast == "blk2_M2"]["dm_blk2_vs_a0"].to_numpy(float)
    macros["mhHoursBlkBetter"] = _int((dms < 0).sum())
    macros["mhHoursTotal"] = _int(len(dms))
    macros["mhHoursBlkSig"] = _int((dms < -1.96).sum())
    macros["mhDMBlkVsAzeroMin"] = _f(np.nanmin(dms), 1, sign=True)
    macros["mhDMBlkVsAzeroMax"] = _f(np.nanmax(dms), 1, sign=True)

    # coefficient summary (descriptive): mean slope on one-step, mean on sofar, by model
    co = pd.read_csv(os.path.join(RES, "ttc_coefs.csv"))
    for tag, mtag in (("a0", "Azero"), ("blk2", "Blk")):
        c = co[co.model == tag]
        macros[f"mh{mtag}SlopeOnestepEarly"] = _f(
            c[c.hour == "10:00"]["b_onestep"].iloc[0], 2
        )
        macros[f"mh{mtag}SlopeOnestepLate"] = _f(
            c[c.hour == "14:30"]["b_onestep"].iloc[0], 2
        )
        macros[f"mh{mtag}SlopeSofarEarly"] = _f(
            c[c.hour == "10:00"]["d_sofar"].iloc[0], 2
        )
        macros[f"mh{mtag}SlopeSofarLate"] = _f(
            c[c.hour == "14:30"]["d_sofar"].iloc[0], 2
        )

    lines = [
        "% GENERATED FILE -- do not edit by hand.",
        "% Written by experiments/multihorizon_tex.py from results/multihorizon/*.csv.",
    ]
    for k in sorted(macros):
        lines.append(f"\\newcommand{{\\{k}}}{{{macros[k]}}}")
    with open(
        os.path.join(GEN, "multihorizon_numbers.tex"),
        "w",
        encoding="utf-8",
        newline="\n",
    ) as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {len(macros)} macros + 2 table bodies", flush=True)
    for k in (
        "mhNDays",
        "mhNaiveQLIKE",
        "mhAzeroMOneQLIKE",
        "mhBlkMOneQLIKE",
        "mhMOneDMBlkVsAzero",
        "mhBlkMTwoQLIKE",
        "mhMTwoDMBlkVsAzero",
        "mhHoursBlkBetter",
        "mhHoursTotal",
    ):
        print(f"  {k} = {macros.get(k, '?')}", flush=True)


if __name__ == "__main__":
    main()
