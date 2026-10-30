"""Gate 8(a)-(b) of results/transform_ablation_2026-10-30/RULE.txt: a variant design cache against the incumbent's.

The variants (RULE.txt section 2) are the incumbent design with one switch of src/backtest/executor.py changed, set on
the design stage with --ex-flag.  This compares a variant's cache (experiments/pool_linear_pretune.py design: X.npy,
y.npy, b_oos.npy, dates_oos.csv, design.json) with V0's and checks what may and what must not change:
  every variant   the rows and the column names equal V0's; dates_oos equal; design.json config.ex_flags = V0's plus
                  exactly the variant's flags
  y, b_oos        equal in V4 .. V7 (predictor-side); y differs in V1 .. V3, V3f, V8 .. V10; b_oos equal except V10
                  (B = 1)
  har_ma_*        bit-equal in V4 .. V6 (the HAR set is untouched: on the HAR-only bucket the whole X is md5-equal)
  predictors      every exogenous column bit-equal in V1 .. V3, V3f, V9, V10 (target-side); some differ in V4 .. V6 (on a
                  bucket the switch acts on)
  V0              (--variant V0) X, y, b_oos md5-equal (the default-invariance gate; dates_oos modulo line endings)

usage: python experiments/transform_design_gate.py gate --ref <V0 cache> --new <variant cache> --variant V4 [--out J]
       python experiments/transform_design_gate.py flags --variant V8      (prints the variant's --ex-flag arguments)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

# RULE.txt section 2: key -> (name, {executor flag: value})
VARIANTS: dict[str, tuple[str, dict[str, bool]]] = {
    "V0": ("incumbent", {}),
    "V1": ("target winsor off", {"TARGET_WINSOR": False}),
    "V2": ("target 5/95", {"TARGET_WINSOR_5_95": True}),
    "V3": ("target log", {"TARGET_LOG": True}),
    # 2026-10-30 (RULE.txt addendum D, before any V3 result): the floored log, RV / B >= TARGET_LOG_FLOOR_VALUE
    "V3f": ("target log, floored", {"TARGET_LOG": True, "TARGET_LOG_FLOOR": True}),
    "V4": ("exog diurnal off", {"EXOG_DIURNAL": False}),
    "V5": ("exog name map off", {"EXOG_SEMANTIC": False}),
    "V6": ("feature winsor off", {"FEATURE_WINSOR": False}),
    "V7": ("robust scaling off", {"PRESCALE": False}),
    "V8": (
        "all off",
        {
            "TARGET_WINSOR": False,
            "TARGET_LEVEL": True,
            "EXOG_DIURNAL": False,
            "EXOG_SEMANTIC": False,
            "FEATURE_WINSOR": False,
            "PRESCALE": False,
        },
    ),
    "V9": ("target level", {"TARGET_LEVEL": True}),
    "V10": ("target diurnal off", {"TARGET_DIURNAL": False}),
}
TARGET_SIDE = ("V1", "V2", "V3", "V3f", "V9", "V10")
EXOG_SIDE = ("V4", "V5", "V6")
CALENDAR = re.compile(r"^(DOW_\d|hour|is_overnight)$")


def flag_args(variant: str) -> list[str]:
    return [a for k, v in VARIANTS[variant][1].items() for a in ("--ex-flag", f"{k}={int(v)}")]


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dates_bytes(p: Path) -> bytes:
    return p.read_bytes().replace(b"\r\n", b"\n")


def gate(ref: Path, new: Path, variant: str) -> dict:
    mr, mn = (json.loads((d / "design.json").read_text()) for d in (ref, new))
    fail: list[str] = []
    out: dict = {"variant": variant, "name": VARIANTS[variant][0], "ref": str(ref), "new": str(new)}
    out["md5"] = {d.name + "/" + f: md5(d / f) for d in (ref, new) for f in ("X.npy", "y.npy", "b_oos.npy")}
    names = mn["names"]
    if names != mr["names"] or mn["bucket"] != mr["bucket"]:
        fail.append("column names or bucket differ")
    if dates_bytes(ref / "dates_oos.csv") != dates_bytes(new / "dates_oos.csv"):
        fail.append("dates_oos differ")
    fr = dict(mr.get("config", {}).get("ex_flags", {}))
    fn = dict(mn.get("config", {}).get("ex_flags", {}))
    want = dict(fr, **VARIANTS[variant][1])
    if fn != want:
        fail.append(f"config.ex_flags {fn} != {want}")
    Xr, Xn = np.load(ref / "X.npy", mmap_mode="r"), np.load(new / "X.npy", mmap_mode="r")
    if Xr.shape != Xn.shape:
        fail.append(f"X shape {Xn.shape} vs {Xr.shape}")
        out.update(fail=fail, passed=False)
        return out
    yr, yn = np.load(ref / "y.npy"), np.load(new / "y.npy")
    br, bn = np.load(ref / "b_oos.npy"), np.load(new / "b_oos.npy")
    y_eq, b_eq = np.array_equal(yr, yn, equal_nan=True), np.array_equal(br, bn, equal_nan=True)
    col_eq = {n: bool(np.array_equal(Xr[:, j], Xn[:, j], equal_nan=True)) for j, n in enumerate(names)}
    har = [n for n in names if n.startswith("har_ma_")]
    cal = [n for n in names if CALENDAR.match(n)]
    exo = [n for n in names if n not in har and n not in cal]
    out["changed"] = dict(
        y=not y_eq,
        b_oos=not b_eq,
        har=[n for n in har if not col_eq[n]],
        calendar=[n for n in cal if not col_eq[n]],
        exog_n=sum(not col_eq[n] for n in exo),
        exog_of=len(exo),
    )
    if variant == "V0":
        if not (y_eq and b_eq and all(col_eq.values())):
            fail.append("V0: the rebuilt cache is not bit-equal to the reference")
        if out["md5"][f"{ref.name}/X.npy"] != out["md5"][f"{new.name}/X.npy"]:
            fail.append("V0: X md5 differs")
    else:
        if variant in ("V4", "V5", "V6", "V7") and not y_eq:
            fail.append("y changed in a predictor-side variant")
        if variant not in ("V4", "V5", "V6", "V7") and y_eq:
            fail.append("y unchanged in a target-side variant")
        if variant == "V10":
            if not np.all(bn == 1.0):
                fail.append("V10: b_oos is not 1")
        elif not b_eq:
            fail.append("b_oos changed")
        if variant in EXOG_SIDE:
            if out["changed"]["har"] or out["changed"]["calendar"]:
                fail.append("HAR or calendar columns changed in a predictor-side variant")
            if exo and not out["changed"]["exog_n"]:
                out["note"] = "no predictor column changed (a bucket the switch does not act on)"
            if not exo and out["md5"][f"{ref.name}/X.npy"] != out["md5"][f"{new.name}/X.npy"]:
                fail.append("HAR-only design: X md5 differs")
        if variant in TARGET_SIDE:
            if out["changed"]["exog_n"] or out["changed"]["calendar"]:
                fail.append("predictor or calendar columns changed in a target-side variant")
            if not out["changed"]["har"]:
                fail.append("no har_ma_* column changed in a target-side variant")
    out.update(fail=fail, passed=not fail)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("gate", "flags"))
    ap.add_argument("--ref", default="")
    ap.add_argument("--new", default="")
    ap.add_argument("--variant", required=True, choices=tuple(VARIANTS))
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.mode == "flags":
        print(" ".join(flag_args(a.variant)))
        return 0
    r = gate(Path(a.ref).resolve(), Path(a.new).resolve(), a.variant)
    txt = json.dumps(r, indent=1)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(txt)
    print(txt)
    return 0 if r["passed"] else 3


if __name__ == "__main__":
    sys.exit(main())
