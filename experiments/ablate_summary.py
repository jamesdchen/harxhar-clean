"""Markdown tables of the ablation scores for SUMMARY.md (results/ablations_2014start_w3000/scores/, from
experiments/ablate_score.py): QLIKE change in percent against the arm's reference (lo_*: the full set; add_*, the
classical arms and the full set: the HAR benchmark) with the Diebold-Mariano statistic in parentheses.

usage: python experiments/ablate_summary.py [--span all|c16|c16_2020] > tables.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ablate_arms as AA  # noqa: E402

RES = HERE.parent / "results" / "ablations_2014start_w3000" / "scores"
CLS = ["ols", "ridge", "lasso", "enet", "lgbm", "xgb", "rf"]
NAME = {"ols": "OLS", "ridge": "Ridge", "lasso": "Lasso", "enet": "EN", "lgbm": "LGBM", "xgb": "XGB", "rf": "RF"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--span", default="all")
    a = ap.parse_args()
    M = pd.read_csv(RES / "metrics.csv")
    D = pd.read_csv(RES / "dm.csv")
    q = M.set_index(["cls", "arm", "span"])["qlike"]
    dq = D[D["loss"] == "qlike"].set_index(["cls", "arm", "vs_cls", "vs_arm", "span"])

    def cell(c: str, arm: str, ref: str, sp: str, rc: str | None = None) -> str:
        rc = rc or c
        if (c, arm, sp) not in q.index or (rc, ref, sp) not in q.index:
            return "--"
        p = 100 * (q[(c, arm, sp)] / q[(rc, ref, sp)] - 1)
        k = (c, arm, rc, ref, sp)
        return f"{p:+.1f} ({dq.loc[k, 'dm']:+.1f})" if k in dq.index else f"{p:+.1f}"

    def table(title: str, rows: list[tuple[str, str, str]], sp: str) -> None:
        print(f"\n**{title}** (span {sp})\n")
        print("| | " + " | ".join(NAME[c] for c in CLS) + " |")
        print("|---" * (len(CLS) + 1) + "|")
        for lab, arm, ref in rows:
            print(f"| {lab} | " + " | ".join(cell(c, arm, ref, sp) for c in CLS) + " |")

    sp = a.span
    table("Full set without the group vs the full set", [(AA.GROUP_LABEL[g], f"lo_{g}", "full") for g in AA.GROUPS], sp)
    table("HAR plus the group vs HAR", [(AA.GROUP_LABEL[g], f"add_{g}", "har") for g in AA.GROUPS], sp)
    table(
        "Classical extensions vs HAR",
        [(b, b, "har") for b in ("shar", "harq", "lhar", "sharql")] + [("Full set", "full", "har")],
        sp,
    )
    print(f"\n**Full set vs augmented HAR** (span {sp}; DM of the class's full set against the benchmark)\n")
    print("| benchmark | " + " | ".join(NAME[c] for c in CLS[1:]) + " |")
    print("|---" * len(CLS) + "|")
    for lab, bc, b in (
        ("own HAR + realized moments", None, "add_rm"),
        ("OLS HAR + realized moments", "ols", "add_rm"),
        ("OLS SHAR+HARQ+LHAR", "ols", "sharql"),
        ("OLS HAR", "ols", "har"),
    ):
        cells = []
        for c in CLS[1:]:
            rc = bc or c
            if lab.startswith("own"):
                # the full set against the own add_rm (the stored test is add_rm vs full: DM sign flipped)
                k = (c, "add_rm", c, "full", sp)
                cells.append(
                    f"{100 * (q[(c, 'full', sp)] / q[(c, 'add_rm', sp)] - 1):+.1f} ({-dq.loc[k, 'dm']:+.1f})"
                    if k in dq.index
                    else "--"
                )
            else:
                cells.append(cell(c, "full", b, sp, rc))
        print(f"| {lab} | " + " | ".join(cells) + " |")
    if sp == "all":
        for key, g in (("sent", "sent"), ("iv", "iv"), ("ivb", "iv"), ("vd", "vd")):
            table(
                f"Live window {key}: group added / removed",
                [("added", f"add_{g}", "har"), ("removed", f"lo_{g}", "full"), ("full set vs HAR", "full", "har")],
                f"live_{key}",
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
