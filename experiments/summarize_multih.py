"""Cross-horizon summary of the multi-horizon probe.

Reads results/multih/h*.npz (one per horizon) and prints the horizon profile:
per-arm QLIKE, paired increment vs the benchmark, and DM, per horizon, plus
the decay ratio dQ(h)/dQ(1) that tells whether the exogenous increment
concentrates at short horizons.
"""

import glob
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "experiments"))

import numpy as np

import score_unification as su

ARMS = ("a0_ols_har", "blk2_user", "blk3_user")


def main():
    rows = []
    for f in sorted(
        glob.glob("results/multih/h*.npz"),
        key=lambda p: int(os.path.basename(p)[1:].split(".")[0]),
    ):
        z = np.load(f)
        h = int(z["horizon"])
        L = {a: z[f"loss_{a}"] for a in ARMS}
        base = L["a0_ols_har"]
        row = {"h": h}
        for a in ARMS:
            ok = np.isfinite(L[a]) & np.isfinite(base)
            row[f"q_{a}"] = float(np.nanmean(L[a][ok]))
            if a != "a0_ols_har":
                d = L[a][ok] - base[ok]
                row[f"d_{a}"] = float(np.mean(d))
                row[f"dm_{a}"] = float(su.dm_test(L[a][ok], base[ok], h=1)["dm"])
        rows.append(row)

    hdr = f"{'h':>4} | {'a0 QLIKE':>9} | {'blk2 dQ':>9} {'DM':>6} | {'blk3 dQ':>9} {'DM':>6} | {'blk3/a0':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        ratio = r["q_blk3_user"] / r["q_a0_ols_har"]
        print(
            f"{r['h']:>4} | {r['q_a0_ols_har']:>9.5f} | "
            f"{r['d_blk2_user']:>+9.5f} {r['dm_blk2_user']:>+6.2f} | "
            f"{r['d_blk3_user']:>+9.5f} {r['dm_blk3_user']:>+6.2f} | {ratio:>7.4f}"
        )
    if rows:
        print("\nincrement retention vs h=1 (dQ(h)/dQ(1)):")
        r1 = next((r for r in rows if r["h"] == 1), None)
        if r1:
            for r in rows:
                print(
                    f"  h={r['h']:>3}: blk2 {r['d_blk2_user'] / r1['d_blk2_user']:6.3f}   "
                    f"blk3 {r['d_blk3_user'] / r1['d_blk3_user']:6.3f}"
                )


if __name__ == "__main__":
    main()
