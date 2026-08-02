"""Is "QLIKE rewards attenuated forecasts" a fact about QLIKE, or a mismatch of
geometry?

The paper reports that across its 45-arm battery QLIKE correlates +0.741 with
the Mincer-Zarnowitz slope (p = 5.8e-9): lower QLIKE goes with a lower slope,
read as QLIKE systematically preferring the more attenuated forecast.

The MZ slope is Cov(true, p) / Var(p) -- a LEVEL-space calibration statistic.
QLIKE depends on the forecast only through the ratio r = true/pred, so it is a
RATIO-space loss. Correlating one against the other may be measuring the
mismatch rather than a property of QLIKE. The direct-attenuation test in
analysis/kernel_ceiling.py pointed the same way: whether QLIKE "prefers
attenuation" flipped sign depending on whether the forecast was compressed in
levels or in logs, so each loss wants shrinkage in its own geometry and the
question has no operator-free answer.

The ratio-space analogues of the MZ slope are the diagnostics that settle it:

    mean ratio      E[true/pred], calibrated at 1. This is exactly the
                    quantity QLIKE's own first-order condition pins: under a
                    rescaling p -> c p, QLIKE is minimised at c* = E[true/pred].
    log-log slope   Cov(log true, log pred) / Var(log pred), the MZ regression
                    run in the geometry QLIKE actually sees.

Run over the 161-arm campaign tile, where per-bar predictions exist for every
arm on identical rows. If QLIKE tracks the LEVEL slope but not the ratio-space
statistics, "QLIKE rewards attenuation" is a geometry artifact and the paper's
evaluation section needs the qualification. If it tracks both, the finding is
about QLIKE and stands as written.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PRED_DIR = os.path.join(ROOT, "results", "geo_preds")
OUT_DIR = os.path.join(ROOT, "writeup", "stats")
MAIN_TILE = "6897c01e"
N_TILE = 2189


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def spearman_p(rho, n):
    """Two-sided p via the t approximation, valid for n well above 10."""
    from inference import _student_t_sf
    if abs(rho) >= 1.0:
        return 0.0
    t = rho * np.sqrt((n - 2) / (1 - rho ** 2))
    return _student_t_sf(abs(float(t)), df=n - 2)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    arms, y = {}, None
    for f in sorted(glob.glob(os.path.join(PRED_DIR, "*.npz"))):
        d = np.load(f)
        if "true_raw" not in d.files or "pred_raw" not in d.files:
            continue
        if d["pred_raw"].shape[0] != N_TILE:
            continue
        h = hashlib.md5(np.ascontiguousarray(d["true_raw"]).tobytes()).hexdigest()
        if h[:8] != MAIN_TILE:
            continue
        y = d["true_raw"]
        arms[os.path.basename(f)[:-4]] = d["pred_raw"]
        if "incumbent_pred_raw" in d.files and "incumbent" not in arms:
            arms["incumbent"] = d["incumbent_pred_raw"]
    print(f"tile: {len(arms)} arms x {N_TILE} bars")

    keep = {k: p for k, p in arms.items()
            if np.all(np.isfinite(p)) and np.all(p > 0)}
    print(f"  {len(keep)} arms with strictly positive finite forecasts")
    ok = np.isfinite(y) & (y > 0)
    print(f"  {int(ok.sum())}/{len(y)} bars with a strictly positive target "
          f"(needed for the log-space statistics)")
    yy = y[ok]
    ly = np.log(yy)

    names, rows = [], []
    for k, p in sorted(keep.items()):
        pp = p[ok]
        r = yy / pp
        lp = np.log(pp)
        rows.append({
            "arm": k,
            "qlike": float(np.mean(r - np.log(r) - 1.0)),
            "mz_level": float(np.cov(yy, pp)[0, 1] / np.var(pp)),
            "mean_ratio": float(np.mean(r)),
            "mz_log": float(np.cov(ly, lp)[0, 1] / np.var(lp)),
            "mse": float(np.mean((yy - pp) ** 2)),
        })
        names.append(k)
    n = len(rows)
    q = np.array([r["qlike"] for r in rows])

    print("\n" + "=" * 74)
    print("QLIKE AGAINST CALIBRATION, IN TWO GEOMETRIES")
    print("=" * 74)
    print(f"  {'statistic':<34}{'geometry':>10}{'rho':>9}{'p':>12}")
    res = {}
    for key, label, geom in (("mz_level", "MZ slope  Cov(y,p)/Var(p)", "level"),
                             ("mean_ratio", "mean ratio  E[y/p]", "ratio"),
                             ("mz_log", "log-log slope", "ratio")):
        v = np.array([r[key] for r in rows])
        rho = spearman(q, v)
        p_ = spearman_p(rho, n)
        print(f"  {label:<34}{geom:>10}{rho:>+9.3f}{p_:>12.3g}")
        res[key] = {"spearman": rho, "p": p_, "geometry": geom}
    print(f"\n  (n = {n} arms; the paper's battery figure is +0.741 on the "
          f"LEVEL slope\n   over its own 45 arms)")

    # how close is each geometry's calibration to its ideal?
    print(f"\n  {'arm set':<20}{'median MZ level':>18}{'median E[y/p]':>16}"
          f"{'median log slope':>18}")
    print(f"  {'all arms':<20}"
          f"{np.median([r['mz_level'] for r in rows]):>18.3f}"
          f"{np.median([r['mean_ratio'] for r in rows]):>16.3f}"
          f"{np.median([r['mz_log'] for r in rows]):>18.3f}")
    best = [rows[i] for i in np.argsort(q)[:10]]
    print(f"  {'10 best by QLIKE':<20}"
          f"{np.median([r['mz_level'] for r in best]):>18.3f}"
          f"{np.median([r['mean_ratio'] for r in best]):>16.3f}"
          f"{np.median([r['mz_log'] for r in best]):>18.3f}")
    worst = [rows[i] for i in np.argsort(q)[-10:]]
    print(f"  {'10 worst by QLIKE':<20}"
          f"{np.median([r['mz_level'] for r in worst]):>18.3f}"
          f"{np.median([r['mean_ratio'] for r in worst]):>16.3f}"
          f"{np.median([r['mz_log'] for r in worst]):>18.3f}")

    # The mean ratio is NOT an independent diagnostic: QLIKE is
    # E[r] - E[log r] - 1 with r = y/p, so E[y/p] is literally its first term
    # and any correlation between them is partly definitional. The clean
    # ratio-space statistic is the log-log slope, which shares no term with the
    # loss. The verdict rests on that one; the mean ratio is reported for
    # completeness and explicitly discounted.
    lvl = abs(res["mz_level"]["spearman"])
    clean = abs(res["mz_log"]["spearman"])
    print("\n" + "=" * 74)
    print("  NOTE: QLIKE = E[r] - E[log r] - 1, so the mean-ratio row shares a")
    print("  term with the loss and is partly definitional. The log-log slope")
    print("  is the ratio-space statistic that shares nothing with it.")
    print()
    if lvl > clean + 0.15:
        verdict = ("GEOMETRY MISMATCH: QLIKE tracks the LEVEL slope more "
                   "strongly than the\n  clean ratio-space statistic, so the "
                   "'rewards attenuation' reading is\n  substantially an "
                   "artifact of pairing a ratio-space loss with a\n  "
                   "level-space calibration measure.")
    elif clean > lvl + 0.15:
        verdict = ("QLIKE tracks the clean RATIO-space calibration more "
                   "closely than the\n  level slope -- the geometry it "
                   "actually sees.")
    else:
        verdict = ("NEITHER GEOMETRY EXPLAINS IT ON THIS TILE. The level "
                   f"slope correlates at\n  {res['mz_level']['spearman']:+.3f}"
                   f" and the clean ratio-space slope at "
                   f"{res['mz_log']['spearman']:+.3f} (p = "
                   f"{res['mz_log']['p']:.2g}),\n  both far weaker than the "
                   "+0.741 the paper reports on its 45-arm battery.\n  These "
                   "are different arm sets, so this neither reproduces nor "
                   "refutes\n  that figure -- it shows the strong "
                   "level-slope relationship is not a\n  general property of "
                   "QLIKE across arm sets, and that no calibration\n  "
                   "geometry orders this tile strongly.")
    print(verdict)
    print("=" * 74)

    out = {"n_arms": n, "n_bars": int(ok.sum()), "correlations": res,
           "per_arm": rows,
           "medians": {
               "all": {k: float(np.median([r[k] for r in rows]))
                       for k in ("mz_level", "mean_ratio", "mz_log")},
               "best10": {k: float(np.median([r[k] for r in best]))
                          for k in ("mz_level", "mean_ratio", "mz_log")},
               "worst10": {k: float(np.median([r[k] for r in worst]))
                           for k in ("mz_level", "mean_ratio", "mz_log")}},
           "verdict": verdict.replace("\n", " ")}
    with open(os.path.join(OUT_DIR, "ratio_calibration.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote ratio_calibration.json")


if __name__ == "__main__":
    main()
