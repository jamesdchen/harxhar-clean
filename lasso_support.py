"""What does L1 kill, and when? -- support analysis of the oracle lasso vs the incumbent enet.

Fits the HAR-OLS base with (a) lasso l1=1.0 a=1.5e-4 (the Optuna oracle winner) and (b) the incumbent
enet l1=0.2 a=1e-3, across the walk-forward refit blocks, then inspects the SUPPORT (nonzero coefs) of
the penalized exog block (HAR is OLS -> never killed):
  * WHAT  -- per feature-family: how many cols kept (nonzero in ANY block) vs always-zero (killed);
  * WHEN  -- nnz per block over time (does the support grow/shrink/shift?), and always-on vs flickering.
Diagnostic only -- the oracle lasso win was -0.00015 (sub-noise, test-set-overfit); this asks whether its
selection is stable/interpretable or a regime-dependent (fragile) thing.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.getcwd())
from resid_amortized import PERIODS_PER_DAY, _cadence_enet_har_unpen  # noqa: E402

B = "results/covid_imp_rank/all_buckets"
HAR = ("har_ma_1", "har_ma_5", "har_ma_25", "har_ma_125", "har_ma_625", "har_ma_3125")
TW = 1000 * PERIODS_PER_DAY
REFIT = 20_000
ITER, TOL = 2000, 1e-3


def family(f):
    if f in set(HAR):
        return "HAR(OLS)"
    fl = f.lower()
    if "avail" in fl or "_ind" in fl or fl.endswith("ind"):
        return "availability_ind"
    if "hour" in fl or "dow" in fl or fl.startswith("day") or "_day" in fl:
        return "calendar"
    if "stocktwits" in fl:
        return "sentiment"
    if "voldemand" in fl:
        return "vol_demand"
    if "vvix" in fl or "vix" in fl:
        return "implied_vol"
    if any(x in fl for x in ("turnover", "spread", "volume", "numobs")):
        return "liquidity"
    if "ewstock" in fl:
        return "market_ew"
    if "vwstock" in fl:
        return "market_vw"
    if (
        fl.startswith(("sum", "adj_sum"))
        or "ret" in fl
        or "bipow" in fl
        or "pret" in fl
    ):
        return "moments"
    return "other"


def analyze(name, coefs, feats, em_idx):
    supp = coefs[:, em_idx] != 0.0  # (nb, n_exog) bool support
    nnz_block = supp.sum(axis=1)
    ever = supp.any(axis=0)  # kept in >=1 block
    always = supp.all(axis=0)  # kept in ALL blocks
    never = ~ever  # killed everywhere
    fam = [family(feats[j]) for j in em_idx]
    tot_by_fam = defaultdict(int)
    ever_by_fam = defaultdict(int)
    always_by_fam = defaultdict(int)
    for k, fm in enumerate(fam):
        tot_by_fam[fm] += 1
        ever_by_fam[fm] += int(ever[k])
        always_by_fam[fm] += int(always[k])
    print(f"\n===== {name} =====", flush=True)
    print(
        f"exog cols={len(em_idx)}  nnz/block: min={nnz_block.min()} max={nnz_block.max()} "
        f"mean={nnz_block.mean():.0f}  |  ever-kept={int(ever.sum())} always-kept={int(always.sum())} "
        f"never (killed)={int(never.sum())}",
        flush=True,
    )
    print(
        "  WHEN (nnz per block, chronological):", list(map(int, nnz_block)), flush=True
    )
    print("  WHAT (family: total | ever-kept | always-kept | killed):", flush=True)
    for fm in sorted(tot_by_fam, key=lambda k: -tot_by_fam[k]):
        t, e, a = tot_by_fam[fm], ever_by_fam[fm], always_by_fam[fm]
        print(
            f"    {fm:18s} {t:4d} | ever {e:4d} | always {a:4d} | killed {t - e:4d}",
            flush=True,
        )
    # flicker = kept sometimes but not always
    flicker = ever & ~always
    print(
        f"  FLICKER (regime-dependent: kept in some blocks, dropped in others): {int(flicker.sum())} cols",
        flush=True,
    )
    return {"support": supp, "ever": ever, "always": always, "em_idx": em_idx}


def main():
    feats = json.load(open(f"{B}/meta.json"))["feats"]
    hm = np.array([f in set(HAR) for f in feats])
    em_idx = np.where(~hm)[0]
    X = np.ascontiguousarray(np.load(f"{B}/X_imp.npy"))
    y = np.load(f"{B}/y.npy").astype(np.float64)
    print(
        f"loaded X{X.shape}  {len(em_idx)} penalized exog cols  tw={TW} refit={REFIT}",
        flush=True,
    )

    out = {}
    for name, a, l1, pen in [
        ("lasso a=1.5e-4 (oracle)", 1.49e-4, 1.0, "lasso"),
        ("enet a=1e-3 l1=0.2 (incumbent)", 1e-3, 0.2, "enet"),
    ]:
        t0 = time.time()
        _s, coefs, _ic = _cadence_enet_har_unpen(
            X,
            y,
            TW,
            REFIT,
            list(hm),
            alpha=a,
            l1=l1,
            penalty=pen,
            max_iter=ITER,
            tol=TOL,
        )
        print(
            f"\nfit {name}: {len(coefs)} blocks ({time.time() - t0:.0f}s)", flush=True
        )
        out[name] = analyze(name, coefs, feats, em_idx)

    # what lasso kills that enet keeps (and vice versa), by family
    la = out["lasso a=1.5e-4 (oracle)"]["ever"]
    en = out["enet a=1e-3 l1=0.2 (incumbent)"]["ever"]
    kills_extra = (~la) & en  # enet keeps, lasso kills entirely
    keeps_extra = la & (~en)  # lasso keeps, enet kills entirely
    fam = [family(feats[j]) for j in em_idx]
    kx, kp = defaultdict(int), defaultdict(int)
    for k, fm in enumerate(fam):
        if kills_extra[k]:
            kx[fm] += 1
        if keeps_extra[k]:
            kp[fm] += 1
    print("\n===== lasso vs enet (ever-kept difference) =====", flush=True)
    print(
        f"  lasso kills entirely but enet keeps: {int(kills_extra.sum())} cols -> {dict(kx)}",
        flush=True,
    )
    print(
        f"  lasso keeps but enet kills entirely: {int(keeps_extra.sum())} cols -> {dict(kp)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
