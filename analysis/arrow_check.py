"""§38.4: do the NEW arrows replicate? — robustness check on the §38 width result.

The q = 40 production move rests on arrows involving directions 21–40, which are individually
gauge-degenerate (§27 C5). The study's signature discipline is split-half replication of every
load-bearing object (map +0.62, daily flow +0.79, intraday flow +0.992) — the D_40 new block
has not had that check. Two probes, no walks:

(i)  split-half replication: D estimated on the first vs second half of the span; entrywise
     correlation, reported separately for the old block (both indices < 20) and the
     new-involving block (either index >= 20).
(ii) refit stability: the quarterly D series' per-entry sign agreement with the full-sample D,
     old vs new distributions.

Recorded expectation: new-block split-half >= +0.5 (the 20-frame flow replicated at +0.99;
some decay is expected in fainter directions). A failure (< +0.3) would say the q = 40 gain
rides NONSTATIONARY arrows — not fatal (the walk gain is real and out-of-sample) but it would
demand per-era attribution before the width claim enters the paper.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.alpha_panel import load_panel  # noqa: E402
from analysis.pool_width import _frame_q  # noqa: E402
from src.features.transforms.target import PERIODS_PER_DAY  # noqa: E402


def _D(G):
    a, b = G[:-1], G[1:]
    az = (a - a.mean(0)) / (a.std(0) + 1e-12)
    bz = (b - b.mean(0)) / (b.std(0) + 1e-12)
    C = (az.T @ bz) / len(az)
    return (C - C.T) / 2.0


def main() -> None:
    p = load_panel()
    G = _frame_q(p, 40)
    ng = len(G)
    iu, ju = np.triu_indices(40, k=1)
    old = (iu < 20) & (ju < 20)
    new = ~old

    D1 = _D(G[: ng // 2])[iu, ju]
    D2 = _D(G[ng // 2 :])[iu, ju]
    for lbl, m in (("old block (190 arrows)", old), ("new-involving (590 arrows)", new)):
        c = np.corrcoef(D1[m], D2[m])[0, 1]
        print(f"  split-half {lbl:28s} corr {c:+.3f}", flush=True)

    Dfull = _D(G)[iu, ju]
    TRAIL, STEP = 504 * PERIODS_PER_DAY, 63 * PERIODS_PER_DAY
    agree = np.zeros(len(iu))
    cnt = 0
    for start in range(TRAIL, ng, STEP):
        Dq = _D(G[start - TRAIL : start])[iu, ju]
        agree += np.sign(Dq) == np.sign(Dfull)
        cnt += 1
    agree /= cnt
    for lbl, m in (("old", old), ("new", new)):
        print(f"  quarterly sign agreement {lbl:4s}: mean {agree[m].mean():.3f}  "
              f"median {np.median(agree[m]):.3f}  frac>0.8 {np.mean(agree[m] > 0.8):.3f}",
              flush=True)
    # magnitude context: are new arrows systematically fainter?
    print(f"  |D| mean old {np.abs(Dfull[old]).mean():.4f}  new {np.abs(Dfull[new]).mean():.4f}",
          flush=True)


if __name__ == "__main__":
    main()
