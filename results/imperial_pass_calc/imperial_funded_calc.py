"""Funded-stage first-payout probability for Imperial Trader Funding.

Funded rules (site FAQ): trailing drawdown trails up and LOCKS at starting balance + $100
("The trailing drawdown moves up with your profits until it reaches your starting balance
plus $100, after which it no longer moves"). Min payout balance = profit target equivalent;
min withdrawal $500. So first payout = reach initial + target before hitting the floor,
with lock_level = initial + 100 (engine's Apex-style lock).
"""

from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\james\CC Allowed\harxhar-clean")

from eval_sim.config import DrawdownType, EvalConfig, SimConfig  # noqa: E402
from eval_sim.metrics import summarize  # noqa: E402
from eval_sim.pnl import BlockBootstrap  # noqa: E402
from eval_sim.sim_core import run  # noqa: E402

LOG = np.load(
    r"C:\Users\james\CC Allowed\harxhar-clean\eval_sim\data\cumrv_close_pnl_honest.npy"
)

# (family, size, payout target, dd)
FUNDED = [
    ("starter", 25_000, 1250, 1000),
    ("starter", 50_000, 3000, 2000),
    ("starter", 100_000, 6000, 3000),
    ("elite", 25_000, 2500, 1750),
    ("elite", 50_000, 5500, 3500),
    ("elite", 100_000, 11_000, 7000),
    ("foundation", 25_000, 875, 1000),
    ("foundation", 50_000, 1750, 2000),
    ("foundation", 100_000, 3500, 4000),
]

rows = []
for fam, size, tgt, dd in FUNDED:
    for mult in [0.5, 1.0, 2.0, 3.0]:
        ec = EvalConfig(
            initial=float(size),
            profit_target=float(tgt),
            max_drawdown=float(dd),
            dd_type=DrawdownType.TRAILING_INTRADAY,
            lock_level=float(size) + 100.0,
            min_days=0,
            max_days=None,
        )
        sc = SimConfig(n_paths=20_000, max_steps=1500, seed=7, size_mult=mult)
        s = summarize(run(BlockBootstrap(LOG, block=5), ec, sc))
        rows.append(
            {
                "family": fam,
                "size": size,
                "mult": mult,
                "p_first_payout": s["pass_rate"],
                "timeout": s["timeout_rate"],
                "median_days": s["median_steps_pass"],
            }
        )
        print(
            f"{fam} {size // 1000}k m={mult}: P(payout)={s['pass_rate']:.3f} "
            f"days~{s['median_steps_pass']} timeout={s['timeout_rate']:.3f}",
            flush=True,
        )

out = rf"C:\Users\james\AppData\Local\Temp\claude\C--Users-james-CC-Allowed-harxhar-clean\7f6148d0-53ba-4bf9-88c8-5f76b70def6f\scratchpad\imperial_funded.json"
with open(out, "w") as f:
    json.dump(rows, f, indent=1)
print("WROTE", out, flush=True)
