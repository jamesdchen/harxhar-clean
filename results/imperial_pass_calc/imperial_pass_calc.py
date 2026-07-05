"""Imperial Trader Funding pass-rate calculation.

Maps Imperial's plan geometries (extracted from their site bundle 2026-07-04) onto the
verified eval_sim engine (harxhar eval_sim package, commit 0ddd683 lineage) and runs the
HONEST cumrv x close edge (Sharpe 0.271/day, skew +12.7, $400/day std scale) through each
barrier, sweeping the position-sizing multiplier (CRN grid).

Rule mapping (best-guess from site FAQ text; dd-type sensitivity bracketed separately):
  starter    (1-step): live trailing (intraday) dd, DLL = 50% of dd, min 3 days, 33% consistency
  elite      (1-step): EOD trailing dd, no DLL, no min days, 50% consistency
  foundation (2-step): STATIC dd, no DLL, no min days, 50% consistency (phase targets t1, t2)
  instant    (funded): live trailing dd, 3 min days, "target" = 10% withdrawal cap

Consistency is modeled as a soft cap: the violating trade is removed -> cap the log's
positive days at frac*target/size_mult BEFORE the engine multiplies by size_mult.
No max_days at this firm ("a few days or a few months") -> horizon 1500 days, timeout reported.
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

SIZE_MULTS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
N_PATHS = 20_000
MAX_STEPS = 1500

# (size, price, reset, targets(list: 1 or 2 phases), dd)
FAMILIES: dict[str, dict] = {
    "starter": {  # GG array
        "tiers": [
            (25_000, 94, 94, [1250], 1000),
            (50_000, 128, 128, [3000], 2000),
            (100_000, 197, 197, [6000], 3000),
            (250_000, 434, 434, [25_000], 10_000),
        ],
        "dd_type": DrawdownType.TRAILING_INTRADAY,
        "dll_frac_of_dd": 0.5,
        "min_days": 3,
        "consist": 0.33,
    },
    "elite": {  # VG array
        "tiers": [
            (25_000, 194, 200, [2500], 1750),
            (50_000, 357, 350, [5500], 3500),
            (100_000, 686, 650, [11_000], 7000),
            (250_000, 1067, 1000, [25_000], 13_500),
            (1_000_000, 2999, 3000, [150_000], 50_000),
        ],
        "dd_type": DrawdownType.TRAILING_EOD,
        "dll_frac_of_dd": None,
        "min_days": 0,
        "consist": 0.50,
    },
    "foundation": {  # XG array (2-step static)
        "tiers": [
            (25_000, 70, 70, [1250, 875], 1000),
            (50_000, 91, 91, [2500, 1750], 2000),
            (75_000, 124, 124, [3750, 2625], 3000),
            (100_000, 158, 158, [5000, 3500], 4000),
            (125_000, 200, 200, [6250, 4375], 5000),
        ],
        "dd_type": DrawdownType.STATIC,
        "dll_frac_of_dd": None,
        "min_days": 0,
        "consist": 0.50,
    },
    "instant": {  # qG array; target = 10% withdrawal cap
        "tiers": [
            (25_000, 342, None, [2500], 1250),
            (50_000, 486, None, [5000], 2500),
            (100_000, 810, None, [10_000], 5000),
            (500_000, 3841, None, [50_000], 17_000),
        ],
        "dd_type": DrawdownType.TRAILING_INTRADAY,
        "dll_frac_of_dd": None,
        "min_days": 3,
        "consist": None,
    },
}


def one_phase(target, dd, fam_cfg, size, mult, cap_frac, seed):
    """P(pass) etc for one phase at one size mult. cap_frac: consistency soft-cap or None."""
    log = LOG
    if cap_frac is not None:
        log = np.minimum(
            LOG, cap_frac * target / mult
        )  # engine multiplies by mult after
    dll = fam_cfg["dll_frac_of_dd"]
    ec = EvalConfig(
        initial=float(size),
        profit_target=float(target),
        max_drawdown=float(dd),
        dd_type=fam_cfg["dd_type"],
        daily_loss_limit=(dll * dd if dll else None),
        min_days=fam_cfg["min_days"],
        max_days=None,
        steps_per_day=1,
    )
    sc = SimConfig(n_paths=N_PATHS, max_steps=MAX_STEPS, seed=seed, size_mult=mult)
    return summarize(run(BlockBootstrap(log, block=5), ec, sc))


def main(family: str):
    fam = FAMILIES[family]
    rows = []
    for size, price, reset, targets, dd in fam["tiers"]:
        for mult in SIZE_MULTS:
            for variant, cap in (("raw", None), ("consist", fam["consist"])):
                if cap is None and variant == "consist":
                    continue
                p_all, days_sum = 1.0, 0.0
                phase_stats = []
                for i, tgt in enumerate(targets):
                    s = one_phase(tgt, dd, fam, size, mult, cap, seed=1000 + i)
                    p_all *= s["pass_rate"]
                    days_sum += s.get("median_steps_pass") or 0
                    phase_stats.append(
                        {
                            k: s[k]
                            for k in ("pass_rate", "timeout_rate", "median_steps_pass")
                        }
                    )
                rows.append(
                    {
                        "family": family,
                        "size": size,
                        "price": price,
                        "reset": reset,
                        "mult": mult,
                        "daily_std": 400 * mult,
                        "variant": variant,
                        "p_pass": p_all,
                        "median_days": days_sum,
                        "cost_per_funded": (price / p_all) if p_all > 0 else None,
                        "phases": phase_stats,
                    }
                )
                print(
                    f"{family} {size // 1000}k m={mult} {variant}: "
                    f"P={p_all:.3f} days~{days_sum:.0f} $per-funded="
                    f"{(price / p_all) if p_all > 0 else float('inf'):.0f}",
                    flush=True,
                )
    out = rf"C:\Users\james\AppData\Local\Temp\claude\C--Users-james-CC-Allowed-harxhar-clean\7f6148d0-53ba-4bf9-88c8-5f76b70def6f\scratchpad\imperial_{family}.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=1)
    print("WROTE", out, flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
