"""Cluster config-sweep worker for the prop-eval control policies -- tune + stress-test the deployable,
literature-assembled solution for OUR measured regime (heavy-tailed / positive-jump empirical returns +
trailing drawdown + deadline):

  base leverage  = Risk-Constrained Kelly (Busseti-Ryu-Boyd 2016) on the EMPIRICAL returns -- convex,
                   distribution-free (eats the jumps), provable drawdown bound, beats fractional Kelly;
  goal-reaching  = Yuan-Li/Browne two-region shape (Merton-far / bang-bang leverage-cap bold near the
                   barrier), literature-optimal AND preserved under positive jumps (our skew +8.2).

One array task = one config. Each evaluates its policy on a COMMON block-bootstrap path set (deterministic
seed -> common random numbers across configs -> paired, low-variance DM/MCS -- the requested seed-cache),
and writes a per-path pass-indicator .npz (for DM/MCS via src.evaluation) + a summary json (pass-rate +
Wilson CI). Mirrors :mod:`eval_sim.hpc_worker`; the modelling lives in :mod:`eval_sim.adp`.

Policy families:
  anchor      -- constant RCK leverage on the empirical returns (the baseline to beat)
  structured  -- StructuredPolicy with the Merton leg = RCK(beta) leverage; sweep (rck_beta, cap, thr).
                 The vol-conditioning arms were dropped: measured phi(vol)~=0 (no persistent vol state),
                 so both global-inverse-vol and conditional-bold lost -- there is nothing to condition on.
"""

from __future__ import annotations

import numpy as np

from eval_sim.adp import PASS, StructuredPolicy, rck_leverage, simulate_geo
from eval_sim.config import EvalConfig
from eval_sim.pnl import BlockBootstrap

DATA = "eval_sim/data/cumrv_close_pnl.npy"


# --------------------------------------------------------------------------------------------------- #
# Contract variants + the config grid.
# --------------------------------------------------------------------------------------------------- #
def _contracts() -> list[dict]:
    """Base contract + robustness variants (friction level and DD:target ratio). ``cost_sigma`` is the
    stated per-step friction in units of the raw PnL sigma; the rest map onto EvalConfig."""
    base = dict(cost_sigma=0.30, profit_target=3000.0, max_drawdown=2500.0, block=20)
    return [
        {**base, "tag": "base"},
        {**base, "cost_sigma": 0.20, "tag": "lowcost"},
        {**base, "cost_sigma": 0.40, "tag": "hicost"},
        {**base, "max_drawdown": 2000.0, "tag": "tightDD"},  # DD:target 0.67 (harder)
        {**base, "profit_target": 4000.0, "tag": "hitarget"},
    ]


def sweep_configs() -> list[dict]:
    """The full (policy x params x contract) coordinate list. array_id indexes this."""
    rck_betas = (
        0.10,
        0.20,
        0.35,
        0.50,
    )  # RCK risk budget -> the base (Merton) leverage on the returns
    caps = (
        2.0,
        2.5,
        3.0,
        4.0,
        5.0,
    )  # extended up: the positive-jump regime rewards leverage (railed @3)
    thrs = (0.05, 0.10, 0.20)
    cfgs: list[dict] = []
    for c in _contracts():
        cfgs.append({**c, "family": "anchor"})
        for rb in rck_betas:
            for cap in caps:
                for thr in thrs:
                    cfgs.append(
                        {
                            **c,
                            "family": "structured",
                            "rck_beta": rb,
                            "cap": cap,
                            "thr": thr,
                        }
                    )
    return cfgs


def coord_for(array_id: int) -> dict:
    cfgs = sweep_configs()
    if not 0 <= array_id < len(cfgs):
        raise SystemExit(f"array_id {array_id} out of range [0, {len(cfgs)})")
    return cfgs[array_id]


# --------------------------------------------------------------------------------------------------- #
# Evaluation. The base leverage comes from Risk-Constrained Kelly on the empirical returns.
# --------------------------------------------------------------------------------------------------- #
def _eval_cfg(coord: dict) -> EvalConfig:
    return EvalConfig(
        profit_target=coord["profit_target"], max_drawdown=coord["max_drawdown"]
    )


def _returns(coord: dict) -> np.ndarray:
    pnl = np.load(DATA)
    return (pnl - coord["cost_sigma"] * pnl.std()) / EvalConfig().initial


def evaluate(
    coord: dict, n_paths: int = 60_000, seed: int = 20260701
) -> tuple[np.ndarray, dict]:
    """Evaluate one config on a COMMON block-bootstrap path set (fixed seed => CRN across configs).
    Returns (per-path pass indicator, summary dict)."""
    cfg = _eval_cfg(coord)
    ret = _returns(coord)
    alpha = (
        1.0 - cfg.max_drawdown / cfg.initial
    )  # DD as a fraction of initial -> the RCK min-wealth level
    sampler = BlockBootstrap(ret, block=coord["block"])
    R = sampler.sample(
        n_paths, 250, np.random.default_rng(seed)
    )  # SAME seed across configs -> paired

    fam = coord["family"]
    if fam == "anchor":
        w0, _ = rck_leverage(
            ret, alpha, 0.35
        )  # RCK constant base at a reference risk budget
        outcome = simulate_geo(R, cfg, float(w0), T=250)[0]
        coord = {**coord, "merton": w0}
    elif fam == "structured":
        merton, _ = rck_leverage(
            ret, alpha, coord["rck_beta"]
        )  # RCK base leverage on empirical returns
        pol = StructuredPolicy(float(merton), coord["cap"], coord["thr"], cfg)
        outcome = simulate_geo(R, cfg, pol, T=250)[0]
        coord = {**coord, "merton": merton}
    else:
        raise SystemExit(f"unknown family {fam!r}")

    passed = (outcome == PASS).astype(np.int8)
    pr = float(passed.mean())
    se = np.sqrt(pr * (1 - pr) / n_paths)
    z = 1.96
    denom = 1 + z * z / n_paths
    centre = (pr + z * z / (2 * n_paths)) / denom
    half = (
        z * np.sqrt(pr * (1 - pr) / n_paths + z * z / (4 * n_paths * n_paths)) / denom
    )
    summary = {
        **{
            k: coord.get(k)
            for k in ("tag", "family", "rck_beta", "merton", "cap", "thr")
        },
        "pass_rate": pr,
        "se": se,
        "wilson_lo": centre - half,
        "wilson_hi": centre + half,
        "n_paths": n_paths,
    }
    return passed, summary


# --------------------------------------------------------------------------------------------------- #
# Array-job entry point (Slurm/SGE), same convention as eval_sim.hpc_worker.
# --------------------------------------------------------------------------------------------------- #
def _resolve_array_id(cli: int | None) -> int:
    import os

    if cli is not None:
        return cli
    for env, base in (("SLURM_ARRAY_TASK_ID", 0), ("SGE_TASK_ID", 1)):
        v = os.environ.get(env)
        if v is not None:
            return int(v) - base
    raise SystemExit(
        "no --array-id and neither SLURM_ARRAY_TASK_ID nor SGE_TASK_ID set"
    )


def main(argv: list[str] | None = None) -> None:
    """Resolve one array task to a config, evaluate on the common CRN paths, and persist per-path pass
    indicators (.npz, for DM/MCS) + a summary json."""
    import argparse
    import json
    import os

    p = argparse.ArgumentParser(
        description="eval_sim ADP policy-sweep worker (one config per task)"
    )
    p.add_argument("--array-id", type=int, default=None)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-paths", type=int, default=60_000)
    p.add_argument(
        "--seed", type=int, default=20260701
    )  # shared -> CRN across all configs
    args = p.parse_args(argv)

    array_id = _resolve_array_id(args.array_id)
    os.makedirs(args.out_dir, exist_ok=True)
    coord = coord_for(array_id)
    passed, summary = evaluate(coord, n_paths=args.n_paths, seed=args.seed)
    summary["coord_id"] = array_id

    np.savez_compressed(
        os.path.join(args.out_dir, f"cell_{array_id:04d}_pass.npz"), passed=passed
    )
    with open(
        os.path.join(args.out_dir, f"cell_{array_id:04d}_summary.json"), "w"
    ) as fh:
        json.dump(summary, fh, indent=2)
    print(
        f"coord_id={array_id} {summary['tag']}/{summary['family']} "
        f"merton={summary.get('merton')} cap={summary.get('cap')} thr={summary.get('thr')} "
        f"pass_rate={summary['pass_rate']:.4f} "
        f"wilson=[{summary['wilson_lo']:.4f}, {summary['wilson_hi']:.4f}]"
    )


if __name__ == "__main__":
    main()
