#!/usr/bin/env python
"""Async Optuna tuning controller for Hero A (XGB on the enetreg2 residualized base).

WHY a hand-rolled async controller (and NOT hpc-agent campaign-advance):
  hpc-agent's `campaign advance` is a SYNC ask->submit->wait->tell ITERATION: it can
  only have one batch in flight and blocks on it. We want K trials genuinely in flight
  at once. The previous attempt (drive_campaign.py) drove the hpc-agent campaign by
  hand but never called `campaign advance`, which silently de-synced the hpc-agent
  journal (batch-status choked, ssh-flooded CARC). This controller sidesteps hpc-agent
  ENTIRELY: it owns its own Optuna study, submits each trial as one independent `sbatch`
  job, and learns the result by POLLING the per-trial JSON that `resid_amortized.py trial`
  writes. Nothing here touches the hpc-agent journal, so there is nothing to poison.

DESIGN (cluster-side controller):
  This script is meant to run ON the cluster (a tiny login-node process or a controller
  SLURM job at /scratch1/jc_905/harxhar-clean) so that `sbatch` and the results dir are
  both local to it -- no ssh polling, no rsync. Each trial:
    study.ask(widened_space("xgb"))  -> a config
    render a per-trial .sbatch with TREE_CFG=<json>
    sbatch it (one job, 4 cpus, runs the WHOLE full-OOS backtest via
      `python resid_amortized.py trial <cell> resid_subset <outname>`)
    poll results/resid_tree/<outname>.json ; when it lands, study.tell(trial, qlike)
  The loop keeps K trials in flight until --n-trials configs have been TOLD.

QOS: CARC `normal` QOS caps concurrency. We respect K*C + 1 <= 100 (C = cpus/trial,
  +1 for the controller). With C=4 that gives K <= 24. (The running-JOB cap K+1 <= 100
  is looser, so the core-budget form binds first; both are printed.)

SAFETY: --dry-run defaults to TRUE. A dry run creates an IN-MEMORY study, runs the real
  Optuna ask()/enqueue, and PRINTS the sbatch scripts + commands + QOS math + study path
  WITHOUT writing the study db, without writing any .sbatch file, and without submitting.
  Only --no-dry-run touches the cluster (and writes the persistent sqlite study).

usage:
  python async_tune.py                 # dry-run: print plan, configs, sbatch, QOS math
  python async_tune.py --k 24 --n-trials 200
  python async_tune.py --no-dry-run    # ACTUALLY submit + poll (cluster-side only)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Make `from src.backtest.tree_space import widened_space` importable regardless of cwd
# (the cluster controller cd's to the repo, but a local dry-run may not).
_REPO_LOCAL = Path(__file__).resolve().parent
if str(_REPO_LOCAL) not in sys.path:
    sys.path.insert(0, str(_REPO_LOCAL))

from src.backtest.tree_space import widened_space  # noqa: E402  (needs the sys.path insert above)


def regime_xgb_space() -> dict:
    """XGB search space for the HERO-B REGIME stage (the h16-19 close-leftover correction, a few-k
    rows/block). Shallower + more regularized than the global widened_space -- the curated sweep showed
    depth>4 overfits the small regime sample (depth6 was worst), so cap depth at 8 and favor strong
    min_child_weight / reg. Used when --global-cfg is given (regime-tune mode)."""
    import optuna.distributions as D

    return {
        "n_estimators": D.IntDistribution(50, 400, step=50),
        "max_depth": D.IntDistribution(2, 8),
        "learning_rate": D.FloatDistribution(0.005, 0.2, log=True),
        "min_child_weight": D.IntDistribution(1, 100, log=True),
        "subsample": D.FloatDistribution(0.5, 1.0),
        "colsample_bytree": D.FloatDistribution(0.3, 1.0),
        "reg_alpha": D.FloatDistribution(1e-9, 50.0, log=True),
        "reg_lambda": D.FloatDistribution(1e-9, 50.0, log=True),
        "gamma": D.FloatDistribution(0.0, 10.0),
    }


# ── Hero A constants ─────────────────────────────────────────────────────────
MODEL = "xgb"
BUCKET = "all_buckets"
BASE_KIND = "enetreg2"
# cell_id(model, bucket, twd, alpha, refit, pipe, base_kind) with base_kind != "ridge"
# -> tag = base_kind -> "xgb_all_buckets_tw1000_enetreg2_rf480_slim" (resid_amortized.cell_id).
CELL = f"{MODEL}_{BUCKET}_tw1000_{BASE_KIND}_rf480_slim"
ARM = "resid_subset"  # tree sees the block's rolling enet survivors only
DEFAULT_CID = "heroA_xgb_enetreg2"
STUDY_NAME = (
    f"resid_{MODEL}_{BUCKET}_{BASE_KIND}"  # distinct from the ridge resid_* studies
)

# ── Hero-B REGIME-tune constants (the --regime / --global-cfg mode) ───────────
# The regime tune freezes the GLOBAL XGB and Optuna-searches the regime XGB (resid_regime arm,
# REGIME_MODEL=xgb) on the h16-19 close/AH leftover -- i.e. "can a PROPERLY TUNED XGB regime beat
# the curated depth3/n400 0.12072, and close on the 0.12033 EBM ceiling?". Base = the locked linbest.
LINBEST_CELL = f"{MODEL}_{BUCKET}_tw1000_{BASE_KIND}_linbest_rf480_slim"
# Frozen Hero-A winner = retuned d8 @ colsample 0.5 (the global UNDER the 0.12033 EBM ceiling, so the
# XGB regime corrects the SAME leftover the EBM did). cs0.3 was the enetreg2-base Hero A; cs0.5 won on linbest.
HEROA_GLOBAL_CS5 = json.dumps(
    {
        "n_estimators": 200,
        "learning_rate": 0.03,
        "subsample": 0.7,
        "colsample_bytree": 0.5,
        "min_child_weight": 50.0,
        "reg_lambda": 2.0,
        "max_depth": 8,
    }
)
DEFAULT_REGIME_CID = "heroBxgb_regime_linbest"
STUDY_NAME_REGIME = f"resid_regime_{MODEL}_{BUCKET}_{BASE_KIND}_linbest"


def regime_seed_configs() -> list[dict]:
    """The 5 curated XGB-regime configs from the pre-DL dig (submit_finaldig.sh Phase 2); the
    depth3/lr0.03/n400/mcw30 one reached the best XGB-regime QLIKE 0.12072. Warm-starts the TPE from
    the known-good region of the small h16-19 sample. The fixed slots (subsample/colsample/reg/gamma)
    match the dig (reg_alpha pinned to the space floor 1e-9 = xgb's effective 0)."""
    base = {
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 1e-9,
        "reg_lambda": 2.0,
        "gamma": 0.0,
    }
    grid = [  # (max_depth, learning_rate, n_estimators, min_child_weight)
        (2, 0.05, 200, 50),
        (3, 0.05, 200, 30),
        (4, 0.05, 200, 20),
        (3, 0.03, 400, 30),  # the 0.12072 winner
        (6, 0.05, 150, 10),
    ]
    return [
        {
            **base,
            "max_depth": d,
            "learning_rate": lr,
            "n_estimators": n,
            "min_child_weight": m,
        }
        for (d, lr, n, m) in grid
    ]


def enqueue_regime_seeds(study: Any, space: dict) -> int:
    """Enqueue the curated regime configs as TPE warm-starts (only those fully covering the space)."""
    n = 0
    for cfg in regime_seed_configs():
        p = {k: cfg[k] for k in space if k in cfg}
        if len(p) == len(space):
            study.enqueue_trial(p, skip_if_exists=True)
            n += 1
    return n


# Cluster (CARC) submission constants — match the existing repo .sbatch templates.
REMOTE_REPO = "/scratch1/jc_905/harxhar-clean"
REMOTE_PY = "/home1/jc_905/.conda/envs/harxhar/bin/python"
ACCOUNT = "pollok_1603"
PARTITION = "main"
# CARC `normal` QOS real limits (sacctmgr): 100 running jobs/user, 2000 cores/user.
# pollok_1603 adds no tighter cap. Each Hero A trial is ONE job (cpus=C), so the JOB
# cap binds first (K+1<=100); the core cap (K<=2000/C=500 at C=4) is far looser.
QOS_MAX_JOBS = 100
QOS_MAX_CORES = 2000

# Default warm-start seed study (the bagging-ward XGB configs from the covid slice sweep).
DEFAULT_SEED_DB = ".hpc/campaigns/covid_xgb_all_buckets/optuna.db"
DEFAULT_SEED_STUDY = "camp_xgb_all_buckets"


# ── Optuna study + search space ──────────────────────────────────────────────
def make_study(storage: str | None, seed: int, study_name: str = STUDY_NAME) -> Any:
    """TPE + constant_liar study (async-correct: pending trials are imputed via the
    constant liar so concurrent in-flight asks don't collapse onto the same point).
    direction=minimize (QLIKE). storage=None -> in-memory (dry-run, no side effects)."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=optuna.samplers.TPESampler(
            constant_liar=True, n_ei_candidates=96, seed=seed
        ),
        direction="minimize",
        load_if_exists=True,
    )


def maybe_seed(
    study: Any, space: dict, seed_db: str, seed_study: str, seed_n: int
) -> int:
    """Enqueue the top-N lowest-QLIKE configs from an existing study (like seed_resid.py).
    enqueue (not add_trial): the objective changed (slice -> full-OOS residualized), so the
    configs are RE-EVALUATED, not trusted. Skips gracefully if the seed db is absent."""
    if seed_n <= 0 or not Path(seed_db).exists():
        return 0
    import optuna

    src = optuna.load_study(study_name=seed_study, storage=f"sqlite:///{seed_db}")
    trials = sorted(
        (t for t in src.get_trials(deepcopy=False) if t.value is not None),
        key=lambda t: t.value if t.value is not None else float("inf"),
    )
    n = 0
    for t in trials[:seed_n]:
        p = {k: t.params[k] for k in space if k in t.params}
        if len(p) == len(
            space
        ):  # only enqueue fully-specified configs (in-distribution)
            study.enqueue_trial(p, skip_if_exists=True)
            n += 1
    return n


def seed_study(study: Any, space: dict, args: argparse.Namespace) -> tuple[int, str]:
    """Dispatch warm-starting: curated regime configs in --regime mode (the covid global seed db is the
    wrong search space there); the warm-start covid-slice study in Hero A mode."""
    if args.regime:
        n = enqueue_regime_seeds(study, space)
        return (
            n,
            f"enqueued {n} curated regime warm-starts (submit_finaldig.sh Phase 2)",
        )
    n = maybe_seed(study, space, args.seed_db, args.seed_study, args.seed_n)
    note = (
        f"enqueued {n} seed configs from {args.seed_study} @ {args.seed_db}"
        if n
        else f"no seeding (seed db {args.seed_db} absent or --seed-n 0)"
    )
    return n, note


# ── sbatch rendering / submission ────────────────────────────────────────────
def render_sbatch(
    cfg: dict,
    outname: str,
    cell: str,
    arm: str,
    walltime: str,
    cpus: int,
    mem: str,
    global_cfg: str = "",
    regime_model: str = "xgb",
) -> str:
    """Render the per-trial sbatch script. The tuned config is single-quoted JSON (xgb params are
    all numeric -> JSON has no single quotes -> shell-safe). One job runs the WHOLE full-OOS backtest
    for this config and writes results/resid_tree/<outname>.json.

    Hero A mode (global_cfg=""): the tuned config is the GLOBAL XGB -> TREE_CFG (arm=resid_subset).
    Regime mode (global_cfg set): freeze GLOBAL_CFG, the tuned config is the REGIME XGB -> REGIME_CFG +
    REGIME_MODEL (arm=resid_regime). TREE_CFG is also set to the regime cfg so do_trial records it in
    the result json (score() ignores TREE_CFG for resid_regime -- it reads GLOBAL_CFG/REGIME_CFG)."""
    cfg_json = json.dumps(cfg)
    if global_cfg:  # REGIME-TUNE mode
        env = (
            f"export GLOBAL_CFG='{global_cfg}'\n"
            f"export REGIME_CFG='{cfg_json}'\n"
            f"export REGIME_MODEL={regime_model}\n"
            f"export TREE_CFG='{cfg_json}'\n"
        )
    else:  # Hero A mode (global XGB only)
        env = f"export TREE_CFG='{cfg_json}'\n"
    return f"""#!/bin/bash
#SBATCH --job-name={outname}
#SBATCH --account={ACCOUNT}
#SBATCH --partition={PARTITION}
#SBATCH --time={walltime}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --output={REMOTE_REPO}/logs/{outname}_%j.out
cd {REMOTE_REPO}
export TQDM_DISABLE=1 OMP_NUM_THREADS={cpus} MKL_NUM_THREADS={cpus}
{env}{REMOTE_PY} resid_amortized.py trial {cell} {arm} {outname} 2>&1 | grep -E 'TRIAL|Error|Traceback'
"""


def submit_sbatch(script_path: Path) -> str | None:
    """sbatch a rendered script (controller is cluster-side -> local sbatch). Returns job id."""
    p = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True)
    if p.returncode != 0:
        print(
            f"  sbatch FAILED rc={p.returncode}: {p.stderr.strip()[:300]}", flush=True
        )
        return None
    # "Submitted batch job 12345"
    out = (p.stdout or "").strip()
    return out.split()[-1] if out else None


def result_path(outname: str) -> Path:
    return Path("results/resid_tree") / f"{outname}.json"


def read_qlike(outname: str) -> float | None:
    try:
        d = json.loads(result_path(outname).read_text())
        return float(d["qlike"])
    except Exception:
        return None


# ── QOS math ─────────────────────────────────────────────────────────────────
def qos_plan(requested_k: int, cpus: int) -> dict:
    """Effective K under the REAL CARC `normal` caps: the running-JOB cap K+1<=100 binds
    first (each trial is one job); the core cap K*C<=2000 is far looser (K<=500 at C=4)."""
    max_k_jobs = QOS_MAX_JOBS - 1  # K + 1 <= 100 (controller + K trial jobs) -> BINDS
    max_k_cores = QOS_MAX_CORES // cpus  # K*C <= 2000 -> 500 at C=4 (looser)
    eff = min(requested_k, max_k_jobs, max_k_cores)
    return {
        "cpus_per_trial": cpus,
        "requested_k": requested_k,
        "max_k_cores": max_k_cores,
        "max_k_jobs": max_k_jobs,
        "effective_k": eff,
        "cores_used": eff * cpus,
    }


def print_qos(plan: dict) -> None:
    print("QOS math (CARC `normal`: 100 jobs/user, 2000 cores/user):")
    print(f"  cpus_per_trial (C)      = {plan['cpus_per_trial']}")
    print(f"  requested K             = {plan['requested_k']}")
    print(f"  running-job cap K+1<=100   -> max K = {plan['max_k_jobs']}  (BINDS)")
    print(f"  core cap       K*C<=2000  -> max K = {plan['max_k_cores']}  (looser)")
    print(
        f"  EFFECTIVE K             = {plan['effective_k']}  (uses {plan['cores_used']} cores)"
    )


# ── dry-run plan ─────────────────────────────────────────────────────────────
def do_dry_run(args: argparse.Namespace, space: dict) -> None:
    plan = qos_plan(args.k, args.cpus)
    storage_url = f"sqlite:///.hpc/campaigns/{args.cid}/optuna.db"
    print("=" * 78)
    print("DRY RUN - nothing submitted, no study db written, no .sbatch file written.")
    print("=" * 78)
    print(
        f"mode             : {'REGIME-tune (resid_regime, XGB regime stage)' if args.regime else 'Hero A (resid_subset, global XGB)'}"
    )
    print(f"cell             : {args.cell}")
    print(f"arm              : {args.arm}")
    print(f"study name       : {args.study_name}")
    print(f"study storage    : {storage_url}   (real-run path; dry-run uses in-memory)")
    print("objective        : minimize QLIKE")
    if args.regime:
        print(f"frozen GLOBAL_CFG: {args.global_cfg}")
        print(f"REGIME_MODEL     : {args.regime_model}")
    print(f"n_trials budget  : {args.n_trials}")
    print(f"poll interval    : {args.poll}s   trial timeout: {args.trial_timeout}s")
    print()
    print_qos(plan)
    print()
    space_label = "regime_xgb_space()" if args.regime else "widened_space('xgb')"
    print(f"XGB search space ({space_label}):")
    for k, dist in space.items():
        print(f"  {k:18s} {dist}")
    print()

    # Real Optuna ask() against an in-memory study (proves seeding + sampler work).
    study = make_study(storage=None, seed=args.seed, study_name=args.study_name)
    _, seed_note = seed_study(study, space, args)
    print(f"seeding          : {seed_note}")
    print()

    n_show = min(args.dry_run_samples, plan["effective_k"], args.n_trials)
    print(
        f"--- first {n_show} suggested trials (of K={plan['effective_k']} kept in flight) ---"
    )
    for _ in range(n_show):
        trial = study.ask(fixed_distributions=space)
        outname = f"{args.cid}_t{trial.number}"
        script = render_sbatch(
            trial.params,
            outname,
            args.cell,
            args.arm,
            args.walltime,
            args.cpus,
            args.mem,
            global_cfg=args.global_cfg,
            regime_model=args.regime_model,
        )
        sbatch_file = f".hpc/campaigns/{args.cid}/trials/{outname}.sbatch"
        print()
        print(f"[trial {trial.number}] outname={outname}")
        print(f"  cfg = {json.dumps(trial.params)}")
        print(f"  would write : {sbatch_file}")
        print(f"  would run   : sbatch {sbatch_file}")
        print(f"  result JSON : {result_path(outname)}")
        print("  --- rendered sbatch ---")
        for line in script.rstrip().splitlines():
            print(f"  | {line}")
    print()
    print(
        "(--no-dry-run would loop: keep K in flight, sbatch each, poll JSON, study.tell.)"
    )


# ── real async loop ──────────────────────────────────────────────────────────
def do_live(args: argparse.Namespace, space: dict) -> int:
    from optuna.trial import TrialState

    plan = qos_plan(args.k, args.cpus)
    K = plan["effective_k"]
    cdir = Path(f".hpc/campaigns/{args.cid}")
    (cdir / "trials").mkdir(parents=True, exist_ok=True)
    Path("results/resid_tree").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{cdir}/optuna.db"

    study = make_study(storage=storage, seed=args.seed, study_name=args.study_name)
    n_seeded, seed_note = seed_study(study, space, args)
    mode = "REGIME" if args.regime else "heroA"
    print(
        f"[live] mode={mode} study={args.study_name} storage={storage} seeded={n_seeded} ({seed_note}) K={K} budget={args.n_trials}",
        flush=True,
    )
    if args.regime:
        print(
            f"[live] frozen GLOBAL_CFG={args.global_cfg} REGIME_MODEL={args.regime_model} arm={args.arm} cell={args.cell}",
            flush=True,
        )
    print_qos(plan)

    in_flight: dict[str, dict] = {}  # outname -> {trial, submitted_at, job_id}
    completed = len(
        [t for t in study.get_trials(deepcopy=False) if t.state.is_finished()]
    )

    while completed < args.n_trials:
        # Fill free slots (never let asks outrun the budget).
        while len(in_flight) < K and (completed + len(in_flight)) < args.n_trials:
            trial = study.ask(fixed_distributions=space)
            outname = f"{args.cid}_t{trial.number}"
            script = render_sbatch(
                trial.params,
                outname,
                args.cell,
                args.arm,
                args.walltime,
                args.cpus,
                args.mem,
                global_cfg=args.global_cfg,
                regime_model=args.regime_model,
            )
            sbatch_file = cdir / "trials" / f"{outname}.sbatch"
            sbatch_file.write_text(script)
            job_id = submit_sbatch(sbatch_file)
            if job_id is None:  # submission failed -> fail the trial, free the slot
                study.tell(trial, state=TrialState.FAIL)
                continue
            in_flight[outname] = {
                "trial": trial,
                "submitted_at": time.time(),
                "job_id": job_id,
            }
            print(
                f"[submit] {outname} job={job_id} cfg={json.dumps(trial.params)}",
                flush=True,
            )

        # Poll for landed results / time-outs.
        for outname, info in list(in_flight.items()):
            if result_path(outname).exists():
                q = read_qlike(outname)
                if q is None:
                    study.tell(info["trial"], state=TrialState.FAIL)
                    print(f"[fail ] {outname} result unreadable -> FAIL", flush=True)
                else:
                    study.tell(info["trial"], q)
                    print(f"[tell ] {outname} qlike={q:.5f}", flush=True)
                completed += 1
                del in_flight[outname]
            elif time.time() - info["submitted_at"] > args.trial_timeout:
                study.tell(info["trial"], state=TrialState.FAIL)
                print(
                    f"[stuck] {outname} job={info['job_id']} > {args.trial_timeout}s -> FAIL",
                    flush=True,
                )
                completed += 1
                del in_flight[outname]

        if completed >= args.n_trials:
            break
        time.sleep(args.poll)

    best = study.best_trial
    print(
        f"[done ] completed={completed} best_qlike={best.value:.6f} best={json.dumps(best.params)}",
        flush=True,
    )
    return 0


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # cid/cell/arm default to None -> resolved per mode in main() (Hero A vs --regime).
    ap.add_argument(
        "--cid",
        default=None,
        help="campaign id (study lives under .hpc/campaigns/<cid>/)",
    )
    ap.add_argument("--cell", default=None, help="resid_amortized cell id")
    ap.add_argument("--arm", default=None, help="tree column arm")
    # ── REGIME-tune mode (Hero B XGB regime stage) ──
    ap.add_argument(
        "--regime",
        action="store_true",
        help="tune the XGB REGIME stage (resid_regime, REGIME_MODEL=xgb) "
        "on the linbest base with the global frozen, instead of the Hero A global XGB",
    )
    ap.add_argument(
        "--global-cfg",
        default="",
        help="JSON of the FROZEN global XGB for --regime mode "
        "(default = the d8@cs0.5 retuned Hero-A winner); presence also implies --regime",
    )
    ap.add_argument(
        "--regime-model",
        default="xgb",
        help="regime-stage learner in --regime mode (xgb|ebm)",
    )
    ap.add_argument(
        "--k",
        type=int,
        default=24,
        help="trials to keep in flight (clamped to QOS cap)",
    )
    ap.add_argument(
        "--cpus", type=int, default=4, help="cpus per trial (C in the K*C+1<=100 cap)"
    )
    ap.add_argument(
        "--n-trials", type=int, default=200, help="total trials to TELL before stopping"
    )
    ap.add_argument("--walltime", default="5:00:00", help="per-trial sbatch --time")
    ap.add_argument("--mem", default="44G", help="per-trial sbatch --mem")
    ap.add_argument(
        "--poll", type=int, default=120, help="seconds between result polls"
    )
    ap.add_argument(
        "--trial-timeout",
        type=int,
        default=16200,
        help="seconds before an in-flight trial is FAILed",
    )
    ap.add_argument("--seed", type=int, default=0, help="TPE sampler seed")
    ap.add_argument(
        "--seed-db",
        default=DEFAULT_SEED_DB,
        help="seed study sqlite path (warm-start, Hero A only)",
    )
    ap.add_argument(
        "--seed-study", default=DEFAULT_SEED_STUDY, help="seed study name (Hero A only)"
    )
    ap.add_argument(
        "--seed-n",
        type=int,
        default=20,
        help="top-N seed configs to enqueue (Hero A only; 0 = none)",
    )
    ap.add_argument(
        "--dry-run-samples", type=int, default=3, help="trials to print in --dry-run"
    )
    # --dry-run defaults TRUE; --no-dry-run is required to actually submit.
    ap.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=True,
        help="(default) plan only",
    )
    ap.add_argument(
        "--no-dry-run",
        dest="dry_run",
        action="store_false",
        help="actually submit + poll",
    )
    return ap.parse_args(argv)


def resolve_mode(args: argparse.Namespace) -> dict:
    """Resolve Hero A vs REGIME mode: fill cell/cid/arm/study_name/global_cfg defaults and pick the
    search space. --global-cfg implies --regime. Returns the space; mutates args in place."""
    args.regime = bool(args.regime or args.global_cfg)
    if args.regime:
        args.cell = args.cell or LINBEST_CELL
        args.cid = args.cid or DEFAULT_REGIME_CID
        args.arm = args.arm or "resid_regime"
        args.global_cfg = args.global_cfg or HEROA_GLOBAL_CS5
        args.study_name = STUDY_NAME_REGIME
        return regime_xgb_space()
    args.cell = args.cell or CELL
    args.cid = args.cid or DEFAULT_CID
    args.arm = args.arm or ARM
    args.global_cfg = ""
    args.regime_model = "xgb"
    args.study_name = STUDY_NAME
    return widened_space(MODEL)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    space = resolve_mode(args)
    if args.dry_run:
        do_dry_run(args, space)
        return 0
    return do_live(args, space)


if __name__ == "__main__":
    raise SystemExit(main())
