"""Campaign strategy: Optuna TPE tree-hyperparameter tuning on the COVID slice.

Path-B closed loop. Each campaign iteration ASKS the persistent Optuna study
(``_load_or_create_study``, TPE + constant_liar) for K trials; each trial becomes a
``run.py`` config that runs ``src.models.<model>.run`` over the FIXED COVID emit
window via the *validated* ``run_executor`` (start/end/halo) — so a trial's QLIKE is
produced by the exact audited pipeline, not a bespoke scorer. The per-iteration
reducer (``aggregate``) calls ``tune_tree.score_trials`` to TELL the study from the
finished trials' metrics, so the next iteration's ASK is informed.

``_optuna_trial_number`` is attached to every task — the load-bearing stochastic
marker that keeps cmd_sha unique (else submit-flow dedupes identical configs and the
campaign silently collapses; validate-campaign's missing_stochastic_marker guards it).

Campaign knobs come from HPC_KW_* (set via the campaign manifest's frozen kwargs):
  MODEL (lgbm|xgb)  BUCKET  K  MAX_TRIALS  N_EST_CAP  TRAIN_WIN_DAYS
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# COVID emit window on the CARC tree matrix (calendar-anchored, 2020-02-25..04-30).
EMIT_START, EMIT_END = 189713, 192023
PERIODS_PER_DAY = 48

_EXOG = {
    "sentiment": "stocktwits_attention|stocktwits_sentiment|stocktwits_sentcount",
    "implied_vol": "vix|vvix|vix3m",
    "vol_demand": "voldemand_spx_open_and_close|voldemand_spx_open_only|voldemand_all_open_and_close|voldemand_all_open_only",
    "all_buckets": (
        "sumret|sumabsret|sumret3|sumret4|sumpret2|sumbipow|sumautocov|sumvolume|numobs|"
        "sumret2_ewstock|sumret3_ewstock|sumret4_ewstock|sumabsret_ewstock|sumbipow_ewstock|sumpret2_ewstock|"
        "turnover_ewstock|buyturnover_ewstock|sellturnover_ewstock|effspread_ewstock|spread_ewstock|"
        "sumret2_vwstock|sumret3_vwstock|sumret4_vwstock|sumabsret_vwstock|sumbipow_vwstock|sumpret2_vwstock|"
        "turnover_vwstock|buyturnover_vwstock|sellturnover_vwstock|effspread_vwstock|spread_vwstock|"
        "stocktwits_attention|stocktwits_sentiment|stocktwits_sentcount|vix|vvix|vix3m|"
        "voldemand_spx_open_and_close|voldemand_spx_open_only|voldemand_all_open_and_close|voldemand_all_open_only"
    ),
}
_MODEL_MODULE = {"lgbm": "lightgbm", "xgb": "xgboost"}

MODEL = os.environ.get("HPC_KW_MODEL", "lgbm")
BUCKET = os.environ.get("HPC_KW_BUCKET", "all_buckets")
K = int(os.environ.get("HPC_KW_K", "8"))
MAX_TRIALS = int(os.environ.get("HPC_KW_MAX_TRIALS", "200"))
N_EST_CAP = int(os.environ.get("HPC_KW_N_EST_CAP", "300"))
TRAIN_WIN_DAYS = int(os.environ.get("HPC_KW_TRAIN_WIN_DAYS", "250"))
CID = os.environ.get("HPC_CAMPAIGN_ID", "covid_tune")

# Residualized objective (regular-TPE tuning of a tree on the Ridge residual, full-OOS QLIKE
# on the slim per-slot-rank matrix via the amortized cache). OBJECTIVE="slice" keeps the
# original COVID-slice fit-once tuning unchanged.
OBJECTIVE = os.environ.get("HPC_KW_OBJECTIVE", "slice")  # "slice" | "residualized"
ALPHA = float(os.environ.get("HPC_KW_ALPHA", "1.0"))
TREE_REFIT = int(os.environ.get("HPC_KW_TREE_REFIT", "480"))
PIPE = os.environ.get(
    "HPC_KW_PIPE", "slim"
)  # MATRIX pipe (the cache cell): slim | std | pca<K>
BASE = os.environ.get(
    "HPC_KW_BASE", "ridge"
)  # residualizer linear base: "ridge" | "enet" (full stack)
ARM = os.environ.get(
    "HPC_KW_ARM", "residualized"
)  # tree column arm: residualized | resid_subset | resid_pruned
CHUNKS = int(
    os.environ.get("HPC_KW_CHUNKS", "90")
)  # time-chunks per trial (cadence-block partition)
SEED_BASE = (
    1000  # base for the per-ask sampler seed (see _build_tasks_resid re-seed note)
)
# Per-campaign seed offset: independent cluster studies (carc/hoffman2) otherwise run
# the IDENTICAL deterministic sequence (same SEED_BASE + same progress) and produce
# identical QLIKEs — the second cluster is then redundant. Give each campaign a disjoint
# seed block so the clusters explore COMPLEMENTARY points. carc stays at offset 0 so its
# in-flight trajectory is unchanged; offsets are >> MAX_TRIALS so blocks never overlap.
_SEED_OFFSETS = {"carc": 0, "hoffman2": 5000}
SEED_OFFSET = next(
    (off for suffix, off in _SEED_OFFSETS.items() if CID.endswith(suffix)), 0
)

_train_win = TRAIN_WIN_DAYS * PERIODS_PER_DAY
_trials_dir = Path(f".hpc/campaigns/{CID}/trials")


def _capped_space():
    # Same widened space the fleet used → its seeded trials are in-distribution.
    from src.backtest.tree_space import widened_space

    return widened_space(MODEL, N_EST_CAP)


def _tell_from_results(study) -> int:
    """Close the loop: tell the study every RUNNING trial whose result has synced.

    Each trial writes results.csv under a combo-scoped path
    (``results/<run_id>/<MODEL>_<BUCKET>/trial_<n>/``); the framework syncs it back
    locally after the iteration. Compute its QLIKE (the audited per-row reduce) and
    ``study.tell(n, qlike)`` so the next ``ask`` accounts for this campaign's own
    trials, not just the seeded ones. Idempotent — only tells RUNNING trials, by the
    optuna trial number embedded in the path (unique per study, so no cross-combo mix).
    """
    import glob
    import re

    import pandas as pd

    from src.backtest.tune_tree import _compute_qlike

    running = {
        t.number for t in study.get_trials(deepcopy=False) if not t.state.is_finished()
    }
    told = 0
    for csv in glob.glob(
        f"results/**/{MODEL}_{BUCKET}/trial_*/results.csv", recursive=True
    ):
        m = re.search(r"trial_(\d+)[/\\]results\.csv$", csv.replace("\\", "/") + "")
        if not m:
            continue
        n = int(m.group(1))
        if n not in running:
            continue
        try:
            q = _compute_qlike(pd.read_csv(csv))
        except Exception:
            continue
        study.tell(n, q)
        running.discard(n)
        told += 1
    return told


def _build_tasks() -> list[dict]:
    from src.backtest.tune_tree import _load_or_create_study

    study = _load_or_create_study(
        "lgbm" if MODEL == "lgbm" else "xgb",
        storage_path=f".hpc/campaigns/{CID}/optuna.db",
        study_name=f"camp_{MODEL}_{BUCKET}",
    )
    _tell_from_results(
        study
    )  # tell finished trials BEFORE asking — closes the TPE loop
    done = sum(1 for t in study.get_trials(deepcopy=False) if t.state.is_finished())
    inflight = sum(
        1 for t in study.get_trials(deepcopy=False) if not t.state.is_finished()
    )
    if done >= MAX_TRIALS:
        return []

    _trials_dir.mkdir(parents=True, exist_ok=True)
    tasks: list[dict] = []
    space = _capped_space()
    for _ in range(min(K, MAX_TRIALS - done - inflight)):
        trial = study.ask(
            fixed_distributions=space
        )  # RUNNING -> constant_liar decorrelates the batch
        params = dict(trial.params)
        if (
            "n_estimators" in params
        ):  # EBM space omits n_estimators (its rounds cap is in-space)
            params["n_estimators"] = min(int(params["n_estimators"]), N_EST_CAP)
        pj = _trials_dir / f"trial_{trial.number}.json"
        pj.write_text(json.dumps(params, indent=2))
        cfg = _trials_dir / f"trial_{trial.number}.yaml"
        cfg.write_text(
            "\n".join(
                [
                    f"model: {_MODEL_MODULE[MODEL]}",
                    f"train_window: {_train_win}",
                    f"refit_frequency: {EMIT_END - EMIT_START}",  # fit-once over the slice
                    f"start: {EMIT_START}",
                    f"end: {EMIT_END}",
                    f"halo: {_train_win}",
                    f'exog_cols: "{_EXOG[BUCKET]}"',
                    f"params_file: {pj.as_posix()}",
                    f"output_file: results/{{run_id}}/{MODEL}_{BUCKET}/trial_{trial.number}/run.json",
                    "",
                ]
            )
        )
        tasks.append({"config": cfg.as_posix(), "_optuna_trial_number": trial.number})
    return tasks


def _chunk_ranges(n_blocks: int, n_chunks: int) -> list[tuple[int, int]]:
    """Partition cadence blocks [0, n_blocks) into <= n_chunks contiguous ranges."""
    import math

    sz = max(1, math.ceil(n_blocks / n_chunks))
    return [(i, min(i + sz, n_blocks)) for i in range(0, n_blocks, sz)]


def _tell_from_chunks(study, n_oos: int) -> int:
    """Close the loop for the CHUNKED residual trials: a trial's full-OOS preds are the concat
    of its chunk_*.csv (columns k,true_raw,pred_raw). Tell only when the chunks cover all OOS
    rows contiguously (the trial is complete); compute the audited full-OOS QLIKE and tell."""
    import glob
    import re

    import pandas as pd

    running = {
        t.number for t in study.get_trials(deepcopy=False) if not t.state.is_finished()
    }
    told = 0
    for d in glob.glob(f"results/**/{MODEL}_{BUCKET}/trial_*/", recursive=True):
        m = re.search(r"trial_(\d+)[/\\]$", d.replace("\\", "/"))
        if not m or int(m.group(1)) not in running:
            continue
        n = int(m.group(1))
        chunks = glob.glob(
            f"{d}/blk_*/chunk_*.csv"
        )  # chunks now live in per-task blk_* subdirs
        if not chunks:
            continue
        try:
            df = (
                pd.concat([pd.read_csv(c) for c in chunks])
                .drop_duplicates("k")
                .sort_values("k")
            )
        except Exception:
            continue
        if (
            len(df) != n_oos
            or int(df["k"].iloc[0]) != 0
            or int(df["k"].iloc[-1]) != n_oos - 1
        ):
            continue  # not all chunks landed yet
        import numpy as _np

        from src.evaluation.metrics import apply_duan_smearing

        pr, tr = apply_duan_smearing(
            df["pred_adj"].to_numpy(), df["y_true"].to_numpy(), df["base"].to_numpy()
        )
        mm = (tr > 0) & (pr > 0)
        rr = tr[mm] / pr[mm]
        study.tell(
            n, float(_np.mean(rr - _np.log(rr) - 1))
        )  # GLOBAL smear on the concat (not per-chunk)
        running.discard(n)
        told += 1
    return told


def _build_tasks_resid() -> list[dict]:
    """Residualized objective: REGULAR-TPE tuning of a tree on the Ridge residual, scored by
    FULL-OOS QLIKE on the slim per-slot-rank matrix. constant_liar is OFF — under a ~100-job
    cap, telling real results beats lying about in-flight trials. Each trial SATURATES by
    fanning its full-OOS backtest across CHUNKS time-chunks (cadence-block partition; the
    amortized Ridge cache makes each chunk a few tree fits). sqlite is fine (single driver)."""
    import json as _json

    import optuna
    from optuna.samplers import TPESampler

    import resid_amortized as ra

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cid = ra.cell_id(MODEL, BUCKET, TRAIN_WIN_DAYS, ALPHA, TREE_REFIT, PIPE, BASE)
    cell = _json.load(
        open(f"{ra.CACHE_ROOT}/{cid}/cell.json")
    )  # cache must be prepped first
    n_oos = int(cell["n"] - cell["train_win"])
    ranges = _chunk_ranges(int(cell["n_refits"]), CHUNKS)
    sampler = TPESampler(constant_liar=False, n_ei_candidates=96, seed=0)
    study = optuna.create_study(
        study_name=f"resid_{MODEL}_{BUCKET}",
        storage=f"sqlite:///.hpc/campaigns/{CID}/optuna.db",
        sampler=sampler,
        direction="minimize",
        load_if_exists=True,
    )
    _tell_from_chunks(
        study, n_oos
    )  # tell completed trials BEFORE asking — closes the TPE loop
    done = sum(1 for t in study.get_trials(deepcopy=False) if t.state.is_finished())
    if done >= MAX_TRIALS:
        return []
    _trials_dir.mkdir(parents=True, exist_ok=True)
    tasks: list[dict] = []
    space = _capped_space()
    # Reuse already-in-flight trials before asking new ones. The driver asks a trial,
    # writes its configs, and deploys the optuna.db; EVERY later import — and the
    # cluster dispatcher re-imports this module once per array task — must rebuild the
    # SAME trial's chunks, NOT ask a fresh trial (which would make each of the CHUNKS
    # array tasks run a DIFFERENT trial and produce incoherent results). So take the
    # in-flight trials first (deterministic), and only ask() to top the batch up to K.
    chosen = [t for t in study.get_trials(deepcopy=False) if not t.state.is_finished()][
        :K
    ]
    # Bug-2 fix: the sampler is re-created in a FRESH PROCESS on every enumeration
    # (each driver iteration / cluster re-import). With a fixed seed and K=1, the
    # startup-phase RNG reset identically every time, so every trial drew the SAME
    # point (trials 0/1/2 were byte-identical and the campaign never explored).
    # Re-seed by study progress before EACH ask so consecutive trials draw distinct
    # points. Cluster re-imports reuse the already-RUNNING trial (chosen=in-flight,
    # no ask), so chunk-rebuild determinism is preserved — only the driver's top-up
    # ask() is affected. len(chosen) makes it correct for K>1 too (no seed collision).
    for _ in range(max(0, min(K - len(chosen), MAX_TRIALS - done - len(chosen)))):
        study.sampler = TPESampler(
            constant_liar=False,
            n_ei_candidates=96,
            seed=SEED_BASE + SEED_OFFSET + done + len(chosen),
        )
        chosen.append(study.ask(fixed_distributions=space))
    for trial in chosen:
        params = dict(trial.params)
        if (
            "n_estimators" in params
        ):  # EBM space omits n_estimators (its rounds cap is in-space)
            params["n_estimators"] = min(int(params["n_estimators"]), N_EST_CAP)
        pj = _trials_dir / f"trial_{trial.number}.json"
        pj.write_text(json.dumps(params, indent=2))
        for blk0, blk1 in ranges:  # fan the trial's backtest across time-chunks
            cfg = _trials_dir / f"trial_{trial.number}_chunk_{blk0}_{blk1}.yaml"
            # Per-chunk result subdir so each cluster array task lands in a UNIQUE
            # dir — the framework's result_dir_template requires a per-task
            # placeholder (else it refuses the submit as clobbering). The blk key
            # encodes trial+chunk so it is unique across the whole array; the
            # output_file's {run_id} is substituted at run time (resid_tree reads
            # HPC_RUN_ID) so the written path matches the framework's result_dir.
            # _tell_from_chunks globs the chunk CSVs back out of these blk_* subdirs.
            blk_key = f"{trial.number}_{blk0}_{blk1}"
            cfg.write_text(
                "\n".join(
                    [
                        "model: resid_tree",
                        f"cid: {cid}",
                        f"arm: {ARM}",
                        f"blk0: {blk0}",
                        f"blk1: {blk1}",
                        f"params_file: {pj.as_posix()}",
                        f"output_file: results/{{run_id}}/{MODEL}_{BUCKET}/trial_{trial.number}/blk_{blk_key}/run.json",
                        "",
                    ]
                )
            )
            tasks.append({"config": cfg.as_posix(), "_optuna_trial_number": blk_key})
    return tasks


def _build_tasks_run10() -> list[dict]:
    """Run-#10 proving campaign (Path A fixed grid): the audited known-answer
    pair — the deployed ridge_market_ew_prod arm and the HAR-only candidate
    (config-varied only; mirrors specs/run10_known_answer.py). Two tasks, no
    optimizer, no stochastic marker (a fixed grid never repeats a cmd). ``arm``
    keys the per-task result dir (result_dir_template's placeholder); the
    free-form scope-lock tags ride the submit spec's ``scopes`` field."""
    d = Path(f".hpc/campaigns/{CID}")
    return [
        {"config": (d / "baseline.yaml").as_posix(), "arm": "baseline"},
        {"config": (d / "candidate.yaml").as_posix(), "arm": "candidate"},
    ]


try:
    if CID == "run10_proving":
        _TASKS = _build_tasks_run10()
    else:
        _TASKS = _build_tasks_resid() if OBJECTIVE == "residualized" else _build_tasks()
except Exception:
    # A bare import of this strategy file with NO campaign context — e.g. the
    # cluster status reporter importing tasks.py to report an UNRELATED run —
    # must not crash: the campaign build reads a cache cell / asks an Optuna
    # study that a foreign run's env does not provide. Re-raise only when we ARE
    # the campaign driver (HPC_CAMPAIGN_ID set) — then a build failure is real
    # and must surface loud. Otherwise degrade to an empty task list so the
    # generic status reporter can still resolve task paths for the other run.
    if os.environ.get("HPC_CAMPAIGN_ID"):
        raise
    _TASKS = []


def total() -> int:
    return len(_TASKS)


def resolve(i: int) -> dict:
    return _TASKS[i] if 0 <= i < len(_TASKS) else {}
