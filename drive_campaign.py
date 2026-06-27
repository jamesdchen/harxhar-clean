#!/usr/bin/env python
"""Autonomous driver for the residualized-EBM hpc-agent campaign (one cluster).

Per iteration: wait the in-flight array to terminal -> scp-pull its chunk CSVs ->
enumerate locally (tells the completed trial via _tell_from_chunks + asks/reuses the
next trial) -> build the per-iteration submit spec (result_dir_template keyed on the
trial number) -> submit-flow the next 82-task array. Loops until the study has
MAX_ITERS finished trials.

Run (background), once per cluster:
    <uv-tool-python> drive_campaign.py carc
    <uv-tool-python> drive_campaign.py hoffman2

State + log live under .hpc/campaigns/<cid>/ (drive_state.json, drive.log).
The driver shells out to `hpc-agent` (submit-flow/compute-run-id/batch-status) and `scp`.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    import msvcrt  # Windows: the local driver always runs here
except ImportError:  # pragma: no cover - POSIX fallback
    msvcrt = None  # type: ignore[assignment]


def _try_lock(fh) -> bool:
    """Non-blocking exclusive lock on byte 0 of fh. True if acquired."""
    if msvcrt is None:
        return True
    try:
        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        return True
    except OSError:
        return False


@contextlib.contextmanager
def _exclusive(path: Path, poll: float = 2.0, timeout: float = 1200.0):
    """Block until an exclusive lock on `path` is held, then release on exit.

    Serializes the submit/deploy critical section ACROSS clusters: carc and
    hoffman2 drivers run submit-flow in the SAME repo and contend on the shared
    local ``.hpc/runs/`` deploy state — concurrent deploys dropped a run's
    sidecar and wedged a whole cluster. One global lock makes deploys sequential.
    """
    fh = open(path, "a+")  # noqa: SIM115 (held for lifetime — released in finally)
    waited = 0.0
    try:
        while not _try_lock(fh):
            time.sleep(poll)
            waited += poll
            if waited >= timeout:
                raise TimeoutError(f"could not acquire {path} within {timeout}s")
        yield
    finally:
        if msvcrt is not None:
            with contextlib.suppress(OSError):
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        fh.close()


REPO = r"C:/Users/james/CC Allowed/harxhar-clean"
UVPY = r"C:/Users/james/AppData/Roaming/uv/tools/hpc-agent/Scripts/python.exe"
SSH = r"C:/Windows/System32/OpenSSH/ssh.exe"
SCP = r"C:/Windows/System32/OpenSSH/scp.exe"
MAX_ITERS = 25
POLL_SEC = 120
RUN_NAME = "ebm_resid"

# Strategy knobs (identical across clusters; campaign_id differs per cluster).
KW = {
    "HPC_KW_MODEL": "ebm",
    "HPC_KW_BUCKET": "all_buckets",
    "HPC_KW_OBJECTIVE": "residualized",
    "HPC_KW_BASE": "enet",
    "HPC_KW_ALPHA": "1",
    "HPC_KW_ARM": "resid_subset",
    "HPC_KW_PIPE": "slim",
    "HPC_KW_TREE_REFIT": "480",
    "HPC_KW_TRAIN_WIN_DAYS": "1000",
    "HPC_KW_CHUNKS": "90",
    "HPC_KW_K": "1",
    "HPC_KW_MAX_TRIALS": "200",
    "HPC_KW_N_EST_CAP": "300",
}
# Derived from KW so study_state() / fail_trial() always query the right Optuna study.
# tasks.py builds study_name=f"resid_{MODEL}_{BUCKET}" from the same env vars; deriving
# here prevents a silent mismatch (wrong study → fail_trial / study_state broken) when
# MODEL or BUCKET changes.
STUDY_NAME = f"resid_{KW['HPC_KW_MODEL']}_{KW['HPC_KW_BUCKET']}"

CLUSTERS: dict[str, dict[str, Any]] = {
    "carc": {
        "campaign_id": "ebm_all_buckets_carc",
        "backend": "slurm",
        "ssh_target": "jc_905@discovery2.usc.edu",
        "ssh_alias": "usc-discovery",
        "remote_path": "/scratch1/jc_905/harxhar-clean",
        "modules": "conda",
        "conda_source": "/apps/conda/miniforge3/25.3.0/etc/profile.d/conda.sh",
        "conda_env": "harxhar",
        # CARC iteration-1 array is already in flight (trial 0); seed the loop with it.
        "seed_state": {"run_id": "ebm_resid-88cb4a2a", "job_id": "9638700", "trial": 0},
    },
    "hoffman2": {
        "campaign_id": "ebm_all_buckets_hoffman2",
        "backend": "sge",
        "ssh_target": "jamesdc1@hoffman2.idre.ucla.edu",
        "ssh_alias": "hoffman2",
        "remote_path": "/u/scratch/j/jamesdc1/harxhar-clean",
        "modules": "",
        "conda_source": "/u/local/apps/anaconda3/2023.03/etc/profile.d/conda.sh",
        "conda_env": "hpc-pi",
        "seed_state": {},  # cold start
    },
}


def main() -> int:
    cluster = sys.argv[1]
    cfg = CLUSTERS[cluster]
    cid = cfg["campaign_id"]
    cdir = Path(REPO) / ".hpc" / "campaigns" / cid
    cdir.mkdir(parents=True, exist_ok=True)
    # Per-cid single-instance lock: refuse to start a second driver for the SAME
    # campaign (the wedge that spawned this fix had duplicate launches). Held for
    # the process lifetime; the OS releases it if the driver dies/is killed.
    _instance_fh = open(cdir / "driver.lock", "a+")  # noqa: SIM115 (held for lifetime)
    if not _try_lock(_instance_fh):
        print(
            f"[{cluster}] another driver for {cid} is already running; exiting.",
            flush=True,
        )
        return 1
    submit_lock_path = Path(REPO) / ".hpc" / "submit_global.lock"
    state_path = cdir / "drive_state.json"
    log_path = cdir / "drive.log"
    study_storage = f"sqlite:///.hpc/campaigns/{cid}/optuna.db"

    env = dict(os.environ)
    env.update(KW)
    env["HPC_CAMPAIGN_ID"] = cid
    env["PYTHONPATH"] = REPO

    def log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [{cluster}] {msg}"
        with open(log_path, "a") as fh:
            fh.write(line + "\n")
        print(line, flush=True)

    def run(args: list[str], timeout: int = 900) -> dict:
        p = subprocess.run(args, env=env, cwd=REPO, capture_output=True, text=True, timeout=timeout)
        out = (p.stdout or "").strip()
        try:
            return json.loads(out.splitlines()[-1]) if out else {}
        except Exception:
            return {"_raw": out, "_stderr": (p.stderr or "")[:500], "_rc": p.returncode}

    def study_state() -> dict:
        code = (
            "import optuna,json;"
            "from optuna.trial import TrialState as TS;"
            f"s=optuna.load_study(study_name='{STUDY_NAME}', storage='{study_storage}');"
            "ts=s.get_trials(deepcopy=False);"
            "r=[t.number for t in ts if not t.state.is_finished()];"
            "comp=[(t.number,t.value) for t in ts if t.state==TS.COMPLETE];"
            "fin=[t.number for t in ts if t.state.is_finished()];"
            "print(json.dumps({'running':r,'finished':len(fin),'complete':len(comp),'trials':comp}))"
        )
        p = subprocess.run([UVPY, "-c", code], env=env, cwd=REPO, capture_output=True, text=True)
        out: dict = json.loads(p.stdout.strip().splitlines()[-1])
        return out

    def fail_trial(n: int) -> None:
        """Mark a stuck/failed Optuna trial FAILED so the next ask() advances past it.
        Used when a trial's cluster job went terminal but produced no foldable chunks
        (all tasks failed / dropped deploy) — otherwise the loop resubmits forever."""
        code = (
            "import optuna;from optuna.trial import TrialState as TS;"
            f"s=optuna.load_study(study_name='{STUDY_NAME}', storage='{study_storage}');"
            f"t=next((t for t in s.get_trials(deepcopy=False) if t.number=={n}), None);"
            f"s.tell({n}, state=TS.FAIL) if (t is not None and not t.state.is_finished()) else None"
        )
        subprocess.run([UVPY, "-c", code], env=env, cwd=REPO, capture_output=True, text=True)

    def job_terminal(job_id: str) -> bool:
        bs = run([UVPY, "-m", "hpc_agent", "batch-status", "--experiment-dir", "."])
        runs = bs.get("data", {}).get("runs", {})
        for r in runs.values():
            st = r.get("job_states", {})
            if job_id in st and st[job_id] in ("running", "pending", "configuring"):
                return False
        return True

    def pull(run_id: str) -> None:
        remote = f"{cfg['ssh_alias']}:{cfg['remote_path']}/results/{run_id}"
        dest = str(Path(REPO) / "results")
        subprocess.run(
            [SCP, "-q", "-o", "BatchMode=yes", "-r", remote, dest],
            capture_output=True,
            text=True,
            timeout=600,
        )

    def submit_trial(trial_n: int) -> dict:
        rdt = f"results/{{run_id}}/ebm_all_buckets/trial_{trial_n}/blk_{{_optuna_trial_number}}"
        extra = {"PYTHONPATH": cfg["remote_path"], **KW}
        rsi = {
            "run_name": RUN_NAME,
            "submit": {
                "profile": RUN_NAME,
                "cluster": cluster,
                "ssh_target": cfg["ssh_target"],
                "remote_path": cfg["remote_path"],
                "run_id": "ebm_resid-00000000",
                "cmd_sha": "0000000000000000",
                "total_tasks": 82,
                "backend": cfg["backend"],
                "modules": cfg["modules"],
                "conda_source": cfg["conda_source"],
                "conda_env": cfg["conda_env"],
                "campaign_id": cid,
                "result_dir_template": rdt,
                "extra_env": extra,
            },
            "sidecar": {
                "run_id": "ebm_resid-00000000",
                "cmd_sha": "0000000000000000",
                "executor": "python run.py --config $CONFIG",
                "result_dir_template": rdt,
                "task_count": 82,
                "cluster": cluster,
                "profile": RUN_NAME,
                "campaign_id": cid,
            },
        }
        rsi_path = cdir / "rsi_spec.json"
        rsi_path.write_text(json.dumps(rsi))
        # Hold the global submit lock across resolve + submit-flow: this is the
        # deploy critical section that contends with the other cluster's driver.
        with _exclusive(submit_lock_path):
            res = run(
                [
                    UVPY,
                    "-m",
                    "hpc_agent",
                    "resolve-submit-inputs",
                    "--spec",
                    str(rsi_path),
                    "--experiment-dir",
                    ".",
                ]
            )
            spec = res.get("data", {}).get("submit_spec")
            if not spec:
                log(f"resolve-submit-inputs FAILED: {json.dumps(res)[:600]}")
                return {}
            spec["canary"] = False
            spec["canary_only"] = False
            spec_path = cdir / "submit_spec.json"
            spec_path.write_text(json.dumps(spec))
            sf = run(
                [UVPY, "-m", "hpc_agent", "submit-flow", "--spec", str(spec_path)],
                timeout=1200,
            )
        d = sf.get("data", {})
        return {"run_id": d.get("run_id"), "job_id": (d.get("job_ids") or [None])[0]}

    # ---- main loop ----
    if state_path.exists():
        state = json.loads(state_path.read_text())
    else:
        state = dict(cfg["seed_state"])
        state_path.write_text(json.dumps(state))
    log(f"START driver; state={state}")

    while True:
        try:
            job_id = state.get("job_id")
            run_id = state.get("run_id")
            waited_trial = state.get("trial")
            if job_id:
                log(f"waiting on array {run_id} job {job_id} ...")
                while not job_terminal(str(job_id)):
                    time.sleep(POLL_SEC)
                log(f"array {run_id} job {job_id} terminal; pulling results")
                pull(run_id)
            # tell completed trial + ask/reuse next (compute-run-id imports tasks.py)
            run(
                [
                    UVPY,
                    "-m",
                    "hpc_agent",
                    "compute-run-id",
                    "--run-name",
                    RUN_NAME,
                    "--experiment-dir",
                    ".",
                ]
            )
            st = study_state()
            log(
                f"study: complete={st['complete']} finished={st['finished']}"
                f" running={st['running']} last={st['trials'][-3:]}"
            )
            # Failed-trial handling: the trial we just waited on went terminal, we
            # pulled, and compute-run-id tried to fold it. If it is STILL running it
            # produced no foldable chunks (all tasks failed / dropped sidecar deploy).
            # Mark it FAILED and advance — never resubmit the same deduped run forever.
            if waited_trial is not None and waited_trial in st["running"]:
                log(
                    f"trial {waited_trial} job terminal but did not fold"
                    " (no foldable chunks); marking FAILED, advancing."
                )
                fail_trial(waited_trial)
                state = {}
                state_path.write_text(json.dumps(state))
                continue
            if st["complete"] >= MAX_ITERS:
                log(f"DONE: complete {st['complete']} >= MAX_ITERS {MAX_ITERS}. Stopping.")
                break
            if not st["running"]:
                log("no running trial after ask; stopping.")
                break
            trial_n = st["running"][0]
            log(f"submitting iteration for trial {trial_n}")
            sub = submit_trial(trial_n)
            if not sub.get("job_id"):
                log(f"submit FAILED for trial {trial_n}; stopping. {sub}")
                break
            state = {"run_id": sub["run_id"], "job_id": sub["job_id"], "trial": trial_n}
            state_path.write_text(json.dumps(state))
            log(f"submitted {state}")
        except Exception as exc:  # noqa: BLE001
            log(f"ITERATION ERROR ({type(exc).__name__}): {exc}; stopping for safety.")
            break

    log("driver exit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
