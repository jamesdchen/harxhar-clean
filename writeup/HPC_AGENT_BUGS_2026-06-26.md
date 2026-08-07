# hpc-agent framework bugs (found driving a Path-B Optuna campaign)

Repo under test: hpc-agent **0.10.65**. Source tree referenced below:
`C:\Users\james\CC Allowed\hpc-agent\src\hpc_agent\` (paths are relative to that root).

Context: driving a campaign whose `.hpc/tasks.py` is a hand-written Optuna strategy (Path B —
`tasks.py` branches on `HPC_CAMPAIGN_ID`, asks the study, fans each trial across N chunk tasks).
The five issues below are **framework-side** (the campaign code's own bugs are tracked separately in
`CAMPAIGN_DRIVE_STATUS_2026-06-26.md`). Ordered by severity.

---

## BUG 1 — local task enumeration omits the experiment repo root from `sys.path`
**Severity: high (blocks every local enumeration of a tasks.py that imports a repo-root module).**

`load_tasks_module` (`__init__.py:186`) imports the user's `tasks.py` via
`importlib.util.spec_from_file_location(...)` + `exec_module` **without putting the experiment dir on
`sys.path`**. So a `tasks.py` that does `import my_root_module` or `from src.x import y` (where
`my_root_module.py` / `src/` live at the experiment root) raises `ModuleNotFoundError` during the
**local** enumeration done by `compute-run-id` (`incorporation/build/compute_run_id.py:97`) and
`build-submit-spec` (`incorporation/build/submit_spec.py:786`).

The **cluster** dispatcher does NOT have this problem — `execution/mapreduce/dispatch.py` (~`_load_tasks_module`, near `here = Path(__file__).resolve().parent`, ~line 511) adds the repo root. So
the same tasks.py imports fine on the cluster but fails locally. Inconsistent.

- **Repro:** `tasks.py` with `import foo` where `foo.py` is at the repo root; run
  `hpc-agent compute-run-id --run-name x --experiment-dir <repo>` → `tasks.py ... is malformed: No
  module named 'foo'`. Workaround: `PYTHONPATH=<repo> hpc-agent compute-run-id ...`.
- **Fix:** in `load_tasks_module` (or its local callers) insert `str(tasks_py.parent.parent)` (the
  experiment dir) at `sys.path[0]` around the `exec_module`, mirroring the cluster dispatcher.

---

## BUG 2 — status reporter discards the real error doc on non-zero exit (bad diagnostics)
**Severity: high (turns every reporter error into an opaque `unable_to_verify`).**

The on-cluster reporter `python -m hpc_agent.execution.mapreduce.reduce.status` writes a **structured
error document to STDOUT** and returns exit code **2** on any handled error
(`execution/mapreduce/reduce/status.py:1030` `_emit_err(..., exit_code=2)`; e.g.
`tasks_py_import_error` at `1064-1066`, `sidecar_not_found` at `1050-1053`).

But the caller `infra/cluster_status.py:69-72` does:
```python
if proc.returncode != 0:
    raise RemoteCommandFailed(f"status reporter failed (rc={proc.returncode}): {proc.stderr.strip()[:200]}")
```
It surfaces only `proc.stderr` (truncated to 200 chars) and **throws away stdout** — which is where the
machine-readable `{errors:[{code, detail}]}` lives. On an Lmod cluster the 200 chars are consumed by
benign `module load` noise ("The following have been reloaded with a version change: python/3.11 =>
3.13" + a `runpy` RuntimeWarning), so the operator sees noise and `reconcile` reports
`verify_state: unable_to_verify` with no actionable cause. (This is what masked BUG 3 below for us.)

- **Fix:** on `rc != 0`, first try to parse `proc.stdout` as the reporter's JSON envelope and surface
  `errors[].code` / `errors[].detail`; fall back to stderr only when stdout is not the structured doc.

---

## BUG 3 — status reporter imports the live `tasks.py` for UNRELATED runs
**Severity: high (one campaign's strategy file can wedge status/reconcile for every other run).**

`status.py:1064` calls `load_tasks_module(.hpc/tasks.py)` to synthesize per-task result dirs
(`_build_per_task_dict_from_sidecar`, `status.py:943-980`) — for **any** run being reported, using the
**current** `.hpc/tasks.py`. If that file is a different campaign's strategy (heavy imports, or
import-time side effects, or it needs campaign env vars that the foreign run doesn't set), the import
fails → `tasks_py_import_error` → exit 2 → (via BUG 2) opaque `unable_to_verify`. The unrelated run can
no longer be reconciled, so it stays `in_flight` and blocks new submits.

Note the import is often **unnecessary**: `_build_per_task_dict_from_sidecar` only uses `resolve(i)`
kwargs if `result_dir_template` references them. For a template like `results/{run_id}/task_{task_id}`
(no kwargs), the `resolve()` output is unused — yet the import (and its failure) still gates the report.

- **Fix:** skip the `tasks.py` import when `result_dir_template` references no `resolve()` kwargs
  (only `{run_id}`/`{task_id}`); or import lazily and, on failure, degrade to `task_id`-only result
  dirs instead of failing the whole report.

---

## BUG 4 — campaign manifest `strategy.params` are never materialized into `HPC_KW_*`
**Severity: high (a campaign submit via the bare worker silently produces empty/wrong tasks).**

A Path-B strategy `tasks.py` reads its knobs from `os.environ["HPC_KW_<PARAM>"]` (the documented
convention). The campaign manifest stores them under `strategy.params`
(`.hpc/campaigns/<id>/manifest.json`). **Nothing wires the two together:**

- `build-submit-spec` (`incorporation/build/submit_spec.py:311-324`) seeds `job_env` with the framework
  defaults + `HPC_CAMPAIGN_ID`, but **never adds `HPC_KW_<param>`** from the manifest.
- The campaign meta layer (`meta/campaign/*`) has no `os.environ` / `HPC_KW_` writes (grep is empty).
- So neither the **local enumeration** env (which imports `tasks.py` to compute `cmd_sha`/`total`) nor
  the **cluster job_env** carries the strategy params.

Result: `hpc-agent run --workflow submit` with only `{experiment_dir, cluster, campaign_id}` reuses the
latest non-campaign profile and enumerates `tasks.py` under default knobs → wrong/empty task list
(observed: `cmd_sha = e3b0c442…` = the empty-string hash → canary "dispatcher_failed"). The operator
must hand-export every `HPC_KW_*` in the shell AND duplicate them into the submit `extra_env`.

- **Fix:** when `campaign_id` is set, read `manifest.strategy.params` and export each as
  `HPC_KW_<KEY.upper()>` into (a) the process env before `load_tasks_module` during local enumeration,
  and (b) the submit-flow `job_env`. (The cluster dispatcher already maps `resolve()` kwargs →
  `HPC_KW_*` at `dispatch.py:815`; this is the symmetric missing half for strategy params.)

---

## BUG 5 — per-array-task re-import re-derives a stochastic `tasks.py` (incoherent results)
**Severity: high (silent wrong results for any non-deterministic strategy). Deepest design issue.**

The cluster dispatcher re-imports `tasks.py` and calls `resolve(task_id)` **independently in each of the
N array tasks** (`execution/mapreduce/dispatch.py:520` `_load_tasks_module`, `:612`
`kwargs = tasks.resolve(task_id)`). For a hand-written strategy whose module top-level is
non-deterministic — e.g. Optuna `study.ask()` at import, the canonical Path-B pattern — each array task
re-runs `ask()` and gets a **different** trial. A run that fanned "trial T across N chunks" instead
executes N *different* trials' single chunks, and nothing reconstructs trial T. The failure is silent
(no error; just wrong numbers).

The framework already materializes the list once for *generated* tasks.py (the planner bakes a literal
list — `incorporation/build/tasks_py.py:524-528`), and `compute_run_id.py:108` already does
`[tasks.resolve(i) for i in range(tasks.total())]` once at submit. But that materialization is **not
shipped to the cluster**; the dispatcher re-derives live.

- **Fix (either):** (a) materialize `[resolve(i) …]` once at submit, ship it as a frozen task manifest,
  and have the dispatcher read it instead of re-importing `tasks.py`; or (b) make the contract explicit
  that `resolve(i)` MUST be deterministic across processes, and ship a helper / validate-campaign check
  that flags a stochastic module top-level. (Our workaround was to make `tasks.py` reuse the already
  in-flight Optuna trial instead of asking a new one — but the framework shouldn't require every Path-B
  author to discover this independently.)

---

## Minor

- **build-submit-spec does not auto-read `modules`/`conda_source`/`conda_env` from `clusters.yaml`.**
  A spec built directly (not via the worker) fails with "submission has no env-activation declared"
  unless these three are re-supplied, even though they're in `clusters.yaml` for that cluster.
  (`incorporation/build/submit_spec.py`; the worker happens to pass them, the raw primitive doesn't.)
