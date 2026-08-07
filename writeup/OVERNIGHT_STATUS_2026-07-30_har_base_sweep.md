# Overnight drive status — har_base_sweep (2026-07-30, ~03:30 PT park)

## What you asked for
Full battery: HAR-ladder base sweep (b ∈ {2,3,4,5,6,8} × caps {240,960,3125} + explicit {1,8,48,240,960} truncated-to-cap; 20 unique arms), OLS-HAR estimator held fixed, empty-exog bucket, QLIKE + DM vs the base-5 OLS-HAR incumbent, driven to completion overnight.

## What COMPLETED (all durable, all committed)
- **Audit `har_base_sweep` PASSED** — 5/5 sections (2 auto-cleared, 3 signed by you); known-answer verified in dry-run: the new `har_lags` seam at (b=5,cap=3125) reproduces the production incumbent EXACTLY.
- **src seam committed**: `run_executor(har_lags=…)` → `generate_har_features(lags=…)`, default None = byte-identical production path; burn-in fixed at 3125 (arm-invariant bar grid, bar-for-bar DM); rungs>3125 refused. Commits: `feat(har): har_lags ladder seam + audited har_base_sweep spec`, `feat(har-sweep): interview + submit wiring + pooled run-level reducer`.
- **Interview materialized**: `tasks.py` + `interview.json` + wrapper, 2100 tasks (21 grid cells × 100 chunks; the (explicit,3125) cell is a disclosed byte-duplicate of explicit@960), cmd_sha `53c27e42…`, run_id `har_base_sweep-53c27e42`.
- **Run-level reducer written+committed**: `specs/reduce_har_base_sweep.py` (pooled per-bar concat per arm + forecast_metrics + DM; register as aggregate_cmd at aggregate time — the built-in JSON reducer cannot reduce the csv table).
- S1 resolved clean on carc AND (after failover) on hoffman2; your `y` journaled; audit gate reported **current** at submit.

## What went WRONG overnight (in order)
1. **S2 canary on CARC died unreadable** (`reporter_unreachable`): discovery2 preamble timeouts → SSH circuit opened.
2. Root cause found: **the USC VPN dropped** (~10:15Z). net-triage verdict: discovery1 + discovery2 `host_unreachable_network_ok`; Hoffman2 fully reachable. I cannot re-auth a Duo VPN.
3. VPN recovery probe (30 min) timed out → executed cluster **failover to Hoffman2** under your typed authority.
4. Blocker A: the dead CARC canary attempt (`…-canary2`) sat non-terminal at "submitting" with **zero scheduler job ids** and blocked every re-resolve; no code path could settle it with USC transport down (`is_kill_confirmed` excludes zero-job records — journal.py:891, defect-shaped gap). **DISCLOSED MANUAL ACTION**: I marked that record `abandoned` via the package's own `state.journal.mark_run` API. Evidence it was safe: `job_ids: []` (nothing was ever queued), kill requested+confirmed over the empty set. The decision-journal note for this was blocked by the permission classifier; this file + the chat transcript are the disclosure.
5. Blocker B (**the park**): the fresh Hoffman2 stage+canary replays the stale CARC canary terminal (idempotency by run_id); the fix is `force_canary: true` — and the Claude Code auto-mode permission classifier **denied that call in both CLI and MCP form**. Retrying a denied action would violate the harness contract, so the chain is parked here.

## Where everything stands
- **Nothing is running on any cluster.** CARC: no jobs ever queued (verified). Hoffman2: staged nothing yet.
- Repo state committed; `.hpc` journals consistent; stale attempt closed out (abandoned, disclosed).
- Old unrelated debris the doctor flagged: 7 dead detached workers from pi-drill/spectral runs (pre-existing, untouched).

## To get your results (one step, then it drives itself)
1. If you want CARC: reconnect the USC VPN first. Otherwise Hoffman2 is ready now.
2. Re-issue the submit: ask the session to re-run `submit-s2` for `har_base_sweep-53c27e42` on hoffman2 with `force_canary: true` and approve the permission prompt when it appears. From your `y`-on-green consent (already journaled), the rest — S3 array submit, watch, S4 harvest, aggregate with `specs/reduce_har_base_sweep.py` — proceeds without further input. Est. compute: 25,200 core-hours ceiling (2100 × 3h × 4 cpu walltime cap); realistically each task is minutes, wall-clock ~1–3h at 100 concurrent.

## Small print for the writeup
- Caps axis is {240, 960, 3125} (not 3840 as first proposed): the fixed 3125 burn-in keeps every arm on the incumbent's exact bar grid; a 3840 rung would break bar-for-bar DM alignment.
- `specs/har_base_sweep.py` carries repo-convention E402s and is deliberately NOT ruff-formatted (formatting would move audited section hashes — standing rule).
- mypy errors in `src/backtest/multi_stage.py` are pre-existing, untouched.
- Tooling defects hit tonight (for the hpc-agent backlog): pack-status/block-drive "no run record" reader blindness; MCP-server env skew (uv env lacks the science stack — lint/gates must run from 285J); `is_kill_confirmed` zero-job gap; S1 prior-run gate has no journaled-decision consumption path.
