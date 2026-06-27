# EBM Campaign Drive — Status (2026-06-26, ~15:12 — SUPERSEDES the earlier version of this file)

> **CAMPAIGN STOPPED ~17:50 — best QLIKE 0.12414 (plateau-limited, deliberate stop). Drivers killed.**
> The hand-rolled driver self-poisons the hpc-agent journal (never calls `campaign advance`) → batch-status
> chokes + ssh-floods CARC → carc stalled at complete=11. A long campaign needs the hpc-agent campaign
> machinery or periodic journal reconcile. **The real payoff is the downstream science:**
> `writeup/intraday_regime_findings_2026-06-26.md`. (The driving status below is historical.)

**Outcome (historical, ~15:12): both clusters were DRIVING & HEALTHY.** The campaign was wedged on trial 2 and silently
*not exploring* (every trial was the same point). Three real bugs found and fixed; all proven live.
carc is at trial 4+, hoffman2 at trial 3+, both asking distinct hyperparameter points and folding.

> The earlier version of this doc said blocker **#5 (output-layout)** was unresolved. That was stale:
> `.hpc/tasks.py` already had the per-task `blk_<trial>_<blk0>_<blk1>/chunk_*.csv` layout, and trials
> 0 & 1 had already folded through the real hpc-agent tick (QLIKE 0.12424 < enet base 0.12516). #5 is done.

---

## What was actually wrong (3 bugs)

### Bug A — the campaign wasn't tuning (scientific)
`_build_tasks_resid` re-created `TPESampler(seed=0)` on **every** enumeration. Each driver iteration /
cluster re-import is a fresh process, so with K=1 and startup-phase random sampling the re-seeded RNG
drew the **same point every trial** → `trial_0/1/2.json` were byte-identical (QLIKE 0.12424 twice).
As-is the campaign would burn all 25 iterations on one point.

**Fix** (`.hpc/tasks.py`): re-seed by study progress before each ask —
`TPESampler(..., seed=SEED_BASE + done + len(chosen))`, `SEED_BASE=1000`. Cluster re-imports reuse the
already-RUNNING trial (no `ask()`), so chunk-rebuild determinism is preserved; only the driver's top-up
ask is affected. **Proven:** trial 3 drew the predicted seed-1003 point (lr 0.04536 / leaves 3 / inter 0 /
bins 663 / rounds 500) → QLIKE **0.12477 ≠ 0.12424** = genuinely exploring.

### Bug B — infinite resubmit (no failed-trial handling)
When a trial's cluster job goes terminal but yields no foldable chunks, `_tell_from_chunks` can't tell it,
so it stays RUNNING and the driver resubmits — but `submit-flow` **dedupes on `cmd_sha`**, so it never
re-runs. Permanent wedge (carc spun on trial 2 from 14:18→15:02; hoffman2 likewise).

**Fix** (`drive_campaign.py`): after pull+tell, if the waited trial is still in `running`,
`study.tell(n, state=FAIL)` and advance. Stop criterion now counts **COMPLETE** trials (not `finished`,
which includes FAIL). **Proven:** on restart both drivers marked trial 2 FAILED and advanced to trial 3.

### Bug C — cross-driver deploy contention (operational root cause of trial 2)
carc + hoffman2 run `submit-flow` in the **same repo** and contend on the shared local `.hpc/runs/`
deploy state. carc's `ebm_resid-c85ebaed.json` sidecar never reached the cluster, so all 82 tasks died
`[dispatch] ERROR: run sidecar not found` (~4s each) → Bug B then wedged it. (Red herring: task k's
`$SLURM_JOB_ID = SLURM_ARRAY_JOB_ID + k` looked like two jobs; it's one.)

**Fix** (`drive_campaign.py`): a global file-lock (`.hpc/submit_global.lock`, msvcrt) around
resolve+submit-flow so deploys are **sequential** across clusters; plus a **per-cid instance lock**
(`.hpc/campaigns/<cid>/driver.lock`) so a campaign can't be double-launched. **Proven:** trial 3 sidecar
deployed, job 9641163 = 82/82 tasks COMPLETED, folded to QLIKE 0.12477.

---

## Validation summary
| Check | Result |
|---|---|
| ruff check / format | pass / clean (both files) |
| seed re-seed unit test | old seed0==seed0 (bug); new 5 seeds → 5 distinct trials |
| failed-trial handling | trial 2 → FAIL → advance (carc + hoffman2) |
| deploy fix | trial 3 sidecar present, 82/82 COMPLETED, folded |
| exploration | trial 3 QLIKE 0.12477 ≠ trials 0/1 0.12424 |
| autonomy | carc advanced 2→3→4 unattended |

mypy: not run to completion locally (slow on the `src.*` import graph); ruff + py_compile + the live run cover it.

## State on disk
- `.hpc/tasks.py`, `drive_campaign.py` — Bugs A/B/C committed in `d0ff1f5`; Bug D (hardcoded study name in `study_state`/`fail_trial`) committed subsequently.
- Study reconciliation: trial 2 is FAILED in both `optuna.db`; trials 0,1 are a (redundant) duplicate
  point — kept as 2 valid observations. Trials 3+ are distinct.
- `.hpc/campaigns/<cid>/driver.lock`, `.hpc/submit_global.lock` — new lock files.

## To babysit / resume
- Watch: `tail -f .hpc/campaigns/ebm_all_buckets_carc/drive.log` (healthy line = `complete=N finished=M running=[next]`).
- Resume after a stop/kill: relaunch `<uvpy> drive_campaign.py {carc|hoffman2}` — the per-cid lock blocks
  double-launch, and the Bug-B fix self-heals any stuck trial. Stops at 25 COMPLETE trials/cluster.
