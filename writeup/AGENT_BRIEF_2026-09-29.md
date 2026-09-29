# Agent brief — overnight run 2026-09-29 (every workstream agent reads this first)

You are one of ~10 Opus agents working concurrently in the SAME worktree
`C:\Users\james\CC Allowed\harxhar-0dte-professor` (Git Bash `/c/Users/james/CC Allowed/harxhar-0dte-professor`),
branch `insurance-selling-2026-09-19`. The user is asleep and wants everything finished by morning: do not ask questions,
make the sensible call, record it, and finish. Read `writeup/SESSION_HANDOFF_2026-09-29.md` in full before starting
(environment, ground rules, where things are). The checklist is `writeup/PROGRESS_2026-09-29.md`.

## Environment (from the handoff; the essentials)
- Python: `/c/Users/james/miniconda3/envs/285J/python.exe` (the msys `python` on PATH has no pandas). pdflatex/pdftoppm/pdftotext on PATH.
- CARC (Slurm, main workhorse): `/c/Windows/System32/OpenSSH/ssh.exe -o BatchMode=yes usc-discovery '<cmd>'`, project root `/scratch1/jc_905/harxhar-subsection`. The CARC python (`python` after the module loads, 3.13) has no torch. QOS normal: 100 running / 5000 submitted jobs per user, 2000 CPUs.
- Hoffman2 (SGE, lighter/faster jobs): `/c/Windows/System32/OpenSSH/ssh.exe -o BatchMode=yes hoffman2 '<cmd>'`, root `/u/scratch/j/jamesdc1/harxhar-subsection` (scratch is purged — check files exist, re-ship what is missing). `qsub -terse -t 1-N`, `-hold_jid`, `-l h_rt=HH:MM:SS,h_data=16G`; env `source /u/local/apps/anaconda3/2023.03/etc/profile.d/conda.sh && conda activate hpc-pi`. Keep its jobs small.
- SUBMITTING: only through a `cluster/.../submit_*.sh` script run over native ssh — `ssh.exe -o BatchMode=yes usc-discovery "cd /scratch1/jc_905/harxhar-subsection && bash cluster/slurm/submit_<x>.sh"` (or the hoffman2 twin). That exact form is pre-approved; a raw `sbatch`/`qsub` over ssh is blocked. `scancel`/`qdel` are denied (cannot cancel jobs — submit carefully, canary first). Ship with a `ship_<x>_carc.sh` twin: tar over native ssh + md5 check (copy `cluster/slurm/ship_treestuned_carc.sh` / `submit_treestuned.sh` / `treestuned_pack.sbatch` / `treestuned_score.sbatch`).
- Do NOT use any `mcp__hpc-agent__*` tool or hpc-agent skill. Reuse its code only if helpful.
- Waiting on a cluster: no foreground `sleep` in Bash; poll with the sleep INSIDE the ssh command, e.g.
  `ssh.exe ... usc-discovery 'for i in $(seq 1 50); do [ -f <flag> ] && break; sleep 10; done; ls <dir>'` with the Bash tool `timeout` set to 600000, repeated as needed. Locally, `python.exe -c "import time; time.sleep(300)"` works.

## Shared-resource protocol (STRICT)
- CHAIN LOCK: anything that loads `data/spxw_chain.parquet` or runs `notebooks/atm_straddle_intraday_holdclose*` / `_write_0dte_intraday_T_nb.py` / `_run_nb.py` on a chain-sized notebook is a "chain-sized job". At most ONE chain-sized job at a time across all agents (the box kills ~3). Acquire the lock with the atomic
  `mkdir "/c/Users/james/CC Allowed/harxhar-0dte-professor/.chain_lock.d" && echo "<owner> $(date)" > .chain_lock.d/owner` — if mkdir fails, wait (python sleep 120 s) and retry; release with `rm -rf .chain_lock.d` in the same Bash call that finishes the job (`... ; rm -rf .chain_lock.d`). Never commit the lock dir. If a lock is older than 3 hours and its owner's output is not progressing, you may break it (note it in your report).
- LOCAL CPU: process-level parallelism inside your scripts is encouraged, but cap it at 4 worker processes per agent locally; anything heavier goes to Hoffman2 (light) or CARC (big fleets).
- LOCAL MEMORY: non-chain pandas loads (per-bar design, yhat parquets, daily tables) are fine.

## Files you may touch
- Your own new scripts under `experiments/` or `writeup/` and your own results dir (named in your task). Prefix new script names with your task's tag so nothing collides. The other Claude session commits studies 84-90 under `writeup/intraday_proposals/NN_*.py` — do not take numbers there.
- Do NOT edit `notebooks/_write_0dte_nb.py`, `notebooks/_write_0dte_intraday_T_nb.py`, `notebooks/atm_straddle_lib.py`, `writeup/main.tex`, `writeup/sections/*`, `experiments/score_*`/`reduce_*`/`check_*`/`compare_*` scripts, or `specs/*` that CARC is running — unless your task says you own that file. Integration into the deck/paper is a later sequential step; write standalone outputs (CSV + a short `.md` and/or standalone PDF via a `make_*_tex.py` you own).
- `writeup/PROGRESS_2026-09-29.md`: update ONLY your own lines (Edit tool with the exact current line; if it fails because another agent changed the file, re-read and retry) and append ONE line to the Agent log at the end when you finish: `- <time> W# <one-line result, key numbers> (<commit sha>)`.

## Git protocol
- Stage by explicit path only (`git add <files>`); NEVER `git add -A`/`-u`/globs. Never stage another workstream's files, `*.npz`, logs, per-arm result folders, `writeup/main.pdf`, LaTeX aux, `results/spxw_pnl/smile_rv_daily.parquet`, `data/*.parquet`, `.chain_lock.d`. Results CSVs that are small (< ~2 MB each) and summary tables ARE committed; big parquets are not.
- Before committing: `python.exe -m ruff check --fix`, `python.exe -m ruff format`, `python.exe -m mypy --ignore-missing-imports` on YOUR staged .py files only (never on shipped specs/scorers — keep committed == shipped).
- Commit with `git -c user.name=jamesdchen -c user.email=bandits.alg@gmail.com commit -F <msgfile>`; the message body ends with the two trailer lines
  `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01NmfcxEZn9c9XdZEPfGfRGm`.
  If `.git/index.lock` exists, another agent is committing: wait 20 s and retry (up to 10 times). Commit once at the end (plus at most one intermediate checkpoint). Do NOT push.

## Research conventions (professor-facing; from the handoff)
- Wording: **sign(s)** rule (never "long-short volatility"); **straddle** (nearest OTM call + nearest OTM put, same-day expiry, one position) — never "package"; **last-30-min trade** — never "paper trade"; no lab/cluster names in prose; features by design column names (`har_ma_*`, `adj_vix_ma_*`, …) with a key; park rather than delete; every number in prose from executed output; no magic numbers (named, justified constants; no ad-hoc clips/floors).
- Panel stamps: naive ET, bar-END labelled (row 16:00 = the 15:30–16:00 bar; the forecast is issued at 15:30). Chain/spot stamps are true UTC. Nothing issued after the entry time may enter a forecast or a sizing rule.
- Same 866 days / same rows for every comparison; paired bootstrap intervals for differences; report intervals, not just point estimates.
- TWO SCORERS, TWO SHARPES: the rv_iv notebook recalibrates on all 13 session bars; the research scorers (`experiments/compare_mfiv_harlag.py`, `score_trees_subsection.py`) recalibrate the 16:00 bar alone. Never mix them in one table — say which one you use; the RESEARCH scorer (16:00-bar recalibration) is the default for the closing-strategy master table.
- Stored per-bar ALL-FEATURES forecasts have rare extreme values in the plain back-transform (16:00 ridge QLIKE 2.65 unrecalibrated) — handle explicitly and say how.

## Your final report (returned to the orchestrator, who relays it)
Under ~40 lines: what was done, the key numbers (with intervals), file paths of outputs, the commit sha, what was NOT done and why, and anything the next step must know. Only facts from executed output.
