# Three-way training-sample comparison, pooled linear models (status)

Rule: `RULE.txt` (commit 419e7ad, 21:38:55 PDT, before any job). Code: `experiments/linear_train3way.py` (orchestration
per arm on the template's orchestrator), `experiments/linear_train3way_runner.py` (the runner plus the close samples
and `dates_all.csv`), `experiments/linear_train3way_score.py` (CARC scoring), `experiments/linear_train3way_econ.py`
(local close economics), `code/stage3.sh`, `code/submit3.sh`. Staging: `/scratch1/jc_905/train3way` on CARC Discovery
(code = 419e7ad + the score script of 34a2edd; the wsweep venv read only).

| Time (PDT, 2026-10-23) | Event |
|---|---|
| 21:38 | RULE.txt committed (419e7ad) and pushed; no job submitted before it |
| 21:39 | primary chain submitted: RTH13 design 13361320, RTH13 pretune/select/gate 13361321, CLOSE derive/pretune/select 13361322, ALL48 design 13361323, ALL48 pretune/select 13361324, CLOSE_NAT design 13361325, CLOSE_NAT pretune/select 13361326, walks ALL48 13361327 (30 tasks), CLOSE 13361328 (7), CLOSE_NAT 13361329 (7) |
| 21:41 | CLOSE_NAT design done (43 s; 6,462 closing rows from 1998-06-04) |
| 21:46 | RTH13 GATE PASS (`RTH13/gate.json`): (i) both design caches md5-identical to w3000; (ii) penalties and pretune_mse.csv identical; (iii) 50-origin walk slices of all-features ridge / lasso / elastic net and the OLS identical (date, true_adj, pred_adj); (iv) CARC w3000/forecasts.parquet md5 4e5c7c17... = the repo's |
| 21:45-21:47 | CLOSE (derived from the rebuilt RTH13 design) and CLOSE_NAT: pretune, select, walks done (each walk task < 30 s) |
| 21:47 | the local background poll was stopped by Claude Code (system critically low on memory); not restarted, per the harness's instruction |
| 21:49 | secondary + scoring chain submitted: select16 ALL48 13361440 (after 13361324), RTH13 13361441; walk16 ALL48 13361442 (29), RTH13 13361443 (13); export 13361444 (after every walk); score 13361445 |
| 21:49 | RTH13@16 selected: lasso HAR 1e-3 (was 1e-8), lasso full 1e-3 (was 1e-4), elastic net HAR 1e-2 (was 1e-7); ridge and elastic net full unchanged |
| 21:49 | ALL48 design 13361323 pending (Priority); every ALL48, export and score job waits on it |
| 21:52 | RULE.txt erratum: header time stamp corrected (no rule changed) |

Frozen penalties so far (fold 3; grid-rule extensions in each `frozen_penalties.json`):

| Model | RTH13 | CLOSE | CLOSE_NAT | RTH13@16 |
|---|---|---|---|---|
| Ridge HAR | 1e-3 | 1e-3 | 1e-3 | 1e-3 |
| Ridge full | 1e3 | 1e3 | 1e3 | 1e3 |
| Lasso HAR | 1e-8 | 1e-4 | 1e-7 | 1e-3 |
| Lasso full | 1e-4 | 1e-2 | 1e-2 | 1e-3 |
| Elastic net HAR | 1e-7 | 1e-7 | 1e-7 | 1e-2 |
| Elastic net full | 1e-3 | 1e-2 | 1e-2 | 1e-3 |

CLOSE / CLOSE_NAT fold 3: fit 2000-11-06 .. 2012-11-19 (3,000 closes), validation 2012-12-31 .. 2013-12-31 (250
closes; 2013 has 249 regular sessions).

To finish (after score 13361445): pull `scores/` (tables, csv, f16.parquet) and the small run files; run
`experiments/linear_train3way_econ.py <main checkout> scores/f16.parquet econ/` locally; sacct CPU-hours.

## Addendum A: equal training rows (RULE.txt addendum A, commit b8e0a5e, 22:10 PDT, before its jobs)

| Time (PDT, 2026-10-23) | Event |
|---|---|
| 21:50-21:55 | RTH13@16 walks 13361443 done (6 jobs: lasso HAR, enet HAR, lasso full in 4 chunks; every task COMPLETED 0:0) |
| ~22:05 | ALL48 design 13361323 started (b10-03) |
| 22:16 | `scancel 13361444 13361445` (the first chain's export / score) denied by the auto-mode classifier; not retried. They run as submitted (equal-calendar scores to `scores/`); the addendum's export / score wait for them (afterany) and write `scores_all/` |
| 22:20 | code on CARC: linear_train3way{,_runner,_score}.py of 06d5871 (additive: new arms, derive_window, --window-rows, the `both` scoring switch; no equal-calendar computation changed) |
| 22:21 | submitted: RTH13_N3k derive/pretune/select 13364083; ALL48_N3k + ALL48_N39k derive/pretune/select 13364084 (after the ALL48 design 13361323); walks RTH13_N3k 13364085 (8), ALL48_N3k 13364086 (14), ALL48_N39k 13364087 (22) |
| 22:23 | export2 13364094 (afterok 13361327, 13361442, 13364085-87; afterany 13361445) and SCORE2 13364095 (both designs, `scores_all/`) - the last job of both chains |

## Addendum B: scored-only refits in the 48-interval arms (RULE.txt addendum B, commit 8225da2, before any affected walk)

| Time (PDT, 2026-10-23) | Event |
|---|---|
| 22:30-22:35 | RTH13_N3k derive/pretune/select 13364083 and walks 13364085 (8 tasks) done, every task COMPLETED |
| 22:46 | ALL48 design 13361323 done (3,111 s; all-features design in one pass) |
| 22:48 | owner-approved scancel of the pending ALL48 walks 13361327, 13361442, 13364086, 13364087 and the dependent export / score jobs 13361444, 13361445, 13364094, 13364095 (none had started) |
| 22:50 | synthetic check on CARC (srun 13366875, 900 rows of a 48-row-a-day cache): OLS and ridge bit-identical, lasso / elastic net <= 3e-14 relative, 18 warm re-solves all by the active-set repair, KKT clean |
| 22:55 | code on CARC: linear_train3way{,_runner}.py of 8225da2 (the scored-only refit, gate_refit), linear_train3way_score.py of 06d5871 |
| 22:56 | submitted: gate_refit 13366980 (after ALL48 pretune/select 13361324); walks ALL48 13366981 (22 tasks), ALL48@16 13366982 (21; after select16 13361440), ALL48_N3k 13366983 (9) and ALL48_N39k 13366984 (13) (after 13364084); every walk afterok on the gate; export3 13366985; SCORE3 13366986 = the last job (both designs, scores_all/) |

## Addendum B gate: FAILED (genuine mismatch, not infrastructure); ALL48 walks held

Gate 13366980 ran to completion (23:02-23:04 PDT, 2 min 13 s, MaxRSS 12 GB, exit 3 = the gate's own failure; every
file under /scratch1, nothing under /home1). `ALL48/gate_refit.json`: 104 scored origins of the 400-row slice.

| Model | Max rel diff (scored vs full) | Tolerance | KKT (both walks) | Pass |
|---|---|---|---|---|
| OLS, ridge HAR, ridge full | 0 (bit-identical) | 1e-9 | - | yes |
| lasso HAR, enet HAR | 4.9e-16, 6.6e-16 | 1e-6 | clean | yes |
| lasso full | 1.55e-5 (45 of 104 origins > 1e-6) | 1e-6 | residual <= 1.9e-14 relative at every solve | NO |
| enet full | 8.9e-6 (33 of 104 origins > 1e-6) | 1e-6 | residual <= 1.1e-14 relative at every solve | NO |

Diagnosis: the differences sit exactly on the stretches where the two walks' last cold seed differs. Under
STANDARDIZE_PENALTY the lasso / elastic-net scales (m, s) are recomputed at every cold seed and held until the next;
WALK_KKT_RESEED's KKT-triggered reseeds happened at different rows in the two walks (full: lasso rows 238, 341, enet
100, 104 - checks at every row, overnight included; scored-only: lasso 82, 282, enet 286 - checks at scored rows only,
after a different warm path), so between such a reseed and the next common seed (the 250-row tune) the two walks
solve differently weighted problems.  Where the seed history is the same, they agree to <= 1.5e-14 (lasso rows 0-82
and 250-282, enet 0-100 and 250-286); each solution is KKT-exact for its own problem.  The multi-row overnight update
and the warm re-solve are not the source (every re-solve by the active-set repair, KKT clean).
Cost on the slice (CPU s, full / scored-only): ridge full 25.6 / 25.5, lasso full 126.2 / 126.5, enet full 69.0 / 53.3 -
the 144,000-row cold seeds (tune every 250 rows, KKT reseeds) dominate, so scored-only saves little.
The held walks 13366981-13366984, export3 13366985 and score3 13366986 stay pending (afterok on the failed gate:
they cannot start).  Nothing loosened; the owner decides.

## Note to addendum B: scored-only refits withdrawn (owner option A, full refit)

| Time (PDT, 2026-10-23) | Event |
|---|---|
| ~23:40 | owner (relayed): option A - every ALL48 walk refits at every row (RULE section 5), no new gate |
| 23:35-23:40 | RULE.txt note to addendum B and this entry committed before any new walk; experiments/linear_train3way.py no longer sets --refit-rows |
| 23:42 | code on CARC: linear_train3way.py of 779aee1 (no --refit-rows), runner of 8225da2, score of 06d5871; scancel of the held 13366981-13366986 (owner-approved) |
| 23:43 | submitted, full refit at every row: walks ALL48 13367775 (19 tasks: ridge 2, lasso 8, enet 5 chunks), ALL48@16 13367776 (18; changed models: ridge full, lasso HAR + full, enet HAR), ALL48_N3k 13367777 (8), ALL48_N39k 13367778 (10); export4 13367779; SCORE4 13367780 = the last job (both designs, scores_all/) |

## Finer splits (coordinator 23:36 PDT: use the CARC resources; split only where bit-identical)

At 23:35 the ALL48 walks 13367775 (tasks 1-8 running) and ALL48_N3k 13367777 (1-7 running) had started: their splits
are kept (19 / 8 tasks).  ALL48@16 13367776 and ALL48_N39k 13367778 had not started: cancelled with export4 13367779 and
score4 13367780 and resubmitted finer (ALL48@16: ridge full 4, lasso full 15 chunks, 21 tasks; ALL48_N39k: ridge 2,
lasso 5, enet 3 chunks, 14 tasks).  Exactness: every chunk starts on the walk's own cold-seed row (every 250th solve
from the walk start, asserted by the runner: start - halo is a re-choice boundary) with a W-row halo, so its first
solve is the tune (identifiability mask + cold seed of the same window) the long walk does there, and the trackers,
standardization scales and OLS sums restart from that window in both.  Checked bit for bit by gate_seam (ALL48,
3,000-day penalties: full-set ridge, lasso, elastic net and the OLS; rows [S0, S0 + 500) as one walk against a chunk
starting at S0 + 250: date, true_adj and pred_adj strings must be identical), on which the re-split walks run afterok.
| Time (PDT, 2026-10-23) | Event |
|---|---|
| 23:52 | by submission time ALL48_N39k 13367778 had started (2 tasks running): kept as is (10 tasks), like ALL48 13367775 and ALL48_N3k 13367777.  Only ALL48@16 re-split: 13367776, export4 13367779, score4 13367780 cancelled (not started); submitted gate_seam 13367866, ALL48@16 walks 13367867 (21 tasks, afterok gate_seam), export5 13367868 (afterok 13367775, 13367777, 13367778, 13367867), SCORE5 13367869 = the last job (both designs, scores_all/) |
