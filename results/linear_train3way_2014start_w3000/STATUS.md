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
