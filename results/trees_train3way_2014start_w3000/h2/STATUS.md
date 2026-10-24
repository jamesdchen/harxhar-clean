# Trees' three-way comparison on Hoffman2 (status)

Assignment (CARC `runs/tasks/h2_assignment.txt`, RULE.txt addendum C): `ASSIGNED w3rf_full.txt 240-368 ALL48 rf_skx
all_features 20` (129 lines, the ALL48 forest's full set).  Owner, 02:40 PDT (relayed): lines go to Hoffman2 in waves
sized to the slots it actually gets; only started lines are Hoffman2's; the assignment is re-read before each wave.
Scripts: `code/` (staging `/u/scratch/j/jamesdc1/t3t` on Hoffman2, user jamesdc1).

| Time (PDT) | Event |
|---|---|
| 02:04-02:07 | transfer 1, CARC -> laptop pipe -> Hoffman2 dtn2 (one SSH session each side, one tar stream): 97 files, 1,812,282,101 bytes in 192 s (9.4 MB/s): the RTH13 and ALL48 design caches (both sets), runs/tasks, timing, CARC's forest gate pieces (gate/g1, gate/g2) and their jsons; manifest `h2/xfer1/MANIFEST.md5` (md5 7371ffd310d137a3e3c9df2f5aca4bc5 on both ends), `md5sum -c` OK on Hoffman2, each design dir's own MD5SUMS OK |
| 02:14 | code = `git archive` of d9d01fd (the commit in CARC's `runs/tasks/h2_code_commit.txt`; LF, no line-ending conversion) at `$B/code`: md5-identical to CARC's `code_d9d01fd` for trees_train3way.py, pooled_trees_rerun_13.py, data/core_stats.parquet, both templates' forecasts.parquet and the three configs |
| 02:14 | `code/submit_h2.sh`: env 15261016; gates per CPU tag (forest only, the assigned class): gold6140 15261017 / 15261018 / 15261019, gold6240 15261020-22, gold6342 15261023-25, e5v4 (E5-2650 v4) 15261026-28 (`t3t_g20L_<tag>` ALL48 full set every row 20 threads, `t3t_g20_<tag>` RTH13 --ends 10 at 20 threads + ALL48 scored-only + HAR every row, `t3t_g8_<tag>` RTH13 --ends 10 at 8 threads + ALL48_N3k every row / scored); check `t3t_gsum` 15261029 (accept + `gate_summary.py`) |

Environment (CARC, `env_carc.txt` of transfer 1): harxhar-r13 = python 3.12.15, numpy 1.26.4 (PyPI wheel, openblas64
0.3.23.dev), pandas 2.3.3, scipy 1.17.1, scikit-learn 1.9.1, lightgbm 4.6.0, xgboost 3.2.0, pyarrow 19.0.1, numba
0.68.0; venv_skx adds daal 2026.2.0, scikit-learn-intelex 2026.2.0, tbb 2023.2.0, tcmlib 1.6.0.  Hoffman2: the same
pins (`req_gb.txt`, `req_skx.txt` derived from CARC's venv_skx listing) in uv venvs (CPython 3.12.3) inside the Debian
bookworm container of the tree learning curve; threads = slots; numpy AVX-512 dispatch off; OpenBLAS Haswell.
