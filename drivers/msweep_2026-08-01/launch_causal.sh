#!/usr/bin/env bash
cd "/c/Users/james/CC Allowed/harxhar-clean"
SP="/c/Users/james/AppData/Local/Temp/claude/C--Users-james-CC-Allowed-harxhar-clean/cf56a2a1-b85d-4e9e-b5e5-f1071299e098/scratchpad"
PY=/c/Users/james/miniconda3/envs/285J/python.exe
export OMP_NUM_THREADS=2 PYTHONPATH="/c/Users/james/CC Allowed/harxhar-clean"

( PB_START=24000 PB_END=26189 TAG=crawc_t1 "$PY" "$SP/causal_rawc.py" > logs/crawc_t1.log 2>&1 && echo DONE > logs/crawc1_done.marker ) &
( PB_START=26189 PB_END=28378 TAG=crawc_t2 "$PY" "$SP/causal_rawc.py" > logs/crawc_t2.log 2>&1 && echo DONE > logs/crawc2_done.marker ) &
wait
