#!/usr/bin/env bash
cd "/c/Users/james/CC Allowed/harxhar-clean"
SP="/c/Users/james/AppData/Local/Temp/claude/C--Users-james-CC-Allowed-harxhar-clean/cf56a2a1-b85d-4e9e-b5e5-f1071299e098/scratchpad"
PY=/c/Users/james/miniconda3/envs/285J/python.exe
export OMP_NUM_THREADS=2 PYTHONPATH="/c/Users/james/CC Allowed/harxhar-clean"
( PB_START=24000 PB_END=26189 TAG=tcorr_t1 "$PY" "$SP/tile_corr.py" > logs/tcorr_t1.log 2>&1 && echo DONE > logs/tc1_done.marker ) &
( PB_START=26189 PB_END=28378 TAG=tcorr_t2 "$PY" "$SP/tile_corr.py" > logs/tcorr_t2.log 2>&1 && echo DONE > logs/tc2_done.marker ) &
wait
