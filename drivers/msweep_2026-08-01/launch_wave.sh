#!/usr/bin/env bash
cd "/c/Users/james/CC Allowed/harxhar-clean"
SP="/c/Users/james/AppData/Local/Temp/claude/C--Users-james-CC-Allowed-harxhar-clean/cf56a2a1-b85d-4e9e-b5e5-f1071299e098/scratchpad"
PY=/c/Users/james/miniconda3/envs/285J/python.exe
export OMP_NUM_THREADS=2 PYTHONPATH="/c/Users/james/CC Allowed/harxhar-clean"

( H=48 DTE_LO=1 DTE_HI=2 DTE_TARGET=1 TAG=h48 "$PY" "$SP/straddle_test.py" > logs/straddle_h48.log 2>&1 && echo DONE > logs/straddle_h48_done.marker ) &
( H=240 DTE_LO=4 DTE_HI=9 DTE_TARGET=7 TAG=h240 "$PY" "$SP/straddle_test.py" > logs/straddle_h240.log 2>&1 && echo DONE > logs/straddle_h240_done.marker ) &
( H=8 TAG=ann_h8 "$PY" "$SP/ann_shape.py" > logs/ann_h8.log 2>&1 && echo DONE > logs/ann_h8_done.marker ) &
( H=16 TAG=ann_h16 "$PY" "$SP/ann_shape.py" > logs/ann_h16.log 2>&1 && echo DONE > logs/ann_h16_done.marker ) &
( "$PY" "$SP/parsimony_battery.py" > logs/parsimony_battery.log 2>&1 && echo DONE > logs/pb_done.marker ) &
( "$PY" "$SP/hybrid_corr.py" > logs/hybrid_corr.log 2>&1 && echo DONE > logs/hc_done.marker ) &
wait
