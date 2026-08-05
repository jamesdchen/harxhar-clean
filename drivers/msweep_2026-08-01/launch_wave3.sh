#!/usr/bin/env bash
cd "/c/Users/james/CC Allowed/harxhar-clean"
SP="/c/Users/james/AppData/Local/Temp/claude/C--Users-james-CC-Allowed-harxhar-clean/cf56a2a1-b85d-4e9e-b5e5-f1071299e098/scratchpad"
PY=/c/Users/james/miniconda3/envs/285J/python.exe
export OMP_NUM_THREADS=2 PYTHONPATH="/c/Users/james/CC Allowed/harxhar-clean"

( H=48 DTE_LO=1 DTE_HI=2 DTE_TARGET=1 TAG=v3h48 "$PY" "$SP/straddle_v3.py" > logs/straddle_v3_h48.log 2>&1 && echo DONE > logs/v348_done.marker ) &
( H=240 DTE_LO=4 DTE_HI=9 DTE_TARGET=7 TAG=v3h240 "$PY" "$SP/straddle_v3.py" > logs/straddle_v3_h240.log 2>&1 && echo DONE > logs/v3240_done.marker ) &
( H=240 DTE_LO=4 DTE_HI=9 DTE_TARGET=7 TAG=ha240 "$PY" "$SP/straddle_alpha.py" > logs/straddle_alpha_h240.log 2>&1 && echo DONE > logs/sa240_done.marker ) &
wait
