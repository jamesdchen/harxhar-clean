#!/usr/bin/env bash
cd "/c/Users/james/CC Allowed/harxhar-clean"
SP="/c/Users/james/AppData/Local/Temp/claude/C--Users-james-CC-Allowed-harxhar-clean/cf56a2a1-b85d-4e9e-b5e5-f1071299e098/scratchpad"
PY=/c/Users/james/miniconda3/envs/285J/python.exe
export OMP_NUM_THREADS=3 PYTHONPATH="/c/Users/james/CC Allowed/harxhar-clean"
( H=4 TAG=pt_h4 "$PY" "$SP/product_tune.py" > logs/product_tune_h4.log 2>&1 && echo DONE > logs/pt4_done.marker ) &
( H=16 TAG=pt_h16 "$PY" "$SP/product_tune.py" > logs/product_tune_h16.log 2>&1 && echo DONE > logs/pt16_done.marker ) &
wait
