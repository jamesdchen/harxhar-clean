#!/usr/bin/env bash
cd "/c/Users/james/CC Allowed/harxhar-clean"
SP="/c/Users/james/AppData/Local/Temp/claude/C--Users-james-CC-Allowed-harxhar-clean/cf56a2a1-b85d-4e9e-b5e5-f1071299e098/scratchpad"
PY=/c/Users/james/miniconda3/envs/285J/python.exe
export OMP_NUM_THREADS=3 PYTHONPATH="/c/Users/james/CC Allowed/harxhar-clean"

( H=4 TAG=hon_h4 "$PY" "$SP/ann_shape.py" > logs/hon_h4.log 2>&1 && echo DONE > logs/hon4_done.marker ) &
( H=8 TAG=hon_h8 "$PY" "$SP/ann_shape.py" > logs/hon_h8.log 2>&1 && echo DONE > logs/hon8_done.marker ) &
( H=16 TAG=hon_h16 "$PY" "$SP/ann_shape.py" > logs/hon_h16.log 2>&1 && echo DONE > logs/hon16_done.marker ) &
wait
