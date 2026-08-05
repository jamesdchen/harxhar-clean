#!/usr/bin/env bash
cd "/c/Users/james/CC Allowed/harxhar-clean"
SP="/c/Users/james/AppData/Local/Temp/claude/C--Users-james-CC-Allowed-harxhar-clean/cf56a2a1-b85d-4e9e-b5e5-f1071299e098/scratchpad"
PY=/c/Users/james/miniconda3/envs/285J/python.exe
export OMP_NUM_THREADS=2 PYTHONPATH="/c/Users/james/CC Allowed/harxhar-clean"
( H=8 TAG=hint_h8 "$PY" "$SP/hybrid_intraday.py" > logs/hint_h8.log 2>&1 && echo DONE > logs/hint8_done.marker ) &
( H=8 USE_ANN=1 TAG=hint_h8ann "$PY" "$SP/hybrid_intraday.py" > logs/hint_h8ann.log 2>&1 && echo DONE > logs/hint8a_done.marker ) &
( H=16 TAG=hint_h16 "$PY" "$SP/hybrid_intraday.py" > logs/hint_h16.log 2>&1 && echo DONE > logs/hint16_done.marker ) &
( H=4 TAG=hint_h4 "$PY" "$SP/hybrid_intraday.py" > logs/hint_h4.log 2>&1 && echo DONE > logs/hint4_done.marker ) &
wait
