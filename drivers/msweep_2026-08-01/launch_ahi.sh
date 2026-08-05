#!/usr/bin/env bash
cd "/c/Users/james/CC Allowed/harxhar-clean"
SP="/c/Users/james/AppData/Local/Temp/claude/C--Users-james-CC-Allowed-harxhar-clean/cf56a2a1-b85d-4e9e-b5e5-f1071299e098/scratchpad"
PY=/c/Users/james/miniconda3/envs/285J/python.exe
export OMP_NUM_THREADS=2 PYTHONPATH="/c/Users/james/CC Allowed/harxhar-clean"
( PB_START=26189 PB_END=28378 PB_ARMS="p32a3_t2:a3x,p32a10_t2:a10x" "$PY" "$SP/parsimony_battery.py" > logs/pb_ahi_t2.log 2>&1 && echo DONE > logs/ahi2_done.marker ) &
( PB_ARMS="p32a10_t1:a10x" "$PY" "$SP/parsimony_battery.py" > logs/pb_ahi_t1.log 2>&1 && echo DONE > logs/ahi1_done.marker ) &
wait
