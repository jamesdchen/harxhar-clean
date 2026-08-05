#!/usr/bin/env bash
cd "/c/Users/james/CC Allowed/harxhar-clean"
SP="/c/Users/james/AppData/Local/Temp/claude/C--Users-james-CC-Allowed-harxhar-clean/cf56a2a1-b85d-4e9e-b5e5-f1071299e098/scratchpad"
PY=/c/Users/james/miniconda3/envs/285J/python.exe
export OMP_NUM_THREADS=2 PYTHONPATH="/c/Users/james/CC Allowed/harxhar-clean"
( PB_ARMS="tex_s01_t1:tex0.1,tex_s003_t1:tex0.0316" "$PY" "$SP/parsimony_battery.py" > logs/pb_tex_t1.log 2>&1 && echo DONE > logs/tex1_done.marker ) &
( PB_START=26189 PB_END=28378 PB_ARMS="tex_s01_t2:tex0.1,tex_s003_t2:tex0.0316" "$PY" "$SP/parsimony_battery.py" > logs/pb_tex_t2.log 2>&1 && echo DONE > logs/tex2_done.marker ) &
wait
