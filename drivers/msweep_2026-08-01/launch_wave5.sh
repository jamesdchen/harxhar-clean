#!/usr/bin/env bash
cd "/c/Users/james/CC Allowed/harxhar-clean"
SP="/c/Users/james/AppData/Local/Temp/claude/C--Users-james-CC-Allowed-harxhar-clean/cf56a2a1-b85d-4e9e-b5e5-f1071299e098/scratchpad"
PY=/c/Users/james/miniconda3/envs/285J/python.exe
export OMP_NUM_THREADS=2 PYTHONPATH="/c/Users/james/CC Allowed/harxhar-clean"

( "$PY" "$SP/rawc_corr.py" > logs/rawc_corr.log 2>&1 && echo DONE > logs/rc_done.marker ) &
( PB_ARMS="rawc_a03:rawc03,rawc_a01:rawc01" "$PY" "$SP/parsimony_battery.py" > logs/pb_rawc_alpha.log 2>&1 && echo DONE > logs/pbra_done.marker ) &
( PB_GATES=1 PB_ARMS="rawc_gates:rawc" "$PY" "$SP/parsimony_battery.py" > logs/pb_rawc_gates.log 2>&1 && echo DONE > logs/pbrg_done.marker ) &
( "$PY" "$SP/pcovr_loadings80.py" > logs/pcovr_loadings80.log 2>&1 && echo DONE > logs/pl80_done.marker ) &
( H=8 TAG=pt_h8 "$PY" "$SP/product_tune.py" > logs/product_tune_h8.log 2>&1 && echo DONE > logs/pt8_done.marker ) &
wait
