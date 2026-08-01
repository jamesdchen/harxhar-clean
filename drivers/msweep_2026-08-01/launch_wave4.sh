#!/usr/bin/env bash
cd "/c/Users/james/CC Allowed/harxhar-clean"
SP="/c/Users/james/AppData/Local/Temp/claude/C--Users-james-CC-Allowed-harxhar-clean/cf56a2a1-b85d-4e9e-b5e5-f1071299e098/scratchpad"
PY=/c/Users/james/miniconda3/envs/285J/python.exe
export OMP_NUM_THREADS=2 PYTHONPATH="/c/Users/james/CC Allowed/harxhar-clean"

( ALPHA_SWEEP=1 ALPHAS="10000,30000,100000,300000,1000000" H=8 TAG=aswx_h8 "$PY" "$SP/ann_shape.py" > logs/aswx_h8.log 2>&1 && echo DONE > logs/aswx8_done.marker ) &
( ALPHA_SWEEP=1 ALPHAS="1,100,1000,10000,100000" H=16 TAG=aswx_h16 "$PY" "$SP/ann_shape.py" > logs/aswx_h16.log 2>&1 && echo DONE > logs/aswx16_done.marker ) &
( ALPHA_SWEEP=1 ALPHAS="1,100,1000,10000,100000" H=4 TAG=aswx_h4 "$PY" "$SP/ann_shape.py" > logs/aswx_h4.log 2>&1 && echo DONE > logs/aswx4_done.marker ) &
( PB_START=26189 PB_END=28378 PB_ARMS="t2_prune32:pramp,t2_rawc:rawc" "$PY" "$SP/parsimony_battery.py" > logs/pb_tile2.log 2>&1 && echo DONE > logs/pbt2_done.marker ) &
( PB_CELLS="sumautocov" PB_ARMS="rawc_autocov:rawc" "$PY" "$SP/parsimony_battery.py" > logs/pb_autocov.log 2>&1 && echo DONE > logs/pbac_done.marker ) &
wait
