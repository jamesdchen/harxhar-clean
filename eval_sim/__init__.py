"""eval_sim -- Monte Carlo first-passage simulator for prop-trading evaluations (trailing drawdown).

Library-first, execution-decoupled. config (rules) + pnl (sources) + sim_core (engine) are pure; grid +
hpc_worker + aggregate are the scheduler-agnostic HPC layer. Movable: numpy core, pandas/pyarrow only
in the HPC layer.
"""
