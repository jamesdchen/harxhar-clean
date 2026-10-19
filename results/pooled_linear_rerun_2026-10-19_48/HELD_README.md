# HELD (2026-10-19 23:55 UTC)

The files here come from the brief's design (base-2 HAR ladder, FOMC excluded; 1,051 columns on all_features).
The main session relayed author decisions at 23:50: base-5 ladder, FOMC included (577 columns expected).
The switch is waiting for the author's confirmation. So the penalties are stored as
`HELD_base2_frozen_penalties.json`, NOT as `frozen_penalties.json`, which the variable-importance session polls for.
`sched/` and `cluster/pooled_linear_rerun_48_tasks.txt` hold the same base-2 values; do not submit them.
