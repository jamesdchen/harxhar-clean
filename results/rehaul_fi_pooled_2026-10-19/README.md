# Pooled intraday FI (note E19): status

- 13/ (13 regular-hours bars): done 2026-10-19 23:30 UTC on the corrected rerun (base-5 HAR ladder, FOMC in;
  penalties of branch insurance-selling-2026-09-19-pooled-linear-13 at f30ad70). See 13/SUMMARY.md.
- 48/ (all 48 intervals): waiting for results/pooled_linear_rerun_2026-10-19_48/frozen_penalties.json on branch
  insurance-selling-2026-09-19-pooled-linear-48.
- Withdrawn: the first 13-bar results (commit 2dbb065) used the base-2, no-FOMC penalty file of 219ec7d, declared
  invalid by the author's main session. experiments/rehaul_fi_pooled_compute.py refuses a penalty file whose
  all-features column list does not have exactly the base-5 HAR components and the four FOMC series (check_landed).
- 2026-10-20 01:56 UTC, 48 still waiting: the 48 branch holds only base-2 pre-tune outputs (HELD_README.md), pending
  the author's base-5 / FOMC confirmation. Dry run, so the sample is ready when its file lands: a 48-interval base-5
  design with FOMC built with the 13 branch's runner (--sample 48 --har-base 5, DROP_ACTIVITY_MA,
  CALENDAR_THROUGH_OVERNIGHT, LEAN_GLOBAL_LAGS): 300,317 x 577, 20 minutes, peak memory about 6.5 GB of 15; three
  monthly fits with placeholder penalties (ridge 1e3, lasso 1e-2) took 9 s each, so the full 52 months plus probes
  project to about 10 minutes. The dry run exposed that weekend-dated overnight rows made a Sunday the "first day" of
  some months; anchors are now the first row of the month's first date with regular-hours bars (the 13-bar anchors
  and blocks are unchanged by this).
