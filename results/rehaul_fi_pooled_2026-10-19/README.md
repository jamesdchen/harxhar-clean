# Pooled intraday FI (note E19): status

- 13/ (13 regular-hours bars): done 2026-10-19 23:30 UTC on the corrected rerun (base-5 HAR ladder, FOMC in;
  penalties of branch insurance-selling-2026-09-19-pooled-linear-13 at f30ad70). See 13/SUMMARY.md.
- 48/ (all 48 intervals): waiting for results/pooled_linear_rerun_2026-10-19_48/frozen_penalties.json on branch
  insurance-selling-2026-09-19-pooled-linear-48.
- Withdrawn: the first 13-bar results (commit 2dbb065) used the base-2, no-FOMC penalty file of 219ec7d, declared
  invalid by the author's main session. experiments/rehaul_fi_pooled_compute.py refuses a penalty file whose
  all-features column list does not have exactly the base-5 HAR components and the four FOMC series (check_landed).
