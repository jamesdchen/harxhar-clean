# Pooled intraday FI (note E19): lasso and ridge, monthly re-estimation, both E15 samples

- 48/ (all 48 intervals, W = 96,000): penalties of branch insurance-selling-2026-09-19-pooled-linear-48 at 8dace13.
  See 48/SUMMARY.md, which also compares the two samples' rankings (compare.csv, compare.json here).
- 13/ (13 regular-hours bars, W = 26,000): penalties of branch insurance-selling-2026-09-19-pooled-linear-13 at
  f30ad70. See 13/SUMMARY.md.
- Both: base-5 HAR ladder, the four FOMC series in, E6 and E9 flags on; 52 re-estimations, 3 January 2020 ..
  1 April 2024; the design caches were built with each branch's own runner (not committed).
- Withdrawn: the first 13-bar results (commit 2dbb065) used the base-2, no-FOMC penalty file of 219ec7d, declared
  invalid by the author's main session. experiments/rehaul_fi_pooled_compute.py refuses a penalty file whose
  all-features column list does not have exactly the base-5 HAR components and the four FOMC series (check_landed).
- Regenerate: python experiments/rehaul_fi_pooled_compute.py {linear,gate,reduce} ... (see each SUMMARY.md), then
  compare --out results/rehaul_fi_pooled_2026-10-19, then python writeup/make_rehaul_fi_pooled.py.
