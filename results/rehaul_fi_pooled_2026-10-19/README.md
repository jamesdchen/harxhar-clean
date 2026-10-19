# Pooled intraday FI (note E19): status

- 2026-10-19 23:05 UTC: the 13-bar results first committed here (commit 2dbb065) are WITHDRAWN. They used
  results/pooled_linear_rerun_2026-10-19_13/frozen_penalties.json at commit 219ec7d of branch
  insurance-selling-2026-09-19-pooled-linear-13, which the author's main session declared invalid (picked on a base-2
  design without the FOMC series). The pooled designs are base 5 (har_ma_{1,5,25,125,625,3125}) and include the four
  FOMC series.
- experiments/rehaul_fi_pooled_compute.py now refuses a penalty file whose all-features column list does not have
  exactly the base-5 HAR components and the four FOMC series (check_landed).
- Waiting for the replaced frozen_penalties.json of both reruns (48 and 13); each sample's results go to 48/ and 13/.
