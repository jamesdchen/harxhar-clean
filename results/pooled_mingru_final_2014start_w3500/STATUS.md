# Final pooled ridge-init MinGRU at W 3,500 (first day 2014-01-02, fold 3): status log

Owner, 2026-10-22 (relayed): "Test mingru 3.5k as well". Rule: `RULE_gru.txt` (the W = 3,000 rule and its addendum of
2026-10-22 21:57 PDT: the same rule, tuned separately at W = 3,500). Code as W = 3,000
(`experiments/pooled_mingru_final_2014_w3000.py` with PMF_WINDOW_DAYS=3500). CARC root `/home1/jc_905/pmf35`. The gate
(previous w1000 configurations on the previous design) is shared with W = 3,000: jobs 13319779 and 13319780.

## Log
- 2026-10-22 22:01 PDT: CPU prep 13320226 (export of the W = 3,500 design caches, plan, fold-3 inits, round-0 tune
  jobs, canary).
- 2026-10-22 21:58 PDT: prep 13320226 COMPLETED (d11-20): export 84,747 x 12 / x 564 (dates of rows >= 45,500 equal
  the previous design's); plan: fold 3 fit [1,702, 47,202), validation [47,527, 50,777) (2013-01-02 15:30 ..
  2013-12-31 16:00); 2,643 daily origins (as W = 3,000); fold-3 keep 295 of 564 (full), 12 of 12 (HAR); round 0
  6,480 tune jobs a set in 19 files a set.
- 2026-10-22 22:00 PDT: canary 13320230 (tune) and 13320231 (refit), POST = check-run; round-0 tune jobs
  13320232-13320269 held afterok on the gate (13319779, 13319780) and these two canaries; every W = 3,500 GPU job
  --nice=1000 (the W = 3,000 jobs first).
