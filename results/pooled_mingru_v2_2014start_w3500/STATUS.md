# Pooled ridge-init MinGRU v2 ("memorize, then regularize"), W 3,500: status log

Rule: `RULE_gru_v2.txt` (written 2026-10-23 18:40 PDT, before any v2 job). Branch insurance-selling-2026-09-19-e34-gru-v2
(from 2cc624c). CARC root `/home1/jc_905/pmf2_35` (W 3,000: `/home1/jc_905/pmf2`, the shared cluster log). Inputs read from
the old run (read-only): `/home1/jc_905/pmf35/data` (design copies, fold-3 ridge inits), `/home1/jc_905/pmf/bin`
(forecast_train md5 44ab6fff), env `/home1/jc_905/envs/harxhar-r13`.

## Log
- Cluster housekeeping (cancellations, feeders): logged in ../pooled_mingru_v2_2014start_w3000/STATUS.md.
- 2026-10-23 18:49 PDT G0 PASS (W 3,500, CPU 13353666; first attempt 13353638 failed on its own assertion: rows < 45,500
  are undated in the W 3,500 export, fixed by dating rows [39,000, 45,500) from the W 3,000 export, equal on every row
  dated in both). Folds: fit [0, 40,702) (89.5 % of W), [0, 43,952) (96.6 %), [1,702, 47,202); validation dates =
  folds14.json. Stage-1 inits recomputed byte for byte (full keep 295, HAR 12); TQ_0 HAR 0.116384 / 0.116390 /
  0.116387 / 0.116427 (L 8 / 20 / 48 / 115), full 0.107735 / 0.107740 / 0.107750 / 0.107789. Stage-2 kept columns full
  284 / 290 / 295; partly zeroed late-start columns 3 / 3 / 6. Shortcut-alone validation QLIKE (L20) by fold: HAR
  0.12640 / 0.13490 / 0.12991, full 0.11586 / 0.12709 / 0.12300.
- 2026-10-23 22:55 PDT: v2 root moved to /scratch1/jc_905/pmf2_35 (the /home1 quota; see the W 3,000 STATUS.md and
  RULE ADDENDUM 1). G4 (Stage 1, full set) submitted on L40S, nice 100: 13367193-98 (6 files, 81 jobs: core grid x lr
  {0.003, 0.01, 0.03} x 3 seeds, E 300; w0 1,702). GPU type of this window's Stage-1 group: L40S.
