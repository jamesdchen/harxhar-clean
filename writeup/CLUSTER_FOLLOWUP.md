# Cluster followup — 2026-08-14

Hoffman2 **reached** via direct SSH (`-o ProxyJump=none`). `usc-discovery` hop is down; TCP to `hoffman2.idre.ucla.edu:22` is fine. Key `~/.ssh/id_hoffman2`.

## qstat (`jamesdc1`, all `qw`, nothing running)

Submitted 2026-08-13 from `dtn1`, cwd `/u/scratch/j/jamesdc1/harxhar-clean`.

| Job | Name | Array | Since | Notes |
|---|---|---|---|---|
| 14318286 | `scorer_blocks.sge` | 1 | 06:53 | 4h wall, 8G/10G |
| 14318447 | `probe_allint_array.sge` | 1–6 | 07:07 | 12h wall |
| 14318448 | `probe_multih.sge` | 1–5 | 07:07 | 8h wall |
| 14318539 | `unif_b2_lasso_tuned` | 1–100 | 07:46 | `ARM=b2_lasso_tuned`, script `/tmp/unification_wide.sge` |
| 14318728 | `probe_hedge_split.sge` | 1–6 | 08:53 | **unsafe 6-way write path** |

`qhold` was not available in the non-interactive PATH (`SGE_ROOT` cell missing). Job 14318728 is still `qw`.

## Remote harvest

- `results/hedge_lin/` exists, **0** `npz`.
- `results/unification/b2_lasso_tuned/` **does not exist**.
- `results/unification/b2_lasso/` exists, **0** `npz`.
- Other unification dirs (ridge, blk2/3/4, trees) are present from earlier waves.

## Hedge script mismatch

- **Remote** (what 14318728 will run): `#$ -t 1-6` — `ridge_hedge`, `ridge_tuned`, `blk2_hedge`, `blk2_tuned`, `blk2_fixed`, `a0_ols_har`. Not harvest-safe: `--arm` still writes a paired table that assumes all four of `a0_ols_har`, `mh`, `mb_h`, `ma`.
- **Local** `ae5e5f6`: `#$ -t 1-3` — only the three blk2 arms.

Do **not** let 14318728 start until the write path is arm-conditional, or delete it and submit the local 3-task script (or the all-arm script).

## Still do not

- Invent a new campaign.
- Submit a second lasso array (14318539 is already the one-protocol `b2_lasso_tuned`).
- Harvest: nothing to score yet (lasso dir missing, hedge 0 files).
