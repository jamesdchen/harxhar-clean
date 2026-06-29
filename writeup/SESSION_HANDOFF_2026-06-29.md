# Session handoff — regime-MoE / FM / routing / order-content (2026-06-29)

Cold-resume doc. Branch `edge-features-legibility`. Repo local-only (no remote). Clusters: **CARC**
(SLURM, `usc-discovery`, `/scratch1/jc_905/harxhar-clean`, env `harxhar`) + **Hoffman2** (SGE, `hoffman2`,
`/u/scratch/j/jamesdc1/harxhar-clean`, env `hpc-pi` = full stack incl torch 2.3.1+cpu + numba 0.65.1).

## TL;DR
We built a differentiable regime stage to slot in for the Hero-B EBM (cell `xgb_all_buckets_tw1000_
enetreg2_linbest_rf480_slim`, residual on h16-19 close). Expert progression closed the gap monotonically
(additive NAM **0.12757** → bilinear FM **0.12108** → hinge basis-FM **0.12080** = no-regime) but plateaued
short of the **EBM 0.12033**. Then a **load-balance BUG** was found that suppressed the gate → the "routing/
architecture is exhausted" conclusion is **SUSPENDED**, re-testing now. Separately, a sequence-vs-state test
found a **faint order hint** (the log-sig "no order content" claim was an overstatement, now corrected) →
robustness re-run in flight.

## LIVE ON THE CLUSTER (harvest after clear — waiters won't survive a context clear)
Hoffman2, RUNNING (cluster JOBS survive; the bg ssh waiters do NOT):
- **`gfix_*`** (jobs 13815521-528): gate-fix re-run of the routing sweep with the FIXED load-balance +
  `aux_weight=0.1`. Labels: `gfix_d0` (no-gate control), `gfix_tree_d2`, `gfix_attn_k4`, `gfix_attn_k6`.
- **`seq_vs_state` robust** (job 13815529, name `seqstate`): order-content closer, 4 walk-forward folds ×
  3 seeds. Output `logs/seqstate.o13815529`.
HARVEST after clear (Hoffman2):
```
ssh hoffman2 'bash -lc "cd /u/scratch/j/jamesdc1/harxhar-clean
  grep -hE \"resid_regime_gfix_.* qlike\" logs/sig_col.o* | sed -E \"s#.*slim ##\" | sort -u
  grep -hE \"fold |ORDER_DELTA|VERDICT\" logs/seqstate.o13815529"'
```
Monitor: `bash wait_next_sge.sh <BASE> <PREFIX>` (SGE) on Hoffman2; `bash wait_next.sh <BASE> <PREFIX>`
(SLURM) on CARC. CARC is IDLE (cleaned). qdel/scancel are DENIED to the agent — ask the user.

## THE TWO VERDICTS PENDING (the only open questions)
1. **Does the FIXED gate help?** The prior "every router → 0.1208, architecture exhausted" was measured on a
   BUGGY gate (load-balance penalized per-sample routing VARIANCE → pinned routing sample-invariant → gate
   inert). Fixed to balance MEAN utilization. `gfix_*` re-tests fairly: if `gfix_tree/attn` beat `gfix_d0`,
   routing was real signal the bug hid (architecture re-opens); if they tie, "routing doesn't help" is earned.
2. **Is there order content?** Single-split `seq_vs_state` gave `order_delta = seq_true - seq_shuffle =
   -0.00029` (true order beat shuffled — the right direction, noise-scale) with history-as-a-set adding
   nothing. The robust run (folds × seeds) settles real-vs-noise: `order_delta` robustly < 0 beyond the
   seed-sd ⇒ genuine (small) order content (the corner the 3-channel log-sig missed); straddles 0 ⇒ noise.

## FINDINGS (numbers = full-OOS QLIKE on linbest; bar = EBM regime 0.12033 / no-regime 0.12081)
- Expressivity ladder (verify-first in `notebooks/results/edge_features/edge_04`): additive can't match
  pairwise; native pairwise (FM) jumps to 0.121; nonlinear-pairwise (hinge basis-FM) = 0.12080 (beats base
  on h16-19 0.15995 vs 0.16070); `lin` HURTS → pure-pairwise (residual has no main effects). Plateau at the
  floor. wd sweep flat. The EBM's 0.12033 edge = bagged-binned regularization.
- kNN (edge_05): local-ridge relevance helps a sliver, placebo-clean, dominated by EBM; naive analog hurts.
- Explainability (edge_06): coef trajectory = close regime ROTATES across timescales (fast har×close fades =
  0DTE-dilution signature); gate readout = gate splits on vol-state not clock.
- The thesis (PENDING the gate-fix re-run): the lever is **information (auction imbalance / GEX / OFI)**, not
  architecture. The MoE/FM is the differentiable substrate to point at that data once acquired (gate=regime,
  experts=new state). Routing levers built: #1 free-gate (REGIME_FULL), #2 context-decoupling (gate_context=
  raw), #3 attention router (AttentionGate). Nonlinear routing is already on via the hinge basis.

## CODE STATE
Committed: `36f88e3` (scaffold) + `b5de38a` (FM/basis-FM/attn/free-gate + 6 verify-first notebooks).
UNCOMMITTED (on disk, survives clear — COMMIT after gfix_/seq results land):
- `src/models/regime_moe.py`: **#2 context-decoupling** (gate_context="raw" → gate routes full raw features,
  experts use basis-expanded subsampled) + **the load-balance BUG FIX** (both gates: mean-util balance, not
  var/mean^2). FMExpert + hinge basis + AttentionGate already in b5de38a.
- `resid_amortized.py`: #1 free-gate (REGIME_FULL) — already committed in b5de38a.
- `moe_dig.sge`: AUXW + gate_kind/num_regimes/REGIME_FULL/basis/gate_context env knobs.
- `seq_vs_state.py` + `.sge` (new), `submit_sge_gatefix.sh` (new), `writeup/mechanism_and_data_to_buy_2026-
  06-28.md` (log-sig overclaim CORRECTED), `results/moe_ladder/*.csv`.
The broader untracked arc (backfit*, ebm_*, heroA/B, interp, pca, persist, plot, ~70 files) is PRE-EXISTING
uncommitted work — left alone. `results/` data + `logs/` + `.claude/` + `*.tgz` are gitignored.

## INFRA / HOW TO RUN
- Deploy: `tar --force-local -czf X.tgz <files>; scp X.tgz hoffman2:.../; ssh hoffman2 "tar xzf X.tgz"`.
  CONTAMINATION LESSON: never deploy regime_moe.py / resid_amortized.py while a job that imports them runs
  (mixes code mid-run) → use fresh labels + deploy only when idle.
- Sweeps (Hoffman2 SGE): `submit_sge_gatefix.sh` (gate-fix), `submit_sge_basisfm.sh`, `submit_sge_fmtune.sh`
  (FM rank×wd_v, staged, not yet run), `submit_sge_sweep.sh <prefix>`. moe_dig.sge knobs via -v: RDEPTH
  EXPERT BASIS NKNOTS RANK FMLIN WDV WD GATEKIND NREG RFULL AUXW LBL.
- collect = `$PY resid_amortized.py chunk_collect <CID> resid_regime <LBL>` → COLLECT qlike + qlike_h16-19.
- preds for the notebooks: per-run CSVs in results/moe_ladder/preds/ (gitignored; concat of chunks).

## NEXT STEPS
1. Harvest `gfix_*` + `seq_vs_state` (above) → the two verdicts. Update writeup/regime_moe_results + the
   edge_04 notebook + results_all.csv.
2. COMMIT the batch (gate fix + #2 + seq_vs_state + writeup correction + gfix sweep).
3. If the fixed gate helps: pursue routing (decoupled #2 sweep gate_context=raw; FM-tuning submit_sge_fmtune).
   If not: the architecture conclusion is earned; pivot fully to the data-to-buy (auction imbalance).
4. The seq robustness verdict decides whether to scaffold a fuller sequence model (only if order_delta robust).
5. Memory: see `.claude/.../memory/MEMORY.md` entry "regime-MoE/FM session".

## KEY FILES
Model: `src/models/regime_moe.py`. Cascade: `resid_amortized.py` (preds_chunk resid_regime arm).
Order test: `seq_vs_state.py`. Explain: `gate_readout.py`, `coef_trajectory.py`. Notebooks:
`notebooks/results/edge_features/edge_01-06.ipynb`. Writeups: `regime_moe_results_2026-06-29.md`,
`explainability_audit_findings_2026-06-29.md`, `discovery_process_methodology_2026-06-29.md`,
`mechanism_and_data_to_buy_2026-06-28.md` (the corrected log-sig claim), `gpu_on_carc_runbook_2026-06-29.md`.
