# Session handoff — regime-MoE/FM/FiLM → starvation → un-starving → new-hero ensemble (2026-06-30)

Cold-resume. Branch `edge-features-legibility`. Repo local-only (no remote). Supersedes
`SESSION_HANDOFF_2026-06-29.md` (whose "routing SUSPENDED" is now RESOLVED).

## TL;DR — the arc resolved, and one forward lever found
The differentiable regime stage (gate/FiLM) does NOT beat the EBM 0.12033 — **not because of a bug** (5
independent confident-routing tests all tie; per-bar preds differ, experts differentiate), but because the
gate is **STARVED** (`r2 = r1 − d8(X)` ⊥ d8's depth-8 partition; depth-ladder regime gain grows 6× as d8
shrinks) and **SUBORDINATE** (enet head-to-head: MoE-as-sole-regime 0.12494 vs d8 0.12255). d8 dissected =
55% additive / 9% pairwise / 36% higher-order, and the higher-order is **diffuse** (context-collapse: only
VIX-term beats a random-bin placebo; clock worse than random). **The lever is information**, named:
HAR × {sentiment, attention, returns, VIX-term}.

**The one forward result: multi-task `r1`-aux UN-STARVING** (aux head predicts the pre-d8 residual →
reintroduces d8-taken structure; causally clean). Recursive form is best: **ĝ-gated** adaptive aux (gate
the aux per-row by `|ĝ|=|r1−r2|`, d8's bite — the *un-starved* conditioner) ~DOUBLES the gain AND removes
the fold reversal (local: −0.00138 vs uniform −0.00078; vol-gating a STARVED feature FAILED). Resolves the
bias/variance trade. **EBM ⊕ MTFM ensemble** (full-OOS linbest, correct EBM): depends on ω (EBM share). MTFM-HEAVY (ω≤0.5) LOSES
(0.121-0.122 > 0.12033 — a heavy weak-FM drags the strong EBM). But a LIGHT MTFM correction WINS: ω≈0.7-0.9
(10-30% MTFM) → **~0.12024-0.12026, a consistent −0.00008 below the EBM** (U-shaped, both aux; EBM-alone
sanity w100=0.12033 ✓). So a **TINY genuine new low (~0.12025 at ω≈0.9)** — the MTFM helps only as a *light
diversity correction*, not a heavy component; ĝ-gating ~neutral at the optimum. CAVEAT: an earlier MTFM-heavy
grid (ω≤0.4) wrongly read "no hero" — the optimum is at HIGH ω; sweep the full ω range. Real but edge-of-
significance; substantial gains still need information.

## CTR/SFV exploration (2026-06-30) — all four architecture threads DEPLOYMENT-NULL
Tested the transferable ad-CTR / short-form-video levers; every one is null at deployment vs the strong EBM,
each for a distinct, instructive reason:

| thread | local | deployment |
|---|---|---|
| ĝ-gating | wins (−0.0005) | null — doesn't transfer (the **linbest base, NOT d8-strength**: isolation shows ĝ helps equally at d8 n_est 80 & 200 on realrank → it's the more-distilled base / thinner `r1`) |
| DIN/BST target-attention | huge (−0.00598, 9× the GRU) | null — **d8 eats ~94%**, only −0.00037 survives over Hero-B (prize-sizing on the d8-leftover) |
| grad-learned soft gate | — | null — **self-disables** (`gw→0`; un-starving is OOS-regularization, invisible to in-sample gradient; needs bilevel/val tuning, not naive joint gradient) |
| SFV multi-objective aux | wins (−0.0005, monotone) | null — `amulti≈auniform≈0.12024` (**EBM subsumes** the `|r2|`/`y` magnitude/vol-level structure) |

AutoFIS + online-streaming: designs noted, NOT run (low priority — every lever is deployment-null; recency already failed; FM already regularizes interactions).
CODE: `MultiTaskFM.fit` takes a `[n,K]` aux stack (1+K heads); `preds_chunk_ggrid` has a `"multi"` variant; `carc_ggrid_multi.sbatch`.
VERDICT: the price-only floor is ~0.12022–0.12033; the lever is **INFORMATION** (data-to-buy), now confirmed from four independent angles.

## OOS-robust feature/HP harness (2026-06-30) — the methodology pivot
The local-mirage failure mode motivated the pivot: stop adding architecture, build the **evaluation discipline**
that gates features/hyperparameters against it. `src/evaluation/feature_cv.py` (committed d8dd553):
- `purged_walk_forward` / `inner_split` — embargoed walk-forward CV + per-step inner-val.
- `score_feature` / `significance_gate` — **bagged** incremental-value delta + permutation **placebo** + pass/fail (`CI<0 ∧ replicates ∧ beats placebo`). Validated: passes real signal (har125→vol, z=11), rejects noise, correctly rejects base-absorbed features (the floor finding).
- `oof_context` / `context_columns` — cross-fit (leak-free) context.
- `tune_hparam` — **per-step CV-ω** on purged inner-val. Validated: CV-ω 0.14192 < in-sample 0.14193 < fixed-0.8 0.14195; adapts per step (0.71–0.97). Fixes the in-sample collapse (ω→1.0/EBM-only).
- `context_omega` — **context-attention ω(x)** = attention over cross-fit regime anchors, each a shrunk OOS CV-fit ω_k. **Rescues the attention gate that self-disabled under gradient.** Validated: 0.14191 < scalar; ω varies by context (std 0.02–0.06).
All gains tiny (weak-signal floor) but REAL through the same purged gate that killed the architecture levers.
NEXT: an `edge_10` notebook (professor-facing, verify-first); cluster-deploy the per-step/context ω into
`preds_chunk`; run the **data-to-buy** through the gate (only gate-passing features ship). Validations in
scratch: `demo_feature_cv`, `pos_control`, `omega_cv`, `omega_context`.

## LIVE ON THE CLUSTERS (harvest after clear)
- **CARC (AUTHORITATIVE) — DONE.** `ggrid` (ω∈{.2,.3,.4}, MTFM-heavy) all LOSE (0.121+). `ggrid2`
  (ω∈{.5,.7,.9,1.0}) is the real curve: **w100=0.12033 (EBM-alone sanity ✓); w90=0.12024-0.12025, w70=0.12026
  → a TINY new low (~−0.00008) at a LIGHT MTFM correction (ω≈0.9)**; w50=0.1205 (worse). U-shaped optimum at
  ω≈0.9. Both EBM cfgs used the CORRECT one (`outer_bags=4,max_rounds=500,max_leaves=3,interactions=10`).
  Harvest: `ssh usc-discovery 'grep -hE "COLLECT .*resid_regime_ggrid2?_.* qlike=" /scratch1/jc_905/harxhar-clean/logs/ggcol_*.out'`. Poll CARC GENTLY (600s).
- **Hoffman2 (RELATIVE-ONLY, redundant)** — `ens_*` + `gg_*` (jobs 13826851-857, 13828240-252): used
  `EBM_CFG={}` (HEAVY library defaults, NOT the ceiling cfg) → `ebmonly` will NOT reproduce 0.12033, so the
  ABSOLUTE comparison is invalid. Only the WITHIN-run relative effects stand (ensemble-vs-EBM, ĝ-vs-uniform)
  as a cross-cfg robustness check. CARC supersedes these — `qdel 13826853 13826856` (+ queued `gg_*`) frees
  Hoffman2 with no loss to the hero verdict. **RESULT (landed, CONFIRMS CARC):** `ens_ebmonly` 0.12046
  (≠0.12033 — heavy default EBM is a different, slightly-WORSE model), `ens_w04` 0.12086, `ens_w02` 0.12147 —
  ensemble loses, MTFM drags → the negative is **robust across both EBM cfgs**. **FiLM deployment**: `film_lin`
  **0.12080** (ties the gate/no-gate baseline exactly → graceful degradation confirmed), `film_enet_full` 0.12252.
- qdel/scancel DENIED to the agent — ask the user. `qstat` needs a login shell (`bash -lc`) on Hoffman2.

## DUAL-CLUSTER NOTES (this session ran Hoffman2-only until the end; now using both)
- CARC: SSH alias `usc-discovery`; `/scratch1/jc_905/harxhar-clean`; SLURM (`--account=pollok_1603 --partition=main`);
  env `harxhar` at `/home1/jc_905/.conda/envs/harxhar/bin/python`. **libstdc++ fix**: the env's interpret/EBM
  needs `export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6` (the env's own libstdc++ has CXXABI_1.3.15; the
  gcc MODULE's doesn't) — OR run the env python DIRECTLY (no `conda activate`/module-load), as `ctrl_ebm.sbatch` does.
  CARC has the linbest cache + masks. **Poll CARC GENTLY (600s)** — fast polling trips its banhammer.
- Hoffman2: SGE, `hoffman2`, `/u/scratch/j/jamesdc1/harxhar-clean`, env `hpc-pi`.
- Local: env **285J** (`miniconda3/envs/285J`) has torch+xgboost; `interpret-core` now installed → EBM runs
  locally; only the **realrank** cache is complete locally (linbest/enet cluster-only). Notebooks run locally via 285J.

## CODE STATE (uncommitted since 29c9da9 — fold into the next commit)
- `src/models/regime_moe.py`: FiLMExpert, gate_temp/gate_lr/gate_weight_decay, routing diag, **MultiTaskFM**
  (per-row aux for ĝ-gating) + `_MultiHeadFMNet`.
- `resid_amortized.py`: preds_chunk `ebm_mtfm` branch (REGIME_MODEL=ebm_mtfm, ENS_W, MT_*, MT_GATE=ghat) +
  REGIME_INVERT/NOGLOBAL/GDEPTH/RMODEL/NOREG; **preds_chunk_ggrid + do_chunk_task_ggrid** (EBM-cached grid).
- Submit: `submit_sge_{liveness,sharpgate,ladder,invert,enet,trainlong,film,ensemble,ensemble_ggated}.sh`,
  `moe_ggrid.sge`+`submit_sge_ggrid.sh`, `moe_dig.sge` (knobs); CARC `carc_ggrid.sbatch`,`carc_collect.sbatch`,`submit_carc_ggrid.sh`.
- Diagnostics: `d8_dissect.py/.sge`, `d8_context.py/.sge`.
- Notebooks: `edge_07_gate_liveness`, `edge_08_starvation`, `edge_09_recursive_unstarving` (+ `edge08_lib.py`).

## NEXT STEPS
1. Harvest the CARC `ggrid` (the hero verdict). Update `edge_09` + a writeup with the deployment numbers.
2. Commit the batch. Optionally `qdel` the redundant Hoffman2 arms.
3. Tuning-later: the EBM-cached `ggrid` is the cheap scorer — point Optuna (`async_tune.py`) at a slice of it
   (fit EBM once/block, sweep ω/aux ~free). Don't tune on full-OOS.
4. Cross-architecture un-starving (memory `ctr-lessons-cross-architecture-test`): does `r1`-aux replicate on
   an EBM via stacked-data + task-indicator? Would put un-starving on the 0.12033 ceiling model.
5. The real lever: data-to-buy (auction imbalance / GEX / OFI / richer sentiment) into the ĝ-MTFM substrate.

## MEMORY
`MEMORY.md` entries: `regime-moe-fm-session` (resolved arc), `ctr-lessons-cross-architecture-test` (ĝ-gated
win + the EBM-stacking TODO + recency/vol-gating rejected), `intraday-regime-findings`, `hero-stack-state-2026-06-27`.
