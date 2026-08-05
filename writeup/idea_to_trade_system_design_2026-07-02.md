# Idea→trade system design — the verdict amplifier (2026-07-02)

Design conversation distilled. Question: "an automated system from idea to trade, verifiable at
scale, each link trading off verifiability vs automation." The month's record (VRP attribution,
CTR/SFV nulls, prop side-quest, campaign ops) *rejected* the framing and produced this design
instead. **Build direction (user decision): grow it out of hpc-agent, modularly, never from
scratch, adding components only as the need arises.**

## 1. The corrected viewpoint (what the month actually showed)

Only **two verbs are genuinely automatable**:

1. **Run** — execute a *frozen, human-specified* experiment at scale, journaled, persisting
   primitives (512 factorial cells in ~25 min; 54/54 battery). Needs an on-call human (driver
   bugs, env crashes) but is semantically safe to automate.
2. **Re-score** — recompute any new metric from persisted per-bar primitives through the
   *identical verified machinery* (the 2^8 factorial answering the MZ/QLIKE question it wasn't
   designed for; rank-1 recompute rule).

Everything else stayed human all month, and each attempt to automate it failed measurably:
- curated Hero-A sweep **beat** the 200-trial Optuna campaign (automated search lost);
- gradient-learned gate self-disabled; regime-MoE tied (automated gate design lost);
- adaptive config-selection net-negative OOS vs pre-committed config ~flat (automated
  selection = in-sample selection at the meta level);
- oracle full-OOS Optuna = test-set overfit by construction;
- **no automated check caught any of the big bugs** — the three scaling bugs, the fill artifact,
  the bool-mask bug, the bd22177 raw-units retraction, the weak-baseline trap were all caught by
  human suspicion about units/baselines/too-clean plots. Gates only kill after a human points
  them at the right object in the right units.

**Consequence:** an "automated idea-to-trade pipeline" is a machine for manufacturing in-sample
selection at scale. The binding constraint is **human verification bandwidth**, which doesn't
parallelize. So don't remove the human — amplify them.

## 2. The reframe: amplifier, not pipeline

The system is a **growing library of frozen judgments** — verified machinery, registered specs,
persisted primitives, kill records. The machine's job at every link: **prepare the verdict,
never make it.** The per-link knob is not verifiability-vs-automation; it is *what fraction of
that link's judgment has already been frozen into verified machinery*. The system matures by
migrating judgments live→frozen **after** they've been human-validated a few times (precedents:
six-gate gauntlet ad-hoc→checklist; incremental Ridge hand-check→default-on; `derive_policy`
a month of sizing judgment→zero-tuning artifact).

Throughput metric: **human decisions per week × amplification per decision** — not automated
decisions per day. The product is validated decisions at minimum human cost; trades are what the
surviving decisions compile to.

## 3. The flow for one idea (cost-ordered: die at the cheapest gate that can kill you)

| Stage | Who | Cost | What |
|---|---|---|---|
| 0. Ledger check | machine | sec | dedupe vs kill records — already died? at which gate? |
| 1. Registration | **human** | min | one-page frozen spec: mechanism, candidate functional, conditioning axis, **claimed units** (QLIKE / P&L / P(pass) — the inversion makes this mandatory), honest baseline, kill criteria. LLM drafts, human signs, frozen before any result seen. |
| 2. Construction | machine + **human picks transform** | min | causal/bounded/no-magic-numbers; fill-swap + clip-scan run as checks; stabilization choice stays human (cumrv: mechanism-reasoned, ~27×/feature). |
| 3. Cheap kills, local | machine | sec–min | spanning R² vs deployed base, placebo, in-sample sanity. Most ideas die HERE. No cluster time until passed. |
| 4. Scaled run | machine (on-call) | hrs | hpc-agent campaign; designed experiment (factorial when multi-factor); **controls + known-value anchors embedded in the run**; per-bar preds+losses persisted cluster-side. Spec frozen — no adaptive selection anywhere. |
| 5. Mechanical gauntlet | machine | hrs | purged walk-forward, DM/MCS, deployment-transfer with **full-range** mixing sweep, re-score in claimed units. |
| 6. Verdict | **human** | min/idea | machine emits verify-first evidence package (numbers → through-line code → controls → gate results). Survive → distill to legible term; kill → ledger entry (gate, number). |
| 7. Economic translation | **human** + machine re-scores | days | honest-baseline P&L, breakeven vs real instrument cost (external datum), spanning/instrument check, P(pass) on actual venue rules post-hoc from persisted components (`results/vrp_pnl/` pattern). |
| 8. Trade | **human**, frozen artifacts | — | derived sizing (no tuning), venue verified vs primary sources, live monitoring emits the same evidence packages; decay-vs-bug-vs-regime verdict stays human. |

Why it scales despite the human bottleneck: stage-6 time is minutes/idea when packages are
standardized; most ideas never reach stage 4; embedded controls make machinery-audit free;
the ledger stops re-litigation. Many ideas in flight through 0–5, human processes verdicts in
batch — the overnight-campaign / morning-verdict rhythm, formalized.

## 4. Three invariants

1. **Frozen specs in, primitives out.** Every run consumes a pre-registered spec and persists
   per-bar preds + losses → any future question is a re-score, not a recompute.
2. **Every run carries its own controls.** Anchor cells with known values, placebo arms, control
   baselines — one read certifies machinery AND result.
3. **Kills are first-class data.** (idea, gate, number, units, date) appended per verdict →
   negative space auditable, saturation measurable, stage-0 dedupe free.

## 5. hpc-agent anchoring (build from it, not from scratch)

Natural mapping onto existing machinery — extend, don't replace:

| Component | hpc-agent seat |
|---|---|
| Frozen spec (stage 1) | sibling of `interview.json` / `tasks.py` — the interview primitive already persists a spec; registration = an extended spec schema |
| Run + persistence invariant (stage 4) | worker-template convention consumed by submit-flow (extract from the patched `winablate_r1.py`: per-bar loss + preds npz, MZ inline) |
| Controls/anchors in-run | task_generator convention: every campaign includes anchor + placebo + control-baseline cells |
| Gauntlet runner (stages 3+5) | local pre-submit step (preflight-like) for stage 3; aggregate-flow custom reducer or post-aggregate local step for stage 5; wraps `feature_cv.py` + DM/MCS + spanning + placebo |
| Evidence package (stage 6) | generated from aggregate output; verify-first notebook template |
| Ledger (stages 0/6) | experiment-repo side (`results/` or `writeup/`), machine-readable JSONL; dedupe query as a preflight-like check |
| Economic re-score (stage 7) | `eval_sim.simulate_topstep` + component dumps — already the pattern |
| Sizing (stage 8) | `adp.derive_policy` — frozen, done |

## 6. Build order — need-triggered, not scheduled

Build each piece the first time its absence costs something (it already has, once each):

| Trigger (the felt need) | Build |
|---|---|
| next time a "win" turns out to be selection/units (bd22177-class) | `specs/` registration template, frozen by commit |
| next cluster submission of an unscreened idea | stage-3 local cheap-kill runner (spanning + placebo + sanity, one command) |
| next campaign whose worker forgets to persist preds | worker template with persistence invariant + anchors baked in |
| next time a killed idea gets re-proposed | ledger JSONL + dedupe query |
| next verdict that takes >30 min to assemble | evidence-package notebook template |
| next new metric question on an old experiment | nothing — re-score already works; protect it |

The month already paid for the hard parts; what remains is connective tissue around human
verdicts. Roughly a week of total build if done at once — but per the user: **don't do it at
once; grow it as the needs recur.**
