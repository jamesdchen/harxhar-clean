# DRAFT PROPOSALS — UNSIGNED

**Nothing here is policy until the human signs each slot via the journaled
utterance path.** LLM-drafted 2026-07-07 per the drafting-is-the-sanctioned-
prelude role (`idea_to_trade_pipeline_v2_2026-07-07.md` header doctrine:
"LLM drafts; human signs"). Each section below is a proposal for one of the
three `HUMAN SIGNS` slots in `writeup/idea_to_trade_pipeline_v2_2026-07-07.md`
§3, plus one clearly-marked bonus draft. Every recommendation names its
reasoning, the alternatives rejected, and the one-line signing utterance.
Edit freely — the words that bind are yours, not these.

Machinery these proposals are shaped to fit (read-only references, hpc-agent
repo): `docs/design/registration-kernel.md` (template mechanism R5, evidence
vocabulary R4), `docs/design/determinism-fingerprint.md` (envelope evidence
`{n, n_full, n_partial, scales, clusters}`, demand vocabulary
`{min_n, min_n_full?, scales, clusters}`), `docs/design/domain-packs.md`
(S6 `registration_fields` / `required_receipts`, `receipt_bindings`),
`docs/design/evidence-memory.md` (conclusions carry tags; dated, advisory,
never blocking), `docs/design/notebook-audit.md` (percent-format template
`.py`; slugs are the required inventory).

No performance number in this document is a result; the only empirical
figures cited are already-recorded repo facts (shipped code defaults, dated
session notes) named by file.

---

## Slot 1 — RV-data scope policy (pipeline §3.1)

### Recommendation

**Scope-tag vocabulary** (all tags shape-valid slugs; lists are pack-data-like
— names + policy, no code):

*Dataset-family tags* — one per input parquet family under `data/`:

| Tag | Names (actual files) |
|---|---|
| `rv-core-bars` | `data/core_stats.parquet` — the 30-min bar RV panel (target `adj_RV` + HAR features) |
| `exog-market-ew` | `data/ewstock_stats.parquet` |
| `exog-market-vw` | `data/vwstock_stats.parquet` |
| `exog-sentiment` | `data/spy_and_sentiment.parquet` |
| `exog-implied-vol` | `data/vix_and_voldemand.parquet` (IV leg) |
| `exog-vol-demand` | `data/vix_and_voldemand.parquet` (vol-demand leg) |
| `calendar` | `data/time_categories.parquet` |
| `options-spx` | `data/optionm_spx_chain.parquet`, `data/optionm_spx_spot.parquet` |
| `options-friction` | `data/om_friction/` (venue/friction exports) |

*Time-slice tags* — the load-bearing pair, cutting across every family
(the bar panel spans Jan 2018 – May 2025, ~74,934 bars —
`writeup/SESSION_HANDOFF_2026-07-02.md` addendum):

| Tag | Slice | Lock state |
|---|---|---|
| `rv-dev-2018-2024h1` | 2018-01-02 … 2024-05-31 | **unlocked** — the working window |
| `rv-holdout-2024h2-2025` | 2024-06-01 … end of data (2025-05) | **LOCKED** — the final ~12 months |

Every run/reduction declares one time-slice tag plus the dataset-family tags
it touches. The look ledger counts against ALL declared tags.

**Look-budget posture:**

- `rv-dev-2018-2024h1`: **no hard budget — counted and disclosed.** Look
  counts + distinct supersession lineages ride every harvest/greenlight
  brief (the shipped look-ledger behavior); the human reads the count as
  deflation context at stage 6. A hard cap on the dev window would only
  drive work off-ledger.
- `rv-holdout-2024h2-2025`: **budget = 1 look per registered program.**
  Mechanized at registration via the `scope-budget` prerequisite kind
  (registration-kernel R3: core compares the ledger count against a
  caller-declared number — see Slot 3's `holdout-look-budget` entry).

**Which stages may touch what:**

| Stage (pipeline v2 §2) | May reduce over |
|---|---|
| Prelude P2–P4, 2 (construction), 3 (cheap kills) | `rv-dev-*` only |
| 4 (scaled run), 5 (gauntlet) | `rv-dev-*` only |
| 6 (verdict) | reads dev results only; holdout untouched |
| 7 (economic translation) | `rv-dev-*` + `options-friction` (cost data is an external datum, not a selection surface — unlocked) |
| 8 (trade go/no-go) | the ONE `rv-holdout-2024h2-2025` look, post-unlock |

**The lock/unlock ceremony:**

1. `rv-holdout-2024h2-2025` is locked NOW, before any post-upgrade submit
   (adoption item, pipeline §4: tag from the first submit onward; the lock is
   only as real as the tagging discipline).
2. Unlock is the typed human ceremony behind the authorship gate
   (scope-lock machinery, pipeline §1.4) — **once per program, at the
   go/no-go decision**, journaled permanently. The unlock utterance must name
   the registered attempt it spends the look on.
3. The look is then **spent**: immediately re-lock. The next program's
   holdout is a NEW boundary rolled forward as data accrues (e.g.
   `rv-holdout-2025h2-2026` once ≥6 fresh months exist) — never a re-read of
   a spent slice. A spent-holdout tag stays in the ledger forever as the
   record that it was spent.

### Reasoning

- **Time-slice locks are the ones that matter.** The look ledger's threat
  model is collective mining of a window (pipeline §1.2); dataset families
  are the disclosure axis (what the attempt touched), the time slice is the
  exhaustion axis (what it spent).
- **~12 months holdout** is the smallest slice that spans more than one
  vol regime at 30-min frequency while leaving 6.4 of 7.4 years for
  development — and it cleanly postdates every result currently in
  `results/` (all campaigns to date evaluated through May 2025 on the full
  window, so the holdout is already partially "seen" by past full-window
  evals; the lock stops FURTHER mining, which is all a lock can honestly do —
  disclosed, not hidden).
- **One look per program** is the classical lockbox posture: the holdout's
  entire value is that its first read is unbiased.

### Alternatives rejected

- **Lock nothing; ledger-count only.** Rejected: counting without a locked
  slice means every tag is eventually exhausted equally; there is then no
  unbiased surface left for the go/no-go read.
- **Lock per-dataset (e.g. lock `options-spx` wholesale).** Rejected:
  exhaustion is temporal, not columnar; locking a family blocks legitimate
  dev work without protecting the go/no-go read.
- **A 2-year holdout (2023-06 onward).** Rejected: it would orphan the
  2023–2025 evaluation window most existing baselines are calibrated on and
  halve the usable recent-regime dev data; the marginal unbiasedness is not
  worth it for a program whose verdicts are re-checked live at stage 8
  anyway.
- **Unlock-per-question (multiple holdout reads per program).** Rejected:
  the second read of the same slice is in-sample selection at the meta
  level — the exact failure `idea_to_trade_system_design_2026-07-02.md` §1
  documents.

**To sign:** an utterance naming the two time-slice tags and the budget —
e.g. *"Sign slot 3.1: lock rv-holdout-2024h2-2025, budget one look per
registered program, unlock only at go/no-go, dev window rv-dev-2018-2024h1
counted-not-capped."*

---

## Slot 2 — Stage-3 gauntlet composition + thresholds (pipeline §3.2)

Stage 3 is the local cheap-kill run: "spanning R² vs deployed base, placebo,
in-sample sanity" (v1 §3), one command, seconds-to-minutes, no cluster time
until passed. The composition below is LLM-drafted from `src/` per the slot's
own text; **the numbers are the human-owned part.**

### Recommendation — composition (cost-ordered, die at the cheapest check)

1. **Sanity/units** — target is the README-invariant transform
   (`winsorize(sqrt(RV/diurnal))`, `.shift(1)` causality present), no NaN
   inflation (candidate NaN fraction ≤ base + 1pp), QLIKE evaluated raw-space
   via Duan smearing (`src/evaluation/metrics.py`). Kill = any failure.
   Binary checks; no threshold to sign beyond the NaN margin.
2. **Spanning vs deployed base** — regress the candidate on the deployed
   base's feature set (base = the production ridge config,
   `configs/ridge_market_ew_prod.yaml` lineage). Kill if in-sample R² of
   candidate-on-base **> 0.95** (the candidate is a linear re-dress of what
   is already deployed).
3. **Incremental OOS gate** — `src/evaluation/feature_cv.py::significance_gate`
   over `purged_walk_forward` folds with `score_feature`'s bagged deltas +
   circular-shift placebo. Kill unless ALL of:
   - CI excludes 0: `mean + k·se < 0` with **k = 2.0**;
   - replication: ≥ **0.7** of (fold,boot) draws AND ≥ **0.7** of folds favor
     the candidate;
   - beats placebo: real mean **2.0σ** below the circular-shift placebo mean;
   - protocol constants: **n_folds = 5, embargo = 0.01, n_boot = 8**.
4. **Economic floor** — mean relative QLIKE improvement vs deployed base
   ≥ **0.05%** (relative delta ≤ −5e-4 of base QLIKE). A statistically clean
   but sub-floor delta is a kill at stage 3 (it may still be recorded as a
   dated conclusion — evidence, not a verdict on the idea).
5. **Cost cap** — the whole gauntlet completes in **≤ 10 min** on the local
   box; a candidate whose stage-3 screen cannot run locally is redesigned,
   not promoted to cluster time.

**Reproducibility demand, in the fingerprint's evidence vocabulary:** the
gauntlet run itself must be deterministic before its kill is trusted —
`reproduction: {min_n: 2, scales: ["canary"]}` (two executions, canary
scale — cheap, and byte-identical-twice honestly supports `exact` per the
fingerprint doc's n=2 rule). Main-scale evidence is deliberately NOT
demanded here — that is registration's seat (Slot 3), where the demand is
`{min_n: 3, min_n_full: 1, scales: ["main"]}`.

### Reasoning per number

| Number | Anchor |
|---|---|
| k = 2.0 | **Anchored (statistical + precedent):** ~2σ one-sided; it is the shipped default of `significance_gate(k=2.0)` that gated the month's real feature work. Raising it belongs at stage 5 (DM/MCS on the scaled run), not at the cheap screen. |
| repl_frac = 0.7 | **Shipped default; partly arbitrary.** Rationale: majority-plus replication across folds without demanding unanimity (one regime-mismatched fold shouldn't veto). Sensitivity ±50%: at 0.35 the check is vacuous (any positive mean passes); at 1.0 a single COVID-regime fold kills nearly every true candidate. The informative band is roughly 0.6–0.8; 0.7 is its center. |
| spanning R² > 0.95 kill | **Arbitrary within a band — flagged.** Rationale: a candidate 95% linearly explained by deployed features cannot carry 0.05% incremental QLIKE except through the 5% residual, so the two thresholds are roughly consistent. Sensitivity ±50% (on the redundancy complement, 0.05): kill at 0.925 starts killing genuinely complementary features that merely correlate with vol level (almost everything does); kill at 0.975 passes near-duplicates that then waste a cluster run each. |
| economic floor 0.05% rel QLIKE | **Anchored (economic precedent), floor placement arbitrary.** The smallest improvement that survived full validation to date is ≈0.075% (the ω-mixing low, w90 0.12024–5 vs 0.12033 sanity, `writeup/SESSION_HANDOFF_2026-06-30.md`) — the floor sits just below the smallest known-real effect so it cannot retro-kill a historical true positive. Sensitivity ±50%: 0.075% would have put the ω finding exactly on the boundary (uncomfortable); 0.025% admits deltas well inside walk-forward noise for this panel, forcing stage-5 to do stage-3's job at cluster prices. |
| n_folds=5, embargo=0.01, n_boot=8 | **Anchored (shipped defaults in `feature_cv.py`).** Protocol constants, not judgment thresholds; they change only with a recorded protocol change. |
| ≤ 10 min local | **Capacity-based, deliberately arbitrary.** It encodes "stage 3 is free" (cost-ordering doctrine, v1 §3). Sensitivity ±50%: 5 min forces subsampled screens (weaker kills); 15 min changes nothing structural — this number can move freely and should be set to whatever the human's patience actually is. |
| repro {min_n: 2, scales:["canary"]} | **Anchored (fingerprint doc):** n=2 is exactly what the double canary mints for free; demanding more at a cheap screen inverts the cost ordering. |

### Alternatives rejected

- **DM/MCS at stage 3.** Rejected: model-level significance
  (`src/evaluation/diebold_mariano.py`, `model_confidence_set.py`) needs the
  scaled run's per-bar losses; running it on the local screen double-counts
  the same small sample the significance gate already used. It stays
  stage 5.
- **A P&L-based stage-3 kill.** Rejected: honest P&L needs instrument costs
  (stage 7's external datum); a cheap-kill P&L would be a fabricated number.
- **i.i.d. permutation placebo.** Rejected in code already
  (`feature_cv.py::score_feature` comment): a flat shuffle white-noises a
  slow feature — mismatched null. Circular shift is the signed default.
- **No economic floor (statistics only).** Rejected: the month's record
  shows statistically-clean micro-deltas that cost more in verification
  bandwidth than they can ever return; the floor is the "is it worth a
  cluster run" gate, and stage 3 is exactly where that question is cheapest.

**To sign:** an utterance naming the numbers — e.g. *"Sign slot 3.2: gauntlet
= sanity → spanning(0.95) → significance_gate(k=2.0, repl 0.7, folds 5,
embargo 0.01, boots 8) → economic floor 0.05% rel QLIKE, 10-min local cap,
repro demand min_n 2 canary."*

---

## Slot 3 — Registration template fields (pipeline §3.3)

Shaped exactly as `docs/design/registration-kernel.md` R5 expects
(`{"fields": [...], "prerequisites": [{slot, kind, requires}]}`), so the
signed artifact is directly usable as the caller-referenced template file.
Home: `specs/registration_template.json` in this repo, frozen by commit
(raw-bytes sha is the kernel's `template_sha`). Field slugs are opaque to
core (counted for presence, never interpreted); the one-line meanings below
are for the humans filling them.

### Recommendation — the template

```json
{
  "fields": [
    "mechanism",
    "candidate-functional",
    "conditioning-axis",
    "claimed-units",
    "honest-baseline",
    "kill-criteria",
    "review-horizon",
    "capacity-estimate",
    "data-scope-citation",
    "scopes-affirmed-locked",
    "conclusion-citation"
  ],
  "prerequisites": [
    {"slot": "audited-source", "kind": "notebook-audit", "requires": {}},
    {"slot": "repro-main-scale", "kind": "reproduction",
     "requires": {"min_n": 3, "min_n_full": 1, "scales": ["main"]}},
    {"slot": "holdout-look-budget", "kind": "scope-budget",
     "requires": {"max_looks": 1}},
    {"slot": "concluding-verdict", "kind": "attestation"}
  ]
}
```

Field meanings (one line each; the pipeline-§3.3 list, confirmed, plus the
v2-decision additions):

| Slug | Meaning |
|---|---|
| `mechanism` | Why the edge should exist — the economic/microstructure story, one paragraph. |
| `candidate-functional` | The exact functional form / feature / model being claimed (names the audited source's section slugs). |
| `conditioning-axis` | What the claim is conditioned on (regime, calendar bucket, context bins) — or "unconditional". |
| `claimed-units` | The units of the claim: QLIKE / P&L / P(pass) — mandatory (the inversion lesson, v1 stage 1). |
| `honest-baseline` | The deployed base being beaten, by config identity (e.g. `configs/ridge_market_ew_prod.yaml` @ commit sha). |
| `kill-criteria` | The pre-committed numbers that kill THIS attempt (Slot 2's thresholds by reference + any attempt-specific ones). |
| `review-horizon` | The date this registration's evidence expires and a live re-verdict is due — time-indexed, never permanent (the no-kill-ledger doctrine: evidence about a regime, not a verdict about an idea). |
| `capacity-estimate` | Claimed deployable size and its binding constraint (venue rules, ADV fraction, friction data cited from `options-friction`). |
| `data-scope-citation` | The scope tags this attempt reduces over (Slot 1 vocabulary), verbatim. |
| `scopes-affirmed-locked` | The tags the signer affirms stay locked throughout (at minimum `rv-holdout-2024h2-2025` until go/no-go). |
| `conclusion-citation` | Prior dated conclusions consulted (evidence-memory tags/ids), or explicitly "none found" — advisory context, never a gate on re-proposal. |

Prerequisite notes (template entries carry `{slot, kind, requires}`; the
registration INSTANCE adds each entry's `subject_id` + `content_sha` at
registration time, per R3's full-address rule):

- `audited-source` — the graduated entry point's notebook audit reads
  current at the registered sha (kernel R3 `notebook-audit` row).
- `repro-main-scale` — the exact demand vocabulary of R4/`evidence_meets`:
  ≥3 fingerprint samples counting full+partial, ≥1 full, main-scale evidence
  present. Registration is "the seat that can demand main-scale evidence
  before 'reproducible' counts" (fingerprint doc §4) — this slot is that
  seat, exercised.
- `holdout-look-budget` — `subject_id` = `rv-holdout-2024h2-2025`; core
  compares the ledger count against `max_looks` and requires the scope
  unlocked-or-within-budget (R3 `scope-budget` row). This is Slot 1's
  budget, mechanized. (`max_looks` is the proposed key name; confirm the
  implemented key against `state/registration.py` when the kernel lands —
  unknown `requires` keys are a loud refusal by design.)
- `concluding-verdict` — the generic `attestation` kind (accepts NO
  `requires`, by design): the stage-6 verdict recorded via append-decision
  on the run scope. When evidence-memory's first-class `conclusion`
  prerequisite kind lands (E6, reserved), migrate this slot to it.
- **Deliberately absent for v1: a `gauntlet-pass` pack-receipt slot.** The
  `pack-receipt` kind refuses loudly until the domain-pack substrate ships
  (kernel T4). Until then the gauntlet's evidence rides the dossier the
  registration already seals. When packs land, add
  `{"slot": "gauntlet-pass", "kind": "pack-receipt"}` with the pack named in
  the caller's `receipt_bindings: [{slot: "gauntlet-pass", pack: "quant"}]`
  — a template edit, which is a disclosed `template: stale` finding on old
  registrations, never a retroactive revocation (R5).

### Alternatives rejected

- **Folding `kill-criteria` and `review-horizon` into one slug.** Rejected:
  one is "what kills the attempt now" (numbers), the other is "when this
  registration's evidence expires" (a date). Conflating them re-invents the
  permanent kill — exactly what the no-kill-ledger decision retired.
- **A `prior-negative-results` mandatory field that blocks on match.**
  Rejected: that is the stage-0 dedupe gate re-admitted; `conclusion-citation`
  keeps the memory advisory (evidence-memory's enforcement-pinned
  never-blocking posture).
- **Demanding cross-cluster repro evidence (`clusters: [...]`) at v1.**
  Rejected: with one production cluster in routine use, the demand would
  block every registration on an artificial errand; add it when
  cross-cluster reproduction is routine.
- **Free-form spec document instead of the fields list.** Rejected: the
  kernel's completeness check is COUNTING over declared slugs; prose can
  omit silently, a slug list cannot.

**To sign:** an utterance adopting the template file — e.g. *"Sign slot 3.3:
commit specs/registration_template.json with these 11 fields and 4
prerequisite slots as drafted (or as edited)."*

---

## BONUS (draft, not one of the three slots) — starter audit template for run #10's prelude

**PREREQ GAP this fills:** the notebook-audit prelude (pipeline §1b, P2)
drafts from "the domain pack's section template", and no template `.py`
exists yet. Sign-off on this section = committing the file below so run
#10's audit prelude has a template to diff against. The template is a
percent-format `.py` whose section markers
(`# hpc-audit-section: <slug>` as first non-blank line in a `# %%` cell) are
the REQUIRED SECTION INVENTORY — structural lint checks the slugs appear as
an order-preserving subsequence; slugs are opaque to core.

Proposed home: `specs/audit_template_rv.py`. Slug inventory, derived from
the repo's actual research pattern (`run.py` → `src/data/loading.py` →
`src/features/transforms/target.py` → `src/backtest/multi_stage.py` →
`src/evaluation/{metrics,feature_cv,diebold_mariano}.py` →
`eval_sim`/friction; the verify-first notebook doctrine of
`notebooks/results/`):

| Section slug | One-line intent |
|---|---|
| `data-load` | Load declared inputs via `src/data/loading.py::load_raw_data`; every path a string literal under the declared `input_roots` (`data/`) — the executes-live lint's surface. |
| `universe-and-alignment` | Timestamp intersection of features/target, overnight-fill policy, bar-count assertion vs the declared slice (Slot 1 scope tags cited here). |
| `target-construction` | The invariant transform (`diurnal_adjust` → `sqrt` → `winsorize`), `horizon=1` shift asserted — units and causality shown, not assumed. |
| `signal-construction` | The candidate feature/functional itself — the section where the drafting LLM exercised judgment; expected to be human-required at audit. |
| `leakage-checks` | Declared assertions: causal shift present, no future timestamp reachable, embargoed fold boundaries, context bins cross-fitted (OOF). |
| `baseline` | The honest baseline reproduced live (HAR / deployed ridge config) — the number the candidate must beat, computed in the same run. |
| `backtest` | Purged walk-forward via the `MultiStageBacktest` harness — the deployment protocol, not a convenience split. |
| `metrics` | QLIKE raw-space (Duan smearing) + MSE/MAE sqrt-space + MZ — claimed units computed by `src/evaluation/metrics.py`, never re-derived inline. |
| `placebo-and-significance` | Circular-shift placebo + `significance_gate` components reported (Slot 2's numbers referenced, each condition's verdict shown). |
| `costs-and-capacity` | Friction/venue costs applied from `options-friction` data; breakeven vs instrument cost; capacity estimate feeding the registration's `capacity-estimate` field. |
| `robustness` | Train-window ablation + subperiod split — the two checks the repo's history shows catching mirages (`window_ablation`, regime studies). |

Rationale for this list over the generic eight (data-load, universe,
signal-construction, leakage-checks, backtest, costs, capacity, robustness):
this repo's recorded failure modes were **units/target bugs and
weak-baseline traps**, so `target-construction`, `baseline`, and `metrics`
earn first-class sections (they are where human suspicion historically paid
off); `costs`+`capacity` merge into one section because both consume the
same friction data in the same cell in practice.

Alternatives rejected: a per-model-family template set (premature — one
template, with `signal-construction` as the variable section, until a second
family's audits actually diverge); folding `leakage-checks` into `backtest`
(the checks are declared assertions the auto-clear tier evaluates — keeping
them a separate section lets a clean re-draft auto-clear everything except
the judgment sections).

**To sign (draft adoption, not a pipeline slot):** *"Adopt
specs/audit_template_rv.py with these 11 sections as the run-#10 audit
template."*
