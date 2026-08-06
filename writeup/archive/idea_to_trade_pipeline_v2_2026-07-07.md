# Idea→trade pipeline v2 — the amplifier, fleshed out (2026-07-07)

Supersedes nothing; extends `idea_to_trade_system_design_2026-07-02.md` with the
decisions made 2026-07-07 while the hpc-agent rigor machinery landed. DRAFT —
LLM-drafted per the doctrine; the three `HUMAN SIGNS` slots below are yours and
this document binds nothing until they are filled in your own words.

## 1. Decisions settled since v1

1. **No kill ledger — the concept is retired, not deferred.** Two cuts, both
   user-made:
   - *Empirical* negatives ("no edge on this window") are **time-indexed
     evidence about a regime, never verdicts about an idea** — markets are
     non-stationary; a stage-0 dedupe gate would block exactly the
     re-proposals non-stationarity makes valuable.
   - *Mechanical* failures (leakage, units bugs, wrong cost models) are not
     kills at all: they are **voided tests** carrying zero information about
     the idea. Their only legitimate residue is a **new automated check** in
     the gauntlet (the live→frozen migration v1 §2 already described) —
     incident → mechanism, never incident → note.
2. **The memory architecture: ideas get no memory; evidence gets full
   memory.** Every discipline mechanism is per-attempt (registration, locks,
   gauntlet — each proposal judged fresh, because the world changes). The one
   cross-attempt store is keyed to *data*, not ideas: the look ledger counts
   every reduction against a scope tag forever, because collectively mining a
   window is meta-selection and data exhaustion does not reset with regimes.
3. **Registration spec: kept, repo-side.** One-page frozen spec per attempt,
   committed to git *before any result exists*. LLM drafts; human signs. The
   core-side kernel (spec-hash journaled, sidecar reference, ordering gate
   proving the freeze predated the results) is **trigger-gated**: build it the
   first time a "frozen" spec turns out to have been edited after results, or
   a run can't prove which spec version it executed under.
4. **The machinery that landed in hpc-agent (all experiment-agnostic —
   identity/ordering/comparison/counting over opaque content):**
   - **Scope locks** — reduction over a locked tag refused at the aggregate
     seam and pre-detach; unlock is a typed human ceremony behind the
     authorship gate, journaled permanently.
   - **Look counts** — prior reductions + distinct supersession lineages per
     scope, code-computed into every harvest brief.
   - **Reproduction receipt** — `reproduce-run` (drift-guarded re-execution
     under the recorded identity) + `verify-reproduction` (caller-tolerance
     comparator, append-only receipt). Seat: distinguishing decay-vs-bug at
     stage 8; re-score-over-recompute stays the first answer when primitives
     were persisted.
   - **Dossier export** — the sealed core-owned record trail (sidecar,
     decisions, briefs, looks, receipts, aggregates-as-bytes) with an
     integrity manifest; repo-side renderers build evidence packages FROM it.

## 1b. The prelude — idea → audited code (added 2026-07-07, user direction)

People come to the table with an **idea**, not fleshed-out experiment code.
The prelude manufactures the code under audit, notebook-first (the
Claude-Science four-component provenance shape, but with graduation GATED on
human sign-off instead of review bolted on after):

| Prelude step | Who | Mechanism |
|---|---|---|
| P1. Idea stated | **human** | free text, authorship-gated (the `goal` doctrine) |
| P2. Notebook drafted | machine | LLM drafts from the domain pack's section template (claim → shown-code → live execution → outputs w/ declared assertions) |
| P3. Audit | **human**, tiered | **TIERED sign-off (user decision 2026-07-07, the auto-mode-classifier pattern): sections code can verify (empty template-diff, zero flags, assertions green) AUTO-CLEAR — journaled as mechanical, zero human attention; sections where the drafting LLM exercised judgment (nonempty diff, flags, failed assertions) require an EFFORTFUL human sign-off engaging the section's specifics — rarity buys seriousness, no rubber-stamp fatigue.** The contract is mechanical either way: section hashes recomputed at append (un-fakeable), executes-live over declared input roots, render determinism. Full plan: hpc-agent `docs/design/notebook-audit.md` |
| P4. Graduation | machine, gated | the audited function is extracted into the entry point; the submit pipeline REFUSES an entry point not hash-linked to an audited notebook |

Boundary: the contract/lints/gate/hashing are experiment-agnostic
(identity/ordering/comparison/counting — core-eligible, being planned in
hpc-agent); the section *vocabulary* (baseline, placebo, units) and the
causality/leakage harness are the domain pack's. Stage 1 (registration) then
consumes the graduated code — the frozen spec references both the notebook
hash and the entry-point identity.

## 2. The flow, v2 (v1 §3 with the deltas applied)

| Stage | Who | v2 delta |
|---|---|---|
| 0. ~~Ledger check~~ | — | **REMOVED** (decision 1). The stage-0 question "did this die before?" is answered by the human and prose artifacts, deliberately. |
| 1. Registration | **human** | Spec additionally declares: the scope tags this attempt will reduce over, and affirms which tags stay locked. Committed to git; the commit sha is the attempt's identity. |
| 2. Construction | machine + human transform choice | unchanged; every mechanical-failure incident adds a check here (decision 1b). |
| 3. Cheap kills, local | machine | unchanged in role; composition + thresholds to be signed (§3.2). |
| 4. Scaled run | machine | runs carry `scopes` tags; anchors/placebo cells in every campaign remain a task_generator **convention** for now (agnostic declared-assertions mechanization is trigger-gated). |
| 5. Mechanical gauntlet | machine | unchanged (purged walk-forward, DM/MCS, claimed-units re-score — all caller-side). |
| 6. Verdict | **human** | the brief now carries look counts per scope; the quant pack deflates in caller code; verdict recorded via append-decision. |
| 7. Economic translation | **human** + machine re-score | unchanged. |
| 8. Trade | **human**, frozen artifacts | dossier accompanies the allocation decision; live monitoring anomalies route through `reproduce-run`/`verify-reproduction` for the decay-vs-bug call. |

## 3. HUMAN SIGNS — the three open slots (nothing binds until filled)

### 3.1 Scope policy for the RV data
> _Which slice(s) of the 30-min bar data are locked, under what tag names,
> and what the unlock criterion is (e.g. "one unlock per program, at the
> go/no-go decision")._
>
> **[unsigned]**

### 3.2 Stage-3 gauntlet composition + thresholds
> _The exact checks in the one-command cheap-kill run (spanning R² vs
> deployed base, placebo, sanity — which functions, in what order) and the
> numeric kill thresholds. Thresholds are human-set numbers; the composition
> can be LLM-drafted from `src/` for signature._
>
> **[unsigned]**

### 3.3 Registration template fields
> _Confirm/edit the field list: mechanism · candidate functional ·
> conditioning axis · claimed units · honest baseline · kill criteria ·
> scopes touched · scopes affirmed locked. Home: `specs/` in this repo,
> frozen by commit._
>
> **[unsigned]**

## 4. Adoption items (no build, no signature — just start doing them)

- Tag every hpc-agent run with its scope(s) from the first post-upgrade
  submit onward; the look ledger only counts what's tagged.
- Anchor/placebo/control cells in every campaign's task_generator (the
  convention half of v1's invariant 2; mechanized later on trigger).
- The overnight-campaign / morning-batch-verdict rhythm, formalized only by
  calendar habit.
- Proving run #10 (the first quant campaign over the new MCP/block-drive
  surface) doubles as this pipeline's first live walk of stages 1→6.

## 5. Standing triggers (build nothing until one fires)

| Felt need | Build |
|---|---|
| A "frozen" spec was edited after results / a run can't prove its spec version | the core registration kernel (hash journaled + sidecar ref + ordering gate) |
| A campaign ships without embedded controls and it costs a verdict | agnostic declared-assertions (opaque role tags + caller-authored numeric assertions core evaluates) |
| A verdict takes >30 min to assemble | the evidence-package notebook template (renders FROM the dossier bundle) |
| A dossier must be handed to someone outside this machine | `archive-dossier` to an object-lock bucket (verb built; bucket setup is one-time human) |
| Team scale makes "did we try this?" unanswerable by memory | revisit idea-indexed records — as dated priors-for-retry, never blockers |
