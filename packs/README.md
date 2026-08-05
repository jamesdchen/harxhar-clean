# `packs/` — the two-layer quant pack stack

This directory holds this lab's **domain packs** for the hpc-agent rigor stack.
A domain pack is a layer ABOVE core that NAMES what opaque content means (which
callables read data, which template the audit drafts from, …). Core never learns
that meaning: it binds a pack AS DATA (relpath + sha), carries an opaque
`{pack, version, sha}` echo on every record that used it, and gates on named
receipts — it never runs or interprets a line of pack logic. Full design:
hpc-agent `docs/design/domain-packs.md` (IMPLEMENTED). Multiple packs per
interview are supported (the `packs` block is a list; `receipt_bindings` compose
across packs), and this stack ships TWO.

## The layers (v0.2.0 — the three-into-two split, user-ruled 2026-07-10)

The four-layer hierarchy is **core / quant domain / per-lab archive / target
program**. v0.1.0 conflated the domain and program layers in one `quant` pack:
the concrete realized-volatility template, the pinned reader, and the swept lab
docs all sat in the DOMAIN seat. v0.2.0 splits them so the domain pack contains
"only the specificity necessary to generalize a quant workflow."

| Pack | Layer | Contains | Portability |
|---|---|---|---|
| **`quant`** | DOMAIN (reusable methodology) | the research-content-free audit SKELETON (`templates/quant_skeleton.py`, five section contracts as prose) + the structural CHECK (`check/check_quant.py`) | reusable by ANY quant lab — **zero references** to this repo's symbols, docs, target, or RV |
| **`rv`** | TARGET PROGRAM (this lab's realized-volatility program) | the concrete audit template (`templates/rv_audit.py`, the signed run-10 content), the pinned data-loader vocab (`vocab/readers.json`), and the swept lab writeup docs | lab-specific by design; it IS the program |

### Why TWO packs, not three (the YAGNI note)

The hierarchy names FOUR layers, but the lower two — the per-lab living archive
and the target program — are NOT split into separate packs here. Splitting them
buys nothing until a **second target program** exists to share the lab archive
between; today there is exactly one program (rv), so the archive/program boundary
has no consumer. We deliberately keep both in the single `rv` pack and record the
YAGNI choice here: introduce the split when a second program lands, not before.

### The portability test for the quant pack

The domain pack passes iff you could hand `packs/quant/` to an unrelated quant
lab and it would be useful without edits. Concretely: no symbol, path, filename,
config name, transform name, metric name, or doc reference from THIS repo appears
anywhere under `packs/quant/`. The skeleton states CONTRACTS ("call the pinned
loader", "reproduce the baseline live, cite its config sha", "metrics come from
the lab's metrics module"), never a lab's concrete cell. The one interface it
fixes is the three-array shape (`pred_raw` / `true_raw` / `baseline_pred_raw`);
see the naming note below. `check/check_quant.py` verifies the FIVE-slug section
inventory as an order-preserving presence — structure, never research content —
and is parameterized on the ACTIVE template path (default: the rv pack's
template).

## Layout

```
packs/
  README.md                        this file
  quant/                           DOMAIN LAYER (v0.2.0)
    .gitattributes                 `* -text` — pin raw bytes so shas don't drift on checkout
    manifest.json                  GENERATED — the sealed integrity set (skeleton + check)
    sweep.json                     build RECIPE (empty sweep; no lab docs at this layer)
    build_quant_pack.py            regenerates manifest.json
    templates/quant_skeleton.py    S4 audit_template seat — five section CONTRACTS, no code bodies
    check/check_quant.py           caller-side structural check → emits the `quant-audit` receipt
  rv/                              TARGET-PROGRAM LAYER (v0.2.0)
    .gitattributes                 `* -text`
    manifest.json                  GENERATED — seals template + vocab + the swept lab docs
    sweep.json                     build RECIPE (reader seam + swept ../../writeup/ globs)
    build_rv_pack.py               regenerates manifest.json (keeps the "sweep docs at build" flow)
    templates/rv_audit.py          S4 audit_template seat — the concrete run-10 template (audit-facing)
    vocab/readers.json             S1 reader_calls vocabulary — the pinned data loader
```

## Seams and slots

| Pack | Seams declared | Fills slot | Notes |
|---|---|---|---|
| `quant` | `audit_template` → `templates/quant_skeleton.py` | `quant-audit` | the slot names the DOMAIN clearance; program identity rides the checked template's sha echo on the receipt |
| `rv` | `audit_template` → `templates/rv_audit.py`, `reader_calls` → `vocab/readers.json` | *(none)* | the AUDIT-FACING template is the rv pack's — audits reference `packs/rv/templates/rv_audit.py`, so the `{pack: rv, version, sha}` echo lands on sidecars. `fills_slots: []`: the rv pack fills no slot of its own in v0.2.0; the quant-audit clearance is quant's |

The `quant-audit` receipt is recorded under the **quant** pack's bind but its
`checked` list points at the **rv** pack's active template — so editing the rv
template flips the quant-audit slot STALE (drift-revocation across the layer
boundary; verified end-to-end).

Seams **not yet declared** (deliberately — each needs content only a human can
sign, and inventing it is forbidden): `failure_patterns` (S2), `axis_hints` (S3),
`tolerances` (S5), `registration_fields` (S6). See the user decisions below.

## "Sweep docs at pack build"

Standing rule (user, 2026-07-07): the PROGRAM pack sweeps the lab's writeup docs
at build time. `domain-packs.md` defines no dedicated sweep mechanism, so the
pack-side realization is the **manifest integrity set**: `build_rv_pack.py`
gathers the `sweep.json` doc globs (`../../writeup/idea_to_trade_*.md`, the
unsigned-slots proposals), references them IN PLACE via `../../writeup/…`
relpaths, and seals their raw-bytes shas into `packs/rv/manifest.json`. Binding
then pins exactly which lab docs the program standards were drafted from; edit a
swept doc and its on-disk sha no longer matches, so the next bind/gate reads
drift and revokes every clearance signed under the old standards. Rebuild =
re-sweep = shas move. The DOMAIN pack sweeps NO docs (a reusable methodology has
no lab docs to pin); its `sweep.json` carries an empty `sweep` list.

```
python packs/rv/build_rv_pack.py            # regenerate rv manifest
python packs/rv/build_rv_pack.py --check    # CI: fail if stale
python packs/quant/build_quant_pack.py      # regenerate quant manifest
python packs/quant/build_quant_pack.py --check
```

## Naming rules (hard, user-ruled)

- The domain pack is **`quant`** — the DOMAIN. The program pack is **`rv`** — the
  realized-volatility TARGET PROGRAM.
- **NEVER** name any pack (or seam, or slot) with a name containing **`harxhar`**.
  `harxhar` is a MODEL name, not a domain and not a program. This is load-bearing:
  `manifest.json`'s `name` keys the journal path (`.hpc/packs/<name>.decisions.jsonl`).
- The interface arrays keep domain-neutral names (`pred_raw` / `true_raw` /
  `baseline_pred_raw`). "raw" = the target's own untransformed scale, which exists
  for any quant target; only the PROSE was generalized from the signed template's
  "raw-variance" (variance is rv-specific) to "raw (untransformed target) scale".

## Build / verify (what passed, v0.2.0)

Both packs bind clean and the `quant-audit` slot reads current+passed:

```
hpc-agent pack-bind --spec <bind_quant.json> --experiment-dir .   # ok: true, seams ["audit_template"]
hpc-agent pack-bind --spec <bind_rv.json>    --experiment-dir .   # ok: true, seams ["audit_template","reader_calls"]
python packs/quant/check/check_quant.py --experiment-dir .        # records slot quant-audit, passed: true
hpc-agent pack-status --spec {"pack":"quant"} --experiment-dir .  # slot quant-audit: status current, passed true
hpc-agent pack-status --spec {"pack":"rv"}    --experiment-dir .  # bound; slots [] ; no dangling
```

End-to-end (validated in a throwaway experiment dir): bind both → check → both
`pack-status` current (quant-audit passed; rv bound with no slot) → the submit
GATE (`assert_pack_receipts_current`) PASSES → emptying the quant receipts makes
the gate REFUSE naming `quant-audit` (proof the gate reads the NEW slot) →
editing `rv_audit.py` flips the quant-audit slot to STALE and the gate refuses
(drift-revocation live, across the layer boundary).

## Signature gate — the one open user decision on this stack

The `rv` pack's `audit_template` seat carries the **SIGNED** run-10 content
(`specs/audit_template_run10.py`, harxhar-clean `e9ff215`) — the working
precedent, moved verbatim from the v0.1.0 quant seat. The FULL **12-slug**
`specs/audit_template_rv.py` is the intended target template but is **UNSIGNED —
awaiting the user's commit-as-signature**. It is treated as read-only: not
copied, not committed, not wired.

The pack machinery does NOT distinguish signed vs. unsigned templates (S4 is just
a file + sha; signature is a human `append-decision` at audit time, not at pack
build), and `pack-bind` does not require a signed template. The binding
constraint is the USER RULE that `specs/audit_template_rv.py` must not be
committed until the user signs it. When the user signs it, **TWO things move
together**:

1. the `rv` pack's `templates/rv_audit.py` seat SWAPS to the 12-slug inventory
   (repoint content + rebuild `packs/rv/`), AND
2. the `quant` pack's `templates/quant_skeleton.py` GROWS the 12-slug section
   inventory (the reusable superset of today's 5-slug kernel), and
   `check/check_quant.py`'s `_EXPECTED_SECTIONS` grows to match, then rebuild
   `packs/quant/`.

See the provenance headers in both template files.

**USER DECISIONS (surfaced):**
1. **Sign `specs/audit_template_rv.py`** (commit-as-signature) → do the two-part
   12-slug swap above.
2. **pipeline_v2 slot 3.1** (scope policy) — unblocks nothing in the packs yet.
3. **pipeline_v2 slot 3.2** (gauntlet thresholds) — required before an S5
   `tolerances` seam is honest.
4. **pipeline_v2 slot 3.3** (registration fields) — required before an S6
   `registration_fields` seam.
5. **Opt-in** — both packs ARE wired into this repo's `interview.json` (`packs`
   block lists quant + rv; `receipt_bindings: [{slot: "quant-audit", pack:
   "quant"}]`), so the submit gate requires a current `quant-audit` receipt.

## Distribution model — SUPERSEDED note (2026-07-10 user ruling)

The sibling layout above is now transitional. The ruled three-tier model
(recorded in hpc-agent `docs/design/domain-packs.md` drift log, 2026-07-10):
the `quant` DOMAIN pack ships upstream in the hpc-agent repo; each lab's
distro carries its own lab bindings (this repo's `rv` = the working
precedent, a lab fork of the skeleton); each experiment's `.hpc/` gets the
lab pack MATERIALIZED at setup, pinning the skeleton's sections per
experiment. Program templates are DERIVATIVES of the domain skeletons —
`rv_audit.py` gains a `derived_from` record naming the skeleton version+sha
it instantiates (the quant-audit receipt is already the conformance
attestation). Migration rides the post-run-#12 batch; nothing here moves
mid-run.
