# `packs/` — the quant domain pack

This directory holds this lab's **domain pack** for the hpc-agent rigor stack.
A domain pack is the layer ABOVE core that NAMES what opaque content means
(which callables read data, which template the audit drafts from, …). Core
never learns that meaning: it binds the pack AS DATA (relpath + sha), carries an
opaque `{pack, version, sha}` echo on every record that used it, and gates on
named receipts — it never runs or interprets a line of pack logic. Full design:
hpc-agent `docs/design/domain-packs.md` (IMPLEMENTED).

## The pack: `quant`

The pack is named **`quant`** — the DOMAIN. This is a hard naming rule
(user-ruled 2026-07-07): the pack is NEVER `harxhar-quant` or any name
containing `harxhar`. `harxhar` is a MODEL name, not the domain. `manifest.json`
declares `"name": "quant"`, and `pack-bind` keys the journal by it, so the name
is load-bearing, not cosmetic.

### Layout (mirrors the reference `examples/packs/toy-widgets/`)

```
packs/quant/
  manifest.json          GENERATED — the sealed integrity set (files + raw-bytes shas)
  sweep.json             build RECIPE (name/version/seams/fills_slots/pack_files/sweep globs)
  build_quant_pack.py    the "sweep docs at pack build" step — regenerates manifest.json
  templates/quant_audit.py   S4 audit_template seat (the P2 section template)
  vocab/readers.json         S1 reader_calls vocabulary
  check/check_quant.py       caller-side domain check → emits a pack receipt (DP2)
  README not needed here; this file covers it
```

### Seams declared

| Seam | File | Content | Source |
|---|---|---|---|
| `audit_template` (S4) | `templates/quant_audit.py` | the section template stage P2 drafts from | SIGNED `specs/audit_template_run10.py` (see signature gate) |
| `reader_calls` (S1) | `vocab/readers.json` | `src.data.loading.load_raw_data` — the pinned data loader | the RV/run10 templates' `universe-and-alignment` / `data-selection` sections |

Seams **not yet declared** (deliberately — each needs content only a human can
sign, and inventing it is forbidden):

- `failure_patterns` (S2) — no stderr/log regexes derivable without invention.
- `axis_hints` (S3) — none derivable without invention.
- `tolerances` (S5) — the gauntlet numbers (`k=2.0`, `repl_frac=0.7`, …) are
  shipped-code DEFAULTS that pipeline_v2 **slot 3.2 leaves UNSIGNED**; they are
  gauntlet thresholds, not reproduction tolerances, so they do not belong here.
- `registration_fields` (S6) — the field list (`mechanism`, `candidate
  functional`, `claimed units`, …) is pipeline_v2 **slot 3.3, UNSIGNED**. It is
  not slugified into the pack until the user signs it (no inventing slugs).

## "Sweep docs at pack build"

Standing rule (user, 2026-07-07): the pack SWEEPS the lab's writeup docs at
build time. `domain-packs.md` defines no dedicated sweep mechanism, so the
pack-side realization is the **manifest integrity set**: `build_quant_pack.py`
gathers the `sweep.json` doc globs (`../../writeup/idea_to_trade_*.md`, the
unsigned-slots proposals), references them IN PLACE via `../../writeup/…`
relpaths, and seals their raw-bytes shas into `manifest.json`. Binding then pins
exactly which lab docs the domain standards were drafted from; edit a swept doc
and its on-disk sha no longer matches, so the next bind/gate reads drift and
revokes every clearance signed under the old standards (the same
drift-revocation core gives source code). Rebuild = re-sweep = shas move.

```
python packs/quant/build_quant_pack.py           # regenerate manifest.json
python packs/quant/build_quant_pack.py --check    # CI: fail if manifest is stale
```

## Build / verify (what passed)

`pack-bind` IS the build/verify verb — it recomputes every listed file's
raw-bytes sha server-side and refuses loudly on any drift. It binds clean:

```
hpc-agent pack-bind --spec <(echo '{"manifest":"packs/quant/manifest.json","pack":"quant"}') \
  --experiment-dir .
# -> ok: true, seams ["audit_template","reader_calls"], manifest_sha 9c69feab…
```

End-to-end (validated in a throwaway experiment dir): bind → `check_quant.py`
records the `rv-audit` receipt → `pack-status` reports the slot `current` +
`passed` → editing the sealed template flips it to `stale` with a dangling-
reference reason (drift-revocation live).

## Signature gate — the one open user decision on THIS pack

The `audit_template` seat is wired to the **SIGNED** `specs/audit_template_run10.py`
content (the committed 5-slug proving-run cut, harxhar-clean `e9ff215`) — the
working precedent. The FULL **12-slug** `specs/audit_template_rv.py` is the
intended target template but is **UNSIGNED — awaiting the user's
commit-as-signature**. It was treated as read-only here: not copied, not
committed, not wired.

The pack machinery does NOT distinguish signed vs. unsigned templates (S4 is
just a file + sha; signature is a human `append-decision` at audit time, not at
pack build), and `pack-bind` does not require a signed template. The binding
constraint is the USER RULE that `specs/audit_template_rv.py` must not be
committed until the user signs it — so this seat carries the signed run10 cut
and leaves a TODO for the swap. See the provenance header in
`templates/quant_audit.py`.

**USER DECISIONS (surfaced):**
1. **Sign `specs/audit_template_rv.py`** (commit-as-signature). Then repoint the
   `audit_template` seat + `check/check_quant.py`'s `_EXPECTED_SECTIONS` at the
   12-slug inventory and rebuild the pack.
2. **pipeline_v2 slot 3.1** (scope policy) — unblocks nothing in the pack yet.
3. **pipeline_v2 slot 3.2** (gauntlet thresholds) — required before an S5
   `tolerances` / threshold seam is honest.
4. **pipeline_v2 slot 3.3** (registration fields) — required before an S6
   `registration_fields` seam.
5. **Opt-in** — this pack is NOT wired into the live campaign `interview.json`
   (that would make the submit gate require the `rv-audit` receipt for the
   run-#10 campaign). To adopt, add a `packs` block to a repo's `interview.json`:
   `{"packs":[{"pack":"quant","manifest":"packs/quant/manifest.json",
   "receipt_bindings":[{"slot":"rv-audit","pack":"quant"}]}]}`.
