# ============================================================================
# QUANT PACK — S4 audit_template seat.
#
# This is the domain pack's section-template seat (idea_to_trade_pipeline_v2,
# stage P2: "the LLM drafts from the domain pack's section template"). Core
# hashes this file and tracks section drift; it never runs it.
#
# PROVENANCE / SIGNATURE GATE:
#   * Content below is sourced VERBATIM from the SIGNED, committed
#     specs/audit_template_run10.py (the 5-slug proving-run cut adopted at
#     harxhar-clean commit e9ff215). It is the working, signed precedent.
#   * The FULL 12-slug RV program template lives at specs/audit_template_rv.py
#     and is UNSIGNED — awaiting the user's commit-as-signature. When the user
#     commits it, this seat SWAPS to the 12-slug template (see packs/README.md,
#     "signature gate"), the pack is rebuilt (build_quant_pack.py), and binding
#     re-seals the new sha. Until then this seat carries the signed 5-slug cut.
#   * TODO(post-signature): repoint templates/quant_audit.py at the 12-slug
#     specs/audit_template_rv.py content once specs/audit_template_rv.py is
#     committed. Do NOT do this before the user signs.
# ============================================================================
"""Run-#10 audit template — the CUT-DOWN proving-run instance.

PURPOSE (user-ruled 2026-07-07): run #10 is an AUTOMATABILITY TRIAL — "is my
PhD-thesis research loop automatable with this copilot?" — not an edge hunt.
Five sections = the kernel of one research iteration (load -> target ->
candidate -> baseline -> score), sized so the audit prelude exercises every
gate tier without 12 sections of drafting surface.

RELATION TO specs/audit_template_rv.py: that file is the FULL RV
target-program template (12 sections, the tier hierarchy in its header) —
adopt it AFTER run #10's lessons fold in. This file's 5 slugs are an
order-preserving subset of the full inventory.

HOW THE TIERS WORK AGAINST THIS FILE (audit_view.py::_tier): a draft section
auto-clears iff it is BYTE-IDENTICAL to the template cell (post-
normalization), has zero lint flags, and carries zero `assert` statements
(no receipt emitter in the demo env — an assertion without a receipt is
never green). Therefore:
  * the three PINNED cells below are COMPLETE, RUNNABLE, ASSERT-FREE code —
    the draft copies them verbatim and they auto-clear;
  * the two VARIABLE cells are prose the draft REPLACES — they diff and
    route to human sign-off. That split is the run's tier evidence.

THE VARIABLE-INTERFACE CONTRACT (what lets the pinned metrics cell be
verbatim): by the end of `baseline`, the draft must have defined three
aligned 1-D arrays on the RAW-VARIANCE scale —
    pred_raw           candidate forecasts (Duan-smeared)
    true_raw           realizations
    baseline_pred_raw  the deployed-baseline forecasts (same rows)

WHAT THE DRAFT SHOULD BE: a real-but-small KNOWN-ANSWER analysis —
reproduce the deployed baseline's QLIKE and score ONE existing exog family
against it (e.g. market_ew). Real enough to test drafting fidelity; known
enough to verify without judgment.

AUDIT CONFIGURATION: input_roots ["data/"], source_roots ["src/"].
VERDICT BOUNDARY: sections show evidence; no section states a conclusion.
RECEIPTS: emitter plugin not in the demo env — evidence is printed/shown,
never assert-gated (an `assert` in any pinned cell would kill its
auto-clear; see above).
"""

# %%
# Path bootstrap (PREAMBLE — outside every audit section by design; interactive
# kernels default cwd to this file's directory, and `src.*` imports need the
# repo root). Walks up from the file (or cwd when __file__ is absent, e.g. an
# interactive cell) to the first directory containing src/.
import sys
from pathlib import Path

import os

_START = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
_ROOT = _START
while not (_ROOT / "src").is_dir() and _ROOT != _ROOT.parent:
    _ROOT = _ROOT.parent
if (_ROOT / "src").is_dir():  # only normalize when the repo root was found
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    os.chdir(_ROOT)  # relative data literals ("data") resolve from the root,
    # and stay bare literals for the executes-live lint's surface.

# %%
# hpc-audit-section: data-selection
# [PINNED — copy verbatim; auto-clears. The pinned loader, path literal
# under input_roots, shape shown as evidence.]
from src.data.loading import load_raw_data

DATA_PATH = "data"
df = load_raw_data(DATA_PATH, allow_missing=True)
print(f"loaded: rows={len(df)} cols={len(df.columns)}")

# %%
# hpc-audit-section: target-construction
# [PINNED — copy verbatim; auto-clears. Mirrors the production call
# (src/backtest/executor.py prepare step): the invariant transform CALLED,
# never re-derived — diurnal -> sqrt -> winsorize, is_target=True; the
# multiplicative baseline kept for raw-space reconstruction.]
from src.features.transforms.target import robust_transform

adj_rv, rv_baseline = robust_transform(df, "RV", is_target=True)
df["adj_RV"] = adj_rv
df["baseline"] = rv_baseline
print(f"target adj_RV: non-null={int(adj_rv.notna().sum())} (diurnal->sqrt->winsorize, is_target=True)")

# %%
# hpc-audit-section: feature-construction
# [VARIABLE — REPLACE this cell body; it will diff and route to human
# sign-off. For run #10: ONE existing exog family (e.g. the market_ew
# moment columns), transformed the production way (robust_transform per
# column, diurnal_mode as configured), assembled with the horizon-1 shift
# (apply_horizon_shift — features at t, target at t+1; show the shift).
# A candidate walk-forward forecast comes from the pinned engine
# (src/backtest/multi_stage.py::MultiStageBacktest) — no hand-rolled loop.
# END STATE: contributes to pred_raw / true_raw per the header contract.]

# %%
# hpc-audit-section: baseline
# [VARIABLE-BY-CONTENT, PINNED-BY-IDENTITY — will diff; human sign-off
# expected. The honest baseline reproduced LIVE in this run: the deployed
# ridge config (configs/ridge_market_ew_prod.yaml — cite its commit sha in
# the cell), same walk-forward engine, never a number quoted from results/.
# END STATE: pred_raw, true_raw, baseline_pred_raw all defined — three
# aligned 1-D raw-variance arrays (the header contract) — and the
# baseline's QLIKE printed for the known-answer check.]

# %%
# hpc-audit-section: metrics
# [PINNED — copy verbatim; auto-clears GIVEN the interface contract.
# Claimed units computed by src/evaluation/metrics.py, never inline:
# QLIKE raw-space (primary), hmse/oos_r2 (level-robust), MZ calibration.
# A NameError below means the variable sections above were not filled —
# DELIBERATE: an unfilled draft must die here, never emit a vacuous report.]
from src.evaluation.metrics import forecast_metrics

report = forecast_metrics(pred_raw, true_raw, benchmark=baseline_pred_raw)
print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in report.items()})
