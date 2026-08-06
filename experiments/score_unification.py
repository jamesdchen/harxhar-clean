"""Harvest + score the unification campaign (paper2) under the paper's smearing contract.

Reads the per-task npz trees written by ``src/unification.py`` (the executor is
the source of truth for the schema: row_id, t, y_fit, yhat, rv_raw, baseline,
valid_mask + scalar suffstats + json meta), scores every (root, arm) pair, and
emits:

  * one CSV row per (root, arm)                    (``--out``)
  * ``<tex-dir>/campaign_numbers.tex``             (scalar macros)
  * ``<tex-dir>/table_buckets.tex``                (tabular body, buckets.tex format)
  * ``<tex-dir>/table_lasso_ridge.tex``            (tabular body, dense_weak.tex format)
  * ``<tex-dir>/table_blocks.tex``                 (tabular body: ladder + diagnostics)

Scoring contract (writeup/sections/smearing.tex; reference implementation
``src/evaluation/metrics.apply_duan_smearing`` — REUSED, not re-derived): one
evaluation window == one CHUNK.  Per chunk, over valid bars,
``sigma2 = mean((y_fit - yhat)^2)``; raw forecast ``f_t = (yhat_t^2 + sigma2) * B_t``;
raw target = the PERSISTED unwinsorized ``rv_raw`` (never y_fit^2 * B).  Per-bar
QLIKE ``r - log r - 1`` with ``r = rv_raw / f``, excluding bars where either
member of the pair is non-positive — the exact metrics.py exclusion rule, via
``src/evaluation/diebold_mariano.qlike_per_bar`` (REUSED).  Pooled arm QLIKE =
mean over all included bars.

Same-environment comparisons: each arm vs ITS OWN root's ``a0_ols_har``, joined
per-bar on row_id.  DM t reuses ``src/evaluation/diebold_mariano.dm_test``
(Newey-West HAC, automatic lag floor(4*(T/100)^(2/9)), Harvey-Leybourne-Newbold
small-sample correction — the repo's established DM utility).  OOS R^2 is on
the sqrt FIT scale: 1 - SSE_arm/SSE_a0 from per-bar (y_fit - yhat)^2 on the
joined rows.  MZ alpha/beta reuse ``src/evaluation/metrics.mz_regression``
(rv_raw on the raw forecast, positive pairs).

Cross-cluster canary: ``a0_ols_har`` present in two roots is joined on row_id
and max|yhat diff| / mean|diff| / pooled-QLIKE diff are reported — the float-
parity bound between clusters.

Partial-harvest robust: runs cleanly at ANY completion level (zero files emits
all-pending macros and pending tables), always exits 0, one summary line per
(root, arm).  Float64 throughout; numpy only for the core scoring.

CLI:
    python experiments/score_unification.py \
        --roots results/unification_carc results/unification_h2 \
        --out results/unification_scores.csv --tex-dir writeup/generated
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass, field

import numpy as np

# ── repo-root bootstrap (script may be invoked from any cwd) ──────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.evaluation.diebold_mariano import dm_test, qlike_per_bar  # noqa: E402
from src.evaluation.metrics import apply_duan_smearing, mz_regression  # noqa: E402

EXPECTED_CHUNKS = 100
A0 = "a0_ols_har"
# Chunks that are illegal BY DESIGN for the 24000-bar-window product /
# transmission arms (selection/frame block + training window -> first legal
# OOS row 48000 -> chunks 0-10 excluded). These arms are complete at 89.
_LEGAL_MISSING: dict[str, set[int]] = {
    arm: set(range(11))
    for arm in (
        "blk3_user",
        "blk4_user",
        "c4_product_alone_user",
        "d3_transmission_alone_user",
    )
}
_CHUNK_RE = re.compile(r"^chunk_(\d+)\.npz$")

# ── arm registry facts (mirrors src/unification.py ARMS; static so the scorer
#    never needs the panel, the executor's heavy deps, or the cluster env) ─────
#    camel: LaTeX macro stem (\unif<camel>QLIKE / \unif<camel>DM). Names for
#    b1/b2/blk* match the pre-existing placeholder writeup/generated/
#    campaign_numbers.tex exactly (BOneRidge, BTwoLasso, BlkTwoUser, ...).

_BUCKET_TEX = {
    "a_bucket_all_features": r"all\_features (joint)",
    "a_bucket_moments": "moments",
    "a_bucket_liquidity": "liquidity",
    "a_bucket_implied_vol": r"implied\_vol",
    "a_bucket_market_vw": r"market\_vw",
    "a_bucket_market_ew": r"market\_ew",
    "a_bucket_vol_demand": r"vol\_demand",
    "a_bucket_sentiment": "sentiment",
}

# (arm, camel) in canonical registry order.
_REGISTRY: list[tuple[str, str]] = [
    (A0, "Azero"),
    ("a_bucket_moments", "Moments"),
    ("a_bucket_liquidity", "Liquidity"),
    ("a_bucket_market_ew", "MarketEw"),
    ("a_bucket_market_vw", "MarketVw"),
    ("a_bucket_sentiment", "Sentiment"),
    ("a_bucket_implied_vol", "ImpliedVol"),
    ("a_bucket_vol_demand", "VolDemand"),
    ("a_bucket_all_features", "AllFeatures"),
    ("b1_ridge", "BOneRidge"),
    ("b2_lasso", "BTwoLasso"),
    # Causally-tuned penalty controls for the head-to-head (battery protocol:
    # periodic causal forward-split re-selection; window 24000, full design).
    ("b1_ridge_tuned", "BOneRidgeTuned"),
    ("b2_lasso_tuned", "BTwoLassoTuned"),
    ("blk2_user", "BlkTwoUser"),
    ("blk3_user", "BlkThreeUser"),
    ("blk4_user", "BlkFourUser"),
    ("blk2_doc", "BlkTwoDoc"),
    ("blk3_doc", "BlkThreeDoc"),
    ("blk4_doc", "BlkFourDoc"),
    ("c4_product_alone_user", "ProductAloneUser"),
    ("c4_product_alone_doc", "ProductAloneDoc"),
    ("d3_transmission_alone_user", "TransAloneUser"),
    ("d3_transmission_alone_doc", "TransAloneDoc"),
]
_CAMEL = dict(_REGISTRY)
_ORDER = {arm: i for i, (arm, _) in enumerate(_REGISTRY)}

# blocks table: (arm, window bars, per-block alpha string) — 6 ladder + 4 diag,
# windows/alphas per src/unification.py (USER_ALPHAS/DOC_ALPHAS, DOC 250d=12000).
_BLOCKS_TABLE: list[tuple[str, int, str]] = [
    ("blk2_user", 24000, "1/100"),
    ("blk3_user", 24000, "1/100/1000"),
    ("blk4_user", 24000, "1/100/1000/1000"),
    ("blk2_doc", 12000, "1/3e3"),
    ("blk3_doc", 12000, "1/3e3/3e4"),
    ("blk4_doc", 12000, "1/3e3/3e4/3e3"),
    ("c4_product_alone_user", 24000, "1/1000"),
    ("c4_product_alone_doc", 12000, "1/3e4"),
    ("d3_transmission_alone_user", 24000, "1/1000"),
    ("d3_transmission_alone_doc", 12000, "1/3e3"),
]


# ── per-(root, arm) container ─────────────────────────────────────────────────


@dataclass
class ArmResult:
    root: str  # root label (basename of the root dir)
    arm: str
    n_chunks: int = 0
    missing: list[int] = field(default_factory=list)
    extra: list[int] = field(default_factory=list)
    incomplete: bool = True
    n_rows: int = 0
    n_valid: int = 0
    n_invalid: int = 0
    n_qlike: int = 0  # bars entering the pooled QLIKE mean
    n_dupes: int = 0
    n_gaps: int = 0
    contiguous_in_chunk: bool = True
    qlike: float | None = None
    sigma2_mean: float | None = None
    masked_col_events: int = 0
    dropped_col_events: int = 0
    dm_t: float | None = None
    oos_r2: float | None = None
    mz_alpha: float | None = None
    mz_beta: float | None = None
    mz_r2: float | None = None
    n_asym: int | None = None
    warnings: list[str] = field(default_factory=list)
    # per-bar arrays (sorted by row_id, deduped)
    row_id: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    loss: np.ndarray = field(default_factory=lambda: np.empty(0))
    err2: np.ndarray = field(default_factory=lambda: np.empty(0))
    f_raw: np.ndarray = field(default_factory=lambda: np.empty(0))
    rv_raw: np.ndarray = field(default_factory=lambda: np.empty(0))
    yhat: np.ndarray = field(default_factory=lambda: np.empty(0))


def _score_chunk(path: str) -> dict:
    """Load one chunk npz and score it as ONE evaluation window (smearing.tex)."""
    with np.load(path, allow_pickle=False) as z:
        row_id = np.asarray(z["row_id"], dtype=np.int64)
        y_fit = np.asarray(z["y_fit"], dtype=np.float64)
        yhat = np.asarray(z["yhat"], dtype=np.float64)
        rv_raw = np.asarray(z["rv_raw"], dtype=np.float64)
        baseline = np.asarray(z["baseline"], dtype=np.float64)
        valid = np.asarray(z["valid_mask"], dtype=bool)
        masked = int(z["ols_masked_cols"]) if "ols_masked_cols" in z.files else -1
        dropped = int(z["ols_dropped_cols"]) if "ols_dropped_cols" in z.files else 0
        if masked < 0 and "meta" in z.files:
            meta = json.loads(str(z["meta"]))
            mc = meta.get("ols_masked_cols", {})
            masked = len(mc) if isinstance(mc, dict) else int(mc)
        masked = max(masked, 0)

    n = len(row_id)
    f = np.full(n, np.nan)
    err2 = np.full(n, np.nan)
    if valid.any():
        # sigma2 + raw-forecast map via the reference implementation (the smear
        # is one scalar per chunk, from the chunk's own valid residuals).
        pred_raw, _ = apply_duan_smearing(yhat[valid], y_fit[valid], baseline[valid])
        f[valid] = pred_raw
        err2[valid] = (y_fit[valid] - yhat[valid]) ** 2
        sigma2 = float(np.mean((y_fit[valid] - yhat[valid]) ** 2))
    else:
        sigma2 = float("nan")
    # raw target is the persisted unwinsorized rv_raw; exclusion rule ==
    # metrics.py: bars where either member is non-positive drop out (NaN).
    loss = qlike_per_bar(f, rv_raw)
    loss[~valid] = np.nan
    return {
        "row_id": row_id,
        "loss": loss,
        "err2": err2,
        "f": f,
        "rv_raw": rv_raw,
        "yhat": yhat,
        "valid": valid,
        "sigma2": sigma2,
        "masked": masked,
        "dropped": dropped,
    }


def _harvest_arm(root_path: str, root_label: str, arm: str) -> ArmResult:
    res = ArmResult(root=root_label, arm=arm)
    arm_dir = os.path.join(root_path, arm)
    found: dict[int, str] = {}
    for fn in sorted(os.listdir(arm_dir)):
        m = _CHUNK_RE.match(fn)
        if m:
            found[int(m.group(1))] = os.path.join(arm_dir, fn)
    res.n_chunks = len(found)
    res.missing = sorted(set(range(EXPECTED_CHUNKS)) - set(found))
    res.extra = sorted(i for i in found if i >= EXPECTED_CHUNKS)
    # Structurally excluded chunks: the 24000-bar-window product/transmission
    # arms refuse rows inside the selection/frame block plus the training
    # window (first legal OOS row 48000 -> chunks 0-10 illegal BY DESIGN, per
    # src/unification.py's legality check). Those arms are complete at 89.
    legal_missing = _LEGAL_MISSING.get(res.arm, set())
    res.incomplete = bool(set(res.missing) - legal_missing)
    if not found:
        return res

    chunks = []
    sigma2s = []
    for idx in sorted(found):
        try:
            c = _score_chunk(found[idx])
        except Exception as err:  # a corrupt chunk must not sink the harvest
            res.warnings.append(f"chunk_{idx}: unreadable ({err}); skipped")
            res.incomplete = True
            if idx not in res.missing:
                res.missing.append(idx)
            continue
        if len(c["row_id"]) > 1 and not np.all(np.diff(c["row_id"]) == 1):
            res.contiguous_in_chunk = False
            res.warnings.append(f"chunk_{idx}: row_id not contiguous within chunk")
        chunks.append(c)
        sigma2s.append(c["sigma2"])
        res.masked_col_events += c["masked"]
        res.dropped_col_events += c["dropped"]
    if not chunks:
        return res

    row_id = np.concatenate([c["row_id"] for c in chunks])
    order = np.argsort(row_id, kind="stable")
    row_id = row_id[order]
    dupe = np.zeros(len(row_id), dtype=bool)
    dupe[1:] = row_id[1:] == row_id[:-1]
    res.n_dupes = int(dupe.sum())
    if res.n_dupes:
        res.warnings.append(
            f"{res.n_dupes} overlapping row_id(s) across chunks; first occurrence kept"
        )
    keep = order[~dupe]
    res.row_id = row_id[~dupe]
    res.n_gaps = int((np.diff(res.row_id) > 1).sum()) if len(res.row_id) > 1 else 0

    for name in ("loss", "err2", "f", "rv_raw", "yhat"):
        arr = np.concatenate([c[name] for c in chunks])[keep]
        setattr(res, {"f": "f_raw"}.get(name, name), arr)
    valid = np.concatenate([c["valid"] for c in chunks])[keep]

    res.n_rows = len(res.row_id)
    res.n_valid = int(valid.sum())
    res.n_invalid = res.n_rows - res.n_valid
    res.n_qlike = int(np.isfinite(res.loss).sum())
    if res.n_qlike:
        res.qlike = float(np.nanmean(res.loss))
    s2 = np.asarray(sigma2s, dtype=np.float64)
    if np.isfinite(s2).any():
        res.sigma2_mean = float(np.nanmean(s2))

    # Mincer-Zarnowitz: rv_raw on the raw forecast, positive finite pairs.
    m = np.isfinite(res.f_raw) & np.isfinite(res.rv_raw)
    m &= (res.f_raw > 0) & (res.rv_raw > 0)
    if m.sum() >= 3:
        mz = mz_regression(res.rv_raw[m], res.f_raw[m])
        res.mz_alpha, res.mz_beta = mz["alpha"], mz["beta"]
        res.mz_r2 = mz["r2"]
    return res


def _compare_vs_a0(res: ArmResult, a0: ArmResult) -> None:
    """Same-environment comparison: join per-bar on row_id vs the root's own a0."""
    common, ia, ib = np.intersect1d(
        res.row_id, a0.row_id, assume_unique=True, return_indices=True
    )
    res.n_asym = (len(res.row_id) - len(common)) + (len(a0.row_id) - len(common))
    if res.n_asym:
        res.warnings.append(
            f"row sets vs {A0} not identical: {len(res.row_id) - len(common)} "
            f"arm-only + {len(a0.row_id) - len(common)} a0-only rows; "
            "compared on the intersection"
        )
    if len(common) == 0:
        return
    dm = dm_test(res.loss[ia], a0.loss[ib], h=1)  # repo DM utility, reused exactly
    if np.isfinite(dm.get("dm", float("nan"))):
        res.dm_t = float(dm["dm"])
    both = np.isfinite(res.err2[ia]) & np.isfinite(a0.err2[ib])
    sse_a0 = float(np.sum(a0.err2[ib][both]))
    if both.any() and sse_a0 > 0:
        res.oos_r2 = 1.0 - float(np.sum(res.err2[ia][both])) / sse_a0


# ── formatting ────────────────────────────────────────────────────────────────


def _fmt(x: float | None, spec: str) -> str:
    return "" if x is None else format(x, spec)


def _tex_escape(s: str) -> str:
    return s.replace("_", r"\_")


_PENDING_CELL = r"\pending{awaiting}"


def _pick_primary(entries: list[ArmResult]) -> ArmResult | None:
    """Best (root, arm) entry for macros/tables: complete first, then most bars."""
    if not entries:
        return None
    return sorted(entries, key=lambda r: (r.incomplete, -r.n_valid))[0]


def _macro_lines(by_arm: dict[str, list[ArmResult]]) -> list[str]:
    from datetime import datetime, timezone

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%MZ")
    lines = [
        "% GENERATED FILE -- do not edit by hand.",
        f"% Written by experiments/score_unification.py at {stamp}.",
        "% Scoring: per-chunk Duan smear (smearing.tex), rv_raw evaluation",
        "% target, DM vs the same root's a0 (src/evaluation/diebold_mariano).",
        r"% \unifAzeroRsq is a0's Mincer-Zarnowitz R^2 (rv_raw on raw forecast).",
        "% Incomplete arms render as red pending boxes until harvest completes.",
    ]

    def pend(arm: str) -> str:
        return r"\pending{%s awaiting harvest}" % _tex_escape(arm)

    def emit(name: str, value: str) -> None:
        lines.append(rf"\newcommand{{\{name}}}{{{value}}}")

    a0 = _pick_primary(by_arm.get(A0, []))
    a0_done = a0 is not None and not a0.incomplete
    emit(
        "unifAzeroQLIKE",
        _fmt(a0.qlike, ".5f") if a0_done and a0.qlike is not None else pend(A0),
    )
    emit(
        "unifAzeroRsq",
        _fmt(a0.mz_r2, ".3f") if a0_done and a0.mz_r2 is not None else pend(A0),
    )
    emit(
        "unifAzeroMZbeta",
        _fmt(a0.mz_beta, ".3f") if a0_done and a0.mz_beta is not None else pend(A0),
    )

    for arm, camel in _REGISTRY:
        if arm == A0:
            continue
        r = _pick_primary(by_arm.get(arm, []))
        done = r is not None and not r.incomplete
        emit(
            f"unif{camel}QLIKE",
            _fmt(r.qlike, ".5f") if done and r.qlike is not None else pend(arm),
        )
        emit(
            f"unif{camel}DM",
            _fmt(r.dm_t, "+.1f") if done and r.dm_t is not None else pend(arm),
        )
    return lines


def _buckets_table(by_arm: dict[str, list[ArmResult]]) -> list[str]:
    a0 = _pick_primary(by_arm.get(A0, []))
    a0q = a0.qlike if a0 is not None and not a0.incomplete else None

    def row(arm: str) -> str:
        name = _BUCKET_TEX[arm]
        r = _pick_primary(by_arm.get(arm, []))
        if r is None or r.incomplete or r.qlike is None:
            return (
                f"{name} & {_PENDING_CELL} & {_PENDING_CELL} & "
                f"{_PENDING_CELL} & {_PENDING_CELL} \\\\"
            )
        q = f"{r.qlike:.5f}"
        if arm == "a_bucket_all_features":
            q = rf"\textbf{{{q}}}"
        # a0 IS the OLS class's no-exogenous baseline, so Delta_own == Delta.
        d = rf"${r.qlike - a0q:+.5f}$" if a0q is not None else _PENDING_CELL
        t = rf"${r.dm_t:+.1f}$" if r.dm_t is not None else _PENDING_CELL
        return f"{name} & {q} & {d} & {d} & {t} \\\\"

    def key(arm: str) -> tuple:
        r = _pick_primary(by_arm.get(arm, []))
        if r is None or r.incomplete or r.qlike is None:
            return (1, _ORDER[arm])
        return (0, r.qlike)

    singles = sorted((a for a in _BUCKET_TEX if a != "a_bucket_all_features"), key=key)
    a0_cell = f"{a0q:.5f}" if a0q is not None else _PENDING_CELL
    return [
        r"\toprule",
        r"Bucket & QLIKE & $\Delta$ vs.\ incumbent & $\Delta_{\text{own}}$ & DM $t$ \\",
        r"\midrule",
        row("a_bucket_all_features"),
        r"\midrule",
        *[row(a) for a in singles],
        r"\midrule",
        rf"(no exogenous $=$ incumbent \texttt{{a0}}) & {a0_cell} & --- & --- & --- \\",
        r"\bottomrule",
    ]


def _lasso_ridge_table(by_arm: dict[str, list[ArmResult]]) -> list[str]:
    a0 = _pick_primary(by_arm.get(A0, []))
    a0q = a0.qlike if a0 is not None and not a0.incomplete else None

    def cell(r: ArmResult | None, bold: bool) -> str:
        if r is None or r.incomplete or r.qlike is None:
            return _PENDING_CELL
        q = f"{r.qlike:.5f}"
        if bold:
            q = rf"\textbf{{{q}}}"
        return f"{q} (${r.dm_t:+.1f}$)" if r.dm_t is not None else q

    def pair_row(label: str, ridge_arm: str, lasso_arm: str) -> str:
        ridge = _pick_primary(by_arm.get(ridge_arm, []))
        lasso = _pick_primary(by_arm.get(lasso_arm, []))
        rq = ridge.qlike if ridge is not None and not ridge.incomplete else None
        lq = lasso.qlike if lasso is not None and not lasso.incomplete else None
        both = rq is not None and lq is not None
        return (
            label
            + " & "
            + cell(ridge, both and rq <= lq)
            + " & "
            + cell(lasso, both and lq < rq)
            + r" \\"
        )

    a0_cell = f"{a0q:.5f}" if a0q is not None else _PENDING_CELL
    return [
        r"\toprule",
        r"Bucket & Ridge & Lasso \\",
        r"\midrule",
        pair_row(r"\texttt{all\_features} (fixed penalty)", "b1_ridge", "b2_lasso"),
        pair_row(
            r"\texttt{all\_features} (causally tuned)",
            "b1_ridge_tuned",
            "b2_lasso_tuned",
        ),
        r"\midrule",
        r"Incumbent: OLS on \texttt{baseline} & "
        rf"\multicolumn{{2}}{{c}}{{{a0_cell}}} \\",
        r"\bottomrule",
    ]


def _blocks_table(by_arm: dict[str, list[ArmResult]]) -> list[str]:
    def row(arm: str, window: int, alphas: str) -> str:
        w = f"${window:,}$".replace(",", "{,}")
        r = _pick_primary(by_arm.get(arm, []))
        if r is None or r.incomplete or r.qlike is None:
            q = t = _PENDING_CELL
        else:
            q = f"{r.qlike:.5f}"
            t = rf"${r.dm_t:+.1f}$" if r.dm_t is not None else _PENDING_CELL
        return (
            rf"\texttt{{{_tex_escape(arm)}}} & {w} & "
            rf"\texttt{{{alphas}}} & {q} & {t} \\"
        )

    ladder = _BLOCKS_TABLE[:6]
    diags = _BLOCKS_TABLE[6:]
    return [
        r"\toprule",
        r"Arm & Window (bars) & $\alpha$ per block & QLIKE & DM $t$ vs.\ \texttt{a0} \\",
        r"\midrule",
        *[row(*x) for x in ladder],
        r"\midrule",
        *[row(*x) for x in diags],
        r"\bottomrule",
    ]


# ── driver ────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--roots",
        nargs="+",
        required=True,
        help="result roots, one per cluster (walked for <arm>/chunk_*.npz)",
    )
    ap.add_argument("--out", required=True, help="scores CSV path")
    ap.add_argument(
        "--tex-dir",
        default=os.path.join("writeup", "generated"),
        help="dir for campaign_numbers.tex + table bodies",
    )
    args = ap.parse_args(argv)

    # 1. DISCOVER + 2/3. VALIDATE + SCORE per (root, arm).
    labels: dict[str, str] = {}
    for root in args.roots:
        lab = os.path.basename(os.path.normpath(root))
        labels[root] = lab if lab not in labels.values() else root
    results: list[ArmResult] = []
    for root in args.roots:
        if not os.path.isdir(root):
            print(f"[{labels[root]}] root missing on disk: {root} (0 arms)")
            continue
        for arm in sorted(os.listdir(root)):
            arm_dir = os.path.join(root, arm)
            if not os.path.isdir(arm_dir):
                continue
            if not any(_CHUNK_RE.match(f) for f in os.listdir(arm_dir)):
                continue
            if arm not in _CAMEL:
                print(
                    f"[{labels[root]}] NOTE: unknown arm dir '{arm}' "
                    "(scored, CSV only, no macro/table slot)"
                )
            results.append(_harvest_arm(root, labels[root], arm))

    # 4. COMPARISONS: within each root, vs that root's own a0.
    by_root: dict[str, list[ArmResult]] = {}
    for r in results:
        by_root.setdefault(r.root, []).append(r)
    for root_label, entries in by_root.items():
        a0 = next((r for r in entries if r.arm == A0), None)
        if a0 is None or a0.n_rows == 0:
            print(f"[{root_label}] no {A0} in this root -> DM/OOS-R2 unavailable here")
            continue
        for r in entries:
            if r.n_rows:
                _compare_vs_a0(r, a0)

    # (Cross-cluster canary removed by author directive 2026-08-06: Hoffman2
    # was abandoned and the campaign is single-cluster (CARC only), so there
    # is no second environment to float-parity against. Intentional — do not
    # restore without a second cluster in the campaign.)

    # 6a. CSV.
    results.sort(
        key=lambda r: (
            args.roots.index(next(k for k, v in labels.items() if v == r.root))
            if r.root in labels.values()
            else 99,
            _ORDER.get(r.arm, 98),
            r.arm,
        )
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "arm",
                "root",
                "n_bars",
                "n_chunks_present",
                "incomplete",
                "qlike",
                "dm_t_vs_a0",
                "oos_r2_vs_a0",
                "mz_alpha",
                "mz_beta",
                "sigma2_mean",
                "masked_col_events",
            ]
        )
        for r in results:
            w.writerow(
                [
                    r.arm,
                    r.root,
                    r.n_valid,
                    r.n_chunks,
                    r.incomplete,
                    _fmt(r.qlike, ".10g"),
                    _fmt(r.dm_t, ".6g"),
                    _fmt(r.oos_r2, ".10g"),
                    _fmt(r.mz_alpha, ".10g"),
                    _fmt(r.mz_beta, ".10g"),
                    _fmt(r.sigma2_mean, ".10g"),
                    r.masked_col_events,
                ]
            )

    # 6b/c. LaTeX macros + table bodies (pending-robust at any completion level).
    by_arm: dict[str, list[ArmResult]] = {}
    for r in results:
        by_arm.setdefault(r.arm, []).append(r)
    os.makedirs(args.tex_dir, exist_ok=True)
    outputs = {
        "campaign_numbers.tex": _macro_lines(by_arm),
        "table_buckets.tex": _buckets_table(by_arm),
        "table_lasso_ridge.tex": _lasso_ridge_table(by_arm),
        "table_blocks.tex": _blocks_table(by_arm),
    }
    for name, lines in outputs.items():
        with open(os.path.join(args.tex_dir, name), "w", newline="\n") as fh:
            fh.write("\n".join(lines) + "\n")

    # 7. Summary: one line per (root, arm); missing arms listed as pending.
    print(f"\n== unification harvest summary ({len(results)} (root, arm) pairs) ==")
    seen = {(r.root, r.arm) for r in results}
    for r in results:
        miss = ""
        if r.missing:
            head = ",".join(map(str, r.missing[:8]))
            miss = f" missing=[{head}{',...' if len(r.missing) > 8 else ''}]"
        extras = f" extra_chunks={r.extra}" if r.extra else ""
        flags = "".join(
            [
                "" if r.contiguous_in_chunk else " NONCONTIG",
                f" dupes={r.n_dupes}" if r.n_dupes else "",
                f" gaps={r.n_gaps}" if r.n_gaps and not r.missing else "",
                f" invalid={r.n_invalid}" if r.n_invalid else "",
                f" asym={r.n_asym}" if r.n_asym else "",
            ]
        )
        print(
            f"[{r.root}] {r.arm:28s} chunks {r.n_chunks:3d}/{EXPECTED_CHUNKS}"
            f" n_valid={r.n_valid:7d} qlike={_fmt(r.qlike, '.5f') or '------'}"
            f" dm={_fmt(r.dm_t, '+.1f') or '----'}"
            f"{' INCOMPLETE' if r.incomplete else ''}{miss}{extras}{flags}"
        )
        for wmsg in r.warnings:
            print(f"    ! {wmsg}")
    for arm, _ in _REGISTRY:
        roots_missing = [
            lab for lab in dict.fromkeys(labels.values()) if (lab, arm) not in seen
        ]
        if len(roots_missing) == len(set(labels.values())):
            print(f"[--] {arm:28s} no chunks in any root -> pending")
    print(
        f"\nwrote {args.out} + "
        f"{', '.join(os.path.join(args.tex_dir, n) for n in outputs)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
