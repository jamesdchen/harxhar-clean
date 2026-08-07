"""Exog-conditional second moment, nesting the scalar contract (2026-08-07).

Layer-3 of the scoring contract is the causal SCALAR variance (prev-window
mean e^2 of calibrated sqrt-space residuals). This sidecar tests whether the
EXOGENOUS STATE predicts the conditional second moment, built so the proven
scalar contract is the INFINITE-SHRINKAGE LIMIT of the model — the two prior
conditional-variance failures are structurally unreachable:

  * affine+floor collapse — no floors, no clips anywhere here;
  * log-linear Duan 1e44 extrapolation — the exp() path is (i) causally
    level-recalibrated per window against the scalar contract's own level,
    and (ii) BYPASSED ENTIRELY whenever the tuner selects alpha=inf, where
    the assembled variance IS the scalar sigma2_k by definition.

DESIGN DECISIONS (documented per directive):
  * Calibration chain: REUSED from experiments/score_unification (_load_chunk,
    _ols2, _y_raw_of, _burnin_state) with the exact _score_chunk_causal rules
    mirrored — prev-window MZ (a,b) with the >=3-row/identity fallback,
    m = a + b*yhat, e = y_raw - m; the first present window is the burn-in.
  * z-target: z_t = log(e_t^2). Exact zeros are guarded by the SMALLEST
    POSITIVE e^2 in the training window — a data-derived floor for the log
    argument only (an exact-zero residual is a measure-zero float event; the
    smallest observed positive square is the natural resolution limit of the
    window, not a tuned epsilon).
  * Design (--design, three modes; see LEVEL-CHANNEL CONTROL below):
      exog     (default) the SAME standardized exogenous columns the tuned
               penalized linears use, BACKBONE EXCLUDED — variance
               conditioning is on the exog state; HAR-level persistence is
               already served by the scalar via the window mean, and
               re-feeding it here would let the variance model re-learn the
               level channel instead of the conditional shape;
      backbone the HAR ladder + session-edge interactions + calendar columns
               ONLY (_backbone_cols; the a0 incumbent design) — the LEVEL
               CHANNEL on its own, i.e. the control arm;
      both     the full wide design (backbone + exog), the union.
    The three selections partition the panel: exog and backbone are DISJOINT
    by construction (kinds {value,indicator} vs {har,regime,calendar}) and
    their union is every column, which is exactly 'both'.
  * Estimator: rolling ridge, window 24000, PER-BAR refit off rank-1-rolled
    sufficient statistics; penalty re-selected every TUNE_PER=250 solves on
    the standard forward split (25-embargo / 125-tail); grid =
    logspace(2, 6, 5) UNION {inf}. alpha=inf IS the scalar contract:
    predicted z = training-window mean of z, no ridge solve, and assembly
    uses sigma2_scalar directly (exact nesting).
  * Selection criterion: FINAL QLIKE of f = (m^2 + sigma2)B on the tail bars
    — we select on what we score, not on z-MSE.
  * Assembly: v_t = exp(zhat_t) * c_k with c_k = prev-window mean(e^2) /
    prev-window mean(exp(zhat)) — a causal multiplicative recalibration that
    pins each window's variance LEVEL to the scalar contract's level; only
    the within-window SHAPE is conditional. At inf-selected bars v_t is the
    scalar sigma2_k itself (see above).
  * alpha=inf serialization: the JSON trajectory stores the STRING 'inf'
    (float('inf') is not valid JSON; a numeric sentinel would collide with
    the real grid).

LEVEL-CHANNEL CONTROL (2026-08-07). The decomposition question is: does the
exogenous state predict forecast uncertainty BEYOND the volatility level it
proxies? Answering it needs the level channel run on its own, hence
--design backbone; the exog-vs-backbone contrast is the decomposition claim
and the backbone-vs-scalar contrast asks whether level-conditioning buys
anything at all.

OUTPUT-PATH ASYMMETRY (deliberate, NOT a bug). --design exog writes to
    <output-dir>/<arm>/chunk_XXX.npz            (NO design tag)
while every other design writes to
    <output-dir>/<arm>__<design>/chunk_XXX.npz  (tagged).
The exog banks were already in flight on CARC under the untagged path when
this flag was generalized; retagging them would have orphaned live output.
exog is the contract default, so the untagged path IS the exog path — read
the absence of a tag as '__exog'. score_unification.py's _CONDVAR_VARIANTS
encodes the same rule (empty dir suffix for the exog variants).

Chunking mirrors the arm's (run_unification chunk_bounds). A chunk is
FEASIBLE when the variance model has a full 24000-bar residual window before
the PREVIOUS chunk's start (the previous chunk's zhat is needed for c_k) —
earlier chunks are a burn-in SKIP (exit 0, nothing written; consistent with
the calibration burn-in convention).

CHAIN RULE (fixed 2026-08-07). LEADING chunk absence is LEGAL: the 24000-bar
product/transmission arms (blk4_trail_tuned and kin) are COMPLETE at 91/100
because chunks 0..8 never exist — their first scorable row lies past the
frozen selection/frame window (score_unification._LEGAL_MISSING = range(9)).
The chain is therefore required contiguous from the arm's OWN FIRST PRESENT
CHUNK to the target; a target below that is a structural SKIP (exit 0), the
first present chunk itself is a burn-in SKIP (no previous chunk to pin the
level against), and any INTERIOR hole still fails loudly — that one is a
genuinely broken causal chain.

Usage (one task = one (arm, chunk, design)):
    python experiments/variance_sidecar.py --arm a0_ols_har --chunk-index 20
    python experiments/variance_sidecar.py --arm a0_ols_har --chunk-index 20 \
        --design backbone
"""

import _bootstrap  # noqa: F401  (sys.path: repo root + experiments/)

import argparse
import json
import os
import re

import numpy as np

from run_unification import chunk_bounds
from score_unification import _burnin_state, _load_chunk, _ols2, _y_raw_of
from src.evaluation.diebold_mariano import qlike_per_bar
from src.models.reclasso_har import forward_window_split
from src.models.rolling_least_squares import RollingLeastSquares
from src.unification import (
    DEFAULT_WINDOW_BARS,
    EMBARGO,
    TUNE_PER,
    VAL_TAIL,
    _backbone_cols,
    _exog_all_cols,
    _load_panel,
    panel_length,
)

VS_WINDOW = DEFAULT_WINDOW_BARS  # 24000-bar rolling variance-model window
# logspace(2,6,5) = 1e2, 1e3, 1e4, 1e5, 1e6; inf = the scalar contract
ALPHA_GRID: tuple[float, ...] = tuple(float(a) for a in np.logspace(2, 6, 5)) + (
    float("inf"),
)

DESIGNS: tuple[str, ...] = ("exog", "backbone", "both")
DESIGN_DEFAULT = "exog"  # the contract design; also the UNTAGGED output path


def design_cols(names: list[str], design: str) -> np.ndarray:
    """Column indices of the requested conditioning design (see docstring).

    exog / backbone are disjoint by the src.unification column taxonomy
    ({value,indicator} vs {har,regime,calendar}); 'both' is their union,
    which is every panel column."""
    if design == "exog":
        return _exog_all_cols(names)
    if design == "backbone":
        return _backbone_cols(names)
    if design == "both":
        return np.arange(len(names), dtype=np.int64)
    raise ValueError(f"unknown design {design!r}; expected one of {DESIGNS}")


def design_out_dir(output_root: str, arm: str, design: str) -> str:
    """Output dir for (arm, design). exog keeps the UNTAGGED historical path
    (in-flight CARC banks); every other design is tagged. See the module
    docstring's OUTPUT-PATH ASYMMETRY note — this is deliberate."""
    if design == DESIGN_DEFAULT:
        return os.path.join(output_root, arm)
    return os.path.join(output_root, f"{arm}__{design}")


def _chain(paths: list[str]) -> dict:
    """Concatenated per-bar chain over contiguous chunks 0..i — the exact
    _score_chunk_causal calibration rules (see module docstring)."""
    acc: dict[str, list[np.ndarray]] = {
        k: [] for k in ("row_id", "yhat", "y_raw", "m", "valid", "baseline", "rv_raw")
    }
    win: list[tuple[int, int]] = []
    prev: dict | None = None
    pos = 0
    for path in paths:
        c = _load_chunk(path)
        y_raw = _y_raw_of(c["rv_raw"], c["baseline"])
        if prev is None:
            m = _burnin_state(c)["m"]  # burn-in window: its own in-sample map
        else:
            a, b = 0.0, 1.0
            s = prev["valid"] & np.isfinite(prev["y_raw"]) & np.isfinite(prev["yhat"])
            if s.sum() >= 3:
                fit = _ols2(prev["yhat"][s], prev["y_raw"][s])
                if fit is not None:
                    a, b = fit
            m = a + b * c["yhat"]
        n = len(y_raw)
        for k, v in (
            ("row_id", c["row_id"]),
            ("yhat", c["yhat"]),
            ("y_raw", y_raw),
            ("m", m),
            ("valid", c["valid"]),
            ("baseline", c["baseline"]),
            ("rv_raw", c["rv_raw"]),
        ):
            acc[k].append(v)
        win.append((pos, pos + n))
        pos += n
        prev = {"yhat": c["yhat"], "y_raw": y_raw, "valid": c["valid"], "m": m}
    out = {k: np.concatenate(v) for k, v in acc.items()}
    out["e"] = out["y_raw"] - out["m"]
    out["win"] = win
    return out


def _window_sel(ch: dict, lo: int, hi: int) -> np.ndarray:
    """The scorer's prev-window selection mask, on chain indices [lo, hi)."""
    s = slice(lo, hi)
    return (
        ch["valid"][s]
        & np.isfinite(ch["y_raw"][s])
        & np.isfinite(ch["yhat"][s])
        & np.isfinite(ch["m"][s])
    )


def _zlog(e2: np.ndarray) -> np.ndarray:
    """log(e^2) with exact zeros lifted to the smallest POSITIVE e^2 in the
    array (data-derived resolution limit of the window; NOT a tuned epsilon)."""
    e2 = np.asarray(e2, dtype=np.float64).copy()
    pos = e2[e2 > 0]
    if pos.size:
        e2[e2 == 0] = pos.min()
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(e2)


def _tail_qlike(
    v_tail: np.ndarray, m_t: np.ndarray, b_t: np.ndarray, rv_t: np.ndarray
) -> float:
    f = (m_t**2 + v_tail) * b_t
    q = qlike_per_bar(f, rv_t)
    return float(np.nanmean(q)) if np.isfinite(q).any() else float("inf")


def variance_walk(
    x_mat: np.ndarray,
    z: np.ndarray,
    ch_m: np.ndarray,
    ch_b: np.ndarray,
    ch_rv: np.ndarray,
    first: int,
    last: int,
) -> tuple[np.ndarray, list[dict]]:
    """Per-bar-refit rolling ridge on (x, z) with causal alpha reselection.

    Emits zhat for sub-series indices [first, last) plus the selection
    trajectory [{idx, alpha}]. alpha=inf -> zhat = training-window mean of z
    (no solve). Selection every TUNE_PER solves on the trailing window's
    forward split, argmin of the FINAL-QLIKE of the assembled tail forecast
    (within-window causal recalibration against the fit block).
    """
    n_cols = x_mat.shape[1]
    solver = RollingLeastSquares(alpha=0.0, fit_intercept=True)  # stats carrier
    solver.init_window(x_mat[first - VS_WINDOW : first], z[first - VS_WINDOW : first])
    zhat = np.full(len(z), np.nan)
    traj: list[dict] = []
    alpha = float("inf")
    n = float(VS_WINDOW)
    for step, t in enumerate(range(first, last)):
        if step % TUNE_PER == 0:
            w0 = t - VS_WINDOW
            fl, fh, vl, vh = forward_window_split(
                VS_WINDOW, VS_WINDOW, VAL_TAIL, EMBARGO
            )
            xf = x_mat[w0 + fl : w0 + fh]
            zf = z[w0 + fl : w0 + fh]
            xv = x_mat[w0 + vl : w0 + vh]
            e2f = np.exp(zf)  # fit-block realized e^2 (z is its log)
            mu_x, mu_z = xf.mean(0), float(zf.mean())
            xc = xf - mu_x
            gram = xc.T @ xc
            rhs = xc.T @ (zf - mu_z)
            best: tuple[float, float] | None = None
            for a in ALPHA_GRID:
                if np.isinf(a):
                    v_tail = np.full(vh - vl, float(np.mean(e2f)))  # the scalar
                else:
                    g = gram.copy()
                    g[np.diag_indices_from(g)] += a
                    coef = np.linalg.solve(g, rhs)
                    z_fit_hat = xc @ coef + mu_z
                    z_tail_hat = (xv - mu_x) @ coef + mu_z
                    c_fit = float(np.mean(e2f)) / float(np.mean(np.exp(z_fit_hat)))
                    v_tail = np.exp(z_tail_hat) * c_fit
                q = _tail_qlike(
                    v_tail,
                    ch_m[w0 + vl : w0 + vh],
                    ch_b[w0 + vl : w0 + vh],
                    ch_rv[w0 + vl : w0 + vh],
                )
                if best is None or q < best[0]:
                    best = (q, a)
            alpha = best[1]
            traj.append({"idx": t, "alpha": alpha})
        gram_c = solver._Sxx - np.outer(solver._sx, solver._sx) / n
        rhs_c = solver._Sxy - solver._sx * (solver._sy / n)
        if np.isinf(alpha):
            zhat[t] = solver._sy / n  # training-window mean of z; no solve
        else:
            g = gram_c
            g = g + alpha * np.eye(n_cols)
            coef = np.linalg.solve(g, rhs_c)
            intercept = solver._sy / n - float(solver._sx @ coef) / n
            zhat[t] = float(x_mat[t] @ coef + intercept)
        solver.roll(x_mat[t], z[t], x_mat[t - VS_WINDOW], z[t - VS_WINDOW])
    return zhat, traj


def assemble_sigma2(
    zhat: np.ndarray,
    e2: np.ndarray,
    sel_prev: np.ndarray,
    prev_lo: int,
    prev_hi: int,
    out_lo: int,
    out_hi: int,
    inf_mask: np.ndarray,
) -> np.ndarray:
    """Causal window-level recalibration (see module docstring): for the
    output window, v_t = exp(zhat_t) * c with c = mean(e^2 over the previous
    window's contract-selected rows) / mean(exp(zhat) over the same rows);
    inf-selected bars carry the scalar sigma2 = mean(prev-window e^2) exactly."""
    e2_prev = e2[prev_lo:prev_hi][sel_prev]
    r_prev = np.exp(zhat[prev_lo:prev_hi][sel_prev])
    sigma2_scalar = float(np.mean(e2_prev))
    c_k = sigma2_scalar / float(np.mean(r_prev))
    v = np.exp(zhat[out_lo:out_hi]) * c_k
    v[inf_mask[out_lo:out_hi]] = sigma2_scalar
    return v


_CHUNK_RE = re.compile(r"^chunk_(\d{3})\.npz$")


def present_chunks(results_dir: str, arm: str) -> list[int]:
    """Sorted chunk indices actually on disk for ``arm``."""
    d = os.path.join(results_dir, arm)
    if not os.path.isdir(d):
        return []
    out = []
    for fn in os.listdir(d):
        m = _CHUNK_RE.match(fn)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def chain_paths(results_dir: str, arm: str, idx: int) -> list[str]:
    """Contiguous chunk paths [first_present .. idx] of ``arm``.

    LEADING absence is LEGAL (score_unification._LEGAL_MISSING): the 24000-bar
    product/transmission arms have no chunks 0..8 at all — their first scorable
    row lies past the frozen selection/frame window, so those chunks are
    structurally absent, not lost. Contiguity is therefore required from the
    arm's OWN FIRST PRESENT CHUNK, not from chunk 0.

    Returns [] when ``idx`` precedes the first present chunk (the caller treats
    that as a structural SKIP). An INTERIOR hole — any gap between the first
    present chunk and ``idx``, ``idx`` itself included — is a genuinely broken
    causal chain and still raises loudly.
    """
    present = present_chunks(results_dir, arm)
    if not present:
        raise SystemExit(
            f"chain broken: no chunk_*.npz under "
            f"{os.path.join(results_dir, arm)} — nothing to chain"
        )
    first = present[0]
    if idx < first:
        return []
    have = set(present)
    missing = [k for k in range(first, idx + 1) if k not in have]
    if missing:
        raise SystemExit(
            f"chain broken: chunk(s) {missing} of '{arm}' absent inside "
            f"[{first}..{idx}] — the sidecar needs an unbroken chain from the "
            f"arm's first present chunk ({first}) to {idx}"
        )
    return [
        os.path.join(results_dir, arm, f"chunk_{k:03d}.npz")
        for k in range(first, idx + 1)
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", required=True)
    ap.add_argument("--chunk-index", type=int, required=True)
    ap.add_argument("--results-dir", default=os.path.join("results", "unification"))
    ap.add_argument("--output-dir", default=os.path.join("results", "variance_sidecar"))
    ap.add_argument(
        "--design",
        choices=DESIGNS,
        default=DESIGN_DEFAULT,
        help="conditioning design: exog (contract, untagged output path) | "
        "backbone (level-channel control) | both (wide union)",
    )
    args = ap.parse_args()
    idx = args.chunk_index

    n_rows = panel_length()
    paths = chain_paths(args.results_dir, args.arm, idx)
    if not paths:
        print(
            f"SKIP structural chunk {idx}: '{args.arm}' has no chunk {idx} "
            "(it precedes the arm's first present chunk — legal leading "
            "absence, see chain_paths)"
        )
        return
    lo_prev, _ = chunk_bounds(idx - 1, n_rows) if idx > 0 else (None, None)
    lo_i, hi_i = chunk_bounds(idx, n_rows)

    ch = _chain(paths)
    sub0 = int(ch["row_id"][0])  # global row of sub-series index 0
    # len(paths) == 1 -> idx IS the arm's first present chunk: there is no
    # previous chunk to pin the level against, so it is a burn-in like idx 0.
    if idx == 0 or len(paths) == 1 or (lo_prev - sub0) < VS_WINDOW:
        print(
            f"SKIP burn-in chunk {idx}: variance model lacks a full "
            f"{VS_WINDOW}-bar residual window before the previous chunk"
        )
        return

    p = _load_panel()
    cols = design_cols(p.names, args.design)
    # design rows aligned to the residual sub-series (global rows are
    # contiguous across chunks by chunk_bounds construction; asserted)
    assert np.all(np.diff(ch["row_id"]) == 1), "sub-series rows not contiguous"
    x_mat = np.ascontiguousarray(p.X[sub0 : sub0 + len(ch["row_id"])][:, cols])
    z = _zlog(ch["e"] ** 2)
    z[~np.isfinite(z)] = float(np.nanmedian(z[np.isfinite(z)]))  # invalid bars:
    # neutral in-window level so the rolling stats stay finite (these bars are
    # never scored; their z only enters as window filler)

    first = lo_prev - sub0
    last = hi_i - sub0
    zhat, traj = variance_walk(
        x_mat, z, ch["m"], ch["baseline"], ch["rv_raw"], first, last
    )
    inf_mask = np.zeros(len(z), dtype=bool)
    for a, b in zip(traj, traj[1:] + [{"idx": last}]):
        if np.isinf(a["alpha"]):
            inf_mask[a["idx"] : b["idx"]] = True

    prev_lo, prev_hi = lo_prev - sub0, lo_i - sub0
    out_lo, out_hi = lo_i - sub0, hi_i - sub0
    sel_prev = _window_sel(ch, prev_lo, prev_hi)
    sigma2_cond = assemble_sigma2(
        zhat, ch["e"] ** 2, sel_prev, prev_lo, prev_hi, out_lo, out_hi, inf_mask
    )
    valid_i = ch["valid"][out_lo:out_hi]
    m_i = ch["m"][out_lo:out_hi]
    b_i = ch["baseline"][out_lo:out_hi]
    f_cond = np.full(out_hi - out_lo, np.nan)
    ok = valid_i & np.isfinite(m_i) & np.isfinite(sigma2_cond) & np.isfinite(b_i)
    f_cond[ok] = (m_i[ok] ** 2 + sigma2_cond[ok]) * b_i[ok]

    t_i = p.t[lo_i:hi_i]
    yr = t_i.astype("datetime64[Y]").astype(int) + 1970
    traj_out = []
    for e in traj:
        g_row = int(ch["row_id"][e["idx"]])
        if lo_prev <= g_row:  # trajectory spans [prev chunk, this chunk]
            yr_e = int(str(p.t[g_row].astype("datetime64[Y]")))
            traj_out.append(
                {
                    "row": g_row,
                    "alpha": "inf" if np.isinf(e["alpha"]) else float(e["alpha"]),
                    "year": yr_e,
                }
            )
    n_inf = sum(1 for e in traj_out if e["alpha"] == "inf")
    by_year: dict[int, list[int]] = {}
    for e in traj_out:
        by_year.setdefault(e["year"], []).append(1 if e["alpha"] == "inf" else 0)
    frac_inf_by_year = {
        str(y): round(sum(v) / len(v), 4) for y, v in sorted(by_year.items())
    }

    out_dir = design_out_dir(args.output_dir, args.arm, args.design)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"chunk_{idx:03d}.npz")
    np.savez_compressed(
        out,
        row_id=ch["row_id"][out_lo:out_hi],
        sigma2_cond=sigma2_cond,
        f_cond_raw=f_cond,
        year=yr,
        meta=json.dumps(
            {
                "arm": args.arm,
                "chunk_index": idx,
                "design": args.design,
                "n_design_cols": int(len(cols)),
                "alpha_grid": ["inf" if np.isinf(a) else a for a in ALPHA_GRID],
                "tuned_alpha": traj_out,
                "frac_inf": round(n_inf / len(traj_out), 4) if traj_out else None,
                "frac_inf_by_year": frac_inf_by_year,
                "recalibration": "v = exp(zhat) * c_k, c_k = mean(e2_prev)/"
                "mean(exp(zhat_prev)); inf bars carry the scalar exactly",
            }
        ),
    )
    print(
        f"DONE {args.arm} [{args.design}] sidecar chunk {idx} [{lo_i},{hi_i}) "
        f"p={len(cols)} frac_inf={n_inf}/{len(traj_out)} -> {out}"
    )


if __name__ == "__main__":
    main()
