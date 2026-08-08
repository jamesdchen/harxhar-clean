"""Synthetics for the 2026-08-07 penalty-shape / PC-basis arm set.

Covers, in order:

  A. GRID ENUMERATION — the wide exponent grid, the shape zoo and the wide
     pcrank grid enumerate the intended number of points with NO duplicates,
     and the wide grids STRICTLY CONTAIN the narrow ones they extend (so any
     wide-vs-narrow increment is attributable to the added points alone).
     Duplicate-freeness is load-bearing: ``_walk_blocks_tuned``'s cyclic
     descent skips a trial when ``_pen_value(a) == sel[k]``, so two grid points
     that normalize to the same descriptor would silently shrink the grid.

  B. PENALTY VECTORS — power/gamma=0 nests the flat penalty BIT-IDENTICALLY;
     every point of every grid produces a monotone penalty across ranks (for
     the SIGNED grids, monotone in either direction and strictly positive —
     gamma<0 is a deliberate part of the bipolar axes); the step family's
     boundary lands at exactly rank K0; the extreme corner is finite and exact
     in float64.

  C. CONDITIONING — the question the wide exponent grid raises: at lambda0=1e4,
     gamma=4, K=40 the largest transmission penalty is 1e4 * 40**4 = 2.56e10,
     ~3 decades above the shaped arm's current worst case. This section builds
     a realistically structured fit gram at the campaign's own fit-window size
     and reports, per grid corner, the condition number of the penalized gram
     AND the fitted-value agreement between the normal-equation solve the code
     performs and the numerically stable augmented-QR ridge solve. The latter
     is the number that decides whether a cap is needed: cond alone overstates
     the danger, because the directions carrying the huge penalty are exactly
     the ones being shrunk to zero.

  D. PC-LADDER REPARAMETERIZATION — at full rank, {ma_j(G_i)} and the
     ladder-expanded standardized base differ by a block-diagonal ORTHOGONAL
     rotation, so a flat-penalty ridge on either must give identical fitted
     values. Verified to machine precision on a fixture, and then re-run WITH
     the arm's trailing standardization to quantify how much that (non-
     orthogonal, time-varying) rescaling breaks the identity.

  E. PER-RUNG BASES — column count, rank-major layout, and the fast-vs-slow
     subspace alignment that predicts whether blk_pcladderPerRung_tuned can
     differ from blk_pcladder_tuned at all.

  F. ARM WIRING — every new arm is registered, resolves its block grids, and
     differs from the arm it extends in EXACTLY the intended field.

  G. FIXED-PENALTY JIGGLE ENVELOPE — every ridge/lasso jiggle arm's penalty is
     literally what its name says, and the two arms already on disk (b1_ridge
     at alpha=1, b2_lasso at alpha=1e-4) are untouched. The experiment is a
     QLIKE-vs-log(alpha) curve, so an arm mislabelled by one decade would
     invert the reading.

  H. HALF-DECADE GRID RESOLUTION — every fine grid's exact membership, and the
     STRICT SUBSET property (bit-exact) that makes fine-vs-coarse attributable
     to the added points alone; plus the cyclic tail-evaluation budget.

  K. REACH-MATCHED ELASTIC-NET GRID — the be_tunedWide alpha axis asserted
     against reclasso's own lam2 = N*alpha*(1-l1_ratio) mapping at every
     mixing value, the old grid a bit-exact strict subset in unchanged order,
     and the degenerate pure-lasso corner driven end-to-end through the warm
     homotopy to confirm it lands on the intercept-only limit.

  N. RANK-DEFICIENT DESIGNS — the regression test for the 2026-08-07 defect.
     Every earlier synthetic used a FULL-RANK fixture, which is why a design
     flaw that disabled the entire grid-free family reached the cluster. This
     section drives the shipped code on a singular design containing dead,
     constant and duplicated indicator columns and asserts the factor is
     strictly inside (0,1) and the walk's mean factor is below 0.99.

  M. PROPER PCR — the real panel's ladder tensor verified uniform and
     reconciled to 1144 columns, K x 12 column counts, scores confirmed
     UNstandardized, the predictive ordering's frame-window-only causality,
     the two orderings coinciding at K=full, and THE GATE: at K=full the
     rotated design's James-Stein fit equals the unrotated one to machine
     precision, end to end.

  L. GRID-FREE SHRINKAGE — causality (post-window perturbation leaves every
     coefficient bit-identical), the James-Stein factor against an independent
     closed-form route, the Schur complement against the literal block
     inverse, the orthogonal-invariance result that dictates the PC arm's
     form, NPEB recovery of a planted two-group prior against the oracle Bayes
     rule, backbone coefficients untouched, and profile persistence.

  I. ENDPOINT RELIEF + ADAPTIVE-VS-FIXED TILT — the bipolar pcrank axis and the
     widened tikhonov power/step axes (membership, duplicate-freeness, strict
     superset, flat nesting), and that the frozen-standardization arm differs
     from its trailing twin in the transmission BLOCK only while carrying the
     SAME exponent grid, so the increment isolates adaptivity rather than grid
     reach. Section C3 measures the conditioning at these new corners,
     including the one deliberately excluded.

Run:  python experiments/verify_unification_shapes.py
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import src.models.reclasso_har as R  # noqa: E402
import src.unification as U  # noqa: E402

FAIL: list[str] = []


def _grid(key: str) -> list[Any]:
    """A block grid as a list of opaque points. BLOCK_TUNE_GRIDS is declared
    ``dict[str, tuple[float, ...]]`` for the scalar blocks; the shaped blocks
    store descriptor tuples under the same key, so the checks read them
    untyped rather than fight the declaration."""
    return list(U.BLOCK_TUNE_GRIDS[key])


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        FAIL.append(name)


# ── A. grid enumeration ───────────────────────────────────────────────────────
def section_a() -> None:
    print("\nA. GRID ENUMERATION")
    g_narrow = _grid("trans_shaped")
    g_wide = _grid("trans_shaped_wide")
    g_zoo = _grid("trans_shaped_zoo")
    g_pcn = _grid("pc_ladder_tilt")
    g_pcw = _grid("pc_ladder_tilt_wide")

    check(
        "trans_shaped unchanged at 12 points", len(g_narrow) == 12, str(len(g_narrow))
    )
    check(
        "TRANS_SHAPE_GAMMAS unchanged",
        U.TRANS_SHAPE_GAMMAS == (0.0, 0.5, 1.0, 2.0),
        str(U.TRANS_SHAPE_GAMMAS),
    )
    check(
        "TRANS_SHAPE_GAMMAS_WIDE = (0, .5, 1, 2, 3, 4)",
        U.TRANS_SHAPE_GAMMAS_WIDE == (0.0, 0.5, 1.0, 2.0, 3.0, 4.0),
        str(U.TRANS_SHAPE_GAMMAS_WIDE),
    )
    # trans_shaped_wide carries the BIPOLAR axis: the pinning it relieves is
    # two-sided (45% at the gamma=0 floor, 41% at the gamma=2 ceiling), so an
    # upward-only extension would have left the larger half in place.
    check(
        "trans_shaped_wide has 24 points (bipolar)", len(g_wide) == 24, str(len(g_wide))
    )
    check(
        "trans_shaped_wide brackets the narrow axis from BOTH sides",
        min(g for _, g in g_wide) < min(g for _, g in g_narrow)
        and max(g for _, g in g_wide) > max(g for _, g in g_narrow),
    )
    check("trans_shaped_zoo has 36 points", len(g_zoo) == 36, str(len(g_zoo)))
    check("pc_ladder_tilt_wide has 18 points", len(g_pcw) == 18, str(len(g_pcw)))

    for label, grid in (
        ("trans_shaped_wide", g_wide),
        ("trans_shaped_zoo", g_zoo),
        ("pc_ladder_tilt_wide", g_pcw),
    ):
        raw = [tuple(a) for a in grid]
        norm = [U._pen_value(a) for a in grid]
        check(f"{label}: no duplicate grid points", len(set(raw)) == len(raw))
        # the tuner compares NORMALIZED descriptors, so those must be distinct
        check(
            f"{label}: no duplicate _pen_value descriptors",
            len(set(norm)) == len(norm),
            f"{len(set(norm))}/{len(norm)}",
        )

    check(
        "wide exponent grid STRICTLY CONTAINS the narrow one",
        set(map(tuple, g_narrow)) < set(map(tuple, g_wide)),
    )
    check(
        "wide pcrank grid STRICTLY CONTAINS the narrow one",
        set(map(tuple, g_pcn)) < set(map(tuple, g_pcw)),
    )
    fams: dict[str, int] = {}
    for a in g_zoo:
        fams[str(a[1])] = fams.get(str(a[1]), 0) + 1
    check(
        "zoo family counts power/exp/step = 18/9/9",
        fams == {"power": 18, "exp": 9, "step": 9},
        str(fams),
    )
    check(
        "zoo contains the exact flat nesting point (power, gamma=0)",
        all(
            (lam, "power", 0.0) in [tuple(a) for a in g_zoo] for lam in (1e2, 1e3, 1e4)
        ),
    )


# ── B. penalty vectors ────────────────────────────────────────────────────────
def section_b(k_span: int = 40) -> None:
    print(f"\nB. PENALTY VECTORS (K={k_span})")

    def fill(value) -> np.ndarray:
        pen = np.zeros(k_span)
        U._fill_pen_span(pen, 0, k_span, U._pen_value(value))
        return pen

    for lam0 in (1e2, 1e3, 1e4):
        flat = np.zeros(k_span)
        U._fill_pen_span(flat, 0, k_span, float(lam0))
        check(
            f"gamma=0 nests flat BIT-IDENTICALLY (lambda0={lam0:g}, 2-tuple)",
            np.array_equal(fill((lam0, 0.0)), flat),
        )
        check(
            f"power/gamma=0 nests flat BIT-IDENTICALLY (lambda0={lam0:g}, zoo form)",
            np.array_equal(fill((lam0, "power", 0.0)), flat),
        )

    # NON-NEGATIVE-exponent grids only: these may be asserted monotone
    # non-DECREASING. trans_shaped_wide is deliberately excluded — it carries
    # the bipolar axis and is checked in the signed loop below.
    bad_mono: list[str] = []
    for label, grid in (
        ("trans_shaped", _grid("trans_shaped")),
        ("trans_shaped_zoo", _grid("trans_shaped_zoo")),
    ):
        for a in grid:
            pv = fill(a)
            if not np.all(np.diff(pv) >= 0) or not np.all(np.isfinite(pv)):
                bad_mono.append(f"{label}{tuple(a)}")
        check(
            f"{label}: penalty monotone non-decreasing + finite at every point",
            not bad_mono,
            "; ".join(bad_mono[:3]),
        )
        bad_mono = []

    # SIGNED-exponent grids: a power law in rank is monotone in EITHER
    # direction, so the invariant is monotone-and-positive, not
    # monotone-INCREASING. gamma<0 (shrink the leading directions hardest) is a
    # deliberate part of the bipolar grids — see TRANS_SHAPE_GAMMAS_BIPOLAR —
    # so asserting non-decreasing here would reject the very points those arms
    # exist to test.
    for label, grid in (
        ("trans_shaped_wide", _grid("trans_shaped_wide")),
        ("exog_tilt_step_wide", _grid("exog_tilt_step_wide")),
    ):
        bad_sign: list[str] = []
        for a in grid:
            pv = fill(a) if label == "trans_shaped_wide" else np.zeros(0)
            if label != "trans_shaped_wide":
                pv = np.zeros(526)
                U._fill_pen_span(pv, 0, 526, U._pen_value(a))
            d = np.diff(pv)
            if not (np.all(d >= 0) or np.all(d <= 0)):
                bad_sign.append(f"{tuple(a)} not monotone in rank")
            if not (np.all(np.isfinite(pv)) and np.all(pv > 0)):
                bad_sign.append(f"{tuple(a)} not finite-positive")
        check(
            f"{label}: penalty monotone in rank (either sign), finite, positive",
            not bad_sign,
            "; ".join(bad_sign[:3]),
        )

    # pcrank grids: monotone across RANK, flat within a rank's rungs
    n_rungs = len(U.PRODUCT_EXOG_WINDOWS)
    for label, grid in (
        ("pc_ladder_tilt", _grid("pc_ladder_tilt")),
        ("pc_ladder_tilt_wide", _grid("pc_ladder_tilt_wide")),
        ("pc_ladder_tilt_bipolar", _grid("pc_ladder_tilt_bipolar")),
    ):
        bad: list[str] = []
        span = 20 * n_rungs
        for a in grid:
            pen = np.zeros(span)
            U._fill_pen_span(pen, 0, span, U._pen_value(a))
            d = np.diff(pen)
            if not (np.all(d >= 0) or np.all(d <= 0)):
                bad.append(f"{tuple(a)} not monotone")
            if not (np.all(np.isfinite(pen)) and np.all(pen > 0)):
                bad.append(f"{tuple(a)} not finite-positive")
            blocks = pen.reshape(20, n_rungs)
            if not np.all(blocks == blocks[:, :1]):
                bad.append(f"{tuple(a)} varies WITHIN a rank")
        check(
            f"{label}: rank-monotone and constant within a rank", not bad, str(bad[:2])
        )

    # step boundary at EXACTLY rank K0
    for k0 in U.TRANS_SHAPE_STEP_KS:
        pv = fill((1e3, "step", float(k0)))
        ok = (
            np.all(pv[:k0] == 1e3)
            and np.all(pv[k0:] == 1e3 * U.STEP_MULTIPLIER)
            and pv[k0 - 1] != pv[k0]
        )
        check(f"step boundary lands at exactly rank K0={k0}", bool(ok))

    # exponential family: geometric, strictly increasing, distinct from power
    for kap in U.TRANS_SHAPE_KAPPAS:
        pv = fill((1e3, "exp", float(kap)))
        ratios = pv[1:] / pv[:-1]
        check(
            f"exp/kappa={kap}: constant ratio exp(kappa) (geometric tail)",
            bool(np.allclose(ratios, np.exp(kap), rtol=1e-12)),
            f"max|ratio-exp(k)|={np.max(np.abs(ratios - np.exp(kap))):.2e}",
        )

    corner = fill((1e4, 4.0))
    check(
        "extreme corner lambda0=1e4, gamma=4, K=40 is exact and finite",
        corner[-1] == 1e4 * 40.0**4 and np.isfinite(corner).all(),
        f"max penalty {corner[-1]:.6g} (= 1e4 * 40**4)",
    )
    print(
        f"    penalty range at the extreme corner: "
        f"[{corner[0]:.4g}, {corner[-1]:.4g}], "
        f"vs the SHIPPED arm's steepest (gamma=2) "
        f"[{fill((1e4, 2.0))[0]:.4g}, {fill((1e4, 2.0))[-1]:.4g}]"
    )


# ── C. conditioning ───────────────────────────────────────────────────────────
def _fit_gram_fixture(
    n_shaped: int = 40, shaped_key: str = "trans", seed: int = 0
) -> tuple[np.ndarray, np.ndarray, list]:
    """Realistically STRUCTURED fit-window design at the campaign's own scale.

    Not the real panel (which needs the feature cache) but built to reproduce
    the three properties the condition number depends on:
      * the row count of the ACTUAL fit block (window - VAL_TAIL - EMBARGO);
      * the heavy near-collinearity of the exogenous MA panel — a low-rank
        factor model whose spectrum is the POWER LAW measured on the real
        frozen frame (d_i ~ i**-1.176) plus small idiosyncratic noise;
      * a WORST-CASE floor: the backbone block, whose grid minimum is the
        smallest penalty in the whole design (0.1), is given a near-duplicate
        column pair so its gram contribution is numerically singular. That
        pins lambda_min of the penalized gram at the penalty floor, which is
        the least favourable configuration the tuner can reach. Real backbones
        contain near-duplicate calendar/session columns, so this is a stress
        test, not a strawman.
    """
    rng = np.random.default_rng(seed)
    n_fit = U.DEFAULT_WINDOW_BARS - U.VAL_TAIL - U.EMBARGO
    n_back, n_exog, n_prod = 40, 526, 300
    n_fac = 106  # live rank of the real base frame
    d = np.arange(1, n_fac + 1, dtype=np.float64) ** -1.176
    fac = rng.standard_normal((n_fit, n_fac)) * np.sqrt(d)
    load = rng.standard_normal((n_fac, n_exog)) / np.sqrt(n_fac)
    exog = fac @ load + 0.01 * rng.standard_normal((n_fit, n_exog))
    back = rng.standard_normal((n_fit, n_back))
    back[:, 1] = back[:, 0] + 1e-9 * rng.standard_normal(n_fit)  # near-duplicate
    prod = exog[:, :n_prod] * exog[:, 1 : n_prod + 1]
    # trailing-standardized scores => unit variance by construction
    shaped = fac[:, :n_shaped] / np.sqrt(d[:n_shaped])
    X = np.hstack([back, exog, prod, shaped])
    y = X[:, :5].sum(1) + rng.standard_normal(n_fit)
    segments = [
        (0, n_back, "backbone"),
        (n_back, n_back + n_exog, "exog"),
        (n_back + n_exog, n_back + n_exog + n_prod, "product"),
        (n_back + n_exog + n_prod, X.shape[1], shaped_key),
    ]
    return X, y, segments


def _cond_report(X, y, segments, shaped_key, corners) -> None:
    Xc = X - X.mean(0)
    yc = y - y.mean()
    G = Xc.T @ Xc
    c = Xc.T @ yc
    p = X.shape[1]
    base = {"backbone": 0.1, "exog": 1e2, "product": 1e3}
    print(
        f"    fixture: {X.shape[0]} fit rows x {p} cols; "
        f"cond(unpenalized gram) = {np.linalg.cond(G):.3e}, "
        f"||G||_2 = {np.linalg.norm(G, 2):.3e}"
    )
    for label, tv in corners:
        pen = np.empty(p)
        for s0, s1, k in segments:
            U._fill_pen_span(pen, s0, s1, base[k] if k in base else U._pen_value(tv))
        A = G.copy()
        A[np.diag_indices_from(A)] += pen
        cond = np.linalg.cond(A)
        b = np.linalg.solve(A, c)
        # STABLE reference: augmented least squares [Xc; sqrt(diag(pen))] via
        # QR, whose conditioning is the SQUARE ROOT of the normal equations'.
        aug = np.vstack([Xc, np.diag(np.sqrt(pen))])
        rhs = np.concatenate([yc, np.zeros(p)])
        b_ref = np.linalg.lstsq(aug, rhs, rcond=None)[0]
        fit, fit_ref = Xc @ b, Xc @ b_ref
        rel_fit = float(
            np.max(np.abs(fit - fit_ref)) / max(np.max(np.abs(fit_ref)), 1e-300)
        )
        rel_coef = float(np.max(np.abs(b - b_ref)) / max(np.max(np.abs(b_ref)), 1e-300))
        print(
            f"    {label:<46s} pen [{pen.min():.3g}, {pen.max():.3g}]  "
            f"cond {cond:.3e}  max rel fitted diff vs stable QR "
            f"{rel_fit:.2e}  (coef {rel_coef:.2e})"
        )
        check(
            f"solve is well-behaved at: {label}",
            bool(np.isfinite(b).all()) and rel_fit < 1e-6,
            f"cond {cond:.3e}, rel fitted diff {rel_fit:.2e}",
        )


def section_c() -> None:
    print("\nC. CONDITIONING AT THE EXTREME GRID CORNERS")
    print("  C1. blk4_trailGShapedWide / blk4_trailGZoo (K=40 transmission block)")
    X, y, segments = _fit_gram_fixture(n_shaped=40, shaped_key="trans")
    _cond_report(
        X,
        y,
        segments,
        "trans",
        [
            ("flat (lambda0=1e2)", 1e2),
            ("SHIPPED steepest: power gamma=2, lambda0=1e4", (1e4, 2.0)),
            ("WIDE steepest: power gamma=4, lambda0=1e4", (1e4, 4.0)),
            ("zoo steepest exp: kappa=0.10, lambda0=1e4", (1e4, "exp", 0.10)),
            ("zoo steepest step: K0=10, lambda0=1e4", (1e4, "step", 10.0)),
        ],
    )
    print("\n  C2. blk_pcladder_fullK_tuned (K=106 ranks x 3 rungs = 318 columns)")
    Xp, yp, segp = _fit_gram_fixture(n_shaped=318, shaped_key="pc")
    _cond_report(
        Xp,
        yp,
        segp,
        "pc",
        [
            ("flat (lambda0=1e2)", 1e2),
            ("narrow steepest: pcrank gamma=2, lambda0=1e4", (1e4, "pcrank", 2.0, 3)),
            ("WIDE steepest: pcrank gamma=4, lambda0=1e4", (1e4, "pcrank", 4.0, 3)),
        ],
    )
    pen = np.empty(106 * 3)
    U._fill_pen_span(pen, 0, pen.size, U._pen_value((1e4, "pcrank", 4.0, 3)))
    check(
        "full-rank pcrank corner stays finite and exact in float64",
        bool(np.isfinite(pen).all()) and pen.max() == 1e4 * 106.0**4,
        f"max penalty {pen.max():.6g} = 1e4 * 106**4, "
        f"{pen.max() / pen.min():.3g}x span",
    )


# ── fixture panel for D/E ─────────────────────────────────────────────────────
def _fixture_panel(n: int = 2400, window: int = 400, seed: int = 1) -> U._Panel:
    """Small synthetic panel with the real NAME grammar, so the production
    column selectors (`_product_base_cols`, `_backbone_cols`, ...) apply
    unchanged. The algebraic claims in D/E are scale-free, so a small fixture
    proves them; the trailing-standardization window is shortened to fit
    (see the TRANS_TRAIL_DAYS override at the call site)."""
    rng = np.random.default_rng(seed)
    stems = [f"s{i:02d}" for i in range(30)]
    names = [f"har_ma_{w}" for w in (1, 2, 4, 8, 16, 32, 64)]
    names += [f"adj_{s}_ma_{w}" for s in stems for w in U.PRODUCT_EXOG_WINDOWS]
    names += [f"{s}_avail_ma_1" for s in stems[:5]]
    names += ["is_open", "is_close", "is_overnight", "hour"]
    p = len(names)
    # factor structure with a power-law spectrum, so the frame is non-trivial
    n_fac = 40
    d = np.arange(1, n_fac + 1, dtype=np.float64) ** -1.176
    fac = rng.standard_normal((n, n_fac)) * np.sqrt(d)
    X = fac @ (rng.standard_normal((n_fac, p)) / np.sqrt(n_fac))
    X += 0.1 * rng.standard_normal((n, p))
    y = X[:, :3].sum(1) + rng.standard_normal(n)
    return U._Panel(
        X=np.ascontiguousarray(X),
        y=y,
        baseline=np.ones(n),
        rv_raw=np.ones(n),
        t=np.arange(n).astype("datetime64[s]").astype("datetime64[ns]"),
        names=names,
        avail=np.ones((n, len(stems)), dtype=bool),
        stem_index={s: i for i, s in enumerate(stems)},
    )


# ── D. PC-ladder reparameterization at full rank ──────────────────────────────
def section_d() -> None:
    print("\nD. PC-LADDER FULL-RANK REPARAMETERIZATION")
    window = 400
    p = _fixture_panel(n=3 * window, window=window)
    k_full = U._frame_live_rank(p, window)
    bc = U._product_base_cols(p.names)
    Z = np.ascontiguousarray(p.X[:, bc])
    check(
        "_frame_live_rank matches the frame builder's own liveness rule",
        k_full == int((Z[window : 2 * window].std(0) > U._DEGENERATE_SD).sum()),
        f"K_full = {k_full} of {Z.shape[1]} base columns",
    )

    # frozen frame, exactly as _transmission_block._frame_of builds it
    zw = Z[window : 2 * window]
    mu, sd = zw.mean(0), zw.std(0)
    live = sd > U._DEGENERATE_SD
    sdl = np.where(live, sd, 1.0)
    lam, v_l = np.linalg.eigh(np.corrcoef(((zw - mu) / sdl)[:, live], rowvar=False))
    v = v_l[:, np.argsort(lam)[::-1]]  # (n_live, n_live) ORTHOGONAL at full rank
    check(
        "full-rank frame is orthogonal",
        bool(np.allclose(v.T @ v, np.eye(v.shape[1]), atol=1e-12)),
        f"max|V'V - I| = {np.max(np.abs(v.T @ v - np.eye(v.shape[1]))):.2e}",
    )

    zs = ((Z - mu) / sdl)[:, live]  # standardized live base series
    rungs = U.PRODUCT_EXOG_WINDOWS
    n_rungs = len(rungs)

    def ladder(mat: np.ndarray) -> list[np.ndarray]:
        out = []
        for w in rungs:
            a = (
                pd.DataFrame(mat)
                .rolling(window=int(w), min_periods=1)
                .mean()
                .shift(1)
                .to_numpy()
            )
            a[~np.isfinite(a)] = 0.0
            out.append(a)
        return out

    # comparator: ladder of the STANDARDIZED BASE (not rotated)
    a_blocks = ladder(zs)
    design_base = np.hstack(a_blocks)
    # PC-ladder WITHOUT trailing standardization: ladder of the rotated series,
    # laid out rank-major exactly as _pc_ladder_design does
    g_blocks = ladder(zs @ v)
    design_pc = np.empty_like(design_base)
    for j in range(n_rungs):
        design_pc[:, j::n_rungs] = g_blocks[j]

    lo = 2 * window
    yv = p.y

    def ridge_fit(F: np.ndarray, alpha: float) -> np.ndarray:
        Fw, yw = F[window:lo], yv[window:lo]
        Fc, yc = Fw - Fw.mean(0), yw - yw.mean()
        A = Fc.T @ Fc
        A[np.diag_indices_from(A)] += alpha
        b = np.linalg.solve(A, Fc.T @ yc)
        return (F[lo:] - Fw.mean(0)) @ b + yw.mean()

    for alpha in (1e0, 1e2, 1e4):
        f1, f2 = ridge_fit(design_base, alpha), ridge_fit(design_pc, alpha)
        rel = float(np.max(np.abs(f1 - f2)) / max(np.max(np.abs(f1)), 1e-300))
        check(
            f"flat-ridge fitted values identical under the rotation (alpha={alpha:g})",
            rel < 1e-9,
            f"max relative difference {rel:.3e}",
        )

    # ...and now the SAME comparison with the arm's actual trailing
    # standardization of the scores, which is a TIME-VARYING PER-COLUMN
    # rescaling and therefore NOT orthogonal.
    trail = 200
    gm = pd.DataFrame(zs @ v).rolling(trail, min_periods=trail).mean().shift(1)
    gs = pd.DataFrame(zs @ v).rolling(trail, min_periods=trail).std().shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        g_t = ((zs @ v) - gm.to_numpy()) / gs.to_numpy()
    g_t[~np.isfinite(g_t)] = 0.0
    gt_blocks = ladder(g_t)
    design_pc_trail = np.empty_like(design_base)
    for j in range(n_rungs):
        design_pc_trail[:, j::n_rungs] = gt_blocks[j]
    rels = []
    for alpha in (1e0, 1e2, 1e4):
        f1 = ridge_fit(design_base, alpha)
        f3 = ridge_fit(design_pc_trail, alpha)
        rels.append(float(np.max(np.abs(f1 - f3)) / max(np.max(np.abs(f1)), 1e-300)))
    print(
        "    WITH the arm's trailing standardization the identity does NOT "
        f"hold: max relative fitted-value difference {max(rels):.3e} "
        "(alpha 1e0/1e2/1e4: " + ", ".join(f"{r:.2e}" for r in rels) + ")"
    )
    check(
        "trailing standardization is correctly identified as NON-orthogonal",
        max(rels) > 1e-6,
        "the equivalence is a span statement, not a bit-level one",
    )

    # WHY it breaks, quantified: standardizing score i to unit variance divides
    # the column by sqrt(d_i). If c_i is the coefficient on the standardized
    # column and b_i the coefficient on the RAW score, b_i = c_i / sqrt(d_i),
    # so a penalty lambda_i on c is a penalty lambda_i * d_i on b. Trailing
    # standardization is therefore ITSELF a spectral tilt: with d_i ~ i**-s the
    # flat penalty already behaves as i**-s on the raw eigen-directions, and an
    # explicit rank tilt gamma lands at gamma_eff = gamma - s. That is the
    # single most useful number for reading the tuner's selected gamma.
    dsort = np.sort(lam)[::-1][: min(40, len(lam))]
    dsort = dsort[dsort > 0]
    i = np.arange(1, len(dsort) + 1, dtype=float)
    slope, intercept = np.polyfit(np.log(i), np.log(dsort), 1)
    pred = intercept + slope * np.log(i)
    r2 = 1.0 - np.sum((np.log(dsort) - pred) ** 2) / np.sum(
        (np.log(dsort) - np.log(dsort).mean()) ** 2
    )
    print(
        f"    fixture frame spectrum: d_i ~ i**{slope:.3f} (log-log R^2 "
        f"{r2:.3f}, d_1/d_K = {dsort[0] / dsort[-1]:.1f}). With the REAL "
        "frame's s = 1.176, a selected gamma maps to gamma_eff = gamma - "
        "1.176 on the raw eigen-directions: the flat penalty (gamma=0) is "
        "already a -1.176 tilt, gamma=1.176 is the FLAT point, the shipped "
        "grid tops out at gamma_eff = +0.824, and the wide grid reaches "
        "+2.824."
    )


# ── E. per-rung bases ─────────────────────────────────────────────────────────
def section_e() -> None:
    print("\nE. PER-RUNG EIGENBASES")
    window = 400
    # 4 windows of rows: _pc_ladder_design's shared-basis path runs the full
    # _transmission_block, whose Ghat rolling scaler needs window bars AFTER
    # the frame + trailing warm-up.
    p = _fixture_panel(n=4 * window, window=window)
    k = 20
    n_rungs = len(U.PRODUCT_EXOG_WINDOWS)
    trail_save = U.TRANS_TRAIL_DAYS
    U.TRANS_TRAIL_DAYS = 4  # 4 * 48 = 192 bars, so the fixture has warm history
    try:
        d_pr = U._pc_ladder_perrung_design(p, window, qpool=k)
        d_sh = U._pc_ladder_design(p, window, qpool=k)
    finally:
        U.TRANS_TRAIL_DAYS = trail_save
    check(
        f"per-rung column count = K x n_rungs = {k} x {n_rungs}",
        d_pr.shape[1] == k * n_rungs,
        str(d_pr.shape),
    )
    check(
        "shared-basis arm has the same column count (basis is the ONLY change)",
        d_sh.shape[1] == d_pr.shape[1],
        f"{d_sh.shape[1]} vs {d_pr.shape[1]}",
    )
    check(
        "per-rung design is NOT the shared-basis design",
        not np.allclose(d_pr, d_sh),
        f"max|diff| {np.max(np.abs(d_pr - d_sh)):.3e}",
    )
    diag = U._LAST_PERRUNG_DIAG
    check("per-rung alignment diagnostic recorded", bool(diag))
    print(
        f"    fast-vs-slow subspace: mean cos(principal angle) "
        f"{diag['mean_principal_angle_cos']:.4f}, min "
        f"{diag['min_principal_angle_cos']:.4f}, mean |dot| of rank-matched "
        f"eigenvectors {diag['mean_matched_abs_dot']:.4f}"
    )
    check(
        "V_fast and V_slow are numerically DISTINCT bases",
        diag["min_principal_angle_cos"] < 1.0 - 1e-9,
        f"min cos = {diag['min_principal_angle_cos']:.6f}",
    )
    # rank-major layout: penalty rank r must map to columns r*n_rungs..+n_rungs
    pen = np.zeros(d_pr.shape[1])
    U._fill_pen_span(
        pen, 0, pen.size, U._pen_value((1.0, "pcrank", 1.0, float(n_rungs)))
    )
    check(
        "pcrank penalty maps onto the rank-major layout (rank r -> lambda0*r)",
        bool(np.array_equal(pen, np.repeat(np.arange(1, k + 1, dtype=float), n_rungs))),
    )


# ── F. arm wiring ─────────────────────────────────────────────────────────────
def section_f() -> None:
    print("\nF. ARM WIRING")
    new_arms = {
        "blk4_trailGShapedWide": "trans_shaped_wide",
        "blk4_trailGZoo": "trans_shaped_zoo",
        "blk_pcladder_fortyK_tuned": "pc_ladder_tilt_wide",
        "blk_pcladder_eightyK_tuned": "pc_ladder_tilt_wide",
        "blk_pcladder_fullK_tuned": "pc_ladder_tilt_wide",
        "blk_pcladderPerRung_tuned": "pc_ladder_tilt",
    }
    for arm, key in new_arms.items():
        spec = U.ARMS.get(arm)
        check(f"{arm} registered", spec is not None)
        if spec is None:
            continue
        keys = [k for _, k in spec.blocks]
        check(
            f"{arm}: kind/grid/oos_mult match the shaped-arm contract",
            spec.kind == "blocks_tuned"
            and spec.grid == "cyclic"
            and spec.oos_mult == 2,
            f"{spec.kind}/{spec.grid}/{spec.oos_mult}",
        )
        check(f"{arm}: uses block grid '{key}'", key in keys, str(keys))
        check(
            f"{arm}: every block grid key resolves",
            all(k in U.BLOCK_TUNE_GRIDS for k in keys),
        )
    ref = U.ARMS["blk4_trailGShaped"]
    wide = U.ARMS["blk4_trailGShapedWide"]
    check(
        "GShapedWide differs from GShaped ONLY in the transmission grid key",
        [b for b, _ in ref.blocks] == [b for b, _ in wide.blocks]
        and [k for _, k in ref.blocks][:3] == [k for _, k in wide.blocks][:3]
        and ref.kind == wide.kind
        and ref.grid == wide.grid
        and ref.oos_mult == wide.oos_mult,
    )
    zoo = U.ARMS["blk4_trailGZoo"]
    check(
        "GZoo differs from GShaped ONLY in the transmission grid key",
        [b for b, _ in ref.blocks] == [b for b, _ in zoo.blocks]
        and [k for _, k in ref.blocks][:3] == [k for _, k in zoo.blocks][:3],
    )
    base = U.ARMS["blk_pcladder_tuned"]
    pr = U.ARMS["blk_pcladderPerRung_tuned"]
    check(
        "PerRung differs from blk_pcladder_tuned ONLY in the PC block builder",
        [k for _, k in base.blocks] == [k for _, k in pr.blocks]
        and [b for b, _ in base.blocks][0] == [b for b, _ in pr.blocks][0],
    )


# ── G. fixed-penalty jiggle envelope ──────────────────────────────────────────
def section_g() -> None:
    """Every jiggle arm's penalty must be EXACTLY what its name says — the whole
    experiment is a QLIKE-vs-log(alpha) curve, so an arm mislabelled by one
    decade would invert the reading."""
    print("\nG. FIXED-PENALTY JIGGLE ENVELOPE")
    ridge = {
        "b1_ridge": 1.0,
        "b1_ridge_a0p1": 0.1,
        "b1_ridge_a0p3": 0.3,
        "b1_ridge_a3": 3.0,
        "b1_ridge_a10": 10.0,
        "b1_ridge_a30": 30.0,
        "b1_ridge_a100": 100.0,
        "b1_ridge_a300": 300.0,
    }
    for arm, alpha in ridge.items():
        spec = U.ARMS.get(arm)
        ok = (
            spec is not None
            and spec.kind == "blocks"
            and [b for b, _ in spec.blocks] == ["wide"]
            and spec.alphas == {"wide": alpha}
        )
        check(
            f"{arm}: fixed ridge alpha == {alpha:g}",
            bool(ok),
            str(spec and spec.alphas),
        )
    check(
        "b1_ridge (alpha=1) NOT perturbed by the extension",
        U.ARMS["b1_ridge"].alphas == {"wide": U.FIXED_RIDGE_ALPHA},
    )

    lasso = {
        "b2_lasso": 1e-4,
        "b2_lasso_a1em6": 1e-6,
        "b2_lasso_a1em5": 1e-5,
        "b2_lasso_a1em3": 1e-3,
        "b2_lasso_a1em2": 1e-2,
    }
    for arm, alpha in lasso.items():
        spec = U.ARMS.get(arm)
        grid = U.ESTIMATOR_GRIDS.get(spec.grid) if spec is not None else None
        ok = (
            spec is not None
            and spec.kind == "tuned"
            and grid is not None
            and len(grid) == 1  # single point => NO selection, a fixed penalty
            and grid[0][0] == "lasso"
            and grid[0][1] == alpha
            and grid[0][2] == 1.0  # l1_ratio=1 => the pure lasso path
        )
        check(
            f"{arm}: fixed lasso alpha == {alpha:g}, l1_ratio == 1", bool(ok), str(grid)
        )
    check(
        "b2_lasso (alpha=1e-4) NOT rebuilt or perturbed",
        U.ESTIMATOR_GRIDS["lasso_fixed"] == [("lasso", U.FIXED_LASSO_ALPHA, 1.0)],
    )
    # the two envelopes must each span their decades with no gaps or repeats
    r_alphas = sorted(v for v in ridge.values())
    l_alphas = sorted(v for _, v in lasso.items())
    check(
        "ridge envelope covers 0.1 .. 300 with 8 distinct points",
        len(set(r_alphas)) == 8 and r_alphas[0] == 0.1 and r_alphas[-1] == 300.0,
        str(r_alphas),
    )
    check(
        "lasso envelope covers 1e-6 .. 1e-2 with 5 distinct points",
        len(set(l_alphas)) == 5 and l_alphas[0] == 1e-6 and l_alphas[-1] == 1e-2,
        str(l_alphas),
    )


# ── H. half-decade grid resolution ────────────────────────────────────────────
def section_h() -> None:
    """The fine-vs-coarse comparison is only clean if the coarse grid is a
    STRICT SUBSET of the fine one — otherwise a fine-arm difference could come
    from having moved a point rather than from having added points."""
    print("\nH. HALF-DECADE GRID RESOLUTION")
    pairs = (
        ("backbone_fine", "backbone", 5),
        ("exog_fine", "exog", 5),
        ("product_fine", "product", 5),
        ("trans_fine", "trans", 5),
        ("trans_shaped_fine", "trans_shaped", 20),
    )
    for fine, coarse, n in pairs:
        gf, gc = _grid(fine), _grid(coarse)
        check(f"{fine}: {n} points", len(gf) == n, str(len(gf)))
        check(
            f"{fine}: no duplicates",
            len(set(map(tuple, gf))) == len(gf)
            if isinstance(gf[0], tuple)
            else len(set(gf)) == len(gf),
        )
        sf = set(map(tuple, gf)) if isinstance(gf[0], tuple) else set(gf)
        sc = set(map(tuple, gc)) if isinstance(gc[0], tuple) else set(gc)
        check(f"{fine} STRICTLY CONTAINS {coarse} (bit-exact)", sc < sf)
    check(
        "backbone_fine membership is exactly 0.1 .. 10 half-decades",
        [f"{v:.6g}" for v in _grid("backbone_fine")]
        == ["0.1", "0.316228", "1", "3.16228", "10"],
        str([f"{v:.6g}" for v in _grid("backbone_fine")]),
    )
    check(
        "exog_fine membership",
        [f"{v:.6g}" for v in _grid("exog_fine")]
        == ["100", "316.228", "1000", "3162.28", "10000"],
        str([f"{v:.6g}" for v in _grid("exog_fine")]),
    )
    check(
        "product_fine membership",
        [f"{v:.6g}" for v in _grid("product_fine")]
        == ["1000", "3162.28", "10000", "31622.8", "100000"],
        str([f"{v:.6g}" for v in _grid("product_fine")]),
    )
    check(
        "trans_shaped_fine keeps the gamma axis UNCHANGED (level axis only)",
        sorted({g for _, g in _grid("trans_shaped_fine")})
        == sorted(set(U.TRANS_SHAPE_GAMMAS)),
    )

    gr_c = [a for _, a, _ in U.ESTIMATOR_GRIDS["ridge_tuned"]]
    gr_f = [a for _, a, _ in U.ESTIMATOR_GRIDS["ridge_tuned_fine"]]
    check("ridge_tuned_fine has 11 points", len(gr_f) == 11, str(len(gr_f)))
    check("ridge_tuned_fine: no duplicates", len(set(gr_f)) == 11)
    check(
        "ridge_tuned_fine STRICTLY CONTAINS ridge_tuned (bit-exact)",
        set(gr_c) < set(gr_f),
    )
    check(
        "ridge_tuned_fine spans 1e-2 .. 1e3 unchanged",
        gr_f[0] == gr_c[0] and gr_f[-1] == gr_c[-1],
        f"{gr_f[0]:g} .. {gr_f[-1]:g}",
    )
    check(
        "ridge_tuned_fine l1_ratio is 0 everywhere (still ridge)",
        all(
            fam == "ridge" and l1 == 0.0
            for fam, _, l1 in U.ESTIMATOR_GRIDS["ridge_tuned_fine"]
        ),
    )
    check(
        "ridge_tuned (coarse) NOT perturbed",
        [f"{a:g}" for a in gr_c] == ["0.01", "0.1", "1", "10", "100", "1000"],
        str([f"{a:g}" for a in gr_c]),
    )

    # arm wiring + the cyclic tail-evaluation budget
    for arm, keys in (
        (
            "blk4_trailGShaped_fine",
            ["backbone_fine", "exog_fine", "product_fine", "trans_shaped_fine"],
        ),
    ):
        spec = U.ARMS[arm]
        check(f"{arm}: block grid keys", [k for _, k in spec.blocks] == keys)
        check(
            f"{arm}: blocks identical to blk4_trailGShaped",
            [b for b, _ in spec.blocks]
            == [b for b, _ in U.ARMS["blk4_trailGShaped"].blocks],
        )
        cost = 1 + U.CYCLIC_PASSES * sum(len(_grid(k)) - 1 for k in keys)
        ref = 1 + U.CYCLIC_PASSES * sum(
            len(_grid(k)) - 1 for _, k in U.ARMS["blk4_trailGShaped"].blocks
        )
        print(
            f"    cyclic tail evaluations per retune: {cost} "
            f"(blk4_trailGShaped: {ref}; observed on disk 52-58, mean 54)"
        )
        check(f"{arm}: tail-eval budget stays under 150/retune", cost <= 150, str(cost))
    spec = U.ARMS["b1_ridge_tuned_fine"]
    check(
        "b1_ridge_tuned_fine wired to ridge_tuned_fine",
        spec.kind == "tuned" and spec.grid == "ridge_tuned_fine",
    )
    check(
        "coarse ancestry registered for both fine arms",
        U.ESTIMATOR_GRID_PARENT == {"ridge_tuned_fine": "ridge_tuned"}
        and set(U.FINE_GRID_PARENT)
        == {
            "backbone_fine",
            "exog_fine",
            "product_fine",
            "trans_shaped_fine",
        },
    )


# ── I. endpoint relief + adaptive-vs-fixed tilt ───────────────────────────────
def section_i() -> None:
    print("\nI. ENDPOINT RELIEF + ADAPTIVE-VS-FIXED TILT")
    # --- bipolar pcrank grid ---
    check(
        "the bipolar axis is SHARED by the transmission and pcrank grids",
        sorted({g for _, g in _grid("trans_shaped_wide")})
        == sorted(set(U.TRANS_SHAPE_GAMMAS_BIPOLAR))
        == sorted({g for _, _, g, _ in _grid("pc_ladder_tilt_bipolar")}),
    )
    gb = _grid("pc_ladder_tilt_bipolar")
    gn = _grid("pc_ladder_tilt")
    check("pc_ladder_tilt_bipolar has 24 points", len(gb) == 24, str(len(gb)))
    check("pc_ladder_tilt_bipolar: no duplicates", len(set(map(tuple, gb))) == 24)
    check(
        "pc_ladder_tilt_bipolar: no duplicate _pen_value descriptors",
        len({U._pen_value(a) for a in gb}) == 24,
    )
    check(
        "pc_ladder_tilt_bipolar STRICTLY CONTAINS pc_ladder_tilt (bit-exact)",
        set(map(tuple, gn)) < set(map(tuple, gb)),
    )
    check(
        "bipolar exponent axis = (-1,-0.5,0,0.5,1,2,3,4)",
        U.TRANS_SHAPE_GAMMAS_BIPOLAR == (-1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0),
        str(U.TRANS_SHAPE_GAMMAS_BIPOLAR),
    )
    check(
        "bipolar grid brackets the OLD grid from BOTH sides",
        min(g for _, _, g, _ in gb) < min(g for _, _, g, _ in gn)
        and max(g for _, _, g, _ in gb) > max(g for _, _, g, _ in gn),
    )

    # --- widened tikhonov step/power grid ---
    gw = _grid("exog_tilt_step_wide")
    go = _grid("exog_tilt_step")
    check("exog_tilt_step_wide has 39 points", len(gw) == 39, str(len(gw)))
    check("exog_tilt_step_wide: no duplicates", len(set(map(tuple, gw))) == 39)
    check(
        "exog_tilt_step_wide: no duplicate _pen_value descriptors",
        len({U._pen_value(a) for a in gw}) == 39,
    )
    check(
        "exog_tilt_step_wide STRICTLY CONTAINS exog_tilt_step (bit-exact)",
        set(map(tuple, go)) < set(map(tuple, gw)),
        "this is what makes the wide-vs-narrow increment attributable to the "
        "ADDED points; the requested step grid (5,10,20,30,60,100) would have "
        "DROPPED K0=40 and K0=80, the two the shipped arm selects most",
    )
    fams: dict[str, int] = {}
    for a in gw:
        fams[str(a[1])] = fams.get(str(a[1]), 0) + 1
    check(
        "wide tikhonov family counts power/step = 21/18",
        fams == {"power": 21, "step": 18},
        str(fams),
    )
    check(
        "step K0 axis extended BOTH ways and keeps (20,40,80)",
        U.TIKHONOV_STEP_KS_WIDE == (5, 10, 20, 40, 80, 100)
        and set(U.TIKHONOV_STEP_KS) < set(U.TIKHONOV_STEP_KS_WIDE),
        str(U.TIKHONOV_STEP_KS_WIDE),
    )
    check(
        "power axis extended DOWN to -1 and capped at 3 (526-wide span)",
        U.TIKHONOV_GAMMAS_WIDE == (-1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0),
        str(U.TIKHONOV_GAMMAS_WIDE),
    )
    check(
        "gamma=4 deliberately ABSENT from the 526-wide power axis",
        4.0 not in U.TIKHONOV_GAMMAS_WIDE,
        "cond 5.3e16 > 1/eps at lambda0=1e4 (measured, section C3); hard "
        "truncation is already covered by the step family",
    )
    for lam0 in (1e2, 1e3, 1e4):
        flat = np.full(526, float(lam0))
        pen = np.zeros(526)
        U._fill_pen_span(pen, 0, 526, U._pen_value((lam0, "power", 0.0)))
        check(
            f"wide tikhonov gamma=0 still nests flat BIT-IDENTICALLY (lambda0={lam0:g})",
            np.array_equal(pen, flat),
        )

    # --- adaptive-vs-fixed arm wiring ---
    froz = U.ARMS["blk4_trailGShapedFrozen"]
    wide = U.ARMS["blk4_trailGShapedWide"]
    check(
        "GShapedFrozen differs from GShapedWide ONLY in the transmission BLOCK",
        [k for _, k in froz.blocks] == [k for _, k in wide.blocks]
        and [b for b, _ in froz.blocks][:3] == [b for b, _ in wide.blocks][:3]
        and [b for b, _ in froz.blocks][3] == "trans_frozenG40"
        and [b for b, _ in wide.blocks][3] == "trans_trailG40",
    )
    check(
        "both tilt arms carry the SAME wide exponent grid (equal gamma_eff reach)",
        [k for _, k in froz.blocks][3]
        == [k for _, k in wide.blocks][3]
        == "trans_shaped_wide",
    )
    for arm, key in (
        ("blk4_trailGShapedFrozen", "trans_shaped_wide"),
        ("blk_pcladderWide_tuned", "pc_ladder_tilt_bipolar"),
        ("blk3_tikhonovStepWide_tuned", "exog_tilt_step_wide"),
    ):
        spec = U.ARMS[arm]
        check(
            f"{arm}: kind/grid/oos_mult match the shaped-arm contract",
            spec.kind == "blocks_tuned"
            and spec.grid == "cyclic"
            and spec.oos_mult == 2,
        )
        check(f"{arm}: uses block grid '{key}'", key in [k for _, k in spec.blocks])
        cost = 1 + U.CYCLIC_PASSES * sum(len(_grid(k)) - 1 for _, k in spec.blocks)
        print(f"    {arm}: {cost} cyclic tail evaluations per retune")
        check(f"{arm}: tail-eval budget under 150/retune", cost <= 150, str(cost))
    # each wide twin must leave its narrow original untouched
    for wide_arm, narrow_arm in (
        ("blk_pcladderWide_tuned", "blk_pcladder_tuned"),
        ("blk3_tikhonovStepWide_tuned", "blk3_tikhonovStep_tuned"),
    ):
        w, n = U.ARMS[wide_arm], U.ARMS[narrow_arm]
        check(
            f"{wide_arm} differs from {narrow_arm} ONLY in the shape grid key",
            [b for b, _ in w.blocks] == [b for b, _ in n.blocks]
            and sum(
                1
                for a, b in zip([k for _, k in w.blocks], [k for _, k in n.blocks])
                if a != b
            )
            == 1,
        )
    check(
        "narrow originals' grids unchanged",
        U.TRANS_SHAPE_GAMMAS == (0.0, 0.5, 1.0, 2.0)
        and U.TIKHONOV_STEP_KS == (20, 40, 80)
        and len(_grid("exog_tilt_step")) == 21
        and len(_grid("pc_ladder_tilt")) == 12,
    )


# ── C3. conditioning at the NEW steep corners ─────────────────────────────────
def section_c3() -> None:
    print("\nC3. CONDITIONING AT THE NEW STEEP CORNERS")
    print("  C3a. exog_rot power family — the 526-COLUMN span (the danger case)")
    X, y, segments = _fit_gram_fixture(n_shaped=526, shaped_key="tilt")
    _cond_report(
        X,
        y,
        segments,
        "tilt",
        [
            ("SHIPPED max: power gamma=2, lambda0=1e4", (1e4, "power", 2.0)),
            ("WIDE max: power gamma=3, lambda0=1e4", (1e4, "power", 3.0)),
            ("NEW low end: power gamma=-1, lambda0=1e2", (1e2, "power", -1.0)),
            ("NEW step: K0=5, lambda0=1e4", (1e4, "step", 5.0)),
            ("NEW step: K0=100, lambda0=1e4", (1e4, "step", 100.0)),
        ],
    )
    print("  C3b. pcrank bipolar — K=20 ranks x 3 rungs (60 columns)")
    Xp, yp, segp = _fit_gram_fixture(n_shaped=60, shaped_key="pc")
    _cond_report(
        Xp,
        yp,
        segp,
        "pc",
        [
            ("bipolar max: pcrank gamma=4, lambda0=1e4", (1e4, "pcrank", 4.0, 3)),
            ("bipolar min: pcrank gamma=-1, lambda0=1e2", (1e2, "pcrank", -1.0, 3)),
        ],
    )
    # the EXCLUDED corner, measured and reported rather than silently dropped
    pen = np.zeros(526)
    U._fill_pen_span(pen, 0, 526, U._pen_value((1e4, "power", 4.0)))
    print(
        f"    EXCLUDED corner (power gamma=4 on a 526-wide span): max penalty "
        f"{pen.max():.4g}, dynamic range {pen.max() / pen.min():.3g}x — "
        "measured cond 5.3e16 EXCEEDS 1/eps = 4.5e15, i.e. numerically "
        "singular to working precision; excluded from TIKHONOV_GAMMAS_WIDE "
        "with that reason recorded in the source."
    )


# ── J. de-confounding control ─────────────────────────────────────────────────
def section_j() -> None:
    """blk4_trailG40_tuned splits a confounded published comparison. Its whole
    value rests on two facts that must be mechanically true: its transmission
    block is K=40 (matching blk4_trailGShaped, NOT blk4_trailG_tuned's K=20),
    and its penalty is genuinely FLAT (no shape axis at all)."""
    print("\nJ. DE-CONFOUNDING CONTROL (blk4_trailG40_tuned)")
    ctl = U.ARMS["blk4_trailG40_tuned"]
    shaped = U.ARMS["blk4_trailGShaped"]
    lvl20 = U.ARMS["blk4_trailG_tuned"]
    check(
        "registered as blocks_tuned / cyclic / oos_mult=2",
        ctl.kind == "blocks_tuned" and ctl.grid == "cyclic" and ctl.oos_mult == 2,
    )
    check(
        "SAME BLOCKS as blk4_trailGShaped (K=40 transmission)",
        [b for b, _ in ctl.blocks] == [b for b, _ in shaped.blocks],
        str([b for b, _ in ctl.blocks]),
    )
    check(
        "differs from blk4_trailGShaped ONLY in the transmission GRID KEY",
        sum(
            1
            for a, b in zip([k for _, k in ctl.blocks], [k for _, k in shaped.blocks])
            if a != b
        )
        == 1
        and [k for _, k in ctl.blocks][3] == "trans"
        and [k for _, k in shaped.blocks][3] == "trans_shaped",
    )
    check(
        "differs from blk4_trailG_tuned ONLY in the transmission BLOCK (K)",
        [k for _, k in ctl.blocks] == [k for _, k in lvl20.blocks]
        and [b for b, _ in ctl.blocks][3] == "trans_trailG40"
        and [b for b, _ in lvl20.blocks][3] == "trans_trailG",
    )
    check(
        "THE CONFOUND, restated mechanically: the shipped pair differs in BOTH "
        "K and shape",
        [b for b, _ in shaped.blocks][3] != [b for b, _ in lvl20.blocks][3]
        and [k for _, k in shaped.blocks][3] != [k for _, k in lvl20.blocks][3],
    )
    # the penalty must be genuinely FLAT — a scalar grid, constant across ranks
    grid = _grid("trans")
    check(
        "transmission grid is the ordinary flat (1e2, 1e3, 1e4)",
        tuple(grid) == (1e2, 1e3, 1e4),
        str(grid),
    )
    for a in grid:
        check(
            f"penalty CONSTANT across all 40 ranks at alpha={a:g}",
            _flat_span(a, 40),
        )
    # ...and the transmission block really is 40 columns wide
    window = 400
    p = _fixture_panel(n=4 * window, window=window)
    trail_save = U.TRANS_TRAIL_DAYS
    U.TRANS_TRAIL_DAYS = 4
    try:
        blk40 = U._build_block(p, "trans_trailG40", window)
        blk20 = U._build_block(p, "trans_trailG", window)
    finally:
        U.TRANS_TRAIL_DAYS = trail_save
    check("trans_trailG40 is 40 columns", blk40.shape[1] == 40, str(blk40.shape))
    check(
        f"trans_trailG is {U.TRANS_QPOOL} columns (the confounded difference)",
        blk20.shape[1] == U.TRANS_QPOOL,
        str(blk20.shape),
    )
    check(
        "the two transmission blocks are genuinely different designs",
        blk40.shape[1] != blk20.shape[1],
        f"K=40 vs K={U.TRANS_QPOOL} — this is what the control holds fixed",
    )


def _flat_span(alpha: float, k: int) -> bool:
    pen = np.zeros(k)
    U._fill_pen_span(pen, 0, k, U._pen_value(alpha))
    return bool(np.all(pen == float(alpha)))


# ── K. reach-matched elastic-net grid ─────────────────────────────────────────
def section_k() -> None:
    """The whole point of be_tunedWide is that its penalty grid can express the
    shrinkage tuned ridge selects. That is an ARITHMETIC claim about the
    reclasso mapping, so it is asserted against the mapping, not eyeballed."""
    print("\nK. REACH-MATCHED ELASTIC-NET GRID")
    n_win = U.DEFAULT_WINDOW_BARS  # N in lam2 = N * alpha * (1 - l1_ratio)
    narrow = U.ESTIMATOR_GRIDS["enet_free"]
    wide = U.ESTIMATOR_GRIDS["enet_free_wide"]
    ridge_top = max(a for _, a, _ in U.ESTIMATOR_GRIDS["ridge_tuned"])

    check("N used by the mapping is the 24000-bar window", n_win == 24000, str(n_win))
    check("enet_free unchanged at 20 points", len(narrow) == 20, str(len(narrow)))
    check("enet_free_wide has 28 points", len(wide) == 28, str(len(wide)))
    check("enet_free_wide: no duplicates", len(set(wide)) == 28)
    check(
        "enet_free_wide STRICTLY CONTAINS enet_free (bit-exact)",
        set(narrow) < set(wide),
    )
    check(
        "the original 20 points keep their ORDER (tuner tie-break unchanged)",
        list(wide[: len(narrow)]) == list(narrow),
    )
    alphas = sorted({a for _, a, _ in wide})
    check(
        "alpha axis == np.logspace(-6, 0, 7)",
        np.allclose(alphas, np.logspace(-6, 0, 7), rtol=0, atol=0),
        str([f"{a:g}" for a in alphas]),
    )
    check(
        "mixings unchanged at (0.25, 0.5, 0.75, 1.0)",
        sorted({x for _, _, x in wide}) == [0.25, 0.5, 0.75, 1.0],
    )
    check(
        "every point is the enet kind (l1=1 rows ARE the lasso)",
        all(k == "enet" for k, _, _ in wide),
    )

    # THE REACH CLAIM, asserted against the documented mapping at each mixing.
    print(
        f"    ridge's selected top alpha = {ridge_top:g}; "
        f"lam2 = N*alpha*(1-l1) with N = {n_win}"
    )
    a_top = max(alphas)
    for l1 in (0.25, 0.5, 0.75, 1.0):
        max_lam2 = n_win * a_top * (1.0 - l1)
        narrow_lam2 = n_win * max(a for _, a, x in narrow if x == l1) * (1.0 - l1)
        need = ridge_top / (n_win * (1.0 - l1)) if l1 < 1.0 else float("inf")
        print(
            f"      l1={l1:<5g} needs alpha {need:>9.4f} | narrow reach "
            f"{narrow_lam2:>8.0f} | WIDE reach {max_lam2:>8.0f} "
            f"({max_lam2 / ridge_top:.1f}x ridge)"
        )
        if l1 < 1.0:
            check(
                f"wide grid REACHES ridge's shrinkage at l1={l1:g}",
                max_lam2 >= ridge_top,
                f"{max_lam2:.0f} >= {ridge_top:.0f}",
            )
            check(
                f"narrow grid did NOT reach it at l1={l1:g} (the defect)",
                narrow_lam2 < ridge_top,
                f"{narrow_lam2:.0f} < {ridge_top:.0f}",
            )
            check(
                f"the grid contains a point at/above the required alpha (l1={l1:g})",
                any(a >= need for a in alphas),
            )
        else:
            check(
                "l1=1.0 has lam2 == 0 at EVERY alpha (structural, not a defect)",
                all(n_win * a * (1.0 - l1) == 0.0 for a in alphas),
            )

    # lam2/mu arithmetic must match reclasso's own solver, not just the comment
    for a, l1 in ((1e-4, 0.5), (1e-2, 0.25), (1e0, 0.75), (1e0, 1.0)):
        rng = np.random.default_rng(3)
        m = 12
        x = rng.standard_normal((200, m))
        yv = x[:, 0] + rng.standard_normal(200)
        g, c = x.T @ x, x.T @ yv
        th_api = R.enet_coef(g, c, 200, a, l1)
        th_manual, _, _ = R.lasso_homotopy(
            g + 200 * a * (1.0 - l1) * np.eye(m), c, 200 * a * l1
        )
        check(
            f"enet_coef == lasso-on-Gram at (alpha={a:g}, l1={l1:g})",
            bool(np.allclose(th_api, th_manual, rtol=0, atol=0)),
        )

    # DEGENERATE CORNER through the FULL warm path, end to end.
    rng = np.random.default_rng(0)
    nrow, ncol = 2000, 40
    xw = rng.standard_normal((nrow + 5, ncol))
    yw = xw[:, :3] @ np.array([0.3, -0.2, 0.15]) + rng.standard_normal(nrow + 5)
    for a, l1, expect_empty in ((1e0, 1.0, True), (1e-6, 0.5, False)):
        solver = U.RollingTunedLinear([("enet", a, l1)])
        solver.init_window(xw[:nrow], yw[:nrow])
        preds = []
        for i in range(4):
            t = nrow + i
            solver.solve()
            preds.append(solver.predict_one(xw[t]))
            solver.roll(xw[t], yw[t], xw[i], yw[i])
        n_active = solver.trace[0][3]
        check(
            f"trace records n_active at (alpha={a:g}, l1={l1:g})",
            isinstance(n_active, int),
            f"n_active={n_active}",
        )
        check(
            f"forecasts are FINITE at (alpha={a:g}, l1={l1:g})",
            bool(np.all(np.isfinite(preds))),
        )
        if expect_empty:
            check(
                "extreme alpha at pure lasso EMPTIES the active set",
                n_active == 0,
                f"mu = {nrow * a * l1:g} exceeds mu_max",
            )
            check(
                "...and the forecast is the INTERCEPT-ONLY limit, not zero/NaN",
                abs(preds[0] - float(yw[:nrow].mean())) < 1e-9,
                f"pred {preds[0]:.6f} vs window mean {yw[:nrow].mean():.6f}",
            )
        else:
            check(f"active set is non-empty at (alpha={a:g}, l1={l1:g})", n_active > 0)

    # arm wiring
    for b in (
        "moments",
        "liquidity",
        "market_ew",
        "market_vw",
        "sentiment",
        "implied_vol",
        "vol_demand",
        "all_features",
    ):
        arm, ref = f"be_tunedWide_{b}", f"be_tuned_{b}"
        spec, rspec = U.ARMS.get(arm), U.ARMS[ref]
        check(f"{arm} registered", spec is not None)
        if spec is None:
            continue
        check(
            f"{arm} differs from {ref} ONLY in the estimator grid",
            spec.kind == rspec.kind
            and spec.blocks == rspec.blocks
            and spec.grid == "enet_free_wide"
            and rspec.grid == "enet_free",
        )


# ── L. grid-free shrinkage ────────────────────────────────────────────────────
def _shrink_fixture(n=1400, window=400, p_back=6, p_exog=40, seed=7):
    """Small design with a KNOWN signal and a backbone/shrunk split."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, p_back + p_exog))
    beta = np.zeros(p_back + p_exog)
    beta[:p_back] = rng.standard_normal(p_back)
    beta[p_back : p_back + 4] = np.array([0.8, -0.6, 0.5, 0.4])
    yv = x @ beta + rng.standard_normal(n)
    segs = [(0, p_back, "backbone"), (p_back, p_back + p_exog, "exog")]
    return x, yv, segs, window


def section_l() -> None:
    print("\nL. GRID-FREE SHRINKAGE")
    x, yv, segs, window = _shrink_fixture()
    lo, hi = window, window + 40

    # --- causality: post-window rows must not touch any coefficient ---
    for est in ("js", "npeb", "js_diag"):
        a, _ = U._walk_shrink(x, yv, window, lo, hi, segs, estimator=est)
        xp, yp = x.copy(), yv.copy()
        xp[hi:] += 1e3 * np.random.default_rng(1).standard_normal(xp[hi:].shape)
        yp[hi:] += 1e3
        b, _ = U._walk_shrink(xp, yp, window, lo, hi, segs, estimator=est)
        check(
            f"CAUSAL: post-window perturbation leaves '{est}' bit-identical",
            np.array_equal(a, b),
            f"max|diff| {np.max(np.abs(a - b)):.3e}",
        )

    # --- everything below drives the SHIPPED code path (_BlockFit +
    #     _js_factor_rank), not a parallel re-implementation ---
    xw, yw = x[:window], yv[:window]
    xc, yc = xw - xw.mean(0), yw - yw.mean()
    gram, rhs = xc.T @ xc, xc.T @ yc
    idx_s = np.arange(segs[0][1], x.shape[1])
    idx_b = np.arange(0, segs[0][1])
    syy_c = float(yc @ yc)
    fit = U._BlockFit(gram, rhs, idx_b, idx_s)
    beta = np.zeros(x.shape[1])
    beta[idx_s] = fit.beta_s
    beta[idx_b] = fit.backbone(fit.beta_s)
    sigma2, dof = U._sigma2_hat(syy_c, beta, rhs, window, fit.rank_b + fit.rank_s)
    f_js = U._js_factor_rank(fit.wald, fit.rank_s, sigma2, dof)
    print(
        f"    fixture: rank_b={fit.rank_b} rank_s={fit.rank_s} "
        f"(of {idx_s.size} shrunk cols), dof={dof}, JS factor {f_js:.4f}"
    )

    # backbone must reproduce the JOINT least-squares backbone exactly
    beta_joint = np.linalg.solve(gram, rhs)
    check(
        "backbone coefficients match the JOINT least-squares solve (FWL identity)",
        bool(np.allclose(beta[idx_b], beta_joint[idx_b], rtol=1e-8, atol=1e-8)),
        f"max|diff| {np.max(np.abs(beta[idx_b] - beta_joint[idx_b])):.3e}",
    )
    check(
        "on a FULL-RANK design the min-norm block equals the ordinary solve",
        bool(np.allclose(fit.beta_s, beta_joint[idx_s], rtol=1e-8, atol=1e-8)),
    )

    # JS factor against an independently computed closed form
    c_mat = np.linalg.inv(gram)[np.ix_(idx_s, idx_s)]
    quad_ref = float(fit.beta_s @ np.linalg.inv(c_mat) @ fit.beta_s)
    check(
        "Wald quantity matches the independent block-inverse route",
        abs(fit.wald - quad_ref) / max(abs(quad_ref), 1e-30) < 1e-8,
        f"{fit.wald:.6e} vs {quad_ref:.6e}",
    )
    k = fit.rank_s
    f_ref = max(0.0, 1.0 - ((k - 2) / (dof + 2)) * dof * sigma2 / quad_ref)
    check(
        "JS factor matches the closed form",
        abs(f_js - f_ref) < 1e-9,
        f"{f_js:.10f} vs {f_ref:.10f}",
    )
    check(
        "at FULL rank the retained rank equals the shrunk column count",
        fit.rank_s == idx_s.size,
        f"r={fit.rank_s}, k={idx_s.size}",
    )

    # positive part engages when there is no signal
    rng = np.random.default_rng(11)
    y_null = rng.standard_normal(window)
    ycn = y_null - y_null.mean()
    f0fit = U._BlockFit(gram, xc.T @ ycn, idx_b, idx_s)
    b0 = np.zeros(x.shape[1])
    b0[idx_s] = f0fit.beta_s
    b0[idx_b] = f0fit.backbone(f0fit.beta_s)
    s0, d0 = U._sigma2_hat(
        float(ycn @ ycn), b0, xc.T @ ycn, window, f0fit.rank_b + f0fit.rank_s
    )
    f0 = U._js_factor_rank(f0fit.wald, f0fit.rank_s, s0, d0)
    check(
        "positive part engages on pure noise (factor collapses toward 0)",
        f0 < 0.5,
        f"factor {f0:.4f} on a null target",
    )

    # ROTATION INVARIANCE of the exact factor (why the PC arm is diagonal)
    q, _ = np.linalg.qr(np.random.default_rng(5).standard_normal((idx_s.size,) * 2))
    xr = x.copy()
    xr[:, idx_s] = x[:, idx_s] @ q
    xrc = xr[:window] - xr[:window].mean(0)
    gr, rr = xrc.T @ xrc, xrc.T @ yc
    fitr = U._BlockFit(gr, rr, idx_b, idx_s)
    br = np.zeros(x.shape[1])
    br[idx_s] = fitr.beta_s
    br[idx_b] = fitr.backbone(fitr.beta_s)
    sr, dr = U._sigma2_hat(syy_c, br, rr, window, fitr.rank_b + fitr.rank_s)
    fr = U._js_factor_rank(fitr.wald, fitr.rank_s, sr, dr)
    check(
        "EXACT JS factor is INVARIANT under an orthogonal rotation of the block",
        abs(fr - f_js) < 1e-8,
        f"{fr:.10f} vs {f_js:.10f} — so an exact-JS PC-basis arm would be a "
        "provable no-op; the shipped PC arm uses the DIAGONAL form",
    )

    # --- NPEB recovers a planted two-group prior ---
    rng = np.random.default_rng(21)
    m = 1200
    is_signal = rng.random(m) < 0.1
    mu_true = np.where(is_signal, rng.normal(3.0, 0.5, m), 0.0)
    z = mu_true + rng.standard_normal(m)
    mu_hat = U._tweedie_shrink(z)
    # exact Bayes posterior mean under the KNOWN planted prior, for reference
    w0, w1 = 0.9, 0.1
    d0 = w0 * np.exp(-0.5 * z**2)
    d1 = w1 / np.sqrt(1 + 0.25) * np.exp(-0.5 * (z - 3.0) ** 2 / 1.25)
    post = (d1 * (3.0 + 0.25 / 1.25 * (z - 3.0))) / (d0 + d1)
    mse_raw = float(np.mean((z - mu_true) ** 2))
    mse_npeb = float(np.mean((mu_hat - mu_true) ** 2))
    mse_oracle = float(np.mean((post - mu_true) ** 2))
    print(
        f"    NPEB on a planted 90/10 two-group prior (m={m}): "
        f"MSE raw z {mse_raw:.4f} -> NPEB {mse_npeb:.4f} "
        f"(oracle Bayes {mse_oracle:.4f})"
    )
    check(
        "NPEB beats the unshrunk estimate on a planted two-group prior",
        mse_npeb < mse_raw,
        f"{mse_npeb:.4f} < {mse_raw:.4f}",
    )
    check(
        "NPEB gets most of the way to the ORACLE Bayes rule",
        mse_npeb < mse_raw - 0.5 * (mse_raw - mse_oracle),
        f"closed {100 * (mse_raw - mse_npeb) / (mse_raw - mse_oracle):.0f}% of "
        "the achievable gap",
    )
    check(
        "NPEB shrinkage is NON-CONSTANT (not ridge in disguise)",
        float(np.std(mu_hat / np.where(np.abs(z) > 1e-9, z, np.nan))) > 0.05,
        "a constant factor would BE ridge — the negative result these arms avoid",
    )
    check(
        "NPEB shrinks the null group harder than the signal group",
        float(np.mean(np.abs(mu_hat[~is_signal])))
        < float(np.mean(np.abs(mu_hat[is_signal]))),
    )
    check("NPEB output is finite", bool(np.all(np.isfinite(mu_hat))))
    check(
        "degenerate spread returns the identity (no shrinkage, no crash)",
        np.array_equal(U._tweedie_shrink(np.zeros(50)), np.zeros(50)),
    )

    # --- profile persistence + zero tuned hyperparameters ---
    _, prof = U._walk_shrink(x, yv, window, lo, hi, segs, estimator="npeb")
    check("shrink profile is persisted", len(prof) > 0, f"{len(prof)} boundaries")
    r0 = prof[0]
    for key in ("row", "n_coef", "mean", "median", "frac_below_0p1", "deciles"):
        check(f"profile carries '{key}'", key in r0)
    check("profile deciles has 9 entries", len(r0["deciles"]) == 9)
    for arm in _SHRINK_ARM_NAMES:
        spec = U.ARMS[arm]
        check(
            f"{arm}: kind='shrink', oos_mult=2",
            spec.kind == "shrink" and spec.oos_mult == 2,
        )
        check(
            f"{arm}: estimator tag is not a BLOCK_TUNE_GRIDS key (no grid exists)",
            spec.grid not in U.BLOCK_TUNE_GRIDS and spec.grid not in U.ESTIMATOR_GRIDS,
            f"grid='{spec.grid}'",
        )
    check(
        "blk3_js_tuned uses blk3_tuned's EXACT design",
        [b for b, _ in U.ARMS["blk3_js_tuned"].blocks]
        == ["backbone", "exog_all", "product"],
    )
    check(
        "the PC-basis twin differs from its raw twin ONLY in the exog block",
        [b for b, _ in U.ARMS["blk3_js_pcbasis_tuned"].blocks][1] == "exog_rot"
        and [b for b, _ in U.ARMS["blk3_jsDiag_tuned"].blocks][1] == "exog_all"
        and U.ARMS["blk3_js_pcbasis_tuned"].grid
        == U.ARMS["blk3_jsDiag_tuned"].grid
        == "js_diag",
    )


# ── M. proper PCR: tensor, gate, ordering ─────────────────────────────────────
def _pcr_panel(n=2000, window=400, n_stem=12, n_ind=5, seed=3) -> U._Panel:
    """Fixture panel with the REAL name grammar and a uniform value tensor, so
    the production selectors (_value_slab_cols, _classify) apply unchanged."""
    rng = np.random.default_rng(seed)
    wins = [1, 2, 4, 8]
    stems = [f"q{i:02d}" for i in range(n_stem)]
    names = [f"har_ma_{w}" for w in (1, 2, 4, 8)]
    names += [f"adj_{s}_ma_{w}" for w in wins for s in stems]
    names += [f"{s}_avail_ma_{w}" for w in wins for s in stems[:n_ind]]
    names += ["is_open", "is_close", "is_overnight", "hour"]
    p = len(names)
    fac = rng.standard_normal((n, 5))
    x = np.empty((n, p))
    for j in range(p):
        x[:, j] = fac @ rng.standard_normal(5) + 0.7 * rng.standard_normal(n)
    y = x[:, :3].sum(1) + rng.standard_normal(n)
    return U._Panel(
        X=np.ascontiguousarray(x),
        y=y,
        baseline=np.ones(n),
        rv_raw=np.ones(n),
        t=np.arange(n).astype("datetime64[s]").astype("datetime64[ns]"),
        names=names,
        avail=np.ones((n, n_stem), dtype=bool),
        stem_index={s: i for i, s in enumerate(stems)},
    )


def section_m() -> None:
    print("\nM. PROPER PCR — TENSOR, GATE, ORDERING")

    # --- tensor structure of the REAL panel, from the dumped name list ---
    import json as _json

    try:
        real = _json.load(open("results/panel_columns.json"))["names"]
    except Exception:
        real = None
    if real:
        kinds: dict[str, int] = {}
        val: dict[str, set] = {}
        ind: dict[str, set] = {}
        for nm in real:
            k, stem, w = U._classify(nm)
            kinds[k] = kinds.get(k, 0) + 1
            if k == "value":
                val.setdefault(stem, set()).add(w)
            if k == "indicator":
                ind.setdefault(nm.rsplit("_ma_", 1)[0], set()).add(w)
        wins = sorted({w for ws in val.values() for w in ws})
        check(
            "REAL panel: value tensor is UNIFORM (every stem at every window)",
            all(sorted(ws) == wins for ws in val.values()),
            f"{len(val)} stems x {len(wins)} windows",
        )
        check(
            "REAL panel: indicator tensor is UNIFORM",
            all(sorted(ws) == wins for ws in ind.values()),
            f"{len(ind)} prefixes x {len(wins)} windows",
        )
        extras = kinds.get("har", 0) + kinds.get("regime", 0) + kinds.get("calendar", 0)
        total = len(val) * len(wins) + len(ind) * len(wins) + extras
        check(
            "REAL panel: tensor reconciles to the full column count EXACTLY",
            total == len(real),
            f"{len(val)}x{len(wins)} value + {len(ind)}x{len(wins)} indicator + "
            f"{extras} extras = {total} == {len(real)}",
        )
        check(
            "ROTATABLE rank is the VALUE stem count, not the indicator count",
            len(val) == 43 and len(ind) == 48 and len(wins) == 12,
            f"K_max = {len(val)} (brief said 92); indicators {len(ind)}; "
            f"windows {len(wins)}",
        )
        print(
            f"    real panel: 12 x {len(val)} value + 12 x {len(ind)} indicator "
            f"+ {extras} extras = {total}"
        )
    else:
        print("    (results/panel_columns.json unavailable — real-panel check skipped)")

    # --- ragged tensor must fail LOUDLY ---
    p = _pcr_panel()
    bad = list(p.names)
    bad.remove("adj_q00_ma_8")
    try:
        U._value_slab_cols(bad)
        check("ragged value tensor fails loudly", False, "no exception raised")
    except SystemExit as exc:
        check("ragged value tensor fails LOUDLY", "RAGGED" in str(exc))

    window = 400
    n_w = len({U._classify(nm)[2] for nm in p.names if U._classify(nm)[0] == "value"})
    _, live, n_live = U._rot_value_frame(p, window)
    print(f"    fixture: {n_live} live base quantities x {n_w} MA windows")

    # --- column counts: K x n_windows ---
    for k_req in (None, 8, 5, 3):
        d = U._rot_value_design(p, window, k_req, "variance")
        kk = n_live if k_req is None else k_req
        check(
            f"rotated design is K x windows = {kk} x {n_w}",
            d.shape[1] == kk * n_w,
            str(d.shape),
        )
    try:
        U._rot_value_design(p, window, n_live + 1, "variance")
        check("K beyond live rank fails loudly", False, "no exception")
    except SystemExit as exc:
        check("K beyond the live rank fails LOUDLY", "exceeds" in str(exc))

    # --- SCORES ARE NEVER STANDARDIZED (the instruction that matters most) ---
    d_full = U._rot_value_design(p, window, None, "variance")
    sds = d_full[window : 2 * window].std(0)
    check(
        "scores are NOT standardized (frame-window sds are NOT all 1)",
        float(np.std(sds)) > 1e-6,
        f"score sd range [{sds.min():.4f}, {sds.max():.4f}] — per-score "
        "standardization is what broke the PC-ladder",
    )

    # --- ORDERINGS coincide at K = full (free correctness check) ---
    d_var = U._rot_value_design(p, window, None, "variance")
    d_pred = U._rot_value_design(p, window, None, "predictive")
    # bitwise column-multiset equality, not a summary statistic
    cols_var = sorted(
        np.ascontiguousarray(d_var[:, j]).tobytes() for j in range(d_var.shape[1])
    )
    cols_pred = sorted(
        np.ascontiguousarray(d_pred[:, j]).tobytes() for j in range(d_pred.shape[1])
    )
    check(
        "at K=FULL the two orderings keep BITWISE-IDENTICAL column multisets",
        cols_var == cols_pred,
        "both keep everything — a free correctness check on the ranking code",
    )
    check(
        "at K=FULL predictive is a PERMUTATION of variance (not a new design)",
        d_var.shape == d_pred.shape,
        str(d_var.shape),
    )
    # ...and at K < full they genuinely differ
    check(
        "at K<full the two orderings select DIFFERENT directions",
        not np.array_equal(
            U._rot_value_design(p, window, 3, "variance"),
            U._rot_value_design(p, window, 3, "predictive"),
        ),
    )

    # --- CAUSALITY of the predictive ranking ---
    p2 = _pcr_panel()
    p2.X[2 * window :] += 50.0 * np.random.default_rng(9).standard_normal(
        p2.X[2 * window :].shape
    )
    p2.y[2 * window :] += 50.0
    check(
        "PREDICTIVE ordering is CAUSAL: post-frame-window rows cannot move it",
        np.array_equal(
            U._rot_value_design(p, window, 5, "predictive")[: 2 * window],
            U._rot_value_design(p2, window, 5, "predictive")[: 2 * window],
        ),
        "frame window is rows [W, 2W), which precede every scored bar",
    )

    # --- THE GATE ---
    # At K=full the rotated design is an orthogonal reparameterization of the
    # standardized unrotated design, and exact JS is rotation-invariant, so the
    # two must agree to machine precision END TO END.
    ind_cols = U._cols(p.names, {"indicator"})
    back = p.X[:, U._backbone_cols(p.names)]
    cols = U._value_slab_cols(p.names)
    wins_sorted = sorted(cols)
    unrot = np.empty((len(p.y), n_live * n_w))
    for j, w in enumerate(wins_sorted):
        slab = p.X[:, cols[w][live]]
        fw = slab[window : 2 * window]
        sd = fw.std(0)
        sd = np.where(sd > U._DEGENERATE_SD, sd, 1.0)
        unrot[:, j::n_w] = (slab - fw.mean(0)) / sd
    f_rot = np.hstack([back, d_full, p.X[:, ind_cols]])
    f_unrot = np.hstack([back, unrot, p.X[:, ind_cols]])
    nb = back.shape[1]
    segs = [
        (0, nb, "backbone"),
        (nb, nb + d_full.shape[1], "exog"),
        (nb + d_full.shape[1], f_rot.shape[1], "exog"),
    ]
    lo, hi = 2 * window, 2 * window + 30
    yr, _ = U._walk_shrink(f_rot, p.y, window, lo, hi, segs, estimator="js")
    yu, _ = U._walk_shrink(f_unrot, p.y, window, lo, hi, segs, estimator="js")
    rel = float(np.max(np.abs(yr - yu)) / max(np.max(np.abs(yu)), 1e-300))
    check(
        "GATE: K=full rotated JS == unrotated JS to machine precision",
        rel < 1e-9,
        f"max relative fitted-value difference {rel:.3e} over {hi - lo} bars",
    )
    print(
        f"    GATE tolerance achieved: {rel:.3e} relative (max abs "
        f"{np.max(np.abs(yr - yu)):.3e})"
    )

    # --- arm wiring ---
    for arm in _PCR_ARM_NAMES:
        spec = U.ARMS[arm]
        check(
            f"{arm}: kind='shrink', js, oos_mult=2",
            spec.kind == "shrink" and spec.grid == "js" and spec.oos_mult == 2,
        )
        blocks = [b for b, _ in spec.blocks]
        check(
            f"{arm}: backbone + rotated value + indicators, NO product",
            blocks[0] == "backbone"
            and blocks[1].startswith("rotval:")
            and blocks[2] == "avail_ind"
            and "product" not in blocks,
            str(blocks),
        )
    g = U.ARMS["blk2_gated_tuned"]
    check(
        "blk2_gated_tuned is blk2_tuned's design at oos_mult=2 (matched rows)",
        [b for b, _ in g.blocks] == [b for b, _ in U.ARMS["blk2_tuned"].blocks]
        and g.kind == "blocks_tuned"
        and g.oos_mult == 2
        and U.ARMS["blk2_tuned"].oos_mult == 1,
        "blk2_tuned scores 273,554 rows; the gated twin scores 248,686",
    )
    check(
        "matched-K ordering pairs exist at BOTH 30 and 20",
        all(
            f"blk2_rot{fam}{k}_js" in U.ARMS
            for fam in ("Var", "Pred")
            for k in ("ThirtyK", "TwentyK")
        ),
    )


# ── N. rank-deficient designs — the regression test for the 2026-08-07 defect ─
def _rank_deficient_fixture(n=1200, window=400, seed=17):
    """Design that REPRODUCES the real defect: live signal columns plus dead /
    constant / duplicated availability indicators, exactly the structure that
    made the production gram singular on 100% of bars.

    Every synthetic before this one used full-rank fixtures, which is precisely
    why the defect reached the cluster. This fixture is the regression test:
    run against the SHIPPED code as of the first grid-free submission it fails
    both assertions below (factor is 1.0, mean factor 1.0).
    """
    rng = np.random.default_rng(seed)
    p_back, p_live, p_dead, p_const, p_dup = 6, 25, 8, 6, 5
    back = rng.standard_normal((n, p_back))
    live = rng.standard_normal((n, p_live))
    dead = np.zeros((n, p_dead))  # never go live inside the window
    const = np.ones((n, p_const)) * rng.standard_normal(p_const)  # constant
    dup = live[:, :p_dup].copy()  # exact duplicates -> exact collinearity
    x = np.hstack([back, live, dead, const, dup])
    beta = np.zeros(x.shape[1])
    beta[:p_back] = rng.standard_normal(p_back)
    beta[p_back : p_back + 4] = np.array([0.7, -0.5, 0.4, 0.3])
    y = x @ beta + rng.standard_normal(n)
    segs = [(0, p_back, "backbone"), (p_back, x.shape[1], "exog")]
    n_shrunk = x.shape[1] - p_back
    true_rank = p_live  # dead + const + dup add nothing to the row space
    return x, y, segs, window, n_shrunk, true_rank


def section_n() -> None:
    print("\nN. RANK-DEFICIENT DESIGNS (regression test for the 2026-08-07 defect)")
    x, y, segs, window, n_shrunk, true_rank = _rank_deficient_fixture()
    idx_b = np.arange(0, segs[0][1])
    idx_s = np.arange(segs[0][1], x.shape[1])
    xw = x[:window]
    xc, yc = xw - xw.mean(0), y[:window] - y[:window].mean()
    gram, rhs = xc.T @ xc, xc.T @ yc

    check(
        "fixture gram IS singular (Cholesky must fail — the production case)",
        _cholesky_fails(gram),
        "this is the NORMAL case on the real design, not an anomaly",
    )
    rcond = U.roll_rank_rcond(window)
    check(
        "retention tolerance is DERIVED from the update path, not a default",
        rcond > U.PINV_RCOND,
        f"n*eps = {rcond:.3e} vs PINV_RCOND {U.PINV_RCOND:.3e} — the library "
        "default sits BELOW the rolled gram's noise floor",
    )
    # THE INVARIANT PROPERTY (whether the library default happens to
    # over-retain is fixture-dependent and is reported, not asserted): the
    # derived tolerance must reproduce the covariance of the EXACTLY-FORMED
    # gram. That is the quantity the shrinkage is computed from, and it is
    # what the library default destroyed on the production design.
    xc_e = x[:window] - x[:window].mean(0)
    gram_e = xc_e.T @ xc_e
    fit_exact = U._BlockFit(gram_e, rhs, idx_b, idx_s, rcond)
    fit_loose = U._BlockFit(gram, rhs, idx_b, idx_s, U.PINV_RCOND)
    fit_derived = U._BlockFit(gram, rhs, idx_b, idx_s, rcond)
    cov_d = fit_derived.diag_cov(1.0).max()
    print(
        f"    diag(G^+) max: exact gram {fit_exact.diag_cov(1.0).max():.4e} | "
        f"rolled@derived {cov_d:.4e} | rolled@PINV_RCOND "
        f"{fit_loose.diag_cov(1.0).max():.4e}  (ranks "
        f"{fit_exact.rank_s}/{fit_derived.rank_s}/{fit_loose.rank_s})"
    )
    check(
        "derived tolerance reproduces the EXACT gram's covariance scale",
        abs(cov_d - fit_exact.diag_cov(1.0).max())
        / max(fit_exact.diag_cov(1.0).max(), 1e-300)
        < 1e-6,
        "the rolled gram is made to agree with the exactly-formed one",
    )
    check(
        "derived tolerance never retains MORE than the library default",
        fit_derived.rank_s <= fit_loose.rank_s,
        "a higher cutoff can only discard directions, never admit them",
    )
    fit = U._BlockFit(gram, rhs, idx_b, idx_s, rcond)
    print(
        f"    shrunk columns {n_shrunk}, retained rank r_s={fit.rank_s} "
        f"(expected {true_rank}), backbone rank r_b={fit.rank_b}, "
        f"row-space fraction {fit.rank_s / n_shrunk:.3f}"
    )
    check(
        "retained rank recovers the TRUE row-space dimension",
        fit.rank_s == true_rank,
        f"r_s={fit.rank_s} vs true {true_rank}",
    )
    check(
        "rank is strictly BELOW the shrunk column count (deficiency is real)",
        fit.rank_s < n_shrunk,
        f"{fit.rank_s} < {n_shrunk}",
    )
    beta = np.zeros(x.shape[1])
    beta[idx_s] = fit.beta_s
    beta[idx_b] = fit.backbone(fit.beta_s)
    sigma2, dof = U._sigma2_hat(
        float(yc @ yc), beta, rhs, window, fit.rank_b + fit.rank_s
    )
    check(
        "sigma^2 dof uses the RETAINED RANK, not the column count",
        dof == window - (fit.rank_b + fit.rank_s) - 1,
        f"dof={dof} = {window} - ({fit.rank_b}+{fit.rank_s}) - 1 "
        f"(column-count convention would give {window - x.shape[1] - 1})",
    )
    f = U._js_factor_rank(fit.wald, fit.rank_s, sigma2, dof)

    # (a) THE ASSERTION THAT WOULD HAVE CAUGHT THE DEFECT
    check(
        "(a) JS factor is a real number strictly in (0, 1) on a SINGULAR design",
        isinstance(f, float) and 0.0 < f < 1.0,
        f"factor = {f!r} — the shipped code returned exactly 1.0 here",
    )
    # dead columns carry negligible coefficient and negligible variance, and
    # are EXCLUDED from the empirical prior by the row-space mask
    dead_local = np.arange(25, 25 + 8)  # dead block inside the shrunk set
    live_local = np.arange(0, 25)
    scale = float(np.max(np.abs(fit.beta_s)))
    check(
        "dead columns carry negligible coefficient under min-norm",
        float(np.max(np.abs(fit.beta_s[dead_local]))) < 1e-12 * max(scale, 1e-300),
        f"max|beta_dead|/max|beta| = "
        f"{np.max(np.abs(fit.beta_s[dead_local])) / max(scale, 1e-300):.2e}",
    )
    cd = fit.diag_cov(sigma2)
    check(
        "dead columns carry negligible sampling variance",
        float(np.max(cd[dead_local])) < rcond * float(np.max(cd)),
        f"max ratio {np.max(cd[dead_local]) / max(np.max(cd), 1e-300):.2e} "
        f"< tol {rcond:.2e}",
    )
    good = np.isfinite(cd) & (cd > rcond * float(np.max(cd)))
    check(
        "the row-space mask EXCLUDES every dead column",
        not bool(good[dead_local].any()),
        f"{int(good[dead_local].sum())} dead columns admitted",
    )
    check(
        "...and RETAINS every live column",
        bool(good[live_local].all()),
        f"{int(good[live_local].sum())} of {live_local.size} live retained",
    )
    check(
        "NPEB's prior is estimated on the ROW-SPACE columns only",
        int(good.sum()) < n_shrunk,
        f"{int(good.sum())} of {n_shrunk} columns enter the empirical prior",
    )

    # (b) the walk must actually shrink, on every estimator
    lo, hi = window, window + 40
    for est in ("js", "js_diag", "npeb"):
        yhat, prof = U._walk_shrink(x, y, window, lo, hi, segs, estimator=est)
        mf = float(np.mean([r["mean"] for r in prof]))
        n_sing = prof[0]["n_singular_bars"]
        print(
            f"    {est:8s} mean_factor={mf:.4f}  n_singular_bars={n_sing}  "
            f"r_s={prof[0]['rank_s']}  finite={bool(np.all(np.isfinite(yhat)))}"
        )
        check(
            f"(b) '{est}' mean shrinkage factor < 0.99 over the walk",
            mf < 0.99,
            f"mean_factor={mf:.4f} — the shipped code produced exactly 1.0000",
        )
        check(
            f"'{est}': n_singular_bars counts PATHOLOGY only (r<=2), not deficiency",
            n_sing == 0,
            f"{n_sing} of {hi - lo} bars — the shipped code reported every bar",
        )
        check(f"'{est}' forecasts are finite", bool(np.all(np.isfinite(yhat))))
        check(
            f"'{est}' profile records the retained rank",
            "rank_s" in prof[0] and prof[0]["rank_s"] == true_rank,
        )
    # causality still holds on the rank-deficient path
    for est in ("js", "npeb"):
        a, _ = U._walk_shrink(x, y, window, lo, hi, segs, estimator=est)
        xp, yp = x.copy(), y.copy()
        xp[hi:] += 1e3
        yp[hi:] += 1e3
        b, _ = U._walk_shrink(xp, yp, window, lo, hi, segs, estimator=est)
        check(f"CAUSAL on the rank-deficient path ('{est}')", np.array_equal(a, b))


def _cholesky_fails(mat: np.ndarray) -> bool:
    try:
        np.linalg.cholesky(mat)
        return False
    except np.linalg.LinAlgError:
        return True


_PCR_ARM_NAMES = (
    "blk2_rotVarFullK_js",
    "blk2_rotVarFortyK_js",
    "blk2_rotVarThirtyK_js",
    "blk2_rotVarTwentyK_js",
    "blk2_rotPredThirtyK_js",
    "blk2_rotPredTwentyK_js",
    "blk2_rotPredTenK_js",
    "blk2_rotPredFiveK_js",
)

_SHRINK_ARM_NAMES = (
    "blk3_js_tuned",
    "blk3_npeb_tuned",
    "blk3_js_pcbasis_tuned",
    "blk3_jsDiag_tuned",
)


if __name__ == "__main__":
    section_a()
    section_b()
    section_c()
    section_c3()
    section_d()
    section_e()
    section_f()
    section_g()
    section_h()
    section_i()
    section_j()
    section_k()
    section_l()
    section_m()
    section_n()
    print("\n" + ("ALL CHECKS PASSED" if not FAIL else f"FAILURES: {FAIL}"))
    sys.exit(1 if FAIL else 0)
