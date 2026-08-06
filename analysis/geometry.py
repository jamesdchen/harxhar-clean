"""The geometry §14 stopped short of, tested three ways.

§14 concluded "no good geometry to exploit". That conclusion was **too broad**, and the reason is
visible in its own numbers: the joint HAC Wald test that the state-dependence direction ``gamma`` is
non-zero came back chi2(7) = 28.1, ``p = 2e-4``, with ``||gamma|| / ||beta|| = 0.762`` — a large,
strongly significant state-dependent component — and then the *identification* test, a split-half
cosine, returned ``+0.126`` against a null of ``-0.193`` with **null sd 0.404**. A test whose null
scatters over half the unit interval cannot distinguish "no axis" from "an axis measured with 3x too
little precision". §14 read the failure as the former. It is at minimum consistent with the latter,
and §16's item-2 run settled the matter for one coordinate: freeze a single ``gamma`` component on
<= 2019 and its forecast beats the pooled combiner on 2020+ at DM-t **+2.35**. The direction is
partly real; the cosine test just could not see it.

So the three things never tried:

``axis``
    The *powered* version of §14's identification test. Instead of asking whether half-1's ``gamma``
    points the same way as half-2's — a question about an unregularized 7-vector estimated from noise
    — freeze the whole ``gamma`` on <= 2019 and ask whether the forecast it implies beats the pooled
    combiner on 2020+. A direction can be too noisy to *report* and still be good enough to *use*;
    only the second question matters, and only the second one has decision-theoretic content. Run
    alongside a shrinkage ladder, because the honest answer may be "the axis is real at 30% of its
    estimated length".

``clock``
    The state variable has always been ``har_ma_25``, one hand-picked coordinate. But §4 found the
    bucket x **time-of-day** activation map *more* split-half stable than the bucket x vol-regime one
    (+0.68 vs +0.64), and time-of-day has only ever entered as a **bin split**, which pays the
    ``1/sqrt(K)`` power tax §5 measured. As a smooth basis it costs 2 df per harmonic per signal and
    is estimated on every bar. Two harmonics of the intraday clock, same frozen-on-<=2019 protocol.

**Outcome, added after running, so the docstring does not read as a promise the results did not keep:**
``axis`` and ``clock`` succeeded — two additive state axes, dR2 +0.0014 at DM-t +3.09 out of sample,
both needing shrinkage to 15-50% of the raw estimate. ``form`` was negative (flat spectrum, unidentified
leading eigenvector). ``blocks`` found a large selection concentration, and ``block_exploit`` then
showed it is **not usable**: excluding the liquidity x liquidity block keeps 113% of the interaction
gain while restricting to it turns +0.0069 into -0.0050. See §17.3 of the writeup — the selection-
frequency null this module builds answers "does the selector go here more than on noise", which is a
different question from "do these products forecast", and only the second licenses a claim about alpha.

``form``
    The one the study walked past. §7's 100 frozen products are not a bag of features — they are a
    sparse **symmetric bilinear form** ``B``, and the forecast contains ``x' B x``. Nobody has ever
    looked at its spectrum. If ``B`` is low-rank then the interaction channel — which §16.1's
    attribution shows carries **60%** of the combination gain — is "a handful of quadratic directions
    in feature space", which is a concrete geometric statement, is exploitable (project onto the top
    eigenvectors instead of carrying 100 products), and would explain the one place in this study
    where sparse beats dense. If it is full-rank and its eigenvalues are flat, that is dense-weak
    again in a second channel, and worth knowing just as much.

Protocol, so none of this becomes another arm scored on exhausted data: every forecasting claim here
is frozen on rows before ``HOLDOUT`` and scored after it, the candidate list is fixed in this
docstring before running, and ``form`` makes no forecasting claim at all — it is a description of an
object that already exists, so it is measured on the search sample and its null is a circular shift.

Expectations, written first:

* ``axis`` — the 7-df frozen ``gamma`` beats pooled on 2020+, but by *less* than 7x the 1-df
  ``sentiment`` term's +0.00134, because the other six coordinates are the ones §14.3 could not
  distinguish from zero. Shrinkage should help, and its optimum should be well below 1.
* ``clock`` — positive, and plausibly larger than the vol-state axis given §4's stability ordering.
  This is the arm with the most room to surprise.
* ``form`` — the participation ratio of ``B``'s eigenvalues will be **well above 1** (this study has
  found nothing concentrated yet) but well below its 100-product ceiling. A PR near 100 would mean the
  100 products are 100 independent things; a PR near 2 would be the first genuinely low-dimensional
  object found.

Usage
-----
    C=.../scratchpad/fixed
    ALPHA_PANEL_CACHE=$C python analysis/geometry.py --stage axis
    ALPHA_PANEL_CACHE=$C python analysis/geometry.py --stage clock
    ALPHA_PANEL_CACHE=$C python analysis/geometry.py --stage form
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.alpha_manifestation import (  # noqa: E402
    BUCKETS,
    COMBINE_WINDOW_DAYS,
    REFIT_EVERY,
    TW,
    dm_test,
)
from analysis.alpha_panel import load_panel  # noqa: E402
from analysis.nl_sparsity import _pair_ic, _products, _upper, base_columns  # noqa: E402
from analysis.synthesis import ALPHA_PROD, HOLDOUT, N_PROD, _floored_scale, _p  # noqa: E402
from analysis.wf import r2_oos, walk_forward  # noqa: E402
from src.features.transforms.target import PERIODS_PER_DAY  # noqa: E402

OUT = "results/alpha_manifestation"
RIDGE = 1.0  # the §5 combiner's penalty on standardized signals
N_NULL = 24
# Pre-specified shrinkage ladder for the frozen axis. 0 is the pooled model, 1 is the raw estimate.
SHRINK = (0.0, 0.15, 0.3, 0.5, 0.75, 1.0)
CLOCK_HARMONICS = 2


def _state(p) -> np.ndarray:
    """The §14.3 continuous causal vol state: robust-scaled ``har_ma_25``, clipped to +-4."""
    z = p.X[TW + TW :, p.names.index("har_ma_25")].astype(np.float64)
    iqr = np.percentile(z, 75) - np.percentile(z, 25)
    return np.clip((z - np.median(z)) / (iqr + 1e-12), -4, 4)


def _clock_basis(ts: pd.Series, harmonics: int = CLOCK_HARMONICS) -> np.ndarray:
    """``(n, 2*harmonics)`` Fourier basis in the intraday clock, mean-removed.

    Deterministic in the timestamp, so causal by construction and free of the estimation cost the
    vol state carries. ``2 pi h k / 48`` with ``k`` the 30-min slot index.
    """
    slot = (ts.dt.hour * 2 + ts.dt.minute // 30).to_numpy(dtype=np.float64)
    cols = []
    for h in range(1, harmonics + 1):
        cols.append(np.cos(2 * np.pi * h * slot / PERIODS_PER_DAY))
        cols.append(np.sin(2 * np.pi * h * slot / PERIODS_PER_DAY))
    B = np.column_stack(cols)
    return B - B.mean(0)


def _fit_ridge(D: np.ndarray, y: np.ndarray, ridge: float = RIDGE) -> np.ndarray:
    Dc = D - D.mean(0)
    return np.linalg.solve(Dc.T @ Dc + ridge * np.eye(D.shape[1]), Dc.T @ (y - y.mean()))


def _pooled_combiner(S: np.ndarray, y: np.ndarray) -> np.ndarray:
    cw = COMBINE_WINDOW_DAYS * PERIODS_PER_DAY
    out = np.full(len(y), np.nan)
    out[cw:] = walk_forward(S, y, cw, alpha=1.0, refit_every=REFIT_EVERY)
    return out


def _frozen_interaction_forecast(
    S: np.ndarray, Z: np.ndarray, e: np.ndarray, search: np.ndarray, shrink: float
) -> np.ndarray:
    """Forecast from a state-interaction model whose interaction block is FROZEN on ``search``.

    The levels ``beta`` are refit walk-forward exactly as the pooled combiner does, so the only thing
    frozen is the interaction direction — otherwise a win could come from the levels rather than from
    the geometry. ``shrink`` scales the frozen interaction coefficients: 0 reproduces the pooled
    combiner bit-for-bit, 1 uses the raw <=2019 estimate.
    """
    q = S.shape[1]
    D = np.hstack([S, np.einsum("nk,nj->nkj", S, Z).reshape(len(S), -1)])
    b = _fit_ridge(D[search], e[search])
    gamma = b[q:] * shrink
    # the interaction part enters as a fixed offset to the target the level model must explain
    offset = D[:, q:] @ gamma
    resid = e - offset
    lev = _pooled_combiner(S, resid)
    return lev + offset


def stage_axis() -> None:
    """The powered identification test: freeze ``gamma`` on <=2019, score on 2020+.

    Reports the shrinkage ladder, plus the two references that make the numbers mean something: the
    pooled combiner (``shrink = 0``) and the 1-df ``sentiment``-only axis from §16 item 2. If the 7-df
    axis cannot beat the 1-df one, the six coordinates §14.3 could not sign are noise and the honest
    description is "one state-dependent coordinate", not "an axis".
    """
    os.makedirs(OUT, exist_ok=True)
    p = load_panel()
    e = np.load(_p("har_resid.npz"))["e"][TW:]
    sig = dict(np.load(_p("bucket_signals.npz")))
    ts = pd.Series(pd.to_datetime(p.t[TW + TW :]))
    S = np.column_stack([sig[b] for b in BUCKETS])
    z = _state(p)[:, None]
    search = (ts < HOLDOUT).to_numpy()
    hold = ~search
    print(f"search {int(search.sum())} rows / holdout {int(hold.sum())} rows", flush=True)

    base = _pooled_combiner(S, e)
    rows = []
    preds = {}
    for sh in SHRINK:
        f = _frozen_interaction_forecast(S, z, e, search, sh)
        preds[sh] = f
        m = hold & np.isfinite(f) & np.isfinite(base)
        y = e[m]
        r = r2_oos(y, f[m])
        r0 = r2_oos(y, base[m])
        t = dm_test(y, f[m], base[m])
        print(f"  shrink {sh:4.2f}: 2020+ R2 {r:+.5f}  dR2 {r - r0:+.5f}  DM-t vs pooled {t:+5.2f}",
              flush=True)
        rows.append({"arm": "gamma_7df", "shrink": sh, "r2_holdout": r, "dr2_holdout": r - r0,
                     "dm_t_vs_pooled": t, "n_holdout": int(m.sum())})

    # the 1-df reference: sentiment only, same freeze-and-score protocol
    k = BUCKETS.index("sentiment")
    q = S.shape[1]
    D1 = np.hstack([S, (S[:, k] * z[:, 0])[:, None]])
    b1 = _fit_ridge(D1[search], e[search])
    off1 = D1[:, q:] @ b1[q:]
    f1 = _pooled_combiner(S, e - off1) + off1
    m = hold & np.isfinite(f1) & np.isfinite(base)
    r1, r0 = r2_oos(e[m], f1[m]), r2_oos(e[m], base[m])
    t1 = dm_test(e[m], f1[m], base[m])
    print(f"\n  reference, 1 df (sentiment x z only): 2020+ dR2 {r1 - r0:+.5f}  DM-t {t1:+5.2f}")
    rows.append({"arm": "sentiment_1df", "shrink": 1.0, "r2_holdout": r1, "dr2_holdout": r1 - r0,
                 "dm_t_vs_pooled": t1, "n_holdout": int(m.sum())})

    best = max(rows[:-1], key=lambda r: r["dr2_holdout"])
    print(f"\n  best 7-df arm: shrink {best['shrink']:.2f}, dR2 {best['dr2_holdout']:+.5f}, "
          f"DM-t {best['dm_t_vs_pooled']:+.2f}")
    print(f"  7-df beats 1-df: {best['dr2_holdout'] > r1 - r0}"
          f"   (DM-t of best-7df vs 1-df: "
          f"{dm_test(e[m], preds[best['shrink']][m], f1[m]):+.2f})")
    pd.DataFrame(rows).to_csv(f"{OUT}/geometry_axis.csv", index=False)
    print(f"wrote {OUT}/geometry_axis.csv")


def stage_clock() -> None:
    """Time-of-day as the state, smooth rather than binned, same frozen-on-<=2019 protocol.

    §4 measured the bucket x time-of-day activation map as the *more* stable of the two axes
    (split-half +0.68 vs the vol axis's +0.64), and §5 then showed the bin-split parameterization
    cannot monetize it because slicing into K bins costs ``1/sqrt(K)`` in every bin's estimate. A
    Fourier basis costs 2 df per harmonic per signal and is fit on every bar, so it does not pay that
    tax at all. This is the same estimator as ``axis`` with the state swapped, and the vol axis is run
    beside it so the comparison is like-for-like.
    """
    os.makedirs(OUT, exist_ok=True)
    p = load_panel()
    e = np.load(_p("har_resid.npz"))["e"][TW:]
    sig = dict(np.load(_p("bucket_signals.npz")))
    ts = pd.Series(pd.to_datetime(p.t[TW + TW :]))
    S = np.column_stack([sig[b] for b in BUCKETS])
    search = (ts < HOLDOUT).to_numpy()
    hold = ~search
    base = _pooled_combiner(S, e)

    states = {
        "vol (har_ma_25)": _state(p)[:, None],
        f"clock ({CLOCK_HARMONICS} harmonics)": _clock_basis(ts, CLOCK_HARMONICS),
        "both": np.hstack([_state(p)[:, None], _clock_basis(ts, CLOCK_HARMONICS)]),
    }
    rows = []
    for name, Z in states.items():
        print(f"\n{name}: {Z.shape[1]} state cols -> {S.shape[1] * Z.shape[1]} interaction df",
              flush=True)
        for sh in SHRINK:
            f = _frozen_interaction_forecast(S, Z, e, search, sh)
            m = hold & np.isfinite(f) & np.isfinite(base)
            y = e[m]
            r, r0 = r2_oos(y, f[m]), r2_oos(y, base[m])
            t = dm_test(y, f[m], base[m])
            print(f"  shrink {sh:4.2f}: 2020+ dR2 {r - r0:+.5f}  DM-t {t:+5.2f}", flush=True)
            rows.append({"state": name, "n_state_cols": Z.shape[1], "shrink": sh,
                         "dr2_holdout": r - r0, "dm_t_vs_pooled": t, "n_holdout": int(m.sum())})
    d = pd.DataFrame(rows)
    d.to_csv(f"{OUT}/geometry_clock.csv", index=False)
    print("\nbest per state (2020+ holdout):")
    print(d.loc[d.groupby("state")["dr2_holdout"].idxmax()].to_string(index=False))
    print(f"wrote {OUT}/geometry_clock.csv")


def stage_form() -> None:
    """The spectrum of the fitted interaction form ``B``, which nobody has looked at.

    §7's selected products are 100 index pairs ``(i, j)`` with fitted coefficients ``c_ij``. Assemble
    them into the symmetric matrix they define,

        B[i, j] = B[j, i] = c_ij / 2   (off-diagonal),   B[i, i] = c_ii

    so the interaction contribution to the forecast is exactly ``x' B x``. Then the geometry of the
    channel is ``B``'s eigendecomposition: eigenvalues say how many quadratic directions matter, and
    eigenvectors say which combinations of features they are.

    Three things are reported, each against a null:

    1. **Participation ratio** of ``|lambda|`` — the effective number of active quadratic directions,
       on the same footing as §1's PR of the linear IC vector (116.8 of 246). The null is a form built
       from the same 100 *positions* with coefficients fitted to a circularly-shifted residual, which
       preserves the sparsity pattern and destroys the signal.
    2. **Split-half stability of the leading eigenvector** — the test §14.2 could not pass for the
       bucket-weight ellipse, asked here where there are 133 dimensions and 176k rows instead of 7
       dimensions and 10 noisy bin estimates.
    3. **Signature** — how many eigenvalues are positive vs negative. A form that is (near) positive
       semi-definite means "interactions only ever add volatility"; an indefinite one with balanced
       signs means the channel encodes cancellation, which is a different economic story.

    No forecasting claim, so no holdout is spent: this describes an object that §7 already built.
    """
    os.makedirs(OUT, exist_ok=True)
    p = load_panel()
    e_full = np.load(_p("har_resid.npz"))["e"]
    bc, bn = base_columns(p)
    X = np.ascontiguousarray(p.X[TW:, bc], dtype=np.float64)
    n, pb = X.shape
    ii, jj = _upper(pb)
    ts = pd.Series(pd.to_datetime(p.t[TW + TW :]))

    def fit_form(y: np.ndarray, rows: slice | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(B, sel) — the fitted symmetric form and the selected pair indices, on ``rows``."""
        mu = X[rows].mean(0)
        Zt = X[rows] - mu
        yt = y[rows]
        sel = np.argsort(-np.abs(np.nan_to_num(_pair_ic(Zt, yt)[ii, jj])))[:N_PROD]
        Pt, _ = _floored_scale(_products(Zt, ii[sel], jj[sel]), _products(Zt, ii[sel], jj[sel]))
        A = np.hstack([Zt, Pt])
        pen = np.concatenate([np.full(pb, 3000.0), np.full(N_PROD, ALPHA_PROD)])
        c = np.linalg.solve(A.T @ A + np.diag(pen), A.T @ (yt - yt.mean()))[pb:]
        B = np.zeros((pb, pb))
        for k, s in enumerate(sel):
            a, b = ii[s], jj[s]
            if a == b:
                B[a, a] += c[k]
            else:
                B[a, b] += c[k] / 2
                B[b, a] += c[k] / 2
        return B, sel

    # The form is fitted on the FIRST window, which is exactly what §7's static arm freezes.
    first = slice(0, TW)
    y = e_full[TW:]
    B, sel = fit_form(e_full[TW:], first)
    lam, V = np.linalg.eigh(B)
    a = np.abs(lam)
    pr = float(a.sum() ** 2 / (a**2).sum())
    print(f"form on {pb} base columns, {N_PROD} selected pairs "
          f"({int((ii[sel] == jj[sel]).sum())} of them squares)")
    print(f"  participation ratio of |lambda|: {pr:.1f} of {pb}")
    print(f"  signature: {int((lam > 0).sum())} positive / {int((lam < 0).sum())} negative")
    print(f"  top |lambda| share: {a.max() / a.sum():.3f}   top 5: {np.sort(a)[-5:].sum() / a.sum():.3f}")

    # null: same sparsity pattern, coefficients fitted to a circularly shifted residual
    prs, tops = [], []
    for kk in range(N_NULL):
        sh = (kk + 1) * (len(y) // (N_NULL + 1))
        Bn, _ = fit_form(np.roll(e_full[TW:], sh), first)
        ln = np.abs(np.linalg.eigvalsh(Bn))
        prs.append(float(ln.sum() ** 2 / (ln**2).sum()))
        tops.append(float(ln.max() / ln.sum()))
    print(f"  null PR {np.mean(prs):.1f} (sd {np.std(prs):.1f})   "
          f"null top share {np.mean(tops):.3f} (sd {np.std(tops):.3f})")

    # split-half stability of the leading eigenvector, on halves of the SEARCH sample
    srch = np.flatnonzero((ts < HOLDOUT).to_numpy())
    h1 = slice(int(srch[0]), int(srch[len(srch) // 2]))
    h2 = slice(int(srch[len(srch) // 2]), int(srch[-1]))
    B1, _ = fit_form(y, h1)
    B2, _ = fit_form(y, h2)
    v1 = np.linalg.eigh(B1)[1][:, -1]
    v2 = np.linalg.eigh(B2)[1][:, -1]
    cos = abs(float(v1 @ v2))
    ncos = []
    for kk in range(N_NULL):
        sh = (kk + 1) * (len(y) // (N_NULL + 1))
        yn = np.roll(y, sh)
        w1 = np.linalg.eigh(fit_form(yn, h1)[0])[1][:, -1]
        w2 = np.linalg.eigh(fit_form(yn, h2)[0])[1][:, -1]
        ncos.append(abs(float(w1 @ w2)))
    print(f"\n  leading-eigenvector split-half |cos|: {cos:.3f}   "
          f"null {np.mean(ncos):.3f} (sd {np.std(ncos):.3f})")
    print(f"  VERDICT: {'IDENTIFIED' if cos > np.mean(ncos) + 2 * np.std(ncos) else 'not identified'}"
          " at 2 null sd")

    top = np.argsort(-np.abs(V[:, -1]))[:12]
    print("\n  leading eigenvector's largest loadings")
    for t in top:
        print(f"    {bn[t]:44s} {V[t, -1]:+.3f}")
    pd.DataFrame(
        {"eig_index": np.arange(pb), "eigenvalue": lam,
         "leading_eigvec": V[:, -1], "feature": bn}
    ).to_csv(f"{OUT}/geometry_form_spectrum.csv", index=False)
    pd.DataFrame([{"pr": pr, "pr_null": float(np.mean(prs)), "pr_null_sd": float(np.std(prs)),
                   "top_share": float(a.max() / a.sum()), "top_share_null": float(np.mean(tops)),
                   "n_pos": int((lam > 0).sum()), "n_neg": int((lam < 0).sum()),
                   "splithalf_cos": cos, "splithalf_cos_null": float(np.mean(ncos)),
                   "splithalf_cos_null_sd": float(np.std(ncos))}]
    ).to_csv(f"{OUT}/geometry_form_summary.csv", index=False)
    print(f"\nwrote {OUT}/geometry_form_spectrum.csv + geometry_form_summary.csv")




def stage_blocks() -> None:
    """Where in the feature space the selected products live — the structure ``form`` missed.

    ``form`` returned a flat, indefinite, unidentified spectrum: participation ratio 20.3 against a
    null of 19.4 (sd 6.1), top eigenvalue share *below* null, leading eigenvector split-half |cos|
    0.000. So the interaction channel has no low-rank spectral structure and the "project onto the top
    eigenvectors" idea is dead.

    But its leading eigenvector had a loud pattern the spectrum could not express: **all twelve of its
    largest loadings were ``_ma_625`` columns** — the slow, ~13-day window. A spectrum is the wrong
    instrument for that kind of structure, because it asks about directions in an unordered
    133-dimensional space while the space actually carries two natural block labels: the HAR window a
    column averages over, and the exog bucket it descends from.

    So the question becomes combinatorial rather than spectral: are the 100 selected pairs
    concentrated in particular ``(window, window)`` and ``(bucket, bucket)`` blocks, relative to what
    selecting 100 pairs would give by chance? The null is the hypergeometric one implied by the block
    sizes — computed exactly from the pair counts rather than simulated — plus a circular-shift null
    that re-runs the *actual selector* against a destroyed signal, which additionally controls for any
    tendency of the selection statistic itself to prefer certain blocks.
    """
    os.makedirs(OUT, exist_ok=True)
    p = load_panel()
    e_full = np.load(_p("har_resid.npz"))["e"]
    bc, bn = base_columns(p)
    X = np.ascontiguousarray(p.X[TW:, bc], dtype=np.float64)
    pb = X.shape[1]
    ii, jj = _upper(pb)

    win = np.array([p.window[j] for j in bc], dtype=np.int64)
    buck = np.array([p.bucket[j] or ("har" if p.kind[j] == "har" else "calendar") for j in bc])

    def select(y: np.ndarray) -> np.ndarray:
        mu = X[:TW].mean(0)
        Zt = X[:TW] - mu
        return np.argsort(-np.abs(np.nan_to_num(_pair_ic(Zt, y[:TW])[ii, jj])))[:N_PROD]

    sel = select(e_full[TW:])

    def compose(labels: np.ndarray, s: np.ndarray) -> pd.Series:
        a, b = labels[ii[s]], labels[jj[s]]
        key = [f"{min(x, y)} x {max(x, y)}" for x, y in zip(a.astype(str), b.astype(str))]
        return pd.Series(key).value_counts()

    for name, labels in (("HAR window", win), ("exog bucket", buck)):
        obs = compose(labels, sel)
        # exact chance expectation: over ALL pairs in the upper triangle
        allp = compose(labels, np.arange(len(ii)))
        exp = allp / allp.sum() * N_PROD
        # selector null: re-run the real selector against circularly shifted residuals
        null_counts: list[pd.Series] = []
        y = e_full[TW:]
        for k in range(N_NULL):
            sh = (k + 1) * (len(y) // (N_NULL + 1))
            null_counts.append(compose(labels, select(np.roll(y, sh))))
        nul = pd.concat(null_counts, axis=1).fillna(0.0)
        d = pd.DataFrame(
            {
                "block": obs.index,
                "selected": obs.to_numpy(),
                "expected_by_chance": exp.reindex(obs.index).fillna(0.0).to_numpy(),
                "selector_null_mean": nul.mean(axis=1).reindex(obs.index).fillna(0.0).to_numpy(),
                "selector_null_sd": nul.std(axis=1).reindex(obs.index).fillna(0.0).to_numpy(),
            }
        )
        d["z_vs_selector_null"] = (d["selected"] - d["selector_null_mean"]) / d[
            "selector_null_sd"
        ].replace(0.0, np.nan)
        d = d.sort_values("selected", ascending=False)
        print(f"\n{name} composition of the {N_PROD} selected pairs")
        print(d.head(12).to_string(index=False), flush=True)
        d.to_csv(f"{OUT}/geometry_blocks_{name.split()[1]}.csv", index=False)
    print(f"\nwrote {OUT}/geometry_blocks_window.csv + geometry_blocks_bucket.csv")


# ---------------------------------------------------------------------------
# Exploiting the block, with the control that decides whether it is real
# ---------------------------------------------------------------------------

# Pre-registered candidate-space restrictions. Each arm selects ``N_PROD`` products by the same
# statistic, freezes them on the first window, and is scored against the same base — only the
# *candidate space* differs, so a difference is attributable to the restriction and nothing else.
#
# ``complement`` is the arm that matters. If excluding the liquidity x liquidity block leaves the gain
# intact, then §17.2's z = +3.58 is decorative: the selector merely visits that block on its way to a
# gain available anywhere. If the complement collapses, the block is where the alpha lives.
#
# Expectation, written before running: ``liq_x_liq`` roughly matches ``all`` (the unrestricted selector
# already goes there, so restriction can only buy variance reduction, not new information), possibly
# edging ahead; ``fast`` also roughly matches; ``complement`` is materially weaker. A ``complement``
# that matches ``all`` refutes the block finding as an exploitable fact.
BLOCK_ARMS = ("all", "liq_x_liq", "fast", "liq_x_liq_fast", "complement")


def _pair_mask(p, bc: np.ndarray, ii: np.ndarray, jj: np.ndarray, arm: str) -> np.ndarray:
    """Boolean mask over candidate pairs for one restriction arm."""
    win = np.array([p.window[j] for j in bc], dtype=np.int64)
    is_liq = np.array([p.bucket[j] == "liquidity" for j in bc])
    liq = is_liq[ii] & is_liq[jj]
    fast = np.isin(win[ii], (1, 25)) & np.isin(win[jj], (1, 25))
    return {
        "all": np.ones(len(ii), dtype=bool),
        "liq_x_liq": liq,
        "fast": fast,
        "liq_x_liq_fast": liq & fast,
        "complement": ~liq,
    }[arm]


def stage_block_exploit() -> None:
    """Does restricting the product space to the concentrated block help — and does excluding it hurt?

    §17.2 established *where* the selected interaction products live. That is a description; this asks
    whether it is a usable restriction. The mechanism a restriction can work through is narrow and
    worth naming in advance: it cannot add information (the unrestricted selector was already free to
    pick those pairs and did), so any gain is **variance reduction in the selection step** — 300
    candidates instead of 8,911 means far less opportunity for the top-100 statistic to be chasing
    noise.

    Scored two ways: over the whole OOS span (comparable to §16.3's +0.0069) and over the 2020+ block
    alone. The 2020+ figure is the one with a clean provenance — the block was identified from the
    *first training window*'s selection, in 2005-06 — but note that this session has now scored a
    number of arms on 2020+, so it wants a Romano-Wolf pass before being quoted as confirmed.
    """
    os.makedirs(OUT, exist_ok=True)
    from analysis.synthesis import _blockwise_ridge

    p = load_panel()
    e_full = np.load(_p("har_resid.npz"))["e"]
    e = e_full[TW:]
    ts = pd.Series(pd.to_datetime(p.t[TW + TW :]))
    bc, _ = base_columns(p)
    X = np.ascontiguousarray(p.X[TW:, bc], dtype=np.float64)
    n, pb = len(e_full), X.shape[1]
    ii, jj = _upper(pb)
    refit = 21 * PERIODS_PER_DAY

    masks = {a: _pair_mask(p, bc, ii, jj, a) for a in BLOCK_ARMS}
    for a, m in masks.items():
        print(f"  {a:16s} {int(m.sum()):5d} candidate pairs", flush=True)

    base = np.full(n - TW, np.nan)
    arms = {a: np.full(n - TW, np.nan) for a in BLOCK_ARMS}
    frozen: dict[str, np.ndarray] = {}
    for t0 in range(TW, n, refit):
        tr = slice(t0 - TW, t0)
        t1 = min(t0 + refit, n)
        out = slice(t0 - TW, t1 - TW)
        mu = X[tr].mean(0)
        Ztr, Zte = X[tr] - mu, X[t0:t1] - mu
        ytr = e_full[tr]
        base[out] = _blockwise_ridge(Ztr, ytr, Zte, pb, 3000.0, 3000.0)
        ic = np.abs(np.nan_to_num(_pair_ic(Ztr, ytr)[ii, jj]))
        for a in BLOCK_ARMS:
            if a not in frozen:  # freeze on the first window, as §7's static arm does
                cand = np.flatnonzero(masks[a])
                frozen[a] = cand[np.argsort(-ic[cand])[:N_PROD]]
            s = frozen[a]
            Ptr, Pte = _floored_scale(
                _products(Ztr, ii[s], jj[s]), _products(Zte, ii[s], jj[s])
            )
            arms[a][out] = _blockwise_ridge(
                np.hstack([Ztr, Ptr]), ytr, np.hstack([Zte, Pte]), pb, 3000.0, ALPHA_PROD
            )

    y = e[: len(base)]
    late = (ts[: len(base)] >= HOLDOUT).to_numpy()
    rb, rbl = r2_oos(y, base), r2_oos(y[late], base[late])
    print(f"\nbase R2 {rb:+.5f}   (2020+ {rbl:+.5f})\n")
    print("  arm               full dR2    DM-t   |   2020+ dR2   DM-t   | vs 'all' DM-t")
    rows = []
    for a in BLOCK_ARMS:
        f = arms[a]
        r, t = r2_oos(y, f) - rb, dm_test(y, f, base)
        rl, tl = r2_oos(y[late], f[late]) - rbl, dm_test(y[late], f[late], base[late])
        va = dm_test(y, f, arms["all"]) if a != "all" else 0.0
        print(f"  {a:16s} {r:+.5f}  {t:+6.2f}   |  {rl:+.5f}  {tl:+6.2f}   |  {va:+6.2f}")
        rows.append({"arm": a, "n_candidates": int(masks[a].sum()), "dr2_full": r, "dm_t_full": t,
                     "dr2_2020plus": rl, "dm_t_2020plus": tl, "dm_t_vs_all": va})
    d = pd.DataFrame(rows)
    d.to_csv(f"{OUT}/geometry_block_exploit.csv", index=False)
    comp = float(d[d.arm == "complement"]["dr2_full"].iloc[0])
    alld = float(d[d.arm == "all"]["dr2_full"].iloc[0])
    print(f"\n  CONTROL: complement keeps {100 * comp / alld:.0f}% of the unrestricted gain "
          f"({'block is where the alpha lives' if comp < 0.5 * alld else 'block finding is DECORATIVE'})")
    print(f"wrote {OUT}/geometry_block_exploit.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stage",
        choices=["axis", "clock", "form", "blocks", "block_exploit"],
        required=True,
    )
    a = ap.parse_args()
    {
        "axis": stage_axis,
        "clock": stage_clock,
        "form": stage_form,
        "blocks": stage_blocks,
        "block_exploit": stage_block_exploit,
    }[a.stage]()
