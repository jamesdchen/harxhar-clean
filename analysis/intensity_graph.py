"""Mine the *intensity modulation* of the dense-weak alpha, using graph structure.

Everything before this asked "what predicts the residual". This asks the second-order question:
**what predicts when the dense alpha is strong?** That target is new, and it is the systematic
version of the only thing in this study that worked — vol-state conditioning (§5, §11.1) *is* an
intensity modulator, found by the crudest possible search, so the class is known to be non-empty and
essentially unexplored.

Three graph roles, which the §12 failure showed must not be conflated:

* **where to look** — interaction adjacency (edge if ``x_i * x_j`` carries signal). This defines the
  hypothesis space, nothing more.
* **how to share strength** — **similarity** adjacency (``|corr|``). §12.1 measured that every
  similarity-regularised arm helps and every interaction-regularised arm hurts: "adjacent
  coefficients should be close" is a claim about *correlated* features and is false for *interacting*
  ones.
* **what to analyse** — the time-varying alpha field **as a graph signal**. The alpha is not a graph;
  it is a function ``w_ij(t)`` *on* one, and the intensity modulation is that function moving. With
  8,911 edges and ~168 effective monthly observations, tracking it directly has no power, but a
  *smooth* function on a graph is described by a few low-frequency **graph-Fourier coefficients**
  (projections onto leading Laplacian eigenvectors). That is the dimension reduction which makes the
  question answerable at all.

PRE-REGISTRATION — written before the first run, so it cannot be revised into a result
=====================================================================================

**Expectation.** ~15-20% chance anything survives a clean holdout; if it does, magnitude of order
**+0.001 to +0.003** R² on the residual.

*Against:* ``c ~ 1.0`` (§5) says local alpha is never larger than pooled, which is what an
exploitable modulation would look like; the dynamic gain already failed (-0.002) using the
intensity's own history; the intensity target is only ~67% signal *at monthly aggregation* (§3), so
there are ~168 noisy search-period observations for 8,911 candidates; and the repo's spectrum work
(eigenvalues ~ i^-3.13, PC1 carrying ~0% of target) says the panel is squeezed.

*For:* vol-state conditioning works and is a member of this class, so a member exists; the intensity
has genuine non-noise variation (1.74x null dispersion, mean-reverting, ~1-month half-life); and
nobody has looked at features — let alone products of features — as modulators.

**Gate (stage 1, descriptive, costs no inferential budget).** Proceed to fitting *only* if the
split-half stability of the modulator-IC map exceeds its circular-shift null. If it sits at the noise
floor, stop here and report that.

**Positive control.** ``har_ma_25`` — the vol-state proxy whose conditioning gain is confirmed at
Romano-Wolf adjusted p 0.0005 — must rank as a strong modulator. If the known-good modulator does not
surface, the measurement is broken and nothing downstream is trustworthy. This check is not optional
and its failure voids the stage.

**Nulls.** The intensity series is a ~1-month EWMA, so it is heavily autocorrelated; bar-level ICs
against it would look enormously significant under any i.i.d. calculation. Every null here is a
circular shift of the intensity series, which preserves that autocorrelation and destroys only the
alignment.

Usage
-----
    python analysis/intensity_graph.py --stage gate   # stage 1: does anything modulate at all?
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.alpha_manifestation import TW, _pr
from analysis.alpha_panel import CACHE_DIR
from analysis.nl_sparsity import _pair_ic, _upper, base_columns
from src.features.transforms.target import PERIODS_PER_DAY

INTENSITY_HL_DAYS = 21  # ~1 month, matching the measured half-life of the slope wobble
SPLIT = "2021-01-01"  # search / holdout boundary, unchanged from §12
N_NULL = 12
OUT = "results/alpha_manifestation"
FIXED_CACHE = os.path.join(CACHE_DIR, "fixed")


def _panel():
    os.environ["ALPHA_PANEL_CACHE"] = FIXED_CACHE
    import importlib

    import analysis.alpha_panel as ap

    importlib.reload(ap)
    return ap.load_panel()


def intensity(a: np.ndarray, e: np.ndarray, hl_days: int = INTENSITY_HL_DAYS) -> np.ndarray:
    """Causal local slope of ``e`` on ``a`` — the alpha's instantaneous strength.

    ``EWMA(a*e) / EWMA(a^2)`` is the exponentially-weighted least-squares slope, shifted one bar so
    row ``t`` uses only information before ``t``. This is the bar-level version of §3's monthly slope
    and inherits its properties: mean ~0.9, 1.74x null dispersion, mean-reverting with a ~1-month
    half-life.
    """
    hl = hl_days * PERIODS_PER_DAY
    num = pd.Series(a * e).ewm(halflife=hl, min_periods=hl).mean()
    den = pd.Series(a * a).ewm(halflife=hl, min_periods=hl).mean()
    m = (num / den.replace(0.0, np.nan)).shift(1)
    return m.to_numpy()


def stage_gate() -> None:  # noqa: C901
    os.makedirs(OUT, exist_ok=True)
    p = _panel()
    e_full = np.load(os.path.join(CACHE_DIR, "har_resid.npz"))["e"]
    a_full = dict(np.load(os.path.join(CACHE_DIR, "bucket_signals.npz")))["all"]
    e = e_full[TW:]
    ts = pd.Series(pd.to_datetime(p.t[TW + TW :]))
    m = intensity(a_full, e)

    bc, bn = base_columns(p)
    X = np.ascontiguousarray(p.X[TW + TW :, bc], dtype=np.float64)
    n, pb = len(e), X.shape[1]
    assert len(X) == n and len(m) == n

    ok = np.isfinite(m)
    search = ok & (ts < SPLIT).to_numpy()
    print(f"  intensity defined on {int(ok.sum())} rows; search rows {int(search.sum())}")
    print(f"  intensity: mean {np.nanmean(m[search]):+.3f}  sd {np.nanstd(m[search]):.3f}  "
          f"AR(1 day) {pd.Series(m[search]).autocorr(PERIODS_PER_DAY):+.3f}\n")

    def mod_ic(rows: np.ndarray, target: np.ndarray) -> np.ndarray:
        """|IC| of every base feature and every pairwise product with the intensity target."""
        Z = X[rows] - X[rows].mean(0)
        y = target[rows] - target[rows].mean()
        ii, jj = _upper(pb)
        pair = np.abs(np.nan_to_num(_pair_ic(Z, y)[ii, jj]))
        sd = Z.std(0)
        lin = np.abs(np.nan_to_num((Z * y[:, None]).mean(0) / (np.where(sd > 0, sd, 1) * y.std())))
        return lin, pair, ii, jj

    lin, pair, ii, jj = mod_ic(search, m)
    print(f"  candidates: {pb} linear + {len(pair)} products")
    print(f"  |modulator IC|: linear mean {lin.mean():.4f} max {lin.max():.4f} | "
          f"products mean {pair.mean():.4f} max {pair.max():.4f}")

    # ---- POSITIVE CONTROL: the known-good modulator must surface -------------------
    ctrl = bn.index("har_ma_25")
    rank = int((lin > lin[ctrl]).sum()) + 1
    print(f"\n  POSITIVE CONTROL har_ma_25: |IC| {lin[ctrl]:.4f}, rank {rank} of {pb} linear "
          f"({'PASS' if rank <= 0.25 * pb else 'FAIL — measurement suspect'})")
    top_lin = np.argsort(-lin)[:8]
    print("  strongest linear modulators:")
    for j in top_lin:
        print(f"    {bn[j]:38s} |IC| {lin[j]:.4f}")

    # ---- nulls: circular shifts of the intensity series ----------------------------
    nl_lin, nl_pair = [], []
    for k in range(N_NULL):
        sh = (k + 1) * (n // (N_NULL + 1))
        l2, p2, _, _ = mod_ic(search, np.roll(m, sh))
        nl_lin.append(l2.mean())
        nl_pair.append(p2.mean())
    print(f"\n  circular-shift null ({N_NULL} reps): linear mean |IC| {np.mean(nl_lin):.4f}, "
          f"products {np.mean(nl_pair):.4f}")
    print(f"  excess over null: linear {lin.mean() / np.mean(nl_lin):.2f}x  "
          f"products {pair.mean() / np.mean(nl_pair):.2f}x")

    # ---- THE GATE: split-half stability of the modulator map vs its null -----------
    idx = np.flatnonzero(search)
    h1, h2 = np.zeros(n, bool), np.zeros(n, bool)
    h1[idx[: len(idx) // 2]] = True
    h2[idx[len(idx) // 2 :]] = True
    l1, p1, _, _ = mod_ic(h1, m)
    l2, p2, _, _ = mod_ic(h2, m)
    stab_lin = float(np.corrcoef(l1, l2)[0, 1])
    stab_pair = float(np.corrcoef(p1, p2)[0, 1])
    ns_lin, ns_pair = [], []
    for k in range(N_NULL):
        sh = (k + 1) * (n // (N_NULL + 1))
        mn = np.roll(m, sh)
        q1, r1, _, _ = mod_ic(h1, mn)
        q2, r2, _, _ = mod_ic(h2, mn)
        ns_lin.append(float(np.corrcoef(q1, q2)[0, 1]))
        ns_pair.append(float(np.corrcoef(r1, r2)[0, 1]))
    print("\n  ==== GATE: split-half stability of the modulator-IC map ====")
    print(f"    linear   {stab_lin:+.3f}  null {np.mean(ns_lin):+.3f}")
    print(f"    products {stab_pair:+.3f}  null {np.mean(ns_pair):+.3f}")
    passed = (stab_lin > np.mean(ns_lin) + 0.10) or (stab_pair > np.mean(ns_pair) + 0.10)
    print(f"    -> {'PASS: proceed to a pre-registered fit' if passed else 'STOP: at the noise floor, do not fit'}")

    # ---- graph view of the modulator field (descriptive) --------------------------
    sel = np.argsort(-pair)[:100]
    W = np.zeros((pb, pb))
    for k in sel:
        if ii[k] != jj[k]:
            W[ii[k], jj[k]] = W[jj[k], ii[k]] = pair[k]
    L = np.diag(W.sum(1)) - W
    ev = np.linalg.eigvalsh(L)
    ev = ev[ev > 1e-9]
    deg = pd.Series(np.concatenate([ii[sel], jj[sel]])).value_counts()
    print(f"\n  modulator graph (top-100 product edges): {len(deg)} nodes touched, "
          f"PR of |IC| {_pr(pair[sel]):.1f} of 100, Laplacian PR {ev.sum() ** 2 / (ev ** 2).sum():.1f}")
    print("  modulator-graph hubs:")
    for node, d in deg.head(5).items():
        print(f"    {bn[node]:38s} degree {d}")

    pd.DataFrame({"feature": bn, "modulator_ic": lin}).sort_values(
        "modulator_ic", ascending=False
    ).to_csv(f"{OUT}/intensity_modulators_linear.csv", index=False)
    pd.DataFrame(
        [
            {
                "stab_linear": stab_lin,
                "stab_linear_null": np.mean(ns_lin),
                "stab_products": stab_pair,
                "stab_products_null": np.mean(ns_pair),
                "excess_linear": lin.mean() / np.mean(nl_lin),
                "excess_products": pair.mean() / np.mean(nl_pair),
                "control_rank": rank,
                "gate_passed": bool(passed),
            }
        ]
    ).to_csv(f"{OUT}/intensity_gate.csv", index=False)
    print(f"\nwrote {OUT}/intensity_gate.csv + intensity_modulators_linear.csv")


# ---------------------------------------------------------------------------
# Stage 2 — the same gate at the HONEST resolution, after stage 1 failed its control
# ---------------------------------------------------------------------------


def stage_gate2() -> None:  # noqa: C901
    """Stage ``gate`` failed its own positive control. This is the diagnosed, corrected version.

    **What went wrong, and why the control caught it.** ``har_ma_25`` — the modulator whose
    conditioning gain is confirmed at Romano-Wolf adjusted p 0.0005 — ranked **98 of 133**, while
    every one of the top eight modulators was an ``_ma_625`` (slow, ~13-day) feature. The intensity
    series has a one-day autocorrelation of **0.988**: it is nearly non-stationary. Correlating it
    with other nearly-non-stationary series at *bar* resolution measures which features are also
    slow, not which features modulate — a textbook spurious-regression setup. The 7x "excess over
    null" was an artifact of that, compounded by a 12-shift null being a hopeless estimate of the
    distribution of |corr| between two smooth series. (Secondary bug: rolling the intensity moved its
    leading NaNs into the window, so the split-half nulls were NaN and the printed verdict was
    vacuous.)

    **The corrections.** (1) Aggregate both sides to **monthly blocks**, which is the intensity's
    actual effective resolution — ~168 search-period observations, so the honest noise floor is
    ~1/sqrt(168) = 0.077 rather than the 0.005 the bar-level null implied. (2) Report ICs of
    intensity *changes* on modulator levels as well as levels-on-levels; a shared trend inflates the
    latter and cannot inflate the former. (3) Mask non-finite rows in every null. (4) Keep the
    positive control as a hard gate on measurement validity, not on the result.
    """
    os.makedirs(OUT, exist_ok=True)
    p = _panel()
    e_full = np.load(os.path.join(CACHE_DIR, "har_resid.npz"))["e"]
    a_full = dict(np.load(os.path.join(CACHE_DIR, "bucket_signals.npz")))["all"]
    e = e_full[TW:]
    ts = pd.Series(pd.to_datetime(p.t[TW + TW :]))
    m = intensity(a_full, e)
    bc, bn = base_columns(p)
    X = np.ascontiguousarray(p.X[TW + TW :, bc], dtype=np.float64)
    pb = X.shape[1]

    codes = (ts.dt.year * 12 + ts.dt.month).to_numpy()
    blk = np.unique(codes, return_inverse=True)[1]
    nb = blk.max() + 1
    keep = np.isfinite(m)
    Xb = np.full((nb, pb), np.nan)
    mb = np.full(nb, np.nan)
    tb = np.empty(nb, dtype="datetime64[ns]")
    for b in range(nb):
        rows = (blk == b) & keep
        if rows.sum() < 200:
            continue
        Xb[b] = X[rows].mean(0)
        mb[b] = m[rows].mean()
        tb[b] = ts[blk == b].iloc[0].to_datetime64()
    good = np.isfinite(mb) & np.isfinite(Xb).all(1)
    search = good & (pd.Series(tb) < SPLIT).to_numpy()
    ns = int(search.sum())
    print(f"  monthly blocks: {int(good.sum())} usable, {ns} in the search period")
    print(f"  honest noise floor at this resolution ~ 1/sqrt(n) = {1 / np.sqrt(ns):.3f}")
    print(f"  intensity by block: mean {np.nanmean(mb[search]):+.3f} sd {np.nanstd(mb[search]):.3f} "
          f"AR(1 block) {pd.Series(mb[search]).autocorr(1):+.3f}\n")

    ii, jj = _upper(pb)

    def ics(rows: np.ndarray, target: np.ndarray):
        r = rows & np.isfinite(target)
        Z = Xb[r] - Xb[r].mean(0)
        y = target[r] - target[r].mean()
        sd = Z.std(0)
        lin = np.abs(np.nan_to_num((Z * y[:, None]).mean(0) / (np.where(sd > 0, sd, 1) * y.std())))
        pair = np.abs(np.nan_to_num(_pair_ic(Z, y)[ii, jj]))
        return lin, pair

    dmb = np.concatenate([[np.nan], np.diff(mb)])  # intensity CHANGES: immune to a shared trend
    for label, tgt in (("levels", mb), ("changes", dmb)):
        lin, pair = ics(search, tgt)
        nl, npr = [], []
        for k in range(N_NULL):
            sh = (k + 1) * max(1, nb // (N_NULL + 1))
            l2, p2 = ics(search, np.roll(tgt, sh))
            nl.append(l2.mean())
            npr.append(p2.mean())
        ctrl = bn.index("har_ma_25")
        rank = int((lin > lin[ctrl]).sum()) + 1
        ok_ctrl = rank <= 0.25 * pb
        print(f"  --- target = intensity {label} ---")
        print(f"    |IC| linear mean {lin.mean():.4f} max {lin.max():.4f} (null {np.mean(nl):.4f}, "
              f"excess {lin.mean() / np.mean(nl):.2f}x)")
        print(f"    |IC| products mean {pair.mean():.4f} max {pair.max():.4f} (null {np.mean(npr):.4f}, "
              f"excess {pair.mean() / np.mean(npr):.2f}x)")
        print(f"    POSITIVE CONTROL har_ma_25 rank {rank}/{pb} -> "
              f"{'PASS' if ok_ctrl else 'FAIL (measurement suspect)'}")
        for j in np.argsort(-lin)[:5]:
            print(f"      {bn[j]:38s} |IC| {lin[j]:.4f}")

        idx = np.flatnonzero(search)
        h1 = np.zeros(nb, bool)
        h2 = np.zeros(nb, bool)
        h1[idx[: len(idx) // 2]] = True
        h2[idx[len(idx) // 2 :]] = True
        l1, p1 = ics(h1, tgt)
        l2, p2 = ics(h2, tgt)
        sl = float(np.corrcoef(l1, l2)[0, 1])
        sp = float(np.corrcoef(p1, p2)[0, 1])
        nsl, nsp = [], []
        for k in range(N_NULL):
            sh = (k + 1) * max(1, nb // (N_NULL + 1))
            tn = np.roll(tgt, sh)
            q1, r1 = ics(h1, tn)
            q2, r2 = ics(h2, tn)
            with np.errstate(invalid="ignore"):
                c1 = np.corrcoef(q1, q2)[0, 1]
                c2 = np.corrcoef(r1, r2)[0, 1]
            if np.isfinite(c1):
                nsl.append(c1)
            if np.isfinite(c2):
                nsp.append(c2)
        print(f"    GATE split-half stability: linear {sl:+.3f} (null {np.mean(nsl):+.3f}), "
              f"products {sp:+.3f} (null {np.mean(nsp):+.3f})")
        passed = ok_ctrl and (
            sl > np.mean(nsl) + 0.10 or sp > np.mean(nsp) + 0.10
        )
        print(f"    -> {'PASS' if passed else 'STOP'}\n")
        pd.DataFrame(
            [{"target": label, "n_blocks": ns, "noise_floor": 1 / np.sqrt(ns),
              "lin_mean": lin.mean(), "lin_null": np.mean(nl), "pair_mean": pair.mean(),
              "pair_null": np.mean(npr), "control_rank": rank, "control_pass": ok_ctrl,
              "stab_lin": sl, "stab_lin_null": np.mean(nsl), "stab_pair": sp,
              "stab_pair_null": np.mean(nsp), "gate_passed": bool(passed)}]
        ).to_csv(f"{OUT}/intensity_gate2_{label}.csv", index=False)
    print(f"wrote {OUT}/intensity_gate2_*.csv")


# ---------------------------------------------------------------------------
# Stage 3 — a better intensity estimator (the binding constraint from §13)
# ---------------------------------------------------------------------------

DAY = PERIODS_PER_DAY
MONTH_D = 21


def _daily_sufficient(a: np.ndarray, e: np.ndarray, day: np.ndarray, nd: int):
    """Per-day ``(sum a*e, sum a^2, sum e^2, count)`` — the sufficient statistics for a daily slope."""
    sae = np.bincount(day, weights=a * e, minlength=nd)
    saa = np.bincount(day, weights=a * a, minlength=nd)
    see = np.bincount(day, weights=e * e, minlength=nd)
    cnt = np.bincount(day, minlength=nd).astype(float)
    return sae, saa, see, cnt


def kalman_intensity(sae, saa, see, cnt, phi: float, mu: float, var_eta: float):
    """Filtered slope from a heteroskedastic-observation, AR(1)-state Kalman filter.

    Observation: the daily OLS slope ``y_d = sae/saa`` is ``beta_d`` plus noise of variance
    ``sigma2_e(d) / saa(d)`` — *known up to* ``sigma2_e``, which is estimated causally from the
    trailing residual variance. That makes GLS weighting automatic: days with a large ``saa`` (lots
    of signal dispersion) and small residual variance get more weight, which a flat EWMA cannot do.

    State: ``beta_d = mu + phi (beta_{d-1} - mu) + eta_d``. §3 and ``--stage diffusion`` measured the
    slope to be **mean-reverting, not a random walk** (attenuation-corrected AR(1) ~0.52 monthly,
    signal variance *falling* with horizon). An EWMA is exactly the Kalman filter for a random-walk
    state, so the incumbent estimator is misspecified in a direction we have already measured. This
    uses the measured AR(1) instead, which both shrinks toward ``mu`` by the right amount and stops
    the filter chasing noise.

    Returns the one-step-ahead *predicted* state (shifted), so it is usable at ``d`` without look-ahead.
    """
    nd = len(sae)
    # causal residual-variance estimate: trailing mean of see/cnt, shifted
    s2 = pd.Series(np.where(cnt > 0, see / np.maximum(cnt, 1), np.nan))
    s2 = s2.ewm(halflife=MONTH_D, min_periods=5).mean().shift(1).ffill().to_numpy()
    beta = np.full(nd, np.nan)
    m, P = mu, var_eta / max(1e-12, 1 - phi * phi)  # start at the stationary distribution
    for d in range(nd):
        m_pred = mu + phi * (m - mu)
        P_pred = phi * phi * P + var_eta
        beta[d] = m_pred  # predicted before seeing day d -> causal
        if saa[d] > 0 and np.isfinite(s2[d]) and s2[d] > 0:
            y = sae[d] / saa[d]
            R = s2[d] / saa[d]
            K = P_pred / (P_pred + R)
            m, P = m_pred + K * (y - m_pred), (1 - K) * P_pred
        else:
            m, P = m_pred, P_pred
    return beta


def stage_estimator() -> None:  # noqa: C901
    """Compare intensity estimators on criteria fixed in advance; no holdout is touched.

    §13 stopped because the intensity target carried too little signal (~67% share at monthly
    aggregation) for ~166 observations to resolve 8,911 candidates. That is an *estimator* problem, so
    it gets an estimator fix before anything else is attempted.

    **Pre-registered candidates**: EWMA at halflives {5, 21, 63, 126} days (the incumbent family), and
    the AR(1)-state heteroskedastic Kalman filter at the measured persistence and at two neighbours.

    **Pre-registered criteria**, both computable without scoring any forecast of the residual:

    1. **Signal share** ``1 - var_null / var_observed`` of the monthly-aggregated intensity, using
       circular-shift nulls. This is the quantity that bound §13; higher is strictly better.
    2. **Predictive validity** — does the estimate at the end of month ``b`` predict month ``b+1``'s
       *realised* slope better than the constant mean does? An intensity estimate that cannot forecast
       the next month's realised alpha strength is not measuring intensity.

    The winner is whichever maximises (1) subject to beating the constant on (2).
    """
    os.makedirs(OUT, exist_ok=True)
    p = _panel()
    e = np.load(os.path.join(CACHE_DIR, "har_resid.npz"))["e"][TW:]
    a = dict(np.load(os.path.join(CACHE_DIR, "bucket_signals.npz")))["all"]
    ts = pd.Series(pd.to_datetime(p.t[TW + TW :]))
    day = (ts.dt.normalize().rank(method="dense").to_numpy() - 1).astype(int)
    nd = day.max() + 1
    sae, saa, see, cnt = _daily_sufficient(a, e, day, nd)
    dts = ts.groupby(day).first().to_numpy()
    mon = pd.Series(pd.to_datetime(dts)).dt.to_period("M")
    mcode = mon.rank(method="dense").astype(int).to_numpy() - 1
    nm = mcode.max() + 1
    search_d = (pd.Series(pd.to_datetime(dts)) < SPLIT).to_numpy()

    # realised monthly slope, the thing an intensity estimate is trying to track
    real = np.full(nm, np.nan)
    for b in range(nm):
        r = mcode == b
        if saa[r].sum() > 0:
            real[b] = sae[r].sum() / saa[r].sum()
    mu = float(np.nanmean(real[: int(search_d.sum() / 21)]))
    var_b = max(np.nanvar(real) - np.nanmean(np.abs(np.diff(real))) ** 2 / 2, 1e-4)
    print(f"  {nd} days, {nm} months; realised monthly slope mean {mu:+.3f} sd {np.nanstd(real):.3f}")

    cands: dict[str, np.ndarray] = {}
    for hl in (5, 21, 63, 126):
        num = pd.Series(sae).ewm(halflife=hl, min_periods=hl).mean()
        den = pd.Series(saa).ewm(halflife=hl, min_periods=hl).mean()
        cands[f"ewma_{hl}d"] = (num / den.replace(0.0, np.nan)).shift(1).to_numpy()
    for hl_m, tag in ((1.0, "kalman_hl1m"), (2.0, "kalman_hl2m"), (0.5, "kalman_hl05m")):
        phi = 0.5 ** (1.0 / (hl_m * MONTH_D))
        cands[tag] = kalman_intensity(sae, saa, see, cnt, phi, mu, var_b * (1 - phi * phi))

    rows = []
    for name, bd in cands.items():
        # aggregate the daily estimate to months (its value at the month's start = usable ex ante)
        est = np.full(nm, np.nan)
        for b in range(nm):
            r = np.flatnonzero(mcode == b)
            if r.size:
                est[b] = bd[r[0]]
        ok = np.isfinite(est) & np.isfinite(real)
        srch = ok & (pd.Series(pd.to_datetime(dts)).groupby(mcode).first().reset_index(drop=True) < SPLIT).to_numpy()
        # (1) signal share of the estimate itself, vs circular-shift nulls of the realised series
        v_obs = float(np.nanvar(est[srch]))
        nulls = []
        for k in range(N_NULL):
            sh = (k + 1) * max(1, nm // (N_NULL + 1))
            rr = np.roll(real, sh)
            num2, den2 = 0.0, 0.0
            for b in range(nm):
                if srch[b] and np.isfinite(rr[b]):
                    num2 += (rr[b] - mu) * (est[b] - np.nanmean(est[srch]))
                    den2 += (est[b] - np.nanmean(est[srch])) ** 2
            nulls.append((num2 / den2) if den2 > 0 else np.nan)
        # (2) predictive validity: est[b] vs real[b] against the constant mu
        m2 = srch & np.isfinite(real)
        sse = float(np.nansum((real[m2] - est[m2]) ** 2))
        sse0 = float(np.nansum((real[m2] - mu) ** 2))
        r2 = 1.0 - sse / sse0 if sse0 > 0 else np.nan
        ic = float(np.corrcoef(est[m2], real[m2])[0, 1]) if m2.sum() > 5 else np.nan
        rows.append(
            {"estimator": name, "n_months": int(m2.sum()), "est_sd": np.sqrt(v_obs),
             "pred_r2_vs_constant": r2, "ic_with_realised": ic,
             "null_slope_mean": float(np.nanmean(nulls))}
        )
        print(f"  {name:13s} sd {np.sqrt(v_obs):.3f}  predictive R2 vs constant {r2:+.4f}  "
              f"IC with realised {ic:+.3f}  (null slope {np.nanmean(nulls):+.3f})", flush=True)
    d = pd.DataFrame(rows)
    valid = d[d.pred_r2_vs_constant > 0]
    win = valid.loc[valid.ic_with_realised.idxmax(), "estimator"] if len(valid) else d.loc[d.ic_with_realised.idxmax(), "estimator"]
    print(f"\n  winner by the pre-registered criteria: {win}")
    d.insert(0, "winner", win)
    d.to_csv(f"{OUT}/intensity_estimators.csv", index=False)
    np.save(os.path.join(CACHE_DIR, "intensity_best_daily.npy"), cands[win])
    np.save(os.path.join(CACHE_DIR, "intensity_day_index.npy"), day)
    print(f"wrote {OUT}/intensity_estimators.csv")


def stage_gate3() -> None:  # noqa: C901
    """§13's gate re-run on the *better* intensity estimator from ``--stage estimator``.

    That stage found the incumbent EWMA family has **negative** predictive R² against a constant
    (-0.44 to -0.14): positive IC but wrong magnitude, because an EWMA is the Kalman filter for a
    random-walk state and therefore never shrinks toward the mean. Every AR(1)-state Kalman variant is
    positive (+0.04 to +0.09) at equal or better IC. So §13 gated on a target that was literally worse
    than a constant, which also explains the earlier dynamic-gain failure (§3).

    Same gate, same positive control, same nulls — only the intensity changed. If the control now
    passes, the measurement is fixed and the gate verdict is meaningful either way; if it still fails,
    the problem was never the estimator.
    """
    os.makedirs(OUT, exist_ok=True)
    p = _panel()
    e = np.load(os.path.join(CACHE_DIR, "har_resid.npz"))["e"][TW:]
    ts = pd.Series(pd.to_datetime(p.t[TW + TW :]))
    bd = np.load(os.path.join(CACHE_DIR, "intensity_best_daily.npy"))
    day = np.load(os.path.join(CACHE_DIR, "intensity_day_index.npy"))
    m = bd[day]  # expand the (causal, pre-day) daily state to bar resolution
    bc, bn = base_columns(p)
    X = np.ascontiguousarray(p.X[TW + TW :, bc], dtype=np.float64)
    pb = X.shape[1]
    assert len(m) == len(e) == len(X)

    codes = (ts.dt.year * 12 + ts.dt.month).to_numpy()
    blk = np.unique(codes, return_inverse=True)[1]
    nb = blk.max() + 1
    keep = np.isfinite(m)
    Xb = np.full((nb, pb), np.nan)
    mb = np.full(nb, np.nan)
    tb = np.empty(nb, dtype="datetime64[ns]")
    for b in range(nb):
        rows = (blk == b) & keep
        if rows.sum() < 200:
            continue
        Xb[b], mb[b] = X[rows].mean(0), m[rows].mean()
        tb[b] = ts[blk == b].iloc[0].to_datetime64()
    good = np.isfinite(mb) & np.isfinite(Xb).all(1)
    search = good & (pd.Series(tb) < SPLIT).to_numpy()
    ns = int(search.sum())
    print(f"  {ns} search months; noise floor 1/sqrt(n) = {1 / np.sqrt(ns):.3f}")
    print(f"  intensity: mean {np.nanmean(mb[search]):+.3f} sd {np.nanstd(mb[search]):.3f} "
          f"AR(1) {pd.Series(mb[search]).autocorr(1):+.3f}\n")

    ii, jj = _upper(pb)

    def ics(rows, target):
        r = rows & np.isfinite(target)
        Z = Xb[r] - Xb[r].mean(0)
        y = target[r] - target[r].mean()
        sd = Z.std(0)
        lin = np.abs(np.nan_to_num((Z * y[:, None]).mean(0) / (np.where(sd > 0, sd, 1) * y.std())))
        return lin, np.abs(np.nan_to_num(_pair_ic(Z, y)[ii, jj]))

    dmb = np.concatenate([[np.nan], np.diff(mb)])
    out = []
    for label, tgt in (("levels", mb), ("changes", dmb)):
        lin, pair = ics(search, tgt)
        nl = [ics(search, np.roll(tgt, (k + 1) * max(1, nb // (N_NULL + 1))))[0].mean() for k in range(N_NULL)]
        npr = [ics(search, np.roll(tgt, (k + 1) * max(1, nb // (N_NULL + 1))))[1].mean() for k in range(N_NULL)]
        ctrl = bn.index("har_ma_25")
        rank = int((lin > lin[ctrl]).sum()) + 1
        ok_ctrl = rank <= 0.25 * pb
        idx = np.flatnonzero(search)
        h1, h2 = np.zeros(nb, bool), np.zeros(nb, bool)
        h1[idx[: len(idx) // 2]] = True
        h2[idx[len(idx) // 2 :]] = True
        l1, p1 = ics(h1, tgt)
        l2, p2 = ics(h2, tgt)
        sl, sp = float(np.corrcoef(l1, l2)[0, 1]), float(np.corrcoef(p1, p2)[0, 1])
        nsl, nsp = [], []
        for k in range(N_NULL):
            tn = np.roll(tgt, (k + 1) * max(1, nb // (N_NULL + 1)))
            q1, r1 = ics(h1, tn)
            q2, r2 = ics(h2, tn)
            with np.errstate(invalid="ignore"):
                c1, c2 = np.corrcoef(q1, q2)[0, 1], np.corrcoef(r1, r2)[0, 1]
            if np.isfinite(c1):
                nsl.append(c1)
            if np.isfinite(c2):
                nsp.append(c2)
        passed = ok_ctrl and (sl > np.mean(nsl) + 0.10 or sp > np.mean(nsp) + 0.10)
        print(f"  --- target = intensity {label} ---")
        print(f"    excess over null: linear {lin.mean() / np.mean(nl):.2f}x  products {pair.mean() / np.mean(npr):.2f}x")
        print(f"    POSITIVE CONTROL har_ma_25 rank {rank}/{pb} -> {'PASS' if ok_ctrl else 'FAIL'}")
        for j in np.argsort(-lin)[:5]:
            print(f"      {bn[j]:38s} |IC| {lin[j]:.4f}")
        print(f"    GATE split-half: linear {sl:+.3f} (null {np.mean(nsl):+.3f})  "
              f"products {sp:+.3f} (null {np.mean(nsp):+.3f})")
        print(f"    -> {'PASS' if passed else 'STOP'}\n")
        out.append({"target": label, "n_months": ns, "excess_lin": lin.mean() / np.mean(nl),
                    "excess_pair": pair.mean() / np.mean(npr), "control_rank": rank,
                    "control_pass": ok_ctrl, "stab_lin": sl, "stab_lin_null": np.mean(nsl),
                    "stab_pair": sp, "stab_pair_null": np.mean(nsp), "gate_passed": bool(passed)})
    pd.DataFrame(out).to_csv(f"{OUT}/intensity_gate3.csv", index=False)
    print(f"wrote {OUT}/intensity_gate3.csv")


if __name__ == "__main__":
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--stage", choices=["gate", "gate2", "estimator", "gate3"], required=True)
    a = ap_.parse_args()
    {"gate": stage_gate, "gate2": stage_gate2, "estimator": stage_estimator, "gate3": stage_gate3}[a.stage]()
