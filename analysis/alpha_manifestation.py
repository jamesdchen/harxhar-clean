"""How does the dense-but-weak exog alpha manifest? (time-sensitivity / rotation / granularity)

The campaign evidence says the exogenous alpha is **dense and weak**: all 8 buckets beat
HAR, 41 exog stacked beats every single bucket, and no single feature carries much. This
study tests the competing reading — that "dense and weak" is an artifact of *pooled*
estimation, and that locally (in time, or conditional on a state) a few exog variables are
strongly active while the identity of the active set rotates.

Three claims, each with an explicit null, all scored out-of-sample:

**A. Density (calibration).** Pooled marginal ICs of the 246 exog value features against
the walk-forward HAR residual: mean/median/max |IC|, participation ratio, and the
all-exog walk-forward composite's OOS corr / R^2. This is the thing to be explained.

**B. Time-sensitivity + rotation.** Per-month realized alpha (concentration: Lorenz /
Gini / top-decile share; persistence: AR(1)), per-month per-feature IC matrices
(participation ratio *within* a month vs pooled), and block-to-block rank persistence of
the IC vector. The null everywhere is a **circular shift** of the residual, which
preserves the autocorrelation of both series and destroys only the true alignment — a
1/sqrt(n) noise floor would badly understate the sampling variation of 30-min bars.

**C. Is coarse testing hiding it?** A granularity ladder. Seven per-bucket walk-forward
alpha signals are combined by a second-stage walk-forward regression that is either
pooled (K=1) or fit *separately within K bins* of a conditioning axis (time-of-day,
day-of-week, causal vol-regime quintile). Splitting into K bins multiplies the local
signal by some concentration factor c but cuts each fit's sample to n/K, so the t-stat
moves as c/sqrt(K): finer testing only wins if alpha concentrates faster than sqrt(K).
The ladder measures c, the OOS gain, and where the optimum sits.

Every fit is strictly causal: rolling-window ridge via the production
:class:`~src.models.rolling_least_squares.RollingLeastSquares`, refit as it walks forward.

Usage
-----
    python analysis/alpha_manifestation.py --stage resid     # HAR baseline residual
    python analysis/alpha_manifestation.py --stage signals   # 7 bucket + all-exog signals
    python analysis/alpha_manifestation.py --stage tests     # the battery + report
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.alpha_panel import CACHE_DIR, load_panel  # noqa: E402
from analysis.wf import nw_tstat, r2_oos, walk_forward  # noqa: E402
from src.features.transforms.target import PERIODS_PER_DAY  # noqa: E402

TRAIN_WINDOW_DAYS = 250  # the Part-4 optimum
TW = TRAIN_WINDOW_DAYS * PERIODS_PER_DAY
COMBINE_WINDOW_DAYS = 60  # second stage has p<=7*K, so it needs far less history
RIDGE_ALPHA = (
    3000.0  # penalty ladder on the all-exog fit: R2 -0.084 / -0.001 / +0.025 / +0.035
)
#                       at alpha = 1 / 30 / 300 / 3000 (OOS corr flat at ~0.19 from 300 up)
REFIT_EVERY = 48  # daily refit; the diagnostic does not need per-bar
N_SHIFT = 24  # circular-shift null replications
BUCKETS = [
    "moments",
    "liquidity",
    "implied_vol",
    "market_vw",
    "market_ew",
    "vol_demand",
    "sentiment",
]
OUT = "results/alpha_manifestation"


def _p(name: str) -> str:
    return os.path.join(CACHE_DIR, name)


# ---------------------------------------------------------------------------
# Stage 1 — the HAR walk-forward residual (what exog alpha must explain)
# ---------------------------------------------------------------------------


def stage_resid() -> None:
    p = load_panel()
    har = np.concatenate([p.cols("har"), p.cols("calendar"), p.cols("regime")])
    pred = walk_forward(
        p.X[:, har].astype(np.float64), p.y, TW, alpha=1.0, refit_every=1
    )
    e = p.y[TW:] - pred
    np.savez_compressed(_p("har_resid.npz"), e=e, pred=pred, tw=TW)
    print(
        f"HAR baseline: {len(har)} cols, R2(sqrt-space) "
        f"{r2_oos(p.y[TW:] - p.y[TW:].mean(), pred - p.y[TW:].mean()):.4f}, "
        f"resid std {e.std():.4f}, n_oos {len(e)}"
    )


# ---------------------------------------------------------------------------
# Stage 2 — per-bucket walk-forward alpha signals
# ---------------------------------------------------------------------------


def stage_signals() -> None:
    p = load_panel()
    e = np.load(_p("har_resid.npz"))["e"]
    sig: dict[str, np.ndarray] = {}
    for b in BUCKETS + ["all"]:
        if b == "all":
            cols = np.concatenate([p.cols("value"), p.cols("indicator")])
        else:
            cols = np.concatenate([p.cols("value", b), p.cols("indicator", b)])
        s = walk_forward(
            p.X[TW:, cols].astype(np.float64),
            e,
            TW,
            alpha=RIDGE_ALPHA,
            refit_every=REFIT_EVERY,
        )
        sig[b] = s
        print(
            f"{b:12s} p={len(cols):4d}  OOS corr {np.corrcoef(s, e[TW:])[0, 1]:+.4f}  "
            f"R2 {r2_oos(e[TW:], s):+.5f}",
            flush=True,
        )
    np.savez_compressed(_p("bucket_signals.npz"), **sig)


# ---------------------------------------------------------------------------
# helpers for the tests
# ---------------------------------------------------------------------------


def _blocks(ts: pd.Series) -> np.ndarray:
    """Month index per row (0-based, contiguous)."""
    codes = (ts.dt.year * 12 + ts.dt.month).to_numpy()
    return np.unique(codes, return_inverse=True)[1]


def _block_ic(X: np.ndarray, y: np.ndarray, blk: np.ndarray, nb: int) -> np.ndarray:
    """(nb, p) within-block correlation of each column of X with y."""
    out = np.full((nb, X.shape[1]), np.nan)
    for b in range(nb):
        m = blk == b
        if m.sum() < 50:
            continue
        xb = X[m] - X[m].mean(0)
        yb = y[m] - y[m].mean()
        sx = np.sqrt((xb**2).sum(0))
        sy = np.sqrt((yb**2).sum())
        with np.errstate(invalid="ignore", divide="ignore"):
            out[b] = (xb * yb[:, None]).sum(0) / (sx * sy)
    return out


def _pr(v: np.ndarray) -> float:
    """Participation ratio of |v|: (sum|v|)^2 / sum v^2. p if flat, 1 if a spike."""
    v = v[np.isfinite(v)]
    return float(np.abs(v).sum() ** 2 / (v**2).sum()) if v.size else np.nan


def _gini(w: np.ndarray) -> float:
    w = np.sort(np.asarray(w, dtype=np.float64))
    n = len(w)
    if n == 0 or w.sum() == 0:
        return np.nan
    return float((2 * np.arange(1, n + 1) - n - 1) @ w / (n * w.sum()))


def _shifts(n: int) -> list[int]:
    step = max(1, n // (N_SHIFT + 1))
    return [step * (k + 1) for k in range(N_SHIFT)]


def _bin_walk_forward(
    S: np.ndarray, y: np.ndarray, bins: np.ndarray, K: int, win_days: int
) -> np.ndarray:
    """Fit the combiner separately within each bin; return pooled-aligned OOS preds.

    Each bin's rolling window holds the same *calendar* span (``win_days``), so a bin
    covering 1/K of the bars trains on 1/K as many rows — the power cost of slicing is
    paid, not hidden. Rows a bin cannot yet predict (its own warm-up) are NaN.
    """
    out = np.full(len(y), np.nan)
    for k in range(K):
        idx = np.flatnonzero(bins == k)
        if idx.size == 0:
            continue
        w = max(30, int(round(win_days * PERIODS_PER_DAY * idx.size / len(y))))
        if idx.size <= w + 30:
            continue
        pk = walk_forward(
            S[idx], y[idx], w, alpha=1.0, refit_every=max(1, REFIT_EVERY // K)
        )
        out[idx[w:]] = pk
    return out


def _score(y: np.ndarray, pred: np.ndarray) -> tuple[float, float, int]:
    m = np.isfinite(pred)
    return (
        float(np.corrcoef(pred[m], y[m])[0, 1]),
        r2_oos(y[m], pred[m]),
        int(m.sum()),
    )


# ---------------------------------------------------------------------------
# Stage 3 — the test battery
# ---------------------------------------------------------------------------


def stage_tests() -> None:  # noqa: C901
    os.makedirs(OUT, exist_ok=True)
    p = load_panel()
    e_all = np.load(_p("har_resid.npz"))["e"]
    sig = dict(np.load(_p("bucket_signals.npz")))
    e = e_all[TW:]
    ts = pd.Series(pd.to_datetime(p.t[TW + TW :]))
    n = len(e)
    assert len(ts) == n
    val = p.cols("value")
    Xv = p.X[TW + TW :, val].astype(np.float64)
    S = np.column_stack([sig[b] for b in BUCKETS])
    blk = _blocks(ts)
    nb = blk.max() + 1
    rep: list[str] = []

    def say(s: str = "") -> None:
        print(s, flush=True)
        rep.append(s)

    say("=" * 78)
    say("A. DENSITY / WEAKNESS  (pooled, the thing to be explained)")
    say("=" * 78)
    ic_pool = np.array([np.corrcoef(Xv[:, j], e)[0, 1] for j in range(Xv.shape[1])])
    say(
        f"  246 exog value features vs HAR residual: mean|IC| {np.abs(ic_pool).mean():.4f}  "
        f"median {np.median(np.abs(ic_pool)):.4f}  max {np.abs(ic_pool).max():.4f}"
    )
    say(
        f"  participation ratio of the pooled |IC| vector: {_pr(ic_pool):.1f} of {len(ic_pool)}"
    )
    say(
        f"  |IC| > 0.02: {(np.abs(ic_pool) > 0.02).sum()} features   > 0.05: {(np.abs(ic_pool) > 0.05).sum()}"
    )
    for b in BUCKETS + ["all"]:
        s = sig[b][TW:] if len(sig[b]) != n else sig[b]
        c, r2, _ = _score(e, s)
        _, t = nw_tstat(s, e)
        say(f"  signal {b:12s} OOS corr {c:+.4f}  R2 {r2:+.5f}  NW-t {t:+7.2f}")
    pd.DataFrame(
        {
            "feature": [p.names[j] for j in val],
            "bucket": [p.bucket[j] for j in val],
            "window": [p.window[j] for j in val],
            "ic_pooled": ic_pool,
        }
    ).to_csv(f"{OUT}/pooled_feature_ic.csv", index=False)

    say()
    say("=" * 78)
    say("B1. TIME-SENSITIVITY  (is the alpha episodic? monthly blocks)")
    say("=" * 78)
    a = sig["all"]
    contrib = np.array([np.sum(a[blk == b] * e[blk == b]) for b in range(nb)])
    tot = contrib.sum()
    order = np.argsort(-contrib)
    csum = np.cumsum(contrib[order]) / tot
    dec = max(1, nb // 10)
    ic_m = _block_ic(a[:, None], e, blk, nb)[:, 0]
    ar1 = float(pd.Series(ic_m).autocorr(1))
    say(f"  {nb} monthly blocks, {ts.iloc[0].date()} .. {ts.iloc[-1].date()}")
    say(f"  share of total alpha from the top decile of months : {csum[dec - 1]:.3f}")
    say(
        f"  share from the top quartile                        : {csum[max(1, nb // 4) - 1]:.3f}"
    )
    say(
        f"  months contributing NEGATIVELY                     : {(contrib < 0).sum()} / {nb}"
    )
    say(
        f"  Gini of |monthly contribution|                     : {_gini(np.abs(contrib)):.3f}"
    )
    say(
        f"  monthly alpha-IC: mean {np.nanmean(ic_m):+.4f}  sd {np.nanstd(ic_m):.4f}  AR(1) {ar1:+.3f}"
    )
    nulls: dict[str, list[float]] = {
        "top_decile": [],
        "neg": [],
        "gini": [],
        "sd": [],
        "ar1": [],
    }
    for sh in _shifts(n):
        en = np.roll(e, sh)
        cn = np.array([np.sum(a[blk == b] * en[blk == b]) for b in range(nb)])
        cs = np.cumsum(np.sort(cn)[::-1]) / max(cn.sum(), 1e-12)
        icn = _block_ic(a[:, None], en, blk, nb)[:, 0]
        nulls["top_decile"].append(cs[dec - 1] if cn.sum() > 0 else np.nan)
        nulls["neg"].append(float((cn < 0).sum()))
        nulls["gini"].append(_gini(np.abs(cn)))
        nulls["sd"].append(np.nanstd(icn))
        nulls["ar1"].append(float(pd.Series(icn).autocorr(1)))
    say(
        f"  circular-shift null ({N_SHIFT} reps): neg-months {np.mean(nulls['neg']):.0f}, "
        f"Gini {np.nanmean(nulls['gini']):.3f}, IC sd {np.mean(nulls['sd']):.4f}, "
        f"AR(1) {np.nanmean(nulls['ar1']):+.3f}"
    )
    say(
        f"  => excess monthly IC dispersion vs null: "
        f"{np.nanstd(ic_m) / np.mean(nulls['sd']):.2f}x"
    )
    pd.DataFrame(
        {
            "month": [str(ts[blk == b].iloc[0])[:7] for b in range(nb)],
            "contribution": contrib,
            "ic": ic_m,
        }
    ).to_csv(f"{OUT}/monthly_alpha.csv", index=False)

    say()
    say("=" * 78)
    say("B2. ROTATION  (do different exog carry it at different times?)")
    say("=" * 78)
    M = _block_ic(Xv, e, blk, nb)  # (nb, 246)
    pr_month = np.nanmean([_pr(M[b]) for b in range(nb)])
    pr_null, sp_null = [], []
    for sh in _shifts(n):
        Mn = _block_ic(Xv, np.roll(e, sh), blk, nb)
        pr_null.append(np.nanmean([_pr(Mn[b]) for b in range(nb)]))
        Dn = Mn - np.nanmean(Mn, axis=0)
        sp_null.append(
            np.nanmean(
                [
                    pd.Series(Dn[b]).corr(pd.Series(Dn[b + 1]), method="spearman")
                    for b in range(nb - 1)
                ]
            )
        )
    D = M - np.nanmean(M, axis=0)  # strip the static (pooled) component
    sp = np.nanmean(
        [
            pd.Series(D[b]).corr(pd.Series(D[b + 1]), method="spearman")
            for b in range(nb - 1)
        ]
    )
    say(f"  participation ratio of |IC| WITHIN a month : {pr_month:.1f} of 246")
    say(f"                                  pooled     : {_pr(ic_pool):.1f}")
    say(f"                                  null       : {np.mean(pr_null):.1f}")
    say(
        f"  month-to-month Spearman of the TIME-VARYING IC part : {sp:+.4f}  (null {np.mean(sp_null):+.4f})"
    )
    Mb = _block_ic(S, e, blk, nb)  # (nb, 7) bucket-level
    Db = Mb - np.nanmean(Mb, axis=0)
    spb = np.nanmean(
        [
            pd.Series(Db[b]).corr(pd.Series(Db[b + 1]), method="spearman")
            for b in range(nb - 1)
        ]
    )
    best = np.nanargmax(Mb, axis=1)
    stay = float(np.mean(best[1:] == best[:-1]))
    say(f"  bucket-level: month-to-month Spearman of time-varying part {spb:+.4f}")
    say(
        f"  best-bucket-of-the-month persistence: {stay:.3f} (chance {1 / len(BUCKETS):.3f})"
    )
    say(
        "  best-bucket shares: "
        + ", ".join(
            f"{BUCKETS[k]} {np.mean(best == k):.2f}" for k in range(len(BUCKETS))
        )
    )
    pd.DataFrame(Mb, columns=BUCKETS).assign(
        month=[str(ts[blk == b].iloc[0])[:7] for b in range(nb)]
    ).to_csv(f"{OUT}/monthly_bucket_ic.csv", index=False)

    say()
    say("=" * 78)
    say("B3. CONDITIONAL ACTIVATION MAPS  (split-half stability is the real test)")
    say("=" * 78)
    # clock slot, rotated so bin 0 starts at the 09:30 open (slot 19) — the coarsening
    # ladder then splits the trading day, not the calendar day.
    slot = ((ts.dt.hour * 2 + (ts.dt.minute >= 30)).to_numpy() - 19) % PERIODS_PER_DAY
    dow = ts.dt.dayofweek.to_numpy()
    vol = _vol_regime(p, ts, 5)
    half = (np.arange(n) >= n // 2).astype(int)
    axes = {
        "time-of-day (48 slots)": slot,
        "day-of-week (5)": dow,
        "vol regime (5, causal)": vol,
    }
    rows = []
    for label, bins in axes.items():
        K = int(bins.max()) + 1
        A = _bin_map(S, e, bins, K)  # (K, 7)
        h0 = _bin_map(S[half == 0], e[half == 0], bins[half == 0], K)
        h1 = _bin_map(S[half == 1], e[half == 1], bins[half == 1], K)
        ok = np.isfinite(h0) & np.isfinite(h1)
        sh_stab = float(np.corrcoef(h0[ok], h1[ok])[0, 1])
        nullst = []
        for s in _shifts(n)[:8]:
            en = np.roll(e, s)
            g0 = _bin_map(S[half == 0], en[half == 0], bins[half == 0], K)
            g1 = _bin_map(S[half == 1], en[half == 1], bins[half == 1], K)
            o = np.isfinite(g0) & np.isfinite(g1)
            nullst.append(float(np.corrcoef(g0[o], g1[o])[0, 1]))
        disp = float(np.nanstd(A))
        say(
            f"  {label:24s} K={K:2d}  |IC| spread {disp:.4f}  "
            f"split-half stability {sh_stab:+.3f}  (null {np.mean(nullst):+.3f})"
        )
        rows.append(
            {
                "axis": label,
                "K": K,
                "ic_sd": disp,
                "split_half": sh_stab,
                "null": np.mean(nullst),
            }
        )
        pd.DataFrame(A, columns=BUCKETS).to_csv(
            f"{OUT}/activation_{label.split()[0].replace('-', '')}.csv", index=False
        )
    pd.DataFrame(rows).to_csv(f"{OUT}/activation_stability.csv", index=False)

    say()
    say("=" * 78)
    say("C. GRANULARITY LADDER  (does finer conditioning WIN out-of-sample?)")
    say("=" * 78)
    say(
        "  combiner: e ~ 7 bucket signals, rolling 60-day, fit separately within K bins"
    )
    pooled = walk_forward(
        S, e, COMBINE_WINDOW_DAYS * PERIODS_PER_DAY, alpha=1.0, refit_every=REFIT_EVERY
    )
    base_pred = np.full(n, np.nan)
    base_pred[COMBINE_WINDOW_DAYS * PERIODS_PER_DAY :] = pooled
    c0, r0, n0 = _score(e, base_pred)
    say(f"  K=1 pooled: corr {c0:+.4f}  R2 {r0:+.5f}  n {n0}")
    ladder = []
    for label, bins_full, Ks in [
        ("time-of-day", slot, [2, 4, 6, 8, 12, 16, 24, 48]),
        ("day-of-week", dow, [5]),
        ("vol regime", None, [2, 3, 5, 10]),
    ]:
        for K in Ks:
            bins = (
                _coarsen(bins_full, K)
                if bins_full is not None
                else _vol_regime(p, ts, K)
            )
            pred = _bin_walk_forward(S, e, bins, K, COMBINE_WINDOW_DAYS)
            m = np.isfinite(pred) & np.isfinite(base_pred)
            c, r2, _ = _score(e[m], pred[m])
            cb, rb, _ = _score(e[m], base_pred[m])
            conc = _concentration(S, e, bins, K)
            cbl, rbl, _ = _score(e[m], 0.5 * (pred[m] + base_pred[m]))
            say(
                f"  {label:12s} K={K:2d}  corr {c:+.4f} (pooled {cb:+.4f})  "
                f"R2 {r2:+.5f} (pooled {rb:+.5f})  dR2 {r2 - rb:+.5f}  "
                f"blend dR2 {rbl - rb:+.5f}  c={conc:.2f}  c/sqrt(K)={conc / np.sqrt(K):.2f}"
            )
            ladder.append(
                {
                    "axis": label,
                    "K": K,
                    "corr": c,
                    "corr_pooled": cb,
                    "r2": r2,
                    "r2_pooled": rb,
                    "dr2": r2 - rb,
                    "corr_blend": cbl,
                    "dr2_blend": rbl - rb,
                    "concentration": conc,
                    "power_ratio": conc / np.sqrt(K),
                    "n": int(m.sum()),
                }
            )
    pd.DataFrame(ladder).to_csv(f"{OUT}/granularity_ladder.csv", index=False)

    with open(f"{OUT}/report.txt", "w") as fh:
        fh.write("\n".join(rep) + "\n")
    print(f"\nwrote {OUT}/report.txt + csvs")


# ---------------------------------------------------------------------------
# Stage 4 — dynamics: the two cheap parameterizations the ladder implies
# ---------------------------------------------------------------------------


def stage_dynamics() -> None:
    """Separate the *gain* channel from the *weights* channel.

    Stage C re-estimates all 7 bucket weights inside every bin, spending 7 df per bin.
    But the stable part of a conditional activation map may be a single per-bin **gain**
    (how much to trust the same composite there), which costs 1 df per bin. And B1's
    persistent monthly alpha intensity (AR(1) +0.375) implies a *dynamic* gain driven by
    trailing realized alpha. Both are tested here against the same pooled baseline.
    """
    os.makedirs(OUT, exist_ok=True)
    p = load_panel()
    e = np.load(_p("har_resid.npz"))["e"][TW:]
    sig = dict(np.load(_p("bucket_signals.npz")))
    ts = pd.Series(pd.to_datetime(p.t[TW + TW :]))
    n = len(e)
    S = np.column_stack([sig[b] for b in BUCKETS])
    cw = COMBINE_WINDOW_DAYS * PERIODS_PER_DAY

    # pooled combiner -> one composite regressor (causal at every row)
    a = np.full(n, np.nan)
    a[cw:] = walk_forward(S, e, cw, alpha=1.0, refit_every=REFIT_EVERY)
    base = np.full(n, np.nan)
    base[2 * cw :] = walk_forward(
        a[cw:, None], e[cw:], cw, alpha=1.0, refit_every=REFIT_EVERY
    )
    m0 = np.isfinite(base)
    c0, r0, _ = _score(e[m0], base[m0])
    rows = [{"model": "pooled composite", "K": 1, "corr": c0, "r2": r0, "dr2": 0.0}]
    print(f"pooled composite: corr {c0:+.4f}  R2 {r0:+.5f}  n {m0.sum()}", flush=True)

    slot = ((ts.dt.hour * 2 + (ts.dt.minute >= 30)).to_numpy() - 19) % PERIODS_PER_DAY
    for label, bins_full, Ks in [
        ("gain x time-of-day", slot, [2, 4, 8, 16, 48]),
        ("gain x vol regime", None, [2, 3, 5, 10]),
    ]:
        for K in Ks:
            bins = (
                _coarsen(bins_full, K)
                if bins_full is not None
                else _vol_regime(p, ts, K)
            )
            Z = np.zeros((n, K))
            Z[np.arange(n), bins] = (
                a  # per-bin gain on the SAME composite: K df, not 7K
            )
            pred = np.full(n, np.nan)
            ok = np.isfinite(a)
            pred[np.flatnonzero(ok)[cw:]] = walk_forward(
                Z[ok], e[ok], cw, alpha=1.0, refit_every=REFIT_EVERY
            )
            m = np.isfinite(pred) & m0
            c, r2, _ = _score(e[m], pred[m])
            cb, rb, _ = _score(e[m], base[m])
            print(
                f"  {label:20s} K={K:2d}  corr {c:+.4f}  R2 {r2:+.5f}  dR2 {r2 - rb:+.5f}",
                flush=True,
            )
            rows.append({"model": label, "K": K, "corr": c, "r2": r2, "dr2": r2 - rb})

    # dynamic gain: scale the composite by trailing realized alpha intensity
    r_real = np.where(np.isfinite(a), a * e, 0.0)
    for hl_days in (5, 20, 60, 250):
        hl = hl_days * PERIODS_PER_DAY
        inten = (
            pd.Series(r_real)
            .ewm(halflife=hl, min_periods=hl)
            .mean()
            .shift(1)
            .to_numpy()
        )
        z = (inten - pd.Series(inten).expanding(hl).mean().to_numpy()) / (
            pd.Series(inten).expanding(hl).std().to_numpy() + 1e-12
        )
        z = np.clip(np.nan_to_num(z), -3, 3)
        Z = np.column_stack([a, a * z])
        ok = np.isfinite(a) & np.isfinite(z)
        pred = np.full(n, np.nan)
        pred[np.flatnonzero(ok)[cw:]] = walk_forward(
            Z[ok], e[ok], cw, alpha=1.0, refit_every=REFIT_EVERY
        )
        m = np.isfinite(pred) & m0
        c, r2, _ = _score(e[m], pred[m])
        cb, rb, _ = _score(e[m], base[m])
        print(
            f"  dynamic gain hl={hl_days:3d}d      corr {c:+.4f}  R2 {r2:+.5f}  dR2 {r2 - rb:+.5f}",
            flush=True,
        )
        rows.append(
            {
                "model": f"dynamic gain hl={hl_days}d",
                "K": 2,
                "corr": c,
                "r2": r2,
                "dr2": r2 - rb,
            }
        )
    pd.DataFrame(rows).to_csv(f"{OUT}/gain_channel.csv", index=False)
    print(f"wrote {OUT}/gain_channel.csv")


# ---------------------------------------------------------------------------
# Stage 5 — the hypothesis in operational form: sparse-rotating vs dense-static
# ---------------------------------------------------------------------------


def dm_test(
    e: np.ndarray, p1: np.ndarray, p2: np.ndarray, lags: int = 480, tol: float = 1e-10
) -> float:
    """Diebold-Mariano HAC t on the squared-error difference (``p1`` vs ``p2``).

    Positive t = ``p1`` has lower squared error. 480 Bartlett lags = 10 trading days,
    long enough for the overlapping-window dependence the walk-forward induces.

    ``tol`` guards a real trap: DM is a *ratio* of a mean difference to its standard error, so
    two numerically-identical forecasts give 0/0 and the statistic explodes on pure float noise.
    Measured here: the same walk-forward re-run in a different process differs by 9.4e-16 (3.8e-15
    of a residual sd) through non-deterministic BLAS reductions, and DM reports **t = +5.00** on
    that. So when the two forecast series agree to within ``tol`` of the target's scale they are
    the same model and 0.0 is returned. A DM-t should never be read without the accompanying
    delta-R2 — a large t on a negligible delta-R2 means nothing.
    """
    scale = float(np.std(e)) or 1.0
    if float(np.sqrt(np.mean((p1 - p2) ** 2))) / scale < tol:
        return 0.0
    d = (e - p2) ** 2 - (e - p1) ** 2
    dc = d - d.mean()
    s = float(dc @ dc)
    for L in range(1, lags + 1):
        s += 2.0 * (1.0 - L / (lags + 1.0)) * float(dc[L:] @ dc[:-L])
    se = np.sqrt(max(s, 0.0)) / len(d)
    return float(d.mean() / se) if se > 0 else 0.0


def stage_sparse() -> None:
    """Does a *locally sparse, rotating* model beat the dense pooled one?

    The hypothesis "few exog are active at any moment, and which ones rotates" has a
    direct operational form: rank the 246 value features by their trailing-250-day |IC|
    and fit only the top ``k``, reselecting and refitting **every day**. If the alpha is
    locally sparse and the active set is identifiable ex ante, small ``k`` wins OOS. If
    dense wins, the density is the truth, not an artifact of pooled estimation.

    Both arms share one machinery, so ``k`` is the only difference: the sliding window's
    sufficient statistics ``(XᵀX, Xᵀy, Σx, Σy, Σy²)`` roll rank-1 across all 246 columns
    (as in :class:`RollingLeastSquares`), the trailing ICs are read off *those same*
    statistics, and each refit solves only the selected ``k×k`` subsystem. ``k=246`` is
    therefore exactly the dense walk-forward, not a separate estimator.

    Alignment (the easy thing to get wrong): ``e[i]`` is the HAR residual of panel row
    ``TW + i``, so the second-stage design is ``p.X[TW:]`` against ``e`` with no further
    offset, and the walk-forward emits from row ``TW`` of that pair.
    """
    os.makedirs(OUT, exist_ok=True)
    p = load_panel()
    e = np.load(_p("har_resid.npz"))["e"]
    val = p.cols("value")
    X = np.ascontiguousarray(p.X[TW:, val], dtype=np.float64)
    n = len(e)
    assert len(X) == n, (len(X), n)
    pv = X.shape[1]
    ks = (5, 10, 25, 50, 100, pv)
    preds = {k: np.empty(n - TW) for k in ks}
    churn: dict[int, list[float]] = {k: [] for k in ks}

    Sxx = X[:TW].T @ X[:TW]
    Sxy = X[:TW].T @ e[:TW]
    sx = X[:TW].sum(0)
    sy = float(e[:TW].sum())
    syy = float(e[:TW] @ e[:TW])
    prev: dict[int, np.ndarray] = {}
    coef: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}
    for i in range(n - TW):
        t = TW + i
        if i % REFIT_EVERY == 0:
            g = Sxx - np.outer(sx, sx) / TW
            r = Sxy - sx * (sy / TW)
            var_x = np.maximum(np.diag(g), 1e-12)
            var_y = max(syy - sy * sy / TW, 1e-12)
            ic = r / np.sqrt(var_x * var_y)
            rank = np.argsort(-np.abs(np.nan_to_num(ic)))
            for k in ks:
                sel = np.sort(rank[:k])
                b = np.linalg.solve(
                    g[np.ix_(sel, sel)] + RIDGE_ALPHA * np.eye(k), r[sel]
                )
                c0 = sy / TW - float(sx[sel] @ b) / TW
                coef[k] = (sel, b, c0)
                if k in prev:
                    churn[k].append(1.0 - len(np.intersect1d(sel, prev[k])) / k)
                prev[k] = sel
        for k in ks:
            sel, b, c0 = coef[k]
            preds[k][i] = float(X[t, sel] @ b) + c0
        xo, xi = X[t - TW], X[t]
        Sxx += np.outer(xi, xi) - np.outer(xo, xo)
        Sxy += xi * e[t] - xo * e[t - TW]
        sx += xi - xo
        sy += float(e[t]) - float(e[t - TW])
        syy += float(e[t]) ** 2 - float(e[t - TW]) ** 2

    y_oos = e[TW:]
    rows = []
    for k in ks:
        c, r2, _ = _score(y_oos, preds[k])
        dmt = dm_test(y_oos, preds[k], preds[pv]) if k != pv else 0.0
        ch = float(np.mean(churn[k])) if churn[k] else 0.0
        print(
            f"  top-k={k:3d}: OOS corr {c:+.4f}  R2 {r2:+.5f}  "
            f"DM-t vs dense {dmt:+5.2f}  daily set churn {ch:.3f}",
            flush=True,
        )
        rows.append({"k": k, "corr": c, "r2": r2, "dm_t_vs_dense": dmt, "churn": ch})
    pd.DataFrame(rows).to_csv(f"{OUT}/sparse_vs_dense.csv", index=False)
    print(f"wrote {OUT}/sparse_vs_dense.csv")


# ---------------------------------------------------------------------------
# Stage 6 — is the time variation in the MAPPING, or just in the residual SCALE?
# ---------------------------------------------------------------------------


def stage_verify() -> None:
    """Two closing tests.

    1. **Mapping vs scale.** Monthly IC varies well beyond its null (B1), but IC is a
       *scaled* object. What a forecaster needs is the regression **slope** of the
       residual on the composite. If the slope is stable while the IC moves, the apparent
       time-sensitivity is heteroskedasticity of the residual — real, but not a signal
       that the exog->vol mapping itself is time-varying, and nothing a conditional model
       can monetize. Compared like-for-like: dispersion of monthly IC and of monthly slope,
       each against its own circular-shift null.
    2. **Significance of the one positive.** Diebold-Mariano HAC t on the vol-regime
       conditional (and its 50/50 blend with pooled) vs the pooled combiner.
    """
    os.makedirs(OUT, exist_ok=True)
    p = load_panel()
    e = np.load(_p("har_resid.npz"))["e"][TW:]
    sig = dict(np.load(_p("bucket_signals.npz")))
    ts = pd.Series(pd.to_datetime(p.t[TW + TW :]))
    n = len(e)
    a = sig["all"]
    blk = _blocks(ts)
    nb = blk.max() + 1

    def per_month(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ic, slope, sd = np.full(nb, np.nan), np.full(nb, np.nan), np.full(nb, np.nan)
        for b in range(nb):
            m = blk == b
            if m.sum() < 50:
                continue
            xb, yb = a[m] - a[m].mean(), y[m] - y[m].mean()
            sxx = float(xb @ xb)
            if sxx <= 0:
                continue
            slope[b] = float(xb @ yb) / sxx
            ic[b] = float(xb @ yb) / np.sqrt(sxx * float(yb @ yb))
            sd[b] = float(np.std(y[m]))
        return ic, slope, sd

    ic, slope, sd = per_month(e)
    nic, nsl = [], []
    for sh in _shifts(n):
        i2, s2, _ = per_month(np.roll(e, sh))
        nic.append(np.nanstd(i2))
        nsl.append(np.nanstd(s2))
    cv_ic = np.nanstd(ic) / abs(np.nanmean(ic))
    cv_sl = np.nanstd(slope) / abs(np.nanmean(slope))
    print("MAPPING vs SCALE (207 monthly blocks)")
    print(
        f"  monthly IC   : mean {np.nanmean(ic):+.4f}  sd {np.nanstd(ic):.4f}  "
        f"null sd {np.mean(nic):.4f}  excess {np.nanstd(ic) / np.mean(nic):.2f}x  CV {cv_ic:.2f}"
    )
    print(
        f"  monthly SLOPE: mean {np.nanmean(slope):+.4f}  sd {np.nanstd(slope):.4f}  "
        f"null sd {np.mean(nsl):.4f}  excess {np.nanstd(slope) / np.mean(nsl):.2f}x  CV {cv_sl:.2f}"
    )
    print(
        f"  corr(monthly IC, monthly residual sd)    : {np.corrcoef(ic[np.isfinite(ic)], sd[np.isfinite(ic)])[0, 1]:+.3f}"
    )
    print(
        f"  corr(monthly slope, monthly residual sd) : {np.corrcoef(slope[np.isfinite(slope)], sd[np.isfinite(slope)])[0, 1]:+.3f}"
    )
    pd.DataFrame(
        {
            "month": [str(ts[blk == b].iloc[0])[:7] for b in range(nb)],
            "ic": ic,
            "slope": slope,
            "resid_sd": sd,
        }
    ).to_csv(f"{OUT}/monthly_mapping.csv", index=False)

    print("\nSIGNIFICANCE of the vol-regime conditional (DM HAC t, + = beats pooled)")
    S = np.column_stack([sig[b] for b in BUCKETS])
    cw = COMBINE_WINDOW_DAYS * PERIODS_PER_DAY
    base = np.full(n, np.nan)
    base[cw:] = walk_forward(S, e, cw, alpha=1.0, refit_every=REFIT_EVERY)
    rows = []
    for K in (2, 3, 5):
        bins = _vol_regime(p, ts, K)
        pred = _bin_walk_forward(S, e, bins, K, COMBINE_WINDOW_DAYS)
        m = np.isfinite(pred) & np.isfinite(base)
        blend = 0.5 * (pred[m] + base[m])
        t_sep = dm_test(e[m], pred[m], base[m])
        t_bl = dm_test(e[m], blend, base[m])
        print(
            f"  vol regime K={K}: separate DM-t {t_sep:+.2f}   50/50 blend DM-t {t_bl:+.2f}   n {m.sum()}"
        )
        rows.append(
            {"K": K, "dm_t_separate": t_sep, "dm_t_blend": t_bl, "n": int(m.sum())}
        )
    pd.DataFrame(rows).to_csv(f"{OUT}/volregime_significance.csv", index=False)
    print(f"wrote {OUT}/monthly_mapping.csv + volregime_significance.csv")


def _bin_map(S: np.ndarray, y: np.ndarray, bins: np.ndarray, K: int) -> np.ndarray:
    """(K, n_signals) within-bin IC of each bucket signal."""
    out = np.full((K, S.shape[1]), np.nan)
    for k in range(K):
        m = bins == k
        if m.sum() < 200:
            continue
        xb = S[m] - S[m].mean(0)
        yb = y[m] - y[m].mean()
        with np.errstate(invalid="ignore", divide="ignore"):
            out[k] = (xb * yb[:, None]).sum(0) / (
                np.sqrt((xb**2).sum(0)) * np.sqrt((yb**2).sum())
            )
    return out


def _concentration(S: np.ndarray, y: np.ndarray, bins: np.ndarray, K: int) -> float:
    """Mean within-bin |IC| divided by pooled |IC| — the ``c`` of ``c/sqrt(K)``."""
    A = _bin_map(S, y, bins, K)
    pooled = _bin_map(S, y, np.zeros(len(y), dtype=int), 1)[0]
    return float(np.nanmean(np.abs(A)) / np.nanmean(np.abs(pooled)))


def _coarsen(bins: np.ndarray, K: int) -> np.ndarray:
    """Group 48 clock slots into K contiguous bins of equal slot count."""
    nb = int(bins.max()) + 1
    return np.minimum((bins * K) // nb, K - 1)


def _vol_regime(p, ts: pd.Series, K: int) -> np.ndarray:
    """Causal K-quantile bins of the recent-vol state (``har_ma_25``, expanding edges)."""
    j = p.names.index("har_ma_25")
    v = p.X[TW + TW :, j].astype(np.float64)
    out = np.zeros(len(v), dtype=int)
    step = PERIODS_PER_DAY * 20
    edges = None
    for i in range(0, len(v), step):
        if i >= step:  # causal: edges from history only
            edges = np.quantile(v[:i], np.linspace(0, 1, K + 1)[1:-1])
        seg = v[i : i + step]
        out[i : i + step] = 0 if edges is None else np.searchsorted(edges, seg)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stage",
        choices=["resid", "signals", "tests", "dynamics", "sparse", "verify"],
        required=True,
    )
    a = ap.parse_args()
    {
        "resid": stage_resid,
        "signals": stage_signals,
        "tests": stage_tests,
        "dynamics": stage_dynamics,
        "sparse": stage_sparse,
        "verify": stage_verify,
    }[a.stage]()
