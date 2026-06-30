"""OOS-robust feature-evaluation harness — the keystone that gates feature development.

Built because every architecture lever explored in the regime-MoE arc was a LOCAL mirage that died at
deployment: a single-split / weak-base / in-sample number looked promising, then collapsed full-OOS. The
lesson is that OOS-robustness is not a model property to bolt on but an EVALUATION DISCIPLINE — a feature
ships only if its incremental OOS QLIKE delta, measured under the deployment protocol, (1) clears k·se,
(2) replicates across purged walk-forward folds, and (3) beats its own shuffled placebo. Context-aware
features are scored with the context assignment CROSS-FITTED (out-of-fold) so the conditioning cannot leak.

Pillars (all three the user asked for):
  - cross-validated : `purged_walk_forward` (embargoed expanding-window; financial-TS-correct, no boundary leak)
  - bagged          : `score_feature` bootstraps the train resample each fold -> a delta distribution, not a point
  - context-aware   : `oof_context` (cross-fitted bins) + `context_columns` (feature x OOF-context, leak-free)
The gate (`significance_gate`) is the single yes/no every candidate must pass.
"""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np

QFn = Callable[[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


class _Estimator(Protocol):
    """Minimal sklearn-style contract for the screening model (small XGB / ridge / FM — anything fast)."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_Estimator": ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...


def purged_walk_forward(n: int, n_folds: int = 5, embargo: float = 0.01) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window walk-forward with an embargo (purge) gap between train-end and test-start.

    The cadence targets overlap, so a naive fold boundary leaks label information forward; the embargo gap
    drops the `embargo`-fraction of rows straddling the boundary. Train always precedes test (causal)."""
    fold = n // (n_folds + 1)
    emb = max(1, int(embargo * n))
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(1, n_folds + 1):
        tr_end = k * fold
        te0 = tr_end + emb
        te1 = min(te0 + fold, n)
        if te1 - te0 < 20:
            break
        out.append((np.arange(0, tr_end), np.arange(te0, te1)))
    return out


def oof_context(values: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]], n_bins: int = 2) -> np.ndarray:
    """Cross-fitted (out-of-fold) context label from an observable: each fold's TEST rows are binned by
    quantile EDGES estimated on that fold's TRAIN rows only — so the context assignment never sees its own
    row's label or the test distribution. Rows never in any test fold stay -1 (unused)."""
    lab = np.full(len(values), -1, dtype=int)
    for tr, te in folds:
        edges = np.quantile(values[tr], np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
        lab[te] = np.digitize(values[te], edges)
    return lab


def context_columns(f: np.ndarray, ctx: np.ndarray, n_bins: int) -> np.ndarray:
    """Expand a feature into context-conditional columns: column b carries f where ctx==b else 0, so a model
    can give the feature a DIFFERENT effect per regime. ctx must be the cross-fitted label from `oof_context`."""
    return np.column_stack([np.where(ctx == b, f, 0.0) for b in range(n_bins)]).astype(np.float32)


def _qlike(resid_pred: np.ndarray, offset_te: np.ndarray, y_te: np.ndarray, base_te: np.ndarray, smear: QFn) -> float:
    """Whole-slice QLIKE of the vol forecast (base+global offset + regime resid prediction), Duan-smeared."""
    pr, trr = smear(offset_te + resid_pred, y_te, base_te)
    m = (trr > 0) & (pr > 0)
    rr = trr[m] / pr[m]
    return float(np.mean(rr - np.log(rr) - 1.0))


def score_feature(
    Xbase: np.ndarray,
    candidate: np.ndarray,
    target: np.ndarray,
    offset: np.ndarray,
    y: np.ndarray,
    base: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    smear: QFn,
    mk_model: Callable[[int], _Estimator],
    n_boot: int = 8,
    placebo: bool = True,
    seed0: int = 0,
) -> dict:
    """Incremental OOS value of `candidate` (one or more columns) ADDED to `Xbase`, for predicting `target`.

    Delta = QLIKE(with candidate) - QLIKE(without); negative => the candidate helps. Each (fold, bootstrap)
    resamples the train rows (bagging) so the result is a DISTRIBUTION of deltas. The placebo permutes the
    candidate (destroying its alignment with the target) and re-scores — a feature that only fits noise will
    match its placebo. Returns per-(fold,boot) deltas + placebo deltas + the per-fold means (for replication).
    """
    cand = candidate.reshape(len(candidate), -1).astype(np.float32)
    Xf = np.column_stack([Xbase, cand]).astype(np.float32)
    rng = np.random.default_rng(seed0)
    deltas: list[float] = []
    plac: list[float] = []
    fold_means: list[float] = []
    for tr, te in folds:
        ot, yt, bt = offset[te], y[te], base[te]
        fold_d: list[float] = []
        for s in range(n_boot):
            tri = tr[rng.integers(0, len(tr), len(tr))]
            q0 = _qlike(mk_model(s).fit(Xbase[tri], target[tri]).predict(Xbase[te]), ot, yt, bt, smear)
            q1 = _qlike(mk_model(s).fit(Xf[tri], target[tri]).predict(Xf[te]), ot, yt, bt, smear)
            d = q1 - q0
            deltas.append(d)
            fold_d.append(d)
            if placebo:
                Xp = np.column_stack([Xbase, rng.permutation(cand)]).astype(np.float32)
                qp = _qlike(mk_model(s).fit(Xp[tri], target[tri]).predict(Xp[te]), ot, yt, bt, smear)
                plac.append(qp - q0)
        fold_means.append(float(np.mean(fold_d)))
    return {
        "deltas": np.array(deltas),
        "placebo": np.array(plac),
        "fold_means": np.array(fold_means),
    }


def inner_split(n_train: int, val_frac: float = 0.2, embargo: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """Per-walk-forward-step inner split: carve a RECENT validation tail from the training window, with an
    embargo gap so the inner-fit cannot leak into inner-val. Returns (inner_fit_idx, inner_val_idx) — both
    indices into the train window. Used to tune hyperparameters (omega, aux weight, ...) per step on OOS
    data rather than fixing them globally or fitting them in-sample (which overfits — e.g. omega->1.0)."""
    nv = max(20, int(val_frac * n_train))
    emb = max(1, int(embargo * n_train))
    fit_end = n_train - nv - emb
    return np.arange(0, fit_end), np.arange(fit_end + emb, n_train)


def tune_hparam(
    fit_predict_val: Callable[[object], np.ndarray],
    candidates: list,
    val_target: np.ndarray,
    val_offset: np.ndarray,
    val_y: np.ndarray,
    val_base: np.ndarray,
    smear: QFn,
    n_boot: int = 200,
    shrink_to: float | None = None,
    shrink_lambda: float = 0.0,
    seed: int = 0,
) -> tuple[object, dict]:
    """Robustly select ONE hyperparameter on a purged inner-validation set by BOOTSTRAP-MEAN QLIKE — the
    deployment-correct alternative to a fixed grid value or an in-sample gradient fit.

    `fit_predict_val(hp)` returns the regime resid prediction on the val rows for candidate `hp` (for omega
    this is just `w*pe_val + (1-w)*pm_val` with EBM/MTFM fit once on inner-fit; for aux_weight/gate params it
    refits the MTFM on inner-fit). Bootstrapping the val rows guards against a noisy single-val argmin; an
    optional shrinkage toward a prior `shrink_to` (e.g. the global-best omega) regularizes a thin val tail.
    Returns the selected hp + each candidate's bootstrap-mean val QLIKE (auditable)."""
    nval = len(val_target)
    rng = np.random.default_rng(seed)
    boots = [rng.integers(0, nval, nval) for _ in range(n_boot)]
    preds = {repr(hp): np.asarray(fit_predict_val(hp), dtype=float) for hp in candidates}

    def _q(p: np.ndarray, idx: np.ndarray) -> float:
        pr, trr = smear(val_offset[idx] + p[idx], val_y[idx], val_base[idx])
        m = (trr > 0) & (pr > 0)
        rr = trr[m] / pr[m]
        return float(np.mean(rr - np.log(rr) - 1.0))

    means = np.array([np.mean([_q(preds[repr(hp)], b) for b in boots]) for hp in candidates])
    best = candidates[int(np.argmin(means))]
    if shrink_to is not None and shrink_lambda > 0 and isinstance(best, (int, float)):
        best = (1.0 - shrink_lambda) * best + shrink_lambda * shrink_to
    return best, dict(zip([repr(h) for h in candidates], means))


def context_omega(
    c_val: np.ndarray,
    pe_val: np.ndarray,
    pm_val: np.ndarray,
    val_off: np.ndarray,
    val_y: np.ndarray,
    val_base: np.ndarray,
    smear: QFn,
    c_test: np.ndarray,
    anchors: np.ndarray,
    candidates: np.ndarray,
    tau: float,
    global_w: float,
    shrink_strength: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Context-aware blend weight omega(x) = attention over context ANCHORS, each carrying a purged-val
    CV-fit omega_k shrunk toward the global CV-omega (partial pooling). This RESCUES the attention gate that
    self-disabled under in-sample gradient: each omega_k is a CV-ARGMIN on OOS val (cannot collapse), the
    shrinkage controls the K-weight capacity, and the attention softmax smooths the bin boundaries.

    Inputs are inner-VAL context/predictions (to fit omega_k) and the test context `c_test` (to apply). Query =
    `c_test`, keys = `anchors` (cross-fit regime prototypes), values = the shrunk omega_k. Returns omega(test)
    and the per-anchor omega_k (post-shrinkage, for inspection)."""
    cand = np.asarray(candidates, dtype=float)
    qrows = []
    for w in cand:  # per-row val QLIKE contribution for each candidate (global Duan smear, per-row term)
        pr, trr = smear(val_off + w * pe_val + (1.0 - w) * pm_val, val_y, val_base)
        rr = np.where(pr > 0, trr / np.where(pr > 0, pr, 1.0), np.nan)
        qrows.append(rr - np.log(np.where(rr > 0, rr, np.nan)) - 1.0)
    Q = np.array(qrows)  # [n_cand, n_val]
    omega_k = np.empty(len(anchors))
    for k, a in enumerate(anchors):
        attn = np.exp(-((c_val - a) ** 2).sum(1) / tau)  # kernel weight of val rows near anchor k
        sw = float(attn.sum()) + 1e-12
        scores = np.nansum(Q * attn, axis=1) / sw  # attention-weighted val QLIKE per candidate
        wk = cand[int(np.nanargmin(scores))]
        neff = sw**2 / (float((attn**2).sum()) + 1e-12)  # effective sample size near the anchor
        lam = shrink_strength / (shrink_strength + neff)  # thin anchors -> shrink hard toward global
        omega_k[k] = (1.0 - lam) * wk + lam * global_w
    A = np.exp(-((c_test[:, None, :] - np.asarray(anchors)[None, :, :]) ** 2).sum(2) / tau)  # [n_test, K]
    A = A / (A.sum(1, keepdims=True) + 1e-12)
    return A @ omega_k, omega_k


def significance_gate(result: dict, k: float = 2.0, repl_frac: float = 0.7) -> tuple[bool, dict]:
    """The single OOS-robust pass/fail. A candidate ships iff ALL hold:
      (1) CI excludes 0   : mean + k·se(mean) < 0           (the gain is significantly negative)
      (2) replicates      : >= repl_frac of (fold,boot) draws favor the feature, AND >= repl_frac of folds
      (3) beats placebo   : real mean is k-sigma below the permuted-placebo mean (real signal, not overfit)
    Reports every component so a rejection is auditable (which condition failed)."""
    d = np.asarray(result["deltas"], dtype=float)
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("inf")
    repl = float((d < 0).mean())
    fm = np.asarray(result["fold_means"], dtype=float)
    fold_repl = float((fm < 0).mean()) if len(fm) else 0.0
    ci_excl0 = (mean + k * se) < 0
    plac = np.asarray(result.get("placebo", []), dtype=float)
    if len(plac) > 1:
        pse = plac.std(ddof=1) / np.sqrt(len(plac))
        z_vs_plac = (float(plac.mean()) - mean) / np.sqrt(se**2 + pse**2 + 1e-300)
        beats_placebo = z_vs_plac > k
    else:
        z_vs_plac, beats_placebo = float("nan"), True
    passes = bool(ci_excl0 and repl >= repl_frac and fold_repl >= repl_frac and beats_placebo)
    return passes, {
        "mean": mean,
        "se": se,
        "repl_frac": repl,
        "fold_repl_frac": fold_repl,
        "ci_excludes_0": bool(ci_excl0),
        "beats_placebo": bool(beats_placebo),
        "z_vs_placebo": z_vs_plac,
        "n_draws": int(len(d)),
    }
