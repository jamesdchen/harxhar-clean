"""Addendum 3 patch, part 1 (module level): the binomial availability rule and the scaled batch fit."""

from pathlib import Path

p = Path(r"C:\Users\james\AppData\Local\Temp\std_wt\experiments\pool_linear_pretune.py")
t = p.read_bytes().decode()


def rep(old: str, new: str, cnt: int = 1) -> None:
    global t
    assert t.count(old) == cnt, (t.count(old), old[:90])
    t = t.replace(old, new)


rep(
    '''def std_weights(
    Xa: np.ndarray, locked: np.ndarray, ref: np.ndarray, full: bool = False
) -> tuple[np.ndarray, ...]:
    """(s, out) of the fit's rows Xa (masked columns already 0): s_j = np.std(Xa[:, j]) (ddof 0); out = the non-locked
    columns with s_j < STD_MIN or (the evidence mask, 2026-10-22, addenda 1 and 2) an evidence below STD_MIN_EVIDENCE:
    the rows that differ from the modal value of the column's reference column (evid_ref: the variable's k = 1 column)
    (masked for the fit, every masked column among them); s_j := 1 on out and on the locked intercept (a weight never
    used: the column is 0, resp. unpenalized).  full: also (modal value, own off-mode count) of every column."""
    s = np.std(Xa, axis=0)
    v, cnt = col_mode(Xa)
    ev_own = Xa.shape[0] - cnt
    ev = group_evidence(ev_own, ref)
    out = ~locked & ((s < STD_MIN) | (ev < STD_MIN_EVIDENCE))
    sw = np.where(out | locked, 1.0, s)
    return (sw, out, v, ev_own) if full else (sw, out)
''',
    '''def avail_k(names: list[str], bb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """2026-10-22 (RULE_std.txt addendum 3): the availability averages <v>_avail_ma_<k> with k > 1 (column indices in
    design order) and their k -- the columns the binomial availability rule applies to (the backbone exempt)."""
    idx, ks = [], []
    for j, n in enumerate(names):
        m = _EV_AVAIL.match(n)
        if m and not bb[j] and int(m.group(2)) > 1:
            idx.append(j)
            ks.append(int(m.group(2)))
    return np.array(idx, dtype=np.int64), np.array(ks, dtype=np.int64)


def av_stats(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(m, s1, C) of the window rows A of the availability averages: the means, the sums of A - m and its cross-products
    (the between-seed tracker's state; mean = m + s1 / n, cov = C / n - (s1 / n)(s1 / n)')."""
    m = A.mean(axis=0)
    D = A - m
    return m, D.sum(axis=0), D.T @ D


def av_mean_cov(st: tuple, n: int) -> tuple[np.ndarray, np.ndarray]:
    """The window means and covariance matrix of the availability averages from av_stats-type sums over n rows."""
    m, s1, C = st
    return m + s1 / n, C / n - np.outer(s1 / n, s1 / n)


def binom_mask(
    mean: np.ndarray, cov: np.ndarray, ks: np.ndarray, eligible: np.ndarray
) -> tuple[np.ndarray, list]:
    """Addendum 3, the binomial availability rule on the availability averages (k > 1) eligible after the existing
    masks: se = sqrt(p (1 - p) / k), p = the window mean; (i) near-constant: masked if the window s.d. <= se; (ii)
    near-duplicate: in design order, masked if against an earlier kept one of the same k the s.d. of the difference
    <= sqrt(2) sqrt(pbar (1 - pbar) / k), pbar the pair's mean level.  Returns (masked over the positions, records
    (position, "near-constant" | "near-duplicate", partner position or -1))."""
    q = len(ks)
    masked = np.zeros(q, dtype=bool)
    recs: list = []
    kept: dict = {}
    var = np.diag(cov)
    for i in range(q):
        if not eligible[i]:
            continue
        k = int(ks[i])
        p = float(mean[i])
        if np.sqrt(max(float(var[i]), 0.0)) <= np.sqrt(max(p * (1.0 - p), 0.0) / k):
            masked[i] = True
            recs.append((i, "near-constant", -1))
            continue
        kk = kept.setdefault(k, [])
        if kk:
            J = np.array(kk)
            pbar = 0.5 * (p + mean[J])
            sep = np.sqrt(np.maximum(pbar * (1.0 - pbar), 0.0) / k)
            sdd = np.sqrt(np.maximum(var[i] + var[J] - 2.0 * cov[i, J], 0.0))
            hit = np.flatnonzero(sdd <= np.sqrt(2.0) * sep)
            if hit.size:
                masked[i] = True
                recs.append((i, "near-duplicate", int(J[hit[0]])))
                continue
        kk.append(i)
    return masked, recs


def std_weights(
    Xa: np.ndarray,
    locked: np.ndarray,
    ref: np.ndarray,
    av: tuple,
    full: bool = False,
) -> tuple:
    """(s, out) of the fit's rows Xa (masked columns already 0): s_j = np.std(Xa[:, j]) (ddof 0); out = the non-locked
    columns with s_j < STD_MIN or (the evidence mask, 2026-10-22, addenda 1 and 2) an evidence below STD_MIN_EVIDENCE:
    the rows that differ from the modal value of the column's reference column (evid_ref: the variable's k = 1 column),
    and then (addendum 3) the availability averages av = avail_k(...) the binomial rule masks (masked for the fit,
    every masked column among them); s_j := 1 on out and on the locked intercept (a scale never used: the column is 0,
    resp. unscaled).  full: also (modal value, own off-mode count) of every column, the availability averages' (m, s1,
    C) and the binomial rule's records (column, kind, partner column)."""
    s = np.std(Xa, axis=0)
    v, cnt = col_mode(Xa)
    ev_own = Xa.shape[0] - cnt
    ev = group_evidence(ev_own, ref)
    out = ~locked & ((s < STD_MIN) | (ev < STD_MIN_EVIDENCE))
    ai, ak = av
    st = av_stats(Xa[:, ai])
    mean, cov = av_mean_cov(st, Xa.shape[0])
    bm, recs = binom_mask(mean, cov, ak, ~out[ai])
    out[ai[bm]] = True
    sw = np.where(out | locked, 1.0, s)
    if not full:
        return sw, out
    rr = [(int(ai[i]), kind, int(ai[j]) if j >= 0 else -1) for i, kind, j in recs]
    return sw, out, v, ev_own, st, rr
''',
)

rep(
    '''def fit_std(
    Xa, yr, locked, kind, a, l1, pf, ref, sw=None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One standardized batch fit on the rows (Xa, yr) (masked columns 0): (theta, s, out) with s, out of std_weights
    (evidence references ref = evid_ref; or sw, std_weights of these rows computed by the caller); the out columns
    zeroed for the fit (theta_j = 0).  Ridge: (X'X + diag(a pf s^2)) theta = X'y on the kept columns; lasso / enet:
    batch_theta_w with w1 = pf s, w2 = pf s^2."""
    s, out = std_weights(Xa, locked, ref) if sw is None else sw
    if out.any():
        Xa = Xa.copy()
        Xa[:, out] = 0.0
    if kind == "ridge":
        # (addendum 2) on the columns centred by their means on these rows; theta back in the original coordinates
        # (intercept - m' beta); the lasso / enet batch residualizes on the intercept already (FWL)
        m = np.where(locked, 0.0, Xa.mean(axis=0))
        Xc = Xa - m
        D = np.where(locked, 0.0, a * (pf * s * s))
        th = ridge_keep(Xc.T @ Xc, Xc.T @ yr, D, ~out)
        th[locked] -= m @ th
        return th, s, out
    return batch_theta_w(Xa, yr, locked, a, l1, pf * s, pf * s * s), s, out
''',
    '''def batch_z(Z, yr, locked, a, l1, pf) -> np.ndarray:
    """The KKT-certified batch fit in the scaled coordinates (addendum 3): the spec's FWL + batch elastic net with the
    penalty factors pf only (batch_theta_pf with the spec's own enet_coef)."""
    import src.models.reclasso_har as RH

    return batch_theta_pf({"enet_coef": RH.enet_coef}, Z, yr, locked, a, l1, pf)


def fit_std(
    Xa, yr, locked, kind, a, l1, pf, ref, av, sw=None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One standardized batch fit on the rows (Xa, yr) (masked columns 0): (theta, s, out) with s, out of std_weights
    (evidence references ref = evid_ref, availability averages av = avail_k; or sw, std_weights of these rows computed
    by the caller); the out columns masked for the fit (theta_j = 0).  Addendum 3: solved in the scaled coordinates
    z = (x - m) / s (m, s of these rows) with the penalty factors pf only, intercept free -- ridge (Z'Z + diag(a pf))
    gamma = Z'y on the kept columns, lasso / enet batch_z -- and back-transformed (beta = gamma / s, intercept
    gamma_0 - m' beta): the weighted + centred problem of addenda 0-2."""
    s, out = std_weights(Xa, locked, ref, av) if sw is None else sw
    m = np.where(locked | out, 0.0, Xa.mean(axis=0))
    Z = (Xa - m) / s
    Z[:, out] = 0.0
    if kind == "ridge":
        D = np.where(locked, 0.0, a * pf)
        g = ridge_keep(Z.T @ Z, Z.T @ yr, D, ~out)
    else:
        g = batch_z(Z, yr, locked, a, l1, pf)
    th = g / s
    th[locked] = g[locked] - m @ th
    return th, s, out
''',
)
p.write_bytes(t.encode())
print("module patched")
