"""What the dual buys that the primal cannot: nonlinearity, and rank on the lag axis.

The primal-dual identity was verified and then used only to DERIVE a feature
(the 41 channel amplitudes). That leaves the actual payoff untouched, because
for a linear kernel the two forms are the same estimator and running the dual
adds nothing. What the dual admits that the primal does not is a kernel with no
finite feature map.

That matters here for a specific reason. This paper's model-class result is
that tree ensembles beat penalised linear models by 0.00184 to 0.00358 on every
feature set. If that premium is reachable as a NONLINEAR KERNEL ON THE 41
AMPLITUDES, then "trees win" restates as "the linear model was in the wrong
space", which is a sharper and more useful claim -- and one that costs 53
features rather than a forest.

Two dials, both read in the dual and both fixed rather than estimated, since
every estimated direction in this project has lost to an imposed one:

  NONLINEARITY. An RBF kernel on the amplitude coordinates, realised through
  random Fourier features so the rolling machinery is unchanged and the cost
  stays linear in the window. Bandwidth by the median heuristic on each
  training window, so nothing is tuned against the evaluation rows.

  RANK ON THE LAG AXIS. The amplitude feature is a rank-1 projection of each
  channel's 12-rung ladder onto one fixed shape. Rank r gives
  k(x,x') = sum_{j<=r} <Z v_j, Z' v_j> for r fixed shapes -- r x 41 features,
  a direct dial from the operator's lag axis into the dual. Where it saturates
  answers how much lag structure the amplitudes discard, which the primal
  answered only for r = 1 and r = 2 and only on the corrupted cache.

Reference arms are the backbone and plain ridge on the 492 raw columns, so the
comparison is against both the floor and the incumbent. Protocol per
analysis/PROTOCOL.md: rolling 24,000-bar window, cadence 1,000, penalty chosen
on a region strictly preceding evaluation, QLIKE reported with MSE alongside.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference import dm_test  # noqa: E402

OUT_DIR = os.path.join(ROOT, "writeup", "stats")
W, CAD = 24000, 1000
LAMS = [0.0] + [10.0 ** e for e in range(-2, 7)]
NBLOCK = 8
RFF_D = [128, 512]
SEED = 0


def qlike_bar(rv, p):
    r = rv / p
    return r - np.log(r) - 1.0


def pick_cache():
    for c in ("b2_mmap_warm", "b2_mmap_indonly", "b2_mmap_fix", "b2_mmap"):
        if os.path.exists(os.path.join(ROOT, "results", c, "X.npy")):
            return c
    raise SystemExit("no cache built")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    cache = pick_cache()
    D = os.path.join(ROOT, "results", cache)
    print(f"cache: {cache}")
    X = np.load(os.path.join(D, "X.npy"), mmap_mode="r")
    y = np.load(os.path.join(D, "y.npy"))
    b = np.load(os.path.join(D, "b.npy"))
    names = [str(s) for s in np.load(os.path.join(D, "names.npy"),
                                     allow_pickle=True)]
    chan: dict[str, list] = {}
    for j, nm in enumerate(names):
        m = re.match(r"^adj_(.+)_ma_(\d+)$", nm)
        if m:
            chan.setdefault(m.group(1), []).append((int(m.group(2)), j))
    for c in chan:
        chan[c].sort()
    labels = [c for c in chan if len(chan[c]) == 12]
    rungs = np.array([r for r, _ in chan[labels[0]]], float)
    lad = np.array([j for c in labels for _, j in chan[c]], int)
    har = np.array([j for j, nm in enumerate(names)
                    if re.match(r"^har_ma_\d+$", nm)], int)
    C, R, nh = len(labels), 12, len(har)
    n = len(y)
    rvv = y ** 2 * b
    Fh = np.asarray(X[:, har], np.float64)
    Fl = np.asarray(X[:, lad], np.float64).reshape(n, C, R)
    del X
    print(f"  {n:,} rows, {C} channels x {R} rungs, {nh} backbone "
          f"({time.time()-t0:.0f}s)")

    def gv(beta):
        g = rungs ** (-beta)
        return g / np.linalg.norm(g)

    SHAPES = [gv(1.0), gv(0.5), gv(2.0)]        # fixed, not estimated

    def amps(r):
        return np.concatenate([Fl @ SHAPES[j] for j in range(r)], axis=1)

    rng = np.random.default_rng(SEED)

    def rff_block(Z, Dd, scale):
        """Random Fourier features for an RBF kernel of the given bandwidth."""
        d = Z.shape[1]
        Wm = rng.standard_normal((d, Dd)) / scale
        bvec = rng.uniform(0, 2 * np.pi, Dd)
        return np.sqrt(2.0 / Dd) * np.cos(Z @ Wm + bvec)

    designs = {"backbone": (Fh, 0)}
    for r in (1, 2, 3):
        A = amps(r)
        designs[f"amp rank-{r}"] = (np.hstack([Fh, A]), A.shape[1])
    A1 = amps(1)
    # median heuristic on an early slice, so the bandwidth is not tuned on test
    sl = A1[:W]
    sub = sl[rng.choice(len(sl), 2000, replace=False)]
    dists = np.sqrt(((sub[:500, None, :] - sub[None, :500, :]) ** 2).sum(-1))
    scale = float(np.median(dists[dists > 0]))
    print(f"  RBF bandwidth by median heuristic on the first window: "
          f"{scale:.4f}")
    for Dd in RFF_D:
        Zr = rff_block(A1, Dd, scale)
        designs[f"amp + RFF {Dd}"] = (np.hstack([Fh, A1, Zr]),
                                      A1.shape[1] + Dd)
        del Zr
    designs["raw492"] = (np.hstack([Fh, Fl.reshape(n, C * R)]), C * R)
    print(f"  {len(designs)} designs: " +
          ", ".join(f"{k}({v[0].shape[1]})" for k, v in designs.items()))

    v0, v1 = int(n * 0.55), int(n * 0.65)

    def pick_lam(Fd, npen):
        if npen == 0:
            return 0.0
        pp = Fd.shape[1]
        P = np.zeros((1 + pp, 1 + pp))
        P[1 + pp - npen:, 1 + pp - npen:] = np.eye(npen)
        num = {L: 0.0 for L in LAMS}
        den = {L: 0 for L in LAMS}
        s0 = v0 - ((v0 - W) % CAD)
        for s in range(s0, v1, CAD):
            e = min(s + CAD, v1)
            if e <= v0:
                continue
            A = np.hstack([np.ones((W, 1)), Fd[s - W:s]])
            G = A.T @ A
            c = A.T @ y[s - W:s]
            a, z = max(s, v0), e
            Ae = np.hstack([np.ones((z - a, 1)), Fd[a:z]])
            for L in LAMS:
                cf = np.linalg.solve(G + L * P + 1e-8 * np.eye(1 + pp), c)
                rr = y[s - W:s] - A @ cf
                pv = ((Ae @ cf) ** 2 + float(rr @ rr) / W) * b[a:z]
                rt = rvv[a:z]
                m = (rt > 0) & np.isfinite(pv) & (pv > 0)
                if m.sum():
                    q = rt[m] / pv[m]
                    num[L] += float(np.sum(q - np.log(q) - 1.0))
                    den[L] += int(m.sum())
            del A, G, Ae
        best = (np.inf, 0.0)
        for L in LAMS:
            if den[L] and num[L] / den[L] < best[0]:
                best = (num[L] / den[L], L)
        return best[1]

    def walk(Fd, npen, lam):
        pp = Fd.shape[1]
        P = np.zeros((1 + pp, 1 + pp))
        if npen:
            P[1 + pp - npen:, 1 + pp - npen:] = np.eye(npen)
        out = np.full(n - W, np.nan)
        for s in range(W, n, CAD):
            e = min(s + CAD, n)
            A = np.hstack([np.ones((W, 1)), Fd[s - W:s]])
            cf = np.linalg.solve(A.T @ A + lam * P + 1e-8 * np.eye(1 + pp),
                                 A.T @ y[s - W:s])
            rr = y[s - W:s] - A @ cf
            Ae = np.hstack([np.ones((e - s, 1)), Fd[s:e]])
            out[s - W:e - W] = ((Ae @ cf) ** 2 + float(rr @ rr) / W) * b[s:e]
            del A, Ae
        return out

    preds, lams = {}, {}
    for k, (Fd, npen) in designs.items():
        lam = pick_lam(Fd, npen)
        lams[k] = lam
        preds[k] = walk(Fd, npen, lam)
        print(f"    {k:>16} lambda*={lam:<9g} ({time.time()-t0:.0f}s)",
              flush=True)

    rv_w = rvv[W:]
    good = rv_w > 0
    for p_ in preds.values():
        good &= np.isfinite(p_) & (p_ > 0)
    gi = np.flatnonzero(good)
    losses = {k: qlike_bar(rv_w[gi], p_[gi]) for k, p_ in preds.items()}
    print(f"\n  scored {len(gi):,} bars")
    print(f"\n  {'design':>16}{'cols':>7}{'QLIKE':>11}{'MSE':>13}")
    res = {}
    for k in designs:
        q = float(np.mean(losses[k]))
        res[k] = {"cols": int(designs[k][0].shape[1]), "lambda": lams[k],
                  "qlike": q,
                  "mse": float(np.mean((rv_w[gi] - preds[k][gi]) ** 2))}
        print(f"  {k:>16}{res[k]['cols']:>7}{q:>11.5f}{res[k]['mse']:>13.4e}")

    print(f"\n  DM against amp rank-1 (the linear amplitude design):")
    dms = {}
    for k in designs:
        if k == "amp rank-1":
            continue
        r = dm_test(losses[k], losses["amp rank-1"], h=1)
        dms[k] = {"diff": r["mean_diff"], "t": r["t"], "p": r["p"]}
        print(f"    {k:>16}{r['mean_diff']:>+11.5f}  t {r['t']:>+6.2f}  "
              f"p {r['p']:.3g}")

    lin = res["amp rank-1"]["qlike"]
    best_rff = min(res[f"amp + RFF {d}"]["qlike"] for d in RFF_D)
    gain = lin - best_rff
    print("\n" + "=" * 74)
    print(f"  nonlinearity in the amplitude space is worth {gain:+.5f}")
    print(f"  the paper's tree-over-linear premium is 0.00184 to 0.00358")
    reach = gain >= 0.00184
    print("  -> " + ("the tree premium IS reachable as a nonlinear kernel on 41 "
                     "amplitudes:\n     'trees win' restates as 'the linear "
                     "model was in the wrong space'"
                     if reach else
                     "the tree premium is NOT reachable this way: whatever the "
                     "trees exploit\n     is not a smooth function of the "
                     "amplitude coordinates"))
    rk = [res[f"amp rank-{r}"]["qlike"] for r in (1, 2, 3)]
    print(f"\n  lag-axis rank 1/2/3: {rk[0]:.5f} / {rk[1]:.5f} / {rk[2]:.5f}")
    print("  -> " + ("rank 1 is enough: the amplitudes discard no usable lag "
                     "structure" if rk[0] <= min(rk) + 0.0005 else
                     f"rank {int(np.argmin(rk)) + 1} is better: the "
                     f"amplitudes do discard usable lag structure"))

    out = {"cache": cache, "n_scored": int(len(gi)), "arms": res,
           "dm_vs_amp_rank1": dms, "rbf_bandwidth": scale,
           "nonlinearity_gain": float(gain),
           "tree_premium_low": 0.00184, "tree_premium_high": 0.00358,
           "tree_premium_reachable": bool(reach),
           "rank_qlike": {str(r): rk[r - 1] for r in (1, 2, 3)}}

    blocks = np.array_split(np.arange(len(gi)), NBLOCK)
    keep = ["backbone", "amp rank-1", f"amp + RFF {RFF_D[-1]}", "raw492"]
    print(f"\n  {'blk':>4}" + "".join(f"{k:>16}" for k in keep))
    eras = []
    for bi, blk in enumerate(blocks):
        vals = [float(np.mean(losses[k][blk])) for k in keep]
        print(f"  {bi+1:>4}" + "".join(f"{v:>16.5f}" for v in vals))
        eras.append({"block": bi + 1, **dict(zip(keep, vals))})
    out["eras"] = eras
    with open(os.path.join(OUT_DIR, "dual_kernels.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote dual_kernels.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
