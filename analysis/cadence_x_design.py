"""Does refit cadence change the ORDERING of designs, or only their levels?

The reconciliation to the paper's 0.12788 was a malformed question. That number
was computed on the FROZEN BATTERY PANEL, which is the pre-fix cache -- the
corrupted one this whole session exists to document. Reproducing it would mean
reproducing the defect. Comparability to it is not merely expensive, it is the
wrong target, and three turns were spent chasing it.

The question that survives is narrower and is about this session's own claims.
Every arm here refits every 1,000 bars while the battery refits every bar, and
cadence was never varied, so it was treated as neutral scaffolding. It is not
obviously neutral: on the full design lambda* already fell from 10,000 to 1,000
when the cadence tightened from 1,000 to 250, so cadence and regularisation are
coupled, and a coupled nuisance parameter held fixed is exactly how a design
ranking becomes an artifact.

So this crosses cadence with design rather than sweeping it on one design:

    cadence   1000 / 250 / 100
    design    backbone(12) / ridge-504 / full-1077 / full + exog x clock

If the cadence effect is roughly common across designs, the within-session
orderings stand and the protocol simply needs stating. If cadence INTERACTS
with design -- and tighter refitting plausibly favours wider designs, since
fresher data needs less shrinkage -- then the rankings reported in this session
are artifacts of a parameter that was never varied. That is the same failure
mode as H3, one level up: a factor held fixed while conclusions are drawn
across the factor it interacts with.

lambda is re-selected per cell on the validation region, so cadence and penalty
are not confounded.
"""

from __future__ import annotations
import json, os, re, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import dm_test                                    # noqa: E402
from fastwalk import RollingGram, block, predict, rss, solve_path  # noqa: E402

OUT_DIR = os.path.join(ROOT, "writeup", "stats")
CACHE = os.environ.get("CX_CACHE", "b2_mmap_warm")
W = 24000
CADS = [1000, 250, 100]
LAMS = [0.0] + [10.0 ** e for e in range(-2, 8)]
NBLOCK = 8


def qlike_bar(rv, p):
    r = rv / p
    return r - np.log(r) - 1.0


def main():
    t0 = time.time()
    D = os.path.join(ROOT, "results", CACHE)
    X = np.load(os.path.join(D, "X.npy"), mmap_mode="r")
    y = np.load(os.path.join(D, "y.npy")); b = np.load(os.path.join(D, "b.npy"))
    names = [str(s) for s in np.load(os.path.join(D, "names.npy"), allow_pickle=True)]
    n = len(y); rvv = y ** 2 * b
    har = np.array([j for j, nm in enumerate(names) if re.match(r"^har_ma_\d+$", nm)], int)
    chan = {}
    for j, nm in enumerate(names):
        m = re.match(r"^adj_(.+)_ma_(\d+)$", nm)
        if m: chan.setdefault(m.group(1), []).append((int(m.group(2)), j))
    for c in chan: chan[c].sort()
    labels = [c for c in chan if len(chan[c]) == 12]
    lad = np.array([j for c in labels for _, j in chan[c]], int)
    rest = np.array([j for j in range(len(names)) if j not in set(har)], int)
    clk = np.array([names.index(c) for c in ("is_open", "is_close", "is_overnight")
                    if c in names], int)
    nh = len(har)
    H = np.asarray(X[:, har], np.float64)
    Z = np.asarray(X[:, lad], np.float64)
    R = np.asarray(X[:, rest], np.float64)
    K = np.asarray(X[:, clk], np.float64)
    del X
    DESIGNS = {
        "backbone": H,
        "ridge-504": np.hstack([H, Z]),
        "full-1077": np.hstack([H, R]),
        "full + exog x clk": np.hstack([H, R] + [Z * K[:, [i]] for i in range(K.shape[1])]),
    }
    print(f"cache {CACHE}: {n:,} rows; designs "
          f"{ {k: v.shape[1] for k, v in DESIGNS.items()} }", flush=True)
    v0, v1 = int(n * 0.55), int(n * 0.65)

    def run(F, npen, CAD):
        T = np.eye(F.shape[1])
        rg = RollingGram(F, y, W)
        num = {L: 0.0 for L in LAMS}; den = {L: 0 for L in LAMS}
        s0 = v0 - ((v0 - W) % CAD)
        for s in range(s0, v1, CAD):
            e = min(s + CAD, v1)
            if e <= v0: continue
            G, c = rg.advance(s); M, vv = block(G, c, T)
            a, z = max(s, v0), e
            for L, cf in solve_path(M, vv, max(npen, 1), LAMS).items():
                pv = (predict(F[a:z], T, cf) ** 2 + rss(M, vv, cf, rg.yy) / W) * b[a:z]
                rt = rvv[a:z]; mk = (rt > 0) & np.isfinite(pv) & (pv > 0)
                if mk.sum():
                    q = rt[mk] / pv[mk]
                    num[L] += float(np.sum(q - np.log(q) - 1.0)); den[L] += int(mk.sum())
        lam = min((num[L] / den[L], L) for L in LAMS if den[L])[1]
        rg = RollingGram(F, y, W)
        out = np.full(n - W, np.nan)
        for s in range(W, n, CAD):
            e = min(s + CAD, n)
            G, c = rg.advance(s); M, vv = block(G, c, T)
            cf = solve_path(M, vv, max(npen, 1), [lam])[lam]
            out[s - W:e - W] = (predict(F[s:e], T, cf) ** 2
                                + rss(M, vv, cf, rg.yy) / W) * b[s:e]
        return out, lam

    cells, lams = {}, {}
    for CAD in CADS:
        for dn, F in DESIGNS.items():
            npen = F.shape[1] - nh
            cells[(CAD, dn)], lams[(CAD, dn)] = run(F, npen, CAD)
            print(f"    cadence {CAD:>5}  {dn:>18}  lambda*="
                  f"{lams[(CAD, dn)]:<9g} ({time.time()-t0:.0f}s)", flush=True)

    rv_w = rvv[W:]; good = rv_w > 0
    for k in cells: good &= np.isfinite(cells[k]) & (cells[k] > 0)
    idx = np.flatnonzero(good)
    Q = {k: float(qlike_bar(rv_w[idx], cells[k][idx]).mean()) for k in cells}
    print(f"\n  scored {len(idx):,} bars\n")
    print(f"  {'design':>18}" + "".join(f"{'cad ' + str(c):>12}" for c in CADS)
          + f"{'1000-100':>11}")
    for dn in DESIGNS:
        row = [Q[(c, dn)] for c in CADS]
        print(f"  {dn:>18}" + "".join(f"{v:>12.5f}" for v in row)
              + f"{row[0] - row[-1]:>+11.5f}")
    print(f"\n  ordering by cadence (best first):")
    orders = {}
    for c in CADS:
        o = sorted(DESIGNS, key=lambda d: Q[(c, d)])
        orders[c] = o
        print(f"    cad {c:>5}: {' < '.join(o)}")
    stable = len({tuple(o) for o in orders.values()}) == 1
    print(f"\n  ordering stable across cadence: {stable}")
    eff = [Q[(CADS[0], d)] - Q[(CADS[-1], d)] for d in DESIGNS]
    print(f"  cadence effect per design: "
          f"{ {d: round(Q[(CADS[0], d)] - Q[(CADS[-1], d)], 5) for d in DESIGNS} }")
    print(f"  spread of the cadence effect across designs: "
          f"{max(eff) - min(eff):+.5f}")
    if stable:
        print("  -> cadence moves levels, not orderings; this session's")
        print("     within-panel comparisons stand, and the protocol needs")
        print("     stating rather than the results revisiting")
    else:
        print("  -> cadence CHANGES the ordering, so design rankings reported")
        print("     in this session are artifacts of a parameter never varied")
    out = {"cache": CACHE, "n": int(len(idx)), "cadences": CADS,
           "cols": {k: int(v.shape[1]) for k, v in DESIGNS.items()},
           "qlike": {f"{c}|{d}": Q[(c, d)] for c in CADS for d in DESIGNS},
           "lambda": {f"{c}|{d}": float(lams[(c, d)]) for c in CADS for d in DESIGNS},
           "ordering": {str(c): orders[c] for c in CADS},
           "ordering_stable": bool(stable),
           "cadence_effect_spread": float(max(eff) - min(eff))}
    with open(os.path.join(OUT_DIR, "cadence_x_design.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote cadence_x_design.json ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
