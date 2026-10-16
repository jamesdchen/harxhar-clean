"""Path-dependent TreeSHAP main effects and SHAP interaction values in numba (no shap dependency).

A port of the shap package's C++ ``tree_shap_recursive`` (Lundberg, Erion and Lee 2018, Algorithm 2, with the
conditioning used for the interaction values of Lundberg et al. 2020, Nature MI 2:56-67):

    phi_j            = TreeSHAP of feature j (path-dependent: node covers stand in for the background),
    Phi_{jk}, j != k = (phi_k | j on  -  phi_k | j off) / 2,
    Phi_{jj}         = phi_j - sum_{k != j} Phi_{jk},

so every row of Phi sums to phi_j and sum_{j,k} Phi_{jk} = f(x) - E f.  Conditioning on j only changes the leaves
whose root path splits on j, so a subtree that neither contains a split on j nor lies below one is skipped in both
the 'on' and the 'off' pass (its two contributions are equal and cancel in the difference): exact, and the cost of
the interactions drops from (#features of the tree) x (one TreeSHAP pass) to roughly (path length) x (one pass).

Trees come from LightGBM (dump_model), XGBoost (the JSON model) or scikit-learn (tree_) in one flat layout:
left / right child (-1 at a leaf), split feature, threshold, leaf value, cover (LightGBM internal / leaf counts,
XGBoost sum_hessian, scikit-learn weighted_n_node_samples: what each library's own TreeSHAP uses).  Decision:
x <= threshold goes left (LightGBM, scikit-learn on float32 inputs), x < threshold goes left (XGBoost on float32
inputs).  The design has no missing values (asserted by the caller), so default directions are not needed.
Validation (experiments/rehaul_fi_compute.py gates): main effects vs LightGBM pred_contrib and XGBoost
pred_contribs, the interaction matrices vs XGBoost pred_interactions, the forest vs the stored shap 0.51 values.
"""

from __future__ import annotations

import json

import numpy as np
from numba import njit, prange


# ============================================================================ tree containers
class Forest:
    """All trees of a model in flat arrays; tree t owns nodes [off[t], off[t + 1])."""

    def __init__(
        self,
        trees: list[dict],
        strict: bool,
        scale: float,
        base: float,
        n_features: int,
    ):
        self.n_trees = len(trees)
        self.off = np.zeros(len(trees) + 1, np.int64)
        for t, tr in enumerate(trees):
            self.off[t + 1] = self.off[t] + len(tr["left"])

        def cat(key: str, dt) -> np.ndarray:
            return np.ascontiguousarray(
                np.concatenate([tr[key] for tr in trees]).astype(dt)
            )

        self.left = cat("left", np.int64)
        self.right = cat("right", np.int64)
        self.feat = cat("feat", np.int64)
        self.thr = cat("thr", np.float64)
        self.val = cat("val", np.float64) * scale
        self.cover = cat("cover", np.float64)
        self.depth = np.array([tr["depth"] for tr in trees], np.int64)
        self.strict = bool(strict)
        self.base = float(base)
        self.p = int(n_features)
        self.nw = (self.p + 63) // 64
        self.bits = subtree_bits(self.left, self.right, self.feat, self.off, self.nw)
        self.expected = self.base + float(
            sum(expected_tree(self, t) for t in range(self.n_trees))
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (
            _predict(
                self.left,
                self.right,
                self.feat,
                self.thr,
                self.val,
                self.off,
                self.strict,
                np.ascontiguousarray(X, np.float64),
            )
            + self.base
        )


def _depth(left, right, root=0) -> int:
    best, stack = 0, [(root, 0)]
    while stack:
        n, dd = stack.pop()
        if left[n] < 0:
            best = max(best, dd)
        else:
            stack.append((left[n], dd + 1))
            stack.append((right[n], dd + 1))
    return best


def expected_tree(F: Forest, t: int) -> float:
    a, b = F.off[t], F.off[t + 1]
    left, cover, val = F.left[a:b], F.cover[a:b], F.val[a:b]
    leaf = left < 0
    return float((cover[leaf] * val[leaf]).sum() / cover[0])


def from_lightgbm(booster, p: int) -> Forest:
    """LightGBM booster -> Forest (x <= threshold goes left; covers = internal_count / leaf_count)."""
    dump = booster.dump_model()
    trees = []
    for ti in dump["tree_info"]:
        nodes: list = []

        def walk(nd):
            idx = len(nodes)
            nodes.append(None)
            if "leaf_index" in nd or "split_index" not in nd:
                nodes[idx] = (
                    -1,
                    -1,
                    -1,
                    0.0,
                    float(nd["leaf_value"]),
                    float(nd.get("leaf_count", 0)),
                )
                return idx
            assert nd["decision_type"] == "<=", nd["decision_type"]
            assert nd.get("missing_type", "None") in ("None", "NaN"), nd.get(
                "missing_type"
            )
            li = walk(nd["left_child"])
            ri = walk(nd["right_child"])
            nodes[idx] = (
                li,
                ri,
                int(nd["split_feature"]),
                float(nd["threshold"]),
                0.0,
                float(nd["internal_count"]),
            )
            return idx

        walk(ti["tree_structure"])
        arr = np.array(nodes, dtype=object)
        left = np.array([n[0] for n in nodes], np.int64)
        right = np.array([n[1] for n in nodes], np.int64)
        trees.append(
            dict(
                left=left,
                right=right,
                feat=np.array([n[2] for n in nodes]),
                thr=np.array([n[3] for n in nodes]),
                val=np.array([n[4] for n in nodes]),
                cover=np.array([n[5] for n in nodes]),
                depth=_depth(left, right),
            )
        )
        del arr
    return Forest(trees, strict=False, scale=1.0, base=0.0, n_features=p)


def from_xgboost(booster, p: int) -> Forest:
    """XGBoost booster -> Forest from its JSON model (x < split goes left on float32 inputs; covers =
    sum_hessian; leaf values = split_conditions at the leaves; base_score added)."""
    js = json.loads(booster.save_raw("json").decode())
    lr = js["learner"]
    bs = lr["learner_model_param"]["base_score"]
    base = float(str(bs).strip("[]"))
    obj = lr["objective"]["name"]
    assert obj == "reg:squarederror", obj
    trees = []
    for tr in lr["gradient_booster"]["model"]["trees"]:
        left = np.array(tr["left_children"], np.int64)
        right = np.array(tr["right_children"], np.int64)
        left[left < 0] = -1
        right[left < 0] = -1
        # XGBoost stores split values and leaf weights as float32 and prints the shortest round-trip decimal; parse
        # back to the exact float32 value, else x == split compares differently from the library
        cond = (
            np.array(tr["split_conditions"], np.float64)
            .astype(np.float32)
            .astype(np.float64)
        )
        leaf = left < 0
        trees.append(
            dict(
                left=left,
                right=right,
                feat=np.where(leaf, -1, np.array(tr["split_indices"], np.int64)),
                thr=np.where(leaf, 0.0, cond),
                val=np.where(leaf, cond, 0.0),
                cover=np.array(tr["sum_hessian"], np.float64),
                depth=_depth(left, right),
            )
        )
    return Forest(trees, strict=True, scale=1.0, base=base, n_features=p)


def from_sklearn_forest(model, p: int) -> Forest:
    """scikit-learn RandomForestRegressor -> Forest (x <= threshold on float32 inputs; covers =
    weighted_n_node_samples; the forest is the mean of its trees)."""
    trees = []
    for est in model.estimators_:
        t = est.tree_
        left = t.children_left.astype(np.int64)
        right = t.children_right.astype(np.int64)
        leaf = left < 0
        trees.append(
            dict(
                left=np.where(leaf, -1, left),
                right=np.where(leaf, -1, right),
                feat=np.where(leaf, -1, t.feature.astype(np.int64)),
                thr=np.where(leaf, 0.0, t.threshold.astype(np.float64)),
                val=np.asarray(t.value, np.float64).reshape(len(left), -1)[:, 0],
                cover=t.weighted_n_node_samples.astype(np.float64),
                depth=_depth(left, right),
            )
        )
    return Forest(trees, strict=False, scale=1.0 / len(trees), base=0.0, n_features=p)


def inputs_for(kind: str, X: np.ndarray) -> np.ndarray:
    """The values each library compares with its thresholds: float32 for XGBoost and scikit-learn."""
    X = np.asarray(X, np.float64)
    if kind in ("xgb", "rf"):
        return np.ascontiguousarray(X.astype(np.float32).astype(np.float64))
    return np.ascontiguousarray(X)


# ============================================================================ numba kernels
@njit(cache=True)
def _predict(left, right, feat, thr, val, off, strict, X):
    n = X.shape[0]
    out = np.zeros(n)
    for i in range(n):
        s = 0.0
        for t in range(len(off) - 1):
            nd = off[t]
            base = off[t]
            while left[nd] >= 0:
                xv = X[i, feat[nd]]
                go_left = (xv < thr[nd]) if strict else (xv <= thr[nd])
                nd = base + (left[nd] if go_left else right[nd])
            s += val[nd]
        out[i] = s
    return out


def subtree_bits(left, right, feat, off, nw):
    """Per node, a bitset of the features split on in its subtree (itself included)."""
    bits = np.zeros((len(left), nw), np.uint64)
    _subtree_bits(left, right, feat, off, bits)
    return bits


@njit(cache=True)
def _subtree_bits(left, right, feat, off, bits):
    for t in range(len(off) - 1):
        a, b = off[t], off[t + 1]
        # children have larger indices than parents in all three libraries' layouts?  Not guaranteed (XGBoost
        # is breadth-first, LightGBM / scikit-learn depth-first); a post-order pass over an explicit stack is.
        stack = np.empty(2 * (b - a) + 2, np.int64)
        state = np.zeros(b - a, np.int64)
        sp = 0
        stack[sp] = 0
        sp += 1
        while sp > 0:
            n = stack[sp - 1]
            g = a + n
            if left[g] < 0:
                sp -= 1
                continue
            if state[n] == 0:
                state[n] = 1
                stack[sp] = left[g]
                sp += 1
                stack[sp] = right[g]
                sp += 1
                continue
            sp -= 1
            f = feat[g]
            bits[g, f >> 6] |= np.uint64(1) << np.uint64(f & 63)
            for w in range(bits.shape[1]):
                bits[g, w] |= bits[a + left[g], w] | bits[a + right[g], w]


@njit(cache=True)
def _extend(pf, pz, po, pw, o, ud, zf, of, fi):
    pf[o + ud] = fi
    pz[o + ud] = zf
    po[o + ud] = of
    pw[o + ud] = 1.0 if ud == 0 else 0.0
    for i in range(ud - 1, -1, -1):
        pw[o + i + 1] += of * pw[o + i] * (i + 1) / (ud + 1)
        pw[o + i] = zf * pw[o + i] * (ud - i) / (ud + 1)


@njit(cache=True)
def _unwind(pf, pz, po, pw, o, ud, pi):
    of = po[o + pi]
    zf = pz[o + pi]
    nop = pw[o + ud]
    for i in range(ud - 1, -1, -1):
        if of != 0.0:
            tmp = pw[o + i]
            pw[o + i] = nop * (ud + 1) / ((i + 1) * of)
            nop = tmp - pw[o + i] * zf * (ud - i) / (ud + 1)
        else:
            pw[o + i] = (pw[o + i] * (ud + 1)) / (zf * (ud - i))
    for i in range(pi, ud):
        pf[o + i] = pf[o + i + 1]
        pz[o + i] = pz[o + i + 1]
        po[o + i] = po[o + i + 1]


@njit(cache=True)
def _unwound_sum(pz, po, pw, o, ud, pi):
    of = po[o + pi]
    zf = pz[o + pi]
    nop = pw[o + ud]
    total = 0.0
    if of != 0.0:
        for i in range(ud - 1, -1, -1):
            tmp = nop * (ud + 1) / ((i + 1) * of)
            total += tmp
            nop = pw[o + i] - tmp * zf * ((ud - i) / (ud + 1))
    else:
        for i in range(ud - 1, -1, -1):
            total += pw[o + i] / (zf * ((ud - i) / (ud + 1)))
    return total


@njit(cache=True)
def _pass(
    a,
    cond,
    cfeat,
    pf,
    pz,
    po,
    pw,
    s_nd,
    s_ud,
    s_po,
    s_zf,
    s_of,
    s_fi,
    s_cf,
    s_seen,
    left,
    right,
    feat,
    thr,
    val,
    cover,
    bits,
    strict,
    x,
    phi,
):
    """shap's tree_shap_recursive on an explicit stack (depth-first, hot child first; numba recursion crashed on
    the forest's depth-20 trees).  a = the tree's first global node; a frame = (local node, unique depth, offset of
    the parent's path segment, zero / one fraction and feature of the edge into the node, condition fraction,
    whether the condition feature is on the path above).  A child only reads its parent's segment
    [o, o + ud + 1], and the hot subtree writes only beyond it, so the cold child, popped after the hot subtree,
    reads it intact (as in the recursion)."""
    sp = 0
    s_nd[0] = 0
    s_ud[0] = 0
    s_po[0] = 0
    s_zf[0] = 1.0
    s_of[0] = 1.0
    s_fi[0] = -1
    s_cf[0] = 1.0
    s_seen[0] = False
    sp = 1
    while sp > 0:
        sp -= 1
        nd = s_nd[sp]
        ud = s_ud[sp]
        po_par = s_po[sp]
        pzf = s_zf[sp]
        pof = s_of[sp]
        pfi = s_fi[sp]
        cfrac = s_cf[sp]
        seen = s_seen[sp]
        if cfrac == 0.0:
            continue
        g = a + nd
        if cond != 0 and not seen:
            if left[g] < 0:
                continue  # a leaf whose path never split on the condition feature: 'on' and 'off' agree
            if (bits[g, cfeat >> 6] >> np.uint64(cfeat & 63)) & np.uint64(
                1
            ) == np.uint64(0):
                continue  # no split on it below either
        o = po_par + ud + 1
        for i in range(ud + 1):
            pf[o + i] = pf[po_par + i]
            pz[o + i] = pz[po_par + i]
            po[o + i] = po[po_par + i]
            pw[o + i] = pw[po_par + i]
        if cond == 0 or cfeat != pfi:
            _extend(pf, pz, po, pw, o, ud, pzf, pof, pfi)
        if left[g] < 0:
            v = val[g]
            for i in range(1, ud + 1):
                w = _unwound_sum(pz, po, pw, o, ud, i)
                phi[pf[o + i]] += w * (po[o + i] - pz[o + i]) * cfrac * v
            continue
        si = feat[g]
        xv = x[si]
        go_left = (xv < thr[g]) if strict else (xv <= thr[g])
        hot = left[g] if go_left else right[g]
        cold = right[g] if go_left else left[g]
        wn = cover[g]
        hzf = cover[a + hot] / wn
        czf = cover[a + cold] / wn
        izf = 1.0
        iof = 1.0
        pi = 0
        while pi <= ud:
            if pf[o + pi] == si:
                break
            pi += 1
        if pi != ud + 1:
            izf = pz[o + pi]
            iof = po[o + pi]
            _unwind(pf, pz, po, pw, o, ud, pi)
            ud -= 1
        hcf = cfrac
        ccf = cfrac
        seen2 = seen
        if cond > 0 and si == cfeat:
            ccf = 0.0
            ud -= 1
            seen2 = True
        elif cond < 0 and si == cfeat:
            hcf *= hzf
            ccf *= czf
            ud -= 1
            seen2 = True
        # push cold first so that the hot child is processed first (the recursion's order)
        s_nd[sp] = cold
        s_ud[sp] = ud + 1
        s_po[sp] = o
        s_zf[sp] = czf * izf
        s_of[sp] = 0.0
        s_fi[sp] = si
        s_cf[sp] = ccf
        s_seen[sp] = seen2
        sp += 1
        s_nd[sp] = hot
        s_ud[sp] = ud + 1
        s_po[sp] = o
        s_zf[sp] = hzf * izf
        s_of[sp] = iof
        s_fi[sp] = si
        s_cf[sp] = hcf
        s_seen[sp] = seen2
        sp += 1


@njit(cache=True)
def _tree_features(feat, a, b, p):
    used = np.zeros(p, np.bool_)
    for g in range(a, b):
        if feat[g] >= 0:
            used[feat[g]] = True
    return np.flatnonzero(used)


@njit(cache=True, parallel=True)
def _shap_rows(
    left, right, feat, thr, val, cover, bits, off, depth, strict, X, p, interactions
):
    """Main effects (n, p) and, if interactions, Phi (n, p, p) per row; trees summed (scale already in val)."""
    n = X.shape[0]
    main = np.zeros((n, p))
    inter = np.zeros((n, p, p)) if interactions else np.zeros((1, 1, 1))
    Dmax = 0
    for t in range(len(depth)):
        Dmax = max(Dmax, depth[t])
    size = (Dmax + 3) * (Dmax + 4) // 2 + 8
    ssz = 2 * (Dmax + 3) + 8
    for i in prange(n):
        x = X[i]
        phi = np.zeros(p)
        on = np.zeros(p)
        offv = np.zeros(p)
        pf = np.full(size, -1, np.int64)
        pz = np.zeros(size)
        po = np.zeros(size)
        pw = np.zeros(size)
        s_nd = np.zeros(ssz, np.int64)
        s_ud = np.zeros(ssz, np.int64)
        s_po = np.zeros(ssz, np.int64)
        s_zf = np.zeros(ssz)
        s_of = np.zeros(ssz)
        s_fi = np.zeros(ssz, np.int64)
        s_cf = np.zeros(ssz)
        s_seen = np.zeros(ssz, np.bool_)
        for t in range(len(off) - 1):
            a = off[t]
            phi[:] = 0.0
            _pass(
                a,
                0,
                0,
                pf,
                pz,
                po,
                pw,
                s_nd,
                s_ud,
                s_po,
                s_zf,
                s_of,
                s_fi,
                s_cf,
                s_seen,
                left,
                right,
                feat,
                thr,
                val,
                cover,
                bits,
                strict,
                x,
                phi,
            )
            for j in range(p):
                main[i, j] += phi[j]
            if interactions:
                fs = _tree_features(feat, a, off[t + 1], p)
                for q in range(len(fs)):
                    j = fs[q]
                    on[:] = 0.0
                    offv[:] = 0.0
                    _pass(
                        a,
                        1,
                        j,
                        pf,
                        pz,
                        po,
                        pw,
                        s_nd,
                        s_ud,
                        s_po,
                        s_zf,
                        s_of,
                        s_fi,
                        s_cf,
                        s_seen,
                        left,
                        right,
                        feat,
                        thr,
                        val,
                        cover,
                        bits,
                        strict,
                        x,
                        on,
                    )
                    _pass(
                        a,
                        -1,
                        j,
                        pf,
                        pz,
                        po,
                        pw,
                        s_nd,
                        s_ud,
                        s_po,
                        s_zf,
                        s_of,
                        s_fi,
                        s_cf,
                        s_seen,
                        left,
                        right,
                        feat,
                        thr,
                        val,
                        cover,
                        bits,
                        strict,
                        x,
                        offv,
                    )
                    for k in range(p):
                        if k != j:
                            dlt = 0.5 * (on[k] - offv[k])
                            inter[i, j, k] += dlt
                            inter[i, j, j] -= dlt
                for j in range(p):
                    inter[i, j, j] += phi[j]
    return main, inter


def tree_shap(F: Forest, X: np.ndarray, interactions: bool = False):
    """(main (n, p), Phi (n, p, p) or None, expected value)."""
    X = np.ascontiguousarray(X, np.float64)
    main, inter = _shap_rows(
        F.left,
        F.right,
        F.feat,
        F.thr,
        F.val,
        F.cover,
        F.bits,
        F.off,
        F.depth,
        F.strict,
        X,
        F.p,
        interactions,
    )
    return main, (inter if interactions else None), F.expected
