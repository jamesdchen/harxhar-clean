"""Quotient the HAR ladder out of the 16:00 all-features tree importance.

For each refit anchor (every 10 sessions, the 2000 rows before it, the shipped
untuned hyperparameters, the per-window identifiability mask):

  h  = the tree fit on har_ma_1 .. har_ma_3125 only
  e  = y - h(x_har), on that window and on its causal tail
  r  = the same tree fit to e on every other column

The tail loss of h, of h + r, and of the stored full-model refit (the masked
fleet's own forecast) is the leftover, in fit-space squared error and in QLIKE.
MDI share, SHAP share, and P2 MSE permutation of r are the makeup of the
residual fit. QLIKE permutation is not computed: r is a forecast of the
residual, not of variance.

``python experiments/feature_importance_1530_har_quotient.py run [models] [k0] [k1]``
writes results/feature_importance_1530_dedup/har_quotient/. QUOTIENT_THREADS
sets the threads of each fit (default: the machine's CPU count).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ["FEATIMP_OUT"] = "results/feature_importance_1530_dedup"
os.environ["FEATIMP_WINDOW_MASK"] = "1"

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))
import feature_importance_1530_trees as ft  # noqa: E402

OUT = ft.OUT / "har_quotient"
MODELS = ("lgbm", "xgb", "rf")
HAR_LAGS = (1, 5, 25, 125, 625, 3125)


def _har_idx(names: list[str]) -> np.ndarray:
    want = [f"har_ma_{k}" for k in HAR_LAGS]
    idx = [names.index(n) for n in want]
    if idx != sorted(idx):
        raise RuntimeError(f"HAR columns out of order: {idx}")
    return np.asarray(idx, dtype=np.int64)


def _series_units(d: dict, har: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Non-HAR source series and their column indices in the full design."""
    kind = d["unit_kind"].astype(str)
    uname = d["unit_name"].astype(str)
    har_set = set(int(j) for j in har)
    out = []
    for u in np.flatnonzero(kind == "series"):
        name = str(uname[u])
        if name == "har_ma":
            continue
        cols = np.asarray(
            [int(j) for j in d["units"][u] if int(j) not in har_set], dtype=np.int64
        )
        if len(cols):
            out.append((name, cols))
    return out


def _stored_preds(d: dict, model: str) -> list[np.ndarray]:
    """The masked fleet's own tail forecast, one array per refit."""
    by_k: dict[int, np.ndarray] = {}
    root = ft.WORK / "trees"
    for f in sorted(root.glob(f"trees_all_features_{model}_*.npz")):
        a, b = (int(x) for x in f.stem.split("_")[-2:])
        if b - a != 21:
            continue
        z = np.load(f)
        off = 0
        pred = np.asarray(z["pred"], dtype=np.float64)
        for k in np.asarray(z["k"], dtype=int):
            _, _, kk = ft.tail_rows(d, int(k))
            by_k[int(k)] = pred[off : off + kk]
            off += kk
        if off != len(pred):
            raise RuntimeError(f"{f.name}: pred length {len(pred)} != tails {off}")
    k_max = ft.n_refits(d["n_oos"])
    if sorted(by_k) != list(range(k_max)):
        raise RuntimeError(f"{model}: stored refits {sorted(by_k)[:3]}.. != 0..{k_max}")
    return [by_k[k] for k in range(k_max)]


def _fit(model: str, threads: int, Xtr: np.ndarray, ytr: np.ndarray, Xt: np.ndarray):
    """Window-masked fit. Returns (model, keep, train pred, tail pred)."""
    keep = ft.window_cols(Xtr)
    if len(keep) == 0:
        ztr = np.zeros(len(Xtr))
        zt = np.zeros(len(Xt))
        return None, keep, ztr, zt
    m = ft.make_model(model, threads)
    m.fit(ft.sub(Xtr, keep), ytr)
    return m, keep, m.predict(ft.sub(Xtr, keep)), m.predict(ft.sub(Xt, keep))


def _predict(m, keep: np.ndarray, Z: np.ndarray) -> np.ndarray:
    if m is None:
        return np.zeros(len(Z))
    return np.asarray(m.predict(ft.sub(Z, keep)), dtype=np.float64)


def _p2_mse(m, keep, Xt, Xtr, units, p2, e, m0) -> np.ndarray:
    """Mean increase in tail squared error from P2 (training-window rows)."""
    kk = len(Xt)
    U = len(units)
    dm = np.zeros(U)
    todo = [u for u in range(U) if len(units[u])]
    if m is None or not todo:
        return dm
    R = p2.shape[1]
    Xb = np.broadcast_to(Xt, (len(todo) * R, kk, Xt.shape[1])).copy()
    Xb = Xb.reshape(len(todo), R, kk, Xt.shape[1])
    for q, u in enumerate(todo):
        cols = units[u]
        Xb[q][:, :, cols] = Xtr[:, cols][p2[u]]
    yh = _predict(m, keep, Xb.reshape(-1, Xt.shape[1])).reshape(len(todo), R, kk)
    se = ((e[None, None, :] - yh) ** 2).mean(axis=2)
    dm[np.asarray(todo)] = se.mean(axis=1) - m0
    return dm


def _shares(v: np.ndarray) -> np.ndarray:
    tot = float(v.sum())
    if tot <= 0:
        return np.zeros_like(v)
    return v / tot


def _one_refit(model, threads, d, har, rest, series, k, full_pred):
    """One anchor. series is (name, columns-into-rest)."""
    i, t, kk = ft.tail_rows(d, k)
    W = d["W"]
    X, y = d["X"], d["y"]
    Xtr, ytr = X[t - W : t], y[t - W : t]
    Xt, yt = X[t : t + kk], y[t : t + kk]
    rv, base = d["rv"][i : i + kk], d["base"][i : i + kk]

    a = time.time()
    _, keep_h, h_tr, h_te = _fit(model, threads, Xtr[:, har], ytr, Xt[:, har])
    sec_h = time.time() - a
    e_tr = ytr - h_tr
    e_te = yt - h_te

    a = time.time()
    m, keep_r, _, r_te = _fit(model, threads, Xtr[:, rest], e_tr, Xt[:, rest])
    sec_r = time.time() - a
    hr = h_te + r_te
    p_rest = len(rest)

    if m is None:
        gain = np.zeros(p_rest)
        phi = np.zeros((kk, p_rest))
    else:
        gain, _ = ft.native(model, m, len(keep_r))
        gain = ft.spread(gain, keep_r, p_rest)
        sh = ft.spread(
            ft.tree_shap(model, m, ft.sub(Xt[:, rest], keep_r)), keep_r, p_rest
        )
        phi = sh[:, :-1]
        if abs(float(np.max(np.abs(sh.sum(axis=1) - r_te)))) > 1e-4:
            raise RuntimeError(
                f"{model} refit {k}: SHAP does not add to the residual forecast"
            )

    mdi = np.array([gain[cols].sum() for _, cols in series])
    g = np.array([np.abs(phi[:, cols].sum(axis=1)).mean() for _, cols in series])
    rng = np.random.default_rng([ft.PERM_SEED, 7, k])
    p2 = np.floor(rng.random((len(series), ft.N_REPEATS, kk)) * W).astype(np.int64)
    kept = np.zeros(p_rest, dtype=bool)
    kept[keep_r] = True
    units = [cols[kept[cols]] for _, cols in series]
    m0 = float(((e_te - r_te) ** 2).mean())
    dm = _p2_mse(m, keep_r, Xt[:, rest], Xtr[:, rest], units, p2, e_te, m0)

    def mse(a_, b_):
        return float(np.mean((a_ - b_) ** 2))

    def ql(yhat):
        return float(np.mean(ft.qlike(rv, yhat, base)))

    return {
        "k": k,
        "kk": kk,
        "sec_h": sec_h,
        "sec_r": sec_r,
        "n_keep_h": int(len(keep_h)),
        "n_keep_r": int(len(keep_r)),
        "mse_h": mse(yt, h_te),
        "mse_hr": mse(yt, hr),
        "mse_f": mse(yt, full_pred),
        "mse_e0": mse(e_te, np.zeros_like(e_te)),
        "mse_er": m0,
        "qlike_h": ql(h_te),
        "qlike_hr": ql(hr),
        "qlike_f": ql(full_pred),
        "mdi": _shares(mdi),
        "shap": _shares(g),
        "shap_abs": g,
        "perm_p2_mse": dm,
    }


def _payload(rows: list[dict], names: list[str]) -> dict:
    keys = (
        "k",
        "kk",
        "sec_h",
        "sec_r",
        "n_keep_h",
        "n_keep_r",
        "mse_h",
        "mse_hr",
        "mse_f",
        "mse_e0",
        "mse_er",
        "qlike_h",
        "qlike_hr",
        "qlike_f",
    )
    out = {key: np.array([r[key] for r in rows]) for key in keys}
    for key in ("mdi", "shap", "shap_abs", "perm_p2_mse"):
        out[key] = np.vstack([r[key] for r in rows])
    out["series"] = np.array(names)
    return out


def _save(path: Path, rows: list[dict], names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **_payload(rows, names))
    tmp.replace(path)


def _load(path: Path) -> list[dict]:
    z = np.load(path, allow_pickle=True)
    rows = []
    n = len(z["k"])
    for i in range(n):
        rows.append(
            {
                "k": int(z["k"][i]),
                "kk": int(z["kk"][i]),
                "sec_h": float(z["sec_h"][i]),
                "sec_r": float(z["sec_r"][i]),
                "n_keep_h": int(z["n_keep_h"][i]),
                "n_keep_r": int(z["n_keep_r"][i]),
                "mse_h": float(z["mse_h"][i]),
                "mse_hr": float(z["mse_hr"][i]),
                "mse_f": float(z["mse_f"][i]),
                "mse_e0": float(z["mse_e0"][i]),
                "mse_er": float(z["mse_er"][i]),
                "qlike_h": float(z["qlike_h"][i]),
                "qlike_hr": float(z["qlike_hr"][i]),
                "qlike_f": float(z["qlike_f"][i]),
                "mdi": z["mdi"][i],
                "shap": z["shap"][i],
                "shap_abs": z["shap_abs"][i],
                "perm_p2_mse": z["perm_p2_mse"][i],
            }
        )
    return rows


def run(models: list[str], k0: int, k1: int, threads: int) -> None:
    d = ft.load_input("all_features")
    names_all = d["names"]
    har = _har_idx(names_all)
    rest = np.asarray(
        [j for j in range(len(names_all)) if j not in set(har)], dtype=np.int64
    )
    series_full = _series_units(d, har)
    # column indices into the residual matrix (rest), not the full design
    pos = {int(j): i for i, j in enumerate(rest)}
    series = [
        (name, np.asarray([pos[int(j)] for j in cols], dtype=np.int64))
        for name, cols in series_full
    ]
    snames = [name for name, _ in series]
    K = ft.n_refits(d["n_oos"])
    k1 = min(k1, K)
    print(
        f"HAR quotient: columns {len(names_all)}, HAR {len(har)}, other {len(rest)}, "
        f"series {len(series)}, refits {k0}..{k1 - 1}, threads {threads}",
        flush=True,
    )
    for model in models:
        if model not in MODELS:
            raise SystemExit(f"unknown model {model}")
        final = OUT / f"har_quotient_{model}_{k0:03d}_{k1:03d}.npz"
        ckpt = OUT / f"_ckpt_{model}_{k0:03d}_{k1:03d}.npz"
        stored = _stored_preds(d, model)
        rows = _load(ckpt) if ckpt.exists() else []
        done = {r["k"] for r in rows}
        t0 = time.time()
        for k in range(k0, k1):
            if k in done:
                continue
            row = _one_refit(model, threads, d, har, rest, series, k, stored[k])
            rows.append(row)
            if len(rows) % 10 == 0 or k == k1 - 1:
                _save(ckpt, rows, snames)
            print(
                f"  {model} refit {k}: HAR {row['sec_h']:.1f}s residual {row['sec_r']:.1f}s "
                f"keep {row['n_keep_h']}/{row['n_keep_r']}; "
                f"MSE h {row['mse_h']:.4f} h+r {row['mse_hr']:.4f} full {row['mse_f']:.4f}; "
                f"elapsed {time.time() - t0:.0f}s",
                flush=True,
            )
        rows.sort(key=lambda r: r["k"])
        _save(final, rows, snames)
        if ckpt.exists():
            ckpt.unlink()
        print(
            f"wrote {final} ({len(rows)} refits, {time.time() - t0:.0f}s)", flush=True
        )


def _wmean(v: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(w * v) / np.sum(w))


def _rank_desc(v: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    return rankdata(-v, method="min").astype(int)


def _years(d: dict) -> np.ndarray:
    K = ft.n_refits(d["n_oos"])
    years = np.zeros(K, dtype=int)
    for k in range(K):
        i, _, _ = ft.tail_rows(d, k)
        years[k] = int(str(d["date_oos"][i])[:4])
    return years


def summarize() -> None:
    import pandas as pd

    d = ft.load_input("all_features")
    years = _years(d)
    frames = []
    loss_rows = []
    for model in MODELS:
        hits = [
            p for p in OUT.glob(f"har_quotient_{model}_*.npz") if "_ckpt" not in p.name
        ]
        if not hits:
            print(f"no file for {model}", flush=True)
            continue
        z = max(
            (np.load(p, allow_pickle=True) for p in hits), key=lambda a: len(a["k"])
        )
        snames = [str(s) for s in z["series"]]
        kk = np.asarray(z["kk"], dtype=float)
        ks = np.asarray(z["k"], dtype=int)
        for key in (
            "mse_h",
            "mse_hr",
            "mse_f",
            "mse_e0",
            "mse_er",
            "qlike_h",
            "qlike_hr",
            "qlike_f",
        ):
            loss_rows.append(
                {
                    "model": model,
                    "quantity": key,
                    "wmean": _wmean(np.asarray(z[key], float), kk),
                    "median": float(np.median(z[key])),
                }
            )
        for measure in ("mdi", "shap", "shap_abs", "perm_p2_mse"):
            V = np.asarray(z[measure], float)
            wmean = (kk[:, None] * V).sum(axis=0) / kk.sum()
            ranks = np.vstack([_rank_desc(V[i]) for i in range(len(V))])
            mean_rank = (kk[:, None] * ranks).sum(axis=0) / kk.sum()
            for j, name in enumerate(snames):
                rec = {
                    "model": model,
                    "measure": measure,
                    "unit": name,
                    "mean_over_time": wmean[j],
                    "rank_of_mean": int(_rank_desc(wmean)[j]),
                    "mean_rank": mean_rank[j],
                    "frac_rank1": _wmean((ranks[:, j] == 1).astype(float), kk),
                }
                for year in sorted(set(years[ks].tolist())):
                    m = years[ks] == year
                    rec[f"y{year}"] = _wmean(V[m, j], kk[m]) if m.any() else np.nan
                frames.append(rec)
        # adjacent-refit rank correlation of the series vector
        from scipy.stats import spearmanr

        for measure in ("mdi", "shap", "perm_p2_mse"):
            V = np.asarray(z[measure], float)
            rho = []
            for i in range(len(V) - 1):
                if np.std(V[i]) == 0 or np.std(V[i + 1]) == 0:
                    continue
                rho.append(float(spearmanr(V[i], V[i + 1]).statistic))
            loss_rows.append(
                {
                    "model": model,
                    "quantity": f"spearman_{measure}",
                    "wmean": float(np.median(rho)) if rho else np.nan,
                    "median": float(np.min(rho)) if rho else np.nan,
                }
            )
    if not frames:
        print("nothing to summarize", flush=True)
        return
    summary = pd.DataFrame(frames)
    loss = pd.DataFrame(loss_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "series_summary.csv", index=False, float_format="%.6g")
    loss.to_csv(OUT / "loss_summary.csv", index=False, float_format="%.6g")
    print(loss.to_string(index=False), flush=True)
    for model in MODELS:
        sub = summary[summary.model == model]
        if sub.empty:
            continue
        print(
            f"\n{model}: top series by mean MDI share / SHAP share / P2 dMSE",
            flush=True,
        )
        for measure in ("mdi", "shap", "perm_p2_mse"):
            t = (
                sub[sub.measure == measure]
                .sort_values("mean_over_time", ascending=False)
                .head(8)
            )
            cells = [
                f"{r.unit} {r.mean_over_time:.4g} (rank {int(r.rank_of_mean)}, "
                f"mean rank {r.mean_rank:.2f}, rank1 {r.frac_rank1:.2f})"
                for r in t.itertuples()
            ]
            print(f"  {measure}: " + "; ".join(cells), flush=True)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    if stage == "run":
        models = sys.argv[2].split(",") if len(sys.argv) > 2 else list(MODELS)
        k0 = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        k1 = int(sys.argv[4]) if len(sys.argv) > 4 else 10**9
        threads = int(os.environ.get("QUOTIENT_THREADS", str(os.cpu_count() or 4)))
        run(models, k0, k1, threads)
    elif stage == "summarize":
        summarize()
    else:
        raise SystemExit(f"unknown stage {stage!r} (run | summarize)")
