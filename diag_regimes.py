import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.mixture import GaussianMixture
from diag_spectral_graph import (TRAIN_WIN, W, delay_views, prepare_series, represent,
                                 view_index, knn_distances, build_adjacency,
                                 normalized_laplacian_basis)
from diag_altdata import build_blocks

STATE_COLS = ["lvix_ma48", "ts_slope_ma48", "vvix_r_ma48", "vrp_ma48",
              "jumpfrac_ma48", "imb_ma48", "sent_ma48", "breadth_ma48",
              "turnover_z", "attention_z", "vd_active_ma48"]

def run_window(X, y, df, ts, end_date):
    t_end = int(np.searchsorted(ts.values, np.datetime64(end_date)))
    t0 = t_end - TRAIN_WIN
    y_win = y[t0:t_end]; df_win = df.iloc[t0:t_end].reset_index(drop=True)
    idx = view_index(len(y_win), 2)
    targets = y_win[idx]
    Z_ms = represent(delay_views(y_win, idx, W), "multiscale").astype(np.float64)
    blocks = build_blocks(df_win)
    S_full = pd.concat(blocks.values(), axis=1)
    S = np.nan_to_num(S_full[STATE_COLS].to_numpy(dtype=np.float64)[idx])
    gates_clock = df_win[["is_open", "is_close"]].to_numpy(dtype=np.float64)[idx]
    ts_slope = S_full["ts_slope_ma48"].to_numpy(dtype=np.float64)[idx]

    split = int(0.8 * len(idx))
    t_tr, t_te = targets[:split], targets[split:]
    dec = np.clip((np.argsort(np.argsort(t_te)) * 10) // len(t_te), 0, 9)
    mu, sd = S[:split].mean(0), S[:split].std(0); sd[sd < 1e-12] = 1
    Sz = (S - mu) / sd

    # regime discovery on TRAIN states only
    d_, i_ = knn_distances(Sz[:split].astype(np.float32), 10)
    A = build_adjacency(d_, i_, split, "heat")
    lam, U = normalized_laplacian_basis(A, 8)
    gaps = np.diff(lam[:6]); K = int(np.argmax(gaps[1:5]) + 2)
    km = KMeans(n_clusters=K, n_init=10, random_state=42).fit(U[:, 1:K])
    lab = km.labels_
    runs = np.diff(np.flatnonzero(np.r_[1, np.diff(lab) != 0, 1]))
    logit = LogisticRegression(max_iter=2000).fit(Sz[:split], lab)
    P = logit.predict_proba(Sz)          # causal OOS memberships
    acc = logit.score(Sz[:split], lab)
    print(f"  K={K} (eigengap {gaps[:5].round(3)}), median run={np.median(runs)*2:.0f} bars, "
          f"distill acc={acc:.2f}, sizes={np.bincount(lab)}")
    gmm = GaussianMixture(n_components=K, random_state=42, n_init=3).fit(Sz[:split])
    Pg = gmm.predict_proba(Sz)

    def gated(base, gates):   # products of every ms coord with every gate col
        prods = [Z_ms * gates[:, [k]] for k in range(gates.shape[1])]
        return np.concatenate([base, gates] + prods, axis=1)

    def ev(name, M):
        m2, s2 = M[:split].mean(0), M[:split].std(0); s2[s2 < 1e-12] = 1
        Mz = (M - m2) / s2
        p = Ridge(alpha=1.0).fit(Mz[:split], t_tr).predict(Mz[split:])
        top = dec == 9
        print(f"  {name:34s} mse={np.mean((t_te-p)**2):.5f}  top={np.mean((t_te[top]-p[top])**2):.5f}")

    ev("M0  multiscale", Z_ms)
    ev("M0c + clock gates", gated(Z_ms, gates_clock))
    inv = (ts_slope < 0).astype(float)[:, None]
    ev("M1  M0c + VIX-inversion gate", gated(Z_ms, np.concatenate([gates_clock, inv], 1)))
    ev("M2  M0c + GMM regime gates", gated(Z_ms, np.concatenate([gates_clock, Pg[:, :-1]], 1)))
    ev("M3  M0c + spectral regime gates", gated(Z_ms, np.concatenate([gates_clock, P[:, :-1]], 1)))

X, y, dates, df = prepare_series("data", return_df=True)
ts = pd.to_datetime(dates)
for end in ("2020-07-01", "2019-07-01"):
    print(f"\n=== window ending {end} ===")
    run_window(X, y, df, ts, end)
