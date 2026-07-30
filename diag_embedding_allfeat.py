import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import spectral_embedding as sk_se
from src.backtest.executor import _build_har_and_calendar, _build_scale_guards, load_and_transform
from src.data.loading import get_bucket
from src.features.extractors.har import resolve_har_lags
from src.features.transforms.scaling import rolling_robust_scale
from diag_spectral_graph import (TRAIN_WIN, W, knn_distances, build_adjacency,
                                 normalized_laplacian_basis, energy_profile, view_index)

df, adj_exog = load_and_transform("data", get_bucket("all_features"),
    target_use_diurnal=True, target_winsor_window=240, dropna_with_exog=True)
df, feats = _build_har_and_calendar(df, adj_exog, add_calendar=True)
df = df.iloc[resolve_har_lags()[-1]:].reset_index(drop=True)
X = df[feats].values.astype(np.float64)
y = df["adj_RV"].values.astype(np.float64)
ref_iqr, fixed = _build_scale_guards(X, df, feats)
X = rolling_robust_scale(X, TRAIN_WIN, ref_iqr=ref_iqr, fixed_cols=fixed)
ts = pd.to_datetime(df["t"])
print(f"{len(feats)} features, sample {ts.iloc[0].date()}..{ts.iloc[-1].date()}")

t_end = int(np.searchsorted(ts.values, np.datetime64("2015-09-25")))
t0 = t_end - TRAIN_WIN
assert t0 >= 0, f"window too early: t_end={t_end}"
X_win, y_win = X[t0:t_end], y[t0:t_end]
on = df["is_overnight"].values[t0:t_end]
idx = view_index(len(y_win), 2)
Z = X_win[idx].astype(np.float32); targets = y_win[idx]; onb = on[idx].astype(bool)
print(f"window {ts.iloc[t0].date()}..{ts.iloc[t_end-1].date()}, {Z.shape} views")

d_, i_ = knn_distances(Z, 10)
A = build_adjacency(d_, i_, Z.shape[0], "binary")
lam, U = normalized_laplacian_basis(A, 64)
prof = energy_profile(U, targets)
rng = np.random.default_rng(42)
null = np.mean([energy_profile(U, rng.permutation(targets))[7] for _ in range(10)])
print(f"d8 captured {prof[7]:.3%} (null {null:.3%}), d16 {prof[15]:.3%}, d64 {prof[-1]:.3%}")

E = sk_se(A, n_components=2, drop_first=True, random_state=42)
SURFACE="#fcfcfb"; INK="#0b0b0b"; INK2="#52514e"; MUTED="#898781"; BASE="#c3c2b7"
BLUE="#2a78d6"; ORANGE="#eb6834"
plt.rcParams.update({"figure.facecolor":SURFACE,"axes.facecolor":SURFACE,
    "savefig.facecolor":SURFACE,"font.size":10,"text.color":INK,
    "axes.labelcolor":INK2,"xtick.color":MUTED,"ytick.color":MUTED})
fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.9), dpi=150)
axes[0].scatter(E[~onb,0], E[~onb,1], s=2.5, c=BLUE, alpha=.45, lw=0, rasterized=True, label="RTH")
axes[0].scatter(E[onb,0], E[onb,1], s=2.5, c=ORANGE, alpha=.45, lw=0, rasterized=True, label="overnight")
axes[0].set_title("colored by session", loc="left", fontsize=10.5, color=INK)
axes[0].legend(frameon=False, fontsize=9, markerscale=4)
clip = np.percentile(targets, [5, 95])
sc = axes[1].scatter(E[:,0], E[:,1], s=2.5, c=np.clip(targets, *clip),
    cmap=matplotlib.colors.LinearSegmentedColormap.from_list("seq", ["#cde2fb","#3987e5","#0d366b"]),
    alpha=.55, lw=0, rasterized=True)
axes[1].set_title("colored by next-bar vol", loc="left", fontsize=10.5, color=INK)
fig.colorbar(sc, ax=axes[1], shrink=.8, label="target (adj √RV)")
for ax in axes:
    for s in ("top","right"): ax.spines[s].set_visible(False)
    for s in ("left","bottom"): ax.spines[s].set_color(BASE)
    ax.set_xlabel("φ1"); ax.set_ylabel("φ2", rotation=0, labelpad=10)
fig.suptitle(f"Laplacian eigenmap of the all_features bucket ({len(feats)} cols, k=10 binary graph)",
             x=0.02, ha="left", fontsize=12.5, color=INK)
fig.tight_layout(rect=(0,0,1,0.94))
fig.savefig("writeup/figures/diag_embedding_allfeatures.png", bbox_inches="tight")
print("wrote writeup/figures/diag_embedding_allfeatures.png")
