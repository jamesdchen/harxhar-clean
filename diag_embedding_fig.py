import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import spectral_embedding as sk_se
from diag_spectral_graph import (TRAIN_WIN, W, delay_views, prepare_series,
                                 represent, view_index, knn_distances, build_adjacency)

SURFACE="#fcfcfb"; INK="#0b0b0b"; INK2="#52514e"; MUTED="#898781"; BASE="#c3c2b7"
BLUE="#2a78d6"; ORANGE="#eb6834"
plt.rcParams.update({"figure.facecolor":SURFACE,"axes.facecolor":SURFACE,
    "savefig.facecolor":SURFACE,"font.size":10,"text.color":INK,
    "axes.labelcolor":INK2,"xtick.color":MUTED,"ytick.color":MUTED})

X, y, dates, df = prepare_series("data", return_df=True)
n = len(y); t_end = int(TRAIN_WIN + 0.5*(n-TRAIN_WIN)); t0 = t_end-TRAIN_WIN
X_win, y_win = X[t0:t_end], y[t0:t_end]
overnight = df["is_overnight"].values[t0:t_end]
idx = view_index(len(y_win), 2)
targets = y_win[idx]; on = overnight[idx].astype(bool)

def embed(Z):
    d_, i_ = knn_distances(Z, 10)
    A = build_adjacency(d_, i_, Z.shape[0], "binary")
    return sk_se(A, n_components=2, drop_first=True, random_state=42)

E_har = embed(X_win[idx].astype(np.float32))
E_ms  = embed(represent(delay_views(y_win, idx, W), "multiscale"))

fig, axes = plt.subplots(2, 2, figsize=(10.6, 9.2), dpi=150)
clip = np.percentile(targets, [5, 95])
for row, (E, name) in enumerate([(E_har, "har27 (pipeline features)"),
                                 (E_ms, "multiscale-5 (level ladder)")]):
    ax = axes[row][0]
    ax.scatter(E[~on,0], E[~on,1], s=2.5, c=BLUE, alpha=.45, lw=0, rasterized=True, label="RTH")
    ax.scatter(E[on,0], E[on,1], s=2.5, c=ORANGE, alpha=.45, lw=0, rasterized=True, label="overnight")
    ax.set_title(f"{name} — colored by session", loc="left", fontsize=10.5, color=INK)
    leg = ax.legend(frameon=False, fontsize=9, markerscale=4, loc="best")
    ax = axes[row][1]
    sc = ax.scatter(E[:,0], E[:,1], s=2.5, c=np.clip(targets, *clip),
                    cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
                        "seq", ["#cde2fb","#3987e5","#0d366b"]),
                    alpha=.55, lw=0, rasterized=True)
    ax.set_title(f"{name} — colored by next-bar vol", loc="left", fontsize=10.5, color=INK)
    fig.colorbar(sc, ax=ax, shrink=.75, label="target (adj √RV)")
for ax in axes.ravel():
    for s in ("top","right"): ax.spines[s].set_visible(False)
    for s in ("left","bottom"): ax.spines[s].set_color(BASE)
    ax.set_xlabel("φ1"); ax.set_ylabel("φ2", rotation=0, labelpad=10)
fig.suptitle("Laplacian eigenmap (k=10 binary graph, window 2013-11 → 2015-09)",
             x=0.02, ha="left", fontsize=12.5, color=INK)
fig.tight_layout(rect=(0,0,1,0.96))
fig.savefig("writeup/figures/diag_embedding_scatter.png", bbox_inches="tight")
print("wrote writeup/figures/diag_embedding_scatter.png")
