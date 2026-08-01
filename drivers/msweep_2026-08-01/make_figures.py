# ruff: noqa: E402, E741
"""Figures: operator shapes, loadings, target kernel, tshape SVD, session geometry."""
import numpy as np
import re
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID = "#1a1a19", "#5f5e56", "#e8e8e4"
DIV = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f4f4f2", "#eb6834"])
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "font.size": 9,
    "axes.titlesize": 10, "axes.spines.top": False, "axes.spines.right": False})

D = "results/b2_mmap"
START, TRAIN_WIN, n1 = 24000, 24000, 26189
X = np.asarray(np.load(f"{D}/X.npy", mmap_mode="r")[:n1 + 300])
y = np.load(f"{D}/y.npy")
names = [str(s) for s in np.load(f"{D}/names.npy", allow_pickle=True)]
chan = {}
for j, nm in enumerate(names):
    m = re.match(r"^adj_(.+)_ma_(\d+)$", nm)
    if m:
        chan.setdefault(m.group(1), []).append((int(m.group(2)), j))
for c in chan:
    chan[c].sort()
labels = list(chan)
rungs = np.array([r for r, _ in chan[labels[0]]], float)
ladder_cols = [j for c in chan for _, j in chan[c]]
other_cols = [j for j in range(len(names)) if j not in set(ladder_cols)]
O = X[:, other_cols]
har_idx = [oi for oi, j in enumerate(other_cols) if re.match(r"^har_ma_\d+$", names[j])]


def ridge_fit(A, yy):
    G = A.T @ A + np.eye(A.shape[1])
    G[0, 0] -= 1.0
    return np.linalg.solve(G, A.T @ yy)


Kbar, Khist = None, []
for seg in range(0, n1 - START, 240):
    t = START + seg
    full = slice(t - TRAIN_WIN, t)
    A0 = np.hstack([np.ones((TRAIN_WIN, 1)), O[full], X[full][:, ladder_cols]])
    cff = ridge_fit(A0, y[full])
    Km = cff[1 + O.shape[1]:].reshape(len(labels), -1)
    Kbar = Km if Kbar is None else 0.9 * Kbar + 0.1 * Km
    Khist.append(cff[1:][har_idx])
Khist = np.array(Khist)
U_, sv, Vt = np.linalg.svd(Kbar, full_matrices=False)
sgn = np.sign(Vt[np.arange(len(Vt)), np.abs(Vt).argmax(axis=1)])
S = (Vt * sgn[:, None])[:5]
U5 = (U_ * sgn[None, :len(U_)])[:, :5]


def eff_kernel(w):
    return np.array([sum(w[k] / rungs[k] for k in range(len(rungs)) if rungs[k] >= u)
                     for u in rungs])


QUESTIONS = ["slow vol-complex curvature", "tails vs priced", "quarter vs fortnight",
             "day vs week band", "participation surprise"]

# FIG 1
fig, axes = plt.subplots(5, 2, figsize=(9, 11), sharex=True)
for k in range(5):
    pairs = [(S[k], "boxcar coefficients"),
             (eff_kernel(S[k]), "effective lag kernel  phi(u) = sum_{r>=u} w_r / r")]
    for j, (vec, ttl) in enumerate(pairs):
        ax = axes[k, j]
        ax.axhline(0, color=INK2, lw=0.8)
        ax.plot(rungs, vec, color=C[k], lw=2, marker="o", ms=4)
        ax.set_xscale("log", base=2)
        if k == 0:
            ax.set_title(ttl, color=INK2)
        if j == 0:
            ax.set_ylabel(f"shape {k + 1}\n{QUESTIONS[k]}", fontsize=8)
    axes[k, 0].text(0.02, 0.9, f"sigma={sv[k]:.2f}", transform=axes[k, 0].transAxes,
                    fontsize=8, color=INK2)
axes[-1, 0].set_xlabel("lag (bars, log-2)")
axes[-1, 1].set_xlabel("lag (bars, log-2)")
fig.suptitle("The five lag profiles (pooled operator SVD, tile 1)", y=0.995)
fig.tight_layout()
fig.savefig("figs_op/fig1_shapes.png", dpi=150)
plt.close(fig)

# FIG 2
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 8), gridspec_kw={"width_ratios": [1, 2.6]})
a1.barh(np.arange(len(sv))[::-1], sv, color="#9ec5f4", height=0.7)
for i in range(5):
    a1.barh(len(sv) - 1 - i, sv[i], color=C[i], height=0.7)
a1.set_yticks(np.arange(len(sv))[::-1])
a1.set_yticklabels([f"{i + 1}" for i in range(len(sv))])
a1.set_title("singular values")
a1.set_ylabel("component")
a1.grid(axis="y", visible=False)
order = np.argsort(np.abs(U5).argmax(axis=1) * 100 - np.abs(U5).max(axis=1))
im = a2.imshow(U5[order], aspect="auto", cmap=DIV, vmin=-0.6, vmax=0.6)
a2.set_yticks(range(len(labels)))
a2.set_yticklabels([labels[i] for i in order], fontsize=6.5)
a2.set_xticks(range(5))
a2.set_xticklabels([f"s{k + 1}" for k in range(5)])
a2.set_title("channel loadings u_k (blue = negative, orange = positive)")
a2.grid(visible=False)
fig.colorbar(im, ax=a2, shrink=0.5)
fig.suptitle("Which channels ask which question", y=0.995)
fig.tight_layout()
fig.savefig("figs_op/fig2_loadings.png", dpi=150)
plt.close(fig)

# FIG 3
ktar = Khist.mean(axis=0)
phi = eff_kernel(ktar)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
a1.axhline(0, color=INK2, lw=0.8)
a1.plot(rungs, ktar, color=C[0], lw=2, marker="o", ms=4)
a1.set_xscale("log", base=2)
a1.set_title("self-kernel: boxcar coefficients (mean over refits)")
a1.set_xlabel("rung (bars)")
pos = phi > 0
a2.plot(rungs[pos], phi[pos], color=C[0], lw=2, marker="o", ms=5,
        label="effective kernel (positive part)")
beta = 1.36
pl = rungs ** (-beta)
pl *= phi[0] / pl[0]
a2.plot(rungs, pl, color=C[1], lw=2, ls="--", label=f"u^-{beta} (fitted power law)")
a2.set_xscale("log", base=2)
a2.set_yscale("log")
a2.set_title("effective kernel vs power law (log-log)")
a2.set_xlabel("lag (bars)")
a2.legend(frameon=False, fontsize=8)
fig.suptitle("HAR as quadrature: the target self-kernel is power-law-like", y=1.0)
fig.tight_layout()
fig.savefig("figs_op/fig3_target_kernel.png", dpi=150)
plt.close(fig)

# FIG 4
u2, sv2, vt2 = np.linalg.svd(Khist, full_matrices=False)
sg2 = np.sign(vt2[np.arange(len(vt2)), np.abs(vt2).argmax(axis=1)])
M = vt2 * sg2[:, None]
amps = u2 * sg2[None, :len(u2)] * sv2[None, :len(u2)]
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
vmax = np.abs(Khist).max()
im = axes[0, 0].imshow(Khist, aspect="auto", cmap=DIV, vmin=-vmax, vmax=vmax)
axes[0, 0].set_title("K_hist: self-kernel across refits")
axes[0, 0].set_xlabel("rung index")
axes[0, 0].set_ylabel("refit (240-bar cadence)")
axes[0, 0].grid(visible=False)
fig.colorbar(im, ax=axes[0, 0], shrink=0.8)
for k in range(3):
    axes[0, 1].plot(rungs, M[k], color=C[k], lw=2, marker="o", ms=3,
                    label=f"mode {k + 1} (sigma={sv2[k]:.2f})")
axes[0, 1].axhline(0, color=INK2, lw=0.8)
axes[0, 1].set_xscale("log", base=2)
axes[0, 1].set_title("temporal modes of the self-kernel")
axes[0, 1].legend(frameon=False, fontsize=8)
axes[1, 0].bar(range(1, len(sv2) + 1), sv2,
               color=["#2a78d6", "#eb6834", "#1baf7a"] + ["#9ec5f4"] * 9, width=0.7)
axes[1, 0].set_title("tshape singular values")
axes[1, 0].set_xlabel("mode")
for k in range(3):
    axes[1, 1].plot(range(1, len(Khist) + 1), amps[:, k], color=C[k], lw=2, marker="o", ms=4)
axes[1, 1].axhline(0, color=INK2, lw=0.8)
axes[1, 1].set_title("mode amplitudes across refits")
axes[1, 1].set_xlabel("refit #")
fig.suptitle("tshape: SVD over the TIME-history of the target kernel (10 refits x 12 rungs)",
             y=0.995)
fig.tight_layout()
fig.savefig("figs_op/fig4_tshape.png", dpi=150)
plt.close(fig)

# FIG 5 (numbers from committed verdict logs: session_kernels2.log)
bins = ["overnight", "early", "open", "midday", "close", "after"]
cosv = [0.826, 0.053, -0.547, 0.316, -0.205, 0.311]
span = [0.62, 0.58, 0.57, 0.76, 0.59, 0.57]
A = np.array([[0.175, 0.101, 0.209, 0.107, 0.049],
              [0.110, 0.289, -0.038, 0.113, 0.068],
              [0.354, 0.160, 0.087, 0.077, 0.028],
              [0.438, -0.048, 0.129, 0.156, 0.090],
              [0.155, 0.240, 0.155, 0.103, 0.111],
              [0.338, 0.096, 0.080, 0.098, 0.191]])
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
im = axes[0, 0].imshow(A, aspect="auto", cmap=DIV, vmin=-0.45, vmax=0.45)
axes[0, 0].set_xticks(range(5))
axes[0, 0].set_xticklabels([f"s{k + 1}" for k in range(5)])
axes[0, 0].set_yticks(range(6))
axes[0, 0].set_yticklabels(bins, fontsize=8)
axes[0, 0].set_title("shape amplitudes by session")
axes[0, 0].grid(visible=False)
fig.colorbar(im, ax=axes[0, 0], shrink=0.8)
cols = ["#2a78d6" if v >= 0 else "#eb6834" for v in cosv]
axes[0, 1].bar(bins, cosv, color=cols, width=0.6)
axes[0, 1].axhline(0, color=INK2, lw=0.8)
axes[0, 1].tick_params(axis="x", rotation=30)
axes[0, 1].set_title("target kernel: cos(session, unconditional)")
axes[1, 0].bar(bins, span, color="#5598e7", width=0.6)
axes[1, 0].axhline(5 / 12, color=C[1], lw=1.5, ls="--")
axes[1, 0].text(0.02, 5 / 12 + 0.015, "random 5-of-12", color=C[1], fontsize=8)
axes[1, 0].set_ylim(0, 0.85)
axes[1, 0].tick_params(axis="x", rotation=30)
axes[1, 0].set_title("5-shape span coverage of session operator")
loop = np.vstack([A[:, 0], A[:, 1]]).T
axes[1, 1].plot(np.append(loop[:, 0], loop[0, 0]), np.append(loop[:, 1], loop[0, 1]),
                color=INK2, lw=1, ls=":", zorder=1)
axes[1, 1].scatter(loop[:, 0], loop[:, 1],
                   c=[C[0], C[1], C[2], C[3], C[4], "#4a3aa7"], s=60, zorder=2)
for i, b_ in enumerate(bins):
    axes[1, 1].annotate(b_, loop[i], textcoords="offset points", xytext=(7, 4), fontsize=8)
axes[1, 1].set_xlabel("amplitude on shape 1 (slow curvature)")
axes[1, 1].set_ylabel("amplitude on shape 2 (tails vs priced)")
axes[1, 1].set_title("the daily loop in amplitude space")
fig.suptitle("Intraday regimes as motion on shared lag structure", y=0.995)
fig.tight_layout()
fig.savefig("figs_op/fig5_session.png", dpi=150)
plt.close(fig)
print("figures written")
