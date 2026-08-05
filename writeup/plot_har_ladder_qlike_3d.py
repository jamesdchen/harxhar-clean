"""3D QLIKE surface over the HAR-ladder sweep: exp base x rung cap.

Reads the run-level reducer table verbatim (no metric recomputed); geometric
arms only (the explicit decaying-ratio arm has no base coordinate and is
stated in the caption instead). Categorical axis positions (index, labeled
with the true values) so the log-spaced caps/bases don't distort the surface.
"""
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SRC = pathlib.Path("_aggregated/har_base_sweep-53c27e42_metrics_table.csv")
OUT = pathlib.Path("writeup/har_ladder_qlike_3d.png")

df = pd.read_csv(SRC)
geo = df[df.ladder_arm != "explicit"].copy()
geo["base"] = geo.ladder_arm.astype(int)
bases = sorted(geo.base.unique())      # [2, 3, 4, 5, 6, 8]
caps = sorted(geo.rungs_cap.unique())  # [240, 960, 3125]

Z = np.empty((len(caps), len(bases)))
for i, c in enumerate(caps):
    for j, b in enumerate(bases):
        Z[i, j] = geo[(geo.base == b) & (geo.rungs_cap == c)].qlike.iloc[0]

X, Y = np.meshgrid(range(len(bases)), range(len(caps)))
incumbent = float(df.incumbent_qlike.iloc[0])

fig = plt.figure(figsize=(9, 6.5), dpi=200)
ax = fig.add_subplot(projection="3d")
surf = ax.plot_surface(
    X, Y, Z, cmap="Blues_r", edgecolor="#5b6472", linewidth=0.4,
    alpha=0.92, antialiased=True,
)
ax.scatter(X.ravel(), Y.ravel(), Z.ravel(), color="#1f2733", s=14, depthshade=False)

# incumbent reference plane (base-5 production ladder, exact tie at b5/c3125)
ax.plot_surface(
    X, Y, np.full_like(Z, incumbent), color="#c25d3a", alpha=0.14, linewidth=0,
)

# direct labels: the winner and the incumbent cell only (selective, not every point)
imin, jmin = np.unravel_index(Z.argmin(), Z.shape)
ax.text(jmin - 0.4, imin, Z.min() - 0.00035,
        f"best: b{bases[jmin]} cap {caps[imin]}  {Z.min():.5f}",
        fontsize=8, color="#1f2733", ha="left", va="top")
j5 = bases.index(5)
i5 = caps.index(3125)
ax.text(j5, i5, Z[i5, j5] + 0.00012, f"incumbent (b5)\n{incumbent:.5f}",
        fontsize=8, color="#c25d3a", ha="center")

ax.set_xticks(range(len(bases)), [str(b) for b in bases])
ax.set_yticks(range(len(caps)), [str(c) for c in caps])
ax.set_xlabel("ladder base (geometric)")
ax.set_ylabel("rung cap (bars)")
ax.set_zlabel("pooled QLIKE (218,934 OOS bars)")
ax.set_title(
    "HAR ladder geometry: QLIKE by base x cap\n"
    "orange plane = base-5 production incumbent; explicit {1,8,48,240,960} arm "
    "(no base coord): 0.13439–0.13447, above the plane",
    fontsize=10,
)
ax.invert_xaxis()  # base 8 near, base 2 far — the downhill-to-b2 slope faces the viewer
ax.view_init(elev=22, azim=-58)
fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
print(f"wrote {OUT}")
