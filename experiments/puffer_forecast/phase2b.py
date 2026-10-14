"""PufferLib-5.0 native MinGRU, Phase 2b (2026-10-13): Phase 2 with three changes, all in our own code.

User (2026-10-13), after Phase 2's all_features blow-ups (near-constant availability averages standardised
to |z| up to 1,118 on a forecast day) and the learning rate selected at the grid's top in 8 of 10 cells:
  1. the linear arms' window mask on the minGRU inputs: src/models/window_mask.window_keep's rule (columns
     constant on the window, or byte-copies of an earlier kept column, float64 bytes) on the fit's own window
     [w0, w1) -- implemented in native/forecast_train.cu (job key mask=1; the float64 design X.f64 written
     here), checked against window_keep by `maskcheck` below;
  2. standardised inputs clipped at +-10 after the standardisation on the window (job key zclip=10);
  3. the learning-rate grid extended upward: 0.003, 0.01, 0.03, 0.06, 0.1 (20 configurations).
Everything else is Phase 2 (experiments/puffer_forecast/phase2.py, imported: 5 blocks of 250 origins,
re-tune at each block's first origin on its 2,000-session window, cold refit at every origin, 5 seeds,
phase2_select.py's rule).

Writes results/puffer_forecast_2026-10-12/_work/phase2b/{blocks.json, tune_block*.txt, maskcheck.txt},
_work/data/<design>/X.f64, and with `check RUN_DIR` compares the native masks with window_keep.

    python experiments/puffer_forecast/phase2b.py            # inputs
    python experiments/puffer_forecast/phase2b.py check RUN  # mask check of RUN/maskcheck/*.keep.i32
    python experiments/puffer_forecast/phase2b.py deltas     # Phase 2 -> 2b deltas (after phase2_score.py)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (
    REPO,
    REPO / "experiments",
    REPO / "experiments" / "puffer_forecast",
    REPO / "specs",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import phase2 as P2  # noqa: E402

WORK = P2.WORK
OUT = WORK / "phase2b"
LRS = (0.003, 0.01, 0.03, 0.06, 0.1)
MASK_ORIGINS = (2381, 2631, 2881, 3131, 3346, 3347, 3361, 3381, 3468)


def designs():
    import feature_mcs_runner as FR

    inp = FR.load_inputs()
    names = inp["names"]
    out = {}
    for m in P2.DESIGNS:
        idx = [names.index(c) for c in inp["models"][m]["cols"]]
        out[m] = np.ascontiguousarray(inp["bar_X"][:, idx], dtype="<f8")
    return out


def main() -> None:
    X = designs()
    for m, x in X.items():
        d = WORK / "data" / m
        f32 = np.fromfile(d / "X.f32", "<f4").reshape(x.shape)
        assert np.array_equal(f32, x.astype("<f4")), (
            f"{m}: X.f32 is not the float32 of the design"
        )
        (d / "X.f64").write_bytes(x.tobytes())
    P2.OUT = OUT
    P2.GRID = [
        {"L": L, "H": H, "lr": lr} for L in (5, 20) for H in (16, 64) for lr in LRS
    ]
    P2.COMMON = dict(P2.COMMON, mask=1, zclip=10)
    P2.main()
    # mask check: the native mask (gradcheck dumps <id>.keep.i32) on tune windows and refit windows
    lines = []
    for m in P2.DESIGNS:
        for o in MASK_ORIGINS:
            lines.append(
                P2.kv(
                    id=f"mask_{m}_r{o}",
                    mode="refit",
                    loss="mse",
                    data=f"data/{m}",
                    w0=o - P2.W,
                    w1=o,
                    L=20,
                    H=16,
                    lr=0.01,
                    seed=20261012,
                    epochs=1,
                    **dict(P2.COMMON, fc_n=1),
                )
            )
    (OUT / "maskcheck.txt").write_bytes(("\n".join(lines) + "\n").encode())
    print(f"phase2b inputs: grid {len(P2.GRID)} configs, {len(lines)} mask-check jobs")


def check(run: Path) -> None:
    from src.models.window_mask import window_keep

    X = designs()
    rows = []
    for m in P2.DESIGNS:
        for o in MASK_ORIGINS:
            k_nat = np.fromfile(run / "maskcheck" / f"mask_{m}_r{o}.keep.i32", "<i4")
            k_ref = window_keep(X[m][o - P2.W : o])
            rows.append(
                dict(
                    design=m,
                    origin=o,
                    n_cols=X[m].shape[1],
                    kept_native=len(k_nat),
                    kept_ref=len(k_ref),
                    identical=bool(np.array_equal(k_nat, k_ref)),
                )
            )
    out = REPO / "results" / "puffer_forecast_2026-10-12" / "phase2b"
    out.mkdir(parents=True, exist_ok=True)
    (out / "maskcheck.json").write_text(json.dumps(rows, indent=1))
    for r in rows:
        print(r)


def deltas() -> None:
    """The minGRU's scores and contrasts, Phase 2 vs Phase 2b (same scorer, same 866 trade days)."""
    import pandas as pd

    base = REPO / "results" / "puffer_forecast_2026-10-12"
    a = pd.read_csv(base / "phase2" / "scores.csv")
    b = pd.read_csv(base / "phase2b" / "scores.csv")
    g = "minGRU (PufferLib 5.0)"
    cols = ["qlike", "mse_fit", "es1", "sharpe_mid", "sharpe_crossed"]
    d = (
        a[a["model"] == g]
        .set_index("design")[cols]
        .join(b[b["model"] == g].set_index("design")[cols], lsuffix="_2", rsuffix="_2b")
    )
    for c in cols:
        d[f"{c}_delta"] = d[f"{c}_2b"] - d[f"{c}_2"]
    ca = pd.read_csv(base / "phase2" / "contrasts.csv").set_index(["design", "vs"])
    cb = pd.read_csv(base / "phase2b" / "contrasts.csv").set_index(["design", "vs"])
    keep = [
        "qlike_pct",
        "dm_qlike",
        "p_qlike",
        "mse_pct",
        "dm_mse",
        "p_mse",
        "hac_t_mid",
        "hac_t_crossed",
    ]
    dc = ca[keep].join(cb[keep], lsuffix="_2", rsuffix="_2b")
    d.to_csv(base / "phase2b" / "deltas_scores.csv")
    dc.to_csv(base / "phase2b" / "deltas_contrasts.csv")
    pd.set_option("display.width", 250)
    print(d.round(4).to_string())
    print(dc.round(3).to_string())


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "check":
        check(Path(sys.argv[2]))
    elif len(sys.argv) > 1 and sys.argv[1] == "deltas":
        deltas()
    else:
        main()
