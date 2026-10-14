"""PufferLib-5.0 native MinGRU, Phase 2c (2026-10-13): the prescaled design fed as is.

Root cause of Phases 2 / 2b's all_features failures (found by the main session): data/<design>/X.f32 is the
regular stack's PRESCALED 16:00 design (run_executor prescale=True: causal rolling median / IQR over the
trailing 2,000 rows with executor._build_scale_guards -- IQR floors for imputed series, hurdle encoding --
and the *_avail_ma_* / *_active_ma_* columns passed raw as [0, 1] fractions); the trainer re-standardised
every column by its window mean / sd, undoing those guards (fomc_*_avail_ma_3125: window sd 1.4e-5 ->
|z| 1,118).  The linear arms use the prescaled row as is.  Phase 2c, our code only:
  xstd=0   the design's kept columns fed unchanged (no input mean / sd, no clip); the target keeps its
           standardisation (mu_y / sd_y)
  mask=1   the regular stack's window mask (window_keep) on the fit's window [w0, w1)
  grid     Phase 2's 12 configurations (lr 0.003 / 0.01 / 0.03); selection unchanged (validation MSE)
Everything else is Phase 2 (phase2.py imported; phase2_select.py).

    python experiments/puffer_forecast/phase2c.py               # inputs -> _work/phase2c/
    python experiments/puffer_forecast/phase2c.py check RUN     # gate: inputs == prescaled X (bit for bit)
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
import phase2b as P2B  # noqa: E402

WORK = P2.WORK
OUT = WORK / "phase2c"


def main() -> None:
    P2.OUT = OUT
    P2.COMMON = dict(
        P2.COMMON, mask=1, xstd=0
    )  # Phase 2's GRID (12 configurations) unchanged
    P2.main()
    lines = []
    for m in P2.DESIGNS:
        for o in P2B.MASK_ORIGINS:
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
                    **P2.COMMON,
                )
            )
    (OUT / "maskcheck.txt").write_bytes(("\n".join(lines) + "\n").encode())
    print(f"phase2c inputs: grid {len(P2.GRID)} configs, {len(lines)} input-check jobs")


def check(run: Path) -> None:
    """The trainer's window inputs (gradcheck Z dumps: n_ctx x (kept + 1)) and its first minibatch (obs dumps)
    equal the prescaled design's kept columns bit for bit; the kept columns equal window_keep's."""
    from src.models.window_mask import window_keep

    rows = []
    for m in P2.DESIGNS:
        meta = (WORK / "data" / m / "meta.txt").read_text().split()
        n, p = int(meta[0]), int(meta[1])
        X32 = np.fromfile(WORK / "data" / m / "X.f32", "<f4").reshape(n, p)
        X64 = np.fromfile(WORK / "data" / m / "X.f64", "<f8").reshape(n, p)
        for o in P2B.MASK_ORIGINS:
            g = run / "maskcheck" / f"mask_{m}_r{o}"
            js = json.loads(Path(f"{g}.json").read_text())
            keep = np.fromfile(f"{g}.keep.i32", "<i4")
            P, n_ctx, B, T, S, row0, N = (
                js[k] for k in ("P", "n_ctx", "B", "T", "S", "row0", "N")
            )
            w0 = o - P2.W
            Z = np.fromfile(f"{g}.Z.f32", "<f4").reshape(n_ctx, P)
            ref = X32[w0 : w0 + n_ctx][:, keep]
            z_ok = Z[:, :-1].tobytes() == np.ascontiguousarray(ref).tobytes() and bool(
                (Z[:, -1] == 1.0).all()
            )
            perm = np.fromfile(f"{g}.perm.i32", "<i4").reshape(-1, S * B)[0, :B]
            real = (perm >= 0) & (perm < N)
            r = row0 + np.where(real, perm, 0)
            obs = np.fromfile(f"{g}.obs.f32", "<f4").reshape(B, T, P)
            obs_ref = Z[r[:, None] + np.arange(-T + 1, 1)[None, :]]
            rows.append(
                dict(
                    design=m,
                    origin=o,
                    kept=len(keep),
                    mask_equals_window_keep=bool(
                        np.array_equal(keep, window_keep(X64[w0:o]))
                    ),
                    Z_equals_prescaled_X=bool(z_ok),
                    obs_equals_X_rows=bool(obs.tobytes() == obs_ref.tobytes()),
                    max_abs_input=float(np.abs(Z[:, :-1]).max()),
                )
            )
    out = REPO / "results" / "puffer_forecast_2026-10-12" / "phase2c"
    out.mkdir(parents=True, exist_ok=True)
    (out / "inputcheck.json").write_text(json.dumps(rows, indent=1))
    for r in rows:
        print(r)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "check":
        check(Path(sys.argv[2]))
    else:
        main()
