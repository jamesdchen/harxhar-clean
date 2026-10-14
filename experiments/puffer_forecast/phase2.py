"""PufferLib Phase 2: walk-forward tune-job files.

The Phase 1 projection (results/puffer_forecast_2026-10-12/projection.csv) prices
one scheme the review left for a go-ahead: refit every 16:00 session, and re-tune
the 12-config grid every 250 sessions. Phase 1 only built inputs for `baseline`
and `all_features`, so Phase 2 walks those two forward. The 96-design hour-count
in the projection is that same per-design cost times 96; those designs are not
in the Phase 1 inputs.

Origins are the 1,088 rows from the Phase 1 window end (2020-01-02) through the
last 16:00 row. Each block of 250 origins (the last block is shorter) gets one
tune window of 2,000 sessions ending at the block's first origin. The cluster
selector (phase2_select.py) picks the config by the Phase 1 rule -- lowest
seed-averaged validation MSE -- and writes the cold refits (fc_n=1, that
config, 5 seeds, each seed's own best epoch).

Stdlib only. Grid, seeds and the training constants are the Phase 1 ones
(experiments/puffer_forecast/prep.py).

    python experiments/puffer_forecast/phase2.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WORK = REPO / "results" / "puffer_forecast_2026-10-12" / "_work"
OUT = WORK / "phase2"
BLOCK = 250
W = 2000
DESIGNS = ("baseline", "all_features")
SEEDS = tuple(20261012 + k for k in range(5))
GRID = [
    {"L": L, "H": H, "lr": lr}
    for L in (5, 20)
    for H in (16, 64)
    for lr in (0.003, 0.01, 0.03)
]
COMMON = {
    "layers": 1,
    "batch": 128,
    "max_epochs": 200,
    "patience": 20,
    "val_tail": 125,
    "embargo": 25,
    "momentum": 0.95,
    "max_grad_norm": 1.0,
    "anneal": 1,
    "min_lr_ratio": 0.0,
    "fc_n": 1,
}


def kv(**kwargs) -> str:
    return " ".join(f"{key}={value}" for key, value in kwargs.items())


def n_dates() -> int:
    with (WORK / "dates.csv").open() as handle:
        next(handle)
        return sum(1 for _ in handle)


def main() -> None:
    meta = json.loads((WORK / "prep.json").read_text())
    n = n_dates()
    origins = list(range(int(meta["w1"]), n))
    if len(origins) != 1088:
        raise SystemExit(f"expected 1088 origins, got {len(origins)}")
    if origins[0] - W != int(meta["w0"]):
        raise SystemExit("first tune window does not match the Phase 1 window")

    OUT.mkdir(parents=True, exist_ok=True)
    blocks = []
    for k, start in enumerate(range(0, len(origins), BLOCK)):
        chunk = origins[start : start + BLOCK]
        tune_w1 = chunk[0]
        tune_w0 = tune_w1 - W
        lines = []
        for design in DESIGNS:
            for g in GRID:
                for seed in SEEDS:
                    tid = f"tune_{design}_L{g['L']}_H{g['H']}_lr{g['lr']}_s{seed}"
                    base = dict(
                        data=f"data/{design}",
                        w0=tune_w0,
                        w1=tune_w1,
                        L=g["L"],
                        H=g["H"],
                        lr=g["lr"],
                        seed=seed,
                        **COMMON,
                    )
                    lines.append(kv(id=tid, mode="tune", loss="mse", **base))
        (OUT / f"tune_block{k}.txt").write_bytes(("\n".join(lines) + "\n").encode())
        blocks.append(
            {
                "k": k,
                "tune_w0": tune_w0,
                "tune_w1": tune_w1,
                "n_origins": len(chunk),
                "origins": chunk,
                "n_tune_jobs": len(lines),
            }
        )
    manifest = {
        "designs": list(DESIGNS),
        "seeds": list(SEEDS),
        "block": BLOCK,
        "W": W,
        "fc_n": 1,
        "common": COMMON,
        "blocks": blocks,
    }
    (OUT / "blocks.json").write_bytes(json.dumps(manifest).encode())
    print(
        json.dumps(
            {
                "origins": len(origins),
                "first": origins[0],
                "last": origins[-1],
                "blocks": [
                    {
                        "k": b["k"],
                        "tune_w1": b["tune_w1"],
                        "n_origins": b["n_origins"],
                        "n_tune_jobs": b["n_tune_jobs"],
                    }
                    for b in blocks
                ],
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
