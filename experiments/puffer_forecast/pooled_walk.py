"""Pooled-panel walk-forward with decoupled weight decay (user, 2026-10-14).

The NN's contract on the pooled panel (prep_pooled.py; 48 half-hour bars round the clock, ET):
  * test period 2020-01-02 .. 2024-04-30 = panel rows [246064, 300317) (the panel ends 2024-04-30 23:30 ET), in the
    per-bar walk's 5 blocks of 250 sessions (blocks.json of Phase 2c; the per-bar session dates);
  * tune at each block start r_b (the first bar at or after 00:00 ET of the block's first session): the trailing
    24,000 bars [r_b - 24000, r_b) split into fit / embargo 48 bars / validation tail 2,400 bars (val_tail=2400,
    embargo=48: the trainer's tune mode, sequences of L bars inside each part), early stopping (patience 10,
    max_epochs 100, cosine over 100), grid H {64, 256} x L {20, 48} x lr {0.001, 0.003}, the seeds; selection = the
    lowest MSE of the seed-averaged validation forecast (as the per-bar walk);
  * refits every 21 sessions within the block (session s of the block at s = 0, 21, 42, ...): the trailing 24,000
    bars ending at the refit origin r (the first bar at or after 00:00 ET of that session), each seed for its own
    best epoch; each refit forecasts every bar [r, r_next) up to the next refit (or the test end).  The bank's trees
    refit at EVERY bar; the NN refits monthly.
  * native_g6, softplus, xstd=0 (the panel is prescaled), mask=1, wd fixed (A) or in the tune grid (B).

    python experiments/puffer_forecast/pooled_walk.py prep          # _work/pooled_walk/: data, rows.csv, plan.json
    python experiments/puffer_forecast/pooled_walk.py inputs SEEDS  # _work/pwalk_wd<x>/, pwalk_wdgrid/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prep_pooled as PP  # noqa: E402

REPO = HERE.parents[1]
WORK = REPO / "results" / "puffer_forecast_2026-10-12" / "_work"
OUT = WORK / "pooled_walk"
W, VAL, EMB = 24_000, 2_400, 48
GRID = [(H, L, lr) for H in (64, 256) for L in (20, 48) for lr in (0.001, 0.003)]
WDS = (0, 0.01, 0.03, 0.1, 0.3, 1)
SETS = ("baseline", "all_features")
COMMON = dict(
    layers=1,
    batch=128,
    max_epochs=100,
    patience=10,
    val_tail=VAL,
    embargo=EMB,
    momentum=0.95,
    max_grad_norm=1.0,
    anneal=1,
    min_lr_ratio=0.0,
    mask=1,
    xstd=0,
    link="softplus",
    loss="mse",
)


def prep() -> None:
    import src.unification as U

    z = np.load(PP.CACHE, allow_pickle=True)
    names = [str(s) for s in z["names"]]
    y = np.asarray(z["y"], np.float64)
    base = np.asarray(z["baselines"], np.float64)
    X = PP.member_memmap(PP.CACHE, "X.npy")
    n, p = X.shape
    tab = pd.read_parquet(PP.YHAT, columns=["t", "baseline", "rv_raw"])
    t_all = pd.DatetimeIndex(tab["t"]).tz_convert(
        "America/New_York"
    )  # panel rows 24,000 ..
    dates = pd.read_csv(WORK / "dates.csv")
    blocks = json.loads((WORK / "phase2c" / "blocks.json").read_text())["blocks"]

    def row_at(day: str) -> int:  # first panel row at or after 00:00 ET of the day
        return (
            int(t_all.searchsorted(pd.Timestamp(day, tz="America/New_York")))
            + PP.WARMUP
        )

    r_end = n
    plan = []
    for b in blocks:
        origins = b["origins"]
        sess = [origins[s] for s in range(0, len(origins), 21)]
        rows = [row_at(dates["date"][o][:10]) for o in sess]
        plan.append(
            dict(
                k=b["k"],
                first_session=dates["date"][origins[0]][:10],
                refit_sessions=[dates["date"][o][:10] for o in sess],
                refit_rows=rows,
            )
        )
    for i, b in enumerate(plan):
        nxt = plan[i + 1]["refit_rows"][0] if i + 1 < len(plan) else r_end
        b["refit_next"] = b["refit_rows"][1:] + [nxt]
    r_test0 = plan[0]["refit_rows"][0]
    assert r_test0 == 246064, r_test0
    r0 = r_test0 - W
    m = r_end - r0
    # every exported row must have y > 0 (softplus fits it); the test rows' 16:00 bars on the per-bar trade days
    # y_fit = 0 where RV printed 0 (33 exported rows): trained on as is (mse; softplus can only approach it),
    # excluded from every QLIKE (undefined at a zero target)
    assert (y[r0:r_end] >= 0).all()
    zero_rows = [int(r) for r in np.flatnonzero(y[r0:r_end] == 0) + r0]
    t = t_all[r0 - PP.WARMUP : r_end - PP.WARMUP]
    assert np.array_equal(
        tab["baseline"].to_numpy()[r0 - PP.WARMUP : r_end - PP.WARMUP], base[r0:r_end]
    )
    sets = {"all_features": np.arange(p), "baseline": U._backbone_cols(names)}
    OUT.mkdir(parents=True, exist_ok=True)
    for s, cols in sets.items():
        d = OUT / f"pooled_walk_{s}"
        d.mkdir(exist_ok=True)
        with open(d / "X.f32", "wb") as f32, open(d / "X.f64", "wb") as f64:
            for a in range(
                r0, r_end, 8000
            ):  # chunks: the panel is a 2.75 GB memory map, this machine has ~2 GB RAM
                c = np.asarray(X[a : min(a + 8000, r_end)][:, cols], np.float64)
                assert np.isfinite(c).all()
                c.astype("<f8").tofile(f64)
                c.astype("<f4").tofile(f32)
        y[r0:r_end].astype("<f4").tofile(d / "y.f32")
        (d / "meta.txt").write_text(f"{m} {len(cols)}\n")
    pd.DataFrame(
        {
            "row": np.arange(m),
            "panel_row": np.arange(r0, r_end),
            "t_et": t.strftime("%Y-%m-%d %H:%M"),
            "slot": (t.hour * 2 + t.minute // 30),
            "baseline": base[r0:r_end],
            "rv_raw": tab["rv_raw"].to_numpy()[r0 - PP.WARMUP : r_end - PP.WARMUP],
        }
    ).to_csv(OUT / "rows.csv", index=False)
    for b in plan:  # slice-relative rows
        b["refit_rows_slice"] = [r - r0 for r in b["refit_rows"]]
        b["refit_next_slice"] = [r - r0 for r in b["refit_next"]]
    first_val = plan[0]["refit_rows"][0] - VAL
    info = dict(
        panel_slice=[r0, r_end],
        test_rows=[r_test0, r_end],
        n_test=r_end - r_test0,
        window=W,
        y_zero_panel_rows=zero_rows,
        y_zero_in_test=int(sum(r >= r_test0 for r in zero_rows)),
        val_tail=VAL,
        embargo=EMB,
        fit_rows=W - VAL - EMB,
        block0_val_tail=[
            str(t_all[first_val - PP.WARMUP]),
            str(t_all[plan[0]["refit_rows"][0] - 1 - PP.WARMUP]),
        ],
        block0_val_days=int(
            pd.Index(
                t_all[first_val - PP.WARMUP : plan[0]["refit_rows"][0] - PP.WARMUP].date
            ).nunique()
        ),
        blocks=plan,
    )
    (OUT / "plan.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: v for k, v in info.items() if k != "blocks"}, indent=1))
    for b in plan:
        print(
            b["k"],
            b["first_session"],
            len(b["refit_rows"]),
            b["refit_rows"][0],
            b["refit_next"][-1],
        )


def kv(**k) -> str:
    return " ".join(f"{a}={b}" for a, b in k.items())


def inputs(seeds: list[int]) -> None:
    info = json.loads((OUT / "plan.json").read_text())
    for tag, wds in [(f"wd{w:g}", [w]) for w in WDS] + [("wdgrid", list(WDS))]:
        d = WORK / f"pwalk_{tag}"
        d.mkdir(parents=True, exist_ok=True)
        for b in info["blocks"]:
            r = b["refit_rows_slice"][0]
            lines = []
            for s in SETS:
                for H, L, lr in GRID:
                    for wd in wds:
                        for seed in seeds:
                            wtag = f"_wd{wd:g}" if tag == "wdgrid" else ""
                            lines.append(
                                kv(
                                    id=f"ptune_{s}_L{L}_H{H}_lr{lr}{wtag}_s{seed}",
                                    data=f"data/pooled_walk_{s}",
                                    mode="tune",
                                    w0=r - W,
                                    w1=r,
                                    L=L,
                                    H=H,
                                    lr=lr,
                                    seed=seed,
                                    fc_n=1,
                                    wd=wd,
                                    **COMMON,
                                )
                            )
            (d / f"tune_block{b['k']}.txt").write_bytes(
                ("\n".join(lines) + "\n").encode()
            )
        man = dict(
            designs=list(SETS),
            seeds=seeds,
            W=W,
            common=dict(COMMON),
            wd_fixed=None if tag == "wdgrid" else wds[0],
            blocks=[
                dict(
                    k=b["k"], origins=b["refit_rows_slice"], next=b["refit_next_slice"]
                )
                for b in info["blocks"]
            ],
        )
        (d / "plan.json").write_bytes(json.dumps(man, indent=1).encode())
        print(
            tag,
            {
                b["k"]: len((d / f"tune_block{b['k']}.txt").read_text().splitlines())
                for b in info["blocks"]
            },
        )


if __name__ == "__main__":
    if sys.argv[1] == "prep":
        sys.path.insert(0, str(REPO))
        prep()
    elif sys.argv[1] == "inputs":
        inputs([int(s) for s in sys.argv[2].split(",")])
