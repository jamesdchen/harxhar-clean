"""Write the task files of the Optuna per-bar tree campaign (cluster/slurm/submit_optuna.sh).

Arms: LightGBM / XGBoost / RF x baseline / live_feasible / all_features at the 16:00 bar
(bar1600), TRAIN_WIN W = 2000 rows, N_OOS = 1469 forecast rows 2018-06-25 .. 2024-04-30
(the bar1600 arms of the earlier per-bar campaigns; the de-duplication drops columns, not
rows).  Stage 1 has a tuning point at every row (1469 x 9 = 13,221 studies of 50 trials).

COST MODEL (CARC core-seconds per unit of work; it only sizes the chunks so that every task
carries about the same expected work -- the time limits in submit_optuna.sh carry >= 10x
margin over the expected task, and stage 1 resumes from its saved points):
  stage 1  per tuning point = TRIALS x the local smoke's mean seconds per trial
           (experiments/gate_trees_optuna.py smoke gate: live_feasible, 6 points x 50
           trials, pool of 4 on the laptop; SMOKE_SEC_PER_TRIAL) / the laptop-to-CARC
           per-core ratio of the untuned refits (LAPTOP_OVER_CARC, from the tuned campaign's
           submit script) x the bucket's cost relative to live_feasible (BUCKET_FACTOR:
           all_features 3.8x, the top of the untuned campaign's 3.1-3.8x; baseline 0.25x,
           above its 0.11x because the per-trial overhead -- TPE sampling, the losses -- does
           not shrink with the columns)
  stage 2  per row = the smoke's distinct fits per row x its mean refit seconds / the ratio x
           the bucket factor
The arms' chunk rows are set so each stage has about TARGET_TASKS tasks of equal expected
work (TARGET_TASKS x 20 cores >= the 2000-CPU cap: the cap binds, not the task count), at
least MIN_ROWS (one wave of the 20-process pool), a multiple of the pool; an arm's last chunk runs
to the end of the series (-1) and absorbs a tail shorter than half a chunk.

CANARY: CANARY_BUCKET x lgbm / xgb / rf, stage 1 on OOS rows [0, CANARY_ROWS) = the fleet's
chunk 0 of those arms (DONE after the canary, so the fleet skips it).

Run from the repo root:  python cluster/optuna_make_tasks.py
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
W = 2000  # TRAIN_WIN rows = the halo every chunk carries
N_OOS = 1469  # bar1600 forecast rows
SEG = "bar1600"
MODELS = ("lgbm", "xgb", "rf")
BUCKETS = ("live_feasible", "all_features", "baseline")  # needed-first order
TRIALS = 50
POOL = 20
MIN_ROWS = POOL  # one wave of the pool
CANARY_ROWS = POOL  # one wave: every canary worker runs one study
# the masked run (2026-09-29, WINDOW_MASK=1): the canary on the cheapest bucket, so the fleet
# waits minutes, not the ~30 min the live_feasible canary of the unmasked run took (its
# LightGBM chunk had one 875-s tuning point, and the repeat at 7 processes)
CANARY_BUCKET = "baseline"
# the smoke gate's measurements (results/linear_subsection_trees_optuna/gates/gate_rows_*.csv)
SMOKE_SEC_PER_TRIAL = {
    "lgbm": 2.54,
    "xgb": 7.93,
    "rf": 3.24,
}  # mean over 6 points x 50 trials
SMOKE_SEC_PER_REFIT = {
    "lgbm": 3.00,
    "xgb": 2.41,
    "rf": 2.59,
}  # mean over the smoke's stage-2 fits
SMOKE_FITS_PER_ROW = {
    "lgbm": 29 / 6,
    "xgb": 32 / 6,
    "rf": 35 / 6,
}  # distinct fits / rows (9 paths)
LAPTOP_OVER_CARC = {"lgbm": 3.1, "xgb": 3.8, "rf": 3.3}
BUCKET_FACTOR = {"live_feasible": 1.0, "all_features": 3.8, "baseline": 0.25}
TARGET_TASKS = {"s1": 150, "s2": 60}  # 150 x 20 cores > the 2000-CPU cap
# REBALANCED 2026-09-29 20:35 (after submission, before the fleet started; the array sizes are
# fixed, so the line counts are kept -- EXACT_LINES): the canary (12479664, live_feasible, OOS
# rows 0-19, 20 workers on CARC) measured the CARC seconds per tuning point directly, and CARC
# was NOT the 3-4x faster core the ratio above assumed: lgbm 64-409 s per point (19 points,
# mean 166 s) plus one point still running after 20 min (counted as 1200 s), xgb 108-392 s
# (mean 229 s), rf 129-202 s (mean 180 s).  Stage 2: the rf canary made 122 distinct fits for
# 20 rows in a 31 s walk on 20 workers (~5 core-s per fit incl. the pool start and TreeSHAP).
CARC_SEC_PER_POINT_LIVE = {"lgbm": (19 * 166 + 1200) / 20, "xgb": 229.0, "rf": 180.0}
CARC_SEC_PER_ROW_LIVE = {"lgbm": 15.0, "xgb": 10.0, "rf": 15.0}
USE_CARC_COSTS = True
# the unmasked run kept its submitted array sizes (s1 110, s2 62; 12479667 / 12479669); the
# masked run is a fresh submission, sized by TARGET_TASKS (costs: the unmasked CARC ones --
# the mask only drops columns, so they are upper estimates)
EXACT_LINES: dict[str, int] = {}


def cost(stage: str, bucket: str, model: str) -> float:
    if USE_CARC_COSTS:
        live = CARC_SEC_PER_POINT_LIVE if stage == "s1" else CARC_SEC_PER_ROW_LIVE
        return live[model] * BUCKET_FACTOR[bucket]
    per = (
        TRIALS * SMOKE_SEC_PER_TRIAL[model]
        if stage == "s1"
        else SMOKE_FITS_PER_ROW[model] * SMOKE_SEC_PER_REFIT[model]
    )
    return per / LAPTOP_OVER_CARC[model] * BUCKET_FACTOR[bucket]


def rows_per_chunk(c: float, budget: float) -> int:
    r = max(MIN_ROWS, int(budget / c))
    return min(N_OOS, POOL * max(1, round(r / POOL)))  # whole waves of the pool


def chunks(first: int, rows: int) -> list[tuple[int, int]]:
    """(start, end) whole-series OOS rows covering [first, N_OOS); the last one open (-1)."""
    out, r = [], first
    while r < N_OOS:
        e = r + rows
        if N_OOS - e < rows // 2:  # a short tail joins the last chunk
            e = N_OOS
        out.append((r, -1 if e >= N_OOS else e))
        r = e
    return out


def line(bucket: str, model: str, k: int, r0: int, r1: int) -> str:
    return f"{bucket} {model} {W} {SEG} {k} {W + r0} {-1 if r1 < 0 else W + r1} {W}\n"


def plan(stage: str, budget: float) -> dict[tuple[str, str], list[tuple[int, int]]]:
    """{(bucket, model): [(first row, end row or -1), ...]} of the fleet (canary chunks excluded)."""
    out = {}
    for bucket in BUCKETS:
        for model in MODELS:
            first = CANARY_ROWS if (stage == "s1" and bucket == CANARY_BUCKET) else 0
            out[(bucket, model)] = chunks(
                first, rows_per_chunk(cost(stage, bucket, model), budget)
            )
    return out


def n_lines(stage: str, pl: dict) -> int:
    return (3 if stage == "s1" else 0) + sum(len(v) for v in pl.values())


def work(stage: str, key: tuple[str, str], c: tuple[int, int]) -> float:
    return ((N_OOS if c[1] < 0 else c[1]) - c[0]) * cost(stage, *key)


def exact_plan(stage: str, lines: int) -> tuple[dict, float]:
    """The plan of the largest budget with at most `lines` lines, then its costliest chunks split
    in two (at a whole wave of the pool when the chunk spans two or more) until it has exactly
    `lines` lines."""
    total = sum(cost(stage, b, m) * N_OOS for b in BUCKETS for m in MODELS)
    best = None
    for k in range(1, 4001):
        b = total / lines * k / 500.0  # 0.002x .. 8x of the even split
        if n_lines(stage, plan(stage, b)) <= lines:
            best = b
            break
    assert best is not None, f"no budget gives <= {lines} lines for {stage}"
    pl = plan(stage, best)
    while n_lines(stage, pl) < lines:
        key, j = max(
            ((k, j) for k, v in pl.items() for j in range(len(v))),
            key=lambda kj: work(stage, kj[0], pl[kj[0]][kj[1]]),
        )
        a, e = pl[key][j]
        n = (N_OOS if e < 0 else e) - a
        assert n >= 2, "cannot split a one-row chunk"
        half = POOL * (n // POOL // 2) if n >= 2 * POOL else n // 2
        pl[key][j : j + 1] = [(a, a + half), (a + half, e)]
    return pl, best


def main() -> None:
    total = {
        s: sum(cost(s, b, m) * N_OOS for b in BUCKETS for m in MODELS)
        for s in ("s1", "s2")
    }
    plans = {
        s: exact_plan(s, EXACT_LINES[s])[0]
        if EXACT_LINES
        else plan(s, total[s] / TARGET_TASKS[s])
        for s in total
    }
    lines: dict[str, list[str]] = {"canary": [], "s1": [], "s2": []}
    longest = {"s1": 0.0, "s2": 0.0}
    print(
        f"{'arm':26s} {'s1 s/pt':>8s} {'tasks':>5s} {'rows':>9s} {'s2 s/row':>8s} {'tasks':>5s} {'rows':>9s}"
    )
    for bucket in BUCKETS:
        for model in MODELS:
            key = (bucket, model)
            k0 = 0
            if bucket == CANARY_BUCKET:
                lines["canary"].append(line(bucket, model, 0, 0, CANARY_ROWS))
                lines["s1"].append(line(bucket, model, 0, 0, CANARY_ROWS))
                k0 = 1
            for j, (a, b) in enumerate(plans["s1"][key]):
                lines["s1"].append(line(bucket, model, k0 + j, a, b))
                longest["s1"] = max(longest["s1"], work("s1", key, (a, b)) / POOL)
            for j, (a, b) in enumerate(plans["s2"][key]):
                lines["s2"].append(line(bucket, model, j, a, b))
                longest["s2"] = max(longest["s2"], work("s2", key, (a, b)) / POOL)
            r1 = [(N_OOS if e < 0 else e) - a for a, e in plans["s1"][key]]
            r2 = [(N_OOS if e < 0 else e) - a for a, e in plans["s2"][key]]
            print(
                f"{bucket + ' ' + model:26s} {cost('s1', *key):8.1f} {len(r1):5d} {min(r1):4d}-{max(r1):4d} "
                f"{cost('s2', *key):8.2f} {len(r2):5d} {min(r2):4d}-{max(r2):4d}"
            )
    for name, ls in lines.items():
        p = ROOT / "cluster" / f"optuna_tasks_{name}.txt"
        p.write_text("".join(ls), encoding="utf-8", newline="\n")
        print(f"wrote {p.relative_to(ROOT)}: {len(ls)} lines")
    n1, n2 = len(lines["s1"]) - 3, len(lines["s2"])
    print(
        f"expected CARC core-hours (fits only): stage 1 {total['s1'] / 3600:.0f}, stage 2 "
        f"{total['s2'] / 3600:.0f}; mean task ~{total['s1'] / n1 / POOL / 60:.0f} / "
        f"{total['s2'] / n2 / POOL / 60:.1f} min on {POOL} cores; longest task "
        f"~{longest['s1'] / 60:.0f} / {longest['s2'] / 60:.1f} min; at the 2000-CPU cap "
        f"{(total['s1'] + total['s2']) / 2000 / 60:.0f} min of compute"
    )


if __name__ == "__main__":
    main()
