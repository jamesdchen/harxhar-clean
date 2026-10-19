"""The pooled linear rerun on all 48 half-hour intervals (2026-10-19): frozen penalties, task lists, scoring.

Author's decisions (2026-10-19; writeup/rehaul notes E6, E9, E15, E28, S4a-c, N18, E26):
  sample    all 48 intervals, overnight included (E15);
  window    2,000 trading days = 96,000 intervals for every pooled linear model, the OLS included
            (a0 used 24,000);
  penalty   one value per model, chosen once on the pre-sample and FROZEN over the evaluation span: the
            pre-tune of experiments/pool_linear_pretune.py (4 folds x 12,000 intervals tiling the 48,000
            intervals before the first interval of 2020-01-03, each fit on the 96,000 intervals before a
            1,200-interval embargo; criterion the fold mean of the validation MSE on the adjusted scale,
            first argmin, as experiments/pool_linear_pretune_report.py); the walk then seeds that value at
            every 250th solve (--sched holding one number); no margin re-tune;
  design    pool_linear_pretune.py --rerun: HAR ladder base 2 (k = 1 .. 2048), DROP_ACTIVITY_MA (E6) and
            CALENDAR_THROUGH_OVERNIGHT (E9) on, every 2026-10-15 data-fix flag at its default (on), FOMC series
            out of all_features; penalty factor r on the HAR benchmark block (1/100 on all_features, 1 on the
            HAR benchmark set, where every column is in the block), as the printed arms;
  models    ridge, lasso, elastic net (l1_ratio 0.5) on the HAR benchmark set (baseline) and on all features;
            the OLS on the HAR benchmark set (the spec's ridge path at alpha 0), a0's successor;
  scoring   tab:rh_oos unchanged: the 16:00 interval on the 1,082 days 3 January 2020 .. 30 April 2024 (early
            closes and non-positive prints dropped), plain forecast pred_adj^2 x B, truth the 16:00 sumret2 of
            data/core_stats.parquet, QLIKE / MSE / R2_OOS and Diebold-Mariano as
            experiments/rolling_eval_2026_10_15.py (its functions, imported); also every interval of the span;
  export    forecasts of EVERY interval (N18, E26), forecasts/<model>_<set>.parquet.

Stages (python experiments/pooled_linear_rerun_2026_10_19.py <stage>):
  pick    _work/pre/<set>/<est>/fold.npz -> frozen_penalties.json, prescreen.csv, _work/sched/<est>_<set>.json
  tasks   cluster/pooled_linear_rerun_48_tasks.txt (the walks; cluster/pool_pretune_task.sh line format)
  score   _work/walk/<set>/<est>/causal_tune_linear/<est>/<set>/results.csv -> forecasts/, scores.csv, dm.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "experiments", REPO / "notebooks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

OUT = REPO / "results" / "pooled_linear_rerun_2026-10-19_48"
WORK = OUT / "_work"
SETS = ("baseline", "all_features")
EST = ("ridge", "reclasso", "reclasticnet")
SHORT = {"ridge": "ridge", "reclasso": "lasso", "reclasticnet": "enet", "ols": "ols"}
R_BB = {
    "all_features": "0.01",
    "baseline": "1.0",
}  # the printed arms' backbone penalty factor
W, TUNE_PER = 96_000, 250
PRE_START, PRE_HALO = (
    246_112,
    145_200,
)  # first interval of 2020-01-03; 4 x 12,000 + 1,200 + 96,000
WALK_START = 246_000  # the re-choice boundary (block 600) just before PRE_START; halo W
SPAN_FIRST, SPAN_LAST, SPAN_N = "2020-01-03", "2024-04-30", 1082
PRINTED = REPO / "results" / "rolling_eval_2026-10-15" / "metrics.csv"
PRINTED_KEY = {
    ("ols", "baseline"): "a0",
    **{(s, b): f"poolpt_{SHORT[s]}_{b}" for s in EST for b in SETS},
}


def models() -> list[tuple[str, str]]:
    return [("ols", "baseline")] + [(e, b) for e in EST for b in SETS]


def walk_dir(est: str, b: str) -> Path:
    return WORK / "walk" / b / est


# ============================================================================ pick
def argmin_first(v) -> int:
    v = np.asarray(v, float)
    return int(np.flatnonzero(v == np.nanmin(v))[0])


def pick() -> None:
    sets = (
        sys.argv[2].split(",") if len(sys.argv) > 2 else SETS
    )  # a subset: sched files only
    rows, frozen, designs = [], {}, {}
    (WORK / "sched").mkdir(parents=True, exist_ok=True)
    for b in sets:
        meta = json.loads((WORK / f"design_{b}" / "design.json").read_text())
        assert meta["rerun"], b
        designs[b] = dict(
            n_columns=len(meta["names"]),
            columns=meta["names"],
            rows=meta["shape"][0],
            backbone_penalty_factor=float(R_BB[b]),
        )
        for est in EST:
            d = WORK / "pre" / b / est
            z = np.load(d / "fold.npz")
            info = json.loads((d / "run.json").read_text())
            assert info["rerun"] and info["r"] == float(R_BB[b]), info
            assert (info["start"], info["halo"]) == (PRE_START, PRE_HALO), info
            mse = (z["resid"] ** 2).mean(axis=2)  # (folds, grid)
            grid = z["grid"]
            i = argmin_first(mse.mean(axis=0))
            alpha = float(grid[i])
            frozen.setdefault(SHORT[est], {})[b] = alpha
            (WORK / "sched" / f"{est}_{b}.json").write_text(repr(alpha) + "\n")
            for g in range(len(grid)):
                rows.append(
                    dict(
                        model=SHORT[est],
                        set=b,
                        alpha=float(grid[g]),
                        picked=g == i,
                        mse_mean=float(mse[:, g].mean()),
                        **{f"mse_fold{f}": float(mse[f, g]) for f in range(len(mse))},
                    )
                )
    frozen["ols"] = {"baseline": 0.0}
    (WORK / "sched" / "ols_baseline.json").write_text("0.0\n")
    if tuple(sets) != SETS:
        print(json.dumps(frozen, indent=1))
        return
    js = dict(
        penalties=frozen,
        note=(
            "sklearn units (Ridge alpha; ElasticNet alpha with l1_ratio 1 for lasso, 0.5 for the elastic net); "
            "penalty on column j = alpha x pf_j, pf_j = backbone_penalty_factor on the HAR benchmark block "
            "(har_ma_* and the 7 calendar columns), 1 elsewhere; intercept unpenalized; chosen once on the "
            "pre-sample (4 folds x 12,000 intervals before 2020-01-03 00:00, fit 96,000, embargo 1,200; "
            "fold-mean validation MSE, adjusted scale, first argmin) and frozen over the evaluation span. "
            "ols = the spec's ridge path at alpha 0."
        ),
        designs=designs,
        window_intervals=W,
        har_ladder=[2**j for j in range(12)],
        flags=dict(
            DROP_ACTIVITY_MA=True,
            CALENDAR_THROUGH_OVERNIGHT=True,
            data_fix_flags="src/backtest/executor.py defaults (all on)",
            fomc_in_all_features=False,
        ),
    )
    (OUT / "frozen_penalties.json").write_text(json.dumps(js, indent=1) + "\n")
    pd.DataFrame(rows).to_csv(OUT / "prescreen.csv", index=False)
    print(json.dumps(frozen, indent=1))


# ============================================================================ tasks
def tasks() -> None:
    """Cluster task list (cluster/pool_pretune_task.sh format, plus --rerun: see the header lines)."""
    lines = []
    for est, b in models():
        lines.append(
            f"rerun48 walk {b} {est} {R_BB[b] if est != 'ols' else '1.0'} "
            f"results/pooled_linear_rerun_2026-10-19_48/_work/sched/{est}_{b}.json "
            f"{WALK_START} -1 {W} results/pooled_linear_rerun_2026-10-19_48/_work/design_{b}"
        )
    p = REPO / "cluster" / "pooled_linear_rerun_48_tasks.txt"
    p.write_text("\n".join(lines) + "\n")
    print(p.read_text())


# ============================================================================ score
def load_walk(est: str, b: str) -> pd.DataFrame | None:
    d = walk_dir(est, b)
    f = d / "causal_tune_linear" / est / b / "results.csv"
    if not (d / "run.json").exists() or not f.exists():
        return None
    info = json.loads((d / "run.json").read_text())
    assert info["rerun"] and info["start"] == WALK_START and info["end"] == -1, info
    r = pd.read_csv(f, parse_dates=["date"])
    r["B"] = r["true_raw"] / r["true_adj"] ** 2
    r["plain"] = r["pred_adj"] ** 2 * r["B"]
    return r


def score() -> None:
    import rolling_eval_2026_10_15 as R

    _, hist = R.truth_history()
    days = hist.index[(hist.index >= SPAN_FIRST) & (hist.index <= SPAN_LAST)]
    assert len(days) == SPAN_N, len(days)
    bench = R.ct_benchmark(hist, days)
    rv = hist.reindex(days)
    # actual RV of every interval (the executor's true_raw is the winsorized target on clipped intervals)
    core = pd.read_parquet(R.CORE, columns=["endbartime", "sumret2"])
    core = pd.Series(
        core["sumret2"].to_numpy(float), index=pd.to_datetime(core["endbartime"])
    )
    assert not core.index.duplicated().any()
    (OUT / "forecasts").mkdir(exist_ok=True)
    plain16, every = {}, {}
    for est, b in models():
        r = load_walk(est, b)
        if r is None:
            continue
        key = f"{SHORT[est]}_{b}"
        r["rv"] = core.reindex(r["date"]).to_numpy()
        r[
            ["date", "true_adj", "pred_adj", "true_raw", "pred_raw", "B", "plain", "rv"]
        ].to_parquet(OUT / "forecasts" / f"{key}.parquet", index=False)
        m16 = (r["date"].dt.hour == 16) & (r["date"].dt.minute == 0)
        s = r.loc[m16].set_index(r.loc[m16, "date"].dt.normalize())
        p = s["plain"].reindex(days)
        # the table's truth (core_stats 16:00 sumret2) is the forecast file's actual RV on every day
        assert np.array_equal(s["rv"].reindex(days).to_numpy(), rv.to_numpy()), key
        plain16[key] = p
        # every interval of the span with a positive actual RV (overnight intervals can print 0)
        sp = r[
            (r["date"] >= SPAN_FIRST)
            & (r["date"] < pd.Timestamp(SPAN_LAST) + pd.Timedelta(days=1))
        ]
        every[key] = sp[sp["rv"] > 0].set_index("date")
    printed = pd.read_csv(PRINTED)
    printed = printed[printed["span"] == "common"].set_index("key")
    rows, dms = [], []
    ols = plain16.get("ols_baseline")
    for est, b in models():
        key = f"{SHORT[est]}_{b}"
        if key not in plain16:
            continue
        p = plain16[key]
        ok = p.notna().to_numpy()
        m = R.metrics(rv[ok], p[ok], bench)
        e = every[key]
        q_all = R.qlike_daily(e["rv"].to_numpy(float), e["plain"].to_numpy(float))
        pk = PRINTED_KEY[(est, b)]
        row = dict(
            model=SHORT[est],
            set=b,
            n_days=int(ok.sum()),
            qlike=m["qlike"],
            mse_e10=m["mse"] * 1e10,
            r2_oos_pct=m["r2_oos_pct"],
            printed_key=pk,
            printed_qlike=float(printed.loc[pk, "qlike"]),
            printed_mse_e10=float(printed.loc[pk, "mse_e10"]),
            qlike_all_intervals=float(np.nanmean(q_all)),
            mse_all_intervals_e10=float(((e["rv"] - e["plain"]) ** 2).mean() * 1e10),
            n_all_intervals=int(len(e)),
        )
        rows.append(row)
        if ols is not None and key != "ols_baseline" and ok.all():
            dms.append(
                dict(
                    model=SHORT[est],
                    set=b,
                    against="ols_baseline (rerun)",
                    **R.dm_pair(rv, p, ols),
                )
            )
            # every interval of the span: the same test on the interval losses
            o = every["ols_baseline"].reindex(e.index)
            dq = R.dm_test(
                q_all,
                R.qlike_daily(o["rv"].to_numpy(float), o["plain"].to_numpy(float)),
            )
            dms.append(
                dict(
                    model=SHORT[est],
                    set=b,
                    against="ols_baseline (rerun), every interval",
                    dm_qlike=float(dq["dm"]),
                    dm_qlike_p=float(dq["p"]),
                    dm_qlike_mean_diff=float(dq["mean_diff"]),
                    dm_T=int(dq["T"]),
                    dm_hac_lag=int(dq.get("hac_lag", -1)),
                )
            )
    pd.DataFrame(rows).to_csv(OUT / "scores.csv", index=False)
    pd.DataFrame(dms).to_csv(OUT / "dm.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(dms).to_string(index=False))


if __name__ == "__main__":
    {"pick": pick, "tasks": tasks, "score": score}[sys.argv[1]]()
