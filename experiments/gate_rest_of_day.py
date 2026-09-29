"""Local gates 2-4 of the rest-of-day campaign (gate 1, IDENTITY, is
experiments/check_restofday_identity.py, run on the 15:30 arm).

GATE 2  TARGET: on the spec's own panel frame, RV_rem (spec: groupby + math.fsum)
        equals an independent construction (day x stamp pivot, all remaining
        bars present, math.fsum across the row) bit for bit, on every row of
        every clock; the incomplete-row counts per clock are tabulated.
GATE 3  LEAK: every bar's RV stamped after the entry clock c on the last
        LEAK_K sessions is multiplied by LEAK_FACTOR, in the raw panel, before
        any transform.  Nested runs P_0 (none) .. P_K (sessions d_1..d_K):
          - P_K vs P_0: every forecast before d_1 and the forecast on d_1 are
            bit-identical; the targets on d_1..d_K are LEAK_FACTOR x;
          - P_j vs P_{j-1} (j = 2..K): every forecast up to and including d_j
            is bit-identical -- session d_j's own post-entry bars are invisible
            at d_j's entry (later sessions legitimately see them through the
            HAR lags, so a single P_K-vs-P_0 comparison cannot test d_2..d_K).
        The spec asserts in code that the sessions of a segment are strictly
        increasing, so the trailing window X[t-W:t] holds only earlier sessions.
GATE 4  SMOKE: one clock (SMOKE_CLOCK) x live_feasible x each estimator on the
        real panel, full series (one chunk): wall time, rows, and mean QLIKE on
        the smoke rows against the honest one-step construction
        (experiments/ft_remaining.py: F_rem = f_next / w_next from the stored
        per-bar table of the same estimator) -- a sanity number only.

Run:  python experiments/gate_rest_of_day.py [--skip-leak] [--skip-smoke]
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "experiments", ROOT / "specs"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import causal_tune_rest_of_day as rod  # noqa: E402  (installs the executor hooks)
import ft_remaining  # noqa: E402

OUT = ROOT / "results" / "linear_subsection_restofday" / "local_gates"
TW = 2000
LEAK_K = 3  # perturbed sessions: >= 2 so the nested P_j-vs-P_{j-1} check runs
LEAK_FACTOR = 50.0
LEAK_CLOCK = "1200"
LEAK_BUCKET, LEAK_EST = "live_feasible", "ridge"
SMOKE_CLOCK = "1200"
SMOKE_BUCKET = "live_feasible"
SHORT = {"ridge": "ridge", "reclasso": "lasso", "reclasticnet": "enet"}


def gate_target() -> pd.DataFrame:
    rod.STATE["clock"] = "1530"
    rod.STATE["cache"] = {}
    df, _ = rod.executor.load_and_transform(
        "data",
        [],
        target_use_diurnal=True,
        target_winsor_window=240,
        dropna_with_exog=False,
        overnight_fill=True,
        impute_indicate=True,
        diurnal_mode="divide",
    )
    t = df["t"]
    mins = (t.dt.hour * 60 + t.dt.minute).to_numpy()
    day = t.dt.normalize()
    rv = pd.DataFrame({"day": day, "m": mins, "RV": df["RV"].to_numpy(float)})
    piv = rv.pivot_table(index="day", columns="m", values="RV", aggfunc="first")
    rows = []
    for c in rod.CLOCKS:
        first = rod.clock_minutes(c) + rod.BAR_MIN
        cols = list(range(first, rod.CLOSE_MIN + 1, rod.BAR_MIN))
        sub = piv.reindex(columns=cols)
        complete = sub.notna().all(axis=1)
        indep = pd.Series(
            [math.fsum(r) if ok else np.nan for r, ok in zip(sub.to_numpy(), complete)],
            index=sub.index,
        )
        spec = rod.rest_of_day(df, c)
        sd = spec["t"].dt.normalize()
        iv = indep.reindex(sd).to_numpy()
        sv = spec["RV_rem"].to_numpy()
        both = np.isfinite(iv) & np.isfinite(sv)
        exact = int((iv[both] == sv[both]).sum())
        nan_agree = bool((np.isnan(iv) == np.isnan(sv)).all())
        rows.append(
            dict(
                clock=c,
                rows=len(spec),
                complete=int(spec["complete"].sum()),
                incomplete=int((~spec["complete"]).sum()),
                compared=int(both.sum()),
                exact=exact,
                nan_pattern_agrees=nan_agree,
            )
        )
        if c == "1530":
            assert (sv == df.loc[spec.index, "RV"].to_numpy()).all()
    tab = pd.DataFrame(rows)
    print("GATE 2  TARGET (RV_rem vs independent pivot + fsum):")
    print(tab.to_string(index=False))
    assert (tab["exact"] == tab["compared"]).all() and tab["nan_pattern_agrees"].all()
    print("GATE 2  PASS: bit-exact on every compared row of all 13 clocks")
    return tab


def _perturber(days: list, clock: str):
    cmin = rod.clock_minutes(clock)

    def f(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        t = df["t"]
        m = (t.dt.hour * 60 + t.dt.minute) > cmin
        hit = m & t.dt.normalize().isin(days)
        df.loc[hit, "RV"] = df.loc[hit, "RV"] * LEAK_FACTOR
        return df

    return f


def gate_leak() -> None:
    root = OUT / "leak"
    runs = {}
    # the sessions carrying a clock row, from the unperturbed run
    rod.STATE["perturb"] = None
    base = rod.run_arm(
        LEAK_CLOCK, LEAK_BUCKET, LEAK_EST, results_root=str(root / "P0"), train_win=TW
    )
    runs[0] = base.set_index("date")
    days = sorted(pd.DatetimeIndex(runs[0].index).normalize().unique())[-LEAK_K:]
    tg0 = rod.STATE["targets"].set_index("t")
    for j in range(1, LEAK_K + 1):
        rod.STATE["perturb"] = _perturber(days[:j], LEAK_CLOCK)
        r = rod.run_arm(
            LEAK_CLOCK,
            LEAK_BUCKET,
            LEAK_EST,
            results_root=str(root / f"P{j}"),
            train_win=TW,
        )
        runs[j] = r.set_index("date")
        if j == LEAK_K:
            tgK = rod.STATE["targets"].set_index("t")
    rod.STATE["perturb"] = None

    def same_through(a, b, last_day, inclusive=True):
        idx = a.index[
            (a.index.normalize() <= last_day)
            if inclusive
            else (a.index.normalize() < last_day)
        ]
        assert idx.equals(b.index[: len(idx)])
        return bool(
            (
                a.loc[idx, "pred_adj"].to_numpy() == b.loc[idx, "pred_adj"].to_numpy()
            ).all()
        ), len(idx)

    d1 = days[0]
    ok, n = same_through(runs[LEAK_K], runs[0], d1)
    print(
        f"GATE 3  LEAK  sessions {[str(d.date()) for d in days]}, clock {LEAK_CLOCK}, "
        f"x{LEAK_FACTOR:g} on every bar after the clock"
    )
    print(f"  P{LEAK_K} vs P0: {n} forecasts through d_1 bit-identical: {ok}")
    assert ok
    after = runs[LEAK_K].index.normalize() > d1
    moved = int(
        (runs[LEAK_K].loc[after, "pred_adj"] != runs[0].loc[after, "pred_adj"]).sum()
    )
    print(
        f"  P{LEAK_K} vs P0: {moved} of {int(after.sum())} forecasts after d_1 move "
        "(legitimate: they see d_1's post-entry bars through the HAR lags)"
    )
    for j in range(2, LEAK_K + 1):
        ok, n = same_through(runs[j], runs[j - 1], days[j - 1])
        print(f"  P{j} vs P{j - 1}: {n} forecasts through d_{j} bit-identical: {ok}")
        assert ok
    kd = tgK.index.normalize().isin(days)
    ratio = (tgK.loc[kd, "RV_rem"] / tg0.loc[tgK.index[kd], "RV_rem"]).to_numpy()
    print(
        f"  targets on the perturbed sessions: RV_rem ratio {ratio.min():.15g} .. {ratio.max():.15g} "
        f"(= x{LEAK_FACTOR:g}); targets before d_1 unchanged: "
        f"{bool((tgK.loc[tgK.index.normalize() < d1, 'RV_rem'].fillna(-1) == tg0.loc[tg0.index.normalize() < d1, 'RV_rem'].fillna(-1)).all())}"
    )
    # bound: one rounding per scaled summand (<= 13 bars) plus the two exactly-rounded
    # fsums, each <= eps relative -> |ratio/LEAK_FACTOR - 1| <= (13 + 2) eps
    bound = (len(rod.CLOCKS) + 2) * np.finfo(float).eps
    print(
        f"  max |ratio/x - 1| = {np.abs(ratio / LEAK_FACTOR - 1).max():.2e} (bound {bound:.1e})"
    )
    assert np.abs(ratio / LEAK_FACTOR - 1).max() <= bound
    print("GATE 3  PASS")


def qlike(y: np.ndarray, f: np.ndarray) -> np.ndarray:
    r = y / f
    return r - np.log(r) - 1.0


def gate_smoke() -> pd.DataFrame:
    root = OUT / "smoke"
    rows = []
    for est in ("ridge", "reclasso", "reclasticnet"):
        t0 = time.perf_counter()
        res = rod.run_arm(
            SMOKE_CLOCK, SMOKE_BUCKET, est, results_root=str(root), train_win=TW
        )
        wall = time.perf_counter() - t0
        tg = rod.STATE["targets"].set_index("t")
        r = res.set_index("date")
        y = tg.loc[r.index, "RV_rem"].to_numpy(float)
        f_dir = r["pred_adj"].to_numpy(float) ** 2 * tg.loc[
            r.index, "baseline_rem"
        ].to_numpy(float)
        ft = ft_remaining.ft_remaining(
            str(
                ROOT
                / "results"
                / "spxw_pnl"
                / f"yhat_sub_{SHORT[est]}_{SMOKE_BUCKET}.parquet"
            )
        )
        hh = f"{SMOKE_CLOCK[:2]}:{SMOKE_CLOCK[2:]}"
        ft = ft[ft["clock"] == hh].set_index("day")["F_rem"]
        f_one = ft.reindex(r.index.normalize()).to_numpy(float)
        ok = (
            np.isfinite(y)
            & (y > 0)
            & np.isfinite(f_dir)
            & (f_dir > 0)
            & np.isfinite(f_one)
            & (f_one > 0)
        )
        rows.append(
            dict(
                estimator=est,
                clock=SMOKE_CLOCK,
                bucket=SMOKE_BUCKET,
                wall_s=round(wall, 1),
                oos_rows=len(r),
                scored=int(ok.sum()),
                qlike_direct_plain=float(qlike(y[ok], f_dir[ok]).mean()),
                qlike_onestep_frem=float(qlike(y[ok], f_one[ok]).mean()),
                first=str(r.index.min().date()),
                last=str(r.index.max().date()),
            )
        )
        print(rows[-1])
    tab = pd.DataFrame(rows)
    print(
        "GATE 4  SMOKE (sanity only; plain = pred_adj^2 x baseline_rem, no recalibration):"
    )
    print(tab.to_string(index=False))
    return tab


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-leak", action="store_true")
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--skip-target", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not a.skip_target:
        gate_target().to_csv(OUT / "gate2_target.csv", index=False)
    rod.STATE["cache"] = {}
    if not a.skip_smoke:
        gate_smoke().to_csv(OUT / "gate4_smoke.csv", index=False)
    if not a.skip_leak:
        gate_leak()


if __name__ == "__main__":
    main()
