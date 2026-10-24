"""The pooled linear runner (experiments/pool_linear_pretune.py) for the three-way training-sample comparison
(results/linear_train3way_2014start_w3000/RULE.txt), run unchanged with three additions made from outside the module:

  * --sample close: segment bar1600 of src/backtest/segmentation.py (the 15:30-16:00 ET half hour, one row a session),
    lag scope global as rth13 (the HAR components and trailing averages on the full 48-interval grid), 1 row a day;
  * --sample close_r: the 16:00 rows of a rth13 design cache (experiments/linear_train3way.py derive_close), for the
    design-cache modes only (pretune, walk with --design); 1 row a day;
  * mode design also writes dates_all.csv, the date label of every design row (the rows _backtest_and_save hands to
    the walk, rows 0 .. W - 1 included), captured from the output of src.backtest.executor.apply_horizon_shift; every
    array the design stores is unchanged.
  * --window-rows N (2026-10-23 addendum, equal training rows): the rolling window is N rows instead of
    --window-days x rows a day, for the design-cache modes only (pretune, walk with --design: a cache derived by
    experiments/linear_train3way.py derive_window, whose config records window = N).  The runner's W is the only
    window those modes use (cached_backtest, the walk's MultiStageBacktest, the pretune's fit block, config_dict);
    the spec's TRAIN_WIN (days) is read only by the design path.
  * --refit-rows all | scored@YYYY-MM-DD (2026-10-23 addendum B, owner: "skip the refits at the overnight rows"): with
    scored@D the walk refits only at the scored origins -- rows whose end stamp is a regular-hours bar (10:00 .. 16:00)
    on or after day D -- and writes NaN as the forecast of every other row.  The window is unchanged (the trailing W
    rows of every origin): every row still enters and leaves the window's state in order; between two scored origins
    the rows in between move the state as data only (OLS and ridge: their rank-one sums, as always; lasso / elastic
    net: Gr += s u u', c += s u w without the homotopy's path tracking, i.e. the overnight rows as one multi-row update
    of the problem), and at the next scored origin the lasso / elastic net solution is restored warm from the last one:
    reclasso_har.walk_kkt on the current (Gr, c, mu); if it fails, pool_linear_pretune._active_repair (support changes,
    exact KKT solve), else _cd_polish, else the cold seed (_seed), all at the KKT tolerance of WALK_KKT_RESEED; then
    the walk's own solve and KKT check / reseed as before.  Every event tied to the row count stays on its row: the
    identifiability mask and cold seed every 250th row (the tune), the between-seed trackers (degenerate columns,
    evidence, binomial rule) on every row, the OLS's exact rebuild every 250 rows.  OLS and ridge forecasts at the
    scored origins are therefore the full-refit walk's up to floating point (the same state, the same solve); lasso /
    elastic net solve the same convex problem (same Gr, c up to the homotopy's segmented accumulation) to the KKT
    tolerance.  The record of the walk: refit_rows.json in --out.  all (or no option): the runner as before.  The option
    is read and removed here, so the runner itself never sees it; it acts on mode walk only.

usage: python experiments/linear_train3way_runner.py <the runner's arguments>
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(HERE), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pool_linear_pretune as P  # noqa: E402

P.SEGMENT_OF["close"] = "bar1600"
P.PPD_OF["close"] = 1
P.SEGMENT_OF["close_r"] = "bar1600"
P.PPD_OF["close_r"] = 1


def _arg(name: str, default: str = "") -> str:
    a = sys.argv
    return a[a.index(name) + 1] if name in a else default


def _pop_all(name: str) -> str:
    """The value of the FIRST occurrence of option name (default ''), every occurrence removed from sys.argv."""
    val = ""
    while name in sys.argv:
        i = sys.argv.index(name)
        val = val or sys.argv[i + 1]
        del sys.argv[i : i + 2]
    return val


REFIT: dict = {}  # addendum B: the scored-only walk's record (refit_rows.json)


def scored_backtest(Base, scored_of):
    """addendum B: MultiStageBacktest whose incremental path refits only at the scored origins (module docstring)."""
    import numpy as np

    import src.models.reclasso_har as RH

    def data_only(Gr, c, mu_vec, u, w, sigma, theta, A, s, locked, **kw):
        Gr += sigma * np.outer(u, u)
        c += sigma * u * w
        return theta, A, s

    def resolve(reg) -> None:
        """The lasso / elastic-net solution of the current (Gr, c, mu), warm from the last one."""
        REFIT["resolves"] += 1
        Gr, c, mu = reg._Gr, reg._c, reg._mu_vec
        ok, _, _ = RH.walk_kkt(Gr, c, mu, reg._th)
        if ok:
            REFIT["resolve_kkt_ok"] += 1
        else:
            tol = RH.KKT_TOL_REL * max(float(np.abs(c).max()), 1.0)
            z = np.zeros(len(c))
            b = P._active_repair(Gr, c, mu, z, reg._th, tol)
            if b is not None:
                REFIT["resolve_active_repair"] += 1
            else:
                b = P._cd_polish(Gr, c, mu, z, reg._th, tol)
                if b is not None:
                    REFIT["resolve_cd_polish"] += 1
            if b is None:
                REFIT["resolve_cold_seed"] += 1
                reg._seed()
            else:
                reg._th = b
                reg._A = list(np.flatnonzero((b != 0.0) | reg._locked))
                reg._s = [0.0 if reg._locked[j] else float(np.sign(b[j])) for j in reg._A]
        reg._warm = True  # the walk's KKT check follows in solve (reseed if it fails)

    class ScoredOnly(Base):
        def _run_incremental(self, X, y, train_win, regressor, *, desc):
            n_test = X.shape[0] - train_win
            sc = scored_of(n_test)
            pred = np.full(n_test, np.nan)
            reinit = int(getattr(regressor, "reinit_every", 0))
            regressor.init_window(X[0:train_win], y[0:train_win])
            regressor.solve()
            if sc[0]:
                pred[0] = regressor.predict_one(X[train_win])
            penal = hasattr(regressor, "_n_solve")  # the penalized class (the OLS class has no solve counter)
            tune_per = REFIT["tune_per"]
            stale = False
            orig = RH.enet_online
            for i in range(1, n_test):
                t = train_win + i
                enet = penal and regressor.kind_ != "ridge"
                if reinit and i % reinit == 0:
                    regressor.init_window(X[t - train_win : t], y[t - train_win : t])
                elif enet and (stale or not sc[i]):
                    RH.enet_online = data_only
                    try:
                        regressor.roll(X[t - 1], y[t - 1], X[t - 1 - train_win], y[t - 1 - train_win])
                    finally:
                        RH.enet_online = orig
                    if regressor._warm:  # a data-only update (no tracker-triggered cold seed on this row)
                        stale = True
                    else:
                        stale = False
                        REFIT["tracker_seeds_in_gaps"] += 1
                else:
                    regressor.roll(X[t - 1], y[t - 1], X[t - 1 - train_win], y[t - 1 - train_win])
                tune = penal and regressor._n_solve > 0 and regressor._n_solve % tune_per == 0
                if sc[i] or tune:
                    if enet and stale and not tune:
                        resolve(regressor)
                    regressor.solve()
                    stale = False
                    REFIT["solves"] += 1
                    REFIT["tune_solves_unscored"] += int(tune and not sc[i])
                else:
                    if penal:
                        regressor._n_solve += 1
                    REFIT["skipped"] += 1
                if sc[i]:
                    pred[i] = regressor.predict_one(X[t])
            REFIT["rows"] = int(n_test)
            REFIT["scored"] = int(sc.sum())
            return pred

    return ScoredOnly


def refit_patch(spec: str) -> None:
    """addendum B: --refit-rows scored@D for mode walk (the walk's MultiStageBacktest replaced in the spec section)."""
    import numpy as np
    import pandas as pd

    first = pd.Timestamp(spec.split("@", 1)[1])
    start = int(_arg("--start"))
    dts = pd.to_datetime(pd.read_csv(Path(_arg("--design")) / "dates_oos.csv", dtype=str)["date"])
    clock = dts.dt.strftime("%H:%M").to_numpy()
    ok = np.isin(clock, sorted(P.RTH)) & (dts >= first).to_numpy()

    def scored_of(n_test: int) -> np.ndarray:
        o = start - P.W  # the OOS index of the walk's first forecast (halo = W)
        assert o >= 0 and o + n_test <= len(ok), (o, n_test, len(ok))
        return ok[o : o + n_test]

    REFIT.update(
        mode=spec,
        resolves=0,
        resolve_kkt_ok=0,
        resolve_active_repair=0,
        resolve_cd_polish=0,
        resolve_cold_seed=0,
        tracker_seeds_in_gaps=0,
        solves=0,
        tune_solves_unscored=0,
        skipped=0,
    )
    sec = P.spec_section

    def spec_section(*a, **k):
        ns = sec(*a, **k)
        REFIT["tune_per"] = int(ns["TUNE_PER"])
        ns["MultiStageBacktest"] = scored_backtest(ns["MultiStageBacktest"], scored_of)
        return ns

    P.spec_section = spec_section


def main() -> None:
    mode, sample = sys.argv[1], _arg("--sample", "48")
    if sample == "close_r":
        assert mode in ("pretune", "walk") and _arg("--design"), "close_r: a derived design cache, pretune / walk only"
    refit = _pop_all("--refit-rows")  # addendum B: read and removed in every mode, acts on walk only
    if mode == "walk" and refit.startswith("scored@"):
        assert _arg("--design") and int(_arg("--halo", "0")) > 0, "--refit-rows scored: a design-cache walk with a halo"
        refit_patch(refit)
    else:
        assert refit in ("", "all") or refit.startswith("scored@"), refit
    rows = _arg("--window-rows")
    if rows:  # 2026-10-23 addendum: a row-count window on a derived design cache
        assert mode in ("pretune", "walk") and _arg("--design"), "--window-rows: a derived design cache only"
        i = sys.argv.index("--window-rows")
        del sys.argv[i : i + 2]
        cfg = P.configure

        def configure(a) -> None:
            cfg(a)
            P.W = int(rows)

        P.configure = configure
    if mode != "design":
        P.main()
        if REFIT:  # addendum B: the scored-only walk's record
            import json

            out = Path(_arg("--out")).resolve()
            (out / "refit_rows.json").write_text(json.dumps({k: v for k, v in REFIT.items()}, indent=1))
        return
    import numpy as np
    import pandas as pd

    import src.backtest.executor as EX

    orig = EX.apply_horizon_shift
    cap: dict = {}

    def spy(X, y, dates, baselines, horizon):
        r = orig(X, y, dates, baselines, horizon)
        cap.setdefault("dates", []).append(pd.Series(r[2]).reset_index(drop=True))
        return r

    EX.apply_horizon_shift = spy
    try:
        P.main()
    finally:
        EX.apply_horizon_shift = orig
    assert len(cap["dates"]) == 1, len(cap["dates"])
    out = Path(_arg("--out")).resolve()
    d = pd.to_datetime(cap["dates"][0]).dt.strftime("%Y-%m-%d %H:%M:%S")
    n = int(np.load(out / "X.npy", mmap_mode="r").shape[0])
    oos = pd.read_csv(out / "dates_oos.csv", dtype=str)["date"]
    assert len(d) == n and d.iloc[n - len(oos) :].reset_index(drop=True).equals(oos), (len(d), n, len(oos))
    pd.DataFrame({"date": d}).to_csv(out / "dates_all.csv", index=False)
    print(f"dates_all.csv: {n} rows, {d.iloc[0]} .. {d.iloc[-1]}", flush=True)


if __name__ == "__main__":
    main()
