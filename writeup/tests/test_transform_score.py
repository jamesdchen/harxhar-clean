"""Tests of the Part A scorer experiments/transform_score.py (results/transform_ablation_2026-10-30/RULE.txt sections
4-7, addenda D-E), on V0 = the paper's final forecasts (the committed Table 1 files) and on synthetic variant files built
from them, whose forecasts are known exactly:
  V1, V1_frz  sqrt map, pred x sqrt(0.8)            -> 0.8 x V0's plain; under C (the causal scalar c) equal to V0
  V3, V3f     log map, pred = log(pred0^2)          -> exp(pred) B = V0's plain
  V9          level map, pred = pred0^2             -> max(pred, 0.01) B: V0's plain except the floored bars
  V10         sqrt map with B = 1, pred = pred0 sqrt(B0) -> V0's plain
and synthetic 2013 selection forecasts (LightGBM files; a linear pre-tune with its design cache) with a known ranking.

Run:  python -m pytest writeup/tests/test_transform_score.py   (reads the committed forecast files and
data/core_stats.parquet; about a minute)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "experiments"), str(ROOT / "notebooks")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ablate_score as AS  # noqa: E402
import transform_backtransform as TB  # noqa: E402
import transform_score as TS  # noqa: E402

SEL_C = {"V5": 1.0, "V2": 1.5, "V3": 2.0, "V1": 3.0, "V4": 5.0}  # 2013 LightGBM: forecast = c x mean RV (V5 best)


def v0_long() -> pd.DataFrame:
    f = pd.concat(
        [pd.read_parquet(p, columns=["model", "predictor_set", "date", "pred_adj", "B"]) for p in TB.FILES],
        ignore_index=True,
    )
    return f


def write(run: Path, f: pd.DataFrame, name: str = "forecasts.parquet") -> None:
    run.mkdir(parents=True, exist_ok=True)
    f.to_parquet(run / name, index=False)


@pytest.fixture(scope="module")
def scored(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("tscore")
    runs, out = tmp / "runs", tmp / "out"
    f0 = v0_long()
    p0, b0 = f0["pred_adj"].to_numpy(), f0["B"].to_numpy()
    with np.errstate(divide="ignore"):
        lg = np.log(p0**2)
    write(runs / "V1", f0.assign(pred_adj=p0 * np.sqrt(0.8)))
    write(runs / "V1_frz", f0.assign(pred_adj=p0 * np.sqrt(0.8)))
    write(runs / "V3", f0.assign(pred_adj=lg))
    write(runs / "V3f", f0.assign(pred_adj=lg))
    write(runs / "V9", f0.assign(pred_adj=p0**2))
    write(runs / "V10", f0.assign(pred_adj=p0 * np.sqrt(b0), B=1.0))
    # 2013 selection: LightGBM full set, forecast = c x the 2013 mean RV of the bar's slot-free mean (constant)
    rv = AS.core_rv()
    t = rv.index[(rv.index >= "2013-01-02") & (rv.index <= "2013-12-31 23:59")]
    t = t[(t.hour * 60 + t.minute >= 600) & (t.hour * 60 + t.minute <= 960)]
    m = float(rv.reindex(t)[rv.reindex(t) > 0].mean())
    Bc = 1e-6
    for k, c in SEL_C.items():
        kind = TS.inverse_kind(k)
        val = c * m / Bc
        pred = {"sqrt": np.sqrt(val), "log": np.log(val), "level": val}[kind]
        write(
            runs / k,
            pd.DataFrame(dict(model="lgbm", predictor_set="all_features", date=t, pred_adj=pred, B=Bc)),
            "forecasts_sel2013_lgbm.parquet",
        )
    # V4 / V5 (predictor-side) need confirmation forecasts too: V0's own
    write(runs / "V4", f0)
    write(runs / "V5", f0)
    # 2013 linear pre-tune of V2 (ridge, full set): a design cache, the fold, the selected penalty
    run = runs / "V2"
    write(run, f0)  # V2's confirmation forecasts: V0's own
    dz = run / "_work" / "design_all_features"
    dz.mkdir(parents=True)
    W = 50
    rng = np.random.default_rng(3)
    y = rng.normal(1.0, 0.3, W + len(t))
    b = rng.uniform(0.5e-6, 2e-6, len(t))
    np.save(dz / "y.npy", y)
    np.save(dz / "b_oos.npy", b)
    pd.DataFrame({"date": t.strftime("%Y-%m-%d %H:%M:%S")}).to_csv(dz / "dates_oos.csv", index=False)
    (dz / "design.json").write_text(json.dumps({"config": {"ex_flags": {"TARGET_WINSOR_5_95": True}}}))
    grid = np.array([1.0, 10.0, 100.0])
    resid = rng.normal(0, 0.1, (1, 3, len(t)))
    pt = run / "_work" / "pretune" / "all_features" / "ridge"
    pt.mkdir(parents=True)
    np.savez(pt / "fold.npz", resid=resid, grid=grid, val=np.array([[W, W + len(t)]]), fit=np.array([[0, W]]))
    (run / "frozen_penalties.json").write_text(json.dumps({"ridge": {"all_features": 10.0}}))
    rc = TS.main(["--runs", str(runs), "--out", str(out), "--mcs-B", "200"])
    return dict(rc=rc, out=out, runs=runs, f0=f0, t=t, m=m, y=y, b=b, resid=resid, W=W)


def table1_bars() -> tuple[pd.DatetimeIndex, pd.Series]:
    rv = AS.core_rv()
    panel = {k: pd.Series(v["plain"].to_numpy(), index=v.index) for k, v in TB.load().items()}
    return AS.scored(panel, rv, AS.evaluation_days(rv)), rv


def metrics(s) -> pd.DataFrame:
    return pd.read_csv(s["out"] / "metrics.csv")


def q(M: pd.DataFrame, key: str, sc: str = "P", span: str = "all") -> pd.Series:
    g = M[(M["key"] == key) & (M["scoring"] == sc) & (M["span"] == span)]
    return g.set_index(["cls", "set"])["qlike"].sort_index()


def test_runs_and_reproduces_table1(scored):
    assert scored["rc"] == 0
    ck = json.loads((scored["out"] / "checks.json").read_text())
    assert ck["pass"] and ck["table1_values_checked"] == 39
    assert ck["spans"]["all"] == {"bars": 33527, "days": 2579} and ck["spans"]["ev"]["days"] == 81
    t1 = pd.read_csv(TB.TABLE1)
    M = metrics(scored)
    v0 = q(M, "V0")
    for _, r in t1.iterrows():
        assert v0.loc[(r["model"], TB.SET_ARM[r["set"]])] == pytest.approx(r["qlike"], rel=1e-12, abs=0)


@pytest.mark.parametrize("key", ["V3", "V3f", "V10"])
def test_inverse_maps_recover_plain(scored, key):
    M = metrics(scored)
    for sc in ("P", "C"):
        for span in ("all", "ev", "c16"):
            a, b = q(M, key, sc, span), q(M, "V0", sc, span)
            assert len(a) == 13 and np.allclose(a.to_numpy(), b.to_numpy(), rtol=1e-11, atol=0), (key, sc, span)


def test_sqrt_scale_and_causal_c_invariance(scored):
    M = metrics(scored)
    idx, rv = table1_bars()
    f0 = scored["f0"]
    g = f0[(f0["model"] == "ridge") & (f0["predictor_set"] == "all_features")]
    s = pd.Series(0.8 * g["pred_adj"].to_numpy() ** 2 * g["B"].to_numpy(), index=pd.DatetimeIndex(g["date"]))
    direct = float(TB.qlike_se(s.reindex(idx).to_numpy(), rv.reindex(idx).to_numpy())[0].mean())
    p1 = q(M, "V1")
    assert p1.loc[("ridge", "full")] == pytest.approx(direct, rel=1e-12, abs=0)
    # the causal scalar c removes a constant level factor once it exists (from the 21st session; the burn-in bars stay
    # plain): V1 under C is nearer V0 under C than under P, and on the function level equal after the burn-in
    gap_c = np.abs(q(M, "V1", "C").to_numpy() / q(M, "V0", "C").to_numpy() - 1)
    gap_p = np.abs(q(M, "V1", "P").to_numpy() / q(M, "V0", "P").to_numpy() - 1)
    assert (gap_c < gap_p / 10).all()
    f = s.to_numpy() / 0.8
    sd = pd.DatetimeIndex(s.index.normalize())
    ss = sd.unique()
    sess = ss.get_indexer(sd)
    r = rv.reindex(s.index).to_numpy(float)
    a, b = TS.scalar_c(0.8 * f, r, sess, len(ss)), TS.scalar_c(f, r, sess, len(ss))
    late = sess >= TS.MIN_HIST
    assert np.allclose(a[late], b[late], rtol=1e-12, atol=0) and np.allclose(a[~late], 0.8 * b[~late], rtol=1e-15)
    # frozen-penalty keys are scored with the variant's map
    assert np.array_equal(q(M, "V1_frz").to_numpy(), p1.to_numpy())


def test_level_map_floor(scored):
    M = metrics(scored)
    ck = json.loads((scored["out"] / "checks.json").read_text())
    f0 = scored["f0"]
    g = f0[(f0["model"] == "lgbm") & (f0["predictor_set"] == "all_features")]
    g = g[g["date"] >= TB.FIRST]
    n_low = int((g["pred_adj"].to_numpy() ** 2 < 0.01).sum())
    assert ck["level_floored_bars"].get("V9/lgbm/full", 0) == n_low
    idx, rv = table1_bars()
    f = pd.Series(
        np.maximum(g["pred_adj"].to_numpy() ** 2, 0.01) * g["B"].to_numpy(), index=pd.DatetimeIndex(g["date"])
    )
    direct = float(TB.qlike_se(f.reindex(idx).to_numpy(), rv.reindex(idx).to_numpy())[0].mean())
    assert q(M, "V9").loc[("lgbm", "full")] == pytest.approx(direct, rel=1e-12, abs=0)


def test_dm_verdicts_mcs(scored):
    D = pd.read_csv(scored["out"] / "dm.csv")
    d3 = D[(D["key"] == "V3") & (D["loss"] == "qlike") & (D["span"] == "all")]
    assert len(d3) == 26 and np.allclose(d3["mean_diff"], 0, atol=1e-12)
    V = pd.read_csv(scored["out"] / "verdicts.csv")
    v1 = V[(V["key"] == "V1")]
    assert len(v1) == 13 and v1["verdict_applies"].all()
    assert (v1["v0_vs_variant_pct_P"] < 0).all()  # V1 = 0.8 x plain: worse than V0 under P in every class
    d1c = D[(D["key"] == "V1") & (D["loss"] == "qlike") & (D["span"] == "all") & (D["scoring"] == "C")]
    d1p = D[(D["key"] == "V1") & (D["loss"] == "qlike") & (D["span"] == "all") & (D["scoring"] == "P")]
    assert len(d1c) == 13 and (d1c["mean_diff"].abs().to_numpy() < d1p["mean_diff"].abs().to_numpy() / 10).all()
    assert set(V[V["key"].isin(["V9", "V10", "V1_frz"])]["verdict_applies"]) == {False}
    S = pd.read_csv(scored["out"] / "mcs.csv")
    prim = S[(S["family"] == "primary") & (S["block"] == 10) & (S["alpha"] == 0.10)]
    assert len(prim) == 26  # 13 classes x 2 scorings
    assert set(prim.iloc[0]["members"].split(",")) == {"V0", "V1", "V2", "V3", "V3f", "V4", "V5"}
    sec = S[S["family"] == "secondary"]
    assert len(sec) and all("V9" in m and "V10" in m and "V1_frz" not in m for m in sec["members"])


def test_selection_2013(scored):
    sel = json.loads((scored["out"] / "selection.json").read_text())
    assert sel["chosen"] == ["V5", "V2", "V3"], sel
    SM = pd.read_csv(scored["out"] / "sel2013_metrics.csv")
    # the linear pre-tune: pred = y + resid at the selected penalty (10 = grid index 1), B from b_oos
    t, y, b, resid, W = scored["t"], scored["y"], scored["b"], scored["resid"], scored["W"]
    pred = y[W:] + resid[0, 1, :]
    rv = AS.core_rv()
    days = TS.eval_days_between(rv, TS.SEL_FIRST, TS.SEL_LAST)
    f = pd.Series(pred**2 * b, index=t)  # V2: sqrt map (5/95 is target-side, map unchanged)
    ok = f.index[(f > 0).to_numpy() & np.isfinite(f.to_numpy())]
    ok = ok[pd.DatetimeIndex(ok.normalize()).isin(days)]
    rr = rv.reindex(ok)
    ok = ok[(rr > 0).to_numpy() & np.isfinite(rr.to_numpy())]
    direct = float(TB.qlike_se(f.reindex(ok).to_numpy(), rv.reindex(ok).to_numpy())[0].mean())
    row = SM[(SM["key"] == "V2") & (SM["cls"] == "ridge") & (SM["set"] == "full")]
    assert len(row) == 1 and row["qlike"].iloc[0] == pytest.approx(direct, rel=1e-12, abs=0)


def test_scalar_c_equals_part_b():
    """transform_score.scalar_c on pred^2 B is transform_backtransform.corrections' c, bit for bit."""
    panel = TB.load()
    f = panel[("ridge", "full")]
    rv = AS.core_rv().reindex(f.index).to_numpy(float)
    sd = pd.DatetimeIndex(f.index.normalize())
    ss = sd.unique()
    sess = ss.get_indexer(sd)
    tod = f.index.hour * 60 + f.index.minute
    slot = np.searchsorted(np.sort(np.unique(tod)), tod)
    fc, _ = TB.corrections(
        f["pred"].to_numpy(), f["true_adj"].to_numpy(), f["B"].to_numpy(), rv, sess, slot, len(ss), 13
    )
    mine = TS.scalar_c(f["pred"].to_numpy() ** 2 * f["B"].to_numpy(), rv, sess, len(ss))
    assert np.array_equal(mine, fc["c"])


def test_design_flags_checked(tmp_path):
    run = tmp_path / "V4"
    dz = run / "_work" / "design_baseline"
    dz.mkdir(parents=True)
    (dz / "design.json").write_text(json.dumps({"config": {"ex_flags": {"EXOG_DIURNAL": True}}}))
    assert TS.design_flags_ok(run, "V4")  # V4 needs EXOG_DIURNAL False
    (dz / "design.json").write_text(json.dumps({"config": {"ex_flags": {"EXOG_DIURNAL": False, "PRESCALE": False}}}))
    assert TS.design_flags_ok(run, "V4")  # a switch beyond the variant's
    (dz / "design.json").write_text(
        json.dumps({"config": {"ex_flags": {"EXOG_DIURNAL": False, "LATE_START_WARMUP": True}}})
    )
    assert not TS.design_flags_ok(run, "V4")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
