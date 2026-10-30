"""Tests of the all-48-interval study's code (results/all48_2026-10-30/RULE.txt; exploratory) on small synthetic samples
only (no design cache, walk table or forecast file of the study is read):
  experiments/all48_score.py   spans, the common scored set, QLIKE / MSE / DM against a hand computation, chunks stitched
                               and de-duplicated, B from the design cache or true_raw / true_adj^2, the checks (coverage,
                               48 end stamps, design B, train3way's reference scores) and their exit code
  experiments/all48_linear.py  the orchestrator's globals for the 48-interval sample (and nothing changed before
                               configure()), check_design, the 19 walk jobs from 250-solve boundaries
  experiments/all48_trees.py   a refit at every interval origin (overnight, Friday-evening and Sunday rows included),
                               pieces = contiguous blocks reproducing the one-piece walk bit for bit and a direct fit,
                               pieces <= the target hours, the merge's checks, gates T1 / T2
  default behaviour            importing the new modules leaves experiments/pooled_linear_rerun_2026_10_19_13.py and
                               experiments/pooled_trees_rerun_13.py as they are (their defaults unchanged)

Run:  python -m pytest writeup/tests/test_all48.py   (LightGBM needed for the tree tests; well under a minute)
"""

from __future__ import annotations

import importlib
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

import all48_score as SC  # noqa: E402
from src.evaluation.diebold_mariano import dm_test  # noqa: E402

FIRST, LAST = "2014-01-02", "2014-02-28"


# ============================================================================ a synthetic 48-interval grid
def grid(start: str = "2013-12-23", end: str = "2014-02-28") -> pd.DatetimeIndex:
    """Bar-end stamps as the grid has them: Mon-Thu 48 (00:00 .. 23:30), Friday 00:00 .. 17:00, Sunday 18:30 .. 23:30,
    no Saturday; the holidays 2014-01-01 and 2014-01-20 have no row after 13:00 except the evening reopen."""
    out = []
    for d in pd.date_range(start, end, freq="D"):
        dow = d.dayofweek
        for k in range(48):
            t = d + pd.Timedelta(minutes=30 * k)
            hm = t.strftime("%H:%M")
            if dow == 5 or (dow == 4 and hm > "17:00") or (dow == 6 and hm < "18:30"):
                continue
            if str(d.date()) in ("2014-01-01", "2014-01-20") and "13:00" < hm < "18:30":
                continue
            out.append(t)
    return pd.DatetimeIndex(out)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """A core_stats file, a linear run in the orchestrator's layout (7 models, chunked full-set ridge with one
    duplicated row), its design cache's b_oos, and a trees forecast file."""
    tmp = tmp_path_factory.mktemp("all48")
    rng = np.random.default_rng(7)
    t = grid()
    rv = np.exp(rng.normal(-12.0, 1.0, len(t)))
    clk = np.asarray(t.strftime("%H:%M"))
    rv[(clk == "17:30") & (np.arange(len(t)) % 3 == 0)] = 0.0  # some zero prints (not scored)
    core = tmp / "core_stats.parquet"
    pd.DataFrame({"endbartime": t, "sumret2": rv}).to_parquet(core, index=False)
    # the design's OOS rows: every grid row from 2013-12-26 on; B a positive diurnal shape
    oos = t[t >= "2013-12-26"]
    hours = np.asarray(oos.hour + oos.minute / 60.0)
    B = np.exp(-12.0 + 0.5 * np.sin(hours / 24 * 2 * np.pi)) * (1 + 0.1 * rng.random(len(oos)))
    work = tmp / "lin" / "_work"
    dz = work / "design_baseline"
    dz.mkdir(parents=True)
    np.save(dz / "b_oos.npy", B)
    pd.DataFrame({"date": oos.strftime("%Y-%m-%d %H:%M:%S")}).to_csv(dz / "dates_oos.csv", index=False)
    rvs = pd.Series(rv, index=t).reindex(oos).to_numpy()
    true_adj = np.sqrt(rvs / B)
    true_raw = true_adj**2 * B
    walk_start = 40  # the walk starts at OOS row 40 (a 250-solve boundary in a real run)
    sl = slice(walk_start, None)
    for s, m in SC.LIN_MODELS:
        pred = true_adj * np.exp(rng.normal(0.0, 0.3, len(oos)))
        f = pd.DataFrame(
            {
                "date": oos.strftime("%Y-%m-%d %H:%M:%S")[sl],
                "horizon": 1,
                "true_adj": true_adj[sl],
                "pred_adj": pred[sl],
                "true_raw": true_raw[sl],
                "pred_raw": pred[sl] ** 2 * B[sl],
            }
        )
        base = work / "walk" / s / m
        tab = f"causal_tune_linear/{SC.LIN_RUNNER[m]}/{s}/results.csv"
        if (s, m) == ("all_features", "ridge"):  # two chunks, the seam row written twice (identically)
            h = len(f) // 2
            for c, part in enumerate((f.iloc[: h + 1], f.iloc[h:])):
                (base / f"c{c}" / tab).parent.mkdir(parents=True)
                part.to_csv(base / f"c{c}" / tab, index=False)
        else:
            (base / tab).parent.mkdir(parents=True)
            f.to_csv(base / tab, index=False)
    # trees: every grid row from 2014-01-02 on
    to = oos[oos >= FIRST]
    bo = pd.Series(B, index=oos).reindex(to).to_numpy()
    ta = pd.Series(true_adj, index=oos).reindex(to).to_numpy()
    parts = []
    for m in ("lgbm", "xgb", "rf_skx"):
        for s in ("baseline", "all_features"):
            pa = ta * np.exp(rng.normal(0.0, 0.25, len(to)))
            parts.append(pd.DataFrame(dict(model=m, predictor_set=s, date=to, pred_adj=pa, true_adj=ta, B=bo)))
    trees = tmp / "trees.parquet"
    pd.concat(parts, ignore_index=True).to_parquet(trees, index=False)
    return dict(tmp=tmp, core=core, lin=tmp / "lin", trees=trees, oos=oos, B=B, walk_start=walk_start)


def run_score(world, out: Path, *extra: str) -> int:
    return SC.main(
        ["--out", str(out), "--part", "2", "--core", str(world["core"]), "--first", FIRST, "--last", LAST, *extra]
    )


@pytest.fixture(scope="module")
def scored(world):
    out = world["tmp"] / "score_all"
    rc = run_score(world, out, "--linear-root", str(world["lin"]), "--trees", str(world["trees"]))
    assert rc == 0
    return out


def manual_panel(world) -> dict:
    """The panel recomputed by hand from the files (plain = pred_adj^2 x B)."""
    oos, B = world["oos"], pd.Series(world["B"], index=world["oos"])
    pan = {}
    for s, m in SC.LIN_MODELS:
        base = world["lin"] / "_work" / "walk" / s / m
        fs = sorted(base.rglob("results.csv"))
        r = pd.concat([pd.read_csv(f, parse_dates=["date"]) for f in fs]).drop_duplicates("date").set_index("date")
        pan[(m, SC.SET_KEY[s])] = r["pred_adj"] ** 2 * B.reindex(r.index)
    tr = pd.read_parquet(world["trees"])
    for (m, s), g in tr.groupby(["model", "predictor_set"]):
        g = g.set_index("date")
        pan[(SC.CLS_OF[str(m)], SC.SET_KEY[str(s)])] = g["pred_adj"] ** 2 * g["B"]
    assert oos.is_unique
    return pan


# ============================================================================ scorer
def test_scorer_matches_hand_computation(world, scored):
    M = pd.read_csv(scored / "metrics.csv")
    D = pd.read_csv(scored / "dm.csv")
    ck = json.loads((scored / "checks.json").read_text())
    assert ck["pass"] and len(ck["panel"]) == 13
    rv = SC.core_rv(world["core"])
    days = SC.evaluation_days(rv, FIRST, LAST)
    assert not days.isin(pd.to_datetime(["2014-01-01", "2014-01-20"])).any()  # no 16:00 print: not in E
    pan = manual_panel(world)
    idx = None
    for f in pan.values():
        ok = f.index[np.isfinite(f.to_numpy()) & (f.to_numpy() > 0)]
        idx = ok if idx is None else idx.intersection(ok)
    idx = idx[(idx >= FIRST) & (idx < pd.Timestamp(LAST) + pd.Timedelta(days=1))]
    idx = idx[(rv.reindex(idx) > 0).to_numpy()].sort_values()
    on_e = pd.DatetimeIndex(idx.normalize()).isin(days)
    clk = np.asarray(idx.strftime("%H:%M"))
    spans = {
        "all48": idx[on_e],
        "non_rth35": idx[on_e & ~np.isin(clk, SC.RTH)],
        "clk_0300": idx[on_e & (clk == "03:00")],
        "grid": idx,
    }
    for sp, ix in spans.items():
        y = rv.reindex(ix).to_numpy()
        for k in (("ridge", "full"), ("rf", "har"), ("ols", "har")):
            p = pan[k].reindex(ix).to_numpy()
            r = y / p
            row = M[(M.span == sp) & (M.cls == k[0]) & (M.set == k[1])].iloc[0]
            assert row.n_bars == len(ix)
            assert row.qlike == pytest.approx(float(np.mean(r - np.log(r) - 1)), rel=1e-12)
            assert row.mse_e10 == pytest.approx(float(np.mean((y - p) ** 2)) * 1e10, rel=1e-12)
        # the full-vs-HAR DM of ridge on daily average QLIKE
        q = {}
        for st in ("full", "har"):
            p = pan[("ridge", st)].reindex(ix).to_numpy()
            r = y / p
            q[st] = pd.Series(r - np.log(r) - 1, index=ix.normalize()).groupby(level=0).mean().to_numpy()
        want = dm_test(q["full"], q["har"])
        got = D[(D.span == sp) & (D.comparison == "full vs HAR") & (D.cls == "ridge")].iloc[0]
        assert got["dm_qlike"] == pytest.approx(want["dm"], rel=1e-10) and got["T"] == want["T"]


def test_spans_partition_and_per_interval(scored):
    ck = json.loads((scored / "checks.json").read_text())
    nb, byc = ck["n_bars"], ck["n_bars_by_clock"]
    assert nb["all48"] == nb["rth13"] + nb["non_rth35"] == sum(byc.values())
    assert nb["rth13"] == sum(v for c, v in byc.items() if "1000" <= c <= "1600")
    assert nb["grid"] > nb["all48"]  # Sunday evenings and the holidays' rows are in the grid span only
    assert len(byc) == 48 and byc["1730"] < byc["0300"]  # zero prints are not scored
    P = pd.read_csv(scored / "per_interval.csv")
    assert set(P["clock"]) == set(SC.CLOCKS) and set(P["cls"]) == set(SC.CLASSES)
    assert P[P.clock == "16:00"]["rth"].all() and not P[P.clock == "16:30"]["rth"].any()
    lin = ck["sources"]["linear_root"]["models"]
    assert lin["ridge/full"]["duplicates_dropped"] == 1 and lin["ridge/full"]["covers_design_to_end"]
    assert all(v["clocks"] == 48 and v["b_source"] == "design b_oos" for v in lin.values())
    assert "full vs OLS HAR" in set(pd.read_csv(scored / "dm.csv")["comparison"])
    assert "EXPLORATORY" in (scored / "tables.txt").read_text()


def test_b_from_true_raw_and_linear_only(world, scored, tmp_path):
    rc = run_score(world, tmp_path / "nd", "--linear-root", str(world["lin"]), "--no-design")
    assert rc == 0
    a = pd.read_csv(tmp_path / "nd" / "metrics.csv")
    ck = json.loads((tmp_path / "nd" / "checks.json").read_text())
    assert len(ck["panel"]) == 7
    assert all(v["b_source"] == "true_raw / true_adj^2" for v in ck["sources"]["linear_root"]["models"].values())
    rc = run_score(world, tmp_path / "wd", "--linear-root", str(world["lin"]))
    b = pd.read_csv(tmp_path / "wd" / "metrics.csv")
    np.testing.assert_allclose(a["qlike"].to_numpy(), b["qlike"].to_numpy(), rtol=1e-12)


def test_reference_check_passes_and_fails(world, tmp_path):
    out = tmp_path / "lin"
    assert run_score(world, out, "--linear-root", str(world["lin"])) == 0
    M, D = pd.read_csv(out / "metrics.csv"), pd.read_csv(out / "dm.csv")
    inv = {v: k for k, v in SC.SET_KEY.items()}
    m = M[M.span == "rth13"]
    qb = pd.DataFrame(
        dict(
            variant="ALL48", model=m.cls, set=[inv[x] for x in m.set], n_bars=m.n_bars, qlike=m.qlike, mse_e10=m.mse_e10
        )
    )
    rows = []
    for sp, target in (("rth13", "every bar"), ("clk_1600", "16:00")):
        d = D[(D.span == sp) & (D.comparison == "full vs HAR")]
        for r in d.itertuples():
            rows.append(
                dict(
                    target=target,
                    variant="ALL48",
                    model=r.cls,
                    qlike_har=r.qlike_vs,
                    qlike_full=r.qlike,
                    dm_qlike=r.dm_qlike,
                )
            )
    ref = tmp_path / "ref"
    ref.mkdir()
    qb.to_csv(ref / "qlike_bar.csv", index=False)
    pd.DataFrame(rows).to_csv(ref / "gap.csv", index=False)
    assert run_score(world, tmp_path / "ok", "--linear-root", str(world["lin"]), "--ref-scores", str(ref)) == 0
    ck = json.loads((tmp_path / "ok" / "checks.json").read_text())
    assert ck["ref_check"]["pass"] and len(ck["ref_check"]["gap"]) == 6
    qb.loc[qb.index[0], "qlike"] *= 1 + 1e-9
    qb.to_csv(ref / "qlike_bar.csv", index=False)
    assert run_score(world, tmp_path / "bad", "--linear-root", str(world["lin"]), "--ref-scores", str(ref)) == 3
    assert not (tmp_path / "bad" / "metrics.csv").exists()


def copy_lin(world, dst: Path) -> Path:
    import shutil

    shutil.copytree(world["lin"], dst)
    return dst


def test_checks_fail_on_disagreeing_duplicates_missing_clock_and_bad_b(world, tmp_path):
    lin = copy_lin(world, tmp_path / "dup")
    f = next((lin / "_work" / "walk" / "all_features" / "ridge" / "c1").rglob("results.csv"))
    r = pd.read_csv(f)
    r.loc[0, "pred_adj"] += 1e-9  # the seam row now differs between the chunks
    r.to_csv(f, index=False)
    with pytest.raises(SystemExit, match="disagree"):
        run_score(world, tmp_path / "o1", "--linear-root", str(lin))
    lin = copy_lin(world, tmp_path / "clk")
    f = next((lin / "_work" / "walk" / "baseline" / "ols").rglob("results.csv"))
    r = pd.read_csv(f)
    r[~r["date"].str.endswith("03:00:00")].to_csv(f, index=False)
    assert run_score(world, tmp_path / "o2", "--linear-root", str(lin)) == 3
    ck = json.loads((tmp_path / "o2" / "checks.json").read_text())
    assert any("47 end stamps" in x for x in ck["fail"]) and any("cover" in x for x in ck["fail"])
    lin = copy_lin(world, tmp_path / "b")
    p = lin / "_work" / "design_baseline" / "b_oos.npy"
    np.save(p, np.load(p) * 1.001)
    assert run_score(world, tmp_path / "o3", "--linear-root", str(lin)) == 3
    assert run_score(world, tmp_path / "o4", "--trees", str(world["trees"]), "--require", "ols/har") == 3


# ============================================================================ linear wrapper
@pytest.fixture
def lin_mods():
    import pooled_linear_rerun_2026_10_19_13 as OR

    OR = importlib.reload(OR)
    import all48_linear as L

    L = importlib.reload(L)
    yield OR, L
    importlib.reload(OR)  # the orchestrator's defaults back for any later test


def test_import_changes_nothing_then_configure(lin_mods, tmp_path):
    OR, L = lin_mods
    assert OR.CFG[:2] == ["--sample", "rth13"] and "DROP_OVERNIGHT_INDICATOR=1" in OR.CFG
    assert (OR.PPD, OR.W, OR.TW_DAYS, OR.FIRST_FOLD, OR.FIRST_DAY) == (13, 26000, 2000, 0, "2020-01-03")
    assert OR.check_design is not L.check_design
    L.configure(tmp_path / "run")
    assert OR.CFG == L.runner_cfg() and OR.CFG[:2] == ["--sample", "48"]
    for f in (
        "DROP_OVERNIGHT_INDICATOR=0",
        "FOMC_SCHEDULE_2020_FIX=1",
        "FOMC_PROXIMITY_NO_WINSOR=1",
        "LATE_START_WARMUP=1",
    ):
        assert f in OR.CFG
    assert "DROP_OVERNIGHT_INDICATOR=1" not in OR.CFG
    assert (OR.PPD, OR.W, OR.TW_DAYS, OR.FIRST_FOLD, OR.FIRST_DAY) == (48, 144000, 3000, 3, "2014-01-02")
    assert OR.PRE_HALO == 12000 + 1200 + 144000 and OR.check_design is L.check_design
    assert L.EX_FLAGS["DROP_OVERNIGHT_INDICATOR"] is False and L.EX_FLAGS["DROP_HOLIDAY_SESSIONS"] is False


def names48(s: str, OR) -> list[str]:
    feats, _ = OR.design_features()
    har = [f"har_ma_{k}" for k in OR.HAR_LADDER]
    if s == "baseline":
        return har + [f"DOW_{d}" for d in range(5)] + ["hour", "is_overnight"]
    val = [f"adj_{f}_ma_{k}" for f in feats for k in OR.HAR_LADDER]
    av = [f"{f}_avail_ma_{k}" for f in feats for k in OR.HAR_LADDER]
    return har + val + av + [f"DOW_{d}" for d in range(5)] + ["hour", "is_overnight"]


def test_check_design_and_walk_jobs(lin_mods, tmp_path):
    OR, L = lin_mods
    L.configure(tmp_path / "run")
    days = pd.bdate_range("2011-06-01", "2015-03-31")  # >= 50 blocks of 250 rows after the walk start
    d = pd.DatetimeIndex([x + pd.Timedelta(minutes=30 * k) for x in days for k in range(48)])
    for s in OR.SETS:
        dz = OR.design_dir(s)
        dz.mkdir(parents=True)
        (dz / "design.json").write_text(json.dumps(dict(names=names48(s, OR))))
        pd.DataFrame({"date": d.strftime("%Y-%m-%d %H:%M:%S")}).to_csv(dz / "dates_oos.csv", index=False)
        assert len(L.check_design(s)) == L.N_COLS[s]
    (OR.design_dir("baseline") / "design.json").write_text(json.dumps(dict(names=names48("baseline", OR)[:-1])))
    with pytest.raises(AssertionError):
        L.check_design("baseline")  # the 13-bar calendar (no is_overnight) is refused
    pen = dict(ridge=dict(baseline=1e3, all_features=1e3), lasso=dict(baseline=1e-4, all_features=1e-4))
    pen |= dict(enet=dict(baseline=1e-3, all_features=1e-4), ols=dict(baseline=None))
    pen["info"] = dict(standardize_penalty=True, off_spec_grid={})
    (OR.OUT / "frozen_penalties.json").write_text(json.dumps(pen))
    jobs = OR.walk_jobs(L.parse_chunks([]), None)
    assert len(jobs) == 19
    S0, n = OR.walk_start(), len(d) + OR.W
    assert (S0 - OR.W) % 250 == 0 and S0 <= OR.first_row() < S0 + 250
    starts = sorted(int(j[0][j[0].index("--start") + 1]) for j in jobs)
    assert all((x - S0) % 250 == 0 for x in starts)
    assert all(j[0][j[0].index("--halo") + 1] == "144000" for j in jobs)
    ends = [int(j[0][j[0].index("--end") + 1]) for j in jobs]
    assert ends.count(-1) == 7 and n > S0  # every model's last chunk runs to the last row


# ============================================================================ trees
def tree_design(root: Path, sample: str, s: str, W: int, d: pd.DatetimeIndex, seed: int = 3) -> None:
    import all48_trees as A

    rng = np.random.default_rng(seed)
    p = A.N_COLS[sample][s]
    dz = root / "_work" / f"design_{s}"
    dz.mkdir(parents=True, exist_ok=True)
    n = W + len(d)
    X = np.asfortranarray(rng.normal(size=(n, p)))
    np.save(dz / "X.npy", X)
    np.save(dz / "y.npy", np.abs(X[:, 0]) + rng.random(n))
    np.save(dz / "b_oos.npy", 0.5 + rng.random(len(d)))
    pd.DataFrame({"date": d.strftime("%Y-%m-%d %H:%M:%S")}).to_csv(dz / "dates_oos.csv", index=False)
    cf = dict(sample=sample, periods_per_day=48 if sample == "48" else 13, window=W, har_base="5")
    (dz / "design.json").write_text(json.dumps(dict(names=[f"c{j}" for j in range(p)], config=cf, shape=[n, p])))


@pytest.fixture(scope="module")
def trees_env(tmp_path_factory):
    pytest.importorskip("lightgbm")
    import all48_trees as A
    import pooled_trees_rerun_13 as T

    tmp = tmp_path_factory.mktemp("trees")
    g = grid("2013-12-20", "2014-01-06")
    d = g[g >= "2013-12-30"]  # OOS rows; origins = the rows dated 2014-01-02 .. (Thu, Fri, Sun, Mon)
    tree_design(tmp / "d48", "48", "baseline", 300, d)
    rv = pd.Series(1.0, index=d)
    orig_rv = T.rv_bars
    T.rv_bars = lambda: rv  # no read of data/core_stats.parquet
    yield dict(tmp=tmp, A=A, T=T, d=d)
    T.rv_bars = orig_rv


def walk(env, *args: str) -> int:
    return env["A"].main(["walk", "--est", "lgbm", "--threads", "1", *args])


def test_tree_walk_every_origin_pieces_and_merge(trees_env):
    A, T, tmp = trees_env["A"], trees_env["T"], trees_env["tmp"]
    D = A.Design(tmp / "d48", "baseline")
    o = D.origins()
    clk = D.d[o].strftime("%H:%M")
    assert len(o) == 48 + 35 + 11 + 48 and "03:00" in set(clk) and "23:30" in set(clk)  # every interval origin
    out = tmp / "run"
    for part in ("1/1", "1/3", "2/3", "3/3"):
        assert (
            walk(trees_env, "--design", str(tmp / "d48"), "--set", "baseline", "--part", part, "--out", str(out)) == 0
        )
    one = pd.read_parquet(out / "pieces" / "a48_lgbm_baseline_1of1.parquet")
    three = pd.concat([pd.read_parquet(out / "pieces" / f"a48_lgbm_baseline_{i}of3.parquet") for i in (1, 2, 3)])
    assert np.array_equal(one["pred_adj"].to_numpy(), three["pred_adj"].to_numpy())
    assert pd.DatetimeIndex(one["date"]).equals(D.d[o])
    # a direct fit on the W rows before the origin, at a Sunday-evening origin
    k = int(np.flatnonzero(np.asarray(D.d[o].dayofweek == 6))[0])
    r = D.W + int(o[k])
    _, make = A.model("lgbm", 1)
    X = np.ascontiguousarray(D.X[r - D.W : r + 1], dtype=np.float64)
    yh = T.fit_predict(make, "lgbm", X[:-1], D.y[r - D.W : r], X[-1:])
    assert yh[0] == one["pred_adj"].iloc[k] and int(one["fit_row"].iloc[k]) == r - D.W
    m = json.loads((out / "pieces" / "a48_lgbm_baseline_2of3.json").read_text())
    assert m["lgbm_deterministic"] and m["lgbm_force_col_wise"] and m["window_rows"] == 300 and m["label"] == "a48"

    # merge: only gated (cluster, CPU model) pairs; the 3-piece and 1-piece sets of one class cannot both be complete
    def gate_dir(name: str, cpu: str, host: str) -> Path:
        g = tmp / name
        (g / "pieces").mkdir(parents=True)
        (g / "passed_lgbm").write_text(cpu + "\n")
        (g / "pieces" / "a48_lgbm_baseline_ends10.json").write_text(json.dumps(dict(host=host)))
        return g

    def merge(g: Path, *extra: str) -> int:
        return A.main(["merge", "--design", str(tmp / "d48"), "--out", str(out), "--gates", str(g), *extra])

    gate = gate_dir("gate_ok", T.cpu_model(), m["host"])
    with pytest.raises(AssertionError, match="complete splits"):
        merge(gate, "--est", "lgbm", "--set", "baseline")
    (out / "pieces" / "a48_lgbm_baseline_1of1.json").unlink()
    assert merge(gate, "--est", "lgbm", "--set", "baseline") == 0
    f = pd.read_parquet(out / "forecasts.parquet")
    assert np.array_equal(f["pred_adj"].to_numpy(), one["pred_adj"].to_numpy())
    assert merge(gate, "--est", "lgbm,xgb", "--set", "baseline", "--require-all") == 1  # XGBoost has no piece set
    assert json.loads((out / "forecasts.json").read_text())["missing"] == ["xgb/baseline"]
    for g in (
        gate_dir("gate_bad", "Some Other CPU", m["host"]),
        gate_dir("gate_carc", T.cpu_model(), "a01.hpc.usc.edu"),
    ):
        with pytest.raises(AssertionError, match="not gated"):  # another model, or this model gated on CARC only
            merge(g, "--est", "lgbm", "--set", "baseline")


def test_tree_tasks_pieces_within_target(trees_env, tmp_path):
    A = trees_env["A"]
    sizes = A.piece_sizes(129332, 249.9, 1.0)
    assert sum(sizes) == 129332 and max(sizes) * 249.9 <= 3600 and len(sizes) == -(-129332 // 14)
    tf = str(tmp_path / "t.txt")
    assert A.main(["tasks", "--design", str(trees_env["tmp"] / "d48"), "--out", tf, "--target-hours", "0.1"]) == 0
    lines = (tmp_path / "t.txt").read_text().splitlines()
    assert all(float(x.split()[3]) <= 0.1 for x in lines)  # a forest refit (249.9 s) is one piece of 0.069 h
    est_set = [tuple(x.split()[:2]) for x in lines]
    assert est_set[0] == ("lgbm", "baseline") and est_set[-1] == ("rf_skx", "all_features")
    n_rf = int(lines[-1].split()[2].split("/")[1])
    assert sum(1 for e in est_set if e == ("rf_skx", "all_features")) == n_rf


def test_tree_gate_t1_t2(trees_env, monkeypatch):
    A, tmp = trees_env["A"], trees_env["tmp"]
    monkeypatch.setattr(A, "SETS", ("baseline",))
    monkeypatch.setattr(A, "WINDOW", {"48": 300, "rth13": 260})
    g13 = grid("2013-12-01", "2014-01-10")
    d13 = g13[np.isin(np.asarray(g13.strftime("%H:%M")), list(SC.RTH)) & (g13 >= "2013-12-20")]
    tree_design(tmp / "d13", "rth13", "baseline", 260, d13, seed=5)
    monkeypatch.setattr(A, "N_RTH13", int((d13 >= "2014-01-02").sum()))
    dz = tmp / "d13" / "_work" / "design_baseline"
    monkeypatch.setattr(A, "CELLB_MD5", {"baseline": {fn: A.md5(dz / fn) for fn in A.CELLB_MD5["baseline"]}})
    gates = {}
    for tag in ("ref", "other"):
        G = tmp / "gate" / tag
        for dd in ("d13", "d48"):
            assert walk(trees_env, "--design", str(tmp / dd), "--set", "baseline", "--ends", "10", "--out", str(G)) == 0
        gates[tag] = G
    t1 = pd.read_parquet(gates["ref"] / "pieces" / "r13_lgbm_baseline_ends10.parquet")
    tp = tmp / "template.parquet"
    t1[["model", "predictor_set", "date", "pred_adj", "true_adj", "B"]].to_parquet(tp, index=False)
    monkeypatch.setattr(A, "TEMPLATE", {"lgbm": tp})
    monkeypatch.setattr(A, "REF_CPUS", {trees_env["T"].cpu_model()})
    assert A.main(["gate", "--out", str(gates["ref"]), "--est", "lgbm"]) == 0  # the reference model
    assert A.main(["gate", "--out", str(gates["other"]), "--ref", str(gates["ref"]), "--est", "lgbm"]) == 0
    assert (gates["other"] / "passed_lgbm").read_text().strip() == trees_env["T"].cpu_model()
    p = gates["other"] / "pieces" / "a48_lgbm_baseline_ends10.parquet"
    f = pd.read_parquet(p)
    f.loc[3, "pred_adj"] = np.nextafter(f.loc[3, "pred_adj"], np.inf)  # one ulp: T2 is bit for bit
    f.to_parquet(p, index=False)
    (gates["other"] / "passed_lgbm").unlink()
    assert A.main(["gate", "--out", str(gates["other"]), "--ref", str(gates["ref"]), "--est", "lgbm"]) == 3
    assert not (gates["other"] / "passed_lgbm").exists()


# ============================================================================ default behaviour of the existing paths
def test_existing_tree_runner_defaults_unchanged(trees_env):
    T, A = trees_env["T"], trees_env["A"]
    assert T.PPD == 13 and T.N_COLS == {"baseline": 12, "all_features": 564}
    assert T.COLS == ["model", "predictor_set", "date", "pred_adj", "true_adj", "B", "plain", "rv_actual"]
    _, make = T.model_factory("lgbm", 1)
    p = make().get_params()
    assert "deterministic" not in p and "force_col_wise" not in p  # the main walker's LightGBM as before
    _, make48 = A.model("lgbm", 1)
    assert make48().get_params()["deterministic"] is True and make48().get_params()["force_col_wise"] is True
