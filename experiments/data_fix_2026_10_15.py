"""Data fix of 2026-10-15: invalid prints nulled at load, the sign of a series read from its name.

User (2026-10-15), on spread_vwstock's one negative print (2018-11-12) making the whole series
"signed": "it shouldn't count as signed though, it's clearly a data problem."  The fix lives in
src/data/loading.py (STRICTLY_POSITIVE_SERIES / NON_NEGATIVE_SERIES / SIGNED_SERIES,
null_invalid_prints, flag NULL_INVALID_PRINTS) and src/features/transforms/target.py
(series_is_signed, flag SIGN_BY_NAME).  This script audits every series and measures the fix's
impact WITHOUT touching any cache a running job reads: everything it writes is under
results/data_fix_2026-10-15/ (heavy files in its _work/, gitignored).

Variants (flags set per process):
  old         NULL_INVALID_PRINTS = SIGN_BY_NAME = False: the pre-fix chain (control)
  fix         both True: the fix as committed to src/
  fix_fomc    fix + the FOMC activity indicators (fomc_release_active, fomc_day_active) set to their
              modal value (0) on UNOBSERVED rows instead of being read off the forward-filled raw
              column (a second finding of the audit: the FOMC dead tail forward-fills the last
              statement bar's 1, so both are 1 on every bar from 2023-11-01 14:30 on, 124 of the
              1,469 forecast sessions).  Applied by wrapping executor.load_and_transform; src/ is NOT
              changed for it.

Stages:  python experiments/data_fix_2026_10_15.py <stage>
  audit                       every series: domain rule, invalid prints, sign by data (old, full
                              sample) / by name (new), prints that decide it, transform branch old /
                              new, log-floor hits, impossible-value checks, FOMC flags; audit.csv
  capture <variant> <segment> <bucket>   one design (capture_design_close.capture, unchanged) into
                              _work/design_<variant>/ (run by `designs` as subprocesses)
  designs                     bar1600 + last30 x all_features + baseline for fix and fix_fomc, and
                              bar1600 all_features for old (gate: old == the 2026-10-06 local rebuild)
  diff                        fixed 16:00 design vs the design of record: per-column change
  linear <variant>            experiments/close_linear_pretune.py prep + compute, its paths pointed
                              at _work/linear_<variant>/ (all_features; the arrays of record except
                              the columns the fix changes)
  score                       arms a-f (ridge / lasso / enet) old vs fix vs fix_fomc: QLIKE, DM vs
                              the old forecasts, Sharpe mid / crossed
  pooled                      the pooled panel (prep_cache_all_features_b2.npz): the affected
                              series' columns rebuilt old vs fix (base-2 ladder, 24,000-bar scaling,
                              masked scaling) -- the cache itself is not rebuilt
"""

from __future__ import annotations

import os

for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
for _p in (
    REPO,
    REPO / "experiments",
    REPO / "notebooks",
    REPO / "specs",
    REPO / "writeup",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

OUT = REPO / "results" / "data_fix_2026-10-15"
WORK = OUT / "_work"
DATA = WORK / "data_nochain"
SKIP = (
    "spxw_chain.parquet",
    "spxw_spot.parquet",
)  # not bar-keyed; load_raw_data skips them
RECORD = REPO / "results" / "close_studies_2026-10-03" / "forecasts"
RECORD_DESIGN = RECORD / "design_bar1600_all_features.parquet"
OLD_LOCAL = (
    REPO / "results" / "linear_pretune_2026-10-06" / "_work" / "design"
)  # 2026-10-06 rebuild here
OLD_STUDY = REPO / "results" / "linear_pretune_2026-10-06"
POOLED_CACHE = (
    REPO.parent / "harxhar-clean" / "results" / "prep_cache_all_features_b2.npz"
)
DESIGN_META = ("row", "date", "forecast_row", "y", "baseline", "true_raw")
VARIANTS = {
    "old": dict(null=False, sign=False, active=False),
    "fix": dict(null=True, sign=True, active=False),
    "fix_fomc": dict(null=True, sign=True, active=True),
}


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def design_dir(variant: str) -> Path:
    return WORK / f"design_{variant}"


def set_variant(variant: str) -> None:
    """Flags of the variant on the src modules of THIS process (+ the active wrapper)."""
    import src.backtest.executor as EX
    import src.data.loading as LD
    import src.features.transforms.target as TG

    v = VARIANTS[variant]
    LD.NULL_INVALID_PRINTS = v["null"]
    TG.SIGN_BY_NAME = v["sign"]
    if v["active"] and not getattr(EX.load_and_transform, "_active_neutral", False):
        orig = EX.load_and_transform

        def load_and_transform(*a, **k):
            df, cols = orig(*a, **k)
            for c in cols:
                # FOMC only: its flags are observed on every bar but the dead tail, so this is
                # exactly the dead-tail repair.  (Applied to every _active column it would also
                # rewrite the sparse voldemand feed's stale-zero bars inside 2010-2023 -- a
                # different question, measured once and set aside: 900-1,365 of the 1,469
                # forecast rows of each voldemand _active_ma_{25..3125} column changed.)
                if c.endswith("_active") and c.startswith("fomc_"):
                    obs = df[f"{c[: -len('_active')]}__obs"].to_numpy(bool)
                    act = df[c].to_numpy(np.float64)
                    if obs.any() and not obs.all():
                        neutral = float(
                            np.round(act[obs].mean())
                        )  # the indicator's modal value
                        df[c] = np.where(obs, act, neutral)
            return df, cols

        load_and_transform._active_neutral = True
        EX.load_and_transform = load_and_transform


def data_dir() -> Path:
    """Hard links (copies if linking fails) to data/*.parquet except SKIP."""
    DATA.mkdir(parents=True, exist_ok=True)
    for f in sorted((REPO / "data").glob("*.parquet")):
        if f.name in SKIP:
            continue
        dst = DATA / f.name
        if dst.is_file() and dst.stat().st_size == f.stat().st_size:
            continue
        if dst.exists():
            dst.unlink()
        try:
            os.link(f, dst)
        except OSError:
            shutil.copy2(f, dst)
    return DATA


# ============================================================================ audit
PROBE = pd.Series([-4.0, 0.25, 4.0, 16.0, 81.0])
CANDIDATES = {
    "sqrt": lambda x: np.sqrt(x),
    "signed sqrt": lambda x: np.sign(x) * np.sqrt(np.abs(x)),
    "cube root": lambda x: np.cbrt(x),
    "signed 4th root": lambda x: np.sign(x) * np.abs(x) ** 0.25,
    "identity": lambda x: x,
    "log": lambda x: np.log(np.clip(x, 1e-12, None)),
}


def semantic_label(col: str, has_neg: bool) -> str:
    import src.features.transforms.target as TG

    with np.errstate(invalid="ignore", divide="ignore"):
        out = TG.apply_semantic_transform(PROBE.copy(), col, has_neg).to_numpy(float)
        hits = [
            k
            for k, f in CANDIDATES.items()
            if np.allclose(out, f(PROBE.to_numpy(float)), equal_nan=True)
        ]
    assert len(hits) == 1, (col, hits)
    return hits[0]


def domain_of(col: str) -> str:
    import src.data.loading as LD

    if col in LD.STRICTLY_POSITIVE_SERIES:
        return "> 0"
    if col in LD.NON_NEGATIVE_SERIES:
        return ">= 0"
    if col in LD.SIGNED_SERIES:
        return "signed"
    return "unregistered"


def audit() -> None:  # noqa: C901 - one table of facts
    import src.backtest.executor as EX
    import src.data.loading as LD
    import src.features.transforms.target as TG

    OUT.mkdir(parents=True, exist_ok=True)
    src = str(data_dir())
    files = {}
    for f in sorted(DATA.glob("*.parquet")):
        for c in pd.read_parquet(f).columns:
            if c != "endbartime":
                files.setdefault("RV" if c == "sumret2" else c, f.stem)

    # ---- the bad print and its neighbours (raw file)
    v = pd.read_parquet(DATA / "vwstock_stats.parquet")
    v["endbartime"] = pd.to_datetime(v["endbartime"])
    i = int(np.flatnonzero((v["spread_vwstock"] < 0).to_numpy())[0])
    nb = v.iloc[i - 6 : i + 7].copy()
    c = pd.read_parquet(
        DATA / "core_stats.parquet", columns=["endbartime", "sumret2", "sumret"]
    )
    c["endbartime"] = pd.to_datetime(c["endbartime"])
    e = pd.read_parquet(
        DATA / "ewstock_stats.parquet",
        columns=["endbartime", "spread_ewstock", "sumret2_ewstock"],
    )
    e["endbartime"] = pd.to_datetime(e["endbartime"])
    nb = nb.merge(
        c.rename(columns={"sumret2": "ES_sumret2", "sumret": "ES_sumret"}),
        on="endbartime",
        how="left",
    )
    nb = nb.merge(e, on="endbartime", how="left")
    nb.to_csv(OUT / "bad_print_neighbours.csv", index=False)
    slot = v["endbartime"].dt.time == v["endbartime"].iloc[i].time()
    ss = v.loc[slot, "spread_vwstock"].dropna()
    say(
        f"bad print: {v['endbartime'].iloc[i]} spread_vwstock = {v['spread_vwstock'].iloc[i]:.6g}; same slot "
        f"n {len(ss)}, median {ss.median():.6g}, p1 {ss.quantile(0.01):.6g}"
    )

    # ---- the panel as the executor loads it, pre-fix (nulling off)
    LD.NULL_INVALID_PRINTS = False
    P = LD.load_raw_data(src, allow_missing=True)
    LD.NULL_INVALID_PRINTS = True
    Q = P.copy()
    counts = LD.null_invalid_prints(Q, verbose=True)
    # OFI rebuilt on the nulled legs (load_raw_data derives it after the nulling)
    LD.add_derived_features(Q)
    series = (
        ["RV"]
        + list(LD.ALL_FEATURES)
        + [
            c
            for c in P.columns
            if c in files
            and c not in LD.ALL_FEATURES
            and c != "RV"
            and c not in ("hour", "DOW")
            and not c.endswith("release")
        ]
    )
    rows = []
    for s in series:
        x = P[s]
        xn = Q[s]
        neg_old = bool((x.dropna() < 0).any())
        TG.SIGN_BY_NAME = True
        sign_new = TG.series_is_signed(s, xn)
        TG.SIGN_BY_NAME = False
        neg_after = bool((xn.dropna() < 0).any())
        TG.SIGN_BY_NAME = True
        excl = TG.is_diurnal_excluded(s) or s in TG.SKIP_VARS
        sem_old = semantic_label(s, neg_old)
        sem_new = semantic_label(s, sign_new)
        dom = domain_of(s) if s != "RV" else ">= 0 (target, never nulled)"
        fin = x.dropna()
        inval = counts.get(s, 0)
        bad = (
            (x <= 0)
            if dom == "> 0"
            else (x < 0)
            if dom.startswith(">= 0")
            else pd.Series(False, index=x.index)
        )
        bt = P.loc[bad.fillna(False).to_numpy(), "t"]
        inflated, mode = EX.mode_inflated(x.ffill().to_numpy(dtype=np.float64))
        rows.append(
            dict(
                series=s,
                source=files.get(s, "derived"),
                domain=dom,
                n_obs=int(fin.size),
                n_neg=int((fin < 0).sum()),
                n_zero=int((fin == 0).sum()),
                min=float(fin.min()) if fin.size else np.nan,
                max=float(fin.max()) if fin.size else np.nan,
                invalid_nulled=int(inval),
                first_invalid=str(bt.min())[:16] if len(bt) else "",
                last_invalid=str(bt.max())[:16] if len(bt) else "",
                sign_data_old=neg_old,
                prints_deciding_sign=int((fin < 0).sum()) if neg_old else 0,
                sign_name_new=sign_new,
                name_rule_registered=domain_of(s) != "unregistered" or s == "RV",
                sign_data_after_null=neg_after,
                name_eq_data_after_null=bool(sign_new == neg_after),
                diurnal_old="none" if excl else ("s.d." if neg_old else "mean"),
                diurnal_new="none" if excl else ("s.d." if sign_new else "mean"),
                transform_old=sem_old,
                transform_new=sem_new,
                log_floor_hits_old=int(((x <= 0) & x.notna()).sum())
                if sem_old == "log"
                else 0,
                log_floor_hits_new=int(((xn <= 0) & xn.notna()).sum())
                if sem_new == "log"
                else 0,
                mode_inflated=bool(inflated),
                mode=float(mode) if inflated else np.nan,
            )
        )
    A = pd.DataFrame(rows)
    A["changed"] = (
        (A["diurnal_old"] != A["diurnal_new"])
        | (A["transform_old"] != A["transform_new"])
        | (A["invalid_nulled"] > 0)
    )
    A.to_csv(OUT / "audit.csv", index=False)

    # ---- impossible values
    chk = []
    for w in ("ewstock", "vwstock"):
        b, s_, t = P[f"buyturnover_{w}"], P[f"sellturnover_{w}"], P[f"turnover_{w}"]
        chk += [
            dict(
                check=f"buyturnover_{w} > turnover_{w}",
                n=int((b > t * (1 + 1e-9)).sum()),
            ),
            dict(
                check=f"sellturnover_{w} > turnover_{w}",
                n=int((s_ > t * (1 + 1e-9)).sum()),
            ),
            dict(
                check=f"buy + sell > turnover ({w}, 1e-9 rel)",
                n=int((b + s_ > t * (1 + 1e-9)).sum()),
            ),
            dict(
                check=f"buy + sell == 0 with turnover > 0 ({w}): ofi NaN",
                n=int(((b + s_) == 0).sum()),
            ),
            dict(
                check=f"ofi_{w} outside [-1, 1]",
                n=int((P[f"ofi_{w}"].abs() > 1 + 1e-12).sum()),
            ),
            dict(
                check=f"effspread_{w} == 0 (nulled)",
                n=int((P[f"effspread_{w}"] == 0).sum()),
            ),
            dict(
                check=f"effspread_{w} == 0 and buy + sell == 0 ({w})",
                n=int(((P[f"effspread_{w}"] == 0) & ((b + s_) == 0)).sum()),
            ),
        ]
    nob = P["numobs"].dropna()
    chk += [
        dict(
            check="numobs outside 1..30 or non-integer",
            n=int(((nob < 1) | (nob > 30) | (nob != nob.round())).sum()),
        ),
        dict(
            check="stocktwits_sentiment outside [-1, 1]",
            n=int((P["stocktwits_sentiment"].abs() > 1 + 1e-12).sum()),
        ),
        dict(
            check="sumvolume == 0 with numobs >= 1 (nulled)",
            n=int(((P["sumvolume"] == 0) & (P["numobs"] >= 1)).sum()),
        ),
        dict(
            check="sumvolume == 0 and sumret2 > 0",
            n=int(((P["sumvolume"] == 0) & (P["RV"] > 0)).sum()),
        ),
        dict(
            check="fomc_release / fomc_day not in {0, 1} (live part)",
            n=int(
                (~P["fomc_release"].dropna().isin([0.0, 1.0])).sum()
                + (~P["fomc_day"].dropna().isin([0.0, 1.0])).sum()
            ),
        ),
        dict(
            check="fomc_until_inv / since_inv outside (0, 1]",
            n=int(
                ((P["fomc_until_inv"] <= 0) | (P["fomc_until_inv"] > 1)).sum()
                + ((P["fomc_since_inv"] <= 0) | (P["fomc_since_inv"] > 1)).sum()
            ),
        ),
    ]
    z = P.loc[P["sumvolume"] == 0, "t"]
    chk.append(
        dict(
            check="sumvolume == 0: first / last bar; ET hours",
            n=int(len(z)),
            detail=f"{str(z.min())[:16]} .. {str(z.max())[:16]}; "
            f"{sorted(set(z.dt.strftime('%H:%M')))[0]}-{sorted(set(z.dt.strftime('%H:%M')))[-1]}",
        )
    )
    pd.DataFrame(chk).to_csv(OUT / "audit_checks.csv", index=False)

    # ---- FOMC flags: what the chain does to them (on the panel the executor hands robust_transform)
    F = P.copy()
    F[LD.FOMC_FEATURES] = F[LD.FOMC_FEATURES].ffill()
    fom = []
    for s in LD.FOMC_FEATURES:
        adj, _ = TG.robust_transform(F, s, use_diurnal=True)
        live = P[s].notna()
        x = P[s]
        r = dict(
            series=s,
            transform=semantic_label(s, False),
            values_raw=json.dumps(sorted(x.dropna().unique().tolist())[:3]),
        )
        if s in ("fomc_release", "fomc_day"):
            ones = live & (x == 1)
            r.update(
                n_ones=int(ones.sum()),
                adj_on_zeros=json.dumps(
                    sorted(set(np.round(adj[live & (x == 0)].dropna(), 3)))[:4]
                ),
                ones_surviving_winsor=int((adj[ones] > -1).sum()),
                ones_clipped_to_floor=int((adj[ones] < -27).sum()),
            )
        fom.append(r)
    pd.DataFrame(fom).to_csv(OUT / "audit_fomc.csv", index=False)
    say(f"audit: {len(A)} series; changed: {A.loc[A['changed'], 'series'].tolist()}")
    print(
        A[
            [
                "series",
                "domain",
                "n_obs",
                "n_neg",
                "n_zero",
                "invalid_nulled",
                "sign_data_old",
                "prints_deciding_sign",
                "sign_name_new",
                "name_eq_data_after_null",
                "diurnal_old",
                "diurnal_new",
                "transform_old",
                "transform_new",
                "log_floor_hits_old",
                "log_floor_hits_new",
            ]
        ].to_string()
    )
    print(pd.DataFrame(chk).to_string())
    print(pd.DataFrame(fom).to_string())


def design_record_checks() -> pd.DataFrame:
    """FOMC and activity columns of the 16:00 design of record."""
    d = pd.read_parquet(RECORD_DESIGN)
    t = pd.to_datetime(d["date"])
    rel = pd.read_parquet(DATA / "releases.parquet")
    ev = pd.to_datetime(rel["endbartime"])[pd.to_numeric(rel["fomc release"]) > 0]
    fd = t.dt.normalize().isin(set(ev.dt.normalize()))
    last = ev.max()
    tail = t > last
    W = int((d["forecast_row"] < 0).sum())
    out = []
    for c in [c for c in d.columns if "fomc" in c or c.endswith("_active_ma_1")]:
        x = d[c]
        out.append(
            dict(
                column=c,
                n_unique=int(x.nunique()),
                value_fomc_days=float(x[fd].mean()),
                value_other_days=float(x[~fd & ~tail].mean()),
                value_dead_tail=float(x[tail].mean()),
                dead_tail_forecast_sessions=int(
                    (tail & (d["forecast_row"] >= 0)).sum()
                ),
            )
        )
    R = pd.DataFrame(out)
    R.to_csv(OUT / "audit_design_record_fomc.csv", index=False)
    say(
        f"design of record: W {W}; FOMC dead tail after {last} = {int(tail.sum())} rows "
        f"({int((tail & (d['forecast_row'] >= 0)).sum())} forecast sessions)"
    )
    return R


# ============================================================================ designs
def capture(variant: str, segment: str, bucket: str) -> None:
    os.environ["CLOSE_DESIGN_DIR"] = str(design_dir(variant))
    os.chdir(REPO)
    set_variant(variant)
    src = str(data_dir())
    import capture_design_close as cdc

    assert cdc.OUT == design_dir(variant), cdc.OUT
    run = cdc.ex.run_executor

    def run_nochain(*args, **kw):
        assert kw.get("data_path") == "data", kw.get("data_path")
        kw["data_path"] = src
        return run(*args, **kw)

    cdc.ex.run_executor = run_nochain
    t0 = time.time()
    cdc.capture(segment, bucket)
    say(f"capture {variant} {segment} {bucket}: {time.time() - t0:.0f}s")


def designs() -> None:
    data_dir()
    jobs = [
        ("fix", s, b)
        for s in ("bar1600", "last30")
        for b in ("all_features", "baseline")
    ]
    jobs += [("fix_fomc", s, "all_features") for s in ("bar1600", "last30")]
    jobs += [("old", "bar1600", "all_features")]
    todo = [
        j
        for j in jobs
        if not (design_dir(j[0]) / f"design_{j[1]}_{j[2]}.npz").is_file()
    ]
    say(f"designs: {len(todo)} of {len(jobs)} to capture")
    logs = WORK / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    procs = []
    for v, s, b in todo:
        fh = open(logs / f"capture_{v}_{s}_{b}.log", "w")
        procs.append(
            (
                subprocess.Popen(
                    [sys.executable, __file__, "capture", v, s, b],
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                    cwd=str(REPO),
                ),
                fh,
                (v, s, b),
            )
        )
        # at most 4 at a time (each holds the ~330k-row panel)
        while sum(p.poll() is None for p, _, _ in procs) >= 4:
            time.sleep(5)
    for p, fh, j in procs:
        rc = p.wait()
        fh.close()
        assert rc == 0, (j, rc)
    # baseline designs carry no exog: identical in every variant (asserted against fix)
    for v in ("fix_fomc",):
        for s in ("bar1600", "last30"):
            dst = design_dir(v) / f"design_{s}_baseline.npz"
            if not dst.is_file():
                shutil.copy2(design_dir("fix") / f"design_{s}_baseline.npz", dst)
    say("designs done")


# ============================================================================ diff
def load_npz(p: Path) -> dict:
    z = np.load(p, allow_pickle=False)
    return {k: z[k] for k in z.files}


def diff() -> None:
    rec = pd.read_parquet(RECORD_DESIGN)
    names = [c for c in rec.columns if c not in DESIGN_META]
    Xrec = rec[names].to_numpy(np.float64)
    old_local = load_npz(OLD_LOCAL / "design_bar1600_all_features.npz")
    gates = []
    old = load_npz(design_dir("old") / "design_bar1600_all_features.npz")
    same = bool(
        np.array_equal(old["X"], old_local["X"])
        and np.array_equal(old["y"], old_local["y"])
    )
    gates.append(
        dict(
            gate="variant old (flags off) == the 2026-10-06 local rebuild, bit for bit",
            ok=same,
        )
    )
    gates.append(
        dict(
            gate="old local vs design of record: max abs diff (host floating point)",
            ok=bool(np.abs(old_local["X"] - Xrec).max() < 1e-9),
            value=float(np.abs(old_local["X"] - Xrec).max()),
        )
    )
    for s in ("bar1600", "last30"):
        a = load_npz(design_dir("fix") / f"design_{s}_baseline.npz")
        b = load_npz(OLD_LOCAL / f"design_{s}_baseline.npz")
        gates.append(
            dict(
                gate=f"{s} baseline design: fix == 2026-10-06 local, bit for bit",
                ok=bool(
                    np.array_equal(a["X"], b["X"]) and np.array_equal(a["y"], b["y"])
                ),
            )
        )
    rows = []
    for v in ("fix", "fix_fomc"):
        for s in ("bar1600", "last30"):
            new = load_npz(design_dir(v) / f"design_{s}_all_features.npz")
            ref = (
                old_local
                if s == "bar1600"
                else load_npz(OLD_LOCAL / f"design_{s}_all_features.npz")
            )
            nn = [str(x) for x in new["names"]]
            assert nn == [str(x) for x in ref["names"]] and np.array_equal(
                new["y"], ref["y"]
            ), (v, s)
            assert (new["date"] == ref["date"]).all()
            W = int(new["W"])
            D = new["X"] - ref["X"]
            for j, c in enumerate(nn):
                d = np.abs(D[:, j])
                if not (d > 0).any():
                    continue
                ch = d > 0
                fr = ch[W:]
                rows.append(
                    dict(
                        variant=v,
                        segment=s,
                        column=c,
                        rows_changed=int(ch.sum()),
                        forecast_rows_changed=int(fr.sum()),
                        max_abs=float(d.max()),
                        mean_abs=float(d.mean()),
                        mean_abs_forecast_rows=float(d[W:].mean()),
                        first_changed=str(new["date"][np.flatnonzero(ch)[0]]),
                        last_changed=str(new["date"][np.flatnonzero(ch)[-1]]),
                        corr_old_new=float(
                            np.corrcoef(new["X"][:, j], ref["X"][:, j])[0, 1]
                        )
                        if np.std(ref["X"][:, j]) > 0 and np.std(new["X"][:, j]) > 0
                        else np.nan,
                    )
                )
            gates.append(
                dict(
                    gate=f"{v} {s}: columns changed vs old local",
                    ok=True,
                    value=int(
                        sum(1 for r in rows if r["variant"] == v and r["segment"] == s)
                    ),
                )
            )
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "design_diff.csv", index=False)
    pd.DataFrame(gates).to_csv(OUT / "design_gates.csv", index=False)
    print(pd.DataFrame(gates).to_string())
    with pd.option_context("display.width", 250, "display.max_rows", 400):
        print(R.to_string())


def composite(variant: str) -> Path:
    """The design of record with the columns the variant changes (vs the old local rebuild,
    exactly) replaced by the variant's: every other column stays the record's, bit for bit."""
    exp_dir = WORK / f"linear_{variant}" / "export"
    exp_dir.mkdir(parents=True, exist_ok=True)
    rec = pd.read_parquet(RECORD_DESIGN)
    names = [c for c in rec.columns if c not in DESIGN_META]
    new = load_npz(design_dir(variant) / "design_bar1600_all_features.npz")
    ref = load_npz(OLD_LOCAL / "design_bar1600_all_features.npz")
    assert [str(x) for x in new["names"]] == names
    ch = np.flatnonzero((new["X"] != ref["X"]).any(axis=0))
    out = rec.copy()
    for j in ch:
        out[names[j]] = new["X"][:, j]
    import close_studies_export as CSE

    CSE.write_parquet(out, exp_dir / "design_bar1600_all_features.parquet")
    for f in ("targets.parquet",):
        if not (exp_dir / f).is_file():
            shutil.copy2(RECORD / f, exp_dir / f)
    (exp_dir / "replaced_columns.json").write_text(json.dumps([names[j] for j in ch]))
    say(f"composite {variant}: {len(ch)} columns replaced")
    return exp_dir


# ============================================================================ linear (close_linear_pretune, repointed)
def clp_patch(variant: str):
    """Point experiments/close_linear_pretune.py's module paths at _work/linear_<variant>/.  Runs at
    import in every process (pool workers re-import this script), so no worker can write into the
    2026-10-06 study's folders."""
    import close_linear_pretune as CLP

    root = WORK / f"linear_{variant}"
    CLP.OUT = root
    CLP.WORK = root / "_work"
    CLP.DESIGN_DIR = design_dir(variant)
    CLP.ARR = CLP.WORK / "arrays"
    CLP.WALK = CLP.WORK / "walk"
    CLP.FOLDS = CLP.WORK / "folds"
    CLP.EXPORT = root / "export"
    CLP.NETLIB = CLP.WORK / "netlib"
    CLP.KERNEL_DLL = CLP.WORK / "close_exogpen_kernel.dll"
    CLP.LAPACK = "netlib"
    CLP.BUCKETS = ("all_features", "baseline")
    if not getattr(CLP, "_datafix_guard", False):
        _ORIG["run_task"] = CLP.run_task
        CLP.run_task = guarded_run_task
        CLP._datafix_guard = True
    return CLP


_ORIG: dict = {}


def guarded_run_task(job: dict) -> dict:
    """close_linear_pretune.run_task, refusing to run unless the module points into this folder."""
    import close_linear_pretune as CLP

    assert str(CLP.WALK).startswith(str(WORK)) and str(CLP.FOLDS).startswith(
        str(WORK)
    ), CLP.WALK
    return _ORIG["run_task"](job)


if os.environ.get(
    "DATAFIX_VARIANT"
):  # pool workers: same patch before any task is unpickled
    clp_patch(os.environ["DATAFIX_VARIANT"])


def linear(variant: str) -> None:
    os.environ["DATAFIX_VARIANT"] = variant
    os.environ.setdefault("CLP_BUCKETS", "all_features")
    CLP = clp_patch(variant)
    composite(variant)
    CLP.WORK.mkdir(parents=True, exist_ok=True)
    if not (CLP.NETLIB / "liblapack.dll").is_file():
        shutil.copytree(OLD_STUDY / "_work" / "netlib", CLP.NETLIB, dirs_exist_ok=True)
    CLP.ARR.mkdir(parents=True, exist_ok=True)
    if not (CLP.ARR / "spec_grids.json").is_file():
        shutil.copy2(
            OLD_STUDY / "_work" / "arrays" / "spec_grids.json",
            CLP.ARR / "spec_grids.json",
        )
    if not (CLP.ARR / "meta.json").is_file():
        CLP.prep()
    CLP.compute()


def determinism_gate() -> dict:
    """One walk of the 2026-10-06 study (ridge, one penalty, spec grid) recomputed here on its own
    arrays: the kernel and library are deterministic on this host, so the old forecasts are a valid
    control for the fix's walks."""
    import close_linear_pretune as CLP

    CLP.ARR = OLD_STUDY / "_work" / "arrays"
    CLP.NETLIB = OLD_STUDY / "_work" / "netlib"
    CLP.LAPACK = "netlib"
    A = CLP.arrays("all_features")
    Xa, y, W = A["bar_Xa"], A["bar_y"], A["W"]
    bl = CLP.blocks_of(len(Xa) - W)
    m = Xa.shape[1]
    lk = np.zeros((len(bl), 1, m), np.uint8)
    mk = np.zeros((len(bl), 1, m), np.uint8)
    pf = np.ones((len(bl), 1, m))
    for k, (i0, _) in enumerate(bl):
        lk[k, 0], mk[k, 0], pf[k, 0] = CLP.structure(Xa[i0 : i0 + W], A["bb_aug"], 1.0)
    bstart = np.array([i0 for i0, _ in bl] + [len(Xa) - W], np.int32)
    o = CLP.krun(Xa, y, W, "ridge", bstart, lk, mk, pf, CLP.spec_grid("ridge"))
    ref = np.load(OLD_STUDY / "_work" / "walk" / "all_features" / "ridge_one_grid.npz")[
        "pred"
    ]
    return dict(
        gate="determinism: 2026-10-06 ridge one-penalty causal walk recomputed here",
        value=float(np.abs(o["pred"] - ref).max()),
        ok=bool(np.array_equal(o["pred"], ref)),
    )


# ============================================================================ score
def score(variants: tuple[str, ...] = ("fix", "fix_fomc")) -> None:  # noqa: C901
    import close_linear_pretune as CLP0

    old = pd.read_parquet(OLD_STUDY / "forecasts.parquet")
    master = pd.read_csv(
        REPO / "results" / "close_master_table" / "master_table.csv"
    ).set_index("key")
    rows, gates = [], [determinism_gate()]
    S = None
    preds: dict = {}
    for v in variants:
        CLP = clp_patch(v)
        CLP._A.clear()
        CLP._ST.clear()
        if S is None:
            S = CLP.Scorer()
        meta = json.loads((CLP.ARR / "meta.json").read_text())
        fdates = pd.to_datetime(meta["_dates"])[meta["_W"] :]
        for est in CLP.EST:
            sh = CLP.SHORT[est]
            P = CLP.plan("all_features", est, CLP.causal_choices("all_features", est))
            bl = P["blocks"]
            for arm, spec in P["arms"].items():
                if arm.endswith("_w"):
                    continue
                if arm in ("a_causal", "b_causal_bb"):
                    pred = np.load(
                        CLP.walk_file("all_features", est, spec["r"], "grid")
                    )["pred"]
                else:
                    pred = CLP.stitch(
                        "all_features", est, spec["r"], spec["alphas"], bl
                    )
                sid = f"linear_pretune/{sh}_all_features_{arm}"
                os_ = old[old["series_id"] == sid]
                o = pd.Series(
                    os_["pred_adj"].to_numpy(float),
                    index=pd.DatetimeIndex(pd.to_datetime(os_["date"])),
                )
                o = o.reindex(fdates).to_numpy(float)
                assert np.isfinite(o).all(), sid
                dn, do = S.daily(pred), S.daily(o)
                sn, so = S.stats(dn), S.stats(do)
                cn = S.contrast(dn, do)
                cols = S.contrast(dn, S.ols)
                preds[(v, sh, arm)] = pred
                rel = np.abs(pred / o - 1)
                row = dict(
                    variant=v,
                    est=sh,
                    arm=arm,
                    alphas_new=json.dumps(spec["alphas"]),
                    qlike_old=so["qlike"],
                    qlike_new=sn["qlike"],
                    dqlike=sn["qlike"] - so["qlike"],
                    dq_pct_vs_old=cn["dq_pct"],
                    dm_vs_old=cn["dm"],
                    dm_p_vs_old=cn["dm_p"],
                    sharpe_mid_old=so["sharpe_mid"],
                    sharpe_mid_new=sn["sharpe_mid"],
                    sharpe_crossed_old=so["sharpe_crossed"],
                    sharpe_crossed_new=sn["sharpe_crossed"],
                    hac_t_mid_vs_old=cn["hac_t_mid"],
                    positions_changed=cn["n_diff_position"],
                    dm_vs_ols_new=cols["dm"],
                    pred_max_rel_change=float(rel.max()),
                    pred_median_rel_change=float(np.median(rel)),
                    sessions_changed_1e9=int((rel > 1e-9).sum()),
                )
                if arm == "f_retune_margin_bb":
                    mk = master.loc[f"subpt_{sh}_all_features"]
                    row.update(
                        master_qlike=float(mk["qlike_raw"]),
                        master_sharpe_mid=float(mk["Sharpe_mid"]),
                        master_sharpe_crossed=float(mk["Sharpe_crossed"]),
                        master_dm_vs_ref=float(mk["dm_vs_ref"]),
                    )
                    gates.append(
                        dict(
                            gate=f"scorer on the old f arm == master table subpt_{sh}_all_features (QLIKE, Sharpe mid, crossed)",
                            value=max(
                                abs(so["qlike"] - mk["qlike_raw"]),
                                abs(so["sharpe_mid"] - mk["Sharpe_mid"]),
                                abs(so["sharpe_crossed"] - mk["Sharpe_crossed"]),
                            ),
                            ok=bool(
                                max(
                                    abs(so["qlike"] - mk["qlike_raw"]),
                                    abs(so["sharpe_mid"] - mk["Sharpe_mid"]),
                                    abs(so["sharpe_crossed"] - mk["Sharpe_crossed"]),
                                )
                                < 1e-9
                            ),
                        )
                    )
                rows.append(row)
        # the f arm's new forecasts, for the record
    del CLP0
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "linear_scores.csv", index=False)
    pd.DataFrame(gates).to_csv(OUT / "linear_gates.csv", index=False)
    fc = []
    for (v, sh, arm), p in preds.items():
        fc.append(
            pd.DataFrame(
                dict(
                    variant=v,
                    series_id=f"linear_pretune/{sh}_all_features_{arm}",
                    date=pd.to_datetime(meta["_dates"])[meta["_W"] :],
                    pred_adj=p,
                )
            )
        )
    pd.concat(fc).to_parquet(OUT / "linear_forecasts.parquet", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(pd.DataFrame(gates).to_string())
        print(R.drop(columns=["alphas_new"]).round(4).to_string())


# ============================================================================ pooled (estimate)
# the series the fix touches (the pooled panel of record has no FOMC columns, so fix_fomc == fix there)
POOLED_SERIES = [
    "spread_vwstock",
    "sumvolume",
    "effspread_ewstock",
    "effspread_vwstock",
]


def pooled_cols(variant: str) -> tuple[np.ndarray, list[str], np.ndarray]:
    """The pooled panel's columns of POOLED_SERIES under the variant, built by
    run_geometry_local.prepare_full's own steps (each exog column's chain is independent of the
    others: rows are dropped on RV only, every transform / ladder / scale is per column)."""
    import src.backtest.executor as EX
    from src.features.extractors.har import resolve_har_lags
    from src.features.transforms.scaling import rolling_robust_scale

    set_variant(variant)
    df, adj = EX.load_and_transform(
        str(data_dir()),
        POOLED_SERIES,
        target_use_diurnal=True,
        target_winsor_window=240,
        dropna_with_exog=False,
        overnight_fill=True,
        impute_indicate=True,
        diurnal_mode="divide",
    )
    rungs, x = [], 1.0
    while round(x) <= resolve_har_lags()[-1]:
        rungs.append(int(round(x)))
        x *= 2.0
    df, names = EX._build_har_and_calendar(
        df, adj, add_calendar=True, har_lags=sorted(set(rungs))
    )
    df = df.iloc[resolve_har_lags()[-1] :].reset_index(drop=True)
    keep = [n for n in names if any(s in n for s in POOLED_SERIES)]
    X = df[keep].values.astype(np.float64)
    TW = 24_000
    ref_iqr, fixed = EX._build_scale_guards(X, df, keep)
    Xs = rolling_robust_scale(X, TW, ref_iqr=ref_iqr, fixed_cols=fixed)
    Xs = EX.apply_masked_scaling(Xs, X, df, keep, TW)
    return Xs, keep, df["t"].to_numpy()


def pooled() -> None:
    import zipfile

    sys.path.insert(0, str(REPO / "experiments" / "puffer_forecast"))
    res = {}
    for v in ("old", "fix"):
        f = WORK / f"pooled_{v}.npz"
        if not f.is_file():
            X, names, t = pooled_cols(v)
            np.savez(
                f, X=X, names=np.array(names), t=t.astype("datetime64[ns]").astype(str)
            )
        z = np.load(f)
        res[v] = (z["X"], [str(n) for n in z["names"]], pd.to_datetime(z["t"]))
        say(f"pooled {v}: {res[v][0].shape}")
    Xo, names, t = res["old"]
    gates = []
    # control: the old rebuild vs the cache's own columns (memmap; the cache is read, never written)
    if POOLED_CACHE.is_file():
        import prep_pooled as PP

        Xc = PP.member_memmap(POOLED_CACHE, "X.npy")
        with zipfile.ZipFile(POOLED_CACHE) as zf, zf.open("names.npy") as fh:
            cn = [str(x) for x in np.lib.format.read_array(fh, allow_pickle=True)]
        common = [n for n in names if n in cn]
        nrow = min(len(Xo), Xc.shape[0])
        diffs = {
            n: float(
                np.nanmax(
                    np.abs(
                        np.asarray(Xc[:nrow, cn.index(n)]) - Xo[:nrow, names.index(n)]
                    )
                )
            )
            for n in common
        }
        gates.append(
            dict(
                gate="pooled cache rows vs this rebuild's rows",
                value=f"{Xc.shape[0]} / {len(Xo)}",
                ok=Xc.shape[0] == len(Xo),
            )
        )
        gates.append(
            dict(
                gate="old rebuild == the cache on these columns (max abs diff over columns)",
                value=json.dumps({k: round(v, 12) for k, v in diffs.items()}),
                ok=bool(max(diffs.values()) < 1e-9),
            )
        )
    rows = []
    test0 = 246_064  # the pooled NN walk's first test row; the forest forecasts from row 24,000
    t_test = t[test0] if len(t) > test0 else pd.NaT
    for v in ("fix",):
        Xn, nn, _ = res[v]
        assert nn == names
        for j, c in enumerate(names):
            d = np.abs(Xn[:, j] - Xo[:, j])
            ch = d > 0
            if not ch.any():
                continue
            rows.append(
                dict(
                    variant=v,
                    column=c,
                    rows_changed=int(ch.sum()),
                    rows_changed_from_2020=int(
                        ch[np.asarray(t >= pd.Timestamp("2020-01-01"))].sum()
                    ),
                    rows_changed_nn_test=int(ch[test0:].sum()),
                    max_abs=float(d.max()),
                    mean_abs=float(d.mean()),
                    first_changed=str(t[np.flatnonzero(ch)[0]])[:16],
                    last_changed=str(t[np.flatnonzero(ch)[-1]])[:16],
                )
            )
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "pooled_diff.csv", index=False)
    pd.DataFrame(gates).to_csv(OUT / "pooled_gates.csv", index=False)
    say(f"pooled NN test starts {t_test}")
    with pd.option_context("display.width", 250, "display.max_rows", 300):
        print(pd.DataFrame(gates).to_string())
        print(R.to_string())


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    if stage == "audit":
        audit()
        design_record_checks()
    elif stage == "capture":
        capture(*sys.argv[2:5])
    elif stage == "designs":
        designs()
    elif stage == "diff":
        diff()
    elif stage == "linear":
        linear(sys.argv[2])
    elif stage == "score":
        score(tuple(sys.argv[2:]) or ("fix", "fix_fomc"))
    elif stage == "pooled":
        pooled()
    else:
        raise SystemExit(__doc__)
