"""Score the DIRECT rest-of-day arms against the honest one-step remaining forecast.

Campaign: specs/causal_tune_rest_of_day.py.  For an entry clock c (09:30 ..
15:30 ET) an arm forecasts, on the adjusted (sqrt-diurnal) scale, the realized
variance from c to the 16:00 close, RV_rem(d, c), on the row stamped c+30
(naive ET, bar-END labelled).  This script turns every arm into variance
forecasts and scores them against the at-entry one-step construction of
experiments/ft_remaining.py on identical rows.

Forecasts of RV_rem(d, c), all issued at c of day d:
  plain      pred_adj^2 * baseline_rem  (baseline_rem = the arm's causal
             per-clock diurnal profile of RV_rem, from its targets sidecar)
  causal     (pred_adj^2 + s) * baseline_rem, s = the mean of the arm's squared
             adjusted-scale errors (true_adj - pred_adj)^2 over the previous
             SMEAR_W sessions (at least SMEAR_MIN), lagged one session:
             score_linear_subsection_causal.causal_forecasts, imported (an arm
             is one clock, so its per-label term is the arm's own)
  as scored  the spec's own pred_raw: its Duan term is the mean squared error
             over the arm's WHOLE out-of-sample run -- LOOK-AHEAD, reference only
  one-step   F_rem(c) = f_next(c) / w_next(c), ft_remaining on the per-bar table
             yhat_sub_<short>_<bucket>.parquet of the same estimator and bucket
             (the forecast of the bar c..c+30, extended by the causal
             prior-days diurnal share of that bar in the rest of the day)

Realized y = RV_rem from the targets sidecar (raw, unclipped; complete rows,
y > 0).  Every forecast is scored on the COMMON rows where all of them are
finite and > 0 and where the one-step's diurnal profile is warm on EVERY
remaining clock: ft_remaining normalizes w_next over whichever remaining clocks
have passed their warm-up, and the per-bar arms start on different dates, so on
the first sessions a late clock can be missing from the denominator (F_rem too
small); those rows are dropped and counted (n_profile_partial).
QLIKE = y/F - log(y/F) - 1 (src.evaluation.diebold_mariano.qlike_per_bar);
DM = dm_test on the per-session loss difference a - b (dm < 0: a better).

GATES (a failure stops the run)
  layout    est / bucket / clock read from the path and the filename agree with
            the campaign's outer <bucket>/rod<CLOCK>/<est>/tw<N> directories;
            every stamp of a file is the bar c+30
  join      every results row has a complete sidecar row, true_adj == adj_RV_rem
  09:30     prepending 09:30 to ft_remaining.TRADE_CLOCKS leaves F_rem at the
            shipped clocks 10:00 .. 15:30 bit-identical
CHECKS (printed; a failed self-check sets exit status 1 after the csv is written)
  identity  at 15:30 the direct arm IS the per-bar bar1600 arm and w_next = 1,
            so plain F equals the one-step F_rem (max rel <= GATE_REL) and the
            two QLIKEs agree to float noise
  RV        RV_rem against the sum of the per-bar table's rv_raw over the bars
            stamped c+30 .. 16:00 on the common rows: max relative difference and
            the count beyond build_subsection_yhat.GATE_REL (same panel: ~1e-12)

Arm layouts found under --root (both):
  campaign  <ROOT>/<bucket>/rod<CLOCK>/<est>/tw<N>/causal_tune_rest_of_day/<est>/<bucket>/
  bare      <ROOT>/causal_tune_rest_of_day/<est>/<bucket>/          (one HPC_RESULT_DIR)
each holding results_bar<HHMM>.csv + targets_bar<HHMM>.csv (HHMM = c+30).
Directories named in EXCLUDE below --root are skipped (the local gate runs).

Output: <ROOT>/score_rest_of_day.csv, one row per bucket x estimator x clock
(x train window), and a printed per-clock table.

Run:  python experiments/score_rest_of_day.py --root <ROOT> [--ref-dir DIR] [--out CSV]
      (default ref dir: <ROOT>/ref if it exists, else results/spxw_pnl)
Needs only numpy / pandas plus the repo's src/ and experiments/.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import ft_remaining  # noqa: E402
from build_subsection_yhat import ESTIMATORS, GATE_REL  # noqa: E402
from score_linear_subsection_causal import (  # noqa: E402
    SMEAR_MIN,
    SMEAR_W,
    causal_forecasts,
)
from src.evaluation.diebold_mariano import dm_test, qlike_per_bar  # noqa: E402

SPEC = ROOT / "specs" / "causal_tune_rest_of_day.py"
SPEC_DIR = "causal_tune_rest_of_day"  # the spec's output subdirectory (its arm_dir)
EXCLUDE = (
    "local_gates",
)  # gate runs kept beside the campaign root, never scored with it
ARM_FILE = re.compile(r"^(results|targets)_bar(\d{4})\.csv$")
ET = "America/New_York"  # the panel's naive stamps are ET wall clock


def spec_constants(names: tuple[str, ...] = ("CLOSE_MIN", "BAR_MIN", "CLOCKS")) -> dict:
    """The spec's own clock constants, executed from its source.

    Importing the spec is not an option (it installs its executor wrappers,
    chdirs and loads the linear machinery), so the named top-level assignments
    are pulled out of its AST -- the idiom the spec itself uses on
    causal_tune_linear.py."""
    tree = ast.parse(SPEC.read_text(encoding="utf-8"))
    body: list[ast.stmt] = [
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and {t.id for t in n.targets if isinstance(t, ast.Name)} & set(names)
    ]
    ns: dict = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(SPEC), "exec"), ns)
    missing = set(names) - set(ns)
    assert not missing, f"{SPEC.name} no longer defines {sorted(missing)}"
    return {k: ns[k] for k in names}


_SPEC = spec_constants()
CLOSE_MIN, BAR_MIN, CLOCKS = _SPEC["CLOSE_MIN"], _SPEC["BAR_MIN"], list(_SPEC["CLOCKS"])
SHIPPED_TRADE_CLOCKS = list(ft_remaining.TRADE_CLOCKS)  # bar starts 10:00 .. 15:30


def hhmm(clock: str) -> str:
    """'1230' -> '12:30'."""
    return f"{clock[:2]}:{clock[2:]}"


def _shift(clock: str, minutes: int) -> str:
    m = int(clock[:2]) * 60 + int(clock[2:]) + minutes
    return f"{m // 60:02d}{m % 60:02d}"


def bar_of(clock: str) -> str:
    """Entry clock c -> the stamp c+30 of the row carrying its forecast (HHMM)."""
    return _shift(clock, BAR_MIN)


def clock_of(bar: str) -> str:
    """Stamp c+30 (HHMM) -> the entry clock c."""
    return _shift(bar, -BAR_MIN)


assert bar_of(CLOCKS[-1]) == f"{CLOSE_MIN // 60:02d}{CLOSE_MIN % 60:02d}", CLOCKS
assert [hhmm(c) for c in CLOCKS[1:]] == SHIPPED_TRADE_CLOCKS, (
    CLOCKS,
    SHIPPED_TRADE_CLOCKS,
)


def extend_trade_clocks() -> None:
    """Admit the first entry clock (CLOCKS[0] = 09:30, serving the 09:35 entry) to ft_remaining.

    ft_remaining.TRADE_CLOCKS starts at the 10:00 bar start.  Both
    ft_remaining.panel_bars and ft_remaining.causal_slice_share read that MODULE
    GLOBAL at call time, so setting the attribute before calling them is
    enough.  Idempotent; the 09:30 gate in onestep_table checks it moves no
    other clock."""
    want = [hhmm(c) for c in CLOCKS]
    if ft_remaining.TRADE_CLOCKS != want:
        assert ft_remaining.TRADE_CLOCKS == want[1:], ft_remaining.TRADE_CLOCKS
        ft_remaining.TRADE_CLOCKS = [want[0]] + ft_remaining.TRADE_CLOCKS
    assert ft_remaining.TRADE_CLOCKS == want


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------


def find_arms(root: Path, exclude: tuple[str, ...] = EXCLUDE) -> dict:
    """{(bucket, est, clock, tw): {"results": Path, "targets": Path}} under ``root``.

    Either file may be absent (an orphan); ``tw`` is None in the bare layout."""
    arms: dict = {}
    for p in sorted(root.rglob("*_bar*.csv")):
        m = ARM_FILE.match(p.name)
        if not m or p.parent.parent.parent.name != SPEC_DIR:
            continue
        rel = p.relative_to(root).parts
        if any(x in exclude for x in rel):
            continue
        kind, bar = m.groups()
        est, bucket = p.parent.parent.name, p.parent.name
        clock = clock_of(bar)
        assert clock in CLOCKS, f"{p}: bar{bar} is no entry clock's row"
        outer = rel[
            :-4
        ]  # the campaign layout: (..., <bucket>, rod<CLOCK>, <est>, tw<N>)
        tws = [int(x[2:]) for x in outer if re.fullmatch(r"tw\d+", x)]
        rods = [i for i, x in enumerate(outer) if re.fullmatch(r"rod\d{4}", x)]
        if rods:
            i = rods[-1]
            assert outer[i][3:] == clock, f"{p}: {outer[i]} holds bar{bar}"
            assert i >= 1 and outer[i - 1] == bucket, f"{p}: outer bucket != {bucket}"
            assert len(outer) > i + 1 and outer[i + 1] == est, (
                f"{p}: outer estimator != {est}"
            )
        key = (bucket, est, clock, tws[-1] if tws else None)
        slot = arms.setdefault(key, {})
        assert kind not in slot, f"two {kind} files for {key}: {slot[kind]} and {p}"
        slot[kind] = p
    return arms


def read_arm(res_path: Path, tgt_path: Path, clock: str) -> pd.DataFrame:
    """One arm's results joined to its targets sidecar on the stamp c+30 (naive ET).

    Indexed by the stamp, sorted.  Gates: every stamp of both files is the bar
    c+30, no stamp repeats, every results row has a complete sidecar row, and
    the results' true_adj is the sidecar's adj_RV_rem (the join is the right
    one)."""
    bar = bar_of(clock)
    res = pd.read_csv(res_path, parse_dates=["date"])
    tg = pd.read_csv(tgt_path, parse_dates=["t"])
    for name, s in (("results", res["date"]), ("targets", tg["t"])):
        assert (s.dt.strftime("%H%M") == bar).all(), (
            f"{name} of {clock}: a stamp is not bar{bar}"
        )
        assert not s.duplicated().any(), f"{name} of {clock}: a stamp repeats"
    j = res.merge(
        tg.rename(columns={"t": "date"}),
        on="date",
        how="left",
        validate="1:1",
        indicator=True,
    )
    assert (j["_merge"] == "both").all(), (
        f"{res_path}: {int((j['_merge'] != 'both').sum())} results rows lack a sidecar row"
    )
    assert j["complete"].astype(bool).all(), (
        f"{res_path}: a results row has an incomplete rest of day"
    )
    rel = float((j["true_adj"] / j["adj_RV_rem"] - 1.0).abs().max())
    assert rel <= GATE_REL, (
        f"{res_path}: true_adj vs sidecar adj_RV_rem max rel {rel:.1e}"
    )
    return j.drop(columns="_merge").set_index("date").sort_index()


# ---------------------------------------------------------------------------
# One-step comparator
# ---------------------------------------------------------------------------


def onestep_table(path: Path) -> pd.DataFrame:
    """ft_remaining's F_rem per (day, entry clock HH:MM) on the per-bar table ``path``.

    Adds ``warm`` (the causal profile is past its warm-up on every remaining
    clock, so w_next's denominator is the full rest of the day) and
    ``rv_rem_ref`` (the sum of the table's rv_raw over the remaining bars;
    NaN unless every one of them is in the table).  Gate: the 09:30 prepend
    leaves F_rem at the shipped clocks bit-identical."""
    extend_trade_clocks()
    clocks = list(ft_remaining.TRADE_CLOCKS)
    bars = ft_remaining.panel_bars(str(path))
    missing = sorted(set(clocks) - set(bars["clock"]))
    assert not missing, f"{path.name} has no bars starting at {missing}"
    ft = ft_remaining.ft_from_bars(bars)

    ft_remaining.TRADE_CLOCKS = SHIPPED_TRADE_CLOCKS
    try:
        ft0 = ft_remaining.ft_remaining(str(path))
    finally:
        extend_trade_clocks()
    m = ft0.merge(
        ft, on=["day", "clock"], how="left", suffixes=("_0", ""), validate="1:1"
    )
    same_nan = bool((m["F_rem_0"].isna() == m["F_rem"].isna()).all())
    both = m["F_rem_0"].notna()
    exact = bool((m.loc[both, "F_rem_0"] == m.loc[both, "F_rem"]).all())
    assert same_nan and exact, (
        f"{path.name}: prepending 09:30 moved F_rem at a shipped clock"
    )

    ws = ft_remaining.causal_slice_share(bars)
    wide = ws.pivot(index="day", columns="clock", values="w_next").reindex(
        columns=clocks
    )
    ok = wide.notna().to_numpy()
    warm = np.logical_and.accumulate(ok[:, ::-1], axis=1)[:, ::-1]
    warm = pd.DataFrame(warm, index=wide.index, columns=clocks)
    rv = bars.pivot(index="day", columns="clock", values="rv").reindex(columns=clocks)
    # remaining-bars sum c .. 15:30 from the right; NaN once any remaining bar is absent
    rv_rem = rv[clocks[::-1]].cumsum(axis=1, skipna=False)[clocks]

    def _long(
        w: pd.DataFrame, name: str
    ) -> pd.DataFrame:  # melt keeps NaN on any pandas
        w = w.rename_axis(index="day", columns="clock").reset_index()
        return w.melt(id_vars="day", var_name="clock", value_name=name)

    extra = _long(warm, "warm").merge(_long(rv_rem, "rv_rem_ref"), on=["day", "clock"])
    out = ft.merge(extra, on=["day", "clock"], how="left", validate="1:1")
    out["warm"] = out["warm"].astype("boolean").fillna(False).astype(bool)
    print(
        f"  one-step {path.name}: {len(out):,} (day, clock) rows, clocks "
        f"{clocks[0]} .. {clocks[-1]}; GATE 09:30 prepend leaves F_rem at the shipped "
        f"{len(SHIPPED_TRADE_CLOCKS)} clocks bit-identical ({len(ft0):,} rows)"
    )
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

KINDS = ("plain", "causal", "as_scored", "onestep")


def score_arm(key: tuple, paths: dict, one: pd.DataFrame | None, ref_name: str) -> dict:
    bucket, est, clock, tw = key
    j = read_arm(paths["results"], paths["targets"], clock)
    n_results = len(j)
    # the rows score_linear_subsection_causal.load_adj keeps (they feed the smear window)
    r = j[(j["true_adj"] > 0) & (j["true_raw"] > 0)].copy()
    r["baseline"] = r["baseline_rem"]
    r["e2"] = (r["true_adj"] - r["pred_adj"]) ** 2
    r["hhmm"] = r.index.strftime("%H:%M")
    r["day"] = r.index.normalize()
    cf = causal_forecasts(r)
    F = pd.DataFrame(
        {
            "plain": r["pred_adj"] ** 2 * r["baseline_rem"],
            "causal": cf["pred_clock"],
            "as_scored": r["pred_raw"],
        },
        index=r.index,
    )
    y = r["RV_rem"].where(r["complete"].astype(bool))
    warm = pd.Series(True, index=r.index)
    rv_ref = pd.Series(np.nan, index=r.index)
    if one is not None:
        o = one[one["clock"] == hhmm(clock)].set_index("day")
        F["onestep"] = o["F_rem"].reindex(r["day"]).to_numpy(float)
        warm = pd.Series(
            o["warm"].reindex(r["day"]).fillna(False).to_numpy(bool), index=r.index
        )
        rv_ref = pd.Series(
            o["rv_rem_ref"].reindex(r["day"]).to_numpy(float), index=r.index
        )
    Fv = F.to_numpy(float)
    yv = y.to_numpy(float)
    fin = (np.isfinite(Fv) & (Fv > 0)).all(axis=1) & np.isfinite(yv) & (yv > 0)
    common = fin & warm.to_numpy(bool)
    c = F.index[common]
    out: dict = {
        "bucket": bucket,
        "estimator": est,
        "train_win": tw,
        "clock": clock,
        "bar": bar_of(clock),
        "onestep_ref": ref_name,
        "n_results": n_results,
        "n_scorable": int(fin.sum()),
        "n_profile_partial": int((fin & ~warm.to_numpy(bool)).sum()),
        "n": int(common.sum()),
        "first": c.min().date() if len(c) else None,
        "last": c.max().date() if len(c) else None,
    }
    loss = {
        k: qlike_per_bar(F.loc[c, k].to_numpy(float), y.loc[c].to_numpy(float))
        for k in F
    }
    for k in KINDS:
        out[f"qlike_{k}"] = float(np.mean(loss[k])) if k in loss and len(c) else np.nan
    if "onestep" in loss and len(c):
        for a in ("causal", "plain"):
            dm = dm_test(loss[a], loss["onestep"])
            out[f"pct_{a}_vs_onestep"] = 100.0 * dm["mean_diff"] / out["qlike_onestep"]
            out[f"dm_{a}_vs_onestep"] = dm["dm"]
            out[f"p_{a}_vs_onestep"] = dm["p"]
        yy, rr = y.loc[c].to_numpy(float), rv_ref.loc[c].to_numpy(float)
        both = np.isfinite(yy) & np.isfinite(rr) & (rr > 0)
        rel = np.abs(yy[both] / rr[both] - 1.0)
        out["rv_xcheck_n"] = int(both.sum())
        out["rv_xcheck_max_rel"] = float(rel.max()) if both.any() else np.nan
        out["rv_xcheck_n_beyond_gate"] = int((rel > GATE_REL).sum())
        if clock == CLOCKS[-1]:
            fr = float(np.max(np.abs(F.loc[c, "plain"] / F.loc[c, "onestep"] - 1.0)))
            dq = abs(out["qlike_plain"] - out["qlike_onestep"])
            out["identity_max_rel_F"] = fr
            out["identity_abs_dqlike"] = dq
            out["identity_pass"] = bool(fr <= GATE_REL)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--root", required=True, help="campaign root (or one bare HPC_RESULT_DIR)"
    )
    ap.add_argument(
        "--ref-dir",
        default=None,
        help="dir with yhat_sub_{ridge,lasso,enet}_{bucket}.parquet "
        "(default <root>/ref if it exists, else results/spxw_pnl)",
    )
    ap.add_argument(
        "--out", default=None, help="csv path (default <root>/score_rest_of_day.csv)"
    )
    a = ap.parse_args()
    root = Path(a.root)
    ref_dir = (
        Path(a.ref_dir)
        if a.ref_dir
        else (
            root / "ref" if (root / "ref").is_dir() else ROOT / "results" / "spxw_pnl"
        )
    )
    extend_trade_clocks()
    print(
        f"root {root}\nref dir {ref_dir}\nentry clocks {CLOCKS[0]} .. {CLOCKS[-1]} "
        f"({len(CLOCKS)}); causal smear window {SMEAR_W} sessions, min {SMEAR_MIN}, lag 1"
    )
    arms = find_arms(root)
    orphans = {k: v for k, v in arms.items() if len(v) < 2}
    for k, v in sorted(orphans.items(), key=str):
        have = next(iter(v))
        lack = "targets" if have == "results" else "results"
        print(f"ORPHAN {'/'.join(map(str, k))}: {have} without {lack} ({v[have]})")
    full = {k: v for k, v in arms.items() if len(v) == 2}
    print(f"{len(full)} arms with results + targets, {len(orphans)} orphans")
    if not full:
        return 1

    rows = []
    tables: dict = {}
    for key in sorted(full, key=lambda k: (k[0], k[1], str(k[3]), k[2])):
        bucket, est, clock, tw = key
        short = ESTIMATORS.get(est)
        ref = ref_dir / f"yhat_sub_{short}_{bucket}.parquet" if short else None
        if (bucket, est) not in tables:
            if ref is not None and ref.exists():
                tables[(bucket, est)] = onestep_table(ref)
            else:
                print(
                    f"  NO one-step table for {est}/{bucket} ({ref}): direct forecasts only"
                )
                tables[(bucket, est)] = None
        one = tables[(bucket, est)]
        ref_name = ref.name if (one is not None and ref is not None) else ""
        rows.append(score_arm(key, full[key], one, ref_name))

    tab = pd.DataFrame(rows)
    out_csv = Path(a.out) if a.out else root / "score_rest_of_day.csv"
    tab.to_csv(out_csv, index=False)

    pd.set_option("display.width", 250)
    show = tab[
        [
            "bucket",
            "estimator",
            "train_win",
            "clock",
            "n",
            "first",
            "last",
            "n_profile_partial",
        ]
        + [f"qlike_{k}" for k in KINDS]
        + [
            c
            for c in (
                "pct_causal_vs_onestep",
                "dm_causal_vs_onestep",
                "p_causal_vs_onestep",
                "pct_plain_vs_onestep",
                "dm_plain_vs_onestep",
                "p_plain_vs_onestep",
            )
            if c in tab
        ]
    ].rename(
        columns={
            "qlike_as_scored": "qlike_as_scored*",
            "n_profile_partial": "n_partial",
        }
    )
    print(
        "\nQLIKE on the common rows (* as scored = the spec's look-ahead back-transform, "
        "reference only); pct/dm/p: direct - one-step, dm < 0 = direct better"
    )
    print(show.to_string(index=False, float_format=lambda v: f"{v:.5g}"))
    if "rv_xcheck_n" in tab:
        print(
            f"\nRV cross-check (targets RV_rem vs the per-bar table's rv_raw summed over "
            f"c+30 .. 16:00, common rows; beyond = rel > GATE_REL {GATE_REL:g}):"
        )
        print(
            tab[
                [
                    "bucket",
                    "estimator",
                    "clock",
                    "rv_xcheck_n",
                    "rv_xcheck_max_rel",
                    "rv_xcheck_n_beyond_gate",
                ]
            ].to_string(index=False)
        )
    status = 0
    if "identity_pass" in tab:
        idt = tab[tab["identity_pass"].notna()]
        for _, r in idt.iterrows():
            verdict = "PASS" if r["identity_pass"] else "FAIL"
            print(
                f"SELF-CHECK identity {r['estimator']}/{r['bucket']} {hhmm(r['clock'])}: "
                f"max |F_plain/F_onestep - 1| = {r['identity_max_rel_F']:.2e} (bound {GATE_REL:g}); "
                f"QLIKE plain {r['qlike_plain']:.12f} vs one-step {r['qlike_onestep']:.12f} "
                f"(|diff| {r['identity_abs_dqlike']:.1e}) on {r['n']:,} rows -> {verdict}"
            )
            status |= 0 if r["identity_pass"] else 1
    print(f"\nwrote {out_csv} ({len(tab)} rows)")
    return status


if __name__ == "__main__":
    sys.exit(main())
