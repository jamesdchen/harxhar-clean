"""Stack the DIRECT rest-of-day forecasts into yhat tables in the per-bar table format.

Campaign: specs/causal_tune_rest_of_day.py -- one regression per entry clock c
in 09:30 .. 15:30 ET (13 clocks) whose target is RV_rem(d, c), the realized
variance from c to the 16:00 close.  For every estimator x bucket whose 13
clock arms all exist this writes

    <out-dir>/yhat_restofday_<short>_<bucket>.parquet      (<short>: ridge / lasso / enet)

with the columns of experiments/build_subsection_yhat.py's per-bar tables:

    t         the stamp c+30, UTC, microsecond unit (the arm's naive-ET stamp
              tz_localize("America/New_York") -> UTC, as build_subsection_yhat.read_arm)
    yhat      pred_adj -- the adjusted-scale (sqrt-diurnal) forecast issued at c
    baseline  baseline_rem -- the arm's causal per-clock diurnal profile of RV_rem
              (targets sidecar); yhat^2 * baseline is the variance forecast
    rv_raw    RV_rem -- the realized variance from c to 16:00, UNCLIPPED (sidecar;
              the results' true_raw is the winsorized target and is not carried)

STAMP CONVENTION.  The row stamped c+30 carries the forecast, ISSUED AT c, of the
variance from c to 16:00 -- the stamp experiments/ft_remaining.py gives
f_next(e) (the row e+30).  A hold-to-close notebook entering at e reads the row
stamped e+30 and trades s = F_rem - IV^2 * h with F_rem = its MZ-recalibrated
yhat^2 * baseline, exactly as it uses ft_remaining's F_rem.  Unlike a per-bar
table, rv_raw is ALREADY the rest-of-day variance: never sum rv_raw over a
day's rows, and recalibrate per clock (each clock is its own target).
Nothing from the spec's look-ahead back-transform (pred_raw) is carried.

Rows: the arm's results rows with true_adj > 0, true_raw > 0 and a finite
pred_adj (build_subsection_yhat.read_arm's filter).

GATES (a failure stops the run)
  per arm   the join of score_rest_of_day.read_arm (every stamp is the bar c+30,
            one sidecar row per results row, complete, true_adj == adj_RV_rem);
            baseline_rem == true_raw / true_adj^2 to GATE_REL on the rows the
            winsorization left alone (true_raw == RV_rem to GATE_REL; the max
            relative difference is printed on all rows and on those)
  per table no duplicate t; every t's ET wall clock is a bar c+30 (10:00 .. 16:00)
            and round-trips to the arm's naive stamp; yhat finite; baseline
            finite and > 0; rv_raw > 0 where finite

--allow-partial writes whatever clocks exist (the printout names the table
PARTIAL and lists the missing clocks); without it such a table is skipped.

Run:  python experiments/build_restofday_yhat.py [--root results/linear_subsection_restofday]
          [--out-dir results/spxw_pnl] [--allow-partial]
(directories named in score_rest_of_day.EXCLUDE below --root -- the local gate
runs -- are skipped.)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from build_subsection_yhat import ESTIMATORS, GATE_REL  # noqa: E402
from score_rest_of_day import CLOCKS, ET, bar_of, find_arms, hhmm, read_arm  # noqa: E402

BAR_STAMPS = {hhmm(bar_of(c)) for c in CLOCKS}  # 10:00 .. 16:00 ET


def arm_rows(paths: dict, clock: str) -> tuple[pd.DataFrame, dict]:
    """One arm as table rows (naive-ET stamp kept as ``stamp``) and its baseline check."""
    j = read_arm(paths["results"], paths["targets"], clock)
    ok = (j["true_adj"] > 0) & (j["true_raw"] > 0) & np.isfinite(j["pred_adj"])
    j = j[ok]
    implied = j["true_raw"] / j["true_adj"] ** 2
    rel = (j["baseline_rem"] / implied - 1.0).abs()
    clean = (
        j["true_raw"] / j["RV_rem"] - 1.0
    ).abs() <= GATE_REL  # winsorization left alone
    chk = {
        "rows": len(j),
        "dropped": int((~ok).sum()),
        "winsorized": int((~clean).sum()),
        "max_rel_all": float(rel.max()),
        "max_rel_clean": float(rel[clean].max()),
    }
    assert chk["max_rel_clean"] <= GATE_REL, (clock, chk)
    et = pd.DatetimeIndex(j.index).tz_localize(ET)  # naive ET, bar-end labelled
    rows = pd.DataFrame(
        {
            # microsecond resolution, as the production and per-bar tables store t
            "t": et.tz_convert("UTC").as_unit("us"),
            "yhat": j["pred_adj"].to_numpy(float),
            "baseline": j["baseline_rem"].to_numpy(float),
            "rv_raw": j["RV_rem"].to_numpy(float),
            "stamp": j.index.to_numpy(),
        }
    )
    return rows, chk


def gate_table(tab: pd.DataFrame, name: str) -> None:
    assert not tab["t"].duplicated().any(), f"{name}: a stamp repeats"
    et = tab["t"].dt.tz_convert(ET)
    assert et.dt.strftime("%H:%M").isin(BAR_STAMPS).all(), (
        f"{name}: a stamp is no bar c+30"
    )
    back = et.dt.tz_localize(None).to_numpy()
    assert (back == tab["stamp"].to_numpy()).all(), (
        f"{name}: a stamp does not round-trip"
    )
    assert np.isfinite(tab["yhat"]).all(), f"{name}: non-finite yhat"
    assert (np.isfinite(tab["baseline"]) & (tab["baseline"] > 0)).all(), (
        f"{name}: baseline <= 0"
    )
    rv = tab["rv_raw"]
    assert (rv[np.isfinite(rv)] > 0).all(), f"{name}: rv_raw <= 0"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--root", default=str(ROOT / "results" / "linear_subsection_restofday")
    )
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "spxw_pnl"))
    ap.add_argument(
        "--allow-partial",
        action="store_true",
        help="write tables with fewer than all 13 clocks (labelled PARTIAL)",
    )
    a = ap.parse_args()
    root, out_dir = Path(a.root), Path(a.out_dir)
    arms = find_arms(root)
    groups: dict = {}
    for (bucket, est, clock, tw), paths in arms.items():
        groups.setdefault((est, bucket), {}).setdefault(tw, {})[clock] = paths
    print(
        f"root {root}: {len(arms)} arm entries, {len(groups)} estimator x bucket groups"
    )
    written = 0
    for (est, bucket), by_tw in sorted(groups.items()):
        short = ESTIMATORS.get(est)
        assert short, f"unknown estimator {est}"
        assert len(by_tw) == 1, (
            f"{est}/{bucket}: several train windows {sorted(map(str, by_tw))}"
        )
        ((tw, arms_c),) = by_tw.items()
        name = f"yhat_restofday_{short}_{bucket}.parquet"
        orphans = sorted(c for c, p in arms_c.items() if len(p) < 2)
        have = [c for c in CLOCKS if c in arms_c and len(arms_c[c]) == 2]
        missing = [c for c in CLOCKS if c not in have]
        if orphans:
            print(
                f"  {name}: results/targets orphan at {', '.join(hhmm(c) for c in orphans)} "
                "(counted missing)"
            )
        if missing and not a.allow_partial:
            print(
                f"skip {name}: {len(missing)} of {len(CLOCKS)} clocks missing: "
                f"{', '.join(hhmm(c) for c in missing)}"
            )
            continue
        if not have:
            print(f"skip {name}: no complete arm")
            continue
        parts, checks = [], {}
        for c in have:
            rows, chk = arm_rows(arms_c[c], c)
            parts.append(rows)
            checks[c] = chk
        tab = pd.concat(parts).sort_values("t").reset_index(drop=True)
        gate_table(tab, name)
        label = (
            ""
            if not missing
            else (
                f" PARTIAL ({len(have)} of {len(CLOCKS)} clocks; missing "
                f"{', '.join(hhmm(c) for c in missing)})"
            )
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        tab.drop(columns="stamp").to_parquet(out_dir / name, index=False)
        written += 1
        et = tab["t"].dt.tz_convert(ET)
        print(
            f"wrote {out_dir / name}{label}: {len(tab):,} rows, "
            f"{et.dt.normalize().nunique():,} sessions, {et.min().date()} .. {et.max().date()} "
            f"(train window {tw})"
        )
        for c in have:
            k = checks[c]
            print(
                f"  entry {hhmm(c)} (row {hhmm(bar_of(c))}): {k['rows']:,} rows "
                f"({k['dropped']} dropped by the filter); baseline_rem vs true_raw/true_adj^2 "
                f"max rel {k['max_rel_all']:.1e} all rows, {k['max_rel_clean']:.1e} on the "
                f"{k['rows'] - k['winsorized']:,} unwinsorized (winsorized {k['winsorized']})"
            )
        print(
            f"  GATES {name}: t unique, ET clocks in {min(BAR_STAMPS)} .. {max(BAR_STAMPS)} and "
            f"round-trip, yhat finite, baseline > 0, rv_raw > 0 "
            f"({int(np.isfinite(tab['rv_raw']).sum()):,} finite) -- pass"
        )
    print(f"{written} tables written")


if __name__ == "__main__":
    main()
