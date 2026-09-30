"""I1 plan: every per-bar 16:00 LINEAR forecast the closing-strategy master table scores, re-run on
the de-duplicated per-bar design (commit 47f7f9c: the segment path no longer builds the twelve
har_ma_k_x_open / har_ma_k_x_close columns).

WHAT IS LISTED.  experiments/master_table_close.discover() is called (never re-implemented) and every
spec of the per-bar linear families is kept: the stacked tables yhat_sub_<est>_<bucket>.parquet (the
per-bar linear family and the VIX-only tables), the per-bar OLS incumbent, and the arm-only bucket
studies (VIX-only arms, implied-vol representations, chain-period buckets at 500 sessions -- table B --
and the HAR-ladder variants of live_feasible at 2000 sessions).  The 500-session HAR-ladder twins the
master table does not score (discover()'s skipped list) are not run.  76 forecasts.

HOW EACH IS RE-RUN.  specs/causal_tune_linear.py (read-only) with the env axes of the run that made the
old forecast: EXOG_BUCKET, ESTIMATOR (ridge | reclasso | reclasticnet; the OLS incumbent is written by
every run of the spec), TRAIN_WIN (2000, or 500 for the chain-period buckets and ivrep_slice_slope),
SEGMENT (one bar), LAG_SCOPE (global; intra for the HAR variant "intra"), HAR_BASE (2 / 3 for the
HAR variants base2 / base3; empty = production powers of 5).  The 16:00 bar only, EXCEPT the arms
behind a stacked table: the canonical tables hold the 13 regular-hours bars (10:00 .. 16:00) and the
stacking builders require all 13, so those 18 arms run every bar (a one-bar linear arm is 1-3 minutes).

WHERE.  New root results/linear_subsection_dedup/, mirroring the old layout one for one:
  <bucket>/<seg>/<est>/tw<tw>/causal_tune_linear/...           (was results/linear_subsection/<bucket>/...)
  arms_hoffman2/<bucket>/<est>/tw<tw>/results_<seg>.csv          (flat copies, built after the pull by
                                                                  experiments/rebuild_yhat_dedup.py)
  arms_carc/live_feasible/<est>/tw<tw>/results_<seg>.csv         (flat copies, build_vixonly's incumbent)
  <study>_carc/linear_subsection[_lassofix]/<bucket>/<seg>/...    (the bucket studies' pulled layout)
  mfiv_carc/linear_subsection_har/<variant>/live_feasible/...     (the HAR-ladder variants)
so a reader re-pointed from results/linear_subsection to results/linear_subsection_dedup finds every file.

Writes
  results/linear_subsection_dedup/arm_list.csv   one row per forecast (key, family, table, spec axes,
                                                 old / new result paths -- new_path is what a reader
                                                 of the mirrored layout opens, run_path the file the
                                                 cluster run wrote -- clusters)
  cluster/linear_dedup_tasks_canary.txt          the canary line
  cluster/linear_dedup_tasks_fleet.txt           one line per task: a pack of one-bar arms
      "<results_root> <bucket> <estimator> <train_win> <har_base|-> <lag_scope> <seg1,seg2,...>"

Run:  python experiments/linear_dedup_plan.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "experiments", ROOT / "notebooks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import master_table_close as mt  # noqa: E402

OLD = "results/linear_subsection"
NEW = "results/linear_subsection_dedup"
OUT = ROOT / NEW
TASKS_CANARY = ROOT / "cluster" / "linear_dedup_tasks_canary.txt"
TASKS_FLEET = ROOT / "cluster" / "linear_dedup_tasks_fleet.txt"
BARS = (
    "bar1000",
    "bar1030",
    "bar1100",
    "bar1130",
    "bar1200",
    "bar1230",
    "bar1300",
    "bar1330",
    "bar1400",
    "bar1430",
    "bar1500",
    "bar1530",
    "bar1600",
)
SEG = "bar1600"
PACK = 4  # one-bar arms per array task for the 13-bar arms (13 -> 4 + 3 + 3 + 3)
EST_OF = {"ridge": "ridge", "lasso": "reclasso", "enet": "reclasticnet"}
MAIN_BUCKETS = ("baseline", "live_feasible", "all_features")
# the column counts of the de-duplicated per-bar design (commit 47f7f9c message)
P_NEW = {"baseline": 22, "live_feasible": 232, "all_features": 628}
# the study each bucket's arms were pulled into (compare_mfiv_harlag.FAMILIES)
PULLED = {b: fam["pulled"] for fam in mt.cmh.FAMILIES.values() for b in fam["tw"]}
FAMILY_OF_STUDY = {
    "vixonly": "vixonly",
    "free": "free",
    "ivrep": "ivrep",
    "mfiv": "mfiv",
    "ivslice": "ivslice",
}
CARC_STUDIES = {"vixonly_carc", "free_carc", "ivrep_carc", "mfiv_carc", "ivslice_carc"}


def rel(p: str | Path) -> str:
    return Path(p).resolve().relative_to(ROOT).as_posix()


def nested(root: str, bucket: str, seg: str, est: str, tw: int) -> str:
    return f"{root}/{bucket}/{seg}/{est}/tw{tw}/causal_tune_linear/{est}/{bucket}/results_{seg}.csv"


def main() -> None:
    specs, skipped = mt.discover()
    keep = {
        "per-bar linear",
        "VIX-only family",
        "implied-vol representations",
        "HAR-ladder variants",
        "chain-period buckets",
    }
    rows = []
    for s in specs:
        if s.family not in keep:
            continue
        run = ""
        r: dict = {"key": s.key, "family": s.family, "source": s.source}
        m = s.meta
        if s.key == "sub_ols_baseline":
            bucket, est, tw, har, lag = "baseline", "ols", 2000, "", "global"
            root = NEW
            old = rel(s.path)
            new = f"{NEW}/baseline/{SEG}/reclasso/tw{tw}/causal_tune_linear/incumbent_ols/results_{SEG}.csv"
            segs: tuple[str, ...] = (SEG,)
            run_est = "reclasso"  # the OLS incumbent rides on the baseline reclasso run
            table = ""
        elif s.family == "HAR-ladder variants":
            v = m["variant"]
            bucket, est, tw = "live_feasible", EST_OF[m["est"]], 2000
            har = {"base2": "2", "base3": "3", "intra": ""}[v]
            lag = "intra" if v == "intra" else "global"
            root = f"{NEW}/mfiv_carc/linear_subsection_har/{v}"
            old = rel(s.path)
            new = nested(root, bucket, SEG, est, tw)
            segs, run_est, table = (SEG,), est, ""
        elif s.source == "table":
            bucket, est = m["bucket"], EST_OF[m["est"]]
            tw, har, lag = 2000, "", "global"
            table = Path(s.path).name
            segs, run_est = BARS, est
            if bucket in MAIN_BUCKETS:
                root = NEW
                old = f"{OLD}/arms_hoffman2/{bucket}/{est}/tw{tw}/results_{SEG}.csv"
                new = f"{NEW}/arms_hoffman2/{bucket}/{est}/tw{tw}/results_{SEG}.csv"
                run = nested(NEW, bucket, SEG, est, tw)
            else:
                sub = (
                    "linear_subsection_lassofix"
                    if est == "reclasso"
                    else "linear_subsection"
                )
                root = f"{NEW}/{PULLED[bucket]}/{sub}"
                old = rel(m["arm_twin"])
                new = nested(root, bucket, SEG, est, tw)
        else:  # arm-only bucket study
            bucket, est, tw = m["bucket"], EST_OF[m["est"]], int(m["tw"])
            har, lag = "", "global"
            sub = (
                "linear_subsection_lassofix"
                if est == "reclasso"
                else "linear_subsection"
            )
            root = f"{NEW}/{PULLED[bucket]}/{sub}"
            old = rel(s.path)
            new = nested(root, bucket, SEG, est, tw)
            segs, run_est, table = (SEG,), est, ""
        assert (ROOT / old).is_file(), old
        pulled = old.split("/")[2]
        r.update(
            master_table="B" if s.lo else "A",
            window_start=s.lo or "",
            yhat_table=table,
            bucket=bucket,
            estimator=est,
            segment="bar1600",
            segments_run=len(segs),
            train_win=tw,
            har_base=har,
            lag_scope=lag,
            cluster_old=(
                "local"  # results/linear_subsection/baseline/... was run on the local box
                if pulled == "baseline"
                else "carc"
                if (pulled in CARC_STUDIES or bucket == "live_feasible")
                else "hoffman2"
            ),
            cluster_new="hoffman2",
            results_root=root,
            run_estimator=run_est,
            p_expected=P_NEW.get(bucket, "") if (not har and lag == "global") else "",
            old_path=old,
            new_path=new,
            run_path=run if (s.source == "table" and bucket in MAIN_BUCKETS) else new,
            runs=",".join(segs),
        )
        rows.append(r)
    arms = pd.DataFrame(rows).sort_values(["family", "key"]).reset_index(drop=True)
    assert arms["key"].is_unique and len(arms) == 76, len(arms)
    OUT.mkdir(parents=True, exist_ok=True)
    arms.drop(columns=["runs"]).to_csv(OUT / "arm_list.csv", index=False)

    # task lines: one per (root, bucket, estimator, tw, har, lag) and pack of segments
    runs = (
        arms.groupby(
            [
                "results_root",
                "bucket",
                "run_estimator",
                "train_win",
                "har_base",
                "lag_scope",
            ],
            sort=True,
        )["runs"]
        .agg(lambda x: sorted({s for v in x for s in v.split(",")}))
        .reset_index()
    )
    lines = []
    for _, r in runs.iterrows():
        pack = [b for b in BARS if b in r["runs"]]
        for k in range(0, len(pack), PACK):
            lines.append(
                f"{r['results_root']} {r['bucket']} {r['run_estimator']} {r['train_win']} "
                f"{r['har_base'] or '-'} {r['lag_scope']} {','.join(pack[k : k + PACK])}"
            )
    # the canary: baseline x recursive lasso x 2000 x bar1600 -- it also writes the OLS incumbent,
    # and both must equal the pre-dedup forecasts (least squares and L1 are blind to a copied column)
    canary = f"{NEW} baseline reclasso 2000 - global bar1600"
    assert any(
        ln.startswith(f"{NEW} baseline reclasso 2000 - global bar1") for ln in lines
    )
    TASKS_CANARY.write_text(canary + "\n", encoding="utf-8", newline="\n")
    TASKS_FLEET.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    n_arm_runs = sum(len(ln.split()[-1].split(",")) for ln in lines)
    print(
        f"{len(arms)} forecasts ({arms['master_table'].value_counts().to_dict()} by master-table part); "
        f"{len(runs)} spec configurations, {n_arm_runs} one-bar arm runs in {len(lines)} tasks"
    )
    print(arms.groupby(["family", "segments_run"]).size().to_string())
    print(f"skipped by discover() (not run): {len(skipped)}")


if __name__ == "__main__":
    main()
