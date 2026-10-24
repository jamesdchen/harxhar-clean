"""Close-only window sweep, pooled trees (results/closeonly_wsweep_2014start/RULE.txt section 3):
experiments/trees_train3way.py with the arms CLOSE_w<W> -- design RTH13_w<W> (the pooled sweep's 13-bar design at W,
the /home1 copy of experiments/closewin.py stage), sample its 16:00 rows, window W closes, origins the three-way's 2,579
16:00 rows from 2014-01-02 -- and its walk / merge code unchanged (frozen configurations, seed 42, a fit from scratch
on the W closes before each origin).  The origins are taken from the W = 3,000 design (the three-way's design, md5;
every W design has the same rows, design_check.json).

usage: python experiments/closewin_trees.py <trees_train3way mode: walk | merge | time> [its options]
       python experiments/closewin_trees.py gate --out <gate dir> --ref <three-way CLOSE forecasts.parquet>
                 --ref-md5 <md5> --runs <runs dir>
         gate T: the --ends 10 pieces of CLOSE_w3000 against the three-way CLOSE forecasts (LightGBM / XGBoost repr
         of pred_adj identical, the forest within 1e-14 relative; true_adj and B equal; every class and set) ->
         <gate dir>/gate.json, exit 3 on a failure; on a pass the three-way forecasts are copied to
         <runs>/CLOSE_w3000/forecasts.parquet (reused, md5-checked)
       python experiments/closewin_trees.py cpus --runs <runs dir> --arm CLOSE_w<W> ...
         the merged pieces' CPU models: LightGBM / XGBoost on EPYC 7513 / 7542 only, the forest on those or Xeon
         Silver 4116 (RULE.txt section 3), exit 3 otherwise
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import trees_train3way as TT  # noqa: E402
from closewin import W_REF, WS, design_name, md5  # noqa: E402

for _W in WS:
    TT.DESIGNS[design_name(_W)] = 13 * _W
    TT.ARMS[f"CLOSE_w{_W}"] = dict(
        design=design_name(_W),
        sample="close",
        W=_W,
        desc=f"the closing half hour: the 16:00 rows of the W = {_W:,} design, {_W:,} sessions",
    )
EPYC = {"AMD EPYC 7513 32-Core Processor", "AMD EPYC 7542 32-Core Processor"}
CPUS_OK = {
    "lgbm": EPYC,
    "xgb": EPYC,
    "rf_skx": EPYC | {"Intel(R) Xeon(R) Silver 4116 CPU @ 2.10GHz"},
}


def template_stamps(root: Path) -> pd.DatetimeIndex:
    """The three-way's origins: every OOS row of the W = 3,000 13-bar design from 2014-01-02 (trees_train3way's
    template_stamps on the RTH13_w3000 copy)."""
    d = TT.Design(root, design_name(W_REF), "baseline")
    o = d.dates[d.Wd :]
    return o[o.normalize() >= pd.Timestamp(TT.FIRST_DAY)]


TT.template_stamps = template_stamps


def arg(name: str) -> str:
    return sys.argv[sys.argv.index(name) + 1]


def cmd_gate() -> int:
    out, ref, runs = Path(arg("--out")).resolve(), Path(arg("--ref")).resolve(), Path(arg("--runs")).resolve()
    g: dict = {"ref": str(ref), "ref_md5": md5(ref)}
    ok = g["ref_md5"] == arg("--ref-md5")
    tf = pd.read_parquet(ref)
    covered = set()
    for est in TT.EST:
        for s in TT.SETS:
            p = out / "pieces" / f"CLOSE_w{W_REF}_{est}_{s}_ends10.parquet"
            if not p.is_file():
                continue
            m = json.loads(p.with_suffix(".json").read_text())
            f = pd.read_parquet(p).set_index("date")
            r = tf[(tf.model == est) & (tf.predictor_set == s)].set_index("date").reindex(f.index)
            a_, b_ = f["pred_adj"].to_numpy(float), r["pred_adj"].to_numpy(float)
            text = [repr(x) for x in a_] == [repr(x) for x in b_]
            rel = float(np.max(np.abs(a_ - b_) / np.abs(b_)))
            same_y = bool(np.array_equal(f["true_adj"].to_numpy(), r["true_adj"].to_numpy()))
            same_b = bool(np.array_equal(f["B"].to_numpy(), r["B"].to_numpy()))
            good = (
                bool(np.isfinite(b_).all() and same_y and same_b and (text if est != "rf_skx" else rel <= 1e-14))
                and len(f) == 20
            )
            g[p.stem] = dict(
                origins=len(f),
                first=str(f.index[0]),
                last=str(f.index[-1]),
                threads=m["threads"],
                cpu=m["cpu"],
                window_rows=m["window_rows"],
                text_identical=text,
                max_rel_diff=rel,
                true_adj_equal=same_y,
                B_equal=same_b,
                pass_=good,
            )
            ok &= good
            covered.add((est, s))
    g["pass"] = bool(ok) and covered == {(e, s) for e in TT.EST for s in TT.SETS}
    if g["pass"]:  # W = 3,000 = the three-way CLOSE forecasts (reused)
        dest = runs / f"CLOSE_w{W_REF}" / "forecasts.parquet"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ref, dest)
        assert md5(dest) == g["ref_md5"]
        dest.with_suffix(".json").write_text(
            json.dumps(dict(arm=f"CLOSE_w{W_REF}", reused=str(ref), md5=g["ref_md5"], gate=str(out / "gate.json")))
        )
        g["reused_to"] = str(dest)
    (out / "gate.json").write_text(json.dumps(g, indent=1))
    print(f"GATE T {'PASS' if g['pass'] else 'FAIL'}: {json.dumps(g)}", flush=True)
    return 0 if g["pass"] else 3


def cmd_cpus() -> int:
    runs = Path(arg("--runs")).resolve()
    arms = [sys.argv[i + 1] for i, x in enumerate(sys.argv) if x == "--arm"]
    ok = True
    rec = {}
    for arm in arms:
        cm = json.loads((runs / arm / "forecasts.json").read_text())["cpu_models"]
        for k, v in cm.items():
            est = k.split("/")[0]
            good = set(v) <= CPUS_OK[est]
            rec[f"{arm}/{k}"] = dict(cpus=v, pass_=good)
            ok &= good
    print(f"CPU CHECK {'PASS' if ok else 'FAIL'}: {json.dumps(rec)}", flush=True)
    (runs / f"cpus_{'_'.join(arms)}.json").write_text(json.dumps(dict(pass_=ok, rec=rec), indent=1))
    return 0 if ok else 3


def main() -> int:
    mode = sys.argv[1]
    if mode == "gate":
        return cmd_gate()
    if mode == "cpus":
        return cmd_cpus()
    assert mode in ("walk", "merge", "time"), mode
    return TT.main()


if __name__ == "__main__":
    sys.exit(main())
