"""MinGRU v2, RULE ADDENDUM 1 VALIDATION of Hoffman2's UGE L40S (2026-10-25, RULE ADDENDUM 3f): the checks of the
GPU-type gate run by cluster/pooled_mingru_v2_gate_gputype_h2.sh (run directories ROOT/runs/gate_<tag>_{tune,refit,
tune2}_<job>, laid out as CARC's gate_<type>_<k>_<job>).

  (1) the CARC gate's own checker (ROOT/code/pooled_mingru_final_gate.py = CARC's /home1/jc_905/pmf/code copy, md5 in the
      transfer manifest) on the tune and refit runs, its P100 reference pointed at ROOT/ref/pm13 (the stored P100 runs
      and out/pretune_qlike_folds.csv, copied from CARC /home1/jc_905/pm13) and ROOT/data (pm13_<set>/y.f32)
      -> gate_check.json in each run directory, as on CARC;
  (2) VALIDATION (RULE ADDENDUM 1): self-consistent (tune and tune2 preds.csv identical as sorted text, the CARC gate's
      cmp) and, per set, the seed-mean validation QLIKE within 0.1 % relative of the P100 reference and the 3 best
      epochs equal to the P100's -> PASS only if both sets pass;
  (3) byte identity with CARC's L40S gate 13346568 (ROOT/ref/carc/gate_l40s_<k>_13346568), run by run: preds.csv,
      epochs.csv and steps.csv (decompressed) as sorted text, jobs_out.csv without its timing columns, the refit
      params files byte for byte -> TYPE L40S if all identical, else H2-L40S (a type of its own).
Writes ROOT/runs/gate_<tag>_verdict_<job>.json and .txt ("VALIDATION PASS|FAIL", "TYPE L40S|H2-L40S"; the canary job
of cluster/pooled_mingru_v2_gpu_h2.sh reads the .txt).

    python pooled_mingru_v2_h2_gate_check.py --root ROOT --job JOB [--tag h2l40s]
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CARC_L40S = "13346568"
TOL = 1e-3  # RULE ADDENDUM 1: seed-mean validation QLIKE within 0.1 % relative of the P100 reference
SETS = ("baseline", "all_features")
TIMING = {"t_setup", "t_capture", "t_train", "wave_wall", "wave_size", "wave_train_wall"}


class RefRoot:
    """Stands in for the gate module's PM13 path: data/... -> ROOT/data/..., everything else -> ROOT/ref/pm13/..."""

    def __init__(self, root: Path):
        self.root = root

    def __truediv__(self, p: str | Path) -> Path:
        s = str(p)
        return self.root / s if s.startswith("data/") else self.root / "ref" / "pm13" / s


def load_gate(root: Path) -> Any:
    spec = importlib.util.spec_from_file_location(
        "pooled_mingru_final_gate", root / "code" / "pooled_mingru_final_gate.py"
    )
    assert spec is not None and spec.loader is not None
    mod: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ref = root / "ref" / "pm13" / "runs"
    mod.PM13 = RefRoot(root)
    mod.STORED_TUNE = {
        "baseline": ref / "tune_baseline_f3_s0_13303258/tune",
        "all_features": ref / "tune_all_features_f3_s0_13303117/tune",
    }
    mod.STORED_WALK = {
        "baseline": ref / "walk_baseline_p0_13303665/refit",
        "all_features": ref / "walk_all_features_p0_13303659/refit",
    }
    return mod


def lines(p: Path) -> list[bytes]:
    b = gzip.decompress(p.read_bytes()) if p.suffix == ".gz" else p.read_bytes()
    return sorted(b.splitlines())


def jobs_core(p: Path) -> pd.DataFrame:
    j = pd.read_csv(p, dtype=str)
    return j.drop(columns=[c for c in j.columns if c in TIMING]).sort_values("id").reset_index(drop=True)


def compare_runs(a: Path, b: Path, sub: str) -> dict[str, Any]:
    """Run directory a vs b (each with <sub>/): which outputs are byte-identical; the max relative difference of f."""
    r: dict[str, Any] = {}
    for f in ("preds.csv", "epochs.csv", "steps.csv.gz"):
        pa, pb = a / sub / f, b / sub / f
        r[f] = (
            bool(pa.is_file() and pb.is_file() and lines(pa) == lines(pb)) if (pa.is_file() or pb.is_file()) else None
        )
    r["jobs_out_core"] = bool(jobs_core(a / sub / "jobs_out.csv").equals(jobs_core(b / sub / "jobs_out.csv")))
    pa, pb = a / sub / "params", b / sub / "params"
    if pa.is_dir() or pb.is_dir():
        na = sorted(x.name for x in pa.iterdir()) if pa.is_dir() else []
        nb = sorted(x.name for x in pb.iterdir()) if pb.is_dir() else []
        r["params"] = bool(na == nb and all((pa / n).read_bytes() == (pb / n).read_bytes() for n in na))
    if not r["preds.csv"]:
        ka, kb = (pd.read_csv(x / sub / "preds.csv") for x in (a, b))
        m = ka.merge(kb, on=["id", "kind", "row"], suffixes=("_a", "_b"))
        r["preds_rows_matched"] = [len(m), len(ka), len(kb)]
        r["f_max_rel_diff"] = float(np.max(np.abs(m["f_a"].to_numpy(float) / m["f_b"].to_numpy(float) - 1)))
    r["identical"] = all(
        v
        for k, v in r.items()
        if k in ("preds.csv", "epochs.csv", "steps.csv.gz", "jobs_out_core", "params") and v is not None
    )
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--job", required=True)
    ap.add_argument("--tag", default="h2l40s")
    a = ap.parse_args()
    root = Path(a.root)
    runs = root / "runs"
    run = {k: runs / f"gate_{a.tag}_{k}_{a.job}" for k in ("tune", "refit", "tune2")}
    gate = load_gate(root)
    res: dict[str, Any] = dict(job=a.job, tag=a.tag, tol_rel=TOL)
    # (1) the CARC gate checker on tune and refit (writes gate_check.json; exit 1 = not text-identical to the P100 runs)
    for k in ("tune", "refit"):
        argv = sys.argv
        try:
            sys.argv = ["pooled_mingru_final_gate.py", str(run[k])]
            res[f"carc_gate_checker_{k}_exit"] = int(gate.main())
        except AssertionError as e:  # a job not ok, or not 3 ids a set
            res[f"carc_gate_checker_{k}_exit"] = f"assertion: {e}"
        finally:
            sys.argv = argv
    # (2) VALIDATION
    gc = json.loads((run["tune"] / "gate_check.json").read_text())
    sc = lines(run["tune"] / "tune" / "preds.csv") == lines(run["tune2"] / "tune" / "preds.csv")
    res["self_consistency"] = dict(preds_identical=sc, detail=compare_runs(run["tune"], run["tune2"], "tune"))
    val: dict[str, Any] = {}
    for s in SETS:
        g = gc[s]
        ids = sorted(g["best_epoch"])
        ref = gate.read_ids(gate.STORED_TUNE[s] / "jobs_out.csv", set(ids)).set_index("id")
        val[s] = dict(
            cfg=g["cfg"],
            val_qlike=g["val_qlike"],
            p100_val_qlike=g["stored_val_qlike"],
            rel_diff=g["val_qlike_rel_diff"],
            within_tol=bool(g["val_qlike_rel_diff"] <= TOL),
            best_epochs=[int(g["best_epoch"][i]) for i in ids],
            p100_best_epochs=[int(ref.loc[i, "best_epoch"]) for i in ids],
            best_epochs_equal=bool(g["best_epoch_equal"]),
            f_max_rel_diff_vs_p100=g["f_max_rel_diff"],
            n_rows=g["n_rows"],
        )
        val[s]["pass"] = bool(val[s]["within_tol"] and val[s]["best_epochs_equal"])
    res["validation"] = val
    res["pass"] = bool(sc and all(val[s]["pass"] for s in SETS))
    # refit (first walk origin): reported, not deciding (as on CARC)
    gr = json.loads((run["refit"] / "gate_check.json").read_text())
    res["refit_vs_p100"] = {
        s: dict(f_max_rel_diff=gr[s]["f_max_rel_diff"], rows_equal=gr[s]["rows_equal"]) for s in SETS
    }
    # (3) byte identity with CARC's L40S gate
    ident = {}
    for k, sub in (("tune", "tune"), ("refit", "refit"), ("tune2", "tune")):
        c = root / "ref" / "carc" / f"gate_l40s_{k}_{CARC_L40S}"
        ident[k] = compare_runs(run[k], c, sub)
    res["vs_carc_l40s"] = ident
    res["byte_identical_to_carc_l40s"] = bool(all(v["identical"] for v in ident.values()))
    res["type"] = "L40S" if res["byte_identical_to_carc_l40s"] else "H2-L40S"
    (runs / f"gate_{a.tag}_verdict_{a.job}.json").write_text(json.dumps(res, indent=1, default=str))
    (runs / f"gate_{a.tag}_verdict_{a.job}.txt").write_text(
        f"VALIDATION {'PASS' if res['pass'] else 'FAIL'}\nTYPE {res['type']}\n"
    )
    print(json.dumps(res, indent=1, default=str))
    print("VALIDATION", "PASS" if res["pass"] else "FAIL", "| TYPE", res["type"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
