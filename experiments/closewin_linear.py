"""Close-only window sweep, pooled linear models (results/closeonly_wsweep_2014start/RULE.txt section 2):
experiments/linear_train3way.py's CLOSE arm with its window WD set to W, nothing else changed.  The close design is
derived (derive_close) from the pooled sweep's W design (the /home1 copy made by experiments/closewin.py stage), the
penalties re-tuned on fold 3 with the native rule (pretune, select), the walk refits at every session's 16:00 origin
(walk1), export writes forecasts_rth.parquet.

usage: python experiments/closewin_linear.py <linear_train3way stage> --W <W> --arm CLOSE --out <dir> [its options]
       python experiments/closewin_linear.py gate --out <W = 3,000 out dir> --ref <the three-way CLOSE out dir>
  gate (RULE.txt gate L): the W = 3,000 run against the three-way CLOSE arm -- derived caches md5-identical, frozen
  penalties and pretune_mse.csv identical, forecasts_rth.parquet values identical -> <out>/gate.json, exit 3 otherwise
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from closewin import WS, md5  # noqa: E402

CACHE = ("X.npy", "y.npy", "b_oos.npy", "dates_oos.csv", "dates_all.csv", "rth13_rows.npy")
FC_COLS = ("date", "alpha", "pred_adj", "true_adj", "B", "plain", "pred_raw_duan", "true_raw", "rv_actual")


def gate(out: Path, ref: Path) -> int:
    g: dict = {"out": str(out), "ref": str(ref)}
    ok = True
    for s in ("baseline", "all_features"):
        a, b = out / "_work" / f"design_{s}", ref / "_work" / f"design_{s}"
        r = {fn: [md5(a / fn), md5(b / fn)] for fn in CACHE}
        ja, jb = (json.loads((d / "design.json").read_text()) for d in (a, b))
        # the keys the three-way's gate (i) compared (host, numpy, rerun, part record where the source was built)
        same_meta = {
            k: ja.get(k) == jb.get(k) for k in ("names", "shape", "hp", "config", "raw_cols", "bucket", "f_order")
        }
        dfa = {k: v for k, v in ja["derived_from"].items() if k != "design"}
        dfb = {k: v for k, v in jb["derived_from"].items() if k != "design"}
        r_ok = all(v[0] == v[1] for v in r.values()) and all(same_meta.values()) and dfa == dfb
        g[f"cache_{s}"] = dict(md5=r, meta_equal=all(same_meta.values()), derived_from_equal=dfa == dfb, pass_=r_ok)
        ok &= r_ok
    fa, fb = (json.loads((d / "frozen_penalties.json").read_text()) for d in (out, ref))
    pen = all(fa[m] == fb[m] for m in ("ridge", "lasso", "enet", "ols"))
    ext = fa["info"].get("grid_extensions") == fb["info"].get("grid_extensions") and fa["info"].get(
        "off_spec_grid"
    ) == fb["info"].get("off_spec_grid")
    pa, pb = pd.read_csv(out / "pretune_mse.csv"), pd.read_csv(ref / "pretune_mse.csv")
    mse = pa.shape == pb.shape and bool((pa.astype(str).to_numpy() == pb.astype(str).to_numpy()).all())
    g["penalties"] = dict(
        equal=pen, grid_rule_equal=ext, pretune_mse_equal=mse, mine={m: fa[m] for m in fa if m != "info"}
    )
    ok &= pen and ext and mse
    xa, xb = pd.read_parquet(out / "forecasts_rth.parquet"), pd.read_parquet(ref / "forecasts_rth.parquet")
    g["forecasts_md5"] = [md5(out / "forecasts_rth.parquet"), md5(ref / "forecasts_rth.parquet")]
    keys = ["model", "predictor_set"]
    same_rows = xa.shape == xb.shape and xa[keys].equals(xb[keys])
    cols = {}
    for c in FC_COLS:
        u, v = xa[c].to_numpy(), xb[c].to_numpy()
        cols[c] = bool(same_rows and (np.array_equal(u, v) if c == "date" else np.array_equal(u, v, equal_nan=True)))
    g["forecasts"] = dict(rows=len(xa), same_rows=same_rows, identical=cols)
    ok &= same_rows and all(cols.values())
    g["pass"] = bool(ok)
    (out / "gate.json").write_text(json.dumps(g, indent=1))
    print(f"GATE L {'PASS' if ok else 'FAIL'}: {json.dumps(g)}", flush=True)
    return 0 if ok else 3


def main() -> int:
    if sys.argv[1] == "gate":
        a = sys.argv
        return gate(Path(a[a.index("--out") + 1]).resolve(), Path(a[a.index("--ref") + 1]).resolve())
    i = sys.argv.index("--W")
    W = int(sys.argv[i + 1])
    del sys.argv[i : i + 2]
    assert W in WS, W
    a = sys.argv
    assert a[a.index("--arm") + 1] == "CLOSE", "the close arm only"
    import linear_train3way as L3

    L3.WD = W  # the arm's window (closes) and the 13-bar source design's window (13 W rows) in derive_close
    L3.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
